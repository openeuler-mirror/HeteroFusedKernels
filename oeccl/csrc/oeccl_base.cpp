#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <experiment/runtime/runtime/mem.h>
#include <fcntl.h>
#include <fstream>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <iostream>
#include <limits.h>
#include <numa.h>
#include <numaif.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define SHMEM_SIZE (1024 * 1024 * 1024) // Shared memory size on each numa node
#define MAX_NUMA_NODES 2 // Currently only supports two numa node scenarios
#define DEFAULT_RATIO 8 // Default ratio of data transferred via PCIe, means 1/ratio data via PCIe
#define RETRY_INTERVAL_US 10 // Retry interval in microseconds
#define MAX_SHM_NAME_LEN 40
#define MAX_AREA_NAME_LEN 40
#define HUGETLBFS_DIR "/hugepages" // Directory for hugetlbfs

#define ACLCHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        aclError ret = cmd;                                                                                            \
        if (ret != ACL_SUCCESS) {                                                                                      \
            printf("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);                         \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                               \
                printf("memory allocation error, check whether the current "                                           \
                       "memory space is sufficient.\n");                                                               \
            }                                                                                                          \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

#define ACLCHECKRET(cmd)                                                                                               \
    do {                                                                                                               \
        aclError ret = cmd;                                                                                            \
        if (ret != ACL_SUCCESS) {                                                                                      \
            printf("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);                         \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                               \
                printf("memory allocation error, check whether the current "                                           \
                       "memory space is sufficient.\n");                                                               \
            }                                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define ACLCHECKsync(cmd)                                                                                              \
    do {                                                                                                               \
        aclError ret = cmd;                                                                                            \
        if (ret != ACL_SUCCESS) {                                                                                      \
            printf("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);                         \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                               \
                printf("memory allocation error, check whether the current "                                           \
                       "memory space is sufficient.\n");                                                               \
            }                                                                                                          \
            return {0, 0};                                                                                             \
        }                                                                                                              \
    } while (0)

pthread_t acl_callback_thread_local;
pthread_t acl_callback_thread_remote;
pthread_t acl_callback_thread_h2h;
HcclComm global_comm;
aclrtContext context_ = nullptr;
uint64_t rank_id;
int rank_size;
bool is_async_op = false;
void *flag[MAX_NUMA_NODES];
void *finish[MAX_NUMA_NODES];
void *all_data[MAX_NUMA_NODES];
int datafd[MAX_NUMA_NODES];
int finishfd[MAX_NUMA_NODES];
int flagfd[MAX_NUMA_NODES];
bool callback_exit = true;
bool begin_exit = false;
bool oeccl_init = false;

aclrtEvent global_event, local_rank_event, h2h_event, remote_rank_event, memcpyasync_event, end_global_event,
    end_memcpyasync_event, end_local_rank_event, end_h2h_event, end_remote_rank_event, set_flag_event;
aclrtStream global_stream, local_rank_stream, h2h_stream, remote_rank_stream, memcpyasync_stream;
int dev_id;
int ratio = DEFAULT_RATIO;
uint64_t root_rank = 0;
void *ready_flag_value = nullptr;                    // Used to set the data ready status
std::map<size_t, size_t> root_rank_list;             // Maps NUMA node ID to its root rank
std::map<size_t, std::vector<size_t>> numa_rank_map; // Maps NUMA node ID to all its ranks

struct put_arg {
    void *new_device_ptr;
    void *new_send_ptr;
    uint64_t new_data_count;
    uint64_t new_data_type;
    void *default_stream;
};

std::map<at::ScalarType, HcclDataType> kScalarTypeToHcclDataType = {
    {at::kByte, HCCL_DATA_TYPE_UINT8},     {at::kChar, HCCL_DATA_TYPE_INT8},   {at::kShort, HCCL_DATA_TYPE_INT16},
    {at::kInt, HCCL_DATA_TYPE_INT32},      {at::kLong, HCCL_DATA_TYPE_INT64},  {at::kHalf, HCCL_DATA_TYPE_FP16},
    {at::kFloat, HCCL_DATA_TYPE_FP32},     {at::kDouble, HCCL_DATA_TYPE_FP64}, {at::kBool, HCCL_DATA_TYPE_UINT8},
    {at::kBFloat16, HCCL_DATA_TYPE_BFP16},
};

HcclDataType getHcclDataType(at::ScalarType scalar_type) {
    try {
        return kScalarTypeToHcclDataType[scalar_type];
    } catch (std::out_of_range &e) {
        throw std::runtime_error("Unsupported data type for HCCL process group");
    }
}

void createShm_rtMallocHostSharedMemory(const char *name, size_t size, int *fd, void **ptr) {
    rtMallocHostSharedMemoryIn input_para = {name, size, O_CREAT | O_RDWR};
    rtMallocHostSharedMemoryOut output_para = {};
    int rt_ret = rtMallocHostSharedMemory(&input_para, &output_para);
    if (rt_ret) {
        throw std::runtime_error("rtMallocHostSharedMemory faild");
    }
    *fd = output_para.fd;
    *ptr = output_para.ptr;
    return;
}

void createShm_hugetlbfs(const char *name, size_t size, int *fd, void **ptr) {
    char shm_name[MAX_SHM_NAME_LEN] = {};
    sprintf(shm_name, "%s/%s", HUGETLBFS_DIR, name);
    *fd = open(shm_name, O_CREAT | O_RDWR, 0666);
    ftruncate(*fd, size);
    *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    return;
}

void createShm_shm_open(const char *name, size_t size, int *fd, void **ptr) {
    *fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    ftruncate(*fd, size);
    *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    return;
}

void (*createShm_funcPtr)(const char *, size_t, int *, void **);

// numa node to which the current process is bound
int local_numa_node = 0;

void mbind_mlock(void **ptr, int size) {
    unsigned long nodemask = 1UL << local_numa_node; // assume number of nodes <= 64
    unsigned long maxnode = sizeof(nodemask) * 8;

    // Bind mapped memory using mbind
    // MPOL_BIND: Strict binding, only allowed to be allocated on specified nodes
    mbind(*ptr, size, MPOL_BIND, &nodemask, maxnode, 0);
    memset(*ptr, 0, size);
    mlock(*ptr, size);
}

static void configure_and_bind_numa(const std::vector<int> &numa_map) {
    if (!numa_map.empty() && rank_id < numa_map.size()) {
        local_numa_node = numa_map[rank_id];

        // Build NUMA node mapping table
        if (root_rank_list.empty()) {
            numa_rank_map.clear();

            for (size_t i = 0; i < numa_map.size(); i++) {
                int numa_id = numa_map[i];

                // Collect all ranks of each NUMA node
                numa_rank_map[numa_id].push_back(i);

                // Also handles root rank selection
                if (root_rank_list.find(numa_id) == root_rank_list.end()) {
                    root_rank_list[numa_id] = i;
                }
            }
        }
    } else {
        perror("NUMA map is empty or rank_id out of range, please check!");
        return;
    }

    // bind numa node
    if (numa_run_on_node(local_numa_node) != 0) {
        perror("numa_run_on_node failed");
    }

    numa_set_preferred(local_numa_node);
}

int RootInitShmem() {
    char area_name[MAX_AREA_NAME_LEN];
    // Create shared memory areas for each NUMA node
    for (int numa_id = 0; numa_id < MAX_NUMA_NODES; numa_id++) {
        // Create data area
        sprintf(area_name, "data_area%d", numa_id);
        createShm_funcPtr(area_name, SHMEM_SIZE, &datafd[numa_id], &all_data[numa_id]);

        // Create finish area
        sprintf(area_name, "finish_area%d", numa_id);
        createShm_funcPtr(area_name, rank_size * sizeof(int), &finishfd[numa_id], &finish[numa_id]);

        // Create flag area
        sprintf(area_name, "flag_area%d", numa_id);
        createShm_funcPtr(area_name, rank_size * sizeof(int), &flagfd[numa_id], &flag[numa_id]);

        // If this process is the root for this NUMA node, perform memory binding
        if (local_numa_node == numa_id && rank_id == root_rank_list[local_numa_node]) {
            mbind_mlock(&all_data[numa_id], SHMEM_SIZE);
            mbind_mlock(&finish[numa_id], rank_size * sizeof(int));
            mbind_mlock(&flag[numa_id], rank_size * sizeof(int));
        }
    }

    return 0;
}

void attachShm(const char *name, int size, int *fd, void **ptr) {
    *fd = shm_open(name, O_RDWR, 0666);
    while (*fd < 0) {
        usleep(RETRY_INTERVAL_US);
        *fd = shm_open(name, O_RDWR, 0666);
    }
    *ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    return;
}

bool ifHuge = true;

int OpenShmem() {
    char area_name[MAX_AREA_NAME_LEN];
    if (ifHuge) {
        // Open shared memory areas by createShm_funcPtr when using hugepages
        for (int numa_id = 0; numa_id < MAX_NUMA_NODES; numa_id++) {
            sprintf(area_name, "data_area%d", numa_id);
            createShm_funcPtr(area_name, SHMEM_SIZE, &datafd[numa_id], &all_data[numa_id]);

            sprintf(area_name, "finish_area%d", numa_id);
            createShm_funcPtr(area_name, rank_size * sizeof(int), &finishfd[numa_id], &finish[numa_id]);

            sprintf(area_name, "flag_area%d", numa_id);
            createShm_funcPtr(area_name, rank_size * sizeof(int), &flagfd[numa_id], &flag[numa_id]);
        }
    } else {
        // Attach to shared memory areas for each NUMA node
        for (int numa_id = 0; numa_id < MAX_NUMA_NODES; numa_id++) {
            sprintf(area_name, "data_area%d", numa_id);
            attachShm(area_name, SHMEM_SIZE, &datafd[numa_id], &all_data[numa_id]);

            sprintf(area_name, "finish_area%d", numa_id);
            attachShm(area_name, rank_size * sizeof(int), &finishfd[numa_id], &finish[numa_id]);

            sprintf(area_name, "flag_area%d", numa_id);
            attachShm(area_name, rank_size * sizeof(int), &flagfd[numa_id], &flag[numa_id]);
        }
    }

    return 0;
}

uint8_t *getRawPointer(at::Tensor &tensor) {
    return reinterpret_cast<uint8_t *>(static_cast<uint8_t *>(tensor.storage().data_ptr().get()) +
                                       tensor.storage_offset() * tensor.dtype().itemsize());
}

void *ProcessCallback(void *arg) {
    aclrtSetCurrentContext(context_);

    // Use the configured local_numa_node (set during configure_and_bind_numa)
    // so callbacks follow the same NUMA affinity policy.
    int node_id = local_numa_node;
    if (numa_run_on_node(node_id) != 0) {
        perror("numa_run_on_node failed");
    }

    numa_set_preferred(node_id);

    while (!begin_exit) {
        (void)aclrtProcessReport(1);
        if (*(static_cast<bool *>(arg)) == true) {
            return NULL;
        }
    }
}

// Copy the data of all ranks on the local node to the device
void local_rank_copy_func(void *para) {
    aclrtSetCurrentContext(context_);
    struct put_arg *input_para = (struct put_arg *)para;

    void *device_ptr = input_para->new_device_ptr;
    uint64_t data_count = input_para->new_data_count;
    uint64_t data_type = input_para->new_data_type;

    uint64_t dataPCIE = (data_count / ratio) * data_type;
    uint64_t dataXCCL = (data_count - data_count / ratio) * data_type;

    int node_idx = local_numa_node;
    void *all_data_ptr = all_data[node_idx];
    void *flag_ptr = flag[node_idx];
    void *finish_ptr = finish[node_idx];

    // Use numa_rank_map to traverse all ranks on the current NUMA node
    for (size_t i : numa_rank_map[local_numa_node]) {
        if (i == rank_id) {
            continue;
        }
        while (((std::atomic<char> *)flag_ptr)[i].load() != 1)
            ;
        ACLCHECKRET(aclrtMemcpy((void *)((char *)(device_ptr) + data_count * data_type * i + dataXCCL), dataPCIE,
                                (void *)((char *)(all_data_ptr) + dataPCIE * i), dataPCIE, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    ((std::atomic<int> *)finish_ptr)[rank_id].store(1, std::memory_order_relaxed);
}

#define INVALID_RANK -1

// Copy the data of the corresponding rank npu from the remote node
void between_node_copy_func(void *para) {
    aclrtSetCurrentContext(context_);
    struct put_arg *input_para = (struct put_arg *)para;

    void *device_ptr = input_para->new_device_ptr;
    uint64_t data_count = input_para->new_data_count;
    uint64_t data_type = input_para->new_data_type;

    uint64_t dataPCIE = (data_count / ratio) * data_type;
    uint64_t dataXCCL = (data_count - data_count / ratio) * data_type;

    // Get the corresponding NUMA node ID (opposite of the current node)
    // Todo: support multi-NUMA node scenarios
    int src_numa = (local_numa_node == 0) ? 1 : 0;
    int dst_numa = local_numa_node;
    void *all_data_src = all_data[src_numa];
    void *all_data_dst = all_data[dst_numa];
    void *flag_src = flag[src_numa];
    void *flag_dst = flag[dst_numa];

    // Find its own position index in the rank list of the current NUMA node
    size_t current_pos = INVALID_RANK;
    auto &dst_ranks = numa_rank_map[dst_numa];
    for (size_t i = 0; i < dst_ranks.size(); i++) {
        if (dst_ranks[i] == rank_id) {
            current_pos = i;
            break;
        }
    }

    // Use the same position index to obtain the corresponding rank on the target NUMA node
    size_t corresponding_rank = INVALID_RANK;
    auto &src_ranks = numa_rank_map[src_numa];
    if (current_pos >= 0 && current_pos < src_ranks.size()) {
        corresponding_rank = src_ranks[current_pos];
    }

    if (INVALID_RANK == corresponding_rank) {
        perror("Error: Could not find matching rank in dst NUMA node\n");
        return;
    }

    while (((std::atomic<char> *)flag_src)[corresponding_rank].load() != 1)
        ;
    ACLCHECKRET(aclrtMemcpy((void *)((char *)(all_data_dst) + dataPCIE * corresponding_rank), dataPCIE,
                            (void *)((char *)(all_data_src) + dataPCIE * corresponding_rank), dataPCIE,
                            ACL_MEMCPY_HOST_TO_HOST));

    ((std::atomic<char> *)flag_dst)[corresponding_rank].store(1, std::memory_order_relaxed);

    ACLCHECKRET(aclrtMemcpy((void *)((char *)(device_ptr) + data_count * data_type * corresponding_rank + dataXCCL), dataPCIE,
                            (void *)((char *)(all_data_dst) + dataPCIE * corresponding_rank), dataPCIE,
                            ACL_MEMCPY_HOST_TO_DEVICE));
}

// Copy the data of all ranks on the remote node to the device
void remote_rank_copy_func(void *para) {
    aclrtSetCurrentContext(context_);
    struct put_arg *input_para = (struct put_arg *)para;

    void *device_ptr = input_para->new_device_ptr;
    uint64_t data_count = input_para->new_data_count;
    uint64_t data_type = input_para->new_data_type;

    uint64_t dataPCIE = (data_count / ratio) * data_type;
    uint64_t dataXCCL = (data_count - data_count / ratio) * data_type;

    void *all_data_ptr = all_data[local_numa_node];
    void *flag_ptr = flag[local_numa_node];
    void *finish_ptr = finish[local_numa_node];

    // Find its own position index in the rank list of the current NUMA node
    // Todo: support multi-NUMA node scenarios
    size_t current_pos = INVALID_RANK;
    auto &local_ranks = numa_rank_map[local_numa_node];
    for (size_t i = 0; i < local_ranks.size(); i++) {
        if (local_ranks[i] == rank_id) {
            current_pos = i;
            break;
        }
    }

    // Use the same position index to obtain the corresponding rank on the target NUMA node
    size_t corresponding_rank = INVALID_RANK;
    auto &remote_ranks = numa_rank_map[(local_numa_node == 0) ? 1 : 0];
    if (current_pos >= 0 && current_pos < remote_ranks.size()) {
        corresponding_rank = remote_ranks[current_pos];
    }

    std::vector<int> cp_flag(remote_ranks.size(), 0);
    while (true) {
        for (size_t i = 0; i < remote_ranks.size(); i++) {
            size_t remote_rank = remote_ranks[i];
            if (cp_flag[i] == 1 || remote_rank == corresponding_rank) {
                continue;
            }

            if (((std::atomic<char> *)flag_ptr)[remote_rank].load() == 1 && cp_flag[i] == 0) {
                ACLCHECKRET(aclrtMemcpy(
                    (void *)((char *)(device_ptr) + data_count * data_type * remote_rank + dataXCCL), dataPCIE,
                    (void *)((char *)(all_data_ptr) + dataPCIE * remote_rank), dataPCIE, ACL_MEMCPY_HOST_TO_DEVICE));
                cp_flag[i] = 1;
            }
        }

        size_t sum = 0;
        for (size_t i = 0; i < cp_flag.size(); i++) {
            sum += cp_flag[i];
        }
        if (sum == remote_ranks.size() - 1) { // -1 because corresponding_rank is skipped
            break;
        }
    }
    ((std::atomic<int> *)finish_ptr)[corresponding_rank].store(1, std::memory_order_relaxed);
}

void flag_clear_func(void *para) {
    // Aggregate finish status across all NUMA nodes defined by MAX_NUMA_NODES
    std::vector<int> status(MAX_NUMA_NODES, 0);
    bool all_ok = false;
    while (!all_ok) {
        // reset statuses
        for (int n = 0; n < MAX_NUMA_NODES; n++) {
            status[n] = 0;
        }

        // sum finish counters for each NUMA node
        for (int i = 0; i < rank_size; i++) {
            for (int n = 0; n < MAX_NUMA_NODES; n++) {
                status[n] += ((std::atomic<int> *)finish[n])[i].load();
            }
        }

        // check whether every NUMA node has all ranks finished
        all_ok = true;
        for (int n = 0; n < MAX_NUMA_NODES; n++) {
            if (status[n] != rank_size) {
                all_ok = false;
                break;
            }
        }

        if (all_ok) {
            // clear finish and flag arrays for all NUMA nodes
            for (int n = 0; n < MAX_NUMA_NODES; n++) {
                memset(finish[n], 0, rank_size * sizeof(int));
                memset(flag[n], 0, rank_size * sizeof(int));
            }
        } else {
            usleep(RETRY_INTERVAL_US);
        }
    }
}

void shm_clear_func() {
    // Clear shared-memory data areas for all configured NUMA nodes
    for (int n = 0; n < MAX_NUMA_NODES; n++) {
        if (all_data[n] != nullptr) {
            memset(all_data[n], 0, SHMEM_SIZE);
        }
    }
}

#define MAX_RANK 8

int InitOEccl(unsigned long input, bool is_huge, int in_r, const std::vector<int> &numa_map, bool is_async) {

    if (oeccl_init)
        return 0;

    ratio = in_r;
    ifHuge = is_huge;
    HcclComm comm = (HcclComm)input;

    if (is_huge) {
        // createShm_rtMallocHostSharedMemory can also be used here
        createShm_funcPtr = createShm_hugetlbfs;
    } else {
        createShm_funcPtr = createShm_shm_open;
    }
    
    HcclGetRankId(comm, (uint32_t *)(&rank_id));
    HcclGetRankSize(comm, (uint32_t *)(&rank_size));

    configure_and_bind_numa(numa_map);

    is_async_op = is_async;

    int ret = 0;
    global_comm = comm;

    if (root_rank_list.count(local_numa_node) && rank_id == root_rank_list[local_numa_node]) {
        ret = RootInitShmem();
    } else {
        ret = OpenShmem();
    }

    callback_exit = false;
    pthread_create(&acl_callback_thread_local, NULL, ProcessCallback, (void *)(&callback_exit));
    pthread_create(&acl_callback_thread_remote, NULL, ProcessCallback, (void *)(&callback_exit));
    pthread_create(&acl_callback_thread_h2h, NULL, ProcessCallback, (void *)(&callback_exit));

    ACLCHECK(aclrtGetDevice(&dev_id));

    ACLCHECK(aclrtGetCurrentContext(&context_));
    ACLCHECK(aclrtCreateEventWithFlag(&global_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&local_rank_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&remote_rank_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&h2h_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&end_global_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&end_local_rank_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&end_remote_rank_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&end_h2h_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&end_memcpyasync_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&memcpyasync_event, ACL_EVENT_SYNC));
    ACLCHECK(aclrtCreateEventWithFlag(&set_flag_event, ACL_EVENT_SYNC));

    ACLCHECK(aclrtCreateStream(&local_rank_stream));  // for local_rank_copy_func
    ACLCHECK(aclrtCreateStream(&remote_rank_stream)); // for remote_rank_copy_func
    ACLCHECK(aclrtCreateStream(&h2h_stream));
    ACLCHECK(aclrtCreateStream(&global_stream));
    ACLCHECK(aclrtCreateStream(&memcpyasync_stream));

    ACLCHECK(aclrtSubscribeReport(static_cast<uint64_t>(acl_callback_thread_local), local_rank_stream));
    ACLCHECK(aclrtSubscribeReport(static_cast<uint64_t>(acl_callback_thread_remote), remote_rank_stream));
    ACLCHECK(aclrtSubscribeReport(static_cast<uint64_t>(acl_callback_thread_h2h), h2h_stream));

    ACLCHECK(aclrtMalloc(&ready_flag_value, sizeof(int), ACL_MEM_MALLOC_NORMAL_ONLY));
    ACLCHECK(aclrtMemset(ready_flag_value, sizeof(int), 1, sizeof(int)));

    oeccl_init = true;
    return ret;
}

int HcclAllGatherCustom(at::Tensor &output, at::Tensor &input) {
    void *aclStream = nullptr;
    if (aclStream == nullptr) {
        int deviceId = 0;
        ACLCHECK(aclrtGetDevice(&deviceId));
        aclStream = c10_npu::getCurrentNPUStream(deviceId).stream();
    }
    uint64_t dataCount = input.numel();
    void *pInput = (void *)getRawPointer(input);
    void *pOutput = (void *)getRawPointer(output);
    HcclDataType dataType = getHcclDataType(input.scalar_type());

    ACLCHECK(aclrtRecordEvent(global_event, aclStream));
    ACLCHECK(aclrtStreamWaitEvent(global_stream, global_event));
    ACLCHECK(aclrtResetEvent(global_event, global_stream));
    ACLCHECK(HcclAllGather((void *)pInput, (void *)pOutput, dataCount, dataType, global_comm, global_stream));
    ACLCHECK(aclrtRecordEvent(end_global_event, global_stream));
    ACLCHECK(aclrtStreamWaitEvent(aclStream, end_global_event));
    ACLCHECK(aclrtResetEvent(end_global_event, aclStream));
    return 0;
}

struct EventToken {
    uintptr_t stream;
    uintptr_t event;
};

void wait_event_on_stream(uintptr_t stream_ptr, uintptr_t event_ptr) {
    aclrtStream stream = reinterpret_cast<aclrtStream>(stream_ptr);
    aclrtEvent event = reinterpret_cast<aclrtEvent>(event_ptr);
    void *aclStream = nullptr;
    if (aclStream == nullptr) {
        int deviceId = 0;
        ACLCHECKRET(aclrtGetDevice(&deviceId));
        aclStream = c10_npu::getCurrentNPUStream(deviceId).stream();
    }
    ACLCHECKRET(aclrtStreamWaitEvent(aclStream, event));
    ACLCHECKRET(aclrtResetEvent(event, aclStream));
}

EventToken OEcclAllGather(at::Tensor &output, at::Tensor &input) {
    if (!oeccl_init) {
        perror("please call init_oeccl first");
        return {0, 0};
    }
    void *aclStream = nullptr;
    if (aclStream == nullptr) {
        int deviceId = 0;
        ACLCHECKsync(aclrtGetDevice(&deviceId));
        aclStream = c10_npu::getCurrentNPUStream(deviceId).stream();
    }
    uint64_t dataCount = input.numel();
    void *pInput = (void *)getRawPointer(input);
    void *pOutput = (void *)getRawPointer(output);
    HcclDataType dataType = getHcclDataType(input.scalar_type());

    struct put_arg *new_put_arg = (struct put_arg *)malloc(sizeof(struct put_arg));

    uint64_t dataPCIE = dataCount / ratio;
    uint64_t dataXCCL = dataCount - dataPCIE;
    uint64_t data_type = input.element_size();
    new_put_arg->new_device_ptr = (void *)pOutput;
    new_put_arg->new_data_count = dataCount;
    new_put_arg->new_send_ptr = (void *)((char *)pInput + dataXCCL * data_type);
    new_put_arg->new_data_type = input.element_size();
    new_put_arg->default_stream = aclStream;
    ACLCHECKsync(aclrtRecordEvent(global_event, aclStream));
    ACLCHECKsync(aclrtStreamWaitEvent(global_stream, global_event));
    ACLCHECKsync(aclrtResetEvent(global_event, global_stream));

    ACLCHECKsync(aclrtRecordEvent(local_rank_event, global_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(local_rank_stream, local_rank_event));
    ACLCHECKsync(aclrtResetEvent(local_rank_event, local_rank_stream));

    ACLCHECKsync(aclrtRecordEvent(h2h_event, local_rank_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(h2h_stream, h2h_event));
    ACLCHECKsync(aclrtResetEvent(h2h_event, h2h_stream));

    ACLCHECKsync(aclrtRecordEvent(remote_rank_event, h2h_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(remote_rank_stream, remote_rank_event));
    ACLCHECKsync(aclrtResetEvent(remote_rank_event, remote_rank_stream));

    ACLCHECKsync(aclrtRecordEvent(memcpyasync_event, remote_rank_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(memcpyasync_stream, memcpyasync_event));
    ACLCHECKsync(aclrtResetEvent(memcpyasync_event, memcpyasync_stream));

    uint64_t recvCounts[rank_size];
    for (int i = 0; i < rank_size; i++) {
        recvCounts[i] = dataXCCL;
    }
    uint64_t recvDispls[rank_size];
    for (int i = 0; i < rank_size; i++) {
        recvDispls[i] = dataCount * i;
    }

    void *all_data_ptr = all_data[local_numa_node];
    void *flag_ptr = flag[local_numa_node];

    ACLCHECKsync(aclrtLaunchCallback(local_rank_copy_func, (void *)new_put_arg, ACL_CALLBACK_BLOCK, local_rank_stream));
    ACLCHECKsync(aclrtLaunchCallback(between_node_copy_func, (void *)new_put_arg, ACL_CALLBACK_BLOCK, h2h_stream));
    ACLCHECKsync(
        aclrtLaunchCallback(remote_rank_copy_func, (void *)new_put_arg, ACL_CALLBACK_BLOCK, remote_rank_stream));

    ACLCHECKsync(aclrtMemcpyAsync((void *)((char *)(all_data_ptr) + dataPCIE * data_type * rank_id), dataPCIE * data_type,
                                  (void *)((char *)(pInput) + dataXCCL * data_type), dataPCIE * data_type,
                                  ACL_MEMCPY_DEVICE_TO_HOST, memcpyasync_stream));
    ACLCHECKsync(aclrtMemcpyAsync((void *)((char *)flag_ptr + rank_id), 1, ready_flag_value, 1, ACL_MEMCPY_DEVICE_TO_HOST,
                                  memcpyasync_stream));
    ACLCHECKsync(aclrtMemcpyAsync((void *)((char *)(pOutput) + dataCount * data_type * rank_id + dataXCCL * data_type),
                                  dataPCIE * data_type, (void *)((char *)(all_data_ptr) + dataPCIE * data_type * rank_id),
                                  dataPCIE * data_type, ACL_MEMCPY_HOST_TO_DEVICE, memcpyasync_stream));

    ACLCHECKsync(HcclAllGatherV((void *)pInput, dataXCCL, (void *)pOutput, recvCounts, recvDispls, dataType,
                                global_comm, global_stream));

    if (rank_id == root_rank) {
        ACLCHECKsync(aclrtLaunchCallback(flag_clear_func, nullptr, ACL_CALLBACK_NO_BLOCK, local_rank_stream));
    }

    ACLCHECKsync(aclrtRecordEvent(end_memcpyasync_event, memcpyasync_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(global_stream, end_memcpyasync_event));
    ACLCHECKsync(aclrtResetEvent(end_memcpyasync_event, global_stream));

    ACLCHECKsync(aclrtRecordEvent(end_h2h_event, h2h_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(global_stream, end_h2h_event));
    ACLCHECKsync(aclrtResetEvent(end_h2h_event, global_stream));

    ACLCHECKsync(aclrtRecordEvent(end_local_rank_event, local_rank_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(global_stream, end_local_rank_event));
    ACLCHECKsync(aclrtResetEvent(end_local_rank_event, global_stream));

    ACLCHECKsync(aclrtRecordEvent(end_remote_rank_event, remote_rank_stream));
    ACLCHECKsync(aclrtStreamWaitEvent(global_stream, end_remote_rank_event));
    ACLCHECKsync(aclrtResetEvent(end_remote_rank_event, global_stream));

    ACLCHECKsync(aclrtRecordEvent(end_global_event, global_stream));
    if (is_async_op)
        return EventToken{reinterpret_cast<uintptr_t>(aclStream), reinterpret_cast<uintptr_t>(end_global_event)};
    else {
        ACLCHECKsync(aclrtStreamWaitEvent(aclStream, end_global_event));
        ACLCHECKsync(aclrtResetEvent(end_global_event, aclStream));
        return {0, 0};
    }
}

int CleanUpFunc() {
    begin_exit = true;
    ACLCHECK(aclrtSetCurrentContext(context_));
    ACLCHECK(aclrtSynchronizeStream(global_stream));
    ACLCHECK(aclrtSynchronizeStream(local_rank_stream));
    ACLCHECK(aclrtSynchronizeStream(remote_rank_stream));
    ACLCHECK(aclrtSynchronizeStream(h2h_stream));
    ACLCHECK(aclrtSynchronizeStream(memcpyasync_stream));

    callback_exit = true;
    (void)pthread_join(acl_callback_thread_local, nullptr);
    ACLCHECK(aclrtUnSubscribeReport(static_cast<uint64_t>(acl_callback_thread_local), local_rank_stream));

    (void)pthread_join(acl_callback_thread_remote, nullptr);
    ACLCHECK(aclrtUnSubscribeReport(static_cast<uint64_t>(acl_callback_thread_remote), remote_rank_stream));

    (void)pthread_join(acl_callback_thread_h2h, nullptr);
    ACLCHECK(aclrtUnSubscribeReport(static_cast<uint64_t>(acl_callback_thread_h2h), h2h_stream));

    int ret = 0;
    if (oeccl_init) {
        ACLCHECK(aclrtDestroyStream(remote_rank_stream));
        ACLCHECK(aclrtDestroyStream(h2h_stream));
        ACLCHECK(aclrtDestroyStream(local_rank_stream));
        ACLCHECK(aclrtDestroyStream(global_stream));
        ACLCHECK(aclrtDestroyStream(memcpyasync_stream));
        ACLCHECK(aclrtDestroyEvent(end_local_rank_event));
        ACLCHECK(aclrtDestroyEvent(end_global_event));
        ACLCHECK(aclrtDestroyEvent(end_remote_rank_event));
        ACLCHECK(aclrtDestroyEvent(end_h2h_event));
        ACLCHECK(aclrtDestroyEvent(local_rank_event));
        ACLCHECK(aclrtDestroyEvent(global_event));
        ACLCHECK(aclrtDestroyEvent(remote_rank_event));
        ACLCHECK(aclrtDestroyEvent(h2h_event));
        ACLCHECK(aclrtDestroyEvent(end_memcpyasync_event));
        ACLCHECK(aclrtDestroyEvent(memcpyasync_event));
    }
    return ret;
}

PYBIND11_MODULE(ops, m) {
    m.def("init_oeccl", &InitOEccl, "init_oeccl");
    m.def("oeccl_cleanup", &CleanUpFunc, "oeccl_cleanup");
    m.def("oeccl_allgather", &OEcclAllGather, "oeccl_allgather");
    m.def("hccl_allgather", &HcclAllGatherCustom, "hccl_allgather");
    m.def("shm_clean", &shm_clear_func, "shm_clean");
    pybind11::class_<EventToken>(m, "EventToken")
        .def_readwrite("stream", &EventToken::stream)
        .def_readwrite("event", &EventToken::event);
    m.def("wait_event_on_stream", &wait_event_on_stream, "wait_event_on_stream");
}
