#include "mem_alloc.h"
#include "managed_mem.h"
#include <sys/mman.h>
#include <numaif.h>
#include <string>
#include <errno.h>
#include <cstring>        
#include <acl/acl.h>
#include <cstdlib>
#include <fstream>
#include "dcmi_management.h"
#include <iostream>

/*
* This function is potentially slow for the mbind
*/
torch::Tensor alloc_numa_pinned_tensor(std::size_t size) {
    auto& dcmiManager = dcmi_ascend::DCMIManager::GetInstance();
    int32_t deviceId = 0;
    auto err = aclrtGetDevice(&deviceId);
    TORCH_CHECK(err == 0, "Unable to obtain device Id");
    dcmi_ascend::dcmi_pcie_info_all pcieInfo;
    std::string pcieBusId = dcmiManager.getDevicePcieInfoV2(static_cast<int>(deviceId), 0, &pcieInfo);

    std::string numaPath = "/sys/bus/pci/devices/" + pcieBusId + "/numa_node";
    std::ifstream file(numaPath);
    if (!file.is_open()) {
        TORCH_CHECK(false, "numa_node file not found");
    }

    std::string content;
    std::getline(file, content);

    if(file.fail() && !file.eof()) {
        TORCH_CHECK(false, "failed to read numa_node file");
    }

    int numaNode = std::stoi(content);

    auto hostPtr = alloc_pinned_numa_ptr(size, numaNode);

    // convert to tensor
    torch::TensorOptions tensorOpsCpu = torch::TensorOptions()
                                                .dtype(torch::kInt8)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
    int64_t numel = static_cast<int64_t>(size);
    std::vector<int64_t> dims = {numel};

    auto deleter = [size](void* ptr) {
        // No need to look up size — we already have it!
        auto unRegErr = unregister_ptr(ptr);
        auto unMapErr = munmap(ptr, size);

        // Do NOT throw in deleter — log only
        if (unMapErr != 0) {
            std::cerr << "munmap failed in tensor deleter: " << unMapErr << std::endl;
        }
    };

    torch::Tensor newTensorFromMyptr = torch::from_blob(
        reinterpret_cast<void*>(hostPtr), dims, deleter, tensorOpsCpu);
    return newTensorFromMyptr;
}

void free_pinned_numa_ptr(void* ptr) {
    auto& memManager = managed_memory::HostRegisteredMemoryManager::GetInstance();
    size_t size = memManager.getRecordSize(ptr);
    auto unRegErr = unregister_ptr(ptr);
    auto unMapErr = munmap(ptr, size);
    if (unRegErr) {
        throw std::runtime_error("unregister_ptr failed: " + std::to_string(unRegErr));
    }
    if (unMapErr) {
        throw std::runtime_error("munmap failed: " + std::to_string(unMapErr));
    }
}

uintptr_t alloc_pinned_numa_ptr(std::size_t size, int node) {
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, \
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
    }

    if (node != -1) {
        // We have a valid numa node number & maximum of 64 numa nodes
        // In machines like Huawei 910C, numa_node can returned as -1, indicating no numa.
        // This is due to the NPUs are connected via switches.
        unsigned long mask = 1UL << node;
        long maxnode = 8 * sizeof(mask);
        int err = mbind(ptr, size, MPOL_BIND, &mask, maxnode, MPOL_MF_MOVE | MPOL_MF_STRICT);
        if (err != 0) {
            munmap(ptr, size);
            throw std::runtime_error(std::string("mbind failed: ") + strerror(errno));
        }
    }

    memset(ptr, 0, size);

    // as before we need to actually save the dev ptr for later reuse,
    // because acl APIs do not allow retrieving register dev ptr
    auto devPtr = register_ptr(ptr, size);
    if (devPtr == nullptr) {
        munmap(ptr, size);
        aclError err = aclrtGetLastError(aclrtLastErrLevel::ACL_RT_THREAD_LEVEL);
        if (err != ACL_SUCCESS) {
            throw std::runtime_error(std::string("unable to register Pinned Numa HostPtr: ") + std::to_string(err));
        } else {
            throw std::runtime_error(std::string("unable to register Pinned Numa HostPtr."));
        }
    }

    return reinterpret_cast<uintptr_t>(ptr);
}

