import torch
import torch_npu
import oeccl
from oeccl.ops import init_oeccl, oeccl_allgather, oeccl_cleanup, hccl_allgather
import acl
import torch.multiprocessing as mp
import numpy as np
import os
import time
import torch.distributed as dist

relative_tol = 1e-6
absolute_tol = 1e-9
err_tol = 1e-5
ifmsprof = int(os.getenv("ifmsprof", "0"))
test_time = 1000
warm_time = 5
num_processes = 8
numa_map = []  # rank到绑定的numa node的映射
root_rank_list = []  # numa node到对应的root rank的映射


def allgather(func_name, func, rank, parameter_ag, parameter0, parameter1, size, barrier, result_queue):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    a = torch.rand((32768, 4096), dtype=torch.bfloat16).npu(rank)
    b = torch.rand((4096, 4096), dtype=torch.bfloat16).npu(rank)
    elapsed_time = 0
    barrier.wait()
    for i in range(test_time + warm_time):
        _ = torch.matmul(a, b) # 掩盖算子下发的时间
        start.record()
        func(parameter1, parameter_ag)
        end.record()
        _ = torch.matmul(a, b)
        torch.npu.synchronize()
        are_equal = torch.allclose(parameter0, parameter1)
        sum_diff = torch.sum(torch.abs(parameter0 - parameter1))

        if are_equal == False and sum_diff > err_tol:
            mask = torch.ne(parameter0, parameter1)
            print(i, "unequal elements in tensor1:", parameter0[mask])
            print(i, "unequal elements in tensor2:", parameter0[mask])
            barrier.wait()

        prameter1 = torch.zeros_like(parameter1)
        if i >= warm_time:
            elapsed_time += start.elapsed_time(end)
    result_queue.put(elapsed_time)
    barrier.wait()
    elapsed_time = 0
    if rank == 0:
        for i in range(num_processes):
            elapsed_time = elapsed_time + result_queue.get()
        bw = test_time * num_processes * 8 * size * 1000 / 1024 / 1024 / 1024 / elapsed_time
        print(f"{func_name}, are_equal:{are_equal}, elapsed_time:{elapsed_time/8/test_time:.6f} ms bw:{bw:6f} GB/s")


def oeccl_step(rank, parameter_ag, parameter0, parameter1, size, barrier, result_queue):
    dist.all_gather_into_tensor(parameter0, parameter_ag)

    func = hccl_allgather
    allgather("hccl", func, rank, parameter_ag, parameter0, parameter1, size, barrier, result_queue)

    func = oeccl_allgather
    allgather("oeccl-hccl", func, rank, parameter_ag, parameter0, parameter1, size, barrier, result_queue)

    return


def oeccl_test(rank, num_processes, barrier, result_queue):
    torch.manual_seed(rank)
    for s in range(0, 7):
        shape = int(2 ** (s) * 1024 * 1024)
        if rank == 0:
            print(f"SHAPE:{2**(s)*4}MB")
        parameter_ag = torch.randn(shape, dtype=torch.bfloat16).npu(rank)
        parameter0 = torch.zeros(shape * num_processes, dtype=torch.bfloat16).npu(rank)
        parameter1 = torch.zeros(shape * num_processes, dtype=torch.bfloat16).npu(rank)
        are_equal = oeccl_step(rank, parameter_ag, parameter0, parameter1, shape * 4, barrier, result_queue)


def parse_cpulist(cpulist_str):
    # 将像“0-3,5,7-9”这样的CPU列表字符串解析为一组CPU整数
    cpus = set()
    if not cpulist_str:
        return cpus
    for part in cpulist_str.strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                continue
            if end >= start:
                cpus.update(range(start, end + 1))
        else:
            try:
                cpus.add(int(part))
            except ValueError:
                continue
    return cpus


def get_numa_node_cpus(numa_node_id):
    try:
        cpulist_path = f'/sys/devices/system/node/node{numa_node_id}/cpulist'
        if os.path.isfile(cpulist_path):
            with open(cpulist_path, 'r') as f:
                return parse_cpulist(f.read())
    except Exception as e:
        print(f"Warning: Failed to get CPUs for NUMA node {numa_node_id}: {e}")

    # 获取失败则返回空集
    return set()


# 解析NUMA节点映射并执行CPU亲和性绑定
def setup_numa_affinity():
    global root_rank_list
    global numa_map
    numa_map_str = os.getenv('OECCL_NUMA_MAP')
    if not numa_map_str:
        return None

    try:
        numa_map = [int(x) for x in numa_map_str.split(',')]
        if len(numa_map) != num_processes:
            print(f"Warning: OECCL_NUMA_MAP length {len(numa_map)} does not match num_processes {num_processes}")
            return None

        # 记录每个numa节点第一次出现的rank作为root rank
        root_rank_list = []
        seen_numa = set()
        for rank, numa in enumerate(numa_map):
            if numa not in seen_numa:
                root_rank_list.append(rank)
                seen_numa.add(numa)

        # 基于NUMA节点直接获取并绑定CPU
        if numa_map is not None:
            numa_node = numa_map[rank]
            cpuset = get_numa_node_cpus(numa_node)
            if cpuset:
                os.sched_setaffinity(0, cpuset)

        return numa_map

    except (ValueError, AttributeError) as e:
        print(f"Warning: Failed to parse OECCL_NUMA_MAP: {e}")
        return None


def worker(rank, num_processes, barrier, result_queue):

    numa_map = setup_numa_affinity()

    torch_npu.npu.set_device(rank)
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()

    dist.init_process_group(
        backend='hccl', world_size=num_processes, rank=rank, pg_options=options, init_method='tcp://127.0.0.1:60087'
    )

    import torch.distributed.distributed_c10d as dist_c10d

    default_pg = dist_c10d._get_default_group()
    _group = default_pg._get_backend(torch.device("npu"))
    ratio = 8
    hccl_comm = _group.get_hccl_comm(rank)
    if rank in root_rank_list:
        init_oeccl(hccl_comm, True, ratio, numa_map, False)
    barrier.wait()
    if rank not in root_rank_list:
        init_oeccl(hccl_comm, True, ratio, numa_map, False)

    if ifmsprof:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        ) as prof:
            oeccl_test(rank, num_processes, barrier, result_queue)
    else:
        oeccl_test(rank, num_processes, barrier, result_queue)

    barrier.wait()
    torch.npu.synchronize()
    oeccl_cleanup()


if __name__ == "__main__":
    processes = []
    result_queue = mp.Queue()
    barrier = mp.Barrier(num_processes)
    for rank in [0, 1, 2, 3, 4, 5, 6, 7]:
        p = mp.Process(target=worker, args=(rank, num_processes, barrier, result_queue), daemon=True)
        processes.append(p)
    for i_p, p in enumerate(processes):
        p.start()
    for p in processes:
        p.join()
