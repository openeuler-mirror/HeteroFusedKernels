# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import random

# Third Party
import pytest
import torch
import torch_npu

from heterofusedkernels import memory

import pcie_through

@pytest.mark.parametrize("block_size", [1024])
@pytest.mark.parametrize("layers", [1, 32, 64])
@pytest.mark.parametrize("heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_blocks", [1, 8, 16]) # how many blocks of tokens to copy
@pytest.mark.parametrize("simultaneous_copy", [2]) # double buffer
@pytest.mark.parametrize("use_fused_copy", [False, True]) # use the fused memcpy for staging
def test_multi_layer_block_kv_transfer(block_size, layers, heads, head_dim, num_blocks, simultaneous_copy, use_fused_copy):
    device = "npu"
    dtype = torch.int8
    total_host_blocks = 1000
    total_dev_blocks = 64
    torch.npu.set_device(0)
    host_shape = (total_host_blocks, block_size, layers, heads, head_dim)
    device_shape = (total_dev_blocks, layers, heads, block_size, head_dim)
    total_host_size = total_host_blocks * block_size * layers * heads * head_dim * dtype.itemsize
    host_blocks_cache = memory.alloc_numa_pinned_tensor(total_host_size)
    host_blocks_cache[:] = 2
    host_blocks_cache = host_blocks_cache.view(dtype).view(host_shape)
    host_blocks_cache = list(host_blocks_cache.unbind(0))
            
    device_blocks_cache = torch.empty(device_shape, dtype=dtype, device=device)
    device_blocks_cache[:] = 1
    device_blocks_cache = list(device_blocks_cache.unbind(0))

    staging_blocks_cache = []

    # obtain device ptr
    device_block_ptrs = []
    for tensor in device_blocks_cache:
        device_block_ptr = torch.empty(layers, dtype=torch.int64, device="cpu")
        for i in range(layers):
            device_block_ptr[i] = tensor[i].data_ptr()
        device_block_ptrs.append(device_block_ptr.to(device))


    streams = [torch.npu.Stream() for _ in range(simultaneous_copy)]
    start_evs = [torch.npu.Event(enable_timing=True) for _ in range(num_blocks)]
    end_evs = [torch.npu.Event(enable_timing=True) for _ in range(num_blocks)]
    
    if use_fused_copy:
        for _ in range(simultaneous_copy):
            staging_blocks_cache.append(torch.empty([block_size, layers, heads, head_dim], dtype=dtype, device=device))
    
    for i in range(num_blocks):
        with torch.npu.stream(streams[i % simultaneous_copy]):
            start_evs[i].record()
            if use_fused_copy:
                torch.ops.pcie_through.fused_memcpy_multi_layer_block_transfer(
                    device_block_ptrs[i],
                    host_blocks_cache[i],
                    staging_blocks_cache[i % simultaneous_copy],
                    4)
            else:
                torch.ops.pcie_through.multi_layer_block_transfer(
                    device_block_ptrs[i],
                    host_blocks_cache[i],
                    4)
            end_evs[i].record()
    
    for stream in streams:
        stream.synchronize()

    expected_host_block_caches = [torch.empty_like(device_blocks_cache[0], device="cpu") for _ in range(num_blocks)]
    for i in range(num_blocks):
        expected_host_block_caches[i].copy_(device_blocks_cache[i])
    for i in range(num_blocks):
        transposed_block = expected_host_block_caches[i].permute(2, 0, 1, 3)
        assert torch.allclose(host_blocks_cache[i], transposed_block), \
            f"Block {i} transfer failed - \n Host: {host_blocks_cache[i][0][0]} \n - Dev: {transposed_block[i][0][0]}."

    # calculate bandwidths in GB/s
    total_size = block_size * layers * heads * head_dim * dtype.itemsize
    time_in_ms = sum([start_ev.elapsed_time(end_ev) for start_ev, end_ev in zip(start_evs, end_evs)]) / num_blocks
    print(f"Bandwidths: {total_size/(time_in_ms*1e6)} GB/s")

