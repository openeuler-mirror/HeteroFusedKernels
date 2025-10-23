# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List
import random

import time
import torch
import torch_npu

from heterofusedkernels import memory

import pcie_through

def verify(t1, t2, ids):
    ids.cpu()
    t1.cpu()
    t2.cpu()
    ok = True
    for i in range(ids.shape[0]):
        index = ids[i]
        line = t1[index].cpu()
        line2 = t2[i].cpu()
        if not torch.allclose(line, line2):
            ok = False
            print(i)
            print(line)
            print(line2)
            break
    return ok  


def benchmark():
    torch.npu.set_device(3)

    embed_size = 4000000
    embed_dim = 512
    total_size = embed_size * embed_dim * 2
    dtype = torch.float16

    embed = memory.alloc_numa_pinned_tensor(total_size)
    embed = embed.view(dtype).view([embed_size, embed_dim])

    step = 1
    for bs in range(8, 2048, 8):
        bs *= step
        run_count = 1000    
        input_ids = torch.randint(0, embed_size, [run_count, bs], dtype=torch.int64).npu()
        embedDst = torch.empty([bs, embed_dim], dtype=dtype).npu()
        torch.npu.synchronize()
        a = time.time()
        for run in range(run_count):
            torch.ops.pcie_through.gather(
                    embed,
                    embedDst,
                    input_ids[run, :]
                )
        torch.npu.synchronize()
        b = time.time()
        ok = verify(embed, embedDst, input_ids[run,:])
        print("Kernel test {} time: {} - bandwidth: {}{}".format(bs, (b - a) / run_count * 10 * 1000, bs * run_count * embed_dim * 2 * 10/ ((b - a) * 10 * 1000000000),ok)) 

        if not ok:
            break


if __name__ == '__main__':
    benchmark()