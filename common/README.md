# HeteroFusedKernels Common

We provide common utils for HeteroFusedKernels in this repo.

## Install

```bash
pip install -v --no-build-isolation -e .
```

## Usage
Example for numa-aware torch tensor buffer

```
from heterofusedkernels import memory             
>>> torch.npu.set_device(7)
>>> a = memory.alloc_numa_pinned_tensor(7*1024*1024*1024)
>>> a
tensor([0, 0, 0,  ..., 0, 0, 0], dtype=torch.int8)
```
