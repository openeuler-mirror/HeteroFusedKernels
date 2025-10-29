# HeteroFusedKernels Common

We provide common utils for HeteroFusedKernels repo in this module. Such utils include the management of device-accessible host memory and the related device communication management.

## Install

```bash
pip install -v --no-build-isolation -e .
```
## APIs
This module exposes three main API functions to Python users:
```
# Allocates on the NUMA node of the current device a pinned tensor
# of total size (in bytes) size_bytes. The memory is registered to the device.
host_registered_tensor = memory.alloc_numa_pinned_tensor(size_bytes)

# Registers an existing tensor to the current device
memory.host_register(tensor_buffer)

# Returns the device pointer for a host pointer that has been previously registered.
device_ptr = memory.get_device_ptr(host_ptr)
```

## Usage
Example for NUMA-aware torch tensor allocation using `alloc_numa_pinned_tensor`.

```
from heterofusedkernels import memory             
torch.npu.set_device(0)
numberOfElements = 400000
sizeOfElements = 512
dtype = torch.float16

total_size = numberOfElements * sizeOfElements * dtype.itemsize 
#2 bytes per element because of float16

pinnedTensor = memory.alloc_numa_pinned_tensor(total_size)
pinnedTensor = pinnedTensor.view(dtype).view([numberOfElements, sizeOfElements])
# Now can access the pinned tensor as a normal tensor
# For example...
print(memory.get_device_ptr(pinnedTensor.data_ptr()))
```

Example usage of registration of pre-existing torch tensor as device-accessible host memory.
```
from heterofusedkernels import memory             
torch.npu.set_device(0)
numberOfElements = 400000
sizeOfElements = 512
dtype = torch.float16

pinnedTensor = torch.empty([numberOfElements, sizeOfElements], dtype=torch.float16)
memory.host_register(pinnedTensor)  # <--- registering an existing tensor
# Now can access the pinned tensor as a normal tensor
# For example...
print(memory.get_device_ptr(pinnedTensor.data_ptr()))
```

## Compatibility

| Component | Tested Versions/Details |
| :--- | :--- |
| `torch` and `torch_npu` | v2.5.1-v2.7.1 |
| `CANN stack` | 8.2.RC1 |
| `Ascend driver` | v25.0.rc1.1 |
| `Hardware Setup` | A2 910B, A3 910C |