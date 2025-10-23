# Test benchmark

## Pre-requisite

- Install the heterogenous package from common following the [common/README.md](../../common/README.md).
- Install the pciethrough package following the [README.md](../README.md)

## File structure
### Embedding Lookup
* test_memcpy.py is for testing embedding lookup via H2D using memcpy via Torch API.
* test_gather.py is for testing embedding lookup via H2D using our pciethrough gather kernel.
### KVCache transfer
* test_transfer_kernel.py is for testing KVCache blocks transfer for H2D, using our transfer kernel.
  * We have two operators in the the transfer_kernel tests
    * PCIEThrough - directly pull kvcache from the host.
    * FusedMemcpyTransferKernel - the idea is to allow us having a staging block cache in scenario where pciethrough doesn't perform well.

#### Testing Steps
Run the following cmd ```pytest -vs test_transfer_kernel.py```


## Test Result
### Embedding Lookup
#### Test Environment
* Linux 5.15
* Cann 8.2.rc1
* Python 3.11
* NPU Ascend910_9392

> Note: We measure the end-to-end host execution time. 

#### Result
| Batch Size | memcpy | gather |
| :-----:| :----: | :----: |
| 16 | 4.3ms(0.03GB/s) | 0.118ms(1.39GB/s) |
| 32 | 4.6ms(0.07GB/s) | 0.115ms(2.83GB/s) |
| 64 | 4.7ms(0.13GB/s) | 0.116ms(5.65GB/s) |
| 128 | 4.8ms(0.27GB/s) | 0.115ms(11.30GB/s) |
| 512 | 5.4ms(0.96GB/s) | 0.166ms(31.67GB/s) |
| 1024 | 5.6ms(1.85GB/s) | 0.306ms(34.26GB/s) |

### KVCache Transfer

#### Test Environment
* Linux 5.15
* Cann 8.2.rc1
* Python 3.11
* NPU Ascend910_9392

> Note: We measure the per op bandwidth.

#### Result

The below result is tested in int8 dtype using 4 AIV.

| KV Block [blockSize, layers, heads, headdim]| PCIEThrough | FusedMemcpyKernel |
| :-----:| :----: | :----: |
| [1024, 32, 1, 128] | 9.65 GB/s  | 15.50 GB/s |
| [1024, 32, 8, 128] | 26.52 GB/s | 19.49 GB/s |
| [1024, 64, 1, 128] | 22.03 GB/s | 20.92 GB/s |
| [1024, 64, 8, 128] | 34.11 GB/s | 22.55 GB/s |
