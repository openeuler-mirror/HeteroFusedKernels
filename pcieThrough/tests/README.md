# Test benchmark

## File structure
### Embedding Lookup
* test_memcpy.py is for testing embedding lookup via H2D using memcpy via Torch API.
* test_gather.py is for testing embedding lookup via H2D using our pciethrough gather kernel.
### KVCache transfer
* test_transfer_kernel.py is for testing KVCache blocks transfer for H2D, using our transfer kernel.
  * To run the test_transfer_kernel.py use `pytest -vs test_transfer_kernel.py`
  * We have two operators in the the transfer_kernel tests
    * PCIEThrough - directly pull kvcache from the host.
    * FusedMemcpyTransferKernel - the idea is to allow us having a staging block cache in scenario where pciethrough doesn't perform well.


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
