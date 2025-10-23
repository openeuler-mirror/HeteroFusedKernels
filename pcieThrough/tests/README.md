# Test benchmark

## File structure
### Embedding Lookup
* test_memcpy.py is for testing embedding lookup via H2D using memcpy via Torch API (simulating copy 10 times).
* test_gather.py is for testing embedding lookup via H2D using our pciethrough gather kernel (simulating copy 10 times).
* test_gather_combine.py is for testing embedding lookup via H2D using our pciethrough gather kernel (copying ten sets of data at once).
### KVCache transfer
* test_transfer_kernel.py is for testing KVCache blocks transfer for H2D, using our pciethrough transfer kernel.

## Test Result
### Embedding Lookup
#### Test Environment
* Linux 5.15
* Cann 8.2.rc1
* Python 3.11
* NPU Ascend910_9392

> Note: We measure the end-to-end host execution time. 

#### Result
| Batch Size | memcpy | gather | gather_combine |
| :-----:| :----: | :----: | :----: |
| 16 | 4.3ms(0.03GB/s) | 0.118ms(1.39GB/s) | 0.011ms(15.82GB/s) |
| 32 | 4.6ms(0.07GB/s) | 0.115ms(2.83GB/s) | 0.014ms(22.99GB/s) |
| 64 | 4.7ms(0.13GB/s) | 0.116ms(5.65GB/s) | 0.026ms(25.22GB/s) |
| 128 | 4.8ms(0.27GB/s) | 0.115ms(11.30GB/s) | 0.049ms(26.96GB/s) |
| 512 | 5.4ms(0.96GB/s) | 0.166ms(31.67GB/s) | 0.181ms(28.93GB/s) |
| 1024 | 5.6ms(1.85GB/s) | 0.306ms(34.26GB/s) | 0.357ms(29.29GB/s) |
