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
| 16 | 4.36ms(0.038GB/s) | 0.114ms(1.43GB/s) | 0.011ms(15.82GB/s) |
| 32 | 4.42ms(0.074GB/s) | 0.108ms(3.04GB/s) | 0.014ms(22.99GB/s) |
| 64 | 4.61ms(0.14GB/s) | 0.108ms(6.09GB/s) | 0.026ms(25.22GB/s) |
| 128 | 4.79ms(0.27GB/s) | 0.107ms(12.20GB/s) | 0.049ms(26.96GB/s) |
| 256 | 5.00ms(0.52GB/s) | 0.121ms(21.59GB/s) | 0.049ms(26.96GB/s) |
| 512 | 5.27ms(0.99GB/s) | 0.211ms(24.11GB/s) | 0.181ms(28.93GB/s) |
| 1024 | 5.52ms(1.90GB/s) | 0.404ms(25.98GB/s) | 0.357ms(29.29GB/s) |

#### Test instruction
```bash
cd {$projectpath}/common
pip install -v --no-build-isolation -e .

cd {$projectpath}/pcie_through
pip install -v --no-build-isolation -e .

cd {$projectpath}/pcie_through/tests
numactl --membind=0 --cpunodebind=0 python test_{$testname}.py
```