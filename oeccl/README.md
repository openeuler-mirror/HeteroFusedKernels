# OECCl
openEuler Collective Communication Library

## Compile and Install
```shell
python setup.py bdist_wheel
pip install dist/oeccl-xxx.whl
```
## Usage
```python
from oeccl.ops import init_oeccl, oeccl_allgather, oeccl_cleanup, hccl_allgather

numa_map = setup_numa_affinity()
init_oeccl(hccl_comm, is_huge, ratio, numa_map, is_async)

oeccl_allgather(output,input)
```
For detailed usage see `tests/test_oeccl_allgather.py`