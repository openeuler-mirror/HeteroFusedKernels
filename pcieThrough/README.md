# PCIEThrough Library

This library provides a torch operator that copy KVCache block with high bandwidth in comparison to memcpy.

> NOTE: Ascend A2 and A3 are supported.

## Install

```bash
pip install -v --no-build-isolation -e .
```

## Usage

```python
import pcie_through

torch.ops.pcie_through.multi_layer_block_transfer(
    a_tensor_thathold_layersptr,
    host_registered_tensor
)
```

For detailed usage see `tests/test_transfer_kernel.py`