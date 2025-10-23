# PCIEThrough Library

This library provides PCIEThrough NPU operators for embedding lookup and kvcache block transfer.

> NOTE: Ascend A2 and A3 are supported.

## Install

```bash
pip install -v --no-build-isolation -e .
```

## Usage

### gather
```python
import pcie_through

torch.ops.pcie_through.gather(
    gmEmbed,
    gmEmbedDst, 
    gmIds
)
```

For detailed usage see `tests/test_gather.py`

### multi_layer_block_transfer
```python
import pcie_through

torch.ops.pcie_through.multi_layer_block_transfer(
    a_tensor_thathold_layersptr,
    host_registered_tensor
)
```

For detailed usage see `tests/test_transfer_kernel.py`