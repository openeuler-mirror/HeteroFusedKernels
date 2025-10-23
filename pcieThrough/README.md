# PCIEThrough Library

This library provides PCIEThrough NPU operators for embedding lookup and kvcache block transfer.

> NOTE: Ascend A2 and A3 are supported.

## Install

```bash
pip install -v --no-build-isolation -e .
```

## Usage

### gather operator
```python
import pcie_through

torch.ops.pcie_through.gather(
    gmEmbed,
    gmEmbedDst, 
    gmIds
)
```

For detailed usage see `tests/test_gather.py`

### KV Transfer operator

```python
import pcie_through
# no staging buffer
torch.ops.pcie_through.multi_layer_block_transfer(
    a_device_tensor_thathold_layersptr,
    host_registered_tensor,
    aiv_num
)

# with staging buffer
torch.ops.pcie_through.fused_memcpy_multi_layer_block_transfer(
    a_device_tensor_thathold_layersptr,
    host_tensor,
    staging_device_tensor,
    aiv_num
)
```

For detailed usage see `tests/test_transfer_kernel.py`