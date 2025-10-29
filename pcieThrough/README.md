# PCIEThrough module

This module provides PCIEThrough Ascend NPU operators for zero-copy embeddings gathering and KV-cache block transfer.

## Install
> Note: this module relies on the common module of HeteroFusedKernels. Please make sure you have that module already installed. 
```bash
pip install -v --no-build-isolation -e .
```

## APIs

This module exposes three Ascend NPU operators to Python users 

```
# Takes the tensor `embed` as input and gathers specific rows of the input tensor into the contiguous tensor `embed_dst`. 
# The rows gathered are the ones indexed by the IDs as specified in the 1-dimensional input tensor `input_ids` . 
# Both `embed` and `input_ids` can be either host registered or on device memory, while `embed_dst` must be a tensor in the device memory. 
torch.ops.pcie_through.gather(
    embed, 
    embed_dst, 
    input_ids)

# Copies a KV-Cache block `srcBlock` from the host memory to the device memory rearranging its memory layout.
# `dstPtrs`: a 1D tensor on NPU that holds a list of pointers, where each pointer points to a tensor that is a layer of
# the KVCache block residing in the device HBM. The expected shape is [ Layers ], and each layer pointer points to a 
# KVCache block in the shape of [ Heads, BlockSize, HeadDim ].
# `srcBlock`: the host KVCache block, that is a hostRegistered tensor that resides in the host main memory and 
# has shape [BlockSize, Layers, Heads, HeadDims].
# (Optional) `aivNum`: the number of AIVector cores used for the zero-copy transfer.
torch.ops.pcie_through.multi_layer_block_transfer(
    dstPtrs,
    srcBlock,
    aivNum )

# Copies a KV-Cache block `srcBlock` from the host memory to the device memory rearranging its memory layout. Performs 
# an intermediate memcpy from host to a device stagingBlock before rearranging the KV Cache in the appropriate layout.
# `dstPtrs`: a 1D tensor on NPU that holds a list of pointers, where each pointer points to a tensor that is a layer 
# of the KVCache block residing in the device HBM. The expected shape is [ Layers ], and each layer pointer points to 
# a KVCache block in the shape of [ Heads, BlockSize, HeadDim ].
# `srcBlock`: the host KVCache block, that is a hostRegistered tensor that resides in the host main memory and has
# shape [BlockSize, Layers, Heads, HeadDims].
# `stagingBlockCache`:  a tensor in the HBM with the same shape as the srcBlock [BlockSize, Layers, Heads, HeadDim]. 
# (Optional) `aivNum`: the number of AIVector cores used for the zero-copy transfer.
torch.ops.pcie_through.fused_memcpy_multi_layer_block_transfer(
    device_block_ptr,
    host_block_cache,
    staging_block_cache,
    aiv_blocks)


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

## Compatibility

| Component | Tested Versions/Details |
| :--- | :--- |
| `torch` and `torch_npu` | v2.5.1-v2.7.1 |
| `CANN stack` | 8.2.RC1 |
| `Ascend driver` | v25.0.rc1.1 |
| `Hardware Setup` | A2 910B, A3 910C |