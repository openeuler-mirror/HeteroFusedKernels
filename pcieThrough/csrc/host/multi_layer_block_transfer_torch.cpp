

#include "securec.h"
#include "tiling/multi_layer_block_transfer.h"
#include <ATen/ATen.h>
#include "managed_mem.h"
#include "acl/acl.h"

namespace py = pybind11;

namespace pciethrough {

at::Tensor mvTilingToDevice(MultiLayerBlockTransferTilingData *tilingData, size_t tilingDataSize) {
    auto buffer = at::empty({static_cast<int64_t>(tilingDataSize)}, at::kByte);
    auto err = memcpy_s(buffer.data_ptr(), tilingDataSize, reinterpret_cast<void*>(tilingData), tilingDataSize);
    TORCH_CHECK(err == EOK, "memcpy_s failed");
    auto tilingDeviceTensor = CopyTensorHostToDevice(buffer);
    return tilingDeviceTensor;
}

DATATYPE fromAtenScalarType(at::ScalarType scalarType) {
    switch (scalarType) {
        case at::ScalarType::Float:
            return DATATYPE::FLOAT;
        case at::ScalarType::Half:
            return DATATYPE::HALF;
        case at::ScalarType::BFloat16:
            return DATATYPE::BF16;
        case at::ScalarType::Char:
            return DATATYPE::INT8_T;
        default:
            auto p = c10::getDtypeNames(scalarType);
            TORCH_CHECK(false, "ScalarType " + p.first + " not supported.")
    }
}


// Potentially need to modify this for graph capturing due to no explicit output (mutable tensor) for torch
void multi_layer_block_transfer(const at::Tensor &deviceBlockCachePtrs, const at::Tensor &hostBlockCache,
                                c10::optional<int64_t> aiv = c10::nullopt) {
    uint32_t aivNum = 2;
    if (aiv.has_value()) {
        aivNum = aiv.value();
    }

    size_t tilingDataSize = sizeof(MultiLayerBlockTransferTilingData);
    auto scalarSize = hostBlockCache.element_size();
    auto blockSize = hostBlockCache.size(0);
    auto numLayers = hostBlockCache.size(1);
    auto numHeads = hostBlockCache.size(2);
    auto headDims = hostBlockCache.size(3);
    at::ScalarType scalarType = hostBlockCache.scalar_type();
    auto tilingData = GenerateMultiLayerBlockTransferTiling(aivNum, scalarSize, blockSize, numLayers,
                                                            numHeads, headDims);
    tilingData.datatype = fromAtenScalarType(scalarType);
    at::Tensor tilingTensor = mvTilingToDevice(&tilingData, tilingDataSize);
    auto& hmm = managed_memory::HostRegisteredMemoryManager::GetInstance();
    void* hostBlockDevPtr = hmm.getDevicePtr(hostBlockCache.data_ptr());
    EXEC_KERNEL_CMD(multi_layer_block_transfer, aivNum, deviceBlockCachePtrs, hostBlockDevPtr, tilingTensor);
}

void fused_memcpy_multi_layer_block_transfer(const at::Tensor& deviceBlockCachePtrs, const at::Tensor &hostBlockCache,
                                             const at::Tensor& stagingBlockCache, c10::optional<int64_t> aiv = c10::nullopt) {
    uint32_t aivNum = 2;
    if (aiv.has_value()) {
        aivNum = aiv.value();
    }

    size_t tilingDataSize = sizeof(MultiLayerBlockTransferTilingData);
    auto scalarSize = hostBlockCache.element_size();
    auto blockSize = hostBlockCache.size(0);
    auto numLayers = hostBlockCache.size(1);
    auto numHeads = hostBlockCache.size(2);
    auto headDims = hostBlockCache.size(3);
    at::ScalarType scalarType = hostBlockCache.scalar_type();
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    auto ret = aclrtMemcpyAsync(stagingBlockCache.data_ptr(), static_cast<size_t>(stagingBlockCache.nbytes()), 
                                hostBlockCache.data_ptr(), static_cast<size_t>(hostBlockCache.nbytes()), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "Unable to copy the kv block to the staging block with Memcpy Async");
    auto tilingData = GenerateMultiLayerBlockTransferTiling(aivNum, scalarSize, blockSize, numLayers,
                                                            numHeads, headDims);
    tilingData.datatype = fromAtenScalarType(scalarType);
    at::Tensor tilingTensor = mvTilingToDevice(&tilingData, tilingDataSize);
    EXEC_KERNEL_CMD(multi_layer_block_transfer, aivNum, deviceBlockCachePtrs, stagingBlockCache, tilingTensor);
}
} // namespace pciethrough


namespace {
TORCH_LIBRARY_FRAGMENT(pcie_through, m)
{
    m.def("multi_layer_block_transfer(Tensor deviceBlockCachePtrs, Tensor hostBlockCache, int? aiv=None) -> ()");
    m.def("fused_memcpy_multi_layer_block_transfer(Tensor deviceBlockCachePtrs, Tensor hostBlockCache, Tensor stagingBlockCache, int? aiv=None) -> ()");
}
}

namespace{
TORCH_LIBRARY_IMPL(pcie_through, PrivateUse1, m) {
    m.impl("multi_layer_block_transfer", TORCH_FN(pciethrough::multi_layer_block_transfer));
    m.impl("fused_memcpy_multi_layer_block_transfer", TORCH_FN(pciethrough::fused_memcpy_multi_layer_block_transfer));
}
}