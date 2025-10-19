#include "kernel_operator.h"
#include "multi_layer_block_transfer_kernel.h"
#include "multi_layer_block_transfer_tiling.h"
#include "types.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void multi_layer_block_transfer(GM_ADDR devBlockCachePtrs, GM_ADDR hostBlockCache, GM_ADDR tilingData) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    auto tilingGM = reinterpret_cast<__gm__ MultiLayerBlockTransferTilingData*>(tilingData);
    if (tilingGM->datatype == pciethrough::DATATYPE::BF16) {
        KVTransfer::MultiLayerBlockTransfer<bfloat16_t> op(&pipe);
        op.Init(devBlockCachePtrs, hostBlockCache, tilingGM);
        op.Process();
    } else if (tilingGM->datatype == pciethrough::DATATYPE::FLOAT) {
        KVTransfer::MultiLayerBlockTransfer<float32_t> op(&pipe);
        op.Init(devBlockCachePtrs, hostBlockCache, tilingGM);
        op.Process();
    } else if (tilingGM->datatype == pciethrough::DATATYPE::HALF) {
        KVTransfer::MultiLayerBlockTransfer<float16_t> op(&pipe);
        op.Init(devBlockCachePtrs, hostBlockCache, tilingGM);
        op.Process();
    } else if (tilingGM->datatype == pciethrough::DATATYPE::INT8_T) {
        KVTransfer::MultiLayerBlockTransfer<int8_t> op(&pipe);
        op.Init(devBlockCachePtrs, hostBlockCache, tilingGM);
        op.Process();
    }
}