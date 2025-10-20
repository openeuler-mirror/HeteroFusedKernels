#include "../../kernel/multi_layer_block_transfer_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <iostream>
#include "acl/acl.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include "../../kernel/types.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

namespace pciethrough{
// each copy iter is 32B
const uint32_t ASCEND_BLOCKCOPYLEN = 32;
const uint32_t BUFF_NUM = 2;

MultiLayerBlockTransferTilingData GenerateMultiLayerBlockTransferTiling(uint32_t aivNum, int64_t scalarSize, int64_t blockSize, 
                                                                        int64_t numLayers, int64_t numHeads, int64_t headDims) {
    MultiLayerBlockTransferTilingData tiling;
    const char* socName = aclrtGetSocName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tiling.numLayers = static_cast<uint32_t>(numLayers);
    tiling.numHeads = static_cast<uint32_t>(numHeads);
    tiling.headDims = static_cast<uint32_t>(headDims);
    tiling.blockSize = static_cast<uint32_t>(blockSize);
    int64_t cacheBlockBufferSize = scalarSize * blockSize * numLayers * numHeads * headDims;
    tiling.cacheBlockBufferSize = static_cast<uint64_t>(cacheBlockBufferSize);
    int64_t layerBufferSize = blockSize * numHeads * headDims * scalarSize;
    tiling.layerBufferSize = static_cast<uint64_t>(layerBufferSize);
    uint32_t tokensPerCore = (blockSize + aivNum - 1) / aivNum;
    tiling.tokensPerCore = tokensPerCore;
    // caclulate tokensPerInnerLoop due to UBSize contraints
    uint64_t basePerTokenBuffSize = BUFF_NUM * numLayers * numHeads * headDims * scalarSize;
    std::string errStr = "Per Token Buffer Size: " + std::to_string(basePerTokenBuffSize) + " is greater than UBSize: " + std::to_string(ubSize);
    TORCH_CHECK(ubSize > basePerTokenBuffSize,  errStr + " The kv cache per token is too large, please contact us.");
    const uint64_t safetyMargin = 256;
    uint32_t maxTokensPerLoopOnChipUB = (ubSize - safetyMargin) / basePerTokenBuffSize;
    maxTokensPerLoopOnChipUB = std::min(tokensPerCore, maxTokensPerLoopOnChipUB);
    TORCH_CHECK(maxTokensPerLoopOnChipUB > 0, "Max tokens per loop on chip is less than or equal to 0. Please contact us. ")
    TORCH_CHECK(ubSize > (basePerTokenBuffSize * maxTokensPerLoopOnChipUB), "Per Token UB size is greater than UB. Please contact us.")
    tiling.tokensPerInnerLoop = maxTokensPerLoopOnChipUB;
    tiling.localTokensBufferSizeUB = numLayers * numHeads * headDims * scalarSize * maxTokensPerLoopOnChipUB;

    // calculate data copy blocklen
    uint64_t perTokenBuffer = numLayers * numHeads * headDims * scalarSize;
    tiling.stageOneCopyParams.perTokenCopyBlockLen = static_cast<uint64_t>(perTokenBuffer / ASCEND_BLOCKCOPYLEN);
    
    uint64_t perHeadBlockLen = (headDims * scalarSize) / ASCEND_BLOCKCOPYLEN;
    tiling.stageTwoCopyParams.perHeadBlockLen = perHeadBlockLen;
    tiling.stageTwoCopyParams.bufferStride = (numLayers * numHeads - 1) * headDims * scalarSize / ASCEND_BLOCKCOPYLEN;
    return tiling;
}
} // namespace pciethrough
