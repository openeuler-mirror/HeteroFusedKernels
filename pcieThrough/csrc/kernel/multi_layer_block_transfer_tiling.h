#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "types.h"

struct StageOneCopyParams {
    uint64_t perTokenCopyBlockLen;
};

struct StageTwoCopyParams {
    uint64_t perHeadBlockLen;
    uint64_t bufferStride;
};

struct MultiLayerBlockTransferTilingData {
    pciethrough::DATATYPE datatype;
    uint32_t numLayers;
    uint32_t numHeads;
    uint32_t headDims;
    uint32_t blockSize;
    uint64_t layerBufferSize;
    uint64_t cacheBlockBufferSize;
    uint32_t tokensPerCore;
    uint32_t tokensPerInnerLoop;
    uint64_t localTokensBufferSizeUB;
    StageOneCopyParams stageOneCopyParams;
    StageTwoCopyParams stageTwoCopyParams;
};