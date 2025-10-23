#pragma once
#include "../launch_utils.h"
#include "../kernel/multi_layer_block_transfer_tiling.h"
#include "aclrtlaunch_kernels.h"

namespace pciethrough {
MultiLayerBlockTransferTilingData GenerateMultiLayerBlockTransferTiling(uint32_t aivNum, int64_t scalarSize, int64_t blockSize, 
                                                                        int64_t numLayers, int64_t numHeads, int64_t headDims);
}