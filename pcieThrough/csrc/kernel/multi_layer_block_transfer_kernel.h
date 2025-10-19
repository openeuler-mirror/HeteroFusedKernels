#ifndef MULTI_LAYER_BLOCK_TRANSFER_KERNEL
#define MULTI_LAYER_BLOCK_TRANSFER_KERNEL

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "multi_layer_block_transfer_tiling.h"

namespace KVTransfer {

template<typename scalar_t> class MultiLayerBlockTransfer {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;
public:
    __aicore__ inline MultiLayerBlockTransfer(AscendC::TPipe* pipe) {
        this->pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR deviceBlockPtrs, GM_ADDR hostBlockCache,
                                __gm__ MultiLayerBlockTransferTilingData* tilingGM) {
        this->numLayers_ = static_cast<int32_t>(tilingGM->numLayers);
        this->numHeads_ = static_cast<int32_t>(tilingGM->numHeads);
        this->headDims_ = static_cast<int32_t>(tilingGM->headDims);
        this->blockSize_ = static_cast<int32_t>(tilingGM->blockSize);
        this->layerBufferSize_ = static_cast<int64_t>(tilingGM->layerBufferSize);
        this->cacheBlockBufferSize_ = static_cast<int64_t>(tilingGM->cacheBlockBufferSize);
        this->tokensPerCore_ = static_cast<int32_t>(tilingGM->tokensPerCore);
        this->tokensPerInnerLoop_ = static_cast<int32_t>(tilingGM->tokensPerInnerLoop);
        this->localTokensBufferSizeUB_ = static_cast<int64_t>(tilingGM->localTokensBufferSizeUB);

        this->hostBlockCache_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(hostBlockCache), this->cacheBlockBufferSize_);
        this->deviceBlockLayersPtr_ = deviceBlockPtrs;

        this->stageOneMultiTokensCopyParams_.blockLen = static_cast<uint16_t>(tilingGM->stageOneCopyParams.perTokenCopyBlockLen);
        this->stageOneMultiTokensCopyParams_.srcStride = 0;
        this->stageOneMultiTokensCopyParams_.dstStride = 0;
        this->stageTwoStridedTokensCopyParams_.blockLen = static_cast<uint16_t>(tilingGM->stageTwoCopyParams.perHeadBlockLen);
        this->stageTwoStridedTokensCopyParams_.srcStride = static_cast<uint16_t>(tilingGM->stageTwoCopyParams.bufferStride);
        this->stageTwoStridedTokensCopyParams_.dstStride = 0;
       
        this->pipe_->InitBuffer(this->localBufferQue, DOUBLE_BUFFER, this->localTokensBufferSizeUB_);
    }

    __aicore__ inline void blockTransposeCopy(int64_t tokenIdx, int64_t actualTokensPerInner) {
        __gm__ uint8_t * __gm__ *deviceBlockLayersPtrs = reinterpret_cast<__gm__ uint8_t* __gm__ *>(this->deviceBlockLayersPtr_);

        local_scalar_t actualBufferTensor = this->localBufferQue.template AllocTensor<scalar_t>();

        int64_t localStartTensorIdx = tokenIdx * this->numLayers_ * this->numHeads_ * this->headDims_;

        this->stageOneMultiTokensCopyParams_.blockCount = actualTokensPerInner;
        this->stageTwoStridedTokensCopyParams_.blockCount = actualTokensPerInner;

        AscendC::DataCopy(actualBufferTensor, this->hostBlockCache_[localStartTensorIdx], this->stageOneMultiTokensCopyParams_);

        this->localBufferQue.EnQue(actualBufferTensor);
        actualBufferTensor = this->localBufferQue.template DeQue<scalar_t>();

        int64_t srcIdx = 0;
        int64_t dstIdx = 0;
        for (int64_t layerIdx = 0; layerIdx < this->numLayers_; layerIdx++) {
            this->deviceBlockCache_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t*>(deviceBlockLayersPtrs[layerIdx]),
                                                    this->layerBufferSize_);
            for (int64_t headIdx = 0; headIdx < this->numHeads_; headIdx++) {
                srcIdx = (layerIdx * this->numHeads_ * this->headDims_) +
                         (headIdx * this->headDims_);
                dstIdx = (headIdx * this->blockSize_ * this->headDims_) +
                         (tokenIdx * this->headDims_);
                AscendC::DataCopy(this->deviceBlockCache_[dstIdx], actualBufferTensor[srcIdx], this->stageTwoStridedTokensCopyParams_);
            }
        }

        this->localBufferQue.FreeTensor(actualBufferTensor);
    }

    __aicore__ inline void Process() {
        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t startTokenIdx = coreIdx * this->tokensPerCore_;
        int32_t endTokenIdx = min(this->blockSize_, startTokenIdx + this->tokensPerCore_);
        for (int32_t tokenIdx = startTokenIdx; tokenIdx < endTokenIdx; tokenIdx += this->tokensPerInnerLoop_) {
            int32_t actualTokensPerInner = min(this->tokensPerInnerLoop_, endTokenIdx - tokenIdx);
            this->blockTransposeCopy(tokenIdx, actualTokensPerInner);
        }
    }

private:
    constexpr static uint32_t DOUBLE_BUFFER = 2;
    AscendC::TPipe* pipe_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, DOUBLE_BUFFER> localBufferQue;

    
    // a list of ptrs (layers) where each ptr hold a cache in the shape of
    // [Heads, BlockSize, HeadDims]
    AscendC::GlobalTensor<scalar_t> deviceBlockCache_;

    // [blockSize, Layers, heads, headDims]
    AscendC::GlobalTensor<scalar_t> hostBlockCache_;

    AscendC::DataCopyParams stageOneMultiTokensCopyParams_;
    AscendC::DataCopyParams stageTwoStridedTokensCopyParams_;

    int32_t numLayers_;
    int32_t numHeads_;
    int32_t headDims_;
    int32_t tokensPerInnerLoop_;
    int32_t blockSize_;
    int32_t tokensPerCore_;
    int64_t layerBufferSize_;
    int64_t cacheBlockBufferSize_;
    int64_t localTokensBufferSizeUB_;

    __gm__ uint8_t* deviceBlockLayersPtr_;

};
}
#endif // MULTI_LAYER_BLOCK_TRANSFER_KERNEL