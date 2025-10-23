/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GATHER_KERNEL
#define GATHER_KERNEL

#include "kernel_operator.h"
#include <stdio.h>
#include "types.h"

namespace Gather {

template <typename scalar_t, typename slot_t>
class MteCopy {
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;
    using local_slot_t = AscendC::LocalTensor<slot_t>;

public:
    __aicore__ inline MteCopy()
    {}

    __aicore__ inline void init(__gm__ uint8_t *gmEmbed, __gm__ uint8_t *gmEmbedDst, __gm__ uint8_t *gmIds,
        const size_t numIds, const size_t embedSizeBytes, int32_t blockNum, int32_t blockIdx, AscendC::TPipe *pipe)
    {
        // UB is 192KB
        // [Ids Size e/g/  numTokens * int8]
        // [localAccumlateBuffer embedSizeBytes*buffer]
        // [local variables .... ]

#define UB_SIZE_SAFE 192 * 1024  // Reserve some space for local vars

        uint32_t numIdsBase = numIds / blockNum;
        uint32_t numIdsRemain = numIds % blockNum;
        uint16_t bufferNum = 2;

        this->pipe_ = pipe;
        this->totalNumIds_ = numIds;
        this->n_ids_thisBlock_ = numIdsBase + (blockIdx < numIdsRemain ? 1 : 0);
        this->first_id_thisBlock_ = blockIdx * numIdsBase + (blockIdx < numIdsRemain ? blockIdx : numIdsRemain);
        this->n_scalarsInEmbedding_ = embedSizeBytes / sizeof(scalar_t);

        // n_embeddingsPerRound_ is floor(UB_SIZE_SAFE/ (buffernum * (embedSizeBytes + sizeof(slot_t))))
        // in other words, to compute the embeddings per round, we say that one embedding requires embedSizeBytes plus
        // the space for its id, multiplied by the number of buffer to keep alive at the same time
        this->n_embeddingsPerRound_ = UB_SIZE_SAFE / (bufferNum * (embedSizeBytes + sizeof(slot_t)));
        // Following line makes sure that n_embeddingsPerRound_ is multiple of 32/Sizeof(slot_t) to always issue
        // DataCopy 32 bytes alligned. For example, for slot_t int64 (8 Bytes), we divide and multiply
        // n_embeddingsPerRound_ by 4
        this->n_embeddingsPerRound_ = this->n_embeddingsPerRound_ / (32 / sizeof(slot_t)) * (32 / sizeof(slot_t));
        this->n_rounds_ = (this->n_ids_thisBlock_ + this->n_embeddingsPerRound_ - 1) /
                          this->n_embeddingsPerRound_;  // ceil(idsPerCore / idsPerRound)

        this->gmEmbed_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(gmEmbed));
        this->gmEmbedDst_.SetGlobalBuffer(reinterpret_cast<__gm__ scalar_t *>(gmEmbedDst));
        this->gmIds_.SetGlobalBuffer(reinterpret_cast<__gm__ slot_t *>(gmIds));

        this->pipe_->InitBuffer(this->embeddingQue_,
            bufferNum,
            this->n_embeddingsPerRound_ * this->n_scalarsInEmbedding_ * sizeof(scalar_t));
        this->pipe_->InitBuffer(this->idxQue_, bufferNum, this->n_embeddingsPerRound_ * sizeof(slot_t));
    }

    __aicore__ inline void reset()
    {}

    __aicore__ inline void LoadStoreFunc()
    {
        // local_scalar_t embeddingTensor = this->embeddingQue_.template AllocTensor<scalar_t>();
        // int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        // The embeddings may not fit all in UB -- loop through multiple rounds of copies in UB
        for (uint32_t r = 0; r < this->n_rounds_; r++) {
            uint32_t n_embeddings_round = this->n_embeddingsPerRound_;
            uint32_t n_scalars_round = n_embeddings_round * this->n_scalarsInEmbedding_;
            if (r == this->n_rounds_ - 1) {
                n_embeddings_round = this->n_ids_thisBlock_ - (r * this->n_embeddingsPerRound_);
                n_scalars_round = n_embeddings_round * this->n_scalarsInEmbedding_;
            }

            // 0. Load the indexes from GM in UB for this round
            local_slot_t ubIds = this->idxQue_.template AllocTensor<slot_t>();
            auto idsSource = this->gmIds_[this->first_id_thisBlock_ + r * this->n_embeddingsPerRound_];
            // AscendC::printf("Round %d -- Copying ids from %d for %d \n", r, firstIdx_this_round,
            // this->n_embeddingsPerRound_);
            AscendC::DataCopy(ubIds, idsSource, this->n_embeddingsPerRound_);
            idxQue_.EnQue(ubIds);
            ubIds = idxQue_.DeQue<slot_t>();

            // 1. Allocate a tensor from the buffer to copy the embeddings in UB
            local_scalar_t embeddingTensor = this->embeddingQue_.template AllocTensor<scalar_t>();

            // 2. Copy the embeddings for one round from GM to the localTensor in UB
            for (uint32_t i = 0; i < n_embeddings_round; i++) {
                slot_t idx = ubIds.GetValue(i);
                auto gmSrc = this->gmEmbed_[idx * this->n_scalarsInEmbedding_];
                // AscendC::printf("Round %d Iteration %d - Copying in from idx %d with stride %d into index %d\n", r,
                // i, idx, this->n_scalarsInEmbedding_, i);
                AscendC::DataCopy(embeddingTensor[i * this->n_scalarsInEmbedding_], gmSrc, this->n_scalarsInEmbedding_);
            }

            // 3. Send the tensor through the pipe and wait the operations on it to finish
            embeddingQue_.EnQue(embeddingTensor);
            embeddingTensor = embeddingQue_.DeQue<scalar_t>();

            // 4. datacopy from the entire UB tensor into GM dst tensor
            auto gmDst = this->gmEmbedDst_[(this->first_id_thisBlock_ + r * this->n_embeddingsPerRound_) *
                                           this->n_scalarsInEmbedding_];

            AscendC::DataCopy(gmDst, embeddingTensor, n_scalars_round);
            // AscendC::PipeBarrier<PIPE_ALL>();
            // AscendC::printf("Copying to HBM round %d position %d stride %d\n", r, (this->first_id_thisBlock_ + r
            // *this->n_embeddingsPerRound_), this->n_scalarsInEmbedding_);

            // 5. free allocated local tensor
            // This free should force the next loop to get another tensor and do double buffering
            this->embeddingQue_.FreeTensor(embeddingTensor);
            this->idxQue_.FreeTensor(ubIds);
            // AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            // AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        }
    }

private:
    AscendC::TPipe *pipe_;
    // a depth of 2
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2> embeddingQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> idxQue_;

    AscendC::GlobalTensor<scalar_t> gmEmbed_;
    AscendC::GlobalTensor<scalar_t> gmEmbedDst_;
    AscendC::GlobalTensor<slot_t> gmIds_;

    uint32_t totalNumIds_;
    uint32_t n_ids_thisBlock_;
    uint32_t first_id_thisBlock_;
    uint32_t n_scalarsInEmbedding_;

    uint32_t n_embeddingsPerRound_;
    uint32_t n_rounds_;
};

}  // namespace Gather

#endif