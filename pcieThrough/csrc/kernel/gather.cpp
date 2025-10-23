#include "kernel_operator.h"
#include "gather.h"
#include "types.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void gather_kernel(uint32_t type, uint32_t slotType, GM_ADDR gmEmbed,
    GM_ADDR gmEmbedDst, GM_ADDR gmIds, size_t numIds, size_t embedSizeBytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    int32_t blockIdx = AscendC::GetBlockIdx();
    int32_t blockNum = AscendC::GetBlockNum();
    pciethrough::DATATYPE type_pcie = static_cast<pciethrough::DATATYPE>(type);
    pciethrough::DATATYPE slotType_pcie = static_cast<pciethrough::DATATYPE>(slotType);

    if (type_pcie == pciethrough::DATATYPE::BF16) {
        if (slotType_pcie == pciethrough::DATATYPE::INT32_T) {
            Gather::MteCopy<bfloat16_t, int32_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        } else if (slotType_pcie == pciethrough::DATATYPE::INT64_T) {
            Gather::MteCopy<bfloat16_t, int64_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        }
    } else if (type_pcie == pciethrough::DATATYPE::HALF) {
        if (slotType_pcie == pciethrough::DATATYPE::INT32_T) {
            Gather::MteCopy<float16_t, int32_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        } else if (slotType_pcie == pciethrough::DATATYPE::INT64_T) {
            Gather::MteCopy<float16_t, int64_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        }
    } else if (type_pcie == pciethrough::DATATYPE::INT8_T) {
        if (slotType_pcie == pciethrough::DATATYPE::INT32_T) {
            Gather::MteCopy<int8_t, int32_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        } else if (slotType_pcie == pciethrough::DATATYPE::INT64_T) {
            Gather::MteCopy<int8_t, int64_t> op{};
            op.init(gmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes, blockNum, blockIdx, &pipe);
            op.LoadStoreFunc();
        }
    }
}