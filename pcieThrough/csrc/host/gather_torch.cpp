#include "securec.h"
#include "aclrtlaunch_kernels.h"
#include <ATen/ATen.h>
#include "managed_mem.h"
#include "../kernel/types.h"
#include "launch_utils.h"

namespace py = pybind11;

namespace pciethrough {

DATATYPE fromAtenScalarTypeGather(at::ScalarType scalarType)
{
    switch (scalarType) {
        case at::ScalarType::Float:
            return DATATYPE::FLOAT;
        case at::ScalarType::Half:
            return DATATYPE::HALF;
        case at::ScalarType::BFloat16:
            return DATATYPE::BF16;
        case at::ScalarType::Char:
            return DATATYPE::INT8_T;
        case at::ScalarType::Long:
            return DATATYPE::INT64_T;
        default:
            TORCH_CHECK(false, std::string("ScalarType ") + c10::toString(scalarType) + " not supported.")
    }
}

void gather(torch::Tensor &gmEmbed, torch::Tensor &gmEmbedDst, torch::Tensor &gmIds,
                                c10::optional<int64_t> aiv = c10::nullopt) {
    uint32_t aivNum = 2;
    if (aiv.has_value()) {
        aivNum = aiv.value();
        TORCH_CHECK(aivNum > 0, "Number of AIV blocks should be positive.")
    }

    uint32_t numIds = gmIds.size(0);
    size_t embedSizeBytes = gmEmbed.element_size() * gmEmbed.size(1);

    const c10::OptionalDeviceGuard device_guard(device_of(gmEmbed));

    at::ScalarType scalar_type = gmEmbed.scalar_type();
    at::ScalarType slot_type = gmIds.scalar_type();
    uint32_t type = static_cast<uint32_t>(fromAtenScalarTypeGather(scalar_type));
    uint32_t slotType = static_cast<uint32_t>(fromAtenScalarTypeGather(slot_type));

    aivNum = std::min(aivNum, numIds);

    auto &hmm = managed_memory::HostRegisteredMemoryManager::GetInstance();
    void *hostgmEmbed = hmm.getDevicePtr(gmEmbed.data_ptr());

    EXEC_KERNEL_CMD(gather_kernel, aivNum, type, slotType, hostgmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes);
}
}  // namespace pciethrough

namespace {
TORCH_LIBRARY_FRAGMENT(pcie_through, m)
{
    m.def("gather(Tensor gmEmbed, Tensor gmEmbedDst, Tensor gmIds, int? aiv=None) -> ()");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(pcie_through, PrivateUse1, m)
{
    m.impl("gather", TORCH_FN(pciethrough::gather));
}
}  // namespace
