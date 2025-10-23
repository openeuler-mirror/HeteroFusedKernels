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
            auto p = c10::getDtypeNames(scalarType);
            TORCH_CHECK(false, "ScalarType " + p.first + " not supported.")
    }
}

void gather(torch::Tensor &gmEmbed, torch::Tensor &gmEmbedDst, torch::Tensor &gmIds)
{
    int numIds = gmIds.size(0);
    size_t embedSizeBytes = gmEmbed.element_size() * gmEmbed.size(1);

    const c10::OptionalDeviceGuard device_guard(device_of(gmEmbed));

    at::ScalarType scalar_type = gmEmbed.scalar_type();
    at::ScalarType slot_type = gmIds.scalar_type();
    uint32_t type = static_cast<uint32_t>(fromAtenScalarTypeGather(scalar_type));
    uint32_t slotType = static_cast<uint32_t>(fromAtenScalarTypeGather(slot_type));

    uint32_t aiv_num = std::min(2, numIds);
    auto &hmm = managed_memory::HostRegisteredMemoryManager::GetInstance();
    void *hostgmEmbed = hmm.getDevicePtr(gmEmbed.data_ptr());

    EXEC_KERNEL_CMD(gather_kernel, aiv_num, type, slotType, hostgmEmbed, gmEmbedDst, gmIds, numIds, embedSizeBytes);
}
}  // namespace pciethrough

namespace {
TORCH_LIBRARY_FRAGMENT(pcie_through, m)
{
    m.def("gather(Tensor gmEmbed, Tensor gmEmbedDst, Tensor gmIds) -> ()");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(pcie_through, PrivateUse1, m)
{
    m.impl("gather", TORCH_FN(pciethrough::gather));
}
}  // namespace
