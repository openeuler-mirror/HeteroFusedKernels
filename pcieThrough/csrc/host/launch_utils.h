#pragma once
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace pciethrough {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    c10_npu::GetDevice(&deviceIndex);
    return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline void *ConvertType(const at::Tensor &at_tensor)
{
    return const_cast<void *>(at_tensor.data_ptr());
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

#define STRINGIZE(x) #x
#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                         \
    do {                                                                                    \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                     \
        auto converted_params = ConvertTypes(__VA_ARGS__);                                  \
        const char* kernel_name_str = STRINGIZE(kernel_name);                               \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                 \
            std::apply([&](auto&&... params) {                                              \
                ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, params...);          \
            }, converted_params);                                                           \
            return 0;                                                                       \
        };                                                                                  \
        at_npu::native::OpCommand::RunOpApi(kernel_name_str, acl_call);                     \
    } while (false)
    
} // namespace pciethrough