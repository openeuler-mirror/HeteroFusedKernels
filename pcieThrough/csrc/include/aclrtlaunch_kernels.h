#ifndef HEADER_ACLRTLAUNCH_KERNELS_H
#define HEADER_ACLRTLAUNCH_KERNELS_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_multi_layer_block_transfer(uint32_t blockDim, aclrtStream stream, void* devBlockCachePtrs, void* hostBlockCache, void* tilingData);
extern "C" uint32_t aclrtlaunch_gather_kernel(uint32_t blockDim, aclrtStream stream, uint32_t type, uint32_t slotType, void* gmEmbed, void* gmEmbedDst, void* gmIds, size_t numIds, size_t embedSizeBytes);

#endif