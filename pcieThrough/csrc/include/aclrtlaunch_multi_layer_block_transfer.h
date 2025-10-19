#ifndef HEADER_ACLRTLAUNCH_MULTI_LAYER_BLOCK_TRANSFER_H
#define HEADER_ACLRTLAUNCH_MULTI_LAYER_BLOCK_TRANSFER_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_multi_layer_block_transfer(uint32_t blockDim, aclrtStream stream, void* devBlockCachePtrs, void* hostBlockCache, void* tilingData);
#endif
