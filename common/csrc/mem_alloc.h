#pragma once
#include <cstdint>
#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor alloc_numa_pinned_tensor(std::size_t size);

uintptr_t alloc_pinned_numa_ptr(std::size_t, int numaNode);

void free_pinned_numa_ptr(void* ptr);