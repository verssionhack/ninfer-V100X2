#pragma once

#include "core/tensor.h"
#include <array>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
// Validated non-overlapping BF16 operands, qualified P2P mapping, and rank-local output.
void allreduce_peer_sum_launch(const Tensor& local, const Tensor& peer, const Tensor& output,
                               cudaStream_t stream);
void allreduce_sum4_launch(const std::array<Tensor, 4>& inputs, const Tensor& output,
                          cudaStream_t stream);
} // namespace ninfer::ops::detail
