#pragma once

// Implements the two-operand sum stage of include/ninfer/ops/allreduce.h. Qualified SM70
// P2P reads only: both operands are immutable until the caller's cross-device completion edge.
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops {

__global__ void allreduce_peer_sum_scalar_kernel(const __nv_bfloat16* local,
                                                const __nv_bfloat16* peer,
                                                __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = __float2bfloat16_rn(__bfloat162float(local[i]) + __bfloat162float(peer[i]));
    }
}

__global__ void allreduce_peer_sum_packed_kernel(const Bf16x8Pack* local,
                                                const Bf16x8Pack* peer,
                                                Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto a = load_vec<Bf16x8Pack>(local + i);
    const auto b = load_vec<Bf16x8Pack>(peer + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a.pair[j] = __floats2bfloat162_rn(__low2float(a.pair[j]) + __low2float(b.pair[j]),
                                        __high2float(a.pair[j]) + __high2float(b.pair[j]));
    }
    store_vec(output + i, a);
}

// Fixed global rank order is essential: rotating the association per destination
// would produce different replicated residuals once there are more than two ranks.
__global__ void allreduce_sum4_scalar_kernel(const __nv_bfloat16* a,
                                            const __nv_bfloat16* b,
                                            const __nv_bfloat16* c,
                                            const __nv_bfloat16* d,
                                            __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = __bfloat162float(a[i]) + __bfloat162float(b[i]);
        sum += __bfloat162float(c[i]);
        sum += __bfloat162float(d[i]);
        output[i] = __float2bfloat16_rn(sum);
    }
}

__global__ void allreduce_sum4_packed_kernel(const Bf16x8Pack* a, const Bf16x8Pack* b,
                                            const Bf16x8Pack* c, const Bf16x8Pack* d,
                                            Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto av = load_vec<Bf16x8Pack>(a + i);
    const auto bv = load_vec<Bf16x8Pack>(b + i);
    const auto cv = load_vec<Bf16x8Pack>(c + i);
    const auto dv = load_vec<Bf16x8Pack>(d + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float low = __low2float(av.pair[j]) + __low2float(bv.pair[j]);
        float high = __high2float(av.pair[j]) + __high2float(bv.pair[j]);
        low += __low2float(cv.pair[j]);
        high += __high2float(cv.pair[j]);
        low += __low2float(dv.pair[j]);
        high += __high2float(dv.pair[j]);
        av.pair[j] = __floats2bfloat162_rn(low, high);
    }
    store_vec(output + i, av);
}

} // namespace ninfer::ops
