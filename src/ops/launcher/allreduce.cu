#include "ops/launcher/allreduce.h"
#include "ops/kernel/allreduce.cuh"
#include "core/device.h"
#include "ops/common/math.h"

namespace ninfer::ops::detail {
void allreduce_peer_sum_launch(const Tensor& local, const Tensor& peer, const Tensor& output,
                               cudaStream_t stream) {
    const auto count = local.numel();
    const auto addresses = reinterpret_cast<std::uintptr_t>(local.data) |
                           reinterpret_cast<std::uintptr_t>(peer.data) |
                           reinterpret_cast<std::uintptr_t>(output.data);
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_peer_sum_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(local.data), static_cast<const Bf16x8Pack*>(peer.data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_peer_sum_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(local.data), static_cast<const __nv_bfloat16*>(peer.data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_sum4_launch(const std::array<Tensor, 4>& inputs, const Tensor& output,
                          cudaStream_t stream) {
    const auto count = output.numel();
    auto addresses = reinterpret_cast<std::uintptr_t>(output.data);
    for (const auto& input : inputs) {
        addresses |= reinterpret_cast<std::uintptr_t>(input.data);
    }
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_sum4_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(inputs[0].data),
            static_cast<const Bf16x8Pack*>(inputs[1].data),
            static_cast<const Bf16x8Pack*>(inputs[2].data),
            static_cast<const Bf16x8Pack*>(inputs[3].data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_sum4_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data),
            static_cast<const __nv_bfloat16*>(inputs[1].data),
            static_cast<const __nv_bfloat16*>(inputs[2].data),
            static_cast<const __nv_bfloat16*>(inputs[3].data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
