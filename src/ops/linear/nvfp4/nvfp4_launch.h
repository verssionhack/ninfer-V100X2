#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {

// Opt-in prefill route (NINFER_NVFP4_VOLTA_MMA=1): the wide-T fused-dequant MMA below, which
// consumes the portable row-major code plane. Selecting it also keeps that layout at load time
// (the load-time QPN prepack is skipped), so small-T/decode fall back to the generic NVFP4 paths.
[[nodiscard]] inline bool nvfp4_volta_mma_prefill_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_NVFP4_VOLTA_MMA");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Quadpair-split-N form (nvfp4_volta_qpn_gemm.cuh), the fourth sibling of the q4/w8/fp8 QPN
// kernels. The tile height is mirrored here because dispatch is host code and cannot see the
// device header; the launcher static_asserts the two agree.
inline constexpr std::int32_t kNvfp4VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kNvfp4VoltaQpnMaxTokens   = 32;
void launch_nvfp4_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_nvfp4_volta_qpn_precast(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                    cudaStream_t);
void nvfp4_volta_qpn_cast_activation(const Tensor&, Tensor&, cudaStream_t);
[[nodiscard]] bool nvfp4_volta_qpn_precast_supported(std::int32_t n, std::int32_t k,
                                                    std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t nvfp4_volta_qpn_precast_workspace_bytes(std::int32_t n, std::int32_t k,
                                                                 std::int32_t max_tokens);
[[nodiscard]] bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;

// Wide-T form (nvfp4_volta_mma_gemm.cuh): T on the mma A axis, dequantizing into shared memory
// once per K-pass instead of QPN2's small-T-chunked re-streaming. Prefill's route.
void launch_nvfp4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream, int splits_override = 0);
[[nodiscard]] bool nvfp4_volta_mma_supported(std::int32_t n, std::int32_t k,
                                             std::int32_t t) noexcept;
[[nodiscard]] std::size_t nvfp4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                                          std::int32_t t) noexcept;
[[nodiscard]] int nvfp4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept;
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
