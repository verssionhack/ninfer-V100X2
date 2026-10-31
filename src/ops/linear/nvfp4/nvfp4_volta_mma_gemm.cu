#include "core/arena.h"
#include "core/device.h"
#include "ops/common/volta_mma_splits.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_mma_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Nvfp4VoltaMmaSchedule;
} // namespace

// The same L2-budget reasoning as volta_mma_split_count (see volta_mma_splits.h), re-expressed
// for this kernel's 64x64 CTA tile. The shared heuristic's 32x32 tile factors would count 4x
// too many CTAs here and split long-K shapes past the point where the concurrent weight
// footprint stops fitting in L2.
int nvfp4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    const int row_ctas = (n + S::kRowsPerCta - 1) / S::kRowsPerCta;
    const int t_tiles  = (t + S::kTTile - 1) / S::kTTile;
    const int blocks   = row_ctas * t_tiles;
    int splits         = blocks >= kVoltaMmaCtaBudget ? 1 : kVoltaMmaCtaBudget / blocks;
    if (k >= kVoltaMmaLongK) { splits = std::min(splits, 2); }
    return std::max(splits, 1);
}

std::size_t nvfp4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                            std::int32_t t) noexcept {
    if (nvfp4_volta_mma_splits(n, k, t) == 1) { return 0; }
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(t) * sizeof(float);
}

bool nvfp4_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    // The code plane needs whole 16-k groups inside a split (k % 16 == 0 always holds for
    // registered NVFP4 shapes, k % 64 == 0 is the tighter constraint actually enforced
    // elsewhere); kKStep must divide the per-split K run.
    if (n <= 0 || t <= 0 || k <= 0) { return false; }
    if (k % 32 != 0) { return false; }
    if ((k / nvfp4_volta_mma_splits(n, k, t)) < S::kKStep) { return false; }
    return true;
}

void launch_nvfp4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream, int splits_override) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const int splits = splits_override > 0 ? splits_override : nvfp4_volta_mma_splits(n, k, t);
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta),
                    static_cast<unsigned>(splits),
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    static_assert(S::kRowsPerCta % S::kNPerWarp == 0);
    static_assert(S::kTTileWarp == 32);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out_data      = static_cast<__nv_bfloat16*>(out.data);

    if (splits == 1) {
        nvfp4_volta_mma_gemm_kernel<true><<<grid, S::kThreads, 0, stream>>>(
            codes, scales, static_cast<const __nv_bfloat16*>(x.data), nullptr, out_data, out_ld, n,
            k, t, inverse_weight_divisor, splits);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Sized from the effective split count (which may be an override), not the auto-computed one
    // -- they can disagree, and workspace_bytes(n,k,t) alone would then under-size against what
    // this specific launch actually needs.
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(t) * sizeof(float);
    auto scope           = ws.scope();
    const DeviceSpan buf = ws.alloc_bytes(workspace_bytes);
    auto* partial        = static_cast<float*>(buf.data);
    CUDA_CHECK(cudaMemsetAsync(partial, 0, workspace_bytes, stream));

    nvfp4_volta_mma_gemm_kernel<false><<<grid, S::kThreads, 0, stream>>>(
        codes, scales, static_cast<const __nv_bfloat16*>(x.data), partial, out_data, out_ld, n, k,
        t, inverse_weight_divisor, splits);
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t count    = static_cast<std::int64_t>(n) * t;
    constexpr int kNarrowThreads = 256;
    nvfp4_volta_mma_narrow_strided_kernel<<<
        static_cast<unsigned>((count + kNarrowThreads - 1) / kNarrowThreads), kNarrowThreads, 0,
        stream>>>(partial, out_data, n, t, out_ld);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
