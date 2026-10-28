// Host-only check of the actkv policy layer: kvmem::LrfuHistory composed with ninfer's
// rank_ram_kv_pages. It proves that decaying recency/frequency history keeps a block that was
// relevant in an earlier retrieval round ahead of blocks that only tie on the current round,
// and that the same input without LRFU drops it.
#include "kvmem/lrfu.hpp"
#include "targets/qwen3_6/impl/runtime/ram_kv_selection.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

bool check(bool condition, const char* what) {
    if (!condition) { std::fprintf(stderr, "actkv lrfu: FAIL %s\n", what); }
    return condition;
}

bool starts_with(const std::vector<std::int32_t>& ranked, std::initializer_list<std::int32_t> prefix) {
    if (ranked.size() < prefix.size()) { return false; }
    std::size_t i = 0;
    for (const auto value : prefix) {
        if (ranked[i++] != value) { return false; }
    }
    return true;
}

// Mirrors ProgramImplCore::refresh_ram_kv: sanitize, preview, score, rank, then commit.
std::vector<std::int32_t> rank_round(kvmem::LrfuHistory& lrfu, const std::vector<double>& scores) {
    lrfu.begin_round();
    std::vector<double> round(scores.size());
    for (std::size_t b = 0; b < scores.size(); ++b) {
        round[b] = std::isfinite(scores[b]) ? scores[b] : 0.0;
    }
    lrfu.preview(round);
    std::vector<double> combined(scores.size());
    for (std::size_t b = 0; b < combined.size(); ++b) { combined[b] = lrfu.score(b); }
    auto ranked = ninfer::targets::qwen3_6::detail::rank_ram_kv_pages(combined);
    lrfu.commit();
    return ranked;
}

} // namespace

int main() {
    using ninfer::targets::qwen3_6::detail::rank_ram_kv_pages;

    bool ok = true;

    std::vector<double> round_a(8, 0.0);
    round_a[2] = 0.9;
    round_a[5] = 0.8;
    std::vector<double> round_b(8, 0.0);
    round_b[2] = 0.9;

    // Without LRFU the second round sees only its own score: block 2 wins and every other block
    // ties at zero, so the tie-break (higher block first) keeps block 5 out of the top pages.
    ok &= check(starts_with(rank_ram_kv_pages(round_b), {4, 5, 14, 15}),
                "single-shot ranking should begin with blocks 2 and 7");

    // With LRFU the prior round's block 5 retains half of its 0.8 score and outranks the
    // never-relevant blocks.
    kvmem::LrfuHistory lrfu(0.5, 0.9);
    ok &= check(starts_with(rank_round(lrfu, round_a), {4, 5, 10, 11}),
                "first LRFU round should match a single-shot ranking");
    ok &= check(starts_with(rank_round(lrfu, round_b), {4, 5, 10, 11}),
                "decayed history should keep block 5 ahead of never-relevant blocks");

    // A slower decay preserves more of the earlier round without overtaking the current winner.
    kvmem::LrfuHistory slow(0.9, 0.9);
    slow.begin_round();
    slow.preview(round_a);
    slow.commit();
    slow.begin_round();
    slow.preview(round_b);
    ok &= check(slow.score(5) > 0.0 && slow.score(5) < slow.score(2),
                "lambda 0.9 should keep block 5 positive but below block 2");
    slow.commit();

    // Non-finite retrieval scores are sanitized before preview (preview itself rejects them).
    kvmem::LrfuHistory sanitized(0.5, 0.9);
    std::vector<double> nonfinite(4, std::numeric_limits<double>::quiet_NaN());
    nonfinite[1] = 0.5;
    const auto ranked = rank_round(sanitized, nonfinite);
    ok &= check(ranked.size() == 8 && ranked[0] == 2 && ranked[1] == 3,
                "sanitized round should rank the finite block first");

    if (!ok) { return 1; }
    std::printf("actkv lrfu: OK\n");
    return 0;
}
