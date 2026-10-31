// Unit tests for the retention cost model. Host-only: no artifact, no device. The estimate drives
// both disk eviction order and the restore admission gate, so it must be monotone in the inputs the
// policy advertises: re-prefill tokens, image bytes, observed hits, and idle time.

#include "targets/qwen3_6/impl/runtime/park_value.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

using ninfer::targets::qwen3_6::detail::ParkedSequence;
using ninfer::targets::qwen3_6::detail::ParkRestorePayoff;
using ninfer::targets::qwen3_6::detail::ParkValueParams;
using ninfer::targets::qwen3_6::detail::estimate_park_value;
using ninfer::targets::qwen3_6::detail::park_restore_passes_gate;

namespace {

int failures = 0;

void check(bool ok, const char* message) {
    if (!ok) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

ParkedSequence entry(std::uint32_t tokens, std::size_t bytes, std::uint32_t hits,
                     std::uint64_t created, std::uint64_t last_hit) {
    ParkedSequence value;
    value.execution_frontier = tokens;
    value.bytes              = bytes;
    value.reuse_hits         = hits;
    value.created_wall_sec   = created;
    value.last_hit_wall_sec  = last_hit;
    return value;
}

ParkValueParams defaults() {
    ParkValueParams params;
    params.prefill_sec_per_token = 1.0 / 1000.0; // 1000 tokens/s
    params.restore_sec_per_byte  = 1.0 / 1.5e9;  // 1.5 GB/s
    params.overhead_sec          = 0.25;
    params.half_life_sec         = 1800.0;
    params.gate_min_saving_sec   = 0.75;
    return params;
}

} // namespace

int main() {
    const ParkValueParams params = defaults();
    const double now             = 1'000'000.0;

    // A fresh, frequently restored conversation has a decent return prior; one never restored has
    // half that prior.
    const auto fresh_hit = estimate_park_value(entry(10000, 1ULL << 30, 3, now - 60, now - 60),
                                               10000, now, params);
    const auto fresh_cold = estimate_park_value(entry(10000, 1ULL << 30, 0, now - 60, 0),
                                                10000, now, params);
    check(fresh_hit.p_return > fresh_cold.p_return,
          "observed restores must raise the return prior");
    check(fresh_hit.net_sec > fresh_cold.net_sec,
          "a hotter conversation must be worth more at the same age");

    // Idle time decays the value of an otherwise identical entry.
    const auto idle = estimate_park_value(entry(10000, 1ULL << 30, 3, now - 7200, now - 7200),
                                          10000, now, params);
    check(idle.p_return < fresh_hit.p_return, "idle time must decay the return probability");
    check(idle.net_sec < fresh_hit.net_sec, "idle time must lower the expected saving");

    // More reusable tokens means more avoided prefill, hence more value.
    const auto small = estimate_park_value(entry(2000, 1ULL << 30, 0, now - 60, 0), 2000, now,
                                           params);
    const auto large = estimate_park_value(entry(40000, 1ULL << 30, 0, now - 60, 0), 40000, now,
                                           params);
    check(large.expected_save_sec > small.expected_save_sec,
          "reused tokens must raise the expected saving");

    // The compaction case: a shallow 2K-token match against a 6 GB image cannot repay the read.
    const auto shallow = estimate_park_value(entry(2000, 6ULL << 30, 0, now - 60, 0), 2000, now,
                                             params);
    check(shallow.net_sec < 0.0, "a shallow match on a large image must have negative net value");
    check(shallow.density < 0.0, "a net-negative entry must have negative density");

    // A full 40K-token restore on a 1 GB image comfortably repays its read.
    const auto deep = estimate_park_value(entry(40000, 1ULL << 30, 0, now - 60, 0), 40000, now,
                                          params);
    check(deep.net_sec > 0.0, "a deep match on a modest image must have positive net value");

    // Shared cross-conversation prefixes do not decay with idle time.
    const auto shared_old = estimate_park_value(entry(4096, 1ULL << 30, 0, now - 500000, 0),
                                                4096, now, params, /*shared_prefix=*/true);
    check(shared_old.p_return == 1.0, "a shared prefix must keep a certainty prior");
    check(shared_old.expected_save_sec > 0.0, "a shared prefix must retain its saving");

    // The gate accepts a deep match and declines a shallow one.
    const ParkedSequence big = entry(40000, 1ULL << 30, 0, now - 60, 0);
    check(park_restore_passes_gate(big, 40000, params), "a deep restore must pass the gate");
    const ParkedSequence heavy = entry(40000, 6ULL << 30, 0, now - 60, 0);
    check(!park_restore_passes_gate(heavy, 2000, params),
          "a shallow restore from a huge image must be declined");
    check(park_restore_passes_gate(heavy, 40000, params),
          "the same image is worth restoring for its full frontier");
    // An on-demand restore is certain to be used, so idle decay must not weaken the gate: the same
    // deep image is still worth loading after a long idle period.
    const ParkedSequence idle_big = entry(40000, 1ULL << 30, 0, now - 200000, 0);
    check(park_restore_passes_gate(idle_big, 40000, params),
          "idle time must not make an on-demand deep restore look worthless");
    ParkRestorePayoff payoff;
    check(park_restore_passes_gate(big, 40000, params, &payoff) && payoff.save_sec > payoff.restore_sec,
          "the restore payoff must expose its saved and load components");

    // Degenerate parameters and timestamps must not produce NaN/inf densities.
    ParkValueParams broken = params;
    broken.prefill_sec_per_token = std::numeric_limits<double>::quiet_NaN();
    broken.restore_sec_per_byte  = -1.0;
    broken.half_life_sec         = 0.0;
    const auto guarded           = estimate_park_value(entry(10000, 1ULL << 30, 0, now + 500, 0),
                                                       10000, now, broken);
    check(std::isfinite(guarded.density), "a future timestamp and bad rates must stay finite");
    check(std::isfinite(guarded.restore_sec), "restore cost must stay finite");

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
