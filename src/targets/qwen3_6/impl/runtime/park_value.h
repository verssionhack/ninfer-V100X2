#pragma once

// Cost/value estimate for a parked or spilled conversation. Keeping an entry occupies bytes and,
// on a restore, costs a disk read; its benefit is the prefill work a future request avoids. The
// estimate is deliberately small and deterministic so eviction order is testable:
//
//   idle  = max(0, now - max(created, last_hit))
//   p     = (hits + 1) / (hits + 2) * exp(-idle / half_life)      (1 for shared prefixes)
//   save  = p * reusable_tokens * prefill_sec_per_token
//   cost  = bytes * restore_sec_per_byte + overhead_sec
//   net   = save - cost;   density = net / bytes
//
// `density` orders eviction inside a byte budget; `net` gates a restore. The hit prior is a
// Laplace-style estimate: a conversation that has never been restored keeps a smaller chance of
// returning than one that has, while recency decay eventually retires both.

#include "targets/qwen3_6/impl/runtime/parked_sequence_store.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::targets::qwen3_6::detail {

struct ParkValueParams {
    // Seconds of prefill one token costs. Seeded by --kv-park-prefill-tok-s or a measured EWMA.
    double prefill_sec_per_token = 1.0 / 1000.0;
    // Seconds one image byte costs to read back. Seeded by --kv-park-read-bytes-per-sec or an
    // EWMA of observed restores.
    double restore_sec_per_byte = 1.0 / 1.5e9;
    // Fixed device-staging and scheduling cost of a restore.
    double overhead_sec = 0.25;
    // Idle-time decay constant for the return probability.
    double half_life_sec = 1800.0;
    // A disk restore whose estimated net saving is below this is declined. 0 disables the gate.
    double gate_min_saving_sec = 0.75;
};

struct ParkValue {
    double p_return         = 0.0;
    double expected_save_sec = 0.0;
    double restore_sec      = 0.0;
    double net_sec          = 0.0;
    double density          = 0.0;
};

namespace park_value_detail {

inline double positive_or(double value, double fallback) noexcept {
    return std::isfinite(value) && value > 0.0 ? value : fallback;
}

inline double finite_or_zero(double value) noexcept {
    return std::isfinite(value) ? value : 0.0;
}

} // namespace park_value_detail

// `shared_prefix` marks a Tools/Leading boundary that is reusable by any conversation sharing the
// block; those are not conversations and must not decay with idle time. `reusable_tokens` is the
// frontier a restore would actually reuse (execution_frontier for retention).
inline ParkValue estimate_park_value(const ParkedSequence& entry, std::uint32_t reusable_tokens,
                                     double now_wall_sec, const ParkValueParams& params,
                                     bool shared_prefix = false) noexcept {
    using park_value_detail::finite_or_zero;
    using park_value_detail::positive_or;

    ParkValue value;
    const double half_life = positive_or(params.half_life_sec, 1800.0);
    const double reference =
        static_cast<double>(std::max(entry.created_wall_sec, entry.last_hit_wall_sec));
    double idle = now_wall_sec - reference;
    if (!(idle >= 0.0) || reference <= 0.0) { idle = 0.0; } // NaN or a future mtime clamps to 0.

    double p_return = 1.0;
    if (!shared_prefix) {
        const double prior =
            (static_cast<double>(entry.reuse_hits) + 1.0) /
            (static_cast<double>(entry.reuse_hits) + 2.0);
        p_return = std::clamp(prior * std::exp(-idle / half_life), 0.0, 1.0);
    }

    value.p_return = p_return;
    value.expected_save_sec = p_return * static_cast<double>(reusable_tokens) *
                              positive_or(params.prefill_sec_per_token, 1.0 / 1000.0);
    value.restore_sec = static_cast<double>(entry.bytes) *
                            positive_or(params.restore_sec_per_byte, 1.0 / 1.5e9) +
                        std::max(0.0, params.overhead_sec);
    value.net_sec = finite_or_zero(value.expected_save_sec - value.restore_sec);
    const double bytes = static_cast<double>(std::max<std::size_t>(entry.bytes, 1));
    value.density      = finite_or_zero(value.net_sec / bytes);
    return value;
}

// An on-demand restore is not a retention bet: the conversation is returning *now*, so its return
// probability is 1. The payoff compares the prefill work the matched frontier replaces against the
// full-image read and fixed staging cost; it deliberately has no idle decay or hit prior.
struct ParkRestorePayoff {
    double save_sec    = 0.0;
    double restore_sec = 0.0;
    double net_sec     = 0.0;
};

inline ParkRestorePayoff estimate_restore_payoff(const ParkedSequence& entry,
                                                 std::uint32_t reusable_tokens,
                                                 const ParkValueParams& params) noexcept {
    using park_value_detail::finite_or_zero;
    using park_value_detail::positive_or;
    ParkRestorePayoff payoff;
    payoff.save_sec = static_cast<double>(reusable_tokens) *
                      positive_or(params.prefill_sec_per_token, 1.0 / 1000.0);
    payoff.restore_sec = static_cast<double>(entry.bytes) *
                             positive_or(params.restore_sec_per_byte, 1.0 / 1.5e9) +
                         std::max(0.0, params.overhead_sec);
    payoff.net_sec = finite_or_zero(payoff.save_sec - payoff.restore_sec);
    return payoff;
}

inline bool park_restore_passes_gate(const ParkedSequence& entry, std::uint32_t reusable_tokens,
                                     const ParkValueParams& params,
                                     ParkRestorePayoff* estimated = nullptr) noexcept {
    const ParkRestorePayoff payoff = estimate_restore_payoff(entry, reusable_tokens, params);
    if (estimated != nullptr) { *estimated = payoff; }
    return payoff.net_sec >= params.gate_min_saving_sec;
}

} // namespace ninfer::targets::qwen3_6::detail
