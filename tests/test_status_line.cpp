// Host-only tests for the status-line formatter. It reads a published snapshot only, so no engine,
// artifact, or device is needed.

#include "serve/live_status.h"
#include "serve/status_line.h"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using ninfer::serve::format_fingerprint;
using ninfer::serve::format_status_line;
using ninfer::serve::LiveRequestSnapshot;
using ninfer::serve::MediaPhase;
using ninfer::serve::StatusSnapshot;

namespace {

int failures = 0;

void check(bool ok, const char* message) {
    if (!ok) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
    check(format_fingerprint(0xDEADBEEFCAFEF00DULL, 8) == "cafef00d",
          "the short fingerprint must be the low eight hex digits");
    check(format_fingerprint(0x1234ULL, 16) == "0000000000001234",
          "the long fingerprint must be zero-padded to sixteen digits");
    check(format_fingerprint(0x1234ULL, 64).size() == 16,
          "the fingerprint width must clamp to sixteen");

    StatusSnapshot idle;
    idle.uptime_seconds = 3723;
    const std::string idle_line = format_status_line(idle, false);
    check(contains(idle_line, "status idle"), "an empty snapshot must report idle");
    check(contains(idle_line, "up=1h02m"), "the status line must carry uptime");
    check(contains(idle_line, "d/p=0/0 t/s"), "an idle line must still show decode/prefill speeds");
    check(idle_line.find('\r') == std::string::npos, "the formatter must never emit carriage returns");

    StatusSnapshot busy;
    busy.stats.prefill_request_id             = 7;
    busy.stats.prefill_prompt_tokens          = 14000;
    busy.stats.prefill_processed_tokens       = 8192;
    busy.stats.prefill_vision_done            = 1;
    busy.stats.prefill_vision_total           = 4;
    busy.stats.running_requests               = 2;
    busy.stats.prefilling_requests            = 1;
    busy.stats.prefill_vision_swap            = 1;
    busy.stats.decode_ready_requests          = 1;
    busy.stats.waiting_requests               = 0;
    busy.stats.kv_park.ram_bytes              = 12ULL << 30;
    busy.stats.kv_park.ram_budget             = 16ULL << 30;
    busy.stats.kv_park.ram_entries            = 7;
    busy.stats.kv_park.disk_bytes             = 41ULL << 30;
    busy.stats.kv_park.disk_budget            = 64ULL << 30;
    busy.stats.kv_park.disk_entries           = 13;
    busy.stats.kv_park.restores_ram           = 3;
    busy.stats.kv_park.restores_disk          = 1;
    busy.stats.kv_park.spills                 = 9;
    busy.stats.kv_park.evictions_value        = 2;
    busy.stats.kv_park.evictions_lru          = 1;
    busy.stats.kv_park.gate_declines          = 2;
    LiveRequestSnapshot request;
    request.http_id      = 42;
    request.engine_id    = 7;
    request.fingerprint  = 0x9F3A1C07ULL;
    request.media_total  = 3;
    request.media_done   = 3;
    request.media_phase  = MediaPhase::Preprocess;
    busy.prefill                = request;
    busy.stats.prefill_tokens_per_second = 1140.0;
    // A decoding request, paired with its live snapshot by engine id.
    busy.stats.decode_requests[0] = ninfer::DecodeRequestStats{
        .engine_id = 9, .generated_tokens = 51, .budget_tokens = 256};
    // A second decoding lane whose engine started before the serve-side id was bound: no live
    // snapshot, so it must render the unknown-request sentinel rather than a spurious id.
    busy.stats.decode_requests[1] = ninfer::DecodeRequestStats{
        .engine_id = 11, .generated_tokens = 20, .budget_tokens = 100};
    busy.stats.decode_request_count   = 2;
    busy.stats.decode_tokens_per_second = 18.4;
    LiveRequestSnapshot decode_snapshot;
    decode_snapshot.http_id   = 44;
    decode_snapshot.engine_id = 9;
    decode_snapshot.has_tools = true;
    decode_snapshot.thinking  = true;
    busy.decodes = std::vector<std::optional<LiveRequestSnapshot>>{decode_snapshot, std::nullopt};
    busy.uptime_seconds         = 500;

    const std::string line = format_status_line(busy, false);
    check(contains(line, "req=42"), "the status line must name the prefilling request");
    check(contains(line, "fp=9f3a1c07"), "the status line must carry the short fingerprint");
    check(contains(line, "58.5%"), "the prefill percentage must reflect processed tokens");
    check(contains(line, "8192/14000"), "the status line must show processed/total tokens");
    check(contains(line, "1.14k t/s"),
          "the status line must show the engine-measured prefill rate");
    check(contains(line, "eta "), "a partial prefill must show an ETA");
    check(contains(line, "vis 1/4"), "the status line must show vision item progress");
    check(contains(line, "vis 1/4 swap"),
          "a vision-swap prefill must tag its vision block with swap");
    check(!contains(line, " kv="), "a fresh prefill must not show a KV-switch tag");
    check(contains(line, "run=2"), "the status line must show scheduler occupancy");
    check(contains(line, "d/p=18/1.14k t/s"),
          "the status line must show decode/prefill speeds as d/p");
    check(contains(line, "decode req=44"), "the status line must name a decoding request");
    check(contains(line, "51/256"), "the status line must show generated/budget for a decode request");
    check(contains(line, "tools+think"), "the status line must tag a decoding request's kind");
    check(contains(line, "decode req=?"),
          "an unregistered decoding request must render the ? sentinel, not a spurious id");
    check(contains(line, "20/100"), "the status line must show generated/budget for the second decode");
    check(contains(line, "\xE2\x96\x88"), "the progress bars must fill with block characters");
    check(!contains(idle_line, "\xE2\x96\x88"), "an idle line must not contain a progress bar");
    check(contains(line, "kv ram 12.0G/16.0G e=7"), "the status line must show RAM park gauges");
    check(contains(line, "disk 41.0G/64.0G e=13"), "the status line must show disk spill gauges");
    check(contains(line, "hit=4"), "the status line must count restores as hits");
    check(contains(line, "gate=2"), "the status line must show restore-gate declines");
    check(line.find('\r') == std::string::npos, "the formatted line must never contain a CR");

    // A resumed prefill: 1000 of 2000 prompt tokens were reused, so the bar tracks the new work
    // [1000, 2000), not the whole prompt. At the midpoint (1500) it must read 50%, not 75%.
    StatusSnapshot resumed;
    resumed.stats.prefill_request_id         = 7;
    resumed.stats.prefill_prompt_tokens      = 2000;
    resumed.stats.prefill_start_tokens       = 1000; // reused prefix
    resumed.stats.prefill_processed_tokens   = 1500; // 500 of the 1000 new tokens done
    resumed.stats.prefill_reuse_path         = 4; // RestoreParkedKv
    resumed.stats.prefill_tokens_per_second  = 1140.0;
    resumed.uptime_seconds                   = 5;
    LiveRequestSnapshot resumed_request;
    resumed_request.http_id     = 55;
    resumed_request.fingerprint = 0x0BADF00DULL;
    resumed.prefill = resumed_request;
    const std::string resumed_line = format_status_line(resumed, false);
    check(contains(resumed_line, "50.0%"), "a resumed prefill at its midpoint must show 50%");
    check(contains(resumed_line, "500/1000"),
          "a resumed prefill must show new-work done/total, not the full prompt");
    check(contains(resumed_line, "kv=park"),
          "a parked-KV resume must tag its KV switch in the prefill section");
    check(contains(resumed_line, "\xE2\x96\x88"),
          "a resumed prefill bar must fill with block characters");

    // Every KV-switch path renders its own tag; the no-switch paths render none.
    const std::pair<std::uint8_t, const char*> kv_cases[] = {
        {2, "kv=turn"},
        {3, "kv=resp"},
        {4, "kv=park"},
        {5, "kv=share"},
    };
    for (const auto& [path, tag] : kv_cases) {
        StatusSnapshot snap  = resumed;
        snap.stats.prefill_reuse_path = path;
        check(contains(format_status_line(snap, false), tag), "each KV-switch path must render its tag");
    }
    StatusSnapshot fresh = resumed;
    fresh.stats.prefill_reuse_path = 0; // FullReset
    check(!contains(format_status_line(fresh, false), "kv="),
          "a fresh prefill must not show a KV-switch tag");
    fresh.stats.prefill_reuse_path = 1; // AppendAtFrontier
    check(!contains(format_status_line(fresh, false), "kv="),
          "a plain-append prefill must not show a KV-switch tag");

    // A request still in host-side preparation has no engine id and is reported separately.
    StatusSnapshot preparing;
    LiveRequestSnapshot pending;
    pending.http_id     = 43;
    pending.media_total = 5;
    pending.media_done  = 2;
    pending.media_phase = MediaPhase::Acquire;
    preparing.preparing = pending;
    const std::string preparing_line = format_status_line(preparing, false);
    check(contains(preparing_line, "prep req=43"), "a preparing request must be named");
    check(contains(preparing_line, "media acq 2/5"), "media acquisition counts must be shown");

    if (failures == 0) { std::printf("ok\n"); }
    return failures == 0 ? 0 : 1;
}
