// Host-only tests for the status-line formatter. It reads a published snapshot only, so no engine,
// artifact, or device is needed.

#include "serve/live_status.h"
#include "serve/status_line.h"

#include <cstdint>
#include <cstdio>
#include <string>

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
    check(idle_line.find('\r') == std::string::npos, "the formatter must never emit carriage returns");

    StatusSnapshot busy;
    busy.stats.prefill_request_id             = 7;
    busy.stats.prefill_prompt_tokens          = 14000;
    busy.stats.prefill_processed_tokens       = 8192;
    busy.stats.prefill_vision_done            = 1;
    busy.stats.prefill_vision_total           = 4;
    busy.stats.running_requests               = 2;
    busy.stats.prefilling_requests            = 1;
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
    busy.prefill          = request;
    busy.prefill_rate_tok_s = 1140.0;
    busy.decode_rate_tok_s  = 18.4;
    busy.uptime_seconds     = 500;

    const std::string line = format_status_line(busy, false);
    check(contains(line, "req=42"), "the status line must name the prefilling request");
    check(contains(line, "fp=9f3a1c07"), "the status line must carry the short fingerprint");
    check(contains(line, "58.5%"), "the prefill percentage must reflect processed tokens");
    check(contains(line, "8192/14000"), "the status line must show processed/total tokens");
    check(contains(line, "1.14k tok/s"), "the status line must show the prefill rate");
    check(contains(line, "eta "), "a partial prefill must show an ETA");
    check(contains(line, "vis 1/4"), "the status line must show vision item progress");
    check(contains(line, "run=2"), "the status line must show scheduler occupancy");
    check(contains(line, "dec=18 tok/s"), "the status line must show the decode rate");
    check(contains(line, "kv ram 12.0G/16.0G e=7"), "the status line must show RAM park gauges");
    check(contains(line, "disk 41.0G/64.0G e=13"), "the status line must show disk spill gauges");
    check(contains(line, "hit=4"), "the status line must count restores as hits");
    check(contains(line, "gate=2"), "the status line must show restore-gate declines");
    check(line.find('\r') == std::string::npos, "the formatted line must never contain a CR");

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
