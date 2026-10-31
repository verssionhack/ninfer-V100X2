#pragma once

// Optional operator-facing status line. Interactive mode redraws one line on a terminal; log mode
// emits the same content as ordinary terminated log lines (the TUI and piped logs remain
// line-oriented). All data comes from published snapshots: RuntimeStats under the executor's
// stats mutex and the serve-side live registry. No engine or scheduler state is read directly.

#include "serve/live_status.h"

#include <ninfer/types.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

class GenerationService;

struct StatusLineOptions {
    bool enabled                  = false;
    std::chrono::milliseconds interval{500};
    std::chrono::milliseconds log_interval{5000};
    bool interactive              = false;
};

struct StatusSnapshot {
    ninfer::RuntimeStats stats;
    std::optional<LiveRequestSnapshot> prefill;
    // Oldest request still in host-side preparation (media acquisition/preprocessing), if any.
    std::optional<LiveRequestSnapshot> preparing;
    // Resolved live snapshot for each entry in `stats.decode_requests`, by index. An entry that
    // has no serve-side registration (the engine started the lane before the id was bound) is
    // std::nullopt. Decode and prefill speeds are not stored here: the engine publishes them in
    // `stats.decode_tokens_per_second` / `stats.prefill_tokens_per_second`, measured where the work
    // runs.
    std::vector<std::optional<LiveRequestSnapshot>> decodes;
    std::uint64_t uptime_seconds = 0;
};

// Pure formatter, exported for tests. `log_mode` only relaxes nothing yet; it exists so future
// output can differ (e.g. elapsed markers) without changing callers.
[[nodiscard]] std::string format_status_line(const StatusSnapshot& snapshot, bool log_mode);

class StatusLineReporter {
public:
    StatusLineReporter(GenerationService& service, StatusLineOptions options);
    ~StatusLineReporter();

    StatusLineReporter(const StatusLineReporter&)            = delete;
    StatusLineReporter& operator=(const StatusLineReporter&) = delete;

    void start();
    void stop();

private:
    void run();
    void render_once();

    GenerationService& service_;
    StatusLineOptions options_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stopping_ = false;
    bool started_  = false;

    std::chrono::steady_clock::time_point started_at_;
    std::chrono::steady_clock::time_point last_log_;
};

} // namespace ninfer::serve
