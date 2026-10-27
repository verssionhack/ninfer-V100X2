#include "serve/status_line.h"

#include "serve/console_log.h"
#include "serve/generation_service.h"

#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace ninfer::serve {
namespace {

using Clock = std::chrono::steady_clock;

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = kKiB * 1024.0;
    constexpr double kGiB = kMiB * 1024.0;
    std::ostringstream out;
    if (static_cast<double>(bytes) >= kGiB) {
        out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / kGiB << 'G';
    } else if (static_cast<double>(bytes) >= kMiB) {
        out << std::fixed << std::setprecision(0) << static_cast<double>(bytes) / kMiB << 'M';
    } else if (static_cast<double>(bytes) >= kKiB) {
        out << std::fixed << std::setprecision(0) << static_cast<double>(bytes) / kKiB << 'K';
    } else {
        out << bytes << 'B';
    }
    return out.str();
}

std::string format_rate(double tok_per_second) {
    std::ostringstream out;
    if (tok_per_second >= 10000.0) {
        out << std::fixed << std::setprecision(1) << tok_per_second / 1000.0 << "k";
    } else if (tok_per_second >= 1000.0) {
        out << std::fixed << std::setprecision(2) << tok_per_second / 1000.0 << "k";
    } else {
        out << std::fixed << std::setprecision(0) << tok_per_second;
    }
    return out.str();
}

std::string format_clock(std::uint64_t seconds) {
    std::ostringstream out;
    const std::uint64_t hours   = seconds / 3600;
    const std::uint64_t minutes = (seconds % 3600) / 60;
    const std::uint64_t rest    = seconds % 60;
    if (hours != 0) {
        out << hours << 'h' << std::setfill('0') << std::setw(2) << minutes << 'm';
    } else if (minutes != 0) {
        out << minutes << 'm' << std::setfill('0') << std::setw(2) << rest << 's';
    } else {
        out << rest << 's';
    }
    return out.str();
}

std::string format_eta(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) { return "?"; }
    std::ostringstream out;
    if (seconds < 10.0) {
        out << std::fixed << std::setprecision(1) << seconds << 's';
    } else if (seconds < 60.0) {
        out << std::fixed << std::setprecision(0) << seconds << 's';
    } else {
        out << format_clock(static_cast<std::uint64_t>(std::round(seconds)));
    }
    return out.str();
}

std::string progress_bar(double percent) {
    constexpr std::size_t kWidth = 24;
    const std::size_t filled =
        static_cast<std::size_t>(std::clamp(percent, 0.0, 100.0) / 100.0 * kWidth);
    return std::string(filled, '#') + std::string(kWidth - filled, '-');
}

const char* media_phase_name(MediaPhase phase) {
    switch (phase) {
    case MediaPhase::Acquire:
        return "acq";
    case MediaPhase::Preprocess:
        return "pp";
    case MediaPhase::VisionEncode:
        return "vis";
    case MediaPhase::None:
        break;
    }
    return "media";
}

std::size_t terminal_width() {
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) {
        return std::max<std::size_t>(size.ws_col, 40);
    }
    return 160;
}

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

} // namespace

std::string format_status_line(const StatusSnapshot& snapshot, bool log_mode) {
    (void)log_mode;
    const ninfer::RuntimeStats& stats = snapshot.stats;
    std::ostringstream out;
    out << "status ";

    bool wrote_work = false;
    if (stats.prefill_request_id != 0) {
        // The engine can start the lane a moment before prepare() binds its id to the serve-side
        // registration; show the work rather than a spurious "idle".
        const LiveRequestSnapshot* request =
            snapshot.prefill ? &*snapshot.prefill : nullptr;
        out << "req=";
        if (request != nullptr) {
            out << request->http_id;
        } else {
            out << '?';
        }
        out << " fp=";
        if (request != nullptr && request->fingerprint != 0) {
            out << format_fingerprint(request->fingerprint);
        } else {
            out << '-';
        }
        const std::uint32_t total = stats.prefill_prompt_tokens;
        const std::uint32_t done  = stats.prefill_processed_tokens;
        const double percent =
            total != 0 ? 100.0 * static_cast<double>(std::min(done, total)) / total : 0.0;
        out << " prefill [" << progress_bar(percent) << "] " << std::fixed << std::setprecision(1)
            << percent << "% " << done << '/' << total;
        const double prefill_rate = stats.prefill_tokens_per_second;
        if (prefill_rate > 0.5) {
            out << ' ' << format_rate(prefill_rate) << " tok/s";
            if (total > done) {
                out << " eta "
                    << format_eta(static_cast<double>(total - done) / prefill_rate);
            }
        }
        wrote_work = true;
    }
    if (snapshot.preparing) {
        const LiveRequestSnapshot& request = *snapshot.preparing;
        if (wrote_work) { out << " | "; }
        out << "prep req=" << request.http_id << " media "
            << media_phase_name(request.media_phase) << ' ' << request.media_done << '/'
            << request.media_total;
        wrote_work = true;
    }
    if (!wrote_work) { out << "idle"; }

    if (stats.prefill_vision_total != 0) {
        out << " vis " << stats.prefill_vision_done << '/' << stats.prefill_vision_total;
    }

    out << " | run=" << stats.running_requests << " pref=" << stats.prefilling_requests
        << " ready=" << stats.decode_ready_requests << " wait=" << stats.waiting_requests
        << " d/p=" << format_rate(snapshot.decode_rate_tok_s) << '/'
        << format_rate(stats.prefill_tokens_per_second) << " tok/s";

    const ninfer::KvParkStats& kv = stats.kv_park;
    if (kv.ram_budget != 0 || kv.ram_entries != 0 || kv.disk_budget != 0 ||
        kv.disk_entries != 0) {
        out << " | kv ram " << format_bytes(kv.ram_bytes) << '/' << format_bytes(kv.ram_budget)
            << " e=" << kv.ram_entries;
        if (kv.disk_budget != 0 || kv.disk_entries != 0 || kv.disk_bytes != 0) {
            out << " disk " << format_bytes(kv.disk_bytes) << '/'
                << format_bytes(kv.disk_budget) << " e=" << kv.disk_entries;
        }
        out << " hit=" << saturating_add(kv.restores_ram, kv.restores_disk)
            << " spill=" << kv.spills
            << " evict=" << saturating_add(kv.evictions_lru, kv.evictions_value);
        if (kv.gate_declines != 0) { out << " gate=" << kv.gate_declines; }
    }

    out << " up=" << format_clock(snapshot.uptime_seconds);
    return out.str();
}

StatusLineReporter::StatusLineReporter(GenerationService& service, StatusLineOptions options)
    : service_(service), options_(options), last_decode_time_(Clock::now()),
      started_at_(Clock::now()), last_log_(Clock::now()) {}

StatusLineReporter::~StatusLineReporter() { stop(); }

void StatusLineReporter::start() {
    if (!options_.enabled || started_) { return; }
    {
        std::lock_guard lock(mutex_);
        stopping_ = false;
    }
    started_ = true;
    thread_  = std::thread([this] { run(); });
}

void StatusLineReporter::stop() {
    if (!started_) {
        clear_console_status_line();
        return;
    }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
    started_ = false;
    clear_console_status_line();
}

void StatusLineReporter::run() {
    using namespace std::chrono_literals;
    const std::chrono::milliseconds tick =
        options_.interactive
            ? options_.interval
            : std::min(options_.interval, options_.log_interval);
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        if (cv_.wait_for(lock, std::max(tick, 100ms), [this] { return stopping_; })) { break; }
        lock.unlock();
        try {
            render_once();
        } catch (...) {
            // The status line is diagnostics; never let a formatting failure take down the server.
        }
        lock.lock();
    }
}

void StatusLineReporter::render_once() {
    const Clock::time_point now       = Clock::now();
    ninfer::RuntimeStats stats        = service_.runtime_stats();
    const auto live                   = service_.live_status();
    StatusSnapshot snapshot;
    snapshot.stats = stats;

    if (stats.prefill_request_id != 0) {
        snapshot.prefill = live->find_by_engine(stats.prefill_request_id);
    }
    for (const LiveRequestSnapshot& entry : live->snapshot_active()) {
        if (entry.engine_id == 0 && entry.media_total != 0) {
            snapshot.preparing = entry;
            break;
        }
    }

    const double decode_dt = std::chrono::duration<double>(now - last_decode_time_).count();
    if (decode_dt > 0.001 && stats.committed_decode_tokens >= last_decode_tokens_) {
        const double instantaneous =
            static_cast<double>(stats.committed_decode_tokens - last_decode_tokens_) / decode_dt;
        decode_rate_ = 0.3 * instantaneous + 0.7 * decode_rate_;
        if (instantaneous == 0.0) { decode_rate_ *= 0.7; }
    }
    last_decode_tokens_ = stats.committed_decode_tokens;
    last_decode_time_   = now;

    snapshot.decode_rate_tok_s = decode_rate_;
    snapshot.uptime_seconds     = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(now - started_at_).count());

    std::string line = format_status_line(snapshot, !options_.interactive);
    if (options_.interactive) {
        const std::size_t width = terminal_width();
        if (line.size() > width) { line.resize(width); }
        set_console_status_line(line);
    } else if (now - last_log_ >= options_.log_interval) {
        write_console_log(ConsoleLogLevel::Info, line);
        last_log_ = now;
    }
}

} // namespace ninfer::serve
