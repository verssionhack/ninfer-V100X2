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

// A "percentage bar": kWidth cells stand for 0..100%, filled left to right with Unicode block
// elements at one-eighth-cell precision so the fill tracks the percentage closely. The fill is the
// DONE fraction (progress); the percentage and done/total count on the right still report the
// work. Cell levels 1..8 are the block elements, and an empty cell is a space.
std::string progress_bar(double percent) {
    constexpr std::size_t kWidth = 10;
    // ▏▎▍▌▋▊▉█ (U+258F..U+2588) as UTF-8 bytes: one-eighth-cell fill levels, least to most.
    const std::string kLevels = "\xE2\x96\x8F\xE2\x96\x8E\xE2\x96\x8D\xE2\x96\x8C"
                                "\xE2\x96\x8B\xE2\x96\x8A\xE2\x96\x89\xE2\x96\x88";
    const double clamped = std::clamp(percent, 0.0, 100.0);
    const int total_eighths =
        static_cast<int>(std::lround(clamped / 100.0 * static_cast<double>(kWidth) * 8.0));
    std::string bar;
    bar.reserve(kWidth * 3);
    for (std::size_t cell = 0; cell < kWidth; ++cell) {
        const int level = std::clamp(total_eighths - static_cast<int>(cell) * 8, 0, 8);
        bar += level == 0 ? " " : kLevels.substr(static_cast<std::size_t>((level - 1) * 3), 3);
    }
    return bar;
}

// The largest prefix of `line` of at most `max_bytes` that ends on a UTF-8 character boundary, so
// truncation never splits one of the wide progress-bar block characters.
std::size_t utf8_prefix_boundary(const std::string& line, std::size_t max_bytes) {
    if (line.size() <= max_bytes) { return line.size(); }
    std::size_t cut = max_bytes;
    // A UTF-8 continuation byte is 0b10xxxxxx; back up past any run so the cut lands on a leading
    // byte (or the start), keeping every emitted character intact.
    while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0U) == 0x80U) { --cut; }
    return cut;
}

// Compact tag for what a decoding request is generating: tools and/or thinking, or plain text.
std::string decode_kind_tag(const LiveRequestSnapshot* request) {
    std::string tag;
    if (request != nullptr) {
        if (request->has_tools) { tag = "tools"; }
        if (request->thinking) { tag += (tag.empty() ? "think" : "+think"); }
    }
    return tag.empty() ? "text" : tag;
}

// The KV switch a prefill performs, as a `PrefixReusePath` value: how it obtains its KV other than
// a fresh prefill (FullReset) or a plain append (AppendAtFrontier). RestoreParkedKv/ShareLanePrefix
// bring back or borrow state that already exists; the checkpoint paths restore from a saved image.
// The no-switch cases render no tag.
std::string prefill_kv_tag(std::uint8_t reuse_path) {
    using ninfer::PrefixReusePath;
    switch (static_cast<PrefixReusePath>(reuse_path)) {
    case PrefixReusePath::RestoreTurnCheckpoint:
        return "turn";
    case PrefixReusePath::RestoreResponseCheckpoint:
        return "resp";
    case PrefixReusePath::RestoreParkedKv:
        return "park";
    case PrefixReusePath::ShareLanePrefix:
        return "share";
    case PrefixReusePath::FullReset:
    case PrefixReusePath::AppendAtFrontier:
        return "";
    }
    return "";
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
        // The prefill bar tracks the *new* work this run does: the reused prefix is already resident, so
        // progress runs from `start` to `total` and reads 0% at the start and 100% at the end even
        // for a resume that began partway. A fresh prompt (start == 0) spans the whole prompt.
        const std::uint32_t total  = stats.prefill_prompt_tokens;
        const std::uint32_t done   = std::min(stats.prefill_processed_tokens, total);
        const std::uint32_t start  = std::min(stats.prefill_start_tokens, total);
        const std::uint32_t work_total = total - start;                     // new tokens to prefill
        const std::uint32_t work_done  = done > start ? done - start : 0;
        const double percent = work_total != 0
            ? 100.0 * static_cast<double>(std::min(work_done, work_total)) / work_total
            : 0.0;
        out << " prefill [" << progress_bar(percent) << "] " << std::fixed
            << std::setprecision(1) << percent << "% " << work_done << '/' << work_total;
        const std::string kv_tag = prefill_kv_tag(stats.prefill_reuse_path);
        if (!kv_tag.empty()) {
            out << " kv=" << kv_tag;
        }
        const double prefill_rate = stats.prefill_tokens_per_second;
        if (prefill_rate > 0.5) {
            out << ' ' << format_rate(prefill_rate) << " t/s";
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
    // One progress bar per decoding request, mirroring the prefill readout: a done/budget bar that
    // fills toward max_tokens, the generated/budget count on the right, the instant decode rate,
    // and a tag for what the request is generating.
    for (std::uint32_t i = 0; i < stats.decode_request_count; ++i) {
        const ninfer::DecodeRequestStats& dreq = stats.decode_requests[i];
        const LiveRequestSnapshot* dreq_snap =
            (i < snapshot.decodes.size() && snapshot.decodes[i].has_value())
                ? &snapshot.decodes[i].value()
                : nullptr;
        if (wrote_work) { out << " | "; }
        out << "decode req=";
        if (dreq_snap != nullptr) {
            out << dreq_snap->http_id;
        } else {
            out << '?';
        }
        const std::uint32_t total = dreq.budget_tokens;
        const std::uint32_t done  = dreq.generated_tokens;
        const double percent =
            total != 0 ? 100.0 * static_cast<double>(std::min(done, total)) / total : 0.0;
        out << " [" << progress_bar(percent) << "] " << std::fixed << std::setprecision(1)
            << percent << "% " << done << '/' << total;
        if (stats.decode_tokens_per_second > 0.5) {
            out << ' ' << format_rate(stats.decode_tokens_per_second) << " t/s";
        }
        out << ' ' << decode_kind_tag(dreq_snap);
        wrote_work = true;
    }
    if (!wrote_work) { out << "idle"; }

    if (stats.prefill_vision_total != 0) {
        out << " vis " << stats.prefill_vision_done << '/' << stats.prefill_vision_total;
        if (stats.prefill_vision_swap != 0) {
            // The vision items are encoded by swapping the persistent KV arena to host, which the
            // operator wants visible next to the count.
            out << " swap";
        }
    }

    out << " | run=" << stats.running_requests << " pref=" << stats.prefilling_requests
        << " ready=" << stats.decode_ready_requests << " wait=" << stats.waiting_requests
        << " d/p=" << format_rate(stats.decode_tokens_per_second) << '/'
        << format_rate(stats.prefill_tokens_per_second) << " t/s";

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
    : service_(service), options_(options), started_at_(Clock::now()), last_log_(Clock::now()) {}

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

    // Resolve each decode-ready request's live snapshot (http id, tools/thinking) by engine id,
    // so the formatter can name the request and tag what it is generating.
    snapshot.decodes.resize(stats.decode_request_count);
    for (std::uint32_t i = 0; i < stats.decode_request_count; ++i) {
        snapshot.decodes[i] = live->find_by_engine(stats.decode_requests[i].engine_id);
    }

    snapshot.uptime_seconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(now - started_at_).count());

    std::string line = format_status_line(snapshot, !options_.interactive);
    if (options_.interactive) {
        const std::size_t width = terminal_width();
        const std::size_t cut = utf8_prefix_boundary(line, width);
        if (cut < line.size()) { line.resize(cut); }
        set_console_status_line(line);
    } else if (now - last_log_ >= options_.log_interval) {
        write_console_log(ConsoleLogLevel::Info, line);
        last_log_ = now;
    }
}

} // namespace ninfer::serve
