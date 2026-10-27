#pragma once

// Live request bookkeeping for the optional console status line. The HTTP layer owns request ids;
// this registry names the requests currently being prepared or executed so a status reader can
// pair the Engine's prefill lane with an HTTP id, a token count, a prompt fingerprint, and media
// preparation counts. It is observation-only: nothing here influences scheduling or output.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

enum class MediaPhase : std::uint8_t {
    None,
    Acquire,
    Preprocess,
    VisionEncode,
};

struct LiveRequestSnapshot {
    std::uint64_t http_id      = 0;
    std::uint64_t engine_id    = 0;
    std::uint64_t fingerprint  = 0;
    std::string protocol;
    bool stream                = false;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t media_total  = 0;
    std::uint32_t media_done   = 0;
    MediaPhase media_phase     = MediaPhase::None;
    std::chrono::steady_clock::time_point accepted;
};

// Formats the low `chars` of a fingerprint as lowercase hex (clamped to 16).
[[nodiscard]] inline std::string format_fingerprint(std::uint64_t fingerprint,
                                                    std::size_t chars = 8) {
    static constexpr char kDigits[] = "0123456789abcdef";
    chars                            = chars > 16 ? 16 : (chars == 0 ? 1 : chars);
    const std::size_t first          = 16 - chars;
    std::string text(chars, '0');
    for (std::size_t i = 0; i < chars; ++i) {
        const std::size_t nibble = (fingerprint >> (4 * (15 - first - i))) & 0xFULL;
        text[i]                  = kDigits[nibble];
    }
    return text;
}

class LiveRequestRegistry {
public:
    explicit LiveRequestRegistry(std::size_t capacity = 0) : capacity_(capacity) {}

    void begin(std::uint64_t http_id, std::size_t media_total, std::string protocol, bool stream);
    void bind_engine(std::uint64_t http_id, std::uint64_t engine_id, std::uint32_t prompt_tokens,
                     std::uint64_t fingerprint);
    void media_tick(std::uint64_t http_id, MediaPhase phase, std::size_t done, std::size_t total);
    void finish(std::uint64_t http_id);

    [[nodiscard]] std::optional<LiveRequestSnapshot>
    find_by_engine(std::uint64_t engine_id) const;
    [[nodiscard]] std::vector<LiveRequestSnapshot> snapshot_active() const;
    [[nodiscard]] std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, LiveRequestSnapshot> entries_;
    std::unordered_map<std::uint64_t, std::uint64_t> engine_index_;
    std::size_t capacity_ = 0;
};

// RAII registration: destruction removes the entry, so an exception, cancellation, or a client
// disconnect cannot leave a stale request in the status line. Move-only.
class LiveRequestLease {
public:
    LiveRequestLease() noexcept = default;
    LiveRequestLease(std::shared_ptr<LiveRequestRegistry> registry, std::uint64_t http_id,
                     std::size_t media_total, std::string protocol, bool stream);
    ~LiveRequestLease();

    LiveRequestLease(LiveRequestLease&& other) noexcept;
    LiveRequestLease& operator=(LiveRequestLease&& other) noexcept;
    LiveRequestLease(const LiveRequestLease&)            = delete;
    LiveRequestLease& operator=(const LiveRequestLease&) = delete;

    [[nodiscard]] std::uint64_t id() const noexcept { return http_id_; }
    void release() noexcept;

private:
    std::shared_ptr<LiveRequestRegistry> registry_;
    std::uint64_t http_id_ = 0;
};

} // namespace ninfer::serve
