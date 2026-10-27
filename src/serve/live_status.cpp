#include "serve/live_status.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ninfer::serve {

void LiveRequestRegistry::begin(std::uint64_t http_id, std::size_t media_total,
                                std::string protocol, bool stream) {
    std::lock_guard lock(mutex_);
    if (capacity_ != 0 && entries_.size() >= capacity_) { return; }
    LiveRequestSnapshot entry;
    entry.http_id      = http_id;
    entry.protocol     = std::move(protocol);
    entry.stream       = stream;
    entry.media_total  = static_cast<std::uint32_t>(media_total);
    entry.accepted     = std::chrono::steady_clock::now();
    entries_[http_id]  = std::move(entry);
}

void LiveRequestRegistry::bind_engine(std::uint64_t http_id, std::uint64_t engine_id,
                                      std::uint32_t prompt_tokens, std::uint64_t fingerprint) {
    std::lock_guard lock(mutex_);
    const auto found = entries_.find(http_id);
    if (found == entries_.end()) { return; }
    found->second.engine_id     = engine_id;
    found->second.prompt_tokens = prompt_tokens;
    found->second.fingerprint   = fingerprint;
    if (engine_id != 0) { engine_index_[engine_id] = http_id; }
}

void LiveRequestRegistry::media_tick(std::uint64_t http_id, MediaPhase phase, std::size_t done,
                                     std::size_t total) {
    std::lock_guard lock(mutex_);
    const auto found = entries_.find(http_id);
    if (found == entries_.end()) { return; }
    LiveRequestSnapshot& entry = found->second;
    entry.media_phase          = phase;
    if (total != 0) { entry.media_total = static_cast<std::uint32_t>(total); }
    constexpr std::size_t kMaximum = std::numeric_limits<std::uint32_t>::max();
    entry.media_done = static_cast<std::uint32_t>(std::min(done, kMaximum));
}

void LiveRequestRegistry::finish(std::uint64_t http_id) {
    std::lock_guard lock(mutex_);
    const auto found = entries_.find(http_id);
    if (found == entries_.end()) { return; }
    if (found->second.engine_id != 0) { engine_index_.erase(found->second.engine_id); }
    entries_.erase(found);
}

std::optional<LiveRequestSnapshot>
LiveRequestRegistry::find_by_engine(std::uint64_t engine_id) const {
    if (engine_id == 0) { return std::nullopt; }
    std::lock_guard lock(mutex_);
    const auto index = engine_index_.find(engine_id);
    if (index == engine_index_.end()) { return std::nullopt; }
    const auto found = entries_.find(index->second);
    if (found == entries_.end()) { return std::nullopt; }
    return found->second;
}

std::vector<LiveRequestSnapshot> LiveRequestRegistry::snapshot_active() const {
    std::lock_guard lock(mutex_);
    std::vector<LiveRequestSnapshot> snapshot;
    snapshot.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        (void)id;
        snapshot.push_back(entry);
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const LiveRequestSnapshot& a,
                                                   const LiveRequestSnapshot& b) {
        return a.accepted < b.accepted;
    });
    return snapshot;
}

std::size_t LiveRequestRegistry::size() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

LiveRequestLease::LiveRequestLease(std::shared_ptr<LiveRequestRegistry> registry,
                                   std::uint64_t http_id, std::size_t media_total,
                                   std::string protocol, bool stream)
    : registry_(std::move(registry)), http_id_(http_id) {
    if (registry_) { registry_->begin(http_id_, media_total, std::move(protocol), stream); }
}

LiveRequestLease::~LiveRequestLease() { release(); }

LiveRequestLease::LiveRequestLease(LiveRequestLease&& other) noexcept
    : registry_(std::move(other.registry_)), http_id_(other.http_id_) {
    other.http_id_ = 0;
}

LiveRequestLease& LiveRequestLease::operator=(LiveRequestLease&& other) noexcept {
    if (this != &other) {
        release();
        registry_      = std::move(other.registry_);
        http_id_       = other.http_id_;
        other.http_id_ = 0;
    }
    return *this;
}

void LiveRequestLease::release() noexcept {
    if (registry_ && http_id_ != 0) { registry_->finish(http_id_); }
    registry_.reset();
    http_id_ = 0;
}

} // namespace ninfer::serve
