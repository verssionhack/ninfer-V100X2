#pragma once

// Deterministic background I/O for parked-conversation images.
//
// The serving worker must not block on a multi-gigabyte image write or read, and spill writes must
// not queue ahead of a restore. One FIFO writer thread owns every write (a fixed order, independent
// of how many workers submit) and a separate FIFO reader thread owns loads, so a restore never
// waits behind an unrelated spill. A pending-byte budget bounds the host RAM held by submitted
// payloads, and completions return to the worker, which is the only thread that touches the parked
// index.

#include "targets/qwen3_6/impl/runtime/parked_sequence_store.h"
#include "targets/qwen3_6/impl/runtime/sequence_image_io.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct DiskSpillJob {
    std::unique_ptr<ParkedSequence> entry;
    std::string path;
    // Matching metadata the worker needs after the write to prune superseded disk states.
    ParkedSequence probe;
    std::uint64_t protect_id = 0;
    std::uint64_t charged    = 0;
};

// One parked-image load. The reader thread owns nothing but the copied path, geometry and the
// immutable pool pointers; the loaded image returns to the worker in the completion.
struct DiskReadJob {
    std::uint64_t id = 0;
    std::string path;
    SequenceImageGeometry geometry;
    std::vector<PagedKVPool*> text_pools;
    std::vector<PagedKVPool*> backend_pools;
    std::uint64_t charged = 0;
};

struct DiskReadCompletion {
    std::uint64_t id = 0;
    std::string path;
    std::uint64_t charged = 0;
    // Wall time of the image read, used to seed the restore-cost estimate.
    double elapsed_seconds = 0.0;
    std::unique_ptr<ParkedSequence> entry;
    std::string error;
};

struct DiskSpillCompletion {
    std::unique_ptr<ParkedSequence> entry;
    ParkedSequence probe;
    std::uint64_t protect_id = 0;
    std::uint64_t charged    = 0;
    std::string error;
};

class ParkedDiskIo {
public:
    ParkedDiskIo(SequenceImageGeometry geometry, std::uint64_t pending_budget_bytes)
        : state_(std::make_shared<State>(std::move(geometry), pending_budget_bytes)) {
        state_->thread = std::thread([state = state_] { run(state); });
    }

    ~ParkedDiskIo() { stop(); }

    ParkedDiskIo(const ParkedDiskIo&)            = delete;
    ParkedDiskIo& operator=(const ParkedDiskIo&) = delete;

    // True when submitting `bytes` keeps the pending payload within the configured budget. The
    // first job is always admitted even when it alone exceeds the budget, so the writer never
    // deadlocks on a single large conversation.
    [[nodiscard]] bool has_capacity(std::uint64_t bytes) const {
        std::lock_guard lock(state_->mutex);
        return state_->pending_bytes == 0 ||
               state_->pending_bytes + bytes <= state_->pending_budget_bytes;
    }

    // Blocks until a completed job is waiting to be drained. The caller drains and rechecks
    // capacity; pending bytes are released only by drain().
    void wait_for_change() {
        std::unique_lock lock(state_->mutex);
        state_->cv.wait(lock, [&] {
            return state_->stopping || !state_->completions.empty();
        });
    }

    void submit(DiskSpillJob job) {
        {
            std::lock_guard lock(state_->mutex);
            state_->pending_bytes += job.charged;
            state_->pending_jobs += 1;
            state_->jobs.push_back(std::move(job));
        }
        state_->cv.notify_all();
    }

    // Moves every finished job to the caller. Pending bytes are released here, on the worker,
    // so the budget also bounds RAM held by completions that have not been processed yet.
    void drain(std::vector<DiskSpillCompletion>& out) {
        std::lock_guard lock(state_->mutex);
        for (DiskSpillCompletion& completion : state_->completions) {
            state_->pending_bytes -= std::min(state_->pending_bytes, completion.charged);
            out.push_back(std::move(completion));
        }
        state_->completions.clear();
    }

    // Waits until every submitted image has been written (or failed). Completions may still be
    // outstanding; callers that need them drain afterwards.
    void wait_idle() {
        std::unique_lock lock(state_->mutex);
        state_->cv.wait(lock, [&] { return state_->pending_jobs == 0; });
    }

    void stop() {
        if (!state_) { return; }
        wait_idle();
        {
            std::lock_guard lock(state_->mutex);
            state_->stopping = true;
        }
        state_->cv.notify_all();
        if (state_->thread.joinable()) { state_->thread.join(); }
    }

private:
    struct State {
        State(SequenceImageGeometry image_geometry, std::uint64_t budget)
            : geometry(std::move(image_geometry)), pending_budget_bytes(budget) {}

        SequenceImageGeometry geometry;
        std::uint64_t pending_budget_bytes = 0;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<DiskSpillJob> jobs;
        std::vector<DiskSpillCompletion> completions;
        // Submitted but not yet drained by the worker.
        std::uint64_t pending_bytes = 0;
        std::size_t pending_jobs    = 0;
        bool stopping               = false;
        std::thread thread;
    };

    static void run(const std::shared_ptr<State>& state) {
        for (;;) {
            DiskSpillJob job;
            {
                std::unique_lock lock(state->mutex);
                state->cv.wait(lock, [&] { return state->stopping || !state->jobs.empty(); });
                if (state->jobs.empty()) { return; } // stopping and drained
                job = std::move(state->jobs.front());
                state->jobs.pop_front();
            }

            DiskSpillCompletion completion;
            completion.probe      = std::move(job.probe);
            completion.protect_id = job.protect_id;
            completion.charged    = job.charged;
            try {
                write_sequence_image(*job.entry, job.path);
                try {
                    write_sequence_metadata(*job.entry, job.entry->bytes, state->geometry,
                                            job.path + ".meta");
                } catch (...) {
                    std::error_code ignored;
                    std::filesystem::remove(job.path, ignored);
                    throw;
                }
                completion.entry = std::move(job.entry);
            } catch (const std::exception& error) {
                completion.error = error.what();
                std::error_code ignored;
                std::filesystem::remove(job.path, ignored);
                std::filesystem::remove(job.path + ".meta", ignored);
                completion.entry = std::move(job.entry);
            }

            {
                std::lock_guard lock(state->mutex);
                state->completions.push_back(std::move(completion));
                state->pending_jobs -= 1;
            }
            state->cv.notify_all();
        }
    }

    std::shared_ptr<State> state_;
};

/**
 * Background reader for parked-conversation images.
 *
 * The serving worker must not block for a minute on a multi-gigabyte image read while the other
 * lane could still decode. One FIFO reader thread owns every load; the worker submits a path and
 * drains completed images on its own thread, where ownership and the parked index live. Reads do
 * not share the writer's queue: a spill queued ahead of a restore must not delay it.
 *
 * The reader only parses host bytes. PagedKVImage::from_host consults immutable pool geometry and
 * stores the pool pointer; no CUDA call, no parked-index access, and no Program state escape this
 * thread. The geometry and pool pointers are captured at submit time and outlive the reader.
 */
class ParkedDiskReader {
public:
    explicit ParkedDiskReader(std::uint64_t pending_budget_bytes)
        : state_(std::make_shared<State>(pending_budget_bytes)) {
        state_->thread = std::thread([state = state_] { run(state); });
    }

    ~ParkedDiskReader() { stop(); }

    ParkedDiskReader(const ParkedDiskReader&)            = delete;
    ParkedDiskReader& operator=(const ParkedDiskReader&) = delete;

    // Same rule as the writer: the first job is always admitted, so a single large conversation
    // can never deadlock the reader.
    [[nodiscard]] bool has_capacity(std::uint64_t bytes) const {
        std::lock_guard lock(state_->mutex);
        return state_->pending_bytes == 0 ||
               state_->pending_bytes + bytes <= state_->pending_budget_bytes;
    }

    void submit(DiskReadJob job) {
        {
            std::lock_guard lock(state_->mutex);
            if (state_->stopping) { return; }
            state_->pending_bytes += job.charged;
            state_->pending_jobs += 1;
            state_->jobs.push_back(std::move(job));
        }
        state_->cv.notify_all();
    }

    // Moves every finished load to the worker and releases its pending bytes.
    void drain(std::vector<DiskReadCompletion>& out) {
        std::lock_guard lock(state_->mutex);
        for (DiskReadCompletion& completion : state_->completions) {
            state_->pending_bytes -= std::min(state_->pending_bytes, completion.charged);
            out.push_back(std::move(completion));
        }
        state_->completions.clear();
    }

    [[nodiscard]] bool busy() const {
        std::lock_guard lock(state_->mutex);
        return state_->pending_jobs != 0;
    }

    // Drops queued jobs and joins the one in progress. Like the writer, this can wait for the job
    // already running; an image read is not chunked, so it cannot be interrupted safely.
    void stop() {
        if (!state_) { return; }
        {
            std::lock_guard lock(state_->mutex);
            state_->stopping = true;
            state_->jobs.clear();
        }
        state_->cv.notify_all();
        if (state_->thread.joinable()) { state_->thread.join(); }
    }

private:
    struct State {
        explicit State(std::uint64_t budget) : pending_budget_bytes(budget) {}

        std::uint64_t pending_budget_bytes = 0;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<DiskReadJob> jobs;
        std::vector<DiskReadCompletion> completions;
        std::uint64_t pending_bytes = 0;
        std::size_t pending_jobs    = 0;
        bool stopping               = false;
        std::thread thread;
    };

    static void run(const std::shared_ptr<State>& state) {
        for (;;) {
            DiskReadJob job;
            {
                std::unique_lock lock(state->mutex);
                state->cv.wait(lock, [&] {
                    return state->stopping || !state->jobs.empty();
                });
                if (state->stopping && state->jobs.empty()) { return; }
                job = std::move(state->jobs.front());
                state->jobs.pop_front();
            }

            DiskReadCompletion completion;
            completion.id      = job.id;
            completion.path    = job.path;
            completion.charged = job.charged;
            const auto read_started = std::chrono::steady_clock::now();
            try {
                completion.entry = read_sequence_image(job.path, job.geometry, job.text_pools,
                                                       job.backend_pools);
            } catch (const std::exception& error) {
                completion.error = error.what();
            }
            completion.elapsed_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - read_started)
                    .count();

            {
                std::lock_guard lock(state->mutex);
                state->completions.push_back(std::move(completion));
                state->pending_jobs -= 1;
            }
            state->cv.notify_all();
        }
    }

    std::shared_ptr<State> state_;
};

} // namespace ninfer::targets::qwen3_6::detail
