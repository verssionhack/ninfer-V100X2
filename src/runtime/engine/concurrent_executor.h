#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options, const DeviceContext& device)
        : instance_(instance), device_(device.device), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          prefill_slice_tokens_(options.prefill_slice_tokens),
          prefill_decode_rounds_(options.prefill_decode_rounds),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          keepalive_interval_(std::chrono::milliseconds(options.stream_keepalive_ms)),
          admission_capacity_(instance.program->admission_capacity()) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            pending_timeout_.count() < 0 || options.max_pending_requests == 0 ||
            options.prefill_decode_rounds == 0 || options.prefill_decode_rounds > 64) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        // Every GPU launch of this executor is issued from worker_, so the thread must be
        // bound to the Engine's DeviceContext device. Without this the worker runs with the
        // CUDA default device (ordinal 0) current while the streams, allocations and
        // per-device kernel attributes belong to device_.
        worker_ = std::thread([this] {
            CUDA_CHECK(cudaSetDevice(device_));
            worker_loop();
        });
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        // Engine-local id assigned at submission. Survives until wait() consumes the request;
        // 0 for a default-constructed handle or the zero-output immediate path.
        [[nodiscard]] std::uint64_t request_id() const noexcept {
            return request_ ? request_->id : 0;
        }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = admission_deadline(submitted);
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

    // Debug-only, OFF by default (see ProgramImplCore::logits_capture): the raw BF16 bits of the
    // full-vocabulary logits behind the most recently sampled token. Empty unless capture was
    // enabled; valid only immediately after a generate()/wait() call returns to the calling thread
    // on a single-lane engine -- there is no per-request association, so concurrent lanes would
    // make this meaningless. Added for the greedy tp1-vs-tp2 parity harness
    // (tools/tp2/parity.cpp); not wire-facing.
    [[nodiscard]] std::vector<std::uint16_t> debug_last_round_logits_bf16() const {
        std::scoped_lock lock(execution_mutex_);
        const std::span<const std::uint16_t> bits = instance_.program->last_round_logits_bf16();
        return std::vector<std::uint16_t>(bits.begin(), bits.end());
    }

    // Takes the execution mutex, so it cannot land between a round's capture and its read.
    void debug_enable_logit_capture(bool enabled) {
        std::scoped_lock lock(execution_mutex_);
        instance_.program->enable_logits_capture(enabled);
    }

    // Debug-only, OFF by default (see ProgramImplCore::check_peer_mtp_egress): cross-rank
    // comparison of the MTP egress record after every speculative round at tp == 2. Takes the
    // execution mutex for the same reason as the capture toggle above.
    void debug_enable_peer_egress_check(bool enabled) {
        std::scoped_lock lock(execution_mutex_);
        instance_.program->enable_peer_egress_check(enabled);
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> debug_peer_egress_check_counts() const {
        std::scoped_lock lock(execution_mutex_);
        return {instance_.program->peer_egress_check_rounds(),
                instance_.program->peer_egress_check_mismatches()};
    }

private:
    // Absolute admission deadline measured from lifetime acquisition. A zero timeout disables it:
    // the request waits behind a prefill owner for as long as the Engine needs. The deadline is
    // never reset while the request waits, so a nonzero value is a hard bound either way.
    [[nodiscard]] Clock::time_point admission_deadline(Clock::time_point from) const noexcept {
        return pending_timeout_.count() > 0 ? from + pending_timeout_ : Clock::time_point::max();
    }

    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        if (prefill_lane_) {
            if (const std::shared_ptr<Request>& active = slots_[*prefill_lane_];
                active != nullptr) {
                snapshot.prefill_prompt_tokens     = active->prompt_summary.prompt_tokens;
                snapshot.prefill_processed_tokens  = active->prefill_processed_tokens;
                snapshot.prefill_start_tokens      = active->prefill_start_tokens;
                snapshot.prefill_request_id        = active->id;
                snapshot.prefill_tokens_per_second = active->prefill_tokens_per_second;
                snapshot.prefill_vision_done =
                    active->prefill_vision_done.load(std::memory_order_relaxed);
                snapshot.prefill_vision_total =
                    active->prefill_vision_total.load(std::memory_order_relaxed);
                snapshot.prefill_vision_swap =
                    active->prefill_vision_swap.load(std::memory_order_relaxed) ? 1U : 0U;
                snapshot.prefill_reuse_path = active->begin
                    ? static_cast<std::uint8_t>(active->begin->prefix_reuse_path) : 0U;
            }
        }
        snapshot.kv_park = instance_.program->kv_park_stats();
        std::uint32_t decode_count = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            const Request& request = *slots_[lane];
            if (!request.decode_ready) { continue; }
            ++snapshot.decode_ready_requests;
            if (decode_count < kMaximumConcurrency) {
                // generated + remaining is the request's original effective output budget, so the
                // status bar is generated/budget and can never run past 100%.
                const std::uint32_t generated = static_cast<std::uint32_t>(request.generated.size());
                const std::uint32_t remaining = request.budget ? request.budget->remaining() : 0;
                snapshot.decode_requests[decode_count] = DecodeRequestStats{
                    .engine_id        = request.id,
                    .generated_tokens = generated,
                    .budget_tokens    = generated + remaining,
                };
                ++decode_count;
            }
        }
        snapshot.decode_request_count = decode_count;
        // Publish the instant decode rate only while a lane is decoding; when none are, cold the
        // window too so a later decode resumes measuring from a fresh window, not a stale one.
        if (snapshot.decode_ready_requests == 0) {
            decode_rate_window_active_ = false;
            decode_rate_window_tokens_ = 0;
            decode_tokens_per_second_  = 0.0;
        }
        snapshot.decode_tokens_per_second =
            snapshot.decode_ready_requests > 0 ? decode_tokens_per_second_ : 0.0;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        Clock::time_point last_activity = Clock::now();
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }
            if (!events.empty()) { last_activity = Clock::now(); }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            // While the request produces no output -- a long prefill, or time queued behind
            // another request's prefill -- hand the consumer a periodic liveness ping on this
            // same thread so it can keep the transport warm and, just as importantly, notice a
            // client that has already gone away. A throw here is handled like a publish throw.
            if (caller_error == nullptr && sink != nullptr && keepalive_interval_.count() > 0 &&
                Clock::now() - last_activity >= keepalive_interval_) {
                try {
                    sink->keepalive();
                    last_activity = Clock::now();
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time)
            : id(request_identity), prompt(std::move(input)), output(std::move(output_session)),
              prompt_summary(summary), prepare_seconds(frontend_seconds),
              options(std::move(request_options)), deadline(limit), submitted(submit_time) {}

        const std::uint64_t id;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        // Vision encode progress of this request's prefill, written by the worker after each
        // prefill step and read by the stats publisher.
        std::atomic<std::uint32_t> prefill_vision_done{0};
        std::atomic<std::uint32_t> prefill_vision_total{0};
        // Whether this request's prefill encodes vision via the KV-swap path; written by the
        // worker, read by the stats publisher.
        std::atomic<bool> prefill_vision_swap{false};
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        // Last tick at which admission was deferred for a disk-backed parked restore. The time
        // spent loading is excluded from the queue deadline so a legitimate image read does not
        // trip `--pending-timeout-ms`.
        std::optional<Clock::time_point> restore_wait_last;
        bool decode_ready = false;
        // True when this request was the protected head of the admission queue: it could not fit
        // beside the active lanes, so the executor drained them and admitted it with the pool.
        // Its prefill takes whole chunks instead of the decode-friendly slice.
        bool exclusive_prefill = false;
        // Prompt tokens of this request made resident so far (reused prefix + prefill work), a
        // live progress readout for the runtime-stats snapshot while it owns the prefill lane.
        std::uint32_t prefill_processed_tokens = 0;
        // Fixed seed for the run: the token position the prefill work begins (the reused prefix).
        // `prefill_processed_tokens` starts here; the new-work window is [start, prompt length).
        std::uint32_t prefill_start_tokens = 0;
        // Instant prefill throughput: processed tokens divided by the last lane step's own wall
        // time, stored unsmoothed. Worker-owned; published in the stats snapshot while the
        // request owns the lane. Reused-prefix tokens are resident before the first step and are
        // not part of any step, so they do not count.
        double prefill_tokens_per_second = 0.0;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
        // The head could not start yet, but nothing failed: it is waiting for its parked image.
        // The caller may admit a backfill candidate while the load runs.
        Deferred,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->retain_cancelled_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        OutputDecision decision;
        try {
            decision = request->output.preview(tokens, request->budget->remaining(),
                                               request->budget->limit_reason());
            if (decision.accepted_tokens != 1) {
                throw std::logic_error("prefill output policy did not accept its licensed token");
            }
        } catch (...) {
            // Request scope: detokenization and the stop/channel policy of ONE OutputSession, run
            // before any Program state moved. See fail_request_scope().
            fail_request_scope(request, lane, std::current_exception());
            return true;
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        try {
            request->budget->commit(1);
            auto published = request->output.commit_preview();
            if (!request->first_token) { request->first_token = Clock::now(); }
            append_output(request, std::move(published));
        } catch (...) {
            // Request scope: publication of ONE request's own text and budget.
            fail_request_scope(request, lane, std::current_exception());
            return true;
        }
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    // A failure whose blast radius is exactly one request: its OutputSession (byte-level
    // detokenization, stop-string/channel policy, published deltas) and its own budget
    // accounting. Nothing about the device, the shared KV pool or another lane is implicated, so
    // the lane is torn down, that one request errors, and the Engine stays serviceable. Failures
    // raised by the Program itself -- CUDA errors from decode_batch/advance_prefill_lane, batch
    // resolution refusing an inconsistent membership -- are NOT routed here: they leave the
    // per-request try blocks untouched, reach worker_loop, and still fail_all.
    //
    // The caller must not touch this request or lane again except to drop the slot.
    void fail_request_scope(const std::shared_ptr<Request>& request, std::uint32_t lane,
                            std::exception_ptr error) {
        instance_.program->abort_lane(lane);
        if (prefill_lane_ && *prefill_lane_ == lane) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        complete_error(request, std::move(error));
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0) { return; }
        if (work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            // A cancelled decode stopped at a resolved round boundary, so its KV and checkpoint
            // are exactly what the conversation's next request can resume from. Retain that
            // state (a cancelled staged prefill is still cleared); dropping it here is what
            // made an ESC during generation cost a full re-prefill.
            instance_.program->retain_cancelled_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    // Record this request's instant prefill speed: the tokens the step processed divided by the
    // step's own wall time. It is stored as-is -- an instant reading, not smoothed -- so the
    // status line tracks the real rate rather than lagging behind it. A reader differencing
    // `prefill_processed_tokens` instead sees progress land in chunk-sized jumps and can only
    // recover the chunk size, not the rate. Steps shorter than the floor are skipped: their time
    // is call overhead, not prefill work.
    static void account_prefill_rate(Request& request, std::uint32_t tokens,
                                     Clock::time_point step_started,
                                     Clock::time_point step_finished) noexcept {
        const double seconds = std::chrono::duration<double>(step_finished - step_started).count();
        constexpr double kMinimumStepSeconds = 0.001;
        if (tokens == 0 || !(seconds >= kMinimumStepSeconds)) { return; }
        const double rate = static_cast<double>(tokens) / seconds;
        if (!std::isfinite(rate) || rate <= 0.0) { return; }
        request.prefill_tokens_per_second = rate;
    }

    // Record this round's contribution to the decode rate over a short wall-clock window. The
    // window divides committed tokens by the actual time elapsed on the worker between rounds --
    // which includes the per-round bookkeeping (detokenization, batch resolution, the gap to the
    // next round) that a decode_batch-only timing excludes. That is what makes the published
    // number the real sustained decode speed instead of an unrealistically high instantaneous
    // spike. Tokens accumulate until the window elapses, then the rate is recomputed and the
    // window restarts. Worker-owned.
    void account_decode_rate(std::uint32_t tokens, Clock::time_point now) noexcept {
        if (tokens == 0) { return; }
        decode_rate_window_tokens_ += tokens;
        if (!decode_rate_window_active_) {
            decode_rate_window_active_ = true;
            decode_rate_window_start_  = now;
            return;
        }
        const double seconds = std::chrono::duration<double>(now - decode_rate_window_start_).count();
        constexpr double kDecodeRateWindowSeconds = 0.25;
        if (seconds < kDecodeRateWindowSeconds) { return; }
        const double rate = static_cast<double>(decode_rate_window_tokens_) / seconds;
        if (std::isfinite(rate) && rate > 0.0) { decode_tokens_per_second_ = rate; }
        decode_rate_window_tokens_ = 0;
        decode_rate_window_start_  = now;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        request->prefill_processed_tokens += step.processed_prompt_tokens;
        consume_service_work(request, step.service_units);
        // The resume decision is fixed when start_prefill_lane consumes the plan and does not
        // change across steps, so record it before the completion/cancel branches: a request
        // cancelled mid-prefill still reports the prefix it actually resumed from instead of
        // the FullReset default.
        request->begin = step.summary;
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            // A cancelled staged prefill has already materialized a KV prefix, so keep it (with
            // the hidden state of its last processed token) and let the conversation's retry
            // resume there instead of re-prefilling the whole prompt. A final step leaves the
            // same continuation state a resolved round would; a partial one is kept only when it
            // can be truncated exactly. Everything else is dropped as before.
            if (step.complete) {
                instance_.program->retain_cancelled_lane(lane);
            } else if (!instance_.program->retain_cancelled_prefill_lane(lane)) {
                instance_.program->abort_lane(lane);
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    // A slice only matters while some other lane has output to make: with no decoder to feed,
    // short steps would just add launch overhead to the same work.
    [[nodiscard]] std::uint32_t prefill_step_tokens() const noexcept {
        if (prefill_slice_tokens_ == 0) { return 0; }
        // The protected head runs the old whole-chunk prefill: slicing it would let a short
        // backfill's decode monopolize the GPU between 256-token steps, which is exactly the
        // starvation this path exists to avoid.
        if (prefill_lane_ && slots_[*prefill_lane_] != nullptr &&
            slots_[*prefill_lane_]->exclusive_prefill) {
            return 0;
        }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr && slots_[lane]->decode_ready) {
                return prefill_slice_tokens_;
            }
        }
        return 0;
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const Clock::time_point step_started = Clock::now();
        const PrefillStepResult step =
            instance_.program->advance_prefill_lane(lane, prefill_step_tokens());
        const Clock::time_point step_finished = Clock::now();
        const auto vision                     = instance_.program->prefill_vision_progress(lane);
        request->prefill_vision_done.store(vision.first, std::memory_order_relaxed);
        request->prefill_vision_total.store(vision.second, std::memory_order_relaxed);
        request->prefill_vision_swap.store(
            instance_.program->prefill_vision_swap(lane), std::memory_order_relaxed);
        account_prefill_rate(*request, step.processed_prompt_tokens, step_started, step_finished);
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        bool selected_retained       = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            // Starting on a retained lane parks its conversation first. Parking copies the whole
            // KV to host and may have to spill it to disk, so when two lanes give the same reuse,
            // take the one that is actually empty: a parked restore must not evict a conversation
            // it did not need to.
            const bool retained = instance_.program->has_retained_lane(lane);
            if (instance_.program->can_admit_lane(lane, plan) &&
                (!selected || reuse > selected_reuse ||
                 (reuse == selected_reuse && selected_retained && !retained))) {
                selected          = LaneChoice{.lane = lane};
                selected_reuse    = reuse;
                selected_retained = retained;
            }
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            // A plan that borrows another lane's prefix cannot survive evicting that lane; wait
            // for capacity instead.
            if (plan.summary().protected_lane != kNoProtectedLane) { continue; }
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                (!selected || reuse > selected_reuse)) {
                selected = LaneChoice{
                    .lane           = lane,
                    .evict_retained = true,
                };
                selected_reuse = reuse;
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        if (choice.evict_retained) {
            const std::uint32_t protected_lane =
                request->lane_plans[lane]->summary().protected_lane;
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && retained_lane != protected_lane &&
                    slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    instance_.program->evict_retained_lane(retained_lane);
                    invalidate_lane_plans(retained_lane);
                }
            }
            if (!instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                // Shared prefixes make the reclaim estimate order-dependent (evicting a source
                // can leave its borrowed pages mapped until their holder is replaced). Retry the
                // request rather than treating a conservative plan as an engine fault; the
                // evictions already performed are parked states, and the next pass sees the
                // updated pool.
                return AdmissionProgress::ControlProgress;
            }
        }
        // A disk-backed parked restore loads on the background reader. Hold the request in the
        // queue until its image is in hand instead of blocking the engine on the read; the other
        // lanes keep decoding. The gate runs AFTER the retained-eviction block: parking another
        // lane can spill the RAM-tier entry this plan names, and the re-check here turns that
        // into a re-plan instead of an engine-fatal "no longer stored".
        switch (instance_.program->gate_parked_restore(*request->lane_plans[lane])) {
        case Program::ParkedRestoreGate::Proceed:
            request->restore_wait_last.reset();
            break;
        case Program::ParkedRestoreGate::Loading: {
            const Clock::time_point now = Clock::now();
            if (request->restore_wait_last && request->deadline != Clock::time_point::max()) {
                request->deadline += now - *request->restore_wait_last;
            }
            request->restore_wait_last = now;
            return AdmissionProgress::Deferred;
        }
        case Program::ParkedRestoreGate::Unavailable:
            // The planned image can no longer be loaded; re-plan against the remaining state
            // instead of spinning on it (the planner skips failed ids).
            request->restore_wait_last.reset();
            request->lane_plans[lane].reset();
            return AdmissionProgress::Deferred;
        }
        // A cross-lane share plan names another lane's state, and that lane keeps running while
        // this request waits. Re-validate the source here: if it moved on, re-plan instead of
        // letting start_prefill_lane throw the engine-fatal "source lane is no longer available".
        if (instance_.program->gate_share_source(*request->lane_plans[lane], request->prompt) ==
            Program::ParkedRestoreGate::Unavailable) {
            request->restore_wait_last.reset();
            request->lane_plans[lane].reset();
            return AdmissionProgress::Deferred;
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        // The protected head was admitted after the drain; it owns the pool and should prefill
        // with whole chunks. A normal head or a backfill candidate keeps the concurrent path.
        request->exclusive_prefill =
            protection_ && protection_->head_request_id == request->id;
        clear_protection_if_head(request);

        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            // A reused prefix is already resident, so progress starts there and the prefill work
            // below carries it to the full prompt length. Both the seed (start) and the running
            // count begin at the reused token count; the new-work window is [start, prompt).
            request->prefill_start_tokens     = summary.reusable_prompt_tokens;
            request->prefill_processed_tokens = summary.reusable_prompt_tokens;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                try {
                    instance_.request_memory.activate(summary.transient_bytes,
                                                      summary.transient_alignment);
                } catch (...) {
                    // The transient is sized per request when the Vision budget is unlimited. A
                    // request whose output buffer does not fit fails on its own; it must not reach
                    // the engine-scope handler below, which stops the worker.
                    const std::exception_ptr error = std::current_exception();
                    instance_.request_memory.deactivate();
                    slots_[lane].reset();
                    invalidate_lane_plans(lane);
                    complete_error(request, error);
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                }
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            target_started                        = true;
            const Clock::time_point first_started = Clock::now();
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient,
                prefill_step_tokens());
            const Clock::time_point first_finished = Clock::now();
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            account_prefill_rate(*request, first.processed_prompt_tokens, first_started,
                                 first_finished);
            request->prefill_vision_swap.store(
                instance_.program->prefill_vision_swap(lane), std::memory_order_relaxed);
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                const AdmissionProgress progress =
                    admit_planned_request(head, *head_lane, BackfillClass::None, 0);
                // A deferred head is waiting for its parked image, not for capacity: leave it in
                // the queue and serve the running lanes. It must not enter the protected-head
                // machinery below, which assumes a capacity-blocked head.
                if (progress == AdmissionProgress::Deferred) {
                    return AdmissionProgress::ControlProgress;
                }
                return progress;
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    const AdmissionProgress progress =
                        admit_planned_request(candidate, *candidate_lane, backfill,
                                              protection_->epoch_id);
                    // A backfill candidate waiting for its parked image is just a retry: the
                    // worker loop must see control progress, not the internal Deferred value.
                    return progress == AdmissionProgress::Deferred
                               ? AdmissionProgress::ControlProgress
                               : progress;
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        std::array<std::array<std::uint64_t, 2>, kMaximumConcurrency> previous_speculative{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const auto& stats = instance_.program->speculative_stats_lane(lanes[row]);
            previous_speculative[row] = {stats.drafted_tokens, stats.accepted_tokens};
        }
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());
        // These are verification counters, not output-publication counters. Include
        // work already performed even if a stop or cancellation truncates the round.
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const auto& current = instance_.program->speculative_stats_lane(lanes[row]);
            cumulative_stats_.speculative_drafted_tokens +=
                current.drafted_tokens - previous_speculative[row][0];
            cumulative_stats_.speculative_accepted_tokens +=
                current.accepted_tokens - previous_speculative[row][1];
        }

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        // A row whose own output session failed. The batch still has to be resolved for every
        // lane -- the Program's pending state is per-round, not per-row -- so such a row is
        // resolved as if it had been cancelled (accepted 0; the Program keeps the pre-round
        // continuation when it can rewind) and the request is errored afterwards, once the shared
        // batch state is consistent again.
        std::array<std::exception_ptr, kMaximumConcurrency> row_errors{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            try {
                // Request scope: detokenization and the stop/channel policy of ONE OutputSession.
                // See fail_request_scope().
                const OutputDecision decision = request->output.preview(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count)) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                accepted[row]       = decision.accepted_tokens;
                terminal[row]       = decision.finished() ? 1 : 0;
                finish_reasons[row] = decision.finish_reason;
            } catch (...) {
                // Resolved exactly like a cancelled row -- accepted 0 with the cancelled flag set,
                // which is the membership resolve_pending_batch routes to clear_lane. `terminal`
                // is deliberately left 0: the Program reads it only on the !cancelled branch, so
                // setting it here would be inert and would misdescribe the row.
                row_errors[row]     = std::current_exception();
                accepted[row]       = 0;
                finish_reasons[row] = FinishReason::None;
                cancelled[row]      = 1;
            }
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            // Held by value: dropping the slot below must not invalidate the handle a failure
            // path still needs.
            const std::shared_ptr<Request> request = slots_[lane];
            if (row_errors[row] != nullptr) {
                // resolve_pending_batch already cleared the lane for this row.
                complete_error(request, row_errors[row]);
                remove_completed_slot(lane);
                continue;
            }
            // ENGINE scope, deliberately outside the fence below: consume_service_work draws down
            // the service projection the admission policy froze for every active request, and it
            // throws when the Program licensed more work than the plan reserved. That divergence
            // invalidates the protection/backfill arithmetic for the whole active set, not just
            // this request, and the drawdown cannot be skipped for one row without leaving the
            // shared accounting overstated for as long as the lane lives.
            if (!cancelled[row]) { consume_service_work(request, accepted[row]); }
            try {
                if (!cancelled[row]) {
                    const auto row_tokens = round.tokens.subspan(
                        row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                    request->generated.insert(request->generated.end(), row_tokens.begin(),
                                              row_tokens.end());
                    request->budget->commit(accepted[row]);
                }
                auto published = request->output.commit_preview();
                if (!request->first_token && accepted[row] != 0) {
                    request->first_token = Clock::now();
                }
                append_output(request, std::move(published));
                if (terminal[row]) {
                    complete_success(request, finish_reasons[row]);
                    remove_completed_slot(lane);
                }
            } catch (...) {
                // Request scope: publication of ONE request's own text and generation budget,
                // after its batch row was already resolved.
                fail_request_scope(request, lane, std::current_exception());
                if (slots_[lane] != nullptr) { remove_completed_slot(lane); }
            }
        }
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        std::uint32_t committed_this_round = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) {
                committed_this_round += accepted[row];
                cumulative_stats_.committed_decode_tokens += accepted[row];
            }
        }
        account_decode_rate(committed_this_round, Clock::now());
        publish_runtime_stats();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    if (!failed_) {
                        // Graceful shutdown: keep retained conversations (and the resident lane)
                        // in the spill directory so the next process can resume them.
                        try { instance_.program->persist_state(); } catch (...) {}
                    }
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                instance_.program->pump_parked_io();
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                if (prefill_lane_) {
                    // A prefill step has a large fixed cost, so let the decoder run a configured
                    // number of rounds between steps; the step itself is bounded by the slice.
                    if (!membership.empty() &&
                        decode_rounds_since_prefill_ < prefill_decode_rounds_) {
                        run_decode_round(membership);
                        ++decode_rounds_since_prefill_;
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        decode_rounds_since_prefill_ = 0;
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        // A request waiting on a parked-image load cannot be admitted this tick,
                        // and there is no decode work to run. Wait briefly instead of spinning;
                        // the condition variable still wakes immediately on a new submission.
                        if (instance_.program->parked_io_pending()) {
                            std::unique_lock lock(queue_mutex_);
                            queue_cv_.wait_for(lock, std::chrono::milliseconds(5),
                                               [&] { return stopping_; });
                        }
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
            } catch (...) {
                // Engine scope only. Everything whose blast radius is a single request is caught
                // and completed at its own site (fail_request_scope for output/publication,
                // remove_pending_error for planning and admission-time refusals), so what reaches
                // here is a failure of the shared execution unit -- a CUDA error, a Program state
                // inconsistency, an exhausted device allocation, or a service-projection drawdown
                // that the Program's licensing contradicts. admit_planned_request cleans its
                // lane up and rethrows on purpose: its region is dominated by start_prefill_lane,
                // a device execution unit. The Engine cannot keep serving after one of those:
                // every in-flight and queued request fails and the worker stops.
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const int device_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::uint32_t prefill_slice_tokens_;
    const std::uint32_t prefill_decode_rounds_;
    const std::chrono::milliseconds pending_timeout_;
    const std::chrono::milliseconds keepalive_interval_;
    const AdmissionResources admission_capacity_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::uint32_t decode_rounds_since_prefill_ = 0;
    // Instant decode throughput: committed tokens over a short wall-clock window (see
    // account_decode_rate), so it reflects the sustained decode speed including per-round
    // bookkeeping, not a decode_batch-only spike. Worker-owned; published only while a lane is
    // decode-ready (and reset to a cold window there), so an idle engine reads 0.
    double decode_tokens_per_second_ = 0.0;
    std::uint64_t decode_rate_window_tokens_ = 0;
    Clock::time_point decode_rate_window_start_{};
    bool decode_rate_window_active_ = false;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
