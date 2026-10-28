#pragma once

#include "serve/generation_service.h"
#include "serve/response_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"
#include "serve/status_line.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ninfer::serve {

// One-shot raw capture of a single agent turn (see --capture-turn).
class TurnCapture;

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    // The process-wide console logger serializes lines from request and reporter threads.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_rejected(const RequestRejectionLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);
    void log_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();

    // Claims the process-wide one-shot capture, if --capture-turn is set and this request is an
    // agent turn (carries tools). Returns nullptr when capture is off, already used, or the file
    // cannot be opened; capture failures never fail the request.
    std::shared_ptr<TurnCapture> begin_turn_capture(const httplib::Request& req,
                                                    const GenerationRequest& request,
                                                    std::uint64_t req_id);

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    // Advertised on GET /v1/models. Clients such as claude-code-best take max_input_tokens as
    // their context window; this is the per-request sequence ceiling the engine admits against.
    std::uint32_t model_max_input_tokens_  = 0;
    std::uint32_t model_max_output_tokens_ = 0;
    ResponseStore response_store_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::atomic<bool> capture_taken_{false};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    std::unique_ptr<StatusLineReporter> status_line_;
};

} // namespace ninfer::serve
