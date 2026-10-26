#pragma once

// TypeSafe / OpenJev `/v1/systemone` wire layer. It is model-independent: the request is
// validated and normalized here, and the protocol service renders it through whatever chat
// template the loaded artifact carries. No decision metadata is read from the model.

#include "product/media_acquire/source.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

enum class SystemoneQuestionType : std::uint8_t {
    Choice,
    Score,
    Noul,
};

struct SystemoneOption {
    std::string key;
    nlohmann::json description; // null when the request omitted it
};

struct SystemoneQuestion {
    std::string id;
    SystemoneQuestionType type = SystemoneQuestionType::Choice;
    nlohmann::json instructions;
    std::vector<SystemoneOption> options;
};

struct SystemoneRequest {
    // The value to evaluate. Image parts inside a state made of chat messages are removed and
    // appended to `images` in encounter order.
    nlohmann::json state;
    std::vector<ninfer::product::media_acquire::Source> images;
    std::vector<SystemoneQuestion> questions;
    // Extension over the reference API: softmax temperature for the option probabilities.
    // 1.0 keeps the raw model distribution; larger values flatten it.
    float temperature = 1.0F;
    std::optional<std::string> model;
};

// Throws ApiException (400) on malformed or unsupported input.
[[nodiscard]] SystemoneRequest parse_systemone_request(const nlohmann::json& body);

[[nodiscard]] nlohmann::json make_systemone_response(std::string model,
                                                     nlohmann::json answers,
                                                     std::int64_t input_tokens);

} // namespace ninfer::serve
