#include "serve/systemone_schema.h"

#include "serve/request.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaximumQuestions = 64;
constexpr std::size_t kMaximumOptions   = 52;
constexpr std::size_t kMaximumImages    = 8;
constexpr std::size_t kMinimumScoreLevels = 2;
constexpr std::size_t kMaximumScoreLevels = 10;

[[noreturn]] void bad_request(std::string message, std::string param,
                              std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

void append_image_url(const std::string& url, std::string_view param,
                      std::vector<ninfer::product::media_acquire::Source>& images) {
    if (!url.starts_with("data:")) {
        bad_request("images must be data URLs (data:image/...;base64,...)", std::string(param));
    }
    if (images.size() >= kMaximumImages) {
        bad_request("too many images, the maximum is " + std::to_string(kMaximumImages),
                    std::string(param));
    }
    ninfer::product::media_acquire::Source source;
    source.kind  = ninfer::product::media_acquire::SourceKind::Data;
    source.value = url;
    images.push_back(std::move(source));
}

// "files" and "images" are aliases; both hold data URLs.
void append_image_array(const Json& value, const char* field,
                        std::vector<ninfer::product::media_acquire::Source>& images) {
    if (value.is_null()) { return; }
    if (!value.is_array()) {
        bad_request(std::string("\"") + field + "\" must be an array", field);
    }
    for (const Json& item : value) {
        if (!item.is_string()) {
            bad_request(std::string("\"") + field + "\" must be an array of data URL strings",
                        field);
        }
        append_image_url(item.get<std::string>(), field, images);
    }
}

std::string image_part_url(const Json& part, std::string_view param) {
    if (!part.contains("image_url")) {
        bad_request("image_url content part must contain image_url", std::string(param));
    }
    const Json& value = part.at("image_url");
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_object() && value.contains("url") && value.at("url").is_string()) {
        return value.at("url").get<std::string>();
    }
    bad_request("image_url must be a data URL string or an object containing url",
                std::string(param));
}

Json parse_state(const Json& body, std::vector<ninfer::product::media_acquire::Source>& images) {
    if (body.contains("videos") && !body.at("videos").is_null() && !body.at("videos").empty()) {
        bad_request("\"videos\" is not supported", "videos", "modality_not_supported");
    }
    if (body.contains("files")) { append_image_array(body.at("files"), "files", images); }
    if (body.contains("images")) { append_image_array(body.at("images"), "images", images); }

    if (!body.contains("state") || body.at("state").is_null()) {
        bad_request("\"state\" must be provided", "state");
    }
    const Json& state = body.at("state");
    const bool wrapped = state.is_object() && state.contains("messages");
    const Json& messages = wrapped ? state.at("messages") : state;
    if (!messages.is_array()) { return state; }

    Json messages_out = Json::array();
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const Json& message = messages.at(i);
        if (!message.is_object() || !message.contains("content") ||
            !message.at("content").is_array()) {
            messages_out.push_back(message);
            continue;
        }
        const std::string param = "state.messages." + std::to_string(i) + ".content";
        Json content_out        = Json::array();
        for (const Json& part : message.at("content")) {
            if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
                content_out.push_back(part);
                continue;
            }
            const std::string type = part.at("type").get<std::string>();
            if (type == "image_url") {
                append_image_url(image_part_url(part, param), param, images);
            } else if (type == "input_audio" || type == "video_url") {
                bad_request("'" + type + "' content is not supported for decisions", param,
                            "modality_not_supported");
            } else {
                content_out.push_back(part);
            }
        }
        Json message_out    = message;
        message_out["content"] = std::move(content_out);
        messages_out.push_back(std::move(message_out));
    }

    if (!wrapped) { return messages_out; }
    Json state_out            = state;
    state_out["messages"]     = std::move(messages_out);
    return state_out;
}

std::vector<SystemoneQuestion> parse_questions(const Json& body) {
    if (!body.contains("questions") || !body.at("questions").is_object() ||
        body.at("questions").empty()) {
        bad_request("\"questions\" must be a non-empty object", "questions");
    }
    std::vector<SystemoneQuestion> questions;
    for (auto it = body.at("questions").begin(); it != body.at("questions").end(); ++it) {
        if (questions.size() >= kMaximumQuestions) {
            bad_request("too many questions, the maximum is " + std::to_string(kMaximumQuestions),
                        "questions");
        }
        const std::string id = it.key();
        const std::string param = "questions." + id;
        const Json& value = it.value();
        if (!value.is_object()) { bad_request(param + " must be an object", param); }
        if (!value.contains("instructions") || value.at("instructions").is_null()) {
            bad_request(param + ".instructions must be provided", param + ".instructions");
        }

        SystemoneQuestion question;
        question.id           = id;
        question.instructions = value.at("instructions");
        const std::string type = value.contains("type") && value.at("type").is_string()
                                     ? value.at("type").get<std::string>()
                                     : std::string();
        const Json criteria = value.contains("criteria") ? value.at("criteria") : Json();

        if (type == "choice") {
            question.type = SystemoneQuestionType::Choice;
            if (!criteria.is_object() || criteria.empty()) {
                bad_request(param + ".criteria must be a non-empty object",
                            param + ".criteria");
            }
            for (auto option = criteria.begin(); option != criteria.end(); ++option) {
                question.options.push_back({option.key(), option.value()});
            }
            if (question.options.size() > kMaximumOptions) {
                bad_request(param + ".criteria has too many options, the maximum is " +
                                std::to_string(kMaximumOptions),
                            param + ".criteria");
            }
        } else if (type == "score") {
            question.type = SystemoneQuestionType::Score;
            if (!criteria.is_array() || criteria.size() < kMinimumScoreLevels ||
                criteria.size() > kMaximumScoreLevels) {
                bad_request(param + ".criteria must be an array of 2 to 10 levels",
                            param + ".criteria");
            }
            for (std::size_t i = 0; i < criteria.size(); ++i) {
                question.options.push_back({std::to_string(i), criteria.at(i)});
            }
        } else if (type == "noul") {
            question.type = SystemoneQuestionType::Noul;
            if (!criteria.is_null() && !criteria.is_object()) {
                bad_request(param + ".criteria must be an object", param + ".criteria");
            }
            for (const char* key : {"false", "true"}) {
                Json description = Json();
                if (criteria.is_object() && criteria.contains(key)) {
                    description = criteria.at(key);
                } else {
                    description = std::string(key) == "true" ? Json("The statement is true.")
                                                             : Json("The statement is false.");
                }
                question.options.push_back({key, std::move(description)});
            }
        } else {
            bad_request(param + ".type must be one of: choice, score, noul", param + ".type");
        }
        questions.push_back(std::move(question));
    }
    return questions;
}

float parse_temperature(const Json& body) {
    if (!body.contains("temperature") || body.at("temperature").is_null()) { return 1.0F; }
    if (!body.at("temperature").is_number()) {
        bad_request("temperature must be a positive finite number", "temperature");
    }
    const double value = body.at("temperature").get<double>();
    if (!std::isfinite(value) || value <= 0.0) {
        bad_request("temperature must be a positive finite number", "temperature");
    }
    return static_cast<float>(value);
}

std::optional<std::string> parse_model(const Json& body) {
    if (!body.contains("model") || body.at("model").is_null()) { return std::nullopt; }
    if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
    return body.at("model").get<std::string>();
}

} // namespace

SystemoneRequest parse_systemone_request(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object", ""); }
    SystemoneRequest request;
    request.state       = parse_state(body, request.images);
    request.questions   = parse_questions(body);
    request.temperature = parse_temperature(body);
    request.model       = parse_model(body);
    return request;
}

Json make_systemone_response(std::string model, Json answers, std::int64_t input_tokens) {
    return Json{
        {"model", std::move(model)},
        {"answers", std::move(answers)},
        {"usage", Json{{"input_tokens", input_tokens}, {"output_tokens", 0}}},
    };
}

} // namespace ninfer::serve
