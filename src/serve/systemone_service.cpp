#include "serve/systemone_service.h"

#include "serve/generation_service.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kSystemInstructions =
    "You are a decision engine. Read the state and answer the question by selecting exactly one "
    "option. Reply with the option's letter and nothing else.";

constexpr std::string_view kCodePool = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

[[noreturn]] void unsupported_labels(std::size_t option_count) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = "the loaded model's tokenizer cannot represent " +
                    std::to_string(option_count) + " option labels as single tokens";
    error.param   = "questions";
    error.code    = "label_not_supported";
    throw ApiException(std::move(error));
}

ChatTurn text_turn(ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = "text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

const char* question_type_name(SystemoneQuestionType type) {
    switch (type) {
    case SystemoneQuestionType::Choice: return "choice";
    case SystemoneQuestionType::Score: return "score";
    case SystemoneQuestionType::Noul: return "noul";
    }
    return "";
}

double choice_confidence(const std::vector<double>& probabilities) {
    if (probabilities.size() < 2) { return 1.0; }
    const double uniform = 1.0 / static_cast<double>(probabilities.size());
    const double top = *std::max_element(probabilities.begin(), probabilities.end());
    return std::max(0.0, (top - uniform) / (1.0 - uniform));
}

double score_confidence(const std::vector<double>& probabilities) {
    if (probabilities.size() < 2) { return 1.0; }
    const std::size_t n = probabilities.size();
    const std::size_t mode =
        static_cast<std::size_t>(std::max_element(probabilities.begin(), probabilities.end()) -
                                 probabilities.begin());
    double distance         = 0.0;
    double uniform_distance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        distance += probabilities[i] * std::fabs(static_cast<double>(i) -
                                                  static_cast<double>(mode));
        uniform_distance +=
            std::fabs(static_cast<double>(i) - static_cast<double>(n - 1) / 2.0) /
            static_cast<double>(n);
    }
    return std::max(0.0, 1.0 - distance / uniform_distance);
}

} // namespace

std::string systemone_json_text(const Json& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return value.dump();
}

std::vector<SystemoneLabel>
select_systemone_labels(std::size_t option_count, const SystemoneEncodeFn& encode) {
    std::vector<SystemoneLabel> labels;
    std::vector<TokenId> used;
    const auto is_used = [&used](TokenId id) {
        return std::find(used.begin(), used.end(), id) != used.end();
    };
    for (const char code : kCodePool) {
        if (labels.size() >= option_count) { break; }
        const std::string text(1, code);
        const std::vector<TokenId> exact = encode(text);
        if (exact.size() != 1 || is_used(exact.front())) { continue; }
        SystemoneLabel label;
        label.code = text;
        label.token_ids.push_back(exact.front());
        used.push_back(exact.front());
        const std::vector<TokenId> spaced = encode(" " + text);
        if (spaced.size() == 1 && !is_used(spaced.front())) {
            label.token_ids.push_back(spaced.front());
            used.push_back(spaced.front());
        }
        labels.push_back(std::move(label));
    }
    if (labels.size() < option_count) { unsupported_labels(option_count); }
    return labels;
}

std::vector<ChatTurn>
build_systemone_messages(std::string_view state_text,
                         const std::vector<ninfer::product::media_acquire::Source>& images,
                         const SystemoneQuestion& question,
                         const std::vector<SystemoneLabel>& labels) {
    if (labels.size() != question.options.size()) {
        throw std::logic_error("decision labels do not match the question options");
    }
    std::vector<ChatTurn> messages;
    messages.push_back(text_turn(ChatRole::System, std::string(kSystemInstructions)));

    ChatTurn state_turn;
    state_turn.role = ChatRole::User;
    for (const auto& source : images) {
        ContentPart media;
        media.kind     = ContentKind::Image;
        media.type_raw = "image_url";
        media.source   = source;
        state_turn.content.push_back(std::move(media));
    }
    state_turn.content.push_back(
        ContentPart{ContentKind::Text, "State:\n" + std::string(state_text), "text"});
    messages.push_back(std::move(state_turn));

    std::string question_text = "Question: " + systemone_json_text(question.instructions);
    if (question.type == SystemoneQuestionType::Score) {
        question_text += " Rate along the ordered levels below (lowest first).";
    }
    question_text += "\nOptions:\n";
    for (std::size_t i = 0; i < question.options.size(); ++i) {
        const SystemoneOption& option = question.options[i];
        // noul keeps false/true as response keys but reads naturally as no/yes in the prompt.
        const std::string display_key =
            question.type == SystemoneQuestionType::Noul
                ? (option.key == "true" ? "yes" : "no")
                : option.key;
        question_text += "[" + labels[i].code + "] " + display_key;
        if (!option.description.is_null()) {
            question_text += ": " + systemone_json_text(option.description);
        }
        question_text += "\n";
    }
    question_text += "\nAnswer with the letter of the best option only.";
    messages.push_back(text_turn(ChatRole::User, std::move(question_text)));
    return messages;
}

std::vector<float> systemone_group_scores(const std::vector<SystemoneLabel>& labels,
                                          std::span<const float> token_scores) {
    std::vector<float> scores;
    scores.reserve(labels.size());
    std::size_t offset = 0;
    for (const SystemoneLabel& label : labels) {
        if (label.token_ids.empty() || offset + label.token_ids.size() > token_scores.size()) {
            throw std::runtime_error("decision score result does not match the label groups");
        }
        float best = token_scores[offset];
        for (std::size_t i = 1; i < label.token_ids.size(); ++i) {
            best = std::max(best, token_scores[offset + i]);
        }
        scores.push_back(best);
        offset += label.token_ids.size();
    }
    if (offset != token_scores.size()) {
        throw std::runtime_error("decision score result has extra values");
    }
    return scores;
}

nlohmann::json answer_from_scores(const SystemoneQuestion& question,
                                  const std::vector<SystemoneLabel>& labels,
                                  std::span<const float> group_scores, float temperature) {
    const std::size_t n = question.options.size();
    if (group_scores.size() != n || labels.size() != n) {
        throw std::runtime_error("decision scores do not match the question options");
    }
    if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
        throw std::runtime_error("decision temperature is invalid");
    }
    const float top = *std::max_element(group_scores.begin(), group_scores.end());
    std::vector<double> probabilities(n, 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        probabilities[i] =
            std::exp(static_cast<double>(group_scores[i] - top) / temperature);
        sum += probabilities[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("the model produced no usable decision scores");
    }
    for (double& probability : probabilities) { probability /= sum; }

    Json answer = Json{{"type", question_type_name(question.type)}};
    if (question.type == SystemoneQuestionType::Noul) {
        for (std::size_t i = 0; i < n; ++i) {
            if (question.options[i].key == "true") {
                answer["noul"] = probabilities[i];
                return answer;
            }
        }
        throw std::runtime_error("a noul question has no true option");
    }

    Json probability_json = Json::object();
    for (std::size_t i = 0; i < n; ++i) {
        probability_json[question.options[i].key] = probabilities[i];
    }
    if (question.type == SystemoneQuestionType::Choice) {
        const std::size_t best = static_cast<std::size_t>(
            std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
        answer["choice"]        = question.options[best].key;
        answer["probabilities"] = std::move(probability_json);
        answer["confidence"]    = choice_confidence(probabilities);
        return answer;
    }

    double expected = 0.0;
    Json legend     = Json::object();
    for (std::size_t i = 0; i < n; ++i) {
        expected += static_cast<double>(i) * probabilities[i];
        legend[question.options[i].key] = question.options[i].description;
    }
    answer["score"]         = expected;
    answer["legend"]        = std::move(legend);
    answer["probabilities"] = std::move(probability_json);
    answer["confidence"]    = score_confidence(probabilities);
    return answer;
}

SystemoneRunResult execute_systemone(const SystemoneRequest& request, const std::string& model,
                                     GenerationService& service,
                                     const std::function<bool()>& is_cancelled) {
    SystemoneRunResult result;
    result.answers = Json::object();
    const std::string state_text = systemone_json_text(request.state);
    const SystemoneEncodeFn encode = [&service](std::string_view text) {
        return service.encode_text(text, /*parse_added_tokens=*/false);
    };

    for (const SystemoneQuestion& question : request.questions) {
        const std::vector<SystemoneLabel> labels =
            select_systemone_labels(question.options.size(), encode);
        std::vector<TokenId> score_ids;
        for (const SystemoneLabel& label : labels) {
            score_ids.insert(score_ids.end(), label.token_ids.begin(), label.token_ids.end());
        }

        GenerationRequest generation;
        generation.model = model;
        generation.messages =
            build_systemone_messages(state_text, request.images, question, labels);
        generation.max_tokens                  = 1;
        generation.max_tokens_set              = true;
        generation.enable_thinking             = false;
        generation.preserve_thinking           = false;
        generation.sampling.temperature        = 0.0; // greedy; the sampled token is discarded
        generation.sampling.top_k              = 1;
        generation.score_token_ids             = std::move(score_ids);

        std::vector<float> token_scores;
        int prompt_tokens = 0;
        {
            PreparedRequest prepared = service.prepare(generation, is_cancelled);
            GenerationOutcome outcome = service.run(prepared, nullptr, is_cancelled);
            prompt_tokens = outcome.prompt_tokens;
            token_scores  = std::move(outcome.token_scores);
        }
        result.input_tokens += prompt_tokens;

        const std::vector<float> group_scores =
            systemone_group_scores(labels, token_scores);
        result.answers[question.id] =
            answer_from_scores(question, labels, group_scores, request.temperature);
    }
    return result;
}

} // namespace ninfer::serve
