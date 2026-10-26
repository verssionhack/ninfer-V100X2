// Host-only contract test for the model-independent /v1/systemone layer: request parsing and
// validation, label selection with a fake tokenizer, prompt rendering, score grouping, and the
// TypeSafe answer shapes. No artifact, device, or Engine is involved.

#include "serve/request.h"
#include "serve/systemone_schema.h"
#include "serve/systemone_service.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::ChatRole;
using ninfer::TokenId;
using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

bool throws_api(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException&) { return true; } catch (...) { return false; }
    return false;
}

bool throws_any(const std::function<void()>& f) {
    try {
        f();
    } catch (...) { return true; }
    return false;
}

std::string api_code(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

SystemoneQuestion choice_question(std::vector<std::pair<std::string, Json>> options) {
    SystemoneQuestion question;
    question.id           = "route";
    question.type         = SystemoneQuestionType::Choice;
    question.instructions = "Which team should handle this?";
    for (auto& option : options) {
        question.options.push_back({std::move(option.first), std::move(option.second)});
    }
    return question;
}

SystemoneEncodeFn fake_encoder(const std::map<std::string, TokenId>& table) {
    auto shared = std::make_shared<const std::map<std::string, TokenId>>(table);
    return [shared](std::string_view text) -> std::vector<TokenId> {
        const auto it = shared->find(std::string(text));
        if (it == shared->end()) { return {1000, 1001}; } // "multi-token" fallback
        return {it->second};
    };
}

int test_parse_choice() {
    const Json body = Json::parse(R"({
        "state": "Customer message: charged twice.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": null, "shipping": "delivery"}
            }
        }
    })");
    int failures = 0;
    const SystemoneRequest request = parse_systemone_request(body);
    failures += check(request.state.is_string(), "state is a string");
    failures += check(request.images.empty(), "no images");
    failures += check(request.questions.size() == 1, "one question");
    failures += check(request.questions[0].id == "route", "question id");
    failures += check(request.questions[0].type == SystemoneQuestionType::Choice, "choice type");
    failures += check(request.questions[0].options.size() == 2, "two options");
    failures += check(request.questions[0].options[0].key == "billing", "option order");
    failures += check(request.questions[0].options[0].description.is_null(), "null description");
    failures += check(request.questions[0].options[1].description == "delivery", "description");
    failures += check(request.temperature == 1.0F, "default temperature");
    failures += check(!request.model.has_value(), "model optional");
    return failures;
}

int test_parse_score_and_noul() {
    const Json body = Json::parse(R"({
        "state": {"ticket": 7},
        "temperature": 0.5,
        "model": "m",
        "questions": {
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["can wait", "this week", "today"]},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"}
        }
    })");
    int failures = 0;
    const SystemoneRequest request = parse_systemone_request(body);
    failures += check(request.state.is_object(), "object state kept");
    failures += check(request.temperature == 0.5F, "temperature parsed");
    failures += check(request.model.has_value() && *request.model == "m", "model parsed");
    failures += check(request.questions.size() == 2, "two questions");
    const SystemoneQuestion& score = request.questions[0].id == "angry" ? request.questions[1]
                                                                       : request.questions[0];
    const SystemoneQuestion& noul  = request.questions[0].id == "angry" ? request.questions[0]
                                                                       : request.questions[1];
    failures += check(score.type == SystemoneQuestionType::Score, "score type");
    failures += check(score.options.size() == 3, "three levels");
    failures += check(score.options[0].key == "0" && score.options[2].key == "2", "level keys");
    failures += check(noul.type == SystemoneQuestionType::Noul, "noul type");
    failures += check(noul.options.size() == 2, "two noul options");
    failures += check(noul.options[0].key == "false" && noul.options[1].key == "true", "noul keys");
    failures += check(noul.options[1].description == "The statement is true.", "noul default");
    return failures;
}

int test_parse_images() {
    const Json body = Json::parse(R"({
        "state": {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}
            ]}
        ]},
        "files": ["data:image/png;base64,BBBB"],
        "images": ["data:image/png;base64,CCCC"],
        "questions": {"q": {"type": "noul", "instructions": "ok?"}}
    })");
    int failures = 0;
    const SystemoneRequest request = parse_systemone_request(body);
    failures += check(request.images.size() == 3, "files, images, then state image");
    failures += check(request.images[0].value.find("BBBB") != std::string::npos, "files first");
    failures += check(request.images[1].value.find("CCCC") != std::string::npos, "images second");
    failures += check(request.images[2].value.find("AAAA") != std::string::npos,
                      "state image third");
    failures += check(request.state.at("messages").at(0).at("content").size() == 1,
                      "state image part removed");
    failures += check(request.state.at("messages").at(0).at("content").at(0).at("type") == "text",
                      "state text kept");
    return failures;
}

int test_parse_rejections() {
    int failures = 0;
    const auto rejects = [&failures](const Json& body, const std::string& what) {
        failures += check(throws_api([&] { (void)parse_systemone_request(body); }),
                          "rejects " + what);
    };
    rejects(Json::array(), "non-object body");
    rejects(Json::parse(R"({"questions": {"q": {"type": "noul", "instructions": "x"}}})"),
            "missing state");
    rejects(Json::parse(R"({"state": "x", "questions": {}})"), "empty questions");
    rejects(Json::parse(
                R"({"state": "x", "questions": {"q": {"type": "bad", "instructions": "x"}}})"),
            "bad type");
    rejects(Json::parse(R"({"state": "x", "questions": {"q": {"type": "choice",
                "instructions": "x", "criteria": {}}}})"),
            "empty choice criteria");
    rejects(Json::parse(R"({"state": "x", "questions": {"q": {"type": "score",
                "instructions": "x", "criteria": ["one"]}}})"),
            "one score level");
    rejects(Json::parse(R"({"state": "x", "questions": {"q": {"type": "score",
                "instructions": "x", "criteria": [0,1,2,3,4,5,6,7,8,9,10]}}})"),
            "eleven score levels");
    rejects(Json::parse(R"({"state": "x", "videos": ["data:video/mp4;base64,AAAA"],
                "questions": {"q": {"type": "noul", "instructions": "x"}}})"),
            "videos");
    rejects(Json::parse(R"({"state": "x", "files": ["https://example.com/a.png"],
                "questions": {"q": {"type": "noul", "instructions": "x"}}})"),
            "non-data URL");
    rejects(Json::parse(R"({"state": "x", "temperature": 0,
                "questions": {"q": {"type": "noul", "instructions": "x"}}})"),
            "zero temperature");

    Json many_options = Json::object();
    for (int i = 0; i < 53; ++i) { many_options["o" + std::to_string(i)] = nullptr; }
    rejects(Json{{"state", "x"},
                 {"questions", Json{{"q", Json{{"type", "choice"},
                                                  {"instructions", "x"},
                                                  {"criteria", many_options}}}}}},
            "53 options");

    Json many_questions = Json::object();
    for (int i = 0; i < 65; ++i) {
        many_questions["q" + std::to_string(i)] =
            Json{{"type", "noul"}, {"instructions", "x"}};
    }
    rejects(Json{{"state", "x"}, {"questions", many_questions}}, "65 questions");
    return failures;
}

int test_labels_and_prompt() {
    int failures = 0;
    const SystemoneEncodeFn encode =
        fake_encoder({{"A", 11}, {" A", 12}, {"B", 13}, {" B", 14}, {"C", 15}, {" C", 16}});
    const std::vector<SystemoneLabel> labels = select_systemone_labels(3, encode);
    failures += check(labels.size() == 3, "three labels");
    failures += check(labels[0].code == "A" && labels[0].token_ids.size() == 2, "A group");
    failures += check(labels[1].token_ids[0] == 13 && labels[1].token_ids[1] == 14, "B group");
    failures += check(labels[2].token_ids[0] == 15, "C group");

    // A multi-token "A" is skipped; the next usable code still renders a valid prompt.
    const SystemoneEncodeFn skip_a = fake_encoder({{"B", 21}, {" B", 22}, {"C", 23}});
    const std::vector<SystemoneLabel> skipped = select_systemone_labels(2, skip_a);
    failures += check(skipped[0].code == "B" && skipped[1].code == "C", "unusable code skipped");

    failures += check(api_code([&] { (void)select_systemone_labels(3, skip_a); }) ==
                          "label_not_supported",
                      "too few labels code");

    const SystemoneQuestion question = choice_question(
        {{"billing", Json()}, {"shipping", "delivery"}, {"technical", Json()}});
    const std::vector<ChatTurn> messages =
        build_systemone_messages("state text", {}, question, labels);
    failures += check(messages.size() == 3, "system + state + question");
    failures += check(messages[0].role == ChatRole::System, "system turn");
    failures += check(messages[1].role == ChatRole::User && messages[1].content.size() == 1,
                      "state turn");
    failures += check(messages[1].content[0].text == "State:\nstate text", "state text");
    const std::string& prompt = messages[2].content[0].text;
    failures += check(prompt.find("Question: Which team should handle this?") == 0,
                      "question prefix");
    failures += check(prompt.find("[A] billing\n") != std::string::npos, "billing option");
    failures += check(prompt.find("[B] shipping: delivery\n") != std::string::npos,
                      "shipping option");
    failures += check(prompt.find("Answer with the letter of the best option only.") !=
                          std::string::npos,
                      "answer tail");
    return failures;
}

int test_group_scores_and_answers() {
    int failures = 0;
    const SystemoneEncodeFn encode =
        fake_encoder({{"A", 1}, {" A", 2}, {"B", 3}, {" B", 4}, {"C", 5}, {" C", 6}});
    const std::vector<SystemoneLabel> labels = select_systemone_labels(2, encode);
    const std::vector<float> token_scores     = {-1.0F, 0.5F, -2.0F, -3.0F};
    const std::vector<float> groups           = systemone_group_scores(labels, token_scores);
    failures += check(groups.size() == 2 && groups[0] == 0.5F && groups[1] == -2.0F,
                      "max over label variants");
    failures += check(throws_any(
                          [&] { (void)systemone_group_scores(labels, std::vector<float>{0.0F}); }),
                      "score arity mismatch rejected");

    const SystemoneQuestion choice =
        choice_question({{"billing", Json()}, {"shipping", Json()}});
    const Json choice_answer =
        answer_from_scores(choice, labels, std::vector<float>{0.0F, 2.0F}, 1.0F);
    failures += check(choice_answer.at("type") == "choice", "choice type");
    failures += check(choice_answer.at("choice") == "shipping", "choice argmax");
    const double p_shipping = choice_answer.at("probabilities").at("shipping").get<double>();
    const double p_billing  = choice_answer.at("probabilities").at("billing").get<double>();
    failures += check(std::abs(p_billing + p_shipping - 1.0) < 1e-9, "probabilities sum to 1");
    failures += check(p_shipping > p_billing, "shipping is more likely");
    const double confidence = choice_answer.at("confidence").get<double>();
    failures += check(confidence >= 0.0 && confidence <= 1.0, "choice confidence range");

    const Json flatter = answer_from_scores(choice, labels, std::vector<float>{0.0F, 2.0F}, 4.0F);
    failures += check(flatter.at("probabilities").at("shipping").get<double>() < p_shipping,
                      "higher temperature flattens");

    SystemoneQuestion score_question;
    score_question.id   = "urgency";
    score_question.type = SystemoneQuestionType::Score;
    score_question.options = {{"0", Json("cold")}, {"1", Json("warm")}, {"2", Json("hot")}};
    const std::vector<SystemoneLabel> score_labels = select_systemone_labels(3, encode);
    const Json score_answer =
        answer_from_scores(score_question, score_labels, std::vector<float>{0.0F, 0.0F, 3.0F},
                           1.0F);
    failures += check(score_answer.at("type") == "score", "score type");
    failures += check(score_answer.at("score").get<double>() > 1.5, "expected level near 2");
    failures += check(score_answer.at("legend").at("2") == "hot", "legend");
    failures += check(score_answer.at("confidence").get<double>() >= 0.0, "score confidence");

    SystemoneQuestion noul_question;
    noul_question.id   = "angry";
    noul_question.type = SystemoneQuestionType::Noul;
    noul_question.options = {{"false", Json()}, {"true", Json()}};
    const Json noul_answer =
        answer_from_scores(noul_question, labels, std::vector<float>{-1.0F, 1.0F}, 1.0F);
    failures += check(noul_answer.at("type") == "noul", "noul type");
    failures += check(noul_answer.contains("noul") && !noul_answer.contains("probabilities"),
                      "noul shape");
    failures += check(noul_answer.at("noul").get<double>() > 0.5, "noul is P(true)");
    return failures;
}

int test_response_shape() {
    const Json response =
        make_systemone_response("model", Json{{"q", Json{{"type", "noul"}, {"noul", 0.5}}}}, 42);
    int failures = 0;
    failures += check(response.at("model") == "model", "response model");
    failures += check(response.at("answers").contains("q"), "response answers");
    failures += check(response.at("usage").at("input_tokens") == 42, "usage input");
    failures += check(response.at("usage").at("output_tokens") == 0, "usage output");
    return failures;
}

int test_json_text() {
    int failures = 0;
    failures += check(systemone_json_text(Json("plain")) == "plain", "string verbatim");
    failures += check(systemone_json_text(Json{{"a", 1}}) == R"({"a":1})", "object compact");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_choice();
    failures += test_parse_score_and_noul();
    failures += test_parse_images();
    failures += test_parse_rejections();
    failures += test_labels_and_prompt();
    failures += test_group_scores_and_answers();
    failures += test_response_shape();
    failures += test_json_text();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
