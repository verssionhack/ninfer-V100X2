#pragma once

// Model-independent typed-decision service. The loaded model's own chat template renders a
// generic "answer with the option letter" prompt; the engine returns the prefill logits for the
// label tokens, and this layer turns them into the TypeSafe answer shapes.

#include "serve/request.h"
#include "serve/systemone_schema.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

class GenerationService;

struct SystemoneLabel {
    std::string code;                // letter shown to the model, e.g. "A"
    std::vector<TokenId> token_ids;  // exact form plus the leading-space variant
};

using SystemoneEncodeFn = std::function<std::vector<TokenId>(std::string_view)>;

// String states are used verbatim; every other JSON value is compact-serialized (the reference's
// `tojson` behaviour).
[[nodiscard]] std::string systemone_json_text(const nlohmann::json& value);

// Picks single-token option codes from A..Z a..z. Throws ApiException when the model's tokenizer
// cannot represent enough labels as one token.
[[nodiscard]] std::vector<SystemoneLabel>
select_systemone_labels(std::size_t option_count, const SystemoneEncodeFn& encode);

[[nodiscard]] std::vector<ChatTurn>
build_systemone_messages(std::string_view state_text,
                         const std::vector<ninfer::product::media_acquire::Source>& images,
                         const SystemoneQuestion& question,
                         const std::vector<SystemoneLabel>& labels);

// One score per label: the maximum logit among the label's token variants (the reference's group
// rule). Throws when `token_scores` does not match the flattened label ids.
[[nodiscard]] std::vector<float>
systemone_group_scores(const std::vector<SystemoneLabel>& labels,
                       std::span<const float> token_scores);

[[nodiscard]] nlohmann::json answer_from_scores(const SystemoneQuestion& question,
                                                const std::vector<SystemoneLabel>& labels,
                                                std::span<const float> group_scores,
                                                float temperature);

struct SystemoneRunResult {
    nlohmann::json answers;
    std::int64_t input_tokens = 0;
};

// Runs one prefill per question, sequentially, so later questions can reuse the state prefix.
[[nodiscard]] SystemoneRunResult execute_systemone(const SystemoneRequest& request,
                                                   const std::string& model,
                                                   GenerationService& service,
                                                   const std::function<bool()>& is_cancelled);

} // namespace ninfer::serve
