// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/prompts.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/json.hpp"

namespace mb {

namespace {

const char* kFallbackText =
    "Machine learning is a branch of artificial intelligence that uses statistical "
    "techniques to give computer systems the ability to learn from data, without being "
    "explicitly programmed. The discipline encompasses several paradigms: in supervised "
    "learning a model is trained on labeled examples and learns a mapping from input "
    "features to output targets; in unsupervised learning the algorithm discovers "
    "structure or patterns in unlabeled data, often via clustering or dimensionality "
    "reduction; in reinforcement learning an agent learns to act in an environment by "
    "receiving rewards or penalties for its choices and gradually optimizes its policy. "
    "Modern neural networks are layered systems of differentiable transformations whose "
    "parameters are tuned by backpropagation. Convolutional networks excel at spatial "
    "data such as images, recurrent and attention-based networks model sequences such as "
    "text, and transformer architectures form the backbone of contemporary large "
    "language models. Training a useful model requires a careful balance of "
    "regularization, optimization, data quality, and evaluation: overfit a model and it "
    "memorizes the training set instead of generalizing; underfit it and the function it "
    "represents is too simple to capture the underlying signal. Practitioners therefore "
    "use techniques such as dropout, weight decay, early stopping, and cross validation "
    "to find the right capacity. Beyond core training, the lifecycle of a machine "
    "learning system includes data ingestion, feature engineering, deployment, monitoring, "
    "and continuous retraining as data distributions shift over time. ";

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

}  // namespace

IndexedPrompts load_prompts(const std::optional<std::string>& prompt,
                            const std::vector<std::string>& prompt_files,
                            const std::vector<int>& prompt_indices) {
    if (prompt && !prompt_files.empty()) {
        die("--prompt and --prompt-file are mutually exclusive");
    }
    if (prompt) {
        return {{0, *prompt}};
    }
    if (prompt_files.empty()) {
        die("provide --prompt or --prompt-file");
    }

    std::vector<std::string> all;
    for (const auto& pf : prompt_files) {
        std::ifstream f(pf);
        if (!f.good()) die("prompt file not found: " + pf);
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            ++lineno;
            // trim
            size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            size_t e = line.find_last_not_of(" \t\r\n");
            std::string trimmed = line.substr(s, e - s + 1);
            if (trimmed.empty()) continue;
            json::ValuePtr obj;
            try {
                json::Parser parser(trimmed);
                obj = parser.parse();
            } catch (...) {
                die(pf + ":" + std::to_string(lineno) + ": invalid JSON");
            }
            std::string val;
            bool found = false;
            if (obj && obj->type == json::Type::Object) {
                for (auto& kv : obj->obj) {
                    if (kv.first == "prompt") {
                        val = kv.second->type == json::Type::String ? kv.second->s : "";
                        found = true;
                        break;
                    }
                }
            }
            if (!found) die(pf + ":" + std::to_string(lineno) + ": missing 'prompt' key");
            all.push_back(val);
        }
    }

    if (!prompt_indices.empty()) {
        std::set<int> seen;
        IndexedPrompts out;
        for (int i : prompt_indices) {
            if (seen.count(i)) continue;
            seen.insert(i);
            if (i < 0 || i >= static_cast<int>(all.size())) {
                die("prompt_index " + std::to_string(i) + " out of range [0, " +
                    std::to_string(all.size()) + ")");
            }
            out.emplace_back(i, all[i]);
        }
        return out;
    }
    IndexedPrompts out;
    for (int i = 0; i < static_cast<int>(all.size()); ++i) out.emplace_back(i, all[i]);
    return out;
}

std::vector<std::string> load_texts_jsonl(const std::vector<std::string>& files) {
    std::vector<std::string> out;
    for (const auto& pf : files) {
        std::ifstream f(pf);
        if (!f.good()) die("texts file not found: " + pf);
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            ++lineno;
            size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos) continue;
            size_t e = line.find_last_not_of(" \t\r\n");
            std::string trimmed = line.substr(s, e - s + 1);
            if (trimmed.empty()) continue;
            json::ValuePtr obj;
            try {
                json::Parser parser(trimmed);
                obj = parser.parse();
            } catch (...) {
                die(pf + ":" + std::to_string(lineno) + ": invalid JSON");
            }
            bool found = false;
            if (obj && obj->type == json::Type::Object) {
                for (const char* key : {"text", "prompt"}) {
                    for (auto& kv : obj->obj) {
                        if (kv.first == key) {
                            out.push_back(kv.second->type == json::Type::String ? kv.second->s : "");
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
            }
            if (!found) die(pf + ":" + std::to_string(lineno) + ": missing 'text'/'prompt' key");
        }
    }
    return out;
}

std::string load_base_text(const std::optional<std::string>& base_text_path,
                           const std::string& project_dir) {
    if (base_text_path) {
        std::ifstream f(*base_text_path, std::ios::binary);
        if (f.good()) {
            std::stringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }
    }
    std::filesystem::path ref = std::filesystem::path(project_dir) /
        "reference/openvino.genai/tools/llm_bench/prompts/llama-2-7b-chat_l.jsonl";
    std::ifstream f(ref);
    if (f.good()) {
        std::string first_line;
        std::getline(f, first_line);
        if (!first_line.empty()) {
            try {
                json::Parser parser(first_line);
                json::ValuePtr obj = parser.parse();
                if (obj && obj->type == json::Type::Object) {
                    for (auto& kv : obj->obj) {
                        if (kv.first == "prompt" && kv.second->type == json::Type::String) {
                            return kv.second->s;
                        }
                    }
                }
            } catch (...) {
            }
        }
    }
    return kFallbackText;
}

std::pair<std::string, int> build_prompt_to_target(
    const std::string& base_text, int target_tokens,
    const std::function<int(const std::string&)>& count_fn) {
    int base_len = count_fn(base_text);
    if (base_len == 0) return {base_text, 0};

    if (base_len >= target_tokens) {
        size_t lo = 1, hi = base_text.size();
        std::string best = base_text;
        while (lo <= hi) {
            size_t mid = (lo + hi) / 2;
            std::string cand = base_text.substr(0, mid);
            if (count_fn(cand) <= target_tokens) {
                best = cand;
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
        return {best, count_fn(best)};
    }

    int repeat = (target_tokens / base_len) + 2;
    std::string long_text;
    long_text.reserve((base_text.size() + 1) * repeat);
    for (int i = 0; i < repeat; ++i) {
        long_text += base_text;
        long_text += " ";
    }
    size_t lo = 1, hi = long_text.size();
    std::string best = long_text;
    while (lo <= hi) {
        size_t mid = (lo + hi) / 2;
        std::string cand = long_text.substr(0, mid);
        if (count_fn(cand) <= target_tokens) {
            best = cand;
            lo = mid + 1;
        } else {
            if (mid == 0) break;
            hi = mid - 1;
        }
    }
    return {best, count_fn(best)};
}

std::pair<std::string, int> build_prompt_from_token_target(
    int target_tokens, const std::function<int(const std::string&)>& count_fn,
    const std::optional<std::string>& base_text_path, const std::string& project_dir) {
    if (target_tokens < 1) die("--input-tokens must be >= 1");
    std::string base = load_base_text(base_text_path, project_dir);
    return build_prompt_to_target(base, target_tokens, count_fn);
}

}  // namespace mb
