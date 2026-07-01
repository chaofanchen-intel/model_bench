// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Prompt/text loading from JSONL + token-targeted prompt construction,
// mirroring tests/lib/prompts.py.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mb {

// (original_index, prompt_text) list.
using IndexedPrompts = std::vector<std::pair<int, std::string>>;

// --prompt / --prompt-file are mutually exclusive. prompt_index applied after
// concatenation. Fatal-exits (throws std::runtime_error) on errors.
IndexedPrompts load_prompts(const std::optional<std::string>& prompt,
                            const std::vector<std::string>& prompt_files,
                            const std::vector<int>& prompt_indices);

// Flat string list from JSONL files; accepts {"text":...} or {"prompt":...}.
std::vector<std::string> load_texts_jsonl(const std::vector<std::string>& files);

// Source text for token-targeted prompt generation (ref llama-2 prompt or fallback).
std::string load_base_text(const std::optional<std::string>& base_text_path,
                           const std::string& project_dir);

// Binary-search a char slice of base_text to land near target_tokens.
// count_fn(text) -> token count. Returns (prompt, actual_count).
std::pair<std::string, int> build_prompt_to_target(
    const std::string& base_text, int target_tokens,
    const std::function<int(const std::string&)>& count_fn);

std::pair<std::string, int> build_prompt_from_token_target(
    int target_tokens, const std::function<int(const std::string&)>& count_fn,
    const std::optional<std::string>& base_text_path, const std::string& project_dir);

}  // namespace mb
