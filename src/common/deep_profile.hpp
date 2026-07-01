// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Deep per-operation profiling that re-implements each model's REAL inference
// on raw ov::Core so we own the InferRequest and can call get_profiling_info()
// on the actual generation workload. This is what makes --profile yield true
// per-layer/per-kernel data even for the GenAI LLM/VLM pipelines, whose
// compiled models are otherwise hidden.
//
// Design (mirrors how GenAI drives these graphs internally):
//   LLM   (stateful decoder): openvino_model.xml with
//         input_ids, attention_mask, position_ids, beam_idx -> logits + KV state.
//         We run a greedy loop and profile PREFILL and DECODE phases separately.
//   VLM   (Qwen3.5): openvino_text_embeddings_model.xml -> inputs_embeds, then
//         openvino_language_model.xml (stateful) with inputs_embeds,
//         attention_mask, position_ids[4,seq] (mRoPE), beam_idx -> logits.
//         Text-only path (no image) — profiles prefill + decode of the LM, plus
//         the text-embeddings graph.
//   Encoder (embedding/reranker/lingua): single forward pass on REAL tokenized
//         inputs (not dummy zeros).
//
// Each phase produces its own ProfRow accumulator + report so prefill and
// decode hotspots are distinguishable.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "common/profiling.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/openvino.hpp"

namespace mb {

struct DeepProfileConfig {
    std::string model_dir;
    std::string device;        // upper-cased
    std::string precision;
    ov::AnyMap plugin_cfg;     // already includes PERF_COUNT=true
    int prompt_tokens = 64;    // synthetic prompt length (fallback when prompt_text is empty)
    // If set, LLM/VLM prefill is profiled on this REAL prompt text (tokenized to
    // its natural length) — mirrors what the GenAI benchmark actually ran with
    // -pf / --prompt. When empty, a synthetic prompt of `prompt_tokens` is used.
    std::optional<std::string> prompt_text;
    int gen_tokens = 32;       // decode steps to profile
    int warmup = 1;            // warmup generations before profiling (not profiled)
    int topk = 40;
    std::optional<std::string> profile_csv;
    // For encoders: the real input text(s) to profile (already resolved).
    std::vector<std::string> encoder_texts;
    int encoder_max_length = 0;  // 0 = let model use dynamic seq
};

// LLM stateful greedy-loop profiling. Reads openvino_model.xml, drives a real
// prefill + `gen_tokens` decode steps, accumulates per-op profiling for each
// phase, prints two reports, and (optionally) writes CSV rows tagged
// "<model>" with stage "llm_prefill" / "llm_decode".
// Returns false if the model/graph isn't the expected stateful LLM shape.
bool deep_profile_llm(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer);

// VLM text-only language-model profiling (text_embeddings -> language_model).
bool deep_profile_vlm(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer);

// Encoder single-pass profiling on REAL tokenized input (embedding/reranker).
// stage e.g. "embedding_graph" / "reranker_graph".
bool deep_profile_encoder(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer,
                          const std::string& stage);

}  // namespace mb
