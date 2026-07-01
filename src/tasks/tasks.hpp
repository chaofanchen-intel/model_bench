// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Entry points for the five benchmark tasks. Each returns process exit code.
#pragma once

#include "common/cli.hpp"
#include "openvino/openvino.hpp"

namespace mb {

int run_llm(const Args& args);        // Qwen3 LLMPipeline
int run_vlm(const Args& args);        // Qwen3.5 VLMPipeline
int run_lingua(const Args& args);     // LLMLingua-2 raw ov::Core
int run_embedding(const Args& args);  // Qwen3-Embedding TextEmbeddingPipeline
int run_reranker(const Args& args);   // Qwen3-Reranker TextRerankPipeline

// Shared helpers (defined in task_common.cpp).

// Build the plugin AnyMap from CACHE_DIR / NUM_STREAMS / load-config, matching
// the Python pipe_config builders. `attention_backend` (if non-empty and not
// NPU and not already present) is injected as ATTENTION_BACKEND.
ov::AnyMap build_plugin_config(const Args& args, const std::string& device_upper,
                               const std::string& attention_backend = "",
                               bool enable_profiling = false);

// Refresh sudo if monitoring enabled (ensures GPU/NPU samplers can read).
void ensure_sudo_if_monitoring(const Args& args);

// project dir = parent of cwd's nearest "model_bench" or just current dir's parent.
std::string detect_project_dir();

}  // namespace mb
