// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// A small zero-dependency argv parser (no cxxopts fetch needed). Supports:
//   --flag                       (store_true)
//   --opt value                  (single value; also --opt=value)
//   --opt v1 v2 v3               (nargs+, consumes until next token starting "-")
//   short aliases (-m, -d, ...)
//
// The Args struct holds the UNION of every task's options so the five task
// modules share one parser. Each task reads only the fields it cares about.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mb {

struct Args {
    std::string task;  // llm | vlm | lingua | embedding | reranker

    // Shared / generation
    std::vector<std::string> model_dir;
    std::vector<std::string> device{"CPU"};
    std::optional<std::string> prompt;
    std::vector<std::string> prompt_file;
    std::vector<int> prompt_index;
    std::optional<int> input_tokens = 1000;  // default: synthesize a ~1000-token prompt
    std::optional<std::string> input_base_text;
    int infer_count = 512;
    bool exact_output_tokens = false;
    int num_iters = 5;
    int batch_size = 1;
    int num_beams = 1;
    bool apply_chat_template = false;
    bool disable_prompt_permutation = false;  // default depends on task (set in defaults)
    bool permutation_default_off = false;     // vlm sets this true
    int seed = 42;
    std::optional<std::string> load_config;
    std::string cache_dir = "";       // matches official default ""
    bool cache_dir_set = true;        // Python sets CACHE_DIR even when empty
    std::optional<std::string> num_streams;
    int cooldown_between_runs = 5;

    // VLM
    std::optional<std::string> image;

    // Lingua
    std::vector<int> input_lengths{64, 128, 256, 512};
    std::vector<double> compression_rates{0.3, 0.5, 0.7};
    int warmup = 5;
    int lingua_iters = 50;
    bool lingua_iters_set = false;

    // Embedding
    std::string mode = "documents";  // documents | query | both
    std::string pooling_type = "last_token";  // cls | mean | last_token
    bool normalize = true;
    std::optional<int> max_length;
    std::optional<std::string> padding_side;  // left | right
    bool pad_to_max_length = false;
    std::optional<std::string> query_instruction;
    std::optional<std::string> embed_instruction;

    // Reranker
    std::vector<std::string> texts;
    std::vector<std::string> texts_file;
    std::optional<int> num_documents;
    int top_n = 3;

    // Monitors
    bool no_monitor = true;       // monitoring OFF by default (matches direct scripts)
    bool memory_monitor = false;
    int memory_monitor_interval = 100;
    int cpu_monitor_interval = 100;
    int gpu_monitor_interval = 500;
    int npu_monitor_interval = 500;
    std::optional<std::string> gpu_command;

    // Profiling mode
    bool profile = false;
    int profile_topk = 40;        // top-N hottest ops per report
    int profile_warmup = 3;
    int profile_iters = 10;
    std::optional<std::string> profile_csv;
    // Which decoder graph to profile. LLMPipeline defaults to
    // ATTENTION_BACKEND=PA, so "pa" is what the reported TTFT actually runs;
    // "sdpa" profiles the stateful graph; "both" profiles each in turn.
    std::string profile_graph = "pa";

    // IO
    std::string csv_out;   // defaults filled per-task
    std::string json_out;

    // misc
    bool help = false;
};

// Parse argv (argv[0] = program, argv[1] = task). Throws std::runtime_error on
// malformed input. Fills task-specific default csv/json paths.
Args parse_args(int argc, char** argv);

// Print usage for a task (or the top-level task list).
void print_usage(const std::string& task);

}  // namespace mb
