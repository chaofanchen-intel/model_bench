// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Per-iteration and aggregated result structs, mirroring tests/lib/metrics.py.
// Field names / units kept identical so the C++ JSON and CSV match the Python
// schema column-for-column (see tests/lib/reports.py).
#pragma once

#include <map>
#include <string>
#include <vector>

#include "common/monitors.hpp"
#include "common/stats.hpp"

namespace mb {

// Shared per-iteration memory block (attached when --memory-monitor on).
struct IterMemory {
    bool memory_available = false;
    int memory_samples = 0;
    double max_rss_mem_mib = 0.0;
    double max_sys_mem_mib = 0.0;
    double max_rss_mem_increase_mib = 0.0;
    double max_sys_mem_increase_mib = 0.0;
    double max_rss_mem_share = 0.0;
    double max_sys_mem_share = 0.0;
};

// Aggregated memory block on a result.
struct ResultMemory {
    bool memory_available = false;
    int memory_samples = 0;
    double initial_rss_mem_mib = 0.0;
    double initial_sys_mem_mib = 0.0;
    double initial_rss_mem_share = 0.0;
    double initial_sys_mem_share = 0.0;
    double compile_max_rss_mem_mib = 0.0;
    double compile_max_sys_mem_mib = 0.0;
    double compile_max_rss_mem_increase_mib = 0.0;
    double compile_max_sys_mem_increase_mib = 0.0;
    double compile_max_rss_mem_share = 0.0;
    double compile_max_sys_mem_share = 0.0;
    double runtime_mean_rss_mem_mib = 0.0;
    double runtime_max_rss_mem_mib = 0.0;
    double runtime_mean_sys_mem_mib = 0.0;
    double runtime_max_sys_mem_mib = 0.0;
    double runtime_mean_rss_mem_increase_mib = 0.0;
    double runtime_max_rss_mem_increase_mib = 0.0;
    double runtime_mean_sys_mem_increase_mib = 0.0;
    double runtime_max_sys_mem_increase_mib = 0.0;
    double runtime_max_rss_mem_share = 0.0;
    double runtime_max_sys_mem_share = 0.0;
};

// ---- Generation (LLM + VLM) -------------------------------------------------

struct GenIterationMetrics {
    int iteration = 0;
    bool is_warmup = false;
    int prompt_idx = 0;
    int input_tokens = 0;
    int output_tokens = 0;
    int infer_count = 0;
    double generate_time_s = 0.0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double first_token_infer_latency_ms = -1.0;
    double other_tokens_infer_avg_latency_ms = -1.0;
    double prefill_latency_ms = 0.0;
    double decode_latency_ms = 0.0;
    double e2e_latency_ms = 0.0;
    double throughput_tps = 0.0;
    double decode_throughput_tps = 0.0;
    double tokenization_time_ms = 0.0;
    double detokenization_time_ms = 0.0;
    std::string result_md5;
    std::string start;
    std::string end;
    IterMemory mem;
};

struct GenBenchmarkResult {
    std::string model_dir, model_name;
    std::string framework = "ov";
    std::string pipeline;  // LLMPipeline | VLMPipeline
    std::string device, precision;
    std::string mode = "text-only";

    int num_beams = 1;
    int batch_size = 1;
    bool apply_chat_template = false;
    bool disable_prompt_permutation = false;
    int prompt_idx = 0;
    int input_tokens = 0;
    int infer_count_arg = 0;
    bool exact_output_tokens = false;
    int num_iters = 0;
    int measured_iters = 0;
    double compile_time_s = 0.0;

    // Aggregated stat groups, indexed by name -> Stats (filled from aggregate()).
    std::map<std::string, Stats> agg;

    ResultMemory mem;
    CpuSummary cpu_util;
    GpuSummary gpu_util;
    NpuSummary npu_util;

    std::vector<GenIterationMetrics> iterations;

    std::string image_path;
    std::string openvino_version, openvino_genai_version, run_started_at;
};

// ---- BERT (lingua) ----------------------------------------------------------

struct BertIterationMetrics {
    int iteration = 0;
    int input_tokens = 0;
    double latency_ms = 0.0;
    double throughput_samples_per_s = 0.0;
    double throughput_tokens_per_s = 0.0;
    int compressed_tokens = 0;
    double compression_ratio = 0.0;
    IterMemory mem;
};

struct BertBenchmarkResult {
    std::string model_dir, model_name;
    std::string framework = "ov";
    std::string device, precision;
    int input_length = 0;
    double compression_rate = 0.0;
    int warmup_iters = 0;
    int measure_iters = 0;
    double compile_time_s = 0.0;
    std::vector<BertIterationMetrics> iterations;

    Stats latency;        // ms
    Stats thr_samples;
    Stats thr_tokens;
    double avg_input_tokens = 0.0;
    double avg_compressed_tokens = 0.0;
    double avg_compression_ratio = 0.0;

    ResultMemory mem;
    CpuSummary cpu_util;
    GpuSummary gpu_util;
    NpuSummary npu_util;

    std::string openvino_version, run_started_at;
};

// ---- Embedding --------------------------------------------------------------

struct EmbedIterationMetrics {
    int iteration = 0;
    bool is_warmup = false;
    int prompt_idx = 0;
    std::string mode = "documents";
    int batch_size = 1;
    int input_tokens = 0;
    int total_input_tokens = 0;
    int embedding_dim = 0;
    double tokenization_time_ms = 0.0;
    double latency_ms = 0.0;
    double e2e_latency_ms = 0.0;
    double throughput_docs_per_s = 0.0;
    double throughput_tokens_per_s = 0.0;
    std::string result_md5;
    std::string start, end;
    IterMemory mem;
};

struct EmbedBenchmarkResult {
    std::string model_dir, model_name;
    std::string framework = "ov";
    std::string pipeline = "TextEmbeddingPipeline";
    std::string device, precision;
    std::string mode = "documents";

    int batch_size = 1;
    std::string pooling_type;
    bool normalize = false;
    int max_length = 0;
    std::string padding_side;
    bool pad_to_max_length = false;
    std::string query_instruction;
    std::string embed_instruction;
    int prompt_idx = 0;
    int input_tokens = 0;
    int total_input_tokens = 0;
    int embedding_dim = 0;
    int num_iters = 0;
    int measured_iters = 0;
    double compile_time_s = 0.0;

    Stats latency, e2e, tokenization, thr_docs, thr_tokens;

    ResultMemory mem;
    CpuSummary cpu_util;
    GpuSummary gpu_util;
    NpuSummary npu_util;

    std::vector<EmbedIterationMetrics> iterations;
    std::string openvino_version, openvino_genai_version, run_started_at;
};

// ---- Reranker ---------------------------------------------------------------

struct RerankIterationMetrics {
    int iteration = 0;
    bool is_warmup = false;
    int prompt_idx = 0;
    int num_documents = 0;
    int top_n = 0;
    int input_tokens = 0;
    int total_input_tokens = 0;
    double tokenization_time_ms = 0.0;
    double latency_ms = 0.0;
    double e2e_latency_ms = 0.0;
    double throughput_docs_per_s = 0.0;
    double throughput_tokens_per_s = 0.0;
    int top1_index = -1;
    double top1_score = 0.0;
    std::string result_md5;
    std::string start, end;
    IterMemory mem;
};

struct RerankBenchmarkResult {
    std::string model_dir, model_name;
    std::string framework = "ov";
    std::string pipeline = "TextRerankPipeline";
    std::string device, precision;

    int num_documents = 0;
    int top_n = 0;
    int max_length = 0;
    std::string padding_side;
    bool pad_to_max_length = false;
    int prompt_idx = 0;
    int input_tokens = 0;
    int total_input_tokens = 0;
    int num_iters = 0;
    int measured_iters = 0;
    double compile_time_s = 0.0;

    Stats latency, e2e, tokenization, thr_docs, thr_tokens;

    ResultMemory mem;
    CpuSummary cpu_util;
    GpuSummary gpu_util;
    NpuSummary npu_util;

    std::vector<RerankIterationMetrics> iterations;
    std::string openvino_version, openvino_genai_version, run_started_at;
};

// Attach a per-iteration memory summary (only when available).
void attach_memory(IterMemory& m, const MemorySummary& s);

// Fill the aggregated ResultMemory from initial/compile summaries + measured iters.
void fill_memory_stats(ResultMemory& out, const MemoryCurrent& initial,
                       const MemorySummary& compile_mem,
                       const std::vector<IterMemory>& measured_mem);

}  // namespace mb
