// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3-Embedding benchmark (ov::genai::TextEmbeddingPipeline) — C++ port of
// tests/benchmark_embedding.py + tests/lib/embed_runner.py.
#include <chrono>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include "common/common.hpp"
#include "common/deep_profile.hpp"
#include "common/md5.hpp"
#include "common/metrics.hpp"
#include "common/profiling.hpp"
#include "common/prompts.hpp"
#include "common/reports.hpp"
#include "common/stats.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/openvino.hpp"
#include "tasks/tasks.hpp"

namespace mb {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int count_tokens(ov::genai::Tokenizer* tok, const std::string& text) {
    if (!tok) return 0;
    try {
        return static_cast<int>(tok->encode(text).input_ids.get_shape().back());
    } catch (...) {
        return 0;
    }
}

ov::genai::TextEmbeddingPipeline::PoolingType parse_pooling(const std::string& s) {
    using P = ov::genai::TextEmbeddingPipeline::PoolingType;
    if (s == "cls") return P::CLS;
    if (s == "mean") return P::MEAN;
    return P::LAST_TOKEN;
}

// Get (n_vectors, dim) and a fingerprint string from EmbeddingResults variant.
void inspect_documents(const ov::genai::EmbeddingResults& res, int& n_vectors, int& dim,
                       std::string& fp) {
    n_vectors = 0;
    dim = 0;
    std::visit(
        [&](const auto& vecs) {
            n_vectors = static_cast<int>(vecs.size());
            if (!vecs.empty()) {
                dim = static_cast<int>(vecs[0].size());
                // fingerprint first vector's first few values
                std::string s;
                size_t lim = std::min<size_t>(vecs[0].size(), 8);
                for (size_t i = 0; i < lim; ++i) s += std::to_string(static_cast<double>(vecs[0][i])) + ",";
                fp = s;
            }
        },
        res);
}

void inspect_query(const ov::genai::EmbeddingResult& res, int& dim, std::string& fp) {
    dim = 0;
    std::visit(
        [&](const auto& vec) {
            dim = static_cast<int>(vec.size());
            std::string s;
            size_t lim = std::min<size_t>(vec.size(), 8);
            for (size_t i = 0; i < lim; ++i) s += std::to_string(static_cast<double>(vec[i])) + ",";
            fp = s;
        },
        res);
}

}  // namespace

int run_embedding(const Args& args) {
    if (args.model_dir.empty()) {
        std::fprintf(stderr, "ERROR: -m/--model-dir is required\n");
        return 1;
    }
    const std::string project_dir = detect_project_dir();
    std::optional<std::string> prompt = args.prompt;
    if (!prompt && args.prompt_file.empty() && !args.input_tokens) {
        std::printf("  [info] no --prompt / --prompt-file / --input-tokens; using a built-in default text\n");
        prompt = "Given a web search query, retrieve relevant passages that answer the query. "
                 "Query: What is the capital of France and what is it famous for?";
    }

    ensure_sudo_if_monitoring(args);

    std::printf("================================================================================\n");
    std::printf("  Qwen3-Embedding OpenVINO GenAI Text-Embedding Benchmark (C++)\n");
    std::printf("================================================================================\n");
    std::printf("  Devices:       ");
    for (auto& d : args.device) std::printf("%s ", d.c_str());
    std::printf("\n  Mode:          %s\n  Pooling:       %s\n  Normalize:     %s\n  Batch size:    %d\n",
                args.mode.c_str(), args.pooling_type.c_str(), args.normalize ? "True" : "False",
                args.batch_size);
    std::printf("  num_iters:     %d (+1 warm-up)\n  profile:       %s\n", args.num_iters,
                args.profile ? "ON" : "OFF");

    MonitorBundle monitors(!args.no_monitor, args.cpu_monitor_interval, args.gpu_monitor_interval,
                           args.npu_monitor_interval, args.gpu_command);
    monitors.start();
    MemoryUsageMonitor mem_mon(args.memory_monitor, args.memory_monitor_interval);
    mem_mon.start();

    std::vector<EmbedBenchmarkResult> all_results;
    const std::string ov_ver = openvino_version_string();
    const std::string genai_ver = openvino_genai_version_string();

    bool first_run = true;
    for (const auto& md : args.model_dir) {
        if (!std::filesystem::exists(md)) {
            std::fprintf(stderr, "ERROR: model directory not found: %s\n", md.c_str());
            continue;
        }
        for (const auto& dev_raw : args.device) {
            const std::string device = to_upper(dev_raw);
            if (!first_run && args.cooldown_between_runs > 0) {
                std::printf("\n  [cooldown] sleeping %ds before next run...\n", args.cooldown_between_runs);
                std::this_thread::sleep_for(std::chrono::seconds(args.cooldown_between_runs));
            }
            first_run = false;

            std::printf("\n################################################################################\n");
            std::printf("  Model: %s | Device: %s | Backend: TextEmbeddingPipeline\n",
                        path_name(md).c_str(), device.c_str());
            std::printf("################################################################################\n");

            ov::AnyMap plugin_cfg = build_plugin_config(args, device);

            ov::genai::TextEmbeddingPipeline::Config cfg;
            cfg.pooling_type = parse_pooling(args.pooling_type);
            cfg.normalize = args.normalize;
            if (args.max_length) cfg.max_length = static_cast<size_t>(*args.max_length);
            if (args.padding_side) cfg.padding_side = *args.padding_side;
            if (args.pad_to_max_length) cfg.pad_to_max_length = true;
            if (args.query_instruction) cfg.query_instruction = *args.query_instruction;
            if (args.embed_instruction) cfg.embed_instruction = *args.embed_instruction;

            std::unique_ptr<ov::genai::Tokenizer> tokenizer;
            try {
                tokenizer = std::make_unique<ov::genai::Tokenizer>(md);
            } catch (const std::exception& e) {
                std::printf("  [warn] could not load tokenizer for token accounting: %s\n", e.what());
            }

            MemoryCurrent initial_mem = mem_mon.current();
            mem_mon.reset();

            std::printf("  Compiling pipeline...\n");
            auto t0 = Clock::now();
            std::unique_ptr<ov::genai::TextEmbeddingPipeline> pipe;
            try {
                pipe = std::make_unique<ov::genai::TextEmbeddingPipeline>(md, device, cfg, plugin_cfg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  ERROR: failed to construct TextEmbeddingPipeline on %s: %s\n",
                             device.c_str(), e.what());
                continue;
            }
            double compile_time_s = ms_since(t0, Clock::now()) / 1000.0;
            MemorySummary compile_mem = mem_mon.summary();
            std::printf("  Compiled in %.2fs\n", compile_time_s);

            // Resolve texts.
            IndexedPrompts texts;
            try {
                if (args.input_tokens) {
                    if (prompt || !args.prompt_file.empty())
                        throw std::runtime_error("--input-tokens is mutually exclusive with --prompt / --prompt-file");
                    auto count_fn = [&](const std::string& t) { return count_tokens(tokenizer.get(), t); };
                    auto [p, actual] = build_prompt_from_token_target(*args.input_tokens, count_fn,
                                                                      args.input_base_text, project_dir);
                    std::printf("  Generated text from --input-tokens: target=%d, actual=%d\n",
                                *args.input_tokens, actual);
                    texts = {{0, p}};
                } else {
                    texts = load_prompts(prompt, args.prompt_file, args.prompt_index);
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "ERROR: %s\n", e.what());
                continue;
            }
            std::printf("  Texts loaded: %zu\n", texts.size());

            const std::string precision = detect_precision(md);
            const std::string run_started = now_iso8601();

            std::vector<std::string> modes;
            if (args.mode == "both") {
                modes = {"documents", "query"};
            } else {
                modes = {args.mode};
            }

            for (const auto& [prompt_idx, raw_text] : texts) {
                for (const std::string& mode : modes) {
                    std::vector<std::string> batch_texts;
                    if (mode == "documents") {
                        int bs = std::max(1, args.batch_size);
                        batch_texts.assign(bs, raw_text);
                    } else {
                        batch_texts = {raw_text};
                    }
                    int per_text = count_tokens(tokenizer.get(), raw_text);
                    std::printf("\n  [P%d] mode=%s batch=%zu per_text_tokens=%d  num_iters=%d (+1 warmup)\n",
                                prompt_idx, mode.c_str(), batch_texts.size(), per_text, args.num_iters);

                    monitors.reset();
                    std::vector<EmbedIterationMetrics> all_iters;
                    std::vector<IterMemory> measured_mem;

                    for (int num = 0; num <= args.num_iters; ++num) {
                        const bool is_warmup = (num == 0);
                        mem_mon.reset();

                        auto tk0 = Clock::now();
                        int per_text_tokens, total_tokens, batch;
                        if (mode == "query") {
                            per_text_tokens = count_tokens(tokenizer.get(), batch_texts[0]);
                            total_tokens = per_text_tokens;
                            batch = 1;
                        } else {
                            int mx = 0, sum = 0;
                            for (auto& t : batch_texts) {
                                int c = count_tokens(tokenizer.get(), t);
                                mx = std::max(mx, c);
                                sum += c;
                            }
                            per_text_tokens = mx;
                            total_tokens = sum;
                            batch = static_cast<int>(batch_texts.size());
                        }
                        double tok_ms = ms_since(tk0, Clock::now());

                        const std::string start_iso = now_iso8601();
                        auto t_s = Clock::now();
                        int dim = 0, n_vectors = 0;
                        std::string fp;
                        try {
                            if (mode == "query") {
                                auto r = pipe->embed_query(batch_texts[0]);
                                auto t_e = Clock::now();
                                inspect_query(r, dim, fp);
                                n_vectors = 1;
                                double latency_ms = ms_since(t_s, t_e);
                                (void)latency_ms;
                            } else {
                                auto r = pipe->embed_documents(batch_texts);
                                auto t_e = Clock::now();
                                inspect_documents(r, n_vectors, dim, fp);
                                (void)t_e;
                            }
                        } catch (const std::exception& e) {
                            std::printf("  [warn] iter %d failed: %s\n", num, e.what());
                            continue;
                        }
                        auto t_e = Clock::now();
                        double latency_ms = ms_since(t_s, t_e);
                        double e2e_ms = latency_ms + tok_ms;

                        EmbedIterationMetrics m;
                        m.iteration = num;
                        m.is_warmup = is_warmup;
                        m.prompt_idx = prompt_idx;
                        m.mode = mode;
                        m.batch_size = batch;
                        m.input_tokens = per_text_tokens;
                        m.total_input_tokens = total_tokens;
                        m.embedding_dim = dim;
                        m.tokenization_time_ms = tok_ms;
                        m.latency_ms = latency_ms;
                        m.e2e_latency_ms = e2e_ms;
                        m.throughput_docs_per_s = latency_ms > 0 ? batch / (latency_ms / 1000.0) : 0.0;
                        m.throughput_tokens_per_s =
                            (latency_ms > 0 && total_tokens) ? total_tokens / (latency_ms / 1000.0) : 0.0;
                        m.result_md5 = MD5::hex(fp);
                        m.start = start_iso;
                        m.end = now_iso8601();
                        MemorySummary itmem = mem_mon.summary();
                        attach_memory(m.mem, itmem);
                        all_iters.push_back(m);

                        const char* tag = is_warmup ? "[warm-up]" : "";
                        std::printf("    %s[%d][P%d] mode=%s batch=%d tokens(max/total)=%d/%d dim=%d "
                                    "lat=%.2fms docs/s=%.1f tok/s=%.0f\n",
                                    tag, num, prompt_idx, mode.c_str(), batch, per_text_tokens,
                                    total_tokens, dim, m.latency_ms, m.throughput_docs_per_s,
                                    m.throughput_tokens_per_s);
                    }

                    if (all_iters.empty()) {
                        std::printf("  [warn] no iterations collected; skipping\n");
                        continue;
                    }

                    std::vector<double> v_lat, v_e2e, v_tok, v_doc, v_ttok;
                    const EmbedIterationMetrics* rep = nullptr;
                    int measured = 0;
                    for (const auto& m : all_iters) {
                        if (m.is_warmup) continue;
                        if (!rep) rep = &m;
                        ++measured;
                        v_lat.push_back(m.latency_ms);
                        v_e2e.push_back(m.e2e_latency_ms);
                        v_tok.push_back(m.tokenization_time_ms);
                        v_doc.push_back(m.throughput_docs_per_s);
                        if (m.throughput_tokens_per_s > 0) v_ttok.push_back(m.throughput_tokens_per_s);
                        measured_mem.push_back(m.mem);
                    }
                    if (!rep) rep = &all_iters.front();

                    EmbedBenchmarkResult r;
                    r.model_dir = md;
                    r.model_name = path_name(md);
                    r.device = device;
                    r.precision = precision;
                    r.mode = mode;
                    r.batch_size = static_cast<int>(batch_texts.size());
                    r.pooling_type = args.pooling_type;
                    r.normalize = args.normalize;
                    r.max_length = args.max_length.value_or(0);
                    r.padding_side = args.padding_side.value_or("");
                    r.pad_to_max_length = args.pad_to_max_length;
                    r.query_instruction = args.query_instruction.value_or("");
                    r.embed_instruction = args.embed_instruction.value_or("");
                    r.prompt_idx = prompt_idx;
                    r.input_tokens = rep->input_tokens;
                    r.total_input_tokens = rep->total_input_tokens;
                    r.embedding_dim = rep->embedding_dim;
                    r.num_iters = args.num_iters;
                    r.measured_iters = measured;
                    r.compile_time_s = compile_time_s;
                    r.iterations = all_iters;
                    r.latency = compute_stats(v_lat);
                    r.e2e = compute_stats(v_e2e);
                    r.tokenization = compute_stats(v_tok);
                    r.thr_docs = compute_stats(v_doc);
                    r.thr_tokens = compute_stats(v_ttok);
                    r.cpu_util = monitors.cpu_summary();
                    r.gpu_util = monitors.gpu_summary();
                    r.npu_util = monitors.npu_summary();
                    fill_memory_stats(r.mem, initial_mem, compile_mem, measured_mem);
                    r.openvino_version = ov_ver;
                    r.openvino_genai_version = genai_ver;
                    r.run_started_at = run_started;
                    print_embed_summary(r);
                    all_results.push_back(std::move(r));
                }
            }

            // Optional REAL per-op profiling of the embedding model graph on the
            // actual tokenized input (not dummy zeros).
            if (args.profile && tokenizer) {
                DeepProfileConfig dp;
                dp.model_dir = md;
                dp.device = device;
                dp.precision = precision;
                dp.plugin_cfg = build_plugin_config(args, device, "", true);
                dp.gen_tokens = std::max(1, args.profile_iters);
                dp.warmup = args.profile_warmup;
                dp.topk = args.profile_topk;
                dp.profile_csv = args.profile_csv;
                dp.encoder_max_length = args.max_length.value_or(0);
                if (!texts.empty()) dp.encoder_texts = {texts.front().second};
                deep_profile_encoder(dp, *tokenizer, "embedding_graph");
            } else if (args.profile) {
                std::printf("  [profile] skipped: tokenizer unavailable for real-input profiling.\n");
            }

            pipe.reset();
        }
    }

    monitors.stop();
    mem_mon.stop();

    if (all_results.empty()) {
        std::printf("\nNo results collected.\n");
        return 0;
    }
    if (!args.csv_out.empty()) write_embed_csv(args.csv_out, all_results);
    if (!args.json_out.empty()) write_embed_json(args.json_out, all_results);
    std::printf("\n  Total runs: %zu\n  Done.\n", all_results.size());
    return 0;
}

}  // namespace mb
