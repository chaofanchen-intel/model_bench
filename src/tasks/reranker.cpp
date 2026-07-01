// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3-Reranker benchmark (ov::genai::TextRerankPipeline) — C++ port of
// tests/benchmark_reranker.py + tests/lib/embed_runner.py (rerank path).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "common/deep_profile.hpp"
#include "common/md5.hpp"
#include "common/metrics.hpp"
#include "common/profiling.hpp"
#include "common/prompts.hpp"
#include "common/reports.hpp"
#include "common/stats.hpp"
#include "openvino/genai/rag/text_rerank_pipeline.hpp"
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

const std::vector<std::string> kDefaultDocuments = {
    "Intel Core Ultra processors incorporate an AI-optimized architecture that supports "
    "new user experiences and the next wave of commercial applications.",
    "Intel Core Ultra processors are designed to provide enhanced performance and "
    "efficiency for a wide range of computing tasks.",
    "The Eiffel Tower is one of the most recognizable landmarks in Paris, France.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy."};

}  // namespace

int run_reranker(const Args& args) {
    if (args.model_dir.empty()) {
        std::fprintf(stderr, "ERROR: -m/--model-dir is required\n");
        return 1;
    }
    std::optional<std::string> query = args.prompt;
    if (!query && args.prompt_file.empty()) {
        std::printf("  [info] no --prompt / --prompt-file query; using a built-in default query\n");
        query = "What is the capital of France?";
    }

    // Resolve documents.
    std::vector<std::string> documents;
    try {
        if (!args.texts.empty() && !args.texts_file.empty()) {
            throw std::runtime_error("--texts and --texts-file are mutually exclusive");
        }
        if (!args.texts_file.empty()) {
            documents = load_texts_jsonl(args.texts_file);
        } else if (!args.texts.empty()) {
            documents = args.texts;
        } else {
            documents = kDefaultDocuments;
        }
        if (documents.empty()) throw std::runtime_error("no candidate documents resolved");
        if (args.num_documents && *args.num_documents > 0) {
            int n = *args.num_documents;
            std::vector<std::string> resized;
            if (static_cast<int>(documents.size()) >= n) {
                resized.assign(documents.begin(), documents.begin() + n);
            } else {
                int reps = (n + static_cast<int>(documents.size()) - 1) / static_cast<int>(documents.size());
                for (int r = 0; r < reps; ++r)
                    resized.insert(resized.end(), documents.begin(), documents.end());
                resized.resize(n);
            }
            documents = resized;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    ensure_sudo_if_monitoring(args);

    std::printf("================================================================================\n");
    std::printf("  Qwen3-Reranker OpenVINO GenAI Text-Reranking Benchmark (C++)\n");
    std::printf("================================================================================\n");
    std::printf("  Devices:       ");
    for (auto& d : args.device) std::printf("%s ", d.c_str());
    std::printf("\n  Documents:     %zu\n  top_n:         %d\n  num_iters:     %d (+1 warm-up)\n",
                documents.size(), args.top_n, args.num_iters);
    std::printf("  profile:       %s\n", args.profile ? "ON" : "OFF");

    MonitorBundle monitors(!args.no_monitor, args.cpu_monitor_interval, args.gpu_monitor_interval,
                           args.npu_monitor_interval, args.gpu_command);
    monitors.start();
    MemoryUsageMonitor mem_mon(args.memory_monitor, args.memory_monitor_interval);
    mem_mon.start();

    std::vector<RerankBenchmarkResult> all_results;
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
            std::printf("  Model: %s | Device: %s | Backend: TextRerankPipeline\n",
                        path_name(md).c_str(), device.c_str());
            std::printf("################################################################################\n");

            ov::AnyMap plugin_cfg = build_plugin_config(args, device);

            ov::genai::TextRerankPipeline::Config cfg;
            cfg.top_n = static_cast<size_t>(args.top_n);
            if (args.max_length) cfg.max_length = static_cast<size_t>(*args.max_length);
            if (args.padding_side) cfg.padding_side = *args.padding_side;
            if (args.pad_to_max_length) cfg.pad_to_max_length = true;

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
            std::unique_ptr<ov::genai::TextRerankPipeline> pipe;
            try {
                pipe = std::make_unique<ov::genai::TextRerankPipeline>(md, device, cfg, plugin_cfg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  ERROR: failed to construct TextRerankPipeline on %s: %s\n",
                             device.c_str(), e.what());
                continue;
            }
            double compile_time_s = ms_since(t0, Clock::now()) / 1000.0;
            MemorySummary compile_mem = mem_mon.summary();
            std::printf("  Compiled in %.2fs\n", compile_time_s);

            IndexedPrompts queries;
            try {
                queries = load_prompts(query, args.prompt_file, args.prompt_index);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "ERROR: %s\n", e.what());
                continue;
            }
            int effective_top_n = std::min(args.top_n, static_cast<int>(documents.size()));
            std::printf("  Queries: %zu  Documents: %zu  top_n: %d\n", queries.size(), documents.size(),
                        effective_top_n);

            const std::string precision = detect_precision(md);
            const std::string run_started = now_iso8601();

            for (const auto& [prompt_idx, q] : queries) {
                int q_tokens = count_tokens(tokenizer.get(), q);
                std::printf("\n  [P%d] query_tokens=%d docs=%zu  num_iters=%d (+1 warmup)\n", prompt_idx,
                            q_tokens, documents.size(), args.num_iters);

                monitors.reset();
                std::vector<RerankIterationMetrics> all_iters;
                std::vector<IterMemory> measured_mem;

                for (int num = 0; num <= args.num_iters; ++num) {
                    const bool is_warmup = (num == 0);
                    mem_mon.reset();

                    auto tk0 = Clock::now();
                    int qt = count_tokens(tokenizer.get(), q);
                    int doc_max = 0, doc_total = 0;
                    for (auto& d : documents) {
                        int c = count_tokens(tokenizer.get(), d);
                        doc_max = std::max(doc_max, c);
                        doc_total += c;
                    }
                    int per_pair_max = qt + doc_max;
                    int total_tokens = qt * static_cast<int>(documents.size()) + doc_total;
                    double tok_ms = ms_since(tk0, Clock::now());

                    const std::string start_iso = now_iso8601();
                    auto t_s = Clock::now();
                    std::vector<std::pair<size_t, float>> result;
                    try {
                        result = pipe->rerank(q, documents);
                    } catch (const std::exception& e) {
                        std::printf("  [warn] iter %d failed: %s\n", num, e.what());
                        continue;
                    }
                    auto t_e = Clock::now();

                    double latency_ms = ms_since(t_s, t_e);
                    double e2e_ms = latency_ms + tok_ms;
                    int n_docs = static_cast<int>(documents.size());

                    RerankIterationMetrics m;
                    m.iteration = num;
                    m.is_warmup = is_warmup;
                    m.prompt_idx = prompt_idx;
                    m.num_documents = n_docs;
                    m.top_n = effective_top_n;
                    m.input_tokens = per_pair_max;
                    m.total_input_tokens = total_tokens;
                    m.tokenization_time_ms = tok_ms;
                    m.latency_ms = latency_ms;
                    m.e2e_latency_ms = e2e_ms;
                    m.throughput_docs_per_s = latency_ms > 0 ? n_docs / (latency_ms / 1000.0) : 0.0;
                    m.throughput_tokens_per_s =
                        (latency_ms > 0 && total_tokens) ? total_tokens / (latency_ms / 1000.0) : 0.0;
                    if (!result.empty()) {
                        m.top1_index = static_cast<int>(result[0].first);
                        m.top1_score = result[0].second;
                    }
                    // fingerprint result
                    std::string fp;
                    for (auto& pr : result)
                        fp += std::to_string(pr.first) + ":" + std::to_string(pr.second) + ",";
                    m.result_md5 = MD5::hex(fp);
                    m.start = start_iso;
                    m.end = now_iso8601();
                    MemorySummary itmem = mem_mon.summary();
                    attach_memory(m.mem, itmem);
                    all_iters.push_back(m);

                    const char* tag = is_warmup ? "[warm-up]" : "";
                    std::printf("    %s[%d][P%d] docs=%d top_n=%d tokens(pair_max/total)=%d/%d "
                                "lat=%.2fms docs/s=%.1f top1=(#%d,%.4f)\n",
                                tag, num, prompt_idx, n_docs, effective_top_n, per_pair_max,
                                total_tokens, m.latency_ms, m.throughput_docs_per_s, m.top1_index,
                                m.top1_score);
                }

                if (all_iters.empty()) {
                    std::printf("  [warn] no iterations collected; skipping\n");
                    continue;
                }

                std::vector<double> v_lat, v_e2e, v_tok, v_doc, v_ttok;
                const RerankIterationMetrics* rep = nullptr;
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

                RerankBenchmarkResult r;
                r.model_dir = md;
                r.model_name = path_name(md);
                r.device = device;
                r.precision = precision;
                r.num_documents = static_cast<int>(documents.size());
                r.top_n = effective_top_n;
                r.max_length = args.max_length.value_or(0);
                r.padding_side = args.padding_side.value_or("");
                r.pad_to_max_length = args.pad_to_max_length;
                r.prompt_idx = prompt_idx;
                r.input_tokens = rep->input_tokens;
                r.total_input_tokens = rep->total_input_tokens;
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
                print_rerank_summary(r);
                all_results.push_back(std::move(r));
            }

            // Optional REAL per-op profiling on an actual (query, document) pair.
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
                // Use the first query + first document joined, a representative pair.
                std::string sample = query.value_or("");
                if (!documents.empty()) sample += " " + documents.front();
                dp.encoder_texts = {sample};
                deep_profile_encoder(dp, *tokenizer, "reranker_graph");
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
    if (!args.csv_out.empty()) write_rerank_csv(args.csv_out, all_results);
    if (!args.json_out.empty()) write_rerank_json(args.json_out, all_results);
    std::printf("\n  Total runs: %zu\n  Done.\n", all_results.size());
    return 0;
}

}  // namespace mb
