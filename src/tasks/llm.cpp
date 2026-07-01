// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3 text-generation benchmark (ov::genai::LLMPipeline) — C++ port of
// tests/benchmark_qwen3.py + tests/lib/genai_runner.py.
#include <chrono>
#include <cstdio>
#include <optional>
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
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/openvino.hpp"
#include "tasks/tasks.hpp"

namespace mb {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// build_generation_config: mirror genai_runner.build_generation_config.
ov::genai::GenerationConfig build_gen_config(const Args& args, const ov::genai::GenerationConfig& base) {
    ov::genai::GenerationConfig cfg = base;
    cfg.max_new_tokens = static_cast<size_t>(args.infer_count);
    if (args.exact_output_tokens) {
        cfg.min_new_tokens = static_cast<size_t>(args.infer_count);
    }
    cfg.max_length = SIZE_MAX;
    cfg.ignore_eos = true;
    cfg.num_beams = static_cast<size_t>(args.num_beams);
    cfg.do_sample = false;
    cfg.rng_seed = static_cast<size_t>(args.seed);
    if (args.num_beams > 1) {
        cfg.frequency_penalty = 0;
        cfg.presence_penalty = 0;
        cfg.repetition_penalty = 1;
    }
    cfg.apply_chat_template = false;  // qwen3 path disables config chat template
    return cfg;
}

// permute_llm_inputs: replace token 0 (or 1 if BOS present) with iter+1,
// avoiding collision with EOS. Returns a new TokenizedInputs.
ov::genai::TokenizedInputs permute_inputs(const ov::genai::TokenizedInputs& tin, int iter_idx,
                                          int64_t bos_id, int64_t eos_id) {
    ov::Tensor src = tin.input_ids;
    ov::Tensor copy(src.get_element_type(), src.get_shape());
    std::memcpy(copy.data(), src.data(), src.get_byte_size());

    const auto shape = copy.get_shape();
    const size_t seq = shape.size() >= 2 ? shape[1] : shape[0];
    size_t replace_idx = 0;
    if (bos_id != -1 && seq > 1) replace_idx = 1;
    int64_t replacement = iter_idx + 1;
    if (eos_id == replacement) replacement += 2;

    if (copy.get_element_type() == ov::element::i64) {
        copy.data<int64_t>()[replace_idx] = replacement;
    } else if (copy.get_element_type() == ov::element::i32) {
        copy.data<int32_t>()[replace_idx] = static_cast<int32_t>(replacement);
    }
    ov::genai::TokenizedInputs out;
    out.input_ids = copy;
    out.attention_mask = tin.attention_mask;
    out.token_type_ids = tin.token_type_ids;
    return out;
}

double fallback_tpot_ms(double e2e_ms, double ttft_ms, int output_size) {
    if (output_size <= 1) return 0.0;
    double v = (e2e_ms - ttft_ms) / (output_size - 1);
    return v < 0 ? 0.0 : v;
}

}  // namespace

int run_llm(const Args& args) {
    if (args.model_dir.empty()) {
        std::fprintf(stderr, "ERROR: -m/--model-dir is required\n");
        return 1;
    }
    const std::string project_dir = detect_project_dir();
    std::string default_prompt =
        "Explain the key concepts of machine learning including supervised "
        "learning, unsupervised learning, and reinforcement learning.";
    std::optional<std::string> prompt = args.prompt;
    // input_tokens defaults to 1000. Explicit --prompt/--prompt-file always win over it;
    // --input-tokens <= 0 falls back to the built-in default prompt.
    const bool use_input_tokens =
        !prompt && args.prompt_file.empty() && args.input_tokens && *args.input_tokens > 0;
    if (!prompt && args.prompt_file.empty() && !use_input_tokens) {
        std::printf("  [info] no --prompt / --prompt-file and --input-tokens<=0; using a built-in default prompt\n");
        prompt = default_prompt;
    }

    ensure_sudo_if_monitoring(args);

    std::printf("================================================================================\n");
    std::printf("  Qwen3 OpenVINO GenAI Text-Generation Benchmark (C++)\n");
    std::printf("================================================================================\n");
    std::printf("  Devices:       ");
    for (auto& d : args.device) std::printf("%s ", d.c_str());
    std::printf("\n  infer_count:   %d\n  num_iters:     %d (+1 warm-up)\n", args.infer_count, args.num_iters);
    std::printf("  permutation:   %s\n", args.disable_prompt_permutation ? "OFF" : "ON");
    std::printf("  profile:       %s\n", args.profile ? "ON" : "OFF");

    MonitorBundle monitors(!args.no_monitor, args.cpu_monitor_interval, args.gpu_monitor_interval,
                           args.npu_monitor_interval, args.gpu_command);
    monitors.start();
    MemoryUsageMonitor mem_mon(args.memory_monitor, args.memory_monitor_interval);
    mem_mon.start();

    std::vector<GenBenchmarkResult> all_results;
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
            std::printf("  Model: %s | Device: %s | Backend: LLMPipeline\n", path_name(md).c_str(), device.c_str());
            std::printf("################################################################################\n");

            ov::AnyMap plugin_cfg = build_plugin_config(args, device, "PA");
            std::printf("  Plugin config:");
            for (auto& kv : plugin_cfg) std::printf(" %s=%s", kv.first.c_str(), kv.second.as<std::string>().c_str());
            std::printf("\n");

            MemoryCurrent initial_mem = mem_mon.current();
            mem_mon.reset();

            std::printf("  Compiling pipeline...\n");
            auto t0 = Clock::now();
            std::unique_ptr<ov::genai::LLMPipeline> pipe;
            try {
                pipe = std::make_unique<ov::genai::LLMPipeline>(md, device, plugin_cfg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  ERROR: failed to construct LLMPipeline on %s: %s\n",
                             device.c_str(), e.what());
                continue;
            }
            double compile_time_s = ms_since(t0, Clock::now()) / 1000.0;
            MemorySummary compile_mem = mem_mon.summary();
            std::printf("  Compiled in %.2fs\n", compile_time_s);

            ov::genai::Tokenizer tokenizer = pipe->get_tokenizer();
            ov::genai::GenerationConfig base_cfg = pipe->get_generation_config();
            const int64_t bos_id = tokenizer.get_bos_token_id();
            const int64_t eos_id = tokenizer.get_eos_token_id();

            auto count_final = [&](const std::string& raw) -> int {
                std::string text = raw;
                if (args.apply_chat_template) {
                    try {
                        ov::genai::ChatHistory h{{{"role", "user"}, {"content", raw}}};
                        text = tokenizer.apply_chat_template(h, true);
                    } catch (...) {
                    }
                }
                return static_cast<int>(tokenizer.encode(text).input_ids.get_shape().back());
            };

            // Resolve prompts.
            IndexedPrompts prompts;
            try {
                if (use_input_tokens) {
                    auto [p, actual] = build_prompt_from_token_target(*args.input_tokens, count_final,
                                                                      args.input_base_text, project_dir);
                    std::printf("  Generated prompt from --input-tokens: target=%d, actual=%d\n",
                                *args.input_tokens, actual);
                    prompts = {{0, p}};
                } else {
                    prompts = load_prompts(prompt, args.prompt_file, args.prompt_index);
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "ERROR: %s\n", e.what());
                continue;
            }
            std::printf("  Prompts loaded: %zu\n", prompts.size());

            const std::string precision = detect_precision(md);
            const std::string run_started = now_iso8601();
            const bool permute = !args.disable_prompt_permutation;

            for (const auto& [prompt_idx, raw_prompt] : prompts) {
                std::string prompt_text = raw_prompt;
                if (args.apply_chat_template) {
                    try {
                        ov::genai::ChatHistory h{{{"role", "user"}, {"content", raw_prompt}}};
                        prompt_text = tokenizer.apply_chat_template(h, true);
                    } catch (const std::exception& e) {
                        std::printf("  [warn] chat template failed (%s); using raw prompt\n", e.what());
                    }
                }
                const int in_tokens =
                    static_cast<int>(tokenizer.encode(prompt_text).input_ids.get_shape().back());
                std::printf("\n  [P%d] input_tokens=%d  infer_count(max_new)=%d  num_iters=%d (+1 warmup)  permute=%s\n",
                            prompt_idx, in_tokens, args.infer_count, args.num_iters,
                            permute ? "true" : "false");

                ov::genai::GenerationConfig gen_cfg = build_gen_config(args, base_cfg);

                monitors.reset();
                std::vector<GenIterationMetrics> all_iters;
                std::vector<IterMemory> iter_mems;

                for (int num = 0; num <= args.num_iters; ++num) {
                    const bool is_warmup = (num == 0);
                    mem_mon.reset();

                    auto tk0 = Clock::now();
                    ov::genai::TokenizedInputs tin = tokenizer.encode(prompt_text);
                    auto tk1 = Clock::now();
                    const int input_tokens = static_cast<int>(tin.input_ids.get_shape().back());

                    ov::genai::EncodedInputs inputs =
                        permute ? ov::genai::EncodedInputs(permute_inputs(tin, num, bos_id, eos_id))
                                : ov::genai::EncodedInputs(tin);

                    std::vector<Clock::time_point> token_times;
                    auto streamer = [&token_times](std::string) -> ov::genai::StreamingStatus {
                        token_times.push_back(Clock::now());
                        return ov::genai::StreamingStatus::RUNNING;
                    };

                    const std::string start_iso = now_iso8601();
                    auto t_start = Clock::now();
                    ov::genai::EncodedResults res = pipe->generate(inputs, gen_cfg, streamer);
                    auto t_end = Clock::now();
                    const std::string end_iso = now_iso8601();

                    const int n_tokens = static_cast<int>(token_times.size());
                    if (n_tokens == 0) {
                        std::printf("  [warn] iter %d: 0 tokens generated (excluded)\n", num);
                        continue;
                    }

                    // Decode for md5 + detok timing.
                    auto dt0 = Clock::now();
                    std::string output_text;
                    if (!res.tokens.empty()) {
                        try {
                            output_text = tokenizer.decode(res.tokens[0]);
                        } catch (...) {
                        }
                    }
                    auto dt1 = Clock::now();

                    // output size: prefer perf_metrics.
                    int output_size = n_tokens;
                    ov::genai::PerfMetrics pm = res.perf_metrics;
                    {
                        size_t g = pm.get_num_generated_tokens();
                        if (g > 0) output_size = static_cast<int>(g);
                    }

                    double e2e_ms = ms_since(t_start, t_end);
                    double ttft_ms = ms_since(t_start, token_times[0]);
                    double tpot_ms = fallback_tpot_ms(e2e_ms, ttft_ms, output_size);
                    // Refine with perf_metrics when >0.
                    {
                        double v = pm.get_ttft().mean;
                        if (v > 0) ttft_ms = v;
                        double w = pm.get_tpot().mean;
                        if (w > 0) tpot_ms = w;
                    }
                    // infer latencies from raw m_token_infer_durations (microseconds).
                    double first_infer_ms = -1.0, other_infer_ms = -1.0;
                    {
                        const auto& d = pm.raw_metrics.m_token_infer_durations;
                        if (!d.empty()) {
                            first_infer_ms = d[0].count() / 1000.0;
                            if (d.size() > 1) {
                                double s = 0;
                                for (size_t i = 1; i < d.size(); ++i) s += d[i].count() / 1000.0;
                                other_infer_ms = s / (d.size() - 1);
                            }
                        }
                    }

                    GenIterationMetrics m;
                    m.iteration = num;
                    m.is_warmup = is_warmup;
                    m.prompt_idx = prompt_idx;
                    m.input_tokens = input_tokens;
                    m.output_tokens = output_size;
                    m.infer_count = output_size;
                    m.generate_time_s = ms_since(t_start, t_end) / 1000.0;
                    m.ttft_ms = ttft_ms;
                    m.tpot_ms = tpot_ms;
                    m.first_token_infer_latency_ms = first_infer_ms;
                    m.other_tokens_infer_avg_latency_ms = other_infer_ms;
                    m.prefill_latency_ms = ttft_ms;
                    m.decode_latency_ms = output_size > 1 ? std::max(0.0, e2e_ms - ttft_ms) : 0.0;
                    m.e2e_latency_ms = e2e_ms;
                    m.throughput_tps = e2e_ms > 0 ? output_size / (e2e_ms / 1000.0) : 0.0;
                    m.decode_throughput_tps = tpot_ms > 0 ? 1000.0 / tpot_ms : 0.0;
                    m.tokenization_time_ms = ms_since(tk0, tk1);
                    m.detokenization_time_ms = ms_since(dt0, dt1);
                    m.result_md5 = MD5::hex(output_text);
                    m.start = start_iso;
                    m.end = end_iso;

                    MemorySummary itmem = mem_mon.summary();
                    attach_memory(m.mem, itmem);
                    all_iters.push_back(m);
                    iter_mems.push_back(m.mem);

                    const char* tag = is_warmup ? "[warm-up]" : "";
                    std::printf("    %s[%d][P%d] in=%d out=%d TTFT=%.1fms TPOT=%.2fms E2E=%.0fms TPS=%.1f\n",
                                tag, num, prompt_idx, input_tokens, output_size, m.ttft_ms, m.tpot_ms,
                                m.e2e_latency_ms, m.throughput_tps);
                }

                // Aggregate over measured (non-warmup) iters.
                std::vector<double> v_out, v_gen, v_ttft, v_tpot, v_first, v_other, v_thr, v_dthr,
                    v_pre, v_dec, v_e2e, v_tok, v_detok;
                std::vector<IterMemory> measured_mem;
                for (const auto& m : all_iters) {
                    if (m.is_warmup) continue;
                    v_out.push_back(m.output_tokens);
                    v_gen.push_back(m.generate_time_s);
                    v_ttft.push_back(m.ttft_ms);
                    if (m.tpot_ms > 0) v_tpot.push_back(m.tpot_ms);
                    if (m.first_token_infer_latency_ms >= 0) v_first.push_back(m.first_token_infer_latency_ms);
                    if (m.other_tokens_infer_avg_latency_ms >= 0) v_other.push_back(m.other_tokens_infer_avg_latency_ms);
                    v_thr.push_back(m.throughput_tps);
                    if (m.decode_throughput_tps > 0) v_dthr.push_back(m.decode_throughput_tps);
                    v_pre.push_back(m.prefill_latency_ms);
                    v_dec.push_back(m.decode_latency_ms);
                    v_e2e.push_back(m.e2e_latency_ms);
                    v_tok.push_back(m.tokenization_time_ms);
                    v_detok.push_back(m.detokenization_time_ms);
                    measured_mem.push_back(m.mem);
                }
                if (v_tpot.empty() && !all_iters.empty()) {
                    // mirror python warning path: all <=1 token
                    for (const auto& m : all_iters)
                        if (!m.is_warmup) v_tpot.push_back(m.tpot_ms);
                }

                GenBenchmarkResult r;
                r.model_dir = md;
                r.model_name = path_name(md);
                r.pipeline = "LLMPipeline";
                r.device = device;
                r.precision = precision;
                r.mode = "text-only";
                r.num_beams = args.num_beams;
                r.batch_size = args.batch_size;
                r.apply_chat_template = args.apply_chat_template;
                r.disable_prompt_permutation = args.disable_prompt_permutation;
                r.prompt_idx = prompt_idx;
                r.input_tokens = in_tokens;
                r.infer_count_arg = args.infer_count;
                r.exact_output_tokens = args.exact_output_tokens;
                r.num_iters = args.num_iters;
                int measured = 0;
                for (const auto& m : all_iters)
                    if (!m.is_warmup) ++measured;
                r.measured_iters = measured;
                r.compile_time_s = compile_time_s;
                r.iterations = all_iters;
                r.agg["output_tokens"] = compute_stats(v_out);
                r.agg["generation_time"] = compute_stats(v_gen);
                r.agg["ttft"] = compute_stats(v_ttft);
                r.agg["tpot"] = compute_stats(v_tpot);
                r.agg["first_token_infer"] = compute_stats(v_first);
                r.agg["other_tokens_infer"] = compute_stats(v_other);
                r.agg["throughput"] = compute_stats(v_thr);
                r.agg["decode_throughput"] = compute_stats(v_dthr);
                r.agg["prefill"] = compute_stats(v_pre);
                r.agg["decode"] = compute_stats(v_dec);
                r.agg["e2e"] = compute_stats(v_e2e);
                r.agg["tokenization"] = compute_stats(v_tok);
                r.agg["detokenization"] = compute_stats(v_detok);
                r.cpu_util = monitors.cpu_summary();
                r.gpu_util = monitors.gpu_summary();
                r.npu_util = monitors.npu_summary();
                fill_memory_stats(r.mem, initial_mem, compile_mem, measured_mem);
                r.openvino_version = ov_ver;
                r.openvino_genai_version = genai_ver;
                r.run_started_at = run_started;
                print_gen_summary(r);
                all_results.push_back(std::move(r));
            }

            // Optional REAL per-op profiling: re-implement the stateful greedy
            // loop on raw ov::Core (prefill + decode profiled separately).
            if (args.profile) {
                DeepProfileConfig dp;
                dp.model_dir = md;
                dp.device = device;
                dp.precision = precision;
                dp.plugin_cfg = build_plugin_config(args, device, "", true);
                dp.prompt_tokens = use_input_tokens ? *args.input_tokens : 64;
                // Profile the REAL prompt that the benchmark ran (-pf / --prompt),
                // not a synthetic one, so per-op hotspots reflect the true prefill
                // length. Only when we are NOT synthesizing from --input-tokens
                // (which already sets the synthetic length explicitly). Mirror the
                // benchmark's chat-template handling.
                if (!use_input_tokens && !prompts.empty()) {
                    std::string ptext = prompts.front().second;
                    if (args.apply_chat_template) {
                        try {
                            ov::genai::ChatHistory h{{{"role", "user"}, {"content", ptext}}};
                            ptext = tokenizer.apply_chat_template(h, true);
                        } catch (...) {
                        }
                    }
                    dp.prompt_text = ptext;
                }
                dp.gen_tokens = std::max(1, std::min(args.infer_count, args.profile_iters));
                dp.warmup = args.profile_warmup;
                dp.topk = args.profile_topk;
                dp.profile_csv = args.profile_csv;
                deep_profile_llm(dp, tokenizer);
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
    if (!args.csv_out.empty()) write_gen_csv(args.csv_out, all_results);
    if (!args.json_out.empty()) write_gen_json(args.json_out, all_results);
    std::printf("\n  Total runs: %zu\n  Done.\n", all_results.size());
    return 0;
}

}  // namespace mb
