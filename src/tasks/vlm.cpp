// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 vision-language benchmark (ov::genai::VLMPipeline) — C++ port of
// tests/benchmark_qwen3_5.py (the GenAI VLMPipeline path; the optimum.intel
// GPU-fallback path has no C++ equivalent and is intentionally dropped).
//
// VLM uses NO streamer: TTFT = get_ttft().mean minus last tokenization
// duration; TPOT = mean of consecutive diffs of raw_metrics.m_new_token_times.
// Permutation is a string prefix ("{iter} ") and is OFF by default.
//
// Images are loaded via OpenCV when available (the GenAI samples link OpenCV).
// To avoid a hard OpenCV dependency we read common image formats through
// stb-like decoding only if --image is given; otherwise text-only runs need no
// image decoder. We load images with OpenVINO's own opencv-free path by
// shelling out is avoided — instead text-only is fully supported and --image
// requires the build to have been configured WITH_OPENCV (see CMake).
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "common/deep_profile.hpp"
#include "common/md5.hpp"
#include "common/metrics.hpp"
#include "common/prompts.hpp"
#include "common/reports.hpp"
#include "common/stats.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/openvino.hpp"
#include "tasks/tasks.hpp"

#ifdef MB_WITH_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace mb {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

ov::genai::GenerationConfig build_gen_config(const Args& args, const ov::genai::GenerationConfig& base) {
    ov::genai::GenerationConfig cfg = base;
    cfg.max_new_tokens = static_cast<size_t>(args.infer_count);
    if (args.exact_output_tokens) cfg.min_new_tokens = static_cast<size_t>(args.infer_count);
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
    // For VLMPipeline the Python path does NOT disable apply_chat_template in cfg.
    return cfg;
}

double mean_micro_to_ms(const std::vector<ov::genai::MicroSeconds>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (auto& x : v) s += x.count();
    return (s / v.size()) / 1000.0;
}

#ifdef MB_WITH_OPENCV
ov::Tensor load_image_tensor(const std::string& path) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("failed to read image: " + path);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    ov::Tensor t(ov::element::u8, {static_cast<size_t>(rgb.rows), static_cast<size_t>(rgb.cols), 3});
    std::memcpy(t.data(), rgb.data, t.get_byte_size());
    return t;
}
#endif

}  // namespace

int run_vlm(const Args& args) {
    if (args.model_dir.empty()) {
        std::fprintf(stderr, "ERROR: -m/--model-dir is required\n");
        return 1;
    }
    const std::string project_dir = detect_project_dir();
    std::optional<std::string> prompt = args.prompt;
    // An --image run needs a real image+prompt, so it forces the inline default and
    // ignores --input-tokens (a synthetic text prompt makes no sense there).
    if (args.image && !prompt && args.prompt_file.empty()) {
        prompt = "Describe this image in detail, including objects, colors, "
                 "spatial relationships, and any visible text.";
        std::printf("  [info] --image with no --prompt; using built-in image default\n");
    }
    // input_tokens defaults to 1000. Explicit --prompt/--prompt-file always win over it;
    // --input-tokens <= 0 (or --image) falls back to the built-in default prompt.
    const bool use_input_tokens =
        !prompt && args.prompt_file.empty() && !args.image &&
        args.input_tokens && *args.input_tokens > 0;
    if (!prompt && args.prompt_file.empty() && !use_input_tokens) {
        prompt = "Explain the key concepts of machine learning including "
                 "supervised, unsupervised, and reinforcement learning.";
        std::printf("  [info] no --prompt / --prompt-file and --input-tokens<=0; using built-in default\n");
    }

    bool have_image = false;
    ov::Tensor image_tensor;
    if (args.image) {
#ifdef MB_WITH_OPENCV
        try {
            image_tensor = load_image_tensor(*args.image);
            have_image = true;
            std::printf("  [image] %s (shape=%s)\n", path_name(*args.image).c_str(),
                        image_tensor.get_shape().to_string().c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ERROR: %s\n", e.what());
            return 1;
        }
#else
        std::fprintf(stderr,
                     "ERROR: --image requires building with OpenCV (cmake -DMB_WITH_OPENCV=ON). "
                     "Text-only VLM runs work without it.\n");
        return 1;
#endif
    }

    ensure_sudo_if_monitoring(args);

    std::printf("================================================================================\n");
    std::printf("  Qwen3.5 OpenVINO GenAI VLM Benchmark (C++)\n");
    std::printf("================================================================================\n");
    std::printf("  Devices:       ");
    for (auto& d : args.device) std::printf("%s ", d.c_str());
    std::printf("\n  Mode:          %s\n  infer_count:   %d\n  num_iters:     %d (+1 warm-up)\n",
                have_image ? "multimodal" : "text-only", args.infer_count, args.num_iters);
    std::printf("  permutation:   %s\n  profile:       %s\n",
                args.disable_prompt_permutation ? "OFF" : "ON", args.profile ? "ON" : "OFF");

    MonitorBundle monitors(!args.no_monitor, args.cpu_monitor_interval, args.gpu_monitor_interval,
                           args.npu_monitor_interval, args.gpu_command);
    monitors.start();
    MemoryUsageMonitor mem_mon(args.memory_monitor, args.memory_monitor_interval);
    mem_mon.start();

    std::vector<GenBenchmarkResult> all_results;
    const std::string ov_ver = openvino_version_string();
    const std::string genai_ver = openvino_genai_version_string();
    const bool permute = !args.disable_prompt_permutation;

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
            std::printf("  Model: %s | Device: %s | Mode: %s | Backend: VLMPipeline\n",
                        path_name(md).c_str(), device.c_str(), have_image ? "multimodal" : "text-only");
            std::printf("################################################################################\n");

            ov::AnyMap plugin_cfg = build_plugin_config(args, device, "PA");
            std::printf("  Plugin config:");
            for (auto& kv : plugin_cfg) std::printf(" %s=%s", kv.first.c_str(), kv.second.as<std::string>().c_str());
            std::printf("\n");

            MemoryCurrent initial_mem = mem_mon.current();
            mem_mon.reset();

            std::printf("  Compiling VLMPipeline...\n");
            auto t0 = Clock::now();
            std::unique_ptr<ov::genai::VLMPipeline> pipe;
            try {
                pipe = std::make_unique<ov::genai::VLMPipeline>(md, device, plugin_cfg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  ERROR: failed to construct VLMPipeline on %s: %s\n",
                             device.c_str(), e.what());
                continue;
            }
            double compile_time_s = ms_since(t0, Clock::now()) / 1000.0;
            MemorySummary compile_mem = mem_mon.summary();
            std::printf("  Compiled in %.2fs\n", compile_time_s);

            ov::genai::Tokenizer tokenizer = pipe->get_tokenizer();
            ov::genai::GenerationConfig base_cfg = pipe->get_generation_config();

            auto count_final = [&](const std::string& raw) -> int {
                std::string text = raw;
                if (args.apply_chat_template) {
                    try {
                        ov::genai::ChatHistory h{{{"role", "user"}, {"content", raw}}};
                        text = tokenizer.apply_chat_template(h, true);
                    } catch (...) {
                    }
                }
                if (permute) text = "1 " + text;
                return static_cast<int>(tokenizer.encode(text).input_ids.get_shape().back());
            };

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
                int in_tokens = 0;
                try {
                    in_tokens = static_cast<int>(tokenizer.encode(prompt_text).input_ids.get_shape().back());
                } catch (...) {
                }
                std::printf("\n  [P%d] raw_text_tokens=%d  infer_count(max_new)=%d  num_iters=%d (+1 warmup)  permute=%s\n",
                            prompt_idx, in_tokens, args.infer_count, args.num_iters,
                            permute ? "true" : "false");

                ov::genai::GenerationConfig gen_cfg = build_gen_config(args, base_cfg);

                monitors.reset();
                std::vector<GenIterationMetrics> all_iters;
                std::vector<IterMemory> measured_mem;

                for (int num = 0; num <= args.num_iters; ++num) {
                    const bool is_warmup = (num == 0);
                    mem_mon.reset();
                    std::string eff_prompt = permute ? (std::to_string(num) + " " + prompt_text) : prompt_text;

                    const std::string start_iso = now_iso8601();
                    auto t_start = Clock::now();
                    ov::genai::VLMDecodedResults res;
                    try {
                        if (have_image) {
                            res = pipe->generate(eff_prompt, std::vector<ov::Tensor>{image_tensor},
                                                 gen_cfg, std::monostate());
                        } else {
                            res = pipe->generate(eff_prompt, std::vector<ov::Tensor>{}, gen_cfg,
                                                 std::monostate());
                        }
                    } catch (const std::exception& e) {
                        std::printf("  [warn] iter %d failed: %s\n", num, e.what());
                        continue;
                    }
                    auto t_end = Clock::now();
                    const std::string end_iso = now_iso8601();

                    ov::genai::VLMPerfMetrics pm = res.perf_metrics;
                    int output_size = static_cast<int>(pm.get_num_generated_tokens());
                    if (output_size == 0) {
                        std::printf("  [warn] iter %d: 0 tokens generated (excluded)\n", num);
                        continue;
                    }
                    int input_tokens = static_cast<int>(pm.get_num_input_tokens());
                    if (input_tokens == 0) input_tokens = in_tokens;

                    std::string output_text = res.texts.empty() ? "" : res.texts[0];

                    double e2e_ms = ms_since(t_start, t_end);
                    // TTFT = get_ttft().mean minus last tokenization duration (ms).
                    double ttft_ms = pm.get_ttft().mean;
                    {
                        const auto& tok_durs = pm.raw_metrics.tokenization_durations;
                        if (!tok_durs.empty()) ttft_ms -= tok_durs.back().count() / 1000.0;
                        if (ttft_ms < 0) ttft_ms = 0;
                    }
                    // TPOT = mean of consecutive diffs of m_new_token_times.
                    double tpot_ms = 0.0;
                    {
                        const auto& nt = pm.raw_metrics.m_new_token_times;
                        if (nt.size() > 1) {
                            double s = 0;
                            for (size_t i = 1; i < nt.size(); ++i) {
                                s += std::chrono::duration<double, std::milli>(nt[i] - nt[i - 1]).count();
                            }
                            tpot_ms = s / (nt.size() - 1);
                        }
                    }
                    if (tpot_ms <= 0) {
                        // fallback
                        tpot_ms = output_size > 1 ? std::max(0.0, e2e_ms - ttft_ms) / (output_size - 1) : 0.0;
                    }
                    double tok_ms = mean_micro_to_ms(pm.raw_metrics.tokenization_durations);
                    double detok_ms = mean_micro_to_ms(pm.raw_metrics.detokenization_durations);

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
                    m.generate_time_s = e2e_ms / 1000.0;
                    m.ttft_ms = ttft_ms;
                    m.tpot_ms = tpot_ms;
                    m.first_token_infer_latency_ms = first_infer_ms;
                    m.other_tokens_infer_avg_latency_ms = other_infer_ms;
                    m.prefill_latency_ms = ttft_ms;
                    m.decode_latency_ms = output_size > 1 ? std::max(0.0, e2e_ms - ttft_ms) : 0.0;
                    m.e2e_latency_ms = e2e_ms;
                    m.throughput_tps = e2e_ms > 0 ? output_size / (e2e_ms / 1000.0) : 0.0;
                    m.decode_throughput_tps = tpot_ms > 0 ? 1000.0 / tpot_ms : 0.0;
                    m.tokenization_time_ms = tok_ms;
                    m.detokenization_time_ms = detok_ms;
                    m.result_md5 = MD5::hex(output_text);
                    m.start = start_iso;
                    m.end = end_iso;
                    MemorySummary itmem = mem_mon.summary();
                    attach_memory(m.mem, itmem);
                    all_iters.push_back(m);

                    const char* tag = is_warmup ? "[warm-up]" : "";
                    std::printf("    %s[%d][P%d] in=%d out=%d TTFT=%.1fms TPOT=%.2fms E2E=%.0fms TPS=%.1f\n",
                                tag, num, prompt_idx, input_tokens, output_size, m.ttft_ms, m.tpot_ms,
                                m.e2e_latency_ms, m.throughput_tps);
                }

                if (all_iters.empty()) {
                    std::printf("  [warn] no iterations collected; skipping\n");
                    continue;
                }

                std::vector<double> v_out, v_gen, v_ttft, v_tpot, v_first, v_other, v_thr, v_dthr,
                    v_pre, v_dec, v_e2e, v_tok, v_detok;
                std::vector<int> measured_in;
                int measured = 0;
                for (const auto& m : all_iters) {
                    if (m.is_warmup) continue;
                    ++measured;
                    measured_in.push_back(m.input_tokens);
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
                int report_in = !measured_in.empty() ? measured_in[0] : in_tokens;

                GenBenchmarkResult r;
                r.model_dir = md;
                r.model_name = path_name(md);
                r.pipeline = "VLMPipeline";
                r.device = device;
                r.precision = precision;
                r.mode = have_image ? "multimodal" : "text-only";
                r.num_beams = args.num_beams;
                r.batch_size = args.batch_size;
                r.apply_chat_template = args.apply_chat_template;
                r.disable_prompt_permutation = args.disable_prompt_permutation;
                r.prompt_idx = prompt_idx;
                r.input_tokens = report_in;
                r.infer_count_arg = args.infer_count;
                r.exact_output_tokens = args.exact_output_tokens;
                r.num_iters = args.num_iters;
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
                r.image_path = args.image.value_or("");
                r.openvino_version = ov_ver;
                r.openvino_genai_version = genai_ver;
                r.run_started_at = run_started;
                print_gen_summary(r);
                all_results.push_back(std::move(r));
            }

            if (args.profile) {
                if (have_image) {
                    std::printf("\n  [profile] VLM deep profiling currently covers the text-only path\n");
                    std::printf("            (text_embeddings + language_model). Image/vision-tower profiling\n");
                    std::printf("            is not yet wired; reporting the LM path below.\n");
                }
                DeepProfileConfig dp;
                dp.model_dir = md;
                dp.device = device;
                dp.precision = precision;
                dp.plugin_cfg = build_plugin_config(args, device, "", true);
                dp.prompt_tokens = use_input_tokens ? *args.input_tokens : 64;
                // Profile the REAL prompt the benchmark ran (-pf / --prompt) so
                // per-op hotspots reflect the true prefill length, unless we are
                // synthesizing from --input-tokens (explicit synthetic length).
                // Mirror the benchmark's chat-template handling.
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
                deep_profile_vlm(dp, tokenizer);
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
