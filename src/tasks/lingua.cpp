// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// LLMLingua-2 (BertForTokenClassification / XLM-RoBERTa) benchmark using raw
// ov::Core — C++ port of tests/benchmark_lingua.py. This is the path where
// per-operation profiling (ov::ProfilingInfo) is directly usable, because we
// own the InferRequest.
//
// Tokenization uses ov::genai::Tokenizer (lingua dirs ship openvino_tokenizer.xml).
// NOTE: token counts may differ slightly from the Python HuggingFace
// AutoTokenizer path; this shifts input_tokens / compressed_tokens but not the
// latency measurement. Documented in README.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "common/common.hpp"
#include "common/metrics.hpp"
#include "common/profiling.hpp"
#include "common/reports.hpp"
#include "common/stats.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/openvino.hpp"
#include "tasks/tasks.hpp"

namespace mb {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

const std::vector<std::string> kSampleTexts = {
    "The quarterly earnings report shows a significant increase in revenue compared to the previous fiscal year. "
    "Our cloud computing division has grown by thirty-seven percent, driven primarily by enterprise adoption of "
    "our AI and machine learning services. The board recommends increasing the dividend payout ratio given the "
    "strong cash flow generation and reduced capital expenditure requirements for the upcoming quarter.",
    "In today's meeting we discussed the product roadmap for Q3 and Q4. The engineering team presented their "
    "progress on the new authentication system which uses biometric verification combined with hardware tokens. "
    "Marketing shared the results of the A/B testing campaign showing a twelve percent improvement in conversion "
    "rates when using the redesigned landing page with the new call-to-action button placement.",
    "The research paper presents a novel approach to distributed consensus in Byzantine fault-tolerant systems. "
    "Unlike previous protocols that require three rounds of communication, our method achieves agreement in just "
    "two rounds under normal conditions while maintaining safety guarantees even when up to one-third of nodes "
    "exhibit arbitrary failures. We prove correctness formally using the TLA+ specification language."};

// Extract int64 token ids from a [1, seq] i64 tensor.
std::vector<int64_t> tensor_to_ids(const ov::Tensor& t) {
    std::vector<int64_t> out;
    const size_t n = t.get_size();
    out.resize(n);
    if (t.get_element_type() == ov::element::i64) {
        std::memcpy(out.data(), t.data(), n * sizeof(int64_t));
    } else if (t.get_element_type() == ov::element::i32) {
        const int32_t* p = t.data<const int32_t>();
        for (size_t i = 0; i < n; ++i) out[i] = p[i];
    }
    return out;
}

// Build an input text of approx `target_tokens` tokens (excluding special)
// by tokenizing the joined sample text, repeating/slicing ids, and decoding.
std::string build_input_text(ov::genai::Tokenizer& tok, int target_tokens) {
    std::string base;
    for (size_t i = 0; i < kSampleTexts.size(); ++i) {
        if (i) base += " ";
        base += kSampleTexts[i];
    }
    ov::genai::TokenizedInputs enc =
        tok.encode(base, ov::AnyMap{{"add_special_tokens", false}});
    std::vector<int64_t> ids = tensor_to_ids(enc.input_ids);
    if (ids.empty()) {
        throw std::runtime_error("sample text tokenized to zero tokens");
    }
    std::vector<int64_t> sliced;
    if (static_cast<int>(ids.size()) < target_tokens) {
        int repeat = (target_tokens + static_cast<int>(ids.size()) - 1) / static_cast<int>(ids.size());
        for (int r = 0; r < repeat; ++r)
            sliced.insert(sliced.end(), ids.begin(), ids.end());
        sliced.resize(target_tokens);
    } else {
        sliced.assign(ids.begin(), ids.begin() + target_tokens);
    }
    return tok.decode(sliced, ov::AnyMap{{"skip_special_tokens", true}});
}

void softmax_keepcol(const float* logits, size_t seq, size_t num_classes,
                     std::vector<float>& keep_probs) {
    // keep prob = softmax(logits)[:,1]
    keep_probs.resize(seq);
    for (size_t i = 0; i < seq; ++i) {
        const float* row = logits + i * num_classes;
        float mx = row[0];
        for (size_t c = 1; c < num_classes; ++c) mx = std::max(mx, row[c]);
        double sum = 0.0;
        for (size_t c = 0; c < num_classes; ++c) sum += std::exp(row[c] - mx);
        double p1 = num_classes > 1 ? std::exp(row[1] - mx) / sum : 0.0;
        keep_probs[i] = static_cast<float>(p1);
    }
}

}  // namespace

int run_lingua(const Args& args) {
    if (args.model_dir.empty()) {
        std::fprintf(stderr, "ERROR: -m/--model-dir is required\n");
        return 1;
    }
    ensure_sudo_if_monitoring(args);

    std::printf("================================================================================\n");
    std::printf("  LLMLingua-2 OpenVINO Benchmark (C++, raw ov::Core)\n");
    std::printf("================================================================================\n");
    std::printf("  Devices:       ");
    for (auto& d : args.device) std::printf("%s ", d.c_str());
    std::printf("\n  Input lengths: ");
    for (int l : args.input_lengths) std::printf("%d ", l);
    std::printf("\n  Keep ratios:   ");
    for (double r : args.compression_rates) std::printf("%.2f ", r);
    std::printf("\n  Warmup/Iters:  %d/%d\n  profile:       %s\n", args.warmup, args.lingua_iters,
                args.profile ? "ON" : "OFF");

    MonitorBundle monitors(!args.no_monitor, args.cpu_monitor_interval, args.gpu_monitor_interval,
                           args.npu_monitor_interval, args.gpu_command);
    monitors.start();
    MemoryUsageMonitor mem_mon(args.memory_monitor, args.memory_monitor_interval);
    mem_mon.start();

    std::vector<BertBenchmarkResult> all_results;
    const std::string ov_ver = openvino_version_string();

    bool first_run = true;
    for (const auto& md : args.model_dir) {
        std::filesystem::path xml = std::filesystem::path(md) / "openvino_model.xml";
        if (!std::filesystem::exists(xml)) {
            std::fprintf(stderr, "ERROR: openvino_model.xml not found in: %s\n", md.c_str());
            continue;
        }
        ov::genai::Tokenizer tokenizer(md);

        for (const auto& dev_raw : args.device) {
            const std::string device = to_upper(dev_raw);
            if (!first_run && args.cooldown_between_runs > 0) {
                std::printf("\n  [cooldown] sleeping %ds before next run...\n", args.cooldown_between_runs);
                std::this_thread::sleep_for(std::chrono::seconds(args.cooldown_between_runs));
            }
            first_run = false;

            std::printf("\n################################################################################\n");
            std::printf("  Model: %s | Device: %s\n", path_name(md).c_str(), device.c_str());
            std::printf("################################################################################\n");

            ov::AnyMap cfg = build_plugin_config(args, device, "", args.profile);

            MemoryCurrent initial_mem = mem_mon.current();
            mem_mon.reset();

            std::printf("  Compiling model on %s...\n", device.c_str());
            auto t0 = Clock::now();
            ov::Core core;
            ov::CompiledModel compiled;
            try {
                auto model = core.read_model(xml.string());
                compiled = core.compile_model(model, device, cfg);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  ERROR: failed to compile model on %s: %s\n", device.c_str(), e.what());
                continue;
            }
            double compile_time_s = ms_since(t0, Clock::now()) / 1000.0;
            MemorySummary compile_mem = mem_mon.summary();
            std::printf("  Compiled in %.2fs\n", compile_time_s);

            std::set<std::string> input_names;
            for (const auto& in : compiled.inputs()) {
                input_names.insert(in.get_any_name());
            }
            const bool has_tti = input_names.count("token_type_ids") > 0;
            const std::string precision = detect_precision(md);
            const std::string run_started = now_iso8601();

            if (args.profile) {
                print_device_properties(compiled, device);
            }
            std::vector<ProfRow> prof_acc;

            for (int input_len : args.input_lengths) {
                int clamped = std::min(input_len, 510);
                std::string text = build_input_text(tokenizer, clamped);
                ov::genai::TokenizedInputs enc_full =
                    tokenizer.encode(text, ov::AnyMap{{"add_special_tokens", true}});
                int actual_tokens = static_cast<int>(enc_full.input_ids.get_shape().back());
                std::printf("\n  Input length target: %d tokens (actual: %d)\n", input_len, actual_tokens);

                for (double comp_rate : args.compression_rates) {
                    std::printf("    Compression rate: %.0f%% keep ... ", comp_rate * 100.0);
                    std::fflush(stdout);
                    monitors.reset();

                    // Pre-tokenize once (outside timing window) — match Python which
                    // re-tokenizes per call but keeps tokenization outside timing.
                    auto make_infer = [&](bool collect_prof) -> std::vector<BertIterationMetrics>* {
                        return nullptr;  // placeholder (unused)
                    };
                    (void)make_infer;

                    std::vector<BertIterationMetrics> iters;
                    std::vector<IterMemory> iter_mems;
                    bool failed = false;

                    auto run_single = [&](bool measure, int iter_idx) -> bool {
                        // Tokenize (truncate to 512) outside timing.
                        ov::genai::TokenizedInputs enc = tokenizer.encode(
                            text, ov::AnyMap{{"add_special_tokens", true}, {"max_length", int64_t(512)}});
                        ov::Tensor input_ids = enc.input_ids;
                        ov::Tensor attn = enc.attention_mask;
                        const int input_tokens = static_cast<int>(input_ids.get_shape().back());

                        ov::InferRequest req = compiled.create_infer_request();
                        try {
                            req.set_tensor("input_ids", input_ids);
                            req.set_tensor("attention_mask", attn);
                            if (has_tti) {
                                ov::Tensor tti;
                                if (enc.token_type_ids && enc.token_type_ids->get_size() ==
                                                              input_ids.get_size()) {
                                    tti = *enc.token_type_ids;
                                } else {
                                    tti = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
                                    std::memset(tti.data(), 0, tti.get_byte_size());
                                }
                                req.set_tensor("token_type_ids", tti);
                            }
                        } catch (const std::exception& e) {
                            std::printf("ERROR: %s\n", e.what());
                            return false;
                        }

                        auto ts = Clock::now();
                        req.infer();
                        auto te = Clock::now();
                        double latency_ms = ms_since(ts, te);

                        if (measure && args.profile) {
                            collect_profiling(req, prof_acc);
                        }
                        if (!measure) return true;  // warmup

                        // Post-process (host, untimed).
                        ov::Tensor logits_t = req.get_output_tensor(0);
                        const auto lshape = logits_t.get_shape();  // [1, seq, C]
                        const size_t seq = lshape.size() >= 2 ? lshape[lshape.size() - 2] : 0;
                        const size_t C = lshape.back();
                        const float* logits = logits_t.data<const float>();
                        std::vector<float> keep;
                        softmax_keepcol(logits, seq, C, keep);
                        // exclude CLS + SEP
                        std::vector<float> content;
                        if (keep.size() > 2) content.assign(keep.begin() + 1, keep.end() - 1);
                        int compressed = 2;
                        if (!content.empty()) {
                            int target_keep = std::max(1, static_cast<int>(content.size() * comp_rate));
                            std::vector<float> sorted_desc = content;
                            std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<float>());
                            int idx = std::min(target_keep - 1, static_cast<int>(content.size()) - 1);
                            float threshold = sorted_desc[idx];
                            int cnt = 0;
                            for (float v : content)
                                if (v >= threshold) ++cnt;
                            compressed = cnt + 2;
                        }

                        BertIterationMetrics m;
                        m.iteration = iter_idx;
                        m.input_tokens = input_tokens;
                        m.latency_ms = latency_ms;
                        m.throughput_samples_per_s = latency_ms > 0 ? 1000.0 / latency_ms : 0.0;
                        m.throughput_tokens_per_s =
                            latency_ms > 0 ? input_tokens * 1000.0 / latency_ms : 0.0;
                        m.compressed_tokens = compressed;
                        m.compression_ratio = input_tokens ? static_cast<double>(compressed) / input_tokens : 1.0;
                        MemorySummary itmem = mem_mon.summary();
                        attach_memory(m.mem, itmem);
                        iters.push_back(m);
                        iter_mems.push_back(m.mem);
                        return true;
                    };

                    for (int w = 0; w < args.warmup; ++w) {
                        if (!run_single(false, 0)) {
                            failed = true;
                            break;
                        }
                    }
                    if (!failed) {
                        for (int i = 0; i < args.lingua_iters; ++i) {
                            mem_mon.reset();
                            if (!run_single(true, i + 1)) {
                                failed = true;
                                break;
                            }
                        }
                    }
                    if (failed || iters.empty()) {
                        std::printf("FAILED\n");
                        continue;
                    }

                    std::vector<double> v_lat, v_ts, v_tt;
                    double sum_in = 0, sum_comp = 0, sum_ratio = 0;
                    for (const auto& m : iters) {
                        v_lat.push_back(m.latency_ms);
                        v_ts.push_back(m.throughput_samples_per_s);
                        v_tt.push_back(m.throughput_tokens_per_s);
                        sum_in += m.input_tokens;
                        sum_comp += m.compressed_tokens;
                        sum_ratio += m.compression_ratio;
                    }

                    BertBenchmarkResult r;
                    r.model_dir = md;
                    r.model_name = path_name(md);
                    r.device = device;
                    r.precision = precision;
                    r.input_length = actual_tokens;
                    r.compression_rate = comp_rate;
                    r.warmup_iters = args.warmup;
                    r.measure_iters = static_cast<int>(iters.size());
                    r.compile_time_s = compile_time_s;
                    r.iterations = iters;
                    r.latency = compute_stats(v_lat);
                    r.thr_samples = compute_stats(v_ts);
                    r.thr_tokens = compute_stats(v_tt);
                    r.avg_input_tokens = sum_in / iters.size();
                    r.avg_compressed_tokens = sum_comp / iters.size();
                    r.avg_compression_ratio = sum_ratio / iters.size();
                    r.cpu_util = monitors.cpu_summary();
                    r.gpu_util = monitors.gpu_summary();
                    r.npu_util = monitors.npu_summary();
                    fill_memory_stats(r.mem, initial_mem, compile_mem, iter_mems);
                    r.openvino_version = ov_ver;
                    r.run_started_at = run_started;

                    std::printf("Lat=%.2fms  Thr=%.0ftok/s  Compressed=%.0f/%.0f (%.0f%%)\n",
                                r.latency.mean, r.thr_tokens.mean, r.avg_compressed_tokens,
                                r.avg_input_tokens, r.avg_compression_ratio * 100.0);
                    all_results.push_back(std::move(r));
                }
            }

            for (const auto& r : all_results) {
                if (r.model_dir == md && r.device == device) print_bert_summary(r);
            }

            if (args.profile && !prof_acc.empty()) {
                long long total_calls = 0;
                // prof rows accumulate per executed iteration across all configs:
                // use the per-row count of the most frequent node as iteration proxy.
                for (const auto& row : prof_acc) total_calls = std::max(total_calls, row.count);
                print_profiling_report(prof_acc, "lingua infer (" + path_name(md) + " / " + device + ")",
                                       args.profile_topk, total_calls);
                if (args.profile_csv) {
                    write_profiling_csv(*args.profile_csv, prof_acc, path_name(md), device, precision,
                                        "lingua_infer", total_calls);
                }
            }
        }
    }

    monitors.stop();
    mem_mon.stop();

    if (all_results.empty()) {
        std::printf("\nNo results collected.\n");
        return 0;
    }
    if (!args.csv_out.empty()) write_bert_csv(args.csv_out, all_results);
    if (!args.json_out.empty()) write_bert_json(args.json_out, all_results);
    std::printf("\n  Total runs: %zu\n  Done.\n", all_results.size());
    return 0;
}

}  // namespace mb
