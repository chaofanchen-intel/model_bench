// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/reports.hpp"

#include <algorithm>
#include <cstdio>
#include <set>

#include "common/common.hpp"
#include "common/format.hpp"
#include "common/json.hpp"

namespace mb {

namespace {

const char* kStatOrder[7] = {"mean", "std", "p50", "p90", "p95", "min", "max"};

double stat_at(const Stats& s, int i) {
    switch (i) {
        case 0: return s.mean;
        case 1: return s.std;
        case 2: return s.p50;
        case 3: return s.p90;
        case 4: return s.p95;
        case 5: return s.min;
        default: return s.max;
    }
}

const Stats kEmptyStats{};

const Stats& agg_get(const std::map<std::string, Stats>& agg, const std::string& key) {
    auto it = agg.find(key);
    return it == agg.end() ? kEmptyStats : it->second;
}

// 7 CSV cells for a stat group, formatted to `prec` decimals.
void push_stat_cells(std::vector<std::string>& row, const Stats& s, int prec) {
    for (int i = 0; i < 7; ++i) row.push_back(fmt_fixed(stat_at(s, i), prec));
}

// 7 JSON fields: <prefix>_<stat><suffix>.
void emit_stat_fields(json::ValuePtr obj, const std::string& prefix,
                      const std::string& suffix, const Stats& s) {
    for (int i = 0; i < 7; ++i) {
        obj->set(prefix + "_" + kStatOrder[i] + suffix, json::Value::number(stat_at(s, i)));
    }
}

// Union of per-core names across results, sorted by int after "cpu".
template <typename ResultT>
std::vector<std::string> all_core_names(const std::vector<ResultT>& results) {
    std::set<std::string> cores;
    for (const auto& r : results) {
        for (auto& kv : r.cpu_util.per_core) cores.insert(kv.first);
    }
    std::vector<std::string> out(cores.begin(), cores.end());
    std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
        auto num = [](const std::string& s) -> long {
            try {
                return std::stol(s.substr(3));
            } catch (...) {
                return 0;
            }
        };
        return num(a) < num(b);
    });
    return out;
}

std::vector<std::string> mem_header() {
    return {
        "initial_rss_mem_mib", "initial_sys_mem_mib", "initial_rss_mem_share", "initial_sys_mem_share",
        "compile_max_rss_mem_mib", "compile_max_sys_mem_mib",
        "compile_max_rss_mem_increase_mib", "compile_max_sys_mem_increase_mib",
        "compile_max_rss_mem_share", "compile_max_sys_mem_share",
        "runtime_mean_rss_mem_mib", "runtime_max_rss_mem_mib", "runtime_mean_sys_mem_mib", "runtime_max_sys_mem_mib",
        "runtime_mean_rss_mem_increase_mib", "runtime_max_rss_mem_increase_mib",
        "runtime_mean_sys_mem_increase_mib", "runtime_max_sys_mem_increase_mib",
        "runtime_max_rss_mem_share", "runtime_max_sys_mem_share", "memory_samples",
    };
}

void push_mem_cells(std::vector<std::string>& row, const ResultMemory& m) {
    row.push_back(f3(m.initial_rss_mem_mib));
    row.push_back(f3(m.initial_sys_mem_mib));
    row.push_back(f3(m.initial_rss_mem_share));
    row.push_back(f3(m.initial_sys_mem_share));
    row.push_back(f3(m.compile_max_rss_mem_mib));
    row.push_back(f3(m.compile_max_sys_mem_mib));
    row.push_back(f3(m.compile_max_rss_mem_increase_mib));
    row.push_back(f3(m.compile_max_sys_mem_increase_mib));
    row.push_back(f3(m.compile_max_rss_mem_share));
    row.push_back(f3(m.compile_max_sys_mem_share));
    row.push_back(f3(m.runtime_mean_rss_mem_mib));
    row.push_back(f3(m.runtime_max_rss_mem_mib));
    row.push_back(f3(m.runtime_mean_sys_mem_mib));
    row.push_back(f3(m.runtime_max_sys_mem_mib));
    row.push_back(f3(m.runtime_mean_rss_mem_increase_mib));
    row.push_back(f3(m.runtime_max_rss_mem_increase_mib));
    row.push_back(f3(m.runtime_mean_sys_mem_increase_mib));
    row.push_back(f3(m.runtime_max_sys_mem_increase_mib));
    row.push_back(f3(m.runtime_max_rss_mem_share));
    row.push_back(f3(m.runtime_max_sys_mem_share));
    row.push_back(std::to_string(m.memory_samples));
}

void emit_mem_fields(json::ValuePtr obj, const ResultMemory& m) {
    obj->set("memory_available", json::Value::boolean(m.memory_available));
    obj->set("memory_samples", json::Value::integer(m.memory_samples));
    obj->set("initial_rss_mem_mib", json::Value::number(m.initial_rss_mem_mib));
    obj->set("initial_sys_mem_mib", json::Value::number(m.initial_sys_mem_mib));
    obj->set("initial_rss_mem_share", json::Value::number(m.initial_rss_mem_share));
    obj->set("initial_sys_mem_share", json::Value::number(m.initial_sys_mem_share));
    obj->set("compile_max_rss_mem_mib", json::Value::number(m.compile_max_rss_mem_mib));
    obj->set("compile_max_sys_mem_mib", json::Value::number(m.compile_max_sys_mem_mib));
    obj->set("compile_max_rss_mem_increase_mib", json::Value::number(m.compile_max_rss_mem_increase_mib));
    obj->set("compile_max_sys_mem_increase_mib", json::Value::number(m.compile_max_sys_mem_increase_mib));
    obj->set("compile_max_rss_mem_share", json::Value::number(m.compile_max_rss_mem_share));
    obj->set("compile_max_sys_mem_share", json::Value::number(m.compile_max_sys_mem_share));
    obj->set("runtime_mean_rss_mem_mib", json::Value::number(m.runtime_mean_rss_mem_mib));
    obj->set("runtime_max_rss_mem_mib", json::Value::number(m.runtime_max_rss_mem_mib));
    obj->set("runtime_mean_sys_mem_mib", json::Value::number(m.runtime_mean_sys_mem_mib));
    obj->set("runtime_max_sys_mem_mib", json::Value::number(m.runtime_max_sys_mem_mib));
    obj->set("runtime_mean_rss_mem_increase_mib", json::Value::number(m.runtime_mean_rss_mem_increase_mib));
    obj->set("runtime_max_rss_mem_increase_mib", json::Value::number(m.runtime_max_rss_mem_increase_mib));
    obj->set("runtime_mean_sys_mem_increase_mib", json::Value::number(m.runtime_mean_sys_mem_increase_mib));
    obj->set("runtime_max_sys_mem_increase_mib", json::Value::number(m.runtime_max_sys_mem_increase_mib));
    obj->set("runtime_max_rss_mem_share", json::Value::number(m.runtime_max_rss_mem_share));
    obj->set("runtime_max_sys_mem_share", json::Value::number(m.runtime_max_sys_mem_share));
}

std::string gpu_cell(const GpuSummary& g) {
    return g.available ? f1(g.avg_util) : std::string("N/A");
}
std::string npu_cell(const NpuSummary& n) {
    return n.available ? f1(n.avg_util) : std::string("N/A");
}

// util dict JSON (mirrors the monitor summary dicts).
json::ValuePtr cpu_util_json(const CpuSummary& c) {
    auto o = json::Value::object();
    o->set("avg_total", json::Value::number(c.avg_total));
    auto pc = json::Value::object();
    for (auto& kv : c.per_core) pc->set(kv.first, json::Value::number(kv.second));
    o->set("per_core", pc);
    return o;
}
json::ValuePtr gpu_util_json(const GpuSummary& g) {
    auto o = json::Value::object();
    o->set("available", json::Value::boolean(g.available));
    o->set("avg_util", json::Value::number(g.avg_util));
    o->set("samples", json::Value::integer(g.samples));
    return o;
}
json::ValuePtr npu_util_json(const NpuSummary& n) {
    auto o = json::Value::object();
    o->set("available", json::Value::boolean(n.available));
    if (!n.cpu_gen.empty()) o->set("cpu_gen", json::Value::str(n.cpu_gen));
    o->set("avg_util", json::Value::number(n.avg_util));
    o->set("samples", json::Value::integer(n.samples));
    return o;
}

void emit_iter_mem(json::ValuePtr obj, const IterMemory& m) {
    obj->set("memory_available", json::Value::boolean(m.memory_available));
    obj->set("memory_samples", json::Value::integer(m.memory_samples));
    obj->set("max_rss_mem_mib", json::Value::number(m.max_rss_mem_mib));
    obj->set("max_sys_mem_mib", json::Value::number(m.max_sys_mem_mib));
    obj->set("max_rss_mem_increase_mib", json::Value::number(m.max_rss_mem_increase_mib));
    obj->set("max_sys_mem_increase_mib", json::Value::number(m.max_sys_mem_increase_mib));
    obj->set("max_rss_mem_share", json::Value::number(m.max_rss_mem_share));
    obj->set("max_sys_mem_share", json::Value::number(m.max_sys_mem_share));
}

void print_table_header() {
    std::printf("  %-18s %10s %8s %10s %10s %10s %8s %8s  Unit\n", "Metric", "Mean", "Std",
                "P50", "P90", "P95", "Min", "Max");
    std::printf("  %-18s %10s %8s %10s %10s %10s %8s %8s  ----\n",
                "------------------", "----------", "--------", "----------", "----------",
                "----------", "--------", "--------");
}
void print_row(const char* name, const Stats& s, const char* unit) {
    std::printf("  %-18s %10.2f %8.2f %10.2f %10.2f %10.2f %8.2f %8.2f  [%s]\n", name, s.mean,
                s.std, s.p50, s.p90, s.p95, s.min, s.max, unit);
}
void print_mem_block(const ResultMemory& m) {
    std::printf("\n  --- Memory ---\n");
    if (!m.memory_available) {
        std::printf("  Memory monitor: unavailable/disabled\n");
        return;
    }
    std::printf("  Initial RSS/System: %.1f / %.1f MiB\n", m.initial_rss_mem_mib, m.initial_sys_mem_mib);
    std::printf("  Compile max RSS/System: %.1f / %.1f MiB\n", m.compile_max_rss_mem_mib, m.compile_max_sys_mem_mib);
    std::printf("  Runtime max RSS/System: %.1f / %.1f MiB\n", m.runtime_max_rss_mem_mib, m.runtime_max_sys_mem_mib);
    std::printf("  Runtime max increase RSS/System: %.1f / %.1f MiB (%d samples)\n",
                m.runtime_max_rss_mem_increase_mib, m.runtime_max_sys_mem_increase_mib, m.memory_samples);
}
void print_util_block(const CpuSummary& cpu, const GpuSummary& gpu, const NpuSummary& npu) {
    std::printf("\n  --- Device Utilization ---\n");
    if (cpu.avg_total > 0) std::printf("  CPU avg: %.1f%%\n", cpu.avg_total);
    if (gpu.available) std::printf("  GPU avg: %.1f%% (%d samples)\n", gpu.avg_util, gpu.samples);
    if (npu.available) {
        std::printf("  NPU avg: %.1f%%%s (%d samples)\n", npu.avg_util,
                    npu.cpu_gen.empty() ? "" : (" [" + npu.cpu_gen + "]").c_str(), npu.samples);
    }
}

}  // namespace

// ===================== Generation (LLM + VLM) ================================

void write_gen_csv(const std::string& path, const std::vector<GenBenchmarkResult>& results) {
    auto cores = all_core_names(results);
    std::vector<std::string> header = {
        "model", "framework", "pipeline", "device", "precision", "mode",
        "num_beams", "batch_size", "apply_chat_template", "disable_prompt_permutation",
        "prompt_idx", "input_tokens", "infer_count_arg", "exact_output_tokens", "num_iters",
        "measured_iters", "compile_time_s",
    };
    auto add7 = [&](const std::string& prefix, const std::string& suffix) {
        for (int i = 0; i < 7; ++i) header.push_back(prefix + "_" + kStatOrder[i] + suffix);
    };
    add7("output_tokens", "");
    add7("generation_time", "_s");
    add7("ttft", "_ms");
    add7("tpot", "_ms");
    add7("first_token_infer", "_ms");
    add7("other_tokens_infer", "_ms");
    add7("tps", "");
    add7("decode_tps", "");
    add7("prefill", "_ms");
    add7("decode", "_ms");
    add7("e2e", "_ms");
    add7("tokenization", "_ms");
    add7("detokenization", "_ms");
    for (auto& h : mem_header()) header.push_back(h);
    header.push_back("gpu_avg_util");
    header.push_back("npu_avg_util");
    header.push_back("cpu_avg_util");
    for (auto& c : cores) header.push_back("cpu_" + c + "_util");
    header.push_back("image_path");
    header.push_back("openvino_version");
    header.push_back("openvino_genai_version");
    header.push_back("run_started_at");

    std::vector<std::vector<std::string>> rows;
    for (const auto& r : results) {
        std::vector<std::string> row = {
            r.model_name, r.framework, r.pipeline, r.device, r.precision, r.mode,
            std::to_string(r.num_beams), std::to_string(r.batch_size), boolstr(r.apply_chat_template),
            boolstr(r.disable_prompt_permutation), std::to_string(r.prompt_idx),
            std::to_string(r.input_tokens), std::to_string(r.infer_count_arg),
            boolstr(r.exact_output_tokens), std::to_string(r.num_iters),
            std::to_string(r.measured_iters), f3(r.compile_time_s),
        };
        push_stat_cells(row, agg_get(r.agg, "output_tokens"), 2);
        push_stat_cells(row, agg_get(r.agg, "generation_time"), 5);
        push_stat_cells(row, agg_get(r.agg, "ttft"), 3);
        push_stat_cells(row, agg_get(r.agg, "tpot"), 3);
        push_stat_cells(row, agg_get(r.agg, "first_token_infer"), 3);
        push_stat_cells(row, agg_get(r.agg, "other_tokens_infer"), 3);
        push_stat_cells(row, agg_get(r.agg, "throughput"), 2);
        push_stat_cells(row, agg_get(r.agg, "decode_throughput"), 2);
        push_stat_cells(row, agg_get(r.agg, "prefill"), 3);
        push_stat_cells(row, agg_get(r.agg, "decode"), 3);
        push_stat_cells(row, agg_get(r.agg, "e2e"), 3);
        push_stat_cells(row, agg_get(r.agg, "tokenization"), 3);
        push_stat_cells(row, agg_get(r.agg, "detokenization"), 3);
        push_mem_cells(row, r.mem);
        row.push_back(gpu_cell(r.gpu_util));
        row.push_back(npu_cell(r.npu_util));
        row.push_back(f1(r.cpu_util.avg_total));
        for (auto& c : cores) {
            auto it = r.cpu_util.per_core.find(c);
            row.push_back(f1(it == r.cpu_util.per_core.end() ? 0.0 : it->second));
        }
        row.push_back(r.image_path);
        row.push_back(r.openvino_version);
        row.push_back(r.openvino_genai_version);
        row.push_back(r.run_started_at);
        rows.push_back(std::move(row));
    }
    write_csv_append(path, header, rows);
}

void write_gen_json(const std::string& path, const std::vector<GenBenchmarkResult>& results) {
    std::vector<json::ValuePtr> objs;
    for (const auto& r : results) {
        auto o = json::Value::object();
        o->set("model_dir", json::Value::str(r.model_dir));
        o->set("model_name", json::Value::str(r.model_name));
        o->set("framework", json::Value::str(r.framework));
        o->set("pipeline", json::Value::str(r.pipeline));
        o->set("device", json::Value::str(r.device));
        o->set("precision", json::Value::str(r.precision));
        o->set("mode", json::Value::str(r.mode));
        o->set("num_beams", json::Value::integer(r.num_beams));
        o->set("batch_size", json::Value::integer(r.batch_size));
        o->set("apply_chat_template", json::Value::boolean(r.apply_chat_template));
        o->set("disable_prompt_permutation", json::Value::boolean(r.disable_prompt_permutation));
        o->set("prompt_idx", json::Value::integer(r.prompt_idx));
        o->set("input_tokens", json::Value::integer(r.input_tokens));
        o->set("infer_count_arg", json::Value::integer(r.infer_count_arg));
        o->set("exact_output_tokens", json::Value::boolean(r.exact_output_tokens));
        o->set("num_iters", json::Value::integer(r.num_iters));
        o->set("measured_iters", json::Value::integer(r.measured_iters));
        o->set("compile_time_s", json::Value::number(r.compile_time_s));
        emit_stat_fields(o, "output_tokens", "", agg_get(r.agg, "output_tokens"));
        emit_stat_fields(o, "generation_time", "_s", agg_get(r.agg, "generation_time"));
        emit_stat_fields(o, "tokenization", "_ms", agg_get(r.agg, "tokenization"));
        emit_stat_fields(o, "detokenization", "_ms", agg_get(r.agg, "detokenization"));
        emit_mem_fields(o, r.mem);
        emit_stat_fields(o, "ttft", "_ms", agg_get(r.agg, "ttft"));
        emit_stat_fields(o, "tpot", "_ms", agg_get(r.agg, "tpot"));
        emit_stat_fields(o, "first_token_infer", "_ms", agg_get(r.agg, "first_token_infer"));
        emit_stat_fields(o, "other_tokens_infer", "_ms", agg_get(r.agg, "other_tokens_infer"));
        emit_stat_fields(o, "throughput", "_tps", agg_get(r.agg, "throughput"));
        emit_stat_fields(o, "decode_throughput", "_tps", agg_get(r.agg, "decode_throughput"));
        emit_stat_fields(o, "prefill", "_ms", agg_get(r.agg, "prefill"));
        emit_stat_fields(o, "decode", "_ms", agg_get(r.agg, "decode"));
        emit_stat_fields(o, "e2e", "_ms", agg_get(r.agg, "e2e"));
        o->set("cpu_util", cpu_util_json(r.cpu_util));
        o->set("gpu_util", gpu_util_json(r.gpu_util));
        o->set("npu_util", npu_util_json(r.npu_util));
        auto iters = json::Value::array();
        for (const auto& it : r.iterations) {
            auto io = json::Value::object();
            io->set("iteration", json::Value::integer(it.iteration));
            io->set("is_warmup", json::Value::boolean(it.is_warmup));
            io->set("prompt_idx", json::Value::integer(it.prompt_idx));
            io->set("input_tokens", json::Value::integer(it.input_tokens));
            io->set("output_tokens", json::Value::integer(it.output_tokens));
            io->set("infer_count", json::Value::integer(it.infer_count));
            io->set("generate_time_s", json::Value::number(it.generate_time_s));
            io->set("ttft_ms", json::Value::number(it.ttft_ms));
            io->set("tpot_ms", json::Value::number(it.tpot_ms));
            io->set("first_token_infer_latency_ms", json::Value::number(it.first_token_infer_latency_ms));
            io->set("other_tokens_infer_avg_latency_ms", json::Value::number(it.other_tokens_infer_avg_latency_ms));
            io->set("prefill_latency_ms", json::Value::number(it.prefill_latency_ms));
            io->set("decode_latency_ms", json::Value::number(it.decode_latency_ms));
            io->set("e2e_latency_ms", json::Value::number(it.e2e_latency_ms));
            io->set("throughput_tps", json::Value::number(it.throughput_tps));
            io->set("decode_throughput_tps", json::Value::number(it.decode_throughput_tps));
            io->set("tokenization_time_ms", json::Value::number(it.tokenization_time_ms));
            io->set("detokenization_time_ms", json::Value::number(it.detokenization_time_ms));
            io->set("result_md5", json::Value::str(it.result_md5));
            io->set("start", json::Value::str(it.start));
            io->set("end", json::Value::str(it.end));
            emit_iter_mem(io, it.mem);
            iters->push(io);
        }
        o->set("iterations", iters);
        o->set("image_path", json::Value::str(r.image_path));
        o->set("openvino_version", json::Value::str(r.openvino_version));
        o->set("openvino_genai_version", json::Value::str(r.openvino_genai_version));
        o->set("run_started_at", json::Value::str(r.run_started_at));
        objs.push_back(o);
    }
    json::append_array(path, objs);
}

void print_gen_summary(const GenBenchmarkResult& r) {
    std::string sep(80, '=');
    std::printf("\n%s\n", sep.c_str());
    std::printf("  Result: %s | %s | %s | prompt[%d] in=%d out_max=%d\n", r.model_name.c_str(),
                r.device.c_str(), r.mode.c_str(), r.prompt_idx, r.input_tokens, r.infer_count_arg);
    std::printf("  Pipeline: %s  measured_iters=%d  compile=%.2fs\n", r.pipeline.c_str(),
                r.measured_iters, r.compile_time_s);
    std::printf("%s\n", sep.c_str());
    print_table_header();
    print_row("TTFT", agg_get(r.agg, "ttft"), "ms");
    print_row("TPOT", agg_get(r.agg, "tpot"), "ms");
    print_row("1st infer", agg_get(r.agg, "first_token_infer"), "ms");
    print_row("2nd infer", agg_get(r.agg, "other_tokens_infer"), "ms");
    print_row("Throughput", agg_get(r.agg, "throughput"), "tok/s");
    print_row("Decode TPS", agg_get(r.agg, "decode_throughput"), "tok/s");
    print_row("Output tokens", agg_get(r.agg, "output_tokens"), "tok");
    print_row("Gen time", agg_get(r.agg, "generation_time"), "s");
    print_row("Prefill", agg_get(r.agg, "prefill"), "ms");
    print_row("Decode", agg_get(r.agg, "decode"), "ms");
    print_row("E2E", agg_get(r.agg, "e2e"), "ms");
    print_row("Tokenization", agg_get(r.agg, "tokenization"), "ms");
    print_row("Detokenization", agg_get(r.agg, "detokenization"), "ms");
    print_mem_block(r.mem);
    print_util_block(r.cpu_util, r.gpu_util, r.npu_util);
    std::printf("%s\n", sep.c_str());
}

// ===================== BERT (lingua) ========================================

void write_bert_csv(const std::string& path, const std::vector<BertBenchmarkResult>& results) {
    auto cores = all_core_names(results);
    std::vector<std::string> header = {
        "model", "framework", "device", "precision", "input_tokens", "compression_rate", "iters",
        "compile_time_s",
    };
    for (auto& h : mem_header()) header.push_back(h);
    auto add7 = [&](const std::string& prefix) {
        for (int i = 0; i < 7; ++i) header.push_back(prefix + "_" + kStatOrder[i]);
    };
    // latency uses _ms suffix.
    for (int i = 0; i < 7; ++i) header.push_back(std::string("latency_") + kStatOrder[i] + "_ms");
    add7("throughput_samples");
    add7("throughput_tokens");
    header.push_back("avg_compressed_tokens");
    header.push_back("avg_compression_ratio");
    header.push_back("cpu_avg_util");
    header.push_back("gpu_avg_util");
    header.push_back("npu_avg_util");
    header.push_back("openvino_version");
    header.push_back("run_started_at");
    for (auto& c : cores) header.push_back("cpu_" + c + "_util");

    std::vector<std::vector<std::string>> rows;
    for (const auto& r : results) {
        std::vector<std::string> row = {
            r.model_name, r.framework, r.device, r.precision, std::to_string(r.input_length),
            f2(r.compression_rate), std::to_string(r.measure_iters), f3(r.compile_time_s),
        };
        push_mem_cells(row, r.mem);
        push_stat_cells(row, r.latency, 3);
        push_stat_cells(row, r.thr_samples, 2);
        push_stat_cells(row, r.thr_tokens, 2);
        row.push_back(f0(r.avg_compressed_tokens));
        row.push_back(f4(r.avg_compression_ratio));
        row.push_back(f1(r.cpu_util.avg_total));
        row.push_back(gpu_cell(r.gpu_util));
        row.push_back(npu_cell(r.npu_util));
        row.push_back(r.openvino_version);
        row.push_back(r.run_started_at);
        for (auto& c : cores) {
            auto it = r.cpu_util.per_core.find(c);
            row.push_back(f1(it == r.cpu_util.per_core.end() ? 0.0 : it->second));
        }
        rows.push_back(std::move(row));
    }
    write_csv_append(path, header, rows);
}

void write_bert_json(const std::string& path, const std::vector<BertBenchmarkResult>& results) {
    std::vector<json::ValuePtr> objs;
    for (const auto& r : results) {
        auto o = json::Value::object();
        o->set("model_dir", json::Value::str(r.model_dir));
        o->set("model_name", json::Value::str(r.model_name));
        o->set("framework", json::Value::str(r.framework));
        o->set("device", json::Value::str(r.device));
        o->set("precision", json::Value::str(r.precision));
        o->set("input_length", json::Value::integer(r.input_length));
        o->set("compression_rate", json::Value::number(r.compression_rate));
        o->set("warmup_iters", json::Value::integer(r.warmup_iters));
        o->set("measure_iters", json::Value::integer(r.measure_iters));
        o->set("compile_time_s", json::Value::number(r.compile_time_s));
        auto iters = json::Value::array();
        for (const auto& it : r.iterations) {
            auto io = json::Value::object();
            io->set("iteration", json::Value::integer(it.iteration));
            io->set("input_tokens", json::Value::integer(it.input_tokens));
            io->set("latency_ms", json::Value::number(it.latency_ms));
            io->set("throughput_samples_per_s", json::Value::number(it.throughput_samples_per_s));
            io->set("throughput_tokens_per_s", json::Value::number(it.throughput_tokens_per_s));
            io->set("compressed_tokens", json::Value::integer(it.compressed_tokens));
            io->set("compression_ratio", json::Value::number(it.compression_ratio));
            emit_iter_mem(io, it.mem);
            iters->push(io);
        }
        o->set("iterations", iters);
        emit_stat_fields(o, "latency", "_ms", r.latency);
        emit_stat_fields(o, "throughput_samples", "", r.thr_samples);
        emit_stat_fields(o, "throughput_tokens", "", r.thr_tokens);
        o->set("avg_input_tokens", json::Value::number(r.avg_input_tokens));
        o->set("avg_compressed_tokens", json::Value::number(r.avg_compressed_tokens));
        o->set("avg_compression_ratio", json::Value::number(r.avg_compression_ratio));
        o->set("cpu_util", cpu_util_json(r.cpu_util));
        o->set("gpu_util", gpu_util_json(r.gpu_util));
        o->set("npu_util", npu_util_json(r.npu_util));
        emit_mem_fields(o, r.mem);
        o->set("openvino_version", json::Value::str(r.openvino_version));
        o->set("run_started_at", json::Value::str(r.run_started_at));
        objs.push_back(o);
    }
    json::append_array(path, objs);
}

void print_bert_summary(const BertBenchmarkResult& r) {
    std::string sep(80, '=');
    std::printf("\n%s\n", sep.c_str());
    std::printf("  Result: %s | %s | input=%d | compression=%.0f%% keep\n", r.model_name.c_str(),
                r.device.c_str(), r.input_length, r.compression_rate * 100.0);
    std::printf("  measured_iters=%d  compile=%.2fs\n", r.measure_iters, r.compile_time_s);
    std::printf("%s\n", sep.c_str());
    print_table_header();
    print_row("Latency", r.latency, "ms");
    print_row("Thrpt(samples)", r.thr_samples, "samp/s");
    print_row("Thrpt(tokens)", r.thr_tokens, "tok/s");
    std::printf("\n  --- Compression ---\n");
    std::printf("  Avg input tokens:      %.0f\n", r.avg_input_tokens);
    std::printf("  Avg compressed tokens: %.0f\n", r.avg_compressed_tokens);
    std::printf("  Avg keep ratio:        %.2f%% (lower = more compressed)\n",
                r.avg_compression_ratio * 100.0);
    print_mem_block(r.mem);
    print_util_block(r.cpu_util, r.gpu_util, r.npu_util);
    std::printf("%s\n", sep.c_str());
}

// ===================== Encoder shared (embed/rerank) ========================

namespace {

template <typename ResultT>
void emit_encoder_iter_common(json::ValuePtr io, const ResultT& it);

// Embedding-specific common header (everything up to compile_time_s).
std::vector<std::string> embed_base_header() {
    return {
        "model", "framework", "pipeline", "device", "precision", "mode", "batch_size",
        "pooling_type", "normalize", "max_length", "padding_side", "pad_to_max_length",
        "query_instruction", "embed_instruction", "prompt_idx", "input_tokens",
        "total_input_tokens", "embedding_dim", "num_iters", "measured_iters", "compile_time_s",
    };
}
std::vector<std::string> rerank_base_header() {
    return {
        "model", "framework", "pipeline", "device", "precision", "num_documents", "top_n",
        "max_length", "padding_side", "pad_to_max_length", "prompt_idx", "input_tokens",
        "total_input_tokens", "num_iters", "measured_iters", "compile_time_s",
    };
}

void add_encoder_stat_headers(std::vector<std::string>& header) {
    auto add7 = [&](const std::string& prefix, const std::string& suffix) {
        for (int i = 0; i < 7; ++i) header.push_back(prefix + "_" + kStatOrder[i] + suffix);
    };
    add7("latency", "_ms");
    add7("e2e", "_ms");
    add7("tokenization", "_ms");
    add7("throughput_docs", "");
    add7("throughput_tokens", "");
}

}  // namespace

void write_embed_csv(const std::string& path, const std::vector<EmbedBenchmarkResult>& results) {
    auto cores = all_core_names(results);
    std::vector<std::string> header = embed_base_header();
    add_encoder_stat_headers(header);
    for (auto& h : mem_header()) header.push_back(h);
    header.push_back("cpu_avg_util");
    header.push_back("gpu_avg_util");
    header.push_back("npu_avg_util");
    for (auto& c : cores) header.push_back("cpu_" + c + "_util");
    header.push_back("openvino_version");
    header.push_back("openvino_genai_version");
    header.push_back("run_started_at");

    std::vector<std::vector<std::string>> rows;
    for (const auto& r : results) {
        std::vector<std::string> row = {
            r.model_name, r.framework, r.pipeline, r.device, r.precision, r.mode,
            std::to_string(r.batch_size), r.pooling_type, boolstr(r.normalize),
            std::to_string(r.max_length), r.padding_side, boolstr(r.pad_to_max_length),
            r.query_instruction, r.embed_instruction, std::to_string(r.prompt_idx),
            std::to_string(r.input_tokens), std::to_string(r.total_input_tokens),
            std::to_string(r.embedding_dim), std::to_string(r.num_iters),
            std::to_string(r.measured_iters), f3(r.compile_time_s),
        };
        push_stat_cells(row, r.latency, 3);
        push_stat_cells(row, r.e2e, 3);
        push_stat_cells(row, r.tokenization, 3);
        push_stat_cells(row, r.thr_docs, 2);
        push_stat_cells(row, r.thr_tokens, 2);
        push_mem_cells(row, r.mem);
        row.push_back(f1(r.cpu_util.avg_total));
        row.push_back(gpu_cell(r.gpu_util));
        row.push_back(npu_cell(r.npu_util));
        for (auto& c : cores) {
            auto it = r.cpu_util.per_core.find(c);
            row.push_back(f1(it == r.cpu_util.per_core.end() ? 0.0 : it->second));
        }
        row.push_back(r.openvino_version);
        row.push_back(r.openvino_genai_version);
        row.push_back(r.run_started_at);
        rows.push_back(std::move(row));
    }
    write_csv_append(path, header, rows);
}

void write_embed_json(const std::string& path, const std::vector<EmbedBenchmarkResult>& results) {
    std::vector<json::ValuePtr> objs;
    for (const auto& r : results) {
        auto o = json::Value::object();
        o->set("model_dir", json::Value::str(r.model_dir));
        o->set("model_name", json::Value::str(r.model_name));
        o->set("framework", json::Value::str(r.framework));
        o->set("pipeline", json::Value::str(r.pipeline));
        o->set("device", json::Value::str(r.device));
        o->set("precision", json::Value::str(r.precision));
        o->set("mode", json::Value::str(r.mode));
        o->set("batch_size", json::Value::integer(r.batch_size));
        o->set("pooling_type", json::Value::str(r.pooling_type));
        o->set("normalize", json::Value::boolean(r.normalize));
        o->set("max_length", json::Value::integer(r.max_length));
        o->set("padding_side", json::Value::str(r.padding_side));
        o->set("pad_to_max_length", json::Value::boolean(r.pad_to_max_length));
        o->set("query_instruction", json::Value::str(r.query_instruction));
        o->set("embed_instruction", json::Value::str(r.embed_instruction));
        o->set("prompt_idx", json::Value::integer(r.prompt_idx));
        o->set("input_tokens", json::Value::integer(r.input_tokens));
        o->set("total_input_tokens", json::Value::integer(r.total_input_tokens));
        o->set("embedding_dim", json::Value::integer(r.embedding_dim));
        o->set("num_iters", json::Value::integer(r.num_iters));
        o->set("measured_iters", json::Value::integer(r.measured_iters));
        o->set("compile_time_s", json::Value::number(r.compile_time_s));
        emit_stat_fields(o, "latency", "_ms", r.latency);
        emit_stat_fields(o, "e2e", "_ms", r.e2e);
        emit_stat_fields(o, "tokenization", "_ms", r.tokenization);
        emit_stat_fields(o, "throughput_docs", "", r.thr_docs);
        emit_stat_fields(o, "throughput_tokens", "", r.thr_tokens);
        o->set("cpu_util", cpu_util_json(r.cpu_util));
        o->set("gpu_util", gpu_util_json(r.gpu_util));
        o->set("npu_util", npu_util_json(r.npu_util));
        emit_mem_fields(o, r.mem);
        auto iters = json::Value::array();
        for (const auto& it : r.iterations) {
            auto io = json::Value::object();
            io->set("iteration", json::Value::integer(it.iteration));
            io->set("is_warmup", json::Value::boolean(it.is_warmup));
            io->set("prompt_idx", json::Value::integer(it.prompt_idx));
            io->set("mode", json::Value::str(it.mode));
            io->set("batch_size", json::Value::integer(it.batch_size));
            io->set("input_tokens", json::Value::integer(it.input_tokens));
            io->set("total_input_tokens", json::Value::integer(it.total_input_tokens));
            io->set("embedding_dim", json::Value::integer(it.embedding_dim));
            io->set("tokenization_time_ms", json::Value::number(it.tokenization_time_ms));
            io->set("latency_ms", json::Value::number(it.latency_ms));
            io->set("e2e_latency_ms", json::Value::number(it.e2e_latency_ms));
            io->set("throughput_docs_per_s", json::Value::number(it.throughput_docs_per_s));
            io->set("throughput_tokens_per_s", json::Value::number(it.throughput_tokens_per_s));
            io->set("result_md5", json::Value::str(it.result_md5));
            io->set("start", json::Value::str(it.start));
            io->set("end", json::Value::str(it.end));
            emit_iter_mem(io, it.mem);
            iters->push(io);
        }
        o->set("iterations", iters);
        o->set("openvino_version", json::Value::str(r.openvino_version));
        o->set("openvino_genai_version", json::Value::str(r.openvino_genai_version));
        o->set("run_started_at", json::Value::str(r.run_started_at));
        objs.push_back(o);
    }
    json::append_array(path, objs);
}

void print_embed_summary(const EmbedBenchmarkResult& r) {
    std::string sep(80, '=');
    std::printf("\n%s\n", sep.c_str());
    std::printf("  Result: %s | %s | %s | mode=%s\n", r.model_name.c_str(), r.device.c_str(),
                r.precision.c_str(), r.mode.c_str());
    std::printf("  Pipeline: %s  pooling=%s normalize=%s batch=%d dim=%d\n", r.pipeline.c_str(),
                r.pooling_type.empty() ? "(default)" : r.pooling_type.c_str(),
                r.normalize ? "True" : "False", r.batch_size, r.embedding_dim);
    std::printf("  input_tokens(max/total)=%d/%d  measured_iters=%d  compile=%.2fs\n",
                r.input_tokens, r.total_input_tokens, r.measured_iters, r.compile_time_s);
    std::printf("%s\n", sep.c_str());
    print_table_header();
    print_row("Latency", r.latency, "ms");
    print_row("E2E (+token)", r.e2e, "ms");
    print_row("Tokenization", r.tokenization, "ms");
    print_row("Thrpt(docs)", r.thr_docs, "doc/s");
    print_row("Thrpt(tokens)", r.thr_tokens, "tok/s");
    print_mem_block(r.mem);
    print_util_block(r.cpu_util, r.gpu_util, r.npu_util);
    std::printf("%s\n", sep.c_str());
}

// ===================== Reranker =============================================

void write_rerank_csv(const std::string& path, const std::vector<RerankBenchmarkResult>& results) {
    auto cores = all_core_names(results);
    std::vector<std::string> header = rerank_base_header();
    add_encoder_stat_headers(header);
    for (auto& h : mem_header()) header.push_back(h);
    header.push_back("cpu_avg_util");
    header.push_back("gpu_avg_util");
    header.push_back("npu_avg_util");
    for (auto& c : cores) header.push_back("cpu_" + c + "_util");
    header.push_back("openvino_version");
    header.push_back("openvino_genai_version");
    header.push_back("run_started_at");

    std::vector<std::vector<std::string>> rows;
    for (const auto& r : results) {
        std::vector<std::string> row = {
            r.model_name, r.framework, r.pipeline, r.device, r.precision,
            std::to_string(r.num_documents), std::to_string(r.top_n), std::to_string(r.max_length),
            r.padding_side, boolstr(r.pad_to_max_length), std::to_string(r.prompt_idx),
            std::to_string(r.input_tokens), std::to_string(r.total_input_tokens),
            std::to_string(r.num_iters), std::to_string(r.measured_iters), f3(r.compile_time_s),
        };
        push_stat_cells(row, r.latency, 3);
        push_stat_cells(row, r.e2e, 3);
        push_stat_cells(row, r.tokenization, 3);
        push_stat_cells(row, r.thr_docs, 2);
        push_stat_cells(row, r.thr_tokens, 2);
        push_mem_cells(row, r.mem);
        row.push_back(f1(r.cpu_util.avg_total));
        row.push_back(gpu_cell(r.gpu_util));
        row.push_back(npu_cell(r.npu_util));
        for (auto& c : cores) {
            auto it = r.cpu_util.per_core.find(c);
            row.push_back(f1(it == r.cpu_util.per_core.end() ? 0.0 : it->second));
        }
        row.push_back(r.openvino_version);
        row.push_back(r.openvino_genai_version);
        row.push_back(r.run_started_at);
        rows.push_back(std::move(row));
    }
    write_csv_append(path, header, rows);
}

void write_rerank_json(const std::string& path, const std::vector<RerankBenchmarkResult>& results) {
    std::vector<json::ValuePtr> objs;
    for (const auto& r : results) {
        auto o = json::Value::object();
        o->set("model_dir", json::Value::str(r.model_dir));
        o->set("model_name", json::Value::str(r.model_name));
        o->set("framework", json::Value::str(r.framework));
        o->set("pipeline", json::Value::str(r.pipeline));
        o->set("device", json::Value::str(r.device));
        o->set("precision", json::Value::str(r.precision));
        o->set("num_documents", json::Value::integer(r.num_documents));
        o->set("top_n", json::Value::integer(r.top_n));
        o->set("max_length", json::Value::integer(r.max_length));
        o->set("padding_side", json::Value::str(r.padding_side));
        o->set("pad_to_max_length", json::Value::boolean(r.pad_to_max_length));
        o->set("prompt_idx", json::Value::integer(r.prompt_idx));
        o->set("input_tokens", json::Value::integer(r.input_tokens));
        o->set("total_input_tokens", json::Value::integer(r.total_input_tokens));
        o->set("num_iters", json::Value::integer(r.num_iters));
        o->set("measured_iters", json::Value::integer(r.measured_iters));
        o->set("compile_time_s", json::Value::number(r.compile_time_s));
        emit_stat_fields(o, "latency", "_ms", r.latency);
        emit_stat_fields(o, "e2e", "_ms", r.e2e);
        emit_stat_fields(o, "tokenization", "_ms", r.tokenization);
        emit_stat_fields(o, "throughput_docs", "", r.thr_docs);
        emit_stat_fields(o, "throughput_tokens", "", r.thr_tokens);
        o->set("cpu_util", cpu_util_json(r.cpu_util));
        o->set("gpu_util", gpu_util_json(r.gpu_util));
        o->set("npu_util", npu_util_json(r.npu_util));
        emit_mem_fields(o, r.mem);
        auto iters = json::Value::array();
        for (const auto& it : r.iterations) {
            auto io = json::Value::object();
            io->set("iteration", json::Value::integer(it.iteration));
            io->set("is_warmup", json::Value::boolean(it.is_warmup));
            io->set("prompt_idx", json::Value::integer(it.prompt_idx));
            io->set("num_documents", json::Value::integer(it.num_documents));
            io->set("top_n", json::Value::integer(it.top_n));
            io->set("input_tokens", json::Value::integer(it.input_tokens));
            io->set("total_input_tokens", json::Value::integer(it.total_input_tokens));
            io->set("tokenization_time_ms", json::Value::number(it.tokenization_time_ms));
            io->set("latency_ms", json::Value::number(it.latency_ms));
            io->set("e2e_latency_ms", json::Value::number(it.e2e_latency_ms));
            io->set("throughput_docs_per_s", json::Value::number(it.throughput_docs_per_s));
            io->set("throughput_tokens_per_s", json::Value::number(it.throughput_tokens_per_s));
            io->set("top1_index", json::Value::integer(it.top1_index));
            io->set("top1_score", json::Value::number(it.top1_score));
            io->set("result_md5", json::Value::str(it.result_md5));
            io->set("start", json::Value::str(it.start));
            io->set("end", json::Value::str(it.end));
            emit_iter_mem(io, it.mem);
            iters->push(io);
        }
        o->set("iterations", iters);
        o->set("openvino_version", json::Value::str(r.openvino_version));
        o->set("openvino_genai_version", json::Value::str(r.openvino_genai_version));
        o->set("run_started_at", json::Value::str(r.run_started_at));
        objs.push_back(o);
    }
    json::append_array(path, objs);
}

void print_rerank_summary(const RerankBenchmarkResult& r) {
    std::string sep(80, '=');
    std::printf("\n%s\n", sep.c_str());
    std::printf("  Result: %s | %s | %s\n", r.model_name.c_str(), r.device.c_str(),
                r.precision.c_str());
    std::printf("  Pipeline: %s  documents=%d top_n=%d max_length=%s\n", r.pipeline.c_str(),
                r.num_documents, r.top_n, r.max_length ? std::to_string(r.max_length).c_str() : "(default)");
    std::printf("  input_tokens(pair_max/total)=%d/%d  measured_iters=%d  compile=%.2fs\n",
                r.input_tokens, r.total_input_tokens, r.measured_iters, r.compile_time_s);
    std::printf("%s\n", sep.c_str());
    print_table_header();
    print_row("Latency", r.latency, "ms");
    print_row("E2E (+token)", r.e2e, "ms");
    print_row("Tokenization", r.tokenization, "ms");
    print_row("Thrpt(docs)", r.thr_docs, "doc/s");
    print_row("Thrpt(tokens)", r.thr_tokens, "tok/s");
    print_mem_block(r.mem);
    print_util_block(r.cpu_util, r.gpu_util, r.npu_util);
    std::printf("%s\n", sep.c_str());
}

}  // namespace mb
