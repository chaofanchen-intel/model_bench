// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/profiling.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <unordered_map>

#include "common/common.hpp"
#include "common/format.hpp"

namespace mb {

namespace {

std::string status_str(ov::ProfilingInfo::Status s) {
    switch (s) {
        case ov::ProfilingInfo::Status::NOT_RUN: return "NOT_RUN";
        case ov::ProfilingInfo::Status::OPTIMIZED_OUT: return "OPTIMIZED_OUT";
        case ov::ProfilingInfo::Status::EXECUTED: return "EXECUTED";
    }
    return "UNKNOWN";
}

std::string truncate(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    return s.substr(0, n - 1) + "…";
}

}  // namespace

void accumulate_profiling(std::vector<ov::ProfilingInfo>&& pinfo, std::vector<ProfRow>& acc) {
    // Index existing rows by node_name for accumulation.
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < acc.size(); ++i) index[acc[i].node_name] = i;

    for (auto& p : pinfo) {
        const long long rt = static_cast<long long>(p.real_time.count());
        const long long ct = static_cast<long long>(p.cpu_time.count());
        auto it = index.find(p.node_name);
        if (it == index.end()) {
            ProfRow r;
            r.node_name = p.node_name;
            r.node_type = p.node_type;
            r.exec_type = p.exec_type;
            r.status = status_str(p.status);
            r.real_time_us = rt;
            r.cpu_time_us = ct;
            r.count = 1;
            index[r.node_name] = acc.size();
            acc.push_back(std::move(r));
        } else {
            ProfRow& r = acc[it->second];
            r.real_time_us += rt;
            r.cpu_time_us += ct;
            r.count += 1;
            // keep last-seen exec_type/status (they don't change across iters)
            if (!p.exec_type.empty()) r.exec_type = p.exec_type;
            r.status = status_str(p.status);
        }
    }
}

void collect_profiling(ov::InferRequest& request, std::vector<ProfRow>& acc) {
    std::vector<ov::ProfilingInfo> pinfo;
    try {
        pinfo = request.get_profiling_info();
    } catch (const std::exception& e) {
        std::printf("  [profile] get_profiling_info() unavailable: %s\n", e.what());
        return;
    }
    if (pinfo.empty()) {
        return;  // GPU/NPU may return nothing; guard silently per-call.
    }
    accumulate_profiling(std::move(pinfo), acc);
}

void print_device_properties(ov::CompiledModel& compiled, const std::string& device) {
    std::printf("\n  --- Device / CompiledModel properties (%s) ---\n", device.c_str());
    auto try_print = [&](const std::string& key) {
        try {
            ov::Any v = compiled.get_property(key);
            std::printf("  %-38s : %s\n", key.c_str(), v.as<std::string>().c_str());
        } catch (...) {
            // not supported on this device; skip
        }
    };
    try_print(ov::optimal_number_of_infer_requests.name());
    try_print(ov::execution_devices.name());
    try_print(ov::hint::performance_mode.name());
    try_print(ov::num_streams.name());
    try_print(ov::inference_num_threads.name());
    try_print(ov::hint::num_requests.name());
    try_print(ov::loaded_from_cache.name());
}

void print_profiling_report(const std::vector<ProfRow>& acc, const std::string& label, int topk,
                            long long iterations) {
    std::string sep(96, '=');
    std::printf("\n%s\n", sep.c_str());
    std::printf("  PER-OPERATION PROFILING — %s  (accumulated over %lld inference call(s))\n",
                label.c_str(), iterations);
    std::printf("%s\n", sep.c_str());
    if (acc.empty()) {
        std::printf("  No profiling data (plugin returned none — common on GPU/NPU).\n");
        std::printf("%s\n", sep.c_str());
        return;
    }

    long long total_real = 0;
    int executed = 0, optimized_out = 0, not_run = 0;
    std::map<std::string, long long> by_type;        // node_type -> real_us
    std::map<std::string, long long> by_type_count;  // node_type -> op count
    std::map<std::string, long long> by_exec;        // exec_type -> real_us
    for (const auto& r : acc) {
        if (r.status == "EXECUTED") {
            total_real += r.real_time_us;
            ++executed;
            by_type[r.node_type] += r.real_time_us;
            by_type_count[r.node_type] += 1;
            by_exec[r.exec_type] += r.real_time_us;
        } else if (r.status == "OPTIMIZED_OUT") {
            ++optimized_out;
        } else {
            ++not_run;
        }
    }

    std::printf("  Ops: %zu total  |  EXECUTED=%d  OPTIMIZED_OUT=%d  NOT_RUN=%d\n", acc.size(),
                executed, optimized_out, not_run);
    const double iters = iterations > 0 ? static_cast<double>(iterations) : 1.0;
    std::printf("  Sum of EXECUTED node real_time: %.3f ms/iter  (parallel exec may exceed wall-clock)\n",
                (total_real / iters) / 1000.0);

    // --- Top-N hottest ops ---
    std::vector<const ProfRow*> exec_rows;
    for (const auto& r : acc) {
        if (r.status == "EXECUTED") exec_rows.push_back(&r);
    }
    std::sort(exec_rows.begin(), exec_rows.end(),
              [](const ProfRow* a, const ProfRow* b) { return a->real_time_us > b->real_time_us; });

    std::printf("\n  Top %d ops by real_time:\n", topk);
    std::printf("  %-34s %-16s %-22s %10s %8s\n", "node_name", "node_type", "exec_type",
                "ms/iter", "%%");
    std::printf("  %-34s %-16s %-22s %10s %8s\n", "----------------------------------",
                "----------------", "----------------------", "----------", "--------");
    int shown = 0;
    for (const ProfRow* r : exec_rows) {
        if (shown++ >= topk) break;
        double ms_iter = (r->real_time_us / iters) / 1000.0;
        double pct = total_real ? 100.0 * r->real_time_us / total_real : 0.0;
        std::printf("  %-34s %-16s %-22s %10.4f %7.2f%%\n", truncate(r->node_name, 34).c_str(),
                    truncate(r->node_type, 16).c_str(), truncate(r->exec_type, 22).c_str(),
                    ms_iter, pct);
    }

    // --- Aggregate by node_type ---
    std::vector<std::pair<std::string, long long>> types(by_type.begin(), by_type.end());
    std::sort(types.begin(), types.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::printf("\n  Aggregate by node_type:\n");
    std::printf("  %-24s %8s %12s %8s\n", "node_type", "count", "ms/iter", "%%");
    std::printf("  %-24s %8s %12s %8s\n", "------------------------", "--------", "------------",
                "--------");
    for (const auto& kv : types) {
        double ms_iter = (kv.second / iters) / 1000.0;
        double pct = total_real ? 100.0 * kv.second / total_real : 0.0;
        std::printf("  %-24s %8lld %12.4f %7.2f%%\n", truncate(kv.first, 24).c_str(),
                    by_type_count[kv.first], ms_iter, pct);
    }

    // --- Aggregate by exec_type (the actual kernel chosen) ---
    std::vector<std::pair<std::string, long long>> execs(by_exec.begin(), by_exec.end());
    std::sort(execs.begin(), execs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::printf("\n  Aggregate by exec_type (kernel):\n");
    std::printf("  %-30s %12s %8s\n", "exec_type", "ms/iter", "%%");
    std::printf("  %-30s %12s %8s\n", "------------------------------", "------------", "--------");
    for (const auto& kv : execs) {
        double ms_iter = (kv.second / iters) / 1000.0;
        double pct = total_real ? 100.0 * kv.second / total_real : 0.0;
        std::printf("  %-30s %12.4f %7.2f%%\n", truncate(kv.first, 30).c_str(), ms_iter, pct);
    }
    std::printf("%s\n", sep.c_str());
}

void write_profiling_csv(const std::string& path, const std::vector<ProfRow>& acc,
                         const std::string& model, const std::string& device,
                         const std::string& precision, const std::string& stage,
                         long long iterations) {
    if (acc.empty()) return;
    std::vector<std::string> header = {
        "model", "device", "precision", "stage", "iterations", "node_name", "node_type",
        "exec_type", "status", "real_time_us_total", "cpu_time_us_total", "real_time_us_per_iter",
        "count",
    };
    std::vector<std::vector<std::string>> rows;
    for (const auto& r : acc) {
        double per_iter = iterations > 0 ? static_cast<double>(r.real_time_us) / iterations : 0.0;
        rows.push_back({
            model, device, precision, stage, std::to_string(iterations), r.node_name, r.node_type,
            r.exec_type, r.status, std::to_string(r.real_time_us), std::to_string(r.cpu_time_us),
            f3(per_iter), std::to_string(r.count),
        });
    }
    write_csv_append(path, header, rows);
}

}  // namespace mb
