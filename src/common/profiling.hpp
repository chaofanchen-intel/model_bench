// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Per-operation profiling utilities — the capability the Python PerfMetrics
// path cannot provide. Given an ov::InferRequest that has run at least once
// with ov::enable_profiling(true) at compile time, collect
// get_profiling_info() and produce:
//   - a hotspot table (top-N ops by real_time)
//   - an aggregate-by-node_type breakdown
//   - an optional CSV dump (node_name,node_type,exec_type,status,real_time_us,
//     cpu_time_us,count)
// plus device-property introspection from a CompiledModel.
#pragma once

#include <string>
#include <vector>

#include "openvino/openvino.hpp"

namespace mb {

struct ProfRow {
    std::string node_name;
    std::string node_type;
    std::string exec_type;
    std::string status;
    long long real_time_us = 0;
    long long cpu_time_us = 0;
    long long count = 1;  // accumulated across iterations
};

// Accumulate one InferRequest's profiling info into `acc` (keyed by node_name).
void accumulate_profiling(std::vector<ov::ProfilingInfo>&& pinfo, std::vector<ProfRow>& acc);

// Collect get_profiling_info() from a request, accumulating into `acc`.
void collect_profiling(ov::InferRequest& request, std::vector<ProfRow>& acc);

// Print the device-level introspection block (compiled-model properties).
void print_device_properties(ov::CompiledModel& compiled, const std::string& device);

// Print a hotspot report (top-N ops + node_type aggregate). `label` names the
// stage (e.g. "lingua infer", "LLM prefill+decode"). divisor is the number of
// accumulated iterations (to print per-iteration averages).
void print_profiling_report(const std::vector<ProfRow>& acc, const std::string& label,
                            int topk, long long iterations);

// Write the profiling rows to CSV (append-mode, header-aware). `tag` columns
// (model/device/stage) prefix every row for cross-run comparison.
void write_profiling_csv(const std::string& path, const std::vector<ProfRow>& acc,
                         const std::string& model, const std::string& device,
                         const std::string& precision, const std::string& stage,
                         long long iterations);

}  // namespace mb
