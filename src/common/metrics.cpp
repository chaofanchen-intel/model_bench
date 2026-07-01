// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/metrics.hpp"

#include <algorithm>

namespace mb {

void attach_memory(IterMemory& m, const MemorySummary& s) {
    if (!s.available) return;
    m.memory_available = true;
    m.memory_samples = s.samples;
    m.max_rss_mem_mib = s.max_rss_mem_mib;
    m.max_sys_mem_mib = s.max_sys_mem_mib;
    m.max_rss_mem_increase_mib = s.max_rss_mem_increase_mib;
    m.max_sys_mem_increase_mib = s.max_sys_mem_increase_mib;
    m.max_rss_mem_share = s.max_rss_mem_share;
    m.max_sys_mem_share = s.max_sys_mem_share;
}

void fill_memory_stats(ResultMemory& out, const MemoryCurrent& initial,
                       const MemorySummary& compile_mem,
                       const std::vector<IterMemory>& measured_mem) {
    if (initial.available) {
        out.memory_available = true;
        out.initial_rss_mem_mib = initial.rss_mem_mib;
        out.initial_sys_mem_mib = initial.sys_mem_mib;
        out.initial_rss_mem_share = initial.rss_mem_share;
        out.initial_sys_mem_share = initial.sys_mem_share;
    }
    if (compile_mem.available) {
        out.memory_available = true;
        out.memory_samples += compile_mem.samples;
        out.compile_max_rss_mem_mib = compile_mem.max_rss_mem_mib;
        out.compile_max_sys_mem_mib = compile_mem.max_sys_mem_mib;
        out.compile_max_rss_mem_increase_mib = compile_mem.max_rss_mem_increase_mib;
        out.compile_max_sys_mem_increase_mib = compile_mem.max_sys_mem_increase_mib;
        out.compile_max_rss_mem_share = compile_mem.max_rss_mem_share;
        out.compile_max_sys_mem_share = compile_mem.max_sys_mem_share;
    }

    std::vector<IterMemory> avail;
    for (auto& m : measured_mem) {
        if (m.memory_available) avail.push_back(m);
    }
    if (avail.empty()) return;

    out.memory_available = true;
    for (auto& m : avail) out.memory_samples += m.memory_samples;

    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0;
        for (double x : v) s += x;
        return s / v.size();
    };
    std::vector<double> rss, sysm, rss_inc, sys_inc;
    for (auto& m : avail) {
        rss.push_back(m.max_rss_mem_mib);
        sysm.push_back(m.max_sys_mem_mib);
        rss_inc.push_back(m.max_rss_mem_increase_mib);
        sys_inc.push_back(m.max_sys_mem_increase_mib);
    }
    out.runtime_mean_rss_mem_mib = mean(rss);
    out.runtime_max_rss_mem_mib = *std::max_element(rss.begin(), rss.end());
    out.runtime_mean_sys_mem_mib = mean(sysm);
    out.runtime_max_sys_mem_mib = *std::max_element(sysm.begin(), sysm.end());
    out.runtime_mean_rss_mem_increase_mib = mean(rss_inc);
    out.runtime_max_rss_mem_increase_mib = *std::max_element(rss_inc.begin(), rss_inc.end());
    out.runtime_mean_sys_mem_increase_mib = mean(sys_inc);
    out.runtime_max_sys_mem_increase_mib = *std::max_element(sys_inc.begin(), sys_inc.end());
    double max_rss_share = 0, max_sys_share = 0;
    for (auto& m : avail) {
        max_rss_share = std::max(max_rss_share, m.max_rss_mem_share);
        max_sys_share = std::max(max_sys_share, m.max_sys_mem_share);
    }
    out.runtime_max_rss_mem_share = max_rss_share;
    out.runtime_max_sys_mem_share = max_sys_share;
}

}  // namespace mb
