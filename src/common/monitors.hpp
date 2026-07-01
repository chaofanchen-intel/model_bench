// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Native C++ port of tests/lib/monitors.py:
//   CpuMonitor   — per-core utilization from /proc/stat (idle = fields[3]+[4],
//                  total = sum of first 7 fields).
//   GpuMonitor   — Intel GPU via `xpu-smi stats -d 1 -j` (engine_util.compute max).
//   NpuMonitor   — Intel NPU via /sys/bus/pci/drivers/intel_vpu/*/npu_busy_time_us
//                  delta + intel_pmt GUID -> cpu generation label.
//   MemoryUsageMonitor — process RSS (incl. recursive children) + system used
//                  memory in MiB, with baseline/increase/share.
// All sample on background std::threads; all default OFF at the bundle level.
#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace mb {

struct CpuSummary {
    double avg_total = 0.0;
    std::map<std::string, double> per_core;  // "cpu0" -> mean util
};

struct GpuSummary {
    bool available = false;
    double avg_util = 0.0;
    int samples = 0;
};

struct NpuSummary {
    bool available = false;
    std::string cpu_gen;
    double avg_util = 0.0;
    int samples = 0;
};

struct MemoryCurrent {
    bool available = false;
    double rss_mem_mib = 0.0;
    double sys_mem_mib = 0.0;
    double rss_mem_share = 0.0;
    double sys_mem_share = 0.0;
};

struct MemorySummary {
    bool available = false;
    int samples = 0;
    double start_rss_mem_mib = 0.0;
    double start_sys_mem_mib = 0.0;
    double max_rss_mem_mib = 0.0;
    double max_sys_mem_mib = 0.0;
    double max_rss_mem_increase_mib = 0.0;
    double max_sys_mem_increase_mib = 0.0;
    double max_rss_mem_share = 0.0;
    double max_sys_mem_share = 0.0;
};

class CpuMonitor {
public:
    explicit CpuMonitor(int interval_ms = 100);
    void start();
    void stop();
    void reset();
    CpuSummary get_summary();

private:
    void loop();
    static std::map<std::string, std::vector<long long>> read_proc_stat();

    double interval_s_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mtx_;
    std::map<std::string, std::vector<long long>> prev_;
    std::vector<std::map<std::string, double>> samples_;
    double aggregate_sum_ = 0.0;
    long aggregate_count_ = 0;
};

class GpuMonitor {
public:
    explicit GpuMonitor(int interval_ms = 500, std::optional<std::string> command = std::nullopt);
    void start();
    void stop();
    void reset();
    GpuSummary get_summary();

private:
    bool check_available();
    void loop();
    std::optional<double> read_gpu_util();

    double interval_s_;
    std::string command_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mtx_;
    double sum_ = 0.0;
    int count_ = 0;
    bool available_ = false;
};

class NpuMonitor {
public:
    explicit NpuMonitor(int interval_ms = 500);
    void start();
    void stop();
    void reset();
    NpuSummary get_summary();

private:
    void detect();
    void loop();
    std::optional<long long> read_busy_time();
    static std::optional<std::string> read_sysfs(const std::string& path);

    double interval_s_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mtx_;
    double sum_ = 0.0;
    int count_ = 0;
    bool available_ = false;
    std::string npu_busy_path_;
    std::string cpu_gen_;
    std::optional<long long> prev_busy_us_;
    double prev_time_ = 0.0;
};

class MemoryUsageMonitor {
public:
    explicit MemoryUsageMonitor(bool enabled = false, int interval_ms = 100);
    void start();
    void stop();
    void reset();
    MemoryCurrent current();
    MemorySummary summary();

private:
    struct Sample {
        double rss_mib = 0.0;
        double sys_mib = 0.0;
        double total_mib = 0.0;
        double ts = 0.0;
        bool valid = false;
    };
    void loop();
    Sample read_sample();
    static long long read_rss_bytes();         // self + children
    static void read_system_bytes(long long& used, long long& total);
    static double share(double v, double total) { return total ? 100.0 * v / total : 0.0; }

    bool enabled_;
    double interval_s_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::mutex mtx_;
    std::vector<Sample> samples_;
    Sample baseline_;
    bool available_ = false;
};

// Bundle owning one of each utilization monitor (matches MonitorBundle).
class MonitorBundle {
public:
    MonitorBundle(bool enabled, int cpu_ms, int gpu_ms, int npu_ms,
                  std::optional<std::string> gpu_command);
    void start();
    void stop();
    void reset();
    CpuSummary cpu_summary();
    GpuSummary gpu_summary();
    NpuSummary npu_summary();
    bool enabled() const { return enabled_; }

private:
    bool enabled_;
    CpuMonitor cpu_;
    GpuMonitor gpu_;
    NpuMonitor npu_;
};

}  // namespace mb
