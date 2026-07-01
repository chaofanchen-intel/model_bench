// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/monitors.hpp"

#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <numeric>
#include <sstream>

namespace mb {

namespace {

double monotonic_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

void sleep_s(double s) {
    std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

std::string run_capture(const std::string& cmd) {
    std::string out;
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return out;
    std::array<char, 4096> buf{};
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), fp)) > 0) {
        out.append(buf.data(), n);
    }
    pclose(fp);
    return out;
}

// Extract the maximum "value": N seen inside the "compute" array of an
// xpu-smi engine_util JSON. We do a tolerant scan rather than a full parse.
std::optional<double> parse_xpu_compute_max(const std::string& json_text) {
    auto eu = json_text.find("\"engine_util\"");
    if (eu == std::string::npos) {
        // fall back to "gpu_util"
        auto gu = json_text.find("\"gpu_util\"");
        if (gu == std::string::npos) return std::nullopt;
        auto colon = json_text.find(':', gu);
        if (colon == std::string::npos) return std::nullopt;
        try {
            return std::stod(json_text.substr(colon + 1));
        } catch (...) {
            return std::nullopt;
        }
    }
    auto comp = json_text.find("\"compute\"", eu);
    if (comp == std::string::npos) return std::nullopt;
    auto start = json_text.find('[', comp);
    auto end = json_text.find(']', comp);
    if (start == std::string::npos || end == std::string::npos || end < start) {
        return std::nullopt;
    }
    std::string seg = json_text.substr(start, end - start);
    double best = -1.0;
    size_t pos = 0;
    const std::string key = "\"value\"";
    while ((pos = seg.find(key, pos)) != std::string::npos) {
        auto colon = seg.find(':', pos);
        if (colon == std::string::npos) break;
        try {
            double v = std::stod(seg.substr(colon + 1));
            if (v > best) best = v;
        } catch (...) {
        }
        pos = colon + 1;
    }
    if (best < 0) return std::nullopt;
    return best;
}

}  // namespace

// ============================ CpuMonitor =====================================

CpuMonitor::CpuMonitor(int interval_ms) : interval_s_(interval_ms / 1000.0) {}

std::map<std::string, std::vector<long long>> CpuMonitor::read_proc_stat() {
    std::map<std::string, std::vector<long long>> result;
    std::ifstream f("/proc/stat");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("cpu", 0) != 0) break;
        std::istringstream iss(line);
        std::string label;
        iss >> label;
        std::vector<long long> vals;
        long long x;
        while (vals.size() < 7 && (iss >> x)) {
            vals.push_back(x);
        }
        if (vals.size() == 7) {
            result[label] = vals;
        }
    }
    return result;
}

void CpuMonitor::start() {
    running_ = true;
    prev_ = read_proc_stat();
    thread_ = std::thread([this] { loop(); });
}

void CpuMonitor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void CpuMonitor::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    samples_.clear();
    aggregate_sum_ = 0.0;
    aggregate_count_ = 0;
}

void CpuMonitor::loop() {
    while (running_) {
        sleep_s(interval_s_);
        auto cur = read_proc_stat();
        std::map<std::string, double> sample;
        double total_util = 0.0;
        int core_count = 0;
        for (auto& kv : cur) {
            const std::string& key = kv.first;
            auto it = prev_.find(key);
            if (it == prev_.end()) continue;
            const auto& pv = it->second;
            const auto& cv = kv.second;
            long long prev_idle = pv[3] + pv[4];
            long long cur_idle = cv[3] + cv[4];
            long long prev_total = std::accumulate(pv.begin(), pv.end(), 0LL);
            long long cur_total = std::accumulate(cv.begin(), cv.end(), 0LL);
            long long dt = cur_total - prev_total;
            long long di = cur_idle - prev_idle;
            double util = 0.0;
            if (dt > 0) {
                util = 100.0 * (1.0 - static_cast<double>(di) / static_cast<double>(dt));
                if (util < 0) util = 0;
                if (util > 100) util = 100;
            }
            if (key != "cpu") {
                sample[key] = util;
                total_util += util;
                ++core_count;
            }
        }
        prev_ = cur;
        if (core_count > 0) {
            std::lock_guard<std::mutex> lk(mtx_);
            samples_.push_back(sample);
            aggregate_sum_ += total_util / core_count;
            ++aggregate_count_;
        }
    }
}

CpuSummary CpuMonitor::get_summary() {
    std::lock_guard<std::mutex> lk(mtx_);
    CpuSummary s;
    if (samples_.empty()) return s;
    s.avg_total = aggregate_count_ ? aggregate_sum_ / aggregate_count_ : 0.0;
    std::map<std::string, std::vector<double>> acc;
    for (auto& sample : samples_) {
        for (auto& kv : sample) acc[kv.first].push_back(kv.second);
    }
    for (auto& kv : acc) {
        double sum = std::accumulate(kv.second.begin(), kv.second.end(), 0.0);
        s.per_core[kv.first] = sum / kv.second.size();
    }
    return s;
}

// ============================ GpuMonitor =====================================

GpuMonitor::GpuMonitor(int interval_ms, std::optional<std::string> command)
    : interval_s_(interval_ms / 1000.0) {
    if (command) {
        command_ = *command;
    } else if (const char* env = std::getenv("XPU_SMI_CMD")) {
        command_ = env;
    } else {
        command_ = "sudo xpu-smi stats -d 1 -j";
    }
}

bool GpuMonitor::check_available() {
    std::string out = run_capture("which xpu-smi 2>/dev/null");
    return !out.empty();
}

void GpuMonitor::start() {
    if (!check_available()) return;
    available_ = true;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void GpuMonitor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void GpuMonitor::reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    sum_ = 0.0;
    count_ = 0;
}

void GpuMonitor::loop() {
    while (running_) {
        sleep_s(interval_s_);
        auto util = read_gpu_util();
        if (util) {
            std::lock_guard<std::mutex> lk(mtx_);
            sum_ += *util;
            ++count_;
        }
    }
}

std::optional<double> GpuMonitor::read_gpu_util() {
    std::string out = run_capture(command_ + " 2>/dev/null");
    if (out.empty()) return std::nullopt;
    return parse_xpu_compute_max(out);
}

GpuSummary GpuMonitor::get_summary() {
    std::lock_guard<std::mutex> lk(mtx_);
    GpuSummary s;
    s.available = available_;
    s.avg_util = count_ ? sum_ / count_ : 0.0;
    s.samples = count_;
    return s;
}

// ============================ NpuMonitor =====================================

NpuMonitor::NpuMonitor(int interval_ms) : interval_s_(interval_ms / 1000.0) {}

std::optional<std::string> NpuMonitor::read_sysfs(const std::string& path) {
    std::ifstream f(path);
    if (f.good()) {
        std::string s;
        std::getline(f, s);
        if (!s.empty()) {
            // trim trailing whitespace
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
                s.pop_back();
            }
            return s;
        }
    }
    // PermissionError fallback: sudo cat
    std::string out = run_capture("sudo cat " + path + " 2>/dev/null");
    if (!out.empty()) {
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
            out.pop_back();
        }
        return out;
    }
    return std::nullopt;
}

void NpuMonitor::detect() {
    static const std::map<std::string, std::string> kGuids = {
        {"0x130670b2", "MTL"}, {"0x1306a0b3", "ARL"}, {"0x1306a0b2", "ARL_H"},
        {"0x1306a0b4", "ARL_S"}, {"0x3072005", "LNL"}, {"0x3086000", "PTL"},
        {"0x308d000", "WCL"},
    };
    const std::string drv = "/sys/bus/pci/drivers/intel_vpu/";
    DIR* d = opendir(drv.c_str());
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string name = ent->d_name;
            if (name.rfind("0000:", 0) == 0) {
                std::string busy = drv + name + "/npu_busy_time_us";
                std::ifstream probe(busy);
                if (probe.good()) {
                    npu_busy_path_ = busy;
                    available_ = true;
                }
                break;
            }
        }
        closedir(d);
    }
    const std::string pmt_root = "/sys/class/intel_pmt";
    DIR* pd = opendir(pmt_root.c_str());
    if (pd) {
        struct dirent* ent;
        while ((ent = readdir(pd)) != nullptr) {
            std::string name = ent->d_name;
            if (name.rfind("telem", 0) != 0) continue;
            auto guid = read_sysfs(pmt_root + "/" + name + "/guid");
            if (guid) {
                auto it = kGuids.find(*guid);
                if (it != kGuids.end()) {
                    cpu_gen_ = it->second;
                    break;
                }
            }
        }
        closedir(pd);
    }
}

std::optional<long long> NpuMonitor::read_busy_time() {
    if (npu_busy_path_.empty()) return std::nullopt;
    auto s = read_sysfs(npu_busy_path_);
    if (!s) return std::nullopt;
    try {
        return std::stoll(*s);
    } catch (...) {
        return std::nullopt;
    }
}

void NpuMonitor::start() {
    detect();
    if (!available_) return;
    running_ = true;
    prev_busy_us_ = read_busy_time();
    prev_time_ = monotonic_s();
    thread_ = std::thread([this] { loop(); });
}

void NpuMonitor::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void NpuMonitor::reset() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        sum_ = 0.0;
        count_ = 0;
    }
    prev_busy_us_ = read_busy_time();
    prev_time_ = monotonic_s();
}

void NpuMonitor::loop() {
    while (running_) {
        sleep_s(interval_s_);
        auto cur_busy = read_busy_time();
        double cur_time = monotonic_s();
        if (cur_busy && prev_busy_us_) {
            long long delta_us = *cur_busy - *prev_busy_us_;
            double delta_s = cur_time - prev_time_;
            if (delta_s > 0) {
                double util = 100.0 * (delta_us / 1.0e6) / delta_s;
                if (util < 0) util = 0;
                if (util > 100) util = 100;
                std::lock_guard<std::mutex> lk(mtx_);
                sum_ += util;
                ++count_;
            }
        }
        prev_busy_us_ = cur_busy;
        prev_time_ = cur_time;
    }
}

NpuSummary NpuMonitor::get_summary() {
    std::lock_guard<std::mutex> lk(mtx_);
    NpuSummary s;
    s.available = available_;
    s.cpu_gen = cpu_gen_;
    s.avg_util = count_ ? sum_ / count_ : 0.0;
    s.samples = count_;
    return s;
}

// ======================= MemoryUsageMonitor =================================

MemoryUsageMonitor::MemoryUsageMonitor(bool enabled, int interval_ms)
    : enabled_(enabled), interval_s_((interval_ms < 1 ? 1 : interval_ms) / 1000.0) {}

long long MemoryUsageMonitor::read_rss_bytes() {
    // Self RSS from /proc/self/statm (pages) — plus children via /proc walk.
    auto rss_of = [](const std::string& status_path) -> long long {
        std::ifstream f(status_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                std::istringstream iss(line.substr(6));
                long long kb;
                iss >> kb;
                return kb * 1024;
            }
        }
        return -1;
    };
    long long self_rss = rss_of("/proc/self/status");
    if (self_rss < 0) return -1;
    long long total = self_rss;

    // Recursive children: build pid -> ppid map, then sum descendants of self.
    pid_t self = getpid();
    std::map<pid_t, pid_t> parent;
    std::map<pid_t, long long> rssmap;
    DIR* d = opendir("/proc");
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            char* endp = nullptr;
            long pid = strtol(ent->d_name, &endp, 10);
            if (endp == ent->d_name || *endp != '\0') continue;
            std::string base = std::string("/proc/") + ent->d_name;
            std::ifstream st(base + "/stat");
            if (!st.good()) continue;
            // stat format: pid (comm) state ppid ...
            std::string content;
            std::getline(st, content);
            auto rp = content.rfind(')');
            if (rp == std::string::npos) continue;
            std::istringstream iss(content.substr(rp + 1));
            std::string state;
            pid_t ppid = 0;
            iss >> state >> ppid;
            parent[static_cast<pid_t>(pid)] = ppid;
            rssmap[static_cast<pid_t>(pid)] = rss_of(base + "/status");
        }
        closedir(d);
    }
    // Sum descendants (BFS).
    std::vector<pid_t> stack{self};
    std::map<pid_t, bool> seen;
    seen[self] = true;
    while (!stack.empty()) {
        pid_t cur = stack.back();
        stack.pop_back();
        for (auto& kv : parent) {
            if (kv.second == cur && !seen[kv.first]) {
                seen[kv.first] = true;
                long long r = rssmap[kv.first];
                if (r > 0) total += r;
                stack.push_back(kv.first);
            }
        }
    }
    return total;
}

void MemoryUsageMonitor::read_system_bytes(long long& used, long long& total) {
    used = -1;
    total = -1;
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long memtotal = -1, memavail = -1;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        std::string k;
        long long v;
        std::string unit;
        iss >> k >> v >> unit;
        if (k == "MemTotal:") memtotal = v * 1024;
        else if (k == "MemAvailable:") memavail = v * 1024;
        if (memtotal >= 0 && memavail >= 0) break;
    }
    if (memtotal >= 0 && memavail >= 0) {
        total = memtotal;
        used = memtotal - memavail;
    }
}

MemoryUsageMonitor::Sample MemoryUsageMonitor::read_sample() {
    Sample s;
    long long rss = read_rss_bytes();
    long long used = -1, total = -1;
    read_system_bytes(used, total);
    if (rss < 0 || used < 0 || total <= 0) return s;
    const double mib = 1024.0 * 1024.0;
    s.rss_mib = rss / mib;
    s.sys_mib = used / mib;
    s.total_mib = total / mib;
    s.ts = monotonic_s();
    s.valid = true;
    return s;
}

void MemoryUsageMonitor::start() {
    if (!enabled_) return;
    reset();
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void MemoryUsageMonitor::stop() {
    if (!enabled_) return;
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void MemoryUsageMonitor::reset() {
    if (!enabled_) return;
    Sample s = read_sample();
    std::lock_guard<std::mutex> lk(mtx_);
    samples_.clear();
    baseline_ = s;
    if (s.valid) {
        samples_.push_back(s);
        available_ = true;
    }
}

void MemoryUsageMonitor::loop() {
    while (running_) {
        sleep_s(interval_s_);
        Sample s = read_sample();
        if (s.valid) {
            std::lock_guard<std::mutex> lk(mtx_);
            samples_.push_back(s);
            available_ = true;
        }
    }
}

MemoryCurrent MemoryUsageMonitor::current() {
    MemoryCurrent c;
    if (!enabled_) return c;
    Sample s = read_sample();
    if (!s.valid) return c;
    c.available = true;
    c.rss_mem_mib = s.rss_mib;
    c.sys_mem_mib = s.sys_mib;
    c.rss_mem_share = share(s.rss_mib, s.total_mib);
    c.sys_mem_share = share(s.sys_mib, s.total_mib);
    return c;
}

MemorySummary MemoryUsageMonitor::summary() {
    MemorySummary out;
    if (!enabled_) return out;
    Sample s = read_sample();
    std::vector<Sample> samples;
    Sample baseline;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (s.valid) {
            samples_.push_back(s);
            available_ = true;
        }
        if (samples_.empty() || !baseline_.valid) return out;
        samples = samples_;
        baseline = baseline_;
    }
    double max_rss = 0, max_sys = 0;
    for (auto& sm : samples) {
        if (sm.rss_mib > max_rss) max_rss = sm.rss_mib;
        if (sm.sys_mib > max_sys) max_sys = sm.sys_mib;
    }
    double total_mib = samples.back().total_mib;
    out.available = available_;
    out.samples = static_cast<int>(samples.size());
    out.start_rss_mem_mib = baseline.rss_mib;
    out.start_sys_mem_mib = baseline.sys_mib;
    out.max_rss_mem_mib = max_rss;
    out.max_sys_mem_mib = max_sys;
    out.max_rss_mem_increase_mib = std::max(0.0, max_rss - baseline.rss_mib);
    out.max_sys_mem_increase_mib = std::max(0.0, max_sys - baseline.sys_mib);
    out.max_rss_mem_share = share(max_rss, total_mib);
    out.max_sys_mem_share = share(max_sys, total_mib);
    return out;
}

// ============================ MonitorBundle ==================================

MonitorBundle::MonitorBundle(bool enabled, int cpu_ms, int gpu_ms, int npu_ms,
                             std::optional<std::string> gpu_command)
    : enabled_(enabled),
      cpu_(cpu_ms),
      gpu_(gpu_ms, std::move(gpu_command)),
      npu_(npu_ms) {}

void MonitorBundle::start() {
    if (!enabled_) return;
    cpu_.start();
    gpu_.start();
    npu_.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

void MonitorBundle::stop() {
    if (!enabled_) return;
    cpu_.stop();
    gpu_.stop();
    npu_.stop();
}

void MonitorBundle::reset() {
    if (!enabled_) return;
    cpu_.reset();
    gpu_.reset();
    npu_.reset();
}

CpuSummary MonitorBundle::cpu_summary() {
    if (!enabled_) return {};
    return cpu_.get_summary();
}
GpuSummary MonitorBundle::gpu_summary() {
    if (!enabled_) return {};
    return gpu_.get_summary();
}
NpuSummary MonitorBundle::npu_summary() {
    if (!enabled_) return {};
    return npu_.get_summary();
}

}  // namespace mb
