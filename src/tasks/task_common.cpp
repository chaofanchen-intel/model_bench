// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "tasks/tasks.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "common/common.hpp"
#include "openvino/runtime/properties.hpp"

namespace mb {

ov::AnyMap build_plugin_config(const Args& args, const std::string& device_upper,
                               const std::string& attention_backend, bool enable_profiling) {
    ov::AnyMap cfg;
    // Python always sets CACHE_DIR (default "") because args.cache_dir defaults to "".
    if (args.cache_dir_set) {
        cfg["CACHE_DIR"] = args.cache_dir;
    }
    if (args.num_streams) {
        cfg["NUM_STREAMS"] = *args.num_streams;
    }
    if (args.load_config) {
        auto extra = load_extra_ov_config(*args.load_config);
        for (auto& kv : extra) cfg[kv.first] = kv.second;
    }
    if (!attention_backend.empty() && device_upper != "NPU" &&
        cfg.find("ATTENTION_BACKEND") == cfg.end()) {
        cfg["ATTENTION_BACKEND"] = attention_backend;
    }
    // GPU default: LATENCY performance hint (matches the python harness). Only set when
    // the caller passes a non-empty attention_backend (i.e. the real generation path, not
    // the deep-profile path) and the user did not already provide it via --load-config.
    if (!attention_backend.empty() && device_upper != "NPU" &&
        cfg.find("PERFORMANCE_HINT") == cfg.end()) {
        cfg["PERFORMANCE_HINT"] = "LATENCY";
    }
    if (enable_profiling) {
        cfg[ov::enable_profiling.name()] = true;
    }
    return cfg;
}

void ensure_sudo_if_monitoring(const Args& args) {
    if (args.no_monitor) return;
    std::printf("  [sudo] Refreshing credentials for hardware monitoring (Ctrl-C to skip)...\n");
    std::fflush(stdout);
    int rc = std::system("sudo -v");
    if (rc == 0) {
        std::printf("  [sudo] OK\n\n");
    } else {
        std::printf("  [sudo] auth failed; GPU/NPU monitors may not work\n\n");
    }
    std::fflush(stdout);
}

std::string detect_project_dir() {
    // Walk up from cwd looking for a dir that contains "reference/openvino.genai".
    std::filesystem::path p = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i) {
        if (std::filesystem::exists(p / "reference" / "openvino.genai")) {
            return p.string();
        }
        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    return std::filesystem::current_path().string();
}

}  // namespace mb
