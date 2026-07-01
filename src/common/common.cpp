// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/common.hpp"

#include <chrono>
#include <ctime>
#include <cstdio>

namespace mb {

std::string now_iso8601() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t tt = system_clock::to_time_t(now);
    const auto us =
        duration_cast<microseconds>(now.time_since_epoch()).count() % 1000000;
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%06ld",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                  static_cast<long>(us));
    return std::string(buf);
}

}  // namespace mb
