// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Summary statistics matching tests/lib/stats.py exactly:
//   - mean  : arithmetic mean
//   - std   : POPULATION standard deviation (divide by N), 0 when N<=1
//   - p50/p90/p95 : numpy.percentile linear-interpolation
//   - min/max
// Empty input -> all zeros (callers detect via count()).
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mb {

struct Stats {
    double mean = 0.0;
    double std = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

// numpy.percentile with the default 'linear' interpolation.
// `sorted_vals` MUST already be sorted ascending and non-empty.
inline double percentile_linear(const std::vector<double>& sorted_vals, double p) {
    const size_t n = sorted_vals.size();
    if (n == 1) {
        return sorted_vals[0];
    }
    const double rank = (p / 100.0) * static_cast<double>(n - 1);
    const double lo = std::floor(rank);
    const double hi = std::ceil(rank);
    const double frac = rank - lo;
    const double v_lo = sorted_vals[static_cast<size_t>(lo)];
    const double v_hi = sorted_vals[static_cast<size_t>(hi)];
    return v_lo + frac * (v_hi - v_lo);
}

inline Stats compute_stats(const std::vector<double>& values) {
    Stats s;
    if (values.empty()) {
        return s;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    s.mean = sum / static_cast<double>(values.size());

    if (values.size() > 1) {
        double sq = 0.0;
        for (double v : values) {
            const double d = v - s.mean;
            sq += d * d;
        }
        s.std = std::sqrt(sq / static_cast<double>(values.size()));  // population
    } else {
        s.std = 0.0;
    }

    std::vector<double> sorted_vals = values;
    std::sort(sorted_vals.begin(), sorted_vals.end());
    s.p50 = percentile_linear(sorted_vals, 50.0);
    s.p90 = percentile_linear(sorted_vals, 90.0);
    s.p95 = percentile_linear(sorted_vals, 95.0);
    s.min = sorted_vals.front();
    s.max = sorted_vals.back();
    return s;
}

inline double fmean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

}  // namespace mb
