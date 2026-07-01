// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Fixed-precision number -> string helpers matching Python f-string formats
// used by tests/lib/reports.py (e.g. f"{x:.3f}", f"{x:.2f}").
#pragma once

#include <cstdio>
#include <string>

namespace mb {

inline std::string fmt_fixed(double x, int prec) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, x);
    return std::string(buf);
}

inline std::string f3(double x) { return fmt_fixed(x, 3); }
inline std::string f2(double x) { return fmt_fixed(x, 2); }
inline std::string f1(double x) { return fmt_fixed(x, 1); }
inline std::string f5(double x) { return fmt_fixed(x, 5); }
inline std::string f0(double x) { return fmt_fixed(x, 0); }
inline std::string f4(double x) { return fmt_fixed(x, 4); }

inline std::string boolstr(bool b) { return b ? "True" : "False"; }

}  // namespace mb
