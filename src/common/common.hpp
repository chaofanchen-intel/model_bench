// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Small shared helpers: device upper-casing, precision detection from model
// dir name, OpenVINO / GenAI version strings, --load-config JSON parsing into
// an ov::AnyMap of string values, and a header-aware append-mode CSV writer
// (mirrors tests/lib/reports.py _write_csv_append: rewrites the whole file when
// the header differs so a changing per-core column count never injects a header
// row mid-file).
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/genai/version.hpp"
#include "openvino/openvino.hpp"
#include "common/json.hpp"

namespace mb {

inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Base name of a (possibly trailing-slash) path, matching pathlib Path.name.
inline std::string path_name(const std::string& p) {
    std::filesystem::path path(p);
    if (path.has_filename()) {
        return path.filename().string();
    }
    return path.parent_path().filename().string();
}

// detect_precision: substring scan of dir name for int4/int8/fp16/fp32/bf16.
inline std::string detect_precision(const std::string& model_dir) {
    const std::string name = to_lower(path_name(model_dir));
    for (const char* tag : {"int4", "int8", "fp16", "fp32", "bf16"}) {
        if (name.find(tag) != std::string::npos) {
            return to_upper(tag);
        }
    }
    return "";
}

// Matches Python ov.get_version(): the single-line buildNumber, not the
// multi-line "OpenVINO Runtime\n Version...\n Build..." stream form.
inline std::string openvino_version_string() {
    const ov::Version v = ov::get_openvino_version();
    return v.buildNumber ? std::string(v.buildNumber) : std::string();
}

inline std::string openvino_genai_version_string() {
    const ov::Version v = ov::genai::get_version();
    return v.buildNumber ? std::string(v.buildNumber) : std::string();
}

// Parse a JSON object of plugin properties into a string->string map.
// Values are stringified (numbers/bools -> their textual form), matching
// tests/lib/cli.py:load_extra_ov_config.
inline std::map<std::string, std::string> load_extra_ov_config(const std::string& path) {
    std::map<std::string, std::string> out;
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        throw std::runtime_error("could not open load-config file: " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    json::Parser parser(ss.str());
    json::ValuePtr v = parser.parse();
    if (!v || v->type != json::Type::Object) {
        throw std::runtime_error("load-config must be a JSON object: " + path);
    }
    for (auto& kv : v->obj) {
        const json::ValuePtr& val = kv.second;
        switch (val->type) {
            case json::Type::String: out[kv.first] = val->s; break;
            case json::Type::Bool: out[kv.first] = val->b ? "true" : "false"; break;
            case json::Type::Int: out[kv.first] = std::to_string(val->i); break;
            case json::Type::Double: {
                std::ostringstream tmp;
                tmp << val->d;
                out[kv.first] = tmp.str();
                break;
            }
            default: out[kv.first] = ""; break;
        }
    }
    return out;
}

// ISO-8601-ish local timestamp (matches datetime.datetime.now().isoformat()
// closely enough for provenance).
std::string now_iso8601();

// Header-aware append-mode CSV writer.
inline void write_csv_append(const std::string& path,
                             const std::vector<std::string>& header,
                             const std::vector<std::vector<std::string>>& rows) {
    std::filesystem::path fp(path);
    if (fp.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(fp.parent_path(), ec);
    }

    bool file_exists = false;
    {
        std::ifstream probe(path, std::ios::binary | std::ios::ate);
        if (probe.good() && probe.tellg() > 0) {
            file_exists = true;
        }
    }

    bool write_header = !file_exists;
    if (file_exists) {
        std::ifstream in(path, std::ios::binary);
        std::string first_line;
        std::getline(in, first_line);
        if (!first_line.empty() && first_line.back() == '\r') {
            first_line.pop_back();
        }
        // Rebuild the expected header line and compare.
        std::ostringstream hl;
        for (size_t i = 0; i < header.size(); ++i) {
            if (i) hl << ",";
            hl << header[i];
        }
        write_header = (first_line != hl.str());
    }

    const bool append = file_exists && !write_header;
    std::ofstream out(path, append ? (std::ios::binary | std::ios::app)
                                   : (std::ios::binary | std::ios::trunc));
    // Match Python csv.writer line terminator (\r\n) so files are byte-compatible.
    auto write_cells = [&out](const std::vector<std::string>& cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            if (i) out << ",";
            const std::string& c = cells[i];
            // Quote if needed (commas, quotes, newlines).
            if (c.find_first_of(",\"\n\r") != std::string::npos) {
                out << '"';
                for (char ch : c) {
                    if (ch == '"') out << "\"\"";
                    else out << ch;
                }
                out << '"';
            } else {
                out << c;
            }
        }
        out << "\r\n";
    };
    if (write_header) {
        write_cells(header);
    }
    for (const auto& row : rows) {
        write_cells(row);
    }
    out.flush();
    std::printf("CSV  %s: %s\n", append ? "appended to" : "written", path.c_str());
}

}  // namespace mb
