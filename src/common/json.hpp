// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Tiny JSON value model + serializer used to build the per-result JSON objects
// and to read+concat an existing top-level array (append-mode), mirroring
// tests/lib/reports.py:write_json (json.dump(existing + new, indent=2)).
//
// We only need: object (ordered), array, string, number (double / int64),
// bool, null. A minimal parser handles re-reading our own previously written
// array so the append semantics match the Python writer.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mb {
namespace json {

class Value;
using ValuePtr = std::shared_ptr<Value>;

enum class Type { Null, Bool, Int, Double, String, Array, Object };

class Value {
public:
    Type type = Type::Null;
    bool b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;
    std::vector<ValuePtr> arr;
    // Ordered object: preserve insertion order to keep stable CSV/JSON layout.
    std::vector<std::pair<std::string, ValuePtr>> obj;

    static ValuePtr null() {
        auto v = std::make_shared<Value>();
        v->type = Type::Null;
        return v;
    }
    static ValuePtr boolean(bool x) {
        auto v = std::make_shared<Value>();
        v->type = Type::Bool;
        v->b = x;
        return v;
    }
    static ValuePtr integer(int64_t x) {
        auto v = std::make_shared<Value>();
        v->type = Type::Int;
        v->i = x;
        return v;
    }
    static ValuePtr number(double x) {
        auto v = std::make_shared<Value>();
        v->type = Type::Double;
        v->d = x;
        return v;
    }
    static ValuePtr str(const std::string& x) {
        auto v = std::make_shared<Value>();
        v->type = Type::String;
        v->s = x;
        return v;
    }
    static ValuePtr array() {
        auto v = std::make_shared<Value>();
        v->type = Type::Array;
        return v;
    }
    static ValuePtr object() {
        auto v = std::make_shared<Value>();
        v->type = Type::Object;
        return v;
    }

    void set(const std::string& key, ValuePtr val) {
        obj.emplace_back(key, std::move(val));
    }
    void push(ValuePtr val) { arr.push_back(std::move(val)); }
};

inline void escape_to(std::ostream& os, const std::string& in) {
    os << '"';
    for (unsigned char c : in) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hexd = "0123456789abcdef";
                    os << "\\u00" << hexd[(c >> 4) & 0xF] << hexd[c & 0xF];
                } else {
                    os << static_cast<char>(c);
                }
        }
    }
    os << '"';
}

// Format a double the way Python's json does for finite values: shortest
// round-trippable-ish representation. We use a high-precision generic format
// and trim. (Stats already pre-round most values; iteration raw fields use
// this.) NaN/Inf -> null to keep valid JSON.
inline void number_to(std::ostream& os, double x) {
    if (!(x == x) /*nan*/ || x == std::numeric_limits<double>::infinity() ||
        x == -std::numeric_limits<double>::infinity()) {
        os << "null";
        return;
    }
    std::ostringstream tmp;
    tmp.precision(12);
    tmp << x;
    os << tmp.str();
}

inline void serialize(std::ostream& os, const ValuePtr& v, int indent, int depth) {
    const std::string pad(static_cast<size_t>((depth + 1) * indent), ' ');
    const std::string pad_close(static_cast<size_t>(depth * indent), ' ');
    switch (v->type) {
        case Type::Null: os << "null"; break;
        case Type::Bool: os << (v->b ? "true" : "false"); break;
        case Type::Int: os << v->i; break;
        case Type::Double: number_to(os, v->d); break;
        case Type::String: escape_to(os, v->s); break;
        case Type::Array:
            if (v->arr.empty()) {
                os << "[]";
                break;
            }
            os << "[\n";
            for (size_t k = 0; k < v->arr.size(); ++k) {
                os << pad;
                serialize(os, v->arr[k], indent, depth + 1);
                if (k + 1 < v->arr.size()) os << ",";
                os << "\n";
            }
            os << pad_close << "]";
            break;
        case Type::Object:
            if (v->obj.empty()) {
                os << "{}";
                break;
            }
            os << "{\n";
            for (size_t k = 0; k < v->obj.size(); ++k) {
                os << pad;
                escape_to(os, v->obj[k].first);
                os << ": ";
                serialize(os, v->obj[k].second, indent, depth + 1);
                if (k + 1 < v->obj.size()) os << ",";
                os << "\n";
            }
            os << pad_close << "}";
            break;
    }
}

inline std::string dump(const ValuePtr& v, int indent = 2) {
    std::ostringstream os;
    serialize(os, v, indent, 0);
    return os.str();
}

// ---- Minimal parser (only robust enough to re-read our own output) ----------
class Parser {
public:
    explicit Parser(const std::string& text) : t_(text) {}

    ValuePtr parse() {
        skip_ws();
        ValuePtr v = parse_value();
        return v;
    }

private:
    // Stored BY VALUE (not a reference): callers construct Parser with temporaries,
    // e.g. `json::Parser parser(ss.str())` in load_extra_ov_config. A `const std::string&`
    // member would dangle after the constructor returns and corrupt parsing (manifested
    // as a spurious `stod` exception on the `-lc`/--load-config path).
    const std::string t_;
    size_t p_ = 0;

    void skip_ws() {
        while (p_ < t_.size() &&
               (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t')) {
            ++p_;
        }
    }
    char peek() { return p_ < t_.size() ? t_[p_] : '\0'; }

    ValuePtr parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value::str(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') {
            p_ += 4;
            return Value::null();
        }
        return parse_number();
    }

    ValuePtr parse_object() {
        auto v = Value::object();
        ++p_;  // {
        skip_ws();
        if (peek() == '}') {
            ++p_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            ++p_;  // :
            ValuePtr val = parse_value();
            v->set(key, val);
            skip_ws();
            if (peek() == ',') {
                ++p_;
                continue;
            }
            if (peek() == '}') {
                ++p_;
                break;
            }
            break;
        }
        return v;
    }

    ValuePtr parse_array() {
        auto v = Value::array();
        ++p_;  // [
        skip_ws();
        if (peek() == ']') {
            ++p_;
            return v;
        }
        while (true) {
            v->push(parse_value());
            skip_ws();
            if (peek() == ',') {
                ++p_;
                continue;
            }
            if (peek() == ']') {
                ++p_;
                break;
            }
            break;
        }
        return v;
    }

    std::string parse_string() {
        std::string out;
        ++p_;  // opening quote
        while (p_ < t_.size()) {
            char c = t_[p_++];
            if (c == '"') break;
            if (c == '\\' && p_ < t_.size()) {
                char e = t_[p_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // Copy through as-is (best effort; we re-emit escaped).
                        if (p_ + 4 <= t_.size()) {
                            out.push_back('?');
                            p_ += 4;
                        }
                        break;
                    }
                    default: out.push_back(e);
                }
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

    ValuePtr parse_bool() {
        if (t_.compare(p_, 4, "true") == 0) {
            p_ += 4;
            return Value::boolean(true);
        }
        p_ += 5;  // false
        return Value::boolean(false);
    }

    ValuePtr parse_number() {
        size_t start = p_;
        bool is_float = false;
        while (p_ < t_.size()) {
            char c = t_[p_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
                ++p_;
            } else if (c == '.' || c == 'e' || c == 'E') {
                is_float = true;
                ++p_;
            } else {
                break;
            }
        }
        std::string num = t_.substr(start, p_ - start);
        if (is_float) {
            return Value::number(std::stod(num));
        }
        try {
            return Value::integer(static_cast<int64_t>(std::stoll(num)));
        } catch (...) {
            return Value::number(std::stod(num));
        }
    }
};

// Append `new_objects` to the JSON array file at `path` (read existing array,
// concat, rewrite) — mirrors tests/lib/reports.py:write_json.
inline void append_array(const std::string& path, const std::vector<ValuePtr>& new_objects) {
    auto root = Value::array();
    {
        std::ifstream in(path, std::ios::binary);
        if (in.good()) {
            std::stringstream ss;
            ss << in.rdbuf();
            std::string content = ss.str();
            if (!content.empty()) {
                try {
                    Parser parser(content);
                    ValuePtr existing = parser.parse();
                    if (existing && existing->type == Type::Array) {
                        for (auto& e : existing->arr) root->push(e);
                    } else if (existing) {
                        root->push(existing);
                    }
                } catch (...) {
                    // Corrupt / unreadable existing file -> start fresh.
                    root = Value::array();
                }
            }
        }
    }
    bool had_existing = !root->arr.empty();
    for (auto& o : new_objects) root->push(o);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << dump(root, 2);
    out.flush();
    std::printf("JSON %s: %s\n", had_existing ? "appended to" : "written", path.c_str());
}

}  // namespace json
}  // namespace mb
