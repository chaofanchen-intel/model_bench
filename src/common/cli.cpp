// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace mb {

namespace {

struct TokenStream {
    std::vector<std::string> toks;
    size_t pos = 0;
    bool has_more() const { return pos < toks.size(); }
    const std::string& peek() const { return toks[pos]; }
    std::string next() { return toks[pos++]; }
    bool next_is_value() const {
        if (pos >= toks.size()) return false;
        const std::string& t = toks[pos];
        // A value does not begin with '-' unless it's a bare negative number.
        if (t.size() >= 1 && t[0] == '-') {
            if (t.size() >= 2 && (std::isdigit(static_cast<unsigned char>(t[1])) || t[1] == '.')) {
                return true;  // negative number
            }
            return false;
        }
        return true;
    }
};

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

std::string take_value(TokenStream& ts, const std::string& opt) {
    if (!ts.next_is_value()) {
        fail("option " + opt + " requires a value");
    }
    return ts.next();
}

std::vector<std::string> take_values(TokenStream& ts, const std::string& opt) {
    std::vector<std::string> out;
    while (ts.next_is_value()) {
        out.push_back(ts.next());
    }
    if (out.empty()) {
        fail("option " + opt + " requires at least one value");
    }
    return out;
}

int to_int(const std::string& s, const std::string& opt) {
    try {
        return std::stoi(s);
    } catch (...) {
        fail("option " + opt + " expects an integer, got '" + s + "'");
    }
}

double to_double(const std::string& s, const std::string& opt) {
    try {
        return std::stod(s);
    } catch (...) {
        fail("option " + opt + " expects a float, got '" + s + "'");
    }
}

}  // namespace

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        a.help = true;
        a.task = "";
        return a;
    }
    a.task = argv[1];
    if (a.task == "-h" || a.task == "--help") {
        a.help = true;
        a.task = "";
        return a;
    }

    // VLM defaults to permutation ON.
    if (a.task == "vlm") {
        a.disable_prompt_permutation = false;
        a.permutation_default_off = false;
    }
    if (a.task == "embedding") a.pooling_type = "last_token";

    // Build token list, splitting --opt=value.
    TokenStream ts;
    for (int i = 2; i < argc; ++i) {
        std::string t = argv[i];
        if (t.rfind("--", 0) == 0) {
            auto eq = t.find('=');
            if (eq != std::string::npos) {
                ts.toks.push_back(t.substr(0, eq));
                ts.toks.push_back(t.substr(eq + 1));
                continue;
            }
        }
        ts.toks.push_back(t);
    }

    auto eq_any = [](const std::string& o, std::initializer_list<const char*> names) {
        for (auto n : names)
            if (o == n) return true;
        return false;
    };

    while (ts.has_more()) {
        std::string o = ts.next();

        if (eq_any(o, {"-h", "--help"})) {
            a.help = true;
        } else if (eq_any(o, {"-m", "--model-dir", "--model_dir"})) {
            a.model_dir = take_values(ts, o);
        } else if (eq_any(o, {"-d", "--device"})) {
            a.device = take_values(ts, o);
        } else if (eq_any(o, {"-p", "--prompt", "--query"})) {
            a.prompt = take_value(ts, o);
        } else if (eq_any(o, {"-pf", "--prompt-file", "--prompt_file"})) {
            a.prompt_file = take_values(ts, o);
        } else if (eq_any(o, {"-pi", "--prompt-index", "--prompt_index"})) {
            for (auto& v : take_values(ts, o)) a.prompt_index.push_back(to_int(v, o));
        } else if (eq_any(o, {"--input-tokens", "--input_tokens"})) {
            a.input_tokens = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--input-base-text", "--input_base_text"})) {
            a.input_base_text = take_value(ts, o);
        } else if (eq_any(o, {"-ic", "--infer-count", "--infer_count"})) {
            a.infer_count = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--exact-output-tokens", "--exact_output_tokens"})) {
            a.exact_output_tokens = true;
        } else if (eq_any(o, {"-n", "--num-iters", "--num_iters"})) {
            a.num_iters = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"-bs", "--batch-size", "--batch_size"})) {
            a.batch_size = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--num-beams", "--num_beams"})) {
            a.num_beams = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--apply-chat-template", "--apply_chat_template"})) {
            a.apply_chat_template = true;
        } else if (eq_any(o, {"--disable-prompt-permutation", "--disable_prompt_permutation"})) {
            a.disable_prompt_permutation = true;
        } else if (eq_any(o, {"--enable-prompt-permutation", "--enable_prompt_permutation"})) {
            a.disable_prompt_permutation = false;
        } else if (eq_any(o, {"-s", "--seed"})) {
            a.seed = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"-lc", "--load-config", "--load_config"})) {
            a.load_config = take_value(ts, o);
        } else if (eq_any(o, {"--cache-dir", "--cache_dir"})) {
            a.cache_dir = take_value(ts, o);
            a.cache_dir_set = true;
        } else if (eq_any(o, {"--num-streams", "--num_streams"})) {
            a.num_streams = take_value(ts, o);
        } else if (eq_any(o, {"--cooldown-between-runs", "--cooldown_between_runs"})) {
            a.cooldown_between_runs = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--image"})) {
            a.image = take_value(ts, o);
        } else if (eq_any(o, {"--input-lengths", "--input_lengths"})) {
            a.input_lengths.clear();
            for (auto& v : take_values(ts, o)) a.input_lengths.push_back(to_int(v, o));
        } else if (eq_any(o, {"--compression-rates", "--compression_rates"})) {
            a.compression_rates.clear();
            for (auto& v : take_values(ts, o)) a.compression_rates.push_back(to_double(v, o));
        } else if (eq_any(o, {"--warmup"})) {
            a.warmup = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--iters"})) {
            a.lingua_iters = to_int(take_value(ts, o), o);
            a.lingua_iters_set = true;
        } else if (eq_any(o, {"--mode"})) {
            a.mode = take_value(ts, o);
        } else if (eq_any(o, {"--pooling-type", "--pooling_type"})) {
            a.pooling_type = take_value(ts, o);
        } else if (eq_any(o, {"--normalize"})) {
            a.normalize = true;
        } else if (eq_any(o, {"--no-normalize", "--no_normalize"})) {
            a.normalize = false;
        } else if (eq_any(o, {"--max-length", "--max_length"})) {
            a.max_length = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--padding-side", "--padding_side"})) {
            a.padding_side = take_value(ts, o);
        } else if (eq_any(o, {"--pad-to-max-length", "--pad_to_max_length"})) {
            a.pad_to_max_length = true;
        } else if (eq_any(o, {"--query-instruction", "--query_instruction"})) {
            a.query_instruction = take_value(ts, o);
        } else if (eq_any(o, {"--embed-instruction", "--embed_instruction"})) {
            a.embed_instruction = take_value(ts, o);
        } else if (eq_any(o, {"--texts"})) {
            a.texts = take_values(ts, o);
        } else if (eq_any(o, {"--texts-file", "--texts_file"})) {
            a.texts_file = take_values(ts, o);
        } else if (eq_any(o, {"--num-documents", "--num_documents"})) {
            a.num_documents = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--top-n", "--top_n"})) {
            a.top_n = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--monitor"})) {
            a.no_monitor = false;
        } else if (eq_any(o, {"--no-monitor", "--no_monitor"})) {
            a.no_monitor = true;
        } else if (eq_any(o, {"--memory-monitor", "--memory_monitor"})) {
            a.memory_monitor = true;
        } else if (eq_any(o, {"--no-memory-monitor", "--no_memory_monitor"})) {
            a.memory_monitor = false;
        } else if (eq_any(o, {"--memory-monitor-interval", "--memory_monitor_interval"})) {
            a.memory_monitor_interval = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--cpu-monitor-interval", "--cpu_monitor_interval"})) {
            a.cpu_monitor_interval = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--gpu-monitor-interval", "--gpu_monitor_interval"})) {
            a.gpu_monitor_interval = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--npu-monitor-interval", "--npu_monitor_interval"})) {
            a.npu_monitor_interval = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--gpu-command", "--gpu_command"})) {
            a.gpu_command = take_value(ts, o);
        } else if (eq_any(o, {"--profile"})) {
            a.profile = true;
        } else if (eq_any(o, {"--profile-topk", "--profile_topk"})) {
            a.profile_topk = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--profile-warmup", "--profile_warmup"})) {
            a.profile_warmup = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--profile-iters", "--profile_iters"})) {
            a.profile_iters = to_int(take_value(ts, o), o);
        } else if (eq_any(o, {"--profile-csv", "--profile_csv"})) {
            a.profile_csv = take_value(ts, o);
        } else if (eq_any(o, {"--profile-graph", "--profile_graph"})) {
            a.profile_graph = take_value(ts, o);
        } else if (eq_any(o, {"-r", "--csv-out", "--csv_out"})) {
            a.csv_out = take_value(ts, o);
        } else if (eq_any(o, {"-rj", "--json-out", "--json_out"})) {
            a.json_out = take_value(ts, o);
        } else {
            fail("unknown option: " + o);
        }
    }

    // Lingua: default iters 50 unless overridden.
    if (a.task == "lingua" && !a.lingua_iters_set) {
        a.lingua_iters = 50;
    }

    // Fill default output paths per task (model_bench/results/<task>.{csv,json}).
    auto set_default_io = [&](const char* base) {
        if (a.csv_out.empty()) a.csv_out = std::string("results/") + base + ".csv";
        if (a.json_out.empty()) a.json_out = std::string("results/") + base + ".json";
    };
    if (a.task == "llm") set_default_io("qwen3");
    else if (a.task == "vlm") set_default_io("qwen3_5");
    else if (a.task == "lingua") set_default_io("lingua");
    else if (a.task == "embedding") set_default_io("embedding");
    else if (a.task == "reranker") set_default_io("reranker");

    return a;
}

void print_usage(const std::string& task) {
    if (task.empty()) {
        std::printf(
            "model_bench — OpenVINO GenAI C++ benchmark suite\n\n"
            "Usage: model_bench <task> [options]\n\n"
            "Tasks:\n"
            "  llm        Qwen3 text-generation (ov::genai::LLMPipeline)\n"
            "  vlm        Qwen3.5 vision-language (ov::genai::VLMPipeline)\n"
            "  lingua     LLMLingua-2 token classification (raw ov::Core)\n"
            "  embedding  Qwen3-Embedding (ov::genai::TextEmbeddingPipeline)\n"
            "  reranker   Qwen3-Reranker (ov::genai::TextRerankPipeline)\n\n"
            "Run 'model_bench <task> --help' for task-specific options.\n"
            "Add --profile to any task for per-operation ov::ProfilingInfo hotspot analysis.\n");
        return;
    }
    std::printf("model_bench %s [options]\n\n", task.c_str());
    std::printf(
        "Common options:\n"
        "  -m, --model-dir DIR [DIR...]   model IR directory(ies) (required)\n"
        "  -d, --device DEV [DEV...]      CPU GPU NPU AUTO ... (default CPU)\n"
        "  -n, --num-iters N              measured iterations (+1 warmup) (default 5)\n"
        "  -lc, --load-config FILE        JSON of extra OV plugin properties\n"
        "      --cache-dir DIR            OV CACHE_DIR (default empty)\n"
        "      --num-streams N            OV NUM_STREAMS\n"
        "      --cooldown-between-runs S  sleep between (model,device) pairs (default 5)\n"
        "      --monitor / --no-monitor   HW utilization monitoring (default off)\n"
        "      --memory-monitor           process/system memory sampling\n"
        "  -r, --csv-out FILE             CSV report (append-mode)\n"
        "  -rj, --json-out FILE           JSON report (append-mode)\n"
        "      --profile                  per-op ProfilingInfo hotspot report\n"
        "      --profile-csv FILE         write profiling rows to CSV\n"
        "      --profile-graph WHICH      llm only: pa (default) | sdpa | both\n\n");
    if (task == "llm" || task == "vlm") {
        std::printf(
            "Generation options:\n"
            "  -p, --prompt TEXT              single inline prompt\n"
            "  -pf, --prompt-file F [F...]    JSONL files {\"prompt\": ...}\n"
            "  -pi, --prompt-index I [I...]   subset of prompt indices\n"
            "      --input-tokens N           synthesize a prompt near N tokens (default 1000; <=0 or --image disables)\n"
            "      --input-base-text FILE     source text for --input-tokens\n"
            "  -ic, --infer-count N           max_new_tokens (default 512)\n"
            "      --exact-output-tokens      pin min_new_tokens=max_new_tokens\n"
            "      --num-beams N              beam count (default 1)\n"
            "      --apply-chat-template      wrap prompt in chat template\n");
        if (task == "llm") {
            std::printf("      --disable-prompt-permutation   disable per-iter input mutation (default ON)\n");
        } else {
            std::printf("      --enable-prompt-permutation    enable per-iter prompt mutation (default OFF)\n"
                        "      --image FILE               multimodal mode\n");
        }
    } else if (task == "lingua") {
        std::printf(
            "Lingua options:\n"
            "      --input-lengths N [N...]   input token lengths (default 64 128 256 512)\n"
            "      --compression-rates R [R...] keep ratios (default 0.3 0.5 0.7)\n"
            "      --warmup N                 warmup iterations (default 5)\n"
            "      --iters N                  measured iterations (default 50)\n");
    } else if (task == "embedding") {
        std::printf(
            "Embedding options:\n"
            "      --mode documents|query|both (default documents)\n"
            "      --pooling-type cls|mean|last_token (default last_token)\n"
            "      --normalize / --no-normalize (default normalize)\n"
            "      --max-length N             max tokens to model\n"
            "      --padding-side left|right\n"
            "      --pad-to-max-length\n"
            "      --query-instruction TEXT\n"
            "      --embed-instruction TEXT\n"
            "  -bs, --batch-size N            texts per embed_documents() call\n"
            "  -p/-pf/-pi/--input-tokens      input text selection\n");
    } else if (task == "reranker") {
        std::printf(
            "Reranker options:\n"
            "  -p, --prompt/--query TEXT      query string\n"
            "  -pf, --prompt-file F [F...]    JSONL query files\n"
            "      --texts T [T...]           candidate documents\n"
            "      --texts-file F [F...]      JSONL document files {\"text\": ...}\n"
            "      --num-documents N          replicate/trim to N candidates\n"
            "      --top-n N                  results to return (default 3)\n"
            "      --max-length N\n");
    }
}

}  // namespace mb
