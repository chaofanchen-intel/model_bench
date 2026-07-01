// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/deep_profile.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>

#include "common/common.hpp"

namespace mb {

namespace {

namespace fs = std::filesystem;

// argmax over the last token's logits row: logits shape [1, seq, vocab].
int64_t argmax_last(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    const size_t seq = shape[shape.size() - 2];
    const size_t vocab = shape[shape.size() - 1];
    const float* p = logits.data<const float>() + (seq - 1) * vocab;
    int64_t best = 0;
    float best_v = p[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (p[i] > best_v) {
            best_v = p[i];
            best = static_cast<int64_t>(i);
        }
    }
    return best;
}

ov::Tensor i64_tensor(const ov::Shape& shape, const std::vector<int64_t>& data) {
    ov::Tensor t(ov::element::i64, shape);
    std::memcpy(t.data(), data.data(), data.size() * sizeof(int64_t));
    return t;
}

ov::Tensor i64_filled(const ov::Shape& shape, int64_t value) {
    ov::Tensor t(ov::element::i64, shape);
    int64_t* p = t.data<int64_t>();
    for (size_t i = 0; i < t.get_size(); ++i) p[i] = value;
    return t;
}

ov::Tensor beam_idx_tensor() {
    ov::Tensor t(ov::element::i32, ov::Shape{1});
    t.data<int32_t>()[0] = 0;
    return t;
}

// Extract input_ids from a tokenizer encode result as a flat int64 vector.
std::vector<int64_t> encode_ids(ov::genai::Tokenizer& tok, const std::string& text) {
    ov::genai::TokenizedInputs enc = tok.encode(text);
    const ov::Tensor& ids = enc.input_ids;
    std::vector<int64_t> out(ids.get_size());
    if (ids.get_element_type() == ov::element::i64) {
        std::memcpy(out.data(), ids.data(), out.size() * sizeof(int64_t));
    } else if (ids.get_element_type() == ov::element::i32) {
        const int32_t* p = ids.data<const int32_t>();
        for (size_t i = 0; i < out.size(); ++i) out[i] = p[i];
    }
    return out;
}

bool has_input(const ov::CompiledModel& cm, const std::string& name) {
    for (const auto& in : cm.inputs()) {
        if (in.get_any_name() == name) return true;
    }
    return false;
}

// Resolve the prefill token ids to profile. If cfg.prompt_text is set, use the
// REAL prompt (tokenized to its natural length) — this matches what the GenAI
// benchmark ran with -pf/--prompt. Otherwise synthesize a prompt of
// cfg.prompt_tokens by repeating a base sentence. `label` is for the log line.
std::vector<int64_t> resolve_prefill_ids(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer,
                                         const char* label) {
    if (cfg.prompt_text && !cfg.prompt_text->empty()) {
        std::vector<int64_t> ids = encode_ids(tokenizer, *cfg.prompt_text);
        std::printf("  [deep-profile] %s using REAL prompt: %zu tokens\n", label, ids.size());
        return ids;
    }
    std::string base =
        "The quick brown fox jumps over the lazy dog while the system processes data. ";
    std::string prompt = base;
    while (static_cast<int>(encode_ids(tokenizer, prompt).size()) < cfg.prompt_tokens) {
        prompt += base;
    }
    std::vector<int64_t> ids = encode_ids(tokenizer, prompt);
    if (static_cast<int>(ids.size()) > cfg.prompt_tokens) {
        ids.resize(cfg.prompt_tokens);
    }
    std::printf("  [deep-profile] %s using synthetic prompt: %zu tokens (target=%d)\n", label,
                ids.size(), cfg.prompt_tokens);
    return ids;
}

}  // namespace

bool deep_profile_llm(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer) {
    fs::path xml = fs::path(cfg.model_dir) / "openvino_model.xml";
    if (!fs::exists(xml)) {
        std::printf("  [deep-profile] LLM: openvino_model.xml not found; skipping.\n");
        return false;
    }
    ov::Core core;
    ov::CompiledModel cm;
    try {
        auto model = core.read_model(xml.string());
        cm = core.compile_model(model, cfg.device, cfg.plugin_cfg);
    } catch (const std::exception& e) {
        std::printf("  [deep-profile] LLM compile failed: %s\n", e.what());
        return false;
    }
    if (!has_input(cm, "input_ids") || !has_input(cm, "position_ids")) {
        std::printf("  [deep-profile] LLM: unexpected graph signature (no input_ids/position_ids); "
                    "skipping real profiling.\n");
        return false;
    }
    const bool has_beam = has_input(cm, "beam_idx");
    const bool has_attn = has_input(cm, "attention_mask");

    print_device_properties(cm, cfg.device);

    std::vector<int64_t> ids = resolve_prefill_ids(cfg, tokenizer, "LLM");
    const int N = static_cast<int>(ids.size());

    ov::InferRequest req = cm.create_infer_request();

    // One greedy generation (prefill + decode). `profile` selects whether to
    // accumulate into prefill/decode accumulators this run.
    auto generate = [&](std::vector<ProfRow>* prefill_acc, std::vector<ProfRow>* decode_acc) {
        req.reset_state();
        // ---- prefill ----
        req.set_tensor("input_ids", i64_tensor({1, (size_t)N}, ids));
        if (has_attn) req.set_tensor("attention_mask", i64_filled({1, (size_t)N}, 1));
        {
            std::vector<int64_t> pos(N);
            for (int i = 0; i < N; ++i) pos[i] = i;
            req.set_tensor("position_ids", i64_tensor({1, (size_t)N}, pos));
        }
        if (has_beam) req.set_tensor("beam_idx", beam_idx_tensor());
        req.infer();
        if (prefill_acc) collect_profiling(req, *prefill_acc);
        int64_t next = argmax_last(req.get_tensor("logits"));

        // ---- decode ----
        int cur = N;
        for (int step = 0; step < cfg.gen_tokens; ++step) {
            req.set_tensor("input_ids", i64_tensor({1, 1}, {next}));
            ++cur;
            if (has_attn) req.set_tensor("attention_mask", i64_filled({1, (size_t)cur}, 1));
            req.set_tensor("position_ids", i64_tensor({1, 1}, {(int64_t)(cur - 1)}));
            if (has_beam) req.set_tensor("beam_idx", beam_idx_tensor());
            req.infer();
            if (decode_acc) collect_profiling(req, *decode_acc);
            next = argmax_last(req.get_tensor("logits"));
        }
    };

    std::printf("  [deep-profile] LLM real greedy loop: prompt=%d tokens, decode=%d tokens, warmup=%d\n",
                N, cfg.gen_tokens, cfg.warmup);
    for (int w = 0; w < cfg.warmup; ++w) generate(nullptr, nullptr);

    std::vector<ProfRow> prefill_acc, decode_acc;
    generate(&prefill_acc, &decode_acc);

    print_profiling_report(prefill_acc, "LLM PREFILL (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, 1);
    print_profiling_report(decode_acc, "LLM DECODE (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, cfg.gen_tokens);
    if (cfg.profile_csv) {
        write_profiling_csv(*cfg.profile_csv, prefill_acc, path_name(cfg.model_dir), cfg.device,
                            cfg.precision, "llm_prefill", 1);
        write_profiling_csv(*cfg.profile_csv, decode_acc, path_name(cfg.model_dir), cfg.device,
                            cfg.precision, "llm_decode", cfg.gen_tokens);
    }
    return true;
}

bool deep_profile_vlm(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer) {
    fs::path te_xml = fs::path(cfg.model_dir) / "openvino_text_embeddings_model.xml";
    fs::path lm_xml = fs::path(cfg.model_dir) / "openvino_language_model.xml";
    if (!fs::exists(te_xml) || !fs::exists(lm_xml)) {
        std::printf("  [deep-profile] VLM: text_embeddings/language_model IRs not found; skipping.\n");
        return false;
    }
    ov::Core core;
    ov::CompiledModel te_cm, lm_cm;
    try {
        te_cm = core.compile_model(core.read_model(te_xml.string()), cfg.device, cfg.plugin_cfg);
        lm_cm = core.compile_model(core.read_model(lm_xml.string()), cfg.device, cfg.plugin_cfg);
    } catch (const std::exception& e) {
        std::printf("  [deep-profile] VLM compile failed: %s\n", e.what());
        return false;
    }
    if (!has_input(lm_cm, "inputs_embeds") || !has_input(lm_cm, "position_ids")) {
        std::printf("  [deep-profile] VLM: unexpected language-model signature; skipping.\n");
        return false;
    }
    const bool has_beam = has_input(lm_cm, "beam_idx");
    const bool has_attn = has_input(lm_cm, "attention_mask");

    print_device_properties(lm_cm, cfg.device);

    // Detect position_ids rank for the LM: Qwen3.5 uses mRoPE [4, batch, seq].
    int pos_rank = 2;
    size_t pos_dim0 = 1;
    for (const auto& in : lm_cm.inputs()) {
        if (in.get_any_name() == "position_ids") {
            auto pshape = in.get_partial_shape();
            pos_rank = static_cast<int>(pshape.rank().get_length());
            if (pos_rank >= 3 && pshape[0].is_static()) pos_dim0 = pshape[0].get_length();
            break;
        }
    }

    std::vector<int64_t> ids = resolve_prefill_ids(cfg, tokenizer, "VLM");
    const int N = static_cast<int>(ids.size());

    ov::InferRequest te_req = te_cm.create_infer_request();
    ov::InferRequest lm_req = lm_cm.create_infer_request();

    const std::string te_in = has_input(te_cm, "input") ? "input" : te_cm.inputs()[0].get_any_name();

    // text_embeddings(input_ids[1,n]) -> inputs_embeds[1,n,hidden]
    auto embed = [&](const std::vector<int64_t>& token_ids, std::vector<ProfRow>* acc) -> ov::Tensor {
        te_req.set_tensor(te_in, i64_tensor({1, token_ids.size()}, token_ids));
        te_req.infer();
        if (acc) collect_profiling(te_req, *acc);
        return te_req.get_output_tensor(0);
    };

    // position_ids builder: rank-3 mRoPE replicates arange across dim0.
    auto make_pos = [&](int start, int len) -> ov::Tensor {
        std::vector<int64_t> seq(len);
        for (int i = 0; i < len; ++i) seq[i] = start + i;
        if (pos_rank >= 3) {
            ov::Shape sh{pos_dim0, 1, (size_t)len};
            ov::Tensor t(ov::element::i64, sh);
            int64_t* p = t.data<int64_t>();
            for (size_t d = 0; d < pos_dim0; ++d)
                for (int i = 0; i < len; ++i) p[d * len + i] = seq[i];
            return t;
        }
        return i64_tensor({1, (size_t)len}, seq);
    };

    auto generate = [&](std::vector<ProfRow>* te_acc, std::vector<ProfRow>* prefill_acc,
                        std::vector<ProfRow>* decode_acc) {
        lm_req.reset_state();
        // ---- prefill ----
        ov::Tensor embeds = embed(ids, te_acc);
        lm_req.set_tensor("inputs_embeds", embeds);
        if (has_attn) lm_req.set_tensor("attention_mask", i64_filled({1, (size_t)N}, 1));
        lm_req.set_tensor("position_ids", make_pos(0, N));
        if (has_beam) lm_req.set_tensor("beam_idx", beam_idx_tensor());
        lm_req.infer();
        if (prefill_acc) collect_profiling(lm_req, *prefill_acc);
        int64_t next = argmax_last(lm_req.get_tensor("logits"));

        // ---- decode ----
        int cur = N;
        for (int step = 0; step < cfg.gen_tokens; ++step) {
            ov::Tensor emb1 = embed({next}, nullptr);  // count text-embeds in prefill phase only
            ++cur;
            lm_req.set_tensor("inputs_embeds", emb1);
            if (has_attn) lm_req.set_tensor("attention_mask", i64_filled({1, (size_t)cur}, 1));
            lm_req.set_tensor("position_ids", make_pos(cur - 1, 1));
            if (has_beam) lm_req.set_tensor("beam_idx", beam_idx_tensor());
            lm_req.infer();
            if (decode_acc) collect_profiling(lm_req, *decode_acc);
            next = argmax_last(lm_req.get_tensor("logits"));
        }
    };

    std::printf("  [deep-profile] VLM (text-only) real greedy loop: prompt=%d tokens, decode=%d, warmup=%d\n",
                N, cfg.gen_tokens, cfg.warmup);
    for (int w = 0; w < cfg.warmup; ++w) generate(nullptr, nullptr, nullptr);

    std::vector<ProfRow> te_acc, prefill_acc, decode_acc;
    generate(&te_acc, &prefill_acc, &decode_acc);

    print_profiling_report(te_acc, "VLM TEXT-EMBEDDINGS (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, 1);
    print_profiling_report(prefill_acc, "VLM LM PREFILL (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, 1);
    print_profiling_report(decode_acc, "VLM LM DECODE (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, cfg.gen_tokens);
    if (cfg.profile_csv) {
        write_profiling_csv(*cfg.profile_csv, te_acc, path_name(cfg.model_dir), cfg.device,
                            cfg.precision, "vlm_text_embeddings", 1);
        write_profiling_csv(*cfg.profile_csv, prefill_acc, path_name(cfg.model_dir), cfg.device,
                            cfg.precision, "vlm_lm_prefill", 1);
        write_profiling_csv(*cfg.profile_csv, decode_acc, path_name(cfg.model_dir), cfg.device,
                            cfg.precision, "vlm_lm_decode", cfg.gen_tokens);
    }
    return true;
}

bool deep_profile_encoder(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer,
                          const std::string& stage) {
    fs::path xml = fs::path(cfg.model_dir) / "openvino_model.xml";
    if (!fs::exists(xml)) {
        std::printf("  [deep-profile] encoder: openvino_model.xml not found; skipping.\n");
        return false;
    }
    ov::Core core;
    ov::CompiledModel cm;
    try {
        cm = core.compile_model(core.read_model(xml.string()), cfg.device, cfg.plugin_cfg);
    } catch (const std::exception& e) {
        std::printf("  [deep-profile] encoder compile failed: %s\n", e.what());
        return false;
    }
    print_device_properties(cm, cfg.device);

    std::set<std::string> names;
    for (const auto& in : cm.inputs()) names.insert(in.get_any_name());
    const bool has_attn = names.count("attention_mask") > 0;
    const bool has_pos = names.count("position_ids") > 0;
    const bool has_tti = names.count("token_type_ids") > 0;

    // Use the real text(s); fall back to a representative sentence.
    std::string text = cfg.encoder_texts.empty()
                           ? std::string("What is the capital of France and what is it famous for?")
                           : cfg.encoder_texts.front();

    ov::AnyMap enc_params;
    enc_params["add_special_tokens"] = true;
    if (cfg.encoder_max_length > 0) {
        enc_params["max_length"] = (int64_t)cfg.encoder_max_length;
        enc_params["pad_to_max_length"] = true;
    }
    ov::genai::TokenizedInputs enc = tokenizer.encode(text, enc_params);
    const ov::Tensor ids = enc.input_ids;
    const int seq = static_cast<int>(ids.get_shape().back());

    ov::InferRequest req = cm.create_infer_request();
    auto set_inputs = [&]() {
        req.set_tensor("input_ids", ids);
        if (has_attn) {
            if (enc.attention_mask && enc.attention_mask.get_size() == ids.get_size())
                req.set_tensor("attention_mask", enc.attention_mask);
            else
                req.set_tensor("attention_mask", i64_filled(ids.get_shape(), 1));
        }
        if (has_pos) {
            std::vector<int64_t> pos(seq);
            for (int i = 0; i < seq; ++i) pos[i] = i;
            req.set_tensor("position_ids", i64_tensor(ids.get_shape(), pos));
        }
        if (has_tti) {
            if (enc.token_type_ids && enc.token_type_ids->get_size() == ids.get_size())
                req.set_tensor("token_type_ids", *enc.token_type_ids);
            else
                req.set_tensor("token_type_ids", i64_filled(ids.get_shape(), 0));
        }
    };

    std::printf("  [deep-profile] encoder real input: seq=%d tokens, iters=%d (warmup=%d)\n", seq,
                cfg.gen_tokens, cfg.warmup);
    for (int w = 0; w < cfg.warmup; ++w) {
        set_inputs();
        req.infer();
    }
    std::vector<ProfRow> acc;
    const int iters = std::max(1, cfg.gen_tokens);
    for (int i = 0; i < iters; ++i) {
        set_inputs();
        req.infer();
        collect_profiling(req, acc);
    }
    print_profiling_report(acc, stage + " REAL INPUT (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                           cfg.topk, iters);
    if (cfg.profile_csv) {
        write_profiling_csv(*cfg.profile_csv, acc, path_name(cfg.model_dir), cfg.device, cfg.precision,
                            stage, iters);
    }
    return true;
}

}  // namespace mb
