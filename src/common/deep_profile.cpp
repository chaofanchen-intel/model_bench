// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
#include "common/deep_profile.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <set>

#include "common/common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

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

// ---------------------------------------------------------------------------
// PagedAttention-graph profiling
// ---------------------------------------------------------------------------
//
// ov::genai::LLMPipeline defaults to ATTENTION_BACKEND=PA, so deep_profile_llm()
// above profiles a graph the benchmark never runs. This path rebuilds what
// ContinuousBatchingPipeline builds:
//
//   ov::pass::SDPAToPagedAttention(false, false, /*score_aggregation*/ true, ...)
//   + gather-before-lm_head (only the sampled rows reach the vocab MatMul)
//
// then compiles it, reads the KV-cache block layout back out of the compiled
// model exactly like GenAI's KVCacheManager does, allocates the blocks, and
// drives a single-sequence prefill plus `gen_tokens` decode steps.

namespace {

// Mirrors ov::genai::utils::find_llm_matmul: locate the vocab-projection MatMul
// so a Gather can be inserted in front of it. For a PA graph tokens live in the
// batch dimension, so the gather axis is 0.
std::shared_ptr<ov::Node> find_vocab_matmul(const std::shared_ptr<ov::Model>& model) {
    auto last = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
    if (auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(last)) return mm;
    if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last))
        return ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
    return nullptr;
}

// Insert the "sampled_tokens_indices" Gather in front of the vocab MatMul, so
// prefill projects only the rows that are actually sampled instead of the whole
// prompt. Returns true when the parameter was added.
bool apply_gather_before_vocab_matmul(const std::shared_ptr<ov::Model>& model) {
    auto matmul = find_vocab_matmul(model);
    if (!matmul) return false;
    const auto rank = matmul->input(0).get_partial_shape().rank();
    if (rank.is_dynamic()) return false;
    // PA moves all tokens into the batch dimension -> gather along axis 0.
    const int64_t axis_val = (rank.get_length() == 3) ? 1 : 0;
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    indices->set_friendly_name("sampled_tokens_indices");
    indices->output(0).get_tensor().set_names({"sampled_tokens_indices"});
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1},
                                                       std::vector<int64_t>{axis_val});
    auto gather = std::make_shared<ov::op::v8::Gather>(matmul->input_value(0), indices, axis);
    matmul->input(0).replace_source_output(gather);
    model->add_parameters({indices});
    model->validate_nodes_and_infer_types();
    return true;
}

ov::Tensor i32_tensor(const ov::Shape& shape, const std::vector<int32_t>& data) {
    ov::Tensor t(ov::element::i32, shape);
    std::memcpy(t.data(), data.data(), data.size() * sizeof(int32_t));
    return t;
}

// argmax over a [rows, vocab] logits tensor's last row.
int64_t argmax_last_row(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    const size_t vocab = shape.back();
    size_t rows = logits.get_size() / vocab;
    const float* p = logits.data<const float>() + (rows - 1) * vocab;
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

}  // namespace

bool deep_profile_llm_pa(const DeepProfileConfig& cfg, ov::genai::Tokenizer& tokenizer) {
    fs::path xml = fs::path(cfg.model_dir) / "openvino_model.xml";
    if (!fs::exists(xml)) {
        std::printf("  [deep-profile/PA] openvino_model.xml not found; skipping.\n");
        return false;
    }

    ov::Core core;
    std::shared_ptr<ov::Model> model;
    bool has_sampled_idx = false;
    try {
        model = core.read_model(xml.string());
        // allow_score_aggregation=true matches ContinuousBatchingImpl.
        ov::pass::SDPAToPagedAttention(false, false, true, false, false, false).run_on_model(model);
        model->validate_nodes_and_infer_types();
        has_sampled_idx = apply_gather_before_vocab_matmul(model);
    } catch (const std::exception& e) {
        std::printf("  [deep-profile/PA] SDPAToPagedAttention failed: %s\n", e.what());
        return false;
    }

    ov::CompiledModel cm;
    try {
        cm = core.compile_model(model, cfg.device, cfg.plugin_cfg);
    } catch (const std::exception& e) {
        std::printf("  [deep-profile/PA] compile failed: %s\n", e.what());
        return false;
    }

    // Read the KV-cache block layout back out of the COMPILED model, the way
    // GenAI's KVCacheManager does — the plugin, not the caller, picks the
    // block layout and cache precision.
    // The GPU plugin picks the cache precision (u8 by default — the KV cache is
    // itself dynamically quantized) and a per-block layout that differs between
    // K and V, so the only safe thing to do is copy dims 1..3 verbatim and vary
    // dim 0 (the block count), exactly like GenAI's KVCacheManager.
    struct CacheInput {
        std::string name;
        ov::element::Type precision;
        ov::Shape per_block;  // dims 1..3, copied from the compiled model
    };
    std::vector<CacheInput> caches;
    for (const auto& in : cm.inputs()) {
        std::string name;
        for (const auto& n : in.get_names()) {
            if (n.rfind("key_cache.", 0) == 0 || n.rfind("value_cache.", 0) == 0) {
                name = n;
                break;
            }
        }
        if (name.empty()) continue;
        auto ps = in.get_partial_shape();
        if (ps.rank().is_dynamic() || ps.rank().get_length() != 4 || ps[1].is_dynamic() ||
            ps[2].is_dynamic() || ps[3].is_dynamic()) {
            std::printf("  [deep-profile/PA] %s has an unusable cache shape %s; skipping.\n",
                        name.c_str(), ps.to_string().c_str());
            return false;
        }
        if (caches.size() < 2)
            std::printf("  [deep-profile/PA] cache input %s : %s %s\n", name.c_str(),
                        in.get_element_type().get_type_name().c_str(), ps.to_string().c_str());
        caches.push_back({name, in.get_element_type(),
                          ov::Shape{static_cast<size_t>(ps[1].get_length()),
                                    static_cast<size_t>(ps[2].get_length()),
                                    static_cast<size_t>(ps[3].get_length())}});
    }
    if (caches.empty()) {
        std::printf("  [deep-profile/PA] no key_cache./value_cache. inputs found; skipping.\n");
        return false;
    }
    // Block size is a device convention, not something derivable from the cache
    // layout: GenAI uses 32 on CPU and 16 on GPU (256 only for XAttention).
    const bool is_gpu = cfg.device.find("GPU") != std::string::npos;
    size_t block_size = is_gpu ? 16 : 32;
    for (const auto& c : caches) {
        if (c.name.rfind("value_cache.", 0) == 0 && is_gpu && c.per_block[1] == 256) {
            block_size = 256;
            break;
        }
    }

    print_device_properties(cm, cfg.device);

    std::vector<int64_t> ids = resolve_prefill_ids(cfg, tokenizer, "LLM/PA");
    const size_t N = ids.size();
    const size_t total_len = N + static_cast<size_t>(cfg.gen_tokens) + 1;
    const size_t num_blocks = (total_len + block_size - 1) / block_size + 1;

    std::printf("  [deep-profile/PA] PagedAttention graph: %zu cache inputs, block_size=%zu, "
                "%zu blocks allocated, gather_before_lm_head=%s\n",
                caches.size(), block_size, num_blocks, has_sampled_idx ? "yes" : "no");

    ov::InferRequest req = cm.create_infer_request();

    // KV cache blocks are persistent across the whole generation and must live
    // in device memory: GenAI allocates them through the compiled model's
    // RemoteContext. Host tensors would work functionally but add a full
    // host<->device copy of the entire cache to every inference, which would
    // show up as (fictional) prefill time.
    auto allocate_cache = [&](ov::CompiledModel& target_cm, ov::InferRequest& target_req) {
        ov::RemoteContext ctx;
        bool have_ctx = false;
        try {
            ctx = target_cm.get_context();
            have_ctx = true;
        } catch (const std::exception&) {
        }
        for (const auto& c : caches) {
            ov::Shape shape{num_blocks, c.per_block[0], c.per_block[1], c.per_block[2]};
            if (have_ctx) {
                target_req.set_tensor(c.name, ctx.create_tensor(c.precision, shape));
            } else {
                ov::Tensor t(c.precision, shape);
                std::memset(t.data(), 0, t.get_byte_size());
                target_req.set_tensor(c.name, t);
            }
        }
    };

    // A PA sequence is described by past_lens / subsequence_begins /
    // block_indices / block_indices_begins; for one sequence the block list is
    // simply every block we allocated.
    std::vector<int32_t> block_ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) block_ids[i] = static_cast<int32_t>(i);

    const bool rank1_ids = cm.input("input_ids").get_partial_shape().rank().get_length() == 1;
    auto set_ids = [&](const std::vector<int64_t>& tok, const std::vector<int64_t>& pos) {
        ov::Shape s = rank1_ids ? ov::Shape{tok.size()} : ov::Shape{1, tok.size()};
        req.set_tensor("input_ids", i64_tensor(s, tok));
        req.set_tensor("position_ids", i64_tensor(s, pos));
    };
    auto set_pa_meta = [&](int32_t past_len, int32_t new_tokens, int32_t max_ctx) {
        req.set_tensor("past_lens", i32_tensor({1}, {past_len}));
        req.set_tensor("subsequence_begins", i32_tensor({2}, {0, new_tokens}));
        req.set_tensor("block_indices", i32_tensor({num_blocks}, block_ids));
        req.set_tensor("block_indices_begins",
                       i32_tensor({2}, {0, static_cast<int32_t>(num_blocks)}));
        auto mc = cm.input("max_context_len");
        req.set_tensor("max_context_len",
                       i32_tensor(mc.get_partial_shape().rank().get_length() == 0 ? ov::Shape{}
                                                                                  : ov::Shape{1},
                                  {max_ctx}));
        if (has_input(cm, "score_aggregation_window"))
            req.set_tensor("score_aggregation_window", i32_tensor({1}, {0}));
    };

    int64_t first_token = -1;
    double last_prefill_wall_ms = 0.0;
    auto generate = [&](std::vector<ProfRow>* prefill_acc, std::vector<ProfRow>* decode_acc,
                        int decode_steps) {
        std::vector<int64_t> pos(N);
        for (size_t i = 0; i < N; ++i) pos[i] = static_cast<int64_t>(i);
        set_ids(ids, pos);
        set_pa_meta(0, static_cast<int32_t>(N), static_cast<int32_t>(N));
        if (has_sampled_idx)
            req.set_tensor("sampled_tokens_indices",
                           i64_tensor({1}, {static_cast<int64_t>(N) - 1}));
        auto t0 = std::chrono::steady_clock::now();
        req.infer();
        last_prefill_wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (prefill_acc) collect_profiling(req, *prefill_acc);
        int64_t next = argmax_last_row(req.get_tensor("logits"));
        first_token = next;

        size_t cur = N;
        for (int step = 0; step < decode_steps; ++step) {
            set_ids({next}, {static_cast<int64_t>(cur)});
            set_pa_meta(static_cast<int32_t>(cur), 1, static_cast<int32_t>(cur + 1));
            if (has_sampled_idx) req.set_tensor("sampled_tokens_indices", i64_tensor({1}, {0}));
            req.infer();
            if (decode_acc) collect_profiling(req, *decode_acc);
            next = argmax_last_row(req.get_tensor("logits"));
            ++cur;
        }
    };

    std::printf("  [deep-profile/PA] real greedy loop: prompt=%zu tokens, decode=%d tokens, warmup=%d\n",
                N, cfg.gen_tokens, cfg.warmup);
    try {
        allocate_cache(cm, req);
        for (int w = 0; w < cfg.warmup; ++w) generate(nullptr, nullptr, cfg.gen_tokens);
        std::vector<ProfRow> prefill_acc, decode_acc;
        generate(&prefill_acc, &decode_acc, cfg.gen_tokens);
        const double profiled_wall_ms = last_prefill_wall_ms;
        std::printf("  [deep-profile/PA] first token id = %lld (%s)\n",
                    static_cast<long long>(first_token),
                    tokenizer.decode(std::vector<int64_t>{first_token}).c_str());

        // Per-op profiling serializes the queue, so the profiled wall time is an
        // upper bound. Re-time the same prefill on a non-profiling compilation:
        // the gap between that and the sum of kernel times is the host-side /
        // per-primitive dispatch cost, which for a many-small-primitive dynamic
        // graph can rival the GPU time itself.
        double clean_ms = 0.0;
        try {
            ov::AnyMap clean_cfg = cfg.plugin_cfg;
            clean_cfg.erase(ov::enable_profiling.name());
            ov::CompiledModel cm_clean = core.compile_model(model, cfg.device, clean_cfg);
            req = cm_clean.create_infer_request();
            allocate_cache(cm_clean, req);
            // TTFT is measured on a prefill that follows another prefill, not
            // one that follows decode steps: re-growing every activation buffer
            // from seq=1 back to seq=N would otherwise be charged to prefill.
            const int reps = 4;
            generate(nullptr, nullptr, 0);  // warm the clean compilation
            for (int i = 0; i < reps; ++i) {
                generate(nullptr, nullptr, 0);
                clean_ms += last_prefill_wall_ms;
            }
            clean_ms /= reps;
        } catch (const std::exception& e) {
            std::printf("  [deep-profile/PA] clean re-timing skipped: %s\n", e.what());
        }
        std::printf("  [deep-profile/PA] prefill wall time: %.2f ms profiled / %.2f ms clean\n",
                    profiled_wall_ms, clean_ms);

        print_profiling_report(prefill_acc,
                               "LLM/PA PREFILL (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                               cfg.topk, 1);
        print_profiling_report(decode_acc,
                               "LLM/PA DECODE (" + path_name(cfg.model_dir) + " / " + cfg.device + ")",
                               cfg.topk, cfg.gen_tokens);
        if (cfg.profile_csv) {
            write_profiling_csv(*cfg.profile_csv, prefill_acc, path_name(cfg.model_dir), cfg.device,
                                cfg.precision, "pa_prefill", 1);
            write_profiling_csv(*cfg.profile_csv, decode_acc, path_name(cfg.model_dir), cfg.device,
                                cfg.precision, "pa_decode", cfg.gen_tokens);
        }
    } catch (const std::exception& e) {
        std::printf("  [deep-profile/PA] inference failed: %s\n", e.what());
        return false;
    }
    return true;
}

}  // namespace mb
