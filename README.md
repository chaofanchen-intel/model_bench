# model_bench — OpenVINO GenAI C++ Benchmark Suite

A C++ port of the Python benchmark suite in [`../tests/`](../tests/), built for
**post-benchmark bottleneck analysis and optimization**. It reproduces
everything the Python benchmarks measure (TTFT/TPOT/throughput/latency, memory,
CPU/GPU/NPU utilization, identical CSV/JSON output) **and adds per-operation
profiling** (`ov::InferRequest::get_profiling_info()`) that the Python
`openvino_genai` API cannot expose.

Targets five model types:

| Task | Pipeline | Models |
|------|----------|--------|
| `llm` | `ov::genai::LLMPipeline` | Qwen3 (text generation) |
| `vlm` | `ov::genai::VLMPipeline` | Qwen3.5 (vision-language) |
| `lingua` | raw `ov::Core` | LLMLingua-2 (BERT/XLM-RoBERTa token classification) |
| `embedding` | `ov::genai::TextEmbeddingPipeline` | Qwen3-Embedding |
| `reranker` | `ov::genai::TextRerankPipeline` | Qwen3-Reranker |

---

## Why C++ for debugging/optimization?

The decisive reason is **per-layer / per-kernel profiling**. The Python
`openvino_genai` pipelines only surface aggregate `PerfMetrics` (TTFT, TPOT,
throughput, infer durations). They cannot tell you *which operation* or *which
kernel* is the bottleneck. The C++ low-level path can:

| Capability | Python (`openvino_genai`) | model_bench (C++) |
|---|---|---|
| TTFT / TPOT / throughput / infer durations | ✅ `PerfMetrics` getters | ✅ same, with tighter timing (no GIL/pybind jitter) |
| **Per-op timing** (`node_name`, `node_type`, `exec_type`, `real_time`, `cpu_time`, `status`) | ❌ | ✅ `--profile` — all 5 models, LLM/VLM split into prefill/decode |
| **Fusion visibility** (`OPTIMIZED_OUT` ops, exec kernel chosen) | ❌ | ✅ |
| Device introspection (`optimal_number_of_infer_requests`, `execution_devices`, supported properties) | partial | ✅ |
| Compile-time tuning knobs (`NUM_STREAMS`, `INFERENCE_NUM_THREADS`, perf hints) | partial | ✅ via `-lc` / `--num-streams` |

### Where `--profile` gives per-op data — now all five models

Per-op `get_profiling_info()` is only available when **you own the
`InferRequest`**. The GenAI pipelines hide the compiled model, so `--profile`
spins up a **second, raw-`ov::Core` re-implementation of each model's real
inference** (in `src/common/deep_profile.cpp`) purely for profiling, alongside
the GenAI-driven timed benchmark. This mirrors how GenAI drives the graphs
internally, so the op timings are from the *actual* workload — not synthetic
stand-ins.

- **`lingua`** — raw `ov::Core` natively; `--profile` profiles the real
  benchmarked inference directly. One hotspot report.
- **`embedding` / `reranker`** — re-runs the model's `openvino_model.xml` on the
  **real tokenized input** (the same text/query+document pair you benchmark) and
  reports per-op hotspots. One report (`*_graph REAL INPUT`).
- **`llm`** (Qwen3, stateful decoder) — drives a **real greedy loop** on
  `openvino_model.xml` (`input_ids/attention_mask/position_ids/beam_idx →
  logits` + KV state) and emits **two separate reports: PREFILL and DECODE**, so
  you can see how the bottleneck shifts (e.g. `lm_head` dominates prefill;
  attention/KV reads grow in decode).
- **`vlm`** (Qwen3.5, text-only path) — drives `openvino_text_embeddings_model`
  → `openvino_language_model` (stateful, 3D mRoPE `position_ids`) and emits
  **three reports: TEXT-EMBEDDINGS, LM PREFILL, LM DECODE**. (Image/vision-tower
  profiling is not yet wired; the language-model path is.)

Each phase is a distinct `stage` in the `--profile-csv` output
(`llm_prefill`, `llm_decode`, `vlm_text_embeddings`, `vlm_lm_prefill`,
`vlm_lm_decode`, `embedding_graph`, `reranker_graph`, `lingua_infer`), so you
can diff phases and devices.

Profiling depth is controlled by `--profile-iters N` (decode steps / encoder
repeats, default 10), `--profile-warmup N` (default 3), `--profile-topk N`
(rows shown, default 40), and `--profile-csv FILE`. For `llm`/`vlm` the
synthetic prompt length defaults to 64 tokens (override with `--input-tokens`).

> **Caveat (from the OpenVINO docs):** `get_profiling_info()` is detailed on
> CPU but may be coarse or empty on GPU/NPU. Also, due to parallel execution
> the **sum of per-op `real_time` can exceed wall-clock** — read it as a
> *relative breakdown*, not an additive time budget. The deep-profile loop is a
> faithful re-implementation, not the GenAI pipeline itself, so absolute op
> timings are representative of the workload but not a substitute for the
> GenAI-measured TTFT/TPOT in the main report.

### If you don't use C++

You can still analyze the Python CSV/JSON: compare CPU/GPU/NPU runs to find the
best device, look at prefill (TTFT) vs decode (TPOT) split, sweep input/output
lengths, compare INT4/INT8/FP16, and watch compile time + memory peaks. But
those are black-box aggregates — they can't point at the slow *operator* or
*kernel*. That's what `--profile` adds.

---

## Build

```bash
source /opt/intel/openvino_2026/setupvars.sh
source /opt/intel/openvino_genai/setupvars.sh
./build.sh                 # Release -> ./build/model_bench
# ./build.sh --opencv      # also enable VLM --image (needs OpenCV dev packages)
# ./build.sh --clean       # wipe ./build first
```

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires OpenVINO 2026.2 + OpenVINO GenAI 2026.2 (the versions under
`/opt/intel`). No internet needed — the argument parser is built-in (no cxxopts
fetch). Image input for `vlm` is the only optional feature and needs OpenCV.

---

## Usage

Every task shares: `-m/--model-dir DIR [DIR...]`, `-d/--device DEV [DEV...]`,
`-n/--num-iters N` (+1 warmup), `-r/--csv-out`, `-rj/--json-out`,
`--monitor`/`--no-monitor`, `--memory-monitor`, `-lc/--load-config`,
`--num-streams`, `--cache-dir`, `--cooldown-between-runs`, `--profile`.
Run `model_bench <task> --help` for the full per-task list.

```bash
# LLM (Qwen3) — TTFT/TPOT/throughput, prompt permutation ON by default
./build/model_bench llm -m ../models/Qwen3-1.7B-INT4 -d CPU GPU NPU -n 10 -ic 512

# VLM (Qwen3.5) — text-only or multimodal (--image needs --opencv build)
./build/model_bench vlm -m ../models/Qwen3.5-2B-INT4 -d GPU -n 5 -ic 256
./build/model_bench vlm -m ../models/Qwen3.5-2B-INT4 -d GPU --image ../test_image.jpg

# Lingua — the per-op profiling sweet spot
./build/model_bench lingua -m ../models/llmlingua-2-bert-base-multilingual-cased-meetingbank-INT8 \
    -d CPU --input-lengths 64 128 256 512 --compression-rates 0.3 0.5 0.7 \
    --warmup 5 --iters 50 --profile --profile-csv results/lingua_profile.csv

# Embedding — sweep max-length / device
./build/model_bench embedding -m ../models/Qwen3-Embedding-0.6B-int8-ov \
    -d CPU GPU NPU -n 100 --max-length 512 --pooling-type last_token

# Reranker — candidate-set-size sweep
./build/model_bench reranker -m ../models/Qwen3-Reranker-0.6B-int8-ov \
    -d GPU -n 100 --num-documents 16 --top-n 3
```

### Runner scripts

`runners/run_<task>.sh` wrap the binary, auto-source the OpenVINO env, and take
env-var overrides (mirroring `tests/runners/`):

```bash
MODELS="../models/Qwen3-1.7B-INT4" ITERS=10 ./runners/run_llm.sh GPU NPU
PROFILE=1 ./runners/run_lingua.sh CPU
MAX_LENGTH=512 ITERS=100 MONITOR=1 ./runners/run_embedding.sh GPU NPU
```

---

## Output

Reports go to `model_bench/results/<task>.{csv,json}` by default (override with
`-r`/`-rj`). Both are **append-mode** and **header-aware** (a changed per-core
CPU column count triggers a clean rewrite, never a mid-file header). The
**CSV column schema and JSON field names are byte-identical to the Python
`tests/lib/reports.py` output** — verified column-for-column — so a C++ run can
append directly to a Python-written CSV and the combined file stays consistent.

`--profile-csv FILE` additionally dumps per-op rows:
`model,device,precision,stage,iterations,node_name,node_type,exec_type,status,real_time_us_total,cpu_time_us_total,real_time_us_per_iter,count`.

Example `lingua --profile` console output (INT8 BERT on CPU):

```
Aggregate by node_type:
  node_type                   count      ms/iter       %
  FullyConnectedCompressed       73      21.1580   82.92%   <- the hotspot
  Subgraph                       12       2.2738    8.91%
  GatherCompressed                3       1.3840    5.42%
Aggregate by exec_type (kernel):
  brgemm_avx2_f32                21.1580   82.92%          <- AVX2 BRGEMM, not AVX-512
  jit_avx2_f32                    2.7438   10.75%
```

That immediately tells you the INT8 matmuls dominate and are running the AVX2
BRGEMM kernel — a concrete signal to investigate AVX-512/VNNI paths,
threading (`--num-streams`, `INFERENCE_NUM_THREADS` via `-lc`), or quantization.

---

## Behavioral parity with the Python suite

Reproduced exactly:

- **Stats**: population std (÷N), numpy-linear percentiles (p50/p90/p95), the
  per-metric filtering (tpot>0, decode_tps>0, throughput_tokens>0, infer≥0).
- **Warmup** = iteration 0, recorded in `iterations[]` but excluded from
  aggregates/memory; zero-token iterations dropped.
- **LLM**: `GenerationConfig` (`max_new_tokens`, `ignore_eos`, `max_length=MAX`,
  greedy, seeded), `ATTENTION_BACKEND=PA` (omitted on NPU), per-iteration
  token-level `input_ids` permutation (ON by default), TTFT/TPOT preferred from
  `PerfMetrics` with manual streamer fallback.
- **VLM**: `ATTENTION_BACKEND=SDPA`, permutation OFF by default (string prefix),
  TTFT = `get_ttft().mean` − last tokenization duration, TPOT = mean of
  `m_new_token_times` diffs.
- **Lingua**: raw `ov::Core`, dynamic `token_type_ids` detection, softmax +
  top-k keep-threshold compression, tokenization outside the timing window.
- **Embedding/Reranker**: single forward pass, token counting via a separate
  `ov::genai::Tokenizer` outside the latency window, `effective_top_n =
  min(top_n, n_docs)`.
- **Monitors**: CPU `/proc/stat`, GPU `xpu-smi`, NPU `intel_vpu` sysfs +
  `intel_pmt` GUID table, memory RSS (incl. recursive children) + system MiB.
  All OFF by default (matches the direct Python scripts).

### Documented divergences

- **Lingua tokenizer**: Python uses HuggingFace `AutoTokenizer`; C++ uses
  `ov::genai::Tokenizer` (the lingua dirs ship `openvino_tokenizer.xml`, so no
  HF dependency). Token counts can differ slightly, shifting
  `input_tokens`/`compressed_tokens`/`compression_ratio` — **the latency
  measurement is unaffected**.
- **`result_md5`**: the Python embedding/rerank fingerprint hashes
  `repr(result)`, which is not byte-reproducible in C++. We hash a canonical
  textual rendering instead (only affects cross-run divergence detection, not
  performance numbers). The LLM/VLM path hashes the decoded output text, same
  as Python.
- **VLM optimum.intel GPU-fallback** path has no C++ equivalent and is dropped.
- **LLM/VLM `--profile`** profiles a faithful raw-`ov::Core` re-implementation
  of the generation loop (prefill + decode), not the GenAI pipeline object
  itself. Op breakdowns are representative of the real workload; the headline
  TTFT/TPOT/throughput in the CSV/JSON still come from the GenAI pipeline.
- **VLM `--profile`** covers the text-only language-model path; the image
  vision tower is not yet profiled.

---

## Layout

```
model_bench/
├── build.sh                 # configure + build helper
├── CMakeLists.txt
├── runners/                 # env-var-driven wrappers (run_<task>.sh)
├── src/
│   ├── main.cpp             # task dispatcher
│   ├── common/              # stats, json, cli, monitors, metrics, reports,
│   │                        #   prompts, md5, profiling, deep_profile,
│   │                        #   version/precision
│   └── tasks/               # llm, vlm, lingua, embedding, reranker
└── results/                 # CSV/JSON output (created on first run)
```
