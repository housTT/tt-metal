# GPT-OSS 120B functional decoder work log

Status: functional-decoder implementation, device gates, independent stage review, and local stage commit complete.

## Target and environment

- Model: `openai/gpt-oss-120b`.
- Checkpoint revision: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Functional mesh: one P150-class Blackhole device (`1x1`).
- Later-stage targets: P150x2 and P150x4; no multichip implementation was started here.
- HF dimensions: hidden 2880, head 64, 64 query heads, 8 KV heads, 128 experts, top 4, intermediate 2880, 36 alternating sliding/full layers, sliding window 128, context 131072.
- Hardware discovery: four local Blackhole devices with firmware bundle 19.13.1. `TT_VISIBLE_DEVICES=0,1,2,3` is required for this host's control-plane graph, while each functional test opens a `1x1` mesh.

All hardware commands used this exact prefix because the inherited shell pointed at a different simulator checkout:

```bash
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME \
  -u TT_METAL_SLOW_DISPATCH_MODE -u TT_METAL_DISABLE_SFPLOADMACRO \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_METAL_HOME=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_CACHE=/home/ttuser/dev/gpt-oss-20b/tt-metal/.tt_metal_cache \
  TTNN_CONFIG_OVERRIDES='{"cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache","model_cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache/models","tmp_dir":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_tmp"}' \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn \
  LD_LIBRARY_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib
```

`HW_ENV` below means that exact prefix.

## Checkpoint and weight evidence

Only the index and shards containing layers 0 and 1 were needed:

```bash
hf download openai/gpt-oss-120b model.safetensors.index.json --quiet
hf download openai/gpt-oss-120b \
  model-00009-of-00014.safetensors model-00010-of-00014.safetensors
```

The two shards total 8.8 GB. `tests/real_weight_utils.py` resolves the index, validates every raw shape/dtype, and dequantizes only the requested layer's MXFP4 expert tensors. Both real layer kinds load through `FunctionalDecoder.from_state_dict`; observed dequantization was 8.7–8.8 seconds per layer with a warm tensor cache.

Exact BF16 population mean/std, dtype, and shape for all 17 tensors in each layer are stored in `real_weight_stats.json`. Statistics were accumulated in 8,388,608-element FP32 chunks with FP64 sums/squared sums. Synthetic fixtures preserve the real mean and use `min(real_std, config.initializer_range)` for independent deterministic noise. A trial using uncapped trained variances without trained cross-tensor correlations was correctly rejected after producing PCC 0.942628 prefill and 0.844111 decode.

## Correctness

The accepted model-specific batch-one whole-decoder bar is PCC `>= 0.95`; the seeded-random batch-two routing/position/page-table stress test uses `0.99`. The `0.995` default is retained for continuous components and identical-route counterfactuals. See `precision_investigation.md` for the complete AutoFix/AutoDebug evidence and why hard top-4 routing requires these exceptions.

Sequence-129 final results:

| Weights | Layer | Prefill PCC | Traced decode PCC | Repeated replay |
| --- | --- | ---: | ---: | --- |
| Real | 0, sliding | 0.977154173283 | 0.992163253293 | bitwise equal |
| Real | 1, full | 0.990045250294 | 0.965931676798 | bitwise equal |
| Stats-derived synthetic | 0, sliding | 0.981616212808 | 0.980701267793 | bitwise equal |
| Stats-derived synthetic | 1, full | 0.987222548868 | 0.993940590124 | bitwise equal |

Exact commands:

```bash
HW_ENV ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_paged_prefill_and_traced_decode -s

HW_ENV \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_real_weight_paged_prefill_and_traced_decode -s
```

Both commands passed two layer-kind cases. Decode PCC was read after complete trace replay, never from the compile or capture forward. The fresh real-weight acceptance rerun passed in 24.04 seconds. Raw transcript: `correctness/real_weight_acceptance.log.gz`; decompressed SHA-256 `04d3acbc0bdc2b88aa8b43b9d9c47d94d8c2e86839b43d60e8b89910942e479e`, compressed SHA-256 `657d965be65bc34822395a11d67231a417582b58165a1fbe71f5ac157099458a`.

The final default hardware suite passed `8` tests and skipped only the explicitly opt-in profiler, batch-32, real-weight, chunk-boundary, and maximum-prefill cases (`10` parametrized skips):

```bash
HW_ENV ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py -s
```

Result after the length-one contract repair: `8 passed, 10 skipped` in 153.39 seconds. Each skipped gate was run separately with its documented opt-in environment variable. Raw transcript: `correctness/default_suite.log.gz`; decompressed SHA-256 `ccaac4ca907985e0803b15947117142fb8614a9d0a28ce34df1361265edaeeb0`, compressed SHA-256 `acdb76f330dfe744155625ec4fcab380018a4ed6775a17479bf6f2576503320f`.

Batch/page/current-position coverage:

- HF-vs-TTNN batch 2 passed prefill and traced decode for both layer kinds with disjoint permuted physical pages and seeded-random distinct device current positions. Layer 0 used `[8, 32]`; layer 1 used `[26, 16]`. Independently reconstructed HF prefix caches prove each position's semantics. Final deterministic-fixture prefill/decode PCCs were 0.983274679196/0.994583692307 for sliding and 0.990537791979/0.997145155509 for full. Prefill uses the `0.95` gate; this independent-per-user hard-routing decode stress uses `0.99`.

```bash
HW_ENV ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_batch_two_paged_prefill_and_traced_decode -s
```

Result: `2 passed` in 38.73 seconds at the final `0.99` gate. Raw transcript: `correctness/batch_two_distinct_positions.log.gz`; decompressed SHA-256 `64789e786fce72192b795dca9acb49fd02dce0b04f4f3c72b1e7fb7d1bc04dc6`, compressed SHA-256 `f786677a2672c5c2d03292789b097702724cfd66163d5a8d57af7215a2e6fb82`.

The stage-review precision finding was closed with a retained, reproducible, real-weight counterfactual for batch-two traced decode and both layer kinds:

```bash
HW_ENV \
  GPT_OSS_120B_ROUTING_ANALYSIS=1 \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_routing_precision_analysis.py -s
```

| Layer | Whole traced PCC | Natural vs fixed tail | Fixed-route HF vs TT tail | Traced vs diagnostic tail |
| --- | ---: | ---: | ---: | ---: |
| 0, sliding | 0.952735916 | 0.984794872 | 0.998856913 | 0.999999445 |
| 1, full | 0.984314131 | 0.986494229 | 0.998906048 | 0.999999326 |

Both cases crossed a route boundary for one of two users and passed the default `0.995` fixed-route bar. Result: `2 passed` in 24.32 seconds. Raw transcript: `correctness/real_weight_routing_counterfactual.log.gz`; decompressed SHA-256 `5d4b84ba89ae1cedc4db770c030fb643be90e02542d6af7b20ad29e13661647a`, compressed SHA-256 `52b97533191cf7b523f7502f37aaab31faa3b576b000965f5151dd0293eefbdf`.
- Batch 32 passed both layer kinds with a 131072-token-per-user paged-cache/page-table allocation, sequence-33 prefill, 32 unique seeded-random device current positions spanning 1–33, complete decode trace, finite output, and bitwise-equal replay. Sliding took 28.42 seconds and full took 28.40 seconds on the cold/warm JIT mix. Raw transcript: `correctness/batch32_random_positions.log.gz`; decompressed SHA-256 `0674496a0c10007b597e7163d3021e2b72727041dc33c39c7da38dd55bf266c3`, compressed SHA-256 `3a7f47f8c927eff362f9b6ee96c99b5135acf2cb5c3a05402bd25ef76b140e7d`.

```bash
HW_ENV GPT_OSS_120B_RUN_BATCH32=1 ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_batch_32_paged_prefill_and_traced_decode_capacity -s
```

## Boundary and full-context evidence

Both layer kinds passed full decoder execution at HF-valid length 1 and all tile/page/window boundaries: 31, 32, 33, 63, 64, 65, 127, 128, and 129. Each case checks HF-vs-TTNN PCC, exact logical output shape, and that the caller's TTNN input remains allocated at the same address. Length 1 exceeded the default `0.995` bar: PCC 0.996835765471 for sliding and 0.998041637348 for full attention.

```bash
HW_ENV ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_prefill_tile_page_and_window_boundaries -s
```

The focused rerun passed both layer kinds in 46.79 seconds. Raw transcript: `correctness/length_one_and_boundaries.log.gz`; decompressed SHA-256 `d28ae67e733dbebc5f916e71f35fb767f6443dc7a59dcb397f6814965e8c5504`, compressed SHA-256 `9948261f65fa7c0d5c8ee9abcd1038291d737d8b3db81c941607819a0ad94fcb`.

Both layer kinds also passed 4095, 4096, and 4097 tokens around the expert chunk boundary:

```bash
HW_ENV GPT_OSS_120B_RUN_CHUNK_BOUNDARIES=1 ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_prefill_chunk_boundaries -s
```

- Layer 0 total test call: 30.18 seconds.
- Layer 1 total test call: 30.00 seconds.

Result: `2 passed` in 62.85 seconds. Raw transcript: `correctness/chunk_boundaries.log.gz`; decompressed SHA-256 `8d02004eb4a9782800c2cafc361d7ee47c74e225570f7899c47a06b62b5c39f2`, compressed SHA-256 `2779377e2a52fdde1a0f65814e0af06f7938bb2b367b4012cb0606d5878130b2`.

The advertised maximum was tested without reduction:

```bash
HW_ENV GPT_OSS_120B_RUN_MAX_PREFILL=1 ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_prefill_at_advertised_context_limit -s

HW_ENV ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_traced_decode_at_advertised_context_limit -s
```

| Layer | Prefill 131071 | Prefill 131072 | Traced decode position |
| --- | ---: | ---: | ---: |
| 0, sliding | 138.601672 s | 138.591955 s | 131071, pass |
| 1, full | 141.221357 s | 141.221283 s | 131071, pass |

The long prefill gates verify full-decoder execution, finite final-token output, exact logical shape, and preserved public input. The fresh maximum-prefill rerun passed both layer kinds in 615.89 seconds. Raw transcript: `correctness/max_prefill_context.log.gz`; decompressed SHA-256 `6fdf18138ac1d153d848c18cdf4c3ab2a4165d6acc2006ebac519032dacaf19b`, compressed SHA-256 `923e2a1e442dbf2e2f924504b604831f55916e562e7db623951a9cdc4f77ab44`. The max-position gate captures and replays the complete decode and requires bitwise equality. `../context_contract.json` records supported context 131072 and no capability reduction.

## Performance and provenance

Profiling was run separately from watcher instrumentation. Each layer kind used sequence 128, one warm prefill, one signposted prefill, a compile/capture/replay sequence, and one signposted warmed trace replay:

```bash
HW_ENV \
  TT_METAL_PROFILER_DIR=/home/ttuser/dev/gpt-oss-20b/tt-metal/generated/profiler \
  GPT_OSS_120B_PROFILE=1 \
  ./python_env/bin/python -m tracy -r -p -v -m pytest \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_warmed_prefill_and_traced_decode_performance \
  -k sliding -s

HW_ENV \
  TT_METAL_PROFILER_DIR=/home/ttuser/dev/gpt-oss-20b/tt-metal/generated/profiler \
  GPT_OSS_120B_PROFILE=1 \
  ./python_env/bin/python -m tracy -r -p -v -m pytest \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_warmed_prefill_and_traced_decode_performance \
  -k full -s
```

Final raw reports came from generated report directories `2026_08_28_00_43_49` (sliding) and `2026_08_28_00_44_24` (full). They were copied into the stage directory before running `tt-perf-report` 1.2.8 with `PERF_PREFILL`/`PERF_PREFILL_END` and `PERF_DECODE`/`PERF_DECODE_END`.

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --csv prefill_perf_report.csv --no-advice --no-color

/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-summary --no-advice --no-color > prefill_perf_report.txt
```

The same two commands were run with decode signposts and output names.

| Layer | Prefill summed device time | Decode summed device time | Ops |
| --- | ---: | ---: | --- |
| sliding | 137.401 ms | 2.060 ms | 58 device/0 host; 64 device/0 host |
| full | 137.418 ms | 2.047 ms | 58 device/0 host; 64 device/0 host |

Original raw CSV SHA-256 (recover the byte-identical CSV with `gzip -dc ops_perf_results.csv.gz`):

- sliding: `5ba52f13c0e014d9c15dc69c42f1d6f934e2707c32db9b1b6d443ccac78e7e40`
- full: `8f55dc138f597834f4d8111669e336855fa8e20226c7102acce172de2e081533`

Compressed artifact SHA-256:

- sliding: `ba01653544b3c03fd68e0142f3976d49a94faa05a5ec486ca877ed6cef4f144e`
- full: `9a9b0741f856133eb35890d7eb38b0b6ae9d6f4cee8eab28403ed811e6d5e3ba`

## Runtime and watcher audit

`runtime_fallback_audit.md` records a clean reachable-call-tree audit. Setup-only PyTorch conversion is outside measured passes; both profiler windows independently report zero host ops.

The final watcher command is run with profiler instrumentation absent:

```bash
HW_ENV \
  TT_METAL_WATCHER=2 TT_METAL_WATCHER_APPEND=1 \
  TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TT_METAL_LOGS_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/models/autoports/openai_gpt_oss_120b/doc/functional_decoder/watcher/final_correctness \
  ./python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_functional_decoder.py::test_paged_prefill_and_traced_decode -s
```

Watcher result: both layer kinds passed in 41.64 seconds. The original log's SHA-256 is `2855e0e6a90a1100154849c126e904fbf481685feff14e8f182e6eb80d4719b4`; the tracked `watcher.log.gz` SHA-256 is `3a7e218c8bda237b366712ab879d466f7482dcd1bf2a4946438d392e9dd28ab9`. A case-insensitive scan for fatal/assert/NoC/bounds/sanitizer/watcher/kernel errors returned no matches; the final dump's minimum stack headroom was 1332 bytes. `watcher/final_correctness/watcher_scan.txt` records the scan.

## Limitations and scope

- This is intentionally a single-device functional decoder. No optimized-decoder, multichip-decoder, full-model, generator, or vLLM work is included.
- Performance numbers are measurements, not improvement claims.
- Batch-32 is a device capacity/trace test; eager-HF numerical correctness at that batch is not run because the host reference's 128 dense expert tensors make it prohibitively expensive. Numerical batch correctness is covered at batch 2.
- Primary batch-one whole-layer PCC uses the documented model-specific `0.95` gate; the seeded-random batch-two hard-routing decode stress uses `0.99`. The retained real-weight batch-two counterfactual exceeds `0.99885` for both layer kinds when routes are fixed; no unproven precision candidate remains in the source.

## Host checks

- `./python_env/bin/pre-commit run`: passed for the complete staged change.
- `python -m py_compile`: passed for the decoder and both test modules.
- `python -m json.tool`: passed for `context_contract.json` and `real_weight_stats.json`.
- Direct decoder-source scan found no `torch`, `ttnn.from_torch`, or `ttnn.to_torch` token.
- No C++ build was required under the repository `AGENTS.md` matrix because this stage adds only Python, JSON, Markdown, CSV/text, and compressed evidence artifacts.

## Review and commit ledger

- Fresh stage-review verdict: `clean-pass` from `/root/stage_review_clean` after a full staged-artifact rereview.
- Closed review findings: retained real-weight routing counterfactual, distinct batch positions/page tables, raw real-weight/chunk/max-prefill provenance, qualified watcher scan, and HF-valid length-one prefill support/evidence.
- Stage commit SHA: `acfbdee588f929b51a22f99c46baa5394eb75216`.
- Ledger commit SHA: this documentation-only follow-up commit; its exact SHA is reported in the final handoff because a commit cannot contain its own SHA.
- Push: never performed.
