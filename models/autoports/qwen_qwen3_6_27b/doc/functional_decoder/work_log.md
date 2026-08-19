# Functional-decoder work log

## 2026-08-19: architecture and implementation

- Resolved `Qwen/Qwen3.6-27B` to `qwen3_5_text`: hidden 5120, intermediate 17408, 64 layers, native context 262144, with 48 Gated DeltaNet and 16 full-attention layers.
- Implemented only `models/autoports/qwen_qwen3_6_27b/tt/functional_decoder.py` plus stage tests/docs.
- Implemented real gated Q/K/V, partial RoPE, paged fill/update/decode SDPA, MLP/residual, vectorized 64-token DeltaNet prefill, recurrent decode, and trace-safe mutable caches/state.
- Added a bounded full-attention path above 32768 tokens: 4096-token projections/cache fills, chunked paged SDPA, and chunked residual/MLP.
- Recorded real layer-0 and layer-3 weight provenance in `real_weight_stats.json`.

## Device and repair evidence

Repository-CWD device open mixed current checkout kernels with an older installed extension and failed on `init_telemetry`. `$autodebug` established the safe `/tmp` invocation with `PYTHONPATH` pointing to the checkout and `TT_METAL_RUNTIME_ROOT` pointing to `/home/ttuser/.local/lib/model-bringup/tt-metal`. See `triage/AUTODEBUG.md`.

The initial DeltaNet chunk implementation overflowed because positive upper-triangle decay exponents were exponentiated before masking. `$autofix` isolated the first non-finite tensor; masking the exponent before `exp` removed `inf * 0` and restored PCC above 0.998. A separate length-63 investigation showed boundary prefill and first decode passing at 0.998366/0.999378; trace/determinism gates remain on exact/over-chunk lengths 64/65.

## Correctness and context commands

All commands used:

```bash
cd /tmp
env PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_functional_tt_cache \
  pytest -q /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -s
```

Targeted acceptance runs passed:

- Real boundary matrix for both layer kinds at 1/31/32/33/63/64/65.
- Linear 64/65 prefill, first decode, traced decode, and determinism: PCC values recorded in `README.md`.
- Full reversed-page prefill/decode and traced replay: PCC values recorded in `README.md`.
- Two active users: linear independent recurrent state; full disjoint reversed pages and distinct positions 32/64. Minimum PCC 0.9970185861.
- Thirty-two active users for both layer kinds at real width with per-user prefill and decode gates. Minimum PCC across all comparisons was 0.9970447455; full decode per-user PCC ranged from 0.9976604581 to 0.9989305735.
- Full non-aligned long path: `QWEN36_FULL_CONTEXT_LENGTH=32769`, passed in 13.94 s.
- Full native prefill 262144: passed in 47.66 s.
- Full forced-chunk real-weight HF comparison, 257 logical / 384 physical with 128-token chunks and a one-token tail: PCC 0.9984344296.
- Linear native prefill 262144: passed in 31.25 s.
- Linear native traced decode after prefix 262143: passed in 36.01 s including state construction.
- Full native traced decode at position 262143: eager PCC 0.9997240988, traced replay PCC 0.9997240988, repeated replay PCC 1.0.

The source/runtime-closure audit passes and forbids `torch`, `from_torch`, and `to_torch` inside measured forwards. Tracy op reports contain device TTNN operations only inside the four signpost windows.

## Profiler commands and artifacts

Profiler runtime: `/home/ttuser/.local/lib/model-bringup/tt-metal-profiler` (`ENABLE_TRACY=ON`). Each layer/phase was isolated because device close/reopen overwrites the device-side log and ten legacy trace replays overflow the installed profiler buffer. Prefill used `QWEN36_PERF_PHASE=prefill`; traced decode used `QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1` and exact pytest node IDs.

The four concrete capture commands were:

```bash
cd /tmp
env QWEN36_PERF_PHASE=prefill \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_functional_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf/raw_linear_prefill \
  -m pytest -q '/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py::test_functional_decoder_perf[blackhole-device_params0-1-linear_attention]' -s

env QWEN36_PERF_PHASE=prefill \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_functional_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf/raw_full_prefill \
  -m pytest -q '/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py::test_functional_decoder_perf[blackhole-device_params0-1-full_attention]' -s

env QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1 \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_functional_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf/raw_linear_decode \
  -m pytest -q '/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py::test_functional_decoder_perf[blackhole-device_params0-1-linear_attention]' -s

env QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1 \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_functional_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --no-runtime-analysis \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf/raw_full_decode \
  -m pytest -q '/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py::test_functional_decoder_perf[blackhole-device_params0-1-full_attention]' -s
```

The older profiler's legacy parser required `pd.options.future.infer_string=False`; two unheadered trace-duration extras were ignored during CSV emission. Blackhole/110-core metadata was restored from the actual device log before running `tt-perf-report`. Results:

- linear prefill: 177 ops, 6046 us device, 8134.594 us signpost wall.
- full prefill: 51 ops, 2594 us device, 3269.159 us signpost wall.
- linear traced decode: 88 ops, 2901 us device, 3178.546 us signpost wall.
- full traced decode, refreshed after the explicit batch-safe SDPA configuration: 50 ops, 2639 us device, 2861.372 us signpost wall.

See `perf/summary.csv`, `perf/reports/`, and the four acceptance `perf/raw_*` directories. Failed combined/high-capacity captures and duplicated multi-hundred-megabyte device logs were removed after their diagnosis; the compact enriched op reports and Tracy host-op provenance remain.

## Watcher and review

The final non-profiler suite ran separately with `TT_METAL_WATCHER=10`: 24 passed, 2 profiler tests deselected, in 593.19 s. After review fixes, a second Watcher run explicitly covered forced-chunk PCC, both 32769/262144 full-prefill cases, and traced maximum-position decode: 4 passed, 24 deselected, in 94.63 s. Its artifacts are `watcher/review_rerun/pytest.log`, `watcher/review_rerun/junit.xml`, and `watcher/review_rerun/generated/watcher/watcher.log`.

The final batch regression used the same environment with this selector:

```bash
pytest -q /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
  -k 'real_weight_batch32 or real_weight_multi_user or full_attention_real_weight_paged_decode_trace or full_attention_advertised_context_decode' \
  --junitxml=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher/batch32/junit.xml -s
```

It passed 6 tests with 24 deselected in 19.72 s. The same Watcher scan returned no matches. Artifacts are `watcher/batch32/pytest.log`, `watcher/batch32/junit.xml`, and `watcher/batch32/generated/watcher/watcher.log`.

Exact post-review Watcher command and scan:

```bash
cd /tmp
env TT_METAL_WATCHER=10 TT_METAL_LOGGER_LEVEL=Info \
  TT_METAL_LOGS_PATH=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher/review_rerun \
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
  TT_METAL_CACHE=/tmp/qwen36_functional_tt_cache \
  pytest -q /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
  -k 'forced_chunked or full_attention_advertised' \
  --junitxml=/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher/review_rerun/junit.xml -s \
  > /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher/review_rerun/pytest.log 2>&1

rg -ni 'watcher.*(error|assert|hang|timeout)|noc.*(error|timeout)|kernel.*assert|device.*hang' \
  /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher/review_rerun/generated/watcher/watcher.log
```

The scan returned no matches. The complete-suite artifacts remain `watcher/final/pytest.log`, `watcher/final/junit.xml`, and `watcher/final/generated/watcher/watcher.log`.

## Anomaly ledger

### Nanobind exit-time lifetime warnings

- **Observed anomaly:** the installed TTNN module reports leaked nanobind instances, types, and functions during Python shutdown.
- **Evidence:** `watcher/final/pytest.log`, `watcher/review_rerun/pytest.log`, `watcher/batch32/pytest.log`, and the hardware-free `anomalies/collect_only.log`.
- **Affected path:** installed TTNN Python-binding teardown after pytest or collection, not a measured decoder pass.
- **Control or comparison:** `anomalies/collect_only.log` reproduces the warning after collecting 30 tests without opening hardware; all retained hardware suites exit successfully, and all Watcher logs stop and detach devices cleanly.
- **Likely subsystem:** installed nanobind/TTNN binding lifetime management.
- **Investigation performed:** compared hardware-free collection, repeated target processes, pytest outcomes, and Watcher shutdown records.
- **Resolution:** controlled external teardown warning; no stage-owned runtime corruption is evidenced.

### Batch-32 decode lane corruption

- **Observed anomaly:** the first batch-32 attempts failed; full decode corrupted lanes 8-10, 19-21, and 30-31, while linear prefill was slightly below threshold.
- **Evidence:** `anomalies/batch32_initial_failure/` and `anomalies/batch32_second_failure/` retain pytest, JUnit, and Watcher logs.
- **Affected path:** active outer-batch functional prefill and default-config paged SDPA decode.
- **Control or comparison:** user-wise TTNN prefill restored linear correctness; physical page remapping did not move the bad full-decode lanes; an explicit 8x8 exact-exp SDPA program config corrected every lane.
- **Likely subsystem:** outer-batch numerical behavior and the installed paged-decode SDPA default program selection.
- **Investigation performed:** isolated prefill from decode, compared per-user PCC, changed page placement, separated projections, and tested explicit SDPA configuration.
- **Resolution:** fixed. Final `watcher/batch32/` evidence passes both layer kinds plus trace, multi-position, and max-position regressions; minimum per-user PCC is 0.9970447455 and the Watcher scan is clean.

### Host topology warnings

- **Observed anomaly:** the runtime reports an unknown `B850M-C` motherboard and warns that opening a subset of MMIO devices can slow remote access.
- **Evidence:** retained Watcher pytest logs.
- **Affected path:** UMD topology discovery and device opening.
- **Control or comparison:** all target tests pass and Watcher repeatedly attaches, checks, stops, and detaches without a device fault.
- **Likely subsystem:** host metadata/topology discovery, outside the decoder.
- **Investigation performed:** compared warnings with the full device lifecycle and Watcher fault scans.
- **Resolution:** controlled external warning; no decoder impact observed.

The retained 262144-token Watcher prefill took 71.869 s; the 47.66 s value above is the earlier normal-runtime feasibility run. Stage-review verdict and local commit SHA are appended after their gates pass. No push is performed.

## Stage review

Fresh final `$stage-review` verdict: `clean-pass`. It reported no required work and no hard-check gaps after inspecting source, all 30 jointly evidenced tests, context/PCC/trace coverage, Watcher scans, anomaly controls, and refreshed profiler reports.

Stage implementation commit: `c0bb9982f0d`. This commit is local only and was not pushed.
