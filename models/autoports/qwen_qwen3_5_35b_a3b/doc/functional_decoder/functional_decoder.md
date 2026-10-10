# Functional decoder performance evidence

Workload: **4096 input tokens, 128 traced decode steps, batch 1, one request**.
Each result is a complete single layer using real pinned Qwen3.5-35B-A3B weights
and supplied hidden states. It is not an autoregressive full-model token rate.
One Blackhole ASIC (device ID 3, mesh 1×1, 110 worker cores) participates; four ASICs
are available. Prefill is warmed eager execution; decode is warmed TTNN replay.

| Layer kind | Whole prefill device us | Mean traced decode device us | Useful prefill FLOPs / nominal peak | Estimated decode DRAM / peak |
|---|---:|---:|---:|---:|
| linear_attention | 1743870.348 | 2190.255 | 0.045586% | 10.583241% |
| full_attention | 1720531.702 | 1608.712 | 0.052297% | 7.188733% |

## Timing definition and completeness

`summarize_perf.py` reads `DEVICE FW START CYCLE`, `DEVICE FW END CYCLE`, and
`DEVICE FW DURATION [ns]` from raw Tracy CSV. The complete-layer interval is the
last firmware end minus the first firmware start, converted from cycles to ns
using the recorded firmware durations, then divided by 1000 for microseconds.
This includes attention, normalization, MoE, every other operation and all gaps
inside the layer. It is not a sum of matmul durations or per-op percentages.

Prefill is one complete 4096-token pass. Decode is the arithmetic mean of 128
complete single-token replay spans, each with its own start/end signpost. Host
input/position refresh, output comparison and profiler drains occur outside
those spans. The eager prefill includes internal dispatch starvation gaps and
profiling overhead; these are instrumented functional-stage measurements, not
optimized-decoder results. The harness performs warmup before measurement.

All measured device rows must have timestamps. Linear prefill contains 4866
programs and each replay 166; full prefill contains 867 and each replay 101.
Trace-definition rows outside measured windows need no execution timestamps.
Each `measurement.json` preserves every replay's span, operation count, device ID,
unit conversion, source CSV path and SHA-256.

## Roofline assumptions

`estimate_roofline.py` counts useful projection FLOPs, eight selected experts of
256 plus the shared expert, router work, causal QK/PV or GDN recurrence, and
convolution. Unselected experts, tile padding and sparse prefill tile-union work
are excluded. Small vector/transcendental operations are omitted from the FLOP
numerator; their time remains in the whole-layer denominator.

Decode traffic is an explicit compulsory-transfer estimate, not a measured DRAM
counter: each active weight once, required KV once, and 14 materialized FP32
recurrent-state volumes in GDN. Extra temporary traffic, padding, multicast
reloads and caching are not modeled. Packed BFP4/BFP8 tiles include exponent
bytes (576/1088 bytes per 1024 values). Component estimates are retained in each
`roofline.json`. No ratio is clamped.

Peak denominators use the full participating ASIC: 120 physical Tensix cores ×
4096 FLOPs/cycle × 1.35 GHz = 663.552 TFLOP/s nominal LoFi peak, and 512 GB/s DRAM.
The [official P300c specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html)
gives two ASICs per card and 1024 GB/s card bandwidth. Per-core phase rates are
also in installed `tt_perf_report/perf_report.py` (`ArchitectureSpec`, Blackhole).
The runtime exposes 110 worker cores and uses mixed fidelity; HiFi4's theoretical
compute peak is one quarter of the stated maximum. These ratios use the same
complete-layer device time above.

Default sparse/shared gate-up weights are BFP4, down weights BFP8, LoFi; the
shared gate is BF16/HiFi2. GDN projections use BF16/HiFi4 with FP32 accumulation
and recurrent state. Full-attention projections use BFP8/LoFi and KV is BF16.
The two outer layer norms and router probabilities use FP32 boundaries;
inherited full-attention Q/K norm offsets are BF16.

## Artifacts and reproduction

For each `tracy/{linear_attention,full_attention}/`:

- `ops.csv.gz`: lossless complete source CSV archive. Decompress with
  `gzip -dc ops.csv.gz > ops.csv` if the local uncompressed copy is absent.
- `measurement.json`, `roofline.json`, `processing_provenance.json`: complete-layer
  measurements, calculation basis, CSV SHA-256, and exact processing commands.
- `{prefill,decode}_perf_report.txt`: human-readable tt-perf-report tables.
- `{prefill,decode}_perf_report.csv`: filtered report data; `Device Time` is us.
- `{prefill,decode}_perf_report.console.log`: CSV-export status and warnings.

The tables include every measured operation. The report tool warns about several
unclassified op names and omits sparse-matmul per-op utilization without numeric
`nnz`; these do not remove rows from the complete-layer calculation. Its summed
per-op summary percentages are not copied into whole-layer telemetry.

From the checkout after `source ../scripts/environment.sh`:

```bash
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.process_profiles --layer-type linear_attention
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.process_profiles --layer-type full_attention
```

These commands process existing evidence only. Capture commands are recorded in
`profile_commands_first_retry.json` (accepted linear) and `profile_commands.json`
(accepted full). Full capture requires the path alias described and reproduced in
`profiler_hash_diagnosis.md`, `profiler_path_alias.json`, and `work_log.md`.
The alias resolves to unchanged checkout bytes. Captures run serially with
32768-program buffers and between-pass drains, never alongside watcher.

The original buffer-overflow capture and both full marker-collision captures
remain rejected and preserved locally. Large raw Tracy binaries stay outside
Git; the complete exported CSV archives, tables and provenance are committed.
Tracy's optional WASM copy warning points at its default output directory while
this run uses an explicit output directory; the actual capture and export exist.
Pandas mixed-column warnings did not lose measured rows, as checked above.


## Cached profiler metadata anomaly

The CSV `PROGRAM CACHE HIT` column contains stale `False` values. The profiler
caches the first call's abbreviated message in `ttnn/cpp/tools/profiler/op_profiler_json.cpp:174–180`
and reuses it without refreshing that field in `ttnn/api/tools/profiler/op_profiler_serialize.hpp:380–384`.
Independent warmup evidence is `profile_integrity.json`: every one of the 105
linear and 84 full measured prefill program hashes already occurred before the
start signpost; zero new hashes appear in either measured prefill. Decode uses
actual trace replays with program-cache misses forbidden. The stale field does
not establish cold execution and is not used for timing or acceptance.

For a fresh full-attention capture, create the alias from the checkout root if
it does not already exist, then use the recorded environment:

```bash
ln -s "$PWD" /tmp/q35-functional-profiler-root  # create only when absent
TT_METAL_RUNTIME_ROOT=/tmp/q35-functional-profiler-root \
TT_METAL_KERNEL_PATH=/tmp/q35-functional-profiler-root \
TT_METAL_CACHE=/tmp/q35-functional-profiler-cache-cwd \
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_profiles --layer-type full_attention
```

`run_profiles.py` supplies the separate process CWD and absolute checkpoint path.
Use a fresh profiler output directory when preserving a previous capture.
