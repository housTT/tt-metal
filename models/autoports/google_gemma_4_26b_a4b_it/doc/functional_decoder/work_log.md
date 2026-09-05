# Functional-decoder work log

Date: 2026-09-05 UTC

Initial checkout: branch `hous/gemma-4-26b-a4b-it`, HEAD
`e983152d5760c80e794280e8f1862169ec36a7e0`. The stage began from a clean
worktree. Only functional-decoder implementation, tests, and documentation
under this autoport directory were modified.

## Model and hardware discovery

- Local checkpoint:
  `/home/hous/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- Target configuration: hidden size 2816, 30 layers, 128 experts/top-8,
  advertised context 262144, sliding window 1024. Representative meaningful
  layer kinds are layer 0 sliding attention and layer 5 full attention.
- `tt-smi -ls --local` found four Blackhole P300 chips. All functional runs
  used a serialized 1x1 mesh on device id 1 after
  `source python_env/bin/activate`.

## Failed gates and repairs

1. Full-attention paged SDPA rejected loose `block_size`/`num_kv_heads`
   keywords. Reproducing the smallest full-layer real-weight parameters and
   inspecting the installed binding proved that reads require
   `ttnn.PagedCacheGeometryOverride`; paged fill/update retain their existing
   keywords. Added the dedicated SDPA view helper and reran natural/shared
   full-attention cases successfully.
2. The first advertised-context run hung during two host-backed 1 GiB
   `ttnn.full` cache uploads. A fresh xhigh AutoTriage capture was taken while
   the process remained live. It found `cq_prefetch` waiting for five NoC0 read
   responses and `cq_dispatch` waiting downstream, with the TTNN op mesh idle.
   The source showed that `ttnn.full` constructs a host vector, contrary to the
   test's direct-device intent. Replaced both initializers with
   `ttnn.moreh_full` and verified small readback plus both 262144-context
   parameters. After the preserved hung job was terminated, `tt-smi -ls`
   reported the stale NOC0 condition; one bounded `timeout 180 tt-smi -r`
   recovered all four devices, followed by a successful list and 1x1 mesh
   open/close smoke. No lock files were removed.
3. The first passing Tracy workload overflowed finite per-RISC profiler
   buffers, dropping 51 device op IDs and causing a correct report-merge
   assertion. AutoFix compared the raw logs, rejected assertion relaxation,
   and added supported `ttnn.ReadDeviceProfiler` drains between complete
   passes. Sliding and full reruns had zero missing IDs and generated complete
   reports.

Detailed evidence is in `AUTOFIX.md`, `AUTOTRIAGE.md`,
`AUTOFIX_TRACY_POSTPROCESS.md`, and `triage/`.

## Correctness and capacity commands

Default suite after repairs:

```bash
source python_env/bin/activate && \
python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py
```

Final result: 28 passed, 10 opt-in tests skipped, 2 warnings in 44.63 s. This includes
all three canonical real-weight prefill/decode cache views, batch-2 prefill,
batch-1 and batch-32 traced decode for both layer kinds, boundary prefill,
bounded-cache wrap, sparse-MoE delegation, and runtime fallback audits.

Real-weight PCC from this run:

| Layer/cache | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| layer 0 sliding/shared | 0.999163 | 0.999739 |
| layer 5 full/natural | 0.998457 | 0.999860 |
| layer 5 full/shared view | 0.998457 | 0.999860 |

Advertised-context traced decode was run separately for both layer kinds:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'advertised_context_traced_decode and sliding_attention'

GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'advertised_context_traced_decode and full_attention'
```

Both passed at current position 262143 with rolled page tables, verified
device-resident history sentinels, finite output, and repeat trace replay PCC
1.0. The sliding and full physical K+V cache footprints were 2 GiB and 1 GiB,
respectively.

Real-weight full-decoder prefill capacity was run as four serialized commands:

```bash
for length in 262143 262144; do
  for layer in sliding_attention full_attention; do
    GEMMA4_PREFILL_CAPACITY_LENGTH="$length" python -m pytest -q -s \
      models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
      -k "prefill_capacity_probe and $layer"
  done
done
```

All four passed. Host elapsed times recorded in the JSON artifacts were:

| Layer kind | 262143 | 262144 |
| --- | ---: | ---: |
| sliding attention | 318.901 s | 325.252 s |
| full attention | 404.387 s | 406.479 s |

The 262143 probes were repeated after the final non-aligned-tail repair. Both
artifacts carry the final decoder SHA-256
`1de6d3e39b2ba645f1d9f0de032178b12faa733b3d51ae4735e5ee33953cdf23`
and final test SHA-256
`a493ef28f2abd5b25cda7d4b009e9b765681dc9b640f26ab5d5923fd44ac526c`.

The chunking-cliff attention-only regression used
`GEMMA4_LONG_ATTN_TEST=1` at length 32800 for both kinds and passed the checked
rows 32767/32768/32799 at PCC >= 0.995. Boundary matrices exercised
1/31/32/33, 63/64/65, 127/128/129, and 1023/1024/1025 with permuted page
tables. Their lowest PCC was 0.995127.

## Determinism, fallback, and weight inventory

```bash
source python_env/bin/activate && \
python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'host or audit or delegates'
```

Result after the final host regression was added: 14 passed, 24 deselected.
The hot-path audit is clean for `torch`, `ttnn.from_torch`, `ttnn.to_torch`,
and host fallback inside one runtime pass.

```bash
source python_env/bin/activate && \
python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_synthetic_weights.py \
  -k 'real_weight_stats_cover or synthetic_generation or full_shape_materializer or from_state_dict_accepts'
```

Result: 6 passed, 2 deselected. `real_weight_stats.json` was computed from the
canonical local checkpoint one tensor at a time and covers 22 layer-0 tensors
and 21 layer-5 tensors. Trace artifacts show eager-to-replay and repeated-input
PCC 1.0.

## Performance

`tt-perf-report` 1.2.9 was installed into the existing `python_env` and used
only to render captured CSVs. The final verified Tracy commands were:

```bash
GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_functional_sliding_v3 \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch1-sliding_attention_1024-device_params0-mesh_device0]'

GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1 \
timeout 1800 \
python -m tracy -r -p -v --check-exit-code \
  -o generated/profiler/gemma4_functional_full_v3 \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_functional_decoder_perf_profile[blackhole-batch1-full_attention_1024-device_params0-mesh_device0]'
```

Both pytest workloads and Tracy post-processing passed. The signpost-sliced
tables report:

| Layer kind | Prefill device total | Traced decode device total |
| --- | ---: | ---: |
| sliding attention | 1,242.489 ms / 557 ops | 3.012 ms / 74 ops |
| full attention | 1,243.618 ms / 557 ops | 3.207 ms / 76 ops |

Measured-phase CSVs, human-readable tables, exact signposts, and host timings
are under `perf/` and the two `*_host_timings.json` files.

## Watcher

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/watcher \
python -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_traced_decode_batch_contract[blackhole-batch1-sliding_attention-device_params0-mesh_device0]' \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_traced_decode_batch_contract[blackhole-batch1-full_attention-device_params0-mesh_device0]'
```

Final current-source result: 2 passed in 5.92 s. The watcher server attached to all visible
devices, stopped cleanly, and its error/assert/fatal/hang/deadlock scan had no
matches. `watcher/summary.json` records the log hash and command.

## First stage-review remediation

The first independent report (`stage_review.md`) returned
`more-work-needed` with three findings. The runtime non-aligned tail still used
host-backed `ttnn.full`; the bounded-wrap result was eager despite being called
traced; and perf/watcher commands contained placeholders.

The tail position now uses device-side rank-2 INT32 `ttnn.moreh_full`, which
the paged update consumes as one interleaved index stick. The initial rank-1
attempt correctly failed the primitive's rank guard and was not retained. The
focused final command was:

```bash
python -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py::test_bounded_modulo_prefill_tail_cache_integrity[blackhole-device_params0-mesh_device0]'
```

Result: pass. The fallback audit now explicitly rejects `ttnn.full(` in the
runtime call graph.

The bounded decode regression now captures the complete `decode_forward` with
`cache_position_modulo=1024` and replays positions 0..1103 through stable
device buffers. It passed against an unbounded eager control with PCC 1.0 at
1023/1024/1025/1103. Strong mutable-input coverage was retained with:

```bash
python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_trace_mutable_buffers.py
```

Result: 2 passed. Both layer kinds replayed batch-32 A/B/A payloads while
overwriting hidden, RoPE, current-position, independently permuted per-user
page tables, and nonzero K/V cache buffers. Eager-control and repeat PCC were
1.0. The final perf captures were rerun as `sliding_v3` and `full_v3` against
the repaired source hash; all profiler and watcher placeholders were replaced
with executable node IDs.

## Static checks and commit log

No C++ or CMake file was changed, so the repository `AGENTS.md` does not
require a build.

```bash
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it \
  --stage functional-decoder --require-contract --strict-caps
```

Result: pass, target 262144 and supported 262144 (full HF context).

The independent final rereview is recorded in `stage_review_rereview.md`; its
verdict is `clean-pass`, with all findings from the initial review fixed and no
new required work. The reviewer performed read-only artifact and source checks
and did not run additional hardware workloads.

```bash
pre-commit run --files \
  models/autoports/google_gemma_4_26b_a4b_it/tt/functional_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  $(find models/autoports/google_gemma_4_26b_a4b_it/doc -type f -print | sort)
```

Result: all applicable hooks passed, including Black/isort/autoflake, JSON/YAML
checks, whitespace, large-file guard, and repository policy hooks. Stage-review
verdict and local commit SHA(s) are appended after the independent review.
Nothing is pushed.
