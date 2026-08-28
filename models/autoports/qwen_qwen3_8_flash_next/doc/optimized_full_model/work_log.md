# Optimized full-model work log

Date: 2026-08-28 America/New_York

## Scope, checkout, and hardware

This stage optimized the completed `Qwen/Qwen3.8-Flash-Next` TTNN model and
generator on P300 Blackhole dies 0 and 1. It did not start vLLM. Source base is
`a38187012fbc55714487c3d53a78a848809c8cf1` on branch
`hous/qwen3.8-flash-next`. Final implementation/evidence runs are tied to
source digest
`e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`.

Every device process used:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

The ttenv default descriptor is P150, so the P300 override is mandatory.
Preflights were healthy. Watcher/allocation tracking and Tracy ran in separate
processes. No reset or push was performed.

## Baseline and retained sequence

Baseline command:

```bash
RUN_QWEN38_PERF=1 timeout 2400 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_batch1_prompt128_generate128_performance \
  --junitxml=.../optimized_full_model/baseline_batch1_prompt128_generate128.xml
```

The 128+128 baseline passed: TTFT 123.909671 s, token-out 644.562319
ms/token, and 1.551440 t/s/u.

Retained changes, in order:

1. Front/back TT trace segments replay nonblocking on CQ0. Reduced/full gates
   passed and decode improved to 605.188601 ms/token.
2. Each layer/rank owns an immutable exact-zero expert. A miss H2D-writes only
   the deterministic EP2 owner and locally D2D-resets the peer. Prompt-128 H2D
   halved from 60.792 to 30.396 GB and decode improved to 454.810104 ms/token.
3. Per-miss device fences were removed. Owner H2D, zero D2D, index publication,
   and the consuming back trace remain CQ0-ordered; completion occurs at the
   next compact route/token boundary.
4. The PLE whole-mesh upload fence was removed. The contiguous host source is
   retained through the next exact route boundary; `close()` provides the final
   safety sync.
5. Optional completed per-token timing records wall, compact read, model/layer,
   expert, route-stall, cache/DMA submission, trace, PLE, and counter deltas.
6. Full construction preloads all 24,576 exact packed experts. The final frozen
   run used 68,080,435,200 bytes and 240.978958 s; steady decode source pack is
   zero.
7. A real-weight endpoint frontier selected BFP8/HiFi2 interleaved LM-head
   weights without changing logits ordering or sampling semantics.

The inherited GDN BFP8/HiFi2, routed/shared/QSA dtype and fidelity, KV/cache,
activation, CCL, program, kernel, and residual policies remain selected. GDN
LoFi was not reopened: retained component evidence has different route/miss
work and no full-model accuracy gate, while broad Pareto selection belongs to
`$datatype-sweep`.

## Final frozen performance

Reportable no-timeline command:

```bash
RUN_QWEN38_PERF=1 QWEN38_COLLECT_DECODE_TIMELINE=0 \
QWEN38_PREPACK_ALL_EXPERTS=1 QWEN38_LM_HEAD_POLICY=bfp8_hifi2 \
QWEN38_EVIDENCE_DIR=.../optimized_full_model/final_frozen_performance_no_timeline \
timeout 2400 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_batch1_prompt128_generate128_performance \
  --junitxml=.../optimized_full_model/final_frozen_batch1_prompt128_generate128_no_timeline.xml
```

Result: TTFT 7.856141 s; 126 measured traced tokens in 29.167731 s;
231.489929 ms/token; 4.319842 t/s/u. Runtime fallback audit is clean, greedy
sampling is device-side, trace replays are 126, and compact readbacks are 128.

The identical selected workload with
`QWEN38_COLLECT_DECODE_TIMELINE=1` is retained separately under
`final_frozen_performance/`: TTFT 24.665760 s; 261.457291 ms/token; 3.824716
t/s/u; p50/p95 262.370/316.480 ms. Timeline totals over 126 tokens are 41,607
misses, 18,873 hits, 115,035,033,600 owner H2D bytes, equal peer-zero D2D,
206,840 index bytes, 680 PLE table rows / 217,600 bytes, zero source pack, and
zero expert/PLE completion syncs.

Timeline means are 259.063 ms model submit, 258.795 ms layer boundary, 252.136
ms expert service, 100.052 ms compact route/device stall, 151.830 ms
cache/control/DMA submit, 0.446 ms trace submit, 5.656 ms PLE, and 2.366 ms
compact caller read/completion. These instrumented numbers are not mixed into
the warmed headline.

Frozen cold/warm exact-store command:

```bash
RUN_QWEN38_HOST_COLD_WARM=1 timeout 1800 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_cold_and_warm_chunked_prefill \
  --junitxml=.../optimized_full_model/final_frozen_cold_warm.xml
```

Cold prompt-128 is 127.019308 s with 10,994 misses, 118.090981 s source pack,
1,816 PLE rows, and token 248046. Warm is 5.460850 s with 10,994 packed hits,
zero pack, zero PLE reads, and the same token.

## Endpoint and sampling AutoFix

Focused endpoint command:

```bash
RUN_QWEN38_FULL_MODEL=1 QWEN38_LM_HEAD_BENCH=1 \
QWEN38_LM_HEAD_POLICY=<policy> timeout 300 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_reduced_real_weight_embedding_to_terminal_gather_smoke \
  --junitxml=.../optimized_full_model/<artifact>.xml
```

Final selected BFP8/HiFi2 interleaved is 0.972307 ms/row, PCC .999922,
top-5 5, top-100 98, exact device greedy, and valid top-k/top-p. DRAM s1/c40
and s4/c40 fail exact L1/CB compile gates. DRAM s5/c40 passes PCC, exact greedy,
top-k/top-p, and a 33-row fallback but takes 1.330563 ms, 36.24% slower.
BFP4/LoFi now typecasts TILE logits to BF16 before sampling; it executes both
samplers but fails accuracy at PCC .976245, top-5 4, top-100 67.

Frozen greedy strategy A/B: specialized full-vocabulary device argmax is
0.664160 ms; correct local-top32/k=1 is 0.898973 ms. Both match host argmax, so
the faster specialized device path remains selected. There is no force-argmax
workaround.

## Host/cache AutoFix and lower bound

Focused service command pattern:

```bash
QWEN38_HOST_MISS_WAVE_POLICY=<policy> QWEN38_HOST_STAGING_DEPTH=<depth> \
RUN_QWEN38_HOST_DMA_BENCH=1 timeout 300 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_completed_cache_service_bandwidth \
  --junitxml=.../optimized_full_model/<artifact>.xml
```

Canonical serial depth 1 passes exact even/odd EP2 owner and peer-zero
readback: 6.820373 GB/s; enqueue p50/p95 3.350906/3.584733 ms; wait
0.697171/0.763217 ms; complete 4.051894/4.077123 ms. Serial depth 10 has no
p50 win and costs 1.1943936 GB/rank across 48 layers. Owner-partitioned depths
1/2/10, coalesced depth 10, and prestarted threaded depth 10 are all slower;
threaded p50/p95 regress 5.315/6.349%.

Pure completed probes separate physical staging from cache service:

- single owner, exactly 2,764,800 bytes: 6.127343 GB/s, p50/p95
  0.447554/0.461020 ms, 77.789% of one raw x4 link;
- concurrent owners, exactly 5,529,600 bytes: 6.497517 GB/s, p50/p95
  0.827907/0.896589 ms, 41.244% of the x4+x4 raw ceiling.

The current service result is not called physical bandwidth. For 912,976,457
owner bytes/token, raw 15.753846 GB/s gives a 57.952608 ms optimistic physical
floor. The decoder stack is 111.817691 ms, selected LM-head plus greedy floor
is 1.387908 ms, and PLE is 5.655693 ms, giving 176.813899 ms/token. Observed
231.489929 ms is 54.676029 ms or 30.923% above that optimistic bound.

At measured pure-dual staging bandwidth, transfer alone is 140.511592 ms/token.
The additive stack+terminal+PLE+transfer diagnostic is 259.372884 ms, 27.883 ms
above observed, showing real overlap. AutoFix exhausted exact Python grouping,
depth, coalescing, and threading candidates without a speedup. The next credible
boundary is native batched TTNN H2D submission, one call/event per owner.

## Accuracy, serving state, and safety

Frozen AIME command:

```bash
RUN_QWEN38_ACCURACY=1 QWEN38_TEACHER_ROWS=99 \
QWEN38_EVIDENCE_DIR=.../optimized_full_model/final_frozen_accuracy \
timeout 2400 pytest -q -s --tt-arch blackhole \
  .../test_full_model.py::test_full_model_aime24_prefill_accuracy \
  .../test_full_model.py::test_full_model_aime24_teacher_forcing_accuracy \
  .../test_full_model.py::test_full_model_aime24_autoregressive_quality \
  --junitxml=.../optimized_full_model/final_frozen_aime_prefill_teacher_autoreg.xml
```

Prefill at non-aligned length 201 is top-1/top-5/top-100 100/100/100. Teacher
forcing is 91.919/100/100 over 99 rows and measures 98 traced rows at 289.150
ms/token / 3.458 t/s/u. It explicitly reads full logits and excludes sampling,
token feedback, and compact-token readback. The 100-token free-run is coherent,
traced, non-degenerate, and measures 224.634 ms/token / 4.452 t/s/u.

The frozen shared three-prompt qualitative suite passes. Split greedy,
top-k/top-p, mixed non-aligned prompts, changed-only page tables, fixed slots,
and inactive rows pass. All-48 eager batch 32 with active slots 0 and 31,
prompt lengths 1 and 33, 30 inactive rows, distinct page tables, reset/reuse,
and request-isolated PLE history passes. Context 262144 construction passes.
The split/mixed XML has four passes plus one environment-gated sampler A/B
skip; the skipped node is rerun with its required environment and passes in
`final_frozen_sampler_strategy_ab.xml`, completing the five-case contract.

Final frozen Watcher/allocation command:

```bash
TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACKING_TRACEBACKS=1 \
TT_METAL_TRACE_ALLOC_TRACKING_TRACEBACK_DEPTH=12 \
RUN_QWEN38_FULL_MODEL=1 timeout 1200 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_48_layer_token_out_trace_smoke \
  --junitxml=.../optimized_full_model/final_frozen_full48_watcher_alloc_tracker.xml
```

The full-48 path passes with no Watcher, NoC, CB, assert, hang, or allocation
fault. Frozen static contracts pass 36/36 in `final_frozen_static_contracts.xml`.

## Profiler

Four frozen-source Tracy captures use:

```bash
RUN_QWEN38_FULL_MODEL_PROFILE=1 \
QWEN38_FULL_MODEL_PROFILE_MODE=<decode|prefill> \
QWEN38_FULL_MODEL_PROFILE_LAYERS=<0|1|3> \
timeout 2400 python -m tracy -p -r --op-support-count=2000 \
  --dump-device-data-mid-run --check-exit-code \
  -o .../optimized_full_model/final_frozen_profiler/<name> \
  -m pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model_perf.py::test_reduced_full_model_profile_window
```

Layer-0 GDN, layer-1 PLE+GDN, layer-3 QSA decode, and warm layer-0 prefill all
pass. `tt-perf-report` runs with advice. Decode device totals are 3.213, 3.632,
and 4.574 ms; sampling is 0.491 ms and LM-head rows are 0.895-0.899 ms. Prefill
device time is 5.736 ms. Exact commands, hashes, roofline conclusions, compact
tables, and raw CSV provenance are in `profiler_provenance.txt` and
`final_frozen_profiler/`.

## Contracts, limitations, review, and commits

`doc/context_contract.json` and `doc/host_weight_contract.json` retain context
262144 and batch-32 eager state. Planned totals are 10,005,165,144 bytes/device
at batch-1 max context and 23,393,673,304 bytes/device at batch-32 sequence
4096. No advertised capability was reduced.

Limitations: model-load prepack costs about 241 s and 68.080 GB host RAM;
segmented traced token-out is batch 1 while eager serving state is batch 32;
CPU-only Torch has no pinned allocator; Python concurrent-owner submission
reaches only 41.24% of raw two-link bandwidth; and CCL emits a future API
warning. Native batched H2D is the remaining credible host-boundary lever.

The first independent review returned `more-work-needed`; its findings drove
the LM-head DRAM/BFP4 frontier, fresh frozen correctness/performance/profiler
evidence, corrected teacher-forcing label, source hashing, and the lower-bound
rewrite.

The final fresh-context `$stage-review` rereview returned `clean-pass` with no
required work. `STAGE_REREVIEW.md` records independently rederived metrics,
source-freshness checks, prior-finding disposition, anomaly ledger, and residual
risk. Its only packaging concern was a stale pre-frozen artifact manifest; the
manifest was regenerated against the final selected package before commit.

Final packaging validation rechecked the aggregate source digest as
`e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`,
parsed all selected JSON and CSV files, compiled the seven touched Python
files, and reran the exact hardware-free contract selection: 36/36 passed.
After an accidentally broad collection opened only the default single-die
fixture and failed its expected 1-vs-2 mesh check, the process closed devices;
an immediate P300 `TT_VISIBLE_DEVICES=0,1` / P300-descriptor `tt-smi -s`
confirmed both dies healthy before the exact static rerun.

Stage implementation, contracts, canonical evidence, profiler tables, and
independent rereview were committed locally as
`8a7ed4dbfb6` (`Optimize Qwen3.8 full-model mesh path`). The commit used explicit
pre-commit skips for Black/isort and whitespace/EOF normalization because those
hooks would change the frozen source digest or retained generated evidence,
and for the 500 KiB limit because the requested raw profiler/JUnit/JSON evidence
is intentionally larger. All remaining applicable hooks passed. This work-log
and SHA-256-manifest ledger are committed separately. No push is made.
