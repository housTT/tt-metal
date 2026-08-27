# Multichip decoder work log

Stage input commits: optimized implementation `d7baac46495` and optimized
decoder signoff `bed265a9c76`.

## 2026-08-27: environment and target lock

- Confirmed devices 0 and 1 are the two Blackhole dies of one P300 board.
- Confirmed the fixed `1x2` mesh, `FABRIC_1D`, 11x10 compute grid, eight DRAM
  channels, and 31.875 GiB nominal DRAM per die.
- The first unsourced probe mixed an installed TTNN with checkout JIT sources.
  `$autodebug` identified the provenance skew; `AUTODEBUG.md` and `AUTOFIX.md`
  retain the proof and corrected repo-local invocation.
- Locked TP2 QSA/MoE with a replicated 10240-wide public boundary.  The tensor,
  activation, cache, state, CCL, MoE, padding, and capacity plan was recorded in
  `mesh_plan.md` before the final implementation.

Common environment for every retained hardware command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

## Implementation and isolated repair

- Added `tt/multichip_decoder.py`, subclassing `OptimizedDecoder`.
- Added setup-only rank-local checkpoint slicing and rank-1 device-to-device
  patching; no runtime Torch conversion.
- Sharded QSA to 12 query/one KV head per die and MoE to local intermediate
  width 320.  Router/indexer stay replicated; all 512 experts remain available
  and execution stays gate-selected top-10.
- Initial 24-head-per-rank GDN decode missed PCC at about 0.992.  State copies,
  ordering, FP32 projection, and FP32 CCL were refuted.  Replicating full GDN
  restored exact optimized kernel geometry and final decode PCC above 0.999999.
- Rejected an L1 all-reduce output candidate after it produced rank divergence
  in combined correctness/trace coverage.
- Selected BF16 row partials/reductions in DRAM and BFP8 QSA caches.  BFP8 cache
  saves 1.7578125 GiB/die at maximum context and passes direct BF16-control PCC.
- Added structural, real-weight PCC, paged cache, state-copy, non-aligned,
  batch-32, stacked-layout, fallback, maximum-context, and trace determinism
  tests plus the real-checkpoint performance/profiler workload.

## Final correctness and trace gates

Command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole \
  --capture=tee-sys -o junit_logging=all -m 'not long_context' \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/final_correctness.xml
```

Result: `28 passed, 1 deselected` in 209.79 seconds.  A later CPU-only
`static_contracts.xml` run covers 19 static/structural checks, including the
extra 262,143/262,144 chunk-plan cases now present in the source.
Representative hardware evidence:

| Layer | Prefill PCC | Decode PCC | Warm/replay PCC |
| --- | ---: | ---: | ---: |
| 0 GDN | 0.99892092 | 0.99999958 | 1.00000000 |
| 1 PLE+GDN | 0.99943250 | 0.99999976 | 1.00000048 |
| 3 QSA | 0.99885875 | 0.99932384 | 0.99999964 |

Each trace test replayed five times and required bitwise-identical repeated
outputs.  Batch-32 passed for all three layer kinds; the stacked layer-0 ->
layer-1 -> layer-3 contract passed; the runtime fallback guard was active.

Maximum-context command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole --long-context \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_qsa_trace_at_advertised_context \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/advertised_context_trace.xml
```

Result: one pass in 11.73 seconds at context 262,144/current position 262,143,
with last-page shuffled addressing and trace replay.

## Latency

Single-chip baseline artifact: `singlechip_perf_count7.xml`.
Multichip artifact: `multichip_perf_count7.xml`.

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 pytest -q --tt-arch blackhole --count=7 \
  --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/multichip_perf_count7.xml
```

Result: 21 passes.  Seven-sample medians:

| Layer | Single prefill | Multi prefill | Speedup / efficiency | Single trace decode | Multi trace decode | Speedup / efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 22.461979 ms | 19.057457 ms | 1.178645x / 58.932% | 1.090756 ms | 1.099877 ms | 0.991707x / 49.585% |
| 1 | 25.621410 ms | 22.127126 ms | 1.157919x / 57.896% | 1.455516 ms | 1.478821 ms | 0.984241x / 49.212% |
| 3 | 48.602010 ms | 32.853951 ms | 1.479335x / 73.967% | 2.867152 ms | 2.880215 ms | 0.995465x / 49.773% |

## Tracy and `tt-perf-report`

Watcher was off.  An initial combined traced capture passed its workload but
post-processing could not correlate one device-1 trace ID.  A combined direct
capture then overflowed profiler marker buffers.  The retained solution uses
one direct-decode process per layer and explicit `ttnn.ReadDeviceProfiler`
checkpoints outside signposted windows:

```bash
QWEN38_MC_PROFILE_DIRECT_DECODE=1 \
QWEN38_MC_PERF_DECODE_REPLAYS=1 \
QWEN38_MC_PERF_LAYERS=<0|1|3> \
python -m tracy -r --check-exit-code -o <layer-output> \
  -m pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py
```

All three profiled tests and post-processors passed.  Each raw
`ops_perf_results_*.csv` was analyzed twice:

```bash
tt-perf-report <ops.csv> \
  --start-signpost MC_PERF_<PREFILL|DECODE>_Lx \
  --end-signpost MC_PERF_<PREFILL|DECODE>_Lx_END \
  --no-color --no-host-ops --active-experts <routing-union|10> \
  --csv <detailed.csv> --summary-file <summary.csv>
```

The reports are under `tracy/layer0_gdn`, `tracy/layer1_ple_gdn`, and
`tracy/layer3_qsa`.  Major findings are recorded in `README.md`: sparse MoE and
reshape traffic dominate GDN prefill; SDPA+MoE dominate QSA prefill; cache/index
gathers dominate QSA decode; one/two fixed fabric rows explain flat decode
speedup.  Detailed reports preserve custom-op warnings where the report tool
cannot categorize Qwen GDN/index kernels.

Raw provenance SHA-256:

```text
ce82fe9035b8298fde583534872f672bcc7b652d06ff4dc32858f3638a0a6525  layer0 ops CSV
6e1d82074c780e095ff2bcbaa0dbd8591d134474a5d2e9498a7638c8dab91025  layer1 ops CSV
2a6bda51846b97483031d5ab0b0fb5b0ed6470fb5bd3f2a1ab147c74e6758e3a  layer3 ops CSV
203887c817b31fc392907d288111b3c63d9fe25fa66d7533fdf4212f50d574a6  single-chip count7 XML
0be77da82910d4a4d1c68466ae5557d77aaa8801b5e6c820fdaf48353c50e129  multichip count7 XML
```

## Watcher and recovery

The first all-tests watcher run made fabric ERISC `(29,25)` fail to return to
base firmware at fixture teardown; subsequent hardware setups failed before
model execution.  A board reset (`tt-smi -r 0 1`) restored healthy DRAM,
firmware heartbeat, and a clean bounded 1x2 mesh open/close.  A single-fixture
stacked-layer watcher run passed the workload and all watcher polls, but with
Ethernet watcher instrumentation enabled the same ERISC teardown failed after
pytest had written a passing XML.

The supported retained mode sets `TT_METAL_WATCHER_DISABLE_ETH=1`: Tensix,
dispatch, NoC/CB sanitization, asserts, stack usage, and waypoints remain
watched; fabric ERISC is not instrumented.  Fabric is independently covered by
the correctness, trace, and profiler runs.  Failed recovery artifacts are
`watcher_pre_reset_failure.xml` and
`watcher_eth_enabled_teardown_failure.xml`; they are retained deliberately.
The clean artifact is `final_watcher.xml`, with raw log under `watcher_final`.

## Full-stack capacity and stage outcome

Standard TP2 BFP4 experts require 31.640625 GiB/die.  With max-context caches,
non-expert weights (including replicated GDN), and a 1 GiB reserve, the
optimistic packing limit is only 53.081868% BFP4 tiles.  Uniform/heavy BFP2,
low-error mixed precision, residual quantization, and low-rank residual tests
all missed PCC.  EP2, pipeline placement, smaller context, and host streaming
do not meet the resident full-stack contract.  Exact trials are in
`AUTOFIX.md` and `mesh_plan.md`; raw probe outputs and command-execution
provenance are retained in `capacity_candidate_probes.log`.

The final independent capacity audit also charged the compressed kernel's
640-to-768 gate/up bank padding.  That raises the physical routed-expert count
to 66,846,720 tiles/die and lowers the practical BFP4 ceiling to 32.131060%;
at least 67.868940% of the physical tiles must be BFP2/zero.

The per-layer path is correct and measured, but the full-stack residency gate
is **autofix-failed** on this fixed board.  Therefore this work must not be
marked pipeline-complete, and a clean stage-review cannot be claimed for the
original goal.  A local commit may record this blocked evidence state, but it is
not a completion signoff.  No remote operation or push was performed.

The first independent stage review found the physical blocker plus one
repairable evidence-provenance gap.  `capacity_candidate_probes.log` fixed the
gap and was added to the SHA-256 manifest.  Rereview found no remaining
repairable gap and returned `more-work-needed` solely for the resident-stack
capacity impossibility; `STAGE_REVIEW.md` records that final verdict.

## Local commits

- `07a1a588e27`: blocked multichip implementation, tests, documentation, and
  retained correctness/performance/watcher/capacity evidence.

This SHA records a blocked stage, not a pipeline-complete signoff.  No push or
other remote operation was performed.

## 2026-08-27 resume-1: exact host backing and trace blocker

The resume used `$host-weight-cache` to replace the resident-stack failure with
an exact bounded path:

- `tt/host_weight_cache.py` mmap-loads one checkpoint expert at a time, packs
  exact TP2 rank-local `[1,1,2560,640]` gate/up and
  `[1,1,320,2560]` down matrices, and uploads BFP4 only on slot misses.
- Each layer has ten generation-checked expert slots and one fixed upload
  staging pair per rank.  Full-stack device cost is 729,907,200 bytes/die.
- Prefill partitions the unique routed-expert union into bounded waves of ten;
  decode uses ordered slots 0--9 so captured back-trace addresses are stable.
  Both retain gate-selected active-expert execution.
- Layer 1 uses the real 128-shard, 320,001,446-logical-row PLE table.  EOS-aware
  HF-equivalent n-gram hashing, isolated two-token histories, reset/cancel,
  exact selected-row mmap, and stable prefill/decode staging are implemented.
- The max-context host-backed plan is 8,066,785,280 bytes/die and leaves
  26,158,735,360 bytes/die of planned headroom.  Exact arithmetic and the
  declared boundary are in `../host_weight_contract.json`.

CPU contract command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py \
  -m 'not requires_device' --disable-warnings --maxfail=1
```

Latest hook-conforming rerun: `11 passed` in 3.28 seconds.  Cases include cold/hit/partial eviction,
capacity-one thrash, reload, stale generations, failed-upload invalidation,
ordered trace replacement, bounded prefill waves, HF row-id parity, real PLE
rows, EOS/chunk carry, reset/cancel, request isolation, non-aligned masks, and
row-cache hits.

Previously completed hardware gates retained from the resume session:

- layer 0 host-backed decode versus resident TP2: PCC >= 0.995, exact ranks;
- layer 1 exact real-PLE prefill/decode versus resident TP2: PCC >= 0.995;
- layer 3 paged non-aligned prefill/decode versus resident TP2: PCC >= 0.995,
  with page table and local KV/index cache checks.

The QSA segmented trace control was rerun after correcting its test geometry
from 128 to 4096 context tokens:

```bash
pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  --disable-warnings --maxfail=1 -s
```

Result: one pass in 4.39 seconds on the healthy 1x2 P300.  Four changing hidden
inputs/positions matched direct TTNN routes and output PCC >= 0.995; final
shuffled-page KV and index caches were exact.  The expert-cache request count
was five (one warm service, one captured token, three external replays).
The retained allocation-tracked rerun passed in 5.53 seconds and is
`host_backed_qsa_segmented_trace_alloc.xml`.

Retained resume artifacts:

- `host_weight_cpu.xml`: 11 CPU exact-cache/PLE contract passes;
- `host_backed_static.xml`: 19 memory/shape/fallback/non-aligned passes;
- `host_backed_correctness.xml`: layer 0, layer 1, and layer 3 host-backed
  hardware correctness passes;
- `host_backed_qsa_segmented_trace_alloc.xml`: changing-input QSA segmented
  trace under `TT_METAL_TRACE_ALLOC_TRACKING=1`;
- `host_backed_gdn_rejection.xml`: direct layer-0 correctness plus explicit
  rejection of the known-corrupt trace mode;
- `SEGMENTED_TRACE_AUTOFIX.md`: isolated GDN failure matrix and exact PCC;
- `evidence_manifest.sha256`: SHA-256 provenance for these artifacts.

### Progressing GDN trace AutoFix

The original resident trace determinism test reset GDN state on every replay.
A new layer-1 gate compared four genuinely progressing tokens
`(23, 91, EOS=248044, 7)` against the eager host-backed TP2 oracle.  Tokens
0--2 were exact; token 3 route ids diverged and output PCC fell to 0.92648160.
Final recurrent PCC was 0.82660490 and the middle FIR tap was -0.01665942,
while all PLE state and the staged newest row remained exact.

`$autofix` isolated and refuted stable source staging, alternate output write,
DRAM destination residency, attention/router split traces, a tiny state trace,
post-back commit, warmed eager D2D commit, and canonical DRAM shadow hydration.
The code now rejects GDN segmented capture explicitly; all failed experimental
paths were removed.  Full observations and the variant matrix are in
`SEGMENTED_TRACE_AUTOFIX.md`.

This is a hard current-runtime blocker for the goal's warmed decode trace gate:
36 of 48 layers use GDN.  Consequently host-backed latency/efficiency,
tt-perf-report acceptance, watcher signoff, and a clean stage-review were not
claimed.  The earlier resident profiler tables remain useful baseline evidence
but are not mislabeled as end-to-end host-backed measurements.

### Resume-1 stage review

The first independent rereview found one repairable accounting contradiction:
the code memory plan omitted 819,200 bytes/die of stable PLE staging that the
host-weight contract already charged.  The code and static test were fixed,
the full-stack plan became 8,066,785,280 bytes/die, and the 19-test static XML
and SHA-256 manifest were regenerated and verified.

The final rereview returned `more-work-needed` solely for the progressing GDN
trace blocker above.  It found no remaining repairable finding, other concern,
or hard-check gap.  `STAGE_REVIEW_RESUME.md` records the verdict.  Because
`$autofix` exhausted the isolated baseline-preserving variants, this resume is
a blocked evidence checkpoint rather than a stage completion.

Final `$tt-device-usage` health check at `2026-08-27T11:58:39-04:00` found
both P300c dies with `dram_status=true`, identical live heartbeat `45918`, and
zero corrected or uncorrected GDDR errors.  No reset was required.

Resume implementation/evidence commit: `849400cf6e0`.  This is a local
blocked-state checkpoint, not a pipeline-complete signoff.  No push or other
remote operation was performed.

## 2026-08-27 resume-1 continuation: final trace and stack resolution

This section supersedes the earlier GDN-blocked outcome above.  The historical
failure evidence remains because it motivated and bounds the accepted fix.

### Accepted GDN state repair

The earlier captured copy into the newest FP32 FIR tap was isolated as the
only corrupt progressing state.  `$autofix` tested stable sources, copy/add,
DRAM tap residency, relocation guards, state commit traces, host/eager commits,
DRAM shadows, PLE/GDN split, and newest-tap ping-pong.  None was accepted.

The final graph keeps `OptimizedDecoder._gdn_decode` unchanged for its output
and ordinary state update, repeats the already optimized packed FP32
`gdn_qkv_b_a` projection, and makes
`ttnn.slice(..., output_tensor=newest_tap)` the final corrective writer.  In
host-backed decode that target is the fixed newest L1 workspace tap and is then
committed to the stable layer-owned canonical DRAM tap; both keep their
identity, address, FP32 TILE layout, and memory configuration.  Batch greater
than one still uses the unmodified optimized path.
`direct_state_correction.xml` passes real optimized-baseline PCC plus exact
eight-token layer-0/layer-1 progressing state, route, output, address, and
allocation checks.  `SEGMENTED_TRACE_AUTOFIX.md` contains the full hypothesis
matrix and rejected PCCs.

### Canonical state and multi-live trace allocation safety

Batch-one host-backed GDN/PLE state is canonical in DRAM.  One
`MultichipDecodeStateWorkspace` supplies fixed L1 recurrent/conv/PLE tensors
to every GDN layer in serialized model order.  The complete state accounting
is 260,702,208 bytes/die decode-canonical plus 208,928,768 bytes/die additional
prefill/user state.  The shared workspace peak is 120,832 bytes/worker,
12,976,128 actual and 13,291,520 bank-reserved bytes/die.

The first two-live-trace allocation test found seven younger live buffers in
an older trace's captured ranges.  Six were `HostDecodeFront` crossings that
are regenerated before use and are now individually marked corruptible.  The
seventh was a persistent lazy BF16 DRAM Clone program buffer and could not be
marked safely.  The accepted two-phase stack protocol is:

1. call `HostBackedSegmentedDecodeTrace.warm_programs` for every distinct
   capture signature before any trace exists;
2. capture in model order with `programs_prepared=True`, freezing
   program-cache misses before snapshots or warm execution;
3. mark only front crossings, PLE staging, and final output corruptible;
4. serialize capture/replay/release through the shared workspace and release
   back trace before front trace.

The isolated final gate `shared_trace_alloc_autofix.xml` passes.  The full
current-source allocation run below retains two traces simultaneously and
replays both with exact independent state, routes, output, fixed workspace
addresses, and no unapproved live allocation overlap.

### Final correctness, stress, and fallback gates

Common hardware environment remained:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

Allocation-tracked current-source matrix (ten node IDs, 15 parameter cases):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 2400 pytest -q -s \
  --tt-arch blackhole --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_real_weights_match_optimized_baseline \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_batch32_decode_contract \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_stacked_decoder_layout_contract \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_shared_decode_state_workspace_stack \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_shared_workspace_segmented_trace_stack \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_layer0_decode_matches_resident_multichip \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_real_ple_decode_matches_resident_multichip \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_qsa_paged_prefill_decode_matches_resident_multichip \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/final_current_correctness_alloc.xml
```

Result: **15 passed in 125.97 s**.  This includes real-weight prefill/decode PCC
for GDN, PLE+GDN, and QSA; non-aligned seq 33; batch 32; stacked layout;
rank-local paged caches; all exact host-backed layer kinds; short progressing
traces; and the two-live-layer workspace.

Progressing stress:

```bash
QWEN38_MC_TRACE_STRESS_STEPS=100 timeout 7200 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/host_backed_trace_stress100.xml
```

Result: **3 passed in 60.67 s**.  Layer 0, layer 1, and layer 3 compare every
token's state/cache, routes, and output against eager TTNN.  GDN checks all
recurrent and FIR state; PLE checks all nine taps and EOS-aware history; QSA
checks final local KV and index caches.

Final static/fallback command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
pytest -q --capture=tee-sys -o junit_logging=all \
  <six static node IDs including all non-aligned parameter cases> \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/final_static_contracts.xml
```

Result: **20 passed in 4.33 s**.  The source whitelist allows only compact
route D2H and declared expert/PLE staging; TT attention, GDN, selected-expert
math, collectives, trace state movement, and trace-open regions contain no
Torch conversion.  Runtime device tests also run the resident mathematical
regions under `ForbidHostFallback`.

The first independent final review found that the older host-cache CPU XML no
longer covered the newly charged runtime state.  The test now derives the total
from every current contract component and asserts 8,536,416,256 bytes/die plus
the exact headroom identity.  The complete current-source rerun is
`host_weight_cpu_final.xml`: **11 passed in 2.88 s**.

The unchanged advertised-context QSA path remains validated by
`advertised_context_trace.xml`: context 262,144, current position 262,143,
shuffled last page, local cache shapes, and warmed replay.  The new code is
GDN/workspace-local and does not reduce this contract.

### Final end-to-end host-backed latency

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 timeout 7200 pytest -q -s \
  --tt-arch blackhole --count=7 --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/host_backed_perf_count7.xml
```

Result: **21 passed in 187.79 s**.  Exact input hashes and PLE rows are shared.
The single-chip baseline is the unchanged `OptimizedDecoder` graph replicated
on the two-die mesh without CCL; the final TP2 measurement includes PLE,
compact route D2H, exact expert lookup/packing/H2D, both traces, state DRAM
movement, and CCL.

| Layer | Prefill PCC | Baseline prefill | Host prefill | Baseline decode | Host decode | Speedup / efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.99898398 | 16.529560 ms | 702.203924 ms | 1.113063 ms | 1.792194 ms | 0.621061x / 31.053% |
| 1 | 0.99639225 | 20.069470 ms | 973.417516 ms | 1.492335 ms | 3.247860 ms | 0.459484x / 22.974% |
| 3 | 0.99869609 | 44.699946 ms | 1,230.619689 ms | 2.867494 ms | 3.039150 ms | 0.943526x / 47.176% |

The raw evidence is `host_backed_perf_count7.xml` and
`host_backed_perf_count7.log`.  Demand-loaded prefill and token decode are
slower than the resident baseline; the result is accepted because the only
resident alternative is physically impossible and all required host work is
included.

### Capacity-correct Tracy and `tt-perf-report`

Watcher was off.  Each layer used its own process and a 2,000-program support
buffer.  `QWEN38_MC_PROFILE_HOST_ONLY=1` omits duplicate baseline setup only in
the profiler process; baseline latency remains the count-seven result above.

```bash
QWEN38_MC_PROFILE_DIRECT_DECODE=1 \
QWEN38_MC_PROFILE_HOST_ONLY=1 \
QWEN38_MC_PERF_DECODE_REPLAYS=1 \
QWEN38_MC_PERF_LAYERS=<0|1|3> \
python -m tracy -p -r --op-support-count=2000 \
  --dump-device-data-mid-run --check-exit-code -o <capacity2000-layer-dir> \
  -m pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode
```

All three pytest processes and postprocessors passed.  The accepted
`tracy_ops_data.csv` files contain zero `Profiler DRAM buffers were full`
messages.  The default-capacity layer-1 attempt had 1,605 warnings and 33
unmatched ops; it was rejected rather than forcing a partial report.

For each final `ops_perf_results_*.csv`:

```bash
tt-perf-report <ops.csv> \
  --start-signpost MC_HOST_<PREFILL|DECODE>_Lx \
  --end-signpost MC_HOST_<PREFILL|DECODE>_Lx_END \
  --no-color --no-host-ops --active-experts 10 \
  --csv <report.csv> --summary-file <summary.csv>

tt-perf-report <ops.csv> \
  --start-signpost MC_HOST_<PREFILL|DECODE>_Lx \
  --end-signpost MC_HOST_<PREFILL|DECODE>_Lx_END \
  --no-color --no-host-ops --active-experts 10 --no-summary > <table.txt>
```

Accepted roots and profiler XMLs:

- `tracy_host/layer0_gdn_capacity2000`, `tracy_host_layer0_capacity2000.xml`;
- `tracy_host/layer1_ple_gdn_capacity2000`, `tracy_host_layer1_capacity2000.xml`;
- `tracy_host/layer3_qsa_capacity2000`, `tracy_host_layer3_capacity2000.xml`.

The human tables and summary CSVs support the communication/DRAM/compute/data
movement findings in `README.md`.  Four failed/overflowed generated profiler
trees were moved to the desktop trash after their small XML records were
retained; they are recoverable and are not accepted evidence.

### Final watcher

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN38_MC_TRACE_STRESS_STEPS=8 timeout 3600 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_shared_workspace_segmented_trace_stack \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/final_host_backed_watcher.xml
```

Result: **4 passed in 25.54 s**.  The 556-line raw log has SHA-256
`096578a1c870c06d6e7c05616879dc470757f49281b30f01faefa79762853e6c`
and no watcher error/assertion.  Watcher and profiler were never enabled in
the same process.  After every failed device/profiler run, both dies were
reset before the next command.

### Final capacity and stage state

The complete plan is now 8,536,416,256 bytes/die including canonical decode
state and additional prefill/user state, leaving 25,689,104,384 bytes/die.
The advertised 262,144-token capability is unchanged.  The delivered decoder
is suitable as the full-model layer-stack baseline, subject to the explicit
batch-one host-backed and measured performance limitations.  Full-model and
vLLM work remain outside this stage.

## 2026-08-27 final-source Autofix and infrastructure recovery

The first count-seven run after persistent fractured-residual integration was
not accepted.  It produced 14 passes and seven deterministic layer-0 failures:
layers 1 and 3 passed, while every layer-0 repetition measured prefill PCC
`0.9911649227142334` against the optimized baseline (required minimum `0.995`).
The current input is the helper's `635cca15...` activation, so the historical
layer-0 `0.99898398` rows with hash `94aa5687...` are not a same-input control.
The failure remains an open Autofix item; the gate was not lowered.

After that failed device run, the required reset/recovery sequence was run.
The first reset targeted logical IDs `0,1`; the second bounded reset targeted
the complete P300 PCI board.  Both chips continued to enumerate, but physical
fabric discovery reported intra-mesh degree `0:2`, and the required bounded
`ttnn.open_mesh_device(ttnn.MeshShape(1, 2))` smoke failed before model code.
The subsequent focused diagnostic XML therefore records infrastructure setup
failure only, not model evidence.  Per `tt-device-usage`, further TT commands
are paused until a host/board reboot restores the P300 link.

A second-turn health audit at 16:14 ET still enumerated both local chip IDs
inside UMD, but again reported physical intra-mesh degree `0:2`; the bounded
1x2 mesh smoke failed with the same topology-mapper error.  No additional
reset was attempted after the skill-mandated two-reset recovery had already
failed.

The third consecutive goal-turn audit at 16:15 ET produced the identical
result: UMD opened local chip IDs `{0,1}`, physical discovery reported
intra-mesh degree `0:2`, and strict mapping rejected the required 1x2 mesh.
This satisfies the blocked-audit threshold.  Resume this same stage after a
physical host/board reboot or reservation re-acquire restores the inter-die
link; the exact Autofix ladder and all prior artifacts remain in the worktree.

The prepared Autofix ladder uses the exact logical 33-row input in separate
processes: optimized baseline versus resident replicated TP2, resident
replicated TP2 versus resident fractured TP2, and resident fractured TP2 versus
host-backed fractured TP2.  The first failing edge will then be localized at
the HC, GDN state/output, router/top-k, and MoE boundaries before any production
precision or bridge change is retained.

## 2026-08-27 routed-expert EP2 resolution and final signoff

The P300 inter-die link was restored before this continuation. `tt-smi -s`,
the fixed `TT_VISIBLE_DEVICES=0,1` mapping, and a bounded 1x2 mesh open all
passed. No additional reset was needed. All TT commands below were serialized.

### Fresh AutoDebug and isolated AutoFix

The fresh xhigh AutoDebug report is `FINAL_SOURCE_AUTODEBUG.md`. Gated
diagnostics proved that HC, GDN, router/top-k, expert weights, and packed
gate/up were not the first failing edge. The routed down projection's
K=320+320 partial sum was about 0.926 PCC and routed output about 0.938 PCC.
HiFi2, FP32 down output, fused precision, and baseline routing substitution
were refuted.

The Autofix candidate placed a full K=640 routed expert on
`expert_id % 2` and an exact-zero full-shaped slot on the peer. The old g20
program still failed layer-0 prefill at 0.99064916
(`expert_ep2_layer0_isolated.xml`, `expert_ep2_boundary.xml`). Changing only
the full-width program to g40 passed at 0.99942303
(`expert_ep2_g40_boundary.xml`). `FRACTURED_EXPERT_AUTOFIX.md` records the
complete retained/refuted ladder.

The corrected per-rank expert charge is 2,764,800 bytes. Ten slots plus one
staging slot across 48 layers consume 1,459,814,400 bytes/die. The reconciled
full plan is 10,103,303,168 bytes/die with 24,122,217,472 bytes/die headroom.

### Final real-checkpoint PCC

`expert_ep2_first_decode_pcc.xml` passed all representative layer kinds:

| Layer | Prefill PCC | First-token decode PCC |
| --- | ---: | ---: |
| 0 | 0.99942303 | 0.99996978 |
| 1 | 0.99949104 | 0.99984211 |
| 3 | 0.99972457 | 0.99988294 |

Decode PCC is checked at the first equally initialized token. Later recurrent
tokens are compared step-for-step with eager TTNN by the progressing trace
tests; comparing different recurrent time indices is not a valid oracle.

Exact owner/zero slots, exact PLE rows, and paged QSA local-cache topology
passed in `expert_ep2_hardware_contracts.xml` (3 passed in 32.86 s).

### Final allocation, fallback, paging, and stress gates

Common hardware environment:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

The allocation-tracked acceptance matrix used the batch-32/layout controls,
all three host-backed optimized-reference comparisons, shared eager/trace
workspaces, and progressing GDN/QSA traces under
`TT_METAL_TRACE_ALLOC_TRACKING=1`. Result:
`expert_ep2_final_correctness_alloc.xml`, **12 passed in 224.56 s**.

The legacy resident routed split-K real-weight test is intentionally excluded
from acceptance because AutoDebug proved it is the rejected numerical path.
The delivered host-backed EP2 path is compared directly with
`OptimizedDecoder`.

The complete host-cache/PLE/capacity/static/fallback/non-aligned matrix is
`expert_ep2_final_static_contracts.xml`: **32 passed in 12.42 s**. It covers
logical lengths through 262,143/262,144 and derives the final capacity sum
from current constants.

The advertised-context QSA command used `--long-context` at context 262,144,
position 262,143, and the fixed 1x2 mesh. Result:
`expert_ep2_advertised_context_trace.xml`, **1 passed in 19.41 s**.

Progressing stress:

```bash
QWEN38_MC_TRACE_STRESS_STEPS=100 timeout 7200 pytest -q -s \
  tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  --junitxml=doc/multichip_decoder/expert_ep2_trace_stress100.xml
```

Result: **3 passed in 90.26 s**. Both GDN kinds and paged QSA compare output,
routes, and all relevant state/cache after every changing token.

### Final count-seven performance

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 timeout 7200 pytest -q -s \
  --count=7 \
  tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode \
  --junitxml=doc/multichip_decoder/expert_ep2_perf_count7.xml
```

Result: **21 passed in 322.92 s**. Median real-checkpoint results:

| Layer | Baseline prefill | Host EP2 prefill | Baseline decode | Host decode | Speedup / efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 18.693426 ms | 3,583.877886 ms | 1.106864 ms | 3.766486 ms | 0.293795x / 14.690% |
| 1 | 20.052107 ms | 1,541.191421 ms | 1.484079 ms | 4.184993 ms | 0.354558x / 17.728% |
| 3 | 43.731704 ms | 2,064.964676 ms | 2.861884 ms | 3.050174 ms | 0.938269x / 46.914% |

The measured path includes route D2H, exact checkpoint service and packing,
owner/zero H2D, PLE, both trace segments, canonical-state movement, and CCL.

### Final Tracy, `tt-perf-report`, and watcher

Watcher was off. Each representative layer used a fresh process with
`QWEN38_MC_PROFILE_DIRECT_DECODE=1`, `QWEN38_MC_PROFILE_HOST_ONLY=1`, one
decode replay, and `python -m tracy -p -r --op-support-count=2000
--dump-device-data-mid-run`. All pytest processes and postprocessors passed:

- `tracy_host_ep2_layer0.xml`, `tracy_host_ep2/layer0_gdn_capacity2000`;
- `tracy_host_ep2_layer1.xml`, `tracy_host_ep2/layer1_ple_gdn_capacity2000`;
- `tracy_host_ep2_layer3.xml`, `tracy_host_ep2/layer3_qsa_capacity2000`.

All three `tracy_ops_data.csv` files contain zero profiler-buffer overflow
messages. `tt-perf-report` accepted each `MC_HOST_PREFILL_Lx` and
`MC_HOST_DECODE_Lx` window. Modeled DRAM rooflines were 119/77 GB/s for L0,
94/75 GB/s for L1, and 64/34 GB/s for L3 prefill/decode.

Profiler was then disabled. The final watcher command used
`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_DISABLE_ETH=1`, and eight progressing
steps for GDN, PLE+GDN, paged QSA, and the shared two-live trace stack. Result:
`expert_ep2_watcher.xml`, **4 passed in 32.88 s**. The archived 556-line raw
log has SHA-256
`50e398a486e7b8290e31e8837284c1e2bcf7b2d4127e855c6eeb76be5b101830`
and no watcher error/assertion/panic/hang/timeout signature.

### Independent stage review and final health

A fresh xhigh `$stage-review` subagent inspected the live worktree read-only,
without opening TT devices. It re-derived the capacity, PCC, test counts, and
performance medians from source artifacts; inspected the cache/PLE, trace,
paging, collective, and residual code; verified the evidence manifest; and
returned **clean-pass** with no required work. The preserved verdict is
`STAGE_REVIEW.md`.

The reviewer noted three controlled anomalies: historical profiler artifacts
coexist with the accepted EP2 set, host Linux sampling is unavailable to
Tracy, and ETH watcher instrumentation is disabled for the final watcher run.
The checkpoint therefore retains only manifest-delimited EP2 evidence plus
diagnostics directly referenced by the AutoDebug/AutoFix reports; the TT ops
reports remain complete, and the watcher control is documented in the README
and review.

The final post-run `tt-smi -s` health check passed. Both target dies reported
healthy DRAM and zero corrected or uncorrected GDDR errors. JSON parsing,
Python bytecode compilation, `git diff --check`, and `sha256sum -c` also
passed. Ruff was unavailable in this environment and is not claimed as a
completed check.
