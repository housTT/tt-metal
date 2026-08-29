# Multichip decoder work log

## Scope and starting state

- Goal: multichip decoder only; no full-model or vLLM implementation.
- Authorized implementation: `tt/multichip_decoder.py`; authorized tests and
  evidence: `tests/test_multichip_decoder.py` and `doc/multichip_decoder/`.
- Baseline commit: `40cd075e` (`Record GPT-OSS optimized decoder stage commit`).
- Baseline implementation: `tt/optimized_decoder.py::OptimizedDecoder`.
- Target meshes: P150 `(1,1)`, P150x2 `(1,2)`, P150x4 `(1,4)`.
- Existing unrelated untracked `.agents/`, cache, AutoDebug/AutoTriage, and
  earlier-stage image/log artifacts were present at start and are excluded
  from this stage.

## 2026-08-28 inventory and strategy selection

Read `$multichip`, `$tt-device-usage`, `$optimize`, `$tt-enable-tracing`,
`$autofix`, `$stage-review`, and `tech_reports/LLMs/llms.md` section 3.3.
Inspected the optimized/fused decoder, canonical GPT-OSS 1D attention, sparse
experts, CCL manager, mesh helpers, paged KV cache, common 1D modules, and the
prior optimized correctness/performance evidence.

Hardware inventory:

```text
timeout 60 tt-smi -ls --local
```

Result: four reset-capable `p300c` Blackhole devices. The first mesh smoke used
the shell's namespace-only Python and failed before hardware with missing
`ttnn.open_mesh_device`; this was environment evidence, not a device failure.
The source-built environment then passed:

```text
timeout 120 env \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn \
  LD_LIBRARY_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib \
  python_env/bin/python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print('MESH_SMOKE_OK 1x4')
PY
```

Result: `MESH_SMOKE_OK 1x4`; all devices closed normally. A subsequent bounded
`tt-smi -s` showed healthy DRAM status, firmware 19.13.1.0, no corrected or
uncorrected GDDR errors, and 31--33 C temperatures.

The selected tensor, activation, cache, collective, and expert plan plus memory
calculation and rejected alternatives are in `mesh_plan.md`. It was written
before `tt/multichip_decoder.py` existed.

## 2026-08-28 implementation and first hardware isolation

Added `tt/multichip_decoder.py`, host-side plan/context/fallback tests, and the
initial real-checkpoint optimized-baseline acceptance harness. Formatting and
`py_compile` passed. The host-only pytest selection passed five tests: all
three tensor plans, the context-capability audit, and the runtime
fallback/active-expert audit.

A direct `(1,2)` fixture open with `FABRIC_1D_RING` failed twice before model
construction because fabric device 2 did not complete its Ethernet handshake.
The failure was unchanged by one bounded all-device reset. Repository fixture
policy explains the contrast: this four-device P300x2 host must open the full
fabric parent before carving a two-device submesh; directly opening only a
subset leaves required fabric partners inactive. These setup failures are not
decoder results.

After a second bounded reset/list recovery, the full `(1,4)` ring initialized
on all four devices. The TP=4 decoder and full-context local caches constructed,
as did an optimized `(1,1)` baseline on a submesh of the same parent. The first
baseline prefill then timed out before any multichip forward. Safe-pytest
captured `generated/tt-triage/triage.csv` and reset the devices. Its exact
reproducer and evidence summary are in `triage/AUTOTRIAGE_INPUT.md`; fresh
AutoTriage/AutoFix agents are separating the concurrent-baseline harness from
the unexecuted TP path before the next hardware command.

## 2026-08-28 AutoFix isolation and real-weight acceptance

Fresh AutoFix agents audited the initial TP2/TP4 failures and the repository's
attention, expert, CCL, and fabric code. The baseline and target were separated
into serialized processes with atomic versioned artifacts. TP2 runs from a
submesh of a full `(1,4)` fabric parent; TP4 uses the parent. This removed the
unsafe overlapping-baseline lifecycle and direct-subset fabric setup.

The final per-rank expert layout pairs gate/up chunks before sharding so every
rank owns both operands of its local SwiGLU. The router is replicated BF16 in
L1 and decode uses indexed sparse matmul for only the selected top-4 experts.
The canonical dense 128-expert decode tensors are deallocated after the compact
indexed weights load.

Numerical program-config isolation found:

- TP4 down subblock width 3 passed and was retained;
- TP4 gate/up width 4 failed decode PCC 0.915330, so width 1 was retained;
- TP2 down width 3 failed prefill PCC 0.874546/0.856799, so width 1 was retained.

The authoritative baseline and target commands are recorded in the log headers
under `artifacts/20260828_final_v2/`. The standalone P150 baseline generated
versioned artifacts for sliding layer 0 and full layer 1. Both TP2 and TP4 then
passed against those exact artifacts in separate safe-pytest processes.

## 2026-08-28 topology selection and final TP4 change

The first TP4 profile exposed two decode `AllBroadcast` reductions at about
35/34 us. The logical hidden width is 2880 = 90 tiles, which cannot be evenly
reduce-scattered over four ranks. Attention's row-parallel O projection already
produces a zero-padded 2944 = 92 tile physical output, or 736 columns/23 tiles
per rank, but canonical decode sliced it before CCL.

The final warmed 200-repeat shape-faithful probe command was:

```text
env GPT_OSS_120B_MULTICHIP_TOPOLOGY_PROBE=1 \
  GPT_OSS_120B_MULTICHIP_TOPOLOGY_REPEATS=200 \
  scripts/run_safe_pytest.sh \
  'models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_residual_sharded_distributed_norm_fused_qkv_probe[blackhole-p150x4-parent-tp4]' -s
```

Results:

- exact canonical attention chain: 0.193192955 ms;
- physical 2944 all-reduce then logical slice: 0.110467005 ms, ratio
  0.571796239, PCC 0.999937 on all ranks;
- coherent reduce-scatter/distributed-norm/fused-AG-QKV chain: 0.129148515 ms,
  ratio 0.668494951, PCC 0.999750;
- expert logical 2880 reduction: 0.070269675 ms;
- expert runtime 2944 pad/reduce/slice: 0.115073950 ms, ratio 1.637604699.

Selected: a model-local TP4 attention specialization that mirrors canonical
decode through O projection, reduces physical width 2944, then slices to 2880.
Prefill already follows that order. TP2 and expert collectives remain
canonical. Rejected: a sharded stack boundary, expert runtime padding, expert
weight-output padding (changes 90 output tiles to 92 and the proven matmul
decomposition), and TP2 fused AG/MM (the operation requires Ring while the
two-device fabric maps Linear). Evidence is under
`artifacts/20260828_final_v2/topology_probe/`.

A fused topology experiment polluted fabric state: unchanged canonical TP4
attention then diverged only on rank 2. The same base class reproduced it, so
the new attention code was not the cause. A bounded warm reset restored the
unchanged path. The final implementation was accepted only after this reset;
pre/reset/post evidence is under `recovery_after_topology_probe/`.

## 2026-08-28 final correctness and trace evidence

P150 baseline, 100 replays per sample and five samples:

- sliding: prefill 36.580892 ms; decode 0.530752270 ms;
- full: prefill 32.245878 ms; decode 0.528672960 ms.

P150x2 final, both tests passed:

- sliding: prefill 47.729884 ms, PCC 0.991984; decode 0.752627020 ms,
  PCC 0.983523, speedup 0.705200, efficiency 0.352600;
- full: prefill 47.502560 ms, PCC 0.991052; decode 0.752160320 ms,
  PCC 0.960387, speedup 0.702873, efficiency 0.351436.

P150x4 final physical-hidden path, both tests passed:

- sliding: prefill 25.874462 ms, PCC 0.991819; decode 0.652859620 ms,
  PCC 0.984313, speedup 0.812965, efficiency 0.203241;
- full: prefill 25.605209 ms, PCC 0.991578; decode 0.652924140 ms,
  PCC 0.962575, speedup 0.809700, efficiency 0.202425.

All four target cases passed two refreshed trace inputs at positions 128/129
after the initial position 127 capture. This crosses a randomized page-table
block boundary and the sliding window. Refresh PCC was 0.974073--0.998842;
reconstructed global K/V blocks from `8/TP` local heads passed PCC
0.999703--0.999996. Every rank output was bitwise replicated with public shape
`[1,1,S,2880]`. Prefill length 127 proves that public sequence lengths need not
be tile or page aligned.

Authoritative logs:

- `artifacts/20260828_final_v2/p150_baseline.log.gz`;
- `artifacts/20260828_final_v2/p150x2_regression_final.log.gz`;
- `artifacts/20260828_final_v2/p150x4_physical_hidden_attention_final.log.gz`.

## 2026-08-28 profiler analysis

Profiler capture and `tt-perf-report` were run separately from watcher. The
final TP4 reports are under
`artifacts/20260828_final_v2/tracy_physical_hidden/p150x4/{sliding,full}` and
contain compressed raw op CSV, sliced prefill/decode CSV, summary CSV/PNG, and
human tables.
The original P150/TP2/TP4 comparison is under `tracy/`.

Final TP4 decode uses 76 device ops and zero host ops. The attention reduction
is now 13 us `ReduceScatterMinimalDirect` plus 10 us `AllGather`, with the
logical slice afterward. The old TP4 path spent about 35 us on its attention
`AllBroadcast` plus pre-CCL slice/copies. The unchanged expert reduction is the
remaining 34 us `AllBroadcast`. Device totals are 581/582 us for sliding/full,
and reported DRAM traffic is 33 GB/s (6.5%). QKV is 44 us, O projection 15 us,
router projection 53--54 us, gate/up sparse matmul 94 us, and down sparse
matmul 22 us. Compute/DRAM utilization shows the active-expert and projection
kernels, rather than the optimized attention CCL, are now the material decode
costs. Prefill totals are 24.710/24.690 ms and remain dominated by packed
all-expert prefill work; its row boundaries already select RS+AG.

The human reports were generated with:

```text
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report \
  --no-summary --signpost PERF_DECODE --csv raw_ops.csv
```

## 2026-08-28 determinism, watcher, and health

The original TP2/TP4 path passed 1,000 trace replays per sample for both layer
kinds. The final changed TP4 path was rerun with three 1,000-replay samples per
layer under:

```text
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_DISPATCH=1 \
TT_METAL_WATCHER_DISABLE_ETH=1
```

Both final tests passed; trace refresh and cache checks also passed. The raw
Tensix watcher log has no NoC, CB, assert, or device error. Full watcher debug
overflowed the trace/model debug capacity, so the supported evidence explicitly
disables dispatch and Ethernet watcher instrumentation. Active-fabric ordinary
dispatch stress was clean. This limitation is not hidden or called a fully
instrumented Ethernet watcher pass.

Evidence:

- `watcher_debug_overflow.log.gz`;
- `watcher_tensix_stress.log.gz` and `watcher_tensix_raw.log.gz`;
- `watcher_tensix_physical_hidden_tp4.log.gz` and its raw log;
- `post_physical_hidden_watcher_tt_smi.log.gz`.

Post-run `tt-smi` showed four devices with DRAM status true and zero corrected
or uncorrected GDDR errors.

## 2026-08-29 stage-review AutoFix

The first independent `$stage-review` verdict was `more-work-needed`. It found
four acceptance-evidence gaps, not a proven implementation defect:

1. decode positions were deterministic 127/128/129 rather than randomized at
   high context;
2. the configured batch 1--32 contract had only batch-1 forward evidence;
3. cache inspection allocations happened after the last replay but before
   `release_trace`, producing an allocation-tracker warning;
4. this log still described the final gates as pending.

Fresh `$autofix` agents separately audited trace lifetime and the batch/high-
context harness. The repaired tests infer capture batch from the input, release
traces before cache slicing, reuse preallocated tensors for refresh, and add a
batch-2 artifact/test family. The batch-2 gate covers non-aligned prefill S=33,
position 131071, randomized upper-half logical blocks, two independent
randomized page tables, changed hidden/position inputs, a page-table-only
refresh, exact rank replication, reconstructed local-head K/V cache entries,
and bitwise repeatability before and after remapping.

The first batch-2 attempt compared TP2 BFP8 attention against the optimized
decoder's automatic batch-2 capacity policy, which selected BFP4/LoFi QKV and
produced prefill PCC 0.8613. A per-user target-prefill hypothesis was tested in
isolation and produced the identical 0.8613, so that code was fully reverted.
Log inspection identified the real controlled-variable error. The accepted
P150 artifact explicitly constructs the same `DEFAULT_OPTIMIZED_POLICY` BFP8
attention policy as TP2/TP4 while still using the real `OptimizedDecoder`; this
isolates parallelization and passes. The failed-policy diagnostic and rejected
per-user experiment are summarized in `AUTOFIX_stage_review.md`.

Accepted allocation-tracked batch-2 results under run ID
`20260828_batch2_highctx_bfp8_v3`:

- P150: both layer artifacts passed, positions
  `[[131071,76544],[93713,82911],[93713,82911]]` for sliding and
  `[[131071,91904],[117841,71839],[117841,71839]]` for full;
- P150x2 sliding/full: prefill PCC 0.992263/0.989610, minimum decode
  PCC 0.998116/0.962366, minimum cache PCC 0.999827/0.999797;
- P150x4 sliding/full: prefill PCC 0.992361/0.990435, minimum decode
  PCC 0.982412/0.961553, minimum cache PCC 0.999827/0.999797.

P150 and TP2/TP4 batch-1 acceptance were also rerun with
`TT_METAL_TRACE_ALLOC_TRACKING=1`: two baseline cases and four target cases
passed without the unsafe-allocation warning. Tracker-instrumented replay is
about 190--199 ms and is intentionally excluded from performance claims; the
authoritative latency remains `20260828_final_v2`.

A final risk-matched watcher follow-up ran TP4 batch-2 full attention with
`TT_METAL_WATCHER=1`, dispatch watcher disabled, and Ethernet watcher disabled.
It passed with prefill PCC 0.990435, decode PCC
0.997775/0.997178/0.961553, and cache PCC 0.999797. Post-run health again showed
four healthy devices and zero corrected/uncorrected GDDR errors. Evidence is
under `artifacts/20260828_stage_review_autofix_v1/`.

## Final static gates and review

The runtime audit is in `runtime_fallback_audit.md`. Final static gates passed:

- `python3 -m py_compile` on the implementation and test module;
- `python3 -m json.tool doc/context_contract.json`;
- `sha256sum -c doc/multichip_decoder/artifacts.sha256` for the complete
  committed evidence set;
- `python_env/bin/pytest -q .../tests/test_multichip_decoder.py`: 5 passed,
  13 hardware-gated skips;
- targeted `pre-commit run --files ...`: every applicable hook passed.

This is Python/docs-only work under the repository `AGENTS.md`; no C++ or CMake
build is required. A fresh xhigh independent `$stage-review` rereview returned
`clean-pass` after directly auditing the implementation, tests, accepted
artifacts, profiler tables, lifecycle logs, and disclosed limitations. Its
verdict is recorded in `stage_review.md`.

Local stage commit SHAs (never pushed): pending.
