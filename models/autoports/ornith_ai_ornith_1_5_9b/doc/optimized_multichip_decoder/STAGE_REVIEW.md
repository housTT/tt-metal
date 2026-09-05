# Stage Review

Verdict: clean-pass

Independent inspection of stage 05, optimized-multichip-decoder, for
`ornith-ai/Ornith-1.5-9B`, completed 2026-09-05 14:28 UTC. Reviewed the live
worktree on `hous/ornith-1.5-9b`, based on
`65abe7f69dbf128012108ef99262bf337f7cdc70`, after the stage owner froze the
implementation and final evidence. No implementation files were changed by
this reviewer and no hardware was accessed.

## Required Work

- None. The stage's technical requirements and independent review gate pass.
  The stage owner must now perform the skill's post-review local checkpoint
  commits and record their SHAs; this verdict does not claim those commits
  already exist.

## Other Concerns

- The final default, rather than an earlier candidate, supplies the headline
  measurements. Linear decode improves from 0.371151 to **0.355672 ms**
  (4.17%); full-attention decode improves from 0.283669 to **0.268965 ms**
  (5.18%). Before/after measurements use B1, 2048-token prefill, and five
  warmed windows of 32 nonblocking trace replays. Final prefill medians are
  3.443891 and 2.690952 ms. The report correctly avoids attributing the
  full-attention prefill median difference to an algorithmic change.
  [Measurements](measurements.json) agree with the archived timing logs.
- Actual final profiler rows confirm packed GDN, BFP4/LoFi packed MLP gate/up
  at 32 input cores/K4/reader3, BFP4/LoFi down at 8 input cores/K6/reader2,
  and BFP8/LoFi QKVG at 32 input cores/K4/reader1. FP32 recurrent state and
  BFP8 KV remain in use. The profiler's 110/80 worker-core counts are correctly
  distinguished from the input shard grids.
- Native TP4 is present in source, weight partitioning and mesh execution.
  The separate single-chip run is a numerical control. It does not replace
  target-mesh performance or direct HF correctness evidence. No host or
  replicated-weight execution fallback was found in the delivered path.
- Material alternatives have adapted, measured evidence: coherent replicated
  and carried-hidden-shard families, native/async collectives, fused AG-MM and
  MM-RS, distributed/fused norms, persistent buffers, packed/separate
  projections, working/residual grids, DRAM readers, and precision/fidelity.
  The current packed family matrix has 32 passing cases, the QKV8 separate-MLP
  family control has 19, and the final packed precision/advice matrix has 34.
  Historical QKV4 matrices remain explicitly historical.
- Review follow-ups are closed with experiments: paired prefill L1 inputs
  pass but are slower for both kinds; all four adapted BFP8 packet4352 cases
  pass but do not beat the retained BF16 collective defaults; exact packed
  gate/up reader1/2/3 microbenchmarks use raw target weights and recorded
  activations. Precision rejection does not rely on a synthetic PCC veto.
  [Optimization evidence](optimization_evidence.md) links the underlying runs.

## Hard-Check Gaps

- No missing stage-required hard check was found. The final default watcher
  suite reports **101 passed** with no failed or expected-failure gate. This
  includes the previously failing B32 full-attention trace selector, logical
  non-aligned lengths, continuation, batch stress, changed-input/page-table
  replay, poisoned pools, refreshed prefill traces, and native-capacity checks.
  Separate async watcher probes pass for both layer kinds.
- Watcher instrumentation is worker-only: `TT_METAL_WATCHER=10` and
  `TT_METAL_WATCHER_DISABLE_ETH=1`. The documented Ethernet instrumentation
  limitation narrows watcher coverage; it is not evidence of an Ethernet
  watcher-clean run. Numerical/trace controls and four-chip collectives were
  exercised. Watcher and profiler runs were separate.
- Native context remains 262144. The final capacity test reserves
  7,153,385,472 DRAM bytes per rank alongside 24 recurrent layers' L1 state;
  native and non-aligned chunk/trace checks pass. Direct HF long-prefill
  evidence reaches 8001 tokens. The position262143 HF cache oracle uses a
  constructed historical cache fixture, not an HF 262144-token rollout.
  The documentation makes this distinction accurately.
- Direct layer0/3/0 composition at B1, prefill131 and eight changed decode
  steps records zero boundary conversions and minimum decode PCC
  0.99992039855. B2-32 outputs are DRAM interleaved in public `[B,1,H]` shape;
  subsequent norms compact/reshard internally. The direct-handoff interface
  is supported by source and batch tests, but the measured three-layer stack
  is B1. No claim of persistent B2-32 boundary L1 storage is accepted.
- The required build wrapper could not launch because Docker is absent from
  the existing build container. The actual fallback command
  `timeout 1200 cmake --build build_Release --target ttnncpp ttnn -j 2`
  compiled and linked 14 targets successfully; `tar` and `tt_pybinds` install
  components completed. Current binaries match the passing final runs.
  The final repository pre-commit log passes applicable hooks. These are
  inspected owner-run results, not commands executed by this reviewer.

## Anomaly Ledger

- Observed anomaly: QKV4 failed real-input B32 user31 output PCC,
  0.9942707597 below the unchanged 0.995 gate.
  Evidence: [QKV AutoFix](AUTOFIX_qkv_trace.md),
  `final_default_v1_watcher_contracts`, and `qkv_trace_*` archives.
  Affected path: full-attention decode projection/gate precision.
  Control or comparison: same user extracted into B1 also fails; QKV8 and
  gate-only BFP8 rescue pass; initial/final KV hashes remain identical for
  the gate-only substitution. HiFi2, HiFi4/FP32 and alternate geometry do
  not rescue QKV4. An actual smaller QKV4/gate8 split was adapted through
  14 configurations and measured slower than packed QKV8.
  Likely subsystem: projection/gate precision, not stale trace/cache state.
  Investigation performed: direct HF, eager/trace/restored replay, component
  substitution and integrated geometry/timing comparisons.
  Resolution: fixed in the selected policy by retaining QKV8 for all batches;
  the final 101-test suite passes the original selector.

- Observed anomaly: mixed-dtype Z fusion produced incorrect batched GDN output.
  Evidence: [Z AutoFix](AUTOFIX_z_fusion.md) and `z_fusion_boundary*` archives.
  Affected path: GDN normalized FP32 output multiplied by the Z gate.
  Control or comparison: unfused/dtype/order/shape controls localized the
  mixed BinaryNG behavior; compact BF16 Z as the left operand with explicit
  FP32 output passes B1/B4/B32.
  Likely subsystem: mixed-dtype fused elementwise tile handling.
  Investigation performed: component comparisons and whole-layer HF/trace
  reruns, followed by final default stress.
  Resolution: fixed by the model-local compact gate adaptation.

- Observed anomaly: an earlier GDN diagnostic reported constant user31 output.
  Evidence: [Z AutoFix](AUTOFIX_z_fusion.md), archived-v2 replays,
  `z_legacy*_batch32_v4`, and `z_user31_hf_current_v2`.
  Affected path: historical B32 diagnostic/current-path regression risk.
  Control or comparison: unchanged archived source replays, legacy/current
  semaphore controls and restored replays are healthy; current real HF
  user31 output PCC is 0.9989239221, with finite nonconstant intermediates.
  Likely subsystem: original cause remains unlocalized.
  Investigation performed: exact archived-source replay and targeted current
  HF diagnostics, rather than attributing the observation to an unrelated fix.
  Resolution: controlled for the current path; the historical cause is not
  claimed fixed. Recurrence requires the prepared failure-only postmortem.

- Observed anomaly: async collective output corruption.
  Evidence: [AG AutoFix](AUTOFIX_ag_trace.md) and collective diagnostics.
  Affected path: mesh CCL semaphore coverage.
  Control or comparison: workers outside the inherited 8x8 semaphore grid
  are covered by the actual 11x10 device grid; caller/full-grid controls pass.
  Likely subsystem: semaphore allocation versus selected worker cores.
  Investigation performed: isolated collective and coherent whole-layer
  family reruns, then separate final async watcher checks.
  Resolution: fixed by `MeshCCLManager` full-grid coverage.

- Observed anomaly: multi-reader matmul descriptor construction used a
  unit-mesh hop assumption on TP4.
  Evidence: [reader AutoFix](AUTOFIX_reader_mesh.md), native build/install
  provenance, reader sweeps and final profiler metadata.
  Affected path: DRAM bank reader placement and mesh program descriptors.
  Control or comparison: explicit per-coordinate descriptors, numeric bank
  ordering, one-reader controls, alternating reader1/2/3 measurements and
  fresh-buffer/changed-trace checks.
  Likely subsystem: native mesh-aware placement.
  Investigation performed: source inspection, native fix/build/install and
  model-weight microbenchmarks plus integrated decoder validation.
  Resolution: fixed. Existing reader defaults remain one; the selected
  higher reader counts appear in the measured final runtime.

- Observed anomaly: BFP4 KV fails unchanged state/real-output gates.
  Evidence: [cache AutoFix](AUTOFIX_cache_precision.md).
  Affected path: paged KV storage precision.
  Control or comparison: matched producers, page mapping, fill/update and CPU
  attention isolate stored K/V PCC around 0.982/0.983, below 0.99. Historical
  QKV4 B32 output failures have matching KV8 rescues; QKV8 B1 output passes
  but the same state gate still fails. Initial cache4 timing adds no benefit.
  Likely subsystem: storage precision.
  Investigation performed: real activations/weights and controlled cache
  comparisons; no claim that current QKV8 B32 cache4 output failure was tested.
  Resolution: controlled rejection; final BFP8 cache remains unchanged.

- Observed anomaly: SDPA chunk1024 exceeded L1 buffer capacity.
  Evidence: [SDPA AutoFix](AUTOFIX_sdpa_chunk1024.md).
  Affected path: full-attention decode scratch and live projection buffers.
  Control or comparison: adapted query/gate DRAM placement, released projection
  intermediates and cap8 scratch reduction make chunk1024 run correctly.
  Likely subsystem: L1 allocation/lifetime.
  Investigation performed: adapted whole-layer comparison; chunk1024 measures
  0.276570 ms versus matched chunk256 at 0.271272 ms.
  Resolution: controlled rejection after adaptation, not a first API error.

- Observed anomaly: allocation warnings while a trace exists, BFP8 packet-size
  advice, unknown tray-name metadata, and one interrupted empty run.
  Evidence: [warning ledger](warning_ledger.md), final watcher archives,
  `packedfinal_ccl8_packet4352_*`, and `interrupted_runs.json`.
  Affected path: trace ownership, experimental CCL throughput, discovery
  metadata and evidence accounting, respectively.
  Control or comparison: exact trace/eager/trace-after-eager output and state,
  poisoned pools/refreshed input checks; four adapted packet cases; physical
  mesh discovery/open evidence; clean owner-run health smoke before resuming.
  Likely subsystem: allocator lifecycle warning, throughput configuration,
  UMD metadata fallback and runner lifecycle, respectively.
  Investigation performed: targeted controls and explicit archive/result
  classification. The interrupted run has no passing result.
  Resolution: controlled within the documented lifetimes and test scope.

## Scope Inspected

- Goal/skill paths: original
  `/home/hous/dev/ornith-1.5-9b/state/multigoal/05-05-optimized-multichip-decoder.prompt.txt`,
  [stage contract](stage_contract.md), repository/model `AGENTS.md`, and
  `.agents/skills/{stage-review,optimize,tt-device-usage}/SKILL.md`.
- Artifact paths: README, work log, optimization evidence, warning ledger,
  AutoDebug/AutoFix reports, candidate matrices/measurements, reader
  microbenchmarks, build/install/check logs, context contract, memory plan,
  final watcher manifest, final validation/timing/stack logs, and all eight
  before/after profile windows with four-rank raw operations and advice tables.
- Code paths: `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, underlying
  state/CCL contracts, changed validation and candidate/diagnostic harnesses,
  native DRAM-sharded matmul factory/utilities/nanobind changes, mesh descriptor
  adapter and per-coordinate device APIs, and evidence/accounting generators.
- Commands run: read-only `rg`, `cat`, `sed`, `git diff`, `git status`,
  `git diff --check`, and standard-library Python analyses of JSON, gzip,
  CSV, hashes and timing arithmetic. No TTNN import, device access, build,
  test, server, reset, implementation edit or nested reviewer was performed.
- Independent integrity result: 467 source archives, 36,591 archived source
  entries and 466 completed log archives have no hash mismatch. One incomplete
  record is explicitly classified as interrupted. All ten final runtime runs
  match 70 current Python-source, 50 native-source and 30 binary hash entries.
  Final `multichip_decoder.py` SHA256 is
  `42c8a9f43429153090d180c5a647686c98b770a8c627c3ab173dd46caf3e1bda`;
  `optimized_decoder.py` is
  `a53232e9b5539fec6361c516203b42b19e2c7ef37a4aaa4d480a26929caada33`.
- Independent performance check: raw per-rank kernel durations plus all
  interior gaps reconcile with rank spans; chip times are never summed as
  layer latency. Final profiled decode host times are 385.308/314.762 us and
  maximum independent rank spans are 368.621/303.530 us. The separate
  60.192/70.050 us storage floors use actual active projection/tiled-KV bytes
  and an explicitly assumed 512 GB/s per chip. They are not measured bus
  utilization or complete end-to-end targets. These results agree with
  [paired performance accounting](performance_accounting.md).

## Residual Risk

- This is decoder-stage evidence. Full-model resource pooling, generation,
  qualitative behavior, serving, and an HF full-native-context rollout are
  outside this delivered stage and are not certified by this review.
- Callers must preserve tested trace input/state/output lifetimes and the
  documented residual boundary shapes. The B1 composition evidence does not
  prove arbitrary caller allocations or every batched multi-layer trace.
- Ethernet watcher coverage and the unlocalized historical GDN observation
  retain the specific limits described above. No current final-path failure
  remains hidden by a lowered tolerance, xfail, synthetic veto or stale result.
