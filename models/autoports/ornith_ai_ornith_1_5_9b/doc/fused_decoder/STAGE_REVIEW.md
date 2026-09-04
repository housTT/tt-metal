# Stage Review

Verdict: clean-pass

Independent reviewer: /root/fused_stage_review. Stage: fused-decoder, after
functional-decoder and before optimized-decoder. Model: ornith-ai/Ornith-1.5-9B,
revision489cb97981b8654bcfcf30ce1f94ed1b62e07b53. Live branch:
hous/ornith-1.5-9b, base ae18ba18bdde6ae20dd628d3f2fd8f59c42ecaf3.
Reviewed runtime SHA256:
**18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d**.

Review used source inspection and read-only artifact scripts, with no TTNN
import, device access, server, reset, or hardware test. Only this report was
written. The required local checkpoint and work-log SHA follow clean-pass;
their absence before review is expected.

## Required Work

- None.

## Other Concerns

No unresolved implementation or evidence contradiction was found. The runtime
owns the block and both mixers' prefill/decode computation; final tests poison
functional block/mixer methods and verify the constructed class. Inherited
setup and prompt orchestration do not provide a computation fallback.

Source and tests preserve BF16 page-64 KV, FP32 recurrent state, fixed state
addresses, padding neutrality, real-token convolution history, partial64 RoPE,
page ownership, and arbitrary logical prompt lengths. Positive-control Torch
and transfer guards pass on fused forward paths. The graph audit accounts for
all surviving movements and all applicable fusion families through measured
adaptations or concrete operator-contract exclusions.

Final real-weight paired results, B1/T2048 with warmed synchronized prefill and
five measured windows of32 traced replays after two discarded windows:

| Layer | Prefill functional → final ms | Decode functional → final ms | Minimum paired PCC |
|---|---:|---:|---:|
| Linear0 |40.662267 →26.180386|1.623890 →1.462064|0.999897954|
| Full3 |35.449361 →23.498724|1.512215 →1.263523|0.999967148|

Native transpose selection also has direct matched evidence: v3 wins25/31
windows, mean saving0.248252µs/replay with sample SE0.040963µs. The integrated
suite reproduces29/31 wins and paired median saving0.252512µs. This establishes
a small layer-level gain; fewer operations alone did not determine selection.

## Hard-Check Gaps

- No required gate is missing: final short v2 passes85 tests, long v2 passes9,
  separate Watcher v2 passes23, and all four final profiler tests pass.
- All seven final run logs, source archives, and contained source hashes match
  provenance and current runtime. Every current file in the final short source
  map still matches. Independently checked all121 run records, including
  preserved failures, against compressed log/source hashes.
- Final raw signpost/report row counts agree:120/196/33/144. Decode has four
  complete49-op linear or36-op full replays, each matching warm operation IDs.
  Raw nanosecond sums equal report microsecond sums; normalized CSV hashes match
  performance.json. Native outer attributes show the selected whole-head reuse,
  transpose_a, FP32 output/destination, HiFi4, and exact math.
- Final kernel sums are linear25.647288/1.4412645ms and
  full23.002021/1.20938125ms, prefill/decode respectively. Measured-output HF PCC
  is0.99951420/0.99976799/0.99949588/0.99889471. Kernel sums exclude dispatch gaps;
  profiler decode uses position128 and four replays, distinct from paired timing.
- Watcher has92 device checks, profiler unset, matching dump hashes, and no
  unexpected corruption/invalid-NoC/error signature. Native-context contract
  check preserves262144. Recorded final pre-commit checks pass and the staged
  diff-check log is empty. Python/tests/docs-only changes require no C++ build.
- README, patterns, graph audit, and work log now distinguish historical source3cf
  evidence from current source18d evidence; inspected evidence links resolve.

## Anomaly Ledger

- **Observed anomaly:** A transpose flag still materialized a kernel; initial
  native/manual-transpose reuse controls both failed PCC0.23859875.
  **Evidence:** v1 profiles, reuse_outer_v1/control logs, matmul/reuse source.
  **Affected path:** Linear decode outer product.
  **Control or comparison:** Whole-head native passes one pair plus six core
  probes; whole-head manual-transpose control passes one pair.
  **Likely subsystem:** Work partitioning versus whole-matrix reader strides.
  **Investigation performed:** Retried per_core_M/N4, blockK1, subblock1×4;
  verified49 final operations and repeated matched timing.
  **Resolution:** fixed; selected configuration passes refreshed stage gates.

- **Observed anomaly:** Coexisting-trace benchmark failed stress PCC0.25064918
  and warned about allocations after capture.
  **Evidence:** reuse_outer_matched_v1, trace_allocation_repro_v1, and
  AUTODEBUG/AUTOFIX_transpose_trace.md.
  **Affected path:** Experimental timing harness.
  **Control or comparison:** With identical decoders, one A replay changes12 B
  buffers before B replays, including425981 recurrent entries and nonfinite taps.
  **Likely subsystem:** Later persistent buffers reuse recorded temporary addresses.
  **Investigation performed:** Allocated both decoders/persistent consumers before
  either capture; v2/v3/final runs pass immediate/final eager equality, stress,
  and state checks. Second-capture temporary/output warnings are explicitly
  classified: those regions are read after their own trace rewrites them.
  **Resolution:** fixed harness; controlled conservative warning. Bad-order
  corruption reproduction is opt-in diagnostic evidence, not model acceptance.

- **Observed anomaly:** Raw flat Q/K passes B1 but fails B32 HF PCC0.99420974.
  **Evidence:** flat_gdn_batch32_trace and hybrid_norm_batch32_trace.
  **Affected path:** Linear prefill normalization and recurrent decode.
  **Control or comparison:** Rank-four BF16 Q/K normalization with flat V passes.
  **Likely subsystem:** Normalization/rounding order.
  **Investigation performed:** Isolated normalization, checked raw core/state,
  and reran the integrated per-user batch gates.
  **Resolution:** fixed by the selected hybrid; minimum final raw-core
  PCC0.99994460, norm ratios0.99783322–1.00369780.

- **Observed anomaly:** KDA conv fails B32 PCC0.99315313; isolated hybrid conv
  fails0.99257131. Ordinary2048-channel groups exceed L1 selection capacity.
  **Evidence:** kda_conv_matrix, hybrid_conv_only, convolution-localization,
  and ordinary Conv1d retry logs.
  **Affected path:** Linear prefill convolution.
  **Control or comparison:** Identical real QKV/taps/history reveal rounding
  differences; disabling FP32 destination worsens them. Ordinary512/1024 groups pass.
  **Likely subsystem:** Activation-before-BF16-pack numerics and L1 footprint.
  **Investigation performed:** Repaired API use, localized numerics, tested legal
  T32 decode adaptation and narrower independent groups without reducing context.
  **Resolution:** controlled KDA rejection; selected1024-channel adaptation passes.

- **Observed anomaly:** Mixed-format SiLU-input multiply gives PCC−0.00138960 or NaN.
  **Evidence:** mixed_silu operand-order failures and same-FP32 controls.
  **Affected path:** Linear output-gating candidate.
  **Control or comparison:** Both same-FP32 adaptations pass but are slower.
  **Likely subsystem:** Mixed-format binary input-activation handling; exact
  kernel defect is not claimed from these controls alone.
  **Investigation performed:** Both operand orders and supported adaptations tested.
  **Resolution:** controlled rejection; corrupt variants are absent from runtime.

- **Observed anomaly:** Bias-add fused softplus zeros5272 negative tails below−5.
  **Evidence:** softplus_localization_v1 and unary/binary codegen.
  **Affected path:** Linear decay candidate.
  **Control or comparison:** Standalone preserves tails; Torch-relative L2 is
  about5.71e−7 standalone versus1.57e−4 fused.
  **Likely subsystem:** Compiled approximation selection.
  **Investigation performed:** Real gate distributions, boundary controls, source
  localization; a passing broad layer PCC was not accepted as sufficient.
  **Resolution:** controlled rejection; standalone FP32 softplus retained.

- **Observed anomaly:** Early adapter/harness errors: rotary padded rows, TTNN
  Shape slicing, missing helper/config imports, missed class-factory construction.
  **Evidence:** dedicated/native/full-width/KDA-conv/localization v1/v2 and
  packed_matrix/packed_matrix_v2 logs.
  **Affected path:** Candidate adapters and test scaffolding.
  **Control or comparison:** Corrected runs pass; final mixer-poison guard is stronger.
  **Likely subsystem:** API/shape and harness wiring.
  **Investigation performed:** Valid shape/API/import/factory corrections followed
  by broader integrated validation.
  **Resolution:** fixed; failed first attempts remain preserved.

- **Observed anomaly:** Full measured-decode HF PCC changes0.99902570→0.99889471;
  discovery, deprecation, convolution-config and reporting warnings appear.
  **Evidence:** Baseline/final HF rows, README warning audit, raw profiles, Watcher.
  **Affected path:** Numerical comparison and evidence tooling.
  **Control or comparison:** Final per-user/long/trace gates exceed0.995; paired
  full-layer minimum PCC0.999967148. Forced Conv1d DRAM output matches its ignored
  argument; optional Tracy web copy loses no actual capture; CSV types are metadata.
  **Likely subsystem:** Valid fused arithmetic and separately classified tooling.
  **Investigation performed:** Compared HF controls, source contracts, full CSV
  coverage/sums and Watcher diagnostics.
  **Resolution:** controlled; no unexplained output/state wrongness or missing data.

## Scope Inspected

- **Goal/skill paths:** Supplied original fused-stage contract, repository/model
  AGENTS.md, stage-review/SKILL.md, graph-fusing/SKILL.md, tt-device-usage/SKILL.md.
- **Artifact paths:** README/patterns/worklog/final_graph_audit, validation and
  performance JSON, candidate inventories, context contract, Watcher audits,
  AutoDebug/AutoFix reports,121 run archives, four final raw/derived CSV/text
  reports, functional-stage controls, and final host-check logs.
- **Code paths:** Entire tt/fused_decoder.py; inherited state/prompt orchestration;
  model config/HF reference; fused/equivalence/contract and allocation tests;
  relevant candidates; normalization, KDA, matmul/reuse, unary/binary, and trace
  allocator operator contracts.
- **Commands run:** Read-only cat/sed/rg/git inspection and Python standard-library
  SHA256/gzip/source-map/AST/JSON/PCC/timing/CSV/replay/Watcher/link analyses.
  Hardware and pre-commit commands were not rerun by the reviewer; their recorded
  evidence was inspected.

## Residual Risk

- Full HF prefill/decode comparison reaches8001 tokens. Native262143/262144
  execution, chunk invariance, and a real-weight high-position HF decode oracle
  over an exact-shape historical KV fixture supplement it; this is not a full
  native-length HF prefill oracle.
- Representative layers are linear0/full3. Native capacity is tested at B1;
  B4/B32 use shorter caches, and irregular B13 exercises grouped rotary geometry.
  Not every batch/context combination was separately executed; no API context
  reduction was introduced.
- Results cover one Blackhole chip on physical P300c boards. The p150 name is
  topology, not a P150-board measurement. Later model/serving/multichip/release
  stages and million-token decoder validation were correctly not started.
- The tiny outer-program gain is supported by repeated within-run controls;
  final standalone numbers remain reported as measured. Trace users must retain
  the documented allocation-before-capture requirement.
