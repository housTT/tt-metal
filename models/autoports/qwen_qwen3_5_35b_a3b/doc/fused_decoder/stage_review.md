# Stage Review

Verdict: clean-pass

Independent review of Stage 02, fused-decoder, for the pinned
`Qwen/Qwen3.5-35B-A3B` revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`. Review completed on 2026-10-10
against the live `feat/qwen3-5-35b-a3b` worktree based on
`53274a46856a80e3872967fd0e79d2d9758f48a7`.
The delivered `tt/fused_decoder.py` SHA256 is
`e570144c58707bb0846be80075c0b26787e2b3cff8539e9a245bca739c1a168d`.
Paths below are relative to this evidence directory unless stated otherwise.

## Required Work

- None. The material fusion omissions, estimator error and default-performance
  reproduction anomaly identified during review were investigated and closed
  before this verdict. The final source, applicable correctness evidence,
  profiler archives, README and telemetry packet agree.
- The local checkpoint and recording its SHA are the stage owner's subsequent
  steps, as required by the review-before-commit sequence. This review does not
  claim a commit or push has already occurred.

## Other Concerns

- Full-attention accuracy has a modest margin above the required 0.995: the
  final headline minimum is 0.995754361 and the B32 position test minimum is
  0.995631337. All tested requests/tokens pass. This is not evidence for lowering
  precision further or for accumulated full-model accuracy.
- The selected indexed RoPE uses an opaque, consistent permutation of the
  first 64 K channels. BF16 allocation, page ownership and capacity are
  preserved, but populated functional/fused K caches cannot be interchanged
  without conversion. This limitation is now explicit in the README and
  `../context_contract.json`; the reviewed contract does not promise raw
  populated-cache interchange between implementations.
- The approximately 42 us default/candidate discrepancy was real. The matched
  lifetime control and final default rerun establish the repair. Physical DRAM
  placement is a plausible mechanism, not an address-level observation; final
  documentation and source comments correctly retain that distinction.
- Constructor repacking uses host Torch and TT tensor transfers. The measured
  prefill/decode paths do not: runtime audits, source inspection and profiler
  rows agree. Remaining layout and sharding work serves the arbitrary-tail
  planner, precise GDN state or native kernel input contracts.

## Hard-Check Gaps

- No unresolved required gate was found. Native-context HF prefill checks 16
  selected outputs while retaining all recurrence/KV contributions; the entire
  TT layer executes every token. The selected-reference method has a dense-HF
  control at 1153 tokens in the accepted functional stage. It should not be
  described as a dense comparison of all 262144 output vectors.
- The custom compact FFN profiler row omits math fidelity. BFP8 weights and
  BF16 output are visible in runtime rows; LoFi is established by the exercised
  `_CompactExperts` configuration and native API, not by inventing a missing
  CSV field. Other dominant projection/packed matmul dtypes and fidelities are
  visible in the final rows.
- DRAM percentages use an explicitly documented compulsory-traffic estimate,
  not hardware bandwidth counters. Useful prefill FLOPs count eight active
  routed experts rather than all 256 experts executed by the dense packed
  prefill adapter. Both ratios use complete layer spans and one participating
  ASIC's nominal peaks. These are suitable for the requested telemetry, with
  the stated limitations.
- This is a single-layer stage. Generation quality, end-to-end accumulation,
  multichip, optimized-decoder and serving qualification are not claimed and
  were explicitly excluded from the authorized stage.

## Anomaly Ledger

- Observed anomaly: Early native FFN output was badly wrong, initially looking
  like a numerical limitation.
  Evidence: `AUTODEBUG.md`, `ffn_hypothesis.md`, `AUTOFIX.md`,
  `ffn_repaired129`, `compact_trials129`, `compact_trials4096`.
  Affected path: Routed decode experts.
  Control or comparison: Repaired full-256-region adapter passes real weights;
  compact eight-region adapter is faster and passes HF/functional comparisons.
  Likely subsystem: Native reader output-region bounds versus activation backing
  capacity, rather than a LoFi precision failure.
  Investigation performed: Inspected the bounds contract and the adapter's
  capacity, then reviewed real repaired trials. The selected compact mapping
  retains all 256 weight descriptors, maps unique top-eight IDs to eight output
  slots, and does not consume unwritten logical rows.
  Resolution: fixed.

- Observed anomaly: The 256-output packed-prefill split experiment hung.
  Evidence: `triage/prefill_split/AUTOTRIAGE.md`, `AUTOFIX.md`,
  `split_full4096`, `split_linear4096`, `split32_full4096`,
  `split32_linear4096`.
  Affected path: Packed routed prefill SwiGLU.
  Control or comparison: Unsplit and single-call-32 alternatives execute;
  eight groups of 32 outputs pass and provide the selected whole-layer speed.
  Likely subsystem: Reader stack use; the recorded compiled frame is 5168 bytes
  against a 5056-byte stack.
  Investigation performed: Preserved failure evidence, source/compiled-frame
  analysis, bounded recovery, and adapted grouped packing. Shared C++ was not
  changed. Later clean watcher runs are separate from failure triage.
  Resolution: fixed.

- Observed anomaly: An early routing/capture path used host-created zeros and
  hung during trace capture.
  Evidence: `AUTOFIX.md`, `triage`, final resident routing templates,
  `FusedDecoder.prefill_forward`, `reuse_final`, `reuse_placement_final`.
  Affected path: Trace-safe temporary and output creation.
  Control or comparison: Device `zeros_like` and resident templates execute
  under allocation tracking, including program-cache allocations, with repeated
  and cross-request replay.
  Likely subsystem: Host-backed tensor creation inside capture.
  Investigation performed: Reviewed preserved triage/recovery and the resulting
  device-only allocation path. Only the intended trace output is acknowledged;
  no blanket capture/warmup scope is exempted.
  Resolution: fixed.

- Observed anomaly: Native TF32 recurrence passes the headline workload but
  fails a real heterogeneous B32 case in the final surrounding graph.
  Evidence: `native_positions_final_candidate.json`,
  `positions_final/linear.json`.
  Affected path: GDN token recurrence and unaligned tails.
  Control or comparison: Native minimum decode PCC 0.988434195 versus selected
  FP32 recurrence 0.997774541 on the same real B32 test; the failing native row
  has prefix length 465.
  Likely subsystem: Recurrence precision/rounding accumulated through token
  execution, not a synthetic-only rejection.
  Investigation performed: Compared real final-graph outputs, the native op
  source contract and selected grouped FP32 in-place update.
  Resolution: controlled; native recurrence is not selected.

- Observed anomaly: Initial projection-residual fusion missed the linear token
  return hook; adapted full-attention residual fusion also fails accuracy.
  Evidence: `residual_linear4096`, `residual_full4096`,
  `_ProjectionResidualLayer`, `_ResidualProjection`.
  Affected path: Attention/GDN output projection and first residual.
  Control or comparison: Repaired linear candidate passes but is slower
  (1275.400 versus 1259.203 us host replay). BF16-upcast full candidate is slower
  and has minimum PCC 0.994860351.
  Likely subsystem: Initial adapter wiring; thereafter numerical rounding and
  the native epilogue's weight-format/dispatch cost.
  Investigation performed: Reviewed corrected residual propagation and adapted
  real 4096/128 trials rather than accepting the first failure as rejection.
  Resolution: controlled; these candidates are not selected.

- Observed anomaly: The original fusion ledger omitted shared-down projection
  plus sigmoid gate plus routed-add epilogue.
  Evidence: `ffn_hypothesis.md`, `tests/run_shared_epilogue_trial.py` under the
  autoport, `shared_epilogue4096/commands.json` and result JSONs.
  Affected path: Shared expert output.
  Control or comparison: BF16-upcast down weights and expanded sigmoid gate
  allow the actual native op to execute on both real 4096/128 paths. Linear
  passes minimum PCC 0.997277260 but is slower, 1279.977 versus 1259.137 us.
  Full is slower, 944.960 versus 924.030 us, and fails PCC at 0.994528234.
  Likely subsystem: Native epilogue format/rounding and execution cost.
  Investigation performed: Requested the missing adapted family, inspected its
  implementation and real controls, and checked the final ledger closure.
  Resolution: controlled; rejection is now earned by adapted measurements.

- Observed anomaly: Indexed partial-width RoPE was initially treated as a
  potentially incompatible tile-local operation, leaving separate tail concat.
  Evidence: Native indexed-RoPE validator/reader/writer/compute source,
  `tests/run_indexed_rope_trial.py`, `indexed_rope4096.json`,
  `indexed_rope_native4096.json`, `indexed_production.diff`.
  Affected path: Full-attention Q/K RoPE, prefill and decode.
  Control or comparison: Coherent 16-value-group permutation preserves BFP8
  exponent groups and maps HF half rotation to tile-local rotation. Native
  layout removes an avoidable caller transpose round trip. Real minimum decode
  PCC is 0.995754361, functional equivalence 0.999609709.
  Likely subsystem: Weight/norm/table basis and input head layout.
  Investigation performed: Independently checked the permutation algebra and
  tile contract; reviewed coherent packing, natural-head caller adaptation and
  both cache consumers. Final rows contain two indexed RoPE operations and no
  separate post-RoPE tail concat.
  Resolution: fixed; adapted indexed RoPE is selected.

- Observed anomaly: Promoting the natural-layout indexed candidate regressed
  host replay from about 920 us to 963 us despite identical PCC.
  Evidence: `headline_final/full.json`, `placement_controls4096/commands.json`,
  `placement_production.diff`, `placement_final/full.json`,
  `profile_placement_final/tracy/full_attention`.
  Affected path: Full-attention default construction and resulting execution.
  Control or comparison: Matched production 962.3595 us, natural candidate
  920.675 us, retained-input control 920.4485 us, prior partial-RoPE baseline
  924.276 us. Final default is 920.198 us with unchanged PCC; final device mean
  is 978.654142 us. The earlier 1018.311540 us full device profile is diagnostic
  evidence and is explicitly superseded.
  Likely subsystem: Construction allocation lifetime, with physical placement
  inferred rather than measured directly.
  Investigation performed: AST/source comparison found equivalent dispatch.
  The test adapter retained old norm/projection allocations through packing;
  production released them earlier. A tests-only lifetime control restored
  speed. Production now retains those references only until construction ends;
  full headline, B32 positions, both page-size reuse tests, watcher and profile
  were repeated. All pass.
  Resolution: fixed.

- Observed anomaly: The fused DRAM estimate charged FP32 convolution taps even
  though selected KDA taps are BF16.
  Evidence: `tests/estimate_roofline.py` under the autoport, final linear
  `roofline.json`, actual KDA profiler input dtype.
  Affected path: Telemetry numerator, not runtime correctness.
  Control or comparison: Fused tap bytes are now 2; functional taps retain 4.
  The named component decreases by 65536 bytes.
  Likely subsystem: Estimator inherited a baseline precision assumption.
  Investigation performed: Cross-checked runtime dtype, estimator arithmetic,
  regenerated roofline, final results and compact packet.
  Resolution: fixed.

- Observed anomaly: Small numerical differences and some aggregate PCC values
  equal to 1.0 appear despite changed native arithmetic.
  Evidence: Final headline per-token arrays, functional-equivalence fields,
  real/synthetic device tests, README precision disclosure.
  Affected path: Native convolution, normalization, expert accumulation and RoPE.
  Control or comparison: All final request/token gates meet 0.995; functional
  decode minima are 0.999664128 linear and 0.999609709 full. Exact determinism is
  separately checked by repeated replay, not inferred from rounded PCC.
  Likely subsystem: Documented rounding changes and FP32 correlation rounding.
  Investigation performed: Checked measured formats, threshold enforcement and
  the separate equality assertions.
  Resolution: controlled.

- Observed anomaly: CPU/device logs contain nanobind teardown leak notices,
  UMD statistics shared-memory warnings and no-op `to_layout` notices.
  Evidence: `cpu_contract_final.log`,
  `../functional_decoder/cpu_contract_final.log`, passing candidate/default
  logs, final watcher raw logs and profiler rows.
  Affected path: Binding teardown, optional statistics, already-matching layout.
  Control or comparison: The unchanged functional CPU control reports the same
  two `CoreRangeSet`/`MemoryConfig` instances, 19 types and 256 functions at
  teardown. Statistics warnings also occur in passing controls. Final watcher
  logs have no assertions, invalid-access or corruption reports.
  Likely subsystem: Existing binding/process teardown and environment warnings.
  Investigation performed: Read actual notices and controls; did not classify
  them as new device-buffer leaks or silently ignore device-health warnings.
  No-op conversions were checked against needed native layouts and device rows.
  Resolution: controlled; no new stage-specific failure established.

- Observed anomaly: Initial test adapters had API/layout errors, including an
  incorrect error-fixture invocation and frozen-dataclass assignment; some
  README references/counts needed synchronization after final promotion.
  Evidence: Preserved initial logs, adapted trial scripts,
  `cpu_contract_final.log`, final README/work log and per-profile
  `graph_analysis.json`.
  Affected path: Test plumbing and evidence navigation.
  Control or comparison: Final CPU run has 15 passing checks; device test run
  has four passes; adapted candidate commands record actual executions. Both
  accepted profile roots contain their graph analysis summaries.
  Likely subsystem: Harness/API adaptation and documentation updates.
  Investigation performed: Verified retries, final references and counts rather
  than treating first errors or ambiguous basename references as kernel blockers.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: The parent-provided original stage contract;
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`;
  `/home/hous/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/`
  `stage-review/SKILL.md`, `graph-fusing/SKILL.md`,
  `tt-device-usage/SKILL.md`, `tt-enable-tracing/SKILL.md`, and the
  model-bringup startup dependency; repository
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Code paths: Entire `tt/fused_decoder.py`; the functional decoder's planner,
  cache/state/trace ownership and relevant GDN/attention paths; changed test
  runners, runtime audit, HF reference, acceptance and contract/device tests;
  fusion, shared-epilogue, indexed-RoPE and placement trial adapters; profile
  collection/processing, span summarization and roofline estimator. Also
  inspected relevant common MoE and native indexed-RoPE/FFN contracts and kernel
  implementations to assess adapted candidates.
- Artifact paths: `README.md`, `work_log.md`, `graph_inventory.md`, all 119
  indexed candidate records and the material trial controls, `AUTODEBUG.md`,
  `AUTOFIX.md`, `ffn_hypothesis.md`, triage records, `final_results.json`,
  `source_provenance.json`, `acceptance_summary.json`, code-precommit logs,
  CPU/device test logs, `../context_contract.json`, accepted functional baseline
  evidence, and the final evidence groups listed below. Read actual watcher
  dumps and raw compressed profiler CSVs, not only summaries.
- Telemetry artifact: Read and checked
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/b499af22-a351-44ce-8553-1dfe93ca6aac.json`.
  It records exact 4096/128, batch/concurrency 1, separate real layer kinds,
  device 3 on a 1x1 Blackhole mesh, final spans and explicit estimate assumptions.
- Commands run: Read-only `git status`, `git diff --stat`, `rg`, `sed`, `cat`
  and standard-library Python analysis using `ast`, `csv`, `gzip`, `hashlib`,
  `json`, `pathlib` and `statistics`. Recomputed source/archive hashes, AST
  equality, permutation algebra, raw device spans, warmed-program coverage,
  row counts, PCC minima and evidence consistency. No TT imports, devices,
  hardware tests, resets, servers or subagents were used by the reviewer. The
  only file written by this reviewer is this report.

Source provenance is sufficient for the accepted evidence. Independently
decompressed the a4c8, eb835 and 4792 archives and checked their hashes.
a4c8 to eb835 changes full attention's indexed-RoPE construction/forward;
GDN and common expert paths are unchanged. eb835 to 4792 changes only transient
construction references in `_enable_indexed_rope`; all runtime forwards are
unchanged. The delivered e570 source has an identical AST to measured 4792;
only the allocation-mechanism comment differs. Thus GDN evidence remains
applicable, and indexed full native-context evidence is not invalidated by the
later lifetime-only repair. The affected full execution/trace paths were rerun.

The final correctness evidence includes:

| Gate | Linear attention | Full attention |
|---|---|---|
| Real 4096/128 HF prefill / minimum decode PCC | `headline_final/linear.json`: 0.999676347 / 0.997841656 | `placement_final/full.json`: 0.998591542 / 0.995754361 |
| Functional equivalence prefill / minimum decode PCC | 1.0 / 0.999664128 | 0.999940872 / 0.999609709 |
| B32 heterogeneous prefixes 1..1006, reversed slots, eight decode steps | `positions_final/linear.json`: minimum decode 0.997774541 | `positions_placement_final/full.json`: minimum decode 0.995631337; guard pages clean |
| Same-instance reuse, unseen length 36, continuations and page boundaries | `reuse_final/linear.json`: 17 cases | `reuse_placement_final/full.json` and `full_page128.json`: 17 cases each, page sizes 32/128 |
| Native 262144 prefill and 262143+1 | `context_final`: prefill 0.999584258; nonaligned prefill/decode 0.999411523 / 0.999906123 | `context_indexed_final`: prefill 0.998267472; nonaligned prefill/decode 0.998284280 / 0.998339355 |
| Watcher | `watcher_final/linear.json` plus actual raw watcher log | `watcher_placement_final`: 129/8 and odd pools 257/B2, 65/B1 with 31 steps; actual logs clean |

Position tests show deterministic warm/replay and repeated outputs. Reuse tests
record one capture, zero request releases and one teardown release, with
`TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations included and cache
misses forbidden. Request reset/restore/copy paths retain fixed backing storage.
State/input/weights predate capture; only the exact intended output is
acknowledged. Reviewed evidence uses one live trace, so the output acknowledgment
does not claim safety across untested coexisting traces or arbitrary aliases.

The final measured performance is independently reproducible from raw CSV:

| Layer kind | Baseline prefill us | Final prefill us | Baseline decode us | Final decode us | Final prefill / decode device rows |
|---|---:|---:|---:|---:|---:|
| linear_attention | 1743870.348148 | 764671.617037 | 2190.255232 | 1399.833297 | 3588 / 103 |
| full_attention | 1720531.702222 | 697717.624678 | 1608.711806 | 978.654142 | 1013 / 65 |

Accepted final roots are `profile_final/tracy/linear_attention` and
`profile_placement_final/tracy/full_attention`. Decompressed CSV SHA256 values
are respectively
`1407fd0f6af035230289a2f681afe82e11a2a818abc559298c90665e1618872e`
and `d3c06e7cde163cc7fa365192018fe1a7ca217a722241a09e4ff8fb043369f0bb`.
Each preserves both prefill/decode `tt-perf-report` CSV and text output,
measurement, processing provenance, graph analysis and roofline estimates.

For every measured window, independently counted all expected device rows and
verified complete firmware timestamps on one ASIC. Computed minimum firmware
start to maximum firmware end, with recorded cycle-to-nanosecond scale, including
every operation and internal gap. Decode is the mean of 128 complete replay
spans. Every measured program hash appears before its corresponding phase:
linear has 78/77 unique prefill/decode hashes, full has 75/56. Baseline archives
were checked with the same method. No host duration, overlapping-row sum or
matmul-only subset substitutes for a device-layer denominator.

Unprofiled final host replay medians are 1259.719 us linear and 920.198 us full.
The linear result reproduces the selected 1259.203 us screen within 0.05%; the
full result reproduces the adapted indexed candidate and improves the matched
924.276 us partial-RoPE control. Both warmed prefill phases improve materially
over functional baselines. Tiny indexed-prefill changes are correctly treated
as timing noise rather than a claimed extra gain.

## Residual Risk

- Single-layer deterministic hidden-state/PCC evidence does not establish
  end-to-end generated-text quality or accumulated accuracy through the full
  model. This stage does not make those claims.
- Native context at batch 1 and heterogeneous short contexts at batch 32 are
  independently qualified. The complete batch/context Cartesian product and
  arbitrary concurrent trace signatures are not established.
- Populated cache interchange requires the documented K-basis conversion;
  future consumers must honor that representation contract.
- Allocation-lifetime sensitivity may affect performance on another allocator,
  device revision or surrounding model. The current exact default is measured;
  the proposed physical mechanism remains an inference.
- Useful-FLOP/DRAM percentages are nominal-peak estimates with the documented
  excluded vector work, reloads, padding and temporary traffic. Host input
  refresh/readback lies outside the device replay window. These are layer
  execution metrics, not serving throughput or measured DRAM utilization.
- Explicit non-default fusion flags remain research controls and include known
  failing/slower variants. This verdict qualifies `DEFAULT_FUSIONS`, the pinned
  checkpoint and the documented execution scope.
