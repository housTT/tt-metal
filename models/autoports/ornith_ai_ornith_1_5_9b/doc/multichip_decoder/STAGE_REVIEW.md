# Stage Review

Verdict: clean-pass

Independent review of **multichip-decoder** for `ornith-ai/Ornith-1.5-9B`.
No numeric stage ordinal was supplied. Reviewed the live worktree on
`hous/ornith-1.5-9b`, starting HEAD
`483920f536f64462ba6c3dc785c59b8ff002cdc1`, against optimized baseline
`d085eb6d1abcc8b25f213fede6b68aa873d6cd6b`.

Final production `tt/multichip_decoder.py` SHA256:
`cf37a95b2b77afa4b6a0131643b444cc26b8f83c42b3ee0938273079d0ab3702`.
The final correctness, stack, timing and profiler archives contain this exact
production source. The reviewer performed source/artifact analysis only and
did not import TTNN, run tests, access devices, reset hardware or collect profiles.

## Required Work

None. The findings raised during review were resolved before this verdict:

- **BFP8 DRAM QKV block32 rejection:** the original failure included excess
  diagnostic L1 allocations. The corrected single-role probe now fits and
  measures 47.71 microseconds; its whole-layer control measures 0.291866 ms,
  slower than the selected DRAM32/block4 path. It is rejected on measurement,
  not the obsolete allocation failure.
- **BFP8 wider-grid coverage:** new precision-matched 64/110-worker sweeps
  cover all eight legal K blocks. Their best whole-layer QKV-only controls
  measure 0.289691/0.287984 ms. The nearest 110-worker control is reproduced
  with the final 24 KiB small-L1 setting at 0.287815 ms, versus selected
  0.283662 ms in the corresponding default run. Other projections remain
  unchanged in these controls.
- **TP-local prefill geometry:** final BFP4/LoFi controls now compare 8x8 and
  11x10 grids at blocks8/16. The 8x8/block16 linear candidate was adapted
  specifically for its FP32 output projection's physical L1 limit. Fifteen
  measured calls favor the selected 11x10/block16 configuration: linear
  3.363816 versus 3.448008 ms for block8; full attention 2.677883 versus
  2.801300 ms. Both 8x8 alternatives are slower.
- **Performance/documentation accounting:** final analysis includes aggregate
  bandwidth lower bounds, actual profiler dtypes, the FP32 DeltaNet output
  intermediate, correct BF16 tile bytes and factory-derived QKV subblocks.

These resolutions are documented in [PERF_ANALYSIS.md](PERF_ANALYSIS.md),
[work_log.md](work_log.md), the candidate ledger and their named source/log
archives. No production change was needed to close the review findings.

## Other Concerns

- Ethernet watcher instrumentation remains limited. The complete final
  99-case run uses `TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_DISABLE_ETH=1`
  and append mode, exits0 and preserves all-fixture worker logs. The earlier
  all-feature attempt hit Ethernet kernel-size limits; its no-inline retry
  completed model checks without a watcher assertion but aborted in Ethernet
  teardown. The report correctly claims a clean **worker watcher** run.
  It does not claim a clean all-feature Ethernet watcher command.
- The original long-output replica discrepancy and host read-completion stall
  have no proven source-level cause. Reset followed by original-order,
  repeated-capacity, focused assembly and final worker-watcher controls
  supports the recorded controlled-recovery classification. Strict checks
  remain in place; the stage does not claim a speculative kernel repair.
- Several matmuls remain marked `SLOW` by the advice tool. Relevant
  precision-matched geometry, larger K blocks, smaller-compute DRAM variants,
  wider grids and whole-layer topology alternatives are measured. The final
  report explains the remaining limits without claiming global optimality
  or saturated bandwidth.

## Hard-Check Gaps

No missing required gate remains for the supplied decoder-stage contract.
The evidence has the following deliberate bounds:

- Native262144 validation combines actual execution, non-aligned lengths,
  chunk-size invariance, final-position replay and an independent permuted
  paged-cache oracle. Direct full-layer HF comparison reaches8001 tokens.
  Native historical fixtures are labeled and are not an HF rollout of a
  262144-token prompt.
- The capacity probe reserves full-stack byte estimates while executing one
  full-attention layer. Three-decoder composition is tested separately. A
  complete32-layer model, embeddings, logits, generation and serving are
  outside this stage. No qualitative-generation gate applies to this
  decoder-only implementation.
- The user constrained this stage to the strongest use of the available
  four-chip mesh; smaller meshes need not work. The explicit1x4 requirement
  does not violate the supplied stage contract.
- Checkpoint commits and recording their SHAs follow independent clean-pass
  under the stage-review workflow. They are the stage owner's next action;
  this report does not claim that they already exist.

## Anomaly Ledger

### Two-link asynchronous collective corruption

- Observed anomaly: sharded candidates produced corrupted gathered activations
  and non-finite layer outputs.
- Evidence: `logs/sharded_norm_localize_bf16.log.gz`,
  `logs/sharded_link1_gather_control.log.gz`,
  `AUTODEBUG_sharded_trace.md`, `AUTOFIX_sharded_trace.md`.
- Affected path: asynchronous AG/RS sharded-residual alternatives.
- Control or comparison: local RMSNorm and gathered statistics agree with
  controls; two-link activation gather corrupts ranks1/3. One-link AG alone
  is insufficient; one-link AG and RS pass exact restored eager/replay.
- Likely subsystem: asynchronous CCL configuration on this P300c ring.
- Investigation performed: focused norm/gather comparisons, link-count
  interventions, whole-layer128/2048 controls and scoped watcher checks.
  Native all-reduce shares RS kernel code but has separately tested
  decomposition/semaphore ownership; final99 gates exercise its actual path.
- Resolution: controlled by `async_links=1` for those alternatives. Historical
  overwritten tensor evidence is disclosed and is not used as surviving proof.

### Batch32 numerical failures

- Observed anomaly: fast BFP4 geometry crossed the unchanged0.995 per-user
  HF threshold even though batch1 aggregate comparisons passed.
- Evidence: `AUTODEBUG_batch32.md`, `AUTOFIX_batch32.md`,
  `logs/finalq8_dram32_b4_accuracy.log.gz`, final99-case gate.
- Affected path: full-attention projection geometry and interacting MLP math.
- Control or comparison: actual single-chip optimized outputs, same recorded
  user inputs, real weights, immutable snapshots and exact restored replay.
- Likely subsystem: numerical accumulation/precision, rather than stale trace
  inputs or cache ownership.
- Investigation performed: isolated projection/block controls, HiFi2 and FP32
  accumulation alternatives, then decode-only raw-HF BFP8 QKV and same-policy
  geometry sweeps. Final minimum diagnostic per-user HF PCC is0.99687457.
- Resolution: fixed by the selected passing geometry/precision configuration;
  no threshold was relaxed and no synthetic-only veto selected the policy.

### Long-output discrepancy and native-capacity read stall

- Observed anomaly: original8001-token replica comparison failed; a later
  native-capacity read did not complete.
- Evidence: `AUTODEBUG_long_replica.md`, `AUTOTRIAGE_capacity.md`, raw triage,
  `AUTOFIX_long_capacity.md`, original failed logs and final gate archives.
- Affected path: long replicated output/readback and reserved native execution.
- Control or comparison: original source and test order after serialized reset;
  physical-chunk assembly versus final concatenation on all ranks.
- Likely subsystem: transient runtime/device-read state; source cause unproven.
- Investigation performed: preserved device/host wait evidence before targeted
  process termination, reset/list/mesh recovery, finite/exact assembly checks,
  all8 original long tests, three original capacity successes and worker
  watcher controls. Final mixed-path99 gates also pass.
- Resolution: controlled recovery; no unsupported production fix retained.

### Watcher instrumentation limits

- Observed anomaly: all-feature Ethernet watcher kernel exceeded its buffer;
  no-inline retry later aborted during Ethernet teardown.
- Evidence: `AUTOFIX_sharded_trace.md`, corresponding failure/recovery logs,
  `release_mixed_watcher_manifest.json` and final99-case log.
- Affected path: instrumented Ethernet setup/teardown, outside decoder math.
- Control or comparison: no-inline retry completes model/PCC/replay checks;
  final worker watcher completes all99 cases and exits0.
- Likely subsystem: Ethernet instrumentation/firmware lifecycle.
- Investigation performed: no-inline adaptation, serialized successful
  recovery and complete final worker instrumentation with append enabled.
- Resolution: controlled, with the exact missing instrumentation disclosed.

### Profiler drain and interpretation

- Observed anomaly: original merged reports include hundreds of milliseconds
  before the first measured operation; QKV output-subblock CSV fields are blank.
- Evidence: original compressed ops CSVs, final rank-accounting JSON,
  advice-enabled reports and `PERF_ANALYSIS.md`.
- Affected path: performance interpretation, not model output.
- Control or comparison: independently reproduced each rank's kernel sums and
  all remaining gaps from raw signposted rows. Public DRAM matmul config and
  its factory explain the absent subblock fields and derive QKV1x5.
- Likely subsystem: profiler-window boundary and report metadata.
- Investigation performed: preserve raw data, remove only each rank's first
  pre-window gap, account for all devices independently, reconcile profiled
  host windows with unprofiled32-replay windows and explicit bandwidth floors.
- Resolution: fixed in supplemental accounting; raw reports remain available.

### Geometry and allocation rejection evidence

- Observed anomaly: QKV block32 was initially rejected with diagnostic L1
  pressure; wider QKV and prefill geometry evidence mixed precision policies.
  A new8x8/block16 prefill control then failed at FP32 `gdn_out`.
- Evidence: corrected QKV geometry JSON, `review_q*`,
  `review_dram4_b32*`, `review_prefill*` logs and source archives.
- Affected path: optimization selection and candidate setup.
- Control or comparison: corrected DRAM capture, BFP8/LoFi-only QKV sweeps,
  QKV-only whole-layer controls and BFP4/LoFi-only prefill controls.
- Likely subsystem: diagnostic resource retention, then a genuine per-op
  static-L1 constraint for FP32 prefill.
- Investigation performed: corrected QKV block32 fits but loses; wider grids
  lose. Prefill static CB demand1,717,248 exceeds physical1,572,864 bytes;
  a `gdn_out`-only block8 adaptation runs and also loses.
- Resolution: controlled by measured adapted alternatives, with all selected
  production configurations retained.

## Scope Inspected

- Goal/skill paths: supplied original multichip-decoder contract; repository
  instructions; model `AGENTS.md`; `.agents/skills/{stage-review,multichip,
  optimize,tt-device-usage,tt-enable-tracing}/SKILL.md`; relevant LLM report
  multi-device guidance and optimized-stage documentation.
- Code paths: complete `tt/multichip_decoder.py`; inherited optimized, fused,
  functional, configuration and RoPE paths; TP partition/weight packing;
  multichip correctness, native-cache, capacity, direct-stack, geometry,
  topology, precision, fallback and profiling harnesses; shared trace/cache
  tests; relevant CCL and DRAM matmul factory/config source.
- Artifact paths: final README, work log, mesh/capacity/context contracts,
  validation/performance summaries, `PERF_ANALYSIS.md`, candidate/geometry
  tables, final and historical source/log archives, AutoDebug/AutoFix/AutoTriage
  reports, raw triage, all-fixture watcher manifest/archives and final per-rank
  prefill/decode reports plus original compressed ops CSVs.
- Commands run: read-only `git status`, `git show`, `rg`, `cat`, `sed`, `head`
  and `tail`; standard-library Python scripts for JSON/gzip inspection,
  SHA256 checks, AST comparisons, CSV accounting and arithmetic. The only
  reviewer-authored file is this report.
- Integrity checks:14 key historical archive sets checked initially;16 final
  run sets checked after closure, including exact current production source,
  source-member hashes and raw-log hashes. All11 watcher manifest archive
  sizes/hashes and all4 final raw profiler CSV hashes match. Final speedup
  and efficiency recompute exactly. Context and capacity plan JSON agree.
- Baseline check: optimized/fused/functional/config/RoPE/reference source is
  byte-identical to the pinned optimized commit. Final production source
  remained unchanged through review remediation.
- Recorded repository verification: `logs/final_precommit.txt` passes all
  applicable hooks; the work log records19 Python compile checks plus JSON
  and diff checks. This Python/docs-only stage requires no C++ build. The
  reviewer did not independently execute these author-side checks.

## Residual Risk

The stage validates a four-chip decoder baseline, not full-model readiness.
Full-stack runtime resource interaction, end-to-end model accuracy, generation
and serving still require their later stages. Native-context reference evidence
and Ethernet watcher coverage have the explicit bounds above. Recovered
runtime anomalies retain their unproven low-level causes. Within those recorded
bounds, the source, strict gates, paired timings, precision-matched alternatives
and final profiler evidence support clean-pass for this stage.
