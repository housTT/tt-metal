# Stage Review

Verdict: clean-pass

Independent review of stage 03, optimized-decoder, for
`ornith-ai/Ornith-1.5-9B`, completed 2026-09-05. The reviewed live worktree is
on `hous/ornith-1.5-9b`, based on fused-decoder checkpoint
`bc8f514f3000da7b24c4d2b289b0ea507674e999`. Final runtime SHA256:
`01a3e3d084f6ca039ab78cc9545607f509afd4aff6852d47cb4f9ca1cc7f1608`.

This is a fresh reviewer verdict, following independent source and artifact
inspection. The earlier review remains historical evidence. The reviewer did
not run TTNN, open devices, start servers, change implementation, or conduct
hardware experiments. The stage owner's final-ready notification preceded
this verdict. The prescribed stage-only local checkpoint follows this review.

## Required Work

- None.

## Other Concerns

- No unresolved correctness, performance-selection, capability, or provenance
  finding remains in the reviewed decoder scope. Final defaults reproduce the
  selected correct decode programs within the declared 0.5% measurement-noise
  tolerance, while the final large-prefill improvements are incorporated and
  validated. The report uses reproduced default timings rather than a best
  earlier sample.
- Historical logs contain real failures. Their controlled repairs and rejected
  candidates remain visible; they are not acceptance waivers. In particular,
  global BFP8 KV has a real-input accuracy and latency justification, rather
  than a synthetic-only veto of BFP4.
- The generic report's derived core-count/FLOPs columns are unsuitable for
  quantifying DRAM-sharded matmul utilization on these captures. The README
  discloses the contradiction with raw rows. Explicit program attributes,
  complete raw kernel timings, reader comparisons, and physical-byte accounting
  support the optimization conclusions independently.

## Hard-Check Gaps

- Native-context checks demonstrate capacity, indexing, chunk handoff, exact
  traced decode, and a full-attention cache-consuming oracle. They do not
  constitute an HF rollout of one contiguous 262144-token prompt. Real-HF
  comparisons extend to 8001 tokens; longer and batched fixtures reuse
  explicitly documented recorded activation segments. This matches the
  preserved decoder contract and is accurately disclosed.
- Native context is validated at batch 1; batches through 32 use shorter
  per-user caches. No full batch-by-native-context capacity claim is made.
  The 2048-token prefill-trace control likewise does not claim native-size
  prefill-trace allocation.
- This layer-only stage cannot establish full-model top-k accuracy, generated
  text quality, sampling, multichip behavior, or serving readiness. None is
  presented as validated or required for this stage.

## Anomaly Ledger

- Observed anomaly: reduced-precision projection packing initially lost real
  accuracy, and random-weight BFP4 diagnostics have much lower PCC.
  Evidence: precision/packing candidate records, source archives,
  `candidate_measurements.csv`, and final real-input gate logs.
  Affected path: attention and MLP projection construction and precision choice.
  Control or comparison: device BF16-to-BFP4 conversion versus direct packing
  from original checkpoint tensors; independent real-weight attention/MLP
  dtype and fidelity comparisons.
  Likely subsystem: quantization and double rounding; synthetic distribution
  sensitivity is a separate diagnostic.
  Investigation performed: inspected packing code, real activation provenance,
  candidate timing/PCC rows, and actual final profiler dtypes/fidelities.
  Resolution: fixed. Original-host packing and BFP4/LoFi projections pass the
  unchanged real-output PCC 0.995 bar. Synthetic diagnostics retain finite-output
  and exact-replay checks and do not veto the real-weight winner.

- Observed anomaly: mixed BF16/FP32 sharded residual addition was not exact on
  restored trace replay.
  Evidence: `AUTOFIX_sharded_trace.md`, isolated controls, and trace regression
  logs/source snapshots.
  Affected path: linear-attention residual handoff during decode.
  Control or comparison: operand/state isolation and homogeneous FP32 addition
  followed by one BF16 rounding.
  Likely subsystem: mixed-dtype sharded binary operation.
  Investigation performed: inspected the repair, ownership/lifetime handling,
  and exact eager/trace/post-stress output and state checks.
  Resolution: fixed. Final runtime uses the homogeneous operation, and final
  correctness, batch, stress, and Watcher evidence passes.

- Observed anomaly: batch-dependent L1 overlap, prime-batch rotary pressure,
  and SDPA query shard-grid mismatch.
  Evidence: `AUTOFIX_batch_l1.md`, batch repair history, and
  `logs/final_release_v2_{batch,watcher_batch}.*`.
  Affected path: batched decode, recurrent-state placement, partial rotary,
  and full-attention SDPA input layout.
  Control or comparison: compact logical batch rows, measured persistent and
  intermediate L1 budgets, direct DRAM rotary tails, and the actual SDPA reader
  grid; DRAM and sharded borrowed-input controls.
  Likely subsystem: allocation and consuming-op shard contracts.
  Investigation performed: inspected selected layout code and all 66 final
  batch trace records, recursively checking 1464 exact comparison leaves.
  Resolution: fixed. Every exact comparison has equality true and maximum
  absolute difference zero. Per-user prefill/decode minima are
  0.998722/0.998429 for linear and 0.998267/0.996066 for full attention.

- Observed anomaly: unaligned continuation retained sharded decode outputs,
  causing allocation/concat specialization failure and a subsequent LLRT wait.
  Evidence: `AUTOFIX_final_continuation.md`, linked triage evidence and
  continuation-resource controls.
  Affected path: leading decode tokens within unaligned prefill.
  Control or comparison: immediate DRAM collection versus retaining sharded
  tensors; concat-batching-only control; all-DRAM recursive concat with 257 and
  1024 inputs.
  Likely subsystem: output lifetime/allocation, followed by generic LLRT
  binary-cache exception safety.
  Investigation performed: inspected continuation collection and borrowed-input
  ownership, the isolated controls, and final arbitrary-length/native gates.
  Resolution: fixed for this model path. Immediate DRAM collection removes the
  failing specialization and preserves exact replay. The generic LLRT failure
  recovery hazard is not claimed repaired upstream.

- Observed anomaly: BFP4 KV passed the short pair but failed real-input users
  8 and 26 in batch 32 and as identical extracted batch-1 inputs.
  Evidence: `AUTOFIX_cache4_batch.md`, `cache_precision_summary.json`,
  `test_cache_precision_regression.py`, and archived diagnostic runs.
  Affected path: full-attention cache precision and cache-consuming output.
  Control or comparison: exact page mapping, untouched rows, independent cache
  updates, CPU SDPA over dequantized cache, BFP8 KV, and legal mixed K8/V4.
  Likely subsystem: numerical cache sensitivity rather than address corruption.
  Investigation performed: inspected controls and measured mixed-cache
  adaptation, including its two update operations.
  Resolution: controlled. Global K8/V8 passes. K8/V4 passes accuracy but traces
  at 0.432902 ms versus 0.427814 ms for K8/V8, so the correct faster decode
  choice is retained without a batch-specific fallback or weakened gate.

- Observed anomaly: reader profiling lost markers and some narrow reader-3
  configurations initially failed legality checks.
  Evidence: `AUTOFIX_reader_profiler.md`, `reader_comparison_summary.*`, and
  `tracy/*/reader_legal_confirmation/` raw rows and accounting.
  Affected path: evidence for DRAM matmul reader selection.
  Control or comparison: support-count 8000, separate profiler drains,
  alternating complete groups, and common legal per-core N18 padding.
  Likely subsystem: profiler buffer capacity and reader alignment constraints.
  Investigation performed: independently rederived all 66 profiled cases from
  raw signposted windows and checked the corresponding 66 unprofiled cases.
  Resolution: fixed. Every complete window contains the expected matmul and
  reader count; minimum PCC is 0.999771377. Selected readers win both actual
  kernel duration and unprofiled trace comparisons.

- Observed anomaly: an older larger-prefill-grid rejection did not predict
  performance under the final separate-MLP topology; the promoted grid then
  exposed inefficient 1x1 MLP subblocks.
  Evidence: `final_prefill_grid*_plan*`, `final_prefill_mlp_grid_plan*`,
  `final_prefill_subblock*_plan*`, and the final candidate ledger.
  Affected path: large-prefill projection program configurations.
  Control or comparison: final-policy 8x8, 10x8, 8x10 and 11x10 grids;
  K4/8/16/32/64/128; successful smaller-output adaptations after L1 failures;
  full-width MLP N35 with legal M1; both 7x1/1x7 subblock orientations,
  MLP-only/all-role changes, and cap8.
  Likely subsystem: matmul geometry and register/output blocking.
  Investigation performed: checked candidate measurements, archived overrides,
  production defaults, final raw program rows, and the scoped rerun decision.
  Resolution: fixed. Final large prefill selects 11x10/K16 and subblock bounds
  1x7, producing actual 1x6/1x7 subblocks. Legal adapted alternatives lose.
  Remaining generic SLOW labels are supported by measured controls, not accepted
  solely because the renderer calls a configuration reasonable. Fixed-shape
  prefill tracing was also measured and validated to address dispatch advice.

- Observed anomaly: native chunk-2048 versus chunk-1024 tail PCC changed from
  1.0 to approximately 0.999958/0.999950 after the large-prefill promotion.
  Evidence: final v4 long logs, source diff, and `../context_contract.json`.
  Affected path: large/small prefill handoff for linear/full attention.
  Control or comparison: unchanged data/weights with different selected K-block
  sizes; real-HF 8001-token comparisons and native exact-cache decode oracle.
  Likely subsystem: floating-point accumulation reassociation.
  Investigation performed: verified the changed program branch, preserved
  chunk-invariance threshold 0.999, unchanged final subblock-trial PCC, and
  final native/HF results.
  Resolution: controlled. The delta is explained and above its unchanged gate;
  both kinds still support 262143/262144 prefill and decode position 262143.

- Observed anomaly: final decode reports show 8 derived cores and sometimes
  over 100% FLOPs for DRAM-sharded matmuls, while raw rows mark CORE COUNT 110.
  Evidence: final `decode_perf_report.txt`, source `decode_ops.csv.gz`, explicit
  program attributes, and `decode_accounting.json` for both kinds.
  Affected path: derived utilization interpretation, not measured outputs.
  Control or comparison: raw duration/reader rows, dedicated reader profiles,
  and bandwidth computed from physical BFP tile bytes and reader padding.
  Likely subsystem: generic report renderer's derived-core model.
  Investigation performed: reconciled final kernel sums and per-role mean
  durations directly from four complete replay sessions.
  Resolution: controlled. The final README excludes these derived columns from
  its utilization evidence and reports sane physical-byte accounting.

- Observed anomaly: two pytest SWIG deprecations, subset-MMIO notices, and an
  initial final-health command returning 1.
  Evidence: final gate logs and
  `logs/final_device_health{,_list}.provenance.json` with compressed logs.
  Affected path: test/tool environment and cleanup checks.
  Control or comparison: separate successful Watcher runs, successful profiles,
  and corrected noninteractive `tt-smi -ls --local` returning 0 with four devices.
  Likely subsystem: metadata warnings, intentional one-chip selection, and CLI
  mode; the failed command explicitly reports no TTY for its interactive UI.
  Investigation performed: read actual warning/error text and successful
  follow-up records. The task-owned profiling server cleanup is recorded.
  Resolution: controlled. No final device assertion or unclassified hardware
  failure is present; no reset was needed for the CLI-mode error.

## Scope Inspected

- Goal/skill paths: original full stage prompt at
  `/home/hous/dev/ornith-1.5-9b/state/multigoal/03-03-optimized-decoder.prompt.txt`;
  repository and model `AGENTS.md`; `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/optimize/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`; relevant repository LLM guidance.
- Code paths: `tt/optimized_decoder.py`; inherited fused/functional decoder
  boundaries; optimized correctness, trace, batch, continuation, prefill-trace,
  profile and cache-precision tests; candidate/geometry harnesses and runners;
  pinned HF reference and activation-recording code. Construction-only torch
  packing is separated from measured TTNN execution. Delivered tests bind the
  optimized implementation and forbid the functional block fallback.
- Artifact paths: final `README.md`, `work_log.md`, `checklist.md`,
  `../context_contract.json`, `final_gates_summary.json`,
  `final_default_measurements.json`, `prefill_trace_measurements.json`,
  activation manifest, candidate CSV/compressed JSON, geometry inventories,
  final combined topology plans, AutoFix/triage reports, source archives, and
  baseline/final/reader raw profiler evidence.
- Provenance checks: at the final bulk audit, all 266 completed compressed logs
  matched recorded decompressed hashes, all 266 source archives matched their
  contained source hashes, and all 356 available recorded compressed-archive
  hashes matched. Independently checked all 208 candidate rows against raw
  logged PCC/timings and recomputed medians. Final host/cleanup records were
  checked separately afterward. No mismatch was found. The recorded layer-0
  and layer-3 activation files matched their manifest hashes and shapes.
- Final acceptance: 73 short + 9 long + 73 Watcher + 2 paired performance +
  2 prefill trace + 2 Watcher prefill trace cases pass on the frozen v4 runtime.
  The 85 batch/resource and 3 targeted batch Watcher cases are retained from v2
  only after comparing archived source and confirming their physical prefill
  sequences stay below 2048 and every decode program is unchanged. Total: 249
  passing cases. No long gate is omitted by relying on its short-run deselection.
- Final unprofiled measurements, rederived from `final_release_v4_pair`:

  | Kind | Fused / optimized prefill ms | Fused / optimized traced decode ms | Optimized HF prefill / decode PCC |
  |---|---:|---:|---:|
  | Linear | 26.258161 / 6.983336 | 1.461685 / 0.524047 | 0.998982 / 0.999388 |
  | Full | 23.509834 / 5.444302 | 1.263289 / 0.427504 | 0.998614 / 0.999037 |

  The same warmed 2048-token, batch-1 harness compares both implementations on
  one Blackhole chip on P300c boards. `p150` identifies the one-chip topology,
  not a measured P150 board. Transfers, state restoration and HF checks remain
  outside timing. Final 32-step linear HF stress PCC is 0.999229.
- Final profiler verification: all four `optimized_release_v4` source windows
  match hashes and current runtime provenance. Prefill contains 45/34 device
  operations and kernel sums 6.672248/5.240061 ms. Decode contains four complete
  replay sessions and the selected per-role BFP4/LoFi reader programs. Directly
  rederived decode kernel sums are 0.49906925/0.411805 ms, firmware spans
  0.54426479/0.45733658 ms, and same-run host times 0.56247599/0.47104101 ms.
  Kernel < firmware span < host in each run. Bandwidth floors
  0.240768/0.241544 ms use physical weight/cache bytes and the explicitly named
  512 GB/s report model; they are not a promise of attainable latency.
- Capacity accounting: independently recomputed native BFP8 KV as 570425344
  bytes (544 MiB), native page-table entries as 16384 bytes, duplicated BFP4
  projection storage as 246104064 bytes linear / 237109248 bytes full, and
  recurrent state as 2097152 bytes per user. The 300/224 KiB per-bank L1 limits
  produce the documented batch-16/batch-12 placement boundaries. Context
  remains 262144; no capability reduction or hidden public alignment cap exists.
- Commands run by the reviewer: read-only `rg`, file reads, `git status`,
  `git diff`/`git diff --check`, and small standard-library Python scripts for
  JSON/CSV/gzip/hash/median/trace-window analysis, source diffs, documentation
  link checks, and AST parsing of 13 runtime/candidate/optimized-test files.
  No TTNN import or hardware command was run by the reviewer. The owner's
  `final_host_checks_v4` pre-commit record returns 0; Python/tests/docs changes
  require no C++ build.

## Residual Risk

- This is evidence for two representative decoder layer kinds on the current
  chip/build and pinned real checkpoint. A new runtime/build, shape regime,
  allocation topology, or precision policy requires appropriate revalidation.
- Layer accuracy does not predict cumulative full-model quality. Optional
  million-token YaRN setup is inherited and explicitly not a million-token
  full-layer validation claim.
- The generic LLRT recovery hazard after a failing binary specialization and
  the report renderer's derived-utilization limitation remain outside the
  repaired decoder path. Both have concrete controls and accurate disclosure;
  neither conceals a failing final acceptance path.
