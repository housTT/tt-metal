# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The stage intentionally supports only the exact 1x4 P300c mesh. This is
  consistent with the user contract, which explicitly does not require
  smaller or alternative meshes.
- Ethernet watcher instrumentation remains disabled because the source-built
  ACTIVE_ETH watcher image exceeds the Blackhole config-buffer allocation.
  This is controlled by target-mesh fused-CCL probes, all-replica checks,
  repeated traces, and a clean selected-path Tensix watcher run; it is not a
  decoder correctness failure.
- The full-stack memory result is a calculated decoder-layer-stack capacity
  plan plus maximum-context layer execution, not a simultaneous 64-layer
  allocation. The calculation preserves the 262,144-token contract with
  15.2 GiB/device remaining for non-decoder components. Simultaneous
  coexistence belongs to the later full-model stage and does not reduce this
  stage's advertised capability.

## Hard-Check Gaps

- None for the multichip-decoder stage. The stage-owned checkpoint commit and
  SHA log are deliberately performed by the stage owner after this clean-pass,
  as required by the stage-review workflow.

## Prior-Finding Disposition

- Real whole-layer selected-topology evidence: fixed. The new
  `test_real_weight_replicated_vs_fractured_decode_perf` uses real checkpoint
  weights and independent recurrent states/KV caches, checks every physical
  output replica, excludes the one-time fracture and validation gather from
  both measured traces, and runs 50 warmed replays. The retained log reports
  linear 772.178 us replicated versus 871.299 us fractured and full 551.352 us
  versus 652.115 us, with four-device PCC 0.999917/1.0. The final default
  `decode_forward` is therefore the measured winner; module documentation,
  README, work log, and context contract consistently select the replicated
  residual contract. Both fractured layer graphs remain independently
  correctness- and trace-valid as a rejected lower-payload family.
- Exhaustive BFP4 geometry/profiler disposition: fixed. The exact TP-local
  gate/up shape has tiled `(K,N)=(160,136)` and down has `(136,160)`, limiting
  an unpadded DRAM-sharded program to eight workers. The BFP4/LoFi sweep covers
  all useful legal K-block divisors: gate/up `{20,10,5,4,2,1}` and down
  `{17,1}`, on all four devices. Block 10 wins gate/up at 57.787 us and block
  17 wins down at 60.470 us. The only useful 16-worker candidate pads 4,352 to
  4,608, passes real-layer PCC, and regresses authoritative profiler rows from
  about 43 us to 45.7--46.3 us. The production code and local-contract test
  encode blocks 10/10/17. The selected Blackhole/110-worker raw, CSV, summary,
  image, and human-readable reports exist. Their remaining `SLOW` label is
  specifically the report's inability to derive an output subblock for this
  DRAM program family, not an untested geometry.
- Decoder-shape fused collective disposition: fixed. The generic fused
  reduce-scatter/matmul and all-gather/matmul probes pass on the target ring.
  `autofix/fused_decoder/AUTOFIX.md` records the exact decoder-shape program
  blocker: MinimalMatmul programs, dedicated CCL cores, persistent buffers,
  two or three global semaphores, shared gather-buffer ownership, and a
  precision-compatible residual are required. These resources/layouts are not
  a drop-in substitution for the selected L1/DRAM-sharded graph. This is an
  earned graph-family rejection rather than an unsupported-API assertion.

## Goal-Contract Evidence

- Real implementation and baseline: `tt/multichip_decoder.py` defines
  `MultichipDecoder(OptimizedDecoder)`, validates the exact Qwen3.6-27B shape
  and 1x4 mesh, and keeps all runtime operations device-side. Column- and
  row-parallel weight materialization, local full-attention heads/cache,
  local linear-attention state, ring reductions, and the optional distributed
  residual/norm graph are implemented rather than mocked.
- Mesh and tensor strategy: `README.md` records the 4x Blackhole P300c 1x4
  ring selected before final-path coding, global/per-device tensor shapes,
  DRAM and Tensix shard specs, activation and residual contracts, padding,
  collective payloads, page/state ownership, and rejected DP/TP2/2D/sequence
  alternatives. Qwen3.6-27B is dense, so MoE/expert routing is correctly
  classified as not applicable.
- Capacity contract: `doc/context_contract.json` recalculates TP-local weights,
  paged KV cache, batch-32 recurrent state, and a 4 GiB reserve. The arithmetic
  is internally exact: 10,035,920,896 + 2,281,701,376 + 1,459,617,792 +
  4,294,967,296 = 18,072,207,360 bytes, or 16.831 GiB/device. The HF-advertised
  262,144-token context is preserved with no capability reduction.
- Correctness against the direct optimized baseline:
  `correctness/direct_optimized/{baseline,tp4_compare}.log` runs the same
  deterministic harness in one-device `OptimizedDecoder` write mode and TP=4
  compare mode. Direct optimized-vs-TP PCC is 0.999504/0.999411/0.999820 for
  linear prefill/decode/trace and 0.997760/0.999443/0.999568 for full
  prefill/decode/trace. Every replicated physical device tensor is separately
  checked as TILE/BF16/DRAM `[B,1,S,5120]` and replicas agree at PCC 1.0.
- Paged cache and stack contracts: local-contract tests prove six Q heads and
  one KV head/device, cache `[blocks,1,64,256]` per tensor/device, four local
  key heads, twelve local value heads, recurrent state `[B,12,128,128]`, and
  convolution state `[B,1,4,2560]`. Real full-attention tests exercise page
  tables, positions, cache prefill/update/read, and warmed replay. Batch-32
  tests cover both layer kinds and all replicated boundaries. The selected
  stack input/output contract is identical replicated TILE/DRAM layout; the
  alternative fractured input/output contract also passes both real layer
  kinds and gathers only at the test boundary.
- Logical lengths and advertised context: real-weight lengths 65 and 33,
  forced chunking, and full prefill at 32,769 prove alignment stays internal.
  Full prefill at 262,144, full traced decode at position 262,143, and linear
  prefill/final traced decode at 262,144 are retained in `context/`. No public
  tile/page/chunk-alignment restriction was introduced.
- Trace, repeatability, fallback, and watcher: both meaningful layer kinds pass
  warmed decode trace replay; repeated replay PCC is 1.0. Batch-32 and
  all-replica coverage match the mesh risk. The static runtime audit rejects
  host conversion, Torch calls, layout fallback, reshard, and untilize/tilize
  calls in multichip runtime methods. The consolidated final acceptance log is
  10 passed. The final selected watcher log records 2 passed and clean detach
  for devices 0--3 with no runtime error/assert/NoC/hang/timeout signature.
- Performance and profiling: like-for-like real-weight warmed trace results are
  1,456.214 -> 772.178 us for linear (1.886x, 47.1% TP efficiency) and
  1,133.241 -> 551.352 us for full (2.055x, 51.4%). Authoritative Blackhole
  `tt-perf-report` artifacts contain human-readable tables, CSVs, provenance,
  raw device rows, and summary plots. They identify full/linear modeled DRAM
  rooflines of 39.6%/22.0%, dominant BFP8/BFP4 matmuls, cache updates, layout
  transformations, and both ring reduce-scatter/all-gather pairs. The
  communication and BFP4 anomalies are investigated rather than dismissed.
- Documentation and scope: `README.md` and `work_log.md` contain commands,
  PCC, latency, topology, limitations, and exact artifact paths. The live
  worktree changes are confined to the requested multichip implementation,
  tests, context contract, and multichip documentation; the pre-existing Tracy,
  UMD, and cluster-descriptor entries remain separately identifiable.

## Anomaly Ledger

- Observed anomaly: the lower-payload fractured micrograph beat the replicated
  boundary by 6.8--8.6%, while the real decoder selected the opposite topology.
  Evidence: `topology_probe.log` and
  `performance/fractured_selected/real_weight_topology_selection.log`.
  Affected path: decoder-stack residual and collective contract.
  Control or comparison: real checkpoint weights, independent state/cache,
  both layer kinds, 50 warmed replays, and all-device output PCC.
  Likely subsystem: two distributed RMSNorms and gathered-normalized activation
  conversion absent from the synthetic boundary's apparent communication win.
  Investigation performed: implemented the full fractured graph, validated it,
  then A/B measured it against the replicated graph without boundary adapters.
  Resolution: controlled; replicated is the measured production winner.

- Observed anomaly: the selected BFP4 MLP rows remain labeled `SLOW` with no
  output-subblock recommendation.
  Evidence: `profiler/linear_decode/selected_bw10_report_blackhole.*`.
  Affected path: linear-attention TP-local MLP.
  Control or comparison: exhaustive same-policy exact-shape block sweep and
  the real-layer-correct 4,608/16-worker profiler candidate.
  Likely subsystem: `tt-perf-report` advice modeling for DRAM-sharded BFP4
  programs, plus the exact shapes' eight-worker divisibility constraint.
  Investigation performed: swept all legal exact blocks; profiled the selected
  graph and the only useful higher-core padded candidate.
  Resolution: controlled; exact 4,352 with blocks 10/10/17 is fastest.

- Observed anomaly: the first fractured MLP attempt failed with
  `bad optional access`.
  Evidence: `correctness/fractured_stacked_decode.log` and `work_log.md`.
  Affected path: gathered distributed-norm output entering the MLP.
  Control or comparison: converting to the inherited 8/16-core L1 norm
  contract makes both layer kinds pass trace and repeat PCC.
  Likely subsystem: DRAM-sharded matmul input memory contract.
  Investigation performed: isolated the layout mismatch and reran both graphs.
  Resolution: fixed.

- Observed anomaly: the initial full-attention TP MLP hit a Blackhole
  L1/static-CB overlap.
  Evidence: `autofix/full_decode_l1/{AUTODEBUG,AUTOFIX}.md` and passing full
  acceptance/context logs.
  Affected path: full-attention decode MLP.
  Control or comparison: gate spill and every smaller legal block were refuted;
  TP-local 4,352 -> 4,608 algebraic-zero padding enabled 16 cores.
  Likely subsystem: program geometry and Blackhole L1 allocation.
  Investigation performed: `$autofix` hypothesis isolation and real-layer
  reruns.
  Resolution: fixed.

- Observed anomaly: ACTIVE_ETH watcher instrumentation exceeded its config
  buffer before decoder execution.
  Evidence: `watcher/pytest.log` and
  `watcher/final_selected_no_eth/{pytest,watcher}.log`.
  Affected path: watcher instrumentation of Ethernet kernels.
  Control or comparison: fused CCL probes, all-replica PCC, repeated traces,
  final selected Tensix watcher run, and normal four-device detach.
  Likely subsystem: watcher image size, not model execution.
  Investigation performed: retained the failure, disabled only Ethernet
  instrumentation, and reran the selected graph.
  Resolution: controlled with documented instrumentation limitation.

- Observed anomaly: the test name/docstring for
  `test_fractured_stacked_decode_trace` still describes the fractured graph as
  selected even though it is now a rejected alternative.
  Evidence: test source versus module, README, context contract, A/B assertion,
  and final selection log.
  Affected path: internal test labeling only.
  Control or comparison: production `decode_forward` remains replicated and
  the A/B test explicitly asserts replicated latency is lower.
  Likely subsystem: historical test naming after topology reselection.
  Investigation performed: traced production/default dispatch and selection
  evidence.
  Resolution: controlled; no runtime or user-facing contract ambiguity.

## Scope Inspected

- Goal/skill paths: the user's full multichip-decoder contract;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/multichip/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`; and `tech_reports/LLMs/llms.md`
  section 3.3.
- Artifact paths: all files under
  `models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/`, including README,
  work log, direct-baseline, consolidated correctness, context, topology,
  performance, profiler, watcher, fused-probe, and autofix evidence; plus
  `doc/context_contract.json`.
- Code paths: `tt/multichip_decoder.py` and
  `tests/test_multichip_decoder.py`, with relevant inherited optimized and
  functional interfaces inspected where needed to establish dispatch.
- Commands run: read-only `sed`, `cat`, `find`, `wc`, `rg`, `tail`,
  `git status`, `git branch`, `git rev-parse`, `git diff --check`, JSON arithmetic,
  and CSV inspection. No TT device was opened and no implementation, test, or
  documentation file was modified apart from this required review report.

## Residual Risk

- The selected path is a production-suitable decoder-layer-stack baseline on
  the exact four-chip mesh, with strong numerical, state/cache, trace,
  performance, profiler, and watcher evidence. Remaining integration risks are
  the normal next-stage work: simultaneous 64-layer allocation, embeddings,
  final norm/logits, and full-stack trace buffers. Those are expressly outside
  this goal and no advertised decoder capability was reduced to defer them.
