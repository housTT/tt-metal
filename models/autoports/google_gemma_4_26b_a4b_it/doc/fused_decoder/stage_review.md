# Stage Review

Verdict: more-work-needed

Independent review of stage 02, fused decoder for `google/gemma-4-26B-A4B-it`.
The delivered correctness, context, watcher, and measured improvement evidence
is substantial and internally consistent. The required work below concerns
the missing direct equivalence gate and unsupported optimization closure.

## Required Work

- P2: Add the required direct unfused-versus-fused equivalence measurement.
  Evidence: `tests/test_fused_decoder.py:63-80,147-168` replaces the oracle
  module's `FunctionalDecoder` with `FusedDecoder`. The invoked functional
  test compares this one decoder to HF
  (`tests/test_functional_decoder.py:319-426`); it never constructs a second
  TTNN functional control. The final PCC artifacts consequently contain
  HF-versus-fused PCC, while the candidate timing harness checks shapes and
  contains no PCC comparison. Neither the candidate artifacts nor the final
  artifacts contain TTNN unfused-versus-fused PCC for the retained rewrites.
  Why this matters: graph-fusing Step 4 explicitly requires an on-device
  "PCC equivalence test (unfused vs fused)" and its verification section applies
  to every rewrite. Separate correlations with HF do not establish that
  correlation. This matters especially for router-scale folding, which moves
  rounding boundaries before expert selection, and packed decode gate/up,
  which gives the up projection the gate's compute configuration.
  Required next step: retain a real-weight, real-shape equivalence test with
  distinct functional and fused instances and independent identically seeded
  caches. Compare prefill and traced decode outputs for both layer kinds at
  the unchanged 0.995 bar. Include controls isolating the retained arithmetic
  rewrites, or their intermediate outputs, so each rewrite's acceptance can
  be established. Keep the existing HF and cache/trace gates.

- P2: Finish the graph-fusing assessment; directly applicable patterns remain
  untried and unclassified.
  Evidence: `_packed_expert_activation` ends in `apply_geglu`
  (`tt/fused_decoder.py:307-329`), whose implementation is a separate accurate
  `ttnn.gelu` followed by `ttnn.mul`
  (`models/demos/gemma4/tt/experts/operations.py:14-18`). TTNN binary bindings
  accept `input_tensor_a_activations` as parameterized unary operations
  (`ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp:119-132,269-309`);
  accurate GELU is exactly `UnaryWithParam(UnaryOpType::GELU, 0.0f)` in
  `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:535-552`. Folding that
  activation into multiply is an explicit graph-fusing pattern, but no
  candidate or blocker exists. The existing `ttnn.geglu` experiment is a
  different implementation: its C++ body splits, launches fast GELU, and then
  launches multiply
  (`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_composite_op.cpp:278-290`).
  It is not a single specialized fused kernel.
  Additional concrete omissions are the constant `router_per_expert_scale`
  multiply (`tt/fused_decoder.py:303`), which can be assessed for folding into
  each expert's down weights, and the three RMS normalizations of the same
  post-attention residual in the dense/router/expert branches
  (`tt/fused_decoder.py:187-192,235-240,285`), which merit a common-normalized-input
  graph assessment with the distinct constant norm weights folded into their
  consumers. The README's statement that other scales are data-dependent or
  after residuals does not cover these constants or shared inputs.
  Why this matters: the stage goal requires all applicable patterns to be
  assessed and tried when possible. These are concrete remaining graph
  transformations, including one expressly listed in the skill; a broad
  "exhaustive" table does not reject them.
  Required next step: measure the accurate GELU-input-activation fusion on
  the selected packed graph in prefill and decode, adapting rank/layout if
  necessary. Assess and test the constant-scale and repeated-normalization
  rewrites, or record an exact semantic/op-contract blocker. Preserve passing
  real-weight equivalence and measure combined latency before selecting them.
  Correct the description of `ttnn.geglu` to reflect its actual composite
  dispatch graph. Document the resulting op sequence and movement boundaries.

- P2: Repeat the fast expert activation comparison with matching surrounding
  fusions before rejecting the compatible combined candidate.
  Evidence: `candidate_runs/expert_geglu_only_layer0_sliding_attention_seq1024_batch1_host_timings.json`
  explicitly disables dense packing, dense GeGLU, router folding, and branch
  fusion, enables packed experts and fast expert GeGLU, and measures
  704.731/2.513 ms. The selected current-source run enables dense packing,
  dense GeGLU, and router folding and measures 704.928/2.456 ms. There is no
  timing artifact with the selected surrounding graph plus fast expert GeGLU.
  The comparable historical packed-expert-only accurate result is
  705.608/2.510 ms, but has a different source hash and only one retained
  screen. The fast candidate's real-weight PCC passes at
  0.999205/0.999694. The README and work log reject it by comparing its
  2.513 ms to the selected combined graph's 2.456 ms.
  Why this matters: this comparison cannot determine whether the compatible
  combined candidate beats the default. Its observed real-weight PCC passes,
  so a small accuracy delta alone is not evidence of a correctness blocker.
  Required next step: on one current source, compare the selected graph with
  only the expert activation changed, using the same dtype/fidelity, weights,
  trace regime, and representative layer kinds. Repeat enough to resolve the
  timing difference. Include the actual fused accurate-activation candidate
  from the preceding finding. Select and report the best correct combined
  graph, or retain a measured, comparable rejection.

- P2: Resolve the dominant packed-projection geometry anomaly under the
  selected precision policy.
  Evidence: the final sliding prefill CSV reports 32 packed gate/up sparse
  matmuls totaling 544.083 ms, or 76.8% of total kernel time. The matching
  decode op is 1.291157 ms, or 53.4% of kernel time. Both are classified
  `SLOW`, use four cores, BF16/HiFi4, `in0_block_w=1`, and 1x1 output subblocks;
  measured DRAM utilization is about 11.7% prefill and 9.8% decode. Full
  attention shows the same anomaly. The source uses the inherited
  `_build_sparse_matmul_config` unchanged for the new packed width 1408
  (`tt/fused_decoder.py:342,424`). No artifact records a precision-matched
  packed geometry or padded-weight candidate. The monolithic 1024-token
  rejection only establishes that its unchanged eight-core down configuration
  requests too many M blocks; it does not establish a hardware-wide limit.
  Why this matters: stage-review explicitly requires investigating a dominant
  `SLOW` row with low utilization and unexplored legal block/core geometries.
  This finding is scoped to the new packed projection and the geometry used
  to reject grouped fusion, not to beginning a later autoport stage.
  Required next step: measure material legal K-block/output-subblock/core
  alternatives for the packed projection under its actual selected BF16/HiFi4
  policy and accumulation flags. Include a padded packed-width candidate if
  that is needed to express a larger legal core grid. Earn the grouped-chunk
  rejection with an adapted configuration or an exact op/L1/divisibility
  blocker. Record measured results and the final bound; do not present the
  helper's current eight-core geometry as a physical device limit.

## Other Concerns

- Historical candidate files retain hashes but not historical source snapshots
  or the resolved defaults of every switch. An empty `environment_overrides`
  object at source `e3f427cf...` is not the same configuration as an empty
  object at final source `16904e08...`. Future comparison artifacts should
  record resolved switches as well as overrides. The four `current_source_*`
  baseline/selected artifacts are already directly comparable and valid.
- The nested inherited `provenance.exact_command` fields still name functional
  tests, while `fused_stage_provenance.exact_command` and
  `measured_decoder_path` identify the actual fused run. This does not make
  the measured path ambiguous after inspection, but commands should be
  labeled as inherited oracle metadata or normalized in the fused stamp.
- Four required perf CSVs and the raw watcher directory are git-ignored.
  Preserve them in the stage checkpoint with explicit force-adds. No local
  stage checkpoint exists yet; this is expected before review remediation.

## Hard-Check Gaps

- The selected-default host test checks the presence of positive-switch names,
  rather than their boolean values. Runtime method counters cover dense
  packing, router folding, and packed experts but do not independently assert
  the dense GeGLU counter. Current source and raw operations establish the
  selected path, so this is test hardening rather than a separate blocker.
- The work log summarizes the 17-pass default suite, opt-in runs, and
  pre-commit output without retaining their complete console logs. Current
  hashes across the regenerated artifacts corroborate the final source.
- Capacity probes establish successful real-weight execution and finite
  last-token output, not full-sequence HF PCC. This is an explicit inherited
  evidence split, supported by full-layer boundary PCC and the unchanged
  functional long-attention implementation/evidence, not a context reduction.

## Anomaly Ledger

- Observed anomaly: Existing `ttnn.geglu` was described as dedicated fusion
  while the accurate activation-plus-multiply merge was unassessed.
  Evidence: unary composite C++ body, binary activation binding, selected
  expert helper, final per-op CSVs.
  Affected path: Dense and expert activation graph assessment.
  Control or comparison: Fast composite candidate versus selected accurate
  helper; neither is an accurate GELU input activation inside multiply.
  Likely subsystem: Incomplete operation discovery and graph audit.
  Investigation performed: Inspected exact unary/binary C++ contracts and
  candidate switches, not only the operation names.
  Resolution: more-work-needed; required work above.

- Observed anomaly: Fast expert candidate was rejected using an unmatched
  combined-graph timing.
  Evidence: Candidate JSON switches and the 2.513 versus 2.456 ms comparison.
  Affected path: Final default selection.
  Control or comparison: Historical accurate expert-only 2.510 ms exists, but
  the current-source combined fast candidate does not.
  Likely subsystem: Experiment comparability.
  Investigation performed: Parsed all candidate provenance and timing fields.
  Resolution: more-work-needed; required work above.

- Observed anomaly: Packed sparse projection dominates runtime while using
  four cores and showing low utilization.
  Evidence: Both layer-kind final CSVs and inherited program builder.
  Affected path: Packed expert gate/up in prefill and decode.
  Control or comparison: Packed versus original graph is faster, but no
  precision-matched packed geometry control exists.
  Likely subsystem: Packed matmul geometry and weight layout.
  Investigation performed: Recomputed grouped op times and inspected all
  candidate files, block sizes, output subblocks, and source configuration.
  Resolution: more-work-needed; required work above.

- Observed anomaly: Fast expert GeGLU initially failed rank handling; large
  grouped prefill initially exceeded its chosen program's M-block geometry.
  Evidence: Work log and current rank-4 adaptation/internal 32-token split.
  Affected path: Exploratory fast activation and packed prefill.
  Control or comparison: Later passing real-weight, boundary, and capacity
  artifacts on final source.
  Likely subsystem: TTNN composite rank contract and sparse program geometry.
  Investigation performed: Inspected adapted code and current-hash evidence.
  Resolution: controlled for current correctness; broader optimization
  rejection remains subject to the required work above.

- Observed anomaly: Final full-attention prefill HF PCC is lower than the
  functional result by 0.000455.
  Evidence: Functional 0.998457 versus fused 0.998002; both full-cache views
  agree and all final boundary rows remain above 0.995.
  Affected path: Full-attention decoder output.
  Control or comparison: Real weights, same seed and HF oracle, both cache views.
  Likely subsystem: Changed arithmetic ordering and projection accumulation.
  Investigation performed: Recomputed the PCC delta and checked boundary,
  batch, and trace results. No HF acceptance failure was found.
  Resolution: controlled for the HF threshold; direct rewrite equivalence
  still requires the first finding's check.

## Scope Inspected

- Goal/skills: `.agents/prompts/model_bringup_multigoal/02-fused-decoder.txt`;
  repository `AGENTS.md`; complete `stage-review`, `graph-fusing`,
  `functional-decoder`, and `tt-device-usage` skill files.
- Snapshot: live worktree, branch `hous/gemma-4-26b-a4b-it`, HEAD
  `a80e3d2911f488c7e207b6cfcfcaedeb9d20be57`. Implementation and test hashes
  at review completion are respectively
  `16904e08cc3840a72aa828bf6f97092903223bceebfb71c399bec4db0b29980d` and
  `8e0c0dd7da2ea931b6d3831e8a02c59a000b8cb39655a191a48ac6be91155106`.
- Code: complete fused implementation/test; inherited weight setup, forward,
  attention/cache, dense/router/expert methods; functional HF, boundary,
  capacity, trace/stress and perf oracles; mutable-buffer test; canonical
  expert operations and sparse config; relevant TTNN GELU/binary code.
- Artifacts: all fused JSONs, README/work log, candidate matrix, four perf
  CSV/text pairs, watcher summary/raw log; context contract; functional
  correctness, perf and review controls; final v2 raw merged Tracy reports and
  raw device logs.
- Commands: read-only `git status`, branch/HEAD, `git diff --check`,
  `git check-ignore`, `rg`, `sed`, `nl`, `sha256sum`; Python JSON/hash,
  candidate-switch, CSV-duration/count, raw signpost/global-ID/trace-session,
  and AST analyses. The strict context checker passed:
  `python3 .agents/scripts/check_context_contract.py --model-dir models/autoports/google_gemma_4_26b_a4b_it --stage fused-decoder --require-contract --strict-caps`.
  No hardware command, TTNN import, pytest workload, server, profiler capture,
  reset, implementation edit, or pre-commit rewrite was run by this reviewer.

Validated results:

| Contract | Independently re-derived result |
| --- | --- |
| Path identity | Fused overrides own the material forward methods; monkeypatched oracle constructs the subclass; selected pack/router/expert counters must advance in correctness runs. |
| HF correctness | Sliding prefill/decode 0.999315708/0.999728306; full 0.998002294/0.999884896, identical across natural/shared cache views. Batch-2 prefill and every boundary row pass 0.995. |
| Trace/cache | Both layer kinds pass batch 1/32 HF trace PCC; eager/replay/repeat PCC is 1.0. A/B/A stable-buffer tests overwrite hidden, RoPE, per-user positions, independently permuted private page tables, and both nonzero caches; control and repeat PCC is 1.0. Bounded sliding stress records 1104 trace replays and PCC 1.0 at 1023/1024/1025/1103. |
| Logical lengths/context | Both kinds cover tile/page/window edges and 262143/262144 real-weight prefill capacity. Advertised-context traces decode at 262143 with sentinel preservation and repeat PCC 1.0. Strict context contract passes with target=supported=262144. |
| Current provenance | All final fused correctness/context JSONs and final host A/B artifacts match the implementation and test hashes above. Historical candidates are separately identified. |
| Device perf | Sliding before/after: prefill 1242.488789/708.482271 ms, decode 3.012500/2.416711 ms. Full: prefill 1243.617644/709.012354 ms, decode 3.206965/2.599937 ms. CSV `Device Time` is microseconds; totals divide by 1000. These are kernel-time totals, excluding reported op gaps. |
| Profile completeness | Final sliding has 1169 distinct raw operation IDs and 1169 device IDs, full 1173/1173, with no missing/extra IDs. Three complete trace sessions contain 72 ops each sliding and 74 each full. Every derived row belongs to the exact signposted raw window; raw durations equal derived totals. |
| Graph change | Prefill operation counts 557 to 494; decode 74 to 72 sliding and 76 to 74 full. Sparse matmuls fall from 96 to 64 prefill and 3 to 2 decode. Final 1024-token prefill has one concat. |
| Host performance | Same-final-source all-off versus selected: sliding 1243.063/3.052 to 704.928/2.456 ms; full 1244.202/3.230 to 706.109/2.653 ms, 20 warmed trace replays. The improvement over the functional graph is established. |
| Host fallback | Local AST/runtime source audit passes, with no Torch/from_torch/to_torch/full/reshard in fused hot methods. The inherited attention sharding transitions and sparse mask layout remain; no new host fallback was found. This does not independently prove that every inherited movement is irreducible. |
| Watcher | Raw log SHA-256 `3e990c41981943935ccccc99e5b370ed02f6bce78982970b115bd0f1f5a9c5b1` matches the summary; recorded fatal-pattern scan rerun has no matches. Summary gives exact two fused real-weight node IDs and a two-pass result. |
| Scope/build | Only fused source, fused tests, and fused docs are stage-owned changes. Both Python files parse, and `git diff --check` passes. No C++/CMake change requires a build. |

## Residual Risk

- Hardware behavior is assessed from existing evidence; the independent
  reviewer deliberately did not rerun device work.
- The final decoder is faster than the functional baseline. That measured
  result should be preserved while completing the remaining graph assessment;
  it does not establish that the best compatible fused candidate was selected.
- Model generation and serving are outside this decoder-only stage. No
  qualitative generation or full-model performance claim was reviewed.
