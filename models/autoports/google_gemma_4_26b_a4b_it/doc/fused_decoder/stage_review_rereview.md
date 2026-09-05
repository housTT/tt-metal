# Fused-decoder independent rereview

Verdict: clean-pass

Independent fresh-context review of stage 02 for
`google/gemma-4-26B-A4B-it`, completed on 2026-09-05. The required findings in
`stage_review.md` are resolved. Two additional adjacent-operation opportunities
identified during this review were measured and selected. The final artifact
checks below were performed after the implementation and tests were frozen.

## Required work

None remains from this review. The stage owner still performs the ordered
post-review local checkpoint and records its SHA; this report does not claim
that a commit or push has occurred.

## Review scope and snapshot

- Live branch: `hous/gemma-4-26b-a4b-it`; starting HEAD:
  `a80e3d2911f488c7e207b6cfcfcaedeb9d20be57`.
- Stage-owned changes: `tt/fused_decoder.py`, `tests/test_fused_decoder.py`,
  and `doc/fused_decoder/**` under this autoport. No other tracked-file change
  appeared in the final scope check.
- Final implementation SHA-256:
  `a9f3d0b776674ecad286500fbd86a5d2cf6bb2fbc1d606312724d484c8aea4ea`.
- Final test SHA-256:
  `fffff4af877208551844a2bd935c86dc4573a54c491c53de1716969db0aeb364`.
- Contract reviewed: `.agents/prompts/model_bringup_multigoal/02-fused-decoder.txt`
  and the supplied AGENTS.md instructions. Read the complete `stage-review`,
  `graph-fusing`, `functional-decoder`, and `tt-device-usage` skill files.
- Inspected the complete fused implementation and tests, relevant functional
  weight loading/forward/cache/trace/performance controls, mutable-buffer
  controls, canonical sparse configuration, GELU composite/unary contracts,
  binary activation support, final documentation, candidate JSONs, final
  correctness/context JSONs, four CSV/text report pairs, raw definitive
  profiler data, and the definitive watcher log.

The reviewer ran read-only source, AST, JSON, CSV, hash, scope, and context
checks. No TT device was opened, no hardware test/profile/server/reset was
launched, and no implementation file was edited by the reviewer. This report
is the only reviewer-authored file.

## Earlier findings resolved

| Finding | Independently checked resolution |
| --- | --- |
| Missing direct functional/fused equivalence | Eight tests construct separate decoder instances and separate identically initialized caches. They compare prefill and actual traced decode replay for dense-only, router-only, expert-only, and selected graphs across both layer kinds. All pass PCC 0.995. Required counters include both binary activations, packed projections, router folding, shared norm, expert-scale folding, and final scalar fusion on the selected path. |
| Accurate expert activation not assessed | The selected expert multiply consumes `UnaryWithParam(GELU, 0.0)`, matching the accurate unary contract. Definitive raw rows show this activation inside `BinaryNgDeviceOperation`; a separate expert GELU is absent. |
| Unmatched fast-expert rejection | `matched_fast_*` and corresponding accurate controls use the same surrounding graph at that comparison point. Both kinds pass HF PCC; fast composite prefill is slower, approximately 279.989/281.256 ms versus 278.484/279.685 ms. Its decode result does not establish a speed advantage. The documentation now labels the historical comparison accurately. |
| Missing scale/shared-normalization assessment | Setup folds router per-expert scale into expert down weights, and learned dense/expert norm weights into their packed projections. One unweighted RMSNorm feeds all three consumers. Isolated candidate PCC/performance artifacts and final combined direct equivalence support the retained rewrites. |
| Dominant packed-projection geometry | BF16/HiFi4 screens cover padding widths 1408/1536/1792/2048 and K blocks 1/2/4/8/11, with three screens each for blocks 4/8/11. Final raw rows confirm width 1536, 48 cores, block 4, and BF16/HiFi4. Reported DRAM utilization is about 62–63%, compared with approximately 10% in the earlier packed graph. The selected one-tile output per core explains its 1x1 output subblock. |
| Unsupported grouped-size rejection | Adapted `per_core_M` candidates made groups 64/128/256/1024 runnable. Recorded prefill times of 392.297/619.982/1072.950/3800.761 ms reject those candidates against the approximately 278.5 ms internal 32-token grouping. Public logical lengths remain unrestricted by that grouping. |
| Weak default/path checks | The host test asserts the complete resolved boolean/integer policy. Runtime wrappers and direct tests require the selected counters to advance. Final policies match the actual source defaults. |
| Composite GEGLU wording | Documentation distinguishes the composite operation from a fused binary activation. The final dense path uses the latter; it does not claim that `ttnn.geglu` itself is a single fused kernel. |
| Final artifact provenance | All 30 stage-root JSON artifacts match both final hashes. Four CSVs match the exact raw measured operations and durations. Four text files contain actual rendered tables. The watcher summary matches the preserved definitive raw log. |

## Findings closed during this rereview

- Dense GELU followed by multiply was still represented by the composite
  GEGLU path. The stage owner assessed a fast binary-input GELU under the
  same graph. The 200-replay full-attention comparison improved decode from
  1.50615 to 1.49672 ms with passing PCC, and the binary path was selected.
  Dense-only direct equivalence is PCC 1.0 for both phases and both kinds.
- Final residual add followed by a learned scalar multiply was unclassified.
  The scalar is now read at weight setup and supplied through
  `MUL_UNARY_SFPU` on the add. The matched 200-replay candidate improved
  sliding/full decode from 1.31564/1.49672 to 1.30943/1.48683 ms. Final
  selected direct and HF gates pass. Both definitive decode graphs contain
  the post-scalar add, two binary GELU operations, and no standalone GELU.
- A redundant BF16-to-BF16 router cast was removed by the stage owner during
  the review. It is absent from both definitive measured decode graphs.
- The initial definitive `.txt` files contained CSV-generation stdout.
  Regenerating without `--csv` restored the actual tables: 471 lines per
  prefill report and 80/82 lines for sliding/full decode. The final reports
  were inspected after regeneration and whitespace normalization.

## Re-derived final results

All device times below are sums of CSV `Device Time` in microseconds divided
by 1000. They exclude the separately reported operation gaps.

| Layer kind / phase | Functional device ms | Final device ms | Operations before / after |
| --- | ---: | ---: | ---: |
| Sliding prefill | 1242.488789 | 277.750937 | 557 / 456 |
| Sliding traced decode | 3.012500 | 1.268977 | 74 / 65 |
| Full prefill | 1243.617644 | 278.890973 | 557 / 456 |
| Full traced decode | 3.206965 | 1.451217 | 76 / 67 |

The canonical final host artifacts report 278.476241/1.310652 ms for sliding
prefill/decode and 279.606996/1.494537 ms for full attention. They use the
selected defaults, 20 additional warmups after the initial replay, and 200
measured trace replays. Documentation uses these reproduced values, rather
than presenting the earlier faster individual scalar-candidate screen as the
final result. The selected configuration is supported by matched candidate
comparisons and remains substantially faster than the functional graph.

| Correctness/capability check | Final result |
| --- | --- |
| HF PCC, sliding prefill/decode | 0.999306908 / 0.999716029 |
| HF PCC, full prefill/decode | 0.998004681 / 0.999900339; natural and shared physical cache views agree |
| Direct selected PCC, sliding prefill/traced decode | 0.999652082 / 0.999866329 |
| Direct selected PCC, full prefill/traced decode | 0.999699890 / 0.999902267 |
| Batch and logical lengths | Batch-2 prefill, batch-1/32 traced decode, and every recorded tile/page/window boundary pass the unchanged 0.995 threshold |
| Mutable trace buffers | A/B/A overwrites hidden state, RoPE, per-user positions, private permuted page tables, and both nonzero caches. All eager-control/replay and A-repeat PCC values are 1.0; A and B outputs differ |
| Sustained cache stress | 1104 bounded-cache trace replays; bounded/unbounded probes at 1023/1024/1025/1103 all have PCC 1.0 |
| Advertised context | Both kinds pass real-weight prefill at 262143 and 262144 and traced decode at position 262143, with sentinel preservation and repeat PCC 1.0 |
| Context metadata | Strict context checker passes: target=supported=262144. The contract file remains unchanged |

The lowest final boundary PCC is sliding length 1023 at 0.995163646.
Its functional control is 0.995126729, so this narrow acceptance margin was
not introduced by fusion. The full-prefill HF delta of approximately
-0.000453 remains above the acceptance bar and is accompanied by passing
direct TTNN equivalence and isolated arithmetic-rewrite controls; it is not
being dismissed as an unexplained failed gate.

## Profiler and watcher integrity

- Definitive raw captures are the sliding `2026_09_05_05_48_35` and full
  `2026_09_05_05_48_59` reports under the exact paths in `perf/README.md`.
- Sliding has 1089 distinct host operation IDs and 1089 device IDs; full has
  1093/1093. Neither has a missing or extra ID.
- The prefill CSVs match their exact signposted raw windows. Decode CSVs
  match the last measured replay session, after two prior replay sessions.
  All three sessions are complete with 65 sliding or 67 full operations.
  Every derived global-call ID and every per-row duration matches raw data.
- The final measured prefill has 64 sparse matmuls and one concat; decode has
  two sparse matmuls. Source and raw graph inspection found no new host
  fallback or unnecessary layout/reshard round trip. The compact row-major
  sparsity mask and inherited attention-head layout boundaries have explicit
  consumer contracts.
- `watcher/summary.json` identifies the two real-weight fused correctness
  nodes and their two-pass result. The 1220-line definitive log has SHA-256
  `959b7e198a53fad82725cb29d800f4e0006a98bc5c2fe922c9bdb153c036badb`.
  The recorded fatal-pattern scan was rerun and found no matches.

## Hard-check gaps and residual risk

- Historical candidate hashes identify earlier experiments, but their source
  snapshots are not all retained. Final selected evidence is independently
  tied to the current source and resolved policy; historical artifacts are
  not substituted for final correctness or performance evidence.
- Some nested oracle provenance still names a functional test. The fused
  provenance and measured-decoder path identify the actual execution. This
  does not obscure path identity after inspecting the wrappers and counters.
- Ordinary suite console output is summarized in the work log rather than
  fully archived. Numeric artifacts, current hashes, and test assertions
  support the recorded gate results; this is not a missing required gate.
- Long prefill probes prove capacity, exact logical output shape, and finite
  last-token output, rather than full-sequence HF PCC. They are supported by
  final boundary PCC and the unchanged functional long-attention path and
  controls. No context reduction is claimed or needed.
- Hardware results were assessed from retained evidence. Full-model
  generation, serving quality, and later optimization stages were not part
  of this decoder-only review. No C++/CMake change requires a build; both
  Python files parse, `git diff --check` passes, and the strict context check
  passes. The stage owner records the successful scoped pre-commit run.

Preserve the four git-ignored perf CSVs and the definitive watcher log in the
authorized stage checkpoint together with this report and the work log.
