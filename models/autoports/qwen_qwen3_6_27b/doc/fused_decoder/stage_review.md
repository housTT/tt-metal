# Stage Review

Verdict: more-work-needed

## Required Work
- P1: The material grouped-convolution fusion family was not adapted and tested.
  Evidence: `README.md` lines 62-68 and 88, and `work_log.md` line 104, reject grouped `conv1d` because the graph needs 10,240 BF16 groups while `tests/ttnn/unit_tests/operations/conv/test_conv1d.py` lines 183-186 and 255-258 skip cases above 5,120 as OOM. That skip is not an op-contract proof. The same repository contains an adaptation for the analogous depthwise Mamba convolution: `models/demos/wormhole/mamba/tt/mamba_conv.py` lines 21-23 and 35-43 split 5,120 channels, and lines 84-114 execute per-split grouped `ttnn.conv1d` and concatenate the outputs. The current Qwen graph still spells the convolution out as concat + four slice/multiply terms + three adds + SiLU in inherited `FunctionalDecoder._linear_causal_conv_chunk` (lines 597-628), and separately as concat + multiply + reduction + SiLU in `FusedDecoder._linear_token` (lines 550-559). No split-2/split-4 (or other legal partition), source snapshot, correctness log, profiler report, or exact minimal-repro blocker exists under `doc/fused_decoder/candidates/`.
  Why this matters: The original contract requires every plausible graph-fusing pattern to be assessed and tried, with unnecessary movement removed from measured paths. The graph-fusing skill expressly requires adapting shape, layout, padding, or weight packing before rejecting a material dedicated-op replacement. This candidate targets a stateful, measured region that contributes several explicit operations and internal tilize/untilize movement in both linear prefill and decode, so a bare OOM skip at the unsplit geometry does not earn rejection.
  Required next step: Build and evaluate a channel-partitioned depthwise `ttnn.conv1d` candidate for both chunk prefill and token decode (for example, legal 2- or 4-way partitions that keep each BF16 group/channel count within the supported range). Preserve the rolling convolution-state semantics and final SiLU. Retain a source hash/snapshot, real-weight PCC for non-aligned prefill and decode, native-context finite/state coverage, warmed prefill and traced warmed decode profiler artifacts, and Watcher evidence if selected. If no adapted form can express the contract, retain a minimal repro that proves the exact blocker after partitioning/layout/packing attempts.

- P2: Material measured/rejected candidate claims lack the promised exact artifacts, and the work log prematurely records a clean review.
  Evidence: `README.md` lines 15 and 62-67 and `work_log.md` line 99 say direct `repeat_interleave` beat reshape/concat end to end, but neither `candidates/index.csv` nor `perf/candidates.csv` contains that comparison and there is no candidate bundle with its source hash, command, correctness, control latency, or profiler CSV. Likewise, the 2,356.778 us and 2,339.680 us identical-base controls cited for concat-heads and decode-RoPE appear only in prose/manifests, not as retained raw/report artifacts. This contradicts `README.md` line 56 and `work_log.md` lines 97 and 135-143, which claim exact candidate artifacts are retained. `work_log.md` line 152 also says the final stage review was `clean-pass`, but no final review artifact existed in commits `5d436142493e9a1085890b483cb86307877d9133` or `48ae700db537b250ee349fe5856ffe2d13f96627`, and this independent review is `more-work-needed`.
  Why this matters: The original contract requires rejected candidates, profiler conclusions, and exact artifacts to be recorded. The missing evidence covers a material movement-heavy choice and the controls used to reject two dedicated fusions; prose alone cannot establish that the final graph beats all correct comparable alternatives. The stale clean-pass statement is also an evidence-integrity contradiction.
  Required next step: Retain and index the exact source snapshots/hashes, commands, correctness logs, raw profiler CSVs, filtered reports, and numeric controls for the repeat-interleave versus reshape/concat comparison and the cited identical-base controls, or rerun them on the frozen final base. Remove the premature clean-pass claim, then update the review record only after remediation and a fresh independent clean review.

## Other Concerns
- The selected tile-tail candidate records 2,723.857 us traced linear decode, while the frozen final reproduction records 2,728.252 us. Diffing its source snapshot against the final implementation shows only two fusion-counter additions, so the device graph is identical and the 4.395 us (0.16%) difference is consistent with run-to-run noise. The final report correctly uses the slower reproduced number, but a repeated-sample policy would make the literal “beat the best correct traced-decode candidates” claim less ambiguous.
- The final correctness log ends with nanobind reference-leak diagnostics and the expected Transformers host-reference fallback warning. Neither appears in the measured TTNN runtime path, and all devices close normally; these are not stage blockers on the retained evidence.
- `source_manifest.sha256` freezes only implementation and test sources, not evidence files. Git commit provenance makes the evidence immutable enough for this review, but a complete artifact manifest would reduce stale-evidence risk.

## Hard-Check Gaps
- No retained multi-sample latency distribution or variance policy exists; final and candidate performance numbers are single signpost windows. This does not invalidate the large functional-to-fused gains, but it leaves sub-percent candidate differences underdetermined.
- The static no-fallback test inspects the fused override methods but not inherited measured methods. Direct inspection of `FunctionalDecoder.prefill_forward`, `decode_forward`, `_finish_layer`, `_finish_layer_chunked`, `_linear_causal_conv_chunk`, `_pad_linear_chunk`, and `_linear_chunk_inverse` found TTNN-only runtime code, so there is no concrete fallback finding; expanding the assertion would make the hard check match the full measured call graph.

## Anomaly Ledger
- Observed anomaly: The grouped `conv1d` family was rejected at the unsplit 10,240-channel geometry without trying the repository's established channel-splitting adaptation.
  Evidence: `README.md` lines 62-68 and 88; `work_log.md` line 104; `tests/ttnn/unit_tests/operations/conv/test_conv1d.py` lines 183-186 and 255-258; `models/demos/wormhole/mamba/tt/mamba_conv.py` lines 21-23, 35-43, and 84-114.
  Affected path: Linear-attention prefill convolution and traced decode convolution.
  Control or comparison: The retained spelled TTNN convolution in `FunctionalDecoder._linear_causal_conv_chunk` and `FusedDecoder._linear_token`; no adapted fused-op control exists.
  Likely subsystem: Dedicated convolution op/layout/weight packing and channel partitioning.
  Investigation performed: Inspected the claimed repository OOM boundary, searched repository `conv1d` implementations, and found the Mamba split-channel workaround.
  Resolution: more-work-needed

- Observed anomaly: A material repeat-interleave alternative and two identical-base performance controls are claimed but not retained or indexed.
  Evidence: `README.md` lines 15, 56, 62-67, 81, and 83; `work_log.md` lines 97-107; absence from `candidates/index.csv`, `perf/candidates.csv`, and the candidate directory listing.
  Affected path: Linear Q/K replication, full prefill concatenate-heads, and full decode RoPE selection.
  Control or comparison: Prose reports reshape/concat slower, concat-heads 9.692 us slower, and lane-axis RoPE 6.533 us slower.
  Likely subsystem: Candidate evidence retention/provenance.
  Investigation performed: Enumerated all stage artifacts, read every candidate manifest/index row, verified all retained snapshot hashes, and searched the complete fused-decoder evidence root for the cited control values and comparison names.
  Resolution: more-work-needed

- Observed anomaly: The final linear decode total is 4.395 us slower than the selected tile-tail candidate total.
  Evidence: `candidates/tile_aligned_beta_a/manifest.txt` reports 2,723.857 us/71 ops; final raw CSV reaggregation gives 2,728.252 us/71 ops.
  Affected path: Traced linear decode.
  Control or comparison: `diff` between the candidate snapshot and final source shows only additions to fusion-counter bookkeeping; the device graph is unchanged.
  Likely subsystem: Profiler run-to-run variation.
  Investigation performed: Re-summed `DEVICE KERNEL DURATION [ns]` strictly between signposts, recounted timed rows, verified candidate/final source hashes, and diffed the sources.
  Resolution: controlled

- Observed anomaly: Dedicated linear-core RMSNorm passes short PCC but becomes nonfinite at native context, including the FP32-accumulating variant.
  Evidence: `candidates/core_rmsnorm_{default,fp32}/short_correctness.log` pass above the 0.995 bar; their native prefill/decode logs fail `torch.isfinite` after the 262,144-token prefix. The final suite passes both native linear tests with the spelled stable normalization.
  Affected path: Linear core normalization in native-context prefill and decode.
  Control or comparison: Final stable mean-square/add/rsqrt/multiply implementation.
  Likely subsystem: Dedicated RMSNorm BF16 output range at extreme recurrent-state magnitude.
  Investigation performed: Verified both failure logs, source hashes, final implementation, and final native-context JUnit cases.
  Resolution: controlled

- Observed anomaly: The work log claims a final clean review before a supporting artifact exists.
  Evidence: `work_log.md` line 152; no `stage_review.md` in either stated stage commit; current verdict is `more-work-needed`.
  Affected path: Stage closure/provenance.
  Control or comparison: This independent report.
  Likely subsystem: Handoff documentation sequencing.
  Investigation performed: Inspected both commits, their file lists, history, and live worktree scope.
  Resolution: more-work-needed

## Scope Inspected
- Goal/skill paths: Original fused-decoder contract supplied in the review request; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/fused_decoder/{README.md,work_log.md,source_manifest.sha256}`; all files under `correctness/final/`, `watcher/final/`, `perf/`, and `candidates/`; `doc/context_contract.json`; functional-decoder performance summary and context tests as controls.
- Code paths: `tt/fused_decoder.py`; `tests/test_fused_decoder.py`; relevant inherited methods in `tt/functional_decoder.py` and `tests/test_functional_decoder.py`; repository `conv1d` test and Mamba split implementation.
- Commands run: Read-only `git status/log/show/diff`; `find`, `wc`, `stat`, `sed`, `nl`, `rg`, `diff`, and `sha256sum`; small read-only Python CSV analyses to locate signpost windows, sum device-kernel durations, count timed operations, and inventory movement operations. No TT device, server, profiler, or test command was run.

## Residual Risk
- The retained frozen implementation is strongly supported for correctness: source hashes match commit `5d436142493`, JUnit independently confirms 12/12 final tests and 6/6 Watcher tests, all meaningful short-path PCC values exceed 0.995, native 262,144 capacity cases pass, and the Watcher log detaches all four devices without an error signature.
- Final profiler arithmetic is reproducible from primary CSVs: linear prefill 5,605.562 us/159 ops, linear traced decode 2,728.252 us/71 ops, full prefill 2,362.104 us/29 ops, and full traced decode 2,334.152 us/45 ops. Movement counts also match the README (full prefill none; full decode 2 ShardedToInterleaved + 2 InterleavedToSharded; linear prefill 11 tilize + 12 untilize-family; linear decode 4 tilize + 5 untilize-family; no explicit reshard).
- At review time, candidate snapshot hashes and retained RMSNorm/SDPA failure dispositions were consistent, but performance completeness was blocked by an untried convolution adaptation and missing controls. The remediation ledger below records the later AutoFix evidence that supersedes this historical risk statement; clean closure still requires fresh independent rereview.

## AutoFix remediation ledger

- P1 grouped convolution: remediated. Split-4 and split-8 geometry probes, flattened batch-32 adaptation, trace reuse, integrated real-weight/native-context correctness, exact failure modes, and prefill/decode performance are retained under `candidates/split_conv1d/`. The correct standalone-SiLU form is slower in both phases and was rejected.
- P2 missing controls: remediated. Fresh identical-frozen-base A/B/A bundles for repeat versus reshape/concat, prefill concatenate-heads, and lane-axis decode RoPE are retained under `candidates/{repeat_interleave_ab,concat_heads_prefill_ab,decode_rope_lane_axis_ab}/`, indexed by both candidate CSVs, and summarized in `AUTOFIX.md` and `work_log.md`.
- Evidence integrity: the work log continues to report `more-work-needed`; no clean-pass is claimed pending a fresh independent rereview.
