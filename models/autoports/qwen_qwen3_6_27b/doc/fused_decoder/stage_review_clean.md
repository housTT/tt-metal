# Stage Review

Verdict: clean-pass

## Required Work

None.

## Other Concerns

- The remediation evidence and documentation are still live stage-owned worktree changes. This does not invalidate the review: implementation and test sources are byte-for-byte restored to the frozen hashes, unrelated dirty third-party paths are distinguishable, and the stage-review workflow calls for the isolated checkpoint commit after a clean review. The stage owner should include this review and the stage-owned remediation files in that checkpoint without including `tt_metal/third_party/{tracy,umd,tt-cluster-descriptors}`.
- Final selected-path profiler rows are single retained signpost windows. The A/B/A remediation provides repeat controls for the close repeat-interleave, concat-heads, decode-RoPE, and split-conv decisions, but the original final and older candidate bundles do not have a general variance policy. This is residual measurement risk, not missing required evidence: all close decisions now have frozen-base controls, the split-conv losses are material in both phases, and the final report uses reproduced final totals rather than a faster earlier selected-candidate scalar.
- Final correctness and Watcher logs end with nanobind reference-leak diagnostics, and the HF linear-attention control reports its expected Torch fallback. The TTNN devices close normally, JUnit has no failures/errors, Watcher detaches all four devices, and code inspection finds no host fallback in the fused runtime graph; these messages do not affect the measured TTNN path.

## Hard-Check Gaps

- `test_fused_runtime_has_no_host_or_layout_fallbacks` statically enumerates fused overrides rather than every inherited measured helper. Direct inspection of inherited `prefill_forward`, `decode_forward`, `_finish_layer`, `_finish_layer_chunked`, `_linear_causal_conv_chunk`, `_pad_linear_chunk`, and `_linear_chunk_inverse` found TTNN-only runtime work and no `from_torch`, `to_torch`, material reshard, tilize/untilize call, or fallback. Expanding the static enumeration would improve the hard check but does not hide a concrete path defect.
- The complete remediation artifact tree has no single top-level checksum manifest. The authoritative split-conv A/B/A bundle does have `evidence.sha256`, which validates every retained source, correctness, raw CSV, report, and summary file; the other A/B/A source hashes and primary CSV arithmetic were independently reconstructed during this review. Git/checkpoint provenance remains the final integrity mechanism for the broader tree.

## Anomaly Ledger

- Observed anomaly: The original split-conv performance rows used an unretained 16-KiB profiler fixture and zero-L1_SMALL controls.
  Evidence: `stage_review_final.md`; `candidates/split_conv1d/manifest.txt`; historical `candidates/split_conv1d/perf/`; authoritative `candidates/split_conv1d/l1_16k_ab/`.
  Affected path: Linear-attention prefill and traced decode candidate disposition.
  Control or comparison: Frozen-base A1/B/A2 with one common 16-KiB fixture.
  Likely subsystem: Candidate experiment provenance and device L1_SMALL reservation.
  Investigation performed: Verified full base/candidate source hashes `f0778a63...`/`f6901109...`, common test hash `ea3c4007...`, exact two-node fixture patch, all bundle checksums, correctness PCC, all six signpost windows, op counts, medians, deltas, fixed device/runtime/cache/replay/node/signposts, and restored final hashes `f0778a63...`/`05c38b76...`.
  Resolution: controlled. The candidate is correct but slower: prefill A1/B/A2 is 5,505.526/5,635.236/5,500.267 us (159/159/159 ops), or +132.3395 us/+2.405% versus median base; decode is 2,625.831/2,873.108/2,625.095 us (71/94/71 ops), or +247.645 us/+9.432% and 23 ops. Historical rows are explicitly superseded.

- Observed anomaly: `dbe94bd3...` was previously presented as though it identified the passing split-conv implementation.
  Evidence: SHA256 of `candidates/split_conv1d/source_diff_standalone_silu.patch`; current `candidates/index.csv`, `perf/candidates.csv`, split manifests, README, work log, and remediation ledger.
  Affected path: Candidate source provenance.
  Control or comparison: Reconstructed candidate implementation snapshot SHA256 `f69011093d5d28c1c3a8ca102aebf156cda1024a4a044eaef3f1661ae236a38d`.
  Likely subsystem: Documentation/index identity labeling.
  Investigation performed: Rehashed the old patch and both full implementation snapshots; checked every remaining `dbe94` reference and both indexes.
  Resolution: fixed. `dbe94...` is now only the patch SHA, while all disposition indexes use `f6901109...` as the candidate implementation identity.

- Observed anomaly: The exact historical terminal transcript for the original split-conv device-hang recovery is unavailable.
  Evidence: `candidates/split_conv1d/triage/tt-triage.txt`; `candidates/split_conv1d/recovery/README.md`; README, work log, AutoFix, split manifest, and remediation ledger.
  Affected path: Hardware-recovery provenance before the original exploratory run.
  Control or comparison: Fresh bounded ownership/list/reset/list/mesh-smoke ledger captured before the authoritative A/B/A.
  Likely subsystem: Evidence retention, not decoder correctness.
  Investigation performed: Confirmed that no historical PID or command is invented; inspected fresh process and lock ownership, preservation of unrelated Laguna PID 512455, topology discovery, targeted free P300 pair reset, bounded lists and exit statuses, second-reset decision, installed-runtime mesh open/close, final process/list/locks, and use of the same isolated visibility pair in the authoritative run manifest.
  Resolution: controlled. The historical limitation is explicit. The fresh pre-P1 sequence killed no process, cleared no lock, excluded the unrelated board, reset topology-complete BDFs `0000:01:00.0` and `0000:02:00.0` with exit 0, retained all-four-chip visibility, required no second reset, and completed `MESH_SMOKE_OK 1` with exit 0.

- Observed anomaly: The recovery attempt first filtered and reset one BDF of a dual-chip P300, which could not form the expected topology.
  Evidence: `recovery/reset_target_0.txt`, `list_after_target_0.txt`, `single_bdf_smoke_failure.txt`, `board_pair_preflight.txt`, `reset_board_pair.txt`, and `mesh_smoke_board_pair_full.log`.
  Affected path: Recovery preflight only; no correctness or profile sample used this visibility.
  Control or comparison: Topology-complete board-pair reset and successful isolated-board smoke.
  Likely subsystem: P300 visibility/topology selection.
  Investigation performed: Checked the exact failure, board ID, follow-up owner check, pair reset/list, and authoritative A/B/A visibility.
  Resolution: controlled. The single-BDF route was rejected before experiments; all authoritative correctness and profiler runs used the verified complete pair.

- Observed anomaly: Three material close candidate claims originally lacked frozen-base raw controls.
  Evidence: Prior `stage_review.md`; `candidates/{repeat_interleave_ab,concat_heads_prefill_ab,decode_rope_lane_axis_ab}/`.
  Affected path: Linear Q/K repetition, full-prefill head concatenation, and full-decode partial RoPE.
  Control or comparison: One-change frozen-base A/B/A bundles.
  Likely subsystem: Candidate evidence completeness.
  Investigation performed: Reconstructed candidate implementation hashes from each retained patch (`bed81f43...`, `c6022275...`, `4fcfb4d2...`), verified correctness logs, and independently summed every raw signpost window. The deltas reproduce as +256.3055 us prefill and +15.6045 us decode for reshape/concat repetition, +8.4055 us for dedicated prefill concat-heads, and +8.1655 us for lane-axis decode RoPE (rounded in docs to 256.306/15.605/8.406/8.166 us).
  Resolution: fixed. All three correct candidates are slower than their median frozen-base controls and are indexed with primary evidence.

- Observed anomaly: Dedicated linear-core RMSNorm passes short PCC but becomes nonfinite at native context, including the FP32-accumulating variant.
  Evidence: `candidates/core_rmsnorm_{default,fp32}/` short and native failure logs; final native-context JUnit cases.
  Affected path: Linear-attention core normalization after recurrent-state accumulation.
  Control or comparison: Retained stable mean-square/add/rsqrt/multiply normalization.
  Likely subsystem: Dedicated RMSNorm output range at extreme recurrent-state magnitude.
  Investigation performed: Checked both candidate dispositions and final native 262,144-token prefill plus native-prefix traced decode passes.
  Resolution: controlled. The numerically stable spelled normalization is intentionally retained while adjacent activation fusion remains.

- Observed anomaly: The selected tile-tail candidate's earlier decode scalar (2,723.857 us) is 4.395 us faster than the frozen final reproduction (2,728.252 us).
  Evidence: `candidates/tile_aligned_beta_a/manifest.txt`; final linear-decode raw CSV; prior independent source diff.
  Affected path: Final traced linear decode reporting.
  Control or comparison: Device graph is unchanged apart from fusion-counter bookkeeping, and the stage reports the slower final reproduction.
  Likely subsystem: Run-to-run profiler variation.
  Investigation performed: Recomputed the final window and reviewed the prior exact source comparison.
  Resolution: controlled. No faster distinct implementation is being displaced and the advertised result is the conservative reproduced number.

## Scope Inspected

- Goal/skill paths: Original Qwen/Qwen3.6-27B fused-decoder contract supplied in the review request; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/fused_decoder/{README.md,work_log.md,AUTODEBUG.md,AUTOFIX.md,review_remediation.md,stage_review.md,stage_review_final.md,source_manifest.sha256}`; all retained files under `correctness/final/`, `watcher/final/`, `perf/`, and `candidates/`, with detailed recomputation of all final and A/B/A raw CSVs; `doc/context_contract.json`; functional-decoder performance summary as the before control.
- Code paths: `tt/fused_decoder.py`; `tests/test_fused_decoder.py`; inherited measured helpers and public entry points in `tt/functional_decoder.py`; relevant functional tests; candidate source snapshots and patches.
- Commands run: Read-only `git status/log/show/diff`; `find`, `wc`, `stat`, `sed`, `rg`, `cmp`, `diff`, `sha256sum`, `gzip -t`, and `zgrep`; small read-only Python CSV/hash analyses to validate index paths, reconstruct candidate source hashes, locate signpost windows, sum device-kernel durations, count operations, and inventory movement. No hardware, server, profiler, or test command was run.

## Residual Risk

- Final source/test hashes match the frozen manifest and live files: `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14` and `05c38b764320bb682248df78d00ff950a1602b8aeee9b2e40cb25fd690a41dad`. JUnit independently records 12/12 final tests and 6/6 separate Watcher tests with zero failures/errors. Coverage includes both layer kinds, real weights, non-aligned lengths, paged cache, repeated trace determinism, forced chunking, batch 32, 262,144-token prefill, and traced decode at position 262,143. All printed material PCC values exceed the 0.995 stage bar.
- Final before/after arithmetic reproduces exactly: linear prefill 5,605.562 us/159 ops versus 6,046/177; linear traced decode 2,728.252 us/71 ops versus 2,901/88; full prefill 2,362.104 us/29 ops versus 2,594/51; full traced decode 2,334.152 us/45 ops versus 2,639/50. Final movement inventory also reproduces: full prefill has none; full decode has two sharded-to-interleaved and two interleaved-to-sharded ops; linear prefill has 11 tilize and 12 untilize-family ops; linear decode has four tilize and five untilize-family ops; no explicit reshard occurs.
- The graph-fusing matrix is exhaustive for the observed graph and every material plausible family is either retained, rejected with correctness/contract evidence, or rejected with measured primary evidence. In particular, the previously missing channel-split depthwise convolution was adapted through split-4/split-8 and logical batch-32 topology, proved correct through native context, and rejected only after like-for-like measurements showed both required phases slower.
- No required work remains. The remaining operational step is the normal post-clean-review isolated stage checkpoint commit, not remediation of the fused decoder stage.
