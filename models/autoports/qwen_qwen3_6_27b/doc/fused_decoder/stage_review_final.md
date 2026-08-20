# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The split-conv performance rejection is not reproducible from the retained source/command and is not yet a like-for-like comparison.
  Evidence: `candidates/split_conv1d/manifest.txt` lines 25-30 says the integrated candidate cannot run with the default zero-byte L1_SMALL reservation and that the correctness-valid standalone-SiLU form requires 16 KiB. The retained passing patch changes only `test_fused_linear_attention_non_aligned_prefill_decode` to `l1_small_size=16384` (`source_diff_standalone_silu.patch` lines 5-15); it does not change the profiler node. The retained profiler command invokes `test_fused_decoder_perf` (`manifest.txt` line 16), whose retained fixture still specifies only `trace_region_size=0` (`tests/test_fused_decoder.py` lines 178-184). Both this checkout and the exact profiler runtime named by the command define `DEFAULT_L1_SMALL_SIZE = 0` (`tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` line 15 and `/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp` line 15). Nevertheless, the raw profiler CSVs contain the four split-2,560 `Conv2dDeviceOperation` rows and recompute to 5,737.134 us/159 ops and 2,983.133 us/94 ops. Therefore an unretained profiler-fixture mutation or launch setting was necessary. The reported controls, 5,605.562 us/159 ops and 2,728.252 us/71 ops, are the frozen final runs under the zero-byte fixture, not a frozen-base run under the candidate's required 16 KiB reservation. In addition, `candidates/index.csv` labels `dbe94bd...` as `candidate_source_sha256`, but that value is the SHA256 of `source_diff_standalone_silu.patch`; reconstructing the passing candidate gives implementation SHA256 `f69011093d5d28c1c3a8ca102aebf156cda1024a4a044eaef3f1661ae236a38d` and test SHA256 `d74c992ffc90ae7aff41eea74514e7d48935334bd5d627b800ba30135db81d4e`.
  Why this matters: The split-4 family is now convincingly adapted for split-4/split-8, logical batch-32 flattening, fused versus standalone SiLU, real-weight correctness, native context, and both measured phases. But the original contract also requires warmed before/after evidence and proof that the final path beats every distinct correct alternative like for like. A candidate requiring a different device reservation cannot be rejected against a base measured with the default reservation when the exact candidate launch is not reproducible from the retained artifacts.
  Required next step: Retain the exact profiler test/config source and its actual source hashes, then run a frozen-base A/B/A (or an equivalently controlled repeated comparison) with the same 16 KiB L1_SMALL reservation, profiler runtime, cache policy, replay count, signposts, and layer/phase inputs for A and B. Recompute both phases from retained primary CSVs and correct `candidates/index.csv`, `perf/candidates.csv`, README, work log, manifest, and source-hash records to use the actual measured source identities and like-for-like control.

- P2: The device-hang recovery record does not satisfy the required TT-device ledger and contradicts the work log.
  Evidence: `candidates/split_conv1d/triage/tt-triage.txt` validly captures device 3 at zero ARC heartbeat and becoming unreadable, so the hang itself is supported. But the evidence root contains no retained output or status for the required bounded `tt-smi` list/reset/list sequence or the claimed mesh smoke, and it does not identify the candidate process terminated, whether a second reset was needed, or whether locks were inspected/cleared. The only recovery statements are prose in `manifest.txt` lines 26 and 51-55 and `work_log.md` line 162. `work_log.md` lines 12-17 simultaneously says no reset or recovery was needed, without limiting that statement to pre-remediation final collection. Later passing native and profiler jobs show the hardware eventually became usable, but they do not establish the exact safe recovery sequence required by the TT-device-usage skill.
  Why this matters: The review contract explicitly asks that hang recovery evidence be validated, and the TT-device-usage skill requires the failure command/signature, processes killed, exact list/reset/list commands and exit status, second-reset status, lock handling, and mesh-smoke result to be recorded. The current prose cannot distinguish a disciplined recovery from an incomplete or differently configured recovery, and the work log is internally inconsistent.
  Required next step: Recover the exact command/status ledger from retained terminal history or other existing run records, including process termination, bounded list/reset/list, second-reset decision, lock decision, and mesh-smoke command/result; retain it under `candidates/split_conv1d/` and reconcile the hardware-discipline section of `work_log.md`. If exact historical details are unavailable, say so explicitly rather than claiming exact recovery evidence, and perform/retain the required bounded health sequence before any remediation rerun used to close P1.

## Other Concerns

- The three missing-control remediations are sound. I reconstructed candidate implementation hashes from each one-change patch and obtained the claimed hashes: repeat/reshape-concat `bed81f43...`, concat-heads `c6022275...`, and lane-axis RoPE `4fcfb4d2...`. Their A1/A2 base hashes are the frozen `f0778a63...`, correctness logs exceed the 0.995 bar, and all raw CSV totals/op counts reproduce exactly.
- The A/B/A arithmetic recomputes as follows: reshape/concat is +256.305 us prefill and +15.605 us decode versus the median bases; dedicated prefill concat-heads is +8.406 us; lane-axis decode RoPE is +8.166 us. Rounding accounts for the 0.001 us differences in prose. These controls are phase- and signpost-matched and do not create required work.
- The split candidate's compact CSVs are intact and clearly show standalone SiLU (`activation: std::nullopt`) with split-2,560 depthwise convolution in both phases. Thus P1 concerns experiment provenance and comparison regime, not fabrication of the reported raw totals.
- The remediation paths are currently uncommitted stage-owned changes, while unrelated dirty third-party submodules/directories remain out of scope. This is acceptable before a clean review, but the stage still needs an isolated checkpoint commit after all required work and a later clean-pass.

## Hard-Check Gaps

- No retained latency distribution exists for the final graph or original candidate set. The new A/B/A controls materially improve confidence for the three sub-percent decisions, but the original single-window final/candidate rows still have no variance policy.
- The no-host/layout-fallback static test inspects fused overrides but not inherited measured methods. Direct inspection found TTNN-only runtime code in the relevant inherited path, so this is not a concrete fallback finding.
- `source_manifest.sha256` covers only the final implementation and test, not the evidence tree. CSV/XML/gzip/PNG integrity checks passed, but an evidence manifest would reduce stale-artifact risk.

## Anomaly Ledger

- Observed anomaly: The split-conv profiler command names a zero-L1_SMALL test fixture although the candidate is documented as requiring 16 KiB.
  Evidence: `candidates/split_conv1d/manifest.txt` lines 16 and 25-30; `source_diff_standalone_silu.patch` lines 5-15; `tests/test_fused_decoder.py` lines 178-184; both runtime `common_values.hpp` files line 15; raw split prefill/decode CSVs.
  Affected path: Correctness-valid standalone-SiLU split-conv warmed linear prefill and traced decode rejection.
  Control or comparison: Frozen final 5,605.562 us/159 ops and 2,728.252 us/71 ops, collected with the default zero-byte reservation.
  Likely subsystem: Profiler test-fixture provenance and device L1_SMALL configuration.
  Investigation performed: Reconstructed candidate source/test hashes in memory, verified the retained patch applies, inspected both runtime defaults and pytest fixtures, reaggregated both raw candidate CSVs, and confirmed split-conv op attributes in the primary rows.
  Resolution: more-work-needed

- Observed anomaly: The DRAM-config split candidate hard-stalled device 3.
  Evidence: `short_linear_correctness_dram_config.log`; `triage/tt-triage.txt` reports zero ARC heartbeat and a device-register timeout.
  Affected path: Split-conv integration with `config_tensors_in_dram=True`.
  Control or comparison: The 16 KiB L1_SMALL route completes; later native and profiler runs completed on the recovered hardware.
  Likely subsystem: Conv1d DRAM configuration metadata/device runtime.
  Investigation performed: Inspected the failing log and full triage report, then searched the complete stage evidence for reset/list/mesh-smoke records and recovery commands.
  Resolution: more-work-needed because the hang is classified but the mandatory recovery ledger is incomplete.

- Observed anomaly: Fused conv1d SiLU passes an isolated probe but fails integrated real-decoder PCC, including with FP32 accumulation.
  Evidence: `probe_split4_l1small.log` reports isolated PCC 0.999771533/0.999769934; `short_linear_correctness_l1small.log` reports 0.957531573; `short_linear_correctness_l1small_fp32acc.log` reports 0.956417713; standalone-SiLU reports 0.997846637 and later checks above the bar.
  Affected path: Split-4 integrated linear prefill/decode.
  Control or comparison: Standalone `ttnn.silu` after the four conv outputs.
  Likely subsystem: Conv1d fused activation numerical behavior.
  Investigation performed: Compared fused activation, exact/FP32 accumulation, and standalone activation; inspected short and native compressed logs.
  Resolution: controlled; the fused-activation form is correctly rejected and the standalone form is the candidate whose performance still needs a valid comparison.

- Observed anomaly: The work log says both that no recovery was needed and that all boards were reset after the remediation hang.
  Evidence: `work_log.md` lines 12-17 and line 162.
  Affected path: Stage provenance and hardware safety record.
  Control or comparison: Subsequent passing native-context and profiler artifacts prove eventual health but not the exact recovery sequence.
  Likely subsystem: Documentation sequencing/scope.
  Investigation performed: Searched the entire fused-decoder evidence root for recovery commands, outputs, statuses, and mesh-smoke text.
  Resolution: more-work-needed

## Scope Inspected

- Goal/skill paths: Supplied fused-decoder stage contract; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/fused_decoder/{README.md,work_log.md,AUTODEBUG.md,AUTOFIX.md,stage_review.md,source_manifest.sha256}`; all files under `candidates/split_conv1d/`; all files under `candidates/{repeat_interleave_ab,concat_heads_prefill_ab,decode_rope_lane_axis_ab}/`; both candidate/performance indexes; final correctness, Watcher, and profiler artifacts; `doc/context_contract.json`.
- Code paths: `tt/fused_decoder.py`; `tests/test_fused_decoder.py`; relevant inherited runtime/tests in `tt/functional_decoder.py` and `tests/test_functional_decoder.py`; repository/profiler runtime L1_SMALL defaults; pytest mesh-device fixture handling.
- Commands run: Read-only `git status/log/show/diff/diff --check/apply --check`; `find`, `stat`, `file`, `sed`, `nl`, `rg`, `sha256sum`, `wc`, `head`, `tail`, and `zcat`; small read-only Python analyses for CSV aggregation/integrity, XML/gzip parsing, source-patch reconstruction/hashing, evidence-index existence, medians, deltas, and op counts. No TT device, server, profiler, or test command was run.

## Residual Risk

- The frozen final implementation remains strongly supported: implementation/test hashes exactly match `f0778a63...` and `05c38b76...`; final JUnit shows 12/12 passing; separate Watcher JUnit shows 6/6 passing with clean detach; both layer kinds, non-aligned lengths, paged cache, repeat determinism, batch 32, and 262,144-token coverage are present.
- Final primary CSV arithmetic independently reproduces 5,605.562 us/159 ops linear prefill, 2,728.252 us/71 ops linear traced decode, 2,362.104 us/29 ops full prefill, and 2,334.152 us/45 ops full traced decode. The three newly retained A/B/A candidate families are correct, identically based, and slower.
- Split-conv exploration is now technically broad enough: split-4/split-8, physical and flattened batch-32, NHWC/NLC, prepared-weight reuse, trace determinism, DRAM metadata, 16 KiB L1_SMALL, fused and standalone SiLU, FP32 accumulation, real weights, native prefill/decode, and both profiler phases were attempted. The remaining blocker is narrower but material: the passing candidate's measured launch/config is not exactly retained and its control uses a different reservation.
- No additional untried dedicated op, structural merge, or adjacent fold was apparent after inspecting the final graph and candidate matrix. Clean closure depends on earning the split-conv performance rejection and repairing the recovery/provenance record, followed by another independent review with no Required Work.
