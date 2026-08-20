# Stage-review remediation ledger

Source review: `stage_review_final.md` (`more-work-needed`). This ledger records the remediation evidence without changing the independent reviewer's verdict.

Final independent rereview: `stage_review_clean.md` (`clean-pass`, no required work). The sections below record the evidence that closed the two findings.

## P1: split-conv like-for-like provenance

Status: addressed and independently accepted in `stage_review_clean.md`.

- Exact frozen implementation SHA256: `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14`.
- Exact standalone-SiLU split-4 candidate implementation SHA256: `f69011093d5d28c1c3a8ca102aebf156cda1024a4a044eaef3f1661ae236a38d`.
- Exact common A1/B/A2 test-fixture SHA256: `ea3c400790428be1f5c888103aa621914607274b741a10c381665fac356d9645`; both relevant nodes specify `trace_region_size=0` and `l1_small_size=16384`.
- The old `dbe94bd3...` value is correctly labeled as a patch SHA, not a candidate implementation SHA.
- A1/B/A2 held the profiler checkout, isolated P300 board, fixed cache, replay count, pytest node, and phase signposts constant.
- Candidate correctness: PCC `0.997846637`, `0.999412803`, `0.999644046`; trace determinism PCC `1.0`; prior native prefill/decode passes remain applicable to this exact implementation.
- Prefill A1/B/A2: `5505.526 / 5635.236 / 5500.267 us`, `159/159/159` ops; B is `132.3395 us` or `2.405%` slower than the `5502.8965-us` median base.
- Traced decode A1/B/A2: `2625.831 / 2873.108 / 2625.095 us`, `71/94/71` ops; B is `247.645 us` or `9.432%` slower and adds 23 ops.
- Disposition: rejected because the correct candidate loses both required phases.
- Authoritative evidence: `candidates/split_conv1d/l1_16k_ab/manifest.txt`, `summary.csv`, source snapshots/patch/hashes, correctness log, six compact raw CSVs, six `tt-perf-report` outputs, and `evidence.sha256`.
- Historical `candidates/split_conv1d/perf/` rows remain labeled exploratory and do not support the final disposition.

After A2, implementation and tests were restored byte-for-byte to SHA256 `f0778a63...` and `05c38b76...`.

## P2: hardware recovery provenance

Status: addressed to the limit of retained historical evidence and independently accepted in `stage_review_clean.md`; the fresh required safety sequence completed before P1.

- Exact historical recovery transcript and original candidate PID are unavailable; this is stated explicitly in README, work log, AutoFix, split manifest, and recovery ledger. No PID or command was invented.
- Fresh process/lock inspection found no remediation hardware process and no owner on the selected board locks. No process was killed and no lock cleared.
- Unrelated Laguna vLLM PID 512455 was preserved and its separate board excluded.
- Bounded pre-list exited 0 with all four chips visible.
- The topology-complete free P300 board pair (`0000:01:00.0`, `0000:02:00.0`) was reset with a 180-second bound; exit 0.
- Bounded post-list exited 0 with all four chips visible. A second reset was not needed.
- Installed-runtime 1x1 mesh smoke with `trace_region_size=0` and `l1_small_size=16384` opened/closed on the isolated pair; exit 0 and `MESH_SMOKE_OK 1`.
- Postflight list exited 0; no pytest/Tracy-capture/EngineCore/device-profiler hardware process remained; selected locks had no owner.
- Authoritative evidence: `candidates/split_conv1d/recovery/`.

The work log's original “no reset” statement is now explicitly scoped to original frozen-final collection; the remediation recovery is documented separately and noncontradictorily.
