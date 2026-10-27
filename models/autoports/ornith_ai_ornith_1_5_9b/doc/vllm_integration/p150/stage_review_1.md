# Stage review 1 (independent subagent, 2026 Oct 27, 19:10 to 19:30 ET)

Verdict: more-work-needed. No serving-correctness defect found; required work
is evidence hygiene, the one-chip teacher-forcing lower bound, and a
disposition record for the controlled sampling failure. The verbatim verdict
follows; the follow-up actions are in `work_log.md` ("Review findings and
follow-up").

## Required Work

- P2: README, work log, serving contract and environment file state facts that
  the artifacts contradict. (1) README cites "counters in b1_primary/server.log"
  that do not exist (teardown never runs; EngineCore is force-killed). (2) The
  haiku HF-control claim is reversed: the HF control counts 5/7/5. (3) Two
  work-log section times are about two hours off (machine clock is UTC). (4)
  The work log says 21 host tests fail before and after but they were
  repaired (112 pass). (5) Serving contract cites 4097 blocks; logs say 4129
  (B32) and 4098 (B1). (6) registration_probe.json still lists
  supports_batched_prefill. (7) README and work log cite b32_compat/ paths;
  the directory is b32_compat_prefinal/. (8) The "Residual" preemption-resume
  paragraph describes a path the pinned plugin does not take (preemption
  re-prefills from position 0). (9) ORNITH_ACTIVE8_MODE=active8 is requested
  but ineligible on one chip (effective_mode original), undocumented. (10)
  Startup growth is one program on B32 servers but eight on the B1 server.
- P2: the one-chip teacher-forcing lower-bound comparison required by the
  contract is absent (only p150x4 TF rows and served baselines in
  state/p150-gpqa-v1/RECORD.md exist).
- P2: the one sampling-gate failure (test_frequency_penalty_changes_output) is
  controlled by the margin probe but has no recorded disposition path.

## Other Concerns

Startup recapture after the warm-up decode (Metal unsafe-allocation warning
at that moment); default p150 server rejects logprobs with HTTP 400 (device
logprobs are TP4-only); 32-row decode graph costs 168 ms per step for any
number of active rows (stage 10); EngineCore force-killed at every shutdown;
preemption untested; source provenance of the served runs tied only by timing
and formatter evidence (files rewritten at 22:55:49 UTC by hooks).

## Hard-Check Gaps

nonaligned_requests.py chat parity field compared a BatchEncoding length;
no generation_config.json in the checkpoint (default-recipe check not
recorded); launch_env.txt should record source hashes; allocation-safety gate
not rerun on committed sources; gate exits 1 on two advisories only.

## Anomaly Ledger (summary)

Frequency-penalty test: controlled, disposition required. Haiku syllable
miscount: controlled for serving (exact standalone parity), README HF claim
to correct. Startup recapture: controlled, startup only. Force-kill shutdown:
controlled on one chip. Work-log times: documentation fix.

## Scope Inspected

Goal and skill files for stage 9, tt-enable-tracing, qualitative-check and
stage-review; all p150 artifacts under doc/vllm_integration/p150 and
readiness_vllm; the serving and context contracts; plugin docs and scheduler
source; the stage diff c0976f241f..5025e96ab8; the host-only gate (exit 1,
advisories) and the eight device-free host test files (112 passed).

## Residual Risk

Preemption and re-prefill with penalty-history restore untested on this
profile; served-source provenance; default server cannot serve logprobs;
single-user latency on the 32-row server; force-kill shutdown on multi-chip
profiles.
