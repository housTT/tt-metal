# Stage review 1 (independent subagent, 2026 Oct 27, 23:05 ET)

Verdict: more-work-needed. Required work (all P2): allocation-tracker run on
the final slot-remap code; stale stage 9 serial-reference copy at the
`readiness_vllm/` root; watcher over the new op configurations. Other
concerns: dummy rows depended on block 0 being the null block; served policy
(`ORNITH_BATCHED_PREFILL_MAX_LEN=512`) was not the code default; serial and
batched servers ran on different chips (about 10 ms of one-row TTFT
variance between chips 2 and 3); `b1_final_v3` ran `9d4de01a6d`, not
`5cff1036cc`; no per-step host-sampling counter on the compatibility server
(inherited from stage 9); evidence preservation (`*.log` ignored by git, the
host sweep log in `/tmp`); the `model.py` sha of `tracked_b32` matches no
commit (pre-format working tree); the host-dispatch-bound batched pass rests
on asserted op limits. Hard-check gaps: `stage_check.log` is scanned by the
context check on a rerun; teardown counters absent; the probe's continuation
uses host argmax in both arms. Anomaly ledger: the four-chip host sweep
(controlled, recorded), the 19 recaptures of attempt 3 (fixed), two aborted
mixed-shape client runs (fixed), the controlled frequency-penalty test
failure (unchanged from stage 9), the single startup allocator warning, and
the +2.9 percent first-request TTFT (within gate, repeat -1.3 percent).
The full review text is in the stage agent's transcript; the work log
section "Independent review 1 and follow-up" lists the actions taken.
