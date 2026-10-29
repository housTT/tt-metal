# Stage review 3 (fresh subagent, read-only, 14:35 to 14:50 ET), HEAD f210559cfc

Verdict: more-work-needed

Review-2 disposition: P1-1 guard fixed, `$autodebug` recorded, wording corrected, but `capacity_probe.py` was not actually corrected and `capacity_bisect.py` called `_apply_sampling_params` without `admit` (P1-A, host-side); P1-2 every device rerun open and honestly recorded as blocked (P1-B, operator-blocked); P2-1 seed admission fixed in code, untested; P2-2 fixed by construction (host-padded ids, 17-length warmup); P2-3 fixed (test order, same-bucket arm); P2-4 fixed; `_write_page_table` fixed.

Required work: P1-A (host, fixed in commit after this review: probe row 0 at 40955 with per-step prints, bisect `admit=True`, documents and AUTODEBUG erratum); P1-B operator reset then the documented rerun list and a rereview.

Other concerns: prefill seed registration by row order self-heals on the plugin's first decode (recorded); standalone and served seeded streams differ (recorded in the README); `prepare()` time with the longer warmup unmeasured; "outside [0, 40960)" wording (fixed); AUTODEBUG section 3 contrast wrong (erratum appended); carried items unchanged.

Hard-check gaps: tracked long-prompt case uses a warmup length (8192) rather than an in-between length; no bucket-1 arm script for positions 40955 to 40958; `check_serving_contract.py` does not verify paths; gdb without Python frames.

Residual risk: every change after 2eaaa0b3f9 is unexecuted; the step-1 hang configuration is excluded statically only; stage 9 will exercise empty_slots, slot_remap, concurrent seeds and continuation under live traces.
