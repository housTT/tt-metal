# Stage review 2 (fresh subagent, read-only, 13:53 to 14:15 ET), HEAD 1cccb01d89

Verdict: more-work-needed

Review-1 items: seeds fixed in code (no device evidence; new prefill-edge defect, P2-1); contract corrected, hang open (P1-1); row counts fixed by construction, continuation open, lengths 4097 to 36864 unwarmed (P2-2); batch-32 divergence open (P2-3); split-sampling artifact renamed but still cited (P2-4); cache ownership fixed, page-table pad rows use the owner's scratch (other concern).

Required work:
- P1-1: hang unlocalized; the probe stepped row 0 past 40959 from its second step and the bisect turned the inactive row into 0; `decode_forward` has no context-bound check; the work log overclaims "the first" decode step; no `$autodebug` attempt. Host-side: guard, corrected probes, static `$autodebug`, wording; after the reset: bisect order (a) three short prompts in bucket 8, (b) one row at 40959 in bucket 1, (c) both; then the corrected probe.
- P1-2: no device evidence for any change since `2eaaa0b3f9`; the tracked gate never replayed the bucket-8 and bucket-32 sampling traces. Host-side: extend the tracked workload (3-row and 32-row prefill after capture, program-cache assertion); after the reset: rerun the module, the tracked gate, both readiness runners and the perf test; refresh the README numbers.
- P2-1: prefill admission deactivated other slots; `slot_remap` ignored; no counter alignment; `salt_duplicate_seeds=True` versus the vLLM contract.
- P2-2: prompt lengths 4097 to 36864 compile after capture (embedding, untilize, chunk slices keyed by the rounded length).
- P2-3: `test_perf_batch1` asserted a cumulative counter that the earlier seeded test makes nonzero; batch-32 bar raised without a same-bucket control.
- P2-4: serving contract cited the renamed split-sampling file in four rows and an obsolete seeded description; work log section 8 empty; top-level README "trace-verified" number; "final code" rows ran commit `2eaaa0b3f9`; trace-gate paragraph overclaimed sampler replays; context contract listed a missing artifact.

Other concerns: `_write_page_table` pad rows use the generator's scratch id; the inactive-row path in the traced full model has no passing run; `supports_prefix_caching` with eager per-offset continuation; `max_tokens_all_users` positional call shape (stage-9 wrapper); no tt-triage on chip 0 and no Inspector data on chip 1; resume thresholds, qualitative citation and LoFi LM head unchanged.

Hard-check gaps: tracked workload coverage (bursts, 32 rows, long prompt, continuation); no run with fewer live users than the bucket; gdb without Python frames; `check_serving_contract.py` does not verify cited paths; new code paths unrun.

Residual risk: nothing after `2eaaa0b3f9` has run on a device; the context-end defect, if real, is undemonstrated either way; stage 9 will exercise continuation and concurrent seeded requests.
