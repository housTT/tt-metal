# Stage review 1 (fresh subagent, read-only, 13:26 to 13:46 ET)

Verdict: more-work-needed

Required work (verbatim titles, evidence in the reviewer's report; the report body was returned as the subagent's final message and is summarized here):

- P1: Request-seed handling is wrong on both edges, and it silently moved the sampler to the eager path for most of the stage's trace, host-work and perf evidence. (`_apply_sampling_params` called `reset_seed` on every decode step and never cleared a seeded slot; after `test_split_sampling_trace` seeded slot 0, every later sampling call in the same pytest process ran the sampler eagerly with a per-token seed write; `sampling_trace_replays` counted calls.)
- P1: The advertised-context path hung on two chips, the hang is unlocalized, the work log's localization is contradicted by its own log (the allocator's "active trace" warning shows the three-row prefill returned and the first bucket-8 decode step was reached), and the context contract asserts tested values that no full-model artifact supports.
- P1: Supported prefill paths compile programs after the first capture (row counts outside 1, 8, 16, 24, 32; continuation rows with nonzero start), and the tracked gate excludes them.
- P2: Batch-32 decode divergence (user 18 at token 6) is uncontrolled.
- P2: The canonical split-sampling artifact was overwritten by the force-argmax run and contradicts the README.
- P2: Cache ownership for external callers is unsafe by default (scratch block defaults to the generator's block 1280) and the DRAM budget ignores the generator-owned cache.

Other concerns: canonical resume check thresholds lowered more than needed (`--pcc-min-min 0.98` would pass); per-position TT-versus-HF PCC down to 0.81 is a precision characteristic for the datatype sweep; qualitative verdict cites margin probes of a different prompt set; `prefill_forward` with `sampling_params` left `device_state_valid`; `max_tokens_all_users` signature versus the plugin hook; duplicate note in the context contract; artifact hashes predate formatting-only commits.

Hard-check gaps: no traced-versus-eager sampler counter; tracked workload lacks a nonzero-start continuation and a 2-to-7-row burst and replays sampling traces only for bucket 1; `check_context_contract.py` does not validate `tested_*` fields; capacity probe lacked a progress print; final-code perf measured only in the force-argmax configuration; LoFi LM head not measured.

Anomaly ledger: capacity probe (more-work-needed), resume top-1 flips (controlled), PCC 0.917 outlier (controlled), batch-32 user 18 (more-work-needed), eager sampler after the first seeded request (more-work-needed), greedy divergence from HF (controlled), chip-1 profile stall (controlled as infrastructure), program-cache growth after capture (fixed for warmed row counts, more-work-needed for others).
