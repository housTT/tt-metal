# Stage review 2 (13:05 to 13:22 ET, fresh read-only subagent, same prompt plus the review-1 record)

Verdict: clean-pass

Required work: none. The three review-1 items are closed by committed evidence (c8 and c10 robustness rows flip both to fail; c9 hit-window margin probes; token budget computed from the loaded model and measured in token_budget_check.json). The selected policy c9_lm_head_bfp4 is the fastest candidate that passes every reference it ran on, every faster candidate has a recorded failing reference, the policy is consumed by the default construction path (36-layer readback, tt-perf-report rows), and the post-selection token-out benchmark ran through that path.

Other concerns (notes): the acceptance-rule sentence overstates coverage (c12, c13, c16, c17 were not run on every robustness reference; each had already failed one) and README section 7's sentence omits c13, c16, c17 among faster main-reference passers; README 3.1/3.3 infer the BFP4-quantization cause from c10 = c5 although the natural-text reference cannot vary fidelity (the inference is supported by c1 against c0); the top-1 frontier passes through c13's concurrent number; max_tokens_all_users defaults kv_cache_dtype to BF16 (the stage-9 hook must pass the generator's dtype); two natural-text JSON files carry the corrected next-token fields while nine carry a note; the sweep_results key round2_concurrent_row is misnamed for chain-1 rows.

Hard-check gaps (notes): no TT_METAL_TRACE_ALLOC_TRACKING=1 run on the selected policy in stage 8 (stage 7's tracked run covers the unchanged capture code; stage 9 should include the selected policy); sweep_results.csv lacks command, hardware, mesh and timing_regime columns (present in the JSON); check_propagation does not assert embedding, final-norm, activation and logits dtypes (reported by describe_runtime and matching by inspection); review placeholders.

Anomaly ledger: review-1 P1 fixed; resume hit-window flips controlled (probed near ties, lowered-threshold run passes); c10 = c5 controlled and documented; trace-allocation warning controlled (stage-7 tracked run, program cache flat); c13 TTFT controlled (concurrent row).

Residual risk: natural-text margin 13 positions above the bar (binomial standard deviation about 11); fresh-versus-continuation agreement moved by one or two disagreements per window under the BFP4 LM head; a BF16 gate or up projection would map to the BFP8-probed prefill table row (no candidate uses BF16 projections); the committed precision.py is the black-reformatted file (parse check passes); the book32 reference is uncommitted by design; logs live only in this worktree; the decode LM-head kernel has 100 tokens of natural-text decode evidence.

The reviewer's full verdict text is in stage_review_2_full.md.
