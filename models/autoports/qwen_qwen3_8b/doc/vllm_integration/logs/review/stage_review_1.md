# Stage Review 1 (fresh read-only subagent, HEAD 85020b4eed, 14:40 to 15:02 ET Oct 30)

Verdict: more-work-needed

## Required Work
- P1: Bucket-rounded continuation windows run past the 40960-token context and the decoder raises on a valid request (tt/model.py prefill_hidden rounds every window to a bucket from the tile below the start; functional_decoder.py raises when start_pos + seq_len > 40960 or the page table is too short; reachable through any prefix-cache hit whose suffix ends within the last bucket-rounding distance of the context; no stage evidence covered that region). Required next step: bound the window so phys_base + compute_len <= max_context, warm the variant, add generator-level evidence at (40000, 40959) and (32, 39969) and one served hit near the context end on a tracked server with program-cache counts.
- P2: Served greedy text leaves the same-precision stage-8 control (doc/datatype_sweep/qualitative/tt_outputs.json) earlier than that control leaves HF in 4 of 6 prompts, with served-only disfluencies; the stage verdict used the stage-7 control. Required next step: a same-regime control (stage-9 code, same chip, bucket 1, 128 tokens); if it reproduces the stage-8 text, compare served logprobs against generator logits at the divergence and classify.
- P2: The test_mixed_params_batch classification rests on an explanation the server log contradicts (the full-run failure is a top_k 100, temperature 2.0, seed 42 request that is host-routed in every step; both runs were entirely host-mode), and the logit reproducibility check through vLLM was not run; README says 3 of 5 runs, artifacts show 4 of 5. Required next step: logit reproducibility through the live server (alone and at several batch positions, two runs, stated tolerance), correct work log 5.10 and the README.
- P2: doc/full_model/serving_contract.json rows prefill_forward and supports_prefix_caching still describe the pre-fix per-offset compile limitation. Required next step: regenerate the notes, rerun the checker, log the commit.

## Other Concerns
- Served continuations never exercised a tile-unaligned start (the plugin disables chunked prefill for qwen3; prefix-cache hits are block-aligned); the lead-row path is covered at generator level only.
- Work log commit table lacked HEAD 85020b4eed.
- vllm_run_summary.json from attached stages records server_command null; the launch line is in README section 3 and the runner log.
- server.log line 198 "Allocating device buffers is potentially unsafe" during capture_decode_traces; the tracked server raised nothing, so transient.
- Serving-stack deviation is the operator's recorded decision; registration resolves to the autoport adapter in both processes.
- supports_device_penalties False routes whole batches with a penalty request to host sampling (1954 of 6660 decode lines in the suite); the primary benchmark window is clean (123 steady device steps, zero refreshes).

## Hard-Check Gaps
- mixed_shape_check.py counts trace releases by grepping for release_trace/retire; the sampler's own trace reset is DEBUG level, so 0 is absence of evidence (risk low: allow_force_argmax False).
- The tracked server observed the deferred host-mode capture only for the bucket test_min_p used.
- No logit-level comparison through vLLM existed at review time.
- Stale-token/current-position/page-table coverage through the adapter is indirect (generator-level split_sampling_trace, served greedy repeats, long-resume pair, benchmark refresh pattern).

## Anomaly Ledger (summary)
- Continuation windows past the context: more-work-needed (P1) -> fixed in 8.1.
- Served greedy diverges from the same-precision control: more-work-needed (P2) -> 8.2, 8.3.
- test_mixed_params_batch 4 of 5 failures: more-work-needed (P2) -> 8.5.
- Serving contract notes stale: more-work-needed (P2) -> fixed.
- First prefix-cache hit on the tracked reduced server compiled 10 programs after capture: fixed (with the P1 regression noted).
- Allocator warning during capture_decode_traces: controlled.
- Resume check top-1 31 of 32 at the 16352 window: controlled (stage-7 margin probe).
- Two API-server tracebacks (logprobs 151936 > 20, HTTP 400, test_chat_logprobs_all_vocab skipped): controlled.
- Orphaned EngineCore after the first reduced-server crash: controlled (killed, chip verified; no holder now).

## Residual Risk
- Chip-to-chip numerics not characterized at review time (settled in 8.2: identical on chips 0, 1 and 2).
- Async overlap correctness rests on the plugin draining pending steps before non-steady steps; no adapter-level stale-input assertion.
- Batched prefill not implemented (stage 10); decode pauses up to 1348 ms during a serial burst.
- supports_device_penalties False (stage 10).
- Deprecated-fork serving stack; a later migration needs the chain rerun.
- Deferred host-mode captures and the sampler's internal trace slot are not observed by per-step logs.
