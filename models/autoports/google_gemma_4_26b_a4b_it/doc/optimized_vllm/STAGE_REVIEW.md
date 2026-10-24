# Stage review (2026 Oct 24 resume)

The independent review of this attempt ran three times. The first pass (on `45376c17e2`) returned
`more-work-needed`; its findings and the stage owner's responses are in [work_log.md](work_log.md),
section "Independent review and follow-up". The second pass (on `3cb8971d5e`) returned
`more-work-needed` with the items below; the remedies are in work_log.md, section "Second
independent review". The 2026 Sep 08 review of the previous attempt is archived as
[STAGE_REVIEW_20260908.md](STAGE_REVIEW_20260908.md).

## Second review (2026 Oct 24, commit 3cb8971d5e): more-work-needed

Required work as returned by the reviewer, condensed:

- P1: out-of-vocabulary token ids reached the detokenizer from the served on-device sampling path
  (42 `Encountered invalid token id` lines in the final server log during the sampling suite's
  temperature-5 batch tests, after `slot_remap` recaptures of the batch-32 sampled trace; the same
  error on both earlier batched-prefill servers; no per-row control; not recorded by the stage).
  Required: reproduce, bisect (batched prefill, remap release with a pending async read, the
  high-temperature sampler path), fix or control, scan the server log in the sampling verdict,
  classify in the anomaly ledger.
- P2: allocation safety of the new eager paths unproven; `corruptible_allocation_scope` wraps whole
  batched forwards and the warmup; the tracked adapter probe never reached the batched path; the
  allocator warning "potentially unsafe due to the existence of an active trace" unclassified.
  Required: one tracked replay run with the scopes narrowed or disabled, the allocations each scope
  encloses listed, the warning classified.
- P2: `doc/full_model/serving_contract.json` cited evidence files that do not exist at the stated
  paths (`readiness_vllm/sampling_tests.log`, `vllm_benchmark.json`, `async_overlap_state_test.json`).
- P2: the documented trace-lifetime open item omitted the largest release class on the final server
  (`trace_key_miss`, 27 of 89 releases).
- P2: `supports_prefix_caching_sliding_window` relies on a plugin interface added by local commit
  `cd25e6602` without recorded per-item authorization. Required next step: human authorization.

Other concerns recorded by the reviewer: padding K/V written into live cache rows up to the
block-aligned length (decode overwrites before any read); the burst serial reference and the
single-user before run came from the per-row server rather than the `GEMMA4_BATCHED_PREFILL=0` A/B;
the final mixed-shape run formed no batched group; greedy decode differs between the batch-1 and
32-slot lanes (pre-existing); `*.log` is gitignored, so the cited logs exist only in the worktree;
`vllm_run_summary.json` has null `enable_prefix_caching` and `max_model_len` in attached mode.

## Third review

Latest verdict: pending re-review after the remedies.
