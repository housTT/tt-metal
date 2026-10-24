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

## Third review (2026 Oct 24, commit acea3bc852): more-work-needed

Required work as returned by the reviewer, condensed:

- P1: the sampler-ceiling remedy (32) routes the model's default recipe (`generation_config.json`
  `top_k: 64`, applied by vLLM to every sampled request without `top_k`) to the plugin's host
  sampler; the README said the opposite; the final-server sampled evidence (qualitative sampled
  half, the temperature-5 reproduction tests, most of the sampling suite) was host-sampled, so the
  device sampler at k = 32 was not re-exercised under the reproduction conditions; the host-sampled
  default-recipe path's latency and its trace-release effect on concurrent users were unmeasured.
  Required: fix the device sampler for k up to 64 or keep 32 and correct the statements, measure
  the host path, rerun the reproduction tests with explicit `top_k <= 32`, and if the intended fix
  is a served default (`--override-generation-config '{"top_k": 32}'`) record it and obtain
  operator acceptance.
- P2: the `batched_group_while_decoding` phase of the final mixed-shape run produced degenerate and
  wrong-script completions with no same-prompt control. Required: standalone-generator control for
  those prompts (`tests/probe_raw_prompt_control.py`) and a rewritten anomaly entry.
- P2: inherited plugin interface without per-item authorization (`supports_prefix_caching_sliding_window`,
  plugin commit `cd25e6602`; the model-type chunked-prefill branch). Required next step: human
  authorization.

Other concerns recorded: per-length single-row programs still release traces (the runtime-index
gather could remove the terminal-slice class); B1 versus 32-slot greedy divergence (pre-existing);
the before run and burst serial reference came from the per-row server; `program_cache_entries` is
null for most prefill steps; a 42-versus-9 invalid-id count discrepancy between the second review's
text and the retained log; the final server ran inert scope wrappers; no sampling-mode check
existed; the readiness batched-equals-solo copies predate the slot-row assembly; the passing tracked
logs carry no tracker fingerprint.

## Fourth review (2026 Oct 24, commit 2bd0d7aeb7): more-work-needed

The reviewer verified all three third-review remedies against the raw artifacts and found no new
correctness defect in the model or adapter code paths. Required work as returned, condensed:

- P2, human authorization: the inherited serving-plugin interface items the adapter relies on
  (`plugin_changes.md` items 1, 5, 6 and the `proposed` chunked-prefill key).
- P2, human authorization: the served default `--override-generation-config {"top_k":32}` (model
  card `top_k` 64; shared `ttnn.sampling` op limited to 32 candidates; host path measured at 97 ms
  per token without it).
- P2, documentation: README reproduction command quoting; stale `optimized_vllm` block in
  `doc/context_contract.json`; merged README table rows; sampling-mode counts attributed to
  over-wide line brackets; CI-burst "after" row labelled chain run but holding repeat-4 numbers;
  serving-contract `top_k <= 64` row and the `batched_prefill` row's pre-assembly citation.

Other concerns recorded: unexplained burst run-to-run variance (single run per server); the chain
run's single 111 ms ITL interval; the 32-slot lane halves per-user decode speed when a second user
joins (pre-existing); partial-chunk prefill steps are host-sampled by the plugin; mixed-shape trace
reuse stays `fail` (open item excluded by the task owner); padding K/V rows in live blocks rely on
position-gated reads; `decode_steps 8` in the final-code acceptance copies; no sliding-window
history-snapshot control (covered indirectly by the `no_history` control failing at the window
starts).

## Fifth review

Latest verdict: pending re-review after the documentation fixes.
