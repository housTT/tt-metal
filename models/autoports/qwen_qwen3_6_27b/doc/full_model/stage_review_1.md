# Independent stage review 1

Verdict: **more-work-needed**.

The first independent review found the following closure gaps:

- reset could leave stale model/sampler traces bound to prior request state;
- token-out sampling was too large a share of latency and the common-sampler
  comparison was not yet a strong semantic-greedy selection;
- direct low-level decode had weak nonzero-position coverage;
- stochastic top-k/top-p parameters, mixed prompts, fixed batch-32 slots,
  duplicate-row determinism, and unchanged-page-table counters needed stronger
  evidence;
- trace setup evidence did not yet prove token/position restoration at the
  maximum context boundary;
- the capacity reserve and runtime fallback audit were incomplete;
- qualitative evidence covered only AIME rather than the shared suite;
- safe-Watcher closure, final repository commit, and commit-SHA logging were
  still absent.

Remediation:

- reset now tears down both traces before in-place state clearing; alternating
  sampling modes and repeated requests pass deterministically;
- canonical common Ring force argmax is 2.416 ms versus 10.796 ms for the same
  greedy token through `Sampling1D`;
- the expanded Ring gate covers direct `start_pos=17`, stochastic parameters,
  mixed non-aligned prompts, batch 32, duplicates, inactive rows, tensor
  identity, and zero/one page-table read counts;
- trace construction restores token, current/rotary positions, and recurrent
  state before capture and after capture; safe Watcher verifies both capture
  boundaries at position 262143;
- `context_contract.json` now accounts for full weights, BFP8 KV cache,
  batch-32 recurrent state, persistent CCL storage, sampler/trace/transient
  storage, and a 4 GiB containing reserve;
- the six-prompt HF/TT shared qualitative suite passes automated and manual
  review;
- AutoTriage/AutoFix artifacts document and close the physical-Ring CCL issue.

Final commit closure and a fresh clean-pass review are intentionally deferred
until all post-fix validation artifacts are complete.
