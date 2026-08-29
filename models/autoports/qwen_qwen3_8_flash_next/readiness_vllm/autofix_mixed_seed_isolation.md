# AutoFix Report: mixed seeded sampling isolation

## Starting Evidence

- The live full sampling suite completed with 71 passed, 1 skipped, and one
  failure: `test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch`.
  The seeded `TopP: ` request (`temperature=1`, `top_p=0.5`, `seed=7`) returned
  `0.95,\nTopK` in the original order and `0.95\n, Temp:` after shuffling.
- `server.log` records that vLLM loaded the model generation defaults
  `top_k=20`, `top_p=0.95`, and `temperature=1`.  The failing request overrides
  top-p but remains individually eligible for Qwen's bounded top-k-32 device
  sampler.
- A fresh independent source/log audit found no request/row/virtual-slot mapping
  defect.  `CachedRequestState` owns the Torch generator, condensation moves it
  with the request, and `_build_host_generators` rekeys it to current rows.

## Hypothesis Experiments

- Hypothesis: virtual-slot TT RNG/signature restoration assigned the wrong seed.
  Experiment: trace the `sampling_params=None` host-compatibility branch and the
  plugin's request/slot/parameter vectors.
  Result: refuted.  Host compatibility never executes the TT sampler, and all
  vectors derive from the same `req_indices` order.

- Hypothesis: vLLM host-generator row mapping changes the seeded stream after a
  row reorder.
  Experiment: run vLLM's host sampler over fixed per-request logits with seed-7
  and seed-42 generators in A/B and B/A order, including the plugin's explicit
  generator advancement.
  Result: refuted.  Both orders produced identical per-request token sequences.

- Hypothesis: cohort composition switches one explicit-seed request between two
  different seeded sampling algorithms.
  Experiment: compare live cumulative metrics across the two 10-request runs.
  Result: verified.  First run: `sampling_seed_host_copies` 34 -> 48,
  token-out `trace_replays` 26 -> 32, and model-only replays 1967 -> 1996.
  Shuffled run: seed copies 48 -> 56, token-out replays stayed at 32, and
  model-only replays 1996 -> 2020.  The first order mixed TT and host sampling;
  the shuffled order used a different mixture.  A seed cannot make different
  RNG/sampling algorithms emit identical text.

## Fix

- Qwen now declares the explicit optional capability
  `force_host_seeded_sampling=True`.
- The shared plugin keeps any active stochastic request with an explicit seed
  (`temperature != 0` and seed != `SEED_NONE_SENTINEL`) on vLLM's host sampler
  for the whole cohort, independent of admission/order.  The capability defaults
  off for other models.
- Unseeded requests and all greedy requests retain the canonical on-device
  traced token-out path.  No adapter sampling implementation, host argmax,
  logits fallback, or Python token-feedback loop was added.

## Verification

- Focused policy/order and adapter capability tests pass.  The regression checks
  the seeded request in both cohort row positions, plus unseeded-random and
  seeded-greedy controls that remain device eligible.
- Main host protocol/state-bank suites: 38 passed.
  Artifact: `autofix_mixed_seed_isolation_host.xml`.
- Full plugin non-TT suite: 169 passed, 8 skipped.
  Artifact: `autofix_mixed_seed_isolation_plugin.xml`.
- No server process or TT device was touched by this isolated repair.

## Final Status

- Source and host-only policy fix: verified.
- The coordinator must restart the server to load the plugin/model capability,
  then rerun the targeted mixed-params test and the full sampling profile.
