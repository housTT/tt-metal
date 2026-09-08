# AutoFix: presence-penalty test stimulus

2026-09-08. Status: stimulus correction, sensitive-distribution proof,
synthetic sampler, targeted canonical and full-suite verification complete.

## Confirmed failure and minimum change

The original repeated `a b c a b c a b c` continuation did not change under
presence penalties, causing two canonical serving tests to fail. The real
host control proved that all 120 sampled choices at penalties 0, -1.5, and
2 were mathematically correct and full-vocabulary certified. Even the
closest repeated-winner/unseen-candidate raw gap was 3.31250003, so a
one-time penalty of 2 left the same winner ahead by 1.31250003. Device outputs
matched the host at every setting. Details and raw evidence are in
[`AUTODEBUG_presence_variation.md`](AUTODEBUG_presence_variation.md).

The plugin's `tests/tt/test_tt_penalties.py` changes exactly the two presence
prompt strings to `Once upon a time`, plus explanatory comments. A concurrent
device screen already measured differing 40-token outputs at presence 0 and
2 for that continuation. All varied-output and deterministic mixed-cohort
assertions remain; parameters and other penalty tests are unchanged. No
runtime presence or precision code changed, and no skip was introduced.

## Verification

- Source diff confirms only two prompt literals and comments changed in the
  canonical tests; an AST comparison against the pinned original proves no
  other behavior changed. Python compilation passed for that file and both
  standalone probes. A CPU synthetic control verifies the distribution
  analyzer detects and certifies a forced presence crossing, and a CPU oracle
  verifies the exact sampler probe's two-step expected outputs.
- Original and shorter abc controls passed their host/CPU/device consistency
  checks, demonstrating why neither reliably exercises output variation.
- Four natural-continuation device screens showed variation, including the
  selected fixture. This is sampling coverage, not chat-quality evidence.
- The selected fixture's complete host/CPU/device control passed all six
  requests and 120 host choices, all full-vocabulary certified. Device token
  IDs equal host output for each penalty. Presence 2 changes the raw argmax
  at positions 16, 18, 21, and 33; presence -1.5 changes it at position 20.
  See [`presence_distribution_sensitive_v1.json`](presence_distribution_sensitive_v1.json).
- The exact hardware probe passed (exit 0), using the actual traced canonical
  sampler with 32 mixed lanes and the full padded vocabulary across four
  shards. It verified every candidate score and two-step token choice,
  presence count independence (1 versus 7), stable trace-bound addresses,
  and no new reset programs after capture. See
  [`presence_sampler_exact.json`](presence_sampler_exact.json) and
  [`presence_sampler_exact.log`](presence_sampler_exact.log).
- Both corrected B32 presence tests passed in
  [`full_b32_targeted_final.log`](full_b32_targeted_final.log), whose three-test
  run also includes the independently corrected bad-words test: 3 passed in
  32.13 seconds.
- The complete B32 canonical suite passed **72 tests with one skip** in
  329.46 seconds, including both corrected presence tests:
  [`full_b32_sampling_before_penalty_order.log`](full_b32_sampling_before_penalty_order.log).
  This closes the presence stimulus repair. A later independent shared
  combined-penalty ordering correction requires its own post-fix rerun and is
  tracked in `AUTOFIX_combined_penalty_order.md`.

No build is required for these Python test and documentation changes.
