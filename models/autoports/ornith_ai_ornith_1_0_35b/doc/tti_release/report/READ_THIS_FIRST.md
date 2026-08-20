# How to read the copied release report

`report_ornith-ai__Ornith-1.0-35B_2026-08-19T215826+0000.md` is the customer-facing markdown exactly as
`tt-inference-server` v0.20.0 generated it. Three things it does not say on its own face:

1. **The accuracy numbers are CI-subset results, not full-set accuracy.** The run used
   `--limit-samples-mode ci-nightly`, which becomes `--limit 0.05`. The table's sample counts are:

   | task | samples run | full set | score in the report |
   |---|---|---|---|
   | `ifeval` | **28** | 541 | 82.08 |
   | `r1_gpqa_diamond` | **10** | 198 | 50.00 |

   `r1_gpqa_diamond` at n=10 has a 16.7-point standard error; an earlier run of the same subset on the
   same code scored 60.0 (`../evals/prior_run_same_subset/`). Do not compare either number with a
   full-set release threshold.

2. **`Acceptance Criteria: PASS` is ungraded.** Every benchmark row is `NA` (no `perf_reference` is
   configured for this model) and both eval rows are `NA` (no published or GPU reference score exists
   for this checkpoint on these tasks). The model status is `EXPERIMENTAL`, which in
   `workflows/workflow_types.py` disables eval and benchmark-tier enforcement entirely. The PASS means
   the release workflow ran end to end with nothing failing — not that a quality bar was cleared.

3. **`Spec Tests: NA` means no API-conformance coverage ran in this configuration**, not that
   conformance passed. The suite was run separately and 15 of 22 rows failed. See `../RUN_NOTES.md` §12
   for the full result, the controls that explain each failure, and what is disclosed rather than waived.

`../RUN_NOTES.md` is the authoritative handoff note.
