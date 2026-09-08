# Context report checker regression

The first final stage gate passed text-degeneracy checks, then rejected the
P150 50,624-token setting in `perf_summary.json` and its qualitative review.
The same value is permitted by the existing physical-capacity contract and by
the already-validated profile-scoped serving manifests. The scanner recognized
an explicit top-level `profiles` mapping only in `readiness_vllm/` aggregate
files, so identical data under `doc/optimized_vllm/` incorrectly inherited the
global 262,144 floor. [Original gate log](final_stage_check.log) preserves the
failure; no served setting was reduced or renamed to evade checking.

Added regressions to `.agents/tests/test_check_context_contract.py` before the
fix: 13 tests ran with five failing assertions/subtests in
[context_report_regression_before.log](context_report_regression_before.log).
The scanner now recognizes the same explicit profile mapping at the three
existing serving stage report roots. Global/sibling settings, unknown profiles,
nested directories and other hardware profiles retain their previous floors.
The P150 qualitative report places its observed setting in an explicit
`profiles.P150.max_model_len` field; the numeric value is unchanged.

`python3 .agents/tests/test_check_context_contract.py` passes all 13 tests after
the fix, including refusal to lower P150 below 50,624 or let P150x2 borrow that
limit. See [passing regression log](context_report_regression_after.log).
Applicable pre-commit checks pass in [context_checker_lint.log](context_checker_lint.log).
This Python-only report-checker change affects no serving/runtime source.

The exact full gate command is:

```bash
PATH=/home/hous/dev/tt-metal/python_env/bin:$PATH MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it HF_MODEL=google/gemma-4-26B-A4B-it bash .agents/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh
```

It returns **0** in [final_stage_check_after.log](final_stage_check_after.log):
no degenerate output and a valid context contract. Remaining plain-text
advisories quote the documented P150 cap and historical checker logs; no
structured serving configuration fails its applicable profile limit.
