# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The new checker was untracked in the live worktree at review time.  The
  remediation checkpoint must include
  `models/common/readiness_check/check_degenerate_output.py`.
- The supplied source commit is not present in this checkout's local object
  database, but the file itself hashes to the requested Git blob
  `18865610c5fec9f1f0cd8c27ff915d99492cc35c` and SHA-256
  `4216f5456f9367c881e8f25b5a26305971dad95a612b296a13038d4cbfec213e`,
  which proves byte-for-byte identity.

## Hard-Check Gaps

- No blocking hard-check gaps found for this remediation.
- Hardware, vLLM, servers, device reset, and TT-device experiments were not
  rerun by instruction.  The remediation changes only the repo-local host
  checker and context metadata, so it does not invalidate the existing
  hardware evidence.

## Anomaly Ledger

- Observed anomaly: the runner-visible full-model gate failed because
  `models/common/readiness_check/check_degenerate_output.py` was missing.
  - Evidence: the original check log reports that Python could not open that
    repo-local path; the work log also shows that the earlier qualitative
    command used the checker from `/home/ttuser/dev/scratch/tt-metal`.
  - Affected path: runner-side full-model gate.
  - Control or comparison: the current repo-local checker runs successfully,
    and the exact `06-full-model.check.sh` exits 0.
  - Likely subsystem: evidence/checker packaging, not model runtime.
  - Investigation performed: inspected the original prompt/check log/check
    script, live diff, new checker, and direct gate output.
  - Resolution: fixed.
- Observed anomaly: the context contract initially used non-canonical
  top-level keys.
  - Evidence: `check_context_contract.py` accepts `hf_advertised_context` and
    `current_supported_context`; the current contract records both as 131072.
  - Affected path: context-contract gate.
  - Control or comparison: `check_context_contract.py --require-contract
    --strict-caps` exits 0.
  - Likely subsystem: metadata schema mismatch.
  - Investigation performed: inspected the checker key expectations and the
    context JSON diff.
  - Resolution: fixed.

## Scope Inspected

- Goal/skill paths: original full-model prompt, failed check log,
  `06-full-model.check.sh`, and the stage-review, full-model,
  tt-device-usage, qualitative-check, and tracing requirements.
- Artifact paths: `doc/context_contract.json`, `doc/full_model/README.md`,
  `work_log.md`, `runtime_fallback_audit.md`, `sampler_decision.md`,
  `qualitative/*`, `artifacts/autoregressive/*`, readiness JSONs, and the
  trace/profiler/reproducibility artifacts.
- Code paths: the new checker plus read-only inspection of `tt/model.py`,
  `tt/generator.py`, and `tests/test_full_model.py`.
- Commands run:
  - `git status --short`
  - `git diff`
  - `git hash-object models/common/readiness_check/check_degenerate_output.py`
  - `sha256sum models/common/readiness_check/check_degenerate_output.py`
  - `python -m py_compile models/common/readiness_check/check_degenerate_output.py`
  - `pre-commit run --files models/common/readiness_check/check_degenerate_output.py`
  - `python -B models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/openai_gpt_oss_120b --missing-artifacts critical --scope autoregressive`
  - `python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/openai_gpt_oss_120b --missing-artifacts critical --scope all`
  - `python .agents/scripts/check_context_contract.py --model-dir models/autoports/openai_gpt_oss_120b --hf-model openai/gpt-oss-120b --stage full-model --require-contract --strict-caps`
  - `MODEL_DIR=models/autoports/openai_gpt_oss_120b HF_MODEL=openai/gpt-oss-120b bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh`

## Residual Risk

- The stage still depends on its saved hardware evidence for the full 36-layer
  TTNN run; the reviewer did not rerun silicon validation.
- The checker was restored with the intended content, the gate is not
  weakened, the current artifacts are found and non-degenerate, and the
  context gate recognizes the full 131072-token contract.
