# AutoFix Report: GPQA Release Harness Initialization

## Starting Evidence

- Failing release log:
  `/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/evidence/tti_release.log`
  (the GPQA traceback beginning near line 111).
- The failure occurs while `ConfigurableTask` calls
  `datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond")`.
- Exception: `DatasetNotFoundError` because the Hub dataset is gated and this
  workspace intentionally has no `HF_TOKEN` or logged-in Hugging Face account.
- The traceback occurs before request construction/evaluation. No model request
  reached the live autoport server.

## Hypothesis Experiments

### Missing server/model capability caused GPQA to fail

- Experiment: reproduce only `TaskManager.load_task_or_group` with no token.
- Result: task initialization raises the same gated-dataset exception before a
  model is invoked.
- Verdict: refuted. This is a client dataset-initialization failure, not model,
  vLLM, TTNN, or hardware behavior.

### A Hugging Face credential is the only valid recovery

- Experiment: stage the publisher's official archive locally at a pinned
  revision, pair it with the cached upstream Hugging Face dataset card, and let
  `datasets==3.1.0` resolve the unchanged `Idavidrein/gpqa` identifier from the
  TTI checkout working directory.
- Result: offline initialization succeeds without a token or network access.
- Verdict: refuted. A credential is valid but not required when the official
  publisher data is supplied through the datasets library's native local-repo
  resolution.

### The official local archive preserves the evaluated task contract

- Publisher repository: `https://github.com/idavidrein/gpqa`.
- Pinned commit: `56686c06f5e19865c153de0fdb11be3890014df7`.
- `dataset.zip` SHA-256:
  `461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc`.
- `dataset/gpqa_diamond.csv` SHA-256:
  `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
- Experiment: run `datasets.load_dataset` and lm-eval `TaskManager` offline,
  printing counts/schema checks only (no examples, prompts, answers, or model
  output).
- Result:
  - raw rows: 198;
  - processed rows: 198;
  - required raw and processed columns: present;
  - processed answer labels: confined to `(A)` through `(D)`;
  - task name: `gpqa_diamond_cot_zeroshot`;
  - dataset path: `Idavidrein/gpqa`.
- Verdict: verified. The original pinned lm-eval YAML, `process_docs`, prompt,
  shuffling, filters, and metric configuration remain in use; only dataset
  acquisition changes from gated Hub download to the publisher archive.

### A TTI source-code override is required

- Experiment: compare an environment-driven custom-task override with native
  local-repo resolution.
- Result: native resolution passes the same initialization contract without
  changing source. The broader source override was removed.
- Verdict: refuted. The minimum recovery is cache-local release wiring.

## Verification

- Offline task validation from the TTI checkout root:

  ```text
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
    HF_HOME=/home/ttuser/dev/gpt-oss-20b/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    .workflow_venvs/.venv_evals_common/bin/lm_eval validate \
    --tasks gpqa_diamond_cot_zeroshot
  ```

  Result: `All tasks found and valid` (exit 0).

- Focused host regression suite:

  ```text
  python -m pytest -q tests/test_module/llm_tests/test_llm_eval_tests.py
  ```

  Result: `33 passed` (one pre-existing pytest collection warning).

- Tracked source diff after discarding the unnecessary override: empty.

## Final Status

Fixed for rerun through cache-local release wiring. The TTI checkout now has an
untracked `Idavidrein` symlink pointing to the pinned official dataset mirror
under `/home/ttuser/dev/gpt-oss-20b/.cache/gpqa-official/`. No secret, raw
example, model request, server process, device, live MMLU process, or container
was accessed.

Rerun GPQA only after the live MMLU/release process is finished. Run from the
TTI checkout root (local resolution is working-directory-sensitive), retain the
existing release command's task name/model arguments/generation settings/output
path, and set the workspace `HF_HOME`; `HF_HUB_OFFLINE=1` is recommended. After
the successful targeted GPQA result is written into the existing release eval
output, regenerate the affected TTI report sections so the prior missing row is
replaced by a scored `gpqa_diamond_cot_zeroshot` row.
