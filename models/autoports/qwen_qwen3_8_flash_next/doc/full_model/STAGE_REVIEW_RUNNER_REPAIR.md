# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- Non-blocking documentation chronology: `README.md` still contains the
  pre-repair statement that
  `models/common/readiness_check/check_degenerate_output.py` is not present.
  The later `work_log.md` repair section, the live file, and the authoritative
  check prove that this has been fixed. This stale README sentence does not
  hide a missing runner gate, but it should be read as pre-repair history unless
  the README is refreshed again.
- Non-blocking text normalization: `autoregressive_meta.json` token arrays
  match `aime24_autoregressive_100_report_final.json` exactly. The
  `hf_completion.txt` and `tt_completion.txt` sidecars match the source strings
  after stripping one trailing newline. The checker tokenizes words and is
  unaffected; this is not evidence drift.
- The repair-critical files are still live/uncommitted at review time,
  including the untracked common checker and autoregressive sidecars. Per the
  stage workflow, the owner must include the repair files in the next local
  checkpoint commit and avoid sweeping in unrelated untracked profiler
  artifacts.

## Hard-Check Gaps

- No blocking hard-check gaps remain for the runner-verification repair. The
  authoritative script
  `.agents/prompts/model_bringup_multigoal/06-full-model.check.sh` now exits
  0 with `MODEL_DIR=models/autoports/qwen_qwen3_8_flash_next` and
  `HF_MODEL=Qwen/Qwen3.8-Flash-Next`.
- This review intentionally did not open TT devices, reserve/reset hardware,
  start servers, or rerun vLLM/hardware experiments. Full-model runtime claims
  therefore rely on the already accepted current-source hardware artifacts and
  the prior clean `STAGE_REVIEW.md`; the repair itself is metadata/static
  runner wiring and was checked CPU-only.

## Anomaly Ledger

- Observed anomaly: the independent post-completion runner failed before the
  repair.
  Evidence: `06-06-full-model.check-1.log` failed with
  `python: can't open file ... models/common/readiness_check/check_degenerate_output.py`;
  separate context checking also required canonical
  `current_supported_context`/`hf_advertised_context` keys.
  Affected path: runner-side full-model completion gate.
  Control or comparison: the live checker hash matches repository verification
  commit `64f9b3ff90101009a7dc484b56ed408c3844d7bf`; `context_contract.json`
  now records both canonical keys as 262144; the authoritative check exits 0.
  Likely subsystem: runner artifact/contract wiring, not TT model execution.
  Investigation performed: inspected the failed check log, check script,
  restored checker source, context contract, generated sidecars, and reran the
  authoritative gate with bytecode writes disabled.
  Resolution: fixed.

- Observed anomaly: final README contains stale pre-repair wording saying the
  common degeneracy script is absent.
  Evidence: `README.md` line containing the absent-script statement; later
  `work_log.md` repair section and live `models/common/readiness_check/check_degenerate_output.py`.
  Affected path: human-facing documentation.
  Control or comparison: `git show
  64f9b3ff90101009a7dc484b56ed408c3844d7bf:models/common/readiness_check/check_degenerate_output.py`
  and the live file have the same SHA-256
  `2629d86ae7be926a21e4839d151d21037e0851e98266f1b9a100e7cf14541db4`;
  the authoritative gate consumes the live checker successfully.
  Likely subsystem: documentation not fully refreshed after a post-completion
  metadata repair.
  Investigation performed: searched README/work log for `check_degenerate`,
  checked file existence and hash, and reran the gate.
  Resolution: controlled.

- Observed anomaly: sidecar text files are not byte-for-byte equal to the JSON
  completion strings because each `.txt` file has one trailing newline.
  Evidence: raw equality is false, newline-stripped equality is true; HF and TT
  token arrays match exactly.
  Affected path: canonical autoregressive text sidecars.
  Control or comparison: `check_degenerate_output.py` reports 79 words,
  adjacent duplication 0.0, trigram-loop fraction 0.0759, and no findings; the
  sidecar token IDs exactly match the detailed final report.
  Likely subsystem: harmless file text normalization.
  Investigation performed: compared lengths, tails, raw equality,
  newline-stripped equality, and token arrays.
  Resolution: controlled.

- Observed anomaly: TT free-running AIME24 output first diverges from HF at
  token 5.
  Evidence: `aime24_autoregressive_100_report_final.json`,
  `autoregressive_meta.json`, `hf_completion.txt`, and `tt_completion.txt`.
  Affected path: all-48 traced greedy token-out generation.
  Control or comparison: final teacher-forcing gate reports prefill
  100/100/100% top-1/top-5/top-100 and 99 decode rows at
  91.9192/100/100%; the exact TT text remains fluent English, on-topic, and
  mechanically non-degenerate; fallback audit marks prohibited host work false.
  Likely subsystem: normal autoregressive amplification of low-precision
  rank-order differences rather than stale token feedback.
  Investigation performed: inspected raw HF/TT text, token IDs, divergence
  index, degeneracy metrics, JUnit properties, and fallback-audit counters.
  Resolution: controlled.

- Observed anomaly: two shared qualitative-suite TT outputs stop inside the
  128-token window.
  Evidence: `qualitative_shared_suite_final.json` and `QUALITATIVE_REVIEW.md`.
  Affected path: explanation and coding qualitative prompts.
  Control or comparison: HF controls also use the same 128-token limit and show
  long visible reasoning; TT outputs remain coherent, English, on-topic, and
  non-degenerate; summarization completes correctly.
  Likely subsystem: checkpoint prompt style / visible reasoning budget, not
  token feedback, cache corruption, or language drift.
  Investigation performed: inspected prompt metadata, rendered prompts,
  decoded TT snippets, prompt-level review metrics, and prohibited-host-work
  audit fields.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths:
  - `/home/ttuser/dev/qwen3.8-flash-next/multigoal-runs/stage4-20260827T204117Z/06-06-full-model.prompt.txt`
  - `/home/ttuser/dev/qwen3.8-flash-next/multigoal-runs/stage4-20260827T204117Z/06-06-full-model.check-1.log`
  - `.agents/prompts/model_bringup_multigoal/06-full-model.check.sh`
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/full-model/SKILL.md`
  - `.agents/skills/host-weight-cache/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
- Artifact paths:
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/README.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/work_log.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/STAGE_REVIEW.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/static_host_contracts_runner_repair.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/aime24_autoregressive_100_report_final.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/aime24_autoregressive_100_final.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/autoregressive_meta.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/hf_completion.txt`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/tt_completion.txt`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/aime24_teacher_99_l1_workspace_final.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch1_prompt128_generate128_performance_final.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/cold_warm_chunked_prefill_final.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_watcher_fixed.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_trace_alloc_tracker.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/non_greedy_split_trace_final.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_batch32_eager_fixed_slots.xml`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_prompt_format.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_shared_suite_final.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/QUALITATIVE_REVIEW.md`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/profiler_provenance.txt`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/context_contract.json`
  - `models/autoports/qwen_qwen3_8_flash_next/doc/host_weight_contract.json`
- Code paths:
  - `models/common/readiness_check/check_degenerate_output.py`
  - `models/autoports/qwen_qwen3_8_flash_next/demo/full_model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tt/generator.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py`
  - `models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py`
- Commands run:
  - Complete `sed` reads of the required skill files and selected stage docs.
  - `git status --short`, `git rev-parse`, `git diff --stat`, focused
    `git diff`, `git diff --check`, and commit-stat inspection for
    `8fca123037b` and `3e0d574a79b`.
  - `rg --files`, `rg -n`, `git ls-files`, and targeted source/doc searches.
  - `git cat-file` / `git show` plus `sha256sum` to compare the restored
    checker against commit `64f9b3ff90101009a7dc484b56ed408c3844d7bf`.
  - Read-only Python parsing of JSON/JUnit artifacts, sidecar equality checks,
    capacity/context summaries, evidence-path existence checks, and in-memory
    `compile()` checks for the modified Python sources.
  - `env PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=models/autoports/qwen_qwen3_8_flash_next HF_MODEL=Qwen/Qwen3.8-Flash-Next bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh`

## Residual Risk

- The review was constrained to existing artifacts and CPU/static checks. I did
  not open TT devices, run hardware/vLLM, collect profiler data, reset
  hardware, or start servers.
- The current worktree contains uncommitted repair changes plus unrelated
  untracked profiler artifacts. A subsequent checkpoint commit must include the
  stage-owned repair files and exclude unrelated untracked profiler outputs.
- The runner-side degeneracy checker is intentionally a mechanical-output gate,
  not a semantic quality judge. Semantic quality remains covered by the
  original full-model AIME24 review, shared qualitative suite, HF controls, and
  prior clean stage review.
