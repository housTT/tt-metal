# Stage Review
Verdict: clean-pass

## Required Work
- None.

## Other Concerns
- Addressed in focused rereview: the README no longer says generic prefill samples the first token on device. It now describes the inherited pre-sampled GPT-OSS channel-token scalar, separates that shortcut from canonical split-sampler decode execution, and states that baseline/optimized TTFT are like-for-like while prefill logits accuracy is validated separately.
- Packaging-focused rereviews confirmed the generated stage `.log` and `.csv` evidence is losslessly gzip-compressed, while profiler `raw_ops.csv` remains losslessly xz-compressed. Stored and decompressed hashes in profiler/watcher provenance match. Final staged rereview also confirmed repository hook normalization only removed trailing padding/final-newline differences from the five human-readable `tt_perf_report.txt` tables, normalized EOFs in `hf_completion.txt`/`tt_completion.txt`, and normalized EOFs in the two degenerate-check JSON files; staged table hashes match `tt_perf_report_txt_sha256`, the JSON files parse with empty findings, and staged inventory had zero mismatches before this report-only edit.
- The final hardware/profile evidence was captured before Black-only formatting changed `models/autoports/openai_gpt_oss_120b/tt/model.py` and `models/autoports/openai_gpt_oss_120b/tests/test_full_model.py`. Provenance records both profiled and formatted hashes, the current hashes match the formatted hashes, `git diff --check` is clean, and a narrow syntax check passes; still, the final hardware evidence is not byte-for-byte on the current formatted source for those two files.
- The final verification ledger claims pre-commit, compileall, JSON parse, 18 host-only pytest cases, and wrapper build. I found supporting build-tree evidence for the wrapper build (`build_codex_optimized_full_model/.ninja_log`, `install_manifest.txt`, linked `ttnn/_ttnn.so`, `ttnn/_ttnncpp.so`, examples, and install target), but the optimized-stage artifact directory does not retain stdout/exit logs for pre-commit, compileall, JSON parse, or the 18 host-only pytest cases. The claims are documented in `work_log.md`; retaining logs would make the audit stronger.
- The watcher validation used `TT_METAL_WATCHER=1` rather than the `TT_METAL_WATCHER=10` convention named in the device-usage skill. The log shows the watcher server initialized with disabled features `None`, checked all four devices, stopped cleanly, and `SAFE_PYTEST_RESULT: PASS`; I treated this as watcher-enabled evidence, not a blocker.

## Hard-Check Gaps
- I did not rerun device, profiler, watcher, vLLM, or server experiments; the review task explicitly prohibited starting hardware/vLLM/server/device jobs. I inspected retained artifacts and logs instead.
- I did not independently rerun the wrapper build or final pre-commit/host test suite. The build tree supports the build claim, but retained command logs are absent for several host-side checks as noted above.
- P150 and P150x2 are not hardware-run resident targets in this stage. Their status is capacity-accounting only: fixed resident state already exceeds 32 GiB/device, so largest feasible resident context is recorded as zero.
- Full profiler raw data is reduced-path profiling, not all 36 layers. This matches the applicable profiling guidance; the 36-layer token-out result is tied back through the optimized-multichip per-layer lower bound plus profiled terminal/sampler rows.

## Anomaly Ledger
- Observed anomaly: Initial review found the README wording "samples the first token on device" overbroad for GPT-OSS prefill.
  Evidence: `models/demos/gpt_oss/tt/model.py` returns `torch.full(..., 200005)` when `sampling_params is not None`; the same code exists at the full-model base commit `5058406ce09eb40a1148b91d4f0f7cc485b1115e`. Optimized `Generator._device_prefill_sample()` calls that path. HF/TT qualitative and AIME artifacts consistently start with the GPT-OSS channel token. Focused rereview confirmed the README now says token-out prefill returns GPT-OSS's inherited pre-sampled channel-token scalar, says this is not generic sampler execution, and says prefill logits accuracy is validated separately.
  Affected path: First-token prefill wording and TTFT interpretation; not the steady-state decode token-out trace.
  Control or comparison: Baseline code comparison at `5058406ce09eb40a1148b91d4f0f7cc485b1115e`; AIME prefill readiness still separately validates logits with top-1/top-5/top-100 `0.94/1.0/1.0`.
  Likely subsystem: GPT-OSS model-specific prefill sampling shortcut/documentation.
  Investigation performed: Compared current and base `process_output_prefill`, inspected optimized generator calls, checked prompt/control outputs and prefill readiness artifact, then reread the updated README wording.
  Resolution: Addressed by README update. Decode-token-out path still satisfies the canonical split sampler, `tt_out_tok`, persistent state, zero steady host-refresh, and zero full-logit-readback contract. Verdict remains clean-pass.
- Observed anomaly: Autoregressive metadata prompt text does not byte-match the referenced prompt file.
  Evidence: `artifacts/autoregressive/autoregressive_meta.json` omits the source file's final newline. Hashes differ, but `prompt_text.strip()` matches the file, prompt token IDs are identical to the full-model autoregressive reference, and the optimized/full-model prompt token count is 214 in both artifacts.
  Affected path: Prompt provenance text field only.
  Control or comparison: Compared optimized metadata to `doc/full_model/prompts/autoregressive_chat_prompt.txt` and `doc/full_model/artifacts/autoregressive/autoregressive_meta.json`.
  Likely subsystem: Artifact serialization/trailing newline handling.
  Investigation performed: Compared lengths, SHA256s, first differing position, stripped text, prompt token IDs, and HF token prefix.
  Resolution: Nonblocking. Token identity, which is the relevant model input, is preserved.
- Observed anomaly: Earlier source-unchanged batch-2 nondeterminism and watcher assert are documented in the stage logs.
  Evidence: `runtime_fallback_audit.md`, `work_log.md`, `batch2_logit_reproducibility.json`, and watcher provenance show recovery by bounded reset and a CCL scatter-state guard; final batch-2 full-logit reproducibility is bitwise exact and final watcher path passes with disabled features `None`.
  Affected path: Cache/request isolation and sampler UINT32 all-gather watcher coverage.
  Control or comparison: Final `batch2_logit_reproducibility.json` has zero differing values for prefill/decode rows and runs; decompressed watcher `watcher_final_both_headers.log.gz` content ends with `SAFE_PYTEST_RESULT: PASS`.
  Likely subsystem: Resident KV/cache lifecycle and all-gather kernel watcher state.
  Investigation performed: Inspected recovery ledger, C++ all-gather header diffs, final watcher provenance/log, and batch-2 reproducibility artifact.
  Resolution: Resolved by current stage evidence. No required work.
- Observed anomaly: Profiler provenance source hashes for two Python files differ from the current formatted tree.
  Evidence: `artifacts/profiler/final_source/provenance.json` records profiled hashes and post-qualification Black formatted hashes for `tt/model.py` and `tests/test_full_model.py`; current SHA256s match the formatted hashes.
  Affected path: Exact byte-for-byte reproducibility of retained hardware/profile evidence.
  Control or comparison: Current hashes: `tt/model.py` `25b18c6f...`, `tests/test_full_model.py` `5ef3073f...`; provenance formatted hashes match. `git diff --check` and `python -m py_compile` over changed Python files passed.
  Likely subsystem: Post-run formatting/provenance.
  Investigation performed: Recomputed source hashes, inspected provenance, checked diff whitespace and syntax.
  Resolution: Nonblocking hard-check gap; no semantic issue found.
- Observed anomaly: Verification logs for some final host-side checks are not retained.
  Evidence: `work_log.md` claims pre-commit, compileall, JSON parse, 18 host-only tests, and wrapper build. `find`/`rg` found no retained pre-commit, compileall, host-contract, or wrapper stdout logs under `doc/optimized_full_model`; build-tree artifacts exist for the wrapper build.
  Affected path: Auditability of final verification ledger.
  Control or comparison: `build_codex_optimized_full_model/.ninja_log` and `install_manifest.txt` show linked/install outputs; packaged evidence records resolved with matching hashes. In the final staged rereview before this report-only edit, all 27 inventory entries matched staged blobs.
  Likely subsystem: Evidence packaging.
  Investigation performed: Searched optimized evidence tree and build tree; checked artifact inventory hashes.
  Resolution: Nonblocking audit gap. Future stages should retain logs for final host-side verification.
- Observed anomaly: `stage_review.md` is self-inventoried, so report-only rereview edits create a moving hash target.
  Evidence: In the final staged state inspected for this rereview, `artifact_inventory.json` matched all 27 recorded staged blobs, including staged `stage_review.md` hash `307755f67e80...`; the earlier stale hash anomaly was addressed by the stage owner. This final report update records hook normalization and therefore changes only the report file after that validation.
  Affected path: Review-report self-hash only.
  Control or comparison: Profiler provenance gzip/xz stored and decompressed hashes all matched; watcher provenance gzip stored and decompressed hashes all matched; decompressed `watcher_final_both_headers.log.gz` and `final_full_stack_acceptance_prompt214_gen100.log.gz` still contain `SAFE_PYTEST_RESULT: PASS`.
  Likely subsystem: Artifact packaging/inventory self-reference.
  Investigation performed: Re-ran read-only staged inventory hash verification and focused staged blob checks for normalized profiler tables, completion text EOFs, degenerate JSON EOFs/semantics, and profiler/watcher compressed provenance.
  Resolution: Previous inventory mismatch addressed. Nonblocking under the explicit instruction to update only `stage_review.md`; if this updated report content is committed with inventory coverage, refresh the `stage_review.md` inventory hash after staging this report.

## Scope Inspected
- Goal/skill paths:
  - `AGENTS.md`
  - `.agents/runs/gpt-oss-120b-p150-family-20260827T214421Z/07-07-optimized-full-model.prompt.txt`
  - `.agents/skills/multichip/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - `.agents/skills/full-model/SKILL.md`
  - `.agents/skills/tt-enable-tracing/SKILL.md`
  - `.agents/skills/qualitative-check/SKILL.md`
  - `.agents/skills/stage-review/SKILL.md`
- Artifact paths:
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/README.md`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/work_log.md`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/perf_summary.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifact_inventory.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/runtime_fallback_audit.md`
  - `models/autoports/openai_gpt_oss_120b/doc/context_contract.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/p150x4_prompt128_gen128_perf.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/p150x4_full_stack_split_isolation.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/batch2_logit_reproducibility.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/full_path_lower_bound.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/prefill_readiness.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/teacher_forcing_readiness.json`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/autoregressive/*`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/qualitative/*`
  - `models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/*`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/profiler/final_source/**`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/watcher/final_source/**`
  - `models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/final_full_stack_acceptance_prompt214_gen100.log.gz`
- Code paths:
  - `models/autoports/openai_gpt_oss_120b/tt/generator.py`
  - `models/autoports/openai_gpt_oss_120b/tt/model.py`
  - `models/autoports/openai_gpt_oss_120b/tt/multichip_decoder.py`
  - `models/autoports/openai_gpt_oss_120b/tt/optimized_decoder.py`
  - `models/autoports/openai_gpt_oss_120b/tests/test_full_model.py`
  - `models/demos/gpt_oss/tt/model.py`
  - `models/tt_transformers/tt/generator.py`
  - `models/common/sampling/generator.py`
  - `models/common/sampling/tt_sampling.py`
  - `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_common.hpp`
  - `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp`
- Commands run:
  - `git diff --name-only 5058406ce09eb40a1148b91d4f0f7cc485b1115e -- ...`
  - Focused `git diff --unified=80` inspections for optimized model/generator/decoder/test and CCL kernel headers.
  - `rg` searches for vLLM leakage, host argmax/logit readback/synchronization boundaries, fallback markers, watcher/pass/fail markers, verification-log retention, and README/ledger claims.
  - Python read-only artifact scripts to recompute artifact inventory hashes, profiler `raw_ops.csv.xz` stored/uncompressed hashes, profiler generated `.csv.gz`/`.log.gz` stored/uncompressed hashes, watcher `.log.gz` stored/uncompressed hashes, lower-bound arithmetic, perf metrics, context contract fields, prompt identity, qualitative-control hashes, source hashes, and file-size gate.
  - `git diff --check 5058406ce09eb40a1148b91d4f0f7cc485b1115e -- <changed files>`.
  - `python -m py_compile models/autoports/openai_gpt_oss_120b/tt/model.py models/autoports/openai_gpt_oss_120b/tt/generator.py models/autoports/openai_gpt_oss_120b/tests/test_full_model.py`.
  - `find` and `tail` inspections of `build_codex_optimized_full_model` and optimized-stage artifact files.
  - Focused rereview commands: `git status --short -- README.md stage_review.md`, `git diff -- README.md stage_review.md` (empty because the optimized evidence directory is untracked), direct `sed` readback of the updated README sections, and `rg` checks for first-token/channel-token wording.
  - Packaging-focused rereview commands: `find` for remaining generated `.log`/`.csv` files, `rg` for stale `.log`/`.csv` references, direct JSON/provenance inspection, and read-only Python verification of gzip/xz stored and decompressed hashes plus decompressed pass markers in `watcher_final_both_headers.log.gz` and `final_full_stack_acceptance_prompt214_gen100.log.gz`.
  - Final hook-normalization rereview commands: `git diff --cached --check`, `git diff --cached --name-status -- models/autoports/openai_gpt_oss_120b/doc/optimized_full_model`, and read-only staged-blob Python verification that all 27 inventory entries matched staged content, the five final-source `tt_perf_report.txt` files have no CR/trailing padding and match `tt_perf_report_txt_sha256`, `hf_completion.txt`/`tt_completion.txt` have normalized EOFs and match autoregressive metadata, both degenerate-check JSON files parse with empty findings, and profiler/watcher compressed provenance hashes match stored and decompressed bytes.

## Residual Risk
- The review relied on retained evidence for hardware behavior. I did not rerun hardware tests, profiler, watcher, or build under the review constraints.
- The all-36-layer performance headline is a measured end-to-end token-out run, but raw profiler decomposition is intentionally reduced-path and tied to the 36-layer result through lower-bound arithmetic. I recomputed the arithmetic: decoder stack `14.2698105 ms`, named terminal `1.058947 ms`, measured split token-out `15.903506748 ms/token`, residual `0.574749248 ms` / `3.749483597%`, gate pass.
- P150/P150x2 support is a hard-limit statement, not runtime validation. The fixed resident state exceeds 32 GiB/device before any feasible context, so this is acceptable for the stated no-fallback contract.
- Fixed-length split token-out cannot stop on EOS without an explicit host collection boundary. This is documented and outside the steady-state no-per-token-host-boundary objective.
- The code still contains explicit host compatibility paths for validation/teacher forcing/host sampling. Runtime fallback audit and token-out counters show these are not on the measured optimized split path.
