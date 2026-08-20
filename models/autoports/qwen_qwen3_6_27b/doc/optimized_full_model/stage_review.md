# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The deferred-read API stores the outstanding logical batch width in mutable generator-global `_last_decode_batch_size`. The final mixed-prompt Watcher gate proves the selected one-outstanding-read lifecycle, but a future scheduler must either enforce that ordering or carry batch/request metadata with each deferred result before interleaving multiple outstanding reads.
- The 3.52% TTFT improvement is one ordered, same-process, exact-shape warmed baseline/selected comparison. It is a valid like-for-like final-default reproduction, though repeated/interleaved samples would give a stronger estimate of the relatively small 23.9 ms delta.

## Hard-Check Gaps

- Final Watcher evidence uses the documented `TT_METAL_WATCHER_DISABLE_ETH=1` retry because unscoped ACTIVE_ETH instrumentation exceeded its kernel-config buffer before model setup. The final Tensix/device-path Watcher artifact passes; Ethernet firmware is outside that scoped run.
- The focused token-out profiler marks inherited BFP4/LoFi decoder rows `SLOW` and omits output-subblock metadata. This is controlled by the accepted optimized-multichip precision-locked geometry evidence for the same shapes, and the current profiler rows reproduce the selected BFP4/BFP8, LoFi, DRAM-sharded, and block-width policy. It is not a new full-model fallback.
- The work log says both “timing lifecycle suite passes 6/6” and later “intended readiness-runner unit suite ... (5/5).” The source contains the relevant timing-separation and warmup/reset lifecycle tests, so this count discrepancy does not conceal a missing contract test, but future log cleanup should name the exact selections.
- This rereview did not rerun host or TT tests and did not access hardware. It used the existing source, JSON, CSV, JUnit, output text, and manifest evidence as required by the read-only review constraint.

## Anomaly Ledger

- Observed anomaly: The prior teacher metric included trace capture because the warmup reset correctly released request-owned traces.
  Evidence: `models/common/readiness_check/run_teacher_forcing.py` now records the first post-prefill callback, reports 99 capture-inclusive intervals separately from 98 later intervals, and `evidence/final/teacher_forcing_metrics.json` reports 4.966378870 s capture-inclusive, 4.344353522 s steady, and 622.025348 ms setup. The arithmetic closes exactly: `4.966378870 - 4.344353522 = 0.622025348` seconds.
  Affected path: Warmed traced teacher-forcing throughput and its headline provenance.
  Control or comparison: The earlier capture-inclusive rereview artifact was 19.919590 t/s/u; the corrected artifact reports 19.934041 t/s/u capture-inclusive and 22.558017 t/s/u steady replay, while retaining 97/100 top-1 and 100/100 top-5/top-100.
  Likely subsystem: Readiness-runner timing boundaries versus generator trace ownership/reset lifecycle.
  Investigation performed: Followed callback placement through `Generator.generate`, `_decode_traced_device`, `_capture_split_traces`, forced-token feedback, and `reset`; inspected `_compute_perf_stats`, its direct interval test, the warmup/reset lifecycle test, final JSON, README, work log, Autofix report, perf summary, and manifest hash.
  Resolution: fixed. Capture remains visible rather than being hidden, and the steady metric begins only after capture and the first decoded callback complete.

- Observed anomaly: The inherited greedy sampler gathered the complete vocabulary, while several compact Ring gather candidates stalled under Watcher.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `evidence/candidates.csv`, focused Autofix JUnits, final Watcher JUnit, and `evidence/final/profiler/token_out_report.csv`.
  Affected path: Default autonomous greedy token-out.
  Control or comparison: Full-vocabulary baseline 2.417212 ms; generic `Sampling1D` 10.736105 ms; selected distributed local argmax/all-broadcast 1.478412 ms. All selected/control tokens equal host greedy token 225721.
  Likely subsystem: Distributed greedy sampling and compact CCL ordering.
  Investigation performed: Inspected local max/argmax, exact global-index reconstruction, persistent `tt_out_tok` copy, model opt-in, stochastic-path separation, profiler topology, candidate shapes, JUnit statuses, and fallback audit.
  Resolution: fixed. The selected physical-Ring path uses two compact all-broadcasts, contains no full-vocabulary greedy all-gather or TopK, passes eager/trace and final scoped Watcher evidence, and keeps stochastic top-k/top-p separate.

- Observed anomaly: Final evidence directories retain failed Watcher/profiler attempts.
  Evidence: `evidence/final/reduced_trace_watcher.junit.xml`, `reduced_trace_watcher_no_eth.junit.xml`, passing `reduced_trace_watcher_distributed_greedy_final.junit.xml`, and profiler repair notes in `capture.md`.
  Affected path: Evidence collection, not the selected runtime.
  Control or comparison: The manifest names the passing final Watcher artifact and complete compact profiler reports; all ten manifest SHA-256 values match the current files.
  Likely subsystem: ACTIVE_ETH instrumentation capacity, an earlier test expectation, and profiler-buffer/merge capacity.
  Investigation performed: Parsed all final and Autofix JUnit statuses, read the failure classifications and profiler provenance, checked artifact existence, and recomputed primary hashes.
  Resolution: controlled/fixed.

- Observed anomaly: The performance-only prompt-128 benchmark records token 198 as both its first and last generated token.
  Evidence: `evidence/final/token_out_metrics.json`.
  Affected path: Synthetic performance prompt only.
  Control or comparison: The prompt is deliberately dominated by repeated token 198; direct AIME HF/TT completions and all six chat-template qualitative HF/TT outputs are coherent, prompt-relevant, nondegenerate, and mechanically passing.
  Likely subsystem: Synthetic benchmark prompt rather than cache, feedback, or sampling corruption.
  Investigation performed: Read the direct completions, token metadata, all six qualitative outputs, prompt-format metadata, degeneracy report, and manual verdict.
  Resolution: controlled.

- Observed anomaly: Accuracy and hardware artifacts were produced with runtime commit `9b415f82002af5d9040eca389d703690e405d91f`, not the live checkout commit `d75b892909a`.
  Evidence: `work_log.md`, profiler `capture.md`, and runtime metadata in final JSON.
  Affected path: Reproducibility/provenance.
  Control or comparison: The work log records the installed-runtime source/header mismatch, places this checkout first for model Python, gives exact environment variables, and manifest hashes bind the reviewed outputs.
  Likely subsystem: Local TT runtime environment selection.
  Investigation performed: Compared branch/HEAD, dirty diff scope, recorded commands, runtime paths, JSON metadata, and profiler provenance.
  Resolution: controlled; provenance is explicit and no cross-runtime before/after claim is used for the selected same-process A/B results.

## Scope Inspected

- Goal/skill paths:
  - Supplied stage 07 optimized-full-model goal contract.
  - `.agents/skills/stage-review/SKILL.md`.
  - `.agents/skills/multichip/SKILL.md`.
  - `.agents/skills/optimize/SKILL.md`.
  - `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths:
  - `doc/optimized_full_model/{README.md,work_log.md,AUTODEBUG.md,AUTOFIX.md}` and `doc/context_contract.json`.
  - Primary final token-out, perf-summary, prefill, teacher-forcing, autoregressive, qualitative, profiler, Watcher, candidate-ledger, fallback-audit, and artifact-manifest evidence.
  - Referenced optimized-multichip layer-floor/context/geometry evidence where needed to validate inherited-policy claims.
- Code paths:
  - Autoport `tt/{generator.py,model.py}` and `tests/test_full_model.py`.
  - `models/common/sampling/tt_sampling.py`.
  - `models/common/readiness_check/{run_teacher_forcing.py,test_run_teacher_forcing.py}`.
- Commands run:
  - Read-only `git status`, branch/commit/log and targeted diff inspection; `rg`, `find`, `sed`, `jq`, `sha256sum`, JUnit/CSV parsing, JSON arithmetic checks, artifact existence checks, and `git diff --check`.
  - No TT device, `tt-smi`, reset, server, vLLM, Tracy, profiler, or hardware test command was run.

## Residual Risk

- Full 262,144-context and batch-32 capability are inherited from accepted exact-policy gates rather than rerun in this review. The optimized batch-1 state-reuse branch is bypassed for larger batches, and current final coverage includes mixed nonaligned lengths, inactive fixed slots, changed-only page tables, deferred reads, reset, stochastic sampling, and traced device feedback; no static regression was found.
- The compact distributed argmax is model-opt-in and exercised for this exact TP4 padded-vocabulary contract. Future opt-ins with different mesh ordering, shard width, or vocabulary padding require their own exact-token, tie-order, trace, and Watcher validation.
- The caller-visible Python path necessarily observes one compact token per step and is slower than the no-readback autonomous device trace; both numbers are correctly separated. Future serving integration must preserve the existing deferred-read ordering and must not reintroduce host argmax or full-logit readback.
- Stage-owned changes are still uncommitted at review time. Per the stage-review workflow, the owner should create isolated local checkpoint commit(s) only after accepting this clean review, excluding the unrelated dirty Tracy/UMD/cluster-descriptor state.
