# Stage Review

Verdict: clean-pass

Independent review of stage 9, vLLM integration for `ornith-ai/Ornith-1.5-9B`,
completed 2026-09-08. This verdict covers the final live implementation and
recorded TP4 serving evidence. Local checkpoint commits follow this review;
this is not a verdict on the later optimized-serving or release stages.

## Required Work

None. Findings raised during review were investigated and resolved before this
verdict. The final shared runner completed successfully with 72 sampling tests
passed and one canonical skip. The first-request B1 benchmark, B32 CI burst,
qualitative outputs, native-context contract, and process cleanup are present.

The reviewed headline is the immediate first request after readiness:
128 input / 128 output / one request / concurrency 1 / server capacity 1,
greedy sampling, native 262144 context. Its TTFT is 62.613036 ms and its
TPOT-derived decode rate is 87.706066 tokens/s/user. The secondary
100 input / 100 output / 32-request burst completes 32/32 requests at
337.942797 output tokens/s. These values were rederived from raw results,
not accepted solely from the README.

## Other Concerns

- The qualitative evidence supports coherence and serving consistency, not
  blanket task accuracy. The selected standalone generator and serving path
  produce the same incorrect 6/7/5 completed haiku; the pinned BF16 HF control
  produces 5/7/5. Their 390-token selected-TT/serving equality controls the
  serving-regression question without claiming HF parity or identifying a
  precision culprit. Final sampled text also contains haiku miscounts,
  awkward analogy wording, and an unsupported assumption about English
  “you.” Both French final translations are correct. Several requests use
  their 256-token budget during thinking, so complete story/code execution
  is not established.
- The advertised context remains 262144. The physical pool supports a
  native-length request and 32 concurrent short requests; it does not promise
  32 simultaneous native-length requests. Prefix caching is explicitly
  disabled. TP1/TP2, million-token YaRN, and broader release accuracy are
  outside this stage's claims.

## Hard-Check Gaps

- `test_chat_logprobs_all_vocab` retains its canonical skip under the server's
  logprob cap. Other shared logprob cases pass. The additional API numerical
  comparisons cover chosen-token logprobs and top-20 maps; complete vocabulary
  reproducibility is established separately in the standalone control. This
  is not full-vocabulary API conformance evidence.
- Trace-allocation and detailed state probes use reduced real layers 0/3 and
  B1/B3. They are combined with final all-layer B1/B32 serving runs, not
  represented as an allocation audit of every full-server operation. The B3
  structural probe exercises the existing decode warmup entry; the B1 startup
  probe directly exercises the changed four-call startup sequence.
- The required primary profile contains one request. Its TTFT and TPOT P50/P99
  are consequently the same value, not a measured population tail. All 127
  token intervals and an identical-workload repeat are retained. No sustained
  load, long-duration memory stability, or universal latency bound is claimed.
- The reviewer performed read-only inspection and artifact analysis. Hardware
  and test results below are recorded execution evidence from the supervising
  lane, not hardware executions performed by this reviewer.

## Anomaly Ledger

- Observed anomaly: The first B1 benchmark contained approximately 1.2 seconds
  of excess decode time, while the unchanged-server repeat did not.
  Evidence: `AUTODEBUG_b1_latency.md`, `AUTOFIX_b1_latency.md`,
  `b1_startup_before.json`, `b1_startup_after.json`,
  `b1_startup_after_tracker.json`, `b1_startup_benchmark_comparison.json`.
  Affected path: Startup, first prefill configuration, first decode admission.
  Control or comparison: Before/after reduced probes and immediate full-model
  first requests with matched repeats; all post-fix ITLs retained.
  Likely subsystem: Admission program compilation and trace recapture.
  Investigation performed: Independently verified eight new width-1 merge
  programs and request-time recapture before the fix, no request-time capture
  afterward, empty startup state, and exact token parity. Only
  `warmup_model_prefill` and its event-synchronization import changed. The final
  full first request has maximum ITL 19.105876 ms; repeat maximum is 16.051877 ms,
  both at interval zero. The original large stall is absent. Reduced timings
  do not account individually for every millisecond of the original full run.
  Resolution: fixed.

- Observed anomaly: Asynchronous scheduling could expose stale host tokens,
  positions, changed physical pages, or permuted request state.
  Evidence: `adapter_device_tracker_release.json`,
  `adapter_device_tracker_startup_final.json`, `serving_contract_device_v3.json`,
  `AUTODEBUG_serving_contract.md`, `AUTOFIX_serving_contract.md`.
  Affected path: Adapter admission, persistent feedback, page tables, slot remap.
  Control or comparison: Synchronous baseline versus two queued deferred steps,
  malicious stale host inputs, physical-page change across position 64, and
  permutation `[2,0,1]`.
  Likely subsystem: Ownership and exact movement of device state.
  Investigation performed: Read implementation and actual vectors/counters.
  Successive outputs differ; deferred results match synchronous results;
  positions/RoPE advance exactly; addresses and caller cache identity persist.
  Steady steps upload no token, position, RoPE, or page table. A real page change
  uploads once, preserves the old page, and writes the new page on all KV shards.
  Exact integer predicates avoid loss of large seed/token values; remap preserves
  state without NaN contamination. The failed wide-repeat L1 approach was
  replaced, and an invalid small physical-pool probe was corrected.
  Resolution: fixed.

- Observed anomaly: Fresh or host-resumed rows could have incorrect penalty
  history or inherited seeds, including unchanged-parameter transitions.
  Evidence: `AUTOFIX_prefill_penalty_admission.md`,
  `AUTOFIX_host_mode_transitions.md`, the host history/seed AutoDebug reports,
  `host_resume_seed_device_exact.json`, and their negative controls.
  Affected path: Prefill admission, optional host sampling, device resumption.
  Control or comparison: Uninterrupted device execution versus a greedy host
  detour, history restoration, and a fresh host-prefilled seed-99 request.
  Likely subsystem: Per-request sampler history and seed ownership.
  Investigation performed: Checked code and exact token/position/history/seed
  values across replicas. Continuing lanes retain their state; resumed penalized
  lanes restore history even with the same parameter key; fresh seeds initialize
  once. These controls do not claim stochastic host/device stream equivalence.
  Resolution: fixed.

- Observed anomaly: Combined penalties applied repetition scaling after
  additive frequency/presence changes.
  Evidence: `AUTODEBUG_combined_penalty_order.md`,
  `AUTOFIX_combined_penalty_order.md`, `combined_penalty_sampler_exact.json`,
  `combined_penalty_live.json`, and before/after CPU logs.
  Affected path: Shared canonical sampler.
  Control or comparison: Independent pinned host penalty implementation,
  captured TP4/B32 sampler scores, neutral and individual controls.
  Likely subsystem: Penalty operation order.
  Investigation performed: Verified repetition now scales original logits
  before frequency/presence subtraction. Independently reconstructed all
  recorded expected token choices, including negative logits and zero crossings;
  captured scores and live host/device token controls agree. No new sampler or
  precision policy was introduced.
  Resolution: fixed.

- Observed anomaly: Bad-word tests confused visible spellings with prohibited
  tokenizer sequences; a separate real multi-token history bug also existed.
  Evidence: `AUTOFIX_bad_words.md`, `AUTODEBUG_bad_words_history.md`,
  `bad_words_original_token_ids.json`, and exact before/after server controls.
  Affected path: Test assertions and plugin host bad-word processing.
  Control or comparison: Actual tokenizer variants, forbidden-subsequence
  negative controls, and matched banned/unbanned multi-token requests.
  Likely subsystem: Token-aware assertion and output-history forwarding.
  Investigation performed: Confirmed legal alternate tokenizations remain
  legal, genuinely forbidden sequences fail, and history forwarding changes
  the banned continuation while preserving its unbanned control.
  Resolution: fixed.

- Observed anomaly: Presence tests produced unchanged output at the tested
  penalties; first-character variety assertions hid distinct sampled tokens;
  a replacement helper initially rejected legitimate immediate EOS.
  Evidence: `AUTOFIX_presence_variation.md`,
  `AUTOFIX_first_token_variety.md`, raw presence and first-token probes, retained
  failures, targeted reruns, and the final full sampling log.
  Affected path: Shared sampling test sensitivity and response validation.
  Control or comparison: Raw score gaps, independent host sampler decisions,
  a sensitive natural-language fixture, actual first-token IDs, and 17 EOS/
  metadata/variety CPU controls.
  Likely subsystem: Test stimulus and assertion semantics.
  Investigation performed: The old presence stimulus's winner gap exceeded
  the penalty; new prompts cross real decision boundaries without weakening
  assertions. First-token checks now use API IDs, retaining seed/full-text/
  variety thresholds. Empty text is accepted only with a tokenizer-defined
  terminal EOS and stop finish reason; all-EOS batches still fail variety.
  Final result is 72 passed, one canonical skip in 329.38 seconds.
  Resolution: fixed for test bugs; controlled for the insensitive old stimulus.

- Observed anomaly: Greeting bans produced Chinese or off-topic continuations
  in two explicit constrained-sampling tests.
  Evidence: `bad_words_distribution_control.json`,
  `AUTODEBUG_bad_words_distribution.md`, and original responses.
  Affected path: Optional host-compatible bad-word sampling over thinking text.
  Control or comparison: Same-seed banned/unbanned requests and exact-prefix
  raw-distribution/mask controls.
  Likely subsystem: The requested token restriction.
  Investigation performed: Read the poor outputs and their normal unbanned
  controls. Both pairs share seven tokens and the same raw distribution before
  masking removes dominant ` hello`; the exact-prefix control removes at least
  97.40% observed probability mass. The actual pinned processor masks forbidden
  IDs and preserves allowed logits. Small prefill-versus-decode score differences
  in the prefix control are disclosed. This explains the branch without calling
  its eventual output good task performance or claiming arbitrary constrained
  outputs are coherent.
  Resolution: controlled.

- Observed anomaly: Synthetic nonaligned prompts had repetitive continuations,
  and the early evidence lacked direct numerical serving determinism controls.
  Evidence: `full_b32_verified_nonaligned.json`,
  `full_b32_startup_final_nonaligned_prose.json`, `logit_determinism_vllm.json`,
  `logit_determinism_standalone.json`, and their probe sources/raw logs.
  Affected path: Logical prompt length, repeated requests, batch positions.
  Control or comparison: Meaningful 131-/65-token prose, repeated/permuted API
  requests, and selected all-layer standalone logits on rows 0/1/31.
  Likely subsystem: Structural stimulus and missing numerical coverage.
  Investigation performed: Checked exact prompt IDs/usage and read the actual
  prose outputs. Reconstructed all nine repeated/permuted API signature matches
  and all eleven standalone/API token/chosen-logprob/top-20 matches. Independently
  hashed the external 43,707,383-byte tensor artifact and compared its nine
  complete-logit repeated/permuted rows byte-for-byte using standard-library ZIP
  storage analysis. Raw repeated-token prompts remain labeled stress probes,
  not the instruction-following quality suite.
  Resolution: controlled; numerical evidence gap closed.

- Observed anomaly: Haiku syllable errors and long thinking prefixes prevent a
  simple “all tasks correct” qualitative verdict.
  Evidence: Final `readiness_vllm/vllm_qualitative_outputs.json`,
  `qualitative_prompt_format.json`, `qualitative_final_review.md/json`,
  `standalone_haiku_512_v1/`, `hf_haiku_512_v1/`, `haiku_serving512.json`, and
  `haiku_standalone_serving_exact_comparison.json`.
  Affected path: Selected-model generation and finite output budget.
  Control or comparison: Original pinned chat template and rendered prompt IDs,
  six previous selected-TT controls, completed standalone/serving/HF haiku runs.
  Likely subsystem: Selected full-model task quality and thinking behavior.
  Investigation performed: Read all twelve final texts, not only verdicts;
  final JSON SHA256 is
  `9ab0691f922fb06551af7cd020f9102bb2b961ac32b862d61ddaafe6925c0140`.
  All six greedy texts exactly equal prior serving controls. Verified all 390
  raw selected standalone/serving haiku IDs including EOS, and read the better
  HF answer. No mechanical repetition, gibberish, language drift, or request
  contamination appears in the normal suite. Final documentation correctly
  distinguishes task limitations from serving consistency.
  Resolution: controlled.

- Observed anomaly: Early serving shutdown left devices difficult to reopen;
  logs also contained trace-allocation advisories and process-exit warnings.
  Evidence: `AUTODEBUG_serving_shutdown.md`, `trace_allocation_audit.md`,
  shutdown negative controls, recorded recovery/reopen logs,
  `full_b1_startup_final.server.log`, `full_b32_startup_final.server.log`, and
  `full_b32_startup_final_process_cleanup.json`.
  Affected path: Trace lifetime, worker teardown, mesh/fabric ownership.
  Control or comparison: Before/after shutdown host tests, repeated fresh
  server/standalone mesh opens without further resets, native allocation guards.
  Likely subsystem: Earlier no-op worker shutdown and warning classification.
  Investigation performed: Verified explicit idempotent teardown releases
  generator traces before mesh close, including exception cleanup. Final B1/B32
  wrappers exit zero, close markers are recorded, and the final process scan
  has no matching vLLM/EngineCore owners. The native tracker includes program
  cache allocations in its passing reduced probes. Final health lists four
  chips. Earlier warnings are preserved rather than deleted or generalized
  into an unsupported full-lifetime guarantee.
  Resolution: fixed for shutdown; controlled for documented advisories.

- Observed anomaly: Provisioning incompatibilities, a 2048 context scanner
  advisory, and raw-file formatting/size hooks could undermine reproducibility.
  Evidence: Environment AutoDebug/AutoFix, dependency/source locks,
  `runner_compatibility_report.md`, final context check, `raw_evidence_manifest.json`,
  final hook logs, and `plugin_final_format_scope.json`.
  Affected path: Task-local environment, evidence interpretation and packaging.
  Control or comparison: Preserved TTNN binary, pinned isolated serving
  interpreter, actual 262144 server manifests, original raw artifact bytes.
  Likely subsystem: Environment compatibility and artifact handling.
  Investigation performed: The 2048 advisory names an earlier CPU registry
  fixture, not a serving cap. Both repositories' final hooks pass. The only
  post-test test-source difference is one blank import line: reconstructed the
  exact tested hash from the final file and independently confirmed AST equality.
  Verified all 208 archive members against lengths, SHA256 values, and local
  originals; all are safe relative regular files. The 408576-byte final archive
  SHA256 is `765921f567399350b0d91b8a0d02309956e4d97bb5b9df134bae13ff023d30c8`.
  README restoration instructions expose the raw logs/JSON at their recorded
  paths in a fresh checkout without rewriting evidence bytes.
  Resolution: fixed or controlled as described.

## Scope Inspected

- Goal/skill paths: Original
  `/home/hous/dev/ornith-1.5-9b/state/multigoal/09-09-vllm.prompt.txt`;
  repository `.agents/skills/{vllm-integration,tt-device-usage,qualitative-check,stage-review}/SKILL.md`;
  applicable root/model instructions. No subagent was spawned by this reviewer.
- Artifact paths: `doc/vllm_integration/` README, work log, source/dependency
  locks, commands, all relevant AutoDebug/AutoFix reports, retained failures and
  negative controls, state/sampling probes, raw numerical and qualitative
  controls, primary before/after archives, final server logs, checks, cleanup,
  and the lossless evidence archive; `readiness_vllm/`; `doc/context_contract.json`;
  selected `doc/datatype_sweep/` policy and teacher-forcing evidence; prior
  `doc/optimized_full_model/` qualitative and canonical sampler comparison.
- Code paths: Model `tt/generator_vllm.py`, `tt/generator.py`, `tt/model.py`,
  precision wiring and new serving probes/tests; shared readiness runner and
  sampling penalties; sibling vLLM TT plugin `platform.py`, `worker.py`,
  `model_runner.py`, relevant async/cache/sampling input code, and modified
  sampling tests/helpers. Benchmark request/warmup and detailed-ITL source was
  checked in the pinned sibling vLLM tree.
- Provenance: Live tt-metal branch `hous/ornith-1.5-9b` at predecessor
  `85710be49fbcce5579aeb6d9571cd3bc1829d953` plus stage changes; sibling branch
  `hous/ornith-1.5-9b-vllm` based on
  `bf98d556bb46a5cda25fac540629251e7f474200` plus stage changes. Final B1/B32
  command manifests match all current runtime source hashes. Final adapter
  SHA256 is `881b38abae6ff278d3f66212e2ab4d8a462ce4ebd42153f65c5497b1d97ad324`.
  User-owned AGENTS.md edits are excluded from stage commits.
- Commands run: Read-only `rg`, `rg --files`, `cat`, `sed`, `head`, `tail`,
  `git status/diff`, and `git diff --check`; small Python standard-library
  JSON, hash, AST, ZIP/tar, byte-comparison, and metric calculations. No TTNN
  import, server request, hardware use, reset, profiler, or long test was run
  by this reviewer. Only this report was written.

## Residual Risk

- The chosen precision policy is inherited and actually consumed by the
  measured runtime, including layer exceptions, BFP8 KV, FP32 recurrence and
  native CCL. This review does not replace its predecessor accuracy evidence
  or establish HF-equivalent free-running text quality.
- The primary measured path uses canonical split sampling, nonblocking trace
  replay, persistent device token feedback, and minimal token readback. Its
  final lifetime counters show 255 model/sampler replays and zero host decodes.
  The optional compatibility path is explicit. The 87.115 tokens/s/user
  predecessor teacher-forcing result is used only as a workload-qualified
  decoder-cost lower bound, not as an interchangeable serving metric.
- Coverage is bounded to the recorded serving workloads and controls. New
  workload shapes or sampling configurations may compile/recapture as guarded
  by the existing runtime. No unresolved concrete correctness or avoidable
  decode-overhead finding remains in the reviewed measured path.
