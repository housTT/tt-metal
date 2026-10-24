# Stage Review

Verdict: clean-pass

Independent review of stage 10, optimized-vLLM, for `google/gemma-4-26B-A4B-it`. This final verdict includes re-review of the targeted qualitative control and refreshed shutdown/reopen evidence. The runtime candidate reviewed is adapter SHA256 `649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325`, against original tt-metal HEAD `6eb0427423392d7c6a7f87a511be892b8bf677ae` and sibling vLLM HEAD `2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`. Implementation changes are uncommitted and stage-owned; checkpoint commits and the SHA ledger follow this clean review, as required by the stage-review skill.

## Required Work

None. The initial P2 request for a same-prefix thermodynamics control is closed by three independently inspected original/current continuation pairs and the refreshed lifecycle evidence described below. No runtime change was warranted by that control.

## Other Concerns

- The inherited TP2 matmul warning is classified, not a demonstrated corrective-copy or CPU-fallback problem. Requested 22-core output metadata conflicts with an 11-core computed output; `compute_output_specs` allocates the latter directly. Matched benchmark prefixes contain 480 instances on each side. The larger host-compatibility sampling suite produces a large warning log. A narrow metadata cleanup may be useful, but there is no measured candidate or rejected device optimization here, and no evidence this warning explains a remaining material serving/device performance gap.
- The API logit diagnostics intentionally use host readback. Full-vocabulary logprobs exceed the configured cap and account for the single expected sampling skip per profile. Neither diagnostic path is the primary `sample_on_device=all` performance path.
- All 36 retained qualitative completions were read. The haiku and French answers are coherent, and each Fibonacci answer contains a complete correct first implementation. Longer learning explanations, stories, thermodynamics answers, and introductions to additional code approaches stop under the shared 256-token output budget. Greedy text and rendered prompt/token IDs match each prior profile exactly for all 18 greedy responses. Complete long answers are not established by these capped outputs.

## Hard-Check Gaps

- The mandatory C++ wrapper build was attempted and failed on Docker socket permissions. Existing logs establish real kernel JIT compilation and targeted watcher execution, not a successful full repository build. This environment limitation is explicitly documented, consistent with the repository's unavailable-build rule.
- The direct trace-lifetime probe uses two actual layer kinds plus the real terminal; its 54 cold/warm/forced-control cases establish the changed state and trace contracts. Full 30-layer serving is separately exercised by the three real plugin servers. The focused probe is not a full-context or serving-performance measurement.
- Context overlap is supported by a source-derived conservative bound in `context_lifetime_audit.md`, independently calculated for all three actual serving configurations. It is not a newly measured allocator high-water mark or proof of identical peak memory. The bound uses the observed 2048-token chunked-prefill budget; it does not extend to an arbitrary larger scheduler budget.
- Primary performance is one timed 128/128 request after one explicit warmup per side/profile. Its request-level TTFT and TPOT percentiles therefore coincide. CI 100/100/32 is secondary capacity evidence and a single B1 warmup does not precompile every later batch shape. No repeatability confidence interval or universal load claim is supported.
- No serving profiler was used, as explicitly required. The separately measured full-model token-out control is comparable work with different host boundaries and positions; the small subtraction from serving TPOT is not a measured device/host decomposition.

## Anomaly Ledger

- Observed anomaly: Initial primary measurements included cold capture overhead because the benchmark default performed no warmup.
  Evidence: Preserved `before` and P150 `after` diagnostics, `before_warmed_summary.json`, `baseline_source_swap.json`, and all six warmed run manifests/benchmark logs.
  Affected path: Headline before/after serving comparison.
  Control or comparison: Original adapter bytes restored temporarily for `before_warmed`; fixed worker, router, benchmark module, server configuration, generation configuration, shapes, sampling and explicit one-request warmup match the candidate.
  Likely subsystem: Benchmark setup and trace capture.
  Investigation performed: Reviewed runner/source-swap restoration and warmup failure handling; independently matched all 12 primary/CI summary rows to raw JSON.
  Resolution: fixed. Cold measurements remain diagnostic and are excluded from the headline.

- Observed anomaly: Trace retirement at every prefill and greedy requests' varying seed keys prevented reuse; prefill compilation also creates a real retained-trace hazard.
  Evidence: Adapter diff, `trace_contract_summary.json`, `trace_reuse_probe/final_router_fix/trace_reuse_tp{1,2,4}.json`, allocation-tracking/watcher logs, and regression source.
  Affected path: Retained model/sampler trace and scheduler-owned inputs.
  Control or comparison: Each profile has 18 cold, warmed and forced-retirement cases across nonaligned prompts and active-slot changes. Warmed later requests retain trace IDs/device addresses; changed and unchanged per-layer page tables and stale host token/current-position inputs are checked; reference logits match exactly.
  Likely subsystem: Adapter trace keys and compilation lifetime.
  Investigation performed: Inspected program-cache-growth retirement, batch/remap boundaries, input refresh, shared page-table aliases and stable device buffers; inspected the generator's nonblocking split replay and token feedback.
  Resolution: fixed. Cold program-cache growth still retires a trace before subsequent decode; warmed reuse does not depend on stale inputs.

- Observed anomaly: Real fabric shutdown/reopen failed with retained NoC packet tags; separate worker cleanup was also incomplete.
  Evidence: `kernel_cleanup_experiment.md`, `worker_cleanup_experiment.md`, AUTODEBUG/AUTOTRIAGE/AUTOFIX reports, failed/fixed router probe logs, raw full-model server shutdown logs and `cleanup_evidence.json`.
  Affected path: Worker mesh ownership and fabric-router teardown.
  Control or comparison: The router clears packet tags only after local/neighbor drain and ERISC synchronization, before termination. Fixed real serving completes worker/UMD shutdown; a subsequent 2x2 parent and 1x2 submesh reopen succeeds without a reset.
  Likely subsystem: Fabric teardown and explicit worker resource ownership.
  Investigation performed: Inspected the three-line router change, packet-tag helper and firmware postcondition, worker idempotent shutdown and parent/submesh close order, host regression evidence and hardware logs. The combined final lifecycle is verified; the original failure is not attributed solely to the worker change.
  Resolution: fixed for the exercised topologies. Full wrapper build remains unverified for the stated environment reason.

- Observed anomaly: Enabling all watcher instrumentation exceeded active ERISC instruction memory.
  Evidence: `kernel_cleanup_experiment.md` and targeted watcher build/run logs.
  Affected path: Instrumented fabric-router kernel.
  Control or comparison: NOINLINE plus disabling WAYPOINT permits the kernel to build/run; assertions, NoC and Ethernet checks remain enabled.
  Likely subsystem: Finite ERISC instruction capacity.
  Investigation performed: Read the concrete build-size failure and subsequent successful targeted watcher runs.
  Resolution: controlled. This is a specifically reduced instrumentation run, not an all-instrumentation watcher claim.

- Observed anomaly: Trace-owned tensors now overlap prefill, so unchanged shapes alone do not preserve the former peak-memory proof.
  Evidence: `context_lifetime_audit.md`, `../context_contract.json`, profile-specific capacity JSON, actual before-warmed server/cache allocation settings, generator/model ownership and attention/MoE allocation source.
  Affected path: Maximum-context serving capacity.
  Control or comparison: The audit separately replaces standalone KV with actual serving KV, accounts for BF16 embedding and the rounded trace region, retains 32 MiB for trace-owned outputs, budgets prefill and concatenation transients, and preserves allocator reserve for each profile.
  Likely subsystem: Allocation lifetime and scheduler chunk size.
  Investigation performed: Independently derived per-profile bounds. Remaining conservative headroom is 579,855,360 / 10,239,876,608 / 13,704,827,904 bytes for P150/P150x2/P150x4 under the observed serving budget.
  Resolution: controlled by a source bound, not measured high-water. Advertised supported limits remain 50,624 / 262,144 / 262,144.

- Observed anomaly: Exact isolated-versus-overlapped long-request text differs; TP2's final overlap tail also differs from its prior integration run.
  Evidence: Each profile's actual `async_overlap_state_test.json`, prior artifacts, `qualitative_P150x2_review.json`, and the runner's wall-clock 0.15-second request injection.
  Affected path: Async scheduling and batch-shape transitions.
  Control or comparison: TP4's entire artifact equals its prior artifact. TP2 differs only in the overlapped long completion and its hash; its isolated long and exact short response are unchanged. All profile-local logit result values equal the prior integration after two artifact-path normalizations. The short request ends first, the long response crosses a 64-token page boundary, and observed text remains coherent without contamination or doubled tokens.
  Likely subsystem: Existing graph-shape numerical sensitivity and time-dependent admission position.
  Investigation performed: Read both long texts and short responses directly, compared nested numeric artifacts, and inspected the admission timing and direct persistent-input/page-table controls.
  Resolution: controlled for the stage's async/state regression scope. Exact lexical equivalence across batch-shape transitions is not claimed.

- Observed anomaly: B1 and padded-B32 full distributions differ even though selected tokens match.
  Evidence: Raw `logit_determinism.json` for all profiles and retained standalone oracle artifacts.
  Affected path: Numeric diagnostic graph-shape comparison.
  Control or comparison: Same-graph and profile-local standalone comparisons pass; current results equal prior profile results. The embedded TP4 B1/B32 standalone control is explicitly inherited in TP1/TP2 reports and is not evidence of their distribution equivalence.
  Likely subsystem: Existing shape-dependent TT matmul/distribution sensitivity.
  Investigation performed: Read the thresholds and diagnostic classification; independently compared current/prior nested values and selected tokens.
  Resolution: controlled. No exact distribution-equivalence assertion is made.

- Observed anomaly: TP2 matmul reports requested/computed output-memory-config mismatch, and tilize reports legacy sharded factory output selection.
  Evidence: `runtime_warning_audit.md`/JSON, decompressed raw server logs, decoder configuration and matmul/tilize output-allocation source.
  Affected path: Existing projection metadata and eager host-compatibility sampling dispatch.
  Control or comparison: Before/after matched benchmark prefixes contain the same 480 matmul and three tilize warnings. Computed matmul output is allocated directly; there is no subsequent correction copy demonstrated by the warning.
  Likely subsystem: Inherited requested output metadata.
  Investigation performed: Checked the 22-core request versus 11-core rank output derivation, shared binary-multiply configuration and output allocation path. The larger sampling log is distinct from the matched benchmark prefix.
  Resolution: controlled. Optional metadata cleanup is unmeasured future work, not a rejected optimized kernel family or an established fallback.

- Observed anomaly: Nanobind reports instance/type/function/keep-alive leaks at interpreter exit.
  Evidence: Raw P150 before/after server logs and `cleanup_evidence.json`.
  Affected path: Interpreter binding teardown.
  Control or comparison: EngineCore counts match at 1493 instances, 48 keep-alive records, 986 types and 4507 functions; API counts also match. Explicit mesh close, UMD close, process disappearance and immediate device reopen succeed.
  Likely subsystem: Inherited interpreter binding lifetime.
  Investigation performed: Compared counters and real close/reopen evidence; verified final compressed server logs contain shutdown markers and no traceback/assertion failure.
  Resolution: controlled as an inherited diagnostic. It is not proof that all bindings are leak-free.

- Observed anomaly: Context guard rejected an already-supported P150 cap inside aggregate stage reports.
  Evidence: `.agents/scripts/check_context_contract.py` and test diff, `AUTOFIX_context_reports.md`, before/after regression logs and `final_stage_check_after.log`.
  Affected path: Stage report validation, not served context.
  Control or comparison: The existing explicit top-level `profiles` mapping semantics now apply at the three existing stage-report roots as well as the readiness root. Global values, unknown profiles, other-profile floors, nested documents and reductions below the recorded P150 cap remain rejected.
  Likely subsystem: Incomplete report-root recognition.
  Investigation performed: Read the scanner and regression tests; independently ran all 13 stdlib tests successfully.
  Resolution: fixed without lowering a served limit or broadly exempting documents.

- Observed anomaly: TP4 sampled thermodynamics adds an inaccurate universal efficiency claim.
  Evidence: Row 3 of the retained TP4 qualitative output, `qualitative_P150x4_review.json`, `AUTODEBUG_thermodynamics.md`, `AUTOFIX_thermodynamics.md`, and `thermodynamics_control/{manifest,original_0,original_1,original_2,optimized_0,optimized_1,optimized_2}.json`.
  Affected path: Positive-temperature generated explanation.
  Control or comparison: The original and optimized adapters received three identical serial continuation requests on fresh TP4 servers, using the actual rendered chat prompt plus the retained prefix immediately before the claim. Temperature 0.7, top-p 0.9 and top-k 32 remain on the supported device-sampling path; there is no explicit seed or requested logprobs. Each request has 247 prompt tokens and 256 completion tokens. Every paired choice object, completion text, usage and finish reason matches exactly. All six continuations reproduce the substantive claim that energy conversion always wastes heat, and also contain the same absolute-zero molecular-motion simplification.
  Likely subsystem: Existing TT-serving explanatory imprecision conditional on this prefix; HF attribution is not established.
  Investigation performed: Requested the missing control, read all six raw completions through their retained ends, verified their hashes and identical paired request bodies, inspected the runner and source restoration, and checked both zero runner exits and worker/UMD/API shutdown markers. The subsequent no-reset 2x2-parent/1x2-submesh reopen succeeds; the device list shows four P300C chips, and the process-check artifact is empty with exit 1. The candidate adapter SHA is restored.
  Resolution: controlled. The initial P2 is closed. The physics wording is not made correct, and neither original full-trajectory stochastic equivalence nor HF causation is inferred. The original qualitative output remains preserved. Repeated sampled requests use different seed keys, so this is warmed program/cache evidence, not reuse of one stochastic trace across request boundaries.

## Scope Inspected

- Goal/skill paths: Original stage-10 contract supplied by the stage owner; `.agents/skills/{stage-review,vllm-integration,optimize,tt-device-usage,tt-enable-tracing,qualitative-check}/SKILL.md`; repository build instructions.
- Artifact paths: This directory's README, work log, performance summary, serving/inherited device contracts, context lifetime audit, trace summary, page-table audit, before-warmed and source-swap records, cleanup experiments/evidence, runtime warning audit, diagnostic reports, lint/host/check logs; all three profiles' before/after warmed manifests and raw primary/CI benchmark JSON; P150 `after` and TP2/TP4 `after_warmed` full-gate logs and actual generated outputs; matching prior integration artifacts; selected datatype-sweep full-model token-out JSON; profile capacity artifacts; same-template shared HF/TT qualitative controls; all six targeted thermodynamics continuation responses, their manifest and both server logs, and the final control-specific reopen/device-list/process-check logs.
- Code paths: Current and baseline `tt/generator_vllm.py`; unchanged `tt/generator.py`, `tt/model.py`, decoder/terminal/allocation dependencies; adapter-contract and trace-reuse tests; fabric-router teardown/helper/postcondition source; context checker/tests; sibling vLLM worker/parent-mesh tests and benchmark warmup handling; stage runners, sampling/feature/async/logit checks and summary generation.
- Commands run: Read-only `git diff`, `git status`, `git show`, `rg`, `cat`, `sed` and source/artifact inspection; `python3 .agents/tests/test_check_context_contract.py` (13 passed); stdlib JSON comparison and arithmetic scripts; SHA256 verification of all four compressed artifacts against their uncompressed index entries; `git diff --check` in both repositories (passed). Independently verified all 12 primary/CI rows' TTFT/TPOT/ITL/E2EL mean/median/P99 and throughput against raw JSON, all six manifests' shared settings/source hashes, all 18 prior/current greedy text and prompt/token-ID pairs, and all three targeted continuation pairs' exact choices/usage and request bodies. No reviewer hardware access, profiling, TTNN import, server launch or vLLM/pytest run occurred. Only this report and the explicitly authorized context audit were written by the reviewer.

## Residual Risk

- Hardware evidence is on physical P300C Blackhole 1/2/4-chip proxies, honestly labeled as established P150 profiles. It does not establish physical P150 card results.
- Warmed primary decode rates are 37.244 / 46.443 / 52.058 t/s/u, versus 36.222 / 43.957 / 48.201 before and selected full-model token-out controls of 37.256 / 46.531 / 52.202. CI capacity remains essentially unchanged; no CI speedup or same-run device/host timing decomposition is claimed.
- The conservative context bound includes an explicit retained-trace reserve and profile-specific serving caches, but does not replace long-running fragmentation/high-water characterization.
- Full repository C++ build remains unverified because the required Docker wrapper was inaccessible; real JIT/watcher and serving lifecycle results have narrower scope.
- This review is not a release evaluation, multimodal certification, all-context stress test, complete-answer evaluation or guarantee of factual correctness for stochastic text. The thermodynamics limitation is controlled against baseline TT serving at the retained prefix; factual error rates and canonical HF attribution remain unmeasured.
