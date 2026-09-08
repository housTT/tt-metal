# Stage Review

Verdict: clean-pass

Independent review date: 2026-09-08. Stage 09-09, vLLM integration of
`google/gemma-4-26B-A4B-it`, checkpoint
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

This review inspected the live staged closure on tt-metal HEAD
`c718255087bc62907ac2882fdfe3fc7eda613c42`, including implementation
`40569292be814f72c62ef30fd06c8b8245957517`, and the clean vLLM checkout at
`2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`. It did not inherit an earlier
review verdict. No server, device open, reset, profiler, or hardware test was
run by this reviewer.

## Required Work

None. No unresolved blocking or non-blocking finding remains in the reviewed
stage. The coordinating agent must perform the normal post-review local
checkpoint of the staged closure and this report, and record that SHA; this is
the explicitly ordered administrative step after clean review, not an unrun
serving gate. Existing implementation SHAs are recorded in `work_log.md`.

## Other Concerns

None requiring remediation. The recovered tokenizer warning found during this
review is now accurately classified in the staged runtime audit, README, and
work log. Its limited evidentiary scope is retained below.

## Contract and Evidence Checks

Paths below are relative to
`models/autoports/google_gemma_4_26b_a4b_it/` unless stated otherwise.

| Requirement | Independently checked evidence and result |
| --- | --- |
| Separate P150/P150x2/P150x4 serving | Decompressed every final `readiness_vllm/<profile>/server.log.xz`. Startup records show 1x1, 1x2, and 1x4 model meshes, respectively; TP2 explicitly creates its 1x2 submesh from a 2x2 physical parent. All run the registered autoport with async scheduling, trace mode `all`, device sampling `all`, block size 64, and 32 sequences. |
| Correct context and valid non-aligned input | Startup and `/v1/models` agree with the profile contract: 50,624 / 262,144 / 262,144. Every feature artifact contains a successful 29-token prompt. The context contract's selected-policy hash matches the current precision file, and all three referenced complete-model boundary artifacts exist and pass at 50,623 / 262,143 / 262,143 logical tokens. P150 retains the prior physical-capacity reduction; serving introduces no additional reduction. |
| Adapter delegates to the full model | Read all of `tt/generator_vllm.py` and the generator's state, prefill, decode, trace, sampling, and construction paths. Prefill/decode and terminal sampling delegate to `Gemma4Generator`. The adapter contains interface translation and scheduler/cache state handling, without a replacement model or sampler. |
| Selected precision | `initialize_vllm_model` defaults to `doc/datatype_sweep/selected_precision_config.json`; `build_generator`, `Gemma4FullModel`, `profile_policy`, `layer_policy`, and `decoder_kwargs` consume it. Checked BF16 activation/residual/CCL/KV, BFP8 terminal head and default expert weights, profile-specific BFP4 exceptions, and layer-specific attention/dense weights and compute fidelity propagation. The explicit environment override is not used by the recorded final commands. |
| Canonical traced device sampling | The device path uses `Sampling1D`, local top-32 plus semantic k=1 for greedy, split model/sampling traces, and `tt_out_tok=token_input`. Both traces replay with `blocking=False`. Persistent device current position advances once in the model trace; sampled tokens remain device feedback. No host argmax, full-logit readback, or Python token feedback occurs on this measured path. |
| Explicit optional host compatibility | Plugin eligibility routes unsupported penalties, seeds, stochastic top-k above 32, structured/logit-processor modes, and requested logprobs to the existing host sampler. The adapter delegates the corresponding logits gather to the full generator. This diagnostic/compatibility path is labeled separately and is not the device qualitative or benchmark path. The shared formatter converts temperature to the reciprocal expected by the sampling kernel and makes greedy rows k=1/p=0. |
| vLLM cache ownership and hybrid attention | Construction passes `create_kv_cache=False`. Plugin `_build_per_layer_specs` and `_allocate_kv_caches` call the adapter's per-layer allocator. The exact returned cache object is required on both forward paths. Sliding/full specs retain different block geometries; shared allocations key on tensor index plus physical volume, separating TP4's wider full-attention storage. Decoder cache-view overrides consistently reinterpret shared buffers on cache writes and attention reads. |
| Async split and mutable scheduler state | Read plugin async submission/finalization and adapter `read_decode_output`/`process_decode_output_host`. Nonblocking token read is paired with an event before host formatting. Adapter host tests prove stale host arguments do not force an active-mask refresh and changed page tables refresh once into existing storage. The separate reduced generator device artifact proves token aliasing, stale position rejection, device increments, stable page addresses, changed/unchanged refreshes, and inactive rows. Live overlap exercises the composed serving path, including a page crossing and slot departure, on every profile. |
| Registration and parsers | `register_tt_models()` selects the autoport with `TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport` and registers Gemma text/unified architecture aliases. Runtime module paths and startup records agree. Every feature artifact contains parsed `get_weather` arguments exactly `{"city":"Paris"}` and nonempty parsed reasoning with final answer `2 + 3 = 5`. |
| Full sampling and concurrency | All three retained logs collect 73 cases and finish with 72 passed / 1 skipped. The skip is the suite's documented all-vocabulary chat-logprobs request exceeding the server cap. There are no xfails or failures. Each CI burst completes 32/32, its raw peak concurrency is 32, and every server lifetime independently logs `Running: 32 reqs`. |
| Qualitative and controls | Read all 36 final greedy/sampled completions, all isolated/overlapped texts, and decoded the retained HF and standalone-TT control token IDs with the exact local checkpoint tokenizer. All 18 rendered prompts and prompt-token lists match the control. Control/source/output hashes agree; all three degeneracy artifacts have no findings and exit code 0. Detailed judgment appears below. |
| Logit determinism | Recomputed the direct sequential, profile-local standalone, concurrent, and graph-shape comparisons from retained raw compact responses. B1 standalone selected-logprob deltas are zero on every profile; repeated and same-group cross-position top-10/logprobs agree exactly. Oracle hashes match. Independently checked all 160 rows across TP4's four homogeneous and one mixed B32 controls: same-group selected tokens/top-10/logprobs are identical across row and composition. |
| Benchmarks and reporting | Recomputed every normalized TTFT/TPOT/ITL/E2EL mean/P50/P99, completion/token count, throughput, and TPOT-derived t/s/u from the six raw result files. All match. README leads with single-user 128/128/1 metrics and presents 100/100/32 burst metrics separately. Teacher-forcing results are expressly a different regime, not an equivalent serving benchmark. |
| Runtime, recovery, and cleanup | Final logs show fallback exceptions enabled, zero HTTP 500/fatal failures, application shutdown, and completed device close/destructor. The cleanup audit records no remaining API/EngineCore/readiness processes and four visible devices. Worker cleanup closes the submesh before its physical parent and does not read the device profiler. Recorded pre-server heartbeat recovery includes bounded reset/list and successful exact mesh smoke before the three successful lifetimes. |
| Evidence preservation and local scope | All 39 hashes in `unified_gate_manifest.json` match, and all corresponding files are staged/tracked, including the nine plain client logs otherwise ignored by Git. Benchmark embedded times and API feature creation times lie inside their final server lifetimes. The current generic archives are complete main-gate lifetimes, resolving the earlier determinism-only overwrite. vLLM is clean; no push was performed by this review. |

### Final lifetime and metric cross-check

All lifetimes are on 2026-09-08 UTC. Each contains exactly 67 chat and 1,514
completion POSTs: 1,580 HTTP 200 responses, one expected HTTP 400 from the
all-vocabulary-logprobs skip, and no HTTP 500 response.

| Profile | Lifetime / API PID | Full sampling | Primary 128/128/1 TTFT P50 | Primary TPOT-derived t/s/u | Secondary 100/100/32 throughput |
| --- | --- | --- | ---: | ---: | ---: |
| P150 | 17:49:09–18:04:15 / 1696137 | 72p / 1s, 592.78 s | 235.121 ms | 36.206 | 100.548 tok/s |
| P150x2 | 18:06:03–18:21:27 / 1699866 | 72p / 1s, 626.31 s | 235.547 ms | 43.958 | 122.205 tok/s |
| P150x4 | 18:23:11–18:39:39 / 1703317 | 72p / 1s, 693.68 s | 203.296 ms | 48.512 | 129.489 tok/s |

Primary concurrency is one; secondary client concurrency is unbounded and
achieves 32. Both workloads explicitly use temperature 0 and ignore EOS.
Primary raw result times are 18:03:31 / 18:20:45 / 18:38:56; secondary raw
result times are 18:04:10 / 18:21:18 / 18:39:29. Raw and normalized metric
files and their retained client logs are mutually consistent.

### Independent qualitative judgment

The exact checkpoint has a nonempty chat template and loads as
`GemmaTokenizer`. The shared suite is sent as chat user messages, with
generation-prompt rendering retained. Greedy uses temperature 0/top-k 1;
sampled uses temperature 0.7/top-p 0.9/top-k 32. Both use a 256-token cap.
The HF and earlier TT controls are greedy and capped at 64 tokens, so their
comparison establishes prompt-correct coherent prefixes and task alignment,
not exact full-length or sampled-text identity.

| Prompt | Judgment after reading both completions on every profile |
| --- | --- |
| `machine_learning_haiku` | Coherent three-line poems about data, patterns, and learning. P150 greedy matches the HF poem, while P150x2/x4 match the earlier TT poem. Identical short greedy/sampled answers are not repetition loops. |
| `supervised_unsupervised` | Correct teacher/labeled-fruit framing. Some answers reach the unsupervised heading or begin its explanation; others stop beforehand at the cap. Evidence does not claim complete two-part instruction coverage. |
| `inventor_story` | Coherent fantasy continuations about Elara/Elian and brass, starlight, or celestial inventions. Restating the supplied story opening is appropriate continuation behavior, not uncontrolled prompt echo. Creative variation is comparable to the HF/TT controls. |
| `thermodynamics_laws` | Correct first/second-law framing with ordinary variation in explaining the zeroth law. Outputs remain coherent through the cap, often during the second law; no claim is made that all three laws were completed. |
| `french_translation` | Correct formal and informal French with appropriate English explanatory framing. Requested French is not wrong-language drift. |
| `fibonacci_function` | Valid iterative sequence or nth-term code, sensible examples, and coherent explanation. Longer alternative sections terminate at the cap; TP4 sampled includes a complete nth-term implementation before its unfinished second implementation. |

All 36 outputs pass coherence, topic, absence of mechanical repetition,
absence of gibberish, appropriate language, and absence of visible request
contamination. The overlap outputs likewise remain coherent, retain their
topic and initial prefix, and return exactly `Brass telescope` for the short
request. Their later long-form wording is not bit-identical across the
B1/multi-request graph transition; that difference is explicitly controlled
below rather than hidden by the pass label.

## Hard-Check Gaps

No unresolved gap blocks this stage. Evidence is composed across host adapter
tests, direct reduced generator state tests, and full-model live serving
checks. `mutable_decode_state.json` correctly identifies itself as standalone
generator evidence; it is not relabeled as direct hardware execution of the
adapter. The full-stack final serving results establish the public integration
contract. No new profiler or runtime-counter artifact is required to replace
this already connected evidence.

## Anomaly Ledger

- Observed anomaly: earlier generic logs preserved only determinism lifetimes,
  and the nine raw client logs were ignored/uncommitted.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, and the final manifest/archive files.
  Affected path: serving evidence provenance, configuration, and cleanup audit.
  Control or comparison: final archive decomposition and 39 SHA256 checks.
  Investigation performed: verified each complete final lifetime, request
  counts, metric dates, feature timestamps, shutdown, and Git retention.
  Resolution: fixed by the three complete final reruns and staged raw logs.

- Observed anomaly: B1/B32 distributions and long overlap completions differ.
  Evidence: per-profile `logit_determinism.json`, TP4 standalone oracle, and
  all three `async_overlap_state_test.json` files.
  Affected path: full-model batch-dependent optimized decode graphs.
  Control or comparison: four real-weight standalone prompt groups in five
  B32 layouts, profile-local B1 oracles, same-graph repeats, and live outputs.
  Investigation performed: rederived comparisons and checked all 160 B32 row
  entries. Same-group row/composition results are exact; selected tokens remain
  equal to the corresponding B1 maxima. The worst raw/centered cosine is
  0.896793/0.958713 and probability total variation is 0.270572.
  Resolution: controlled as measured standalone graph-shape sensitivity, not
  claimed as general numerical equivalence. No new serving-specific corruption
  is established; live quality and full sampling pass.

- Observed anomaly: one active-trace allocation warning in each final server.
  Evidence: final server logs, adapter trace release/recapture code, generator
  capture scopes, and TP4 `mutable_decode_state.json`.
  Affected path: trace allocation lifetime and persistent decode state.
  Control or comparison: stable token alias/address, unchanged page addresses,
  exact changed-page refresh, inactive-row and stale-input assertions, then live
  overlap and full serving gates.
  Investigation performed: traced device feedback and state ownership through
  adapter/generator/plugin. The allocation warning is once-per-thread and its
  count is not proof of safety.
  Resolution: controlled by explicit lifetime and state evidence, consistent
  with the corrected stage documentation.

- Observed anomaly: one TP4 invalid-prefix detokenization warning at
  18:33:59 UTC, request `cmpl-b675f7e6ff20f971-0-84e2e2dc`.
  Evidence: decompressed TP4 server line 3626 and
  `/home/hous/dev/vllm/vllm/v1/engine/detokenizer.py::_protected_step`.
  Affected path: framework per-request UTF-8/DecodeStream output decoding
  during the full sampling suite.
  Control or comparison: the exact guarded exception branch resets the stream
  and retries the same token; the lifetime continues with successful requests,
  full sampling finishes 72p/1s, and subsequent qualitative/overlap gates pass.
  Investigation performed: inspected the surrounding log, exception handling,
  test results, and absence of fatal/500 responses. The coordinator added the
  precise classification during this review; the revised audit/README/work-log
  wording was independently reread.
  Resolution: controlled as successful framework recovery. The affected stress
  request's text/token sequence was not retained and was not manually judged;
  no stronger claim about that request is made.

- Observed anomaly: TP2 computed/provided matmul memory-config warnings;
  multi-device deprecated all-gather, packet-size, and L1-semaphore warnings;
  nanobind diagnostics on interpreter exit.
  Evidence: final logs and `matmul_device_operation.cpp` validation branch.
  Affected path: inherited TT operation configuration and runtime diagnostics.
  Control or comparison: matmul verifies layout/buffer compatibility and uses
  its computed shard geometry; real-weight numerical checks and serving gates
  pass. Clean device destruction follows the exit diagnostics.
  Investigation performed: grouped repeated warning messages and read their
  source/contexts. These are not exceptions, hidden CPU computation, or proof
  of an allocator/sampling failure. The installed vLLM version string names
  its older build metadata, while resolved imports and tested worker source
  point to the current clean plugin checkout.
  Resolution: controlled runtime/configuration diagnostics; no measured new
  serving-specific overhead or correctness regression is established.

- Observed anomaly: active-Ethernet heartbeat timeout before a serving launch.
  Evidence: `hardware_recovery_20260908.json`, historical failed-open archive,
  cleanup audit, and subsequent final server lifetimes.
  Affected path: infrastructure mesh initialization before model construction.
  Control or comparison: bounded reset/list and successful exact 2x2 mesh smoke.
  Investigation performed: checked the recorded recovery sequence and all
  subsequent profile opens/closes; no hardware action was performed here.
  Resolution: recovered. The final audit records all four devices visible and
  no remaining serving processes.

## Scope Inspected

- Goal/skills: the complete original `09-09-vllm.prompt.txt` and complete
  `stage-review`, `vllm-integration`, `tt-device-usage`, `qualitative-check`,
  and `autofix` skill files under `/home/hous/dev/tt-metal/.agents/skills/`.
- Artifacts: stage README/work log/AutoDebug/AutoFix; context and precision
  contracts and referenced boundary results; unified manifest and cleanup/
  recovery audits; final profile feature, sampling, qualitative, degeneracy,
  overlap, determinism, raw/normalized benchmark/log files; decompressed final
  server lifetimes; standalone oracle/JUnit controls; earlier prompt-correct
  HF/TT control output and assessment.
- Code: `tt/generator_vllm.py`, relevant full-generator/model/precision/decoder
  cache-view paths, adapter/oracle/feature/overlap tests, shared sampling and
  readiness runner; external plugin registration, worker, cache allocation,
  input sampling eligibility, async submission/finalization, parser tests,
  and the framework detokenizer recovery branch.
- Read-only checks: Git status/log/show/diff and staged whitespace check;
  `sed`/`rg`/`jq`; lossless XZ decompression; small host scripts checking hashes,
  retained paths, raw metrics, timestamps, prompt format, and numeric control
  consistency. Local tokenizer loading used `local_files_only=True` and the
  exact checkpoint snapshot. No network or hardware dependency was needed.

Host verification run by this reviewer:

```bash
# From /home/hous/dev/tt-metal
env PYTHONDONTWRITEBYTECODE=1 python_env/bin/pytest -q --noconftest \
  -p no:cacheprovider \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_adapter_contract.py \
  models/common/readiness_check/test_run_vllm_server.py
# 21 passed, 3 warnings in 3.34 seconds

# From /home/hous/dev/vllm
env PYTHONDONTWRITEBYTECODE=1 /home/hous/dev/tt-metal/python_env/bin/pytest \
  -q --noconftest -p no:cacheprovider plugins/vllm-tt-plugin/tests \
  --ignore=plugins/vllm-tt-plugin/tests/tt
# 118 passed, 16 warnings in 5.17 seconds

git diff --cached --check
# passed
```

The host tests use mocks and no device fixtures. Pytest emitted temporary test
files and the repository's default ignored
`generated/test_reports/most_recent_tests.xml`; no implementation or retained
stage artifact was modified by these tests. This review's only authored file
is this report. The stage changes are Python/evidence changes, so no C++ or
CMake build is required.

## Residual Risk

- This is evidence-backed integration review, not a new hardware run or a
  customer release/evaluation certification. It does not add a long-duration
  stress or all-input correctness claim.
- Supported capabilities are text-only; image/video and prefix caching are
  excluded. P150's existing reduced context and the optional host compatibility
  modes remain explicit.
- The measured B1/B32 distribution differences are real; selected-token and
  same-graph determinism controls do not imply general distribution equality.
  P150's standalone full-vocabulary B32 probe is memory-limited, while its live
  32-request serving coverage passes.
- Qualitative answers are capped. Incomplete trailing sections at that cap are
  not evidence of complete instruction fulfillment. The single recovered
  tokenizer stress request has no retained text for independent quality review.
- The post-clean-pass closure commit SHA cannot exist before this review; it
  must be appended by the stage owner after committing locally. Never push as
  part of autonomous stage closure.
