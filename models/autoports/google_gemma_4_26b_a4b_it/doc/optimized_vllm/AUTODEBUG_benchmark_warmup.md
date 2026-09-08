# AutoDebug: serving benchmark warmup was skipped

Source and saved-log audit, 2026-09-08. No device access, TTNN imports, server requests, profiling, or implementation changes were performed for this investigation.

Follow-up: [the warmup validation fix](AUTOFIX_benchmark_warmup.md) now rejects unsuccessful warmup results and reports successful/requested counts. The source observations below describe the original audited behavior; the preserved cold logs remain unchanged.

## Finding

**Verified:** the earlier optimized-vLLM `before` benchmarks on P150, P150x2, and P150x4, and the current P150 `after` benchmarks, requested zero explicit warmups. Their initial endpoint test was also skipped. The primary 128/128/1 measurement therefore includes the first serving request's decode capture; it does not measure reuse across an earlier completed serving request. Correct retained-trace replay in the functional probe does not imply a gain in this cold primary workload.

The saved P150 mean TPOT values, 28.433678 ms before and 28.422195 ms after, are the symptom motivating this audit, not evidence of a warmed improvement or regression. A paired run with explicit warmup is still required. The secondary CI workload follows the primary on the same server, so describe its previous state precisely as **no explicit workload warmup**, rather than claiming every internal cache was cold.

## Source and artifact evidence

In the installed sibling checkout, [vLLM `serve.py`](/home/hous/dev/vllm/vllm/benchmarks/serve.py:656):

- Line 656 prints `Starting initial single prompt test run...` unconditionally. Lines 657–684 only construct a request object.
- Lines 686–702 send that request only when `ready_check_timeout_sec > 0`; otherwise they print `Skipping endpoint ready check.` The CLI default is zero at lines 1508–1512, and the CLI value is passed into the benchmark at line 1730. The function signature's different default does not govern this invocation.
- Lines 704–727 send explicit warmups only when `num_warmups > 0`. The CLI default is zero at lines 1322–1325. Each warmup uses the first main input's prompt, expected output length, and sampling body. All warmup tasks finish before the main benchmark timer starts at line 788.
- The adapter's [startup warmup methods](../../tt/generator_vllm.py) `warmup_model_prefill` and `warmup_model_decode` are no-ops. Polling `/health` in the stage wrapper does not send an inference request.

All eight inspected logs contain `num_warmups=0`, `ready_check_timeout_sec=0`, and `profile=False` in the parsed namespace, then the same lines 32–34:

```text
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
```

| Saved run | Primary log | Secondary CI log |
| --- | --- | --- |
| P150 before | [128/128/1](../../readiness_vllm/P150/optimized_vllm/before/vllm_benchmark.log) | [100/100/32](../../readiness_vllm/P150/optimized_vllm/before/vllm_ci_serving_benchmark.log) |
| P150x2 before | [128/128/1](../../readiness_vllm/P150x2/optimized_vllm/before/vllm_benchmark.log) | [100/100/32](../../readiness_vllm/P150x2/optimized_vllm/before/vllm_ci_serving_benchmark.log) |
| P150x4 before | [128/128/1](../../readiness_vllm/P150x4/optimized_vllm/before/vllm_benchmark.log) | [100/100/32](../../readiness_vllm/P150x4/optimized_vllm/before/vllm_ci_serving_benchmark.log) |
| P150 after | [128/128/1](../../readiness_vllm/P150/optimized_vllm/after/vllm_benchmark.log) | [100/100/32](../../readiness_vllm/P150/optimized_vllm/after/vllm_ci_serving_benchmark.log) |

The corresponding normalized summaries preserve the actual benchmark command, which lacks either warmup or ready-check overrides. The P150 before summary retains absolute paths from its original `readiness_vllm/optimized_vllm/before/P150` location; the current evidence is under `readiness_vllm/P150/optimized_vllm/before`. This relocation does not change the recorded flags.

## Smallest corrected experiment

The [shared readiness runner](/home/hous/dev/tt-metal/models/common/readiness_check/run_vllm_server.py:1001) already supports the necessary override:

```sh
--additional-benchmark-args='--num-warmups 1'
```

It parses the value with `shlex.split` at line 1030, passes it to **both** primary and CI calls at lines 1113 and 1134, and appends the resulting arguments to each `vllm bench serve` command at line 706. No shared-harness change is necessary. At audit completion, [the model stage wrapper](run_profiles.py) already includes this override and its comment correctly identifies the skipped default ready check. This investigator did not edit that wrapper.

Run the following separately, with the specified adapter installed before each fresh server starts. These are a recipe, not commands executed by this audit:

```sh
# Original adapter; current cleanup-only worker/kernel runtime held fixed.
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_profiles.py \
  --phase warm_before --profiles P150 P150x2 P150x4
```

```sh
# Candidate adapter; exactly the same worker/kernel runtime and server settings.
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_profiles.py \
  --phase warm_after --profiles P150 P150x2 P150x4 --full-gates
```

The wrapper starts a fresh server per profile and runs benchmarks before the optional correctness gates. `--full-gates` therefore adds post-benchmark validation without preceding the measurements with those requests. Use new phase names if either output directory already exists; preserve the original `before` and `after` cold evidence. The shared runner deletes existing raw and normalized benchmark JSON files at lines 671–673 and overwrites the log, so directly reattaching with an old output directory would destroy evidence.

Both sides must use these exact paired settings:

| Profile | TP / mesh label | Max model length | TT configuration |
| --- | --- | --- | --- |
| P150 | 1 / `N150` | 50624 | `{"trace_region_size":220000000}` |
| P150x2 | 2 / `N300` | 262144 | `{"trace_region_size":220000000,"fabric_config":"FABRIC_2D","parent_mesh_shape":[2,2],"submesh_offset":[0,0]}` |
| P150x4 | 4 / `P300x2` | 262144 | `{"trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}` |

Common server settings are model `google/gemma-4-26B-A4B-it`, max sequences 32, block size 64, port 8000, sampling mode `all`, and `--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`. Preserve the wrapper's offline HF cache environment, model registration variable, and exception-on-fallback configuration. No profiler flags or profiler environment may be enabled.

Primary remains random 128 input / 128 output / 1 request, max concurrency 1. CI remains random 100 / 100 / 32, no explicit concurrency limit. Both retain seed 0, temperature 0, `--ignore-eos`, infinite request rate, and now exactly one explicit warmup. Keep primary then CI ordering identical on both servers.

The baseline adapter must be the original `6eb04274233` adapter, SHA-256 `80fcc940cffaccdde91bb480318015ca9fd25bfc2a23320b51a668ddd6a77f46`. The inspected candidate hash is `649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325`. Keep canonical `model.py` and `generator.py` unchanged. Rebaseline with the **current cleanup-only** worker and kernel changes on both sides, rather than reverting the whole worktree or comparing against an older runtime. The current manifest records vLLM revision `2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`, worker hash `428629f00e6ab06c9fc2ab0156f5cd2a23538b959be80735c432ebacf7fc7ed6`, and fabric-router source hash `753619896b67583c9be01ea76523b68682cd88cde20910832b1847584c1ba978`. Record the actual hashes for both new runs and verify they match. Earlier before manifests omit those runtime hashes, so they cannot establish this pairing merely by sharing the same tt-metal HEAD.

## Acceptance and interpretation limits

1. Both primary and CI logs must show `num_warmups=1`, `Warming up with 1 requests...`, and `Warmup run completed.` before the main run. The normalized command must contain the override. Preserve raw results and confirm every measured request completed with all requested output tokens.
2. vLLM discards the returned warmup result objects at line 723 without checking their `success` fields. The completion banner proves the tasks returned, not that inference succeeded. Verify successful warmup completion using the associated server/request evidence and check for errors before accepting the comparison. The shared runner's completed-request check validates measured requests only.
3. One CI warmup is a single batch-1 request with the CI prompt/output length. It does not establish that every padded batch shape or chunked-prefill signature in the subsequent 32-request burst is warm. Label this protocol explicitly as “one warmup request per workload”; report CI separately from primary decode throughput.
4. The baseline may intentionally recapture at the next request boundary despite having completed a warmup. That is the baseline behavior this paired experiment should measure. The candidate can reuse only when its safety guards permit it; new prefill cache entries or physical shapes can still force recapture.
5. Do not compare a new warmed candidate headline against the old cold baseline. Preserve the old metrics as cold/no-explicit-warmup evidence, and use matched new runs for conclusions. A single 128-token request also gives limited evidence about variability; any improvement claim must come from the resulting measurements, not the source diagnosis or the earlier exact-logit probes.

This audit establishes the missing benchmark warmup and a reproducible experiment. It does not establish a new performance result.
