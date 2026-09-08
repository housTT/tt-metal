# AutoFix: first B1 serving decode latency

Starting report: [fresh source AutoDebug](AUTODEBUG_b1_latency.md), written before
runtime changes and accepted as hypotheses by the supervising agent. The original
all32-layer B1 native-context primary benchmark has mean TPOT 20.846 ms but median
ITL 11.358 ms; its unchanged-server repeat has mean TPOT 11.375 ms and median ITL
11.359 ms. The original and repeated 128/128/1 results are retained in
`b1_cold_first/` and `b1_warm_repeat/`. These data establish a first-use anomaly,
not a persistent 20.8-ms decode cost.

## H1 experiment

Hypothesis: production startup omits real decode admission merges. The first B1
request compiles width-1 UINT32/INT32 merge programs and then recaptures the full
decode/sampler graphs, delaying its first decode. Initial sampling configuration
may separately rewarm traces and snapshot caller cache during the first prefill.

The [focused probe](../../tests/b1_startup_latency_probe.py) exercises production
code without runtime monkey-patches other than instance-only observation wrappers.
It opens a reduced real-weight B1 TP4 model with layers 0 and 3, native logical
context 262144, and exactly 4097 caller-owned physical KV blocks. It invokes the
same four warmup calls as pinned plugin `model_runner.warmup_model`, then issues
two ordinary identical 128-token admissions, each followed by three device-sampled
decode steps through the deferred read/format split. It preserves adapter sampling
keys across requests instead of recreating the adapter or artificially resetting
the key between controls.

The wrapper records host `perf_counter_ns` spans, program-cache counts, captured
program counts, trace IDs, and phase names around `_merge_serving_vector`,
`refresh_serving_inputs`, `_capture`, `_ensure_replay_safe`, `ensure_traces`,
`configure_sampling`, and adapter prefill/decode. Each merge includes its exact
target shape/dtype. Submission, queued-copy submission, and completion spans are
separate. No synchronization is added inside an observed submission. Program-cache
queries are host-side. These reduced/instrumented times are attribution evidence,
not full-model or serving benchmark numbers.

Run only in the supervising serialized hardware lane, with the server stopped:

```bash
USER=hous ../state/serving-env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.b1_startup_latency_probe \
  --model-path ../upstream \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/b1_startup_before.json
```

The probe checks exact repeated token output, valid sampled IDs, current/RoPE
positions, stable persistent input addresses, exact caller cache identity, native
context, and absence of host decode fallback. It retains all nested spans and
source hashes, including failed-run evidence and successful cleanup metadata.
An after-fix fresh invocation can add `--reference .../b1_startup_before.json` to
require exact output parity. Optional native trace-allocation tracking can be
required with `--require-trace-allocation-tracking`, with the environment enabled
before process launch and program-cache allocations included. No profiler is used.

CPU preparation passed: script syntax compilation and the actual AST-executed
`HostSpans` wrapper correctly attribute phase, elapsed host time, and before/after
program-count changes. Repository pre-commit passed the probe. The authoring
subagent has not imported TTNN, opened hardware, or made server requests.

## Baseline result: H1 verified in the reduced adapter

The supervising lane ran the unmodified runtime probe with exit 0 and clean mesh
closure at 2026-09-08 14:53:57 UTC. Independently inspected
[JSON](b1_startup_before.json) and [log](b1_startup_before.log) show:

| Boundary | Program-cache / host evidence |
| --- | --- |
| End of production startup | 235 programs; trace captured at 166; adapter and generator sampling keys unset |
| First real prefill configuration | 5 new programs; 20.456 ms configuration, including 19.928 ms `ensure_traces` |
| First decode token merge, UINT32 width 32 | No new program; 0.214 ms |
| First decode RoPE merge, UINT32 width 1 | 4 new programs; 2.949 ms |
| First decode current-position merge, INT32 width 1 | 4 new programs; 4.783 ms |
| First decode recapture | 9.677 ms `_capture`; submission totals 18.103 ms |
| Repeated request boundary | No new merge programs or capture; about 0.773 ms submission |

This verifies omitted admission compilation followed by complete trace recapture
in the first decode. Reduced two-layer spans do not independently assign the full
all32-layer 1.2-second outlier; the first-request full-model benchmark remains
required. The initial prefill sampling-key/capture work is a separate startup
omission observed directly in the same baseline.

## Narrow startup repair

Only adapter `warmup_model_prefill` and its TTNN import change runtime behavior.
The method now primes canonical neutral sampling before the existing
`ensure_traces(preserve_cache=False)`, initializes the adapter key, executes an
actual adapter prefill and device decode admission, and queues/finishes its
deferred token read. It then resets cache/recurrent/conv state, canonical seeds,
collector index, token/current/RoPE inputs, and adapter row/pending/mode flags.
The warmed sampling keys and traces remain reusable; the plugin's second warmup
phase is idempotent. Steady decode and canonical sampling selection are unchanged.

Pinned plugin `input_batch.py` represents disabled logprobs as `num_logprobs=-2`.
The generator key includes this field even when `enable_log_probs=false`, so
warming with zero would leave a first-API-prefill reconfiguration. Startup uses
the actual sentinel and per-row lists matching adapter formatting. Greedy
unrestricted `top_k` and ordinary `top_p` values still normalize through the
canonical formatter to the same captured `k=1, p=0` distribution. No separate
adapter sampling algorithm is introduced.

Warmup uses neutral penalties throughout. Canonical `_prepare_prompt_sampling`
returns before modifying prompt masks/host shadow, and neutral sampling does not
update penalty counts. Those histories remain constructor-empty; collector
history is explicitly reset. The extended probe independently checks empty prompt
host metadata, zero generated-history rows/index, device seeds matching reset
host values, zero token/position/RoPE inputs, clear admission flags, and zero
recurrent/conv buffers on every replica. It does not read full KV or logits.

The [new CPU regression](../../tests/test_serving_startup_warmup.py) executes actual
adapter warmup/admission and canonical parameter formatting with CPU TT boundaries.
It covers B1/B4 and mRoPE/non-mRoPE, a single finished dummy decode across the four
warmup calls, reset state/cache/input identity, retained keys, and no full-cache
snapshot even when the first request supplies plugin-style unrestricted greedy
`top_k` and an explicit seed. All four cases reject the saved original startup
source ([before log](b1_startup_host_before.log)). The current nearby suite passes
44 tests ([after log](b1_startup_host_after.log)); all three changed/authored Python
files pass [pre-commit](b1_startup_precommit.log).

## Current status

The fresh after-fix reduced run passed with exit 0 and clean mesh closure at
2026-09-08 15:02:10 UTC. Independently inspected
[JSON](b1_startup_after.json) and [log](b1_startup_after.log) confirm startup ends
with 243 programs and traces captured at that exact count; both sampling keys are
set and `_live=false`. Neither real request adds a program or calls `_capture`.
The command added `--reference .../b1_startup_before.json` and
`--require-clean-warmup`; both requests' sampled IDs exactly match the saved
before-fix outputs. Request parameters now reproduce the plugin's disabled-logprob
sentinel and unrestricted greedy top-k, numerically equivalent to the original
probe's settings through canonical greedy formatting.

| Reduced instrumented span | Before | After |
| --- | --- | --- |
| First prefill, including output | 28.973 ms | 5.756 ms |
| First decode submission | 18.103 ms | 0.814 ms |
| Repeated-request first decode submission | 0.773 ms | 0.715 ms |
| First decode including finalized token read | 19.821 ms | 2.503 ms |

All 32 token lanes and the B1 current/RoPE positions are zero after startup;
device/prefilled/pending-seed flags are false, last mode is `None`, prompt host
history is empty, collector history rows/index are zero, reset host/device seed
values match, and all four recurrent/conv buffers are zero on every replica.
Caller cache identity and persistent input addresses remain unchanged. The two
real requests contribute exactly six device decodes and six async reads, with
zero host decodes. Neutral reduced outputs alone are a weak contamination check;
these direct state assertions provide the independent reset proof.

The runtime diff was compared to the saved pre-fix adapter: only its TTNN import
and `warmup_model_prefill` changed. The supervisor's independent
[AST/source comparison](warmup_source_scope.json) confirms every other adapter
method and the generator/model source are unchanged.

Both subsequent native-guard runs passed with exit 0 and clean teardown:
[B1 startup](b1_startup_after_tracker.json) checks the complete warmup/reset path
with native trace-allocation tracking and program-cache allocations included;
[B3 adapter](adapter_device_tracker_startup_final.json) checks stale asynchronous
token/position inputs, changed scheduler pages, and slot remapping on the updated
source. The [actual adapter host gate](adapter_host_startup_final.log) also passed
all five tests. These supplement the 44 CPU boundary regressions.

## Full-model first-request verification: performance bug fixed

The supervising lane reproduced the original all32-layer primary workload on a
fresh B1 server immediately after readiness, then repeated it on the same server:
128 input/128 output tokens, one request, concurrency 1, greedy temperature 0,
TP4 on P300c, native context 262144, and the same selected precision. There were
no benchmark warmup requests. The only additional benchmark argument,
`--save-detailed`, saves all intervals after execution. The
[comparison artifact](b1_startup_benchmark_comparison.json) binds all four raw
results by SHA256 and links the before/after server manifests; those hashes were
independently checked against the retained files.

| Same primary workload | Before first | Before repeat | After first | After repeat |
| --- | --- | --- | --- | --- |
| TTFT (ms) | 241.559 | 51.064 | 62.613 | 50.127 |
| Mean TPOT (ms) | 20.846 | 11.375 | 11.402 | 11.377 |
| Decode tokens/s/user, `1000 / mean TPOT` | 47.970 | 87.914 | **87.706** | 87.898 |
| Median ITL (ms) | 11.358 | 11.359 | 11.364 | 11.364 |
| p99 ITL (ms) | 11.804 | 11.658 | 11.690 | 11.653 |
| Output throughput (tokens/s) | 44.301 | 85.565 | 84.717 | 85.604 |

The true fresh-first [raw result](b1_final_primary/vllm_result.json), timestamp
2026-09-08 15:05:49 UTC, contains all 127 intervals. The maximum is 19.106 ms at
zero-based index 0. The [exact repeat](b1_final_repeat/vllm_result.json) also
contains 127 intervals, with maximum 16.052 ms at index 0. Both completed exactly
one request and generated identical text; the retained detailed interval arrays
and generated strings were independently checked. The fresh-first result, not the
repeat, is the repaired headline measurement.

This measured first-request result removes the cold decode stall while preserving
the previously observed steady latency. Combined with the reduced source/host
span proof, it verifies the startup omission and successful repair. It does not
assign exactly 1.2 seconds to one reduced `_capture` call: the old aggregate
mean/standard deviation only supported an inferred outlier, and the two-layer
probe measured the mechanism at a different model size. The full before/after
same-workload benchmark supplies the actual end-to-end performance verdict.

The [final B1 server log](full_b1_startup_final.server.log) records lifetime
counters of 255 model replays, 255 canonical sampling replays, 255 device decodes,
and 255 async reads: one startup decode plus 127 for each of the two real requests.
There are zero host decodes, history readbacks, generator synchronizations, or
generator read waits. Each of token/current/RoPE/page refresh counts is six across
the whole lifetime. These are startup/request-boundary totals, not per-token
uploads. The three prefills comprise one dummy startup request and the two real
requests. Engine initialization including cache/warmup took 3.91 seconds as logged;
this is not total process startup time.

The [server manifest](full_b1_startup_final.command.json) records return code 0;
mesh closure completed at 2026-09-08 15:07:55 UTC. Runtime remains frozen. The B1
performance bug is fixed, and the final full B32 serving regression gate passed:72 sampling tests,1 canonical skip,
all12 qualitative outputs reviewed,32/32 CI requests completed, shared runner exit0.
See full_b32_startup_final_checks.runner.log.
