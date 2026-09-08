# AutoDebug: first B1 serving decode latency

Inspection date: 2026-09-08. Fresh isolated investigator, source-only. No TTNN
imports, device access, server requests, profiler collection, or runtime edits.
This report precedes any proposed fix or hardware hypothesis experiment.

## Finding and confidence

**The serving warmup omits the adapter's decode admission path.** The first real
decode after prefill executes integer merge operators that startup never warms,
then checks whether the program-cache count changed and recaptures the complete
decode/sampler graph if it did. This omission is established by source. It is
the strongest explanation for the approximately 1.2-second one-time decode
delay, but the causal timing link still requires the focused experiment below.

The unchanged-server repeat removes the anomaly. That supports a cold-path
cause and refutes a persistent 20.8-ms decode cost. It does not by itself prove
which cold operation caused the delay or justify leaving startup incomplete.

## Evidence examined

- Server manifest: `full_b1_primary.command.json`; full 32-layer model, TP4 on
  four Blackhole chips/P300c boards, B1, native 262144 context, async scheduling,
  canonical device sampling, host compatibility off, profiler and watcher off.
- First raw result: `b1_cold_first/vllm_result.json`, timestamp
  `20260908-144334`: 128 input/128 output tokens, one completed request,
  TTFT 241.559 ms, mean TPOT 20.846 ms, median ITL 11.358 ms,
  p99 ITL 11.804 ms, ITL population standard deviation 106.477 ms.
- Same-server exact repeat, supplied/run by the supervisor and read from
  `readiness_vllm/vllm_result.json` before its archive to `b1_warm_repeat/`,
  timestamp `20260908-144514`: TTFT 51.064 ms, mean TPOT 11.375 ms,
  median ITL 11.359 ms, p99 ITL 11.658 ms, standard deviation 0.206 ms.
- `full_b1_primary.runner.log`, benchmark runner log, live `server.log`,
  generator/adapter/model, shared readiness runner, sibling vLLM benchmark and
  TT plugin model-runner/async controller sources, existing adapter device probe.

There are 127 output-token intervals. Their first-run versus repeat mean
difference represents 1202.902 ms additional decode time. A model with 126 equal
intervals and one outlier gives 11.361 ms for ordinary intervals and 1216.046 ms
for the outlier from the recorded mean/standard deviation. This is a consistency
calculation, not the recovered interval vector. The original JSON has no `itls`,
so the exact outlier index is unknown. With 127 samples, p99 need not include the
largest interval; its small value does not contradict a single severe stall.

The benchmark did **not** secretly warm the endpoint: pinned sibling
`vllm/vllm/benchmarks/serve.py:656` prints "Starting initial single prompt test
run", but lines 686-702 skip the actual request when
`ready_check_timeout_sec=0`. The logged arguments have that value and
`num_warmups=0`. Lines 1773-1787 remove individual intervals unless
`--save-detailed` is supplied. The shared runner supports that flag through
`--additional-benchmark-args` (`run_vllm_server.py:978`).

Source hashes matched the B1 server manifest during inspection:

| File | SHA256 |
| --- | --- |
| `tt/generator_vllm.py` | `c127718c510c9702cbe0530b7d7e25b53523401455de5af9fa6dd6a777344992` |
| `tt/generator.py` | `7a8c046f33768bab079812e6d4c048b60045625fbd2e383b564dbadca3600993` |
| `tt/model.py` | `f9da26100bdf48dd1a2e487a3687895e75b2b385f0a32cf21d10c2786907e615` |
| `models/common/readiness_check/run_vllm_server.py` | `601d9b00c3c0ef51b50caf398c35e5a3b24d88f9901e90c913c53b35b842e064` |

## H1: admission helper compilation causes first-decode recapture

### Complete source path

1. Plugin `model_runner.py:3295-3342` calls prefill/decode warmup in two phases.
   Adapter `generator_vllm.py:330-348` calls `gen.ensure_traces`, runs a direct
   `gen.prefill_forward` at length 128, then resets cache. It never calls
   `adapter.prefill_forward` or `adapter.decode_forward`.
2. Adapter `warmup_model_decode` at lines 350-351 calls only `ensure_traces`.
   Generator lines 644-645 immediately return when `_model_trace` exists. This
   does not run `_ensure_replay_safe`, refresh admission inputs, or replay a
   decode. The second warmup phase also returns through these same guards.
3. A real adapter prefill marks `_prefilled_rows` true at adapter line 236.
   Its first decode therefore enters `refresh_serving_inputs` at lines 295-300
   even if the plugin does not set `reset_batch`.
4. `generator.py:215-237` invokes `_merge_serving_vector` for tokens, RoPE and
   current position. At B1 their logical widths/dtypes are respectively
   32/UINT32, 1/UINT32, 1/INT32. The helper at lines 201-213 uploads INT32
   predicates and typed values, reshapes the old persistent vector, converts
   it to tile layout, calls `where`, converts the result to row-major layout,
   reshapes to the original target shape, and copies back. All transient
   tensors are explicitly deallocated. The width-1 variants differ from the
   32-lane UINT32 token/seed preservation path warmed by prefill
   (`_sample_prefill_device`, lines 516-543). Startup uses `_write_positions`,
   which performs host copies, instead of these merge operators.
5. Adapter line 303 invokes generator `decode_forward`. Lines 852-853 call
   `ensure_traces` and `_ensure_replay_safe`; the latter compares `_programs`
   with the device program-cache count and calls `_capture` on any difference
   (lines 675-679). `_capture` at lines 573-593 records the full model,
   sampler and history sampler. For serving there is no prefill trace.
6. Actual decode/sampling replay is nonblocking at lines 691/695. The plugin
   calls the adapter first and only then queues readback
   (`async_decode.py:624-637`), so a synchronous compilation/recapture cost
   before submission directly delays the next streamed token. The deferred
   completion waits at `async_decode.py:665-667`; that wait can expose queued
   device time but cannot overlap time spent before submission exists.

**Prediction:** after normal production warmup, the first B1 admission merge
increases the program-cache count, followed by one `_capture` inside that decode;
steady decode and a second identical request do not. Most of the cold excess
falls in the merge and/or capture host spans. Prefill itself can leave additional
program-cache changes, so record counts before and after both prefill and merge
instead of attributing the entire count delta to the merge by assumption.

### Smallest verify/refute experiment

Have the supervising hardware owner run one standalone, reduced real-weight B1
adapter process using layers `[0, 3]`, TP4, 262144 logical context, the production
64-token page size, selected precision and caller-owned cache. Reuse the setup
patterns in `tests/adapter_serving_device_probe.py:263-300`; that existing probe
only permits B3/B4, so use a separate focused probe instead of relabeling it B1.

Before calling any warmup, install process-local wrappers in the probe around
`gen._merge_serving_vector`, `gen._capture`, `gen._ensure_replay_safe`,
`gen.configure_sampling`, and adapter prefill/decode. Record host
`time.perf_counter_ns()`, program-cache count, trace IDs before/after, and the
phase/step label. For each merge also record target shape/dtype. Do not alter
production modules or add synchronization inside a measured submission span.
The program-count call is a host query (`mesh_device.cpp:1151`), not a device
synchronization. Use the existing async output split and record both submission
and finalized wall time. Dump collected records after the short run.

Run the actual four-call plugin warmup sequence, then a 128-token greedy
adapter prefill and three decodes with `reset_batch=True` on the first decode.
Repeat the identical request with the normal new-request adapter prefill, not
an artificial adapter recreation or a reset of its sampling cache key. No
concurrent server, profiler, watcher or reset command is needed. This one
process gives cold-first, steady, and request-repeat controls.

Verification needs both the new program-count delta and `_capture` attribution,
not just a faster second request. If counts do not change, no recapture occurs,
or both spans are short, H1's causal explanation is refuted; retain the
instrumentation records and test H2 below. Reduced capture timing will scale
with two layers and is not a full-model speed claim.

If verified, the narrow fix belongs in adapter startup warmup: exercise the
same real adapter prefill/decode admission path with valid positions and page
mapping, finish its readback, and restore an empty serving state. Preserve the
warmed sampling key and program-cache/trace state; clear dummy request row,
seed and cache state so no warmup history becomes a request. A second warmup
phase must not repeat expensive work or leave mutated model state. Do not
remove the `_ensure_replay_safe` guard: new legitimate prefill shapes and
sampling variants still need its protection. Do not replace the canonical
integer merge with an untested shortcut as part of this fix.

Then repeat the focused probe in a fresh process and the original full-model
B1 server benchmark immediately after readiness, with `--save-detailed`, no
benchmark warmup, and an identical warm repeat. Both cold-first and repeated
requests must remain correct; record startup time and all ITLs, including the
maximum and its index. Re-run the appropriate adapter state/async correctness
probe and nearby host tests because warmup changes persistent serving state.

## H2: scheduler/readback or unrelated host first-use delay

This remains possible until H1 is timed. Plugin async scheduling correctly has
an admission drain boundary (`model_runner.py:2349-2368`; `async_decode.py:389-400`)
and a steady path that permits overlap. There is no fixed approximately
one-second sleep in the inspected decode controller. `ensure_finalized` uses
an idempotent lock and invokes the pending readback when draining
(`async_decode.py:102-110,378-387`), rather than waiting forever for a future
the engine has not yet resolved. The stable repeat weakens a recurring
scheduler-overlap fault as the explanation for this anomaly.

If H1 fails or leaves most time unexplained, use a fresh full server with
temporary host timestamp wrappers only at plugin `execute_model`,
`build_model_input`, `submit_decode`, `finalize_decode`, and
`wait_for_all_pending_async_steps`. Correlate those spans with detailed client
ITLs and request/step IDs. A delay before adapter entry assigns the cost to
host scheduling/input construction; a long submission assigns it to the model
adapter; a long event-finalize span after a short submit assigns it to queued
device/readback completion. Do not toggle async scheduling first: that broad
change can move the symptom without locating its cause.

## Other source findings, not explanations for 1.2-second steady delay

- First-request sampling setup starts with both adapter and generator
  `_sampling_key=None`. Adapter `_sampling` can invoke `configure_sampling`,
  whose non-live branch releases/rewarms traces when the actual formatted
  key differs (`generator.py:367-383,478-490`). For caller-owned native cache,
  `ensure_traces` normally snapshots cache buffers (`648-649`). This can
  contribute to the first TTFT difference and should be logged separately
  from the decode outlier. Exercising the actual adapter path during startup
  may address it too; that needs evidence, not a second speculative fix.
- All serving prefill is eager: the generator receives caller-owned cache at
  adapter allocation (`generator_vllm.py:107-112`), while `_prepare_prefill_trace`
  requires `owns_cache` (`generator.py:608`). This explains why earlier
  standalone prefill trace performance is not the serving TTFT lower bound.
  It is outside this bounded first-decode fix.
- Steady serving decode already preserves device token/position feedback;
  adapter lines 295-311 avoid refresh outside scheduler boundaries, generator
  model/sampler replay uses `blocking=False`, and readback copies one replicated
  token shard (`generator.py:717-721`). Page tables are compared every step but
  copied only on change (`193-199`). Full-logit host sampling is not silently
  used for this host-compatibility-off greedy benchmark. None of those source
  paths proves that every remaining microsecond of host overhead is necessary.

## Status

Source warmup omission: verified. Exact 1.2-second causal attribution: pending
focused hardware experiment. Persistent 20.8-ms decode claim: refuted by the
unchanged-server repeat. No runtime fix has been made by this investigator.
