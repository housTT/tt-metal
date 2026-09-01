# GPT-OSS 120B optimized vLLM serving

## Result

**Primary warmed P150x4 vLLM serving, 1 request × (128 prompt → 128
output), concurrency 1, temperature 0:** **502.93 ms TTFT P50/P99** and
**57.67 decode tokens/s/user** from **17.339 ms mean/P50/P99 TPOT**. ITL
P50/P99 is **16.238/29.191 ms** and aggregate output throughput is
**47.316 tok/s**. The request completed 128/128 output tokens.

The same clean-server `run_vllm_server` workload before this change measured
510.30 ms TTFT, 21.691 ms mean TPOT, 21.243/35.636 ms ITL P50/P99,
39.200 tok/s aggregate output throughput, and 46.10 decode tokens/s/user.
The retained optimization therefore improves TPOT-derived decode rate by
**25.10%**, reduces mean TPOT by **20.06%**, and leaves TTFT flat (-1.44%).

| Primary workload: P150x4, 128/128/1, concurrency 1 | Before | After | Change |
| --- | ---: | ---: | ---: |
| TTFT P50/P99 | 510.297 / 510.297 ms | **502.928 / 502.928 ms** | -1.44% |
| TPOT mean/P50/P99 | 21.691 / 21.691 / 21.691 ms | **17.339 / 17.339 / 17.339 ms** | -20.06% |
| ITL mean/P50/P99 | 21.691 / 21.243 / 35.636 ms | **17.339 / 16.238 / 29.191 ms** | -20.06% mean |
| Output throughput | 39.200 tok/s | **47.316 tok/s** | +20.70% |
| TPOT-derived decode | 46.103 t/s/u | **57.674 t/s/u** | **+25.10%** |

The comparable optimized full-model P150x4 prompt-128/output-128 split-greedy
token-out path is 15.904 ms/token or 62.879 t/s/u. Final vLLM reaches 91.7%
of that rate; its 16.238 ms ITL P50 is within 0.334 ms of the full-model wall
time. This is close parity for the same model, precision, trace, LM-head,
sampling, and token-feedback work with vLLM scheduling and a caller-visible
token boundary added.

**Secondary CI serving burst, 32 requests × (100 prompt → 100 output),
unbounded client admission, server `max-num-seqs=32`, temperature 0:** 32/32
completed; TTFT P50/P99 is 15.272/15.273 s, TPOT mean/P50/P99 is
589.409/589.402/589.546 ms, ITL P50/P99 is 589.396/594.845 ms, and aggregate
output throughput is **43.464 tok/s**. The before throughput was 43.450 tok/s.
This is capacity/nightly-parity evidence only; its interleaved TPOT is not the
headline decode t/s/u.

## What changed

The shared generator previously called `.cpu(blocking=False)` on the complete
mesh sampled-token tensor. GPT-OSS TP4 sampling replicates the token tensor
across the four TP columns, while output formatting later keeps only the first
replica. The adapter was therefore transferring four device shards and
discarding three at every vLLM decode boundary.

The GPT-OSS generator now resolves the distinct token rows before starting the
read. On the production `(1, 4)` mesh it submits one device-shard read, records
the same mesh event after that nonblocking submission, and converts the
host-resident shard only after the plugin synchronizes the event. Row-sharded
meshes select one replica per distinct row. Logprob payloads and full-logit
compatibility calls retain the shared generic collector.

The adapter also implements the plugin's terminal
`release_persistent_capture()` hook. Orderly worker shutdown now writes the
final serving counters before releasing model and sampling traces and before
the mesh is closed. This made the measured-path audit deterministic instead of
depending on Python `atexit` in EngineCore multiprocessing.

The full-depth trace-allocation gate found two startup/prefill hazards hidden
by the earlier two-layer probe. Shared block-size inspection indexed a TT
tensor and materialized an unnecessary device slice, and generic prefill
warmup did not compile the explicit per-request page-table signature that
vLLM uses. Block-size inspection now examines Python containers only, and the
adapter compiles one exact B1 per-layer paged-fill signature before any decode
trace is captured. A 36-layer run with allocation tracking and traceback
capture enabled, with program-cache tracking still active, completed all 226
B1/B32 model and sampler replays with zero unsafe survivors.

## Measured-path contract

The final benchmark-only shutdown snapshot contains only the primary and CI
workloads:

| Counter | Final clean benchmark |
| --- | ---: |
| vLLM decode calls / async reads | 226 / 226 |
| Device-sampled / host-sampled decodes | **226 / 0** |
| Model / sampling nonblocking trace submissions | 226 / 226 |
| Unclassified trace submissions | 0 |
| Fixed sampling-state replays / pushes | 224 / 2 |
| Minimal token readbacks | 226 |
| Device token shards read | **226** |
| Redundant TP replica shards skipped | **678** |
| Full-logits readbacks / host argmax calls | **0 / 0** |
| Validation full-logit synchronizations | 0 |
| Page-table refreshes / reuses | 9 / 221 |
| Full input refreshes | 2, once for each B1/B32 phase |

`sample_on_device_mode=all` is set by the actual server command. The adapter
returns device tensors from `decode_forward(..., read_from_device=False)`;
model and sampler replay use `ttnn.execute_trace(..., blocking=False)`; the
plugin starts the selected-shard `.cpu(blocking=False)` read; and the event is
synchronized only in plugin finalization. `supports_async_decode` remains true
because this exact path exercised 226 async submissions.

Token, current-position, RoPE, page-table, KV-cache, and sampler state remain
persistent device inputs. The first call for each B1/B32 phase initializes
state. The next 224 calls reuse the fixed on-device sampler state, with token
feedback and position progression owned by the traces. Scheduler page-table
changes are copied; 221 unchanged steps reuse the resident table. Host tests
cover stale host token/current-position values, changed page tables, unchanged
page tables, drained/removal-only resets, and B1/B32 bucket transitions.

The terminal path is still the selected full-model split sampler:
vocab-sharded LM head, local TopK, gathered candidates, canonical greedy
parameters, sampling trace, and `tt_out_tok` feedback. There is no adapter
sampler, force-argmax route, generic slow greedy sampler, full-vocabulary
all-gather, host top-1/argmax, or Python token read/write feedback loop in the
measured path.

## Preserved configuration

| Item | Final value |
| --- | --- |
| Model / revision | `openai/gpt-oss-120b` / `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| Hardware | four Blackhole p300c boards as P150x4 mesh `(1, 4)` |
| vLLM | official workspace checkout, v0.26.0, `568afb3a13806beb53bb2e6bd518269357b237c0` |
| TT plugin | standalone official checkout, `414b870b57844115d37e8b8abf3ef0e8b62765bf` |
| Adapter | `models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py` |
| Resident layers | 36/36 |
| `max_model_len` | 131072, unchanged from `doc/context_contract.json` |
| `max-num-seqs` | 32 |
| KV cache | vLLM-owned hybrid paged cache, page size 64, BFP8 |
| Tracing | B1 and B32 decode traces; eager prefill |
| Sampling | `sample_on_device_mode=all`, full-model split sampling |
| Fabric / trace region | `FABRIC_1D_RING` / 750,000,000 bytes |

The selected datatype-sweep `ds00_baseline` policy is unchanged: BFP8
attention weights/projection inputs/attention CCL/LM head/KV cache, BFP4
expert weights, BF16 residual/router/norm/expert intermediates/expert CCL,
LoFi decode attention and expert math, HiFi2 prefill attention/router/LM-head,
and HiFi4 SDPA. Policy SHA-256 is
`b8e1e655581ffca37dd8b841285940c494a5e43d6b5deec3ebc10078ddb52c57`.

The public context contract and valid non-aligned prefill path were not
changed. Focused coverage passes prompt lengths 65 and 97 through the canonical
generator, and the completed integration's live 65-token HTTP request remains
valid evidence because this optimization changes only post-sampling token
collection.

## Correctness and qualitative gates

- Full plugin sampling profile: **73 passed, 1 expected skip** in 1506.58 s.
  It covers greedy and stochastic sampling, per-request seeds, B1/B32 and mixed
  batches, penalties, logprobs, host-only compatibility controls, request
  isolation, full-capacity structured output, and async decode.
- Focused adapter/full-model host suite after the changes: **62 passed, 12
  hardware-gated skips**. Pre-commit passed for every touched Python file.
- Qualitative suite: all six prompts were generated in greedy and sampled
  modes. All 12 outputs passed coherence, topic, repetition, gibberish,
  language, and cross-request-contamination review. The two thermodynamics
  responses are explicitly classified incomplete at the fixed 256-token cap
  because they do not reach the Third Law; the regenerated comparison makes
  no completeness claim for them.
- Scoped degeneracy checker: exit 0, no degenerate output detected.
- Primary benchmark: 1/1 request and 128/128 output tokens completed.
- CI burst: 32/32 requests and 3200/3200 output tokens completed.
- Final process audit found no API server or EngineCore process. `tt-smi`
  listed all four boards after shutdown; no reset was needed.
- Full 36-layer trace-allocation gate: primary 1/1 and CI burst 32/32;
  226 model plus 226 sampler replays; zero live unsafe buffers. No
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` exclusion was used.

## Commands

Every device command first sourced
`.agents/scripts/gpt_oss_workspace_env.sh`. An import-origin assertion checked
TTNN, vLLM, the plugin, and the adapter below
`/home/ttuser/dev/gpt-oss-20b` before every server start.

The final clean before/after benchmark command was:

```bash
VLLM_SYSTEM_START_DATE=2026-08-31 \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --block-size 64 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":750000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args "--async-scheduling --disable-log-stats --structured-outputs-config '{\"reasoning_parser\":\"openai_gptoss\",\"enable_in_reasoning\":false}'" \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 1 \
  --benchmark-concurrency 1 \
  --benchmark-temperature 0 \
  --ci-benchmark-prompt-len 100 \
  --ci-benchmark-output-len 100 \
  --ci-benchmark-num-requests 32
```

The final all-gates server used the same arguments with
`--stages serve,sampling,qualitative,benchmark --sampling-profile full`.

The full-depth allocation-safety rerun used the benchmark command above with:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 \
TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python -m models.common.readiness_check.run_vllm_server ...
```

`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` remained unset.

No Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or
`ReadDeviceProfiler` was collected. That omission is intentional for this
vLLM-serving stage. Device-time and roofline fields are therefore `null` in
`perf_summary.json`; the optimization is justified by same-harness benchmark
JSON plus runtime trace/async/fallback counters.

## Evidence

- `perf_summary.json`: machine-readable primary, burst, and full-model
  comparison.
- `artifacts/before_reproduced/`: clean unchanged-integration before metrics
  and server log.
- `artifacts/after_minimal_token_read/`: first isolated candidate A/B,
  54.859 t/s/u.
- `artifacts/after_full_validation/`: 73-pass sampling log, qualitative
  outputs/verdict, degeneracy report, long-lived warmed benchmark, full
  runtime counters, and compressed server log.
- `artifacts/after_final_clean/`: final clean primary and CI JSON/logs,
  pre-review measured-path-only capability snapshot and compressed server log.
- `artifacts/trace_allocation_full36/`: exact production full-depth tracker
  result, benchmark JSON, zero-survivor summary, server log, and post-run
  cleanup/device-health evidence.
- `artifacts/after_final_remediation/`: exact final-code normal primary and CI
  JSON/logs, capability snapshot, compressed server log, and post-run audits.
- `AUTODEBUG_trace_allocation.md`: fresh-context diagnosis that separates the
  real prefill survivors from the generic trace-storage warning path.
- `work_log.md`: implementation chronology, decisions, commands, checks, and
  optimization checklist.

## Limitations

- Production serving requires P150x4. P150 and P150x2 fail the physical
  resident-state capacity gate rather than lowering the advertised context.
- Prefix caching remains disabled for this alternating full/sliding hybrid KV
  contract.
- Explicit host sampling remains available for unsupported sampling controls
  and correctness tests. It is not used by the reported benchmarks.
- Process exit still prints known nanobind reference-leak diagnostics after
  orderly trace release and mesh close. The runner nevertheless terminates
  cleanly, leaves no vLLM/EngineCore process, and leaves all boards visible.
- Normal logging emits one generic active-trace warning while reserved
  `BufferType::TRACE` storage is created for the split sampling trace. Metal's
  allocation tracker explicitly excludes that reserved trace-storage type;
  the stricter full-36 tracked run completed every replay with zero survivors.
