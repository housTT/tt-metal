# Qwen3.8-Flash-Next vLLM readiness

## Headline serving result

**Primary single user — 128 prompt tokens / 128 output tokens / 1 request /
concurrency 1 / greedy device sampling:** TTFT P50/P99 **5,358.05 / 5,358.05
ms** and decode **3.337 t/s/u** from mean TPOT **299.713 ms**. This is the
headline decode result.

| Workload | TTFT P50 / P99 | TPOT P50 / P99 / mean | ITL P50 / P99 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Primary: 128 in / 128 out / 1 request / concurrency 1 / temperature 0 | 5,358.05 / 5,358.05 ms | 299.713 / 299.713 / 299.713 ms | 302.326 / 363.098 ms | 3.337 TPOT t/s/u; 2.948 output tok/s; 0.0230 req/s; 5.896 total tok/s |
| Secondary CI burst: 100 in / 100 out / 32 requests / unbounded admission, one active traced slot / temperature 0 | 546,147.12 / 1,064,667.79 ms | 302.471 / 325.746 / 298.668 ms | 300.555 / 367.852 ms | 3.348 TPOT t/s/u (secondary only); 2.894 output tok/s; 0.0289 req/s; 5.788 total tok/s |

The primary workload completed 1/1 request and all 128 requested output tokens
in 43.422 s. The CI serving-burst workload completed 32/32 requests and all
3,200 requested output tokens in 1,105.777 s. Its high TTFT is queued admission
behind the model's one active traced sequence and is not used as the headline
decode result. Raw and normalized artifacts are
[`vllm_result.json`](../../readiness_vllm/vllm_result.json),
[`vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json),
[`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json),
and [`vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json).

## Status

The model serves through the shared OpenAI-compatible vLLM TT plugin path on a
P300 board (Blackhole dies 0 and 1), TP2/DP1. The adapter is
[`tt/generator_vllm.py`](../../tt/generator_vllm.py); both the plain Qwen4Exp
architecture and TT-prefixed alias are registered by
`vllm_tt_plugin.platform.register_tt_models()`.

The adapter only translates the plugin protocol. Prefill, decode, asynchronous
read/format, split on-device sampling, persistent device token feedback,
request state, expert service, and PLE lookup delegate to the canonical
full-model implementation in `tt/generator.py` and `tt/model.py`. There is no
adapter sampler, host argmax, generic top-k fallback, full-logits readback, or
Python token readback/writeback loop in the performance path.

## Server configuration

The final server used `max_model_len=262144`, `max-num-seqs=1`, block size 64,
asynchronous scheduling, and decode tracing. The worker adds one allocator
headroom block, so vLLM owns 4,097 blocks / 262,208 physical token positions
while the served logical contract remains the full 262,144 tokens from
`doc/context_contract.json`.

`max-num-seqs=1` is a current segmented token-out **implementation** limit,
not a claimed DRAM/device limit. The selected full-stack trace ABI has one
routed-expert service row and one fixed recurrent-state workspace. The existing
full-model eager control runs batch 32 at context 4,096, and the recorded DRAM
plans have substantial headroom; a larger active traced batch therefore needs a
new multi-user expert-wave/state ABI. vLLM still admits and queues concurrent
HTTP requests, but those are not presented as active device batching. Exact
gates and capacity controls are in
[`max_num_seqs_limit.json`](../../readiness_vllm/max_num_seqs_limit.json).

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export PYTHONPATH=/home/ttuser/dev/vllm-tt-plugin/src:$PWD:/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
export VLLM_PLUGINS=tt,tt_model_registry
export QWEN38_VLLM_LOG_METRICS=1
unset QWEN38_VLLM_LAYER_INDICES
python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages serve --mesh-device P300 --port 8018 --max-num-seqs 1 \
  --sampling-profile full --block-size 64 --max-model-len 262144 \
  --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' \
  --additional-server-args='--async-scheduling'
```

The runner adds `sample_on_device_mode=all` to the effective TT config. Qwen4Exp
declares `supports_async_decode=True`; prefix caching and chunked prefill are
disabled for this model, while arbitrary logical prompt lengths remain valid.

## Selected precision

Serving loads the datatype-sweep winner
`qsa_bfp8_hifi2_lm_head_bf16_hifi2` without a vLLM override:

- routed experts: BFP4 TILE / LoFi, all 512 experts per layer prepacked on host,
  ten fixed device slots per layer/rank;
- shared experts: BFP8 / LoFi;
- QSA input/output, GDN projections, and final hyper projections: BFP8 / HiFi2;
- LM head: BF16 / HiFi2; norms: BF16 / HiFi4;
- activations, residual, CCL payload, logits, and sampling: BF16;
- K/V and raw QSA index cache: BFP8 TILE with BF16 updates; compressed QSA
  index cache: BF16; page size 64;
- PLE: BF16 row-major mmap table, BF16 host assembly/staging/execution,
  8,192-row host cache, and 128-row prefill staging;
- no layer exceptions.

## Correctness and quality evidence

- Canonical sampling profile: **72 passed, 1 skipped** in 4,485.77 s. This
  includes seeded/unseeded variety, first-token temperature variety, mixed
  device top-k, exact top-1/greedy equality, request isolation, structured
  output mixing, logprobs, and explicit host-only penalties. See
  [`sampling_tests.log`](../../readiness_vllm/sampling_tests.log).
- Focused current-source adapter/host/plugin contracts: **24 passed**. Upload
  failure recovery, protected hits, expert hit/miss/eviction/stale guards,
  real PLE mmap rows/history/reset/cache metrics, threaded cancel/isolation,
  precision, paging, stale-token/current-position/page-table behavior, and
  sampling capability routing are covered in
  [`host_weight_cache_tests.log`](../../readiness_vllm/host_weight_cache_tests.log).
- Non-aligned device-greedy requests at logical lengths 1, 63, 64, 65, 67,
  127, and 129 were all HTTP 200 with exact prompt-token accounting and
  repeat-identical outputs. See
  [`non_aligned_prompt_check.json`](../../readiness_vllm/non_aligned_prompt_check.json).
- The live cancellation workload completed a 128-token survivor, cancelled an
  overlapping queued client, and then produced two non-empty, byte-identical
  deterministic follow-ups. vLLM emitted no explicit abort log line; client
  cancellation plus clean subsequent reset/isolation is the recorded evidence.
  See [`host_serving_lifecycle.json`](../../readiness_vllm/host_serving_lifecycle.json).
- The primary qualitative suite uses the checkpoint's exact chat template and
  token IDs and compares three served greedy outputs against the existing HF
  control. The explanation and summary stop cleanly and are correct; the coding
  output stays coherent and contains the correct implementation/example in its
  reasoning but reaches 256 tokens before its final answer. No case shows
  gibberish, language drift, looping, or request contamination. See
  [`qualitative_tt_chat.json`](../../readiness_vllm/qualitative_tt_chat.json),
  [`qualitative_prompt_format.json`](../../readiness_vllm/qualitative_prompt_format.json),
  and [`QUALITATIVE_REVIEW.md`](../../readiness_vllm/QUALITATIVE_REVIEW.md).
  The earlier twelve untemplated outputs remain secondary raw-continuation
  stress evidence only.
- A direct full-48 vLLM rerun with `TT_METAL_TRACE_ALLOC_TRACKING=1`
  completed seven HTTP-200 requests and 1,033 decode trace replays with zero
  unsafe-live-allocation errors, zero generic active-trace allocation warnings,
  zero model-only replays, and every prohibited host-work flag false. This
  directly closes the earlier untracked generic allocation warning; see
  [`trace_allocation_tracker_audit.json`](../../readiness_vllm/trace_allocation_tracker_audit.json).
- The vLLM hard gate reports no degenerate output and confirms the full 262,144
  context contract. See [`stage_gate.log`](../../readiness_vllm/stage_gate.log).

## Host-backed serving metrics

The model owns the 36 linear-attention recurrent states and the declared
expert/PLE stores; vLLM owns the twelve-layer QSA attention object. Adoption
released 48 standalone QSA tensors (K, V, raw index, compressed index for each
QSA layer). The fixed host preload contains 24,576 packed experts /
68,080,435,200 bytes and took 241.257 s. All 480 device expert slots were
resident during both benchmark windows; the PLE row cache held 8,192 entries.

| Workload | Expert cache/service | PLE lookup/H2D | Traced timing boundary |
| --- | --- | --- | --- |
| Primary: 128 in / 128 out / 1 request / concurrency 1 | 25,707 hits; 41,974 misses/evictions/packed-host hits; 0 packed-host misses; 116,049,715,200 H2D bytes in 14.857 s; 222,480 index bytes in 0.628 s | 129 calls; 4,096 selected/unique rows; 3,240 mmap rows / 1,036,800 bytes read; 1,310,720 host-assembly bytes; 2,621,440 logical/device H2D bytes in 0.017 s; 0.655 s lookup | 126 device trace replays; 37.526 s total submit; 37.037 s expert service; 23.060 s route-read/TT stall; 13.944 s cache-control DMA submit; 0.329 s PLE service |
| CI burst: 100 in / 100 out / 32 requests / unbounded admission, one active traced slot | 639,424 hits; 1,082,044 misses/evictions/packed-host hits; 0 packed-host misses; 2,991,635,251,200 H2D bytes in 383.664 s; 5,461,120 index bytes in 15.778 s | 3,232 calls; 102,400 selected / 102,392 unique rows; 81,575 mmap rows / 26,104,000 bytes read; 32,768,000 host-assembly bytes; 65,536,000 logical / 74,711,040 physical H2D bytes in 0.383 s; 9.794 s lookup | 3,136 device trace replays; 932.630 s total submit; 922.865 s expert service; 573.725 s route-read/TT stall; 348.333 s cache-control DMA submit; 5.729 s PLE service |

Both benchmark windows have zero model-only trace replays, zero optional host
sampling calls, zero seed-copy changes, stable vLLM cache ownership, and every
prohibited fallback flag false. Exact deltas and end-state lifecycle counters
are in [`serving_host_metrics.json`](../../readiness_vllm/serving_host_metrics.json).
The 185 MB raw server log is retained losslessly as `server.log.gz` to keep the
stage commit below repository blob-size limits.

## Full-model lower-bound comparison

The selected full-model teacher-forcing measurement is 228.101 ms/token or
4.384 t/s/u. It is used only as an optimistic lower-bound latency reference:
that run constructs a 4,096-token model and excludes sampling/token feedback,
whereas serving must retain the 262,144-token contract and performs declared
PLE/expert host service plus client-visible sampling.

For the primary serving request, 37.526 s of the observed 38.064 s TPOT decode
window (98.6%) is inside the model's measured traced submit/host boundary. The
remaining 0.538 s is about 4.23 ms per decode interval, and telemetry proves the
path used 126 device trace replays with no model-only trace, host compatibility,
full-logit, or prohibited fallback work. There is no identified avoidable
vLLM-specific decode path left in this measurement.

## Limitations

- The current segmented traced token-out implementation is one active sequence.
  This is not a physical-memory claim: eager batch 32 at context 4,096 is
  already proven, while multi-active traced serving needs a multi-row
  routed-expert/state ABI. vLLM accepts concurrent HTTP admissions and queues
  them; burst TTFT therefore reflects serialization.
- Device random sampling supports default penalties and `top_k <= 32`.
  Unsupported penalties, logprobs/all-vocabulary cases required by shared tests,
  or other unsupported controls use an explicit optional plugin host path. It
  never replaces the device-greedy benchmark path.
- Serving is text-only. Request-specific multimodal MRoPE is disabled while
  ordinary text RoPE remains intact.
- Chunked prefill and prefix caching are disabled. Non-aligned logical prompt
  lengths are nevertheless supported and directly tested.
- Cold construction preloads about 68.08 GB of packed expert weights and maps
  the 102.40 GB BF16 PLE table; measured expert preload is 241.257 s.
- vLLM does not emit a server abort line for the queued-client cancellation in
  this configuration. It owns disconnect/cancellation; the adapter resets PLE,
  GDN, convolution, page, token, and position state at every admitted request.

SIGINT used the runner's abort-mode cleanup (the log includes one process-manager
force-kill), followed by an explicit clean audit: no APIServer, EngineCore, or
runner process and no TT device holder remained, and both dies reported healthy
DRAM and synchronized heartbeats. The tracker-enabled rerun has the same clean
post-stop result. See
[`process_cleanup_audit.json`](../../readiness_vllm/process_cleanup_audit.json)
and
[`autofix_chat_template/process_cleanup_audit.json`](../../readiness_vllm/autofix_chat_template/process_cleanup_audit.json).
