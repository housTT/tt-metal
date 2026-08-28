# Qwen3.8-Flash-Next vLLM integration work log

## Scope and starting point

- Model: `Qwen/Qwen3.8-Flash-Next`, checkpoint revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Starting tt-metal commit: `fe285f8c974` on `hous/qwen3.8-flash-next`.
- Starting vLLM TT plugin commit: `106744c` on `housTT/register-muse-glimmer-30b`.
- Input stage: completed datatype sweep. The selected policy is
  `qsa_bfp8_hifi2_lm_head_bf16_hifi2`; no vLLM-specific precision override is
  permitted.
- Target: P300 Blackhole dies 0 and 1, TP2/DP1, one active segmented traced
  request slot. Concurrent HTTP requests are admitted and queued by vLLM.
  This is a current token-out implementation limit, not a claimed physical
  device or DRAM limit.

## Implementation

`tt/generator_vllm.py` is a protocol bridge. It registers the Qwen4Exp plain
and TT-prefixed architectures and delegates prefill, decode, asynchronous
read, and host formatting to `tt/generator.py`. The adapter contains no sampler,
argmax/top-k fallback, logits conversion, or token feedback loop. The canonical
full-model split sampler owns on-device sampling and keeps its sampled token in
the persistent TT input buffer. The one compact token read required by the
declared host PLE lookup is never written back to TT.

The shared plugin chooses the device sampler only when the model capability is
implemented. Qwen3.8 declares default penalties and random `top_k <= 32` on
device. Unsupported penalties, all-vocabulary random sampling, logprobs, and
other shared-test-only features use the plugin's explicit host compatibility
path. Greedy serving and the benchmark stay on the device path.

vLLM owns the exact twelve-layer QSA attention object. Each entry contains K,
V, raw index keys, and compressed index keys. Linear-attention recurrence and
the expert/PLE stores remain model-owned and are excluded from vLLM KV. The
adapter validates object identity and exposes cache lifecycle, host service,
decode timing, fallback, and ownership metrics.

## Precision and context

The adapter constructs the ordinary full-model generator with
`doc/datatype_sweep/selected_precision_config.json`. The selected runtime policy
is:

- routed expert weights BFP4 TILE / LoFi, all 512 experts prepacked on host,
  ten exact device slots per layer/rank;
- shared expert BFP8 / LoFi;
- QSA, GDN projections, attention output, and final hyper projections BFP8 /
  HiFi2;
- BF16 activations, residuals, CCL payload, logits, and sampling; BF16/HiFi4
  norms and BF16/HiFi2 LM head;
- BFP8 TILE KV/raw-index cache with BF16 updates; BF16 compressed index cache;
- BF16 PLE mmap rows, host assembly, staging, and execution, with an 8192-row
  host cache and 128-row prefill staging.

`doc/context_contract.json` advertises 262,144 tokens. vLLM adds one 64-token
allocator-headroom block, so the physical pool has 4,097 blocks / 262,208
tokens while the served `max_model_len` remains the full 262,144-token logical
contract. There is no capability reduction.

## AutoFix investigations

The full server exposed two independent QSA paging bugs and two sampling ABI
bugs; each was isolated with the AutoFix workflow before its fix was retained.

1. The compressed QSA cache was still allocated with 4,096 physical blocks
   when vLLM passed its intentional 4,097-block pool. The fused index selection
   tried to reshape 8,388,608 elements as 8,390,656 elements, exactly one
   compressed page too many. The vLLM adoption path now reallocates and owns all
   four QSA state tensors at 4,097 blocks and releases the standalone tensors.
2. Prefill compute pads every final chunk to 128 tokens, but vLLM allocates
   cache pages only for logical tokens. A prompt shorter than 64 tokens received
   a block table `[real_page, 0]`; writing the padded second half could overwrite
   physical page zero. Chunk page tables now round the logical length to 64 and
   functional/fused QSA cache fills slice K/V/raw/compressed inputs to those
   actual pages while retaining all 128 padded query rows for compute.

Request reset was also hardened: stable exact-zero TT tensors are copied into
GDN/conv/PLE state in place, avoiding allocation leaks and NaN-preserving
multiply-by-zero. Immutable expert weights now persist across requests, so a
request reset does not destroy valid cache hits.

3. `ttnn.sampling` consumes inverse temperature as a logits multiplier, while
   the canonical Qwen model API had uploaded the user-facing temperature. At
   temperature 2.0 this sharpened EOS into an almost-certain first token. The
   canonical full-model boundary now validates positive user temperature and
   uploads its reciprocal. An `ignore_eos` probe had already shown varied
   tokens, refuting stuck RNG, top-k, and host-fallback hypotheses.
4. `top_k=1` at temperature 2.0 was stable but disagreed with greedy. A
   one-token support is mathematically argmax regardless of temperature or
   top-p, and the stochastic local-top32 route does not provide the exact
   global tie behavior. The canonical model now routes every all-top1 cohort
   through its existing global device argmax and disables unused seed state.

## Server command

The final server command is run from the tt-metal root after:

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

The effective TT config additionally contains the readiness runner default
`sample_on_device_mode=all`. Chunked prefill is disabled by the shared plugin
for Qwen4Exp. Decode trace mode is enabled and asynchronous scheduling is on.

## Validation chronology

Host-only/current-source checks:

```bash
python -m pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py::<focused-host-nodes> \
  /home/ttuser/dev/vllm-tt-plugin/tests/test_device_sampling_capabilities.py
```

Result: 24 passed. This covers adapter delegation/capabilities, exact vLLM
cache adoption, stale-token PLE input, page/current-position flow, non-aligned
paging, cache cold/hit/miss/eviction/stale protection, upload-failure recovery,
real PLE row/history/reset/cache metrics, threaded concurrency/cancellation,
inverse-temperature/top1 semantics, and device/host sampling capability selection. Raw log:
`readiness_vllm/host_weight_cache_tests.log`.

Full server page-boundary probe: two repeat-identical on-device greedy requests
at each logical prompt length 1, 63, 64, 65, 67, 127, and 129. Every response
was HTTP 200, every reported prompt length was exact, and both responses in
every case were identical. The requests do not ask for logprobs and therefore
stay on the canonical device argmax path. Raw artifact:
`readiness_vllm/non_aligned_prompt_check.json`.

Targeted AutoFix sampling rerun: ten runtime sampling checks passed, including
shuffled seed batches, explicit seeds, unseeded variety, and first-token
temperature variety. A fresh-process plugin import-order error was isolated as
test infrastructure; the canonical first test primes vLLM imports, after which
the mixed isolation node passed. Raw log:
`readiness_vllm/autofix_targeted_sampling.log`.

Canonical full sampling command against the active-batch-one server:

```bash
python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages sampling --server-url http://localhost:8018 \
  --max-num-seqs 10 --sampling-profile full
```

Here `--max-num-seqs 10` is only the canonical pytest fixture fanout: it lets
the penalty and mixed-parameter tests construct both sides of comparisons.
The live server remains `--max-num-seqs 1` and queues the submitted requests.
The final result is recorded in `readiness_vllm/sampling_tests.log`.

## Host-backed serving and lifecycle

Expert directory publication occurs only after all required uploads submit;
failure invalidates reserved slots and a retry successfully reloads them while
protected hits survive. PLE uses the real mmap-backed two/three-gram table,
request-isolated two-token history, explicit reset/cancel, and locked row cache.
TT compute remains in front/back segmented traces around PLE lookup/upload and
compact route-id read/expert service.

Request-boundary telemetry is opt-in and outside the token loop. It reports
expert hits/misses/evictions/prepack/H2D, PLE selected/unique/read/cache/history
and host/device H2D, route-read/TT-stall and trace-submit times, expert slot and
packed-host occupancy, preload totals, attention lifecycle, trace/copy
counters, host compatibility calls, ownership, and prohibited fallback flags.
Primary and CI-burst deltas are stored in
`readiness_vllm/serving_host_metrics.json`.

Live cancellation evidence overlaps requests against the single physical slot,
cancels one client, drains the survivor, and then requires two deterministic
follow-up requests to be identical. Artifact:
`readiness_vllm/host_serving_lifecycle.json`.

Final shutdown used SIGINT and the readiness runner's abort-mode cleanup; its
log includes one process-manager force-kill. The subsequent audit records empty
APIServer/vLLM/EngineCore process matches and healthy devices in
`readiness_vllm/process_cleanup_audit.json`. It is cleanup evidence, not a
claim that every child exited gracefully.

## Final sampling and qualitative evidence

The final canonical sampling suite completed in 4,485.77 seconds with **72
passed and 1 skipped**. The skip is the shared conditional case, not a runtime
failure. The suite includes device greedy, random top-k/top-p, inverse
temperature, exact top-1 equivalence, seed behavior, mixed request isolation,
structured-output routing, logprobs, and explicit optional host-compatibility
coverage. Raw output: `readiness_vllm/sampling_tests.log`. The focused fixes
also pass in `readiness_vllm/autofix_topk.log`.

The prompt-correct primary suite reran the three shared full-model prompts with
`Qwen2Tokenizer` and exact
`apply_chat_template(add_generation_prompt=True)` token IDs. The template hash
is `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`;
prompt lengths 76, 89, and 94 matched server usage. Exact HF control completions
and tokens come from `doc/full_model/qualitative_shared_suite_final.json`.

The explanation and one-sentence summary stop cleanly and are correct. The
coding response is coherent and includes the correct set/list implementation
and `[3, 1, 2]` example in its visible reasoning, but reaches the 256-token cap
before emitting its final answer. All three are free of mechanical loops,
gibberish, unintended language drift, and request contamination. Exact prompts,
token IDs, controls, outputs, and review data are in
`readiness_vllm/qualitative_tt_chat.json`,
`readiness_vllm/qualitative_prompt_format.json`, and
`readiness_vllm/QUALITATIVE_REVIEW.md`.

The earlier six raw `/v1/completions` prompts, each greedy and sampled, are
retained only as secondary untemplated continuation stress because this
checkpoint has a non-empty chat template. All twelve remain coherent; seven
reach the 256-token cap. Artifact:
`readiness_vllm/vllm_qualitative_outputs.json`.

A second full-48 server was run with
`TT_METAL_TRACE_ALLOC_TRACKING=1` before TTNN import. Seven HTTP-200 requests
completed with 1,033 decode trace replays, zero generic active-trace allocation
warnings, zero unsafe-live-allocation errors, zero model-only replays, zero
host sampling compatibility calls, and every prohibited host-work flag false.
This is direct vLLM-stage evidence that closes the untracked allocation warning
from the original server. Artifacts:
`readiness_vllm/trace_allocation_tracker_audit.json` and
`readiness_vllm/autofix_chat_template/server.log`.

The final hard gate passed with no degenerate output and confirmed served and
target context length 262,144. Raw output: `readiness_vllm/stage_gate.log`.

## Final benchmarks

Headline single-user workload: 128 prompt tokens, 128 requested/generated
tokens, one request, concurrency one, temperature zero, ignore EOS, canonical
device argmax.

- TTFT P50/P99/mean: 5,358.047 / 5,358.047 / 5,358.047 ms.
- TPOT P50/P99/mean: 299.713 / 299.713 / 299.713 ms.
- ITL P50/P99/mean: 302.326 / 363.098 / 299.713 ms.
- TPOT-derived decode: **3.337 t/s/u**; output throughput 2.948 tok/s;
  request throughput 0.0230 req/s; total token throughput 5.896 tok/s.
- Wall duration: 43.422 s; 1/1 request and 128/128 output tokens complete.

Raw client result: `readiness_vllm/vllm_result.json`. Normalized primary
artifact: `readiness_vllm/vllm_benchmark.json`. Command output:
`readiness_vllm/vllm_benchmark.log`.

Secondary CI serving-burst workload: 100 prompt tokens, 100 requested/generated
tokens, 32 requests, unbounded client admission into one active traced slot,
temperature zero, ignore EOS.

- TTFT P50/P99/mean: 546,147.117 / 1,064,667.786 / 544,262.447 ms.
- TPOT P50/P99/mean: 302.471 / 325.746 / 298.668 ms.
- ITL P50/P99/mean: 300.555 / 367.852 / 298.668 ms.
- TPOT-derived decode: 3.348 t/s/u; output throughput 2.894 tok/s;
  request throughput 0.0289 req/s; total token throughput 5.788 tok/s.
- Wall duration: 1,105.777 s; 32/32 requests and 3,200/3,200 output tokens
  complete.

Raw client result: `readiness_vllm/vllm_ci_serving_result.json`. Required
normalized CI artifact: `readiness_vllm/vllm_ci_serving_benchmark.json`.
Command output: `readiness_vllm/vllm_ci_serving_benchmark.log`. The CI number
is secondary because admission serialization and prefill scheduling dominate
its TTFT; it is not the headline decode number.

The selected full-model teacher-forcing result, 228.101 ms/token or 4.384
t/s/u at a constructed context of 4,096, is only an optimistic latency lower
bound. It omits sampling/token feedback and does not carry the served 262,144
context. During the primary vLLM request, 37.526 s of the 38.064 s TPOT decode
window (98.6%) was inside the measured model trace/host-service boundary. The
remaining 0.538 s is about 4.23 ms per interval. The measured path has 126
device trace replays, no model-only replay, no host compatibility sampling, no
full-logit readback, and no prohibited fallback, so no avoidable vLLM-specific
decode path remains identified.

## Final host-service metrics

The final model-load gauges are 24,576 packed expert entries / 68,080,435,200
bytes, 241.257 s preload, 480/480 device slots resident, 1,592,524,800 device
bytes per rank, and 8,192 PLE cache rows. Attention-cache lifecycle is 48
standalone tensors allocated and released plus one vLLM adoption; ownership is
vLLM throughout both benchmark windows.

Primary 128/128/1/concurrency-one deltas:

- expert service: 6,791 requests/waves, 25,707 hits, 41,974 misses/evictions
  and packed-host hits, zero packed-host misses, 116,049,715,200 H2D bytes in
  14.857 s, and 222,480 expert-index bytes in 0.628 s;
- PLE: 129 calls, 4,096 selected/unique rows, 3,240 mmap rows / 1,036,800 bytes
  read, 1,310,720 host-assembly bytes, 2,621,440 logical/physical H2D bytes,
  0.655 s lookup, and 0.017 s device H2D;
- boundary timing: 37.526 s total submit, 37.037 s expert service, 23.060 s
  route-read/TT stall, 13.944 s cache-control DMA submit, 0.329 s PLE service,
  and 126 device trace replays.

CI 100/100/32/unbounded-admission deltas:

- expert service: 172,843 requests/waves, 639,424 hits, 1,082,044
  misses/evictions and packed-host hits, zero packed-host misses,
  2,991,635,251,200 H2D bytes in 383.664 s, and 5,461,120 expert-index bytes
  in 15.778 s;
- PLE: 3,232 calls, 102,400 selected / 102,392 unique rows, 81,575 mmap rows /
  26,104,000 bytes read, 32,768,000 host-assembly bytes, 65,536,000 logical /
  74,711,040 physical H2D bytes, 9.794 s lookup, and 0.383 s device H2D;
- boundary timing: 932.630 s total submit, 922.865 s expert service, 573.725 s
  route-read/TT stall, 348.333 s cache-control DMA submit, 5.729 s PLE
  service, and 3,136 device trace replays.

Both windows have zero model-only trace replays, host sampling compatibility
calls, and seed-copy changes; every prohibited host-work flag is false. The
machine-readable deltas and end-state ownership/fallback records are in
`readiness_vllm/serving_host_metrics.json`.

## Final lifecycle and cleanup

The live cancellation check completed a 128-token survivor while overlapping
and client-cancelling a second queued request after one second, then required
two non-empty HTTP-200 follow-ups to match byte-for-byte. The server did not
emit an explicit abort marker; that limitation is retained in the artifact and
README rather than inferred away. vLLM owns admission/cancellation, and every
model admission initializes fresh state. The focused current-source tests also
cover real PLE history/reset/cancel and threaded request isolation. Artifact:
`readiness_vllm/host_serving_lifecycle.json`.

The final server was interrupted with SIGINT; vLLM entered abort mode and the
process manager force-killed one remaining child. The process audit then found
no APIServer, vLLM, EngineCore, or runner process, no holder of
`/dev/tenstorrent/0` or `/dev/tenstorrent/1`, healthy DRAM on both P300 dies,
and synchronized heartbeats. The direct tracker-enabled rerun has the same
clean post-stop audit. Artifacts: `readiness_vllm/process_cleanup_audit.json`
and `readiness_vllm/autofix_chat_template/process_cleanup_audit.json`.

## Active-sequence boundary

The largest active traced `max_num_seqs` proven by this stage is one; the
32-request CI workload is concurrent HTTP admission queued behind that slot,
not active device batching. This is a fail-fast implementation boundary in the
full-stack segmented token-out trace: it has one recurrent-state workspace,
one PLE row, and one top-8 routed-expert service row. It is explicitly **not**
classified as a physical-memory limit. The full 48-layer eager control already
runs batch 32 at context 4,096, and both recorded DRAM plans retain headroom.
True multi-active traced serving requires a new multi-user expert-wave and
state-trace ABI rather than an adapter-only relaxation. Exact code gates,
capacity numbers, and supporting artifacts are recorded in
`readiness_vllm/max_num_seqs_limit.json`.

## Review and commits

The pre-stage code audit returned clean-pass after ownership wording was made
consistent with vLLM adoption. The first final stage review returned
`more-work-needed` for raw-prompt qualitative evidence, physical-limit wording,
and an unclassified allocation warning. AutoFix produced prompt-correct chat
evidence, an honest segmented-ABI boundary report, and a direct tracker-enabled
full-48 vLLM rerun. The fresh independent rereview then returned **clean-pass**
with no P0/P1/P2 findings; see `doc/vllm_integration/STAGE_REVIEW.md`.

Stage-owned local commit SHAs are appended after the commits below. Nothing is
pushed.
