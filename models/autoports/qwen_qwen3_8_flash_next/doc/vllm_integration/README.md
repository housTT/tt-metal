# Qwen3.8-Flash-Next vLLM integration

Status: ready for shared TT vLLM serving on one P300 board (Blackhole devices
0 and 1). Sampling, quality, performance, lifecycle, trace-allocation, and
cleanup gates pass; the final independent stage review verdict is
`clean-pass`.

## Headline result

Primary single-user workload — random prompt 128 tokens, requested output 128 tokens, one request, concurrency 1, temperature 0, ignore EOS, server `max_model_len=262144`, `max_num_seqs=2` (physical traced batch 1 with two model-owned virtual state slots):

- TTFT P50/P99: **4007.855929 / 4007.855929 ms**.
- TPOT mean/P50/P99: **268.980406 / 268.980406 / 268.980406 ms**.
- ITL P50/P99: **271.870091 / 303.711892 ms**.
- Headline decode: **3.717742915 tokens/s/user**, derived as `1000 / mean TPOT`.
- Aggregate output throughput: **3.353537 tokens/s**; 128/128 output tokens completed.

Raw data is in [`vllm_result.json`](../../readiness_vllm/vllm_result.json); the normalized summary and exact command are in [`vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json).

Secondary CI serving-burst workload — random prompts 100 tokens, requested output 100 tokens, 32 requests, unbounded client concurrency, temperature 0, ignore EOS, same server configuration:

- 32/32 requests and 3200/3200 output tokens completed.
- TTFT P50/P99: **495900.588359 / 984280.693984 ms**.
- TPOT mean/P50/P99: **608.992298 / 607.325343 / 633.142100 ms**.
- ITL P50/P99: **622.197126 / 675.488786 ms**.
- Aggregate output throughput: **3.059929 tokens/s**.
- Mean-TPOT-derived value: **1.642056890 tokens/s/user**, secondary only. Burst admission, physical-B1 virtual scheduling, and chunked prefill affect this TPOT, so it is not the headline decode result.

Raw data is in [`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json); the normalized summary is in [`vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json).

The selected full-model teacher-forcing median, 4.384021 tokens/s/user, is only
an optimistic decode-latency lower bound (equivalently a throughput upper
bound), not a serving parity target. The measured vLLM windows contain zero
model-only trace replays, zero sampling-seed host copies, and zero host-sampling
compatibility calls. Primary decode records 126 trace replays plus one
admission-time trace invalidation/recapture, the expected 127 post-prefill
steps for 128 output tokens, and zero virtual-bank commit/restore/reset copies.
CI records 3167 replays plus one admission-time invalidation, the expected 3168
steps for 32 requests x 100 output tokens. No avoidable vLLM-specific decode
fallback remains in the measured physical-B1 canonical token-out loop.

## Serving contract

- Model/checkpoint: `Qwen/Qwen3.8-Flash-Next`, revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Shared plugin registration: both `Qwen4ExpForConditionalGeneration` and `TTQwen4ExpForConditionalGeneration` resolve to `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm:Qwen4ExpForConditionalGeneration` in `register_tt_models()`.
- Adapter: `tt/generator_vllm.py` is a protocol bridge. It delegates prefill, decode, output handling, cache adoption, virtual state, sampling, and cleanup to `tt/generator.py` and `tt/model.py`.
- Attention ownership: vLLM owns the 12 QSA K/V/raw/compressed-index paged caches and block IDs. The runtime recorded one vLLM adoption. Model-owned temporary attention allocations were all released (48 allocated, 48 released).
- Model ownership: 36 linear-attention recurrent states, request-local PLE history, packed expert host store, and fixed device expert slots.
- Context: `max_model_len=262144`, exactly matching [`context_contract.json`](../context_contract.json). There is no advertised-context reduction.
- Paging: block size 64. Logical prompt lengths need not align with block, 128-token prefill chunks, tiles, or trace sizes.
- Advertised and validated capacity: `max_num_seqs=2`, physical traced batch 1, virtual state capacity 2. The adapter rejects larger active widths until they complete the same full-model proof ladder.
- Scheduling: async scheduling enabled; `supports_async_decode=True`; decode token-out trace enabled.
- Sampling profile: `full`. Greedy and ordinary unseeded stochastic sampling use canonical on-device split sampling and direct device token feedback. Explicitly seeded stochastic requests and cohorts requiring unsupported host logits processing use the optional, caller-visible vLLM host compatibility sampler. This compatibility route is explicit, is not the optimized path, and is absent from both performance windows.

## Selected precision

Serving loads [`selected_precision_config.json`](../datatype_sweep/selected_precision_config.json) through the ordinary full-model constructor and rejects any path mismatch. The selected ID is `qsa_bfp8_hifi2_lm_head_bf16_hifi2`:

- weights/compute: QSA input, attention output, and GDN projections BFP8/HiFi2; shared projections BFP8/LoFi; routed experts BFP4/LoFi with the selected `g40b16/d40b5` programs; LM head BF16/HiFi2; final hyper down/up BFP8/HiFi2; BF16 row-major embeddings.
- activations and communication: BF16 ingress, residuals, matmul outputs, PLE, and CCL payload; linear CCL with two links and 8192-byte packets.
- caches: BFP8 tiled QSA KV/index caches, 64-token pages, BF16 update tensors.
- exceptions: BF16/HiFi4 norms and BF16 router/top-k outputs; the selected layer-exception map is empty.
- host representations: all 512 BF16 checkpoint experts per layer are prepacked to BFP4 host storage; ten BFP4 device slots per layer/rank. The 102.400 GB PLE table is BF16 mmap storage with BF16 host assembly/device execution and an 8192-row host cache.

## Correctness and quality

- Final shared sampling profile: **72 passed, 1 skipped, 0 failed** in 3167.48 s. It covers greedy, top-k/top-p/temperature, seeded isolation, penalties/host compatibility, mixed parameters, stale token, current position, page table, and async decode behavior. See [`sampling_tests.log`](../../readiness_vllm/sampling_tests.log).
- Direct non-aligned serving: logical lengths `1, 63, 64, 65, 67, 127, 129`, each requested twice with exact usage and identical deterministic output. See [`non_aligned_prompt_check.json`](../../readiness_vllm/non_aligned_prompt_check.json).
- Prompt-correct chat outputs: all three responses are coherent, correct, on-topic, non-repetitive, non-gibberish, in the requested language, and free of cross-request contamination. See [`qualitative_tt_chat.json`](../../readiness_vllm/qualitative_tt_chat.json), [`qualitative_prompt_format.json`](../../readiness_vllm/qualitative_prompt_format.json), and [`QUALITATIVE_REVIEW.md`](../../readiness_vllm/QUALITATIVE_REVIEW.md).
- Raw completion outputs are mechanically healthy and topic-following but non-gating because they bypass the checkpoint chat template: 7/12 reach the visible 256-token cap and 8/12 expose `<think>` markup. The sampled story echoes the prompt and both thermodynamics continuations drift into learned follow-up Q&A. There is still no mechanical loop, gibberish, wrong-language drift, or cross-request leakage. See [`vllm_qualitative_outputs.json`](../../readiness_vllm/vllm_qualitative_outputs.json).
- Hard stage check: pass, including the 262144-token context contract and qualitative degeneration check. See [`stage_gate.log`](../../readiness_vllm/stage_gate.log).

Focused adapter coverage maps the async state contract directly: `test_decode_reset_once_then_steady_async_delegation`, `test_generator_virtual_decode_microbatches_real_rows_and_ignores_padding`, `test_generator_virtual_decode_rejects_stale_generation_before_device_execution`, and `test_generator_multi_host_decode_preflights_later_stale_row` cover current-position, page-row, stale-owner, and deferred-read behavior. `test_device_bank_copies_only_request_local_state` proves token/current-position/page-table and recurrent state are request-local while vLLM QSA KV is excluded. Plugin tests `test_virtual_decode_forwards_only_unpadded_slot_metadata` and `test_stale_virtual_generation_cannot_apply_a_deferred_token` cover padded scheduler rows and cancelled/reused generations. Host-B2 compatibility is exercised by the adapter's `test_generator_virtual_prefill_supports_multi_active_host_compatibility` and `test_generator_virtual_decode_supports_multi_active_host_compatibility`, plus plugin `test_two_host_only_prefill_rows_consume_stacked_torch_logits`.

## Host-backed serving and lifecycle

Cold construction preloaded 24,576 packed experts (68,080,435,200 bytes) in 238.860717 s. The final metrics report distinguishes packed-host hits from fixed device-slot misses and reports expert, PLE, host/H2D, and stall work for each workload.

Primary 128/128/1/concurrency-1 workload:

- expert device-slot hits/misses/evictions: `24,868 / 42,796 / 42,796`; packed-host hits/misses: `42,796 / 0`.
- expert physical-owner H2D: `118,322,380,800 bytes`, 9.501212 s; expert control/index H2D: 219,800 bytes, 0.375017 s.
- PLE: 129 lookups, 4,096 selected/unique rows, 2,621,440 logical and physical H2D bytes, 0.702702 s lookup and 0.012474 s H2D.
- route-read plus TT stall: 24.020644 s; traced replay submit: 0.039884 s.
- lifecycle within the benchmark window: one assignment/release, one prefill trace invalidation, zero bank commits/restores/resets, and zero stale rejections.

CI 100/100/32 burst:

- expert device-slot hits/misses/evictions: `181,758 / 1,506,457 / 1,506,457`; packed-host hits/misses: `1,506,457 / 0`.
- expert physical-owner H2D: `4,165,052,313,600 bytes`, 329.659205 s; expert control/index H2D: 2,822,000 bytes, 6.247189 s.
- PLE: 3,201 lookups, 101,904 selected and 101,896 unique rows, 65,218,560 logical H2D bytes and 74,393,600 physical H2D bytes, 8.992924 s lookup and 0.363380 s H2D.
- route-read plus TT stall: 621.833877 s; traced replay submit: 0.962293 s.
- virtual state: 32 assignments/releases, 3,200 commits, 3,167 restores, 32 resets, one prefill trace invalidation, zero stale rejections, and zero active/valid slots at the boundary.

The complete deltas and ownership/fallback assertions are in [`serving_host_metrics.json`](../../readiness_vllm/serving_host_metrics.json). TT compute remains inside the declared front/back trace boundaries; only exact route IDs, expert/PLE lookup and transfer, and caller-visible optional host sampling are declared host work. All prohibited host-work flags are false.

Upload-failure publication/retry, cold/hit/partial/evict/reload/stale/thrash behavior, real PLE n-gram row lookup/history/EOS/reset, concurrency, cancellation, and request isolation are covered by the focused host suite and live lifecycle proof. The final focused host/readiness suite reports **65 passed** in 12.98 s in [`host_weight_cache_tests.log`](../../readiness_vllm/host_weight_cache_tests.log).

The live lifecycle run admitted two streams concurrently, observed the first token from both before cancelling one client, completed the survivor, and produced two identical clean follow-ups. It recorded 152 canonical trace replays, zero model-only replays or host-sampling calls, eight virtual-bank commits, five restores, one reset, four generation-tagged releases, zero stale rejections, zero leaked PLE history, and zero active/valid slots at the end. See [`host_serving_lifecycle.json`](../../readiness_vllm/host_serving_lifecycle.json).

SIGINT completed the API shutdown; the vLLM process manager force-terminated its one remaining EngineCore child during teardown. The post-shutdown audit passed: no vLLM, APIServer, EngineCore, or test process remained, neither device had an open holder, and both device DRAM checks and heartbeats were healthy. No reset was required. See [`process_cleanup_audit.json`](../../readiness_vllm/process_cleanup_audit.json) and the compressed final [`server.log.gz`](../../readiness_vllm/server.log.gz).

### Trace-allocation lifetime closure

The superseded pre-fix B2 tracker reproduction found a real unsafe lifetime: 75 buffers survived before replay (73 replaced GDN/PLE recurrent or convolution state buffers and two program-cache buffers). Prefill now copies state into construction-time fixed buffers, guards the full virtual-prefill/release boundary with the exact program-cache entry count, releases a live trace before returning when the program set grows, and permanently treats cache initialization as uncertain after a failed first enqueue.

The source-frozen exact-B2 rerun used unfiltered allocation tracking (`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` unset) with the same `max_model_len=262144`, physical B1/virtual B2, block 64, async, decode-only trace, and all-device sampling configuration. It passed active overlap/cancellation/follow-ups with 150 replays, one safe prefill invalidation/recapture, zero generic tracker warnings, zero unsafe-live errors, zero runtime errors/tracebacks, and clean process/device release. [`trace_allocation_tracker_b2_audit.json`](../../readiness_vllm/trace_allocation_tracker_b2_audit.json) and [`AUTOFIX_TRACE_ALLOCATION_LIFETIME.md`](../../readiness_vllm/AUTOFIX_TRACE_ALLOCATION_LIFETIME.md) are the authoritative closure. The ordinary frozen final `server.log` has one conservative active-trace allocation warning during startup/capture, but no trace-tracker live-buffer/corruption error, traceback, or EngineCore failure; the exact unfiltered tracker run is the controlling safety evidence.

## Reproduction

From the tt-metal root after sourcing the model environment:

```bash
export PYTHONPATH=/home/ttuser/dev/vllm-tt-plugin/src:$PWD:/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
export VLLM_PLUGINS=tt,tt_model_registry
export QWEN38_VLLM_LOG_METRICS=1

python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages serve --mesh-device P300 --port 8018 --max-num-seqs 2 \
  --sampling-profile full --block-size 64 --max-model-len 262144 \
  --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' \
  --additional-server-args='--async-scheduling'
```

Run the canonical readiness stages against that server:

```bash
python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages sampling --server-url http://localhost:8018 \
  --max-num-seqs 10 --sampling-profile full

python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages qualitative,benchmark --server-url http://localhost:8018 \
  --max-num-seqs 2 --sampling-profile full
```

The `--max-num-seqs 10` on the sampling client controls pytest fanout; the live server remains capped at two active requests.

## Limitations

- The public adapter is capped at the validated two simultaneous active requests over a physical-B1 trace. Wider active capacity is rejected pending an equivalent full-model proof ladder.
- Explicitly seeded stochastic requests deliberately use the optional host sampler for cross-cohort reproducibility. Seedless and greedy performance requests remain on-device and traced.
- The 32-request CI workload is a serving-admission burst, not 32 simultaneous TT decode lanes; the server schedules at most two active virtual requests.
- The source-frozen unfiltered exact-B2 tracker rerun is the controlling allocation-lifetime proof; it reports zero tracker warnings and zero unsafe buffers/errors without a program-cache skip.
