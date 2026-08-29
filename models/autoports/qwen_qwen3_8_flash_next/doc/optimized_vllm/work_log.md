# Optimized vLLM work log

## Frozen serving contract

- Model/checkpoint: `Qwen/Qwen3.8-Flash-Next`, revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Real path: vLLM TT plugin registration of
  `Qwen4ExpForConditionalGeneration` from
  `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm`.
- Mesh: P300 Blackhole dies 0 and 1, `FABRIC_1D`, packet payload 8192.
- Precision: datatype-sweep selection
  `qsa_bfp8_hifi2_lm_head_bf16_hifi2`; BFP4/LoFi exact routed experts,
  BF16/HiFi2 LM head, BFP8 KV cache, BF16 CCL payloads.
- Server: `max_num_seqs=2`, physical decode batch 1, virtual slot capacity 2,
  block size 64, `max_model_len=262144`, async scheduling, decode-only trace,
  trace region 1 GiB, `sample_on_device_mode=all`.
- Host stores: all 24,576 experts prepacked into 68,080,435,200 host bytes;
  ten persistent expert slots/layer/rank; serial miss waves; configured
  staging depth 1; PLE row-cache capacity 8192; unpinned contiguous host rows.
- Both performance collections used 12 sequential qualitative warm-up
  requests before the primary benchmark, then the primary and CI burst on the
  same live server. No Tracy, `tt-perf-report`, live device profiler, adapter
  profiler, or `ReadDeviceProfiler` was collected in this stage.

## Reproducible server and benchmark commands

The final server command (port is not a performance parameter) was:

```bash
PYTHONPATH=/home/ttuser/dev/vllm-tt-plugin/src:$PWD:/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages \
TT_VISIBLE_DEVICES=0,1 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
VLLM_PLUGINS=tt,tt_model_registry \
QWEN38_VLLM_LOG_METRICS=1 \
QWEN38_HOST_MISS_WAVE_POLICY=serial \
QWEN38_HOST_STAGING_DEPTH=1 \
python_env/bin/python \
  /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after \
  --stages serve --mesh-device P300 --port 8021 --max-num-seqs 2 \
  --sampling-profile full --block-size 64 --max-model-len 262144 \
  --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' \
  --additional-server-args='--async-scheduling'
```

The warmed qualitative/primary/CI sequence was:

```bash
PYTHONPATH=/home/ttuser/dev/vllm-tt-plugin/src:$PWD:/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages \
python_env/bin/python \
  /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after \
  --stages qualitative,benchmark --server-url http://localhost:8021 \
  --max-num-seqs 2 --sampling-profile full
```

The runner expands that into an explicit greedy `vllm bench serve` primary
with random input length 128, output length 128, one request, concurrency 1,
temperature 0, and `ignore_eos`; then a separate CI-parity burst with random
input/output length 100/100, 32 requests, unbounded client concurrency,
temperature 0, and `ignore_eos`. The `before/` run used the same flags, env,
mesh, config, warm-up sequence, cache/store capacities, and generation mode;
only the port and source implementation differed.

After one post-burst sentinel established the final cumulative metrics marker,
the exact windows are selected with:

```bash
python_env/bin/python \
  models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/derive_serving_host_metrics.py \
  --server-log models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after/server.log \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/after/serving_host_metrics.json \
  --primary-requests 1 --ci-requests 32 \
  --primary-trace-replays 127 --primary-trace-captures 1 \
  --ci-trace-replays 3168 --ci-trace-captures 0 \
  --max-num-seqs 2 --physical-batch 1 --virtual-slot-capacity 2
```

## Measured before/after

Primary values are the warmed 128-input / 128-output / 1-request /
concurrency-1 greedy workload. CI values are the secondary 100-input /
100-output / 32-request unbounded-client-concurrency burst; they are capacity
evidence, not the headline decode rate.

| Run and shape | TTFT p50/p99 (ms) | TPOT mean/p99 (ms) | ITL p50/p99 (ms) | Aggregate output (token/s) | TPOT-derived t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary 128/128/1 before | 4007.856 / 4007.856 | 268.980 / 268.980 | 271.870 / 303.712 | 3.353537 | 3.717743 |
| Primary 128/128/1 after, definitive | 4484.084 / 4484.084 | 269.661 / 269.661 | 272.196 / 303.694 | 3.304815 | 3.708356 |
| Primary 128/128/1 after, same-source repeat A | 4146.631 / 4146.631 | 267.016 / 267.016 | 266.566 / 301.020 | 3.363287 | 3.745088 |
| CI 100/100/32 before | 495900.588 / 984280.694 | 608.992 / 633.142 | 622.197 / 675.489 | 3.059929 | 1.642057 |
| CI 100/100/32 after, definitive | 486420.899 / 965326.408 | 594.470 / 622.156 | 605.783 / 642.479 | 3.123112 | 1.682172 |

The optimized single-user repeats bracket baseline (-0.25% and +0.74% in
decode t/s/u), so the result is recorded as neutral. The definitive CI burst
improves mean TPOT 2.38% and aggregate output 2.06%. Comparable optimized
full-model 128+128 batch-1 token-out is 231.594 ms / 4.317901 t/s/u; definitive
vLLM is 269.661 ms / 3.708356 t/s/u, a 16.44% latency overhead.

The retained exact cache window improves completed service from 6.820373 to
12.496113 GB/s and p50/p95 from 4.051894/4.077123 to
2.171385/2.699363 ms. In definitive primary/CI windows, expert H2D enqueue is
8.795/283.602 s, PLE lookup is 1.489/16.237 s, PLE H2D enqueue is
0.015/0.350 s, exposed route stall is 24.569/639.758 s, expert service is
32.891/917.919 s, PLE service is 0.760/7.976 s, and total submit is
33.756/928.506 s. Owner D2D, expert completion syncs, and PLE completion syncs
are all zero.

## Optimization sequence

1. Froze the completed integration and datatype-sweep policy. Collected the
   full-profile baseline under `before/`.
2. Audited `generator_vllm.py`, the canonical generator, TT plugin async
   runner, trace inputs, on-device sampling, vLLM KV ownership, host expert
   store, PLE store, and lifecycle/cleanup paths.
3. Changed expert misses to write each exact owner shard directly into the
   persistent slot coordinate. Removed one 2,764,800-byte upload scratch per
   layer/rank and two owner D2D submissions per miss.
4. Added a conservative physical-owner ledger. Same-owner slot reuse skips
   the already-zero peer copy; owner flips enqueue the exact immutable-zero
   peer reset. Failures and diagnostic probes mark ownership unknown.
5. Reused the plugin's already-enqueued compact async token read for the next
   request/generation/virtual-slot PLE lookup. A lock makes conversion and
   event completion one-shot even when plugin finalization and the next decode
   contend. Initial request steps retain the exact device-token fallback.
6. Exported async-reuse/fallback and direct-H2D/peer-reset counters, then made
   the exact benchmark-window parser reject host sampling, model-only trace,
   seed copies, owner D2D, completion synchronizations, missing async reuse,
   or incomplete lifecycle accounting.
7. Tried a decode-specialized PLE 16-row dedup. It passed exact CPU tests and
   improved the isolated lookup microbenchmark 6.89x cold / 1.14x hot, but
   real primary TPOT regressed to 270.037506 ms. The source/test were reverted;
   evidence is under `candidates/ple_decode_dedup/`.
8. Ran the full sampling, prompt-correct qualitative, generic qualitative,
   non-aligned serving, reduced real-weight virtual-B2 stale-input/isolation,
   cancellation/lifecycle, context, benchmark, and cleanup gates on the
   retained implementation.

## Focused commands and results

```bash
python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py -q
# retained direct-slot source: 17 passed

python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py -q
# 34 passed

RUN_QWEN38_HOST_DMA_BENCH=1 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_completed_cache_service_bandwidth -q
# pass: 12.496113 GB/s, 2.171385 ms p50, 2.699363 ms p95

RUN_QWEN38_REDUCED_VLLM_TT=1 python_env/bin/python -m pytest \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_vllm_virtual_slots_tt.py::test_reduced_real_weight_vllm_virtual_b2_trace_isolation -q
# pass in 20.823 s: exact A/B isolation, lengths 63/67, changed token/current
# position, changed/unchanged page tables, stale generation, cancellation,
# async reuse 8, initial fallbacks 4
```

The full plugin sampling gate used the live real server and `--sampling-profile
full`: 72 passed, one expected all-vocabulary logprobs skip. Prompt-correct
qualitative used `run_prompt_correct_qualitative.py` with the checkpoint chat
template and inherited full-model HF controls; all three outputs were coherent,
topical, non-degenerate, and HTTP 200. Non-aligned serving passed exact prompt
lengths 1, 63, 64, 65, 67, 127, and 129 twice each. The active-overlap cancel
test released all four logical requests and produced identical follow-ups.

## AutoFix record

The focused layer-0 host-backed/reference test reports decode PCC
`0.8637402653694153`. AutoFix forced every peer-zero copy, repeated on clean
starting HEAD `adcfaa2191584bdb6e56c1d2e8b49ecc278a9a51`, and inserted an
explicit post-`ensure_indexed` synchronization; all three controls produced
the identical PCC and values. The diagnostic sync was removed and no
speculative fix was retained. `AUTODEBUG.md` and `AUTOFIX.md` record the
pre-existing regression classification and artifacts.

## Device and process discipline

Hardware runs were serialized on dies 0 and 1. Watcher/profiler collection was
kept separate from this serving stage, and no prohibited profiler was started.
After each live-server campaign, vLLM/API/EngineCore processes and device
holders were checked explicitly. Final cleanup and device-health results are
recorded in `after/process_cleanup_audit.json`.

## Independent stage review

A fresh-context, read-only `$stage-review` inspected the goal contract, skill
contracts, source, raw benchmark/qualitative/host-metric artifacts, exactness
tests, cleanup evidence, and the external plugin wiring. Its verdict is
`clean-pass` with no Required Work. `stage_review.md` contains the full scope,
anomaly ledger, and non-blocking residual risks. The stage-owned implementation
and evidence checkpoint SHA is recorded below after the local commit; nothing
was pushed.

The raw baseline and final server logs remain at `before/server.log.gz` and
`after/server.log` in this checkout. They were reviewed and are retained
locally, but are not added to Git because both exceed the repository's 500 KB
artifact limit. Versioned JSON/XML and the small sampling, benchmark, and stage
gate logs preserve the selected measurements and derived contract assertions.

Local checkpoint record:

- Starting tt-metal HEAD: `adcfaa2191584bdb6e56c1d2e8b49ecc278a9a51`.
- Stage-owned implementation, tests, review, and selected-evidence commit:
  `b3233ad4ee106f63e23b1c525c535ab96c62a80d` (`Optimize Qwen3.8 vLLM
  serving`).
- External `/home/ttuser/dev/vllm-tt-plugin`: no stage-owned commit; the
  pre-existing dirty `src/vllm_tt_plugin/platform.py` was used read-only and
  preserved unchanged.
- Push status: not pushed.
