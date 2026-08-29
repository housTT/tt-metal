# Qwen3.8-Flash-Next vLLM work log

## Frozen final-source state

The model serves through the shared TT vLLM plugin on a P300 1x2 mesh. The frozen final source uses `max_model_len=262144`, block size 64, async scheduling, full sampling profile, physical traced batch 1, and two model-owned virtual state slots. vLLM owns all QSA attention cache state; the model owns linear-attention recurrence, the PLE request history/table, and expert host/device stores. Sampling, qualitative, benchmark, host lifecycle, exact trace-allocation, and cleanup artifacts pass. The final independent stage review verdict is `clean-pass`.

The adapter is `tt/generator_vllm.py`. It delegates execution to `Qwen38Generator` and `Qwen38FullModel`; the performance path uses the canonical split-sampling token-out trace and direct device feedback. There is no adapter-local argmax/top-k fallback, logits readback, token reconstruction, or independent KV allocation. Optional host sampling exists only for shared-test parameters unsupported by the bounded TT sampler and for explicitly seeded stochastic requests whose vLLM/TT RNG algorithms are not identical.

## Implementation chronology

1. Registered plain and TT-prefixed Qwen4Exp architectures in the shared plugin and normalized the Qwen sparse-attention layer metadata so vLLM allocates exactly 12 attention cache groups.
2. Added the thin adapter, selected-precision construction, exact 262144-token capability, vLLM cache allocation/adoption, async decode, and lifecycle metrics.
3. Reused the full-model split sampler and compact token-out path for greedy and supported stochastic sampling. Corrected TT sampler inverse-temperature handling and exact `top_k=1` global argmax behavior in the canonical model.
4. Made arbitrary logical prompt lengths safe. The QSA cache fill now ignores padded block-table tails, and the 4097-block vLLM allocator-headroom cache is adopted without retaining the constructor-time 4096-block compressed tensor.
5. Added stable request/slot/generation metadata through the plugin. A physical-B1 virtual state bank snapshots only model-owned recurrence, PLE convolution/history state, token/position/page state, and sampler state; vLLM QSA cache and global expert weights are excluded. Finish, preemption, cancellation, stale generation, reset, and survivor transitions are generation-guarded.
6. Added explicit optional host-logits compatibility for unsupported sampling cohorts. Host-mode prefill/decode serializes over the canonical physical-B1 model-only trace and materializes each reused logits result before advancing the next virtual row. The canonical device token-out path is unchanged.
7. Forced explicitly seeded stochastic requests to the request-owned vLLM host generator, preventing cohort-dependent switches between different TT/host RNG algorithms. Greedy and seedless performance traffic remains on-device.
8. Diagnosed a repeated non-aligned full-model prefill stall. The endpoint expansion avoids unsafe tiled reshape reuse, and route sparsity now performs max reduction, row-major conversion, then a metadata reshape. The upstream sparse-matmul FP32-intermediate CB sizing correction was also applied. Focused TT regressions pass.
9. Added an allocator-safe coordinate H2D primitive. Expert service now uses one parent-mesh staging allocation and writes only the owning rank coordinate, so physical H2D counters no longer double-count replicated staging. The topology/H2D/D2D TT gate passes.
10. Expanded metrics to cover attention ownership, runtime fallback, exact host service, virtual state, cold preload, PLE gauges, and process cleanup. Benchmark-window derivation uses logical request deltas plus canonical replay-plus-invalidation signatures: primary 126 replays plus one invalidation equals 127 decode steps; CI 3167 plus one equals 3168.
11. Reproduced the B2 allocator warning with exact tracking and found a real 75-buffer pre-fix lifetime (73 replaced recurrent/conv/PLE state buffers and two program-cache buffers). Prefill now writes fixed construction-time state buffers; the full virtual-prefill/release boundary invalidates live traces on program-cache growth; failed lazy initialization leaves a persistent uncertainty flag. Reduced and exact-B2 unfiltered tracker gates pass.
12. Froze the repaired source and regenerated the full sampling, qualitative, single-user, CI burst, host-metric, active-cancellation, cleanup, and hard-gate evidence. The final benchmarks and metrics below come only from this source generation.

## Final server command

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
unset QWEN38_VLLM_LAYER_INDICES QWEN38_HOST_EXPERT_WAVE_FENCE QWEN38_HOST_EXPERT_WAVE_REUSE_FENCE QWEN38_HOST_EXPERT_STAGE_FENCE QWEN38_VLLM_DEBUG_PREFILL_PROGRESS
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

The server became ready after the 68.080 GB packed expert preload. Exact preload time in the final metrics is 238.860717 s. The served context equals `doc/context_contract.json`; no reduction was used. The runner supplies `sample_on_device_mode=all`; `trace_mode=decode_only`, the 1 GiB trace region, `FABRIC_1D`, 8192-byte fabric payload, and 24576-byte L1-small reservation are explicit in the TT config above.

## Final validation commands and results

Non-aligned direct serving:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/run_non_aligned_prompt_check.py \
  --server-url http://localhost:8018 \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/non_aligned_prompt_check.json
```

Pass for `1,63,64,65,67,127,129`, twice each, with exact usage and deterministic equality.

Canonical full sampling profile:

```bash
python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages sampling --server-url http://localhost:8018 \
  --max-num-seqs 10 --sampling-profile full
```

Result: 72 passed, 1 skipped, 0 failed in 3167.48 s. The larger client value controls test fanout only; the server remained at two active requests.

Shared qualitative and benchmark stages:

```bash
python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm \
  --stages qualitative,benchmark --server-url http://localhost:8018 \
  --max-num-seqs 2 --sampling-profile full
```

Primary random 128 input / 128 requested output / one request / concurrency 1 / temperature 0 / ignore EOS: 1/1 complete and 128/128 output tokens; TTFT P50/P99 4007.855929/4007.855929 ms; TPOT mean/P50/P99 268.980406/268.980406/268.980406 ms; ITL P50/P99 271.870091/303.711892 ms; output throughput 3.353537 tokens/s; TPOT-derived headline decode 3.717742915 tokens/s/user. Raw: `readiness_vllm/vllm_result.json`; normalized: `readiness_vllm/vllm_benchmark.json`.

Secondary CI random 100 input / 100 requested output / 32 requests / unbounded client concurrency / temperature 0 / ignore EOS: 32/32 complete and 3200/3200 output tokens; TTFT P50/P99 495900.588359/984280.693984 ms; TPOT mean/P50/P99 608.992298/607.325343/633.142100 ms; ITL P50/P99 622.197126/675.488786 ms; aggregate output throughput 3.059929 tokens/s. The 1.642056890 TPOT-derived tokens/s/user is recorded only as a burst result, never as the headline. Raw: `readiness_vllm/vllm_ci_serving_result.json`; normalized: `readiness_vllm/vllm_ci_serving_benchmark.json`.

The selected full-model traced teacher-forcing median is 4.384021 tokens/s/user. It is retained only as an optimistic decode-latency lower bound (throughput upper bound); it excludes normal serving orchestration and is not a parity target.

Prompt-correct qualitative:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/run_prompt_correct_qualitative.py \
  --server-url http://localhost:8018 \
  --snapshot /home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --control models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_shared_suite_final.json \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/qualitative_tt_chat.json \
  --max-tokens 256
```

All three prompt-correct outputs passed manual review: coherent, correct, on-topic, no mechanical repetition, no gibberish, no wrong-language drift, and no request contamination. They stopped naturally with prompt/completion usage `76/155`, `89/148`, and `94/82`. The 12 raw completions were also read: 7 hit the visible 256-token cap, 8 expose thinking markup, the sampled story echoes its prompt, and thermodynamics drifts into learned follow-up Q&A. None shows a mechanical loop, gibberish, wrong-language drift, or cross-request leakage.

Hard gate:

```bash
MODEL_DIR=models/autoports/qwen_qwen3_8_flash_next \
HF_MODEL=Qwen/Qwen3.8-Flash-Next \
bash .agents/prompts/model_bringup_multigoal/09-vllm.check.sh \
  > models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage_gate.log 2>&1
```

Pass: no degenerate output; target and served context both 262144.

Metrics derivation:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/derive_serving_host_metrics.py \
  --server-log models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/server.log \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/serving_host_metrics.json \
  --primary-requests 1 --ci-requests 32 \
  --primary-trace-replays 127 --ci-trace-replays 3168 \
  --max-num-seqs 2 --physical-batch 1 --virtual-slot-capacity 2
```

Markers 631/632/649 were selected. Primary has 126 raw trace replays plus one prefill trace invalidation, exactly 127 effective post-prefill steps; CI has 3167 plus one, exactly 3168. Both windows have zero model-only replays, seed copies, and host sampling calls; all prohibited fallback flags are false. Primary bank commit/restore/reset deltas are zero. CI deltas are 32 assignments/releases, 3200 commits, 3167 restores, 32 resets, one prefill trace invalidation, and zero stale state.

Primary host deltas: device-slot hits/misses/evictions `24,868/42,796/42,796`; packed-host hits/misses `42,796/0`; physical-owner expert H2D 118,322,380,800 bytes in 9.501212 s; control/index H2D 219,800 bytes in 0.375017 s; 129 PLE lookups, 4,096 selected/unique rows, 2,621,440 logical/physical H2D bytes, 0.702702 s lookup and 0.012474 s H2D; 24.020644 s route-read plus TT stall; 0.039884 s trace submit.

CI host deltas: device-slot hits/misses/evictions `181,758/1,506,457/1,506,457`; packed-host hits/misses `1,506,457/0`; physical-owner expert H2D 4,165,052,313,600 bytes in 329.659205 s; control/index H2D 2,822,000 bytes in 6.247189 s; 3,201 PLE lookups, 101,904 selected/101,896 unique rows, 65,218,560 logical and 74,393,600 physical H2D bytes, 8.992924 s lookup and 0.363380 s H2D; 621.833877 s route-read plus TT stall; 0.962293 s trace submit.

Live cancellation/isolation:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/run_host_serving_lifecycle.py \
  --server-url http://localhost:8018 \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/host_serving_lifecycle.json \
  --server-log models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/server.log \
  --max-num-seqs 2 --physical-batch 1 --virtual-slot-capacity 2
```

Pass: both streams produced a token before client cancellation; survivor completed; two deterministic follow-ups matched; 152 canonical trace replays, zero model-only replays or host-sampling calls, four assignments/releases, eight commits, five restores, one reset, zero stale rejection, no active slot, and no PLE history remained.

Focused host/readiness contracts:

```bash
python_env/bin/python -m pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_vllm.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_virtual_decode_state_bank.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_readiness_vllm_scripts.py
```

Post-format result: 65 passed in 12.98 s. The state-contract map is explicit: `test_decode_reset_once_then_steady_async_delegation`, `test_generator_virtual_decode_microbatches_real_rows_and_ignores_padding`, `test_generator_virtual_decode_rejects_stale_generation_before_device_execution`, and `test_generator_multi_host_decode_preflights_later_stale_row` cover current-position, page-table row, stale-token/owner, and async behavior; `test_device_bank_copies_only_request_local_state` proves request-local token/current-position/page/recurrent copies without copying vLLM QSA KV. Adapter B2 host compatibility is covered by `test_generator_virtual_prefill_supports_multi_active_host_compatibility` and `test_generator_virtual_decode_supports_multi_active_host_compatibility`; plugin counterparts are `test_virtual_decode_forwards_only_unpadded_slot_metadata`, `test_stale_virtual_generation_cannot_apply_a_deferred_token`, and `test_two_host_only_prefill_rows_consume_stacked_torch_logits`.

Shared plugin non-TT regression after the seeded-policy repair: 169 passed, 8 skipped in 4.87 s. Focused TT artifacts include virtual B2 state/page/token parity, repeated non-aligned endpoint reuse, route-sparsity RM-first reuse, sparse FP32 CB sizing, and rank-local staging coordinate H2D.

After `SIGINT`, API shutdown completed and the vLLM process manager force-terminated its one remaining EngineCore child. The process-cleanup audit then found no vLLM/APIServer/EngineCore/test process, no holders on `/dev/tenstorrent/0` or `/dev/tenstorrent/1`, healthy DRAM, and live heartbeats. No reset was required.

Exact allocation-lifetime tracker closure:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 \
TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12 \
python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/final_b2_trace_tracker_fixed \
  --stages serve --mesh-device P300 --port 8019 --max-num-seqs 2 \
  --sampling-profile full --block-size 64 --max-model-len 262144 \
  --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' \
  --additional-server-args='--async-scheduling'
```

`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE` was unset. The historical pre-fix exact-B2 reproduction found 75 live buffers before replay: 73 replaced GDN/PLE recurrent or convolution state buffers and two program-cache buffers. Fixed state updates now retain construction-time addresses, and the exact program-cache boundary releases/recaptures traces on growth or uncertain initialization. `readiness_vllm/trace_allocation_tracker_b2_audit.json` passes with source hashes frozen before launch, active B2 overlap/cancellation/follow-ups, 150 replays, one safe prefill invalidation, zero tracker warnings/unsafe errors/runtime errors/tracebacks, and zero leftover processes/device holders. The ordinary frozen final `server.log` has one conservative active-trace allocation warning during startup/capture but no trace-tracker live-buffer/corruption error, traceback, or EngineCore failure; the exact unfiltered tracker run controls the safety verdict.

## Final evidence index

- serving/sampling: `readiness_vllm/server.log.gz`, `sampling_tests.log`, `non_aligned_prompt_check.json`, `stage_gate.log`.
- quality: `vllm_qualitative_outputs.json`, `qualitative_tt_chat.json`, `qualitative_prompt_format.json`, `QUALITATIVE_REVIEW.md`.
- performance: `vllm_result.json`, `vllm_benchmark.json`, `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json` and their `.log` files.
- host/runtime: `serving_host_metrics.json`, `host_serving_lifecycle.json`, `host_weight_cache_tests.log`, `process_cleanup_audit.json`.
- allocation lifetime: `trace_allocation_tracker_b2_audit.json`, `AUTOFIX_TRACE_ALLOCATION_LIFETIME.md`, `final_b2_trace_tracker_fixed/server.log.gz`, `final_b2_trace_tracker_fixed/host_serving_lifecycle.json`, and `final_b2_trace_tracker_fixed/process_cleanup_audit.json`.
- focused fixes: `autofix_virtual_b2_tt.xml`, `autofix_repeated_nonaligned_embedding.xml`, `autofix_reused_program_virtual_adapter.xml`, `autofix_route_sparsity_rm_first_tt.xml`, `autofix_sparse_matmul_fp32_cb.xml`, `autofix_rank_local_staging_tt.xml`, and the associated `autofix_*.md` reports.

Historical max-one and pre-fix trace-allocation artifacts are retained only as root-cause history. The current-source exact-B2 tracker audit above is authoritative for allocation lifetime. The authoritative capacity file is `readiness_vllm/max_num_seqs_limit.json`: the public adapter and final server are both capped at the two active full-model requests validated over the physical-B1 trace.

## Review and commits

The first frozen-source review returned `more-work-needed` because the adapter accepted active B8 while the full-model proof stopped at B2. The public adapter/test/readiness contract was narrowed to the validated B2 surface, the refreshed focused suite passed 65/65, and the hard stage gate passed again. A fresh xhigh rereview then returned `clean-pass` with no required work; the authoritative report is `doc/vllm_integration/STAGE_REVIEW.md`.

Local checkpoint SHAs are appended below after each repository commit. No push is performed.

- vLLM TT plugin: branch `housTT/register-muse-glimmer-30b`, commit `a48857ac68b17c31303e4809f348caaebbf10f74` (`Support Qwen3.8 virtual-slot vLLM serving`).
- tt-metal implementation/evidence checkpoint: recorded by the follow-up bookkeeping commit after the implementation commit is created.
