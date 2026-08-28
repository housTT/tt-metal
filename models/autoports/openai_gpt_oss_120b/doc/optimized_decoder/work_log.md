# GPT-OSS 120B optimized decoder work log

Status: optimized-decoder implementation and evidence complete after operator recovery. The full
final-topology 80-node matrix, reversed-order finalist, capacity 1/2/32,
integrated, watcher-only, marker-clean repeated profiler/roofline, and
post-health gates all ran. The prior blanket DRAM rejection and temporary
blocked status are superseded by the final split: BFP8/HiFi2 at configured
capacity 1 and BFP4/LoFi DRAM-QKV 15-core at capacities 2--32.

## Scope, parent, and target

- Model: `openai/gpt-oss-120b`, pinned revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Parent fused-decoder commit:
  `95e2665577e2511007f918d09b2dd4748eb261f7`.
- Authorized source: `tt/optimized_decoder.py`; authorized tests and evidence:
  `tests/test_optimized_decoder.py` and `doc/optimized_decoder/`.
- Hardware: four local p300c Blackhole boards, firmware 19.13.1, exposing four
  P150-class devices. The runtime mesh is intentionally `1x1`; P150x2/P150x4
  are host inventory targets, not multichip decoder implementations.
- This stage did not start multichip-decoder, full-model, or vLLM work.

The canonical hardware environment was:

```bash
env \
  -u TT_METAL_SIMULATOR \
  -u TT_METAL_SIMULATOR_HOME \
  -u TT_METAL_SLOW_DISPATCH_MODE \
  -u TT_METAL_DISABLE_SFPLOADMACRO \
  -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_WATCHER \
  -u TT_METAL_WATCHER_DUMP_ALL \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_METAL_HOME=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_CACHE=/home/ttuser/dev/gpt-oss-20b/tt-metal/.tt_metal_cache \
  TTNN_CONFIG_OVERRIDES='{"cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache","model_cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache"}' \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn \
  LD_LIBRARY_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  GPT_OSS_120B_TENSOR_CACHE=/tmp/gpt_oss_120b_functional_decoder_tensor_cache \
  PYTHONUNBUFFERED=1 \
  timeout 1800 scripts/run_safe_pytest.sh COMMAND
```

The safe wrapper serialized device use, ran pre/post health checks, and closed
the device on every completed run. Each retained transcript starts with the
expanded pytest command and contains its selectors and terminal result.

## Starting baseline

The best correct fused FullLocal baseline at sequence 128 was:

| Layer kind | Prefill wall / Tracy device | Decode wall, 500 replays / Tracy device | Ops / host ops |
| --- | ---: | ---: | ---: |
| sliding | 37.081505 / 36.220 ms | 0.719116 / 0.678 ms | 66/34, 0 host |
| full | 36.431644 / 35.558 ms | 0.728915 / 0.689 ms | 66/34, 0 host |

The fused real-weight correctness anchors were layer-0 sliding
0.978123157/0.990064440 prefill/decode PCC, layer-1 full
0.990890/0.955781, and layer-1 batch-2 FullLocal
0.992492223/0.992720098.

## Measured operation-topology audit

The audit preceded changes and used the fused Tracy reports plus a fresh
optimized control capture. Q/K/V were already one packed QKV projection,
expert gate/up were already packed, and decode executed only the top four
experts. A legal three-matmul Q/K/V alternative was added and measured to
justify keeping the same-input pack. The remaining material opportunities were
projection precision, bias folding, small-weight residency, norm sharding, and
dominant MoE geometry.

| Boundary | Measured starting graph | Candidate replacement | Action and evidence |
| --- | --- | --- | --- |
| Input norm | one-core/interleaved RMSNorm | L1 width-sharded RMSNorm | Applied ten-core `32x288` shards; grid sweep found 10 cores fastest. Batch-32 input-only exception is evidence-backed below. |
| QKV | packed BF16 DRAM matmul then bias add | BFP8/BFP4 weights, folded linear bias, separate Q/K/V, explicit fidelity/config, DRAM sharding | Capacity-selected BFP8 at batch 1 and BFP4/LoFi DRAM15 at multibatch applied. DRAM QKV's unsupported fused bias was repaired. The cumulative 20-row final-topology matrix ran for both layer kinds and batches; 15 cores was the fastest universally correct row. Three independently configured DRAM-sharded separate projections are legal, but the exact-attention shell measured 13.70--15.48% slower than packed. |
| Head/cache/attention | create packed heads, RoPE, BFP8 paged cache update, paged decode SDPA, concat heads | composite/SDPA or lower-movement replacement | Existing composite paged SDPA and head ops retained. Their required height-sharded boundaries are sub-2 us and no better public composite spans projection/cache mutation/SDPA. |
| Output projection | BF16 DRAM projection plus bias, cast BFP8 | BFP8/BFP4 weights, output activation BFP8, explicit 1D program and width-shard geometries | Capacity-selected BFP8/BFP4 weight retained. The first explicit config floor-covered only 64/90 N tiles and was invalid, not an accuracy result. Legal 64/32/16/8-core candidates all passed; 32 cores was promoted after improving both whole-layer traces by about 5.2%. |
| Post norm/router | interleaved RMSNorm; BF16 router in DRAM | sharded norm; persistent L1 router; explicit router config | ten-core norm and L1 router applied. Router matmul is only 6.9-7.0 us after residency; prefill router is 67 us, <0.25% of the path. |
| Decode experts | BFP4/LoFi FullLocal, four active experts, height shard 4, fused reduce | BFP4/LoFi geometry and output-shard sweeps | Inherited canonical precision retained. Height 2 is physically illegal; height 8 is correct but slower. |
| Prefill experts | unified routed expert composite | alternate primitive/packed graph | Existing composite retained: 89.00-90.65% of prefill and already 6.25x faster than the functional primitive implementation. |
| Runtime movement | three small sharding conversions and MoE route adapters | direct layout chaining / host fallback | No host fallback exists. Retained conversions satisfy RMSNorm, SDPA, and fused-MoE contracts. One additional device reshard from SDPA output to a 32-core width shard is profitable in whole-trace measurements and feeds the production output linear directly. |

## Final implementation

`OptimizedDecoder.from_state_dict` is a local constructor and never calls the
functional or fused constructor. The optimized component graph is asserted in
every reusable correctness harness:

- `_OptimizedAttention` owns decode QKV linear+bias folding, the existing
  paged cache/SDPA composite sequence, BFP8 attention weights at batch 1, and
  the production 32-core decode output projection;
- `_DecodeShardedRMSNorm` uses a `10x1` grid, width shard `(32, 288)`,
  `subblock_w=3`, `block_w=9`;
- `_L1Router` keeps BF16 router weight and bias in L1;
- `_OptimizedMLP` retains the exact-revision fused FullLocal BFP4/LoFi active
  expert path and the indexed fallback outside its qualified domain;
- automatic precision uses BFP8/HiFi2 at configured batch 1 and BFP4/LoFi at
  configured batch 2 through 32; selection uses `max_batch_size`, not a call's
  logical runtime batch. Explicit BFP8 and BF16 requests stay exact. Every
  policy shares the same `8x4`, `ibw4/pcn3/sb3` output-projection topology;
- attention tensor caches use separate `attention_bf16`, `attention_bfp8`, and
  `attention_bfp4` namespaces.

Paged KV cache stays BFP8 with page size 64. Allocation remains
`batch * ceil(context/page_size)` and the maximum context remains 131072.
No source or evidence required reducing `doc/context_contract.json`, so it is
unchanged.

## Candidate ledger

Whole-decoder wall numbers use the exact checkpoint, sequence 128, warmed
prefill, and captured decode replay. Correctness tests use the same public path
and unchanged PCC bars.

| Candidate | Correctness | Warmed prefill / traced decode | Decision |
| --- | --- | ---: | --- |
| BF16 attention control | correct | 35.120607 / 0.736451 ms sliding | Rejected: slower than fused decode. |
| BFP8 HiFi2 initial | layer-0 0.973633/0.995493 | 35.669842 / 0.722264 ms | Kept as precision base; not yet a final speedup. |
| Historical BFP4 HiFi2 attention before final 32c topology | random-activation layer-0 prefill 0.826763 | 35.634133 / 0.739606 ms | Diagnostic only; superseded by the exact prompt-derived final-topology matrix below. |
| Fold packed QKV bias into `ttnn.linear` | layer-0 0.973633/0.995096 | 35.844865 / 0.706474 ms | Applied; removes the separate decode bias add. |
| Persistent L1 router | layer-0 0.973261/0.995096 | 35.573716 / 0.695411 ms | Applied. |
| Decode sharded RMSNorm, initial 9-core | correct | 35.629121 / 0.588086 ms | Applied, then geometry swept. |
| RMSNorm 10-core | correct | 35.667910 / 0.564826 ms | Applied; best legal geometry. |
| RMSNorm 15-core | correct | 35.817654 / 0.571573 ms | Rejected: 1.19% slower than 10-core. |
| RMSNorm 30-core | correct | 35.732708 / 0.596151 ms | Rejected: 5.55% slower than 10-core. |
| Historical BFP8 LoFi projections before final 32c topology | layer-0 0.973261/0.993335 | 35.688295 / 0.577443 ms | Superseded by the cumulative final-topology matrix below. |
| Explicit QKV decode program config | layer-0 decode 0.992752 | 35.652666 / 0.564862 ms | Rejected: neutral/slower than automatic tuned config. |
| Explicit QKV + floor-covered output config | layer-0 decode 0.838096 | not timed | Invalid diagnostic: `per_core_N=90//64=1` covered only 64 of 90 N tiles. Replaced by the legal ceiling-covered sweep below. |
| Historical BFP8 output activation before final 32c topology | random-activation sliding 0.973261/0.992718; full decode 0.944517 | 0.564134 sliding / 0.576347 full | Superseded by the exact prompt-derived final-topology matrix below. |
| DRAM-sharded QKV before bias repair | layer-0 decode 0.913927-0.916125 | not timed | Invalid diagnostic: this TTNN matmul family does not correctly fuse bias. Repaired candidates below omit fused bias and add it on device after interleaving. |
| Separate BFP8 Q/K/V linears + concat | layer-0 0.973261/0.992752, identical to packed | 35.672645 / 0.584969 ms | Rejected: packed 0.564745 ms is 3.46% faster under the same 1000-replay harness. |
| FullLocal output height shard 2 | cannot allocate: 737280-byte CB exceeds 368640-byte L1 bank | not timed | Rejected on a hard physical limit. |
| FullLocal output height shard 8 | layer-0 decode 0.992752 | 0.565029 ms | Rejected: slower than inherited height 4. |
| Pre-remediation BFP8/HiFi2 + folded QKV + L1 router + ten-core norms | sliding 0.973261/0.992752; full 0.990511/0.957151 | 35.768797/0.564745 sliding; 29.972959/0.577178 full | Correct prior best; superseded by the 32-core production output projection. |
| Historical batch-2 BFP8 vs explicit BF16, both orders | sliding decode PCC 0.997050/0.997082; full 0.998446/0.998450 | sliding decode 0.800628--0.800679 / 0.857780--0.858143 ms; full 0.806238--0.806708 / 0.863633--0.863978 ms | Established BFP8 over BF16 before the cumulative final-topology sweep; both pass and remain bitwise equal after 1000 replays. |
| Final 32c BFP8 LoFi/output-activation cross-product | all exact prompt-derived rows pass unchanged bars and 1000 replay determinism | all three variants are 0.371--1.163% slower in decode than BFP8/HiFi2 | Rejected on directly comparable whole-trace latency. |
| Final 32c BFP4/HiFi2 | sliding 0.993719/0.993575; full 0.994044/0.993988 | 1.239--1.291% faster sliding; 7.352--7.407% faster full than BFP8 | Correct and competitive; superseded by BFP4/LoFi. |
| Final 32c BFP4/LoFi, BF16 output activation | sliding 0.993719/0.993575; full 0.994044/0.993988 | 1.480--1.561% faster sliding; 7.606--7.695% faster full than BFP8 | Promoted for configured batch 2 through 32. |
| Final 32c BFP4 output-activation variants | prefill 0.993719/0.994044; decode 0.993354/0.993927 | slower than matching BF16-output BFP4 rows | Rejected: lower PCC and no latency win over BFP4/LoFi without output quantization. |
| All-capacity BFP4/LoFi trial at batch 1 | finite and deterministic | 0.554860 ms sliding / 0.542234 ms full | Rejected for batch 1: sliding regressed 3.67% versus the reproduced 0.535201 ms BFP8 path. |

### Stage-review projection remediation

| Finding | Root cause | Repair and decisive evidence |
| --- | --- | --- |
| DRAM-QKV decode PCC 0.913927-0.916125 | Bias was fused into a DRAM-sharded matmul family whose bias path is broken. | Omit fused bias, convert the sharded result to interleaved DRAM, then in-place add bias before head creation. BFP8 90c decode PCC became 0.993434 in `optimized_autofix_dram_qkv_biasfix_bfp8_90c_correctness_sliding.log.gz`. |
| Explicit output decode PCC 0.838096 | Floor-derived `per_core_N=1` on 64 cores covered only 64/90 N tiles. | Add an explicit coverage validator and sweep ceiling-covered `per_core_N` values 2/3/6/12. Every legal row passed. |
| First legal 64c output run raised a TTNN validation error | The 1D factory requires `fuse_batch=True` when input A is sharded. | Enable batch fusion, preserve the width-sharded input, and rerun. Failure and passing retry are `optimized_autofix_oproj_bfp8_hifi2_64c_correctness_sliding.log.gz` and `optimized_autofix_oproj_bfp8_hifi2_64c_correctness_sliding_retry1.log.gz`. |

The DRAM-QKV bias repair was validated first with the nearest 90-core BF16 and
BFP8 controls. All rows below use real layer-0 weights; passing candidates were
timed over 1000 traced whole-decoder replays. A failing precision row was not
timed.

| DRAM QKV dtype/fidelity/geometry | Prefill/decode PCC | Warmed prefill / traced decode | Decision |
| --- | --- | ---: | --- |
| BF16/HiFi2, 90c `ibw1/pcn2` | 0.979107 / 0.972571 | 35.250080 / 0.616186 ms | Correct control, slower. |
| BFP8/HiFi2, 90c `ibw1/pcn2` | 0.973261 / 0.993434 | 35.692926 / 0.614718 ms | Bias repair verified; slower. |
| BFP8/LoFi, 90c `ibw1/pcn2` | 0.973261 / 0.993570 | 35.875343 / 0.605637 ms | Correct, slower. |
| BFP4/LoFi, 90c `ibw1/pcn2` | prefill 0.829411, below bar | not timed | Rejected on real weights. |
| BFP8/HiFi2, 45c `ibw2/pcn4` | 0.973261 / 0.993070 | 35.688312 / 0.597911 ms | Correct, slower. |
| BFP8/HiFi2, 30c `ibw3/pcn5` | 0.973261 / 0.989524 | 35.766500 / 0.592643 ms | Correct, slower. |
| BFP8/HiFi2, 15c `ibw6/pcn10` | 0.973261 / 0.991963 | 35.712646 / 0.587581 ms | Correct, slower. |
| BFP8/HiFi2, 10c `ibw9/pcn16` | 0.973261 / 0.993010 | 35.579755 / 0.568760 ms | Closest DRAM row, still 0.71% slower than 0.564745 ms. |

These rows are historical diagnostics, not the final DRAM decision. They used
random activations and an output path that predates the promoted 32-core
projection. The stage-review AutoFix replaced the source-defined matrix with
the complete cumulative cross-product:

- BFP8 and BFP4 attention weights;
- HiFi2 and LoFi decode projection math;
- 90/45/30/15/10-core DRAM QKV geometries;
- final `32c_ibw4_pcn3_sb3` output topology on every row;
- configured batch 1 and 2;
- exact checkpoint prompt-derived activations for sliding layer 0 and full
  layer 1;
- unchanged 0.95 prefill and 0.99 decode PCC bars, warmed prefill, and
  first/second plus post-1000 traced replay determinism;
- same-process latency against the capacity-selected automatic baseline.

The first execution attempt stopped before model construction during firmware
initialization. The safe wrapper captured triage and reset; two bounded resets,
both pair-pinned mesh smokes, and a recoverable stale-lock move established an
ERISC heartbeat infrastructure failure rather than a candidate failure. After
operator recovery, both paired inventories opened a 1x1 mesh and all 80 nodes
ran. Those initial failures reject no candidate. The recovery chronology,
complete matrix, and exact artifacts are in
`AUTOFIX_dram_qkv_final32c_matrix.md`.

The output projection has K/N dimensions 4096/2880, or 128/90 tiles. The
production 32-core row uses an `8x4` grid, input width shard `(32, 128)`,
`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `in0_block_w=4`,
`per_core_M=1`, `per_core_N=3`, `out_subblock_h/w=1/3`, `mcast_in0=True`, and
`fuse_batch=True`. The complete measured path includes the SDPA-output reshard.

| Output projection | Prefill/decode PCC | Warmed prefill / traced decode | Decision |
| --- | --- | ---: | --- |
| 64c `ibw2/pcn2/sb2` | 0.973261 / 0.992752 | 35.795665 / 0.561594 ms | Correct after repairing the first `fuse_batch` API error; slower than 32/16c. |
| 32c `ibw4/pcn3/sb3` | 0.973261 / 0.992055 | 35.626650 / 0.535187 ms | Best candidate; promoted. |
| 16c `ibw8/pcn6/sb6` | 0.973261 / 0.992520 | 35.791649 / 0.536127 ms | Correct, 0.18% slower than 32c. |
| 8c `ibw16/pcn12/sb6` | 0.973261 / 0.993205 | 35.780629 / 0.566041 ms | Correct, slower than the prior best. |
| Promoted 32c default, sliding | 0.973261 / 0.992055 | 35.609699 / 0.535223 ms | Final batch-1 production result; 5.23% below prior best. |
| Promoted 32c default, full | 0.990511 / 0.956693 | 30.277082 / 0.547673 ms | Final batch-1 production result; 5.11% below prior best. |

The first legal 64-core execution failed with TTNN's explicit
`fuse_batch must be enabled when input A is sharded` validation. Setting
`fuse_batch=True` addressed the stated contract, and the retained retry passed;
the first API error was therefore debugged rather than used as rejection
evidence. All DRAM and non-production output candidates remain private opt-in
reproduction policies and are omitted from `__all__`.

Exact new projection artifacts, relative to `evidence/logs/`, are:

- BF16 90c control:
  `optimized_autofix_dram_qkv_biasfix_bf16_90c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_biasfix_bf16_90c_perf1000_sliding.log.gz`;
- BFP8 90c control:
  `optimized_autofix_dram_qkv_biasfix_bfp8_90c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_biasfix_bfp8_90c_perf1000_sliding.log.gz`;
- BFP8/LoFi and BFP4/LoFi 90c:
  `optimized_autofix_dram_qkv_bfp8_lofi_90c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_lofi_90c_perf1000_sliding.log.gz`, and
  `optimized_autofix_dram_qkv_bfp4_lofi_90c_correctness_sliding.log.gz`;
- BFP8/HiFi2 geometry rows:
  `optimized_autofix_dram_qkv_bfp8_hifi2_45c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_45c_perf1000_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_30c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_30c_perf1000_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_15c_correctness_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_15c_perf1000_sliding.log.gz`,
  `optimized_autofix_dram_qkv_bfp8_hifi2_10c_correctness_sliding.log.gz`, and
  `optimized_autofix_dram_qkv_bfp8_hifi2_10c_perf1000_sliding.log.gz`;
- output 64c failure, retry, and timing:
  `optimized_autofix_oproj_bfp8_hifi2_64c_correctness_sliding.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_64c_correctness_sliding_retry1.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_64c_perf1000_sliding.log.gz`;
- output 32/16/8c correctness and timing:
  `optimized_autofix_oproj_bfp8_hifi2_32c_correctness_sliding.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_32c_perf1000_sliding.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_16c_correctness_sliding.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_16c_perf1000_sliding.log.gz`,
  `optimized_autofix_oproj_bfp8_hifi2_8c_correctness_sliding.log.gz`, and
  `optimized_autofix_oproj_bfp8_hifi2_8c_perf1000_sliding.log.gz`;
- promoted default gates:
  `optimized_autofix_promoted_32c_default_correctness_sliding.log.gz`,
  `optimized_autofix_promoted_32c_default_correctness_full.log.gz`,
  `optimized_autofix_promoted_32c_default_perf1000_sliding.log.gz`, and
  `optimized_autofix_promoted_32c_default_perf1000_full.log.gz`;
- health:
  `optimized_autofix_projection_controls_post_tt_smi.log.gz`,
  `optimized_autofix_dram_qkv_geometry_sweep_post_tt_smi.log.gz`,
  `optimized_autofix_oproj_geometry_sweep_post_tt_smi.log.gz`, and
  `optimized_autofix_promoted_32c_batch1_post_tt_smi.log.gz`.

The BFP4/LoFi requirement was evaluated at both attention and dominant expert
boundaries. Expert BFP4/LoFi was already the fastest qualified fused baseline;
this stage swept its material output shard geometries. Synthetic/random
failures were treated only as diagnostics. Production precision decisions use
exact real checkpoint weights.

## Correctness and context gates

| Gate | Result | Exact artifact |
| --- | --- | --- |
| Pre-DRAM capacity-selected integrated module | 12 passed, 45 explicitly gated skips; exact snapshot; automatic BFP8 batch 1 and non-DRAM BFP4/LoFi batch 2; both real-activation kinds; 1000 replays | `optimized_autofix_attention_precision_capacity_auto_integrated_final.log.gz` |
| Final automatic batch-1 BFP8 performance/determinism | sliding 35.637099/0.535201 ms; full 29.937398/0.547660 ms; 1000 replay deterministic | `optimized_autofix_attention_precision_capacity_auto_batch1_bfp8_retry_final.log.gz` |
| Rejected all-capacity BFP4/LoFi batch-1 trial | sliding 35.560438/0.554860 ms; full 35.104688/0.542234 ms; sliding regresses 3.67% | `optimized_autofix_attention_precision_bfp4_lofi_production_batch1_perf.log.gz` |
| Final prompt-derived automatic BFP4/LoFi vs explicit BFP8, automatic first | sliding decode 0.993575/0.997050, BFP8 cost 1.528%; full 0.993988/0.998446, cost 8.178%; both post-loop bitwise deterministic | `optimized_autofix_attention_precision_capacity_auto_batch2_automatic_then_bfp8_final.log.gz` |
| Final prompt-derived automatic BFP4/LoFi vs explicit BFP8, BFP8 first | sliding decode 0.993575/0.997050, BFP8 cost 1.571%; full 0.993988/0.998446, cost 8.267%; both post-loop bitwise deterministic | `optimized_autofix_attention_precision_capacity_auto_batch2_bfp8_then_automatic_final.log.gz` |
| Final automatic BFP4/LoFi batch 32 capacity/determinism | both layer kinds pass at 131072 context/user; replay difference count 0 | `optimized_autofix_attention_precision_capacity_auto_batch32_bfp4_lofi_final.log.gz` |
| Watcher-only final multibatch production correctness | both prompt-derived layer kinds pass unchanged bars and 1000 replay determinism; no watcher error | `optimized_autofix_attention_precision_capacity_auto_batch2_watcher_only_final.log.gz` |
| Heavy `--dev` watcher diagnostic | paged SDPA cannot compile: 80896--82032-byte instrumented program exceeds 70656-byte TENSIX config buffer | `optimized_autofix_attention_precision_capacity_auto_batch2_watcher_final.log.gz` |
| Final post-run health | all four p300c devices enumerated and reset-capable | `optimized_autofix_attention_precision_capacity_auto_final_tt_smi.log.gz` |
| Promoted 32c real representative layers | sliding 0.973260504/0.992054854; full 0.990510676/0.956693412 | `optimized_autofix_promoted_32c_default_correctness_{sliding,full}.log.gz` |
| Historical synthetic automatic-BF16 control | sliding 0.984148544/0.990688969; full 0.991103757/0.996522189; superseded as a policy selector | `optimized_autofix_batch2_synthetic_automatic_bf16_final.log.gz` |
| Opt-in synthetic BFP8 rejection diagnostic | sliding 0.979787472/0.992090839 passes; full 0.989791339/0.983495361 fails the unchanged decode bar; diagnostic only | `optimized_autofix_batch2_synthetic_explicit_bfp8_rejected_final.log.gz` |
| Opt-in exact-weight/random-activation BFP8 rejection diagnostic | decode 0.975027444 fails the unchanged 0.99 bar; diagnostic only | `optimized_final_batch2_real_full_local.log.gz` |
| Historical real FullLocal explicit-BF16 control | 0.992492223/0.992070182 | `optimized_autofix_batch2_real_weight_automatic_bf16_final.log.gz` |
| Historical real prompt-derived BFP8 vs explicit-BF16 A/B, BFP8 first | sliding decode 0.997049920/0.997081894, BF16 cost 7.132%; full 0.998445515/0.998449670, cost 7.056%; both post-loop bitwise deterministic | `optimized_autofix_batch2_production_bfp8_vs_bf16_automatic_first_final.log.gz` |
| Historical real prompt-derived BFP8 vs explicit-BF16 A/B, BF16 first | sliding decode 0.997049920/0.997081894, BF16 cost 7.184%; full 0.998445515/0.998449670, cost 7.162%; both post-loop bitwise deterministic | `optimized_autofix_batch2_production_bfp8_vs_bf16_bf16_first_final.log.gz` |
| Promoted-32c batch-32 regression | both layers allocated full context but six rows changed across trace replay | `optimized_autofix_batch32_promoted_32c_nondeterminism.log.gz` |
| Historical automatic-BFP8 batch 32 capacity/determinism | omitted request then resolved BFP8; both layer kinds pass at 131072 context/user; replay difference count 0 | `optimized_autofix_batch32_automatic_bfp8_capacity_final.log.gz` |
| Final focused repository hooks | all applicable hooks pass, including the `expect_error` policy | `optimized_autofix_production_bfp8_final_precommit.log.gz` |
| Historical host policy/cache and syntax checks | omitted capacity 1/2/32 then resolved BFP8; explicit BF16 exact; dtype cache namespaces distinct | `optimized_autofix_production_bfp8_final_host_checks.log.gz` |
| Final device health | all four p300c devices enumerated and reset-capable | `optimized_autofix_production_bfp8_final_tt_smi.log.gz` |
| Non-aligned tile/page/window lengths | 20 cases pass across both kinds | `optimized_final_non_aligned_boundaries.log.gz` |
| Chunk boundaries 4095/4096/4097 | both kinds pass | `optimized_final_chunk_boundaries.log.gz` |
| Max prefill 131071/131072 | both kinds pass | `optimized_final_max_prefill.log.gz` |
| Decode at context position 131071 | both kinds pass | `optimized_final_advertised_context_decode.log.gz` |
| Repeated performance/determinism | both kinds pass, 1000 trace replays | `optimized_final_source_stress_perf_1000.log.gz` |
| Promoted 32c repeated performance/determinism | sliding 0.535223344 ms; full 0.547672939 ms; both bitwise deterministic over 1000 replays | `optimized_autofix_promoted_32c_default_perf1000_{sliding,full}.log.gz` |
| Watcher real correctness | both kinds pass, no watcher error | `optimized_final_watcher_real_correctness.log.gz` |

All artifact paths in tables are relative to `doc/optimized_decoder/evidence/logs/`.
The module's static tests additionally prove distinct construction, optimized
component types, policy selection, and dtype-specific attention cache keys.

## AutoFix record

The first batch-32 optimized trace was nondeterministic while the fused control
passed. AutoDebug reports are retained as `AUTODEBUG.md` and
`AUTODEBUG_round2.md`. AutoFix isolated one component per run:

1. DRAM router instead of L1 router: still failed; H1 refuted.
2. Canonical post-attention norm only: still failed; H2 refuted.
3. Canonical QKV only: still failed; H3 refuted.
4. All canonical components within the optimized constructor: bitwise pass,
   establishing a positive control.
5. Canonical input norm only: bitwise pass while optimized attention, post
   norm, MLP, and router remained selected; hypothesis verified.

The first production fix disabled input-norm decode sharding only when
configured batch is a full tile (`>=32`). Both layer kinds then passed batch 32
with zero differing elements, while batch 1/2 retained the fast ten-core path.

A subsequent batch-2 synthetic diagnostic exposed ambiguity between omitted
and explicitly requested BFP8 policies. `AUTODEBUG_batch2_policy.md` identified
the default policy object being used as an automatic sentinel. The fix uses
`None` exclusively for automatic selection, preserves `requested_policy` for
audit, and stores the resolved policy separately. Explicit BFP8 or BF16
requests remain exact and receive dtype-specific cache namespaces.

Stage review then requested real activations and a batch-2 precision/latency
A/B. The delivered production gate pins two ordinary 33-token text sequences,
reads their exact checkpoint embedding rows, and runs the exact HF layer 0 to
obtain true layer-1 prefill/decode inputs at positions 26 and 16; layer 0 uses
the exact embedding rows directly at positions 8 and 32. It compares omitted
policy (then requested automatic, effective BFP8) with explicit BF16 in both orders.
Both policies pass the unchanged 0.95/0.99 bars, first/second replay equality,
and bitwise equality of the post-1000-loop output with the first replay.
Sliding BFP8 was 0.800628--0.800679 ms and BF16 was 0.857780--0.858143 ms;
full-layer BFP8 was 0.806238--0.806708 ms and BF16 was
0.863633--0.863978 ms. This established BFP8 over BF16 before the cumulative
final-topology precision matrix below.

The exact-weight/random-activation BFP8 result of 0.975027444 and promoted
synthetic full-layer result of 0.983495361 remain opt-in rejected diagnostics
at the unchanged 0.99 bar. They attest a distribution sensitivity but do not
select production precision: the optimization contract explicitly prohibits
synthetic/random PCC from vetoing a real-weight win. BF16 remains available as
the slower explicit control. The final default integrated suite directly runs
both real-activation layer kinds with 1000 replays and leaves both diagnostics
skipped.

The final attention-precision AutoFix reran the complete non-DRAM cross-product
on the promoted 32-core output topology: BFP8/BFP4 weights, HiFi2/LoFi decode
projection math, and BF16/BFP8 output activations. Every candidate ran exact
checkpoint prompt-derived inputs for sliding layer 0 and full layer 1 in both
orders against the then-automatic BFP8 baseline. All 28 nodes passed unchanged
PCC bars and remained bitwise equal after 1000 trace replays. BFP8 LoFi and/or
output-activation variants were 0.371--1.163% slower. All BFP4 rows passed;
BFP4/LoFi without output quantization was fastest at 1.480--1.561% below BFP8
sliding and 7.606--7.695% below BFP8 full attention. Its exact prefill/decode
PCC was 0.993718903/0.993574877 sliding and
0.994043785/0.993987661 full.

An all-capacity BFP4/LoFi trial then measured batch-1 sequence-128 decode at
0.554860 ms sliding and 0.542234 ms full. Sliding regressed 3.67% relative to
the reproduced BFP8 result of 0.535201 ms, so the pre-DRAM automatic policy at
that checkpoint became configured-capacity based: BFP8/HiFi2 at capacity 1 and
non-DRAM BFP4/LoFi at capacity 2 through 32. It was never selected from a
call's smaller logical batch. That batch-2 two-order A/B measured BFP4 at 0.788169--0.788481 ms sliding and
0.744771--0.745105 ms full versus BFP8 at 0.800528--0.800549 ms and
0.806037--0.806344 ms. Batch 32 retained the full 131072-token-per-user cache
and zero replay differences for both layer kinds.

Watcher was kept separate from latency. The heavy `--dev` preset cannot
compile paged SDPA because lightweight/LLK assert instrumentation grows the
program to 80896--82032 bytes beyond the 70656-byte TENSIX kernel-config
buffer. A watcher-only rerun, with polling watcher and NoC/CB checks but without
the extra assert instrumentation, passed the exact multibatch production gate
for both layer kinds and 1000 replays with no watcher error.

The projection AutoFix then resolved two separate invalid diagnostics. First,
DRAM-sharded QKV had attempted to fuse bias in a TTNN matmul family with a
known broken bias path. Moving bias to an in-place device add after
sharded-to-interleaved conversion raised 90-core BFP8 decode PCC from 0.916125
to 0.993434. The historical dtype/fidelity/geometry sweep above then found its
tested rows slower, but that is not a final rejection: those rows predate the
promoted output topology and exact-activation acceptance harness. Second, the
old explicit output program used floor division for N coverage and produced
only 64 of 90 tiles. Legal ceiling-covered configs all passed. The 32-core
candidate won the whole-trace sweep and was applied to every production and
explicit precision policy so comparisons retain identical topology.

Promotion exposed one final capacity-only lifetime interaction: at configured
batch 32, the new width-sharded output projection followed by the sharded
post-attention RMSNorm changed six logical rows between trace replays in both
layer kinds. Keeping the 32-core projection but using the canonical
post-attention norm at the full-tile boundary restored bitwise equality. The
the historical automatic-BFP8 gate passed both layer kinds with full
131072-token-per-user cache allocation; batch 1/2 retain both ten-core sharded
norms. The final BFP4/LoFi batch-32 rerun also passed unchanged.

The exact hardware command was:

```bash
GPT_OSS_120B_BATCH2_POLICY_AB_REPEATS=1000 \
GPT_OSS_120B_BATCH2_POLICY_AB_ORDER=automatic,bfp8 \
scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_activation_batch_two_precision_and_latency_ab \
  -q
```

The same command was repeated with
`GPT_OSS_120B_BATCH2_POLICY_AB_ORDER=bfp8,automatic`. The final batch-32 gate
used the same safe wrapper with `GPT_OSS_120B_RUN_BATCH32=1` on
`test_optimized_batch_32_paged_prefill_and_traced_decode_capacity`. The final
integrated command ran the entire test file with the exact snapshot and
`GPT_OSS_120B_BATCH2_POLICY_AB_REPEATS=1000`; it passed 12 tests and skipped 45
opt-in gates. One post-promotion batch-1 retry encountered a transient SIGBUS
before any model result; the immediate health probe could not map device 0, so
one whole-system reset was performed. All four devices recovered, the identical
retry passed, and final health enumerated every board as reset-capable. The
failed transcript, reset, health checks, and passing retry are retained.

## Final performance and profiler results

The final uninstrumented command was:

```bash
GPT_OSS_120B_OPTIMIZED_PERF=1 \
GPT_OSS_120B_OPTIMIZED_PERF_REPEATS=1000 \
scripts/run_safe_pytest.sh -q \
  'models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_weight_warmed_prefill_and_traced_decode_performance[blackhole-1x1-attention_bfp8_hifi2-sliding]' \
  'models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_weight_warmed_prefill_and_traced_decode_performance[blackhole-1x1-attention_bfp8_hifi2-full]'
```

| Layer kind | Fused wall | Optimized wall | Change | Replays |
| --- | ---: | ---: | ---: | ---: |
| sliding | 0.719116 ms | 0.535223344 ms | 25.57% lower | 1000 |
| full | 0.728915 ms | 0.547672939 ms | 24.86% lower | 1000 |

Prefill was 35.609699 ms sliding and 30.277082 ms full. Both outputs were
bitwise deterministic over the measurement loop. This is the final measured
source, including the added SDPA-output reshard, not an intermediate cleaner
topology. Compared with the fused warmed wall baseline above, prefill is 3.97%
lower for sliding and 16.89% lower for full attention. Relative to the prior
best correct optimized decoder, decode is 5.23% lower sliding and 5.11% lower
full.

The final-source Tracy capture was run for ten replays per exact automatic
capacity-1 node with watcher unset. The two raw CSVs came from report
directories `2026_08_28_18_38_45` and `2026_08_28_18_39_58`:

```bash
GPT_OSS_120B_OPTIMIZED_PERF=1 \
GPT_OSS_120B_OPTIMIZED_PERF_REPEATS=10 \
scripts/run_safe_pytest.sh --profile \
  'models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_weight_warmed_prefill_and_traced_decode_performance[NODE_ID]' \
  -q -s
```

The exact nodes were `blackhole-1x1-automatic-sliding` and
`blackhole-1x1-automatic-full`. Each raw capture contains one
`PERF_PREFILL...END` interval and ten complete decode iterations inside
`PERF_DECODE...END`. A preceding 100-replay sliding attempt passed correctness
and printed 0.549525 ms wall, but profiler DRAM buffers filled and markers were
dropped. It was rejected from all device/roofline accounting and retained as
`optimized_final_same_run_profile_sliding_repeats100_overflow.log.gz`.
The safe wrapper printed a terminal warning because this profiler version
names the file `ops_perf_results_<timestamp>.csv` rather than the wrapper's
older exact filename. Both generated paths are present in the transcripts,
were losslessly compressed into `tracy/{sliding,full}/ops_perf_results.csv.gz`,
and successfully produced every report below.

Reports were produced for both intervals with and without advice:

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --csv prefill_perf_report.csv --summary-file prefill_summary

/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --active-experts 4 --csv decode_perf_report.csv \
  --summary-file decode_summary
```

The `_no_advice.csv` variants add `--no-advice --no-color`.

| Layer kind | Prefill device / ops / host | Decode device / ops / host |
| --- | ---: | ---: |
| sliding | 34.982222 ms / 66 / 0 | 0.504847 ms / 36 / 0 |
| full | 29.655483 ms / 66 / 0 | 0.517103 ms / 36 / 0 |

Sliding prefill is 90.64% unified routed experts, 3.84% sort, and 1.44%
embeddings. Full prefill is 89.00%, 4.53%, and 1.69%. Sliding decode is 45.14%
FullLocal MoE, 20.60% three matmuls, 8.85% head creation, 4.39% TopK, 4.34%
SDPA, 3.22% cache updates, and 2.32% norms. Full decode is materially the same.
The production SDPA-output reshard is one of the 36 decode ops and costs only
1.45/1.49 us on average for sliding/full; its downstream 32-core projection produces a
net end-to-end win.

### Decode bytes/token and same-run three-number roofline

The P150 peak used by `tt-perf-report` is 512 GB/s; for example, raw projection
rows label about 397 GB/s as 77.5% of peak. A compulsory physical-traffic lower
bound for one batch-1 layer-token at the measured context 128 is:

| Component | Physical TT-tile derivation | Bytes |
| --- | --- | ---: |
| Packed QKV BFP8 weight | `90 * 160 * 1088` | 15,667,200 |
| Output projection BFP8 weight | `128 * 90 * 1088` | 12,533,760 |
| Router BF16 weight | `90 * 4 * 2048` | 737,280 |
| Four active FullLocal BFP4 experts | `4 * ((8*6*98*4*576) + (8*3*98*4*576))` | 65,028,096 |
| K/V read at context 128 | `2 * 8 * 4 * 2 * 1088` | 139,264 |
| K/V physical tile write | `2 * 8 * 1 * 2 * 1088` | 34,816 |
| **Minimum total** | sum | **94,140,416** |

The expert formula is the actual eight-position FullLocal ring allocation:
six four-tile W0/W1 groups and three four-tile W2 groups per position, with the
bias dimension padded from 91 to 98 tiles. TT BFP8/BFP4 physical tiles are
1088/576 bytes including headers. Internal activation spill/reloads are not
counted, so this is a conservative lower bound, not a claim of exact traffic.

The measured device and wall values below come from each layer kind's same
marker-clean ten-replay profile invocation. Device time sums all 360 decoded
operation rows and divides by ten; wall is the test timer around the same ten
trace replays. The gap therefore has a valid common accounting scope.

| Layer kind | Theoretical at 512 GB/s | Tracy device | Same-run wall | Wall-device gap | Effective compulsory GB/s, device/wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding | 0.183868 ms | 0.504847 ms (2.75x) | 0.555876 ms (3.02x) | 0.051029 ms (9.18% wall) | 186.5 / 169.4 |
| full | 0.183868 ms | 0.517103 ms (2.81x) | 0.567259 ms (3.09x) | 0.050155 ms (8.84% wall) | 182.1 / 166.0 |

The same-run wall includes profiler/runtime overhead and does not replace the
uninstrumented 1000-replay latency used for candidate selection. The report's
separate 52-53 GB/s number is not whole-decoder bandwidth:
the dominant `MoEComputeDeviceOperation` has blank DRAM/FLOP model fields, so
the aggregate model omits it. Exact machine-readable component formulas and
same-run comparison rows are in `tracy/decode_roofline.csv`.

The raw profiler labels the outer `MoEComputeDeviceOperation` as HiFi4 even
though the FullLocal compute kernel factory hard-codes LoFi. This is an outer-op
metadata default: the row's kernel source/hash points to `compute.cpp`, whose
program factory sets `MathFidelity::LoFi`; it is not evidence that the expert
matmuls ran at HiFi4. The BFP4 weight dtype remains visible in the input tensor
metadata.

`tt-perf-report` advice was reconciled as follows:

- decode reports say the automatic QKV/output block and subblock choices look
  good. HiFi4 is suggested only for additional accuracy; HiFi2 already clears
  real-weight bars, and the measured LoFi candidate is 2.23% slower;
- prefill suggests placing projection/router input 0 in L1. Sequence-128
  activation and projection working sets cannot stay wholly in L1. The
  inherited large prefill configs already use the reported good block shapes;
- the prefill router has no explicit program config, but its entire 67 us
  matmul is only 0.23-0.30% of the 29.5-35.0 ms path. Even eliminating it would
  be below the run-to-run wall delta, so it is not an actionable material
  candidate;
- BFP4, LoFi, output activation, explicit configs, DRAM-sharded QKV, and MoE
  shard geometry all have before/after correctness or latency evidence above.

There is no `torch`, `from_torch`, `to_torch`, or host fallback inside measured
prefill/decode. Host tensor creation occurs only in the test harness before
capture. Remaining tilize/untilize and reshard ops are the device-side routing,
fused-MoE, norm, and attention contract adapters already accounted for in the
tables; none is an unmeasured fallback.

## Device and platform qualification

- Pre-run and post-run `tt-smi -s` found four p300c boards with healthy DRAM,
  firmware 19.13.1, and no workload owner after completion.
- All final tests used `TT_VISIBLE_DEVICES=0,1,2,3`; the `1x1` test fixture
  selected one P150-class device from the P150x4 host inventory.
- A separate `TT_VISIBLE_DEVICES=0,1` P150x2-host smoke passed exact advertised
  context decode.
- This lab exposes each physical p300c as a paired fabric cluster. Restricting
  visibility to only device 0 makes the board a `CUSTOM` cluster before pytest
  collection and requires an external custom fabric graph; the decoder never
  ran in that attempt. No unverified graph was invented. Single-P150 decoder
  coverage is supplied by the same `1x1` mesh selected from the healthy x2/x4
  inventories.
- Watcher and profiler were run separately. Two bounded device-reset attempts
  preceded operator firmware recovery, and one later whole-system reset
  recovered a transient post-promotion SIGBUS; neither event was attributed to
  a retained runtime candidate.
- The final same-run-profile `tt-smi -s` found all four boards at 35-37 C, healthy
  DRAM on all eight channels, zero corrected/uncorrected GDDR errors, and no
  workload process other than the inventory command itself.

Exact transcripts:
`optimized_final_p150x2_host_smoke.log.gz`,
`optimized_final_p150_host_smoke.log.gz`, and
`optimized_final_same_run_profile_post_tt_smi.log.gz`.
The exact final watcher, integrated-module, and profiler transcripts are
`optimized_autofix_dram_qkv_final32c_final_split_batch2_watcher.log.gz`,
`optimized_autofix_dram_qkv_final32c_final_split_integrated.log.gz`, and
`optimized_final_same_run_profile_{sliding,full}_repeats10.log.gz`.

## Final-topology DRAM-QKV AutoFix

Stage review found that the earlier DRAM table was not cumulative: its policies
omitted the final 32-core output geometry and its precision/fidelity/geometry
cross-product was incomplete. The repaired source defines BFP8/BFP4 x
HiFi2/LoFi x 90/45/30/15/10 cores, with every row feeding
`32c_ibw4_pcn3_sb3`. The exact matrix gate uses prompt-derived checkpoint
activations, both representative layer kinds, configured batch 1 and 2,
unchanged 0.95/0.99 bars, first/second and post-1000 replay equality, and
same-process automatic/candidate timing.

After an operator recovered the boards, four serialized pair-pinned groups ran
all 80 nodes. BFP4/LoFi 15-core was the fastest universally correct candidate:

| Order | Batch 1 sliding/full | Batch 2 sliding/full | Result |
| --- | ---: | ---: | --- |
| automatic,candidate | -4.724% / -4.712% | -1.458% / -2.561% | all PCC/determinism pass |
| candidate,automatic | -4.736% / -4.727% | -1.368% / -2.480% | strict reversed-order win |

Every delta exceeded 1%, so the three-order median rule did not apply. The
10-core BFP4/HiFi2 and BFP4/LoFi rows were rejected at batch-1 sliding decode
PCC 0.989157931; BFP8 policies were not strict all-kind winners; BFP4/LoFi
45/30-core passed but lost to 15-core. Exact per-policy ranges are in
`AUTOFIX_dram_qkv_final32c_matrix.md`.

The required production sequence-128 gate prevented an overbroad promotion:
capacity-1 DRAM15 sliding was 0.541827 ms versus the established correct BFP8
0.535201 ms, although full improved to 0.528912 ms. Capacity 1 therefore stays
BFP8/HiFi2. At capacity 2, DRAM15 measured 0.778263 ms sliding and 0.727294 ms
full in the final integrated suite, improving the former non-DRAM BFP4/LoFi
0.788169--0.788481/0.744771--0.745105 ms. Capacities 2--32 therefore use the
promoted BFP4/LoFi DRAM15 policy.

Final gates and artifacts:

| Gate | Result | Artifact under `evidence/logs/` |
| --- | --- | --- |
| Four 20-candidate matrix groups | 78 pass; two batch-1 sliding 10-core PCC failures | `optimized_autofix_dram_qkv_final32c_matrix_batch{1,2}_{sliding,full}.log.gz` |
| Reversed-order DRAM15 finalist | 4 passed; strict wins | `optimized_autofix_dram_qkv_final32c_finalist_15c_candidate_first.log.gz` |
| Final capacity-1 automatic | BFP8 0.535430/0.547912 ms, deterministic | `optimized_autofix_dram_qkv_final32c_final_split_batch1.log.gz` |
| Promoted capacity-2 | 2 passed, unchanged PCC/determinism | `optimized_autofix_dram_qkv_final32c_promoted_batch2.log.gz` |
| Promoted capacity-32 | both kinds, full allocation, zero replay differences | `optimized_autofix_dram_qkv_final32c_promoted_batch32.log.gz` |
| Integrated final split | 12 passed, 125 gated skips | `optimized_autofix_dram_qkv_final32c_final_split_integrated.log.gz` |
| Watcher-only capacity-2 | 2 passed; no watcher error | `optimized_autofix_dram_qkv_final32c_final_split_batch2_watcher.log.gz` |
| Final health | four local boards enumerate | `optimized_autofix_dram_qkv_final32c_final_split_post_tt_smi.log.gz` |

The source/test promotion changes are host-checked with `py_compile`, focused
policy/cross-product tests, `git diff --check`, and repository pre-commit. No
context capacity, KV dtype, public length contract, or advertised capability
changed, so `doc/context_contract.json` remains unchanged.

## Optimize checklist

- [x] Audit measured topology for repeated same-input matmuls, conversions,
  packed projections, composites, and lower-movement replacements.
- [x] Establish fused correctness, warmed wall, Tracy device, op-count, and
  host-op baselines.
- [x] Keep QKV and gate/up packed. The corrected cumulative control proves
  that three distinct DRAM-sharded Q/K/V program configs are legal, then
  measures their exact-attention shell 13.70--15.48% slower than packed for
  both capacities and layer kinds; anchored whole estimates are
  3.526--6.377% slower. Gate/up retains the fused stage's previously measured
  packed win.
- [x] Use canonical per-tensor precision/fidelity policy and dtype-specific
  tensor caches.
- [x] Sweep BF16/BFP8/BFP4 attention and HiFi2/LoFi compute with real weights.
- [x] Retain BFP4/LoFi FullLocal experts and sweep material expert shard
  geometries.
- [x] Add profitable L1 sharding and sweep 9/10/15/30-core norm geometries.
- [x] Repair DRAM-QKV bias handling, then test BF16/BFP8/BFP4,
  HiFi2/LoFi, and five legal 90/45/30/15/10-core geometries. Reject only after
  real-weight correctness and whole-trace latency evidence.
- [x] Diagnose floor-covered output N as invalid, then sweep legal
  ceiling-covered 64/32/16/8-core configs. Repair the first `fuse_batch` API
  error and promote the best 32-core result.
- [x] Retain inherited large prefill program configs and reconcile all
  `tt-perf-report` advice.
- [x] Retain paged SDPA/composite ops and fused active-expert MoE; no applicable
  lower-movement public replacement is left untried.
- [x] Audit runtime movement, output dtypes, memory configs, and trace capture;
  final measured path has zero host ops/fallbacks.
- [x] Preserve non-aligned lengths, batch 2/32, BFP8 paged cache, deterministic
  replay, and full 131072-token context.
- [x] Run real representative layer kinds, stress, watcher, and post-run health
  checks.
- [x] Capture final sliding/full Tracy and generate prefill/decode CSV reports,
  advice tables, grouped summaries, and plots.
- [x] Reproduce the promoted final source at 1000 replays and beat both the
  fused baseline and the previous best correct optimized candidate on both
  layer kinds.
- [x] Record non-applicable items: collectives/fused CCL and persistent
  multidevice buffers belong to multichip-decoder; LM-head optimization belongs
  to full-model. Neither was started here.

## Repository verification and artifacts

Python/docs-only changes do not require a C++ build under the repository
`AGENTS.md`. The final verification set is:

```bash
scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py -q

python_env/bin/python -m black --check \
  models/autoports/openai_gpt_oss_120b/tt/optimized_decoder.py \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py

python_env/bin/python -m py_compile \
  models/autoports/openai_gpt_oss_120b/tt/optimized_decoder.py \
  models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py
```

The post-repair integrated result is 8 passed and 183 opt-in skips. Expensive gates
were run separately as listed in the correctness table. The repository
pre-commit suite passed on the two Python files and all stage Markdown; its
managed isort hook passed even though there is no standalone host `isort`
executable. `py_compile` and direct host assertions for automatic BFP8 at
configured capacity 1, automatic BFP4/LoFi at capacities 2/32, explicit
BFP8/BF16 identity, and distinct dtype cache namespaces also passed.

`tracy/{sliding,full}/` contains losslessly compressed raw ops CSV, uncompressed
advice/no-advice phase CSVs, and summary CSV/PNG files. `evidence/logs/`
contains deterministic gzip transcripts including the full final-32-core
precision matrix, two-order capacity-selected A/B, automatic-BFP4 batch-32
gate, watcher-only run, default integrated suite, transient initialization
diagnostic/recovery, and final health evidence. `artifacts.sha256` is regenerated
after all final evidence is integrated and passes `sha256sum -c`.

## 2026-08-28 final-topology stage-review AutoFix

The direct exact-prompt capacity-2 DRAM10/DRAM15 comparison ran three A/B
replicates (15-first, 10-first, 15-first), both layer kinds, final32, and 1000
replays. DRAM10's equal-kind median was 0.484% faster, but the follow-up valid
logical-batch-1 call on a configured-capacity-2 decoder failed sliding decode
PCC at 0.989158. DRAM15 passed at 0.990655 and remains production. This
follow-up found and repaired configured-capacity K/V input sharding: decode now
creates one K/V height shard per explicit runtime user.

Configured-32 exact-weight logical-1 attempts were bounded after expert loading
was killed and then grew to about 226 GB RSS before candidate execution. The
lightweight replacement now checks the intended semantics rather than random
PCC: both layer kinds pass finite shape checks, selected-only paged K/V
mutation, and deterministic traced replay. The two earlier random-PCC failures
remain first-diagnostic artifacts and are not policy vetoes.

The eventual DRAM15 cumulative output sweep reconfirmed 32 cores. Relative
sliding/full deltas for 64/32/16/8 were +4.611/+4.898%, timer-noise
-0.041/-0.029% for the identical 32-core reference, +0.248/+0.250%, and
+3.027/+3.206%. The first non-DRAM packed-versus-separate control passed all
PCC/determinism bars but did not settle whether a DRAM-sharded separate graph
was legal. The follow-up implements three distinct DRAM-sharded program
configs (`per_core_N=(9,2,2)`) under the same final32 topology. Its low-memory
exact-attention shell passed direct PCC, selected-page K/V mutation, and
first/second/post-1000 determinism for both capacities and layer kinds, but was
13.70--15.48% slower; adding each measured delta to the prior packed whole
trace estimates a 3.526--6.377% penalty. Packed therefore remains final
without a near-tie reverse-order requirement.

The initial combined, per-node, and shared-MLP whole-decoder DRAM-separate
attempts were stopped or killed at 226--239 GB RSS while the first layer's 128
experts were still materializing, before any candidate result. They remain
resource diagnostics. The replacement harness loads only exact attention,
sink, and norm tensors and injects one identical stateless zero-MLP into both
complete decoder shells. Layer 1 deliberately uses exact checkpoint embedding
rows as a labeled proxy rather than claiming an unavailable post-layer-0
activation. The measured shell delta/direct PCC are decision evidence; the
whole-trace number is explicitly an anchored estimate.

The automatic performance harness has collected capacity-2 automatic
sliding/full nodes and asserts TT dtype, fidelity, DRAM-sharded input/program,
and final output geometry. Its non-profiled 1000-replay qualification printed
41.094766/41.420308 ms warmed prefill and 0.687493/0.779065 ms traced decode.
The subsequent marker-clean ten-replay Tracy capture passed both nodes:

| Layer kind | Prefill device / wall / ops | Decode device / wall / ops | Host rows |
| --- | ---: | ---: | ---: |
| sliding | 40.157148 / 41.894460 ms / 74 | 0.628083 / 0.705526 ms / 57 | 0 |
| full | 41.044686 / 42.394656 ms / 74 | 0.728872 / 0.799062 ms / 57 | 0 |

```bash
GPT_OSS_120B_OPTIMIZED_PERF=1 \
GPT_OSS_120B_OPTIMIZED_PERF_REPEATS=10 \
scripts/run_safe_pytest.sh --profile \
  'models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_real_weight_warmed_prefill_and_traced_decode_performance[blackhole-1x1-capacity2-automatic-NODE_KIND]' \
  -q -s
```

`NODE_KIND` was run separately as `sliding` and `full`; watcher variables were
unset, and no other hardware process overlapped either invocation.

Each decode capture has ten BFP4/LoFi DRAM-sharded packed-QKV rows and ten
BFP4/LoFi 32-core output rows. The raw capture, advice/no-advice phase tables,
and summary CSV/PNG files are under `tracy/capacity2/{sliding,full}`. Reports
were generated with the same commands as the capacity-1 reports, adding
`--active-experts 4`; the tool multiplies that per-input-group value across
the two users. Exact transcripts are
`optimized_final_source_capacity2_same_run_profile_{sliding,full}_repeats10.log.gz`
and `optimized_final_source_capacity2_tt_perf_report_generation.log.gz`.

The two-user physical lower bound is 146,071,552 bytes and 0.285296 ms at
512 GB/s. Sliding device/wall is 2.202x/2.473x roofline and full is
2.555x/2.801x. `tracy/capacity2/decode_roofline.csv` records every component
and the same-run comparison.

The capacity-2 advice tables repeat three actionable themes. Final-policy
64/16/8-core output controls were slower than final32 for both layer kinds;
BFP4/HiFi2 was covered by the real-weight precision matrix and lost to correct
LoFi; and prefill L1 residency is not legal for the full activation/projection
working set. The inherited large prefill configs already use the advised good
blocks, while the unconfigured router is just 0.22--0.24% of device time.

Final local gates after the KV-shard repair:

| Gate | Result | Artifact |
| --- | --- | --- |
| Configured-32/logical-1 semantic | 2 passed | `optimized_autofix_stage_review_logical_batch1_capacity32_lightweight_semantic.log.gz` |
| Packed versus separate | 4 passed | `optimized_autofix_stage_review_packed_vs_separate_final_packed_first.log.gz` |
| DRAM-sharded separate legality/perf | 4 passed; legal, packed faster | `optimized_autofix_stage_review_dram_separate_qkv_low_memory_batch{1,2}_{sliding,full}_packed_first.log.gz` |
| Automatic capacity-2 perf harness | 2 passed | `optimized_autofix_stage_review_automatic_capacity2_profile_harness.log.gz` |
| Production logical batch 32 | 2 passed | `optimized_autofix_stage_review_final_batch32.log.gz` |
| Integrated ordinary dispatch | 8 passed, 183 opt-in skips | `optimized_autofix_stage_review_final_integrated.log.gz` |
| Focused watcher sliding/full | 2 passed, clean | `optimized_autofix_stage_review_final_watcher_focused.log.gz` |
| Final health | four boards enumerate | `optimized_autofix_stage_review_final_post_tt_smi.log.gz` |

The delivered-source rerun after the DRAM-separate control plumbing also
passed the ordinary integrated suite (12 passed, 183 opt-in skips), both
logical-batch-32 layer kinds, and the focused watcher gate. Its exact artifacts
are `optimized_final_source_integrated.log.gz`,
`optimized_final_source_batch32.log.gz`, and
`optimized_final_source_watcher_focused.log.gz`. The final profiler-only
health snapshot enumerating all four boards is
`optimized_final_source_post_tt_smi.log.gz`.

The broad watcher diagnostic is retained as
`optimized_autofix_stage_review_final_watcher.log.gz`: four tests passed before
the advertised-context sliding SDPA program exceeded watcher's reduced kernel
config buffer (81984 versus 70656 bytes). It is not a production-path failure;
ordinary dispatch passes and the separate representative watcher run is clean.
See `AUTOFIX_stage_review_final_topology.md` for the exact evidence ledger.

## Final independent review

A fresh xhigh `$stage-review` audited the exact staged source, tests, docs,
parent fused baseline, context contract, candidate evidence, final profiler
rows, 280-entry checksum manifest, and stage scope without using hardware. It
returned `clean-pass` with no required work or hard-check gaps. Its sole
nonblocking note was that the newest focused watcher rerun is capacity 1; the
retained capacity-2 watcher artifact directly exercises and passes the
promoted BFP4/LoFi DRAM15 policy. The full verdict is retained in
`STAGE_REVIEW.md`.

## Local commit

The optimized-decoder implementation, tests, documentation, and retained
evidence were committed locally as
`f00119daaa7adbb65d36948f0e79de76dbf79f51`. The follow-up ledger-only commit
is reported in the stage handoff because a commit cannot record its own SHA.
No commit was pushed.
