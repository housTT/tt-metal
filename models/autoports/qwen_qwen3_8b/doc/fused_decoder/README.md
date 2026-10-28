# Qwen3-8B fused decoder

Status: fused-decoder gates pass on one Blackhole P300 chip (1x1 mesh). This
stage claims nothing about optimized-decoder, multichip, full-model or vLLM
readiness.

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient),
KV (key/value), RoPE (rotary position embedding), SDPA (scaled dot-product
attention), MLP (multi-layer perceptron), DRAM (device dynamic RAM), L1
(per-core local memory), GQA (grouped-query attention), ET (Eastern Time).

## What the fused decoder is

`tt/fused_decoder.py` defines `FusedDecoder(FunctionalDecoder)`. It keeps the
public contract, weights, paged KV-cache behaviour, chunk planning, padding
rules and the two tt-metal SDPA workarounds of the functional decoder
(`../functional_decoder/README.md`) and replaces the op graph of the measured
prefill and decode paths. `doc/context_contract.json` is unchanged: no memory
layout, sharding or KV-cache dtype changed, and the advertised 40960-token
context is re-verified below with the fused graph.

## Fusing patterns applied

Every rewrite was verified on device against the unfused graph (bit-equal or
PCC >= 0.9999, bf16 rounding only) and kept only when faster on the measured
workload (`experiments/` holds the scripts and logs; `work_log.md` section 3
has the numbers).

| Path | Rewrite | Skill class | Measured effect |
| --- | --- | --- | --- |
| prefill QKV, gate, up projections (chunks >= 256 rows) | `ttnn.linear` -> `ttnn.experimental.minimal_matmul` | dedicated op (faster kernel for the same math) | 1024 rows: QKV 1,635 -> 556 us, gate+up+mul 4,185 -> 2,317 us; 4096 rows: QKV 5,583 -> 1,906 us, gate+up+mul 29,756 -> 8,658 us |
| prefill O and down projections + residual add (chunks >= 256 rows) | `ttnn.linear` + `ttnn.add` -> `ttnn.experimental.dit_minimal_matmul_addcmul_fused(x, W, 1.0, residual, ones)` | op merging (residual add folded into the matmul kernel) | 1024 rows: O+res 1,179 -> 387 us, down+res 3,352 -> 1,056 us; two `ttnn.add` ops removed |
| decode Q/K per-head RMSNorm | `sharded_to_interleaved -> rms_norm -> interleaved_to_sharded` -> `sharded_to_interleaved -> rms_norm(memory_config=height-sharded)` | structural (data movement removed) | 2 reshards removed, PCC 1.0 |
| decode RoPE cos/sin gather | tiled `embedding` + tilize + `transpose` + reshard -> row-major `embedding` + free reshape + tilize + reshard | structural (permute/reshape removed) | 2 transposes removed, bit-equal, 13.0 -> 11.2 us per table |
| decode K/V cache write | 2 x `paged_update_cache` -> V reshard + `paged_fused_update_cache` | dedicated fused op (adapted: the op needs K and V on disjoint cores) | 16 -> 8 us + 0 us reshard, bit-equal caches |
| decode residual adds, logical batch 1 | `ttnn.linear` + `ttnn.add` -> `ttnn.linear(..., bias=residual)` | op merging (matmul + bias -> linear) | 2 adds removed, PCC 0.9999998 vs separate add |

Prefill at 1024 tokens: 21 -> 19 ops, 10,966 -> 4,803 us device. Decode at
batch 1: 35 -> 29 ops, 1,265 -> 1,253 us device. Decode is DRAM-bound on the
four bf16 weight matmuls (948 us, 400 GB/s) and the two single-core RMSNorms
(169 us), none of which a fusing rewrite changes; see "Next levers".

## Fusing patterns assessed and rejected (with evidence)

| Candidate | Result | Evidence |
| --- | --- | --- |
| `ttnn.rms_norm` directly on the height-sharded Q/K heads (sharded program config, and default) | exact op-contract blocker: `layernorm_device_operation.cpp:167` rejects `HEIGHT_SHARDED` input ("Height sharded inputs are not supported"); the only other placement (block-sharded column grid) breaks the SDPA Q-placement rule recorded in stage 1 | `experiments/exp1_contracts.py` E1/E1b, `experiments/logs/exp1_b1.log` |
| `nlp_create_qkv_heads_decode` with interleaved output | the op always returns height-sharded output | E2 |
| SDPA decode writing its output height-sharded (skip one reshard) | exact op-contract blocker for GQA: `sdpa_decode_device_operation.cpp:414` "Sharded output not supported for GQA" | E4 |
| `ttnn.linear` reading the width-sharded concat-heads output (skip one reshard) | numerically equal but the matmul picks a 32-core program (85 us) instead of the 64-core one used for the interleaved input (82 us + 2 us reshard); reverted after the first fused profile | `perf/full_attention` first run (work log 5), E5 |
| `paged_fused_update_cache` with K on separate cores (`overlap_qk_coregrid=False`) | validation forces overlapping grids for an interleaved QKV input (`nlp_create_qkv_heads_decode_device_operation.cpp:83`); the adapted path would add an input reshard plus two cos/sin copies for one saved op; the V-reshard variant was adopted instead | `experiments/exp2_adapt.py`, `exp3_inf.py` |
| gate/up packing (`[gate|up]` weight, one matmul + 2 slices or `ttnn.swiglu`) | bit-equal; decode +3 us (slices), prefill under `ttnn.linear` -175 us but slower than two `minimal_matmul` calls; `ttnn.swiglu` is a composite (split + silu + mul) | `exp2_adapt.py` E9b, `exp1_contracts.py` E9 |
| `minimal_matmul(fuse_swiglu=True)` (one kernel for gate, up and silu-gate) | numerically fine (PCC 0.99997 vs linear) but slower than two plain `minimal_matmul` calls + fused mul at 1024 rows (2,654 vs 2,317 us) and 4096 rows (9,724 vs 8,658 us); needs a tile-pair-interleaved weight that `ttnn.linear` cannot consume in decode (reshape is not a view: +230 us) | `experiments/exp7_prefill_mlp.py`, `exp6_view.py` |
| `minimal_matmul` / addcmul-fused kernels in decode (M = 32 rows) | 1.5x to 2x slower than `ttnn.linear` with default config; a 48-config `MinimalMatmulConfig` sweep per op (grids 11x10, 8x8, 11x8, 8x10; K/N blocks 2..8; subblocks 1..4) leaves QKV 213 vs 128 us, fused SwiGLU 734 vs 522 us, down 401 vs 241 us | `experiments/exp5_decode_configs.py` |
| `minimal_matmul` / addcmul for prefill chunks shorter than 256 rows | `ttnn.linear` wins below 192 rows for every projection and ties at 192; crossover set at 256 rows (`PREFILL_MINIMAL_MATMUL_MIN_ROWS`) | `experiments/exp8_crossover.py` |
| `ttnn.experimental.matmul_decode` | exact contract blocker: requires L1 width-sharded activations and L1-resident width-sharded weights (`matmul_decode_device_operation.cpp:35`); per-layer 100 MB weights cannot be L1 resident | `exp5_decode_configs.py` |
| `minimal_matmul_split(chunks=3)` for QKV | would still need per-tensor head splits (reshape + permute each) that `nlp_create_qkv_heads` does in one op | assessed by contract (docstring), not run |
| `rms_norm(residual_input_tensor=...)` for the residual adds | the norm consumes the sum but does not return it; the residual stream still needs the add | assessed by contract (layernorm op), not run |
| matmul bias as residual for decode batches > 1 | matmul bias must have one logical row (`matmul_device_operation.cpp:390-401`) | contract |
| `rotary_embedding_llama_fused_qk` (one RoPE op for Q and K) | needs Meta-interleaved weights and transformation matrices, K on separate cores and a second cos/sin copy; at most one op saved | assessed, not run |
| RMSNorm, SDPA, create/concat heads, RoPE, SiLU-in-mul, QKV packing, softmax, topk patterns from the skill | already the dedicated ops in the functional decoder (stage 1); no spelled-out math remains in the graph | op tables in `perf/` |

## Correctness evidence (fused graph, real weights)

Acceptance bar PCC >= 0.995 against the float32 HF layer; layer-contribution
("delta") PCC also reported. Every row below was produced by the fused decoder
(`provenance.decoder == "fused"`, `fused_decoder_sha256 330c9fe1…`).

| Gate | Result | Artifact |
| --- | --- | --- |
| layer 0 prefill 1024 / decode x2 | PCC 0.999983 (delta 0.999982) / 0.999995, 0.999988; repeat bit-equal | `pcc_layer0_full_attention.json` |
| layer 17 prefill 1000 / decode x2 (real captured activations) | PCC 0.999999 (delta 0.999422) / 0.999987, 0.999991 | `pcc_layer17_full_attention.json` |
| 19 boundary lengths (1 .. 8225, 33 after 8225) | worst prefill/decode PCC 0.999964, worst delta 0.999963 | `prefill_boundaries_full_attention.json` |
| continuation at starts 32, 37, 1000, 4096, 4101, 4499 | window per-row delta PCC min 0.999919; no-history and fresh-suffix controls score <= 0.894 | `prefill_continuation_full_attention.json` |
| multi-row prefill (5, 200, 4097, 33@70) | every row PCC >= 0.99999, zero padding verified | `prefill_multi_row_full_attention.json` |
| block size 64 | continuation PCC 0.999984, decode after 0.999954 | `block64_full_attention.json` |
| traced decode batches 1, 13, 31, 32 | eager PCC >= 0.999942, replay == eager (1.0), repeat bit-equal, refreshed-input steps PCC >= 0.999947 and differ | `trace_full_attention_batch{1,13,31,32}.json` |
| advertised context 40960 and 40959 prefill, batch-32 decode at 40959 | passed; live-user decode PCC 0.999922, replay == eager | `prefill_capacity_full_attention_{40960,40959}.json`, `advertised_context_decode_full_attention.json` |
| fused vs functional on device (same process, same weights) | prefill PCC >= 0.999994 (1024, 300, 37, 4500@37), caches >= 0.99999997, decode PCC >= 0.999982, batch-32 decode per-user >= 0.999995, max abs diff 0.0625 (bf16) | `fused_vs_functional_full_attention.json` |
| stress: 10 repeated prefills, 50 same-input replays, 48 refreshed decode steps (batch 8) | prefill bit-equal, replays bit-equal, worst step PCC 0.999941, all finite | `stress_full_attention.json` |

The layer-17 delta PCC (0.99942) equals the functional value (0.99942); the
fused graph introduces no material delta anywhere (every fused-vs-functional
PCC >= 0.99998).

## Performance evidence (same chip, same build, separate processes)

Batch 1, 1024 tokens, layer 0, HiFi4 with fp32 accumulation, `perf/README.md`:

| Phase | Before (functional, `perf/before/`) | After (fused, `perf/full_attention/`) |
| --- | ---: | ---: |
| warmed prefill, 1024 tokens | 10,966 us device, 21 ops, 11.32 ms host | 4,803 us device, 19 ops, 5.29 ms host |
| warmed continuation prefill, 100 tokens at 37 | 2,870 us, 381 ops | 2,870 us, 381 ops (64-row chunk stays on `ttnn.linear`) |
| warmed traced decode, batch 1, position 1024 | 1,265 us device, 35 ops, 1.335 ms host | 1,253 us device, 29 ops, 1.319 ms host |

The "before" numbers are a fresh run of the stage-1 harness on 2026 Oct 28
and reproduce the stage-1 tables (`../functional_decoder/perf/`) within 1 us.
The decode win is small because the decode graph was already made of
dedicated ops in stage 1 and 90 percent of its time is the four DRAM-bound
weight matmuls plus two single-core RMSNorms.

## Safety and fallback gates

- `test_fused_hot_path_fallback_audit`: no torch, `from_torch`, `to_torch`,
  host `ttnn.full`, `.cpu()`, `to_device` or synchronization in any fused hot
  method. Tracy: every op in the three windows is a device op ("0 host ops").
- No tilize/untilize beyond the row-major RoPE table gather already present in
  stage 1 (one tilize per table), no reshard that a consuming op does not
  require (the remaining decode reshards feed ops whose contracts demand the
  layout; see the rejected table), no host fallback.
- Trace allocation tracking (`trace_alloc_tracking/console.log`): traced
  decode batches 1, 13, 31, 32 and the stress test pass under
  `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
  (`*.tracked.json`); `positive_control.log` shows the tracker raising on a
  deliberately live buffer in the same environment.
- Watcher (`watcher/summary.json`): the same five tests pass with
  `TT_METAL_WATCHER=10` armed (`*.watcher.json`); the committed `watcher.log`
  holds the last device session (3 dumps, 0 fatal-pattern matches).
- Determinism: repeated prefill bit-equal, repeated replay bit-equal.

## Next levers (not fusing; owned by the optimized-decoder stage)

- Decode `rms_norm` on `[1, 1, 32, 4096]` DRAM-interleaved runs on one core
  (84 and 85 us, 13.5 percent of decode); a width-sharded input with a sharded
  program config is a layout/program-config change (stage 3).
- Decode matmuls at 78 to 84 percent DRAM utilization in bf16 (948 us);
  dtype and DRAM-sharded weights are stage 3 and 8 work.
- `nlp_create_qkv_heads_decode` at batch 1 runs on one core (33 us).
- Prefill `minimal_matmul` block sizes are the op defaults (27 percent DRAM,
  66 to 71 percent FLOPs); the continuation path still writes 27 leading rows
  token by token.

## Reproduction

From the checkout root with the stage environment (`TT_METAL_HOME`,
`PYTHONPATH`, `TT_MESH_GRAPH_DESC_PATH` for the p150 profile,
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`,
`TT_VISIBLE_DEVICES=<chip>`):

```bash
python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -p no:cacheprovider
QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k advertised_context -p no:cacheprovider
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN3_EVIDENCE_SUFFIX=tracked python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k 'test_traced_decode or test_repeated_runs_stress' -p no:cacheprovider
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8b/doc/fused_decoder/watcher QWEN3_EVIDENCE_SUFFIX=watcher python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k 'test_traced_decode or test_repeated_runs_stress' -p no:cacheprovider
```

Profiler commands are in `perf/README.md`; every command run, with times and
logs, is in `work_log.md`.

## Artifact index

- `*.json`: default-suite evidence (fused decoder), each with provenance
  (fused/functional decoder and test hashes, git SHA, device, exact command,
  environment). `*.tracked.json` and `*.watcher.json` are the tracker and
  watcher runs of the same tests.
- `fused_vs_functional_full_attention.json`, `stress_full_attention.json`:
  fused-specific gates.
- `perf/before/`, `perf/full_attention/`: tt-perf-report tables and CSV
  before and after; `perf/layer0_*_host_timings.json`: host wall times.
- `trace_alloc_tracking/`, `watcher/`: debug-gate runs.
- `experiments/`: the scratch scripts and logs behind every kept and rejected
  rewrite (random inputs, timing by trace replay).

Limitations: the decode fusing gain is 12 us (1 percent) because the graph
was already dedicated-op based; prefill chunks shorter than 256 rows keep the
functional kernels; the batch-1 bias-residual path is a separate code path
from batches > 1 (both are covered by the traced-decode tests); the
`dit_minimal_matmul_addcmul_fused` op is an experimental DiT op used here with
a ones multiplier; at logical batch 1 the bias-residual matmul multiplies the
uninitialized padding rows 1..31 of the concat-heads output and the slice
discards them (row-independent, covered by the batch-1 gates); the fused
cache update needs a core area twice the height of the Q/K rectangle (8 rows
for batch 32), which narrows headroom on harvested grids.
