# Qwen3-8B optimized decoder

Status: optimized-decoder gates pass on one Blackhole P300 chip (1x1 mesh,
p150 mesh graph descriptor). This stage claims nothing about multichip,
full-model or vLLM readiness.

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient),
KV (key/value), RoPE (rotary position embedding), SDPA (scaled dot-product
attention), MLP (multi-layer perceptron), DRAM (device dynamic RAM), L1
(per-core local memory), BFP8/BFP4 (block floating point with 8- or 4-bit
mantissa), LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), ET (Eastern
Time).

## What the optimized decoder is

`tt/optimized_decoder.py` defines `OptimizedDecoder(FusedDecoder)`. It keeps
the public contract, paged KV-cache behaviour, chunk planning, padding rules
and the SDPA workarounds of the functional and fused decoders and changes
the weight precision, the weight placement and the decode activation layout.
Every choice is a field of `OptimizedPolicy`; the shipped values are
`DEFAULT_POLICY` (`ALL_BFP8_POLICY` is the conservative alternative kept in
the module for the accuracy stages).

### Precision and fidelity policy (one weight copy serves prefill and decode)

| Tensor group | dtype | decode fidelity | prefill fidelity | proven by the measured row |
| --- | --- | --- | --- | --- |
| QKV projection (packed Q, K, V) | BFP8 | LoFi, bf16 accumulation, 1 DRAM reader per bank | LoFi, fp32 accumulation | `perf/full_attention/decode_batch1.txt` row `MatmulDeviceOperation 32 x 4096 x 6144`, `LoFi BF16 x BFP8_B` |
| attention output projection | BFP8 | LoFi, 1 reader | LoFi, fp32 | row `32 x 4096 x 4096`, `LoFi BF16 x BFP8_B` |
| MLP gate projection | BFP8 | LoFi, 1 reader | LoFi, fp32 | first row `32 x 4096 x 12288`, `LoFi BF16 x BFP8_B` |
| MLP up projection | BFP4 | LoFi, 3 readers per bank | LoFi, fp32 | second row `32 x 4096 x 12288`, `LoFi BF16 x BFP4_B` |
| MLP down projection | BFP8 | LoFi, 1 reader | LoFi, fp32 | row `32 x 12288 x 4096`, `LoFi BF16 x BFP8_B` |
| activations, residual stream | BF16 | | | |
| RMSNorm (input, post-attention, Q, K), RoPE, cache update, SDPA | BF16 | HiFi4, fp32 accumulation | HiFi4, fp32 | `LayerNormDeviceOperation`, `SdpaDecodeDeviceOperation` rows |
| KV cache | BF16 (BFP8 measured and rejected: one 32-token-prefix user at PCC 0.9942 against HF, work log 8.4) | | | `trace_full_attention_batch*.json` provenance `kv_cache_dtype` |

Weight bytes per decoded token: 180 MB (fused decoder: 386 MB BF16); resident weight bytes 360 MB per layer with the interleaved prefill copy (work log section 8 has the DRAM budget).

### Decode layout (batch 1 to 32 users; the activation is one 32-row tile)

* Weights are width-sharded over the 8 DRAM banks
  (`dram_sharded_weight_memory_config`); every decode projection is a
  `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` matmul with
  `per_core_M=1` on a 32-core width-sharded activation (4 K-tiles per core
  for K=4096, `in0_block_w=4`; 12 for K=12288, `in0_block_w=6`), explicit
  `num_workers_per_dram_bank`, explicit compute-kernel config per role.
* The residual stream, both RMSNorms (`LayerNormShardedMultiCoreProgramConfig`,
  `block_w=4`, `subblock_w=2`, 11x3 program grid), the residual adds and the
  SiLU-gated multiply stay width-sharded on the same 32 cores: the first 32
  cores of the compute grid in row-major order, which is the core set the
  DRAM-sharded matmul writes its output on. No reshard separates them (the
  only `ReshardDeviceOperation` row is the 0-us V move the fused cache update
  needs).
* The QKV projection output goes to L1 interleaved for
  `nlp_create_qkv_heads_decode` (both sharded-input readers of that op trip
  the watcher, see "Rejected"); `nlp_concat_heads_decode` writes the attention
  output straight into the O-projection input layout.
* The layer accepts a DRAM-interleaved residual `[1, 1, B, 4096]` (the
  functional contract: one `interleaved_to_sharded` at entry, one
  `sharded_to_interleaved` plus a row slice at exit) or a residual already in
  `decode_residual_memory_config` (`[1, 1, 32, 4096]`, users in the first
  physical-batch rows, padded users at position -1 with any page-table row;
  their RoPE index is clamped to 0 on device), which it returns in the same
  layout. The second form is the layer-stack contract for the full-model
  stage; `test_sharded_residual_contract` proves it equals the DRAM contract
  at batches 1, 8, 13 (run as 16 with padded users) and 32
  (`sharded_residual_full_attention.json`), also under the watcher.
* Head split, Q/K norm, RoPE, the fused K/V cache update and SDPA (8x8 grid,
  q chunk 32, k chunk from the page-table extent, HiFi4) are the stage-2 ops.

### Prefill

Prefill reads an interleaved copy of each projection weight
(`OptimizedPolicy.prefill_interleaved_copy_max_rows`; 0 drops the copy and
every chunk falls back to `minimal_matmul` on the DRAM-sharded weight):
chunks up to 256 rows run `ttnn.linear` with the auto config (the
DRAM-sharded read is 1.3x to 2x slower at those sizes, work log 3.10, 8.2);
chunks of 257 to 1024 rows run `ttnn.linear` with an explicit 2D program
config (11x10 grid, `in0_block_w` 16) for QKV, O and down, which beats
`minimal_matmul` by 10 to 20 percent there (work log 8.8), while gate and up
keep `minimal_matmul` (their 2D configs hit L1 limits); the 4096-row
production chunk runs `minimal_matmul` for every role (every 2D config fails
on L1 at `per_core_M` 13; a 54-config `MinimalMatmulConfig` sweep does not
beat the default, work log 8.3). All at LoFi with fp32 accumulation. The residual projections use the matmul
plus `ttnn.add` because the stage-2 addcmul-fused kernel needs equal weight
and residual tile sizes. A tile-unaligned continuation writes its leading
cache rows with one fused K/V update per token over a prepared pseudo-user
tensor, in groups of up to 32 rows (3 ops per token instead of 13; the
one-hot position vector is a slice of one `ttnn.arange` row because
`moreh_full` of a scalar trips the watcher; a single batched call is a data
race because the update kernel rewrites whole cache tiles).

## Operation-topology audit of the measured decode path

`work_log.md` section 2 holds the audit table of the fused path. Outcome:

| Finding on the fused path | Action in the optimized path |
| --- | --- |
| two single-core DRAM RMSNorms, 169 us | sharded RMSNorm on the residual grid, 7 to 8 us each |
| five BF16 interleaved `ttnn.linear` decode matmuls, 948 us, 386 MB per token | DRAM-sharded BFP8/BFP4 matmuls with per-role readers and fidelity, 392 us, 180 MB |
| head split from DRAM interleaved, 33 us | head split from L1 interleaved, 20 us (sharded read is 7 us but trips the watcher) |
| concat heads -> DRAM -> O projection | concat heads output is the O-projection input (no DRAM round trip) |
| residual adds folded into matmul bias at batch 1 only | sharded `ttnn.add` (2 us) for every batch |
| separate gate and up matmuls | kept separate: they carry different dtypes (BFP8 gate, BFP4 up); packing was measured slower in stage 2 and would force one dtype |
| no reshard/layout rows other than the required head layouts | same; the two DRAM-contract conversions (entry, exit) are the only layout rows and are absent from the layer-stack contract |

## Candidate tables

Every table lives in `work_log.md` with its script and log under
`experiments/`:

* 3.1 and 3.4: decode matmul family (DRAM-sharded versus 1D mcast), core grid
  (8/16/32/64), `in0_block_w` (2 to 48), DRAM readers (1/2/3), dtype
  (BFP8/BFP4), fidelity (LoFi/HiFi2) and fp32 accumulation, per role.
* 3.2: sharded RMSNorm grids and subblocks, residual add, reshards, SiLU-mul,
  head split from sharded input, SDPA configs (grid, chunk, approx, HiFi2).
* 3.3 and 3.9: prefill `minimal_matmul` dtype/fidelity/fp32 and sharded
  versus interleaved weights.
* 3.5 and 3.10: SiLU fusion variants, short-chunk matmul kernels.
* 3.6: SDPA-decode two-op versus fused cache update A/B (stage-2 review item).
* 3.8 and 3.9: whole-layer precision policies on layers 0 and 17 (BFP8
  HiFi2, BFP8 LoFi, BFP4 MLP, BFP4 attention, mixed gate/up/down).
* 3.11: BF16 versus BFP8 KV cache.
* 3.14: head-split input layouts under the watcher.

## Correctness evidence (optimized graph, real weights)

Acceptance bars: output PCC >= 0.995 and layer-contribution ("delta") PCC
>= 0.99 against the float32 HF layer. Every row below was produced by the
committed optimized decoder (commit `0b00bbdf25`; `provenance.decoder == "optimized"`,
`optimized_decoder_sha256 063e7582b25e…` and `test_sha256 b509901be226…` in each JSON).

| Gate | Result | Artifact |
| --- | --- | --- |
| layer 0 prefill 1024 / decode x2 | PCC 0.999632 (delta 0.999615) / 0.999740, 0.999544 (delta 0.999724, 0.999519); repeat bit-equal | `pcc_layer0_full_attention.json` |
| layer 17 prefill 1000 / decode x2 (real captured activations) | PCC 0.999979 (delta 0.996277) / 0.999623, 0.999642 (delta 0.996188, 0.996680) | `pcc_layer17_full_attention.json` |
| 19 boundary lengths (1 .. 8225, 33 after 8225) | worst prefill PCC 0.999570 (delta 0.999552), worst decode-after PCC 0.998814 (delta 0.998773) | `prefill_boundaries_full_attention.json` |
| continuation at starts 32, 37, 1000, 4096, 4101, 4499 | passed at every start (window row-delta bar 0.98, no-history and fresh-suffix controls below the bar) | `prefill_continuation_full_attention.json` |
| multi-row prefill (5, 200, 4097, 33@70) | every row at the HF bar, zero padding verified | `prefill_multi_row_full_attention.json` |
| block size 64, start 37 (27 lead rows) and start 1 (63 lead rows, two pseudo-user groups) | at the HF bar; 63-row case: continuation PCC 0.999650 (delta 0.999632), decode after 0.999701 (delta 0.999671) | `block64_full_attention.json`, `block64_long_lead_full_attention.json` |
| traced decode batches 1, 13, 31, 32 | eager PCC >= 0.999072, replay == eager (1.0), repeat bit-equal, refreshed-input steps PCC >= 0.998851 and differ | `trace_full_attention_batch{1,13,31,32}.json` |
| advertised context 40960 and 40959 prefill, batch-32 decode at 40959 | passed; window PCC >= 0.999570 at 40960 and 40959; batch-32 decode at 40959, replay == eager | `prefill_capacity_full_attention_{40960,40959}.json`, `advertised_context_decode_full_attention.json` |
| optimized versus fused on device (prefill 1024, 300, 37, 4500@37, 200@4101; decode; batch 32) | prefill PCC >= 0.999576 (delta >= 0.999558), K/V caches >= 0.999950, decode PCC >= 0.999071 (delta >= 0.999040); batch-32 per user: >= 0.999232 against the fused decoder, >= 0.999202 against HF on the same inputs (fused: >= 0.999932); bars 0.995 | `optimized_vs_fused_full_attention.json` |
| sharded residual contract, batches 1, 8, 13 (as 16 with padded users), 32 | bit-equal to the DRAM contract (PCC 1.0) in every case, also under the watcher; the test runs eagerly (its `.tracked.json` twin only proves the tracker environment did not object), the traced sharded contract is for the full-model stage | `sharded_residual_full_attention.json` |
| stress: 10 repeated prefills, 50 same-input replays, 48 refreshed decode steps (batch 8) | prefill bit-equal, replays bit-equal, worst step PCC 0.999100, all finite | `stress_full_attention.json` |

## Performance evidence (same chip, same build, separate processes)

Layer 0, real weights, Tracy device time summed over the signposted window
(`perf/README.md` has the commands); host wall times from
`layer0_full_attention_seq1024_batch{1,32}_host_timings*.json`.

| Window | Before (fused, `perf/before/`) | After (optimized, `perf/full_attention/`, `perf/short_prompts/`) | Ratio |
| --- | ---: | ---: | ---: |
| warmed prefill, 1024 tokens | 19 ops, 4,814 us device, 5.19 ms host | 21 ops, 2,841 us, 3.18 ms | 1.69x device |
| warmed first prefill, 64 / 128 / 256 tokens | 1,359 / 1,587 / 2,082 us | 1,061 / 1,311 / 2,106 us | 1.28x / 1.21x / 0.99x |
| warmed continuation prefill, 100 tokens at start 37 | 381 ops, 2,864 us, 5.64 ms | 122 ops, 1,771 us | 1.62x device |
| warmed traced decode, batch 1, position 1024 | 29 ops, 1,257 us, 1.321 ms | 33 ops, 543 us, 0.622 ms | 2.31x device, 2.12x host |
| warmed traced decode, batch 32, positions 1024 | 30 ops, 1,818 us, 1.886 ms | 32 ops, 1,101 us, 1.258 ms | 1.65x device |

Dominant decode rows at batch 1 (`perf/full_attention/decode_batch1.txt`):

| Row | dtype, fidelity | DRAM readers per bank (compute cores) | time | bandwidth |
| --- | --- | --- | ---: | ---: |
| QKV `32 x 4096 x 6144` | BF16 x BFP8, LoFi | 1 (8) | 58 us | 432 GB/s (84 percent) |
| O `32 x 4096 x 4096` | BF16 x BFP8, LoFi | 1 (8) | 41 us | 404 GB/s (79 percent) |
| gate `32 x 4096 x 12288` | BF16 x BFP8, LoFi | 1 (8) | 113 us | 445 GB/s (87 percent) |
| up `32 x 4096 x 12288` | BF16 x BFP4, LoFi | 3 (24; the report column shows the 8 bank readers) | 68 us | 368 GB/s (72 percent) |
| down `32 x 12288 x 4096` | BF16 x BFP8, LoFi | 1 (8) | 113 us | 458 GB/s (89 percent) |
| SDPA decode | BF16, HiFi4 | 64 | 44 us | |
| RMSNorm x2 (sharded) | BF16, HiFi4 | 33 | 7 + 8 us | |
| head split (L1 interleaved input) | BF16 | 1 | 17 us | |
| SiLU-gated multiply | BF16 | 32 | 23 us | |

Performance accounting (decode, batch 1, same run): roofline 180 MB of
weights plus 0.5 MB of KV reads at 512 GB/s = 352 us; device time 543 us
(65 percent of roofline; the five matmuls alone are 393 us, 90 percent of
their own roofline); end-to-end traced replay 622 us on the host (one
blocking `ttnn.execute_trace` under Tracy; 79 us of replay launch and
readback on top of the device window). The remaining device gap is the 150
us of non-matmul ops (SDPA 44, SiLU-mul 23, head split 17, norms 15, RoPE
gather and layout 18, cache update 8, adds and conversions 25); the
sharded-input head split would save 13 us but trips the watcher (tt-metal
kernel item).

## Checklist (optimize skill)

| Item | Status and evidence |
| --- | --- |
| Decoder path fully traced with no host fallbacks | done: `test_optimized_hot_path_fallback_audit`; Tracy windows show 0 host ops; `trace_full_attention_batch*.json` |
| Decode activations width-sharded in L1 across norm, attention, residual, MLP and output projection | done: `perf/full_attention/decode_batch1.txt` (no reshard rows; one conversion at entry and exit of the DRAM contract only) |
| Prefill activations DRAM interleaved, large matmul program configs | done: explicit 2D `ttnn.linear` configs for QKV, O and down at 257 to 1024 rows, `minimal_matmul` elsewhere, both measured against the alternatives under the final policy (work log 3.3, 3.10, 8.3, 8.8) |
| Operation-topology audit | done: work log section 2 and the audit table above |
| Multi-device families, lower-movement residual, fused CCL, persistent CCL buffers | not applicable (one chip, no collectives) |
| Best-candidate comparison | done: work log 3.8, 3.9 (eight whole-layer policies on layers 0 and 17); the shipped default is the fastest policy that passes the delta bar on layer 17 |
| Final default reproduced the selected candidate | done: candidate 569 us (trace replay, host wall) versus final 538 us device / 646 us host in `perf/full_attention/` |
| Dtype/fidelity policy visible in measured rows | done: `LoFi BF16 x BFP8_B` and `LoFi BF16 x BFP4_B` rows in `decode_batch1.txt`; `runtime_weights` in `optimized_vs_fused_full_attention.json` |
| SDPA and composite ops | done: SDPA decode with explicit config, config sweep in work log 3.2 (grid, chunk, approx, HiFi2), A/B in 3.6 |
| Fused or packed same-input projections | QKV packed (stage 1); gate/up kept separate: they carry different dtypes in the selected policy and packing measured slower in stage 2 under one dtype |
| Explicit memory, program and compute-kernel configs | done: every decode op in `OptimizedDecoder.__init__` |
| Dominant matmul program-config sweep | done: work log 3.1 and 3.4 (core grid 8/16/32/64, `in0_block_w` up to 48, readers 1/2/3, L1 blockers recorded) |
| Compute fidelity swept per group | done: LoFi versus HiFi2 per role (3.4) and whole layer (3.8); LoFi kept |
| Attention weight dtype swept separately | done: BFP4 attention fails layer-17 delta (0.9737), BFP8 LoFi kept (3.8) |
| MLP BFP4/LoFi trials | done: gate/up/down BFP4 all tried on real weights and real captured activations; only one BFP4 projection passes layer 17; up kept BFP4 (3.8, 3.9) |
| Clean shard specs and core grids | done: 32 cores x 4 tiles (K 4096), 32 x 12 (K 12288), no padding |
| DRAM-sharded decode matmuls with reader sweep | done: 1/2/3 readers per role under the production dtype (3.4); selected counts explicit in `DEFAULT_POLICY` |
| MoE, LM head, sampling, batched prefill, perf_summary.json | not applicable to a dense decoder layer stage |
| Reduced precision experiments with real weights | done (3.8, 3.9, 3.11) |
| Performance accounting reconciled | done (performance evidence above) |
| Batch capability preserved | done: batches 1, 13, 31, 32 traced; batch-32 full context; batch-32 profile; padded batch 13 on the sharded contract |
| Stress and watcher-clean run | done: `stress_full_attention.json`; `watcher/summary.json` (9 tests including the padded-user sharded contract and the 63-lead-row continuation, 0 fatal patterns) |

## Rejected candidates (with evidence)

Kept after measurement: LoFi for every BFP8/BFP4 matmul in decode and prefill (3.8, 3.9), one DRAM reader for BFP8 rows and three for the BFP4 up row (3.4), the fused K/V cache update (3.6), the interleaved prefill weight copy for chunks up to 256 rows (8.2).

| Candidate | Result | Evidence |
| --- | --- | --- |
| BFP4 MLP (gate, up and down) | fastest (482 us) but layer-17 delta PCC 0.9852 against the 0.99 bar with real captured activations | work log 3.8 |
| BFP4 attention projections | layer-17 delta 0.9737 | 3.8 |
| BFP4 gate/up with BFP8 down | layer-17 delta 0.9899 | 3.8 |
| BFP4 gate (instead of up) or BFP4 down as the single BFP4 projection | pass, within 6 us of the kept policy, smaller PCC margin | 3.9 |
| BFP8 HiFi2 decode | 669 us versus 610 us at LoFi for all-BFP8, PCC change in the fourth digit | 3.8 |
| fp32 accumulation in decode matmuls | no PCC change beyond the fifth digit, 0 to 2 percent slower | 3.4, 3.8 |
| `in0_block_w` 2 versus 4 for gate/up | within 1 us | 3.8 |
| more DRAM readers for BFP8 rows | 2 and 3 readers are 10 to 20 percent slower than 1 at LoFi | 3.4 |
| 1D mcast decode matmuls on interleaved weights | 1 to 25 percent slower than DRAM-sharded with readers | 3.1 |
| 8-core / 64-core activation grids | 8 cores: L1 blockers at the larger blocks and no gain; 64 cores: 2-tile shards, slower | 3.1, 3.4 |
| SDPA 11x10 grid, k chunk 256/512, exp approx, HiFi2 | slower, or wrong at batch 32 (11x10, PCC 0.855), or 2 us for a precision drop | 3.2 |
| SiLU fused into the gate matmul; `ttnn.silu` then `mul` | 32 us slower; equal | 3.5, 3.10 |
| BFP8 KV cache | 2 to 3 percent faster at batch 32, half the cache bytes, but one user with a 32-token prefix falls to PCC 0.9942 against HF (0.9993 with BF16; same drop with all-BFP8 weights) | 3.11, 8.4, `experiments/exp10.json` |
| `MinimalMatmulConfig` sweep at 1024 rows (54 configs per role) | none beats the op default (QKV 283, O 193, gate 494, up 463, down 500 us) | 8.3, `experiments/exp9.json` |
| 2D `ttnn.linear` configs for gate/up at 1024 rows and for every role at 4096 rows; activation in L1 for `minimal_matmul` | L1 circular-buffer failures (gate/up `in0_block_w` 8 to 32; all 4096-row configs); in0-in-L1 gains 1 to 4 percent for 8 to 32 MB of L1 | 8.8, `experiments/exp12.json` |
| explicit `MinimalMatmulConfig` for short chunks | 15 to 35 percent slower than the default | 3.10 |
| `dit_minimal_matmul_addcmul_fused` with BFP8/BFP4 weights | op contract: residual tile size must equal the weight tile size | 3.5 |
| `ttnn.linear` on DRAM-sharded weights for prefill | TT_FATAL (auto config) or circular-buffer errors (1D configs) | 3.10 |
| one batched `paged_update_cache` for all lead rows | data race: the writer rewrites whole cache tiles | 3.5 |
| sharded input to `nlp_create_qkv_heads_decode` (7 us) or one 8x4 reshard first (10 us) | both sharded readers trip the watcher (out-of-bounds runtime arg read) | 3.14 |
| HiFi2 prefill for BFP8 | 9 percent slower than LoFi, PCC change in the fourth digit | 3.9 |
| two-op cache update instead of the fused one | SDPA time identical; fused is 8 us versus 16 us at batch 1 | 3.6 |

## Safety and fallback gates

- `test_optimized_hot_path_fallback_audit`: no torch, `from_torch`, `to_torch`, host `ttnn.full`, `.cpu()`, `to_device` or synchronization in any hot method. Tracy: every op in the six windows is a device op.
- No tilize/untilize beyond the row-major RoPE table gather (one tilize per table, stage 1), no reshard beyond the fused V move, one DRAM conversion at entry and at exit of the DRAM contract (absent from the layer-stack contract).
- Trace allocation tracking (`trace_alloc_tracking/console.log`): traced decode batches 1, 13, 31, 32 and the stress test (the traced tests) plus the eager sharded-residual test pass under `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` (`*.tracked.json`); `trace_alloc_tracking/positive_control.log` shows the tracker raising on a deliberately live buffer in the same environment. No corruptible-buffer acknowledgment or allocation scope is used.
- Watcher (`watcher/summary.json`): traced decode batches 1, 13, 31, 32, the stress test, both PCC tests, the sharded residual contract (padded users) and the 63-lead-row continuation pass with `TT_METAL_WATCHER=10` (`*.watcher.json`); `watcher/generated/watcher/watcher.log` holds the last device session (3 dumps, 0 fatal-pattern matches). Two ops that trip the watcher are off the shipped path: the sharded-input head-split readers and `moreh_full` of a scalar (work log 3.14, 8.6).
- Determinism: repeated prefill bit-equal, repeated replay bit-equal, sharded contract bit-equal to the DRAM contract.
- Host-work counters for the traced decode loop (`test_traced_decode`): one capture per batch size, three measured replays, token and current_pos refreshed by `ttnn.copy_host_to_device_tensor` only when the test changes them, page table never refreshed after setup, no synchronization inside the replayed region; positions advance on the host in this layer-level harness (the generator stage owns on-device advancement).

## Limitations

- The continuation window at a tile-unaligned start still spends 3 ops per lead row (27 rows at start 37) and more than 1 ms of host op-to-op gap in eager prefill; one batched call is unsafe (3.5), and a 2-op variant (one-hot rows built once per group) is the next lever.
- Prefill chunks up to 256 rows need the interleaved weight copy (another 180 MB per layer, 6.47 GB over 36 layers; budget in work log section 8); with `prefill_interleaved_copy_max_rows=0` they fall back to `minimal_matmul` on the DRAM-sharded weights, 15 to 20 percent slower than stage 2 at 64 to 128 rows (3.10, 8.2).
- `nlp_create_qkv_heads_decode` reads its input from L1 interleaved (17 us) instead of the 7-us sharded read because both sharded-input readers trip the watcher (3.14).
- The SiLU-gated multiply costs 21 us on 32 cores (SFPU bound); fusing it into the matmul is slower (3.10).
- Short prefixes are the precision-sensitive case: with a BFP8 cache one 32-token-prefix user fell to PCC 0.9942 against HF; with the BF16 cache the batch-32 minimum is 0.9992 (`optimized_vs_fused_full_attention.json`, per-user HF control).
- SDPA decode at batch 32 is 52 percent of the batch-32 window at 1024-token contexts and 4.9 ms at 8192-token contexts; a BFP8 cache saved only 3 percent there (3.11): a tt-metal SDPA-decode kernel item, not a layout item.
- `DEVICE_PARAMS` keeps `trace_region_size=0`, which on this checkout means dynamically allocated trace buffers (`tt_metal/distributed/mesh_trace.cpp`), not "no trace"; the Tracy decode windows (0 to 2 us op-to-op gaps) are real replays.
- `prefill_minimal_matmul_min_rows` (inherited from the fused decoder) is not consulted by `OptimizedDecoder`; `prefill_interleaved_copy_max_rows` is the chunk-size threshold.
- The layer-stack (sharded) residual contract has eager evidence only; its first trace capture and the 1 to 2 us `ttnn.relu` row belong to the full-model stage.
- Rows of the 32-row residual tile past the physical batch hold whatever the producer left there; every decode op is row-independent and `test_sharded_residual_contract` and the padded-batch traced tests cover it, but a future op that reduces across rows would need the zero fill restored.

## Reproduction

From the checkout root with the stage environment (`TT_METAL_HOME`,
`PYTHONPATH`, `TT_MESH_GRAPH_DESC_PATH` for the p150 profile,
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`,
`TT_VISIBLE_DEVICES=<chip>`; `QWEN3_KV_CACHE_DTYPE=bfp8` selects the rejected BFP8
cache for experiments):

```bash
python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k 'not host and not audit and not perf_profile' -p no:cacheprovider
python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k 'host or audit' -p no:cacheprovider
QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k advertised_context -p no:cacheprovider
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN3_EVIDENCE_SUFFIX=tracked python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k 'test_traced_decode or test_repeated_runs_stress or test_sharded_residual_contract' -p no:cacheprovider
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8b/doc/optimized_decoder/watcher QWEN3_EVIDENCE_SUFFIX=watcher python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_optimized_decoder.py -k 'test_traced_decode or test_repeated_runs_stress or test_real_weights_prefill_decode_pcc or test_sharded_residual_contract or test_block64_long_lead_rows' -p no:cacheprovider
```

Profiler commands are in `perf/README.md`; every command run, with times and
logs, is in `work_log.md`.

## Artifact index

- `*.json`: default-suite evidence (optimized decoder, BF16 cache), each with provenance (functional, fused and optimized decoder hashes, test hash, git SHA, device, policy, cache dtype, exact command, environment). `*.tracked.json` and `*.watcher.json` are the tracker and watcher runs of the same tests; `*.fused_before.json` are the fused-baseline host timings.
- `optimized_vs_fused_full_attention.json`, `sharded_residual_full_attention.json`, `stress_full_attention.json`: optimized-specific gates.
- `perf/before/`, `perf/full_attention/`: tt-perf-report tables (advice enabled) and CSV before and after, batch 1 and 32; `perf/README.md`: commands and conclusions.
- `trace_alloc_tracking/`, `watcher/`: debug-gate runs; `logs/`: filtered console logs of the final battery and profiles.
- `experiments/`: the scripts, JSON results and filtered logs behind every kept and rejected candidate (exp1 decode matmuls, exp2 small ops, exp3 prefill and contracts, exp4 SDPA A/B with its tt-perf-report tables, exp5 whole-layer policies, exp6 layout and short chunks, exp7 KV-cache dtype, exp8 head-split input under the watcher, exp9 1024-row `MinimalMatmulConfig` sweep, exp10 short-prefix cache attribution, exp11 one-hot position construction under the watcher).
- `perf/short_prompts/`: first-prefill tables at 64, 128 and 256 tokens, before and after.
- `triage/repro_cached_tensor_to_device_hang.py`: minimal repro of the tt-metal host stall on a cached-tensor device write (work log section 4).
- `../context_contract.json`: `optimized_decoder` section (weight and cache dtypes, bytes, residual layout).
