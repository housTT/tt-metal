# Phase 0: 4-chip (2× P300, TP4 + EP4, resident experts) decode/prefill profile

Captured 2026-09-10 on the shipped build (tt-metal `60f1562e8ec` dirty tree) with
the reduced full-model profiler window (`tests/test_full_model_perf.py`), one
representative layer per capture plus embedding, final hyper mixer, LM head and
device argmax. Tracy device data, device 0 only, last complete decode step
(between the last two `ArgMax` ops). Command lines: `phase0_profile.sh`
(scratchpad); reports under `layer{0_gdn,1_ple_gdn,3_qsa}/reports/` and
`layer0_prefill/reports/`. Op-level dumps with shapes:
`layer0_gdn_decode_window_ops.txt`, `layer3_qsa_decode_window_ops.txt`.

Boards had to be reset (`tt-smi -r`) before the first capture: device 0 reported
"Timed out while waiting for active ethernet core 29-25" at mesh open.

## Decode, one layer + endpoints (device 0, traced)

| capture | ops | device FW sum | layout/TM | matmul | CCL | eltwise | MoE plumbing |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer 0 GDN + endpoints | 173 | 5.53 ms | 35 % (93 ops) | 24 % (14) | 17 % (13) | 12 % (39) | 11 % (6) |
| layer 1 PLE+GDN + endpoints | 241 | 6.04 ms | 36 % (138) | 23 % (15) | 15 % (14) | 14 % (56) | 10 % (6) |
| layer 3 QSA + endpoints | 257 | 5.80 ms | 37 % (157) | 20 % (13) | 16 % (14) | 12 % (51) | 11 % (6) |

Profiler on adds per-op overhead, so absolute sums are ~1.5× production; the
proportions and per-op ranking are what matter.

### Endpoints (≈ 2.6 ms of every capture)

| op | µs | note |
|---|---:|---|
| LM head matmul 2560×32768 + 2560×29312 (BF16 HiFi2) | 426 + 391 | 318 MB → ~390 GB/s, bandwidth-bound, fine |
| `Concat` of the two LM-head splits | 401 | pure waste: one matmul with N=62080 removes it |
| `AllGatherAsync` of 248,320 BF16 logits | 450 | replace by local argmax + gather of 4 (value,index) pairs |
| `UntilizeWithUnpadding` of 248,320 logits | 491 | goes away with local argmax on the 62,080 shard |
| `ArgMax` | 99 | |
| final hyper mixer: matmul 10240×320 on **10 cores** | 120 | wrong grid; ~15 µs on 108 cores |
| final hyper mixer: `BinaryNg` over a 10240×320 tensor | 123 | applies to the weight-sized tensor every token; investigate |

Total endpoint waste ≈ 1.5 ms per token.

### GDN layer body (ops 27–87 of the layer-0 dump, ≈ 1.15 ms)

Compute that must happen: qkv matmul 85 µs, z matmul 55 µs, out matmul 41 µs,
`RecurrentGatedDeltaRule` 26 µs → ~210 µs. Everything else is overhead:

| item | µs | fix |
|---|---:|---|
| duplicate `gdn_qkv_b_a` matmul + fp32 `Slice` to refresh the conv tap | 121 + 86 | write the tap from the first matmul (1b) |
| first `Slice` of the fp32 10336-wide projection | 90 | emit BF16 qkv + separate tiny fp32 a/b matmul (1b) |
| DRAM↔L1 state copies (recurrent 3.1 MB + 3 taps, in and out) | 38 + 42 | keep state in DRAM, kernel writes in place (1b) |
| conv 4-tap FIR (`BinaryNg`, 3× `Ternary`, `Unary`) + 3 tap-rotation copies | 57 + 29 | one conv-update kernel with ring index (1b) |
| head reshapes of fp32 q/k/v (`ReshapeView` 12+12+33+40+29 µs) and the 16→48 expansion done as untilize+concat+tilize ×2 (48 µs) | ~175 | fold L2-norm/scale/expansion into the recurrent kernel (1b) |
| `softplus` `Unary` on padded 128×48 fp32 | 57 | tiny; fuse into gate prep |
| sigmoid `Unary` on 48×128 kept separate because of the fused-activation bug | 19 | fix the ttnn bug (1b) |

### Hyper-connection mixers (2 per layer, ops 7–26 and 88–111, ≈ 170 µs each)

`LayerNormPreAllGather` → `AllGather` (18–27 µs) → `LayerNormPostAllGather` →
matmul K=640 N=324 → `AllBroadcast`+`Concat`+`MorehSum` (the all-reduce, ~22 µs)
→ slices, silu, matmul, reshape, multiply, `FillPad`, `Reduce` → `AllGather`
(9 µs). About 20 ops and 3 collectives per mixer; 6 collectives and ~40 ops per
layer, ~16 ms per token across 48 layers. Phase 1a replaces this with a local
mixer on the replicated 10240-wide residual.

### MoE (ops 112–147, ≈ 1.3 ms with skew)

| op | µs | note |
|---|---:|---|
| router matmul + slices | 15 + 20 | |
| `TopK` k=10 over 512 | 76 | |
| pad/typecast/fillpad glue | ~50 | |
| `MaskedBincount` (64-core grid) | 123 | |
| `AllGather` of 512 uint32 counts (ROW_MAJOR) | 120 | includes inter-chip skew |
| `OffsetCumsum` | 43 | |
| `Dispatch` + `Tilize` 320×2560 | 27 + 57 | |
| `UnifiedRoutedExpertFfn` | 152 | the only necessary op besides the router |
| `Combine` + `ReshapeView` of scores | 80 + 83 | |
| `PostCombineReduce` (32 rows × 10 × 2560 weighted sum) | 202 | absurd for 3.3 MB; replace with matmul/sum |
| `ReduceScatter` of 32×2560 | 217 | includes skew wait; the shared-expert reduce_scatter of the same shape takes 18 µs |
| shared expert (2 `BinaryNg`, matmul 160→2560, `ReduceScatter`) | 18 + 7 + 6 + 18 | |

Phase 1d (replicated-token EP) removes `MaskedBincount`, counts `AllGather`,
`OffsetCumsum`, `Dispatch`, `Tilize`, `Combine`, and one `ReduceScatter`; the
weighted sum shrinks to a 10-row matmul.

### QSA layer (layer-3 dump, ops 26–174, ≈ 1.6 ms)

Projection matmul 31 µs, `SdpaDecode` 48 µs, out matmul 15 µs: ~95 µs of real
work. The selector and page map (ops 69–158, ~90 ops of int32 glue: gathers,
ternaries, typecasts, tilize/untilize, permutes, concats) cost ~700 µs, then
two `Embeddings` K/V gathers of 2080×256 rows plus untilize/repeat/tilize of the
mask (~60 µs). Below the 2,048-token budget the whole selector is a no-op by
construction (Phase 1c: dense paged SDPA when `current_pos ≤ 2048`).

## Prefill, one 128-token microchunk through layer 0 + endpoints (eager)

204 ops, 7.0 ms device time on device 0 for a 1-token prompt (padded to 128):
matmul 1.35 ms, `ReshapeView` 1.03 ms (29 ops), `Concat` 0.53 ms, `BinaryNg`
0.40 ms, tilize/untilize 0.9 ms, slices 0.32 ms, `ChunkGatedDeltaRule` 0.19 ms,
expert FFN 0.13 ms. Over 48 layers that is ~10,000 eager ops per microchunk,
which with ~30 µs host dispatch each is the ~1 s fixed TTFT floor. Phase 3
traces the microchunk.
