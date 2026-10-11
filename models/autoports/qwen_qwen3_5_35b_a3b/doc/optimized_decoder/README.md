# Qwen3.5-35B-A3B optimized decoder: stage 03

Target checkpoint: `Qwen/Qwen3.5-35B-A3B` revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
Hardware: one Blackhole ASIC (1x1 mesh) on a host with four ASICs (two P300c cards).
Implementation: `tt/optimized_decoder.py`. `OptimizedDecoder.from_state_dict(optimization=None)`
selects `DEFAULT_POLICIES[layer_type]`. An explicit dict selects an experiment policy.
Functional and fused runtimes are unchanged.

Final frozen evidence: `final_evidence15/` only (chain
`runs/main/stage03_claude/evidence_chain15.sh`, log `final_evidence15_chain.log`, summary
`final_evidence15/summary.json`, start and end source manifests equal). Earlier chains are
superseded history. History, candidate matrices and review responses: `work_log.md`.
Checklist: `checklist.md`. Reviews: `stage_review.md` through `stage_review_5.md`.

Abbreviations: PCC (Pearson correlation coefficient), GDN (Gated DeltaNet, the
linear-attention layer), MoE (mixture of experts), KV (key/value cache), HF (Hugging Face
reference), BF16/FP32 (16/32-bit float), BFP8/BFP4 (8/4-bit block floating point), SDPA
(scaled dot-product attention), RM (row-major), DRAM (device memory), L1 (core SRAM),
ULP (unit in the last place), TTFT (time to first token).

## Headline result

Workload: 4096 prefill tokens, 128 traced decode steps, batch 1, recorded HF layer-input
activations (`runs/main/stage03_inputs/layer{0,3}.pt`, 4352 tokens of
`tech_reports/LLMs/llms.md`, real checkpoint). Host times: wall clock around the
synchronized warmed prefill, and median of 128 blocking trace replays. Physical device 3
(the device of the Stage-2 runs). Device times: whole-layer spans from a separate profiler run.

| Layer kind | Stage-2 warmed prefill (host) | Stage-3 warmed prefill (host) | Speedup | Stage-2 traced decode (host) | Stage-3 traced decode (host) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| linear_attention (30 layers) | 764010.552 us | 32491.0 us | 23.5x | 1259.203 us | 709.10 us | 1.78x |
| full_attention (10 layers) | 697554.472 us | 25904.8 us | 26.9x | 920.399 us | 434.60 us | 2.12x |

Sources: `../fused_decoder/README.md`; `final_evidence15/perf_{linear,full}/default.json`.

| Layer kind | Device prefill (stage 2 / 3) | Device decode (stage 2 / 3) | Prefill PCC | Min decode PCC (128 tokens) | DRAM per layer (allocator) |
|---|---:|---:|---:|---:|---:|
| linear_attention | 764671.6 / 32250.3 us | 1399.8 / 812.2 us | 0.99972 | 0.99874 | 1577.0 MiB |
| full_attention | 697717.6 / 25762.0 us | 978.7 / 481.5 us | 0.99952 | 0.99835 | 1536.6 MiB |

Profiler markers add time to the many small decode ops, so profiled device decode exceeds
plain traced host replay. Host and device numbers are never mixed in one comparison.

Decoder-stack estimate for one chip (layers only): 30 x 709.10 + 10 x 434.60 = 25.62 ms per
token (Stage 2: 46.98 ms). 4096-token prefill: 1.23 s (Stage 2: 29.9 s).

### Device placement finding (`final_evidence15/device_ab`, device 3 is `perf_*`)

| Physical device | full decode | linear decode | full prefill | linear prefill |
|---|---:|---:|---:|---:|
| 0 | 450.34 us | 730.39 us | 25840 us | 32560 us |
| 1 | 433.20 us | 706.99 us | 25858 us | 32525 us |
| 2 | 433.51 us | 706.94 us | 26048 us | 32542 us |
| 3 | 434.60 us | 709.10 us | 25905 us | 32491 us |

Physical device 0 replays traces 16-23 us slower from the host; prefill does not change. The
cause is not investigated in this stage (unverified: host or PCIe path). The multi-chip stages
must measure per-device placement.

### Prefill lengths (`final_evidence15/prefill_*`, device 3, 4 decode steps)

| Tokens | Plan full / linear (chunk rows / valid rows) | full prefill | linear prefill | full prefill PCC | linear prefill PCC |
|---|---|---:|---:|---:|---:|
| 2 | 2 decode-path steps | 4120 us | 4376 us | 0.99986 | 0.99983 |
| 3 | 3 decode-path steps / masked 32/3 | 4451 us | 4952 us | 0.99983 | 0.99978 |
| 4 | masked 32/4 | 4419 us | 5263 us | 0.99978 | 0.99977 |
| 5 | masked 32/5 | 4814 us | 5890 us | 0.99977 | 0.99977 |
| 100 | masked 128/100 | 7636 us | 8128 us | 0.99932 | 0.99970 |
| 1000 | masked 1024/1000 | 11948 us | 13680 us | 0.99956 | 0.99971 |
| 1024 | tiled 1024 | 12206 us | 12998 us | 0.99954 | 0.99971 |
| 1100 | 128 + masked 1024 | 16355 us | 18167 us | 0.99955 | 0.99971 |
| 1500 | masked 2048 / 256+256+masked 1024 | 16321 us | 19541 us | 0.99952 | 0.99969 |
| 2048 | tiled 2048 | 16231 us | 19427 us | 0.99953 | 0.99970 |
| 2176 | 128 + 2048, MoE group padded to 4096 | 24002 us | 27735 us | 0.99953 | 0.99971 |
| 2600 | 1024 + masked 2048 / ladder | 25182 us | 29834 us | 0.99955 | 0.99971 |
| 3072 | 1024 + 2048, one MoE group | 25564 us | 30158 us | 0.99952 | 0.99973 |
| 4000 | masked 4096/4000 | 26587 us | 33592 us | 0.99951 | 0.99972 |
| 4095 | masked 4096/4095 | 26602 us | 33870 us | 0.99952 | 0.99971 |
| 4096 | tiled 4096 | 25905 us | 32491 us | 0.99952 | 0.99972 |
| 4097 | 4096 + 1 decode-path token | 26832 us | 33736 us | 0.99952 | 0.99971 |
| 4098 | 4096 + 2 decode-path tokens | 27186 us | 34342 us | 0.99952 | 0.99971 |
| 4100 | 4096 + 4 decode-path tokens / 4096 + masked 32/4 | 28215 us | 35997 us | 0.99951 | 0.99971 |
| 2048 after 128 cached (continuation) | tiled | 15837 us | 19209 us | 0.99953 | 0.99971 |

Control without the KV scratch page (`final_evidence15/noscratch_full_4000.json`): full 4000
tokens take 30674 us, because a page table covering 4032 positions does not allow a masked
4096-row chunk; the planner then uses a ladder.

Remaining steps above the next aligned or longer length (full / linear): 1000 versus 1024
(-0.26 / +0.68 ms), 1100 and 1500 versus 2048 (full +0.12 / +0.09 ms), 4000 and
4095 versus 4096 (full +0.68 / +0.70 ms, linear +1.10 / +1.38 ms). Short whole prompts, single
samples (the same plan varies by up to about 0.6 ms between chains)
(full 2-5 tokens 4120, 4451, 4419, 4814 us; linear 4376, 4952, 5263, 5890 us). A 2176-token
prompt pads its MoE group to the 4096 dispatch bucket.

Planner rule (`OptimizedDecoder._chunk_plan`):

- Token steps until the start is aligned (page size for full attention, 32 rows for GDN),
  then full 4096-row chunks.
- For the remainder, a small dynamic program chooses exact ladder buckets
  (4096/2048/1024/512 (full)/256/128/64/32) and at most one masked bucket at the end. Cost =
  chunks x chunk_cost + rows x row_cost, with per-layer constants calibrated on device
  (`review3/plan_calibration/`: full 0.8 ms + 1.4 us per row, linear 0.5 ms + 3.0 us per row).
  A remainder of up to 8 tokens may use decode-path steps. Step cost after earlier chunks: 0.8 ms
  (full), 1.1 ms (linear); for a whole prompt of decode steps: 1.05 ms (full), 1.6 ms (linear);
  a separate masked tail group costs 3.0 ms more.
- Masked chunks never end past `max_seq_len` (262144), so padded RoPE positions stay inside the
  tables.
- GDN masked chunk: beta and decay are zero on padded rows and the convolution history is
  gathered from the last valid rows, so padding never enters state.
- Full-attention masked chunk: padded K/V rows land in owned pages beyond the logical length,
  or, when the page table is tight, in a scratch page bound with the cache
  (`bind_paged_kv_cache(..., scratch_block=...)`). Without a scratch page, the chunk must fit
  in the page table. Decode-path steps always use the original page table.
- One MoE call per group of up to 4096 planned rows; chunk outputs are concatenated in tile
  layout, padded to the next dispatch bucket and sliced back.
- Aligned unpadded inputs whose plan is one exact chunk per group use the tiled slicing path.
  All other inputs use index gathers, so a fixed physical bucket compiles no offset-specific
  programs while a decode trace is alive.

## Roofline and accounting (same final source)

`final_evidence15/profile/tracy/*/roofline.json`, `tests/estimate_roofline.py`.

| Layer kind | Compulsory decode DRAM bytes | Roofline decode (512 GB/s) | Device decode | Host decode | decode_dram_pct | prefill_flops_pct |
|---|---:|---:|---:|---:|---:|---:|
| linear_attention | 87683072 | 171.3 us | 812.2 us | 709.1 us | 21.09 | 2.46 |
| full_attention | 55215712 | 107.8 us | 481.5 us | 434.6 us | 22.40 | 3.49 |

Unaligned prefill profile (`final_evidence15/profile_unaligned/l0_4095_report.txt`,
`l0_1000_report.txt`, op device-time sums): linear 4095 tokens cost about 0.6 ms more device time
than aligned 4096 tokens. The largest extra item is one BF16 concat (+358 us); the embedding
gathers add +83 us and untilize +84 us (review-4 attribution of `final_evidence13`). The full-layer unaligned
profile is blocked by a profiler limitation: two zone names in
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`
("Reduce max" and "Softmax(Q@KT)@V") have the same 16-bit hash, and
`tt_metal/impl/profiler/profiler.cpp:376` throws when a new streaming SDPA variant compiles.

Largest prefill op: `moe_fused_swiglu`, 7.5-7.7 ms at 4096 tokens on 64 cores (BFP8), about
4.9-5.5 ms at 1000-1024 tokens. It processes the 256 experts in sequence. The sibling op
`unified_routed_expert_moe` is slower (+0.9 to +3.1 ms in the layer, `review3/routed_ffn/`;
7.62 versus 5.91 ms and 8.25 versus 8.03 ms in isolation, `review4/hybrid/`).
`hybrid_routed_expert_moe` needs all L1 above the allocator base (the router keeps 8 KiB per core
of persistent L1 tensors that decode traces capture) and, with 256 local experts, a program larger
than the default 70656-byte kernel config buffer. With `worker_l1_size` reduced by 16 KiB and the
router tensors freed in isolation, it runs and is slower at every threshold (32-256): 8.1-8.7 ms
at 1024 tokens and 9.8-10.9 ms at 4096 tokens versus 5.9 / 8.1 ms for `moe_fused_swiglu`
(`review5/hybrid_worker_l1/`).

Named limitations (evidence in `work_log.md`):

- GDN in-projection stays BF16 (BFP8 fails real-activation gates); full QKVG BFP8 runs at
  about 71% of DRAM peak.
- GDN decode keeps the exact FP32 SFPU recurrence (Stage-2 correctness contract).
- Sharded RMSNorm is not precise enough for GDN layers (0.9899 at 4096/128).
- Prefill dispatch keeps a one-core scatter (1.75 ms per 4096 tokens).
- GDN prefill head split (reshape and transpose, about 6.5 ms per 4096 tokens): the in-kernel
  flat q/k path fails decode token 32 at 4096/128 (0.9899).

## Selected policy (cumulative contract)

| Item | Full attention | Linear attention (GDN) | Evidence |
|---|---|---|---|
| Logical batch / tile rows | 1 / 32 padded rows (padding zero-filled at decode entry) | same | `review1/tail2_*` |
| Decode norms | sharded 16 cores, BF16 in, FP32 accumulate, padding zero-filled, DRAM out | interleaved FP32 | norm2, c10, c11 |
| Attention in-projection | packed QKVG BFP8 LoFi; prefill `minimal_matmul` 11x10 | packed QKV/A/B/Z BF16 HiFi4; same prefill config | inproj, review1/p1 |
| Attention out-projection | BFP8 LoFi 1D 8x4 k8 per_N 2; prefill `minimal_matmul` 8x8 | BFP8 HiFi2 1D 8x4 k8 per_N 2 | projection_extra, review1/p1 |
| Attention core | paged SDPA decode, BFP8 paged KV | FP32 recurrence, L1 intermediates, RM conv history | gate, gdn2, gdn3 |
| Router + shared gate | one packed BF16 HiFi4 matmul N=288, 1D 8x2 k32 | same | gate, c4 |
| Routed experts, decode | indexed `sparse_matmul`: gate/up BFP4 LoFi; down BFP8 LoFi; fused weighted reduce | same | indexed_fine, c5abl |
| Shared expert | packed BFP4 + SwiGLU; down BFP8 DRAM-sharded 8 cores; prefill 2D 8x8 k4 | same | shared_full, reruns |
| Routed experts, prefill | device dispatch, matmul counts/ranks, gather, `moe_fused_swiglu` BFP8 LoFi (4x4 grid; 8x8 for groups of 2048+), restore gather; `post_combine_reduce` below 2048, k-major tile combine from 2048 | same | review2/dispatch_dtype, review2/bfp8_grid, review3/tile_combine |
| Prefill planner | cost-model plan, masked buckets, KV scratch page | cost-model plan, masked buckets | review3/plan_calibration, final_evidence15/prefill_* |

## Correctness and robustness (final source, `final_evidence15/`)

| Gate | Linear | Full |
|---|---|---|
| 4096/128 recorded activations | pass 0.99972 / 0.99874 | pass 0.99952 / 0.99835 |
| Prefill per-token PCC, 4096 recorded activations | min 0.9940 at token 1720; 1 of 4096 below 0.995 | min 0.9914 at token 911; 1 of 4096 below 0.995 |
| B32 heterogeneous positions, random activations, traced, bitwise replay identity | pass 0.99923 / 0.99739 | pass 0.99855 / 0.99594 (scratch page bound, guards clean) |
| Reuse: 18 cases, B2, incl. a scratch-page write (2047 tokens, cut 1152); full also page 128 | pass 0.99900 / 0.99819 | pass 0.99852 / 0.99739; page 128 0.99838 / 0.99696; guards clean; scratch written |
| Watcher: 129/8, small pools, 4096, 4095, 1000 and full 1500 (scratch write) prefill | 0 errors in 11 logs | (same) |
| pytest real + synthetic weights, 129 and 1000 tokens | 8 passed | (same run) |
| Planner invariants and short-plan choices (CPU, `tests/test_prefill_plan.py`) | 1125 passed | (same run) |
| Native context: 262143-token prefill + decode at 262143; 262144 prefill | 0.99939 / 0.99988; 0.99955 | 0.99853 / 0.99837; 0.99835 |
| Dispatch-map exactness (32 to 4096 tokens, both combine modes) | 27/27 and 27/27 | (shared op) |
| Recorded-activation reuse (18 cases) | pass 0.99904 / 0.99892 | pass 0.99657 / 0.99634 |
| Recorded-activation B32 positions | one token 0.99393 (controlled, below) | pass 0.99913 / 0.99664 |

Reference-precision controls (CPU, HF layer in BF16 versus FP32, `review2/ref_precision/`,
`review2/prefill_ref/`):

- B32 positions row 17 token 4: BF16 reference versus FP32 0.99484; TT versus BF16
  reference 0.99393. The BF16 reference itself deviates at this token. Controlled.
- Full prefill token 911: BF16 reference versus FP32 0.99156; TT 0.99136. Controlled.
- Linear prefill token 1720: BF16 reference versus FP32 0.99942; TT 0.99402. This is a
  TT-side error at one token. Over the same 4096 tokens the BF16 reference itself has two
  tokens below 0.995 (minimum 0.99323). This is a different rule from the decode rule that
  rejected BFP4 decode down (one token at 0.9946); the datatype sweep must revisit both with one
  rule and full-model accuracy.

- Linear token 950 at 0.9922 (2600, 4097 and 4098 tokens, `review4/token950/`): TT outputs at that
  token are bitwise identical between 4096 and 4097 tokens. The HF BF16 reference has an exact tie
  between experts 27 and 173 in the router logits (-4.59375, 8th and 9th places); the CPU top-k picks
  a different 8th expert at different lengths (reference cross-length PCC 0.99260). TT versus the
  4096 reference: 0.99975. Controlled.

- Full-layer pytest (real weights, random activations seed 1234, 1000 tokens), token 684 at 0.99306
  (`review5/token684_ref/`, `review5/token684_tt/`): the BF16 router logits have a three-way exact
  tie for the 8th expert (experts 70, 10, 92 at 1.04688); the reference picks 10. With 70 or 92
  forced into the top 8, TT versus reference is 0.99751 / 0.99771. The FP32 reference also picks
  expert 10, but only 2.2e-4 ahead of expert 70, far below one BF16 ULP at 1.0. Classified as a
  BF16 router-logit tie; TT follows a valid tie-break.
- Same pytest, token 592 at 0.99499: no tie (BF16 versus FP32 reference 0.999975). TT-side error at
  one of 1000 tokens, same class as linear token 1720.

Determinism: traced decode replay is bitwise identical in every gate. Repeated prefill is not
bitwise reproducible. MoE groups below 2048 rows (`post_combine_reduce`) differ in 1-3 32-row
blocks; groups of 2048+ rows (k-major tile combine) can differ in most blocks (up to 61 of 64,
`final_evidence15/control_grid_linear_2048/default.json`). Maximum absolute difference up to
0.0029 in the final chain (`final_evidence15/prefill_full_2176_s128`), 0.0078 anywhere
(`review2/bfp8_grid/full_1024/small4x4_large11x8.json`); repeat PCC at least 0.99999994.
Experiments (`review2/drift/`, `post_combine_reduce` path only): back-to-back calls with the same
inputs agree; a later call on the same input buffers gives the majority result; a neutral op before
the call does not change the rate. State left by preceding ops causes the difference; the root cause
is unresolved. `run_layer.py` records magnitude, rows and repeat PCC for every warmed run.

## Memory (allocator and buffer census, `final_evidence15/inventory/`)

| Consumer | Linear layer | Full layer |
|---|---:|---:|
| Routed experts, prefill dispatch (BFP8 per-expert lists plus maps) | 852.7 MiB | 852.7 MiB |
| Routed experts, decode (BFP4 gate/up, BFP8 down, stacked) | 560.0 MiB | 560.0 MiB |
| Attention weights and constants (includes GDN slot 0) | 67.1 MiB | 27.9 MiB |
| GDN state, slots 1-31 (32 configured: 32 x (2 MiB FP32 + 48 KiB conv) = 65.5 MiB) | 63.5 MiB | n/a |
| RoPE tables (BF16 262144 x 64, cos and sin) | n/a | 64.0 MiB |
| Module caches shared per mesh (prefix matrices only) | 1.1 MiB | 1.1 MiB |
| Shared expert, router, norms, index tables | 9.1 MiB | 9.1 MiB |
| Allocator total after construction | 1577.0 MiB | 1536.6 MiB (incl. 1.2 MiB test KV) |

Forty layers on one chip need about 60 GiB (59.7 GiB from allocator totals minus named shared
buffers, 60.6 GiB from raw allocator totals with RoPE and module caches counted once); one chip has
31.8 GiB. Both routed-expert
representations are consumed: `moe_fused_swiglu` needs per-expert lists with one dtype
(`TT_FATAL` in `moe_fused_swiglu_device_operation.cpp:142`), and decode `sparse_matmul`
needs stacked tensors. BFP8 dispatch costs 408 MiB per layer more than BFP4. Expert placement
across chips (Stage 4) and the datatype sweep must resolve this. Recheck of the memory-layout
handoff: `final_evidence15/memory_handoff_recheck.json`.

## Rejected options (with evidence)

| Option | Result | Evidence |
|---|---|---|
| BFP4 prefill dispatch | 54 (linear) / 96 (full) of 4096 tokens below 0.995 per-token PCC, min 0.9846; BFP8 costs +0.4-0.5 ms at 4096 | review2/dispatch_dtype |
| BFP4 gate/up with BFP8 down in dispatch | op contract: one dtype for all weights | review2/dispatch_down |
| `unified_routed_expert_moe` | +0.9 to +3.1 ms in the layer; slower in isolation | review3/routed_ffn, review4/hybrid |
| `hybrid_routed_expert_moe` | with a 16 KiB smaller worker L1 it runs and is 1.6-2.8 ms slower than `moe_fused_swiglu` at thresholds 32-256 | review4/hybrid, review5/hybrid_worker_l1 |
| DRAM ND-sharded expert weights | -0.18 to +0.33 ms | review3/routed_ffn |
| 8x8 grid below 2048 tokens; 8x4 or 11x8 grids | +0.17 ms at 1024 (8x8); +0.9 to +1.0 ms at 4096 (8x4, 11x8) | review2/bfp8_grid |
| k-major tile combine below 2048 tokens | +0.24 to +0.30 ms at 1024 | review3/tile_combine |
| BFP8 GDN in-projection (LoFi/HiFi2) | 4096/128 token 32 0.9897 | review1/p1 |
| BFP8 GDN out-projection LoFi | 4096/128 0.9888 (HiFi2 passes, adopted) | review1/p1 |
| Flat q/k GDN prefill | -5.5 ms prefill; token 32 0.9899 | review1/flatqk |
| BFP4 GDN / full attention in-projection | 0.9638 / 0.9893 | gdn_real, attention_real |
| BFP4 routed down projection (decode) | token 127 0.9946 | c5v_full, c5abl_full |
| Sharded decode norm on GDN layers | 0.9899 / 0.9897 | c10_linear, c11_linear |
| Sharded residual stream (16 cores) | 454 / 448 us vs 446 us | review1/reruns/mm_full |
| DRAM-sharded in-projections | slower or L1 overflow | attention_real, gdn_real, geom, review1/reruns |
| Explicit SDPA decode configs | no win | gate_full, c4_full |
| Heads-as-rows GDN decode | -2 us | gdn2 |
| GDN state in L1 | -5 us; 2 MiB L1 per slot per layer | review1/reruns |
| Native compact decode FFN | 873-930 us vs indexed sparse | expert_real, sparse_retry |

## Context contract

Unchanged at 262144 tokens; validated in `final_evidence15/context_*`. Full-attention KV is
BFP8. See `../context_contract.json`.

## Reproduction

```bash
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_claude/evidence_chain15.sh
```

Profiles run through the symlinked root `/tmp/q35-functional-profiler-root` with cache
`/tmp/q35-opt-profiler-cache` and tt-perf-report 1.4.1, advice enabled. Candidate
matrices: `tests/run_optimized_trials.py --matrix <json>` (source manifest, stops if sources
change). `QDEV=<id>` selects a physical chip. Timing rows run alone; parallel lanes are used
only for correctness gates.
