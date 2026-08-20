# AutoDebug: TP=4 full-attention decode MLP L1 clash

## Scope and direct observations

This is a source-only diagnosis of `OptimizedDecoder._mlp`; no TT devices were opened and no implementation files were changed.

- The TP path fixes gate/up/down to 8 DRAM-sharded compute cores and local MLP shapes `5120x4352`, `5120x4352`, and `4352x5120` (`multichip_decoder.py:384-391`).
- Decode has padded `m=32`, so `_mlp` selects `use_sharded=True`; a prefill of logical length 33 has `m=33` and selects the non-DRAM-sharded prefill path (`optimized_decoder.py:875-878`). This exactly explains why PCC-valid prefill does not exercise the failing allocation contract.
- Gate/up use `K_tiles=160`, `N_tiles=136`, 8 cores. `dram_program` therefore selects `in0_block_w=20` and `per_core_N=17` (`optimized_decoder.py:890-912`).
- `_decode_l1_memory(32, 4352, 8)` gives each worker a `32 x 544` BF16 output shard: `32 * 544 * 2 = 34,816` bytes/core (`optimized_decoder.py:61-83, 920-952`).
- The gate output remains live because the following multiply needs it. The up `ttnn.linear` is therefore the first identical matmul launched while another 34,816-byte L1 output occupies the same worker cores (`optimized_decoder.py:920-959`).
- The allocator reports the live L1 frontier at 1,330,752 and the up program's static-CB end at 1,339,904, an overlap of 9,152 bytes. `ProgramImpl::validate_circular_buffer_region` throws exactly when a live L1 allocation begins below the program's static-CB end (`tt_metal/impl/program/program.cpp:1542-1549`).
- The common 1D MLP has an explicit decode option whose stated purpose is to spill W1 to DRAM before W3 so W3 CB validation does not overlap the W1 L1 result, and its implementation also puts the multiply in DRAM before resharding for W2 (`models/common/modules/mlp/mlp_1d.py:102-106, 219-245`). This is the same operation/lifetime pattern.

## Ranked hypotheses

### H1 — Verified by source: the live gate L1 result collides with the up program's static CB region

The failure boundary and addresses are predicted by this lifetime: the gate call can pass, but the otherwise identical up call sees the added live gate shard. Prefill passes because it does not use this `m<=32` decode path. Current position 33 and real-weight values are incidental; shape, phase, memory placement, and liveness determine the failure.

Smallest likely intervention: mirror the common MLP's spill contract—move `gate` to DRAM and deallocate its L1 tensor before launching `up`; request the multiply result in DRAM; then explicitly reshard the activated tensor to the down projection's `_decode_l1_memory` before down. A gate-only spill without controlling multiply placement and the down-input reshard is incomplete.

### H2 — Likely alternative: `in0_block_w=20` makes the decode matmul's static CB footprint unnecessarily large

The automatic selector takes the largest listed divisor of 20. Smaller legal divisors `(10, 5, 4, 2, 1)` should reduce one or more input CBs, but the exact Blackhole factory footprint must be measured. The overlap is only 9,152 bytes, so a smaller block may allow gate and up to coexist in L1 and could outperform a DRAM spill.

### H3 — Plausible topology fix: 8-core TP-local MLP concentrates both replicated-input and output shards too heavily

Eight cores give a 544-element gate/up output width per core. The current code cannot directly choose 16 cores because `136 N_tiles % 16 != 0`. Internally padding local intermediate width from 4,352 (136 tiles) to 4,608 (144 tiles) would permit 16 cores with 9 N tiles/core, cutting each BF16 output shard to 18,432 bytes. This costs 5.9% padded MLP compute/weight capacity and requires slicing/masking at the internal boundary. It is a broader performance candidate, not the first correctness fix.

### H4 — Refuted as the primary cause: page position, KV cache, CCL, or numerical policy

The exception is host-side L1/CB admission before up executes. Page position can alter attention work but not this fixed MLP geometry. BFP8 weights and real values affect CB formats/sizes, not data-dependent allocation. The MLP all-reduce occurs only after down returns, so ring CCL cannot cause this up-launch failure.

## Focused verify/refute experiments

Run each experiment independently; keep only a change that passes the original real-weight full-attention prefill+decode command.

1. **Confirm H1 with an allocation-only A/B.** After gate, convert it to `ttnn.DRAM_MEMORY_CONFIG`, deallocate the original L1 gate tensor, then launch up unchanged. It is sufficient for this narrow experiment to stop immediately after up. Prediction: the up launch passes. Record gate/up memory configs and the failing/passing allocator addresses. If it still reports the same 1,330,752 allocation, verify the original L1 alias was actually released/synchronized before rejecting H1.

2. **Complete the H1 functional path.** Keep the spill, force the fused SiLU multiply output to DRAM, convert the activated tensor to `width_sharded_memory("down", 4352)`, and run down plus all-reduce. Compare decode PCC to the same single-chip optimized baseline and run two consecutive decode iterations to catch lifetime reuse. Prediction: no L1 clash and unchanged PCC within the existing threshold.

3. **Test H2 without the spill.** Override gate/up `in0_block_w` one at a time with legal values `10, 5, 4, 2, 1`, preserving all other shapes, dtypes, core counts, and memory configs. First run gate+up only and record pass/fail plus static-CB end; then run the full decoder for the smallest fast passing candidate. A passing value with CB end `<= 1,330,752` verifies H2. If all legal values retain an end above the live frontier, H2 is refuted for this topology.

4. **Discriminate output-liveness from unrelated upstream L1 pressure.** Run up once with the gate output absent/deallocated and the original residual/MLP-input tensors otherwise unchanged. Prediction: up passes. Then allocate a same-memory-config `32x4352` BF16 stand-in before up. Prediction: the clash returns. This directly proves the differential allocation is the first projection result rather than KV/cache/attention state.

5. **Evaluate H3 only if spill/block tuning is too slow or still fails.** Pad local gate/up N and down K to 4,608, use 16 cores (`per_core_N=9` for gate/up), slice only at documented internal boundaries, and run the exact gate+up+multiply+down probe. Record PCC and warmed latency against the best H1/H2 candidate. Reject padding if it is slower; do not select it merely because it fits.

6. **Regression checks for the selected fix.** Run full- and linear-attention decode (both inherit `_mlp`), non-aligned prefill followed by decode, two-step determinism, trace capture/replay, and a separate watcher-clean run. Profile separately to verify whether added DRAM traffic dominates and to compare H1 against any passing H2 candidate.

## Recommended repair order

Use experiment 1 as the decisive cause check. If verified, implement experiment 2 as the smallest known-safe repair, then measure the no-spill smaller-`in0_block_w` family from experiment 3 and retain it instead only if it is correct and faster. Consider 16-core padded geometry only after those narrower candidates. No evidence supports changing KV-cache, page-table, attention, CCL, or precision policy for this exception.
