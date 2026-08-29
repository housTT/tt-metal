# GPT-OSS 120B optimized multichip decoder: operation-topology audit

Date: 2026-08-29
Starting commit: `b0d2fb3c`
Checkpoint: `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
Targets: P150, P150x2, P150x4; optimization decisions are made on the real TP2/TP4 decoder paths.

This audit was written before implementation changes. It describes the completed
multichip default and fixes the comparison rules for this pass.

## Stack boundary and dataflow

The TP2/TP4 decoder enters and exits every layer as a replicated logical
`[1, 1, B, 2880]` BF16 tensor. The completed implementation currently clones
that input to DRAM at the start of every decode call. Both decode RMSNorms use
the canonical interleaved implementation. Decode attention and active-expert
work in L1 temporarily, then each row-parallel projection performs one ring
all-reduce before the residual add. TP4 decode attention alone produces a
natural 2944-column physical result, all-reduces it, and slices to 2880. TP2
decode attention and TP2/TP4 experts reduce the logical 2880 columns. Prefill
uses the inherited BF16 DRAM path and fused reduce-scatter plus all-gather for
attention at the measured logical length 127.

There is no gather, reshard, or all-reduce *between* decoder layers, but the
DRAM clone at each layer entrance and the interleaved norms discard the L1
layout produced inside the preceding layer. A final candidate must state its
inter-layer residual contract and include the next consumer in any comparison.

## Material operations and movement

| Family | Starting topology | Repeated/same-input work | Required action and evidence |
| --- | --- | --- | --- |
| Residual layout | Replicated logical BF16; per-layer DRAM clone; two interleaved RMSNorms | Both norms read the same layer residual family; the second reads the post-attention residual | Compare canonical DRAM/interleaved against a layer-persistent L1 width-sharded replicated contract, including both norms, both residual adds, and the next layer's first norm/QKV consumer. Do not restore immediately for timing. |
| Attention projections | One packed column-parallel QKV matmul; one row-parallel O matmul per rank | Q, K, and V consume the same normalized input | Compare packed with separately tuned Q/K/V; try DRAM-sharded decode weights, explicit program configs, and lower precision/fidelity. Include bias, head conversion, CCL, and PCC. |
| Attention collective | Decode: TP2 logical 2880 all-reduce; TP4 physical 2944 all-reduce plus slice. Prefill: inherited BF16 fused RS+AG. | One material CCL after O projection | Compare decode plain ring all-reduce, asynchronous RS+AG, persistent output buffers, fused matmul-CCL, and collective placement. Measure through the next residual/norm/QKV consumer. |
| Router | Replicated BF16 weight/bias persistent in L1; matmul + softmax + top-k | Gate input is the post-attention normalized residual | Sweep explicit geometry and BF16/BFP8/BFP4 fidelity while checking selected experts and full-layer PCC. |
| Expert projections | One packed gate+up active-expert sparse matmul, then one row-parallel sparse down matmul | Gate and up consume identical activations and expert indices | Compare packed versus separate gate/up with identical active-expert semantics. Sweep legal sparse program geometry and precision/fidelity. Dense all-expert decode is disallowed. |
| Expert collective | Logical 2880 ring all-reduce after routing-weight reduction | One material CCL after down projection | Compare async RS+AG, persistent buffers, physical padding, fused paths if shape-applicable, and placement through the next residual/norm consumer. |
| Activation/CCL dtype | BF16 residual; decode BFP8 attention payload; BF16 expert payload; prefill BF16 attention payload at length 127; BFP8 KV | Typecasts precede the material reductions | Compare the custom decode attention CCL and both-phase expert CCL at BF16 and lower supported dtypes as coherent full-layer policies; retain only PCC-accepted candidates. |
| Prefill | DRAM-interleaved dense matmuls; packed all-expert sparse prefill with router mask; ring reductions | Packed QKV and packed gate/up share inputs | Sweep applicable large-matmul/sparse configs, collective placement, and dtype. Public non-aligned sequence lengths remain internally padded/sliced. |
| Persistent state | Weights, router, KV cache, rotary tensors, and CCL semaphores persist; collective output buffers do not | Trace replays allocate/reuse temporary CCL outputs | Try preallocated/persistent buffers for each material async CCL and verify replay freshness and alias safety. |

## Starting profiler evidence

The accepted multichip stage's final TP4 decode profile contains 76 operations
and about 581--582 us summed device duration. Its largest kernels are packed
active-expert gate/up sparse matmul (94 us), router (53--54 us), packed QKV
(44 us), two single-core RMSNorms (42 and 40 us), expert broadcast/all-reduce
(34 us), active-expert down sparse matmul (22 us), top-k (22 us), O projection
(15 us), and the attention RS+AG pair (13 + 10 us). Layout conversions,
typecasts, tilizes, and the layer-entry DRAM clone are therefore a coherent
movement family, not isolated micro-ops.

The accepted topology probe already established two useful controls on TP4:
logical all-reduce was 0.193193 ms; physical-2944 all-reduce plus slice was
0.110467 ms. A reduce-scattered residual plus distributed norm plus fused
all-gather/QKV consumer measured 0.129149 ms. This pass will remeasure the
families in the optimized full-layer context instead of treating that earlier
probe as a final rejection.

## Decision and retry rules

Every candidate is compared with warmed medians, real-checkpoint PCC for both
sliding and full-attention layer kinds, and profiler evidence when material.
An API or shape failure triggers layout, padding, weight-placement, and program
configuration adaptation before rejection. Material kernel/CCL failures enter
the `$autofix` workflow. Results must come from the final default path. Watcher
and profiler runs remain separate.

The public context remains 131072 tokens with page size 64 and accepts valid
non-aligned logical sequence lengths. Any internal padding owns its masking
and final slice. No context or capacity reduction is planned.

## Final audit disposition

| Audited family | Action taken | Final evidence |
| --- | --- | --- |
| Residual layout | Removed the decode DRAM clone, borrowed caller input, wrote residual adds into branch outputs, and used two 10-core L1 width-sharded norms | Decode boundary is replicated logical BF16 L1; prefill keeps replicated BF16 DRAM; neither phase has an inter-layer collective or reshard |
| Attention projections | Compared packed, separate, DRAM-sharded QKV, explicit O, TP2/TP4 DRAM O grids, BFP4/BFP8, and LoFi/HiFi2/HiFi4 | Packed BFP8 QKV uses LoFi in decode and HiFi2 in prefill; O is LoFi; TP2 alone uses 16-core DRAM-sharded decode O |
| Attention collective | Compared decode logical/physical widths, sync/async RS+AG, persistent buffers, fused MMRS, and delayed-gather residuals | BFP8 physical-width ring reduction wins decode; inherited prefill remains BF16 fused RS+AG at the measured logical length 127 |
| Router | Swept BF16/BFP8/BFP4, decode geometry, explicit 4x4 prefill program config, and DRAM-to-L1 input placement both alone and combined | BF16 automatic config retained; explicit prefill changed PCC for a sub-1.1% timing delta, while L1 placement was mixed or slower |
| Expert projections | Compared packed/separate gate-up, BFP4/BFP8/BF16, gate/up 9/12/15/30/45 cores, and down 15/18/30/45/48 cores; reran 15/30/45 down as phase-specific full-layer prefill candidates on both meshes and layer kinds | Packed top-4 BFP4/LoFi; gate/up 45; decode down 15; prefill down 45 |
| Expert collective | Compared BF16/BFP8/BFP4, logical/physical width, async and persistent placement | BF16 logical-width reduction retained |
| Activation/CCL dtype | Crossed the custom decode attention payload and both-phase expert payloads separately with the best topology | Decode attention BFP8 and both-phase expert BF16 selected; inherited prefill attention stays BF16 at length 127 and was outside the decode-tail CCL sweep |
| Prefill | Reran both kinds at non-aligned logical length 127 after every material default change, including phase-specific expert geometry and router advice | TP2 improves 53.95--54.30%; TP4 44.49--45.03%; PCC accepted |
| Persistent state | Measured persistent async and fused CCL intermediate/output buffers | Helpful inside rejected families; selected sync physical collective has no reusable output-buffer API |

The inter-layer contract that full-model bringup must preserve is logical
`[1, 1, batch, 2880]`, BF16, replicated across TP ranks, and producer-owned in
both phases. Decode is L1-interleaved and consumer-borrowed, with a 10-core L1
width-sharded RMSNorm as its next operation. Prefill remains DRAM-interleaved
and is consumed directly by the inherited prefill norm. Full-model code must
not insert a gather, reshard, or all-reduce between either phase's boundaries,
nor insert a DRAM clone on the decode boundary.
