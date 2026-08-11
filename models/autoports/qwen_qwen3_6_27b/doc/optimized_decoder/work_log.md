# Qwen/Qwen3.6-27B — optimized decoder work log

Stage 3 of the repo-local TTNN autoport pipeline: take the graph-fused decoder
([`../../tt/fused_decoder.py`](../../tt/fused_decoder.py)) and make it fast on this hardware
without changing what it computes, what context it supports, or what its public API accepts.

Hardware and environment are unchanged from stages 1 and 2: one Blackhole chip
(`/dev/tenstorrent/2`, `TT_VISIBLE_DEVICES=2`) of the intact p300c board, a 1x1 mesh, this
checkout's own `python_env`, sourced through
[`../functional_decoder/ttenv.sh`](../functional_decoder/ttenv.sh). The compute grid is 11x10 =
110 worker cores and the DRAM grid is 8x1, both read from the device rather than assumed.

* Implementation: [`../../tt/optimized_decoder.py`](../../tt/optimized_decoder.py)
* Baseline it is measured against: [`../../tt/fused_decoder.py`](../../tt/fused_decoder.py)
* Tests: [`../../tests/test_optimized_decoder.py`](../../tests/test_optimized_decoder.py),
  [`../../tests/test_optimized_decoder_perf.py`](../../tests/test_optimized_decoder_perf.py),
  [`../../tests/test_optimized_decoder_docs.py`](../../tests/test_optimized_decoder_docs.py)
* Probes: [`probes/`](probes)
* Report: [`README.md`](README.md)

## 0. Method

The same discipline as the two earlier stages, with the additions the `$optimize` skill requires:

* **One change at a time, measured in the model.** A model-free probe
  ([`probes/probe_matmul_policy.py`](probes/probe_matmul_policy.py)) maps the *legal* envelope of
  each dominant matmul cheaply - which dtype, fidelity, core count and `in0_block_w` combinations
  even allocate - and an in-model probe
  ([`probes/probe_optimized.py`](probes/probe_optimized.py)) is what actually **decides**, because
  a matmul that is faster in isolation can lose once the reshards, slices and residual layout
  around it are counted.
* **Traced decode only.** Every decode number in this document is a captured-trace replay, never
  an eager pass. Prefill is one warmed 2048-token pass.
* **The baseline is re-measured here, not copied.** `--impl fused` runs stage 2's implementation
  through *this* stage's harness on the same machine and the same build, so the before/after pair
  is one measurement session rather than two.
* **The precision change and the layout change are measured separately.** The optimized module
  takes a `PrecisionPolicy` and a `DecodeGeometry`, and both have a `fused-baseline` value, so
  "the same code at the old precision" and "the old precision on the new layout" are runnable arms
  rather than estimates. That is the only way to say which of the two levers earned what.
* **Watcher and profiler runs are separate.** Never `TT_METAL_WATCHER` together with Tracy.

## 1. Step 1 — operation-topology audit of the measured path

Before any knob tuning, this is what the fused stage's own committed reports say the measured path
*is*. Aggregated from
[`../fused_decoder/tracy/fused/*/[phase]_perf_report.csv`](../fused_decoder/tracy/fused) by op code,
device time per pass (per trace replay for the decode windows), with the profiler's own `Bound`
classification and `Math Fidelity` column:

| pass | total | ops/pass | largest rows (device time per pass, count, bound, measured fidelity/dtype) |
|---|---|---|---|
| `linear_attention` prefill 2048 | 25.681 ms | 66 | `2048x5120x34816` 6.107 ms FLOP HiFi4 BF16xBF16; `2048x17408x5120` 3.306 ms FLOP; 11x `BinaryNg` 2.264 ms; `2048x5120x10240` 2.242 ms **SLOW** HiFi4 BF16xBF16=>FP32; 15x `Slice` 1.897 ms; `ChunkGdnPrep` 1.629 ms; 5x `Tilize` 1.262 ms; `2048x5120x6144` 1.238 ms FLOP |
| `linear_attention` decode b1 | 2.352 ms | 67 | `32x5120x34816` 0.867 ms DRAM; `32x17408x5120` 0.431 ms DRAM; `32x5120x10240` 0.262 ms DRAM; `32x6144x5120` 0.156 ms DRAM; `32x5120x6144` 0.149 ms DRAM; 13x `ReshapeView` 0.145 ms; 3x `b={48}` recurrence 0.061 ms **SLOW** |
| `linear_attention` decode b32 | 5.166 ms | 70 | 3x `b={1536}` recurrence 1.252 ms DRAM FP32xFP32; `32x5120x34816` 0.868 ms; 11x `ReshapeView` 0.866 ms; 3x `Ternary` (state update) 0.788 ms; `32x17408x5120` 0.432 ms |
| `full_attention` prefill 2048 | 17.807 ms | 28 | `2048x5120x34816` 6.114 ms FLOP; `2048x17408x5120` 3.305 ms FLOP; `2048x5120x8192` 1.596 ms FLOP; `SDPA` 1.281 ms; `2048x5120x6144` 1.238 ms FLOP; `2048x6144x5120` 1.219 ms FLOP; 4x `BinaryNg` 1.020 ms; 6x `Slice` 0.905 ms |
| `full_attention` decode b1 | 2.070 ms | 50 | `32x5120x34816` 0.868 ms DRAM; `32x17408x5120` 0.431 ms DRAM; `32x5120x8192` 0.202 ms DRAM; `32x6144x5120` 0.155 ms DRAM; `32x5120x6144` 0.149 ms DRAM; `SdpaDecode` 0.104 ms |
| `full_attention` decode b32 | 2.869 ms | 49 | `32x5120x34816` 0.866 ms; `SdpaDecode` 0.864 ms; `32x17408x5120` 0.437 ms; `32x5120x8192` 0.202 ms; `32x6144x5120` 0.156 ms; `32x5120x6144` 0.149 ms |

### 1.1 What the audit says, item by item

The skill asks this audit to answer six specific questions. It does, and the answers are what
this stage's whole plan is derived from.

**Repeated same-input matmuls.** Three groups consume the same activation:

| group | inputs | fused-stage form | audit verdict |
|---|---|---|---|
| `wqkv` + `wgate` (`full_attention`) | the normed hidden state | two matmuls, `5120x8192` and `5120x6144` | the fused stage **measured** the packed form and rejected it: at 2048 rows packing is 51 % slower because cutting a 14336-wide TILE tensor apart costs more than the saved activation read, and at decode the two forms are inside one stdev (its work log §6.2). Re-checked here at the new dtype (§3.3) - the balance does not move, because the slice cost scales with the *output* width, which the dtype does not change. **Kept separate.** |
| `in_proj_qkv` + `in_proj_z` (+`in_proj_ab`) | the normed hidden state | three matmuls, `5120x10240`, `5120x6144`, `5120x128` | same measurement, same conclusion (fused §6.2): packed is 58 % slower at prefill, a tie at decode. `in_proj_ab` additionally cannot join: it is float32 with a fused bias row and the others are not. **Kept separate.** |
| `mlp_gate` + `mlp_up` | the post-attention normed state | **already packed** into one `5120x34816` matmul plus two slices | this is the reverse question, and it is the one the audit reopens. The two slices are 2 of the 15 `Slice` rows in the `linear_attention` prefill; at decode the packed output's width doubles `per_core_N`, which halves the `in0_block_w` that fits L1. **Both families measured here, per phase** (§3.3, OPT-010). |

**Avoidable reshard / layout conversions.** The fused decode pays 4 (`linear_attention`) and 9
(`full_attention`) memory-config conversions, of which **four in each** exist only because its
residual stream lives in DRAM interleaved: each of the two decode RMS norms is bracketed by an
`InterleavedToSharded` and a `ShardedToInterleaved` so the norm can run width-sharded on 20 cores.
Those four are removable by making the residual stream itself width-sharded (OPT-003), and doing
so also removes the *reason* every projection reads from DRAM. The remaining conversions are forced
by op contracts (`nlp_create_qkv_heads_decode` wants interleaved, `paged_update_cache` and
`nlp_concat_heads_decode` want height-sharded heads, the causal conv is a ROW_MAJOR chain).

**Candidate fused / packed projections.** Covered by the table above; additionally the split gate
matmul can carry its SiLU in the matmul's own epilogue, which the packed form cannot (it has to
slice first).

**Optimized composite ops.** Already in place from stage 2 and re-verified unchanged here:
`chunk_gated_delta_rule`, `chunked_scaled_dot_product_attention`,
`paged_scaled_dot_product_attention_decode`, `paged_fill_cache`, `paged_update_cache`,
`nlp_create_qkv_heads[_decode]`, `nlp_concat_heads[_decode]`, `rotary_embedding_hf`, `rms_norm`,
`addcmul`. There is no hand-built attention primitive left to replace. The one composite this stage
adds is `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, which is the decode matmul the
audit's five `Bound=DRAM` rows are asking for.

**Lower-movement replacements.** Single chip, 1x1 mesh: there is **no collective in either layer
kind**, so the whole multi-device family of the skill (residual layout across a fractured
boundary, reduce-scatter versus all-gather, fused CCL+matmul, persistent CCL buffers) has no
instance here. That is recorded as not-applicable rather than as untried, and §6 says so explicitly.
The on-device movement that *is* there is the five DRAM-bound decode matmuls and the DRAM residual
between them, and both are addressed above.

**Dtype / fidelity constraints per candidate.** This is the audit's most consequential column, and
it is the reason this stage is a precision stage as much as a layout one:

| pass family | profiler verdict | what it means |
|---|---|---|
| every decode projection | `Bound=DRAM`, 407-428 GB/s, 79-84 % of the roofline the profiler models | already near the bandwidth ceiling **at bfloat16**. No graph rewrite and no program config can move 2 bytes per weight; only storing fewer bytes can. |
| every prefill projection | `Bound=FLOP`, 82-94 GB/s (16-18 % DRAM), 63-79 % of the **HiFi4** FLOP roofline | prefill is not short of bandwidth; it is paying four compute passes per multiply. LoFi is a quarter of that work for the same tensors. |
| `2048x5120x10240` (`in_proj_qkv` prefill) | `Bound=SLOW` | its output is float32 (the causal conv carries float32 state), so it runs with float32 destination accumulation, which halves matmul throughput. Its *weight* dtype and its *accumulation* precision are independent, and this stage separates them. |
| the `b={48}` / `b={1536}` recurrence matmuls | `SLOW` at batch 1, `DRAM` at batch 32 | float32 carried state. The fused stage swept eighteen core grids for these at both regimes; they are state arithmetic, and this stage does not touch their precision. |
| `SdpaDecode` | 0.104 ms at batch 1, **0.864 ms at batch 32** | one core per head, pinned by stage 1 to work around an upstream cross-core tree-reduction defect. The KV cache it reads is bfloat16; halving it halves the read (OPT-002). The kernel defect itself is investigated in [`sdpa/`](sdpa). |

### 1.2 The plan the audit produces

1. **Precision and fidelity per tensor group.** Block-float weights for the attention and MLP
   projections; LoFi where the group tolerates it; `bfloat8_b` KV cache; float32 destination
   accumulation kept *only* on the two roles whose output is carried float32 state. Norms, the
   recurrent/conv state, the gated-delta-rule core and the recurrence matmuls unchanged.
2. **A width-sharded L1 decode stream with DRAM-sharded matmuls.** One shard grid for the whole
   layer, so the residual, both norms, the attention epilogue and the MLP never touch DRAM for an
   activation, and every dominant projection becomes a DRAM-sharded matmul with a swept
   `in0_block_w`.
3. **Phase-specific MLP packing and explicit prefill program configs** - the knobs that only pay
   off once (1) and (2) have changed which resource is scarce.

<!-- GENERATED:matmul_envelope -->
| phase | role | K x N | fused-stage form | best measured candidate | cores | in0_block_w | median | PCC vs float32 | L1 blockers hit |
|---|---|---|---|---|---|---|---|---|---|
| decode | `in_proj_qkv` | 5120 x 10240 | 292.0 us | dram-sharded bfp4/LoFi | 16 | 10 | 124.9 us | 0.993664 | 10 |
| decode | `in_proj_z` | 5120 x 6144 | 178.6 us | dram-sharded bfp4/LoFi | 8 | 20 | 81.3 us | 0.993605 | 3 |
| decode | `mlp_down` | 17408 x 5120 | 458.8 us | dram-sharded bfp4/LoFi | 8 | 17 | 176.7 us | 0.993571 | 8 |
| decode | `mlp_gate_up` | 5120 x 34816 | 897.7 us | dram-sharded bfp4/LoFi | 8 | 2 | 351.2 us | 0.993586 | 15 |
| decode | `o_proj` | 6144 x 5120 | 181.6 us | dram-sharded bfp4/LoFi | 8 | 24 | 80.4 us | 0.993534 | 3 |
| decode | `wgate` | 5120 x 6144 | 177.4 us | dram-sharded bfp4/LoFi | 8 | 20 | 81.5 us | 0.993605 | 3 |
| decode | `wqkv` | 5120 x 8192 | 228.6 us | dram-sharded bfp4/LoFi | 8 | 20 | 101.0 us | 0.993579 | 3 |
| prefill | `in_proj_qkv` | 5120 x 10240 | 2546.0 us | interleaved bf16/LoFi | — | — | 1943.5 us | 0.999899 | 0 |
| prefill | `mlp_down` | 17408 x 5120 | 3659.7 us | interleaved bfp4/LoFi | — | — | 1671.2 us | 0.993291 | 0 |
| prefill | `mlp_gate_up` | 5120 x 34816 | 7226.9 us | interleaved bfp4/LoFi | — | — | 2808.9 us | 0.993614 | 0 |
| prefill | `o_proj` | 6144 x 5120 | 1289.4 us | interleaved bfp8/LoFi | — | — | 660.7 us | 0.999741 | 0 |
| prefill | `wqkv` | 5120 x 8192 | 1730.5 us | interleaved bfp4/LoFi | — | — | 845.3 us | 0.993563 | 0 |
<!-- END GENERATED:matmul_envelope -->

## 2. The candidate ledger

Every configuration measured in this stage, with the arm it belongs to. The generated blocks below
are written by [`probes/make_doc_tables.py`](probes/make_doc_tables.py) out of the probe logs, so a
re-measurement cannot leave them stale.

### 2.1 Precision and fidelity candidates, one tensor group at a time

Synthetic-weight PCC and in-model latency, at the shipped decode layout:

<!-- GENERATED:policy_sweep -->
| layer kind | candidate | prefill | traced decode b1 | prefill PCC | decode PCC |
|---|---|---|---|---|---|
| `linear_attention` | fused-baseline bf16/HiFi4 | 28.660 ms | 2.3870 ms | 0.999906 | 0.999917 |
| `linear_attention` | bf16 weights, LoFi prefill / HiFi2 decode | 21.988 ms | 2.1321 ms | 0.999353 | 0.999664 |
| `linear_attention` | bfp8 all, LoFi prefill / HiFi2 decode | 20.229 ms | 1.9773 ms | 0.999203 | 0.999371 |
| `linear_attention` | bfp8 all, LoFi both phases | 20.265 ms | 1.5325 ms | 0.999203 | 0.999212 |
| `linear_attention` | bfp8 all, HiFi2 both phases | 22.845 ms | 1.9777 ms | 0.999455 | 0.999371 |
| `linear_attention` | bfp8 all + bf16 KV cache | 20.258 ms | 1.9779 ms | 0.999203 | 0.999371 |
| `linear_attention` | bfp4 gate/up only (rest bfp8) | 19.641 ms | 1.9353 ms | 0.996548 | 0.997026 |
| `linear_attention` | bfp4 gate/up, HiFi2 prefill | 22.756 ms | 1.9341 ms | 0.996806 | 0.997026 |
| `linear_attention` | bfp4 MLP incl. down (rest bfp8) | 19.593 ms | 1.8574 ms | 0.995205 | 0.995964 |
| `linear_attention` | bfp4 attention only (rest bfp8) | 20.289 ms | 1.9736 ms | 0.982556 | 0.983799 |
| `linear_attention` | bfp4 MLP + bfp4 attention | 19.600 ms | 1.8534 ms | 0.978831 | 0.980388 |
| `linear_attention` | shipped | 19.585 ms | 1.8533 ms | 0.978831 | 0.980388 |
| `full_attention` | fused-baseline bf16/HiFi4 | 20.360 ms | 2.1229 ms | 0.999484 | 0.999202 |
| `full_attention` | bf16 weights, LoFi prefill / HiFi2 decode | 13.037 ms | 1.8395 ms | 0.997321 | 0.996765 |
| `full_attention` | bfp8 all, LoFi prefill / HiFi2 decode | 10.773 ms | 1.6694 ms | 0.997081 | 0.996567 |
| `full_attention` | bfp8 all, LoFi both phases | 10.821 ms | 1.1705 ms | 0.997081 | 0.995928 |
| `full_attention` | bfp8 all, HiFi2 both phases | 13.955 ms | 1.6696 ms | 0.997876 | 0.996798 |
| `full_attention` | bfp8 all + bf16 KV cache | 11.034 ms | 1.6757 ms | 0.997147 | 0.996588 |
| `full_attention` | bfp4 gate/up only (rest bfp8) | 10.140 ms | 1.6262 ms | 0.987883 | 0.985375 |
| `full_attention` | bfp4 gate/up, HiFi2 prefill | 13.956 ms | 1.6263 ms | 0.988606 | 0.985885 |
| `full_attention` | bfp4 MLP incl. down (rest bfp8) | 10.136 ms | 1.5503 ms | 0.983345 | 0.979733 |
| `full_attention` | bfp4 attention only (rest bfp8) | 10.738 ms | 1.6607 ms | 0.934821 | 0.920129 |
| `full_attention` | bfp4 MLP + bfp4 attention | 10.149 ms | 1.5408 ms | 0.922029 | 0.906528 |
| `full_attention` | shipped | 10.108 ms | 1.5407 ms | 0.922029 | 0.906528 |
<!-- END GENERATED:policy_sweep -->

And the table that actually **decides** the policy - the same candidates on the **real checkpoint**,
at the sequence length the suite uses, including the cache-consuming traced-replay check OPT-007
asks for when attention-projection precision is what changed:

<!-- GENERATED:real_weight_policy -->
| layer kind | candidate | prefill PCC @2049 | decode PCC, 4 steps | traced decode PCC, 5 replays |
|---|---|---|---|---|
| `linear_attention` | fused-baseline bf16/HiFi4 | 0.999935 | 0.999625 | 0.999973 |
| `linear_attention` | shipped policy | 0.999451 | 0.998421 | 0.999049 |
| `linear_attention` | bfp8 all + LoFi | 0.999636 | 0.999356 | 0.999691 |
| `linear_attention` | bfp4 gate/up only (rest bfp8) | 0.999451 | 0.998421 | 0.999049 |
| `linear_attention` | bfp4 MLP incl. down (rest bfp8) | 0.999183 | 0.997206 | 0.998338 |
| `linear_attention` | bfp4 attention only (rest bfp8) | 0.995052 | 0.962769 | 0.996297 |
| `linear_attention` | bfp4 MLP + bfp4 attention | 0.994544 | 0.961522 | 0.994921 |
| `full_attention` | fused-baseline bf16/HiFi4 | 0.999964 | 0.999986 | 0.999977 |
| `full_attention` | shipped policy | 0.999103 | 0.999249 | 0.998257 |
| `full_attention` | bfp8 all + LoFi | 0.999624 | 0.999684 | 0.999585 |
| `full_attention` | bfp4 gate/up only (rest bfp8) | 0.999103 | 0.999249 | 0.998257 |
| `full_attention` | bfp4 MLP incl. down (rest bfp8) | 0.992919 | 0.992522 | 0.991277 |
| `full_attention` | bfp4 attention only (rest bfp8) | 0.995119 | 0.997564 | 0.987765 |
| `full_attention` | bfp4 MLP + bfp4 attention | 0.988435 | 0.990592 | 0.978232 |
<!-- END GENERATED:real_weight_policy -->

### 2.2 Decode layout and geometry candidates

<!-- GENERATED:geometry_sweep -->
| layer kind | candidate | traced decode b1 | prefill PCC | decode PCC |
|---|---|---|---|---|
| `linear_attention` | shipped geometry | 1.3422 ms | 0.996548 | 0.996738 |
| `linear_attention` | cores=1 | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 0-0] grow to 1778560 B which is beyond max L1 size of 1572864 B
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/_ttnncpp.so(+0x1fef3ca) [0x753d16def3ca]
 --- tt::tt_metal::detail::ProgramImpl::validate_circul | | |
| `linear_attention` | cores=2 | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1779: tt::exception
info:
Statically allocated circular buffers in program 198 clash with L1 buffers on core range [0-0 - 3-0]. L1 buffer allocated at 393216 and static circular buffer region ends at 611200
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/libtt_metal.so(+0x9c7753) [0x753d141c7 | | |
| `linear_attention` | cores=4 | 2.0422 ms | 0.996548 | 0.996707 |
| `linear_attention` | cores=8 | 1.4786 ms | 0.996548 | 0.996864 |
| `linear_attention` | cores=16 | 1.4153 ms | 0.996548 | 0.996815 |
| `linear_attention` | cores=16, in0_block_w=2 everywhere | 1.4155 ms | 0.996548 | 0.996815 |
| `linear_attention` | cores=16, in0_block_w=1 everywhere | 1.4154 ms | 0.996548 | 0.996815 |
| `linear_attention` | packed gate/up at decode | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 10-9] grow to 1585536 B which is beyond max L1 size of 1572864 B
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/_ttnncpp.so(+0x1fef3ca) [0x753d16def3ca]
 --- tt::tt_metal::detail::ProgramImpl::validate_circu | | |
| `linear_attention` | cores=16, split gate/up (OPT-010 pair) | 1.4158 ms | 0.996548 | 0.996815 |
| `linear_attention` | cores=16, packed gate/up (OPT-010 pair) | 1.4440 ms | 0.996524 | 0.996830 |
| `linear_attention` | fused decode layout (no sharded stream, no DRAM-sharded matmuls) | 1.5897 ms | 0.996438 | 0.996736 |
| `linear_attention` | sharded residual, interleaved matmuls (no DRAM sharding) | 1.5486 ms | 0.996438 | 0.996694 |
| `linear_attention` | SiLU fused into the gate matmul epilogue | 1.3891 ms | 0.996548 | 0.996738 |
| `linear_attention` | SDPA 1 core per head (stage 1's pinned value) | 1.3422 ms | 0.996548 | 0.996738 |
| `linear_attention` | SDPA 8 cores per head | 1.3421 ms | 0.996548 | 0.996738 |
| `full_attention` | shipped geometry | 0.9907 ms | 0.987883 | 0.985080 |
| `full_attention` | cores=1 | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 0-0] grow to 1778560 B which is beyond max L1 size of 1572864 B
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/_ttnncpp.so(+0x1fef3ca) [0x753d16def3ca]
 --- tt::tt_metal::detail::ProgramImpl::validate_circul | | |
| `full_attention` | cores=2 | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1779: tt::exception
info:
Statically allocated circular buffers in program 610 clash with L1 buffers on core range [0-0 - 7-7]. L1 buffer allocated at 786432 and static circular buffer region ends at 939520
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/libtt_metal.so(+0x9c7753) [0x753d141c7 | | |
| `full_attention` | cores=4 | 1.6999 ms | 0.987883 | 0.984469 |
| `full_attention` | cores=8 | 1.1462 ms | 0.987883 | 0.984959 |
| `full_attention` | cores=16 | 1.0710 ms | 0.987883 | 0.985146 |
| `full_attention` | cores=16, in0_block_w=2 everywhere | 1.0708 ms | 0.987883 | 0.985146 |
| `full_attention` | cores=16, in0_block_w=1 everywhere | 1.0709 ms | 0.987883 | 0.985146 |
| `full_attention` | packed gate/up at decode | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 10-9] grow to 1585536 B which is beyond max L1 size of 1572864 B
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/_ttnncpp.so(+0x1fef3ca) [0x753d16def3ca]
 --- tt::tt_metal::detail::ProgramImpl::validate_circu | | |
| `full_attention` | cores=16, split gate/up (OPT-010 pair) | 1.0707 ms | 0.987883 | 0.985146 |
| `full_attention` | cores=16, packed gate/up (OPT-010 pair) | 1.0991 ms | 0.987857 | 0.985117 |
| `full_attention` | fused decode layout (no sharded stream, no DRAM-sharded matmuls) | 1.2717 ms | 0.987673 | 0.984623 |
| `full_attention` | sharded residual, interleaved matmuls (no DRAM sharding) | 1.2292 ms | 0.987673 | 0.984647 |
| `full_attention` | SiLU fused into the gate matmul epilogue | 1.0375 ms | 0.987883 | 0.985080 |
| `full_attention` | SDPA 1 core per head (stage 1's pinned value) | 1.0344 ms | 0.987883 | 0.985231 |
| `full_attention` | SDPA 8 cores per head | 0.9794 ms | 0.987883 | 0.985080 |
<!-- END GENERATED:geometry_sweep -->

### 2.3 `in0_block_w` per role, at the shipped core count

<!-- GENERATED:in0_block_w_sweep -->
| layer kind | candidate | traced decode b1 |
|---|---|---|
| `linear_attention` | in_proj_qkv in0_block_w=1 | 1.4722 ms |
| `linear_attention` | in_proj_qkv in0_block_w=5 (shipped) | 1.3422 ms |
| `linear_attention` | in_proj_z in0_block_w=1 | 1.4056 ms |
| `linear_attention` | in_proj_z in0_block_w=5 (shipped) | 1.3422 ms |
| `linear_attention` | mlp_down in0_block_w=1 | 1.5536 ms |
| `linear_attention` | mlp_down in0_block_w=17 (shipped) | 1.3422 ms |
| `linear_attention` | mlp_gate in0_block_w=1 | 1.4884 ms |
| `linear_attention` | mlp_gate in0_block_w=5 (shipped) | 1.3421 ms |
| `linear_attention` | mlp_up in0_block_w=1 | 1.4875 ms |
| `linear_attention` | mlp_up in0_block_w=5 (shipped) | 1.3420 ms |
| `linear_attention` | out_proj in0_block_w=1 | 1.4170 ms |
| `linear_attention` | out_proj in0_block_w=2 | 1.3589 ms |
| `linear_attention` | out_proj in0_block_w=3 | 1.3471 ms |
| `linear_attention` | out_proj in0_block_w=6 (shipped) | 1.3420 ms |
| `full_attention` | mlp_down in0_block_w=1 | 1.2020 ms |
| `full_attention` | mlp_down in0_block_w=17 (shipped) | 0.9907 ms |
| `full_attention` | mlp_gate in0_block_w=1 | 1.1363 ms |
| `full_attention` | mlp_gate in0_block_w=5 (shipped) | 0.9905 ms |
| `full_attention` | mlp_up in0_block_w=1 | 1.1362 ms |
| `full_attention` | mlp_up in0_block_w=5 (shipped) | 0.9906 ms |
| `full_attention` | o_proj in0_block_w=1 | 1.0651 ms |
| `full_attention` | o_proj in0_block_w=2 | 1.0066 ms |
| `full_attention` | o_proj in0_block_w=3 | 0.9950 ms |
| `full_attention` | o_proj in0_block_w=6 (shipped) | 0.9904 ms |
| `full_attention` | wgate in0_block_w=1 | 1.0533 ms |
| `full_attention` | wgate in0_block_w=5 (shipped) | 0.9904 ms |
| `full_attention` | wqkv in0_block_w=1 | 1.0601 ms |
| `full_attention` | wqkv in0_block_w=5 (shipped) | 0.9905 ms |
<!-- END GENERATED:in0_block_w_sweep -->

### 2.4 Prefill candidates

<!-- GENERATED:prefill_sweep -->
| layer kind | candidate | prefill | prefill PCC | decode PCC |
|---|---|---|---|---|
| `linear_attention` | split gate/up both phases (shipped) | 19.415 ms | 0.996524 | 0.996738 |
| `linear_attention` | packed gate/up at prefill | 19.634 ms | 0.996548 | 0.996738 |
| `linear_attention` | derived grid but 10 rows of cores (8x10) | 22.465 ms | 0.996524 | 0.996738 |
| `linear_attention` | derived grid but 4 rows of cores (8x4) | 25.755 ms | 0.996524 | 0.996738 |
| `linear_attention` | in0_block_w=2 on every prefill projection | 20.381 ms | 0.996453 | 0.996738 |
| `linear_attention` | in0_block_w=8 on every prefill projection | ERROR | ERROR | ERROR |
| | ↳ blocker | RuntimeError: TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 7-7] grow to 1594240 B which is beyond max L1 size of 1572864 B
backtrace:
 --- /home/ttuser/dev/qwen/tt-metal/build_Release/lib/_ttnncpp.so(+0x1fef3ca) [0x7af8949ef3ca]
 --- tt::tt_metal::detail::ProgramImpl::validate_circul | | |
| `linear_attention` | explicit 2D grid 8x8 on the MLP | 19.371 ms | 0.996524 | 0.996738 |
| `linear_attention` | explicit 2D grid 11x10 on the MLP | ERROR | ERROR | ERROR |
| | ↳ blocker | AssertionError: actual tensor contains non-finite values | | |
| `linear_attention` | explicit 2D grid 8x8 on every projection | 19.372 ms | 0.996524 | 0.996738 |
| `full_attention` | split gate/up both phases (shipped) | 9.908 ms | 0.987857 | 0.985080 |
| `full_attention` | packed gate/up at prefill | 10.160 ms | 0.987883 | 0.985080 |
| `full_attention` | derived grid but 10 rows of cores (8x10) | 13.507 ms | 0.987857 | 0.985080 |
| `full_attention` | derived grid but 4 rows of cores (8x4) | 15.323 ms | 0.987857 | 0.985080 |
| `full_attention` | in0_block_w=2 on every prefill projection | 10.721 ms | 0.987729 | 0.985145 |
| `full_attention` | in0_block_w=8 on every prefill projection | 9.762 ms | 0.987695 | 0.984954 |
| `full_attention` | explicit 2D grid 8x8 on the MLP | 9.937 ms | 0.987857 | 0.985080 |
| `full_attention` | explicit 2D grid 11x10 on the MLP | ERROR | ERROR | ERROR |
| | ↳ blocker | AssertionError: actual tensor contains non-finite values | | |
| `full_attention` | explicit 2D grid 8x8 on every projection | 10.011 ms | 0.987857 | 0.985080 |
<!-- END GENERATED:prefill_sweep -->

## 3. The changes, in the order they were applied

Each subsection names the probe log that decided it.  Latencies quoted inline are in-model wall
clock, median over the repeats the probe reports, measured back to back in one session; the
profiler's device times are §5.

### 3.1 Precision and fidelity, per tensor group

Moved one group at a time from the fused baseline.  The synthetic-weight table is §2.1's first block
and the **real-checkpoint** table - the one that decides - is its second.  What the two together say:

| group | shipped | why |
|---|---|---|
| MLP `gate`/`up` | **BFP4**, LoFi | real-weight PCC clears the 0.995 bar at every measurement in §2.1's real-weight table, minimum 0.998334 over the whole suite; worth 5-6 % of traced decode over BFP8 |
| MLP `down` | BFP8, LoFi | BFP4 here takes `full_attention` below the bar (see the real-weight table).  Real-weight rejection, and exactly the asymmetry the skill predicts for FF2 |
| attention `wqkv`/`wgate`/`o_proj`, GDN `in_proj_z`/`out_proj` | BFP8, LoFi | the mandatory BFP4 attention trial (OPT-007) was run on real weights *with* the cache-consuming traced follow-on, and it fails: `linear_attention` decode 0.962769 and `full_attention` traced 0.989 - both below the bar.  Rejected on model-visible output PCC, not on a raw-cache diagnostic |
| GDN `in_proj_qkv` | BFP8, **HiFi2**, float32 destination accumulation | its output *is* the float32 state the causal conv carries and compares against HF's cache object; the weight dtype and the accumulation precision are separate levers and only the first is reduced |
| GDN `in_proj_ab` | float32, HiFi4 | four output tiles feeding the recurrence decay; nothing to win |
| KV cache | **BFP8** | halves the cache footprint and every SDPA read; real-weight prefill/decode PCC unchanged to 1e-4 (OPT-002's mandatory reduced-cache trial) |
| norms, recurrent/conv state, the gated-delta-rule core, the recurrence matmuls | unchanged (BF16 / float32, HiFi4, fp32 destination) | state, not weights.  Stage 2 swept eighteen core grids for the recurrence; this stage does not touch its precision |

Two results here are worth more than their size, because both are the opposite of the obvious guess:

**Math fidelity at decode is not free.** The audit found every decode projection `Bound=DRAM` at
79-84 % of the roofline *at bfloat16*, which suggests raising decode fidelity should cost nothing.
Measured, at the same BFP8 weights, LoFi -> HiFi2 at decode costs **+29 %** of the traced
`linear_attention` step (1.5325 -> 1.9773 ms) and **+43 %** of the `full_attention` one (1.1705 ->
1.6694), because halving the weight bytes moved those matmuls off the bandwidth ceiling and onto the
compute one.  It buys 0.995928 -> 0.996567 of `full_attention` decode PCC on synthetic weights.  So
LoFi in both phases, and `PrecisionPolicy.decode_fidelity` is the lever that records it.

**Float32 destination accumulation was a blanket setting and is now targeted.** Stage 2 used one
`HiFi4 + fp32_dest_acc_en` config for every matmul in the layer.  Destination accumulation halves
matmul throughput at these shapes, and only the two roles whose output is carried float32 state need
it.  `PrecisionPolicy.fp32_dest_acc_all` restores the old behaviour for the baseline arm.

### 3.2 The decode stream: one width-sharded L1 grid, DRAM-sharded matmuls, and the core count

The audit's headline finding was that four of the fused decode's memory-config conversions per layer
kind exist only to bracket the two RMS norms, because the residual lives in DRAM interleaved - and
that the same fact is why every projection reads its activation from DRAM.  Both are fixed by the
same change: put the residual stream in width-sharded L1 and keep it there from the first norm to the
last residual add, then give every dominant projection the DRAM-sharded matmul that layout enables.

The core count is a single number for the whole layer, and it has to divide the tile count of every
activation width the decode path carries - 160, 192, 256/320, 544 and 1088 tiles - whose greatest
common divisor is 32.  So the legal set on this device is `{1, 2, 4, 8, 16, 32}`, computed from the
real shapes by `_legal_stream_cores` rather than assumed, and swept
(`logs/probe_optimized_geometry.log`):

| cores | `linear_attention` traced decode | `full_attention` traced decode |
|---|---|---|
| 1 | L1 blocker: circular buffers grow to 1778560 B against 1572864 B of L1 | same |
| 2 | L1 blocker: static circular buffers clash with the L1 buffers | same |
| 4 | 2.0422 ms | 1.6999 ms |
| 8 | 1.4786 ms | 1.1462 ms |
| 16 | 1.4153 ms | 1.0710 ms |
| **32** | **1.3422 ms** | **0.9907 ms** |

32 is both the largest legal value and the fastest, and it is fastest for a second reason worth
recording: the failure mode at low core counts is *not* a bare circular-buffer overflow but
"statically allocated circular buffers clash with L1 buffers" - the width-sharded activations are
resident in L1 for the whole step, so fewer cores means wider shards means less room for the matmul's
own blocks.  More cores buys block size, and block size is what the next paragraph is about.

At 32 cores every dominant projection ends up at the **largest legal `in0_block_w`** for its shape -
equal to its input shard's whole K-tile count - and the sweep confirms that is where the time is:

| role | shipped `in0_block_w` | at 1 | cost of the small block |
|---|---|---|---|
| `mlp_down` (`linear`) | 17 | 1.5538 ms vs 1.3424 | +15.7 % |
| `mlp_gate` / `mlp_up` (`linear`) | 5 | 1.4879 / 1.4875 vs 1.3423 | +10.8 % |
| `in_proj_qkv` | 5 | 1.4723 vs 1.3420 | +9.7 % |
| `in_proj_z` | 5 | 1.4055 vs 1.3425 | +4.7 % |
| `out_proj` | 6 | 1.4170 / 1.3587 / 1.3472 at 1 / 2 / 3 | monotone in the block size |
| `mlp_down` (`full`) | 17 | 1.2022 vs 0.9912 | +21.3 % |
| `mlp_gate` / `mlp_up` (`full`) | 5 | 1.1365 vs 0.9907 | +14.7 % |
| `wqkv` | 5 | 1.0598 vs 0.9904 | +7.0 % |
| `wgate` | 5 | 1.0535 vs 0.9907 | +6.3 % |
| `o_proj` | 6 | 1.0653 / 1.0070 / 0.9958 at 1 / 2 / 3 | monotone in the block size |

There is no role left at `in0_block_w <= 2`, and the value is **computed** at construction from the
role's own K tiles and weight dtype rather than tabulated - a table went stale once during this stage
(it was written for 16 cores, the constructor silently clamped it, and the sweep that was supposed to
validate it measured a configuration the layer never ran), which is why
:data:`~...optimized_decoder.DEFAULT_IN0_BLOCK_W` is now empty by design.

What the layout is worth, isolated:

| arm | `linear_attention` | `full_attention` |
|---|---|---|
| fused decode layout (DRAM residual, interleaved matmuls) | 1.5897 ms | 1.2717 ms |
| sharded residual, interleaved matmuls | 1.5486 ms | 1.2292 ms |
| **sharded residual, DRAM-sharded matmuls (shipped)** | **1.3422 ms** | **0.9907 ms** |

So the sharded residual alone is worth 2.6 % / 3.3 % and the DRAM-sharded matmuls it enables another
13.3 % / 19.4 %.  The reshard count fell from stage 2's 4 / 9 to 5 / 8, and the composition changed:
the four norm brackets are gone in both kinds, and what remains is exactly the set each op contract
forces, enumerated in `tests/test_optimized_decoder.py::EXPECTED_DECODE_RESHARDS` and asserted as an
equality rather than a budget.

`sharded_stream=False` forces `dram_sharded=False`, and that is a property of the op rather than a
choice: the DRAM-sharded matmul requires a width-sharded L1 activation and produces one, so
"DRAM-interleaved residual with DRAM-sharded projections" is not a configuration - the projections
would hand a sharded tensor straight back into the residual add.  The first version of this stage's
isolation arm did exactly that and failed with `input_tensor_a.is_allocated()`, because the fused
norm's `to_memory_config` became a no-op view and its unconditional `deallocate` then freed the
residual.

### 3.3 The MLP gate/up split, per phase (OPT-010)

Stage 2 measured the packed `[hidden, 2 * intermediate]` gate/up matmul 51 % faster than two separate
matmuls at 2048 rows, with a DRAM-interleaved bfloat16 weight and `ttnn.linear`'s heuristic on both
sides.  At this stage's dtypes and layout the answer flips in **both** phases:

| phase | packed | split | verdict |
|---|---|---|---|
| `linear_attention` prefill 2048 | 19.634 ms | **19.415 ms** | split |
| `full_attention` prefill 2048 | 10.160 ms | **9.908 ms** | split |
| `linear_attention` decode, 32 cores | L1 blocker (CBs clash with the L1 buffers) | **1.3422 ms** | split, and packed does not allocate |
| `full_attention` decode, 32 cores | L1 blocker | **0.9907 ms** | split, and packed does not allocate |
| `linear_attention` decode, 16 cores | 1.4440 ms (stdev 0.1075) | **1.4158 ms** | split |
| `full_attention` decode, 16 cores | 1.0991 ms | **1.0707 ms** | split |

The two slices of a 34816-wide tensor did not get cheaper; the matmul they were amortising did, so
they stopped paying for themselves.  The 16-core rows exist because at 32 cores the packed form's
`per_core_N` is twice the split form's and its circular buffers clash with the resident sharded
activations - so the comparison OPT-010 asks for is made where both families are legal, and the
blocker is recorded where one is not.

Splitting **both** phases also means the packed weight is never built, which makes the layer about
89 MB smaller per layer at BFP4 - the opposite of the memory cost an earlier draft of this stage
assumed it would pay.

The SiLU can ride the split gate matmul's own epilogue, and that is measurably *worse*: 1.3891 vs
1.3422 (`linear_attention`) and 1.0375 vs 0.9907 (`full_attention`), so it stays on the multiply that
consumes it, exactly as in the packed form.  `DecodeGeometry.fuse_gate_silu` is the lever.

### 3.4 Prefill 2D program configs

The prefill side is not optional here: `ttnn.linear`'s heuristic falls back to
`MatmulMultiCoreProgramConfig` when `in1` is sharded, and that config rejects a sharded B outright
("Input B memory layout must be INTERLEAVED").  So the decode-side DRAM-sharded weight and the
prefill-side explicit 2D config are two halves of one decision, and the alternative - a
DRAM-interleaved copy of every weight for prefill - would cost about 210 MB per layer.

Two things about the geometry had to be measured rather than derived:

* **the column count must equal the DRAM bank count.**  With any other divisor of the output tile
  count the matmul returns non-finite values rather than failing validation: `x = 10` on the
  160-tile-wide `o_proj` / `mlp_down` / `out_proj` and the 320-tile-wide `in_proj_qkv` all produced
  NaN, while `x = 8` (the bank count) is correct to PCC 0.99937 against the heuristic's own output on
  the same weights.  Every `N` this model projects to is a multiple of 8 tiles, so `per_core_N` stays
  exact.
* **the output block has to be bounded.**  The natural `per_core_M x per_core_N` for the
  `2048 x 5120 x 34816` matmul is 8 x 136 = 1088 tiles = 2.2 MB, which does not fit L1;
  `out_block_h` is therefore the largest divisor of `per_core_M` that keeps the block under 160
  tiles.

The candidate table (`logs/probe_optimized_prefill.log`), against the shipped derived grid:

| candidate | `linear_attention` | `full_attention` |
|---|---|---|
| **shipped (derived grid 8xy, `in0_block_w` per role)** | **19.107 ms** | **9.644 ms** |
| explicit 8x8 on the MLP (identical to derived) | 19.371 ms | 9.937 ms |
| explicit 8x8 on every projection | 19.372 ms | 10.011 ms |
| 10 rows of cores (8x10, non-exact `per_core_M`) | 22.465 ms | 13.507 ms |
| 4 rows of cores (8x4) | 25.755 ms | 15.323 ms |
| 11x10 on the MLP | non-finite output (see the bank-alignment rule above) | non-finite output |
| `in0_block_w = 2` everywhere | 20.381 ms | 10.721 ms |
| `in0_block_w = 8` everywhere | L1 blocker on `in_proj_qkv` | 9.762 ms |

`in0_block_w` is swept upward per role to a cap of 8 subject to an L1 estimate, which lands every
role at 8 except `in_proj_qkv`, whose float32 output block and BFP8 weight put 8 over L1.  That is the
one role this bound holds back and it costs about 1.5 % of the `linear_attention` prefill; it is
recorded in the README's limitations rather than left implicit.

### 3.5 `in_proj_ab` keeps its interleaved, bias-folded form

Four output tiles of float32 state arithmetic with `dt_bias` folded in as the matmul's bias row.  The
DRAM-sharded matmul has no bias slot, so making this role DRAM-sharded would mean a separate bias add,
and at 15 us of a ~1.3 ms step there is nothing to win.  It keeps stage 2's measured `core_grid`.  The
cost is one `ShardedToInterleaved` of the normed stream tensor - a 320 KB copy - and it is one of the
five reshards `EXPECTED_DECODE_RESHARDS` accounts for.

### 3.6 The decode SDPA: the kernel defect stage 1 handed over

Stage 1 pinned `max_cores_per_head_batch = 1` and said in as many words that it was "a real
decode-latency cost - 4 active cores instead of 64 - and it is a correctness-first choice, not a tuned
one", with a model-free reproducer and an explicit hand-off to this stage.  This stage took it.

The recorded rule - correct only when `num_k_chunks % (2 * cores_per_head) == 0` - turned out **not**
to be the predicate: `num_k_chunks = 5` on 2 cores violates it and is exact.  The real cause is a
destination-register bounds violation: the fused SFPU softmax correction
(`calculate_fused_max_sub_exp_add_tile`) addresses five DEST tiles, and under float32 destination
accumulation only four per half are addressable - the op's own host side says so
(`sdpa_decode_program_factory.cpp`: `dst_size = fp32_dest_acc_en ? 4 : 8`).  The fifth tile lands
outside the live half, and *which* live value it destroys depends on the active half, which alternates
with the reducer core's k-chunk-loop parity.  The decisive experiment needed no code change: with
`fp32_dest_acc_en` off, five tiles fit and every position is correct.

The fix replaces that one fused call with the same arithmetic unfused, under `if constexpr
(DST_ACCUM_MODE)`, so no other caller's path changes; it is a device kernel, so no C++ rebuild.  Full
diagnosis, the refuted hypotheses, the correctness sweep and the blast-radius run are
[`sdpa/AUTOFIX_SDPA.md`](sdpa/AUTOFIX_SDPA.md).  In summary: at `k_chunk` 512 the device/float32-golden
scale over stage 1's eight characterisation positions is 0.9929-1.0011 at 8 cores and 0.9933-0.9999 at
4, against 0.9951-1.0168 at 1 core - so multi-core is *more* accurate as well as faster; the op goes
9774 us -> 2795 us at position 262143; and the existing SDPA decode unit tests are unchanged at 30
passed, 1 skipped.

In the whole traced `full_attention` decode step at position 2048 the choice is worth:

| `max_cores_per_head_batch` | traced decode |
|---|---|
| 1 (stage 1's pinned value) | 1.0344 ms |
| 4 | 0.9907 ms |
| **8 (shipped)** | **0.9794 ms** |
| 16 | L1 blocker at this k chunk, identical on the stock build |

2 cores is avoided deliberately: with more than one core the program factory's float32 flash
accumulators stay off and the residual bf16 merge error tracks the **per-core** merge count, which at
2 cores is 256 and lands at 0.980 - inside the bar but on its edge, where 8 cores' 64 merges are not.
At the advertised `max_batch` the setting is inert by construction, because `batch * num_kv_heads` =
128 already exceeds the grid, so the batch-32 SDPA win comes from the BFP8 cache instead.

### 3.7 Shared-input projection packing, re-measured at the new dtypes (OPT-001)

Stage 2 rejected packing `wqkv`+`wgate` and `in_proj_qkv`+`in_proj_z`, but it measured them with
bfloat16 weights, a DRAM-interleaved layout and the heuristic on both sides - all three of which this
stage changed - so the comparison was redone at the shipped dtypes, on the shipped decode layout, and
charging the packed arm for the slices it needs
(`probes/probe_projection_packing.py`, `logs/probe_projection_packing.log`):

<!-- GENERATED:projection_packing -->
| pair | phase | rows | candidate | median | in0_block_w | blocker |
|---|---|---|---|---|---|---|
| full_attention wqkv + wgate | decode | 32 | separate (shipped) | 197.5 us | [5, 5] | — |
| full_attention wqkv + wgate | decode | 32 | packed + 2 slices | 204.0 us | 5 | — |
| linear_attention in_proj_qkv + in_proj_z | decode | 32 | separate (shipped) | 329.5 us | [5, 5] | — |
| linear_attention in_proj_qkv + in_proj_z | decode | 32 | packed + 2 slices | 334.9 us | 5 | — |
| full_attention wqkv + wgate | prefill | 2048 | separate (shipped) | 1302.1 us | — | — |
| full_attention wqkv + wgate | prefill | 2048 | packed + 2 slices | 1993.5 us | — | — |
| linear_attention in_proj_qkv + in_proj_z | prefill | 2048 | separate (shipped) | 2985.4 us | — | — |
| linear_attention in_proj_qkv + in_proj_z | prefill | 2048 | packed + 2 slices | 3988.0 us | — | — |
<!-- END GENERATED:projection_packing -->

At decode the two forms are inside each other's spread on both pairs, and at prefill the separate
form wins by a third on the pair that can be measured. So stage 2's conclusion survives the change
of dtype and layout, and it survives it for a reason that is now visible: the packed form's cost is
two slices whose size scales with the *output* width, and the separate form's cost is one extra
matmul launch plus a second read of the activation - reducing the weight bytes shrinks the second
term and leaves the first alone. Both pairs stay separate.

### 3.8 The full-context regression, and what it turned out to be

The 2049-token suite was green and the full-context test was not: at the shipped policy the
262143-token `full_attention` prefill tail came back at PCC **0.959397** with a best-fit scale of
**0.971309**, against stage 2's 0.998030 / 0.997496.  The same policy is at 0.999103 at 2049 tokens,
so nothing shorter showed it.  This is the anomaly this stage spent the most effort on, and the answer
was not where the shape of the problem suggested.

Attributed one group at a time on the real 262143-token prompt, with the same reference construction
the test uses (`logs/probe_long_context_precision.log`, `logs/probe_long_context_mlp.log`):

| arm | tail PCC | tail scale | paged K cache PCC |
|---|---|---|---|
| shipped | 0.959397 | 0.971309 | 0.999799 |
| + bfloat16 KV cache, SDPA 4 cores per head | 0.959518 | 0.971615 | 0.999824 |
| + bfloat16 KV cache, SDPA 1 core per head | 0.959518 | 0.971615 | 0.999824 |
| shipped (BFP8 cache), SDPA 1 core per head | 0.959397 | 0.971309 | 0.999799 |
| + bfloat16 attention weights | 0.964391 | 0.976327 | 0.999820 |
| + bfloat16 KV **and** bfloat16 attention weights | 0.964474 | 0.976373 | 0.999845 |
| + bfloat16 MLP weights | 0.971467 | 0.913160 | 0.999799 |
| + BFP8 MLP gate/up | 0.971508 | 0.949062 | 0.999799 |
| + HiFi2 on every projection | 0.950487 | 0.974342 | 0.999859 |
| + HiFi4 on every projection | **0.947617** | 0.977627 | 0.999858 |
| **precision fully reverted, this stage's layout kept** | 0.972423 | 0.983181 | 0.999924 |
| fused-stage policy *and* layout (control) | **0.998030** | 0.997496 | **0.999989** |

Three things fall out of that table, and the first two are what made the diagnosis:

1. **The reduced KV cache is innocent.**  It was the prime suspect - the chunked SDPA reads it 262144
   times - and a bfloat16 cache measures 0.959518 against the BFP8 cache's 0.959397.  The decode SDPA
   core count moves nothing either.  (The 8-core setting is only *legal* with a BFP8 cache: at
   bfloat16 the receive buffers clash with L1, which is a real constraint and is recorded as one.)
2. **Raising precision makes it worse.**  HiFi4 on every projection is 0.947617 - below the shipped
   LoFi's 0.959397.  A quantity that moves the wrong way with precision is not a precision effect.
3. **Reverting the precision policy entirely and keeping this stage's layout still leaves it at
   0.972423, and the paged K cache PCC at 0.999924 against the control's 0.999989.**

The cache is 6x noisier, and the cache is what a 262144-key attention integrates.  What produces it
is `wqkv`, whose reduction is 160 K tiles deep - and stage 2 accumulated every matmul in **float32
destination registers**, because it used one `HiFi4 + fp32_dest_acc_en` compute config for the whole
layer.  This stage had dropped that to the two roles whose *output* is float32 state, on the grounds
that destination accumulation halves matmul throughput.  For `wqkv` that trade is wrong, and only the
full context shows it: a 160-tile reduction in bfloat16 destination registers costs the cached K/V
about six times its error, which is invisible at 2049 keys and is amplified by an attention over
262144 of them.

The fix is `PrecisionPolicy.prefill_fp32_acc_roles = ("wqkv",)`: float32 destination accumulation for
that role **at prefill only**.  Prefill is what fills the cache; a decode step writes one token's K/V
into a 262144-entry cache and cannot move its PCC, and decode is where destination accumulation costs
the most relative to the work done.  The verification is `logs/probe_long_context_fp32acc.log`, and it says two things.  The fix restores
the **scale** exactly - 0.971309 -> 0.997382, against the control's 0.997496 - and takes the tail PCC
to 0.984588, comfortably inside the bar; what is left is the genuine block-float cost of BFP8
attention weights and BFP4 MLP weights at 262144 keys, and the paged K cache moves 0.999799 ->
0.999850 with it.  And the *targeted* fix is better than the blanket one: float32 destination
accumulation on every projection in both phases gives the same PCC (0.985160) with a **worse** scale
(0.970051), as well as costing decode - so this is not "the fused stage was right all along", it is a
narrower and measurably better setting than either.

<!-- GENERATED:long_context_fp32acc -->
| arm | tail PCC | tail scale | decode PCC | decode scale | paged K PCC | paged V PCC | blocker |
|---|---|---|---|---|---|---|---|
| shipped policy (fp32 dest acc on prefill wqkv) | 0.984588 | 0.997382 | 0.985773 | 0.997669 | 0.999850 | 0.999856 | — |
| no fp32 dest acc anywhere but the state roles | 0.959397 | 0.971309 | 0.961608 | 0.979283 | 0.999799 | 0.999807 | — |
| fp32 dest acc on every projection, both phases | 0.985160 | 0.970051 | 0.985960 | 0.967598 | 0.999850 | 0.999856 | — |
| fused-stage policy (control) | 0.998030 | 0.997496 | 0.999267 | 0.994318 | 0.999989 | 0.999993 | — |
<!-- END GENERATED:long_context_fp32acc -->

### 3.9 Chasing the one `Bound=SLOW` row that is a third of the step

The committed optimized decode reports classify every BFP8 projection `Bound=DRAM` at 85-90 % of the
roofline, but the two BFP4 gate/up rows come back `Bound=SLOW` at 54 % of the DRAM roofline *and*
54 % of the FLOP roofline, on the 12 compute cores the DRAM-sharded matmul allocates for itself.
Together they are about a third of the traced `full_attention` decode step, so the label was worth a
search rather than a footnote: quartering the weight bytes moved that matmul off the bandwidth
ceiling, and 12 cores is then not enough compute to reach the other one.

`probes/probe_bfp4_gateup.py` measures every legal way to run that exact matmul - the shipped
DRAM-sharded config, `ttnn.linear`'s heuristic on an interleaved weight, an explicit 1D multicast
config over four core grids with the weight both interleaved and DRAM width-sharded, and an explicit
2D config - and repeats the whole thing at BFP8 so the comparison separates "BFP4 is compute-bound"
from "this shape prefers a different config":

<!-- GENERATED:bfp4_gateup -->
| shape | dtype | candidate | cores | in0_block_w | per_core_N | median | PCC vs float32 | blocker |
|---|---|---|---|---|---|---|---|---|
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | dram-sharded (shipped) | 32 | 5 | 17 | 196.3 us | 0.993616 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | interleaved, ttnn.linear heuristic | — | — | — | 206.4 us | 0.993617 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 8x8 w=interleaved | 64 | 8 | 9 | 205.3 us | 0.993634 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 9 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 11x10 w=interleaved | 110 | 8 | 5 | 204.1 us | 0.993634 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 5 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 8x4 w=interleaved | 32 | 8 | 17 | 212.9 us | 0.993634 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 17 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 11x5 w=interleaved | 55 | 8 | 10 | 206.0 us | 0.993634 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 10 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp4 | 2D 8x1 w=dram-sharded | 8 | — | 68 | 248.9 us | 0.993613 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | dram-sharded (shipped) | 32 | 5 | 17 | 229.0 us | 0.999840 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | interleaved, ttnn.linear heuristic | — | — | — | 270.9 us | 0.999815 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 8x8 w=interleaved | 64 | 8 | 9 | 291.1 us | 0.999831 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 9 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 11x10 w=interleaved | 110 | 8 | 5 | 273.0 us | 0.999831 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 5 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 8x4 w=interleaved | 32 | 8 | 17 | 286.7 us | 0.999831 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 17 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 11x5 w=interleaved | 55 | 8 | 10 | 269.4 us | 0.999831 | — |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 10 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate / mlp_up (split) `32x5120x17408` | bfp8 | 2D 8x1 w=dram-sharded | 8 | — | 68 | 354.8 us | 0.999835 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | dram-sharded (shipped) | 32 | 5 | 34 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 10-9] grow to 15855 |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | interleaved, ttnn.linear heuristic | — | — | — | 374.4 us | 0.993602 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 8x8 w=interleaved | 64 | 8 | 17 | 387.7 us | 0.993617 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 17 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 11x10 w=interleaved | 110 | 8 | 10 | 382.9 us | 0.993617 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 10 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 8x4 w=interleaved | 32 | 8 | 34 | 402.8 us | 0.993617 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 34 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 11x5 w=interleaved | 55 | 8 | 20 | 386.4 us | 0.993617 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 20 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp4 | 2D 8x1 w=dram-sharded | 8 | — | 136 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 7-0] grow to 167616 |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | dram-sharded (shipped) | 32 | 1 | 34 | 595.1 us | 0.999768 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | interleaved, ttnn.linear heuristic | — | — | — | 519.2 us | 0.999815 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 8x8 w=interleaved | 64 | 8 | 17 | 541.6 us | 0.999830 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 17 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 11x10 w=interleaved | 110 | 8 | 10 | 515.7 us | 0.999830 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 10 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 8x4 w=interleaved | 32 | 8 | 34 | 542.0 us | 0.999830 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 34 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 11x5 w=interleaved | 55 | 8 | 20 | 518.9 us | 0.999830 | — |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 20 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_gate_up (packed) `32x5120x34816` | bfp8 | 2D 8x1 w=dram-sharded | 8 | — | 136 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/program/program.cpp:1722: tt::exception
info:
Statically allocated circular buffers on core range [0-0 - 7-0] grow to 279027 |
| mlp_down `32x17408x5120` | bfp4 | dram-sharded (shipped) | 32 | 17 | 5 | 186.4 us | 0.993571 | — |
| mlp_down `32x17408x5120` | bfp4 | interleaved, ttnn.linear heuristic | — | — | — | 370.9 us | 0.993499 | — |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 8x8 w=interleaved | 64 | 8 | 3 | 196.5 us | 0.993610 | — |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 3 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 11x10 w=interleaved | 110 | 8 | 2 | 203.7 us | 0.993610 | — |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 2 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 8x4 w=interleaved | 32 | 8 | 5 | 201.4 us | 0.993610 | — |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 5 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 11x5 w=interleaved | 55 | 8 | 3 | 195.7 us | 0.993610 | — |
| mlp_down `32x17408x5120` | bfp4 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 3 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp4 | 2D 8x1 w=dram-sharded | 8 | — | 20 | 236.6 us | 0.993586 | — |
| mlp_down `32x17408x5120` | bfp8 | dram-sharded (shipped) | 32 | 17 | 5 | 228.4 us | 0.999782 | — |
| mlp_down `32x17408x5120` | bfp8 | interleaved, ttnn.linear heuristic | — | — | — | 370.9 us | 0.999682 | — |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 8x8 w=interleaved | 64 | 8 | 3 | 300.2 us | 0.999796 | — |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 8x8 w=dram-sharded | 64 | 8 | 3 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 11x10 w=interleaved | 110 | 8 | 2 | 267.8 us | 0.999796 | — |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 11x10 w=dram-sharded | 110 | 8 | 2 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 8x4 w=interleaved | 32 | 8 | 5 | 272.0 us | 0.999796 | — |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 8x4 w=dram-sharded | 32 | 8 | 5 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 11x5 w=interleaved | 55 | 8 | 3 | 272.7 us | 0.999796 | — |
| mlp_down `32x17408x5120` | bfp8 | 1D mcast 11x5 w=dram-sharded | 55 | 8 | 3 | — | — | TT_THROW @ /home/ttuser/dev/qwen/tt-metal/tt_metal/impl/buffers/circular_buffer_config.cpp:222: tt::exception
info:
Only L1 buffers can have an associated circular buffer!
backtrac |
| mlp_down `32x17408x5120` | bfp8 | 2D 8x1 w=dram-sharded | 8 | — | 20 | 362.7 us | 0.999801 | — |
<!-- END GENERATED:bfp4_gateup -->


## 4. Correctness

Two properties are worth stating before the table, because they are what make the rest of it mean
something:

* **The optimized code reproduces the fused layer exactly at the fused settings.**
  `test_optimized_matches_fused` builds this stage's module with stage 2's `PrecisionPolicy` and
  `DecodeGeometry` and compares against the real `FusedDecoder` on identical weights and inputs:
  **PCC 1.0** for both layer kinds, prefill and decode.  So the before/after pair is one
  implementation at two settings, and nothing in the speed-up is attributable to a different
  computation.
* **At the shipped settings the two agree at 0.999092 on real checkpoint weights**
  (`test_optimized_output_agrees_with_fused`), which is the precision change and only the precision
  change.

The acceptance bar is **PCC >= 0.995** against the HF layer on real checkpoint weights; the
synthetic-weight cases hold the looser `SYNTHETIC_PCC_BAR` for the measured reason in §3.1 and the
README.  Over the whole suite: 472 PCC records, synthetic minimum **0.983746**, real-weight minimum
**0.998334**.

<!-- GENERATED:correctness -->
| measurement | `linear_attention` | `full_attention` |
|---|---|---|
| `optimized_alt_block_size_decode_pcc` | — | min 0.984853 (2) |
| `optimized_alt_block_size_prefill_pcc` | — | min 0.986281 (2) |
| `optimized_batched_decode_pcc` | min 0.995919 (52) | min 0.983746 (52) |
| `optimized_batched_prefill_pcc` | min 0.994891 (52) | min 0.985936 (52) |
| `optimized_batched_traced_decode_pcc` | min 0.995826 (72) | min 0.983783 (72) |
| `optimized_bf16_cache_decode_pcc` | — | min 0.985500 (1) |
| `optimized_bf16_cache_prefill_pcc` | — | min 0.986295 (1) |
| `optimized_bfp8_policy_decode_pcc` | min 0.999872 (1) | min 0.999405 (1) |
| `optimized_bfp8_policy_prefill_pcc` | min 0.999636 (1) | min 0.999677 (1) |
| `optimized_conv_state_after_decode_pcc` | min 0.999961 (4) | — |
| `optimized_conv_state_pcc` | min 0.999961 (1) | — |
| `optimized_decode_pcc` | min 0.993345 (16) | min 0.984222 (16) |
| `optimized_full_context_conv_state_pcc` | min 0.999961 (1) | — |
| `optimized_full_context_decode_pcc` | min 0.997126 (1) | min 0.985773 (1) |
| `optimized_full_context_decode_scale` | min 0.982905 (1) | min 0.997669 (1) |
| `optimized_full_context_paged_k_cache_pcc` | — | min 0.999850 (1) |
| `optimized_full_context_paged_v_cache_pcc` | — | min 0.999856 (1) |
| `optimized_full_context_prefill_tail_pcc` | min 0.996715 (1) | min 0.984588 (1) |
| `optimized_full_context_prefill_tail_scale` | min 0.988131 (1) | min 0.997382 (1) |
| `optimized_full_context_recurrent_state_pcc` | min 0.999852 (1) | — |
| `optimized_long_prefill_pcc` | min 0.996719 (1) | min 0.985438 (1) |
| `optimized_packed_gate_up_decode_pcc` | min 0.996916 (1) | min 0.984662 (1) |
| `optimized_pad_alias_decode_pcc` | min 0.996522 (6) | min 0.984857 (6) |
| `optimized_pad_alias_prefill_pcc` | min 0.996586 (6) | min 0.987635 (6) |
| `optimized_paged_k_cache_pcc` | — | min 0.999850 (1) |
| `optimized_paged_v_cache_pcc` | — | min 0.999855 (1) |
| `optimized_prefill_pcc` | min 0.986326 (7) | min 0.985615 (7) |
| `optimized_real_weight_decode_pcc` | min 0.999852 (1) | min 0.999245 (1) |
| `optimized_real_weight_prefill_pcc` | min 0.999451 (1) | min 0.999159 (1) |
| `optimized_real_weight_traced_decode_pcc` | min 0.999049 (5) | min 0.998334 (5) |
| `optimized_recurrent_state_pcc` | min 0.999856 (1) | — |
| `optimized_traced_decode_replay_pcc` | min 0.996467 (3) | min 0.984644 (3) |
| `optimized_vs_fused_real_weight_pcc` | min 0.999517 (2) | min 0.999153 (2) |
| `optimized_vs_fused_same_policy_pcc` | min 1.000000 (2) | min 1.000000 (2) |

Minimum over all 480 PCC records: **0.983746**, against a bar of 0.995.
<!-- END GENERATED:correctness -->

## 5. Performance — before and after

<!-- GENERATED:before_after -->
| layer kind | phase | device time before | device time after | speed-up | end-to-end before | end-to-end after | ops before | ops after |
|---|---|---|---|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 25.830 ms | **18.018 ms** | **1.43x** | 28.572 ms | 19.820 ms | 66 | 65 |
| `linear_attention` | traced decode, 1 token, batch 1 | 2.352 ms | **1.334 ms** | **1.76x** | 2.418 ms | 1.391 ms | 67 | 68 |
| `linear_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 5.169 ms | **4.136 ms** | **1.25x** | 5.241 ms | 4.195 ms | 70 | 69 |
| `full_attention` | prefill, 2048 tokens | 17.780 ms | **9.450 ms** | **1.88x** | 19.279 ms | 9.841 ms | 28 | 29 |
| `full_attention` | traced decode, 1 token, batch 1 | 2.068 ms | **0.950 ms** | **2.18x** | 2.134 ms | 1.007 ms | 50 | 48 |
| `full_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 2.862 ms | **1.415 ms** | **2.02x** | 2.950 ms | 1.491 ms | 49 | 47 |
<!-- END GENERATED:before_after -->

### 5.1 Isolating the precision change from the layout change

The 2x2 below is the reason this stage carries two `fused-baseline` values rather than one "before"
number, and it says something that neither lever's own measurement does: **the two are not
additive, and the layout is only a win once the weights are block-float.**

At the fused stage's bfloat16 weights the shipped layout is *slower* than the fused layout - 2.8704
against 2.1278 ms for `full_attention` traced decode - and for `linear_attention` it does not
allocate at all. Two mechanisms, both visible in the configs the arms record:

* the DRAM-sharded matmul allocates its own **12** compute cores, while `ttnn.linear`'s heuristic
  spread the interleaved bfloat16 matmuls over 80-109. At 2 bytes per weight those matmuls are
  bandwidth-bound either way, so the wider grid wins; at 1.06 or 0.56 bytes they are not, and 12
  cores reading DRAM-sharded weights wins instead;
* the L1 budget is shared with the resident width-sharded activations, so a bfloat16 weight block
  forces `in0_block_w` down to 1 for the MLP and the down projection - which §3.2 measures at 10-21 %
  of the step - or does not fit at all.

So the honest attribution is: the precision change is worth 1.50x / 1.61x of traced decode on its
own, the layout change is worth **nothing** on its own, and together they are 1.78x / 2.17x. That is
a stronger claim than "each lever contributed half", and it is the one the measurement supports.

<!-- GENERATED:isolation -->
| arm | `linear_attention` traced decode | `linear_attention` prefill | `full_attention` traced decode | `full_attention` prefill |
|---|---|---|---|---|
| fused precision + fused layout | 2.3899 ms | 30.622 ms | 2.1278 ms | 22.625 ms |
| shipped precision + fused layout | 1.5933 ms | 20.327 ms | 1.3209 ms | 10.953 ms |
| fused precision + shipped layout | — | — | 2.8704 ms | 27.438 ms |
| shipped precision + shipped layout | 1.3428 ms | 19.224 ms | 0.9794 ms | 9.751 ms |
<!-- END GENERATED:isolation -->

### 5.2 Where the time goes now

<!-- GENERATED:breakdown -->
| bucket | `linear_attention` prefill | `linear_attention` decode | `linear_attention` decode_batch32 | `full_attention` prefill | `full_attention` decode | `full_attention` decode_batch32 |
|---|---|---|---|---|---|---|
| `matmul` | 7.4384 ms | 0.8596 ms | 0.8849 ms | 6.0687 ms | 0.7523 ms | 0.7548 ms |
| `layout` | 4.0832 ms | 0.1961 ms | 0.9319 ms | 0.3663 ms | 0.0284 ms | 0.0417 ms |
| `gated_delta_rule` | 2.7626 ms | — | — | — | — | — |
| `elementwise` | 2.5878 ms | 0.1356 ms | 0.2505 ms | 1.0625 ms | 0.0635 ms | 0.0623 ms |
| `state_update` | 0.7634 ms | 0.0553 ms | 0.7860 ms | — | — | — |
| `norm` | 0.3829 ms | 0.0277 ms | 0.0306 ms | 0.5543 ms | 0.0268 ms | 0.0291 ms |
| `batched_matmul` | — | 0.0599 ms | 1.2522 ms | — | — | — |
| `sdpa` | — | — | — | 0.9862 ms | 0.0504 ms | 0.4870 ms |
| `heads_and_cache` | — | — | — | 0.4116 ms | 0.0290 ms | 0.0402 ms |
| **total** | **18.0184 ms** | **1.3342 ms** | **4.1362 ms** | **9.4497 ms** | **0.9504 ms** | **1.4151 ms** |

Every op is classified: the `other` bucket is empty in all measured passes.
<!-- END GENERATED:breakdown -->

### 5.3 The dominant matmul rows, with the fidelity and dtype the profiler measured

This is the OPT-013 artifact: a precision policy is not implemented until the measured rows say so.

<!-- GENERATED:dominant_matmuls -->
| pass | op | instances per pass | device time per pass | math fidelity (measured) | bound | cores | DRAM % | FLOPs % |
|---|---|---|---|---|---|---|---|---|
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 17408` | 2 | 3012.7 us | `LoFi BF16 x BFP4 => BF16` | FLOP | 64 | 17.7 | 68.5 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 10240` | 1 | 1787.7 us | `HiFi2 BF16 x BFP8 => FP32` | FLOP | 64 | 17.2 | 67.9 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 17408 x 5120` | 1 | 1260.7 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 28.1 | 81.8 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 6144` | 1 | 515.1 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 29.4 | 70.7 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 6144 x 5120` | 1 | 498.2 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 30.4 | 73.1 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 128` | 1 | 134.5 us | `HiFi4 BF16 x FP32 => FP32` | SLOW | 32 | 35.8 | 45.1 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.8 us | `LoFi BF16 x BFP4 => BF16` | SLOW | 12 | 54.5 | 53.8 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 17408 x 5120` | 1 | 194.0 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 89.6 | 44.3 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 10240` | 1 | 184.3 us | `HiFi2 BF16 x BFP8 => FP32` | SLOW | 12 | 55.7 | 55.0 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 6144` | 1 | 72.7 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 85.9 | 42.4 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 6144 x 5120` | 1 | 72.7 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 80.6 | 39.8 |
| `linear_attention` decode | `MatmulDeviceOperation b={48} x 32 x 128 x 128` | 3 | 60.0 us | `HiFi4 FP32 x FP32 => FP32` | SLOW | 24 | 48.3 | 8.0 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation b={1536} x 32 x 128 x 128` | 3 | 1252.2 us | `HiFi4 FP32 x FP32 => FP32` | DRAM | 40 | 68.3 | 6.7 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.0 us | `LoFi BF16 x BFP4 => BF16` | SLOW | 12 | 54.4 | 53.8 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 17408 x 5120` | 1 | 194.0 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 89.9 | 44.4 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 10240` | 1 | 184.0 us | `HiFi2 BF16 x BFP8 => FP32` | SLOW | 12 | 55.5 | 54.8 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 6144 x 5120` | 1 | 76.3 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 80.0 | 39.5 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 6144` | 1 | 72.1 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 83.6 | 41.3 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 17408` | 2 | 3017.9 us | `LoFi BF16 x BFP4 => BF16` | FLOP | 64 | 17.7 | 68.4 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 17408 x 5120` | 1 | 1261.8 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 28.1 | 81.8 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 8192` | 1 | 776.3 us | `LoFi BF16 x BFP8 => BF16` | SLOW | 64 | 24.3 | 62.5 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 6144` | 1 | 515.6 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 29.4 | 70.6 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 6144 x 5120` | 1 | 497.2 us | `LoFi BF16 x BFP8 => BF16` | FLOP | 64 | 30.5 | 73.2 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 321.0 us | `LoFi BF16 x BFP4 => BF16` | SLOW | 12 | 54.1 | 53.4 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 17408 x 5120` | 1 | 193.8 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 89.8 | 44.3 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 5120 x 8192` | 1 | 94.2 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 86.7 | 42.8 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 6144 x 5120` | 1 | 72.0 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 85.8 | 42.4 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 5120 x 6144` | 1 | 71.2 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 86.2 | 42.6 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.3 us | `LoFi BF16 x BFP4 => BF16` | SLOW | 12 | 54.2 | 53.6 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 17408 x 5120` | 1 | 193.9 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 90.1 | 44.5 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 8192` | 1 | 94.3 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 86.6 | 42.8 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 6144 x 5120` | 1 | 75.0 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 81.9 | 40.4 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 6144` | 1 | 71.3 us | `LoFi BF16 x BFP8 => BF16` | DRAM | 12 | 86.2 | 42.6 |
<!-- END GENERATED:dominant_matmuls -->

### 5.4 Performance accounting

Three numbers from the same run, as the skill requires: the bytes the measured path must move
divided by this board's peak DRAM bandwidth, the profiler's device time, and the host-visible wall
time of the same signposted window. The peak is **derived** - every ``Bound=DRAM`` row reports both
the GB/s it achieved and the fraction of the roofline that is, so their ratio is the peak as the
profiler models it, and the median over those rows comes out at 512 GB/s.

<!-- GENERATED:accounting -->
| pass | bytes per step | implied peak DRAM | roofline | device time | end-to-end | op-to-op gap | fraction of roofline |
|---|---|---|---|---|---|---|---|
| `linear_attention/decode` | 329.9 MB | 512.0 GB/s | 0.6444 ms | 1.3342 ms | 1.3910 ms | 0.0510 ms | 0.483 |
| `linear_attention/decode_batch32` | 632.6 MB | 512.0 GB/s | 1.2356 ms | 4.1362 ms | 4.1950 ms | 0.0486 ms | 0.299 |
| `full_attention/decode` | 312.0 MB | 512.0 GB/s | 0.6093 ms | 0.9504 ms | 1.0070 ms | 0.0513 ms | 0.641 |
| `full_attention/decode_batch32` | 484.6 MB | 512.0 GB/s | 0.9466 ms | 1.4151 ms | 1.4910 ms | 0.0690 ms | 0.669 |
<!-- END GENERATED:accounting -->

### 5.5 What the remaining gaps are

**End-to-end minus device time is 51-69 us per decode step, and it is dispatch, not host work.**
The measured window contains eight `execute_trace` calls and one `synchronize_device`, with every
input tensor already resident; the gap equals the report's own summed op-to-op gap to within a few
microseconds, so it is the per-op dispatch cost of a 48-68 op trace rather than anything the host is
doing. Removing it would mean fewer ops, and the op count is already at the graph stage-2 arrived at.

**Device time minus roofline is the honest remaining headroom, and one row owns most of it.** The
`full_attention` decode step reaches 64 % of its roofline and the `linear_attention` one 48 %. Every
BFP8 projection is at 85-90 % of the DRAM roofline; the two **BFP4** gate/up rows come back
``Bound=SLOW`` at 54 % of the DRAM roofline *and* 54 % of the FLOP roofline on the 12 compute cores
the DRAM-sharded matmul allocates itself. Quartering the weight bytes moved those two matmuls off
the bandwidth ceiling, and 12 cores is then not enough compute to reach the other one. Section 3.9
is the search that followed from reading that row.

`linear_attention` decode at the advertised `max_batch` is the one pass where this stage's levers
reach least - 1.25x against 2.02x for `full_attention` - and the breakdown says why: at batch 32 the
step is dominated by the gated-delta-rule recurrence, whose state is float32 by contract and whose
core grids stage 2 already swept at both decode regimes. The weight-bound part of that step improved
exactly as much as at batch 1; the state-bound part is not a precision or layout question, and its
remaining lever is the recurrent-state *format*, which stage 2 explicitly handed to the stage that
owns the decode state rather than to this one.

## 6. What was assessed and not taken

Recorded here so "no remaining decoder optimization" is a claim with evidence behind it.

| candidate | why not |
|---|---|
| **BFP4 attention projections** (`wqkv`, `wgate`, `o_proj`, `in_proj_z`, `out_proj`) | **measured on real weights and rejected**: `linear_attention` decode 0.962769 and `full_attention` traced decode below the bar (§2.1's real-weight table).  OPT-007's mandatory trial, run with the cache-consuming traced follow-on it asks for, so the rejection is on model-visible output PCC rather than on a raw-cache diagnostic |
| **BFP4 MLP down projection** | **measured on real weights and rejected**: `full_attention` falls below the bar (§2.1's real-weight table).  The gate/up pair at BFP4 is kept; this is the FF2 sensitivity the skill predicts, with the number attached |
| **BFP4 `in_proj_qkv`** | **measured and rejected**: synthetic PCC 0.952312 / 0.956293, and it is the role whose float32 output is the carried conv state.  BFP8 + HiFi2 is kept |
| **HiFi2 at decode** | **measured and rejected**: +29 % / +43 % of the traced step for 6e-4 of PCC (§3.1).  The obvious assumption - that fidelity is free on a DRAM-bound matmul - stops being true once the weights are block-float |
| **blanket float32 destination accumulation** (stage 2's setting) | **measured and rejected**: it halves matmul throughput at these shapes and only `in_proj_qkv`/`in_proj_ab` need it.  Retained as `PrecisionPolicy.fp32_dest_acc_all` for the baseline arm |
| **bfloat16 KV cache** | BFP8 is kept: identical PCC to 1e-4 and half the bytes.  bfloat16 remains selectable and `test_bfloat16_kv_cache_still_works` keeps it working, because a serving stage may want the headroom |
| **decode core counts 1, 2, 4, 8, 16** | measured (§3.2).  1 and 2 are L1 blockers with the exact byte counts recorded; 4, 8 and 16 are 52 %, 10 % and 5-8 % slower than 32 |
| **packed gate/up at decode** | L1 blocker at the winning 32 cores (its `per_core_N` is twice the split form's and the circular buffers clash with the resident sharded activations), and 2 % slower at 16 cores where both are legal (§3.3) |
| **packed gate/up at prefill** (stage 2's choice) | **measured and reversed**: 19.634 vs 19.415 and 10.160 vs 9.908 (§3.3).  Splitting both phases also removes the packed weight entirely |
| **SiLU fused into the split gate matmul's epilogue** | **measured and rejected**: 1.3891 vs 1.3422 and 1.0375 vs 0.9907 (§3.3).  It stays on the multiply |
| **packing `wqkv`+`wgate` and `in_proj_qkv`+`in_proj_z`** | re-measured at this stage's dtypes and layout rather than inherited from stage 2 (§3.7) |
| **explicit prefill grids other than the derived one** | measured (§3.4): 8x10 and 8x4 are 15-40 % slower, 11x10 returns non-finite output because a DRAM width-sharded weight needs one bank per compute column, and an explicit 8x8 reproduces the derived grid to within the spread |
| **prefill `in0_block_w` 2 and 8-everywhere** | measured (§3.4): 2 is 5-11 % slower; 8 is 1.5 % faster for `full_attention` and an L1 blocker on `linear_attention`'s `in_proj_qkv`, which is why the bound is computed per role |
| **SDPA decode at 1, 2, 4 and 16 cores per head-batch** | measured (§3.6).  1 is stage 1's pinned value and 5.6 % slower; 2 sits on the edge of the scale tolerance for a documented reason; 4 is 1.2 % slower than 8; 16 exceeds L1 at this k chunk, identically on the stock build |
| **a smaller SDPA `k_chunk`** | measured by the kernel investigation: at one core per head-batch, `k_chunk` 64 / 128 / 256 / 512 give 22587 / 15418 / 11626 / 9774 us and worst alpha 1.676 / 1.336 / 0.989 / 1.017.  512 wins on **both** axes, so the inherited value stands |
| **the default (auto) SDPA decode program config** | measured: 37152 us and alpha 1.669 even after the kernel fix, because it picks `k_chunk` 128 and turns `exp_approx_mode` on.  It is 23x better than before the fix and still must not be used |
| **fixing the five-DEST-tile declaration at its source** (`compute_common.hpp` / the LLK SFPU header) | deliberately not done: it is outside this stage's writable scope and would change every caller of that helper.  Named as the recommended upstream follow-up in `sdpa/AUTOFIX_SDPA.md` |
| **multi-device topology work** - residual layout across a fractured boundary, reduce-scatter vs all-gather, fused CCL+matmul, persistent CCL buffers | **not applicable, not untried**: this stage is a single Blackhole chip on a 1x1 mesh and there is no collective in either layer kind.  The whole family has no instance here; the multichip stage owns it |
| **`ttnn.sparse_matmul` and routed active-expert execution** | not applicable: both layer kinds have a dense SwiGLU MLP, no router and no experts |
| **LM head, logits movement, sampling and token feedback** | not applicable to a decoder-layer stage: this module ends at the layer output.  The full-model stage owns the terminal path |
| **larger-batch throughput tuning** | out of scope by the skill's own rule - batch 1 is the optimization target - but batch-32 correctness and latency are both measured, and the batch-32 decode is a different graph rather than a wider tensor |
| **the recurrence matmul core grids, the gated-norm group threshold, the conv-state layout** | stage 2's, re-verified unchanged.  Its work log swept eighteen grids at both decode regimes; nothing this stage changed moves that measurement, and its own §6 hands the recurrent-state *format* question to the stage that owns the decode state |

### 6.1 Every piece of `tt-perf-report` advice in the committed optimized reports

Advice is left **on** in the reports these decisions were made against, and every distinct line in
the six committed optimized reports is accounted for here.  In the two prefill reports and in the
`full_attention` decode reports every weight-projection row reads `✅ Optimized` except the two BFP4
gate/up rows, chased in §3.8.  What is left is a cluster on the three float32 gated-delta-rule
recurrence matmuls and on `in_proj_ab` - stage 2's ops, inherited here:

| advice line | rows it fires on | tried | outcome |
|---|---|---|---|
| `Use HiFi2 or HiFi4 with BF16 activations for improved accuracy` | every LoFi projection | yes, §3.1 | **rejected with evidence**: HiFi2 at decode costs +29 % / +43 % of the traced step for 6e-4 of PCC, because block-float weights moved those matmuls off the bandwidth ceiling |
| `No output subblock size found` | every DRAM-sharded row | n/a | the DRAM-sharded program config has no output-subblock fields to set; the report cannot see them because they do not exist.  §3.8 measures the config classes that *do* have them |
| `HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4` | the 3 recurrence matmuls (float32 x float32) and `in_proj_ab` | yes, table below | measured at HiFi4 / HiFi2 / LoFi on the real shape at both decode regimes |
| `in0_block_w=1 is small, try in0_block_w=2 or above` | the 3 recurrence matmuls | yes, table below | measured with an explicit `MatmulMultiCoreReuseProgramConfig` at `in0_block_w` 1 / 2 / 4 |
| `Output subblock 1x1 is small, try out_subblock_h * out_subblock_w >= 2` | the 3 recurrence matmuls | yes, table below | measured at output subblock 1x1 / 1x2 / 1x4 |
| `If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)` | the 3 recurrence matmuls | yes, table below | measured with the activation uploaded to L1 |
| `Try a DRAM-sharded program config` | the batch-32 recurrence matmuls | **rejected with a calculation, not a measurement** | the "weight" of those matmuls is the carried recurrent state, and a DRAM-sharded matmul needs it width-sharded in DRAM.  That is a change to the *format* of a persistent tensor that `prepare_decode_state` writes and that the HF cache comparison reads - the same state-format boundary stage 2's own §6 handed to the stage that owns the decode state.  It is also 100 MB at the advertised `max_batch`, so the L1 variant of the same advice cannot apply there either: 3.1 MB at batch 1 could be sharded into L1, 100 MB cannot, and a knob that only works at batch 1 is not a decode policy |

<!-- GENERATED:recurrence_advice -->
| head problems | candidate | median | PCC vs float32 | blocker |
|---|---|---|---|---|
| 48 (batch 1) | core_grid 6x4, HiFi4 (shipped), in0 DRAM | 37.3 us | 1.000000 | — |
| 48 (batch 1) | core_grid 6x4, HiFi2 (report advice), in0 DRAM | 35.1 us | 0.999994 | — |
| 48 (batch 1) | core_grid 6x4, LoFi, in0 DRAM | 35.5 us | 0.999897 | — |
| 48 (batch 1) | core_grid 6x4, HiFi4, in0 L1 (report advice) | 35.5 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x1 | 35.7 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x2 | 34.6 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x4 | 34.7 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x1 | 35.2 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x2 | 34.4 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x4 | 34.6 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x1 | 35.8 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x2 | 35.5 us | 1.000000 | — |
| 48 (batch 1) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x4 | 35.9 us | 1.000000 | — |
| 1536 (batch 32) | core_grid 10x4, HiFi4 (shipped), in0 DRAM | 455.2 us | 1.000000 | — |
| 1536 (batch 32) | core_grid 10x4, HiFi2 (report advice), in0 DRAM | 456.3 us | 0.999994 | — |
| 1536 (batch 32) | core_grid 10x4, LoFi, in0 DRAM | 456.1 us | 0.999897 | — |
| 1536 (batch 32) | core_grid 10x4, HiFi4, in0 L1 (report advice) | 455.1 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x1 | 453.2 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x2 | 454.6 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=1 subblock 1x4 | 452.8 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x1 | 451.3 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x2 | 452.9 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=2 subblock 1x4 | 456.0 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x1 | 473.9 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x2 | 478.2 us | 1.000000 | — |
| 1536 (batch 32) | MatmulMultiCoreReuse in0_block_w=4 subblock 1x4 | 478.9 us | 1.000000 | — |
<!-- END GENERATED:recurrence_advice -->

Every one of those four is **inside the run-to-run spread** on the real shape at both decode regimes:
at 48 head problems the shipped HiFi4/DRAM/`core_grid` form is 37.3 us and the eleven alternatives -
HiFi2, LoFi, the activation in L1, and `in0_block_w` 1/2/4 crossed with output subblock 1x1/1x2/1x4 -
land between 34.4 and 35.8 us against a stdev of 15 us.  HiFi2 is nominally 2.2 us faster and costs
PCC 1.000000 -> 0.999994 on the carried state; LoFi costs 0.999897.  So the advice is *tried and
measured*, none of it is a win outside the noise on a group that is 4.5 % of the step, and none of it
is taken - which is a different statement from "not applicable", and the numbers are here so a later
stage does not have to rediscover it.

### 6.2 Every `Bound=SLOW` row in the committed optimized reports

<!-- GENERATED:slow_rows -->
| pass | op | instances per pass | device time per pass | share | cores | DRAM % | FLOPs % |
|---|---|---|---|---|---|---|---|
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 128` | 1 | 134.5 us | 0.75 % | 32 | 35.8 | 45.1 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 6144 x 64` | 1 | 119.7 us | 0.66 % | 64 | 43.2 | 15.2 |
| `linear_attention` prefill | `MatmulDeviceOperation 2048 x 64 x 6144` | 1 | 109.7 us | 0.61 % | 110 | 46.7 | 9.7 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.8 us | 24.05 % | 12 | 54.1-54.5 | 53.4-53.8 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 10240` | 1 | 184.3 us | 13.81 % | 12 | 55.3-55.7 | 54.6-55.0 |
| `linear_attention` decode | `MatmulDeviceOperation b={48} x 32 x 128 x 128` | 3 | 59.9 us | 4.49 % | 22-24 | 40.7-49.9 | 7.3-8.2 |
| `linear_attention` decode | `MatmulDeviceOperation 32 x 5120 x 128` | 1 | 15.0 us | 1.12 % | 4 | 38.3-39.0 | 50.2-51.1 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.0 us | 7.74 % | 12 | 54.2-54.5 | 53.5-53.8 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 10240` | 1 | 184.0 us | 4.45 % | 12 | 55.5-55.7 | 54.8-55.0 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 6144 x 64` | 1 | 16.3 us | 0.39 % | 2 | 14.1-14.3 | 55.4-55.9 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 128` | 1 | 15.0 us | 0.36 % | 4 | 38.4-38.9 | 50.3-50.9 |
| `linear_attention` decode_batch32 | `MatmulDeviceOperation 32 x 64 x 6144` | 1 | 7.3 us | 0.18 % | 16 | 31.3-32.7 | 15.4-16.1 |
| `full_attention` prefill | `MatmulDeviceOperation 2048 x 5120 x 8192` | 1 | 776.3 us | 8.21 % | 64 | 24.3 | 62.5 |
| `full_attention` decode | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 321.0 us | 33.78 % | 12 | 53.9-54.5 | 53.2-53.8 |
| `full_attention` decode_batch32 | `MatmulDeviceOperation 32 x 5120 x 17408` | 2 | 320.3 us | 22.64 % | 12 | 54.1-54.5 | 53.5-53.8 |

15 `Bound=SLOW` op groups across the six committed optimized reports.
<!-- END GENERATED:slow_rows -->

## 7. Commands

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh

# the optimized suite (long-context cases skipped without --long-context)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py -v -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/suite_main.log

# full advertised context
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
  -k test_full_advertised_context --long-context -v -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/long_context.log

# watcher-clean optimized correctness run (kept separate from every profiler run)
TT_METAL_WATCHER=10 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
  -k "traced_decode or repeated_runs or decode_pcc or batched_users or linear_state" -v -s \
  2>&1 | tee models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/watcher_run.log

# document/artifact gate (no device)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder_docs.py -v

# candidate sweeps
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_matmul_policy.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_optimized.py policy
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_optimized.py geometry
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_optimized.py prefill
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_optimized.py isolation
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_projection_packing.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_real_weight_policy.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_blockfloat_distribution.py

# before/after profiling, one (kind, phase, arm) triple at a time
for kind in linear_attention full_attention; do
  for phase in prefill decode decode_batch32; do
    for impl in fused optimized; do
      doc/optimized_decoder/probes/run_perf.sh "$kind" "$phase" "$impl"
    done
  done
done
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/make_perf_summary.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/make_doc_tables.py

# collect every recorded number
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/logs/{suite_main,long_context,watcher_run}.log \
  --out models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/pcc_evidence.json
```

## 8. The `$optimize` checklist, with evidence

| item | status | evidence |
|---|---|---|
| Decoder path fully traced with no host fallbacks | done | `test_no_runtime_host_fallback` (source scan **and** a live pass with `from_torch`/`to_torch`/`as_tensor` stubbed to raise); every decode number here is a captured-trace replay |
| Decode activations width-sharded in L1 across norm, attention, residual, MLP and output projection | done | §3.2; `test_decode_stream_stays_in_l1` intercepts the norms and residual adds and asserts no `hidden_size`-wide decode tensor is interleaved; `test_no_relayout_or_host_ops_in_measured_decode` pins the reshard count as an equality |
| Prefill activations DRAM interleaved; 2D program configs for large prefill matmuls | done | §3.4; `config_summary()["roles"][*]["prefill_program_config"]`-equivalent derivation in `_prefill_program_cfg`, and the candidate table |
| Operation-topology audit completed | done | §1 and §1.1: current op sequence, repeated same-input matmuls, collectives (none), reshard/layout conversions, candidate replacements, dtype/fidelity constraints, action taken |
| Multi-device topology families measured as coherent families | **not applicable** | single chip, 1x1 mesh, no collective in either layer kind (§6) |
| Lower-movement residual candidates measured without an old-contract restore | **not applicable** for CCLs; the on-device analogue **is** measured | §3.2: the sharded residual is carried through the whole step, and the isolation arm that restores the DRAM residual is measured separately rather than bundled |
| Best-candidate comparison completed | done | §2 candidate ledger, §5.1 isolation 2x2, and `test_optimized_beats_fused_traced_decode`, which measures both arms back to back in one session |
| Final default performance reproduced the selected best candidate | done | §5's before/after table is a profiler run of the shipped default, and the isolation arm's shipped row agrees with it |
| Final dtype/fidelity policy verified in the measured runtime rows | done | §5.3 lists the dominant matmul rows with the profiler's own Math Fidelity column; `test_dominant_matmul_rows_prove_the_policy` fails if a dominant row does not carry a block-float operand while the policy claims one |
| SDPA and other optimized composite ops used | done | §1.1: `chunk_gated_delta_rule`, chunked/paged SDPA, `paged_fill_cache`/`paged_update_cache`, `nlp_create_qkv_heads[_decode]`, `nlp_concat_heads[_decode]`, `rotary_embedding_hf`, `rms_norm`, `addcmul` - inherited and re-verified; no hand-built attention primitive remains |
| Fused/packed repeated same-input projections | done | §3.3 (gate/up, kept split with measurements) and §3.7 (`wqkv`+`wgate`, `in_proj_qkv`+`in_proj_z`, re-measured at the new dtypes) |
| Explicit `memory_config`, `program_config`, `compute_kernel_config` for important ops | done | every projection in both phases, both norms, and SDPA; `config_summary()` serialises all of it and the docs test cross-checks it against the report |
| Program configs swept per dominant role: core grid, larger `in0_block_w`, output subblocks, memory configs, compute kernel config | done | §3.2 (decode: core count and per-role `in0_block_w`, every role at its maximum legal value), §3.4 (prefill: grid, `in0_block_w`, output block), §3.1 (fidelity per group) |
| Decode compute fidelity swept as a performance knob per projection group | done | §3.1: LoFi vs HiFi2 vs HiFi4 at fixed dtype, with the measured +29 %/+43 % cost of HiFi2 at decode |
| Attention projection dtype/fidelity swept separately from MLP | done | §3.1 and §2.1: the BFP4 attention trial is a separate arm, run on real weights with a traced cache-consuming follow-on, and rejected with the exact PCCs |
| BFP4/LoFi trials for the dense MLP before lower-priority prefill advice | done | the MLP was the first group moved; gate/up **shipped** at BFP4, down rejected with real-weight PCC |
| Shard specs and core grids dividing tensor dimensions cleanly, as large as the shape allows | done | §3.2: 32 cores is the largest value that divides every activation width, computed from the shapes; prefill columns pinned to the DRAM bank count for exact `per_core_N` |
| DRAM-sharded decode matmuls | done | §3.2; worth 13.3 %/19.4 % isolated |
| Collective topology minimized | **not applicable** | no collective (§6) |
| Fused matmul-CCL ops | **not applicable** | no collective (§6) |
| Persistent/preallocated CCL buffers | **not applicable** | no collective (§6) |
| MoE routed active-expert path | **not applicable** | dense SwiGLU MLP, no router (§6) |
| LM head, sampling, token feedback in the optimized token-out path | **not applicable** | decoder-layer stage; the module ends at the layer output (§6) |
| LM head optimized for DRAM-sharded matmuls | **not applicable** | same |
| Reduced precision/fidelity experiments on real weights and input activations | done | §2.1's real-weight table plus `logs/probe_real_weight_policy.log`; the synthetic/real discrepancy is quantified in `logs/probe_blockfloat_distribution.log` rather than waved away |
| Performance accounting reconciled: roofline, device time, end-to-end from the same run | done | §5.4, generated from `perf_summary.json`, with the peak DRAM bandwidth derived from the report's own columns |
| Batch capability preserved; larger batch tested to 32 | done | `test_batched_users` (4/16/32), `test_traced_decode_batched` (4/32), `test_repeated_runs_stable` (1/32), and the batch-32 decode window is measured and reported separately |
| Functional checks still pass against the optimized path | done | `logs/suite_main.log` |
| Prefill and decode PCC at the acceptance bar for every layer kind | done | §4, and the README's note on which evidence carries the bar |
| Paged KV cache and warmed trace replay still correct | done | `test_linear_state_and_kv_cache_match_reference`, `test_alternate_page_block_size`, `test_traced_decode_pcc`, `test_traced_decode_batched` |
| Runtime fallback audit clean | done | `test_no_runtime_host_fallback`, `test_no_relayout_or_host_ops_in_measured_decode`, `test_no_redundant_relayout_in_measured_prefill` |
| Stress / repeated-run coverage | done | `test_repeated_runs_stable` (six whole cycles at batch 1 and 32, bit-identical, per-bank DRAM unchanged) and `test_traced_decode_stress` (200 traced replays) |
| Warmed prefill and decode latency before/after | done | §5 |
| `tt-perf-report` output with advice enabled | done | `tracy/*/*/[phase]_perf_report.txt` (advice on) plus a `.noadvice.txt` companion; [`probes/run_perf.sh`](probes/run_perf.sh) leaves advice enabled in the guiding report |
| Watcher clean, separate from profiler runs | done | `logs/watcher_run.log`, `TT_METAL_WATCHER=10`, no offender line; `test_watcher_log_is_clean` |

## 9. Best-candidate signoff, stage review and checkpoint commits

### 9.1 The final default is the fastest **correct** candidate measured

The skill's rule is that the shipped path must beat the strongest correct baseline *and* every
material candidate from this stage, and that a candidate is not rejected for being faster if it is
also correct.  Both halves are checked:

* **against the baseline**: 1.76x / 2.18x traced decode and 1.44x / 1.90x prefill, measured by the
  same script in the same session (§5).  `test_optimized_beats_fused_traced_decode` re-measures both
  arms back to back on the device so the claim cannot drift.
* **against every candidate**: three candidates in the ledger are *faster* than the shipped one and
  all three are rejected on **real-checkpoint** correctness, not on preference - BFP4 attention
  (`linear_attention` decode 0.962769), BFP4 MLP including the down projection (`full_attention`
  below the bar) and BFP4 `in_proj_qkv` (0.952312 synthetic, and it is the carried conv state's
  producer).  Every candidate that passes the bar is slower than the shipped configuration.
* **the final default reproduces the selected candidate**: the geometry sweep's best row is
  1.3422 ms / 0.9794 ms in-model wall clock, and the shipped default's profiler run comes out at
  1.334 ms / 0.950 ms device time with 1.392 / 1.008 ms end to end - the same configuration measured
  two ways, not a candidate number copied forward.

### 9.2 Checkpoint commits

Repo `tt-metal`, branch `agentic-research/hous/qwen3.6-27b-v2`, parent `a6a91a89da5` (the fused
stage's last commit).  Local only; nothing pushed.

| commit | subject |
|---|---|
| `8916f99e2b7` | Qwen3.6-27B optimized decoder: implementation, tests and probes |
| `0f04f65c50a` | Qwen3.6-27B optimized decoder: sdpa_decode diagnosis artifacts and advice probes |
| `684db2301f5` | Qwen3.6-27B optimized decoder: float32 destination accumulation for prefill wqkv |
| `63aeb65dd30` | Qwen3.6-27B optimized decoder: evidence, documents and capability contract |

The stage's changes are isolated: `git diff --stat a6a91a89da5..HEAD` touches only
`models/autoports/qwen_qwen3_6_27b/**` and the one shared device kernel
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`,
whose diagnosis, correctness sweep and blast-radius run are `sdpa/AUTOFIX_SDPA.md`.  The three files
that remain dirty in the worktree - `.agents/notes/gdn.md`,
two files under `.agents/prompts/model_bringup_multigoal/`, plus the untracked
`scripts/check_agent_prompt_lengths.py` - were already
dirty when this stage started and are deliberately **not** in any of these commits.

### 9.3 Stage review

