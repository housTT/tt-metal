# Qwen3.6-27B optimized decoder — work log

Stage: `optimized_decoder`. Code under test: `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`.
Baseline: `tt/fused_decoder.py` (previous stage), re-measured in this stage on the same device
through the same harness. The two headline runs are back-to-back invocations of `probes/sweep.py`
rather than one process; the in-process comparison is the suite's
`test_optimized_{decode,prefill}_beats_fused`, which agrees with them (§16).

Hardware: one Blackhole chip of a p300c board (`TT_VISIBLE_DEVICES=2`), 11x10 worker grid,
8 DRAM banks. Environment: `doc/fused_decoder/ttenv.sh` for correctness/latency,
`ttenv_profiler.sh` for Tracy runs (the shared install is built with `ENABLE_TRACY=OFF`).

Everything below is measured. Commands are copy-pasteable from
`/home/ttuser/dev/qwen/rundir`.

---

## 0. Method

Two harnesses do all the work.

**`probes/sweep.py`** — opens the mesh once, builds the HF reference once per layer kind, then
for each named candidate builds the layer, checks prefill and traced-decode PCC against that
reference and times a warmed prefill and 32 warmed traced-decode replays. Running candidates in
one process is what makes a sweep affordable: the real-weight load and the CPU reference
forward dominate a per-candidate pytest invocation.

```bash
cd /home/ttuser/dev/qwen/rundir && source ./ttenv.sh
ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder
python $ART/probes/sweep.py --group precision --kinds linear,full --seq 2048 --real
python $ART/probes/sweep.py --group topology  --kinds linear,full --seq 2048 --real
```

`--real` uses the real Qwen3.6-27B checkpoint weights for the layer. Every precision and
topology decision below was taken on real weights, per `$optimize` OPT-012; the synthetic
weights of `weight_stats.json` are used only by the test suite's stress cases.

**`probes/matmul_sweep.py`, `probes/decode_matmul_families.py`, `probes/prefill_sweep.py`** —
model-free program-config searches on the real matmul shapes and dtypes. A whole geometry sweep
costs seconds instead of a layer rebuild per candidate, which is what made an exhaustive
`in0_block_w` / block-geometry search practical.

Baseline check: the optimized module built with `FUSED_BASELINE_PRECISION` and the fused module
measure identically (`linear_attention` 2.277 ms decode / 51.7 ms prefill for both), so the copy
is faithful and every delta below is attributable to a named change.

---

## 1. Operation-topology audit of the measured path

Read from the fused stage's `tt-perf-report` tables
(`doc/fused_decoder/tracy/*/{prefill,decode}_perf_report.csv`) together with the code.

### Decode, per token, one layer (fused stage, 8 traced replays)

| op | linear_attention | full_attention | note |
|---|---|---|---|
| `Matmul 32 x 5120 x 34816` (MLP gate+up, packed) | 867 us / 38.4 % | 868 us / 39.3 % | BF16 weights, HiFi4 |
| `Matmul 32 x 17408 x 5120` (MLP down) | 432 us / 19.2 % | 438 us / 19.8 % | BF16, HiFi4 |
| `Matmul 32 x 5120 x 10240` (GDN in_proj_qkv) | 262 us / 11.6 % | — | |
| `Matmul 32 x 6144 x 5120` (out / o_proj) | 153 us / 6.8 % | 155 us / 7.0 % | |
| `Matmul 32 x 5120 x 6144` (in_proj_z / wgate) | 149 us / 6.6 % | 148 us / 6.7 % | |
| `Matmul 32 x 5120 x 8192` (wqkv) | — | 202 us / 9.1 % | |
| `SdpaDecode` | — | 257 us / 11.7 % | |
| `ReshapeView` x10 | 104 us / 4.6 % | — | GDN head reshapes |
| `LayerNorm` x5/x4 | 40 us | 32 us | already width-sharded (F15) |
| **total device** | **2257 us** | **2206 us** | |

Findings, in the order the audit produced them:

1. **Every weight matmul ran BF16 at HiFi4 with fp32 accumulation.** 82 % of decode is weight
   matmuls and they are DRAM-bandwidth bound, so the stored weight dtype *is* the decode time.
   This is the single largest lever and it is a precision question, not a topology one → §2.
2. **No matmul used a program config.** All ran the ttnn-chosen interleaved program → §3.
3. **The decode residual stream round-tripped through DRAM** between every sub-block; the two
   norms sharded and unsharded around themselves → §4.
4. **The MLP gate/up projection was packed into one matmul** — the fusing stage's choice, which
   costs two output slices. Whether that is still right under a DRAM-sharded decode matmul is a
   different question → §5.
5. **No repeated same-input matmul remained unfused**: the fused stage already packed Q/K/V into
   `wqkv`, `b|a` into `in_proj_ba`, and gate/up into `mlp_gate_up`. `wgate` and `in_proj_z`
   consume the same activation as `wqkv`/`in_proj_qkv` and could in principle join them; both
   are measured in §5.
6. **No collectives** — this is a single-chip decoder stage, so the whole multi-device topology
   family of `$optimize` (residual layout across a mesh, fused CCL+matmul, persistent CCL
   buffers) does not apply here. It is the multichip stage's contract.
7. **No MoE** — Qwen3.6-27B's `qwen3_5` decoder has a dense SwiGLU MLP, so the
   `ttnn.sparse_matmul` active-expert path does not apply.
8. **Prefill is a different regime**: `linear_attention` prefill spends 39 ms of 49 ms in the
   gated-delta-rule machinery (196 `BinaryNg`, 332 `Slice`, 3 `Ternary` on 84 MB fp32 tensors)
   and only 9.6 ms in weight matmuls → §6 and §7.

### Candidate table

| # | candidate | action | evidence |
|---|---|---|---|
| O1 | per-tensor-group weight dtype + math fidelity | **kept** | §2 |
| O2 | DRAM-sharded decode matmuls | **kept** | §3 |
| O3 | width-sharded L1 decode residual through both norms and both residual adds | **kept** | §4 |
| O4 | split gate/up instead of packed | **kept** | §5 |
| O5 | `ttnn.transformer.gated_delta_attn_seq` for the GDN prefill | **rejected** (accuracy) | §7 |
| O6 | 2D program configs for prefill matmuls | **kept** (forced by O2) | §6 |
| O7 | fuse `wgate` into `wqkv` / `in_proj_z` into `in_proj_qkv` | **rejected** | §5 |
| O8 | gated-delta-net causal conv in bfloat16 | **kept** | §8 |
| O9 | one reshape instead of two in the gated-delta-net decode head split | **kept** | §14 |
| O10 | shape-aware program configs for the batched delta-rule matmuls | **kept** | §17 |
| O11 | matching ttnn's own row-wise decode shard grid | **rejected** (sharded layernorm needs a rectangle) | §19 |
| O12 | the gated-delta-rule chunk loop's recurrent state in L1 | **kept** | §19 |
| — | `ttnn.conv1d` for the gated-delta-net causal conv | **rejected** (exact L1 blocker) | §18 |

---

## 2. O1 — precision and fidelity, one tensor group at a time

`PrecisionPolicy` names each group separately: MLP gate/up, MLP down, attention QKV / gate /
output, gated-delta-net QKV / z / out / b|a, KV cache, and the math fidelity of the weight
matmuls and of SDPA. The delta-rule *state* math is deliberately outside the policy: it keeps
HiFi4 + `fp32_dest_acc_en` because the functional stage measured real catastrophic cancellation
in the triangular inverse, and it is a few percent of decode time.

Sweep: `probes/sweep.py --group precision --kinds linear,full --seq 2048 --real`
→ `logs/sweep_precision_real.log`. Real Qwen3.6-27B weights, layer 0 and layer 3.

### linear_attention (real weights)

| candidate | prefill ms | decode ms | prefill PCC | decode PCC |
|---|---|---|---|---|
| fused stage baseline | 51.75 | 2.277 | 0.999967 | 0.999998 |
| optimized module, fused precision | 51.72 | 2.277 | 0.999967 | 0.999998 |
| MLP BFP8 + HiFi2 | 47.43 | 1.805 | 0.999939 | 0.999997 |
| MLP BFP8 + LoFi | 46.24 | 1.808 | 0.999595 | 0.999969 |
| MLP gate/up BFP4, down BFP8, LoFi | 46.22 | 1.664 | 0.999377 | 0.999965 |
| + MLP down BFP4 | 46.42 | 1.664 | 0.999114 | 0.999959 |
| + attention/GDN projections BFP8 | 46.43 | **1.492** | 0.999465 | 0.999909 |
| + attention/GDN projections BFP4 | 46.62 | 1.470 | **0.987644** | 0.997102 |
| + BFP8 KV cache (= default policy) | 46.36 | 1.491 | 0.999465 | 0.999909 |
| default, SDPA HiFi4 | 46.37 | 1.493 | 0.999465 | 0.999909 |
| default, SDPA LoFi | 46.40 | 1.493 | 0.999465 | 0.999909 |

### full_attention (real weights)

| candidate | prefill ms | decode ms | prefill PCC | decode PCC |
|---|---|---|---|---|
| fused stage baseline | 19.56 | 2.231 | 0.999969 | 0.999993 |
| optimized module, fused precision | 20.22 | 2.265 | 0.999969 | 0.999993 |
| MLP BFP8 + HiFi2 | 14.98 | 1.755 | 0.999936 | 0.999968 |
| MLP BFP8 + LoFi | 14.10 | 1.756 | 0.999793 | 0.999885 |
| MLP gate/up BFP4, down BFP8, LoFi | 13.97 | 1.614 | 0.999281 | 0.999637 |
| + MLP down BFP4 | 13.97 | 1.613 | **0.993112** | **0.993970** |
| + attention/GDN projections BFP8 | 14.10 | 1.467 | 0.999284 | 0.999654 |
| + attention/GDN projections BFP4 | 13.97 | 1.456 | **0.994815** | 0.998573 |
| + BFP8 KV cache (= default policy) | 13.41 | **1.435** | 0.999270 | 0.999655 |
| default, SDPA HiFi4 | 13.96 | 1.469 | 0.999268 | 0.999656 |
| default, SDPA LoFi | 13.08 | 1.432 | 0.999146 | 0.998990 |

### Decisions

* **MLP gate/up = BFP4, LoFi.** Worth 144 us of decode against BFP8 and 1.2 ms of prefill
  against HiFi2, at a prefill PCC of 0.9993 against a 0.995 bar. Kept.
* **MLP down = BFP8.** BFP4 on the down projection buys **nothing** in decode (1.664 → 1.664 ms
  on `linear_attention`, 1.614 → 1.613 ms on `full_attention`) and costs
  `full_attention` prefill PCC 0.999281 → **0.993112**, i.e. below the bar. Rejected on real
  weights with both numbers. This is the mandatory FF2/down BFP4 trial of `$optimize`.
* **Attention and gated-delta-net projections = BFP8, LoFi.** Worth 172 us / 147 us of decode.
* **BFP4 attention/GDN projections rejected on accuracy, not on preference.** It is the
  mandatory OPT-007 trial and it was run on real weights: `linear_attention` prefill PCC
  **0.987644** and `full_attention` prefill PCC **0.994815**, both below the 0.995 bar, for
  22 us and 11 us of decode respectively. The `linear_attention` failure is the informative
  one: those projections feed the gated-delta-rule state recurrence, whose conditioning the
  functional stage already documented. Re-measured on the final DRAM-sharded topology in §5 so
  the rejection is not a screening-only result.
* **KV cache = BFP8.** Free accuracy-wise (0.999284 → 0.999270 prefill, 0.999654 → 0.999655
  decode) and worth 32 us of decode plus 0.7 ms of prefill, and it halves the cache footprint.
* **SDPA fidelity = HiFi2.** HiFi4 costs 34 us of decode for no accuracy gain; LoFi saves 3 us
  of decode and 0.33 ms of prefill but costs decode PCC 0.999655 → 0.998990. HiFi2 is the
  middle and is kept.
* **`in_proj_ba` stays float32 at HiFi4.** It is a 2.6 MB weight feeding the `sigmoid` and
  `softplus` gates of the state recurrence; its dtype is an accuracy choice, not a bandwidth
  one.
* **LoFi vs HiFi2 for the weight matmuls is a real knob, not an implied one.** At BFP8 the two
  are identical in decode (1.805 vs 1.808 ms) because decode is bandwidth-bound, but LoFi is
  1.2 ms faster in prefill, where the same matmuls are compute-bound. LoFi is kept because the
  final policy is BFP4-dominated, where LoFi is the matching fidelity.

Net effect of O1 alone: decode 2.277 → 1.491 ms (`linear_attention`) and 2.231 → 1.435 ms
(`full_attention`); prefill 51.7 → 46.4 ms and 19.6 → 13.4 ms.

---

## 3. O2 — DRAM-sharded decode matmuls

### Family comparison (model-free, real shapes and dtypes, M = 32)

`probes/decode_matmul_families.py` → `logs/decode_matmul_families.log`, plus the DRAM-sharded
geometry sweep in `probes/matmul_sweep.py` → `logs/matmul_sweep.log`. Each row is the best
configuration found for that family.

| role (K x N, dtype) | interleaved, ttnn-chosen | mcast-1D, L1-sharded act | **DRAM-sharded** | best GB/s |
|---|---|---|---|---|
| MLP gate, split (5120x17408, BFP4) | 182.5 us | 177.6 us | **167.8 us** | 299 |
| MLP gate+up, packed (5120x34816, BFP4) | 350.6 us | 361.3 us | 526.6 us | 190 |
| MLP down (17408x5120, BFP8) | 349.9 us | 314.9 us | **198.4 us** | 477 |
| `wqkv` (5120x8192, BFP8) | 123.3 us | 125.2 us | **98.8 us** | 451 |
| `wgate` (5120x6144, BFP8) | 120.1 us | 94.3 us | **75.6 us** | 442 |
| `o_proj` (6144x5120, BFP8) | 128.0 us | 88.0 us | **75.5 us** | 442 |
| GDN `in_proj_qkv` (5120x10240, BFP8) | 150.9 us | 152.5 us | **121.8 us** | 457 |
| GDN `in_proj_z` (5120x6144, BFP8) | 118.9 us | 93.4 us | **71.7 us** | 466 |
| GDN `out_proj` (6144x5120, BFP8) | 126.4 us | 87.4 us | **71.4 us** | 468 |

DRAM-sharded wins every role except the packed gate/up, which is §5's subject.

### Geometry sweep — `in0_block_w` is the whole story

`logs/matmul_sweep.log` sweeps every activation core grid whose core count divides the tiled K
dimension, and every legal `in0_block_w` for each. Extract for `o_proj` (6144x5120, BFP8):

| cores | `in0_block_w` | `per_core_N` | us |
|---|---|---|---|
| 8 | 1 | 20 | 204.4 |
| 8 | 2 | 20 | 119.5 |
| 8 | 4 | 20 | 77.3 |
| 8 | **6** | 20 | **75.5** |
| 8 | 8 | 20 | 76.0 |
| 8 | 12 | 20 | 76.9 |
| 8 | 24 | 20 | L1 overflow |
| 16 | 6 | 10 | 75.9 |
| 32 | 6 | 5 | 76.0 |
| 48 | 4 | 4 | 78.5 |
| 64 | 3 | 3 | 89.3 |

The core count barely matters; `in0_block_w` matters by 2.7x. `in0_block_w = 1` costs 204 us,
`2` costs 120 us, and the curve flattens at 4-8. This is `$optimize` OPT-004's point: treat
`in0_block_w >= 2` as a floor, not a success condition. The same shape holds for every role —
`mlp_down` runs 564 us at `in0_block_w=1`, 323 us at `2` and 198 us at `17`.

The layer's selection rule (`_find_grid`, target 32 cores, then the largest legal
`in0_block_w`) lands on the measured optimum or within 0.3 % of it for **every** role:

| role | chosen cores / `in0_block_w` | chosen us | best measured us |
|---|---|---|---|
| MLP gate, MLP up | 32 / 5 | 168.3 | 167.8 (8 cores / 5) |
| MLP down | 32 / 17 | 199.3 | 198.4 (16 cores / 17) |
| `wqkv` | 32 / 5 | 99.4 | 98.8 (8 cores / 5) |
| `wgate` | 32 / 5 | 76.1 | 75.6 (16 cores / 5) |
| `o_proj` | 32 / 6 | 76.0 | 75.5 (8 cores / 6) |
| GDN `in_proj_qkv` | 32 / 5 | 123.2 | 121.8 (8 cores / 5) |
| GDN `in_proj_z` | 32 / 5 | 72.2 | 71.7 (10 cores / 4) |
| GDN `out_proj` | 32 / 6 | 71.9 | 71.4 (8 cores / 6) |

No dominant decode matmul is left at `in0_block_w <= 2`.

### The one row that stays slow

The BFP4 gate/up projection reaches only **299 GB/s** against 442-477 GB/s for every BFP8 row,
in *every* family and at *every* geometry (168 us at 8, 16 and 32 cores alike). The bytes are
right — 50.1 MB per projection — so this is not a geometry or a policy problem; the BFP4 read
path itself does not saturate DRAM at M = 32 on this part. It is still the fastest option
available (BFP8 gate/up would move 94.7 MB at 477 GB/s = 198 us per projection against BFP4's
168 us), so BFP4 is kept and the gap is recorded as a ttnn-side observation.

### Whole-layer effect

`probes/sweep.py --group topology` → `logs/sweep_topology.log`, `full_attention`, real weights:

| candidate | prefill ms | decode ms |
|---|---|---|
| default (O2 on) | 9.26 | **1.094** |
| `dram_sharded_decode=False` | 17.53 | 1.702 |
| `dram_sharded_decode=False`, prefill program configs on | 10.32 | 1.703 |

O2 is worth **0.61 ms of decode** and, through the prefill program configs it forces (§6),
8.3 ms of prefill.

---

## 4. O3 — width-sharded L1 decode residual

The decode residual stream now stays width-sharded in L1 on the 32-core grid across input
RMSNorm, the mixer boundary, the attention residual add, post-attention RMSNorm, the MLP and
the final residual add. The norm program config is derived from the *residual* shard, so the
norm output is already the projections' activation shard and no layout op sits between them.

| candidate (`full_attention`, real weights) | decode ms |
|---|---|
| default (sharded residual) | **1.094** |
| `sharded_decode_residual=False` (DRAM-interleaved residual) | 1.109 |

Worth 15 us (1.4 %). Small because the fused stage had already width-sharded the two norms
themselves (F15); what O3 removes is the four `to_memory_config` round trips around them.
`linear_attention` behaves the same way (1.167 vs 1.180 ms).

First the accounting, straight from the two decode profiles, so the explanations below can be
checked against a total rather than taken on trust:

<!-- generated:decode-layout -->
| layer kind | layout ops per token | us per token | breakdown |
|---|---|---|---|
| `linear_attention` | 5 | 8.0 | 2x `InterleavedToSharded`, 2x `ShardedToInterleaved`, 1x `Reshard` |
| `full_attention` | 11 | 10.1 | 6x `InterleavedToSharded`, 4x `ShardedToInterleaved`, 1x `Reshard` |
<!-- /generated:decode-layout -->

Three families make up those counts. **The sharded norms' own conversions** — an
interleaved-to-sharded on the way in and one per consumer on the way out, around each of the two
residual-stream RMSNorms — are the largest group in `full_attention` and are the cost `O3`
already minimised rather than removed (§4's first table: removing them entirely by keeping the
residual sharded end to end is what `O3` does, and what is left is the boundary with ops that
will not take a shard). **The `full_attention` head path** adds the q/k norm round trip and the
SDPA input/output conversions. **One `Reshard`** is ttnn overriding the caller's grid (§19).

The individual boundaries, and why each one is paid:

| boundary | op | why |
|---|---|---|
| `wqkv` → `nlp_create_qkv_heads_decode` | one sharded→L1-interleaved conversion | the head-creation op takes an interleaved input, and a DRAM-sharded matmul must write a sharded one |
| `in_proj_qkv` → causal conv | one sharded→L1-interleaved conversion | the gated-delta-net decode path works on interleaved tensors; the tensor is 1.3 MB |
| residual → `in_proj_ba` | one sharded→L1-interleaved conversion | `in_proj_ba` is outside the DRAM-sharded family (its 112-column output is smaller than one shard row) |
| SDPA-decode output | `to_memory_config` to the head shard | the decode SDPA kernel rejects a sharded output for GQA (`sdpa_decode_device_operation.cpp:405`), carried over from the fused stage |
| mixer output → output projection | one `ReshardDeviceOperation`, **1.83 us** (`linear_attention`) / **1.56 us** (`full_attention`) per token | the DRAM-sharded matmul writes its own row-wise core set instead of the rectangle this layer asks for, and the rectangle cannot be given up because the sharded layernorm rejects a non-rectangular grid — see §19 |
| q/k head norm (`full_attention`) | two sharded→interleaved (0.66, 0.68 us) and two interleaved→sharded (0.72, 0.73 us) per token, **2.79 us** total | the per-head RMSNorm on q and k takes a DRAM-interleaved input and the head tensors arrive height-sharded from `nlp_create_qkv_heads_decode`, so each of the two norms pays a round trip. Inherited from the fused stage's head layout; it is 0.26 % of the step and no sharded variant of that norm accepts the height-sharded head shape |

---

## 5. O4/O7 — projection packing

### Gate/up: packed vs split

`$optimize` OPT-010 requires both families measured with everything else fixed.

| | packed (1 matmul + 2 slices) | split (2 matmuls) |
|---|---|---|
| decode matmul, best legal DRAM-sharded | 526.6 us | 2 x 167.8 = **335.6 us** |
| decode matmul, best interleaved | 350.6 us | 2 x 182.5 = 365.0 us |
| decode, best DRAM-sharded at the *selected* 32-core grid | **L1 overflow** | 2 x 168.3 = 336.6 us |
| prefill matmul, best legal 2D config | 5182 us | 2 x 1619.5 = 3239 us |
| whole layer, `full_attention` decode | fails (2,142,464 B of CBs vs 1,572,864 B L1) | **1.094 ms** |
| whole layer, `full_attention` prefill | 11.90 ms | **9.26 ms** |
| whole layer, `linear_attention` prefill | 44.64 ms | **41.92 ms** |

Split wins on both phases and the packed form is not even legal for decode at the geometry the
selection rule picks: doubling N doubles `per_core_N` to 34, and 34 output tiles per core plus
the weight block overflows L1. The fusing stage's packed choice was right for an interleaved
BF16 matmul and is wrong for a DRAM-sharded BFP4 one — the packed row is *slower per byte*
(190 GB/s vs 299 GB/s), and the two output slices it needs are now width slices of a
width-sharded tensor. Split also removes those two slices from prefill (694 us in the fused
`full_attention` profile).

The split path keeps the activation fused where it counts: `ttnn.multiply(up, gate,
input_tensor_b_activations=[SILU])` is one binary op with the SiLU as an input activation, so
the separate family costs one extra matmul launch and nothing else.

### `wgate` into `wqkv`, `in_proj_z` into `in_proj_qkv` (O7)

Both pairs consume the same post-norm activation, so packing them is the OPT-001 pattern.

The **gated-delta-net pair is blocked on a contract, not a preference**: `in_proj_qkv` must emit
float32 for the conv and state path while `in_proj_z` is bfloat16 and must land in `out_proj`'s
activation shard. One matmul has one output dtype, so packing them forces a typecast plus a
width slice of a width-sharded tensor onto whichever half loses.

The **attention pair has no such blocker and was measured**, model-free, PCC-gated, on the real
shapes and dtype (`probes/probe_packed_attn.py`, `logs/probe_packed_attn.log`). An earlier
revision of this section rejected it by analogy with packed gate/up — "packing widens N and hits
the same L1 wall". That argument does not survive this stage's own sweep: the BFP4 gate runs
fine at `per_core_N = 17` while packed gate/up fails at 34, and packed `wqkv|wgate` is only 14.
So it was measured instead:

<!-- generated:o7-matmuls -->
| | separate (`5120x8192` + `5120x6144`) | packed (`5120x14336`) |
|---|---|---|
| decode matmul, DRAM-sharded, 32 cores | 96.4 + 72.6 = **169.0 us** | **164.1 us** |
| prefill matmul, best legal 2D | 767.1 + 541.5 = **1308.6 us** | **1779.1 us** |
| PCC against the separate path | 0.999883 | 0.999883 |
<!-- /generated:o7-matmuls -->

Packed wins the decode matmul by **4.9 us of a 1094 us step (0.4 %)** and loses the prefill
matmul by **470 us of a 9286 us prefill (5.1 %)** — and that is *before* the packed form pays for
its consumers. (Those two deltas are read off the generated table above, so a re-run moves them;
an earlier revision quoted 5.6 us and 466 us from a previous sitting of the same probe.) The two
halves go to different places: `qkv` has to reach `nlp_create_qkv_heads_decode` interleaved and
`gate` has to reach `o_proj`'s activation shard. Separate, each projection writes its consumer's
layout directly and the only conversion is one sharded-to-interleaved on the 8192-wide half.
Packed, that conversion is 14336 wide (1.75x the bytes) and is followed by two width slices.

An earlier revision *estimated* that consumer cost at 3.6 us from the profile's four small
conversions. The fifth stage review was right that an estimate is not a measurement, so the same
probe now times the consumer path directly, model-free, at the decode shape
(`PACKATTN` rows with `"phase": "decode_consumers"`):

<!-- generated:o7-consumers -->
| family | matmul output width | sharded-to-interleaved | width slices | consumer total |
|---|---|---|---|---|
| separate | 8192 | 11.8 us | none | **11.8 us** |
| packed | 14336 | 7.4 us | 39.9 us | **47.3 us** |
<!-- /generated:o7-consumers -->

**Rejected on measured evidence, by a much larger margin than the estimate suggested.** The
packed consumer path costs **47.3 us against 11.8 us** — a 35.5 us penalty, ten times the 3.6 us
the earlier revision guessed, and six times the 5.6 us the packed matmul saves. The reason is
visible in the split: the wider conversion is actually *cheaper* (7.4 us against 11.8 us, because
`sharded_to_interleaved` on more columns still moves the data once), and the cost is the two
width slices at 20.7 and 19.3 us. Slicing a 14336-wide interleaved tensor at decode is
expensive in a way that packing cannot amortise.

So packed loses **~30 us of a 1094 us decode step** as well as 466 us of prefill, 5.0 %. The
estimate happened to reach the right verdict for the wrong magnitude, which is exactly why the
measurement was worth taking. Both projection-packing groups are now measurements rather than
analogies, and no part of the comparison is an estimate.

### BFP4 attention weights on the final topology (OPT-007 follow-up)

The BFP4 attention/GDN trial in §2 ran on the pre-O2 topology, which OPT-007 calls screening
evidence. Re-measured on the final DRAM-sharded, split-gate/up topology:
`logs/sweep_attn_precision.log`. Result and decision unchanged — see §9.

---

## 6. O6 — prefill program configs, and a ttnn correctness bug

O2 stores the dominant weights width-sharded across the DRAM banks. That is not free for
prefill: ttnn's auto-selected prefill program **rejects** a sharded `input_tensor_b` outright
(`matmul_device_operation.cpp:1805`, `Input B memory layout must be INTERLEAVED`). The 2D
multicast program does accept one, but only when `per_core_N` equals the weight's DRAM shard
width in tiles and the shard grid is a single row, which pins the grid's x extent to the 8 DRAM
banks. So an explicit prefill program config is a *requirement* of O2, not an optional tuning
step.

### The bug

The first pinned-`per_core_N` implementation shrank `out_block_w` to fit L1 and produced
**PCC 0.0003** end-to-end. `probes/probe_matmul_correctness.py` isolates it
(`logs/probe_matmul_correctness.log`):

| K x N | dtype | program | PCC vs torch |
|---|---|---|---|
| 5120 x 17408 | BFP4 | interleaved weight, ttnn-chosen | 0.9937 |
| 5120 x 17408 | BFP4 | 2D, **interleaved** weight, `out_block_w=17 < per_core_N=68` | 0.9937 |
| 5120 x 17408 | BFP4 | 2D, **DRAM-sharded** weight, `out_block_w=17 < per_core_N=68` | **0.2402** |
| 5120 x 8192 | BFP8 | 2D, DRAM-sharded weight, `out_block_w=16 < per_core_N=32` | **0.4965** |
| 17408 x 5120 | BFP8 | 2D, DRAM-sharded weight, `out_block_w=20 == per_core_N=20` | 0.99988 |

A 2D `MatmulMultiCoreReuseMultiCast` matmul with a DRAM-width-sharded `input_tensor_b` computes
the wrong answer when `out_block_w < per_core_N`, and nothing rejects it: the
`out_block_w == per_core_N` guard at `matmul_device_operation.cpp:1659` only runs when the
*output* is sharded, and this output is interleaved. **This is a silent-wrong-answer ttnn bug**
and is worth reporting upstream; the reproducer is model-free and 60 lines.

Consequence for the decoder: `out_block_w` is pinned to `per_core_N`, and the search moves to
the grid's y extent, `out_block_h` and `in0_block_w`. `probes/prefill_sweep.py` re-ran the whole
prefill search under that constraint **with a PCC gate on every timed candidate**, so a
fast-but-wrong configuration cannot win a sweep again.

### Prefill results (M = 2048, real shapes and dtypes)

| role | ttnn-chosen, interleaved weight | best legal 2D, DRAM-sharded weight | speed-up |
|---|---|---|---|
| MLP gate / up (5120x17408, BFP4) | 4152.5 us | **1619.5 us** (8x8, `out_block_h`=2, `in0_block_w`=4) | 2.56x |
| MLP down (17408x5120, BFP8) | 2818.7 us | **1221.3 us** (8x10, 7, 8) | 2.31x |
| `wqkv` (5120x8192) | 1329.6 us | **763.7 us** (8x8, 4, 4) | 1.74x |
| `wgate` (5120x6144) | 1058.5 us | **519.6 us** (8x10, 7, 4) | 2.04x |
| `o_proj` (6144x5120) | 1030.6 us | **473.3 us** (8x10, 7, 8) | 2.18x |
| GDN `in_proj_qkv` (5120x10240) | 1789.1 us | **954.7 us** (8x8, 4, 4) | 1.87x |
| GDN `in_proj_z` (5120x6144) | 1064.3 us | **515.3 us** (8x10, 7, 4) | 2.07x |
| GDN `out_proj` (6144x5120) | 1027.9 us | **474.2 us** (8x10, 7, 8) | 2.17x |
| MLP gate+up packed (5120x34816) | 3559.4 us | 5182.2 us | 0.69x |

The candidate list sweeps `in0_block_w` well past 8 — up to 34 — because OPT-004 is explicit
that 8 is not a stopping point and that legal divisors need not be powers of two (K = 5120 is
160 tiles, so 10, 16, 20 and 32 divide it; K = 17408 is 544, so 16, 17 and 34 do). Where those larger values allocate they are **measured slower**, because the L1 budget then
forces `out_block_h` down and `out_block_h` is worth more; where they do not allocate the failure
is the same circular-buffer overflow. For the 5120x17408 gate only `in0_block_w = 10` allocates
at all; 16, 20 and 32 overflow at every grid and `out_block_h`. Either way the cap at 8 is a
measured result:

| role | best overall | best with `in0_block_w > 8` |
|---|---|---|
| MLP gate / up | 1629 us (`in0_block_w` 4, `out_block_h` 2) | 2454 us (10, 1) |
| MLP down | 1218 us (8, 7) | 1320 us (16, 4) |
| `wqkv` | 764 us (4, 4) | 984 us (10, 2) |
| `wgate` / `in_proj_z` | 512 us (4, 7) | 534 us (10, 4) |
| `o_proj` / `out_proj` | 476 us (8, 7) | 504 us (16, 4) |
| GDN `in_proj_qkv` | 957 us (4, 4) | 1215 us (10, 2) |

So the cap at 8 is a measured result, not a default. The selection rule maximises
`out_block_h * in0_block_w` — tiles of work per buffer refill — among the configurations that
fit the per-core circular-buffer budget, breaking ties toward the larger `out_block_h`. It reproduces the measured optimum for the gate, the down projection, QKV
and `in_proj_qkv`, and lands on 541.5 us against 519.6 us (4 % off) for the two 5120x6144
projections. The budget itself is calibrated, not guessed: ttnn reported the exact byte count
for one overflow (1,630,976 B at `per_core_M=8, out_block_h=8, out_block_w=24, in0_block_w=4`
on a BFP8 weight), which fixes the output-block multiplier at 3.283, and 3.3 with a
1,565,000 B ceiling reproduces every allocate/overflow outcome in `logs/prefill_sweep.log`
against the part's 1,572,864 B L1.

---

## 7. O5 — `ttnn.transformer.gated_delta_attn_seq`: tried, rejected on accuracy

`.agents/notes/gdn.md` names this kernel as the largest available prefill win and gates it on
one question: the kernel hard-codes four 32x32 diagonal blocks and therefore only supports
`chunk_size = 128`, while this autoport (and HF) use 64.

**Gate 1 — is chunk 128 mathematically the same?** Yes. `probes/probe_chunk_size.py` (CPU only,
no device) runs HF's own `torch_chunk_gated_delta_rule` at both chunk sizes on the real head
shapes: output and state PCC **1.000000000** at every length from 128 to 5000, with a maximum
absolute difference of 3.7e-8 against an output scale of 5e-2. The chunked gated delta rule is
an exact reformulation, so chunk size changes float association only. `logs/probe_chunk_size.log`.

**Gate 2 — does the kernel hold the accuracy bar?** No.
`probes/probe_gdn_kernel.py` loads the built tree's canonical caller
(`models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py`) by file path
and compares it against HF at the real head shapes (48 heads, 128/128 dims):

| inputs | output PCC | state PCC |
|---|---|---|
| synthetic N(0, 0.5), seq 128 | 0.992810 | 0.993056 |
| synthetic N(0, 0.5), seq 2048 | 0.992869 | 0.992909 |

Both below the 0.995 bar, but `$optimize` OPT-012 is explicit that a synthetic distribution
cannot by itself veto a candidate. So `probes/probe_gdn_kernel_real.py` repeats it at the real
operating point: it replaces `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule` on an instance built
from the **real Qwen3.6-27B layer-0 weights**, captures the exact `q/k/v/g/beta` that layer
feeds the kernel during a 2048-token prefill, and runs the kernel path on those tensors.

| inputs | output PCC | state PCC |
|---|---|---|
| **real weights, real activations**, seq 2048 | **0.989740** | **0.982896** |

Captured activation ranges, for the record: `|q| <= 3.02`, `|k| <= 5.52`, `|v| <= 6.54`,
`g in [-7.23, -1.5e-6]`, `beta in [0.037, 0.973]`. `logs/probe_gdn_kernel_real.log`.

The real operating point is *worse* than the synthetic one, so the rejection is not a
synthetic-distribution artifact. For comparison, the decoder's own chunked implementation
reaches a recurrent-state PCC of **0.99998** against the same reference. A state PCC of 0.9829
is a model-visible accuracy loss: the recurrent state is the layer's entire memory of the
prompt, it is carried across every chunk and into decode, and the stage's acceptance bar is
0.995 on the layer output.

**Gate 3 — is the loss a Python-side knob?** No, and this is the part that makes the
rejection earned rather than assumed.

Two things in the canonical caller are Python-side choices rather than kernel properties, and
both were in the loop for the numbers above: every preprocessing matmul and the whole `L_inv`
solve run at **HiFi2 without `packer_l1_acc`** (`ttnn_delta_rule_seq.py:198-203, 330-334`),
while this decoder's own triangular inverse uses HiFi4 + `fp32_dest_acc_en` precisely because
the functional stage measured cancellation there; and the adapter **typecasts the returned final
state to bfloat16** (`ttnn_delta_rule_seq.py:138-141`). An earlier revision of this section
blamed the first of those and asserted it was not reachable from Python. That was wrong on both
counts, so it was measured.

`probes/probe_gdn_kernel_precision.py` rewrites the loaded source's compute-kernel configs to
HiFi4 + fp32 accumulation and calls the inner `chunk_gated_delta_rule_seq` directly, so the
float32 final state is compared as float32. Same real activations:

| candidate | final-state dtype at readback | output PCC | state PCC |
|---|---|---|---|
| as shipped (HiFi2; adapter would have downcast the state) | FLOAT32 | 0.989740 | 0.982905 |
| **HiFi4 + `fp32_dest_acc_en` everywhere Python controls** | FLOAT32 | 0.989763 | 0.982880 |

**Nothing moves.** Output PCC changes by 2e-5 and state PCC by -2e-5. Two things follow. The
bfloat16 downcast never mattered — the inner function already returns float32, so the first
probe's number was not measured through it. And the precision is not lost in the Python
preprocessing or in the `L_inv` solve: raising every knob Python owns to the highest setting
ttnn offers leaves the result exactly where it was. The loss is inside the C++ sequential scan.

**Gate 4 — how big is the prize, actually?** Earlier revisions of this section and of the
README called this "the largest `linear_attention` prefill opportunity" without ever timing it,
which the fifth stage review flagged: a rejected candidate's prize should be a number. So
`probes/probe_gdn_kernel.py` now times the adapted path warmed, five calls after a warm-up, at
the real shape (`logs/probe_gdn_kernel.log`):

| seq | first call (includes JIT) | **warmed** |
|---|---|---|
| 128 | 77.7 ms | **5.07 ms** |
| 2048 | 196.1 ms | **28.89 ms** |

**28.89 ms for the delta rule alone, against 28.899 ms of warmed wall time for the entire
shipped `linear_attention` layer** — projections, causal conv, delta rule, both norms and the
MLP included (`logs/sweep_final_default.log`; the layer's *device* time is 27.11 ms, and the
warmed wall figure is the like-for-like one because the kernel number is also wall). In the form
that exists and can be called, this kernel path is not a prefill win at all: it takes as long as
everything the layer currently does put together. It is also measured at `chunk_size=128`, which
the kernel requires and the layer does not use, and it carries the adapter's own preprocessing —
both noted so the number is not read as a clean kernel latency.

That does not mean the *kernel* is slow — the adapter carries its own preprocessing matmuls and
the `L_inv` solve, and those are Python-side and untuned. It does mean the opportunity was
asserted, not measured, and the honest statement is narrower: **a fused sequential scan is the
right shape for this problem, and the one available today is neither accurate enough nor, as
adapted, faster.**

**Decision: rejected on measured real-weight accuracy after an adapted retry, and separately
not a latency win as it stands.** The adapted path is *equally inaccurate* whatever Python does,
which is the stronger form of the result: the fix has to happen in the kernel and no Python-side
contract change reaches it. Recorded as a ttnn improvement candidate with four model-free
reproducers — `probe_chunk_size.py`, `probe_gdn_kernel.py`, `probe_gdn_kernel_real.py` and
`probe_gdn_kernel_precision.py` (`logs/probe_gdn_kernel_precision.log`).

The prefill win it would have bought is not lost entirely: §6's program configs take
`linear_attention` prefill from 51.7 ms to 41.9 ms without touching the delta-rule math, and
`O8`, `O10`, `O12` and `O13` take it from there to 28.9 ms.

---

## 8. O8 — the gated-delta-net causal conv in bfloat16

The `linear_attention` prefill profile after O1/O2/O6 still spends 23 % of its device time in
`BinaryNg` and 16 % in `Ternary`, on three ops. Those are the causal conv: `mixed_qkv` is
`[1, 1, 2051, 10240]`, which is **84 MB in float32**, and the conv is one multiply plus three
`addcmul` passes plus a `concat` over tensors of that size. The delta-rule state math genuinely
needs float32 — the functional stage measured that — but the conv is a four-tap FIR followed by
a SiLU, and its output is L2-normalised immediately afterwards.

`PrecisionPolicy.gdn_conv` names that stage separately. The conv output is cast back to float32
at the boundary inside `_causal_conv`, so nothing downstream changes.

| candidate (real weights) | linear prefill | linear decode | linear prefill PCC | linear decode PCC |
|---|---|---|---|---|
| `gdn_conv = float32` | 41.93 ms | 1.167 ms | 0.999460 | 0.999908 |
| **`gdn_conv = bfloat16`** | **32.16 ms** | **1.153 ms** | 0.999432 | 0.999899 |

**9.8 ms of prefill, 23 %, for 2.8e-5 of PCC.** Kept. `full_attention` is unaffected (9.26 ms
both ways) because it has no gated-delta-net.

---

## 9. Attention and gated-delta-net precision, field by field, on the final code

This section was re-run twice. The first version measured the pre-`O2` topology, which OPT-007
calls screening evidence only. The second measured the final topology but set `attn_*` and
`gdn_*` fields **together**, which conflates two layer kinds: a `gdn_*` field is dead in
`full_attention` and an `attn_*` field is dead in `linear_attention`, so a rejection could not be
attributed to one of them. The table below moves one field at a time on the shipped
post-`O8`/`O9`/`O10` code, real weights (`probes/sweep.py --group precision_v2 --real`,
`logs/sweep_precision_v2.log`). Each row is read on the layer kind whose weights it touches.

The `default` row here is the *pre-`O14`* default, all-BFP8 projections; `attn_gate` BFP4 is a
candidate in this table and became the default later (§23), so read this table as the state of
the search at the time, not as the shipped policy.

| candidate | kind | decode ms | delta | prefill PCC | decode PCC |
|---|---|---|---|---|---|
| **default** (all BFP8 projections) | full | 1.0940 | — | 0.999270 | 0.999651 |
| `attn_qkv` BFP4 | full | **1.0822** | -11.8 us | 0.996002 | 0.998927 |
| `attn_out` BFP4 | full | **1.0869** | -7.1 us | 0.998535 | 0.999397 |
| `attn_gate` BFP4 | full | **1.0873** | -6.7 us | 0.998836 | 0.999508 |
| `proj_fp32_acc=False` | full | **1.0886** | -5.4 us | 0.999120 | 0.999550 |
| **default** | linear | 1.0985 | — | 0.999436 | 0.999897 |
| `gdn_qkv` BFP4 | linear | 1.0812 | -17.3 us | **0.992516** | 0.998496 |
| `gdn_out` BFP4 | linear | **1.0912** | -7.3 us | 0.998507 | 0.999269 |
| `gdn_z` BFP4 | linear | 1.0920 | -6.5 us | **0.995821** | 0.999521 |
| `proj_fp32_acc=False` | linear | **1.0930** | -5.5 us | 0.999226 | 0.999955 |

Two clean rejections come straight out of this:

* **`gdn_qkv` BFP4 fails the bar** on real weights (prefill PCC 0.992516). It feeds the delta
  rule's state recurrence, whose conditioning the functional stage documented — the same reason
  the state math keeps HiFi4. Note this is *not* an argument about `attn_qkv`: splitting the
  fields shows `full_attention`'s QKV passes at 0.996002 where the gated-delta-net's does not.
* **`gdn_z` BFP4 at 0.995821** is 0.0008 above the bar, and `attn_gate`+`gdn_z`+both outputs
  together lands `linear_attention` at **0.995004** — four parts per million above 0.995.
  A value that close is not a pass, it is noise around the bar.

That last sentence started as an assertion and is now a measurement. The draw-sensitivity table
below puts a number on "noise around the bar": holding the code, the weights and the prompt
fixed and changing only which decode token is drawn moves real-weight decode PCC by up to
**0.0048** on `linear_attention` and **0.0013** on `full_attention`. So the working rule for the
rest of this stage, applied to `O14` as well as to the rejections here, is: **a candidate whose
worst measured real-weight PCC is within ~0.001 of the bar has not been shown to hold it, and
needs the draw sweep before it can be adopted.**

The rest — `attn_qkv`, `attn_out`, `gdn_out`, `proj_fp32_acc=False` — all looked **correct on
real weights and faster on both layer kinds** at this point, and OPT-007 is explicit that margin
above a passing bar is not on the list of permitted rejections. So they were stacked and measured
properly rather than waved away (`logs/sweep_v3_real.log`, `logs/sweep_v3_synth_*.log`,
`logs/sweep_v4_synth.log`):

| stack | kind | decode ms | real prefill PCC | **synthetic** prefill PCC | **synthetic** decode PCC |
|---|---|---|---|---|---|
| default | full | 1.0947 | 0.999270 | 0.986537 | 0.984493 |
| `proj_fp32_acc=False` | full | 1.0882 | 0.999120 | 0.984732 | **0.980829** |
| outputs BFP4 | full | 1.0866 | 0.998535 | **0.974126** | **0.971992** |
| outputs BFP4 + no fp32 acc | full | **1.0802** | 0.998382 | **0.972372** | **0.968534** |
| + `attn_qkv` BFP4 | full | **1.0674** | 0.995052 | **0.916320** | **0.900899** |
| default | linear | 1.0983 | 0.999436 | 0.996553 | 0.996778 |
| outputs BFP4 + no fp32 acc | linear | **1.0849** | 0.998291 | 0.988272 | 0.989150 |

**Decision: not adopted — and the deciding evidence is real-weight, not synthetic.**

The stacked candidate looked good at 2048 tokens on real weights (0.998382 / 0.998291) and an
earlier revision of this section rejected it only because it would have needed the synthetic
suite bar lowered. That was the wrong reason, so the stack was adopted and the suite was run.
`test_real_weight_pcc_at_disputed_lengths` — which exists precisely because 2048 tokens is not
the whole contract — failed it:

```
real-weight prefill PCC 0.9936081444689238 < 0.995 at seq_len 17
real-weight decode  PCC 0.9907796831867199 < 0.995 at seq_len 743
```

Re-measured field by field at the short lengths on real weights
(`logs/sweep_v5_real_short_17.log`, `logs/sweep_v5_real_short_743.log`):

| candidate | kind | seq 17 prefill | seq 17 decode | seq 743 prefill | seq 743 decode |
|---|---|---|---|---|---|
| default | full | 0.998295 | 0.997929 | 0.998961 | 0.999620 |
| **outputs BFP4** | full | **0.993815** | **0.994999** | 0.997772 | 0.999311 |
| `proj_fp32_acc=False` | full | 0.998117 | 0.997682 | 0.998775 | 0.999511 |
| outputs BFP4 + no fp32 acc | full | **0.993608** | **0.994716** | 0.997561 | 0.999151 |
| default | linear | 0.999213 | 0.999651 | 0.999367 | 0.999212 |
| **outputs BFP4** | linear | 0.998054 | 0.999007 | 0.998430 | 0.996838 |
| `proj_fp32_acc=False` | linear | 0.998731 | 0.999618 | 0.999084 | 0.998899 |

So the two halves separate cleanly, and only one of them is a synthetic story:

* **`attn_out` BFP4 is rejected on real target weights.** `full_attention` at seq 17 gives
  0.993815 prefill and 0.994999 decode, both below the 0.995 bar. That is model-visible
  correctness loss on the model's own weights at a length the contract requires — an
  OPT-012-permitted rejection, and one that a 2048-token measurement could not see. It is also
  the answer to why the disputed-length real-weight test exists.
* **`gdn_out` BFP4 is rejected too, and needed its own evidence.** The sweep measured the two
  output projections together as `out_bfp4`, and the fifth stage review pointed out that the
  numbers above are `full_attention` rows, where **`gdn_out` does not exist** — §9 itself says
  (above) that a `gdn_*` field is dead in `full_attention`. Read strictly, this section was
  rejecting `gdn_out` on a measurement of a different tensor in a different layer kind, while
  its own `linear_attention` rows showed it passing at 0.998054 / 0.999007 / 0.998430 / 0.996838
  and saving 7.3 us of decode.

  So it was measured on its own, with the draw sweep the rule above requires
  (`probes/probe_draw_sensitivity.py --kinds linear --candidates default,gdn_out_bfp4`,
  `logs/probe_draws_gdnout_seq17.log`, `logs/probe_draws_gdnout_seq743.log`,
  `logs/probe_draws_gdnout_seq5000.log`, real weights, `linear_attention` decode):

  | seq | config | 777 | suite's draw | 11 | 12345 | 2024 | 4242 | worst |
  |---|---|---|---|---|---|---|---|---|
  | 743 | shipped BFP8 | 0.999211 | 0.997799 | 0.999365 | 0.999748 | 0.996689 | 0.999864 | **0.996689** |
  | 743 | **`gdn_out` BFP4** | 0.996842 | **0.994368** | **0.994194** | 0.999234 | **0.994902** | 0.999233 | **0.994194** |
  | 17 | `gdn_out` BFP4 | 0.999006 | 0.999366 | 0.999491 | 0.999396 | 0.998575 | 0.999338 | 0.998575 |
  | 5000 | `gdn_out` BFP4 | 0.997974 | 0.998325 | 0.996595 | 0.998715 | 0.997778 | 0.999351 | 0.996595 |

  **`gdn_out` BFP4 is below the bar on three of six draws at seq 743**, worst 0.994194. The
  rejection stands; the reason is now `gdn_out`'s own numbers. Note also that the row this
  section originally leaned on — 0.996838 at seq 743 — is seed 777, reproduced here to the digit
  as 0.996842, and it is the third-best of the six draws.
* **`proj_fp32_acc=False` fails the real-weight bar too** — and an earlier revision of this
  section got that wrong, so the correction is spelled out below rather than edited away.

`attn_qkv` BFP4 is rejected on the same real-weight grounds as the outputs, one step earlier:
0.996002 at 2048 and 0.995052 in the stack are already inside noise of the bar before the short
lengths are considered. It was re-tested alone at the short lengths for §23 and fails there
outright: **0.987711** prefill at seq 17 and **0.993865** at seq 743.

### `proj_fp32_acc=False`, and why the sweep and the suite disagreed by 0.0045

This field was first written up here as *"holds the real-weight bar everywhere (worst 0.997682),
rejected because two synthetic structural cases land at 0.9789 against the 0.98 bar"*. The
synthetic half of that is true — `logs/suite_o13_trial.log` has
`test_decode_pcc[5000-full_attention]` at 0.9788857 and `test_batched_users[32-full_attention]`
at 0.9793360 — but it was not the deciding evidence, and the same log says so four failures
further down:

```
FAILED test_real_weight_pcc_at_disputed_lengths[linear_attention]
  AssertionError: real-weight decode PCC 0.9943651152143212 < 0.995 at seq_len 743
```

That is a **real-weight** bar failure, at a disputed length, in the shipped test. It also flatly
contradicts `logs/sweep_v5_real_short_743.log`, which reads **0.998899** for the same field, the
same layer kind, the same length and the same weights. Two harnesses, one config, 0.0045 apart:
one of them had to be wrong about something, and "the sweep says it passes" is exactly the claim
the rest of this section rests on.

The difference is the **decode token draw**. The sweep draws it with `seed=777`; the suite draws
it with `seed=900 + seq_len`, which is 1643 at seq 743. `probes/probe_draw_sensitivity.py` holds
everything else fixed — same weights, same prompt, same post-prefill state, same golden path —
and sweeps only that seed (`logs/probe_draws_fp32acc_seq743.log`, real weights, seq 743):

| kind | config | 777 | 1643 | 11 | 12345 | 2024 | 4242 | worst | spread |
|---|---|---|---|---|---|---|---|---|---|
| `linear_attention` | default | 0.999211 | 0.997799 | 0.999365 | 0.999748 | 0.996689 | 0.999864 | **0.996689** | 0.0032 |
| `linear_attention` | `proj_fp32_acc=False` | 0.998899 | **0.995077** | 0.998845 | 0.999735 | 0.997574 | 0.999843 | **0.995077** | 0.0048 |
| `full_attention` | default | 0.999620 | 0.999116 | 0.999187 | 0.999257 | 0.998560 | 0.999534 | 0.998560 | 0.0011 |
| `full_attention` | `proj_fp32_acc=False` | 0.999511 | 0.998930 | 0.999049 | 0.999112 | 0.998356 | 0.999334 | 0.998356 | 0.0012 |

Both harnesses were reporting honestly. The sweep's 0.998899 at seed 777 reproduces to the digit;
so does the suite's failure, at seed 1643, where this run reads 0.995077 against the trial run's
0.994365 (`O13` perturbs `linear_attention` in the sixth digit, and a value sitting on the bar
moves visibly). **Neither number characterises the config — the spread does.**

So the rejection is restated on the evidence that actually decides it:

* **turning `fp32_dest_acc_en` off takes the worst-case-over-draws real-weight decode PCC of
  `linear_attention` from 0.996689 to 0.995077** — from 0.0017 of margin to 0.00008, i.e. onto
  the bar. It does not "hold the bar everywhere"; it holds it at five draws out of six and lands
  on it at the sixth, and the shipped suite happens to draw the sixth.
* the synthetic structural failures (0.9789, 0.9793) are then a *second*, independent reason,
  not the first one.
* it is worth 5.4-5.5 us of decode, 0.5 %. Against a real-weight correctness bar it does not
  robustly clear, that is not a trade this stage takes.

Two things follow for the rest of this document. First, every "holds the bar" claim from a sweep
row is a claim about **one draw**, and the per-field sweeps in the table above should be read
with the ±0.003 that this table measures; the binding gate is
`test_real_weight_pcc_at_disputed_lengths`, which is why adopted candidates are re-run through
the suite and not signed off from the sweep (§23 does exactly this for `attn_gate`). Second, the
shipped default's own worst draw is **0.996689**, not the 0.999 the headline table shows — that
is the honest margin over the 0.995 bar, and it is recorded in `pcc_evidence.json`.

One more data point from the same experiment, worth keeping because it is the hardest place to
measure: with the BFP4 output projections in, the **full advertised context** degrades further
than any short length suggested — the `full_attention` 262143-token prefill tail falls from
0.985447 to **0.957102** and the `linear_attention` tail from 0.996595 to 0.988325
(the `logs/long_context.log` written by that trial run). Whatever the short-length real-weight
failure is, it compounds over 262k tokens, which is the opposite of what a 2048-token
measurement would have predicted.

Two further fields, unchanged for reasons that are not about margin:

* **`in_proj_ba` stays float32.** bfloat16 is *free* accuracy-wise and also free
  performance-wise (1.166 vs 1.167 ms, inside noise), so there is nothing to buy; it is 2.6 MB
  and it feeds the two gates of the state recurrence.
* **KV cache stays BFP8.** BF16 is the same speed (1.094 both ways) at the same PCC and twice
  the memory.
* **Weight matmuls stay LoFi.** HiFi2 costs 0.52 ms of decode and 4.4 ms of prefill.

## 10. Decode SDPA: a 4-9x row that cannot be taken (OPT-002)

After O1/O2 the paged decode SDPA is the second-largest row of a `full_attention` decode step
(226 us of 1082 us, 21 %), and it runs on the op's **default** configuration. `$optimize`
OPT-002 makes an explicit `SDPAProgramConfig` candidate mandatory when that is the case.

`probes/sdpa_decode_sweep.py` (`logs/sdpa_decode_sweep.log`) found explicit configurations
**4x faster at position 2048 and 9x faster at 8192** — and returning garbage at some positions.
`probes/sdpa_decode_positions.py` (`logs/sdpa_decode_positions.log`) then swept 18 positions
across the whole advertised context against a float32 torch attention over the *quantised*
cache (the op's own default cannot be the golden here, because it is itself wrong at long
positions):

| position | default | 24-core grid | 32-core grid | 64-core grid |
|---|---|---|---|---|
| 63 | 0.99997 / 45 us | 0.99998 / 31 us | 0.99998 / 34 us | 0.99998 / 70 us |
| 255 | 0.99998 / 48 us | **0.71039** | **0.71039** | **0.71039** |
| 511 | 0.99997 / 78 us | **0.51400** | **0.51400** | **0.51400** |
| 1023 | 0.99996 / 134 us | **0.34803** | **0.35205** | **0.35205** |
| 2048 | 0.99990 / 244 us | **0.43240** | 0.99998 / 62 us | **0.21718** |
| 5003 | 0.99970 / 556 us | **0.45694** | **0.16400** | **0.18775** |
| 8191 | 0.99942 / 892 us | **0.38561** | 0.99998 / 101 us | 0.99998 / 105 us |
| 32767 | 0.99354 / 3476 us | 1.00000 / 337 us | 0.99996 / 283 us | 0.99997 / 247 us |
| 131071 | **0.92734** / 13815 us | 1.00000 / 1233 us | 0.99979 / 998 us | 0.99991 / 798 us |
| 262143 | **0.78999** / 27596 us | 1.00000 / 2429 us | 0.99941 / 1951 us | 0.99979 / 1531 us |

Two conclusions, both worth keeping.

1. **No explicit config is correct at every position, so none can be adopted.** The kernel's
   cross-core merge of partial softmax results is only valid when `num_k_chunks == 1` or
   `num_k_chunks % (2 * cores_per_head) == 0`; `num_k_chunks` depends on the runtime position
   while the program config is compile-time, and under trace capture it is baked once for every
   step. The one-core-per-head hypothesis (a 24-core grid, so no k split and therefore no merge)
   was tested explicitly and is **false**: that grid is exact at 12287 and above and wrong from
   255 to 8191. This is the same defect `doc/context_contract.json` already records, now
   measured across the context instead of at one point.
2. **The default itself degrades with position**, smoothly: 0.99997 at 63, 0.99354 at 32767,
   0.92734 at 131071, 0.78999 at 262143. That is the upstream `Float16_b` softmax-denominator
   defect, and it is why the context contract caps HF-verified `full_attention` decode at
   position 5003. The optimized decoder inherits the behaviour unchanged — it neither fixes nor
   worsens it — and the 4-9x that a working program config would buy is recorded here as the
   size of the prize for an upstream fix.

Kept: the default configuration. Rejected: every explicit `SDPAProgramConfig`, with an
18-position correctness table rather than a first API error.

---

## 11. `tt-perf-report` advice, item by item

Collected with advice enabled (`probes/run_perf.sh`, `tracy/<kind>/<phase>_perf_report.{txt,csv}`).
Two things about how the tool emits advice matter for reading this section, both from
`tt_perf_report/perf_report.py`: advice is only generated for rows the tool classifies as
`DRAM`, `COMPUTE` or `SLOW`-bound (`perf_report.py:1380-1446`), and a row that stops being
`SLOW` stops carrying advice. So the shipped profile's advice set is *smaller* than the one
that drove this stage's changes — the rows that advice sent us to were fixed, and the advice
went with them. Both sets are recorded below; conflating them would read as if the tool never
said anything about the delta-rule loop.

### The advice in the shipped profile

Every advice string in the four final CSVs, with nothing omitted:

| advice, verbatim | rows it fires on | device time | action |
|---|---|---|---|
| "No output subblock size found" | `32 x 5120 x 17408` decode (the BFP4 gate/up pair), both kinds | 2603.4 us / 2605.9 us over 16 dispatches = 325.4 us per token | **not actionable**: this row runs under `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, which has no output-subblock field at all (`in0_block_w`, `per_core_M`, `per_core_N`, `fused_activation`). Reported as a `tt-perf-report` improvement candidate — the advice should not ask a DRAM-sharded program for a field its config class does not have. |
| "Use HiFi2 or HiFi4 with BF16 activations for improved accuracy" | the same two rows | as above | **rejected with measurement**: those rows are LoFi by policy (§2/§3). HiFi2 across the projections costs 0.53 ms of decode and 4.4 ms of prefill (§9) for real-weight PCC that is already 0.9977 worst-case. |
| "Output subblock 1x1 is small, try out_subblock_h * out_subblock_w >= 2 if possible" | `32 x 5120 x 128` decode, the `b\|a` projection, `linear_attention` | 221.3 us over 8 dispatches = 27.7 us per token | **not taken**: the whole output is four tiles wide and one tile high, so there is no larger subblock to have. |
| "HiFi2 may also work, it discards the lowest bit of the activations and has 2x the throughput of HiFi4" | `32 x 5120 x 128` decode and `2048 x 5120 x 128` prefill, both the `b\|a` projection | 27.7 us per token, 152.4 us of prefill | **not taken**: `b\|a` feeds the `sigmoid`/`softplus` gates of the state recurrence and is the one weight this stage deliberately keeps at float32/HiFi4 (§9, and the functional stage's cancellation finding). It is 2.5 % of a decode step. |
| "If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)" | `2048 x 5120 x 128` prefill, the `b\|a` projection | 152.4 us | **not taken**: input 0 is the 2048x5120 bfloat16 post-norm activation, 20 MB. It is shared by the `qkv`, `z` and `b\|a` projections in the same block, and §6's configs already bind on L1 for the two large consumers; pinning a 20 MB interleaved-L1 residency to save part of 152 us would take the block's headroom away from them. |
| "in0_block_w=2 and output subblock 1x4 look good 🤷" | the same row | — | already satisfied. |
| (`full_attention` prefill carries **no** advice on any row) | — | — | — |

### Advice from the earlier profiles, and what it turned into

These rows no longer appear because the change the advice pointed at was made; the batched
delta-rule matmuls are the whole story here. They were `SLOW`-bound with no program config and
a DRAM-interleaved input 0; after `O10` gave them a program config and `O12` moved the
recurrent state to L1 they are neither, and their `Bound` column is now empty.

| advice, verbatim | where | action |
|---|---|---|
| "No program_config specified, try using one to override in0_block_w and out_subblock_h/w" (`perf_report.py:1417-1419`) | the batched delta-rule matmuls, `linear_attention` prefill, 2.6 ms | **taken — `O10`, §17.** An earlier revision rejected this on the grounds that a program config would fix a core grid across a loop whose shapes change with the ragged final chunk. That is the same objection `_prefill_linear` already solves by caching per shape, so it was measured instead: 17.55 -> 9.30 us and 14.55 -> 9.21 us on the two largest groups, and 32.12 -> 31.18 ms of whole-layer prefill. |
| "in0_block_w=1 is small, try in0_block_w=2 or above" (`perf_report.py:1425`) | `b={48} x 64 x 128 x 128`, the largest batched delta-rule group | **rejected with measurement — §17.** 2 and 4 were swept: 10.1 us and 10.4 us against 9.8 us at 1. The operands are L1-resident, which is why the DRAM-sharded rows' preference for a large `in0_block_w` does not carry over. |
| "If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)" (`perf_report.py:1411`) | the same group | **taken — `O12`, §19.** This one was right and was misread for a while: the DRAM operand was the recurrent state, and moving it to L1 took 1.83 ms off the prefill. The group now runs 998.9 us over 128 dispatches, **7.8 us each**, below the 9.8 us the isolated sweep measured. |
| "HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4" (`perf_report.py:681-683`) | `b={48} x 64 x 128 x 128`, `HiFi4 FP32 x FP32 => FP32` | **not taken, and the advice is wrong about the row.** There is no BFP8 operand anywhere in the delta-rule state path. The cause is in the tool: the BFP8 branch is selected by `in1_bits >= 7 and out_bits >= 7`, which float32 satisfies, so an FP32 x FP32 => FP32 row is told it is a BFP8 multiplication. Recorded as a `tt-perf-report` defect with the source line rather than only the string, because the string itself is no longer reproducible from the shipped profile. |
| (no advice; found by auditing the profile) | the causal conv's 1.61 ms of tilize/untilize over 11 ops, after `O13` removed a 12th and 13th that were not the conv at all | **rejected with an exact blocker — §18, §22.** |

---

## 12. Performance accounting

Per-token decode, one layer, from the same run (`tracy/<kind>/decode_perf_report.csv` for
device time, `probes/sweep.py` for end-to-end, both at position 2048, batch 1).

Roofline: bytes the measured path must move per token / aggregate DRAM bandwidth. Weight bytes
at their stored dtypes plus the KV-cache read.

**`linear_attention`** — `mlp_gate` + `mlp_up` 2 x 50,135,040 B (BFP4), `mlp_down` 94,699,520 B
(BFP8), `in_proj_qkv` 55,705,600 B, `in_proj_z` 33,423,360 B, `out_proj` 33,423,360 B (BFP8),
`in_proj_ba` 2,293,760 B (FP32), the gated-delta-net recurrent state read and written once
(2 x 3,145,728 B at batch 1), conv taps 61,440 B and norms 40,960 B
⇒ **326,209,536 B = 326.21 MB**.

**`full_attention`** — `mlp_gate` + `mlp_up` 100.3 MB, `mlp_down` 94.7 MB, `wqkv` 44.6 MB,
`wgate` 33,423,360 B, `o_proj` 33,423,360 B, norms 40,960 B ⇒ 306,421,760 B, plus the paged KV
read at position 2048 (2 x 4 heads x 2049 x 256 x 1.0625 B = 4,458,624 B)
⇒ **310,880,384 B = 310.88 MB**.

Both totals are computed from the shipped `PrecisionPolicy` by
`probes/derive_report_numbers.py`, not typed in, so a dtype change moves them.

At the 512 GB/s aggregate DRAM bandwidth of one Blackhole chip, and against the device time
from `tracy/<kind>/decode_perf_report.csv` and the end-to-end time from `probes/sweep.py`:

<!-- generated:accounting -->
| | roofline | device time | end-to-end | device to e2e gap | roofline / device |
|---|---|---|---|---|---|
| `linear_attention` decode | 0.637 ms | 1.074 ms | 1.098 ms | 0.024 ms | 59.3 % |
| `full_attention` decode | 0.607 ms | 1.082 ms | 1.094 ms | 0.012 ms | 56.1 % |
<!-- /generated:accounting -->

`tt-perf-report`'s own modeled figure agrees: 53.9 % (276 GB/s) for `linear_attention` and
51.1 % (262 GB/s) for `full_attention` (`tracy/*/decode_perf_report.console.log`).

<!-- generated:accounting-narrative -->
**end-to-end = device time + dispatch gap + host work.** The measured op-to-op gap inside
the signposted window is 43.3 us per token (`linear_attention`) and 39.6 us (`full_attention`), which is *larger* than the 24 us / 12 us end-to-end excess -
i.e. the replay pipeline overlaps some of it, and there is **no host term left in the traced
decode loop**: the window contains `execute_trace` calls and nothing else, and the harness
uploads every input before the start signpost.  The distance between roofline and device time
is concentrated in two named places, both measured rather than assumed:

* the **BFP4 gate/up rows**: 325.6 us of the 1074 us step for 100.3 MB, i.e. ~308 GB/s where every BFP8 row reaches
  442-477 GB/s.  At BFP8 those two rows would move 189 MB and take 397 us, so BFP4 is still the
  right choice; the efficiency gap is a ttnn-side property of the BFP4 read path at M = 32 (§3);
* the **gated-delta-net decode's small state ops**: 44.6 us of `ReshapeView` (the per-head
  layout change, a last-dim reshape and therefore an untilize/retilize), 67.7 us of `BinaryNg`,
  35.9 us of `LayerNorm`, 22.4 us of `Copy` and 17.5 us of `Ternary`
  - about 188 us of the 1074 us `linear_attention` step spent on
  tensors small enough that fixed per-op cost dominates, after `O9` removed 55 us of it.
  `full_attention` has the equivalent in its 226.2 us `SdpaDecode` row (§10).

Prefill reconciles the same way: 27.11 ms of device time against 29.87 ms measured inside the
signpost for `linear_attention` (1.13 ms of op-to-op gap over 873 ops), and 8.91 ms against
9.37 ms for `full_attention` (11 us of gap over 23 ops).  Prefill is compute-bound,
not DRAM-bound, so its 5.4 % / 15.5 % DRAM figure is expected rather than a finding; the FLOP
column of `tracy/full_attention/prefill_perf_report.txt` is the relevant one there.
<!-- /generated:accounting-narrative -->

Machine-readable form: `perf_summary.json`.


---

## 13. The BFP4 MLP against synthetic weights — the one place the bars differ

The test suite runs on the **synthetic** weights of `weight_stats.json`: per-tensor Gaussians
with the real mean and standard deviation. Every earlier stage passed the 0.995 bar on them
comfortably. The optimized decoder does not, and the reason is worth stating precisely because
it is the only place where this stage's delivered suite differs from the previous one's.

Measured both ways, same code, same lengths (`logs/bisect_synth_seq64.log`,
`logs/bisect_synth_seq2049.log` for synthetic; `logs/bisect_seq64_precision.log` and
`logs/sweep_final_default.log` for real):

| weights | length | kind | fused stage | **BFP4 gate/up** | BFP8 gate/up |
|---|---|---|---|---|---|
| synthetic | 64 | linear | 0.999942 | **0.994762** | 0.998990 |
| synthetic | 64 | full | 0.999722 | **0.993334** | 0.998745 |
| synthetic | 2049 | linear | 0.999944 | 0.996554 | 0.999069 |
| synthetic | 2049 | full | 0.999443 | **0.986536** | 0.997650 |
| **real** | 64 | linear | 0.999961 | **0.999007** | — |
| **real** | 64 | full | 0.999869 | **0.997758** | — |
| **real** | 2048 | linear | 0.999967 | **0.999432** | — |
| **real** | 2048 | full | 0.999969 | **0.999270** | — |

The gap is an order of magnitude: BFP4 costs 5e-4 of PCC on the real checkpoint and 1.3e-2 on a
Gaussian of the same variance. That direction is the expected one. A block-float format with
one shared exponent per 16 values loses least when the 16 values are correlated and the output
has dominant directions — which is what a trained weight matrix produces and a Gaussian does
not. The synthetic case is a stress probe, and it is behaving like one.

`$optimize` OPT-012 and this stage's goal both say the same thing about that situation: a
synthetic PCC cannot veto a real-weight win, and the resolution is to add a same-contract
real-weight check under the disputed condition rather than to select the slower policy.

What was actually done, so a reader can judge it:

* the **default policy keeps BFP4 gate/up** — worth 145 us of a 1.15 ms decode step (12.6 %) and
  2.5 ms of prefill;
* the inherited suite keeps **all** of its structural coverage — every unfriendly length, the
  pad-aliasing regression range, the paged cache, determinism, tracing, batch 4 and 32 — at
  `SYNTHETIC_PCC_BAR = 0.98`, which is wide enough for the synthetic distribution and still
  catches any structural break (a wrong page table, a corrupted chunk, a mis-shaped mask all
  land far below 0.98, as the failures earlier in this stage did at 0.24 and 0.50);
* `test_real_weight_pcc_at_disputed_lengths` re-runs **exactly the lengths that relaxation
  covers** — 1, 17, 64, 743, 2049, 5000 — on the **real checkpoint**, prefill and a following
  decode step, at the **unmodified 0.995 bar**, for both layer kinds.

That real-weight test then did its job three times over, which is the honest way to read this
section: it **rejected** BFP4 output projections (§9), it **rejected** `proj_fp32_acc=False`
(§9), and it very nearly **admitted** a BFP4 output gate that does not hold the bar (§23) — that
last one needed the draw sweep on top of the test to catch. The relaxed synthetic bar bought
BFP4 gate/up's 12.6 %; it did not become a licence for every later BFP4 candidate, and §20 now
puts a precision-independent 0.999 gate back under the prefill path as well.

The alternative was measured rather than assumed. BFP8 gate/up passes the synthetic bar
everywhere (worst 0.997650) and costs, on synthetic weights at 2049: `linear_attention` prefill
36.37 → 38.30 ms and `full_attention` prefill 13.98 → 16.06 ms. Its decode could not be
measured at all without a further change: the BFP8 gate/up decode matmul at the selected
32-core geometry overflows L1, because a BFP8 tile is 1.9x a BFP4 one and `per_core_N` is 17.
That is recorded as the cost of the fallback, not as a reason to avoid it.

### The control at the full advertised context

The bisect above is at 64 and 2049 tokens. The place the relaxed bar is doing the most work is
the *other* end: at 262143 the synthetic `full_attention` prefill tail is 0.985447, which clears
0.98 by 0.0054 and would have failed the 0.995 bar the earlier stages used. Extrapolating the
BFP4 attribution to that length would be an inference, so it was controlled directly:
`OPT_DECODER_PRECISION=bfp8_gate_up` re-runs the same `test_full_advertised_context` with the
BFP8 fallback and nothing else changed (`logs/long_context_bfp8_control.log`).

<!-- generated:bfp4-control -->
| metric at 262143, synthetic weights | BFP4 gate/up (default) | **BFP8 gate/up control** |
|---|---|---|
| `full_attention` prefill tail | **0.985447** | **0.997937** |
| `linear_attention` prefill tail | **0.996595** | **0.999062** |
| `linear_attention` recurrent state | 0.999666 | 0.999666 |
| `linear_attention` conv state | 0.999880 | 0.999880 |
| `full_attention` paged K cache | 0.999849 | 0.999849 |
| `full_attention` paged V cache | 0.999856 | 0.999856 |
<!-- /generated:bfp4-control -->

The attribution holds at the full context: swapping only the MLP gate/up dtype moves both
prefill tails back above 0.995, and every quantity the MLP does not touch — the recurrent state,
the conv state, both KV caches — is **identical to the last digit** between the two runs. There
is no length-dependent second effect hiding under the relaxed bar.

(Both runs report two failures. The default's are the inherited `full_attention` decode-SDPA
defect plus its own; the control's are both the BFP8 gate/up decode matmul overflowing L1 at the
selected geometry, which is the same thing this section records as the cost of the fallback.
Neither affects the prefill-tail numbers above, which are recorded before the decode step.)



---

## 14. O9 — the gated-delta-net decode head split

After everything above, the `linear_attention` decode profile still spends **103.5 us of a
1132 us step (9.1 %) in ten `ReshapeViewDeviceOperation`s** — the only material layout cost left
in either decode path (`full_attention` spends 10.1 us, 0.9 %, on layout in total).

They come from `_linear_attention_decode`'s head split. `to_heads` turned
`[1, 1, batch, heads * head_dim]` into `[1, batch * nv, 1, head_dim]` in two steps, via
`[1, batch, heads, head_dim]`. Both are row-major views of the same flat buffer, so the
intermediate step is redundant — and it is not free, because each is a **last-dimension** change
and TTNN implements those as untilize + retilize.

Collapsing the pair into one reshape, for `q`, `k` and `v`:

| | `linear_attention` prefill | `linear_attention` decode | prefill PCC | decode PCC |
|---|---|---|---|---|
| two reshapes | 32.17 ms | 1.153 ms | 0.999432 | 0.999899 |
| **one reshape** | **32.12 ms** | **1.098 ms** | 0.999432 | 0.999899 |

**55 us, 4.8 % of the step, bit-identical accuracy.** `logs/sweep_o9_check.log`.

The `tt-perf-report` tables in `tracy/` were re-collected after this change and after `O10`, so
they are the shipped code's: `linear_attention` decode device time is 1072.3 us against the
1132.1 us that table showed before, which is exactly the 55 us plus noise.

### What is left, counted from the shipped profile

An earlier revision said "the remaining reshapes are the four the layer genuinely needs". The
shipped profile has **seven** per token, 44.6 us in total (56 dispatches over 8 replays,
`tracy/linear_attention/decode_perf_report.csv`). Naming all of them, since a count that does not
match the profile is exactly the kind of thing this document should not contain:

| op ID (first token) | us | site | why |
|---|---|---|---|
| 1126 | 6.9 | `to_heads(q)` | `[1, 1, batch, heads*D] -> [1, batch*nv, 1, D]`, a last-dim change |
| 1128 | 6.4 | `to_heads(k)` | as above |
| 1129 | 3.6 | `to_heads(v)` | as above, and narrower - no `v_per_k` widening concat |
| 1132 | 4.0 | `beta` to per-head scalars | `[1, 1, batch, nv] -> [1, batch*nv, 1, 1]` |
| 1133 | 3.9 | `g` to per-head scalars | as above |
| **1144** | **16.0** | `core = reshape(out, (1, batch, nv, head_v_dim))` before the gated norm | de-pads `batch*nv` single-row heads into `ceil(batch*nv/32)` dense tile-rows |
| 1147 | 3.3 | `normed_flat` back to `[1, 1, batch, value_dim]` | the output projection wants the flat residual shape |

The 16.0 us one is worth a measurement rather than an assumption, because RMSNorm reduces over
the last dimension and both shapes already have `head_v_dim` last — so the reshape looks
removable. `probes/probe_decode_reshapes.py` times the two *complete* arrangements at the real
shapes, median of five batches of 200 with the input uploaded once
(`logs/probe_decode_reshapes.log`):

| arrangement | us per call | spread over 5 batches |
|---|---|---|
| **shipped**: reshape to `[1, 1, 48, 128]` → typecast → norm → flatten | **33.01** | 0.72 |
| alternative: typecast → norm on `[1, 48, 1, 128]` → flatten | 33.04 | 1.13 |
| (norm alone on the padded tensor, neither flatten nor cast — for scale) | 13.80 | 0.43 |

**A dead heat: 0.04 us apart, inside either spread**, at PCC 0.999999999999997 against each
other. The reshape does not add work, it *relocates* it — whichever arrangement is chosen, the
de-pad happens once, either before the norm or in the flatten afterwards. Kept, on the grounds
that it is free and the shipped order is the one every other number in this document was
measured on.

Two things about how this was measured are worth recording, because the first version of the
probe got both wrong and reported a spurious 7.0 us win for the alternative. It left the
`ttnn.from_torch` upload inside the timed loop, so a single-digit-microsecond difference was
riding on a ~285 us host-bound baseline; and it omitted the **float32 → bfloat16 typecast** that
sits between the reshape and the norm in the layer, which is precisely the op the de-padding
helps most (48 tile-rows at 1/32 occupancy against 2 dense ones). Hoisting the upload and adding
the cast turned a 7.0 us win into a 0.04 us tie.


---

## 15. The policy in the measured rows (OPT-013)

A dtype policy is intent until the profiler shows it. Read straight out of
`tracy/<kind>/<phase>_perf_report.csv`, one row per distinct matmul:

### decode, per token

<!-- generated:policy-decode -->
| matmul | role | fidelity / dtypes | DRAM-sharded program | `in0_block_w` | us |
|---|---|---|---|---|---|
| `32 x 5120 x 17408` | MLP gate, MLP up | LoFi BF16 x BFP4 => BF16 | yes | 5 | 325.6 |
| `32 x 17408 x 5120` | MLP down | LoFi BF16 x BFP8 => BF16 | yes | 17 | 193.7 |
| `32 x 5120 x 10240` | GDN `in_proj_qkv` | LoFi BF16 x BFP8 => BF16 | yes | 5 | 117.5 |
| `32 x 5120 x 6144` | GDN `in_proj_z` | LoFi BF16 x BFP8 => BF16 | yes | 5 | 71.2 |
| `32 x 6144 x 5120` | GDN `out_proj` | LoFi BF16 x BFP8 => BF16 | yes | 6 | 71.1 |
| `32 x 5120 x 128` | `b\|a` | HiFi4 BF16 x FP32 => FP32 | no | 5 | 27.7 |
| `b={48} x 32 x 128 x 128` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | — | 17.9 |
| `b={48} x 128 x 32 x 128` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | — | 13.4 |
| `32 x 5120 x 8192` | `wqkv` | LoFi BF16 x BFP8 => BF16 | yes | 5 | 94.5 |
| `32 x 6144 x 5120` | `o_proj` | LoFi BF16 x BFP8 => BF16 | yes | 6 | 71.5 |
| `32 x 5120 x 6144` | `wgate` | LoFi BF16 x BFP8 => BF16 | yes | 5 | 71.3 |
<!-- /generated:policy-decode -->

### prefill, 2048 tokens

<!-- generated:policy-prefill -->
| matmul | role | fidelity / dtypes | DRAM-sharded program | `in0_block_w` | us |
|---|---|---|---|---|---|
| `2048 x 5120 x 17408` | MLP gate, MLP up | LoFi BF16 x BFP4 => BF16 | no | 4 | 3077.0 |
| `2048 x 17408 x 5120` | MLP down | LoFi BF16 x BFP8 => BF16 | no | 8 | 1115.5 |
| `b={48} x 64 x 128 x 128` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | 1 | 995.6 |
| `2048 x 5120 x 10240` | GDN `in_proj_qkv` | LoFi BF16 x BFP8 => BF16 | no | 4 | 903.6 |
| `b={48} x 64 x 64 x 128` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | 1 | 574.4 |
| `2048 x 5120 x 6144` | GDN `in_proj_z` | LoFi BF16 x BFP8 => BF16 | no | 8 | 514.2 |
| `2048 x 6144 x 5120` | GDN `out_proj` | LoFi BF16 x BFP8 => BF16 | no | 8 | 446.2 |
| `b={768} x 32 x 32 x 32` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | — | 273.6 |
| `b={384} x 64 x 128 x 64` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | — | 219.5 |
| `2048 x 5120 x 128` | `b\|a` | HiFi4 BF16 x FP32 => FP32 | no | 2 | 156.0 |
| `b={384} x 32 x 32 x 32` | delta-rule batched | HiFi4 FP32 x FP32 => FP32 | no | — | 41.2 |
| `2048 x 5120 x 8192` | `wqkv` | LoFi BF16 x BFP8 => BF16 | no | 4 | 723.7 |
| `2048 x 5120 x 6144` | `wgate` | LoFi BF16 x BFP8 => BF16 | no | 8 | 514.2 |
| `2048 x 6144 x 5120` | `o_proj` | LoFi BF16 x BFP8 => BF16 | no | 8 | 445.7 |
<!-- /generated:policy-prefill -->

Rows are deduplicated across the two layer kinds: a shape that both kinds run under the same
dtype and fidelity appears once. Both tables are regenerated from
`tracy/<kind>/<phase>_perf_report.csv` by `probes/derive_report_numbers.py --write`.

Everything the policy claims is on the row: BFP4 exactly where the policy says BFP4, BFP8
everywhere else, LoFi on every weight matmul, and HiFi4 + float32 preserved on the delta-rule
state recurrence and on `b|a`. No dominant decode matmul is left at `in0_block_w <= 2` — the
smallest is 5 and the MLP down projection runs at 17. `test_precision_policy_reached_the_weights`
asserts the same thing from the device tensors so a regression fails the suite rather than
waiting for the next profile.

The "DRAM Sharded" column is `False` for the prefill rows by design: the weight *is* width
sharded across the DRAM banks, but prefill runs the 2D multicast program over it rather than the
DRAM-sharded one, which is exactly the coupling §6 describes.


---

## 16. Final gates

All re-run on the shipped code after the last change (`O14`), in one sitting: the suite, the
full-context case and its BFP4-attribution control, the watcher run, the headline pair and the
four Tracy profiles.

<!-- generated:gates -->
| gate | result | log |
|---|---|---|
| `tests/test_optimized_decoder.py` | **72 passed, 2 skipped, 3 warnings in 758.77s (0:12:38)** | `logs/suite_main.log` |
| `--long-context`, prompt 262143 and decode at 262143 | 1 failed, 1 passed, 72 deselected, 3 warnings in 360.57s (0:06:00) - the failure is the inherited `full_attention` decode-SDPA defect at 0.547175 (section 10) | `logs/long_context.log` |
| watcher, `TT_METAL_WATCHER=10` | **30 passed, 44 deselected, 3 warnings in 295.49s (0:04:55)**, `watcher.log` clean | `logs/watcher_run.log` |
| stress, repeated prefill+decode passes | min PCC 0.996503 / 0.987445 prefill, 0.996257 / 0.985865 decode | in `logs/suite_main.log` |
| BF16/HiFi4 structural prefill over the disputed lengths, bar 0.999 | worst 0.999898 / 0.999434 | in `logs/suite_main.log` |
| worst real-weight PCC over the disputed lengths, prefill and decode | **0.996689** / **0.996181** | in `logs/suite_main.log` |
| BFP4-attribution control at 262143 | prefill tails 0.985447 -> 0.997937 and 0.996595 -> 0.999062 with BFP8 gate/up; everything the MLP does not touch identical | `logs/long_context_bfp8_control.log` |
| runtime host-fallback audit | passes (source scan plus `forbid_host_fallback` around a measured prefill and decode) | in `logs/suite_main.log` |
| batch 4 and 32, per-user page tables and positions | pass, eager and traced | in `logs/suite_main.log` |
| in-process speed-up, measured by the suite itself | decode 2.074x / 2.039x, prefill 1.785x / 2.179x | in `logs/suite_main.log` |
<!-- /generated:gates -->

The suite re-measures the headline itself rather than trusting this document:
`test_optimized_decode_beats_fused` and `test_optimized_prefill_beats_fused` build both decoders
in-process on the same weights and compare them, and
`test_real_weight_pcc_at_disputed_lengths` recorded a worst real-weight PCC of **0.996689** and
**0.996181** across lengths 1, 17, 64, 743, 2049 and 5000, prefill and decode, against the
unmodified 0.995 bar — three decode-token draws at the two tightest points and one elsewhere,
which is why those numbers are lower than the single-draw 0.997799 / 0.997501 earlier revisions
reported (§23).

`test_precision_policy_reached_the_weights` and `test_decode_matmuls_are_dram_sharded` assert
§15's table from the device tensors, so a silent fallback to interleaved weights or to BF16
fails the suite rather than waiting for the next profile.

### A log line that looks like a fault and is not

The first prefill of a new chunk length emits one or more
`TT_THROW: Statically allocated circular buffers ... beyond max L1 size` lines at `critical`
level. That is `_prefill_linear` walking its candidate program configs and catching the ones
that do not allocate (§6); the winner is cached per `(weight, M tiles)` and no later call
retries. `O10` uses the same pattern for the batched delta-rule matmuls. The tests around those
lines pass, and the determinism and 12-repeat stress cases pass bit-identically.

The residual risk in that pattern is worth naming: the search keeps the *first* configuration
that allocates, and L1 occupancy at the moment of the first call is part of what "allocates"
means. Every failure observed in every shipped run here is the shape-only kind (`grow to
1585920 B which is beyond max L1 size`), which is occupancy-independent, so the choice was
deterministic. A first call under materially different L1 pressure could in principle pin a
slower configuration for the rest of the process.

---

## 17. O10 — program configs for the batched delta-rule matmuls

`tt-perf-report` puts "No program_config specified, try using one to override `in0_block_w` and
`out_subblock_h/w`" on the `linear_attention` prefill's batched delta-rule matmuls (the fusing
stage's `F21` loop). An earlier revision of this section rejected it on the grounds that a
program config would fix a core grid across a loop whose shapes change with the ragged final
chunk — the same objection `_prefill_linear` already solves by caching per shape. So it was
measured.

**The first implementation was too broad and this section records that.** It measured two shapes
and then applied one fixed 8x6 config to *every* tile-aligned batched matmul. The triangular
inverse's single-tile base case got **slower**: comparing the profile at the first stage commit
with the one after, over identical op counts, `b={768} x 32 x 32 x 32` went **271.5 -> 535.9 us**
and `b={384} x 32 x 32 x 32` went **41.5 -> 69.4 us**. A 48-core grid is narrower than what the
default program picks for a 768-deep batch of one-tile matmuls.

So every shape the prefill actually dispatches was swept — grid x `in0_block_w` x
`out_subblock_h`, PCC-gated against the default (`probes/batched_matmul_sweep.py`,
`logs/batched_matmul_sweep.log`):

| shape | share of prefill | default | **11x10** | 8x6 | 8x8 |
|---|---|---|---|---|---|
| `b=48 x 64 x 128 x 128` | 1976 us / 128 ops | 20.3 us | **9.8** | 12.7 | 10.5 |
| `b=384 x 64 x 128 x 64` | 218 us / 4 ops | 57.0 us | **25.4** | 34.6 | 29.8 |
| `b=48 x 64 x 64 x 128` | 576 us / 96 ops | 14.5 us | 13.6 | **9.4** | 9.7 |
| `b=768 x 32 x 32 x 32` | 536 us / 32 ops | **11.9** | 15.6 | 17.8 | 14.8 |
| `b=384 x 32 x 32 x 32` | 69 us / 8 ops | 11.6 us | 12.0 | **10.0** | 21.7 |

Two rules fall out, and both are in `_batched_program_config`:

* **a matmul one tile tall is left alone.** There is nothing to spread over a fixed grid, and the
  default program beats every explicit candidate at the batch depth that matters (11.9 us against
  a best of 14.8). This is what fixes the regression above.
* **above that, the grid follows K**: the full 11x10 worker grid when K is 4 tiles or more, the
  48-core grid (one per value head) for the narrower K.

`in0_block_w = 1` and `out_subblock_h = 1` are the measured optimum for every shape here, not an
unexplored floor: 2 and 4 were swept and are 3-10 % slower (`11x10_ibw2` 10.1 us and `ibw4`
10.4 us against 9.8 us on the largest shape). That is the answer to the
`in0_block_w=1 is small, try in0_block_w=2 or above` advice this group carries in the profile,
and it belongs in §11's table as a measured rejection.

Whole-layer effect on `linear_attention` prefill, real weights: **32.12 -> 31.18 ms** with the
first implementation, **-> 30.96 ms** with the shape-aware one. In the profile, device time over
the signposted window goes 30,480 us (pre-`O10`) -> 29,406 (first implementation) -> **29,140**
(shape-aware), and the two regressed rows are back where they started:

| row | pre-`O10` | first `O10` | shape-aware `O10` |
|---|---|---|---|
| `b={768} x 32 x 32 x 32` | 271.5 us | **535.9** | **274.7** |
| `b={384} x 32 x 32 x 32` | 41.5 us | **69.4** | **41.6** |
| `b={48} x 64 x 64 x 128` | 863.2 us | 576.0 | 572.7 |
| `b={48} x 64 x 128 x 128` | 2074.4 us | 1976.2 | 1994.7 |
| `b={384} x 64 x 128 x 64` | 475.8 us | 217.5 | 218.3 |

Worth noting where the model and the isolated sweep disagree: the sweep has
`b={48} x 64 x 128 x 128` at 9.8 us on 11x10 against 12.7 on 8x6, but in the layer both land at
~15.5 us per dispatch.

**This paragraph originally attributed that gap to contention with the chunk loop's other ops.
That was wrong, and `O12` (§19) is what disproved it**: the slow dispatches in the group were
exactly the two that read the recurrent state, which `ttnn.zeros` had left DRAM-interleaved.
With the state in L1 the group runs at **998.9 us over 128 dispatches — 7.8 us each**, i.e.
*below* the isolated sweep's 9.8 us rather than 60 % above it. The lesson kept here is the
opposite of the one first written down: when a shape is slower in the layer than in a probe,
look at where its operands live before blaming the neighbours. The rule is still chosen on the
sweep because that is the only place the shapes can be compared cleanly, and the whole-layer
number is what is reported.

Decode is untouched (1.099 ms) and PCC moves 0.999432 -> 0.999436
(`logs/sweep_o10_check.log`, `logs/sweep_o10b_check.log`).

The candidates are **not** bit-identical, as an earlier revision of this section said: the
measured PCC against the default program is 0.99999997, and the whole-layer prefill PCC moves in
the seventh digit. Shapes the reuse program refuses are cached as `None` after one failed
dispatch and fall back to the default.

---

## 18. The causal conv's tilize/untilize, audited

The goal contract asks for no *unnecessary* tilize/untilize in the measured path. After `O8`
and `O13` the `linear_attention` prefill carries **1.61 ms** of it over 11 ops (5.6 % of
28.9 ms); `full_attention` carries **none** (no layout op at all on any row). Every site, from
`tracy/linear_attention/prefill_perf_report.csv`. Op IDs move with every re-profile, so the
sites are named here and the table below is regenerated from the shipped CSV by
`probes/derive_report_numbers.py --write`:

* the conv window **concat**, `ttnn.concat([prefix, mixed_qkv], dim=-2)` - two untilizes and a
  retilize, because the prefix is `conv_kernel_size - 1 = 3` rows;
* the **tap slices** `window[j : j+L]` for `j = 1, 2` - a causal conv's shifted views start at
  rows 1 and 2 and a row shift of a tiled tensor is never aligned. `j = 0` starts at row 0 and
  needs no layout op at all, which is visible in the table as a slice with no untilize before it;
* the **conv state save**, `window[logical-1 : logical-1+K]` - the state is taken at the logical
  end of the chunk, and the untilize is of the whole window even though only 3 rows are kept;
* the **ragged output trim** after the output projection, because `seq_len` is not a tile
  multiple.

A fifth site used to be here and is now gone: `g_last = g_cum[:, :, chunk-1 : chunk, :]`, the
per-chunk total decay, cost 308.7 us of untilize/slice/tilize. `O13` (§22) replaces it with a
reduction.

<!-- generated:prefill-layout -->
| op ID | us | op |
|---|---|---|
| 921 | 3.6 | `UntilizeWithUnpaddingDeviceOperation` |
| 922 | 195.7 | `UntilizeWithUnpaddingDeviceOperation` |
| 924 | 260.2 | `TilizeWithValPaddingDeviceOperation` |
| 928 | 201.1 | `UntilizeWithUnpaddingDeviceOperation` |
| 930 | 266.0 | `TilizeDeviceOperation` |
| 932 | 202.5 | `UntilizeWithUnpaddingDeviceOperation` |
| 934 | 261.9 | `TilizeDeviceOperation` |
| 936 | 201.1 | `UntilizeWithUnpaddingDeviceOperation` |
| 938 | 8.3 | `TilizeWithValPaddingDeviceOperation` |
| 1775 | 4.0 | `UntilizeWithUnpaddingDeviceOperation` |
| 1777 | 7.8 | `TilizeWithValPaddingDeviceOperation` |
| **total** | **1612.2** | 11 ops; `full_attention` prefill has 0 |
<!-- /generated:prefill-layout -->

These are **structural, not accidental**: a causal convolution is a sum of row-shifted views, and
a row shift of a tile-laid-out tensor is never tile-aligned. The fusing stage already removed the
one avoidable member of this family (`F22`: tap `K-1` is `mixed_qkv` itself, no slice), and `O8`
already halved every one of these tensors by moving the stage to bfloat16.

The dedicated op that would replace the whole thing is `ttnn.conv1d`. It was tried
(`probes/probe_gdn_prefill_ops.py`) with the real contract — 10240 channels, kernel 4, depthwise
(`groups = 10240`), length 2048 — and refused:

```
DRAM Auto slice could not find valid slice configuration. Tried up to 1 slices for
height-slicing on output dimension 1. Available L1: 1461504 bytes. Operation requires
more memory than available even with maximum slicing.
```

That is an exact op-contract blocker, not a first API error: the op's own auto-slicer reports it
cannot fit a 10240-channel depthwise conv in 1.46 MB of L1 at any slicing it can generate.

Because a depthwise conv does not mix channels, splitting the width is an **exact**
decomposition rather than an approximation, so the shape was adapted and retried at 2, 4, 8 and
16 splits. Every one is blocked, and the blocker changes:

| splits | channels each | result |
|---|---|---|
| 1 | 10240 | `DRAM Auto slice could not find valid slice configuration ... even with maximum slicing` |
| 2 | 5120 | the same auto-slicer failure |
| 4 | 2560 | `Out of Memory: Not enough space to allocate 1040 B L1_SMALL buffer across 65 banks ... bank size is 0 B` |
| 8 | 1280 | the same `L1_SMALL` failure |
| 16 | 640 | the same `L1_SMALL` failure |

Below 2560 channels the op stops failing on capacity and starts failing on a device contract:
`ttnn.conv1d` wants an `l1_small_size` region reserved at `open_mesh_device` time, and this
decoder's mesh is opened without one because nothing else in the layer needs it.

An earlier revision stopped there and dismissed reserving one with an argument — "it would take
L1 away from the DRAM-sharded decode matmuls". The sixth stage review called that the weakest
rejection in the document, and it was right: a device-open flag is a one-line change, not a
blocker. So the probe now reopens the mesh with `l1_small_size = 32768` and retries
(`probes/probe_gdn_prefill_ops.py`, `logs/probe_gdn_prefill_ops.log`):

| splits | channels each | with `l1_small_size` reserved | per split | **total for 10240 channels** |
|---|---|---|---|---|
| 2 | 5120 | still the auto-slicer failure | — | — |
| 4 | 2560 | **runs** | 1158.4 us | **4633.8 us** |
| 8 | 1280 | **runs** | 768.4 us | 6147.4 us |
| 16 | 640 | **runs** | 547.2 us | 8754.5 us |

**`ttnn.conv1d` is legal after all, and it loses.** The shipped causal conv — every op from the
prefix concat through the typecast back to float32, layout and arithmetic together — is
**4451.1 us** in the profile (op IDs 921-940, `tracy/linear_attention/prefill_perf_report.csv`).
The best conv1d decomposition is 4633.8 us, **4 % slower**, and that is before it does any of
the work it does not cover: the ragged-chunk prefix concat and the conv-state save still have to
happen for chunked prefill to continue across calls, and they are two of the four layout sites
in the table above. Splitting further makes it worse, monotonically — the per-split cost falls
much more slowly than the split count rises, which is the signature of a fixed per-dispatch cost
dominating a 2048-row depthwise conv.

So the rejection is now a measurement rather than a contract error: **the dedicated op runs, and
the hand-written sum of row-shifted views is faster.** The `l1_small_size` reservation is not
needed and is not taken.

The structural fix that would subsume the conv, the chunk loop and the triangular inverse in one
op is `O5`'s kernel, rejected upstream on accuracy (§7) — and, once it was finally timed, not a
latency win either: 28.89 ms warmed at 2048 tokens against this layer's 28.90 ms warmed prefill
for *everything* (28.899 ms, `logs/sweep_final_default.log` — the like-for-like comparison is
warmed wall against warmed wall, not against the 27.11 ms of device time an earlier revision
quoted; it is a dead heat, not a loss, and either way not a win). So the honest ranking of what
is left is: **1.61 ms of conv layout, whose dedicated replacement is measurably slower**, and no
fused alternative that is currently both accurate and fast.


---

## 19. O11 and O12 — the decode shard grid, and the state operand of the chunk loop

Two findings from auditing the shipped profile rather than the code.

### O11 — ttnn picks its own decode shard grid, and this layer cannot follow it

`logs/suite_main.log` carries **555** occurrences of

```
Mismatch between computed MemoryConfig(... grid=[{0,0}-{10,1}, {0,2}-{9,2}] ...)
and provided MemoryConfig(... grid=[{0,0}-{7,3}] ...) ... Using computed config
(matmul_device_operation.cpp:865)
```

for shard shapes `[32,160]`, `[32,192]` and `[32,544]` — the residual, the 6144-wide projections
and the MLP intermediate. The DRAM-sharded matmul lays 32 cores out **row-wise across the whole
11-wide device grid**, not as the 8x4 rectangle `_find_grid` produces, and silently uses its own.
The visible cost is one `ReshardDeviceOperation` per decode step in the attention epilogue —
**1.83 us (`linear_attention`) and 1.56 us (`full_attention`)** — where the next op expects the
rectangle.

Matching the op's choice was implemented and **is not legal for this decoder**:

```
Sharded layernorm does not support non-rectangular core grids. The shard spec grid has
32 cores but its bounding box spans 33 cores (11 x 3).
(layernorm_device_operation.cpp:187)
```

The residual norm carries the same shard, so the whole `O3` contract would have to give up the
sharded layernorm. A core count that is rectangular *both* as a grid and row-wise on an 11-wide
device would have to be at most 11 or a multiple of 11, while it also has to divide the tiled K
of every role — 160 for the 5120-wide projections and 544 for the down projection — and no such
count exists. So the rectangle stays and the reshard is now in §4's boundary table with its cost,
rather than being contradicted by a docstring that claimed the epilogue added no layout op.

Reported as a ttnn observation: the op should either accept the caller's grid or reject it,
not override it silently — the warning fires once per affected dispatch, **555** times in the
shipped `logs/suite_main.log`, which is noise that hides real warnings.

### O12 — the recurrent state was the only DRAM operand left in the chunk loop

The `b={48} x 64 x 128 x 128` prefill group ran at 20.6 us per dispatch in the layer against
9.8 us for the same shape in isolation, and §17 first put that down to contention. It was not:
the two dispatches in that group that read the recurrent state (`v_prime = kc_i @ state` and
`inter = qd_i @ state`) were the slow ones, while the neighbouring `b={48} x 64 x 64 x 128`
dispatches ran *faster* in the layer than in the probe. The difference is operand placement —
`state` came from `ttnn.zeros` with no memory config, so it was DRAM-interleaved for all 32
chunks, and this module's own `_l1_groups` docstring records the rule that breaks: a batched
matmul over a DRAM-resident operand costs ~1.06 us per batch element against ~0.043 us out of L1.

The state is 3.1 MB at batch 1 against a 96 MB L1 budget, so it now moves to L1 for the duration
of the loop and back once at the end, and `decayed_state`, `update` and `new_state` stay there
with it.

| | `linear_attention` prefill | decode | prefill PCC | decode PCC |
|---|---|---|---|---|
| state in DRAM | 30.96 ms | 1.099 ms | 0.999436 | 0.999897 |
| **state in L1** | **29.13 ms** | 1.098 ms | 0.999436 | 0.999897 |

**1.83 ms, 6 % of the prefill, at identical PCC** (`logs/sweep_o12_check.log`). `full_attention`
is unaffected (9.29 -> 9.20 ms, inside noise) because it has no chunk loop.


---

## 20. The precision-independent structural gate — half of it turned out to be possible

§13 relaxes the synthetic-weight bar to 0.98 because BFP4 weights are lossy on a Gaussian. The
obvious way to get back the structural sensitivity that gives up is a second gate that runs the
same unfriendly lengths with precision taken out of the picture — BF16 weights, HiFi4 — on the
optimized code path.

An earlier revision of this section recorded that as **impossible**, on the strength of two
throws seen while building it:

```
Statically allocated circular buffers on core range [0-0 - 7-9] grow to 2638592 B
which is beyond max L1 size of 1572864 B                        (program.cpp:1582)
Statically allocated circular buffers in program 224 clash with L1 buffers on core
range [0-0 - 7-9]. L1 buffer allocated at 1511424 and static circular buffer region
ends at 1524480                                                  (program.cpp:1639)
```

and attributed both to the prefill program-config search of §6 being sized for the shipped
weights. **That attribution was wrong, and `probes/probe_bf16_structural.py` is what showed it.**
The probe builds the layer at `FUSED_BASELINE_PRECISION` and walks the disputed lengths one at a
time, reporting a PCC or an exception for each (`logs/probe_bf16_structural.log`):

| kind | seq 1 | 17 | 64 | 743 | 2049 | 5000 | decode |
|---|---|---|---|---|---|---|---|
| `linear_attention` prefill | 0.999898 | 0.999932 | 0.999941 | 0.999944 | 0.999943 | 0.999942 | throws |
| `full_attention` prefill | 0.999973 | 0.999833 | 0.999720 | 0.999500 | 0.999443 | 0.999434 | throws |

**Prefill at BF16 works at every disputed length, including the short chunks the earlier
revision said were the blocker**, and it works to five nines. What throws is the *decode* step
that follows it — `program 222`, not a prefill program. The cause is the other half of the
topology: `O2`'s DRAM-sharded decode config picks `per_core_N` from the tiled output width for
the shipped BFP4/BFP8 weights, and a BF16 tile is 1.9-3.6x larger, so the `in1` circular buffer
overflows L1 before the first dispatch. The two throws are different failures — 1582 is the
`full_attention` decode matmul exceeding L1 outright, 1639 is the `linear_attention` one
clashing with what is already allocated — and neither is the prefill search.

Two things follow.

**The gate exists now, for prefill.** `test_structural_prefill_at_high_precision` runs the
disputed lengths at BF16/HiFi4 on the optimized path at a **0.999** bar, against a measured worst
of 0.999434. With precision removed, anything that moves that number is structure — padding,
masking, chunking, cache fill, output slicing — which is exactly the coverage the 0.98 synthetic
bar gives up. It is a real gate rather than a wide one: the structural breaks this stage actually
hit landed at 0.24-0.50.

**What is still not covered is decode at high precision**, and that is a genuine limitation of
the optimized decode topology rather than a documentation gap: the layer cannot be built with
BF16 weights and DRAM-sharded decode matmuls at the same time. It is recorded rather than fixed
because the fix — teaching `_decode_matmul_plan` to fall back to a narrower `per_core_N` or to
the interleaved program when the weight dtype is wide — would add a code path that only ever runs
in a test, at a precision the layer never ships. For decode, the structural argument stays the
statistical one: a break is not a small PCC loss.

**The consequence for the contract** is unchanged in substance and narrower in scope than the
earlier revision claimed: `from_state_dict(weight_dtype=...)` remains a supported way to pin
dtypes and is how `test_bfloat8_kv_cache` and every sweep policy work, but pinning every weight
to BF16 is not a supported *decode* configuration. Prefill at BF16 is fine at every length.

One more thing came out of the original investigation and stands. The 1639 throw is the
**occupancy-dependent** failure mode rather than the shape-only one — a config that allocated
once can clash later depending on what else is live in L1. `_prefill_linear` now re-searches when
a *cached* config fails instead of trusting it forever, which is a real robustness improvement
for the shipped policy, and it closes a residual risk earlier revisions could only name.


---

## 21. Hardware recovery, once

Recorded because `$tt-device-usage` asks for it, and because it is infrastructure rather than a
model result.

**Failure signature.** Two gate chains were accidentally launched overlapping, so a second
`pytest` waited on `CHIP_IN_USE_2_PCIe` while the first still held it. Killing the pair left the
device unusable: `timeout 60 tt-smi -ls --local` hung (exit 124), and a bounded
`timeout 180 tt-smi -r` failed with

```
Resetting all PCI devices: [0, 1, 2, 3]
Error when re-initializing chips!
Read 0xffffffff over PCIe ID 0: the board should be reset.
```

**Recovery.** `Read 0xffffffff` is the ARC signature the skill lists as recoverable.

1. Killed only the stale processes from this run (`pgrep -f test_optimized_decoder`), nothing else
   — no `CODEX_HOME`, no logs, no repo state.
2. `timeout 240 tt-smi -r /dev/tenstorrent/2 /dev/tenstorrent/3` — the reset of the two chips this
   stage uses succeeded; the command still exits non-zero because its post-reset re-init scan
   walks **all four** PCI devices and trips over PCIe ID 0.
3. That board is the one `doc/context_contract.json` already records as wedged and awaiting an
   operator power cycle (`hardware.boards.000004613192404C`: "chip 0 ARC wedged"). It is not the
   board this stage runs on, and `tt-smi -ls` therefore cannot complete on this host at all —
   which is why the mesh smoke, not the device list, is the health check here.
4. Mesh smoke on the stage's own device, per the skill:
   `TT_VISIBLE_DEVICES=2 python -c "open_mesh_device(MeshShape(1,1)); close_mesh_device()"` →
   **1 device, opened and closed cleanly**.
5. Resumed the same stage from preserved state: no earlier evidence was regenerated, and the gate
   chain was re-run once from the top.

No profiler or watcher collection was attempted while the card was unhealthy, and no result in
this document comes from a run that touched the failure.

---

## 22. `O13` — the per-chunk decay slice, found by re-reading §18's own table

The §18 audit exists to prove the remaining tilize/untilize is structural. Regenerating its
table against the shipped profile put a **fifth** site in it that the prose never explained —
308.7 us, larger than the ragged trim and the concat's own untilizes, and nowhere near the
causal conv:

```
1089  250.6 us  UntilizeWithUnpaddingDeviceOperation
1090   22.1 us  SliceDeviceOperation
1091   58.1 us  TilizeWithValPaddingDeviceOperation
```

Its position in the dispatch order — immediately after `ttnn.concat(pieces, dim=0)` closes the
triangular inverse and after `exp(g_cum)` and `k_beta * exp_gcum` — identifies it exactly:

```python
g_last = ttnn.slice(g_cum, [0, 0, chunk - 1, 0], [nc, nv, chunk, 1], memory_config=mem)
```

the per-chunk total decay, read as the **last row** of the cumulative sum. Row `chunk - 1 = 63`
of a tiled tensor is not a tile boundary, so ttnn untilizes all of `g_cum`, takes one row, and
retilizes.

The fix is arithmetic, not layout: **the last element of a cumulative sum is the sum.** `g_cum`
is `ttnn.cumsum(g_h, dim=-2)`, so `g_last` is `ttnn.sum(g_h, dim=-2, keepdim=True)` — a reduction
over the chunk axis, which is a tile-friendly operation and produces the `[nc, nv, 1, 1]` shape
both consumers (`decay_to_end = exp(g_last - g_cum)` and `exp_g_last`) already broadcast against.
The reduction runs immediately after the `cumsum`, so `g_h` is deallocated on the very next line
as before and nothing is kept alive across the triangular inverse; what survives to the two
consumers is the `[nc, nv, 1, 1]` `g_last`, which is smaller than the slice it replaces.

| | `linear_attention` prefill | decode | prefill PCC | decode PCC | log |
|---|---|---|---|---|---|
| `g_last` as a row slice | 29.132 ms | 1.098 ms | 0.999436 | 0.999897 | pre-`O13` `sweep_final_default.log` |
| **`g_last` as a reduction** | **28.882 ms** | 1.098 ms | 0.999428 | 0.999899 | `sweep_o13_glast_check.log` |
| shipped, after the full re-run | **28.899 ms** | 1.098 ms | 0.999428 | 0.999899 | `sweep_final_default.log` |

**250 us, 0.9 % of the prefill** (real weights, seq 2048). The first two rows are separate
invocations of the same harness rather than one process, so the third row is included: the
shipped re-run lands at 28.899 ms, inside 17 us of the check, which is the run-to-run noise of
this measurement and well under the effect. The PCC moves in the sixth digit in both directions —
a cumulative sum and a plain sum over the same 64 values do not round identically, and neither is
more correct than the other. `full_attention` has no chunk loop and is untouched.

The layout audit it came from confirms it directly: §18's table went from 13 ops totalling
1942.4 us to **11 ops totalling 1612.2 us**, and the `UntilizeWithUnpadding` / `Slice` /
`TilizeWithValPadding` triple that was `g_last` is simply absent from the shipped profile.

The general lesson, and the reason this section exists rather than a one-line changelog entry:
**an audit table is only as good as the number of rows the prose accounts for.** Four sites were
explained and five were measured, and the unexplained one was the second-cheapest fix in the
whole stage.

---

## 23. `O14` — the BFP4 output gate that passed the test and still does not hold the bar

This one is worth reading as a method result rather than a precision result. It is the case the
draw-sensitivity rule of §9 was written for, and it is the only candidate in this stage that was
adopted, measured end to end, and then **un**adopted on evidence.

### Why it looked right

§9 rejects BFP4 for `attn_qkv`, `attn_out` and `gdn_out` on real weights at the disputed
lengths — each on its own layer kind's numbers, after the fifth stage review found `gdn_out`
being rejected on an `attn_out` measurement. `attn_gate` was measured in the same field-by-field sweep and was *not* rejected there —
it was folded into the stacked candidate that failed and went down with the stack. Split back out
and run alone through the same lengths (`logs/sweep_v6_attn_short_17.log`,
`logs/sweep_v6_attn_short_743.log`, real weights, `full_attention`):

| candidate | seq 17 prefill | seq 17 decode | seq 743 prefill | seq 743 decode | 2048 prefill | 2048 decode |
|---|---|---|---|---|---|---|
| BFP8 gate (shipped) | 0.998295 | 0.997929 | 0.998961 | 0.999620 | 0.999270 | 0.999651 |
| `attn_gate` BFP4 | 0.997039 | 0.996975 | 0.998388 | 0.999461 | 0.998836 | 0.999508 |
| `attn_qkv` BFP4, for contrast | **0.987711** | **0.988919** | **0.993865** | 0.998292 | 0.996002 | 0.998927 |

Every number clears 0.995. The decode saving is real and reproduces at all three lengths — 7.0,
6.4 and 6.7 us, three independent measurements inside 0.6 us of each other, 0.6 % of a
`full_attention` step. `attn_qkv` fails at both short lengths, so the field split was doing its
job. On synthetic weights the cost was measured too (`logs/sweep_v7_gate_synth.log`): prefill
0.986537 → 0.983532 and decode 0.984493 → 0.981423, both still above §13's 0.98 bar, so
**no bar would have had to move**.

So it was adopted, and the whole gate chain was re-run on it: suite **70 passed, 2 skipped**
(the count before §20's structural gate was added), worst real-weight PCC 0.996131, watcher
clean, 262143-token case unchanged except for its inherited failure. By every gate this stage
had, `O14` shipped.

### What was wrong with that

Its worst number, `full_attention` decode at seq 17, was **0.996131** — 0.0011 above the bar.
§9's rule says a candidate that close has not been shown to hold the bar, because changing only
which decode token is drawn moves real-weight PCC by more than that. Applying the rule
(`probes/probe_draw_sensitivity.py --seq 17 --kinds full --candidates default,bfp8_gate`,
`logs/probe_draws_o14_seq17.log`, real weights, `full_attention`, seq 17):

| gate dtype | 777 | **917** (the suite's draw) | 11 | 12345 | 2024 | 4242 | worst | prefill |
|---|---|---|---|---|---|---|---|---|
| **BFP4 (`O14`)** | 0.996975 | 0.996131 | **0.994316** | 0.996195 | 0.995060 | 0.995320 | **0.994316** | 0.997039 |
| **BFP8 (shipped)** | 0.997929 | 0.997501 | 0.996181 | 0.997430 | 0.996721 | 0.997203 | **0.996181** | 0.998295 |

(The run above is on the shipped code, so `default` in the log *is* the BFP8 row and `bfp4_gate`
is the candidate; an earlier run of the same probe with `O14` still adopted produced the same two
sets of six numbers with the labels the other way round, which is the reproducibility check.)

**`O14` is below the 0.995 bar at seed 11, and within 0.00006 of it at seed 2024.** It passed the
suite because `test_real_weight_pcc_at_disputed_lengths` draws `900 + seq_len` = 917, which is
the second-best of the six. The shipped BFP8 gate is above the bar on all six with 0.0012 to
spare — thin, and now honestly stated, but not the same thing.

**Decision: rejected.** 0.6 % of `full_attention` decode — 0.16 % of the model's decode step,
since 16 layers of 64 are `full_attention` — is not worth a policy that fails the accuracy bar
on one draw in six.

### What this cost and what it bought

It cost a full gate chain on a configuration that is not shipping. It bought three things worth
more than that:

* **the rule in §9 is now load-bearing rather than decorative.** It was written from the
  `proj_fp32_acc=False` reconciliation and immediately caught a different candidate that every
  other gate had passed;
* **`test_real_weight_pcc_at_disputed_lengths` had a blind spot, and the gate itself is now
  narrower for it.** One draw per length is one sample of a distribution whose spread this stage
  measures at 0.0011-0.0048. The test still runs one draw at most lengths — six lengths x two
  kinds x three draws would triple a twelve-minute suite for coverage the probe gives on demand —
  but it now runs **three draws at the two tightest points** (`full_attention` seq 17,
  `linear_attention` seq 743), which are exactly where `O14` and `gdn_out` BFP4 hid. Seeds 11 and
  2024 are in that set, so both would now fail the gate directly. A candidate landing within
  ~0.001 of the bar anywhere else still needs `probes/probe_draw_sensitivity.py`;
* **the shipped policy's real margin is measured, not assumed, and the suite now reports it.**
  `full_attention` decode at seq 17 is the tightest point in the whole precision policy at
  **0.996181**, and `linear_attention` decode at seq 743 is next at **0.996689**. Those two
  numbers are what the 0.995 bar actually has under it, and since the gate samples three draws
  at exactly those points they are also what `real_weight_worst_pcc` reports in
  `logs/suite_main.log` — the headline accuracy number went *down* from 0.997799 / 0.997501 to
  0.996689 / 0.996181 as a result of this round, not because the layer got worse but because the
  measurement got honest.

