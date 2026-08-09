# Qwen3.6-27B optimized decoder — work log

Stage: `optimized_decoder`. Code under test: `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`.
Baseline: `tt/fused_decoder.py` (previous stage), re-measured in this stage on the same device,
in the same process, through the same harness.

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

Where the shard contract is deliberately broken, and why:

| boundary | op | why |
|---|---|---|
| `wqkv` → `nlp_create_qkv_heads_decode` | one sharded→L1-interleaved conversion | the head-creation op takes an interleaved input, and a DRAM-sharded matmul must write a sharded one |
| `in_proj_qkv` → causal conv | one sharded→L1-interleaved conversion | the gated-delta-net decode path works on interleaved tensors; the tensor is 1.3 MB |
| residual → `in_proj_ba` | one sharded→L1-interleaved conversion | `in_proj_ba` is outside the DRAM-sharded family (its 112-column output is smaller than one shard row) |
| SDPA-decode output | `to_memory_config` to the head shard | the decode SDPA kernel rejects a sharded output for GQA (`sdpa_decode_device_operation.cpp:405`), carried over from the fused stage |

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

Both pairs consume the same post-norm activation, so packing them is the OPT-001 pattern. It
was **not** taken, for the reason the packed gate/up case demonstrates: packing widens N, which
raises `per_core_N` for the DRAM-sharded decode matmul and forces the same L1 wall — `wqkv|wgate`
would be N = 14336 (`per_core_N` = 14 at 32 cores) and `in_proj_qkv|in_proj_z` N = 16384
(`per_core_N` = 16), against the 34 that already fails. More decisively, the pair would have to
share one output dtype and one shard contract, and they do not: `in_proj_qkv` must emit float32
for the conv/state path while `in_proj_z` is bfloat16 and must land in `out_proj`'s activation
shard, and the packed output would then need a width slice of a width-sharded tensor plus a
typecast, which is exactly the cost the split gate/up measurement above shows to be decisive.
The measured evidence is the gate/up row: at these shapes and this dtype policy, one wider
DRAM-sharded matmul loses to two narrower ones.

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

The selection rule maximises `out_block_h * in0_block_w` — tiles of work per buffer refill —
among the configurations that fit the per-core circular-buffer budget, breaking ties toward the
larger `out_block_h`. It reproduces the measured optimum for the gate, the down projection, QKV
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

**Decision: rejected on measured real-weight accuracy**, not on effort or preference. The
kernel's own preprocessing recipe is what loses the precision — it inverts the intra-chunk
triangular factor with Neumann doubling plus two Newton-Schulz steps at HiFi2, which is the
method the functional stage already measured as the cancellation-prone one at these magnitudes,
and the sequential scan then compounds it across chunks. Making it usable needs a change inside
the kernel (fp32 accumulation in the scan) or a higher-precision `L_inv` contract, neither of
which is reachable from Python in this checkout. Recorded as a ttnn improvement candidate with
two model-free reproducers.

The prefill win it would have bought is not lost entirely: §6's program configs take
`linear_attention` prefill from 51.7 ms to 41.9 ms without touching the delta-rule math.

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

## 9. Attention-precision follow-up on the final topology (OPT-007)

The §2 BFP4-attention trial ran on the pre-O2 topology, which OPT-007 calls screening evidence
only. Re-measured on the final topology — DRAM-sharded decode matmuls, split gate/up, sharded
residual — with everything else at the default policy
(`probes/sweep.py --group attn_precision --real`, `logs/sweep_attn_precision.log`):

| candidate | linear decode | linear prefill PCC | full decode | full prefill PCC | decision |
|---|---|---|---|---|---|
| BFP8 attention + GDN (default) | 1.167 ms | 0.999460 | 1.094 ms | 0.999270 | **kept** |
| BFP4 on QKV / `in_proj_qkv` | 1.149 ms | **0.992530** | 1.082 ms | 0.996002 | rejected |
| BFP4 on output projection | 1.158 ms | 0.998537 | 1.087 ms | 0.998535 | rejected |
| BFP4 on the gate / `in_proj_z` | 1.159 ms | 0.995861 | 1.087 ms | 0.998836 | rejected |
| BFP4 on all attention/GDN projections | 1.135 ms | **0.987644** | 1.068 ms | **0.994787** | rejected |
| BFP4 attention + bfloat16 conv | 1.153 ms | 0.999432 | 1.068 ms | **0.994787** | rejected |
| MLP down BFP4 | 1.128 ms | 0.999221 | 1.057 ms | **0.993099** | rejected |
| `in_proj_ba` bfloat16 instead of float32 | 1.166 ms | 0.999460 | 1.094 ms | 0.999270 | rejected (no gain) |
| BF16 KV cache instead of BFP8 | 1.167 ms | 0.999460 | 1.094 ms | 0.999281 | rejected (no gain, 2x memory) |
| weight matmuls at HiFi2 instead of LoFi | 1.694 ms | 0.999702 | 1.612 ms | 0.999395 | rejected |
| LoFi without `fp32_dest_acc_en` | 1.160 ms | 0.999259 | 1.088 ms | 0.999120 | rejected |

The BFP4 attention family was tried group by group, on real weights, on the final topology, and
every member either falls below the 0.995 bar (`in_proj_qkv` at 0.9925, the whole family at
0.9876) or erodes the margin for 7-18 us of a 1.09-1.17 ms step. The `linear_attention` numbers
are the decisive ones and the reason is structural: those projections feed the gated delta
rule's state recurrence, whose conditioning the functional stage already documented — the same
reason the state math keeps HiFi4.

`in_proj_ba` at bfloat16 is *free* accuracy-wise but also free performance-wise (1.166 vs
1.167 ms, inside noise), so the float32 weight is kept: it is 2.6 MB and it feeds the two gates
of the recurrence.

`proj_fp32_acc=False` buys 6-7 us and costs a digit of PCC on both kinds; kept `True`.

---

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

Collected with advice enabled (`probes/run_perf.sh`, `tracy/<kind>/<phase>_perf_report.txt`).

| advice | where | action |
|---|---|---|
| "No program_config specified, try using one to override in0_block_w and out_subblock_h/w" | `b={48} x 64 x 128 x 128` and `b={48} x 128 x 64 x 128` batched matmuls, `linear_attention` prefill, 2.6 ms | **not taken.** These are the per-chunk delta-rule matmuls inside the L1-resident loop (F21). They already run out of L1, which is the 24x factor the fusing stage measured; a program config would fix a core grid across a loop whose operand shapes change with the ragged final chunk. Left as the largest remaining `linear_attention` prefill item. |
| "If possible place input 0 in L1 (currently in DRAM_INTERLEAVED)" | `2048 x 5120 x 10240` and `2048 x 5120 x 128` prefill matmuls | **not taken.** Input 0 is the 2048x5120 bfloat16 post-norm activation, 20 MB; an interleaved-L1 residency for it competes with the per-core circular buffers of the very matmul it feeds, and §6's configs already bind on L1. |
| "in0_block_w=4 and output subblock 1x4 look good" | prefill matmuls | already satisfied by §6. |
| "Use HiFi2 or HiFi4 with BF16 activations for improved accuracy" | every LoFi matmul | **rejected with measurement**: HiFi2 costs 0.53 ms of decode and 4.4 ms of prefill (§9) for PCC that is already far above the bar. |
| "No output subblock size found" | the DRAM-sharded decode matmuls | **not actionable**: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no output-subblock field (`in0_block_w`, `per_core_M`, `per_core_N`, `fused_activation` only). Reported as a `tt-perf-report` improvement candidate: the advice does not apply to this program class. |
| "Output subblock 1x1 is small" | `32 x 5120 x 128`, the `b\|a` projection, 28 us | **not taken**: the whole output is four tiles wide, so there is no larger subblock to have. |

---

## 12. Performance accounting

Per-token decode, one layer, from the same run (`tracy/<kind>/decode_perf_report.csv` for
device time, `probes/sweep.py` for end-to-end, both at position 2048, batch 1).

Roofline: bytes the measured path must move per token / aggregate DRAM bandwidth. Weight bytes
at their stored dtypes plus the KV-cache read.

**`linear_attention`** — `mlp_gate` + `mlp_up` 2 x 50.2 MB (BFP4), `mlp_down` 94.7 MB (BFP8),
`in_proj_qkv` 55.7 MB, `in_proj_z` 33.4 MB, `out_proj` 33.4 MB (BFP8), `in_proj_ba` 2.6 MB
(FP32), norms/taps/state 3.4 MB ⇒ **323.6 MB**. The gated-delta-net recurrent state is read and
written once (3.1 MB each way at batch 1) ⇒ **329.8 MB**.

**`full_attention`** — `mlp_gate` + `mlp_up` 100.3 MB, `mlp_down` 94.7 MB, `wqkv` 44.6 MB,
`wgate` 33.4 MB, `o_proj` 33.4 MB, norms 0.02 MB ⇒ 306.4 MB, plus the paged KV read at position
2048: 2 x 4 heads x 2049 x 256 x 1.0625 B = 4.5 MB ⇒ **310.9 MB**.

At the 512 GB/s aggregate DRAM bandwidth of one Blackhole chip, and against the device time
from `tracy/<kind>/decode_perf_report.csv` and the end-to-end time from `probes/sweep.py`:

| | roofline | device time | end-to-end | device→e2e gap | roofline / device |
|---|---|---|---|---|---|
| `linear_attention` decode | 0.637 ms | 1.132 ms | 1.153 ms | 0.021 ms | 56.3 % |
| `full_attention` decode | 0.607 ms | 1.083 ms | 1.094 ms | 0.011 ms | 56.1 % |

`tt-perf-report`'s own modeled figure agrees: 51.1 % of the DRAM roofline, 262 GB/s, for both
kinds (`tracy/*/decode_perf_report.console.log`).

**end-to-end = device time + dispatch gap + host work.** The measured op-to-op gap inside the
signposted window is 42.7 us per token (`linear_attention`) and 39.5 us (`full_attention`),
which is *larger* than the 21 us / 11 us end-to-end excess — i.e. the replay pipeline overlaps
some of it, and there is **no host term left in the traced decode loop**: the window contains
`execute_trace` calls and nothing else, and the harness uploads every input before the start
signpost. The remaining 0.5 ms between roofline and device time is concentrated in two named
places, both measured rather than assumed:

* the **BFP4 gate/up rows**: 325.7 us of the 1132 us step for 100.3 MB, i.e. 308 GB/s where
  every BFP8 row reaches 442-477 GB/s. At BFP8 those two rows would move 189 MB and take
  397 us, so BFP4 is still the right choice; the 60 % efficiency is a ttnn-side property of the
  BFP4 read path at M = 32 (§3);
* the **gated-delta-net decode's small state ops**: 103.5 us of `ReshapeView` (the per-head
  layout change, a last-dim reshape and therefore an untilize/retilize), 67.6 us of `BinaryNg`,
  35.8 us of `LayerNorm`, 31.5 us of `Copy` and 29.0 us of `Ternary` — about 300 us of the
  1132 us `linear_attention` step spent on tensors small enough that fixed per-op cost
  dominates. `full_attention` has the equivalent in its 226.2 us `SdpaDecode` row (§10).

Prefill reconciles the same way: 30.48 ms of device time against 33.19 ms measured inside the
signpost for `linear_attention` (1.11 ms of op-to-op gap over 936 ops), and 8.92 ms against
9.41 ms for `full_attention` (11 us of gap over 23 ops). Prefill is compute-bound, not
DRAM-bound, so its 6.7 % / 15.5 % DRAM figure is expected rather than a finding; the FLOP
column of `tracy/full_attention/prefill_perf_report.txt` is the relevant one there.

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

The alternative was measured rather than assumed. BFP8 gate/up passes the synthetic bar
everywhere (worst 0.997650) and costs, on synthetic weights at 2049: `linear_attention` prefill
36.37 → 38.30 ms and `full_attention` prefill 13.98 → 16.06 ms. Its decode could not be
measured at all without a further change: the BFP8 gate/up decode matmul at the selected
32-core geometry overflows L1, because a BFP8 tile is 1.9x a BFP4 one and `per_core_N` is 17.
That is recorded as the cost of the fallback, not as a reason to avoid it.


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

**55 us, 4.8 % of the step, bit-identical accuracy.** The remaining reshapes are the four the
layer genuinely needs: the gated norm has to see `[..., nv, head_v_dim]` to reduce over the value
dimension and the output has to come back flat, and `beta`/`g` have to reach the per-head scalar
shape. `logs/sweep_o9_check.log`.

The `tt-perf-report` tables in `tracy/` were collected before this change, so the
`linear_attention` decode table shows the ten reshapes rather than the seven. Everything else in
it — every matmul row, the SDPA row, the op-to-op gap — is unchanged, and the headline latency
rows in `README.md` and `perf_summary.json` are the post-O9 measurement.


---

## 15. The policy in the measured rows (OPT-013)

A dtype policy is intent until the profiler shows it. Read straight out of
`tracy/<kind>/<phase>_perf_report.csv`, one row per distinct matmul:

### decode

| matmul | fidelity / dtypes | DRAM-sharded program | `in0_block_w` | us |
|---|---|---|---|---|
| `32 x 5120 x 17408` (MLP gate, MLP up) | **LoFi BF16 x BFP4 => BF16** | yes | 5 | 163.2 |
| `32 x 17408 x 5120` (MLP down) | LoFi BF16 x BFP8 => BF16 | yes | **17** | 192.9 |
| `32 x 5120 x 10240` (GDN `in_proj_qkv`) | LoFi BF16 x BFP8 => BF16 | yes | 5 | 117.6 |
| `32 x 5120 x 6144` (GDN `in_proj_z` / `wgate`) | LoFi BF16 x BFP8 => BF16 | yes | 5 | 71.2 |
| `32 x 6144 x 5120` (GDN `out_proj` / `o_proj`) | LoFi BF16 x BFP8 => BF16 | yes | 6 | 70.9 |
| `32 x 5120 x 8192` (`wqkv`) | LoFi BF16 x BFP8 => BF16 | yes | 5 | 94.8 |
| `32 x 5120 x 128` (`b\|a`) | HiFi4 BF16 x FP32 => FP32 | no | 5 | 27.7 |
| `b={48} x 32 x 128 x 128` (state recurrence) | **HiFi4 FP32 x FP32 => FP32** | no | — | 8.9 |

### prefill

| matmul | fidelity / dtypes | `in0_block_w` | us |
|---|---|---|---|
| `2048 x 5120 x 17408` (MLP gate, MLP up) | **LoFi BF16 x BFP4 => BF16** | 4 | 1535.2 |
| `2048 x 17408 x 5120` (MLP down) | LoFi BF16 x BFP8 => BF16 | 8 | 1116.2 |
| `2048 x 5120 x 10240` (GDN `in_proj_qkv`) | LoFi BF16 x BFP8 => BF16 | 4 | 901.9 |
| `2048 x 5120 x 8192` (`wqkv`) | LoFi BF16 x BFP8 => BF16 | 4 | 723.7 |
| `2048 x 5120 x 6144` (`in_proj_z` / `wgate`) | LoFi BF16 x BFP8 => BF16 | 8 | 515.9 |
| `2048 x 6144 x 5120` (`out_proj` / `o_proj`) | LoFi BF16 x BFP8 => BF16 | 8 | 445.6 |
| `b={384} x 64 x 128 x 64` etc. (delta rule) | HiFi4 FP32 x FP32 => FP32 | — | 54.4 |

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

| gate | result | log |
|---|---|---|
| `tests/test_optimized_decoder.py` | **70 passed, 2 skipped** (the two `--long-context` cases) in 11m42s | `logs/suite_main.log` |
| `--long-context`, prompt 262143 + decode at 262143 | 1 passed, 1 failed — the inherited `full_attention` decode-SDPA defect, identical to the functional (0.550293) and fused (0.551271) stages at 0.547175 | `logs/long_context.log` |
| watcher, `TT_METAL_WATCHER=10` | **30 passed**, `watcher.log` clean | `logs/watcher_run.log`, `watcher/watcher.log` |
| stress, 12 back-to-back prefill+decode passes | min PCC 0.996264 / 0.985865, DRAM bytes per bank constant | in `logs/suite_main.log` |
| runtime host-fallback audit | passes (source scan + `forbid_host_fallback` around a measured prefill and decode) | in `logs/suite_main.log` |
| batch 4 and 32, per-user page tables and positions | pass, eager and traced | in `logs/suite_main.log` |

The suite re-measures the headline itself rather than trusting this document:
`test_optimized_decode_beats_fused` recorded **2.074x** (`linear_attention`) and **2.041x**
(`full_attention`) traced decode against the fused decoder in-process on the same weights, and
`test_optimized_prefill_beats_fused` recorded **1.612x** and **2.187x**.
`test_real_weight_pcc_at_disputed_lengths` recorded a worst real-weight PCC of **0.997777** and
**0.997501** across lengths 1, 17, 64, 743, 2049 and 5000, prefill and decode, against the
unmodified 0.995 bar.

`test_precision_policy_reached_the_weights` and `test_decode_matmuls_are_dram_sharded` assert
§15's table from the device tensors, so a silent fallback to interleaved weights or to BF16
fails the suite rather than waiting for the next profile.

### A log line that looks like a fault and is not

The first prefill of a new chunk length emits one or more
`TT_THROW: Statically allocated circular buffers ... beyond max L1 size` lines at `critical`
level. That is `_prefill_linear` walking its candidate program configs and catching the ones
that do not allocate (§6); the winner is cached per `(weight, M tiles)` and no later call
retries. The tests around those lines pass.
