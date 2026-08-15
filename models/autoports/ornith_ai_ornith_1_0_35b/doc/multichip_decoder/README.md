# Ornith-1.0-35B — multichip decoder (TTNN, 4-chip Blackhole ring)

One `ornith-ai/Ornith-1.0-35B` decoder layer, both layer kinds, sharded across the four Blackhole
`p300c` chips on this host.

| | |
|---|---|
| Implementation | `models/autoports/ornith_ai_ornith_1_0_35b/tt/multichip_decoder.py` |
| Tests | `models/autoports/ornith_ai_ornith_1_0_35b/tests/test_multichip_decoder.py` |
| Single-chip baseline | `tt/optimized_decoder.py` — `MultichipDecoder` **subclasses** `OptimizedDecoder` unmodified |
| Target mesh | `ttnn.MeshShape(1, 4)`, `FabricConfig.FABRIC_1D_RING`, 2 links per hop |
| Parallelism | `TP = 4` for every dense tensor, `EP = 4` for the 256 routed experts, on the same four chips |
| Residual contract | replicated in, **bit-identical on every device** out |
| Suite | **132 passed, 3 skipped** (`logs/pytest_full_suite.txt.gz`) |
| PCC vs the single-chip TTNN baseline | 34 values, min **0.999892**, bar 0.999 |
| PCC vs the float32 HF golden | 86 values, min **0.999851**, bar 0.995 |
| Warmed 2048-token prefill | **3.47x** linear_attention, **3.39x** full_attention (87% / 85% of linear) |
| Warmed traced decode | **1.69x** linear_attention, **1.65x** full_attention (42% / 41% of linear) |
| Advertised context | **262144, unchanged**; per-device layer footprint falls 60–75% |
| Advertised batch | **32, unchanged**, exercised on the mesh at 1 / 4 / 13 / 32, aligned and not |
| Watcher | 0 fatal-class matches over 48 dumps (`watcher/census_summary.txt`) |

Everything below is produced by one script:

```bash
bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/run_evidence.sh
```

It runs six steps in the order the hardware discipline requires — suite, benchmark, whole-layer A/B,
isolated probes, Tracy + `tt-perf-report`, and watcher **last and alone**. Individual steps can be
selected with `STEPS="suite bench ab probes tracy watcher"`.

---

## 1. Contract

Identical to the optimized decoder's, including the parts a sharded implementation most easily
breaks:

* `prefill_forward(x, *, page_table=None, start_pos=0, ...)` and
  `decode_forward(x, *, current_pos, rot_idxs, page_table=None)`, same shapes, same semantics.
* **Any** logical `seq_len` in `[1, max_context - start_pos]`. The 4-way sharding adds **no**
  divisibility requirement to any public argument. `test_prefill_matches_single_chip` covers
  `seq_len` in `{1, 7, 32, 128, 129, 250, 2048, 2049, 3000}` — tile-, page-, chunk- and
  alignment-unaligned — and `test_unaligned_max_context` builds with `max_context=5000`, which is
  not a multiple of the 128-token physical alignment.
* Paged KV cache with `page_block_size=64`, arbitrary block mappings (`test_permuted_page_table`),
  per-user disjoint block spans and distinct absolute positions
  (`test_batched_decode_ragged_positions`), and chunked-prefill continuation
  (`test_prefill_continuation`).
* Decode and prefill batch up to 32, exercised **on the mesh** at 1, 4, 13 and 32, at a tile-aligned
  `seq_len` (192) and a non-aligned one (130) (`test_batched_prefill_decode_pcc`). This is not
  inherited from the single-chip stage: TP=4 changes the per-device head counts that every
  batch-sensitive decode op is bounded by, and batched non-aligned prefill is the only shape where
  the `CCL_COMPACT_ROWS` fold fires in prefill.
* The advertised 262144-token context. `test_full_context_prefill_and_decode` prefills at 262144 —
  the last legal prefill, which leaves no slot to decode into — and prefills **and** decodes at the
  non-aligned 262141.

The one deliberate difference is the **distribution** of the layer's input and output, not their
shape or semantics: inputs are replicated across the mesh and outputs are bit-identical on every
device. That is the layout a stack of these layers passes between them with no boundary conversion,
and it is asserted bitwise rather than by PCC in `test_output_is_identical_on_every_device` — a
per-device divergence would compound over 40 layers and PCC would not see it for many of them.

---

## 2. The mesh plan

### 2.1 The hardware, and what it forces

`tt-smi -ls --local` reports four Blackhole `p300c` chips: two dual-ASIC p300 cards,
`ClusterType.P300_X2`. Every chip has degree 2 and each hop carries 2 Ethernet links, i.e. the
topology is a **physical 4-ring**, not a line and not a 2D torus. Three consequences fix the plan
before any code:

1. **1D, not 2D.** There is no second mesh axis to give expert parallelism or sequence parallelism.
   Every parallel decomposition has to share the same four devices.
2. **`FABRIC_1D_RING`.** `logs/probe_ccl.txt` measures every collective at every shape the layer
   produces under both the ring and the line fabric. The ring wins at every single shape **in the
   traced rows**, which are the ones a decode step pays (e.g. the batch-1 decode tile: 22.10 us vs
   27.00 us traced all-reduce; 2048-token prefill: 196.77 us vs 267.00 us). Several *eager* rows go
   the other way — they are dominated by per-call dispatch rather than by the fabric — so the
   artifact's `eager` column is not evidence for the ring and is not used as any. The ring is real in
   the rows that matter, so the layer configures the ring.

   The fabric config, set before `ttnn.open_mesh_device`, is what actually selects it.
   `ttnn.Topology.Ring` is additionally passed to the collectives that still accept it — `all_reduce`
   and `reduce_scatter`. `ttnn.all_gather` marks both `topology` and `num_links` deprecated and
   **ignored** (`all_gather_nanobind.cpp`), so the shipped decode collective takes the ring from the
   fabric config alone. `probe_ccl.py` uses the identical spelling, so the Ring-vs-Linear rows are a
   comparison of fabric configs, which is where the difference lives.
3. **Per-device DRAM is not the constraint.** 31.75 GiB allocatable per chip against a 0.31 GiB
   worst-case per-device layer at the full advertised context (§6). Nothing about this plan is
   forced by memory; it is chosen for latency.

### 2.2 Strategy

`TP = 4` for every dense tensor and `EP = 4` for the 256 routed experts, on the same four chips,
with a **replicated residual stream**.

The replicated residual is the load-bearing choice, and it was made against the usual advice.
A sharded (reduce-scatter) residual removes half the collective bytes, but on this model it costs
more than it saves:

* Both RMSNorms would need `rms_norm_pre_all_gather` → stats all-gather → `rms_norm_post_all_gather`
  instead of one exact local norm. That is a **third and fourth** collective per layer, on the
  latency-critical path, to save part of one.
* The MoE router's top-8 is a decision over all 256 experts. Under a sharded residual the router
  input is a quarter of the hidden dimension on each device, so the routing matmul needs the full
  width gathered back — reintroducing exactly the collective the sharded residual removed, at the
  point in the layer where it cannot be overlapped.
* `logs/probe_ccl.txt` measures the sharded-residual family directly: `rs_only_*` is the
  reduce-scatter half alone. At the decode tile it is 14.38 us against 22.10 us for the full
  all-reduce — a 7.7 us saving per collective, 15.5 us per layer. The distributed-norm pair plus the
  router gather costs more than that, and `probe_ccl.txt`'s `rs_ag` rows show the round trip back to
  a replicated tensor is 22.09 us, i.e. **identical** to the stable all-reduce, because the stable
  all-reduce lowers to exactly that pair.

So the layer pays **exactly two collectives**, one after the token mixer and one after the MoE, and
`test_collectives_per_forward` pins that count and their positions by intercepting the CCL entry
points. Nothing else crosses the fabric — not attention, not the router, not the norms.

### 2.3 Per-tensor plan, with the calculated per-device shapes

`local_decoder_config()` builds the **per-device** `OrnithDecoderConfig`, so every shape the
inherited single-chip code derives from `self.cfg` is already the sharded shape and no forward path
does per-op sharding arithmetic.

| tensor / activation | global | per device (TP=4 / EP=4) | how | measured bytes/device |
|---|---|---|---|---|
| residual `x` | `[b, t, 2048]` | `[b, t, 2048]` | replicated | — |
| `attn_in` (packed QKV+gate) | `[2048, 9216]` | `[2048, 2560]` | column-parallel | in `projection_weights` |
| q heads | 16 | 4 | column-parallel | — |
| kv heads | 2 | **1** | device `d` owns kv head `d // 2` | — |
| paged K cache | `[nb, 2, 64, 256]` | `[nb, 1, 64, 256]` | local kv head | 71 303 168 |
| paged V cache | `[nb, 2, 64, 256]` | `[nb, 1, 64, 256]` | local kv head | 71 303 168 |
| `o_proj` | `[4096, 2048]` | `[1024, 2048]` | row-parallel → all-reduce | in `projection_weights` |
| `gdn_in` (packed DeltaNet in) | `[2048, 12352]` | `[2048, 3136]` | column-parallel | in `projection_weights` |
| DeltaNet key/value heads | 16 / 32 | 4 / 8 | column-parallel | — |
| DeltaNet `a` / `b` gates | 32 each | 8 each, **padded to 32** | column-parallel + tile pad | +104 448 |
| DeltaNet recurrent state | `[b, 32, 128, 128]` | `[b, 8, 128, 128]` | local value heads | 524 288 |
| prepared `ttnn.conv1d` weights | from `[8192, 1, 4]` taps | from `[2048, 1, 4]` taps | column-parallel | 8 912 896 |
| `gdn_out` | `[4096, 2048]` | `[1024, 2048]` | row-parallel → all-reduce | in `projection_weights` |
| `expert_gate_up` | `[1, 256, 2048, 1024]` | `[1, 64, 2048, 1024]` | **expert-parallel** | 113 246 208 (both) |
| `expert_down` | `[1, 256, 512, 2048]` | `[1, 64, 512, 2048]` | **expert-parallel** | ↑ |
| `shared_in` | `[1, 1, 2048, 1056]` | `[1, 1, 2048, 288]` | column-parallel | in `moe_shared_and_router` |
| `shared_down` | `[1, 1, 512, 2048]` | `[1, 1, 128, 2048]` | row-parallel → all-reduce | ↑ |
| `router` | `[1, 1, 2048, 256]` | `[1, 1, 2048, 256]` | **replicated** | ↑ |
| `expert_select` | — | `[1, 1, 256, 64]` | one-hot block of `I_256` | ↑ |
| RoPE cos/sin/trans | — | full | replicated | 67 633 152 |
| RMSNorm gains | — | full | replicated | 294 912 / 565 248 |

Byte counts are **measured**, not modelled: `logs/probe_footprint_local.py` walks shard 0 of every
device tensor of a built layer at its allocated padded size and compares each term against the
committed single-chip artifact `doc/optimized_decoder/logs/probe_footprint.txt`. Output:
`logs/probe_footprint_local.txt`. The probe reports the four MoE items above as one
`moe_shared_and_router_weights` term (1 986 560 B per device); the split between them is arithmetic
from the shapes and dtypes, not a separate measurement.

The `conv_weights` figure is the *prepared* `ttnn.conv1d` weights the layer builds at
`allocate_state`, one set per accepted prefill block length — the raw tap shape is given only to show
what is sharded.

### 2.4 The two dimensions that do not divide by 4

Both are handled explicitly rather than by rounding the model down.

**`n_kv_heads = 2` over 4 devices.** Devices 0,1 own kv head 0 and devices 2,3 own kv head 1. This
is exactly the GQA grouping the 16 query heads already impose — query head `h` uses kv head `h // 8`,
and TP=4 gives each device query heads `4d .. 4d+3`, all of which share kv head `d // 2`. So:

* each device stores **one whole** kv head, i.e. **half** the single-chip cache, not a quarter
  (`vs_ideal = 2.000` in `probe_footprint_local.txt`);
* the k/v projection rows are duplicated across each sharing pair — 2048 extra columns in `attn_in`,
  557 056 B per layer per device — 256 extra weight columns of 2048 rows at bfloat8_b, not 2048
  columns (`projection_weights` `vs_ideal = 1.077`);
* **no attention traffic crosses the fabric.** SDPA is entirely local to each device's 4 query heads
  and 1 kv head.

The alternative — splitting the 2 kv heads over 4 devices by head_dim — would make SDPA a
cross-device reduction on the decode critical path. Half a cache for zero attention collectives is
the right trade at TP=4. `test_kv_cache_is_local_heads` asserts the per-device cache shape
`[nb, 1, 64, 256]` and that pairs (0,1) and (2,3) hold equal caches while the pairs differ.

**DeltaNet `a` / `b` gates, 32 wide over 4 devices.** 8 columns per device is a quarter tile. Each
gate therefore gets its own 32-column block in the packed `gdn_in` weight whose trailing 24 columns
are exact zeros, and `MultichipDecoder._gdn_project` slices the 8 real ones back out. The padding is
internal, costs 104 448 B per layer per device, and never reaches the delta rule.

### 2.5 Rejected alternatives

| alternative | why rejected | evidence |
|---|---|---|
| Sharded (reduce-scatter) residual | needs distributed RMSNorm (2 extra collectives) **and** a router-input gather; the saving is 7.7 us/collective and the additions cost more | `logs/probe_ccl.txt` `rs_only_*` vs `all_reduce_*`; §2.2 |
| Line fabric / `Topology.Linear` | slower at **every** measured shape on a physically-ringed 4-chip host | `logs/probe_ccl.txt` |
| `ttnn.experimental.all_reduce_async` | **refuses Blackhole DRAM input outright**: `all_reduce_async_device_operation.cpp` — "does not support blackhole dram as it does not use an accessor to get the noc address" | `logs/probe_ccl.txt`, `all_reduce_async` rows FAIL at every shape |
| Explicit `reduce_scatter` + `all_gather` | identical to `ttnn.all_reduce` to two decimals at every shape — the stable all-reduce lowers to exactly that pair | `logs/probe_ccl.txt` `rs_ag_*`; `ab_layer_knobs.txt` `ccl rs_ag` |
| Sharding the MoE **intermediate** (512-wide) instead of expert parallelism | keeps 8 `sparse_matmul` loop iterations and cuts `Nt` from 32 to 8 tiles — removes parallelism the op is already short of; the `[1, E, tokens, 2048]` down output does not shrink at all | `logs/probe_expert_parallel.txt`: decode `tp` 473.28 us at 8 active vs `ep` 198.75 us at 3; prefill `tp` 1297.71 us at 162 active vs `ep` 701.31 us at 63. Each arm is at **its own** representative active count, which is what the layer runs — the counts differ because EP is what changes them |
| No expert parallelism at all (all 256 experts on every device) | 668.72 us decode against 198.75 us — **3.4x slower**. This is the `single` arm; it is still *gate-selected*, so it prices EP, not density | `logs/probe_expert_parallel.txt` `single` rows |
| Replicating all 256 experts on every device | pointless here: EP already fits with 102x DRAM headroom, and replication would quadruple expert weight traffic for zero parallelism gain | §6 |
| `ttnn.gather` for the local-expert narrowing | 57–58 us/step slower at the layer than the one-hot selection matmul, and both are bit-equal | `logs/ab_layer_knobs.txt` `routing` arms; `test_routing_select_modes_agree` |
| Inheriting the single-chip decode matmul geometry | 22 us/step slower on linear_attention, 14 us/step on full_attention | `logs/ab_layer_knobs.txt` `geometry` arms |
| Inheriting the single-chip routed sparse-matmul core rule | **4.2–4.7 ms/layer** slower on 2048-token prefill, and 16–71 us/step slower at every decode batch above 1 | `logs/ab_layer_knobs.txt` `sparse` arms and `logs/probe_decode_batch.txt` `SPARSEB` rows; `logs/probe_sparse_matmul_local.txt`; §5.7 |
| Casting the MoE's block-float output to bf16 before its collective | costs 5–6 us on every decode step and moves prefill by nothing, despite the profiler attributing ~1500 us to that collective — see §5.8, where the null result doubles as the control for that anomaly | `logs/ab_layer_knobs.txt` `cast` arms |

---

## 3. MoE: the routed decode pipeline stays gate-selected

The active-expert path from the optimized stage is preserved and made mesh-aware. **Dense
all-expert execution is not used anywhere in the shipped path.**

1. **Router — replicated, global, uncommunicated.** The `[2048, 256]` router weight is replicated,
   so every device computes the identical 256-wide logits, `topk(k=8)`, softmax and scatter. Nothing
   is communicated: replicating a `[2048, 256]` matmul is cheaper than an all-gather of its output
   plus the synchronisation it would impose, and it makes the routing decision bit-identical across
   the mesh by construction rather than by a collective's accumulation order.
2. **Narrowing — one exact matmul.** `self.w["expert_select"]` is this device's `[1, 1, 256, 64]`
   one-hot block of `I_256`; the four shards concatenate to the identity. One matmul turns the
   replicated 256-wide score vector into this device's 64-wide one, exactly: each output sums 255
   structural zeros and one bfloat16 score. `test_local_routing_selects_this_devices_experts`
   reassembles the four 64-wide local vectors and asserts they rebuild the 256-wide global one.
3. **Sparse expert projections — unchanged, on 64 local experts.** `self.cfg.num_experts` is 64, so
   the inherited `_routed_experts` chain (packed gate/up `ttnn.sparse_matmul`, SwiGLU in the sparse
   layout, score-on-the-down-input, `deepseek_moe_fast_reduce_nc` over the expert axis) runs
   untouched and produces this device's **partial** sum over experts. Only its program-config core
   target is rescaled, because EP changed the active-expert count without changing `Nt` (§5.7).
4. **Expert reduce → the layer's second all-reduce.** The four partial sums are summed by the
   collective that restores the residual.

**The zero-local-expert case.** With top-8 of 256 and 64 experts per device, a device has zero
locally-active experts with probability `(3/4)**8` ≈ 10% per token per layer. Rather than depend on
`ttnn.sparse_matmul` tolerating an all-zero sparsity — the optimized stage recorded a device **wedge**
inside this op when a count was wrong — the sparsity mask is floored at local expert 0
(`MOE_MASK_FLOOR`). That expert's routing **score** stays exactly zero and the score multiplies the
down projection's *input*, so its contribution is exactly zero.
`test_zero_local_active_experts` constructs a router state where all 8 experts land on device 0 and
asserts devices 1–3 each run exactly 1 floored expert and contribute exactly zero.

**It is still gate-selected.** `test_gate_selected_experts_not_dense` intercepts the sparsity tensors
and records the per-device non-zero counts: decode `max 4 of 64` per device, prefill `39–44 of 64`
per 32-token group. Dense would be 64.

The expected maximum over the four devices of the local active-expert count at top-8 is **3.512**
(`probe_expert_parallel.txt` header — exact, over the multivariate hypergeometric that 8 *distinct*
experts drawn from 256 into 4 blocks of 64 actually follow), which is why the sparse matmul loop
drops from 8 iterations to ~3.5. That is the single largest item in the decode window.

---

## 4. Correctness

### 4.1 Against the single-chip TTNN baseline — the primary bar

`OptimizedDecoder` is built **in the same process, on the same 1x4 mesh, from the same weights**,
with every weight replicated. On a 1x4 mesh it computes four identical copies of the single-chip
result, so device 0's copy *is* the single-chip answer. This isolates sharding and collective bugs
from the HF-vs-TTNN numerical difference the earlier stages already characterised.

Bar `BASELINE_BAR = 0.999`, deliberately far tighter than the HF bar: the two implementations compute
the same math from the same weights in the same dtypes and differ only by where the sums are
reassociated.

| | values | min | max |
|---|---|---|---|
| `test_prefill_matches_single_chip` (9 lengths x 2 layer kinds) | 18 | 0.999913 | 0.999998 |
| `test_decode_matches_single_chip` (4 steps x 2 prefill lengths x 2 kinds) | 16 | 0.999892 | 0.999993 |

Lengths: `1, 7, 32, 128, 129, 250, 2048, 2049, 3000`. Prefill lengths for decode: 130 (decode writes
cross a 64-token page boundary) and 2048 (past an internal prefill chunk boundary).

### 4.2 Against the float32 HF golden

Bar `PCC_BAR = 0.995`, inherited from the functional stage and not lowered. 86 values, min
**0.999851**, over prefill (20), decode (16), batched (32), ragged-position (4), long-context (4),
traced-decode replay (6), permuted-page-table (2) and continuation (2) cases on the mesh.

### 4.3 Sharding, layout and mesh-specific properties

| test | asserts |
|---|---|
| `test_local_config_is_the_per_device_view` | `q 16→4, kv 2→1, gdn k/v 16/32→4/8, experts 256→64, shared inter 512→128` |
| `test_local_config_rejects_an_indivisible_mesh` | a `tp` that does not divide is a construction error, not silent rounding |
| `test_mesh_is_the_target_shape` | `(1, 4)`, 4 devices, `FABRIC_1D_RING` |
| `test_weights_are_sharded_not_replicated` | every TP/EP weight's shards actually **differ** — a silently replicated weight would still look correct for the TP tensors and be wrong for the EP ones. Six weights per layer kind are checked directly; the small constants (`conv_taps`, `A_neg`, `dt_bias`, `expert_select`) are covered only indirectly, by the single-chip baseline PCC |
| `test_expert_partition_is_disjoint_and_complete` | the 4 x 64 expert blocks rebuild the 256-expert checkpoint tensor **in order** |
| `test_kv_cache_is_local_heads` | per-device `[nb, 1, 64, 256]`; pairs (0,1) and (2,3) agree, pairs differ |
| `test_collectives_per_forward` | exactly two collectives per forward, and where they sit |
| `test_output_is_identical_on_every_device` | prefill and decode outputs **bitwise** equal on all 4 devices |
| `test_ccl_modes_agree` | `all_reduce` / `rs_ag` / `stack_sum` agree with the shipped `auto` (PCC 1.000000, 1.000000, and 0.999999 / 0.999997 for `stack_sum` on the two layer kinds) |
| `test_routing_select_modes_agree` | `gather` and `select_matmul` give identical layer output |
| `test_decode_runs_the_multichip_program_configs` | the re-swept geometry actually reaches `attn_in`/`gdn_in`/`shared_down`, and its `in0_block_w` stays compatible with the residual norm's shard carry |
| `test_chunk_size_invariance` | a 6000-token prefill run with internal chunk 2048 and 1024 agrees to PCC **1.000000** — the control for the long-prefill path, where the HF golden is intractable |
| `test_sparse_cores_match_the_local_sweep` | the routed sparse matmuls run on the core counts §5.7's sweep selected, at every advertised decode batch (8/8 at batch 1, 16/16 at 2, 32/32 from 4 up) |

### 4.4 Paged KV cache on the target mesh

* `test_permuted_page_table` — a shuffled block mapping, prefill PCC 0.999914 / decode 0.999903.
* `test_batched_prefill_decode_pcc[1,4,13,32]` x `seq_len ∈ {192, 130}` — disjoint per-user block
  spans, up to the advertised batch bound, at a tile-aligned and a non-aligned prefill length.
* `test_batched_decode_ragged_positions` — 4 users at **distinct** absolute positions
  (37, 130, 200, 64) over disjoint per-user block spans, each compared to its own HF golden. The
  block mapping here is contiguous per user; the shuffled-mapping case is `test_permuted_page_table`.
* `test_prefill_continuation` — chunked prefill resumed at a non-chunk-aligned `start_pos`.
* The page table is **replicated** and indexes blocks; the head axis is what is sharded, so the page
  table contract is byte-for-byte the single-chip one.

### 4.5 Non-aligned lengths after sharding

This is the property multichip padding most easily breaks, so it is asserted at three levels:

* **public prefill lengths** — `1, 7, 129, 250, 2049, 3000` all compared to the single-chip baseline
  (§4.1). None is a multiple of the 32-token tile, the 64-token page, the 128-token physical
  alignment, or the 2048-token internal chunk.
* **`max_context`** — `test_unaligned_max_context` builds with `max_context=5000`, prefills 5000
  tokens and decodes.
* **the advertised context** — `test_full_context_prefill_and_decode` at 262144 (prefill) and 262141
  (prefill and decode).

The internal padding introduced by this stage (the DeltaNet gate blocks) is sliced at a documented
boundary inside `_gdn_project` and never reaches the delta rule or any public argument.

### 4.6 Trace, determinism, stress, fallback

| | result |
|---|---|
| `test_traced_decode_pcc` | warmed capture + replay, 3 steps per layer kind, replay PCC 0.999872–0.999981 vs the HF golden, and bitwise equal across devices on every replay |
| `test_traced_replay_does_not_leak` | 128 trace replays, **DRAM allocated growth 0 bytes** on both layer kinds |
| `test_determinism_repeated_inputs` | 3/3 runs **bit-identical on all 4 devices**, both layer kinds |
| `test_repeated_run_stress` | 12 prefill + 4-step-decode cycles over lengths `[96, 130, 257]`, repeats bit-identical, **DRAM allocated growth 0 bytes** on both kinds |
| `test_no_host_fallback_in_forward` | prefill and decode clean for `ttnn.{from_torch, to_torch, as_tensor, from_device, to_device}` and **all** `torch` ops; both guards are verified to fire, so a silently-broken guard cannot pass |

The two zero-growth results are the specific multichip risk they exist for: a CCL semaphore or
persistent buffer leaked per call would show up there and nowhere else. They cover the untraced and
the traced path separately, because a captured trace re-runs the same program and a leak inside it is
invisible to both PCC and the untraced test.

---

## 5. Performance

### 5.1 Method

`logs/bench.py`, real Ornith-1.0-35B weights, batch 1, three arms in three separate processes with
the same harness:

* **`single-chip-baseline`** — `OptimizedDecoder` on a `1x1` mesh. The "before" of every number.
* **`replication-control`** — `OptimizedDecoder` on the `1x4` mesh, every weight replicated. This
  separates "opening a 4-chip mesh changed dispatch" from "the parallelisation helped".
* **`multichip`** — this stage on the `1x4` mesh.

Prefill is a warmed 2048-token forward; decode is 32 replays of a captured trace, reported per
replay. Raw output: `logs/ab_single_vs_multichip.txt`.

### 5.2 Result

| layer kind | phase | single-chip | 1x4 replication control | multichip | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.56 ms | 101.94 ms | **29.29 ms** | **3.467x** | 86.7% |
| linear_attention | decode (traced) | 1.031 ms | 1.031 ms | **0.612 ms** | **1.685x** | 42.1% |
| full_attention | prefill 2048 | 95.37 ms | 95.68 ms | **28.14 ms** | **3.389x** | 84.7% |
| full_attention | decode (traced) | 0.827 ms | 0.827 ms | **0.502 ms** | **1.647x** | 41.2% |

The replication control is within +0.0% to +0.4% of the single-chip arm on every row, so none of the
speedup is a mesh-dispatch artifact.

`test_multichip_beats_single_chip_traced_decode` gates the decode claim in the suite itself
(`multichip 0.502 ms (1.65x)`, `0.611 ms (1.68x)`).

### 5.3 Why decode efficiency is 41–42% and prefill is 85–87%

Prefill is compute-bound and scales close to linearly. Decode at batch 1 is latency-bound: each
device's share of the work shrinks 4x but the per-op launch cost, the fixed collective latency and
the serial op count do not. The profiler says exactly where the remaining time goes.

### 5.4 `tt-perf-report`

Four captures (`tracy/{linear,full}_attention/{prefill,decode}_perf_report.*` — per-op table, CSV and
roofline summary, plus a stacked-by-op-code CSV and PNG; everything over the repo's 500 KB limit or
matching its blanket `*.csv` ignore is committed gzipped). Prefill and decode are captured in
**separate** runs so each Tracy capture holds one signposted window and one device session, and the
routed matmul rows are modelled with `--active-experts` (4 per device at batch-1 decode, 63 for a
32-token prefill group).

**Top of the stack, share of merged 4-device time:**

| | linear_attention decode | full_attention decode | linear_attention prefill | full_attention prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 17.69% | 21.54% | 75.02% | 73.94% |
| `TopK` (router) | 8.98% | 9.43% | 0.17% | 0.18% |
| dense `Matmul` (all in0 layouts) | 13.08% | 9.40% | 1.10% | 1.08% |
| **collectives (`AllGather` / `ReduceScatter`)** | **6.79%** | **9.37%** | **2.98%** | **5.78%** |
| all data movement (`DM` category) | 8.48% | 11.20% | 3.25% | 5.78% |
| DRAM roofline (modeled ops) | 6.5% (33 GB/s) | 5.9% (30 GB/s) | 27.1% (139 GB/s) | 27.3% (140 GB/s) |

`--active-experts` — the count `tt-perf-report` needs to model a `sparse_matmul` row at all — is the
**measured per-device, per-group** active count: 4 at batch-1 decode, and 41 for a 32-token prefill
group. 41 is `64 * (1 - (1 - 1/64)^(32*top_k/tp))` = 40.6, and
`test_gate_selected_experts_not_dense` measures 44 (linear) and 39–40 (full) for exactly that
quantity. Review round 2 found this at 63 — the same formula with the draws *not* divided by `tp`,
so every global draw was credited to every device — which inflated the modelled DRAM bytes and FLOPs
of the rows that are 74–75% of the prefill window by about 50%. The roofline figures above are at the
corrected input, and are **not** comparable to any roofline this stage quoted before round 2.

Findings this drove:

* **Communication is 6.8–9.4% of decode and 3.0–5.8% of prefill.** That is an upper bound on the
  cost of the whole parallelisation — §5.8 shows part of the prefill figure is barrier wait, not data
  movement — and it is small enough that no further collective-shape work would change the headline.
  In decode the two collectives are `AllGather` because `CCL_MODE="auto"` picks `stack_sum` at the
  batch-1 tile (§5.5); in prefill they are `ReduceScatter` + `AllGather`, which is what
  `ttnn.all_reduce` lowers to.
* **The routed experts dominate, and are where this stage's largest win came from.** 74–75% of
  prefill and 18–22% of decode. Every `SparseMatmul` row in all four captures is marked `SLOW`
  (257/257 rows per prefill capture, 128/128 per decode capture); 66 of the 257 prefill rows and
  96–128 of the 128 decode rows additionally carry a `1x1 output subblock` advisory, the rest
  carrying `output subblock 1x2 look good`. Following that flag led to §5.7, which took 4.2–4.7 ms a
  layer off prefill. The remaining `SLOW` marks sit at a geometry the sweep confirms is the fastest
  available — the sweep covers `out_block_w` and `sub_w` as well as core count — so they are the op's
  own advisory, not an unexplored candidate.
* **Decode DRAM utilisation is 6%.** Decode is launch- and latency-bound, not bandwidth-bound; this
  is the direct explanation for the 41–42% parallel efficiency, and it is inherited from the
  single-chip stage rather than caused by sharding.
* **`TopK` is 9% of decode.** It is the replicated router's `topk(k=8)` over 256 experts. It is
  replicated *deliberately* (§3) and is the price of an uncommunicated, bit-identical global routing
  decision; the alternative costs a collective on the critical path.
* Other rows the report marks `SLOW` are the small batch-1 dense projections, whose geometry was
  re-swept for the per-device shapes in `logs/probe_dense_matmul.txt` (§5.6). They are launch-bound
  at `M = 32`; the sweep's winner is what ships.

### 5.5 Whole-layer A/B, `logs/ab_layer_knobs.txt`

Each arm is built fresh in the same process on the same device with the same weights, three builds
per arm, so the spread is visible next to the difference. All values in ms.

| knob | arm | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|---|
| `ccl` | **`auto`** (shipped) | **0.611** | **0.502** | 29.06–29.25 | 28.13–28.60 |
| `ccl` | `stack_sum` | 0.611–0.612 | 0.502–0.503 | 29.35–29.49 | 28.57–28.82 |
| `ccl` | `all_reduce` | 0.621 | 0.508 | 28.95–29.79 | 28.40–28.54 |
| `ccl` | `rs_ag` | 0.621 | 0.508–0.509 | 29.12–29.42 | 28.10–28.44 |
| `geometry` | **multichip-retuned** (shipped) | **0.611–0.612** | **0.502** | 29.17–29.76 | 28.10–28.43 |
| `geometry` | single-chip-inherited | 0.635 | 0.516–0.517 | 29.11–29.63 | 28.39–28.64 |
| `routing` | **`select_matmul`** (shipped) | **0.611–0.612** | **0.502** | 29.14–29.34 | 28.09–28.29 |
| `routing` | `gather` | 0.669 | 0.560 | 29.32–29.48 | 28.54–28.96 |
| `sparse` | **tp-rescaled** (shipped) | **0.611** | **0.502–0.503** | **29.03–29.22** | **28.30–28.83** |
| `sparse` | single-chip-inherited | 0.611–0.612 | 0.502 | 33.00–33.35 | 32.45–32.78 |
| `cast` | **block-float** (shipped) | **0.611–0.612** | **0.502** | 28.91–29.22 | 28.38–28.42 |
| `cast` | bf16 | 0.616 | 0.507–0.508 | 29.08–29.23 | 28.54–28.55 |

All values in ms, and all at **batch 1** — which is why the `sparse` arm ties in the decode columns
here: that knob only changes decode geometry above batch 1, and `logs/probe_decode_batch.txt` is
where its decode effect is measured (§5.7).

`CCL_MODE="auto"` picks `stack_sum` at or below 64 **physical** activation rows and `all_reduce`
above. Since `CCL_COMPACT_ROWS` folds first (§5.9), a decode activation is `align_up(b, 32)` = 32
physical rows at every advertised batch, so **decode always takes `stack_sum`** and prefill (2048
rows) always takes the all-reduce. The crossover still matters, because it is what makes the fold
worth doing: unfolded, batch 4 is 128 rows and would take the all-reduce (§8.4). It is measured, not
modelled — `probe_ccl.txt`'s traced rows have `stack_sum` clearly ahead at 32 rows, a tie at 64, and
losing by a widening margin from 96 up. `stack_sum` moves 4x the bytes and still wins below the
crossover because at that size both are latency-bound and it is one fabric phase instead of two.

The `sparse` arm is the largest single item in the table (§5.7) and the only one that moves prefill
here. The others move decode only, as expected: at 2048 tokens the collectives are 3–6% of the window
and the decode geometry does not apply.

### 5.6 Dense decode geometry, re-swept at the per-device shapes

The single-chip table was tuned at the unsharded widths and does not transfer: `mcast_in0` streams
the whole `K` through each core in `in0_block_w`-tile blocks, so the winning cap is a function of how
much `N` each core owns, and TP=4 cuts `N` by four on the two wide in-projections.
`logs/probe_dense_matmul.txt` has the full ladder (9 core targets x 6 caps x every role).

Comparing each inherited entry at its **realised** grid against the local winner, in microseconds at
the batch-1 decode tile:

| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |
|---|---|---|---|---|---|---|---|---|
| `attn_in` | (32,2) → 33/2 | 30.92 | 110/8 | 17.91 | 13.01 | 0.02 | 0.11 | **retuned (110, 8)**, 17.91 |
| `gdn_in` | (110,2) → 110/2 | 48.90 | 33/8 | 20.43 | 28.47 | 0.08 | 0.18 | **retuned (110, 8)**, 20.53 |
| `shared_down` | (48,16) → 55/4 | 9.10 | 4/4 | 8.51 | 0.59 | 0.54 | 0.25 | **retuned (8, 4)**, 8.56 |
| `expert_select` | new role | — | 8/8 | 8.26 | — | 0.19 | 0.74 | **(8, 8)**, 8.26 |
| `o_proj` | (16,16) → 22/16 | 10.53 | 88/8 | 9.97 | 0.56 | 0.38 | 0.27 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.58 | 22/16 | 9.35 | 0.23 | 0.30 | 0.18 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.90 | 110/32 | 8.77 | 0.13 | 0.67 | 0.13 | inherited |
| `router` | (32,32) → 33/32 | 9.01 | 8/32 | 8.81 | 0.20 | 0.46 | 1.49 | inherited |

`repeatability` is this probe's own noise floor for that role, read out of the same file: several
distinct core *targets* collapse onto the same realised grid (16 and 24 both land on 22 and 33), so
the file contains independent repeat measurements of identical configs, and `repeatability` is the
widest disagreement between such repeats. A delta smaller than it is not a result.

Three of the four inherited rows sit inside the winner's own spread and the probe's repeatability
and keep the inherited entry. `o_proj`'s 0.56 us is outside both **in this sweep** and was 0.00–0.30
us in the four before it, with the winning core count moving between 22, 33 and 88 — so it is not a
stable result and the inherited entry stays. That instability is the reason the `repeatability`
column exists at all: judging these rows against a single sweep's spread is what put `shared_down` on
the wrong side of the line in round 1.

The retuned rows are outside that band by a wide margin. `attn_in` and `gdn_in` are the two wide
in-projections — the inherited cap of 2 is the *worst* legal value for the local shape, costing 1.7x
and 2.4x on the op — and together they move the layer by 24 us/step on linear_attention and 14
us/step on full_attention (§5.5 `geometry` arm). `shared_down` is retuned because TP=4 cuts its `K`
from 512 to 128 — 4 tiles — and a 55-core grid for a 4-tile `K` is launch overhead rather than
parallelism. It ships at **8** cores, not the 4 an earlier round chose: 4 measured 8.44–8.62 us on
three sweeps and 9.98–10.29 on another, while 8 reads 8.55–8.77 on every sweep, so 8 is the stable
choice and is never behind. At the layer the whole `geometry` arm moves by ≤0.002 ms for this row,
inside the build-to-build spread (see §8).

`in0_block_w` is additionally bounded above by the residual norm's per-core shard width where the
shard is carried into the projection: the 2048-wide norm over 8 cores gives 8 tiles per core, so 8 is
the largest cap that keeps `mcast_in0`'s `block_w % in0_block_w == 0` legal. Caps of 16 and 32 do
build against an interleaved `in0` and are measured — `attn_in` reads 18.6–18.7 and 19.5–19.6 us
against 17.91 at cap 8 — so the shipped value is the measured winner as well as the largest legal
one. The only two `FAIL` rows in the sweep are a different constraint entirely: `cores=4,
in0_block_w=32` on `attn_in` and `gdn_in`.

### 5.7 The routed sparse matmuls, re-swept at the per-device operating point

The largest single item this stage's re-tuning found, and it came from following the `SLOW` flag in
§5.4.

`OptimizedMoE` targets `clamp(bound // SPARSE_CORES_PER_ACTIVE[role], 8, 32)` cores for the routed
matmuls, where `bound` is `_active_expert_bound` = `min(num_experts_local, rows * top_k)`. The
divisor means "work per core, per active expert" and was calibrated at `E = 256`, where that bound
saturates at 256. Expert parallelism does **not** change `Nt` — `moe_intermediate_size` (512) and
`dim` (2048) are not sharded, so the op still produces 32 and 64 output tiles — but it cuts the
bound's ceiling to 64, which shrinks the core target exactly where the available parallelism did not
change.

`MultichipMoE._sparse_cfg` scales the bound back by the same divisor, i.e. the net rule becomes
`cores = clamp(bound, 8, 32)`. Read off the shipped constants, that changes:

| call | bound | `gate_up` realised, off → on | `down` realised, off → on |
|---|---|---|---|
| decode b=1 | 8 | 8 → 8 | 8 → 8 |
| decode b=2 | 16 | 8 → **16** | 8 → **16** |
| decode b=4 | 32 | 16 → **32** | 8 → **32** |
| decode b≥8 | 64 | 32 → 32 | 16 → **32** |
| prefill group | 64 | 32 → 32 | 16 → **32** |

So at prefill only `down` moves, and decode geometry changes at **every batch above 1**. Round 1 of
this stage recorded the opposite in four places — "16/8 → 32/32 at prefill, decode unchanged" — from
reading the rule at an assumed *active count* rather than at the bound the code computes. Round 2
caught it. `test_sparse_cores_match_the_local_sweep` now pins the realised grid at every advertised
decode batch so the table above cannot drift from the code again.

`logs/probe_sparse_matmul_local.py` re-runs the single-chip sweep's candidate ladder at the
per-device operating points (`E = 64`, `active ∈ {4, 8, 16, 32, 41, 63}`) under the same BFP4/LoFi
policy. Best microseconds per realised core count:

| active | 4 cores | 8 cores | 16 cores | 32 cores | 64 cores | winner |
|---|---|---|---|---|---|---|
| 4, `gate_up` | 82.8 | **62.6** | 64.4 | 70.0 | — | 8 |
| 4, `down` | — | **52.3** | 59.6 | 66.7 | 85.4 | 8 |
| 8, `gate_up` | 122.6 | 81.5 | **73.2** | 78.0 | — | 16 |
| 8, `down` | — | 62.2 | **61.8** | 68.9 | 88.0 | 16 |
| 16, `gate_up` | 180.5 | 110.3 | **86.8** | 90.0 | — | 16 |
| 16, `down` | — | 74.3 | **65.3** | 72.4 | 92.4 | 16 |
| 32, `gate_up` | 329.5 | 193.6 | 125.6 | **120.0** | — | 32 |
| 32, `down` | — | 119.9 | 83.1 | **80.3** | 101.5 | 32 |
| 41, `gate_up` | 415.0 | 240.2 | 150.2 | **139.3** | — | 32 |
| 41, `down` | — | 146.3 | 96.0 | **85.8** | 107.8 | 32 |
| 63, `gate_up` | 691.6 | 397.8 | 234.9 | **199.3** | — | 32 |
| 63, `down` | — | 235.7 | 144.8 | **105.1** | 125.2 | 32 |

The ladder's winner tracks the shipped rule: 8 cores at the batch-1 decode point (measured active 4),
16 in the middle, 32 from active 32 up — which includes the corrected prefill point of 41 and the
over-counted 63 the earlier calibration used, so the prefill geometry decision is the same either way.

Both effects are then measured at the layer rather than argued from the ladder:

* **prefill** — `ab_layer_knobs.txt` `sparse` arm: **33.00–33.35 → 29.03–29.22 ms** on
  linear_attention and **32.45–32.78 → 28.30–28.83 ms** on full_attention. About 4.2–4.7 ms a layer,
  and the difference between a 3.0x and a 3.4x prefill speedup.
* **decode** — `probe_decode_batch.txt` `SPARSEB` rows, three builds per arm per batch: a tie at
  batch 1 by construction, and **+33 / +50 / +23 / +38 / +49 / +71 us** a step at batch
  2 / 4 / 8 / 13 / 16 / 32 on full_attention (**+0 / +29 / +16 / +28 / +39 / +64** on
  linear_attention). Round 2 predicted this range might be *adverse*, because the isolated ladder
  favours 8 cores at low active counts; the layer says otherwise at every batch, which is why the
  layer A/B is the one that decides.

### 5.8 The block-float collective — an anomaly, and its control

The prefill profile shows the layer's two collectives at wildly different cost for the same logical
shape: `tracy/full_attention/prefill_perf_report.txt` has the BF16 reduce-scatter at 114 us on 20
cores and the BFP8 one at 1465 us on 12 cores — **4.7%** of the whole prefill window on its own, and
the largest non-sparse item in it. (The 5.27% in the stacked report is the two rows summed.) The only
difference between them is the operand dtype: the MoE half produces `bfloat8_b` under the inherited
precision policy.

That reads as an obvious win, and it is not one. `CCL_CAST_BLOCKFLOAT` casts the operand up to
`bfloat16` before the collective; the `cast` arm of `ab_layer_knobs.txt` measures it at the layer.
Casting up **costs 5–6 us on every decode step and moves warmed prefill by nothing** — the arms'
three-build ranges overlap on both layer kinds. It is therefore off.

The null result is the useful part: it is the control that classifies the anomaly. Removing the
block-float operand changes the block-float row's cost and not the layer's, which means the
profiler's 1465 us is not data movement this layer pays. The remaining explanation is the
collective's barrier absorbing whatever the four devices did not finish together, attributed to the
op that waits — expert-load imbalance under EP being the obvious candidate, since the routed matmuls
are 74% of the window and their per-device active counts are only equal on average.

That mechanism is **not** established by the numbers this README used to cite for it: the suite's
`39–44 of 64` is a spread *across layer kinds* (linear_attention logs `[44, 44, 44, 44]` — no
imbalance at all — and full_attention `[39, 39, 40, 40]`), not across devices within a run. What is
established is the negative: it is not the operand dtype and it is not data volume. The consequence
for §5.4 stands either way — the prefill CCL percentage is an upper bound on communication cost, not
a measurement of it — and pinning the mechanism would need per-device op timelines from the profiled
run, which is recorded here as an open item rather than claimed.

### 5.9 Decode at batch > 1

Every headline number above is batch 1. `logs/probe_decode_batch.py` measures what the collective is
handed at larger batches, by wrapping `_all_reduce` rather than reading it off the source:

```
SHAPE full_attention batch=32 site=mixer shape=[32, 1, 2048] padded=[32, 32, 2048]
      physical_rows=1024 useful_rows=32 waste=32.0x
SHAPE full_attention batch=32 site=moe   shape=[1, 1, 32, 2048] padded=[1, 1, 32, 2048]
      physical_rows=32 useful_rows=32 waste=1.0x
```

A tile's row axis is the *sequence* axis, so a decode activation `[b, 1, dim]` occupies `b` tile rows
of which `b` carry data. The MoE call site never had this problem — `_block` already reshapes to
`[1, 1, tokens, dim]` before it — so without a fix the same logical reduction costs 32x more at one
call site than the other in the same forward. `CCL_COMPACT_ROWS` folds `[b, t, dim]` to
`[1, 1, b*t, dim]` around the collective:

| batch | mixer physical rows | off | on | delta |
|---|---|---|---|---|
| 1 | 32 | 0.502 / 0.611 ms | 0.502 / 0.611 | 0 / 0 us |
| 2 | 64 | 0.582 / 0.709 | 0.576 / 0.704 | +6 / +5 us |
| 4 | 128 | 0.637 / 0.818 | 0.641 / 0.822 | **−4 / −4 us** |
| 8 | 256 | 0.791 / 1.007 | 0.785 / 1.001 | +6 / +6 us |
| 13 | 416 | 1.627 / 2.068 | 1.609 / 2.053 | +18 / +15 us |
| 16 | 512 | 1.828 / 2.272 | 1.799 / 2.244 | +29 / +28 us |
| 32 | 1024 | 3.023 / 3.947 | 2.934 / 3.864 | **+89 / +83 us** |

(full_attention / linear_attention; positive delta = the fold is faster.) At batch 1 the guard skips
the fold entirely, because `[1, 1, dim]` is already one tile row. It ships always-on above that; the
single 4 us regression at batch 4 is explained and quantified in §8.

The same probe carries the `SPARSEB` rows §5.7 uses, so both round-1/round-2 findings are measured on
one harness at one set of batches.

---

## 6. Context contract

`doc/context_contract.json` gains a `multichip_decoder` section. **No capability is reduced.** The
advertised 262144-token context, the batch-32 bound, non-aligned support and the public contract are
all unchanged, and the per-device footprint **falls**.

Per-device layer footprint at the full 262144-token context, batch 1
(`logs/probe_footprint_local.txt`, measured):

| | per device | single chip | `vs_ideal` |
|---|---|---|---|
| full_attention layer, total | **333 565 956 B** (0.311 GiB) | 839 553 028 B | 1.589 |
| linear_attention layer, total | **134 680 576 B** (0.125 GiB) | 533 123 072 B | 1.011 |
| paged K + V cache | 142 606 336 B | 285 212 672 B | 2.000 |
| routed expert weights | 113 246 208 B | 452 984 832 B | 1.000 |

`vs_ideal = local / (single_chip / 4)`; 1.000 is a perfect four-way split. The full_attention layer's
1.589 is dominated by two deliberately replicated terms — the RoPE tables (67.6 MB, replicated
because RoPE is applied to local heads and the tables are position-indexed, not head-indexed) and the
kv cache (2.000, §2.4) — plus the replicated router.

Whole model, projected from the measured per-layer figures (40 layers: 10 full_attention at
`full_attention_interval=4`, 30 linear_attention):

| | per device (1x4) | single chip |
|---|---|---|
| all 40 layers at 262144 tokens | **7 376 076 840 B (6.87 GiB)** | 24 389 222 440 B (22.71 GiB) |
| of which paged KV cache | 1 426 063 360 B (1.33 GiB) | 2 852 126 720 B (2.66 GiB) |
| headroom vs measured 31.75 GiB allocatable DRAM | **4.62x** | 1.40x |
| left over for embeddings, trace region and activations | **24.88 GiB** | 9.04 GiB |

The untied embedding and `lm_head` matrices are 248 320 x 2048 x 2 B = 1 017 118 720 B each, so the
single-chip path's 9.04 GiB has to hold both of them plus the trace region and every activation
buffer. The projection is conservative: it charges every full_attention layer its own copy of the
RoPE tables, which a full model shares.

**No hard physical device limit is reached, so no context reduction is considered or taken.** This
stage strictly enlarges the per-device budget.

---

## 7. Runtime audit and watcher

**Fallback audit — clean.** `test_no_host_fallback_in_forward` patches `ttnn.from_torch`,
`ttnn.to_torch`, `ttnn.as_tensor`, `ttnn.from_device`, `ttnn.to_device` and every `torch` op to raise,
runs a full prefill and a full decode, and passes. It first **verifies both guards fire** on a
deliberate violation, so a guard that silently stopped working cannot pass the test. Both layer
kinds. There is no `import torch` on any shipped forward path: the routing mask floor and the
`expert_select` one-hot are built at construction, and the only lazily-built host tensor in the file
belongs to the non-shipped `ROUTING_SELECT_MODE="gather"` arm.

**Watcher — clean.** `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_DISABLE_ETH=1`
over the state-, trace- and collective-critical subset: **43 passed, 3 skipped**
(`logs/watcher_pytest.txt.gz`). `watcher/census.py` partitions all 52 542 log lines into disjoint
buckets — an unknown line kind lands in `UNCLASSIFIED` and trips an assert rather than hiding in a
catch-all — and reports **0 fatal-class matches** over 48 dumps, minimum stack headroom 1332 bytes
free over 40 detail lines (`watcher/census_summary.txt`).

`TT_METAL_WATCHER_DISABLE_ETH=1` is a **hard tool limit here, not a choice**. With watcher
instrumenting the ACTIVE_ETH cores, the 1D-fabric ERISC program grows to 29 040 B against a 25 600 B
ACTIVE_ETH kernel config buffer on Blackhole, so every test in the subset fails at `mesh_device`
setup before any model code runs:

```
TT_FATAL: Program size (29040) too large for kernel config buffer (25600) on ACTIVE_ETH (assert.hpp:104)
```

That run is preserved verbatim as `logs/watcher_pytest_eth_enabled.txt.gz` — every selected test
errors in ~15 s, zero tests executed. No environment knob grows that buffer, so the choice is watcher
coverage on the 110 Tensix worker cores per chip or none at all. The uninstrumented cores are the
fabric routers, which this stage does not author: they are stock `ttnn` 1D-fabric kernels, identical
to what any `ttnn` CCL user runs. Every op this stage does author runs on the covered Tensix cores.

---

## 8. Known limitations

1. **Watcher does not cover the ACTIVE_ETH cores on this configuration.** Hard tool limit, evidence
   and reasoning in §7.
2. **Decode parallel efficiency is 41–42%, not ~100%.** Batch-1 decode is launch- and latency-bound
   (5.9–6.5% of the DRAM roofline), so dividing the work by four does not divide the time by four.
   The collectives are only 6.8–9.4% of the window, so this is not a communication problem and no
   collective-shape change would fix it. Prefill, which is compute-bound, reaches 85–87%.
3. **The `shared_down` retune is an op-level win that the layer cannot resolve.** The inherited
   55-core point is 0.51–0.68 us behind the winner on every sweep, against a 0.25–0.35 us probe
   repeatability, so the op-level win is real — but the whole-layer `geometry` arm moves by ≤0.002 ms
   for this row, inside the build-to-build spread. It ships at 8 cores because that reading is stable
   across sweeps where 4 cores is not (§5.6), and because the shipped table should match its own
   sweep; it is **not** claimed as a layer-level speedup.
4. **`CCL_COMPACT_ROWS` costs 4 us at batch 4 specifically.** The fold is not free, and its cost
   grows with the batch before saturating (~4 us at batch 2, ~20 at batch 4, ~26 at batch 8, ~31 at
   batch 16 and above, derived from §5.9 against `probe_ccl.txt`'s per-shape collective costs). The
   saving grows faster, so the fold wins at every measured batch except 4, where the saving
   (all_reduce at 128 rows minus stack_sum at 32 rows, ~16 us) is just under the fold's ~20 us. It
   ships always-on: a rule that excluded exactly 128 physical rows would be fitting to one point, and
   the regression is 0.6% at one batch against 2.1–3.0% at the advertised bound.
5. **The kv cache is halved, not quartered.** `n_kv_heads = 2 < tp = 4`. Deliberate (§2.4): it buys
   zero attention traffic across the fabric. A head_dim split would quarter the cache and put a
   cross-device reduction on the decode critical path.
6. **`ttnn.experimental.all_reduce_async` is unavailable on this hardware.** The op refuses Blackhole
   DRAM inputs at the device-op level. Recorded as a hardware-side refusal in `probe_ccl.txt`, not as
   a slow arm.
7. **The mechanism behind the prefill collective's device time is not pinned, only bounded.** §5.8
   establishes what it is *not* (operand dtype, data volume) with a measured control, and leaves the
   barrier/imbalance explanation as the remaining candidate rather than a demonstrated one. Pinning
   it needs per-device op timelines from the profiled run. Nothing in the stage's conclusions rests
   on it: the collective share is quoted as an upper bound throughout.
8. **Long-context PCC is validated to 8000 tokens, not 262144.** Inherited from the earlier stages
   and unchanged by this one: the eager HF full-attention reference materialises a `[seq, seq]` score
   matrix and is not tractable on host beyond that. Beyond 8000 the mesh path is cross-checked
   against *itself* under a different internal decomposition (`test_chunk_size_invariance`, PCC
   1.000000 at 6000 tokens), and the full advertised context is validated for shape, finiteness and
   non-degeneracy at 262144 and 262141.
9. **One mesh shape.** `DEFAULT_MESH_SHAPE = (1, 4)` targets this host's hardware, as the stage goal
   directs. `local_decoder_config` is written against a general `tp` and validates divisibility, and
   `test_local_config_rejects_an_indivisible_mesh` covers the rejection path, but no other mesh shape
   is measured or claimed.

---

## 9. Exact artifacts

```
doc/multichip_decoder/
├── README.md                                  this file
├── work_log.md                                what was done, in order, with measurements
├── logs/
│   ├── run_evidence.sh                        regenerates everything below, in order
│   ├── pytest_full_suite.txt.gz               132 passed, 3 skipped
│   ├── bench.py                               3-arm warmed prefill / traced decode harness
│   ├── ab_single_vs_multichip.txt             §5.2 — the speedup table's raw rows
│   ├── ab_layer_knobs.py / .txt               §5.5 — ccl / geometry / routing / sparse / cast arms
│   ├── probe_ccl.py / .txt                    §2.2 — every collective spelling x shape x topology
│   ├── probe_dense_matmul.py / .txt           §5.6 — decode matmul geometry ladder, per-device shapes
│   ├── probe_sparse_matmul_local.py / .txt    §5.7 — routed sparse geometry at E=64, active 4..63
│   ├── probe_expert_parallel.py / .txt        §2.5, §3 — EP vs intermediate-TP vs unsharded
│   ├── probe_footprint_local.py / .txt        §6 — measured per-device footprint, term by term
│   ├── probe_decode_batch.py / .txt           §5.7/§5.9 — collective shapes, and the fold and
│   │                                          sparse-core A/Bs at batch 1..32
│   ├── watcher_pytest.txt.gz                  §7 — 43 passed, 3 skipped, watcher on
│   └── watcher_pytest_eth_enabled.txt.gz      §7 — the ACTIVE_ETH kernel-buffer failure, preserved
├── tracy/
│   ├── run_profiling.sh                       4 captures, prefill and decode in separate runs
│   ├── linear_attention/prefill_perf_report.{txt,csv.gz,summary.txt,console.txt}
│   ├── linear_attention/decode_perf_report.{txt.gz,csv.gz,summary.txt.gz,console.txt}
│   ├── linear_attention/{prefill,decode}_perf_report_stacked.{csv.gz,png}
│   ├── full_attention/…                       same set
│   └── */{prefill,decode}_tracy_run.txt       the capture's own console log
└── watcher/
    ├── census.py                              disjoint line-kind partition + fatal-class check
    ├── census_summary.txt                     0 fatal-class matches, 48 dumps
    └── watcher_log.txt.gz                     the raw watcher log
```

The **raw** tt-metal op CSV (`ops_perf_results_*.csv`) is regenerated by `run_profiling.sh` but is
deliberately not committed for either phase. Merging four devices makes it ~10 000 rows x ~420
columns — 1.42 MB and 1.52 MB gzipped against the repo's hard 500 KB pre-commit limit — and trimming
it to the signposted window (454 KB / 468 KB) does **not** reproduce the committed report byte for
byte, because `tt-perf-report`'s four-device merge is sensitive to what precedes the window and
silently relabels the Device column. The committed machine-readable provenance is therefore the
per-op table `tt-perf-report` itself wrote for the exact window
(`{prefill,decode}_perf_report.csv.gz`), the stacked-by-op-code CSV, and each capture's console log,
plus `run_profiling.sh` as the exact command. The reasoning is repeated inline in that script.

Also updated by this stage: `doc/context_contract.json` (`multichip_decoder` section, `stage`,
`target`).
