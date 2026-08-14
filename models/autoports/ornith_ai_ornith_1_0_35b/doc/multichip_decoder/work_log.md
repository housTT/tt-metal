# Ornith-1.0-35B — multichip decoder, work log

Companion to `README.md`. The README states what the stage delivers; this log states what was done,
in what order, what each decision was measured against, and what was tried and rejected.

Stage boundary: `models/autoports/ornith_ai_ornith_1_0_35b/tt/multichip_decoder.py`, its test file,
and this documentation directory. No full-model or vLLM work.

---

## 0. Starting point

The single-chip baseline is `tt/optimized_decoder.py`, closed by operator decision at commit
`b6d427d6c4e` after 32 review rounds. Its warmed numbers on one Blackhole `p300c`, from
`logs/ab_single_vs_multichip.txt` (re-measured in this stage's own harness, not copied):

```
layer 0 (linear_attention)  prefill 2048  101.49 ms      decode (traced)  1.031 ms
layer 3 (full_attention)    prefill 2048   95.34 ms      decode (traced)  0.827 ms
```

It was chosen as the baseline because it is the fastest correct single-chip implementation in the
tree and because its program configs, precision policy, sharded-norm decisions, sparse-matmul
geometry rules, paged-cache contract and trace-safety properties are all already tuned. The design
decision that follows from that is in §2.

---

## 1. Hardware survey, before any code

```bash
timeout 60 tt-smi -ls --local
```

Four Blackhole `p300c` chips: two dual-ASIC p300 cards, `ClusterType.P300_X2`. Every chip has degree
2 and each hop carries 2 Ethernet links — a **physical 4-ring**. This fixes three things before a
line of the final path was written:

* **1D, not 2D.** There is no second mesh axis, so TP, EP and any sequence parallelism must share the
  same four devices. The `$multichip` skill's Galaxy 2D planning section does not apply here; its
  "1D meshes up to 8 chips → start with 1D tensor parallelism" default does.
* **Ring fabric.** `FABRIC_1D_RING` before `ttnn.open_mesh_device`, `ttnn.Topology.Ring` on every
  collective. Confirmed by measurement, not assumption — see §3.
* **DRAM is not the constraint.** 31.75 GiB allocatable per chip (measured in the functional stage)
  against a 0.78 GiB single-chip worst-case layer at the full advertised context. Nothing about the
  plan is forced by memory; it is chosen for latency.

---

## 2. Structural decision: subclass the optimized decoder

`MultichipDecoder(OptimizedDecoder)` and `MultichipMoE(OptimizedMoE)`, with the single-chip file
**unmodified**.

The mechanism that makes this work is `local_decoder_config()`: it builds the **per-device**
`OrnithDecoderConfig` (`n_heads 16→4`, `n_kv_heads 2→1`, `linear_num_key_heads 16→4`,
`linear_num_value_heads 32→8`, `num_experts 256→64`, `shared_expert_intermediate_size 512→128`;
`dim`, `head_dim`, `moe_intermediate_size` and `num_experts_per_tok` deliberately **not** divided).
Every shape the inherited code derives from `self.cfg` — packed projection widths, head splits, GQA
repeat factors, cache shapes, sparse-matmul `Nt`, expert-group bounds — is then already the local
shape, so no forward path does per-op sharding arithmetic.

What the subclass adds is exactly four things:

1. the local config (above);
2. setup-time weight sharding in `from_state_dict`, handing each device the slice its local config
   describes;
3. two collectives per layer, `_all_reduce`, one after the token mixer and one after the MoE;
4. a globally-correct router that emits **device-local** routing, `MultichipMoE.routing_weights`.

Everything else — precision policy, sharded norms, sparse-matmul geometry rules, trace safety, the
paged-cache contract — is inherited rather than re-derived. `test_local_config_is_the_per_device_view`
pins the config transform and `test_local_config_rejects_an_indivisible_mesh` pins that a `tp` which
does not divide is a construction error rather than a silent rounding of the model.

---

## 3. The collective topology table, measured before committing to a path

`logs/probe_ccl.py` → `probe_ccl.txt`. Every candidate spelling of the layer's collective, at every
activation shape the layer actually produces, **eager and inside a captured trace**, on the 4-device
mesh with `num_links=2`. Trace is what a decode step pays, so trace rows decide.

Traced microseconds:

| shape | `all_reduce` Ring | `rs_ag` Ring | `rs_only` Ring | `all_reduce` Linear | `stack_sum` |
|---|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.09 | 22.10 | 14.36 | 27.00 | **17.17** |
| 64 rows | 26.98 | 26.97 | 16.97 | 31.66 | 26.38 |
| 96 rows | 29.13 | 29.16 | 17.19 | 35.69 | 35.51 |
| 128 rows | 33.19 | 33.21 | 19.41 | 40.06 | 43.75 |
| 256 rows | 49.13 | 49.20 | 27.61 | 57.52 | 77.18 |
| 512 rows | 76.40 | 76.41 | 44.27 | 92.87 | 125.33 |
| decode batch 32 | 135.06 | 135.37 | 77.61 | 157.84 | 233.08 |
| prefill 2048 | 196.79 | 196.78 | 104.20 | 265.95 | 447.14 |

Five conclusions, all of which shaped the implementation:

1. **`Ring` beats `Linear` at every single shape.** The physical ring is real. `FABRIC_1D_RING` +
   `Topology.Ring` is what the layer configures; `DEFAULT_CCL_NUM_LINKS = 2`.
2. **`ttnn.all_reduce` and explicit `reduce_scatter` + `all_gather` are the same number to two
   decimal places at every shape.** The stable all-reduce lowers to exactly that pair. The `rs_ag`
   arm therefore exists to make that identity *checkable*, not as a separate candidate — and
   `ab_layer_knobs.txt` confirms it at the layer (0.622 vs 0.622 ms, 0.510 vs 0.510 ms).
3. **`ttnn.experimental.all_reduce_async` refuses Blackhole DRAM inputs outright**, at the device-op
   level:
   `all_reduce_async_device_operation.cpp` — *"does not support blackhole dram as it does not use an
   accessor to get the noc address"*. Every shape FAILs identically. This is recorded as a
   hardware-side refusal, not as a slow arm, and it is why the tuned experimental-tier CCL is not in
   the shipped path.
4. **`stack_sum` (`all_gather` onto a new leading axis, then a local `ttnn.sum`) wins below 64
   rows** even though it moves 4x the bytes, because at that size both are latency-bound and it is
   one fabric phase instead of two. It loses by a widening margin from 96 rows up. Hence
   `CCL_MODE="auto"` with `CCL_STACK_SUM_MAX_ROWS = 64`: `stack_sum` at decode batch 1–2,
   `all_reduce` for every larger batch and all of prefill.
5. **The sharded-residual family is priced here too.** `rs_only` is the reduce-scatter half alone:
   14.36 us against 22.09 us at the decode tile, i.e. a 7.7 us saving per collective and 15.5 us per
   layer. That is the entire budget a sharded residual has to pay for itself out of. §4 spends it.

---

## 4. The residual contract: replicated, decided against the usual advice

The `$multichip` skill is explicit that a replicated residual must not be the only measured contract,
and that a `reduce_scatter → all_gather` round trip mostly recreates the communication a fused path
was meant to avoid. Both points are taken, and the sharded-residual family was priced before being
rejected rather than after.

The budget from §3 is **15.5 us per layer**. Against it, a sharded residual on *this* model must pay:

* **Both RMSNorms become distributed.** With the hidden dimension fractured across devices, a local
  RMSNorm computes the wrong statistics, so each norm becomes
  `rms_norm_pre_all_gather` → stats all-gather → `rms_norm_post_all_gather`. That is a **third and
  fourth** collective per layer, on the latency-critical path, to save part of one. Even at the
  smallest measured shape a stats all-gather is a fabric phase, and §3 row 1 shows a fabric phase at
  the decode tile is 17–22 us. Two of them exceed the 15.5 us saving on their own.
* **The router needs the full hidden width.** The MoE top-8 is a decision over all 256 experts from
  the full 2048-wide residual. Under a sharded residual the routing matmul input is a quarter width
  on each device, so it needs an all-gather immediately before the router — reintroducing exactly the
  collective the sharded residual removed, at the point in the layer where it cannot be overlapped
  with anything.
* **Two of the four sublayer boundaries are not row-parallel outputs at all.** The DeltaNet path's
  recurrent state and conv state are per-value-head, and the shared expert's gate is applied after
  its own down projection; neither produces a naturally scattered residual.

So the layer pays **exactly two collectives**, and `test_collectives_per_forward` pins that count and
their positions by intercepting the CCL entry points (`prefill ['all_reduce', 'all_reduce']`,
`decode ['all_gather', 'all_gather']` — the decode pair is `stack_sum`). Nothing else crosses the
fabric: not attention, not the router, not the norms.

The consequence for the *next* stage is the point of the contract: the layer takes a replicated
activation and returns a tensor that is **bit-identical on all four devices**, so a stack of 40 of
them passes activations straight through with no boundary conversion.
`test_output_is_identical_on_every_device` asserts that bitwise rather than by PCC — a per-device
divergence would compound over 40 layers, and PCC on one layer would not see it.

`tt-perf-report` retires the question: collectives are **6.77%** (linear_attention) and **9.35%**
(full_attention) of merged 4-device decode time, and 3.06% / 6.27% of prefill. The parallelisation's
whole communication cost is under a tenth of the decode window; no collective-shape change could move
the headline.

---

## 5. Attention: `n_kv_heads = 2` over 4 devices

The only genuinely awkward dimension in the model. Three options were considered:

| option | per-device cache | attention traffic | verdict |
|---|---|---|---|
| duplicate kv heads across pairs | 1/2 | **none** | **taken** |
| split the 2 kv heads by `head_dim` | 1/4 | a cross-device reduction inside SDPA, per decode step | rejected |
| keep kv replicated on all 4 | 1/1 | none | rejected — strictly worse than the pair split |

Devices 0,1 own kv head 0 and devices 2,3 own kv head 1. This is *exactly* the GQA grouping the query
heads already impose: query head `h` uses kv head `h // 8`, and TP=4 gives device `d` query heads
`4d..4d+3`, all of which share kv head `d // 2`. So SDPA is entirely local to each device's 4 query
heads and 1 kv head, and the k/v projection rows are simply duplicated across each sharing pair.

Measured price (`logs/probe_footprint_local.txt`): the cache is `vs_ideal = 2.000` (halved, not
quartered) and `attn_in` carries 557 056 B/device/layer of duplicated k/v rows
(`projection_weights vs_ideal = 1.077`). Half a cache and 0.5 MB of weights for **zero attention
collectives** on the decode critical path is the right trade at TP=4, and the DRAM headroom (§8)
means the cache size is not a binding constraint anyway.

`test_kv_cache_is_local_heads` asserts the per-device shape `[nb, 1, 64, 256]`, that pairs (0,1) and
(2,3) hold equal caches, and that the pairs differ.

---

## 6. MoE: expert parallelism, and the routed decode pipeline

### 6.1 EP vs sharding the expert intermediate — measured

`logs/probe_expert_parallel.py` → `probe_expert_parallel.txt` runs the whole routed-expert chain
(packed gate/up sparse matmul → SwiGLU → score-on-down-input → expert reduce) in three arms:

| arm | experts/device | intermediate/device | decode us | prefill us |
|---|---|---|---|---|
| `single` (all 256, unsharded) | 256 | 512 | 664.38 | 1637.05 |
| **`ep` (expert-parallel)** | **64** | **512** | **197.91** @3 active | **429.08** @31 active |
| `tp` (intermediate-sharded) | 256 | 128 | 472.35 | 1297.74 |

Expert parallelism wins by 2.4x over intermediate sharding at decode and 3.0x at prefill. The reason
is structural, and the probe was built to expose it: `ttnn.sparse_matmul` loops once per **active**
expert and its parallelism is capped by the output tile count.

* Sharding the intermediate keeps all 8 loop iterations and cuts `Nt` from 32 to 8 tiles — it removes
  parallelism the op was already short of.
* Expert parallelism keeps the full `Nt` and the tuned geometry, and cuts the loop count to a mean of
  2 with an expected **maximum over the four devices of 3.512** at top-8, computed exactly in the
  probe header. (That figure was originally 3.538, from modelling the 8 draws as independent uniform
  assignments. The gate picks 8 **distinct** experts out of 256, so the per-device counts are
  multivariate *hyper*geometric; the multinomial shortcut overstates clustering. Corrected in
  `probe_expert_parallel.py::expected_max_load`, which now takes the block size and enumerates the
  hypergeometric exactly. The error was 0.026 experts and changed no decision, but the figure is
  quoted as exact.) The `ep` rows show the cost as a function of local active count: 200.37 us at 1
  active rising to 236.23 us at 8, so the realised ~3.5 sits near the flat part of that curve.
* Every `num_experts`-wide intermediate — packed gate/up output, its two unpacking slices, the SwiGLU
  product, the scored activation, the down output, the expert reduction — becomes 4x narrower under
  EP. Under intermediate sharding only the `moe_intermediate`-wide ones shrink; the
  `[1, E, tokens, 2048]` down output, the largest, does not shrink at all.

Dense all-expert execution is the `single` arm: **3.4x slower at decode** (664.38 us against
197.91 us). It is not used.

### 6.2 Keeping the routing decision global and uncommunicated

The top-8 is over all 256 experts, so the router weight is **replicated** and every device computes
identical 256-wide logits, `topk`, softmax and scatter. Replicating a `[2048, 256]` matmul is cheaper
than an all-gather of its output plus the synchronisation it would impose, and it makes the routing
decision bit-identical across the mesh *by construction* rather than by a collective's accumulation
order — which matters because expert selection is a discrete decision where a rounding difference
swaps an expert rather than perturbing a value.

Only the final narrowing is mesh-aware. `w["expert_select"]` is this device's `[1, 1, 256, 64]`
one-hot block of `I_256`; the four shards concatenate to the identity. It is exact — each output sums
255 structural zeros and one bfloat16 score.

`test_local_routing_selects_this_devices_experts` reassembles the four 64-wide local vectors and
asserts they rebuild the 256-wide global one exactly (256 non-zero entries over 32 tokens).

### 6.3 The zero-local-expert case

With top-8 of 256 and 64 experts per device, a device has **zero** locally-active experts with
probability `(3/4)**8` ≈ 10% per token per layer. The optimized stage had already recorded a device
**wedge** inside `ttnn.sparse_matmul` when a count was wrong (its §4.8), so depending on the op to
tolerate an all-zero sparsity was not acceptable.

`MOE_MASK_FLOOR` floors the sparsity mask at local expert 0. The floored expert's routing **score**
stays exactly zero — the global top-8 did not select it — and the score multiplies the down
projection's *input*, so its contribution is exactly zero. This is the same argument the single-chip
stage uses for its tile-padding rows.

`test_zero_local_active_experts` constructs a router state where all 8 experts land on device 0 and
asserts devices 1–3 each run exactly 1 floored expert and contribute exactly zero.

### 6.4 It is still gate-selected

`test_gate_selected_experts_not_dense` intercepts the sparsity tensors and records per-device non-zero
counts: **decode max 4 of 64 per device**, prefill 39–44 of 64 per 32-token group. Dense would be 64.

---

## 7. Re-sweeping the decode matmul geometry at the per-device shapes

The single-chip geometry table was tuned at the unsharded widths and does not transfer: `mcast_in0`
streams the whole `K` through each core in `in0_block_w`-tile blocks, so the winning cap is a
function of how much `N` each core owns, and TP=4 cuts `N` by four on the two wide in-projections.

`logs/probe_dense_matmul.py` → `probe_dense_matmul.txt` sweeps 9 core targets x 6 caps for every dense
decode role at its **per-device** shape. Comparing each inherited entry at its *realised* grid
(`_decode_1d_matmul_config` lays a target out as `ceil(target/11)` rows of `min(11, target)` columns,
and clamps `in0_block_w` to `Kt`) against the local winner:

| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |
|---|---|---|---|---|---|---|---|---|
| `attn_in` | (32,2) → 33/2 | 30.85 | 88/8 | 17.89 | 12.96 | 0.24 | 0.27 | **retuned (110, 8)**, 18.01 |
| `gdn_in` | (110,2) → 110/2 | 48.80 | 33/8 | 20.34 | 28.46 | 0.13 | 0.10 | **retuned (110, 8)**, 20.40 |
| `shared_down` | (48,16) → 55/4 | 9.10 | 4/4 | 8.54 | **0.56** | 0.13 | 0.36 | **retuned (4, 4)**, 8.54 |
| `expert_select` | new role | — | 8/8 | 8.27 | — | 0.15 | 0.25 | **(8, 8)**, 8.27 |
| `o_proj` | (16,16) → 22/16 | 9.26 | 22/8 | 9.23 | 0.03 | 0.49 | 0.12 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.42 | 22/8 | 9.20 | 0.22 | 0.27 | 0.08 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.70 | 33/32 | 8.70 | 0.00 | 0.27 | 0.12 | inherited |
| `router` | (32,32) → 33/32 | 9.04 | 66/32 | 8.87 | 0.17 | 0.35 | 0.14 | inherited |

`repeatability` is the probe's own noise floor for that role, read out of the same file: several
distinct core *targets* collapse onto the same realised grid, so the file already contains
independent repeat measurements of identical configs, and this column is the widest disagreement
between such repeats. It is the threshold a delta has to clear to be a result at all — and adding it
is the direct fix for the mistake described below.

* The two wide in-projections are where the inherited table is badly wrong: the inherited cap of 2
  is the **worst legal value** for both local shapes, costing a factor of 1.7 on `attn_in` and 2.4
  on `gdn_in`.
* Four roles are ties inside the winner's own spread and keep the inherited entry.
* Four roles are ties and keep the inherited entry — but *how* that was established had to be fixed
  twice, and both fixes are worth recording because the first version of this table was wrong in a
  way that is easy to repeat.

  The first version compared each inherited entry against the local winner using the **requested**
  core target rather than the **realised** grid. `_decode_1d_matmul_config` lays a target out as
  `ceil(target/11)` rows of `min(11, target)` columns and clamps `in0_block_w` to `Kt`, so targets
  16/24/32/48 realise as 22/33/33/55 cores and a 16-wide cap on a 4-tile `K` becomes 4. Comparing at
  the requested target compares points the shipped code never runs.

  The second version fixed that but judged the resulting deltas against the *winner's own spread*, a
  single-measurement statistic. Re-running the whole sweep three times over this stage showed that
  is not stable: `o_proj`'s delta moved between 0.00 and 0.30 us with no consistent sign, `gdn_out`'s
  between 0.07 and 0.22 us. The honest threshold is the probe's repeatability on repeated identical
  configs (0.08–0.36 us depending on role), which is now a column in the table. Under it, the four
  inherited roles are ties in every sweep.

* `shared_down` was **initially left inherited on that same reasoning, and that was wrong**: its
  realised point is 55 cores at `in0_block_w` 4, which came out 0.58, 0.68 and 0.56 us behind the
  winner on the three sweeps against a 0.36 us repeatability — consistently outside the noise, unlike
  the four above. TP=4 cuts its `K` from 512 to 128 (4 tiles), so a 55-core grid for a 4-tile `K` and
  a 64-tile `N` is launch overhead rather than parallelism. Retuned to `(4, 4)`. Honest accounting of
  the result: the op-level win is real and reproducible, but the **whole-layer** `geometry` arm moves
  by ≤0.002 ms, which is inside the build-to-build spread. It ships because it is faster where it can
  be measured and never slower, and because the shipped table now matches its own sweep — not as a
  claimed layer-level speedup. Recorded as limitation 3 in the README.

`in0_block_w` is additionally bounded above by the residual norm's per-core shard width where the
shard is carried into the projection: the 2048-wide norm over 8 cores gives 8 tiles per core, so 8 is
the largest cap that keeps `mcast_in0`'s `block_w % in0_block_w == 0` legal. The sweep's 16 and 32
rows fail to build against a sharded `in0` for exactly that reason and are **recorded as failures**
rather than omitted.

`test_decode_runs_the_multichip_program_configs` asserts that the retuned configs actually reach
`attn_in`/`gdn_in` and `shared_down` in a real decode step, checks them against
`MULTICHIP_DECODE_MATMUL_GEOMETRY` rather than a literal, and re-derives the shard-carry divisibility
rule. This is the guard the single-chip stage's review round 32 forced after a faster geometry was
found to be computing the wrong thing: a program config that does not reach the op it was swept for
is indistinguishable from one that does, in every measurement except this one.

---

## 8. Context contract

`doc/context_contract.json` gains a `multichip_decoder` section, and `stage` / `target` are updated.

The per-device footprint had to be **measured**, not divided by four on paper, because several terms
are not 1/4 of the single-chip term. `logs/probe_footprint_local.py` walks shard 0 of every device
tensor of a built layer at its allocated padded size and compares each term against the committed
single-chip artifact `doc/optimized_decoder/logs/probe_footprint.txt`:

```
LOCALFOOTPRINT full_attention   total  bytes=333565956  single_chip=839553028  vs_ideal=1.589
LOCALFOOTPRINT linear_attention total  bytes=134680576  single_chip=533123072  vs_ideal=1.011
```

The 1.589 decomposes into exactly the deliberate replications: RoPE tables `vs_ideal=4.000`
(position-indexed, not head-indexed — there is nothing to shard), kv cache 2.000 (§5), norms 4.000
(the replicated residual makes them exact and local), router inside
`moe_shared_and_router vs_ideal=1.781`. Everything genuinely shardable is at 1.000, and the two rows
above it are the two internal paddings: `projection_weights` 1.077 on full_attention (duplicated k/v
rows) and 1.012 on linear_attention (the DeltaNet `a`/`b` gate blocks, 8 real columns in a 32-column
tile-aligned block, sliced back out in `_gdn_project`).

Projected over the whole model (40 layers, 10 full_attention at `full_attention_interval=4`):

```
per device (1x4):  7 376 076 840 B =  6.87 GiB   headroom vs 31.75 GiB allocatable: 4.62x
single chip:      24 389 222 440 B = 22.71 GiB   headroom:                          1.40x
```

**No capability is reduced.** The advertised 262144-token context, the batch-32 bound and the
non-aligned support are all unchanged; this stage strictly *enlarges* the per-device budget, and the
1.40x → 4.62x change is the reason the multichip decoder is the right layer stack for the full model
(the single-chip 1.40x has to hold the untied embedding and `lm_head`, 1 017 118 720 B each, plus the
trace region and every activation buffer, in ~9.4 GiB).

---

## 9. Non-aligned sequence lengths

The property multichip padding most easily breaks. The internal padding this stage introduces is the
DeltaNet gate block (24 zero columns per gate), and it is sliced at a documented boundary inside
`_gdn_project` before the delta rule ever sees it. Nothing else changed, and no public argument
gained a divisibility requirement.

Asserted at three levels rather than one:

* **public prefill lengths** — `1, 7, 32, 128, 129, 250, 2048, 2049, 3000`, each compared to the
  single-chip TTNN baseline. `7`, `129`, `250`, `2049` and `3000` are not multiples of the 32-token
  tile, the 64-token page, the 128-token physical alignment or the 2048-token internal chunk.
* **`max_context`** — `test_unaligned_max_context` builds with `max_context=5000`, prefills 5000
  tokens and decodes.
* **the advertised context** — `test_full_context_prefill_and_decode` at 262144 **and** 262141.

---

## 10. Validation strategy

The primary bar is the **single-chip TTNN baseline**, not HF. `_baseline_pair` builds
`OptimizedDecoder` in the same process, on the same 1x4 mesh, from the same weights, with every
weight replicated — so it computes four identical copies of the single-chip result and device 0's
copy *is* the single-chip answer. That isolates sharding and collective bugs from the HF-vs-TTNN
numerical difference the earlier stages already characterised, which is the whole point of the
comparison.

`BASELINE_BAR = 0.999`, deliberately far tighter than the HF `PCC_BAR = 0.995`: the two
implementations compute the same math from the same weights in the same dtypes and differ only by
where the sums are reassociated. Anything looser would let a real sharding bug through on the
strength of the HF bar's slack.

Results from `logs/pytest_full_suite.txt.gz` (**104 passed, 3 skipped**; the 3 skips are
`linear_attention` cases for KV-cache, page-table and ragged-position tests, which that layer kind
does not have):

| | n | min | max |
|---|---|---|---|
| vs single-chip baseline, prefill | 18 | 0.999913 | 0.999998 |
| vs single-chip baseline, decode | 16 | 0.999892 | 0.999993 |
| vs float32 HF golden (all cases) | 49 | 0.999851 | — |

All three are unchanged across the three full evidence sweeps this stage ran, to the last digit.

Everything else the mesh has to prove:

* `test_weights_are_sharded_not_replicated` — every TP/EP weight's shards actually **differ**. A
  silently replicated weight would still look correct for the TP tensors and be wrong for the EP
  ones, and would make every speedup number meaningless.
* `test_expert_partition_is_disjoint_and_complete` — the 4 x 64 expert blocks rebuild the 256-expert
  checkpoint tensor **in order** (worst per-expert PCC 0.9935, which is bfloat4_b weight
  quantisation from the inherited precision policy, not a partition error).
* `test_permuted_page_table` (shuffled block mapping, prefill 0.999914 / decode 0.999903),
  `test_batched_prefill_decode_pcc[1,4,13,32]`, `test_batched_decode_ragged_positions`
  (4 users at distinct absolute positions 37/130/200/64 over a shuffled disjoint block mapping, each
  against its own HF golden), `test_prefill_continuation`.
* `test_traced_decode_pcc` — warmed capture and replay on the mesh, replay PCC 0.999872–0.999981.
* `test_determinism_repeated_inputs` — 3/3 runs bit-identical **on all 4 devices**.
* `test_repeated_run_stress` — 12 prefill + 4-step-decode cycles over `[96, 130, 257]`, repeats
  bit-identical, **DRAM allocated growth 0 bytes** on both layer kinds. This is the specific
  multichip risk it exists for: a CCL semaphore or persistent buffer leaked per call shows up here
  and nowhere else.
* `test_ccl_modes_agree` and `test_routing_select_modes_agree` — the rejected arms agree with the
  shipped one (PCC 1.000000 / 1.000000 / 0.999999, and bit-equal respectively), so the A/B numbers in
  §3 and §11 compare implementations that compute the same thing.
* `test_no_host_fallback_in_forward` — prefill and decode clean for
  `ttnn.{from_torch, to_torch, as_tensor, from_device, to_device}` and **all** `torch` ops, on both
  layer kinds, with **both guards verified to fire first** so a silently-broken guard cannot pass.

---

## 11. Whole-layer A/B for every knob the stage introduced

`logs/ab_layer_knobs.py` → `ab_layer_knobs.txt`. Op-level probes rank candidates op by op; this
measures what the *layer* does with each choice, because an op-level win can be absorbed by the rest
of the graph and an op-level tie can still move the layer. Each arm is built fresh in the same
process on the same device with the same weights, three builds per arm.

| knob | arm | linear_attention decode | full_attention decode | prefill 2048 |
|---|---|---|---|---|
| `ccl` | **`auto`** (shipped) | **0.612–0.613 ms** | **0.503 ms** | 32.9–33.2 / 32.4–32.5 ms |
| `ccl` | `stack_sum` | 0.612–0.613 ms | 0.503–0.504 ms | flat |
| `ccl` | `all_reduce` | 0.622 ms | 0.510 ms | flat |
| `ccl` | `rs_ag` | 0.622 ms | 0.510 ms | flat |
| `geometry` | **multichip-retuned** (shipped) | **0.612–0.613 ms** | **0.503 ms** | flat |
| `geometry` | single-chip-inherited | 0.635 ms | 0.516–0.517 ms | flat |
| `routing` | **`select_matmul`** (shipped) | **0.612–0.613 ms** | **0.503–0.504 ms** | flat |
| `routing` | `gather` | 0.670 ms | 0.561–0.562 ms | flat |

* `auto` reproduces `stack_sum` exactly in the decode column, as it must at the batch-1 tile — the
  arms cross-check the switch rather than just measuring it.
* The retuned geometry is worth **22–23 us/step** on linear_attention and **13–14 us/step** on
  full_attention.
* `ttnn.gather` for the local-expert narrowing costs **57–59 us/step** against the one-hot selection
  matmul, and `test_routing_select_modes_agree` shows they are bit-equal, so this is pure op cost.
* Prefill is flat across every arm (32.2–33.7 ms): at 2048 tokens the collectives are 3–6% of the
  window and the decode geometry does not apply.

---

## 12. Performance result and where the time goes

Three arms, three processes, same weights, same harness (`logs/bench.py` →
`ab_single_vs_multichip.txt`). The middle arm is the control that separates "opening a 4-chip mesh
changed dispatch" from "the parallelisation helped".

| layer kind | phase | single-chip 1x1 | 1x4 replication control | multichip 1x4 | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.49 ms | 102.16 ms (+0.7%) | **33.40 ms** | **3.039x** | 76.0% |
| linear_attention | decode traced | 1.031 ms | 1.031 ms (+0.0%) | **0.613 ms** | **1.682x** | 42.0% |
| full_attention | prefill 2048 | 95.34 ms | 96.02 ms (+0.7%) | **32.31 ms** | **2.951x** | 73.8% |
| full_attention | decode traced | 0.827 ms | 0.827 ms (+0.0%) | **0.503 ms** | **1.644x** | 41.1% |

`test_multichip_beats_single_chip_traced_decode` gates the decode claim inside the suite itself.

`tt-perf-report` (`tracy/`, four separate captures — prefill and decode never share a Tracy session,
and `--active-experts` is passed so the routed `sparse_matmul` rows are modelled rather than skipped)
explains the 41–42% decode efficiency and confirms it is **not** a communication problem:

| | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 17.64% | 21.45% | 77.75% | 76.18% |
| `TopK` (router) | 8.95% | 9.40% | 0.15% | 0.15% |
| dense `Matmul` | 13.38% | 9.73% | 0.96% | 0.92% |
| **collectives** | **6.77%** | **9.35%** | **3.06%** | **6.27%** |
| all data movement | 8.45% | 11.17% | 3.29% | 6.27% |
| DRAM roofline | 6.4% (33 GB/s) | 5.9% (30 GB/s) | 36.0% (184 GB/s) | 36.1% (185 GB/s) |

The decode window is at **6% of the DRAM roofline** — batch-1 decode is launch- and latency-bound, so
dividing the work by four does not divide the time by four. That is inherited from the single-chip
stage, not caused by sharding, and it is the whole explanation for the gap between 74–77% prefill
efficiency and 41–42% decode efficiency. The routed experts remain the dominant term (76–78% of
prefill, 18–21% of decode) and are the right target for any future work; the collectives are not.

---

## 13. Watcher, and a hard tool limit

Watcher and profiler evidence were kept in separate runs throughout, per `$tt-device-usage`.

The first watcher attempt **failed outright**, and the failure is preserved rather than papered over:
`logs/watcher_pytest_eth_enabled.txt.gz`, 30 errors in 15.23 s, **zero tests executed**, every one at
`mesh_device` setup:

```
TT_FATAL: Program size (29040) too large for kernel config buffer (25600) on ACTIVE_ETH (assert.hpp:104)
```

With watcher instrumenting the ACTIVE_ETH cores, the 1D-fabric ERISC program does not fit the
Blackhole ACTIVE_ETH kernel config buffer. No environment knob grows that buffer, so the choice is
watcher coverage on the 110 Tensix worker cores per chip or none at all. `TT_METAL_WATCHER_DISABLE_ETH=1`
was added to `run_evidence.sh` with that reasoning inline, and the uninstrumented cores are the fabric
routers — stock `ttnn` 1D-fabric kernels this stage does not author. Every op this stage *does* author
runs on the covered Tensix cores.

Result: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_DISABLE_ETH=1` over the state-,
trace- and collective-critical subset — **27 passed, 3 skipped**. `watcher/census.py` partitions all
32 760 log lines into disjoint buckets, asserts the partition is complete (an unknown line kind lands
in `UNCLASSIFIED` and trips the assert rather than hiding in a catch-all), and reports **0
fatal-class matches** over 30 dumps.

The committed log contains no stack-usage lines, and the census prints "stack headroom: not reported
in this log" rather than a number. That is deliberate in `census.py`: watcher only emits a stack-usage
summary for dumps where firmware had already recorded a high-water mark, so a log can legitimately
contain none, and silence there means *not measured*, not *no overflow*. Earlier runs of the same
subset in this stage did produce them (minimum headroom 1332 bytes free over 220 detail lines); this
one's 30 dumps did not land on one. Recorded as part of README limitation 1 rather than quietly
dropped.

`TT_METAL_WATCHER_APPEND=1` is needed because watcher truncates its log on each device open and this
subset opens the mesh once per test; without it the committed log would hold only the last test's
session.

---

## 14. Hardware discipline

* Every device-facing command was run one at a time with the mesh otherwise idle.
* Watcher runs and Tracy/profiler runs were never combined in one process; `run_evidence.sh` orders
  them so watcher is step 6, last and alone.
* No hang, no ARC/ERISC fault, no Ethernet-link loss and **no device reset** was needed at any point
  in this stage. `tt-smi -ls --local` reported all four `p300c` chips before and after.
* `tools/tt-triage.py` was therefore not needed, and `$autofix` was not needed: no failure in this
  stage was mysterious. The two that occurred — the ACTIVE_ETH kernel-buffer overflow (§13) and the
  `shared_down` geometry gap (§7) — were both diagnosed directly from the failing command's own
  output and the committed sweep, and both are recorded above with their evidence.

---

## 15. Evidence regeneration

Everything in this directory is produced by one script, in the order the hardware discipline
requires:

```bash
bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/run_evidence.sh
```

Steps: `suite` → `bench` → `ab` → `probes` → `tracy` → `watcher`. Selectable with
`STEPS="..."`. Large text logs and every CSV are committed gzipped (the repo's pre-commit hook
rejects files over 500 KB and `.gitignore` carries a blanket `*.csv`); every generator reads
`foo.ext` or `foo.ext.gz` transparently.

Three full sweeps were run in this stage, all on 2026-08-14 UTC: 17:50–18:12, 18:26–18:48 and
18:52–19:14. Each one followed a source change and each regenerated every artifact, so the committed
artifacts are the third and were produced by exactly the bytes that were committed — including the
formatting the repo's `black`/`isort`/`autoflake` hooks apply, which were run *before* the final
sweep rather than after it. Running the sweep three times is also what produced the run-to-run
stability evidence in §7: the retune decisions hold in all three, and the near-tie deltas do not.

The one-line manual command used to confirm the ETH fix before wiring it into the script was:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/ornith_ai_ornith_1_0_35b/tests/test_multichip_decoder.py -v -p no:randomly \
  -k "kv_cache_is_local"
```

---

## 16. Review and checkpoint

Recorded below as the stage closes: `$stage-review` verdicts and the local checkpoint commit SHAs.
