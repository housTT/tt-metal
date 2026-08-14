# Ornith-1.0-35B — multichip decoder, work log

Companion to `README.md`. The README states what the stage delivers; this log states what was done,
in what order, what each decision was measured against, and what was tried and rejected.

Stage boundary: `models/autoports/ornith_ai_ornith_1_0_35b/tt/multichip_decoder.py`, its test file,
and this documentation directory. No full-model or vLLM work. `tt/optimized_decoder.py` is not
modified.

---

## 0. Starting point

The single-chip baseline is `tt/optimized_decoder.py`, closed by operator decision at commit
`b6d427d6c4e` after 32 review rounds. Its warmed numbers on one Blackhole `p300c`, from
`logs/ab_single_vs_multichip.txt` (re-measured in this stage's own harness, not copied):

```
layer 0 (linear_attention)  prefill 2048  101.49 ms      decode (traced)  1.030 ms
layer 3 (full_attention)    prefill 2048   95.43 ms      decode (traced)  0.827 ms
```

It was chosen as the baseline because it is the fastest correct single-chip implementation in the
tree and because its program configs, precision policy, sharded-norm decisions, sparse-matmul
geometry rules, paged-cache contract and trace-safety properties are all already tuned. The design
decision that follows from that is in §2 — and its limits are in §7 and §7b, which are the two places
those inherited tunings turned out **not** to transfer to the per-device shapes.

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
* **Ring fabric.** `FABRIC_1D_RING` before `ttnn.open_mesh_device`. Confirmed by measurement, not
  assumption — see §3.
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

What the subclass adds is exactly five things:

1. the local config (above);
2. setup-time weight sharding in `from_state_dict`, handing each device the slice its local config
   describes;
3. two collectives per layer, `_all_reduce`, one after the token mixer and one after the MoE;
4. a globally-correct router that emits **device-local** routing, `MultichipMoE.routing_weights`;
5. three retunings of inherited program-config *rules* whose calibration point EP or TP moved — the
   dense decode geometry (§7), the routed sparse-matmul core target (§7b), and the collective's
   activation layout at batch > 1 (§11b).

Everything else — precision policy, sharded norms, trace safety, the paged-cache contract — is
inherited rather than re-derived. `test_local_config_is_the_per_device_view` pins the config
transform and `test_local_config_rejects_an_indivisible_mesh` pins that a `tp` which does not divide
is a construction error rather than a silent rounding of the model.

Item 5 is the general lesson of this stage: subclassing correctly inherits the single-chip *values*,
and the values were fitted to shapes that sharding changed. Every inherited constant that is a
function of a sharded dimension had to be re-derived, and two of the three were found only after a
review round pointed at the profiler.

---

## 3. The collective topology table, measured before committing to a path

`logs/probe_ccl.py` → `probe_ccl.txt`. Every candidate spelling of the layer's collective, at every
activation shape the layer actually produces, **eager and inside a captured trace**, on the 4-device
mesh with `num_links=2`. Trace is what a decode step pays, so trace rows decide.

Traced microseconds:

| shape | `all_reduce` Ring | `rs_ag` Ring | `rs_only` Ring | `all_reduce` Linear | `stack_sum` |
|---|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.10 | 22.10 | 14.36 | 26.97 | **17.17** |
| 64 rows | 26.99 | 26.98 | 16.96 | 31.65 | 26.39 |
| 96 rows | 29.11 | 29.17 | 17.18 | 35.67 | 35.49 |
| 128 rows | 33.19 | 33.22 | 19.41 | 39.91 | 43.75 |
| 256 rows | 49.11 | 49.16 | 27.57 | 57.54 | 77.28 |
| 512 rows | 76.44 | 76.43 | 44.25 | 92.83 | 125.49 |
| decode batch 32 (1024 rows) | 135.05 | 135.13 | 77.61 | 157.78 | 233.10 |
| prefill 2048 | 196.73 | 196.78 | 104.16 | 267.13 | 447.35 |

Five conclusions, all of which shaped the implementation:

1. **The ring beats the line at every single shape.** `FABRIC_1D_RING` is what the layer configures.
   `ttnn.Topology.Ring` is passed to the ops that still accept it; `ttnn.all_gather` marks both
   `topology` and `num_links` deprecated and ignored, so the `stack_sum` path takes the ring from the
   fabric config alone. `probe_ccl.py` uses the identical spelling, so these rows compare fabric
   configs, which is where the difference lives.
2. **`ttnn.all_reduce` and explicit `reduce_scatter` + `all_gather` are the same number to two
   decimal places at every shape.** The stable all-reduce lowers to exactly that pair. The `rs_ag`
   arm therefore exists to make that identity *checkable*, not as a separate candidate — and
   `ab_layer_knobs.txt` confirms it at the layer (0.622 vs 0.622 ms, 0.510 vs 0.510 ms).
3. **`ttnn.experimental.all_reduce_async` refuses Blackhole DRAM inputs outright**, at the device-op
   level: `all_reduce_async_device_operation.cpp` — *"does not support blackhole dram as it does not
   use an accessor to get the noc address"*. Every shape FAILs identically. Recorded as a
   hardware-side refusal, not a slow arm, and it is why the tuned experimental-tier CCL is not in the
   shipped path.
4. **`stack_sum` (`all_gather` onto a new leading axis, then a local `ttnn.sum`) wins below 64
   rows** even though it moves 4x the bytes, because at that size both are latency-bound and it is
   one fabric phase instead of two. It loses by a widening margin from 96 rows up. Hence
   `CCL_MODE="auto"` with `CCL_STACK_SUM_MAX_ROWS = 64`.
5. **The sharded-residual family is priced here too.** `rs_only` is the reduce-scatter half alone:
   14.36 us against 22.10 us at the decode tile, i.e. a 7.7 us saving per collective and 15.5 us per
   layer. That is the entire budget a sharded residual has to pay for itself out of. §4 spends it.

The `decode_b32` row — 1024 physical rows for 32 useful ones — sat in this table from the first sweep
and its significance was missed until review round 1. §11b is that thread.

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

`tt-perf-report` retires the question: collectives are **6.78%** (linear_attention) and **9.35%**
(full_attention) of merged 4-device decode time, and 2.94% / 5.68% of prefill — and §12b shows part
of even that prefill figure is barrier wait rather than data movement. The parallelisation's whole
communication cost is under a tenth of the decode window.

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
| `single` (all 256, unsharded, still gate-selected) | 256 | 512 | 665.86 @8 active | 1638.94 @162 active |
| **`ep` (expert-parallel)** | **64** | **512** | **198.77** @3 active | **698.35** @63 active |
| `tp` (intermediate-sharded) | 256 | 128 | 472.69 @8 active | 1298.09 @162 active |

Read like for like — each arm at *its own* representative active count, which is what the layer
actually runs — expert parallelism beats intermediate sharding by **2.4x at decode** and **1.86x at
prefill**. (An earlier draft of this table quoted the prefill `ep` row at 31 active against `tp` at
162 and claimed 3.0x. That compared arms at different operating points; 63 is the per-device
expected distinct union for a 32-token group, is what `run_profiling.sh` passes as
`--active-experts`, and is consistent with the suite's measured 39–44 of 64. Review round 1 caught
it.)

The reason is structural, and the probe was built to expose it: `ttnn.sparse_matmul` loops once per
**active** expert and its parallelism is capped by the output tile count.

* Sharding the intermediate keeps all 8 loop iterations and cuts `Nt` from 32 to 8 tiles — it removes
  parallelism the op was already short of.
* Expert parallelism keeps the full `Nt` and cuts the loop count to a mean of 2 with an expected
  **maximum over the four devices of 3.512** at top-8, computed exactly in the probe header. (That
  figure was originally 3.538, from modelling the 8 draws as independent uniform assignments. The
  gate picks 8 **distinct** experts out of 256, so the per-device counts are multivariate
  *hyper*geometric; the multinomial shortcut overstates clustering. Corrected in
  `probe_expert_parallel.py::expected_max_load`, which now takes the block size and enumerates the
  hypergeometric exactly. The error was 0.026 experts and changed no decision, but the figure is
  quoted as exact.) The `ep` rows show the cost as a function of local active count: 199.51 us at 1
  active rising to 236.37 us at 8, so the realised ~3.5 sits near the flat part of that curve.
* Every `num_experts`-wide intermediate — packed gate/up output, its two unpacking slices, the SwiGLU
  product, the scored activation, the down output, the expert reduction — becomes 4x narrower under
  EP. Under intermediate sharding only the `moe_intermediate`-wide ones shrink; the
  `[1, E, tokens, 2048]` down output, the largest, does not shrink at all.

The `single` arm is the "no expert parallelism" control, **not** a dense-execution control: it runs
all 256 experts' weights on one device but still with the gate's 8 active, which is what its `active`
column says. It prices EP at 3.35x on decode. No dense (all-64-active) decode arm was measured,
because nothing in the shipped path can produce one — `test_gate_selected_experts_not_dense` shows
decode activates at most 4 of 64 per device.

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

`test_gate_selected_experts_not_dense` intercepts the sparsity tensors and records per-device
non-zero counts: **decode max 4 of 64 per device**, prefill 39–44 of 64 per 32-token group. Dense
would be 64.

---

## 7. Re-sweeping the *dense* decode matmul geometry at the per-device shapes

The single-chip geometry table was tuned at the unsharded widths and does not transfer: `mcast_in0`
streams the whole `K` through each core in `in0_block_w`-tile blocks, so the winning cap is a
function of how much `N` each core owns, and TP=4 cuts `N` by four on the two wide in-projections.

`logs/probe_dense_matmul.py` → `probe_dense_matmul.txt` sweeps 9 core targets x 6 caps for every
dense decode role at its **per-device** shape. Comparing each inherited entry at its *realised* grid
(`_decode_1d_matmul_config` lays a target out as `ceil(target/11)` rows of `min(11, target)` columns,
and clamps `in0_block_w` to `Kt`) against the local winner:

| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |
|---|---|---|---|---|---|---|---|---|
| `attn_in` | (32,2) → 33/2 | 30.90 | 110/8 | 17.91 | 12.99 | 0.44 | 0.18 | **retuned (110, 8)** |
| `gdn_in` | (110,2) → 110/2 | 48.94 | 33/8 | 20.40 | 28.54 | 0.01 | 0.09 | **retuned (110, 8)**, 20.49 |
| `shared_down` | (48,16) → 55/4 | 9.13 | 4/4 | 8.62 | **0.51** | 0.17 | 0.35 | **retuned (4, 4)** |
| `expert_select` | new role | — | 4/8 | 8.31 | — | 0.27 | 0.28 | **(8, 8)**, 8.31 |
| `o_proj` | (16,16) → 22/16 | 9.26 | 22/8 | 9.25 | 0.01 | 0.48 | 0.34 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.73 | 22/16 | 9.43 | 0.30 | 0.29 | 0.12 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.69 | 88/32 | 8.61 | 0.08 | 0.55 | 0.12 | inherited |
| `router` | (32,32) → 33/32 | 9.05 | 88/32 | 8.89 | 0.16 | 0.67 | 0.07 | inherited |

`repeatability` is the probe's own noise floor for that role, read out of the same file: several
distinct core *targets* collapse onto the same realised grid, so the file already contains
independent repeat measurements of identical configs, and this column is the widest disagreement
between such repeats. It is the threshold a delta has to clear to be a result at all — and adding it
is the direct fix for the mistake described below.

* The two wide in-projections are where the inherited table is badly wrong: the inherited cap of 2 is
  the **worst legal value** for both local shapes, costing a factor of 1.7 on `attn_in` and 2.4 on
  `gdn_in`. Together they are worth 22 us/step on linear_attention and 14 us/step on full_attention
  (§11 `geometry` arm).
* Four roles are ties and keep the inherited entry — but *how* that was established had to be fixed
  twice, and both fixes are worth recording because the first version of this table was wrong in a
  way that is easy to repeat.

  The first version compared each inherited entry against the local winner using the **requested**
  core target rather than the **realised** grid. Targets 16/24/32/48 realise as 22/33/33/55 cores and
  a 16-wide cap on a 4-tile `K` becomes 4. Comparing at the requested target compares points the
  shipped code never runs.

  The second version fixed that but judged the resulting deltas against the *winner's own spread*, a
  single-measurement statistic. Re-running the whole sweep five times over this stage showed that is
  not stable: `o_proj`'s delta moved between 0.00 and 0.30 us and `gdn_out`'s between 0.07 and 0.30
  us, with no consistent sign. The honest threshold is the probe's repeatability on repeated
  identical configs (0.07–0.35 us depending on role), which is now a column. Under it, the four
  inherited roles are ties in every sweep.

* `shared_down` was **initially left inherited on that same reasoning, and that was wrong**: its
  realised point is 55 cores at `in0_block_w` 4, which came out 0.58, 0.68, 0.56 and 0.51 us behind
  the winner on four sweeps against a 0.35 us repeatability — consistently outside the noise, unlike
  the four above. TP=4 cuts its `K` from 512 to 128 (4 tiles), so a 55-core grid for a 4-tile `K` and
  a 64-tile `N` is launch overhead rather than parallelism. Retuned to `(4, 4)`. Honest accounting:
  the op-level win is real and reproducible, but the **whole-layer** `geometry` arm moves by ≤0.002
  ms, inside the build-to-build spread. It ships because it is faster where it can be measured and
  never slower, and because the shipped table now matches its own sweep — not as a claimed
  layer-level speedup. README limitation 3.

`test_decode_runs_the_multichip_program_configs` asserts that the retuned configs actually reach
`attn_in`/`gdn_in` and `shared_down` in a real decode step, checks them against
`MULTICHIP_DECODE_MATMUL_GEOMETRY` rather than a literal, and re-derives the shard-carry divisibility
rule. This is the guard the single-chip stage's review round 32 forced after a faster geometry was
found to be computing the wrong thing: a program config that does not reach the op it was swept for
is indistinguishable from one that does, in every measurement except this one.

---

## 7b. Re-sweeping the *routed sparse* matmul core rule — the stage's largest win

Found by review round 1, by following a flag the first draft of the README had misread: every
`SparseMatmul` row in all four profiler captures is marked `SLOW` (128/128 per decode capture,
257/257 per prefill capture), and the draft attributed those `SLOW` rows to the small dense
projections instead. They are the dominant op — 74–75% of prefill.

`OptimizedMoE` targets `clamp(active // SPARSE_CORES_PER_ACTIVE[role], 8, 32)` cores for the routed
matmuls. The divisor means "work per core, per active expert" and was calibrated at `E = 256`, where
a 32-token prefill group activates ~162 experts. Expert parallelism does **not** change `Nt` —
`moe_intermediate_size` (512) and `dim` (2048) are not sharded, so the op still produces 32 and 64
output tiles — but it divides the *active* count by `tp`, to ~63 per device. Inherited unchanged,
that collapses the core target exactly where the available parallelism did not change:

```
# SHIPPED role=gate_up active=63 Nt=32 inherited_realised_cores=16 multichip_realised_cores=32
# SHIPPED role=down    active=63 Nt=64 inherited_realised_cores=8  multichip_realised_cores=32
# SHIPPED role=gate_up active=4  Nt=32 inherited_realised_cores=8  multichip_realised_cores=8
# SHIPPED role=down    active=4  Nt=64 inherited_realised_cores=8  multichip_realised_cores=8
```

`logs/probe_sparse_matmul_local.py` is the single-chip sweep's candidate ladder with `E` made a flag,
re-run at the per-device operating point under the same BFP4/LoFi policy. Best microseconds per
realised core count:

| | 8 cores | 16 cores | 32 cores | 64 cores |
|---|---|---|---|---|
| `gate_up`, active=4 | **62.4** | 64.2 | 70.0 | — |
| `down`, active=4 | **52.3** | 59.6 | 66.7 | 85.4 |
| `gate_up`, active=63 | 398.6 | 234.5 | **199.3** | — |
| `down`, active=63 | 235.8 | 144.8 | **105.1** | 124.8 |

So the inherited rule is already optimal at decode (both roles clamp to `SPARSE_MIN_CORES` = 8, which
is the winner) and is 2.0x/2.2x off the winner at prefill.

The fix is one number per role, and it is the same number the divisor already means: with `tp` times
fewer active experts per device, each core should be given `tp` times fewer of them.
`MultichipMoE._sparse_cfg` therefore scales the *bound* the parent's rule is applied to, rather than
re-declaring the rule — so the parent's clamping, its realised-core reduction and its `in0_block_w`
cap all still run exactly once and in the parent, and `tt/optimized_decoder.py` stays untouched.

Whole-layer effect (`ab_layer_knobs.txt` `sparse` arm), warmed 2048-token prefill:

```
linear_attention   32.96–33.14 ms  ->  28.94–29.41 ms
full_attention     32.20–32.88 ms  ->  28.18–28.67 ms
```

Decode is unchanged to the millisecond, exactly as the sweep predicts. This is a **4.0 ms per layer**
prefill win — the difference between a 2.9–3.1x and a 3.4–3.5x prefill speedup — and it also moved the
prefill DRAM roofline from 36% to 41%, i.e. it moved the machine measurably closer to bandwidth-bound.

The general lesson is §2 item 5: this is the second inherited constant that was a function of a
dimension sharding changed, and the first one (§7) was found by re-sweeping on principle while this
one was found only because a reviewer asked why the dominant op was flagged.

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
non-aligned support are all unchanged; this stage strictly *enlarges* the per-device budget. The
1.40x → 4.62x change is the reason the multichip decoder is the right layer stack for the full model:
the single-chip path leaves 9.04 GiB for the untied embedding and `lm_head` (1 017 118 720 B each)
plus the trace region and every activation buffer, and the multichip path leaves 24.88 GiB per device
for the same.

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
* **the advertised context** — `test_full_context_prefill_and_decode` at 262144 (prefill only: it is
  the last legal prefill, so there is no slot left to decode into) and at 262141 (prefill and
  decode).

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

Results from `logs/pytest_full_suite.txt.gz` (**114 passed, 3 skipped**; the 3 skips are
`linear_attention` cases for KV-cache, page-table and ragged-position tests, which that layer kind
does not have):

| | n | min | max |
|---|---|---|---|
| vs single-chip baseline, prefill | 18 | 0.999913 | 0.999998 |
| vs single-chip baseline, decode | 16 | 0.999892 | 0.999993 |
| vs float32 HF golden (all cases) | 70 | 0.999851 | — |

The two baseline rows are unchanged to the last digit across all five full evidence sweeps this stage
ran, including the two that changed shipped code.

Everything else the mesh has to prove:

* `test_weights_are_sharded_not_replicated` — every TP/EP weight's shards actually **differ**. A
  silently replicated weight would still look correct for the TP tensors and be wrong for the EP
  ones, and would make every speedup number meaningless. Six weights per layer kind are checked
  directly; the small constants (`conv_taps`, `A_neg`, `dt_bias`, `expert_select`) are covered only
  indirectly, by the baseline PCC.
* `test_expert_partition_is_disjoint_and_complete` — the 4 x 64 expert blocks rebuild the 256-expert
  checkpoint tensor **in order** (worst per-expert PCC 0.9935, which is bfloat4_b weight
  quantisation from the inherited precision policy, not a partition error).
* `test_batched_prefill_decode_pcc[1,4,13,32]` — batch on the mesh up to the advertised bound. This
  test was **pinned at batch 4** until review round 1, while the README and two `context_contract`
  fields claimed 32; TP=4 changes the per-device head counts that `nlp_create_qkv_heads_decode`,
  `paged_fused_update_cache` and `sdpa_decode` are all bounded by, so the claim was not merely
  undocumented but unverified. Batch 32 runs and is correct (prefill 0.999938, decode 0.999924 on
  full_attention).
* `test_permuted_page_table` (shuffled block mapping, prefill 0.999914 / decode 0.999903),
  `test_batched_decode_ragged_positions` (4 users at distinct absolute positions 37/130/200/64 over
  disjoint per-user block spans, each against its own HF golden), `test_prefill_continuation`.
* `test_chunk_size_invariance` — a 6000-token prefill under internal chunk 2048 and 1024 agrees to
  PCC **1.000000**. Added in review round 1: the mesh had no correctness cross-check at all beyond
  8000 tokens, where the HF golden becomes intractable, and this is the same control the single-chip
  stage uses for its long-prefill path.
* `test_traced_decode_pcc` — warmed capture and replay, replay PCC 0.999872–0.999981, bitwise equal
  across devices on every replay.
* `test_traced_replay_does_not_leak` — 128 replays, **DRAM allocated growth 0 bytes**. Added in
  review round 1: `test_repeated_run_stress` bounds the *untraced* path, and a CCL semaphore or
  persistent buffer allocated per replay inside a captured trace is invisible to both PCC and that
  test.
* `test_determinism_repeated_inputs` — 3/3 runs bit-identical **on all 4 devices**.
* `test_repeated_run_stress` — 12 prefill + 4-step-decode cycles over `[96, 130, 257]`, repeats
  bit-identical, **DRAM growth 0 bytes** on both layer kinds.
* `test_ccl_modes_agree` and `test_routing_select_modes_agree` — the rejected arms agree with the
  shipped one (PCC 1.000000 / 1.000000 / 0.999999–0.999997, and bit-equal respectively), so the A/B
  numbers in §3 and §11 compare implementations that compute the same thing.
* `test_no_host_fallback_in_forward` — prefill and decode clean for
  `ttnn.{from_torch, to_torch, as_tensor, from_device, to_device}` and **all** `torch` ops, on both
  layer kinds, with **both guards verified to fire first** so a silently-broken guard cannot pass.

---

## 11. Whole-layer A/B for every knob the stage introduced

`logs/ab_layer_knobs.py` → `ab_layer_knobs.txt`. Op-level probes rank candidates op by op; this
measures what the *layer* does with each choice, because an op-level win can be absorbed by the rest
of the graph and an op-level tie can still move the layer. Each arm is built fresh in the same
process on the same device with the same weights, three builds per arm. All values in ms.

| knob | arm | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|---|
| `ccl` | **`auto`** (shipped) | **0.612–0.613** | **0.503** | 29.07–29.21 | 28.07–28.33 |
| `ccl` | `stack_sum` | 0.612–0.613 | 0.503 | 29.34–29.49 | 28.48–28.65 |
| `ccl` | `all_reduce` | 0.622 | 0.510 | 28.92–29.00 | 28.30–28.76 |
| `ccl` | `rs_ag` | 0.622 | 0.510 | 28.94–29.16 | 28.21–28.43 |
| `geometry` | **multichip-retuned** (shipped) | **0.613** | **0.503** | 28.89–29.26 | 28.21–28.54 |
| `geometry` | single-chip-inherited | 0.635 | 0.517 | 29.02–29.11 | 28.30–28.43 |
| `routing` | **`select_matmul`** (shipped) | **0.613** | **0.503** | 28.97–29.25 | 28.08–28.68 |
| `routing` | `gather` | 0.670–0.671 | 0.561 | 29.36–29.42 | 28.64–28.84 |
| `sparse` | **tp-rescaled** (shipped) | **0.612–0.613** | **0.503** | **28.94–29.41** | **28.18–28.67** |
| `sparse` | single-chip-inherited | 0.612–0.613 | 0.503–0.504 | 32.96–33.14 | 32.20–32.88 |
| `cast` | **block-float** (shipped) | **0.613** | **0.503** | 28.96–29.30 | 28.17–28.33 |
| `cast` | bf16 | 0.617–0.618 | 0.509 | 29.00–29.40 | 28.46–28.68 |

* `auto` reproduces `stack_sum` exactly in the decode column, as it must at the batch-1 tile — the
  arms cross-check the switch rather than just measuring it.
* The retuned dense geometry is worth **22 us/step** on linear_attention and **14 us/step** on
  full_attention.
* The rescaled sparse core rule is worth **4.0 ms** of prefill on both layer kinds (§7b) and nothing
  on decode, exactly as its sweep predicts.
* `ttnn.gather` for the local-expert narrowing costs **57–58 us/step** against the one-hot selection
  matmul, and `test_routing_select_modes_agree` shows they are bit-equal, so this is pure op cost.
* The `cast` arm is a **rejected** candidate and a control; §12b.

---

## 11b. Decode at batch > 1: the collective was moving 32x the rows it needed to

Review round 1's second finding. Every performance number in this stage was batch 1, where the token
mixer's decode output `[1, 1, dim]` is exactly one tile row. A tile's row axis is the *sequence*
axis, so at batch `b` that tensor occupies `b` tile rows — `b * 32` physical rows — of which `b` carry
data. The layer's **second** collective never had this problem: `_block` already reshapes to
`[1, 1, tokens, dim]` for the MoE. So in one batch-32 decode forward the same logical reduction ran
once on 1024 physical rows and once on 32.

The `decode_b32` row in §3's table had priced the wasteful shape from the first sweep (135.05 us
against 17.17 us for `stack_sum` at 32 rows) without the connection being made.

`logs/probe_decode_batch.py` closes it, and measures the shapes rather than inferring them — it wraps
`_all_reduce` and prints what each call site is handed:

```
SHAPE full_attention batch=32 site=mixer shape=[32, 1, 2048] padded=[32, 32, 2048]
      physical_rows=1024 useful_rows=32 waste=32.0x
SHAPE full_attention batch=32 site=moe   shape=[1, 1, 32, 2048] padded=[1, 1, 32, 2048]
      physical_rows=32 useful_rows=32 waste=1.0x
```

`CCL_COMPACT_ROWS` folds `[b, t, dim]` to `[1, 1, b*t, dim]` around the collective and views it back.
Traced decode, three builds per arm, full_attention / linear_attention:

| batch | mixer physical rows | off | on | delta |
|---|---|---|---|---|
| 1 | 32 | 0.503 / 0.613 | 0.503 / 0.612 | 0 / +1 us |
| 2 | 64 | 0.583 / 0.711 | 0.577 / 0.705 | +6 / +6 us |
| 4 | 128 | 0.639 / 0.820 | 0.642 / 0.823 | **−3 / −3 us** |
| 8 | 256 | 0.792 / 1.008 | 0.786 / 1.002 | +6 / +6 us |
| 13 | 416 | 1.628 / 2.070 | 1.610 / 2.054 | +18 / +16 us |
| 16 | 512 | 1.829 / 2.273 | 1.800 / 2.246 | +29 / +27 us |
| 32 | 1024 | 3.024 / 3.949 | 2.935 / 3.866 | **+89 / +83 us** |

At batch 1 the guard skips the fold entirely (the tensor is already one tile row), which is why the
headline batch-1 numbers are unchanged. Subtracting these from §3's per-shape collective costs gives
the fold's own cost: ~3 us at batch 2, ~20 at batch 4, ~27 at batch 8, ~31 from batch 16 up. The
saving grows faster than that, so the fold wins everywhere except batch 4, where the saving
(all_reduce at 128 rows minus stack_sum at 32, ~16 us) is just under the fold's ~20 us.

It ships always-on. A rule that excluded exactly 128 physical rows would be fitting to one point, and
the regression is 0.4% at one batch against 2.3–2.9% at the advertised bound. README limitation 4.

Correctness is covered by the same batch parametrization the round added:
`test_batched_prefill_decode_pcc[1,4,13,32]` passes on both layer kinds with the fold on.

---

## 12. Performance result and where the time goes

Three arms, three processes, same weights, same harness (`logs/bench.py` →
`ab_single_vs_multichip.txt`). The middle arm is the control that separates "opening a 4-chip mesh
changed dispatch" from "the parallelisation helped".

| layer kind | phase | single-chip 1x1 | 1x4 replication control | multichip 1x4 | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.49 ms | 101.88 ms (+0.4%) | **29.16 ms** | **3.480x** | 87.0% |
| linear_attention | decode traced | 1.030 ms | 1.031 ms (+0.1%) | **0.613 ms** | **1.680x** | 42.0% |
| full_attention | prefill 2048 | 95.43 ms | 95.58 ms (+0.2%) | **28.15 ms** | **3.390x** | 84.8% |
| full_attention | decode traced | 0.827 ms | 0.827 ms (+0.0%) | **0.503 ms** | **1.644x** | 41.1% |

`test_multichip_beats_single_chip_traced_decode` gates the decode claim inside the suite itself.

`tt-perf-report` (`tracy/`, four separate captures — prefill and decode never share a Tracy session,
and `--active-experts` is passed so the routed `sparse_matmul` rows are modelled rather than skipped)
explains the 41–42% decode efficiency and confirms it is **not** a communication problem:

| | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 17.64% | 21.46% | 75.03% | 74.03% |
| `TopK` (router) | 8.94% | 9.40% | 0.17% | 0.18% |
| dense `Matmul` | 13.38% | 9.71% | 1.11% | 1.07% |
| **collectives** | **6.78%** | **9.35%** | **2.94%** | **5.68%** |
| all data movement | 8.47% | 11.17% | 3.21% | 5.68% |
| DRAM roofline | 6.4% (33 GB/s) | 5.9% (30 GB/s) | 41.4% (212 GB/s) | 41.7% (214 GB/s) |

The decode window is at **6% of the DRAM roofline** — batch-1 decode is launch- and latency-bound, so
dividing the work by four does not divide the time by four. That is inherited from the single-chip
stage, not caused by sharding, and it is the whole explanation for the gap between 85–87% prefill
efficiency and 41–42% decode efficiency. The routed experts remain the dominant term and were the
right target: §7b took 4.0 ms/layer off prefill and moved the prefill roofline from 36% to 41%.

---

## 12b. The block-float collective: an anomaly, and the null result that classified it

The prefill profile shows the layer's two collectives at wildly different cost for the same logical
shape — `tracy/full_attention/prefill_perf_report.txt` has the BF16 reduce-scatter at ~100 us on 20
cores and the BFP8 one at ~1500 us on 12 cores, 5.2% of the whole prefill window. The only difference
is the operand dtype: the MoE half produces `bfloat8_b` under the inherited precision policy.

Review round 1 flagged the prefill collective time as an unclassified anomaly (7–15x the isolated
probe for the same shape, and 2.1x different between layer kinds). The hypothesis it offered — the
barrier absorbing EP load imbalance — was right, and the way to test it was to remove the only
variable that distinguishes the two collectives.

`CCL_CAST_BLOCKFLOAT` casts the operand up to `bfloat16` before the collective. Measured at the layer
(§11 `cast` arm): it **costs 5–6 us on every decode step and moves warmed prefill by nothing** — the
arms' three-build ranges overlap on both layer kinds. So it is off, and the null result is the
control:

```
Observed anomaly:  prefill ReduceScatter device time is 7-15x the isolated probe for the same
                   [1,1,2048,2048] shape, and 2.1x different between layer kinds.
Evidence:          tracy/*/prefill_perf_report.txt (~100 us BF16 on 20 cores vs ~1500 us BFP8 on 12);
                   logs/probe_ccl.txt prefill_2048 rs_only_ring trace 104.16.
Affected path:     the second per-layer collective, prefill, both layer kinds.
Control:           decode reproduces the probe closely (AllGather ~14-18 us/op against a 17.17 us
                   probe), so the discrepancy is prefill-specific; and the `cast` arm removes the
                   block-float operand without moving the layer at all.
Likely subsystem:  the collective's barrier absorbing per-device expert-load imbalance under EP.
                   The suite logs 39-44 active experts of 64 across devices for one prefill group.
Investigation:     CCL_CAST_BLOCKFLOAT implemented and A/B'd at the layer, three builds per arm.
Resolution:        controlled. The ~1500 us is not data movement this layer pays: removing the
                   block-float operand changes that row's cost and not the layer's. Recorded in
                   README §5.4 as the reason the prefill CCL percentage is an upper bound on
                   communication cost rather than a measurement of it. The knob is kept, off, because
                   the null result *is* the control and a future dtype policy could move the boundary.
```

---

## 13. Watcher, and a hard tool limit

Watcher and profiler evidence were kept in separate runs throughout, per `$tt-device-usage`.

The first watcher attempt **failed outright**, and the failure is preserved rather than papered over:
`logs/watcher_pytest_eth_enabled.txt.gz` — every selected test errors in ~15 s, **zero tests
executed**, all at `mesh_device` setup:

```
TT_FATAL: Program size (29040) too large for kernel config buffer (25600) on ACTIVE_ETH (assert.hpp:104)
```

With watcher instrumenting the ACTIVE_ETH cores, the 1D-fabric ERISC program does not fit the
Blackhole ACTIVE_ETH kernel config buffer. No environment knob grows that buffer, so the choice is
watcher coverage on the 110 Tensix worker cores per chip or none at all.
`TT_METAL_WATCHER_DISABLE_ETH=1` was added to `run_evidence.sh` with that reasoning inline, and the
uninstrumented cores are the fabric routers — stock `ttnn` 1D-fabric kernels this stage does not
author. Every op this stage *does* author runs on the covered Tensix cores.

Result: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_DISABLE_ETH=1` over the
state-, trace- and collective-critical subset — **35 passed, 3 skipped**. `watcher/census.py`
partitions all 43 875 log lines into disjoint buckets, asserts the partition is complete (an unknown
line kind lands in `UNCLASSIFIED` and trips the assert rather than hiding in a catch-all), and reports
**0 fatal-class matches** over 40 dumps with minimum stack headroom 1292 bytes free over 40 detail
lines.

`TT_METAL_WATCHER_APPEND=1` is needed because watcher truncates its log on each device open and this
subset opens the mesh once per test; without it the committed log would hold only the last test's
session.

One census behaviour is worth naming because it looks like a gap and is not: watcher only emits a
stack-usage summary for dumps where firmware had already recorded a high-water mark, so a log can
legitimately contain none. `census.py` prints "stack headroom: not reported in this log" in that case
rather than a reassuring number — silence there means *not measured*, not *no overflow*. One
intermediate sweep in this stage produced such a log; the committed one does report headroom.

---

## 14. Hardware discipline

* Every device-facing command was run one at a time with the mesh otherwise idle.
* Watcher runs and Tracy/profiler runs were never combined in one process; `run_evidence.sh` orders
  them so watcher is step 6, last and alone.
* No hang, no ARC/ERISC fault, no Ethernet-link loss and **no device reset** was needed at any point
  in this stage. `tt-smi -ls --local` reported all four `p300c` chips before and after.
* `tools/tt-triage.py` was therefore not needed, and `$autofix` was not needed: no failure in this
  stage was mysterious. The three that occurred — the ACTIVE_ETH kernel-buffer overflow (§13), an
  `L1_SMALL` exhaustion inside `probe_decode_batch.py` when it built ~50 decoders in one process
  (CCL semaphores are not reclaimed while the mesh stays open; fixed by running one process per
  batch, which `run_evidence.sh` now does), and the `shared_down` geometry gap (§7) — were each
  diagnosed directly from the failing command's own output.

---

## 15. Evidence regeneration

Everything in this directory is produced by one script, in the order the hardware discipline
requires:

```bash
bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/run_evidence.sh
```

Steps: `suite` → `bench` → `ab` → `probes` → `tracy` → `watcher`. Selectable with `STEPS="..."`.
Large text logs and every CSV are committed gzipped (the repo's pre-commit hook rejects files over
500 KB and `.gitignore` carries a blanket `*.csv`); every generator reads `foo.ext` or `foo.ext.gz`
transparently. The repo's trailing-whitespace and end-of-file hooks rewrite the committed
`*_perf_report.txt` and `*.summary.txt` files; that is cosmetic and the numbers are untouched.

Five full sweeps were run in this stage, all on 2026-08-14 UTC: 17:50–18:12, 18:26–18:48, 18:52–19:14,
20:21–20:56 and 21:06–21:41. Each followed a source change and each regenerated every artifact, so
the committed artifacts are the fifth and were produced by exactly the bytes that were committed —
including the formatting the repo's `black`/`isort`/`autoflake` hooks apply, which were run *before*
the final sweep rather than after it. Running the sweep five times is also what produced the
run-to-run stability evidence in §7: the retune decisions hold in all five, and the near-tie deltas
do not.

The one-line manual command used to confirm the ETH fix before wiring it into the script was:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/ornith_ai_ornith_1_0_35b/tests/test_multichip_decoder.py -v -p no:randomly \
  -k "kv_cache_is_local"
```

---

## 16. Review rounds and checkpoint

### Round 1 — `more-work-needed`

An independent `$stage-review` subagent, read-only, against the goal contract and the committed
artifacts. Three P1 findings, five P2, and a set of smaller concerns. All were taken; none was
dismissed. What each one turned into:

| finding | resolution |
|---|---|
| P1 — batch-32 claimed in three documents, `test_batched_prefill_decode_pcc` pinned at batch 4 | parametrized at `[1, 4, 13, 32]` and run on the mesh; batch 32 is correct (§10). The claim was unverified, not merely undocumented |
| P1 — decode at batch > 1 all-reduces a 32x tile-padded activation | `CCL_COMPACT_ROWS`, measured across batch 1–32; +83–89 us/layer at the advertised bound (§11b) |
| P1 — dominant `SparseMatmul` flagged `SLOW` in all four captures, misattributed in the README, geometry never re-swept at the per-device point | `probe_sparse_matmul_local.py` + `MultichipMoE._sparse_cfg`; **4.0 ms/layer** off prefill, and README §5.4 corrected (§7b) |
| P2 — "dense all-expert 3.3x slower" is not what the `single` arm measures | restated: the arm is unsharded-but-gate-selected, so it prices EP, not density (§6.1) |
| P2 — EP-vs-TP prefill compared at different active counts | requoted like-for-like at 63 active: 1.86x, not 3.0x (§6.1) |
| P2 — HF-golden PCC count wrong | recounted from the log by a script rather than by hand |
| P2 — "leaves ~9.4 GiB / ~26.7 GiB" were GB labelled GiB | corrected to 9.04 / 24.88 GiB (§8) |
| P2 — "the sweep's 16 and 32 rows fail to build" contradicted by the sweep's two `FAIL` rows | restated: cap 8 is the measured winner *and* the largest the shard carry allows (README §5.6) |
| anomaly — prefill collective time 7–15x the isolated probe | controlled by the `cast` null result (§12b) |
| `ttnn.all_gather` ignores `topology`/`num_links`, so "every collective runs Ring" is not literal | module docstring and README §2.1 now say what selects the ring and what ignores the argument |
| smaller: `test_ccl_modes_agree` value, "shuffled" ragged mapping, `3.54` in `run_profiling.sh`, `CLASSIFICATION.md` reference, conv-taps label, 262144-decode wording, router/`expert_select` "measured" | each corrected in place |
| gap — no traced-replay leak check | `test_traced_replay_does_not_leak`, 128 replays, 0 bytes growth |
| gap — no full-context correctness cross-check on the mesh | `test_chunk_size_invariance`, PCC 1.000000 at 6000 tokens |

Two of the three P1s produced real speedups (4.0 ms/layer prefill, 83–89 us/layer decode at batch 32)
and the third produced a capability that was claimed but unverified. The review was worth more than
the stage's own optimization pass.

### Checkpoint

Local checkpoints only; nothing is pushed. `.agents/skills/tt-device-usage/SKILL.md` and
`.agents/fast-models-fast-feedback.md` are dirty in the worktree but are **not** stage-owned and are
deliberately excluded from every commit.

| repo | branch | round | commit |
|---|---|---|---|
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 0 | `914163b92b1` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 1 | recorded below after the round-1 commit |
