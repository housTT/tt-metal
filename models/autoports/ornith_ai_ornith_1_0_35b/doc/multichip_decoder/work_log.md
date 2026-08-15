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
layer 0 (linear_attention)  prefill 2048  101.56 ms      decode (traced)  1.031 ms
layer 3 (full_attention)    prefill 2048   95.37 ms      decode (traced)  0.827 ms
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
| decode (batch 1, 32 rows) | 22.10 | 22.09 | 14.38 | 27.00 | **17.17** |
| 64 rows | 26.98 | 26.97 | 16.96 | 31.69 | 26.38 |
| 96 rows | 29.12 | 29.14 | 17.21 | 35.67 | 35.50 |
| 128 rows | 33.21 | 33.23 | 19.39 | 39.94 | 43.71 |
| 256 rows | 49.11 | 49.17 | 27.60 | 57.54 | 77.28 |
| 512 rows | 76.44 | 76.26 | 44.18 | 92.80 | 125.32 |
| decode batch 32 (1024 rows) | 135.04 | 135.35 | 77.57 | 157.78 | 233.20 |
| prefill 2048 | 196.77 | 196.75 | 104.15 | 267.00 | 447.19 |

These are the **traced** rows. `probe_ccl.txt` also carries an `eager` column, and several of its
rows favour the line fabric — those are dominated by per-call dispatch rather than by the fabric, and
none of this stage's decisions reads them.

Five conclusions, all of which shaped the implementation:

1. **The ring beats the line at every single shape.** `FABRIC_1D_RING` is what the layer configures.
   `ttnn.Topology.Ring` is passed to the ops that still accept it; `ttnn.all_gather` marks both
   `topology` and `num_links` deprecated and ignored, so the `stack_sum` path takes the ring from the
   fabric config alone. `probe_ccl.py` uses the identical spelling, so these rows compare fabric
   configs, which is where the difference lives.
2. **`ttnn.all_reduce` and explicit `reduce_scatter` + `all_gather` are the same number to two
   decimal places at every shape.** The stable all-reduce lowers to exactly that pair. The `rs_ag`
   arm therefore exists to make that identity *checkable*, not as a separate candidate — and
   `ab_layer_knobs.txt` confirms it at the layer (0.621 vs 0.621 ms, 0.508 vs 0.508-0.509 ms).
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
   14.38 us against 22.10 us at the decode tile, i.e. a 7.7 us saving per collective and 15.4 us per
   layer. That is the entire budget a sharded residual has to pay for itself out of. §4 spends it.

The `decode_b32` row — 1024 physical rows for 32 useful ones — sat in this table from the first sweep
and its significance was missed until review round 1. §11b is that thread.

---

## 4. The residual contract: replicated, decided against the usual advice

The `$multichip` skill is explicit that a replicated residual must not be the only measured contract,
and that a `reduce_scatter → all_gather` round trip mostly recreates the communication a fused path
was meant to avoid. Both points are taken, and the sharded-residual family was priced before being
rejected rather than after.

The budget from §3 is **15.4 us per layer**. Against it, a sharded residual on *this* model must pay:

* **Both RMSNorms become distributed.** With the hidden dimension fractured across devices, a local
  RMSNorm computes the wrong statistics, so each norm becomes
  `rms_norm_pre_all_gather` → stats all-gather → `rms_norm_post_all_gather`. That is a **third and
  fourth** collective per layer, on the latency-critical path, to save part of one. Even at the
  smallest measured shape a stats all-gather is a fabric phase, and §3 row 1 shows a fabric phase at
  the decode tile is 17–22 us. Two of them exceed the 15.4 us saving on their own.
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

`tt-perf-report` retires the question: collectives are **6.79%** (linear_attention) and **9.37%**
(full_attention) of merged 4-device decode time, and 2.98% / 5.78% of prefill — and §12b shows part
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
quartered) and `attn_in` carries 557 056 B/device/layer of duplicated k/v rows — 256 extra weight
columns of 2048 rows at bfloat8_b, not 2048 columns
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
| `single` (all 256, unsharded, still gate-selected) | 256 | 512 | 668.72 @8 active | 1641.23 @162 active |
| **`ep` (expert-parallel)** | **64** | **512** | **198.75** @3 active | **701.31** @63 active |
| `tp` (intermediate-sharded) | 256 | 128 | 473.28 @8 active | 1297.71 @162 active |

Read like for like — each arm at *its own* representative active count, which is what the layer
actually runs — expert parallelism beats intermediate sharding by **2.4x at decode** and **1.85x at
prefill**. The two arms' active counts differ (162 vs 63) because that difference *is* what expert
parallelism buys; comparing them at a common count would compare something neither arm runs. An
earlier draft quoted the prefill `ep` row at 31 active against `tp` at 162 and claimed 3.0x, which is
a different operating point again — review round 1 caught that, and round 2 asked for each arm's
count to be stated, which the table now does.

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
  quoted as exact.) The `ep` rows show the cost as a function of local active count: 200.30 us at 1
  active rising to 236.82 us at 8, so the realised ~3.5 sits near the flat part of that curve.
* Every `num_experts`-wide intermediate — packed gate/up output, its two unpacking slices, the SwiGLU
  product, the scored activation, the down output, the expert reduction — becomes 4x narrower under
  EP. Under intermediate sharding only the `moe_intermediate`-wide ones shrink; the
  `[1, E, tokens, 2048]` down output, the largest, does not shrink at all.

The `single` arm is the "no expert parallelism" control, **not** a dense-execution control: it runs
all 256 experts' weights on one device but still with the gate's 8 active, which is what its `active`
column says. It prices EP at 3.4x on decode. No dense (all-64-active) decode arm was measured,
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
| `attn_in` | (32,2) → 33/2 | 30.92 | 110/8 | 17.91 | 13.01 | 0.02 | 0.11 | **retuned (110, 8)** |
| `gdn_in` | (110,2) → 110/2 | 48.90 | 33/8 | 20.43 | 28.47 | 0.08 | 0.18 | **retuned (110, 8)**, 20.53 |
| `shared_down` | (48,16) → 55/4 | 9.10 | 4/4 | 8.51 | 0.59 | 0.54 | 0.25 | **retuned (8, 4)**, 8.56 |
| `expert_select` | new role | — | 8/8 | 8.26 | — | 0.19 | 0.74 | **(8, 8)**, 8.26 |
| `o_proj` | (16,16) → 22/16 | 10.53 | 88/8 | 9.97 | 0.56 | 0.38 | 0.27 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.58 | 22/16 | 9.35 | 0.23 | 0.30 | 0.18 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.90 | 110/32 | 8.77 | 0.13 | 0.67 | 0.13 | inherited |
| `router` | (32,32) → 33/32 | 9.01 | 8/32 | 8.81 | 0.20 | 0.46 | 1.49 | inherited |

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
  single-measurement statistic. Re-running the whole sweep seven times over this stage showed that is
  not stable: `o_proj`'s delta moved between 0.00 and 0.56 us with the winning core count wandering
  between 22, 33 and 88, and `gdn_out`'s between 0.07 and 0.30. The honest threshold is the probe's
  repeatability on repeated identical configs (0.11–1.49 us depending on role and sweep), which is
  now a column. Under it the inherited roles are ties, and where a single sweep says otherwise —
  `o_proj` in the committed one — the cross-sweep instability is the reason the inherited entry
  stays rather than a reason to chase it.

* `shared_down` was **initially left inherited on that same reasoning, and that was wrong**: its
  realised point is 55 cores at `in0_block_w` 4, which came out 0.51-0.68 us behind the winner on
  every sweep against a 0.25-0.35 us repeatability — consistently outside the noise, unlike the rows
  above. TP=4 cuts its `K` from 512 to 128 (4 tiles), so a 55-core grid for a 4-tile `K` and a
  64-tile `N` is launch overhead rather than parallelism.

  It was first retuned to **4** cores, the winner on three sweeps. A fourth sweep read 4 cores at
  9.98-10.29 us against 8 cores' 8.55-8.77 — a 1.4 us flip, outside repeatability — while the 8-core
  point read 8.55-8.77 on every sweep. So it ships at **8**: the stable reading, never behind, and
  the one that does not depend on which sweep is committed. That instability is also why the
  `repeatability` column exists; judging a row against one sweep's spread is exactly what put this
  entry on the wrong side of the line twice.

  Honest accounting: the op-level win over the inherited entry is real and reproducible, but the
  **whole-layer** `geometry` arm moves by ≤0.002 ms for this row, inside the build-to-build spread.
  It ships because it is faster where it can be measured and never slower, and because the shipped
  table should match its own sweep — not as a claimed layer-level speedup. README limitation 3.

`test_decode_runs_the_multichip_program_configs` asserts that the retuned configs actually reach
`attn_in`/`gdn_in` and `shared_down` in a real decode step, checks them against
`MULTICHIP_DECODE_MATMUL_GEOMETRY` rather than a literal, and re-derives the shard-carry divisibility
rule. This is the guard the single-chip stage's review round 32 forced after a faster geometry was
found to be computing the wrong thing: a program config that does not reach the op it was swept for
is indistinguishable from one that does, in every measurement except this one.

---

## 7b. Re-sweeping the *routed sparse* matmul core rule — the stage's largest win

Found by review round 1, by following a flag the first draft of the README had misread: every
`SparseMatmul` row in all four profiler captures is marked `SLOW` (257/257 per prefill capture,
128/128 per decode capture), and the draft attributed those rows to the small dense projections
instead. They are the dominant op — 74–75% of prefill.

`OptimizedMoE` targets `clamp(bound // SPARSE_CORES_PER_ACTIVE[role], 8, 32)` cores for the routed
matmuls, where `bound` is `_active_expert_bound` = `min(num_experts_local, rows * top_k)`. The
divisor means "work per core, per active expert" and was calibrated at `E = 256`, where that bound
saturates at 256. Expert parallelism does **not** change `Nt` — `moe_intermediate_size` (512) and
`dim` (2048) are not sharded, so the op still produces 32 and 64 output tiles — but it cuts the
bound's ceiling to 64, shrinking the core target exactly where the available parallelism did not
change.

`MultichipMoE._sparse_cfg` scales the bound back by the same divisor, so the net rule is
`cores = clamp(bound, 8, 32)`. Read off the shipped constants:

| call | bound | `gate_up` realised, off → on | `down` realised, off → on |
|---|---|---|---|
| decode b=1 | 8 | 8 → 8 | 8 → 8 |
| decode b=2 | 16 | 8 → **16** | 8 → **16** |
| decode b=4 | 32 | 16 → **32** | 8 → **32** |
| decode b≥8 | 64 | 32 → 32 | 16 → **32** |
| prefill group | 64 | 32 → 32 | 16 → **32** |

**Round 1 got this table wrong in five places and round 2 caught it.** Round 1 recorded "16/8 → 32/32
at prefill, decode unchanged", in the README, the work log, two source docstrings and
`context_contract.json`. Both halves were wrong: at prefill only `down` moves (`gate_up`'s bound is
already 64, so the inherited rule already targets 32), and decode geometry changes at every batch
above 1. The error came from evaluating the rule at an assumed *active count* — the probe's
`--active` — instead of at `_active_expert_bound`, which is what the code computes. The prefill win
was real and measured; the mechanism recorded for it was not.

Two things now stop that recurring. `probe_sparse_matmul_local.py::shipped_choice` prints the
inherited and shipped **realised** core counts, derived from the same functions the layer calls, into
the artifact's header. And `test_sparse_cores_match_the_local_sweep` intercepts `ttnn.sparse_matmul`
at batch 1/2/4/8/32 and asserts the realised grid against a table transcribed from the policy — the
same guard the dense geometry has had since the stage opened.

`logs/probe_sparse_matmul_local.py` is the single-chip sweep's candidate ladder with `E` made a flag,
re-run at the per-device operating points under the same BFP4/LoFi policy. Best microseconds per
realised core count, and the winner:

| active | 4c | 8c | 16c | 32c | 64c | winner |
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

The `active ∈ {8, 16, 32, 41}` rows were added in round 2. Round 1 swept only 4 and 63, which left
the whole decode range between them unmeasured and made 63 — itself an over-count, see §12c — the
only prefill evidence. The ladder's winner now tracks the shipped rule across the range, and the
prefill decision is the same at the corrected 41 as at the old 63.

Both effects are measured at the **layer**, which is what decides:

* **prefill** — `ab_layer_knobs.txt` `sparse` arm: **33.00–33.35 → 29.03–29.22 ms** on
  linear_attention and **32.45–32.78 → 28.30–28.83 ms** on full_attention, i.e. about 4.2–4.7 ms a
  layer and the difference between a 3.0x and a 3.4x prefill speedup.
* **decode** — `probe_decode_batch.txt` `SPARSEB` rows, three builds per arm per batch: a tie at
  batch 1 by construction, then **+33 / +50 / +23 / +38 / +49 / +71 us** a step at batch
  2 / 4 / 8 / 13 / 16 / 32 on full_attention and **+0 / +29 / +16 / +28 / +39 / +64** on
  linear_attention.

That decode measurement is the one round 2 asked for, and it is worth recording that it came out the
*opposite* way to what round 2 predicted: the isolated ladder favours 8 cores at low active counts,
so a rescale that forces 16 or 32 there looked adverse. It is not, at any batch. The reason is that
the ladder holds the active count fixed while the layer's real per-device count grows with batch, and
the bound the rule keys off tracks that growth. It is a clean example of why the layer A/B decides
and the op ladder only shapes the candidate.

The general lesson is §2 item 5: this is the second inherited constant that was a function of a
dimension sharding changed, and the first (§7) was found by re-sweeping on principle while this one
was found only because a reviewer asked why the dominant op was flagged.

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

Results from `logs/pytest_full_suite.txt.gz` (**132 passed, 3 skipped**; the 3 skips are
`linear_attention` cases for KV-cache, page-table and ragged-position tests, which that layer kind
does not have):

| | n | min | max |
|---|---|---|---|
| vs single-chip baseline, prefill | 18 | 0.999913 | 0.999998 |
| vs single-chip baseline, decode | 16 | 0.999892 | 0.999993 |
| vs float32 HF golden (all cases) | 86 | 0.999851 | — |

The two baseline rows are unchanged to the last digit across all seven full evidence sweeps this
stage ran, including the four that changed shipped code.

Everything else the mesh has to prove:

* `test_weights_are_sharded_not_replicated` — every TP/EP weight's shards actually **differ**. A
  silently replicated weight would still look correct for the TP tensors and be wrong for the EP
  ones, and would make every speedup number meaningless. Six weights per layer kind are checked
  directly; the small constants (`conv_taps`, `A_neg`, `dt_bias`, `expert_select`) are covered only
  indirectly, by the baseline PCC.
* `test_expert_partition_is_disjoint_and_complete` — the 4 x 64 expert blocks rebuild the 256-expert
  checkpoint tensor **in order** (worst per-expert PCC 0.9935, which is bfloat4_b weight
  quantisation from the inherited precision policy, not a partition error).
* `test_batched_prefill_decode_pcc[1,4,13,32]` x `seq_len ∈ {192, 130}` — batch on the mesh up to
  the advertised bound, at a tile-aligned and a non-aligned prefill length. The test was **pinned at
  batch 4** until review round 1, while the README and two `context_contract` fields claimed 32; TP=4
  changes the per-device head counts that `nlp_create_qkv_heads_decode`, `paged_fused_update_cache`
  and `sdpa_decode` are all bounded by, so the claim was not merely undocumented but unverified. The
  `seq_len` parameter came from round 2, which found the same shape of gap one level down: batched
  non-aligned prefill is the only path where `CCL_COMPACT_ROWS` fires in **prefill**, and every
  prefill test was either batch 1 or tile-aligned. Both now pass on both layer kinds (batch 32:
  prefill 0.999937/0.999934, decode 0.999926/0.999919 on full_attention).
* `test_permuted_page_table` (shuffled block mapping, prefill 0.999914 / decode 0.999903),
  `test_batched_decode_ragged_positions` (4 users at distinct absolute positions 37/130/200/64 over
  disjoint per-user block spans, each against its own HF golden), `test_prefill_continuation`.
* `test_chunk_size_invariance` — a 6000-token prefill under internal chunk 2048 and 1024 agrees to
  PCC **1.000000**. Added in review round 1: the mesh had no correctness cross-check at all beyond
  8000 tokens, where the HF golden becomes intractable, and this is the same control the single-chip
  stage uses for its long-prefill path.
* `test_traced_decode_pcc` — warmed capture and replay, replay PCC 0.999872–0.999981, bitwise equal
  across devices on every replay.
* `test_sparse_cores_match_the_local_sweep` — the realised routed-matmul grid at batch 1/2/4/8/32,
  against a table transcribed from the shipped policy. Added in review round 2; §7b is why.
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

Every row is at **batch 1**. That is why the `sparse` arm ties in the decode columns here and not in
§7b: the rescale only changes decode geometry above batch 1, and `probe_decode_batch.txt` is where
its decode effect is measured. Round 2 found the batch-1-only reading being generalised into "decode
is unchanged", which is why that split is now stated everywhere the knob is described.

* `auto` reproduces `stack_sum` exactly in the decode column, as it must at the batch-1 tile — the
  arms cross-check the switch rather than just measuring it.
* The retuned dense geometry is worth **24 us/step** on linear_attention and **14 us/step** on
  full_attention.
* The rescaled sparse core rule is worth **4.2–4.7 ms** of prefill on both layer kinds (§7b), and
  16–71 us/step at decode batches above 1, which this table cannot see.
* `ttnn.gather` for the local-expert narrowing costs **58 us/step** against the one-hot selection
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

The `decode_b32` row in §3's table had priced the wasteful shape from the first sweep (135.04 us
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
| 1 | 32 | 0.502 / 0.611 | 0.502 / 0.611 | 0 / 0 us |
| 2 | 64 | 0.582 / 0.709 | 0.576 / 0.704 | +6 / +5 us |
| 4 | 128 | 0.637 / 0.818 | 0.641 / 0.822 | **−4 / −4 us** |
| 8 | 256 | 0.791 / 1.007 | 0.785 / 1.001 | +6 / +6 us |
| 13 | 416 | 1.627 / 2.068 | 1.609 / 2.053 | +18 / +15 us |
| 16 | 512 | 1.828 / 2.272 | 1.799 / 2.244 | +29 / +28 us |
| 32 | 1024 | 3.023 / 3.947 | 2.934 / 3.864 | **+89 / +83 us** |

At batch 1 the guard skips the fold entirely (the tensor is already one tile row), which is why the
headline batch-1 numbers are unchanged. Subtracting these from §3's per-shape collective costs gives
the fold's own cost: ~4 us at batch 2, ~20 at batch 4, ~26 at batch 8, ~31 from batch 16 up. The
saving grows faster than that, so the fold wins everywhere except batch 4, where the saving
(all_reduce at 128 rows minus stack_sum at 32, ~16 us) is just under the fold's ~20 us.

It ships always-on. A rule that excluded exactly 128 physical rows would be fitting to one point, and
the regression is 0.6% at one batch against 2.1–3.0% at the advertised bound. README limitation 4.

Correctness is covered by the same batch parametrization the round added:
`test_batched_prefill_decode_pcc[1,4,13,32]` passes on both layer kinds with the fold on. Round 2
then found the prefill half of the fold untested — it fires only when `b > 1` **and** `seq_len` is
not tile-aligned, and every prefill test was one or the other — so that test now also runs
`seq_len = 130`. The same probe was extended with the `SPARSEB` arm §7b needed, so both rounds'
batch-scaling questions are answered on one harness at one set of batches.

---

## 12. Performance result and where the time goes

Three arms, three processes, same weights, same harness (`logs/bench.py` →
`ab_single_vs_multichip.txt`). The middle arm is the control that separates "opening a 4-chip mesh
changed dispatch" from "the parallelisation helped".

| layer kind | phase | single-chip 1x1 | 1x4 replication control | multichip 1x4 | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.56 ms | 101.94 ms (+0.4%) | **29.29 ms** | **3.467x** | 86.7% |
| linear_attention | decode traced | 1.031 ms | 1.031 ms (+0.0%) | **0.612 ms** | **1.685x** | 42.1% |
| full_attention | prefill 2048 | 95.37 ms | 95.68 ms (+0.3%) | **28.14 ms** | **3.389x** | 84.7% |
| full_attention | decode traced | 0.827 ms | 0.827 ms (+0.0%) | **0.502 ms** | **1.647x** | 41.2% |

`test_multichip_beats_single_chip_traced_decode` gates the decode claim inside the suite itself.

`tt-perf-report` (`tracy/`, four separate captures — prefill and decode never share a Tracy session,
and `--active-experts` is passed so the routed `sparse_matmul` rows are modelled rather than skipped)
explains the 41–42% decode efficiency and confirms it is **not** a communication problem:

| | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 17.69% | 21.54% | 75.02% | 73.94% |
| `TopK` (router) | 8.98% | 9.43% | 0.17% | 0.18% |
| dense `Matmul` | 13.08% | 9.40% | 1.10% | 1.08% |
| **collectives** | **6.79%** | **9.37%** | **2.98%** | **5.78%** |
| all data movement | 8.48% | 11.20% | 3.25% | 5.78% |
| DRAM roofline | 6.5% (33 GB/s) | 5.9% (30 GB/s) | 27.1% (139 GB/s) | 27.3% (140 GB/s) |

The decode window is at **6% of the DRAM roofline** — batch-1 decode is launch- and latency-bound, so
dividing the work by four does not divide the time by four. That is inherited from the single-chip
stage, not caused by sharding, and it is the whole explanation for the gap between 85–87% prefill
efficiency and 41–42% decode efficiency. The routed experts remain the dominant term and were the
right target: §7b took 4.2–4.7 ms/layer off prefill.

The prefill roofline figures are at the **corrected** `--active-experts` input (§12c) and are not
comparable to the ones this stage quoted before round 2, which were computed at an input about 50%
too large.

---

## 12b. The block-float collective: an anomaly, and the null result that classified it

The prefill profile shows the layer's two collectives at wildly different cost for the same logical
shape — `tracy/full_attention/prefill_perf_report.txt` has the BF16 reduce-scatter at 114 us on 20
cores and the BFP8 one at **1465 us on 12 cores, 4.7%** of the whole prefill window. (The 5.27% in
the stacked report is the two rows summed; three different numbers for this one row appeared in the
round-1 documents, which round 2 flagged.) The only difference is the operand dtype: the MoE half
produces `bfloat8_b` under the inherited precision policy.

Round 1 flagged the prefill collective time as an unclassified anomaly (7–15x the isolated probe for
the same shape, and ~2x different between layer kinds). The way to test the offered explanation was
to remove the only variable that distinguishes the two collectives.

`CCL_CAST_BLOCKFLOAT` casts the operand up to `bfloat16` before the collective. Measured at the layer
(§11 `cast` arm): it **costs 5–6 us on every decode step and moves warmed prefill by nothing** — the
arms' three-build ranges overlap on both layer kinds. So it is off, and the null result is the
control:

```
Observed anomaly:  prefill ReduceScatter device time is 7-15x the isolated probe for the same
                   [1,1,2048,2048] shape, and ~2x different between layer kinds.
Evidence:          tracy/*/prefill_perf_report.summary.txt (114 us BF16 on 20 cores vs 1465 us BFP8
                   on 12, full_attention; 101 vs 643 us, linear_attention);
                   logs/probe_ccl.txt prefill_2048 rs_only_ring trace 104.15.
Affected path:     the second per-layer collective, prefill, both layer kinds.
Control:           decode reproduces the probe closely (AllGather ~14-21 us/op against a 17.17 us
                   probe), so the discrepancy is prefill-specific; and the `cast` arm removes the
                   block-float operand without moving the layer at all.
Investigation:     CCL_CAST_BLOCKFLOAT implemented and A/B'd at the layer, three builds per arm.
Resolution:        controlled, with the mechanism left open. The 1465 us is NOT data movement this
                   layer pays: removing the block-float operand changes that row's cost and not the
                   layer's. The remaining candidate is the collective's barrier absorbing whatever
                   the four devices did not finish together, attributed to the op that waits.
                   Round 1 supported that with the suite's "39-44 active experts of 64", and round 2
                   correctly pointed out that range is across *layer kinds*, not across devices
                   (linear_attention logs [44,44,44,44] — no imbalance at all). So the mechanism is
                   a candidate, not a finding; pinning it needs per-device op timelines from the
                   profiled run. Nothing rests on it: README section 5.4 quotes the collective share
                   as an upper bound on communication cost throughout, which the control justifies.
                   The knob is kept, off, because the null result *is* the control.
```

---

## 12c. `--active-experts`: the modelling input the profiler runs on

Round 2's third finding, and the one with the widest blast radius in the documents even though it
changed no shipped code.

`tt-perf-report` early-returns out of its advice for a `sparse_matmul` row whose `nnz` is
`std::nullopt`, so without `--active-experts` the two routed matmuls — 74–75% of the prefill window —
carry no Bound, no DRAM %, no FLOPs % and no advice at all. The flag therefore decides every prefill
utilisation figure this stage quotes.

`run_profiling.sh` passed **63** for prefill, justified as the expected distinct union
`64 * (1 - (1 - 1/64)^(32*top_k))`. That formula credits all `32 * 8 = 256` global draws to this
device. Only `256 / tp = 64` of them land here, so the correct expectation is
`64 * (1 - (1 - 1/64)^64)` = **40.6**, and the shipped value is now 41.

It is also measured rather than only modelled: `test_gate_selected_experts_not_dense` records the
per-device, per-group non-zero count directly — 44 on linear_attention, 39–40 on full_attention,
bracketing 40.6 — and the test now asserts on the prefill count as well as the decode one, so the
modelling input cannot drift from the measurement silently. The decode value, 4, was already
measured and is unchanged.

Two consequences, both taken:

* the prefill roofline is **27.1% / 27.3%**, not the 41% the round-1 documents recorded. Neither the
  old 36% nor the old 41% is comparable to it — both were computed at the inflated input — so this
  stage no longer claims a roofline *improvement* from §7b, only the wall-clock one, which is
  measured at the layer and needs no model at all;
* `probe_sparse_matmul_local.py` now sweeps `active ∈ {4, 8, 16, 32, 41, 63}`. The winner is 32 cores
  for both roles at 32, 41 and 63, so the geometry decision is the same at the corrected point as at
  the inflated one — which is the specific thing round 2 could not verify without hardware.

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
state-, trace- and collective-critical subset — **43 passed, 3 skipped**. `watcher/census.py`
partitions all 52 542 log lines into disjoint buckets, asserts the partition is complete (an unknown
line kind lands in `UNCLASSIFIED` and trips the assert rather than hiding in a catch-all), and reports
**0 fatal-class matches** over 48 dumps with minimum stack headroom 1332 bytes free over 40 detail
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

Seven full sweeps were run in this stage: five on 2026-08-14 UTC (17:50–18:12, 18:26–18:48,
18:52–19:14, 20:21–20:56, 21:06–21:41) and two spanning into 2026-08-15 (22:35–23:20 and
23:22–00:07). Each followed a source change and each regenerated every artifact, so the committed
artifacts are the seventh and were produced by exactly the bytes that were committed — including the
formatting the repo's `black`/`isort`/`autoflake` hooks apply, which were run *before* the final
sweep rather than after it. Running the sweep seven times is also what produced the run-to-run
stability evidence in §7: the retune decisions hold in all seven, the near-tie deltas do not, and one
of them (`shared_down` at 4 cores) flipped far enough on the sixth to change the shipped value.

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
| P1 — dominant `SparseMatmul` flagged `SLOW` in all four captures, misattributed in the README, geometry never re-swept at the per-device point | `probe_sparse_matmul_local.py` + `MultichipMoE._sparse_cfg`; 4.2–4.7 ms/layer off prefill, and README §5.4 corrected (§7b) |
| P2 — "dense all-expert 3.3x slower" is not what the `single` arm measures | restated: the arm is unsharded-but-gate-selected, so it prices EP, not density (§6.1) |
| P2 — EP-vs-TP prefill compared at different active counts | each arm now labelled with its own operating point (§6.1) |
| P2 — HF-golden PCC count wrong | recounted from the log by a script rather than by hand |
| P2 — "leaves ~9.4 GiB / ~26.7 GiB" were GB labelled GiB | corrected to 9.04 / 24.88 GiB (§8) |
| P2 — "the sweep's 16 and 32 rows fail to build" contradicted by the sweep's two `FAIL` rows | restated: cap 8 is the measured winner *and* the largest the shard carry allows (README §5.6) |
| anomaly — prefill collective time 7–15x the isolated probe | controlled by the `cast` null result (§12b) |
| `ttnn.all_gather` ignores `topology`/`num_links`, so "every collective runs Ring" is not literal | module docstring and README §2.1 now say what selects the ring and what ignores the argument |
| smaller: `test_ccl_modes_agree` value, "shuffled" ragged mapping, `3.54` in `run_profiling.sh`, `CLASSIFICATION.md` reference, conv-taps label, 262144-decode wording, router/`expert_select` "measured" | each corrected in place |
| gap — no traced-replay leak check | `test_traced_replay_does_not_leak`, 128 replays, 0 bytes growth |
| gap — no full-context correctness cross-check on the mesh | `test_chunk_size_invariance`, PCC 1.000000 at 6000 tokens |

Two of the three P1s produced real speedups and the third produced a capability that was claimed but
unverified.

### Round 2 — `more-work-needed`

A second independent reviewer, given round 1's findings as context and told not to treat them as
evidence that anything worked. It found that two of round 1's three fixes were right in effect and
wrong in explanation, which is a failure mode worth naming: a measured win can validate a change
without validating the story told about it.

| finding | resolution |
|---|---|
| P1 — the sparse rescale changes decode geometry at every batch ≥ 2, unmeasured, while four artifacts said "decode is unchanged" | `SPARSEB` arm added to `probe_decode_batch.py`, measured at batch 1–32 on both layer kinds. It is a **win** of 16–71 us/step everywhere above batch 1, not the regression the isolated ladder predicted. All four statements corrected, and `test_sparse_cores_match_the_local_sweep` now pins the realised grid so they cannot drift again (§7b) |
| P1 — the "inherited realises 16/8" attribution is wrong; the shipped inherited path realises 32/16 and only `down` moves at prefill | re-derived from `_active_expert_bound` rather than from an assumed active count, corrected in all five places, and the derivation is now printed into the artifact by `shipped_choice()` (§7b) |
| P2 — `--active-experts 63` contradicted by the stage's own measured 39–44 | the formula was missing the `/tp` division. Corrected to 41, cross-checked against the measurement, and the prefill rooflines restated at the corrected input; the sweep was extended to `active ∈ {4,8,16,32,41,63}` so the geometry decision is verified at the corrected point (§12c) |
| P2 — README §5.5's crossover sentence describes the pre-fold behaviour | restated in terms of the post-fold row count: decode always takes `stack_sum`, and the crossover is what makes the fold worth doing |
| P2 — batched non-aligned prefill is the only path where the fold fires in prefill, and nothing tested it | `test_batched_prefill_decode_pcc` now runs `seq_len ∈ {192, 130}` at every batch |
| the `1x1 output subblock` advisory count, the three different figures for the BFP8 collective row, "2048 extra columns", "the ring wins at every shape" (true only of the traced rows), the EP-vs-TP label, `census.py`'s remaining `CLASSIFICATION.md` comments, the artifact tree's `.gz` names | each corrected in place |
| `SPARSE_SCALE_CORES_BY_TP` was not `tp`-gated, so a 1-device build would not reproduce `OptimizedMoE` | gated on `self.tp > 1` |
| §5.8 cited "39–44 across devices" for the imbalance mechanism; that range is across layer kinds | the mechanism is now stated as an open candidate, not a finding, and README limitation 7 records it (§12b) |
| gap — nothing asserted the realised sparse core count | `test_sparse_cores_match_the_local_sweep` |
| gap — `o_proj`/`gdn_out` kept on a cross-sweep-instability claim not reproducible from the committed artifact | the `repeatability` column, which is computed from the committed file, is now the stated threshold; the cross-sweep history is context, not the argument |

### Checkpoint

Local checkpoints only; nothing is pushed. `.agents/skills/tt-device-usage/SKILL.md` and
`.agents/fast-models-fast-feedback.md` are dirty in the worktree but are **not** stage-owned and are
deliberately excluded from every commit.

| repo | branch | round | commit |
|---|---|---|---|
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 0 | `914163b92b1` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 1 | `e3915a376f0`, SHA record `ca6674393a1` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 2 | `80563b120e4` |
