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

§12's generated table carries them as the `single-chip` column, re-measured by every sweep, and they
are not copied into prose here for the reason §16 round 3 records: a hand-transcribed figure goes
stale on the next re-measurement and nothing notices.

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
  against a single-chip worst-case layer of 839553028 B at the full advertised context. Nothing about the
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

<!-- TABLE:ccl -->
| shape | `all_reduce` Ring | `rs_ag` Ring | `rs_only` Ring | `all_reduce` Linear | `stack_sum` | `async` |
|---|---|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.11 | 22.12 | 14.43 | 26.50 | 15.83 | 33.31 |
| 64 rows | 25.93 | 25.92 | 16.45 | 30.76 | 22.71 | 38.88 |
| 96 rows | 27.63 | 27.66 | 16.71 | 34.04 | 29.31 | 43.14 |
| 128 rows | 31.04 | 31.04 | 18.49 | 37.87 | 43.65 | 48.00 |
| 256 rows | 42.53 | 42.57 | 24.63 | 53.11 | 76.52 | 66.15 |
| 512 rows | 69.84 | 69.98 | 37.39 | 91.91 | 108.03 | 103.22 |
| decode batch 32 (1024 rows) | 121.84 | 121.89 | 63.28 | 156.55 | 193.93 | 177.70 |
| prefill 2048 | 179.80 | 179.75 | 102.21 | 250.76 | 373.07 | 325.13 |
<!-- /TABLE:ccl -->

These are the **traced** rows, and every one of them is under `FABRIC_1D_RING`: they vary the ops'
`topology` argument, not the fabric. `probe_ccl.txt` also carries an `eager` column, and several of
its rows favour `Topology.Linear` — those are dominated by per-call dispatch, and none of this
stage's decisions reads them.

The **fabric config** is the separate question review round 3 pulled apart from the argument, and it
needs its own process because `set_fabric_config` runs before `open_mesh_device`. `probe_ccl.py
--fabric line` is that process; its rows are tagged `CCLFAB` in the same artifact:

<!-- TABLE:fabric -->
| shape | `all_reduce` ring fabric | `all_reduce` line fabric | `stack_sum` ring fabric | `stack_sum` line fabric |
|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.11 | 22.48 | 15.83 | 18.08 |
| 64 rows | 25.93 | 26.68 | 22.71 | 26.87 |
| 128 rows | 31.04 | 32.61 | 43.65 | 59.89 |
| 512 rows | 69.84 | 88.42 | 108.03 | 152.63 |
| decode batch 32 (1024 rows) | 121.84 | 159.17 | 193.93 | 287.43 |
| prefill 2048 | 179.80 | 220.89 | 373.07 | 556.13 |
<!-- /TABLE:fabric -->

Six conclusions, all of which shaped the implementation:

1. **The ring wins both comparisons, and they are two comparisons.** As the ops' *argument*,
   `Topology.Ring` beats `Topology.Linear` at every traced shape (the table above). As the *fabric
   config*, `FABRIC_1D_RING` ties `FABRIC_1D` up to 128 rows — latency-bound, so the ring's second
   direction has nothing to carry, and individual rows there fall either way — and wins from 512 rows
   up, which is the prefill regime. `FABRIC_1D_RING` is what the layer configures, and `ttnn.Topology.Ring` is passed to the
   ops that still accept it; `ttnn.all_gather` marks both `topology` and `num_links` deprecated and
   ignored, so the `stack_sum` path takes the ring from the fabric config alone. Rounds 0-3 of this
   stage asserted the fabric comparison while measuring only the argument one; the line-fabric rows
   above are what closed that.
2. **`ttnn.all_reduce` and explicit `reduce_scatter` + `all_gather` are the same number at every
   shape**, within the probe's own repeatability (a couple of tenths of a microsecond; the largest
   gap in the table above is at decode batch 32). The stable all-reduce lowers to exactly that pair. The `rs_ag`
   arm therefore exists to make that identity *checkable*, not as a separate candidate — and
   the `ccl` arm of `ab_layer_knobs.txt` confirms it at the layer, where the two spell the same number.
3. **`ttnn.experimental.all_reduce_async` runs here, and is slower.** Rounds 0-3 of this stage
   recorded it as *refusing Blackhole DRAM inputs outright* on the strength of a `TT_FATAL` in
   `all_reduce_async_device_operation.cpp`. That was wrong twice over: the guard is DRAM-specific
   rather than architecture-specific, and the arm was calling the op with one barrier semaphore where
   it asserts two and with `cluster_axis=None` where it requires an axis — so it never reached any
   Blackhole check. Called correctly it is correct on a DRAM operand (and on an L1 one), and it is
   1.5-1.9x slower than the shipped arm at every measured shape. Rejected on the measurement, which
   is what the `async` column above now carries. §16 round 3 is the thread.
4. **`stack_sum` (`all_gather` onto a new leading axis, then a local `ttnn.sum`) wins below 64
   rows** even though it moves 4x the bytes, because at that size both are latency-bound and it is
   one fabric phase instead of two. It loses by a widening margin from 96 rows up. Hence
   `CCL_MODE="auto"` with `CCL_STACK_SUM_MAX_ROWS = 64`.
5. **The fabric's packet payload was left at the build default for four rounds, and the runtime had
   been saying so all along.** Every CCL dispatch logs `Fabric packet size 4352 B is suboptimal for
   transporting 2048 B pages. Configure 8192 B` (`ccl_common.cpp:63`) — 864 of them in one suite log
   — and no document, probe or limitation mentioned it until review round 5. It is settable from
   Python (`FabricRouterConfig.max_packet_payload_size_bytes`, through `set_fabric_config` or the
   pytest `device_params`), and taking the advice is worth up to 18% on the collectives (`CCLPKT`
   rows) and a few tenths of a percent at the layer — never adverse on any measured row, and a tie on
   one of the four bench rows. Now shipped as
   `DEFAULT_FABRIC_PACKET_BYTES = 8192`. The lesson this stage keeps re-learning: a warning on the
   critical path is evidence, and "it still passes" is not a classification.
6. **The sharded-residual family is priced here too.** `rs_only` is the reduce-scatter half alone:
   about two thirds of the full all-reduce at the decode tile (the table above), i.e. a saving of
   roughly a third of a collective, twice a layer. That is the entire budget a sharded residual has to pay for itself out of. §4 spends it.

The `decode_b32` row — 1024 physical rows for 32 useful ones — sat in this table from the first sweep
and its significance was missed until review round 1. §11b is that thread.

---

## 4. The residual contract: replicated, decided against the usual advice

The `$multichip` skill is explicit that a replicated residual must not be the only measured contract,
and that a `reduce_scatter → all_gather` round trip mostly recreates the communication a fused path
was meant to avoid. Both points are taken, and the sharded-residual family was priced before being
rejected rather than after.

The budget from §3 is the reduce-scatter saving, twice a layer — about 15 us at the decode tile,
and the table above has the two rows it comes from. Against it, a sharded residual on *this* model
must pay:

* **Both RMSNorms become distributed.** With the hidden dimension fractured across devices, a local
  RMSNorm computes the wrong statistics, so each norm becomes
  `rms_norm_pre_all_gather` → stats all-gather → `rms_norm_post_all_gather`. That is a **third and
  fourth** collective per layer, on the latency-critical path, to save part of one. Even at the
  smallest measured shape a stats all-gather is a fabric phase, and §3 row 1 shows a fabric phase at
  the decode tile is 17–22 us. Two of them exceed the whole saving on their own.
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

`tt-perf-report` retires the question: the collective share of merged 4-device time is under a tenth of the decode window and under a
sixteenth of prefill on both layer kinds (README §5.4 has the generated table) — and §12b shows part
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

<!-- TABLE:moepar -->
| phase | arm | experts/device | active/device | us |
|---|---|---|---|---|
| decode | unsharded, still gate-selected | 256 | 8 | 665.02 |
| decode | **expert parallelism** (shipped) (mean local) | 64 | 2 | 196.96 |
| decode | **expert parallelism** (shipped) (>= the expected maximum, 3.512) | 64 | 4 | 204.56 |
| decode | intermediate sharded 4 ways | 256 | 8 | 472.74 |
| prefill | unsharded, still gate-selected | 256 | 162 | 1640.63 |
| prefill | **expert parallelism** (shipped) | 64 | 41 | 513.77 |
| prefill | intermediate sharded 4 ways | 256 | 162 | 1299.29 |
<!-- /TABLE:moepar -->

Read like for like — each arm at *its own* representative active count, which is what the layer
actually runs — expert parallelism beats intermediate sharding by about **2.1x at decode** and
**2.5x at prefill**, and the unsharded arm by about 2.9x at decode. README §3 generates those ratios
from this artifact; three review rounds each found a hand-derived version of them wrong, which is
why they are no longer written by hand. The two arms' active counts differ (162 globally against ~41 locally) because
that difference *is* what expert parallelism buys; comparing them at a common count would compare
something neither arm runs. Three drafts of this comparison were wrong before this one and each was
caught by a review round: round 1 found the prefill `ep` row quoted at 31 active against `tp` at 162
(a third operating point), round 2 asked for each arm's count to be stated, and round 4 found the
probe still choosing the EP arm's prefill point from an undivided draw count — 63 rather than the
~41 that actually lands on a device — which had been understating the EP win at prefill. The sweep
keeps both, so the two are directly comparable in the artifact.

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
  quoted as exact.) The artifact's `ep` decode rows sweep the cost as a function of local active
  count, 1 through 8; it rises by about a fifth across that whole range, so the realised ~3.5 sits on
  the flat part of the curve and the variance in the count costs little.
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

<!-- TABLE:dense -->
| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |
|---|---|---|---|---|---|---|---|---|
| `attn_in` | (32,2) → 33/2 | 30.92 | 110/8 | 17.85 | 13.07 | 0.21 | 0.22 | **retuned (110, 8)**, 17.85 |
| `gdn_in` | (110,2) → 110/2 | 48.79 | 33/8 | 20.35 | 28.44 | 0.10 | 0.15 | **retuned (110, 8)**, 20.41 |
| `shared_down` | (48,16) → 55/4 | 9.17 | 4/4 | 8.44 | 0.73 | 0.33 | 3.86 | **retuned (8, 4)**, 8.52 |
| `expert_select` | new role | — | 55/8 | 8.29 | — | 0.23 | 0.20 | **(8, 8)**, 8.37 |
| `o_proj` | (16,16) → 22/16 | 9.39 | 22/8 | 9.21 | 0.18 | 0.42 | 0.19 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.49 | 33/16 | 9.41 | 0.08 | 0.15 | 0.19 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.64 | 33/32 | 8.64 | 0.00 | 0.13 | 0.09 | inherited |
| `router` | (32,32) → 33/32 | 8.88 | 22/32 | 8.85 | 0.03 | 0.25 | 0.03 | inherited |
<!-- /TABLE:dense -->

`repeatability` is the probe's own noise floor for that role, read out of the same file: several
distinct core *targets* collapse onto the same realised grid, so the file already contains
independent repeat measurements of identical configs, and this column is the widest disagreement
between such repeats. It is the threshold a delta has to clear to be a result at all — and adding it
is the direct fix for the mistake described below.

* The two wide in-projections are where the inherited table is badly wrong: the inherited cap of 2 is
  the **worst legal value** for both local shapes, costing a factor of 1.7 on `attn_in` and 2.4 on
  `gdn_in`. Together they are worth tens of microseconds a step on both layer kinds; §11's `geometry`
  arm has the measured pair, and it is the number quoted there rather than a second copy here.
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
  repeatability on repeated identical configs (a tenth of a microsecond to several, depending on role
  and sweep), which is
  now a column. Under it the inherited roles are ties, and where a single sweep says otherwise —
  `o_proj` in the committed one — the cross-sweep instability is the reason the inherited entry
  stays rather than a reason to chase it.

* `shared_down` was **initially left inherited on that same reasoning, and that was wrong**: its
  realised point is 55 cores at `in0_block_w` 4, which came out 0.51-0.68 us behind the winner on
  every sweep against a 0.25-0.35 us repeatability — consistently outside the noise, unlike the rows
  above. TP=4 cuts its `K` from 512 to 128 (4 tiles), so a 55-core grid for a 4-tile `K` and a
  64-tile `N` is launch overhead rather than parallelism.

  It was first retuned to **4** cores, the winner on three sweeps. A fourth sweep read 4 cores at
  about 10 us against 8 cores' 8.71-8.96 — a flip of more than a microsecond, outside repeatability — while the 8-core
  point has never read above 9 us on any sweep. So it ships at **8**: the stable reading, never behind, and
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

Two things now stop that recurring, and round 3 found both of them still half-done. Round 2 said
`probe_sparse_matmul_local.py::shipped_choice` had been re-derived from `_active_expert_bound`; it
had not — it still took the probe's `--active` flag, so the committed artifact's header printed the
round-1 basis next to a README table that had been corrected. `shipped_choice` now takes the bound,
`active_bound(rows, experts)` computes it the way the layer does, and the header prints one line per
group row count with the bound named, so the two cannot disagree again. And
`test_sparse_cores_match_the_local_sweep` installed its `ttnn.sparse_matmul` spy *after*
`prefill_forward` returned, so it pinned decode only and the prefill row of the table above was
derivation-only; the spy is now installed first and tags each call with its phase, so prefill (32/32,
where the bound saturates) is asserted on the mesh alongside batch 1/2/4/8/32 decode.

`logs/probe_sparse_matmul_local.py` is the single-chip sweep's candidate ladder with `E` made a flag,
re-run at the per-device operating points under the same BFP4/LoFi policy. Best microseconds per
realised core count, and the winner:

<!-- TABLE:sparse_ladder -->
| active | 4 cores | 8 cores | 16 cores | 32 cores | 64 cores | winner |
|---|---|---|---|---|---|---|
| 4, `down` | — | **52.3** | 59.5 | 66.7 | 85.4 | 8 |
| 4, `gate_up` | 82.7 | **62.5** | 64.2 | 70.1 | — | 8 |
| 8, `down` | — | 62.2 | **61.9** | 68.9 | 88.0 | 16 |
| 8, `gate_up` | 122.5 | 81.5 | **73.1** | 77.9 | — | 16 |
| 16, `down` | — | 75.2 | **65.3** | 72.4 | 92.4 | 16 |
| 16, `gate_up` | 180.3 | 109.6 | **86.8** | 90.0 | — | 16 |
| 32, `down` | — | 119.9 | 83.0 | **80.3** | 102.2 | 32 |
| 32, `gate_up` | 330.1 | 192.8 | 125.5 | **120.1** | — | 32 |
| 41, `down` | — | 146.2 | 96.0 | **85.8** | 107.7 | 32 |
| 41, `gate_up` | 414.5 | 240.9 | 151.1 | **140.2** | — | 32 |
| 63, `down` | — | 235.7 | 145.6 | **105.8** | 124.8 | 32 |
| 63, `gate_up` | 691.6 | 398.6 | 234.9 | **199.3** | — | 32 |
<!-- /TABLE:sparse_ladder -->

The `active ∈ {8, 16, 32, 41}` rows were added in round 2. Round 1 swept only 4 and 63, which left
the whole decode range between them unmeasured and made 63 — itself an over-count, see §12c — the
only prefill evidence. The ladder's winner now tracks the shipped rule across the range, and the
prefill decision is the same at the corrected 41 as at the old 63.

Both effects are measured at the **layer**, which is what decides:

* **prefill** — the `sparse` rows of §11's generated table, i.e. about 4 ms a
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
  tokens and decodes at the last slot, both against the HF golden (round 3: it asserted shape, finiteness
  and variance only, at a length where the reference runs).
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

Results from `logs/pytest_full_suite.txt.gz` (**137 passed, 3 skipped**; the 3 skips are
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
  checkpoint tensor **in order**: every one of the 256 experts scores at least 0.992281 against its
  own checkpoint weights (the residue is bfloat4_b quantisation from the inherited precision policy,
  not a partition error) and at most 0.004249 against the two neighbours a block-boundary error would
  produce. Three orders of magnitude apart, which is what makes it a disjointness proof rather than a
  smell test; round 4 widened it from a 1-in-17 sample to all 256, and round 5 found the old sampled
  figure still quoted here.
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
  shipped one (worst PCC 0.999996 over the 12 cases, and bit-equal respectively), so the A/B
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

<!-- TABLE:ablayer -->
| knob | arm | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|---|
| `ccl` | **`auto`** (shipped) | 0.611 | 0.500–0.501 | 28.97–29.09 | 28.31–28.49 |
| `ccl` | `stack_sum` | 0.611 | 0.500–0.501 | 29.23–29.50 | 28.35–28.51 |
| `ccl` | `all_reduce` | 0.621 | 0.508–0.509 | 28.94–29.41 | 28.19–28.55 |
| `ccl` | `rs_ag` | 0.621 | 0.508 | 29.01–30.46 | 28.56–28.69 |
| `geometry` | **multichip-retuned** (shipped) | 0.611 | 0.500–0.501 | 28.97–29.27 | 28.30–28.48 |
| `geometry` | single-chip-inherited | 0.635 | 0.516–0.517 | 29.02–29.37 | 28.07–28.39 |
| `routing` | **`select_matmul`** (shipped) | 0.611 | 0.500–0.501 | 28.88–28.97 | 28.25–28.66 |
| `routing` | `gather` | 0.668 | 0.559–0.560 | 29.29–29.54 | 28.53–28.88 |
| `sparse` | **tp-rescaled** (shipped) | 0.611 | 0.500–0.662 | 28.85–28.96 | 28.42–28.51 |
| `sparse` | single-chip-inherited | 0.611 | 0.500–0.501 | 32.98–33.19 | 32.19–32.66 |
| `cast` | **block-float** (shipped) | 0.611 | 0.500 | 28.91–29.10 | 28.15–28.45 |
| `cast` | `bf16` | 0.614 | 0.504 | 29.09–29.35 | 28.25–28.63 |
<!-- /TABLE:ablayer -->

Every row is at **batch 1**. That is why the `sparse` arm ties in the decode columns here and not in
§7b: the rescale only changes decode geometry above batch 1, and `probe_decode_batch.txt` is where
its decode effect is measured. Round 2 found the batch-1-only reading being generalised into "decode
is unchanged", which is why that split is now stated everywhere the knob is described.

* `auto` reproduces `stack_sum` exactly in the decode column, as it must at the batch-1 tile — the
  arms cross-check the switch rather than just measuring it.
* The retuned dense geometry is worth **24 us/step** on linear_attention and **14 us/step** on
  full_attention.
* The rescaled sparse core rule is worth about **4.0–4.6 ms** of prefill on both layer kinds (§7b),
  and tens of microseconds a step at decode batch 4 and above, which this table cannot see.
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

The `decode_b32` row in §3's table had priced the wasteful shape from the first sweep — an order of
magnitude above `stack_sum` at 32 rows — without the connection being made.

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

<!-- TABLE:decode_batch -->
| batch | mixer physical rows | off | on | delta |
|---|---|---|---|---|
| 1 | 32 | 0.500 / 0.611 | 0.500 / 0.611 | +0 / +0 us |
| 2 | 64 | 0.578 / 0.705 | 0.575 / 0.703 | +3 / +2 us |
| 4 | 128 | 0.634 / 0.815 | 0.639 / 0.820 | **-5 / -5 us** |
| 8 | 256 | 0.787 / 1.003 | 0.783 / 1.000 | +4 / +3 us |
| 13 | 416 | 1.620 / 2.062 | 1.607 / 2.051 | +13 / +11 us |
| 16 | 512 | 1.827 / 2.271 | 1.794 / 2.243 | +33 / +28 us |
| 32 | 1024 | 3.018 / 3.945 | 2.932 / 3.862 | **+86 / +83 us** |
<!-- /TABLE:decode_batch -->

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
<!-- TABLE:bench -->
| layer kind | phase | single-chip | 1x4 replication control | multichip | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.58 ms | 101.78 ms | **29.33 ms** | **3.463x** | 86.6% |
| linear_attention | decode (traced) | 1.031 ms | 1.032 ms | **0.610 ms** | **1.690x** | 42.3% |
| full_attention | prefill 2048 | 95.41 ms | 95.64 ms | **28.24 ms** | **3.379x** | 84.5% |
| full_attention | decode (traced) | 0.827 ms | 0.827 ms | **0.500 ms** | **1.654x** | 41.3% |
<!-- /TABLE:bench -->

`test_multichip_beats_single_chip_traced_decode` gates the decode claim inside the suite itself.

`tt-perf-report` (`tracy/`, four separate captures — prefill and decode never share a Tracy session,
and `--active-experts` is passed so the routed `sparse_matmul` rows are modelled rather than skipped)
explains the 41–42% decode efficiency and confirms it is **not** a communication problem:

| | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
See README §5.4 for the generated table; the shape of it is that the routed sparse matmuls are
74–75% of prefill and 18–22% of decode, the collectives are under a tenth of the decode window, and
the decode DRAM roofline is about 6%.

The decode window is at **6% of the DRAM roofline** — batch-1 decode is launch- and latency-bound, so
dividing the work by four does not divide the time by four. That is inherited from the single-chip
stage, not caused by sharding, and it is the whole explanation for the gap between 84–87% prefill
efficiency and 41–42% decode efficiency. The routed experts remain the dominant term and were the
right target: §7b took about 4.0–4.6 ms/layer off prefill.

The prefill roofline figures are at the **corrected** `--active-experts` input (§12c) and are not
comparable to the ones this stage quoted before round 2, which were computed at an input about half
as large again.

---

## 12b. The block-float collective: an anomaly, and the null result that classified it

The prefill profile shows the layer's two collectives at wildly different cost for the same logical
shape — `tracy/full_attention/prefill_perf_report.txt` has the BF16 reduce-scatter at 97 us on 20
cores and the BFP8 one at **about 1500 us on 12 cores**, roughly 5% of the whole prefill window.
(The stacked report's single `ReduceScatter` row is the two summed, which is why it reads higher;
three different numbers for this one row appeared in the round-1 documents, which round 2 flagged.) The only difference is the operand dtype: the MoE half
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
Evidence:          tracy/*/prefill_perf_report.summary.txt (101 us BF16 on 20 cores vs 1477 us BFP8
                   on 12, full_attention; 100 vs 631 us, linear_attention; every sweep re-measures
                   these and the ratio, not the absolute, is the finding);
                   logs/probe_ccl.txt prefill_2048 rs_only_ring trace 102.19.
Affected path:     the second per-layer collective, prefill, both layer kinds.
Control:           decode mostly reproduces the probe (AllGather 12.71-13.19 us/op on the L1-operand
                   row, against the `stack_sum` probe row) -- with one exception, below, so the discrepancy is prefill-specific; and the `cast` arm removes the
                   block-float operand without moving the layer at all.
Investigation:     CCL_CAST_BLOCKFLOAT implemented and A/B'd at the layer, three builds per arm.
Second instance:   the same shipped `stack_sum` all-gather costs 33.73 us/op on full_attention and
                   21.85 us/op on linear_attention at the same decode shape (the DRAM-operand rows of
                   `tracy/*/decode_perf_report_stacked.csv.gz`, 32 ops each), while the L1-operand
                   rows of the same two captures agree to within 4%. Review round 4 found this and it
                   is the same signature as the prefill one -- a collective's cost varying with what
                   the four devices did *before* it, not with what it moves. It is ~1.5% of the decode
                   window and nothing rests on it; recorded here so the ledger holds both instances.
Resolution:        controlled, with the mechanism left open. That row is NOT data movement this
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

* the prefill roofline is **27% on both layer kinds**, not the 41% the round-1 documents recorded. Neither the
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
state-, trace- and collective-critical subset — **50 passed, 3 skipped**. `watcher/census.py`
partitions all 59 031 log lines into disjoint buckets, asserts the partition is complete (an unknown
line kind lands in `UNCLASSIFIED` and trips the assert rather than hiding in a catch-all), and reports
**0 fatal-class matches** over 54 dumps with minimum stack headroom 1332 bytes free over 20 detail
(from `stack reporting cores: 1`, i.e. a single core's spot check rather than a survey — round 4)
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
* **One hang, one reset, one triage capture.** `ttnn.experimental.all_gather_matmul_async` wedged the
  mesh during the round-3 fused-CCL measurement: `AllGatherMatmulAsyncDeviceOperation` on
  `[1,1,32,512] x [1,1,2048,2560]` sat on all four devices and 40 cores for ten minutes with no
  progress. Captured with `tools/tt-triage.py` before killing it (`triage/tt-triage.txt`,
  `triage/triage-summary.txt`, 21 checks), killed, and the devices were recovered with `tt-smi -r`.
  Recovery was verified with an actual CCL smoke test rather than an open/close — the first attempt
  passed an open/close check while the fabric was still wedged, which produced a second false hang.
  The arm is off by default in `probe_fused_ccl.py` and `run_evidence.sh` runs that probe one process
  per shape under `timeout`, so a repeat cannot take a sweep with it (README limitation 5). Review
  round 6 found this section claiming the opposite; it was written before the round-3 measurement and
  never revisited.
* No ARC/ERISC fault and no Ethernet-link loss at any point. `tt-smi -ls --local` reported all four
  `p300c` chips before and after every sweep, including after the reset above.
* `$autofix` was not needed: no other failure in this stage was mysterious. The three that occurred —
  the ACTIVE_ETH kernel-buffer overflow (§13), an `L1_SMALL` exhaustion inside
  `probe_decode_batch.py` when it built ~50 decoders in one process (CCL semaphores are not reclaimed
  while the mesh stays open; fixed by running one process per batch, which `run_evidence.sh` now
  does), and the `shared_down` geometry gap (§7) — were each diagnosed directly from the failing
  command's own output.

---

## 15. Evidence regeneration

Everything in this directory is produced by one script, in the order the hardware discipline
requires:

```bash
bash models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/run_evidence.sh
```

Steps: `suite` → `bench` → `ab` → `probes` → `tracy` → `watcher`, then a seventh, unconditional step
that regenerates every table in these two documents from the artifacts (`logs/make_tables.py`),
records the behavioural fingerprint of the measured sources (`audit_figures.py --stamp`) and audits
every remaining hand-written figure. The first six are selectable with `STEPS="..."`.
Large text logs and every CSV are committed gzipped (the repo's pre-commit hook rejects files over
500 KB and `.gitignore` carries a blanket `*.csv`); every generator reads `foo.ext` or `foo.ext.gz`
transparently. The repo's trailing-whitespace and end-of-file hooks rewrite the committed
`*_perf_report.txt` and `*.summary.txt` files; that is cosmetic and the numbers are untouched.

Eleven full sweeps were run in this stage, each after a source change and each regenerating every
artifact, so the committed artifacts come from exactly the bytes that were committed — including the
formatting the repo's `black`/`isort`/`autoflake` hooks apply, which is run *before* each final sweep
rather than after it. Running the sweep this many times is also what produced the run-to-run
stability evidence in §7: the retune decisions hold in every one of them, the near-tie deltas do not,
and one of them (`shared_down` at 4 cores) flipped far enough on the sixth to change the shipped
value.

Two things make a re-sweep cheap enough to be the default answer to any doubt. Every table in these
documents is **generated** from the artifacts, so a re-measurement cannot leave a stale table behind;
and freshness is checked against `logs/source_stamp.json` — a hash of each measured source's
comment- and docstring-stripped AST — rather than against file mtimes, so the documentation edits a
review round produces do not invalidate a 90-minute measurement while any change to the code still
does. Both were added after round 3, whose recurring finding was stale hand-transcribed figures.

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
| P1 — dominant `SparseMatmul` flagged `SLOW` in all four captures, misattributed in the README, geometry never re-swept at the per-device point | `probe_sparse_matmul_local.py` + `MultichipMoE._sparse_cfg`; about 4.0–4.6 ms/layer off prefill, and README §5.4 corrected (§7b) |
| P2 — "dense all-expert 3.3x slower" is not what the `single` arm measures | restated: the arm is unsharded-but-gate-selected, so it prices EP, not density (§6.1) |
| P2 — EP-vs-TP prefill compared at different active counts | each arm now labelled with its own operating point (§6.1) |
| P2 — HF-golden PCC count wrong | recounted from the log by a script rather than by hand |
| P2 — the two "leaves N GiB" figures were GB labelled GiB | corrected to 9.04 / 24.88 GiB (§8) |
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
| §5.8 cited "39–44 across devices" for the imbalance mechanism; that range is across layer kinds | the mechanism is now stated as an open candidate, not a finding, and README limitation 9 records it (§12b) |
| gap — nothing asserted the realised sparse core count | `test_sparse_cores_match_the_local_sweep` |
| gap — `o_proj`/`gdn_out` kept on a cross-sweep-instability claim not reproducible from the committed artifact | the `repeatability` column, which is computed from the committed file, is now the stated threshold; the cross-sweep history is context, not the argument |

### Round 3 — `more-work-needed`

A third independent reviewer. Its two P1s were both *unearned rejections*: an optimization family
this stage had claimed to measure and had not, and one it had rejected on an error message it had
caused itself. Neither changed the shipped path in the end — both alternatives are genuinely slower —
but neither was evidence until round 3 forced the measurement.

| finding | resolution |
|---|---|
| P1 — the fused CCL+matmul family was never measured, and the shipped docstring said it was | `logs/probe_fused_ccl.py`: `matmul_reduce_scatter_async` on the `o_proj` row-parallel boundary and `all_gather_matmul_async` on the `attn_in` column-parallel one, at the real per-device shapes, traced, at the decode tile / batch 32 / the 2048-token prefill chunk, with the persistent intermediate and output buffers and the multi-device semaphores the ops require and the matmul confined to a sub-grid so the CCL workers have cores. Fused is slower than the unfused pair at every shape; the *sharded*-residual variant wins at the producer and loses more at the consumer; `all_gather_matmul_async` **hung the mesh** and is recorded as an exact op-contract blocker with a `tt-triage` capture (§14). README §2.5 has all three rows |
| P1 — `all_reduce_async` was rejected on the one memory config its own assert forbids | the assert is DRAM-specific, not architecture-specific — and the arm had never reached it: it passed one barrier semaphore where the op asserts two, and `cluster_axis=None` where it requires an axis. Fixed, plus an L1 arm. It runs, it is correct, and it is 1.5–1.9x slower than the shipped arm at every shape. README limitation 8 was a limitation that did not exist and is now a measured rejection (§3 conclusion 3) |
| P2 — `probe_sparse_matmul_local.txt`'s `SHIPPED` header still encoded round 1's assumed-active basis, contradicting the corrected README table; and the sparse-core test pinned decode only | `shipped_choice()` now takes `_active_expert_bound` (`active_bound(rows, experts)`) and prints one line per group row count with the bound named; `test_sparse_cores_match_the_local_sweep` installs its spy before `prefill_forward` and asserts the prefill grid (32/32, measured on the mesh) as well as batch 1/2/4/8/32 decode (§7b) |
| P2 — `context_contract.json` carried figures round 1 had already replaced, and a batch-32 PCC pair in no committed log | refreshed from the committed artifacts, and `non_aligned_max_context` reconciled with the test that runs it |
| P2 — five stale claims in the shipped module's docstrings (16/32 dense caps "fail to build", the imbalance mechanism stated as fact, a line *fabric* said to be measured, `stack_sum` "loses from 128 rows up", "both collectives are `ttnn.all_reduce`") | each corrected in place; the `stack_sum` crossover now says 96 rows, which is what the artifact shows six lines below it |
| P2 — Ring-vs-Linear was an op-argument comparison presented in three places as a fabric-config comparison | `probe_ccl.py --fabric line` reconfigures the fabric in its own process and tags its rows `CCLFAB`. Both comparisons are now measured and stated separately: the fabrics tie at the decode tile and at 64 rows, and the ring wins 1.3–1.4x from 512 rows up (§3, README §2.1) |
| P2 — README §5.4's `1x1 output subblock` advisory counts were attributed to the wrong op family | recounted from the committed reports by pairing each advisory to its row: 64 of 257 sparse prefill rows, **none** of the 128 sparse decode rows; the decode 1x1 advisories are on the small dense projections, where `per_core_N = 1` at the swept core count forces it |
| P2 — two README figures still at their pre-round-2 values (`--active-experts 63`, "2048 extra columns" asserted and denied in one sentence) | both deleted |
| P2 — this stage lacked the `audit_figures.py` guard the three preceding stages carry, while `census.py` referenced it | ported and extended (`derived_from_artifacts` computes the aggregates rather than whitelisting them), plus `logs/make_tables.py`, which **generates** all ten tables in these two documents from the artifacts. Both run as step 7 of `run_evidence.sh`. Every quoted figure in the documents, the contract and the probe scripts is now either in an artifact, generated, or listed with its provenance |
| `test_unaligned_max_context` asserted shape and variance where the HF golden is tractable | it now asserts prefill **and** decode PCC against the reference at `max_context = 5000` |
| `_all_reduce` freed the pre-fold tensor guarded only by object identity, which would not catch a `ttnn.reshape` that returned a view | `_free_unless_aliased` compares buffer addresses, which is what `deallocate` operates on |
| `logs/watcher_pytest_eth_enabled.txt` predated the sweep it sat next to | the ACTIVE_ETH control is now re-run from the committed bytes as part of step 6, so it is an artifact of the same sweep as everything else |
| smaller: work log §7 quoting a different `geometry` figure from §11's, `--active`'s help text, README §5.6 quoting a range where the artifact has one measurement | each corrected in place, or replaced by a pointer to the generated table that carries the number |

### Round 3 follow-through, found while fixing it

Two things surfaced while closing round 3 that no reviewer had raised, both recorded here because
they are the same failure mode round 3 named:

* `probe_expert_parallel.py::arms()` still chose the EP arm's prefill operating point from the
  **undivided** draw count — 63 of 64 local experts, i.e. every expert active — while its own
  docstring said ~41. That is round 2's `--active-experts` error surviving in the one place round 2
  did not look. Corrected to `32 * top_k / tp` draws for the arm that only sees a quarter of them;
  the sweep keeps 63 as well, so the two are side by side in the artifact. It had been *understating*
  the EP win at prefill (2.5x against intermediate-TP at the corrected point, not 1.85x).
* the freshness half of `audit_figures.py` compared mtimes, so a documentation-only edit to
  `multichip_decoder.py` after a sweep reported every one of the 31 artifacts as stale and the only
  remedies were a 90-minute re-run or a hand-restored mtime. It now compares a hash of the
  comment- and docstring-stripped AST recorded by the sweep (`logs/source_stamp.json`), so the
  question "did the code change?" is answered mechanically.

### Round 4 — `more-work-needed`

A fourth independent reviewer, given rounds 1-3 as claims to verify. No correctness defect; five P2s,
all of them evidence-integrity — three of them the same failure mode round 3 named, now inside the
*correctness* table rather than the performance one.

| finding | resolution |
|---|---|
| P2 — README §4.4 and §1 said `test_prefill_continuation` resumed at a **non-chunk-aligned** `start_pos`, which `OptimizedDecoder.prefill_forward` rejects outright; the test resumed at exactly `chunk`, and its docstring's "equals a single call over the concatenation" comparison was never made | the test now resumes on the chunk boundary and **ends** at 130 tokens (tile-, page- and chunk-unaligned), and checks the HF golden *and* a single 258-token TTNN prefill. `test_prefill_continuation_rejects_unaligned_start` pins the inherited `start_pos % chunk_size` restriction as a clean `ValueError`, `context_contract.json` records it as `public_prefill_start_pos`, and §4.4 says what runs |
| P2 — §5.4 used the profiler to check "data movement" while omitting `TM`, the layout category, which is 19-22% of the decode window — larger than the `SparseMatmul` row the section calls top of the stack | §5.4 now carries the whole window by `Op Category`, with the **single-chip stage's share of the same phase in brackets** as the control: a few points lower there than here on the same phase, i.e. inherited. The eight rows behind it are tabulated and attributed to call sites; the one this stage moves (`ReshapeView`, the `CCL_COMPACT_ROWS` fold) is measured a net win in §5.9 |
| P2 — two rows of §4.3 described comparisons the tests did not make: `test_ccl_modes_agree` ran at 128 rows, where `auto` already resolves to `all_reduce`, so its `all_reduce` arm compared the shipped path with itself; `test_routing_select_modes_agree` ran a `select_matmul` arm against `select_matmul` and compared routing vectors while the README claimed layer output | `test_ccl_modes_agree` now runs at 32 **and** 128 rows, i.e. both sides of the `auto` crossover, so every arm meets a genuinely different spelling; `test_routing_select_modes_agree` compares `gather` against `select_matmul` once and asserts the **layer** PCC as well as bit-equality of the vectors |
| P2 — §3 priced the EP arm at the *mean* local active count (2) while the arms it beats were priced at their exact count, and the same section says the step waits for the *expected maximum* (3.512) | the generated table now carries the EP arm at both points, and the ratios are quoted at the conservative one |
| P2 — §4.2's HF-golden PCC breakdown was 4 short and misattributed after round 3 added two assertions to an existing test | §4.1 and §4.2 are **generated** from the suite log by per-test patterns (`make_tables.py::PCC_INVENTORIES`), so an added assertion updates the count. The baseline inventory is generated the same way |
| the "gates the decode claim" wording (the test asserted `> 1.0`), `test_perf_prefill` asserting nothing, `test_weights_are_sharded_not_replicated` comparing only devices 0 and 1, `test_expert_partition` sampling 1 expert in 17 at `pcc > 0.99` with disjointness never asserted, `test_decode_runs_the_multichip_program_configs` accepting `realised <= target`, `test_gate_selected_experts_not_dense` not pinning the band its `--active-experts` modelling input sits in | each tightened: a 1.4x decode bar and a 48 ms prefill bar, all six device pairs, every expert at the baseline bar plus an explicit mismatched-expert bound, equality against the realisable core count, and a 30-55 band on the prefill active count |
| the `+0.0% to +0.2%` replication-control range (it is +0.27%), a `~110 us` collective row (101), the "no others" collectives claim, the `stack_sum` local reduce missing from the collectives share, the unmeasured distributed-norm cost stated as fact, the single-core stack-headroom figure | each corrected or qualified in place |
| hard-check gap — `run_evidence.sh` wrote the source stamp outside the step guard, so a partial sweep would certify unmeasured artifacts as fresh | the stamp is written only when all six measuring steps ran; a partial sweep says so and leaves freshness on mtimes |
| hard-check gap — the fingerprint covered only the two stage-owned files, not `optimized_decoder.py`/`model_config.py`, which every artifact also measures | `INHERITED_SOURCES` added to the fingerprint (and deliberately **not** to the scanned document set — their figures are the optimized stage's own audit) |
| anomaly ledger — the decode `stack_sum` all-gather costs 33.9 us/op on full_attention against 21.8 on linear_attention for the same shape, unrecorded | added as the second instance of the §12b anomaly, same signature, ~1.5% of the decode window; README limitation 9 now covers both |

### Round 5 — `more-work-needed`

A fifth independent reviewer. No correctness defect again; five P2s, all evidence integrity, plus one
runtime warning on the critical path that turned out to be a real optimization nobody had taken.

| finding | resolution |
|---|---|
| P2 — **864 `Fabric packet size 4352 B is suboptimal … Configure 8192 B` warnings** on this stage's own collectives, classified nowhere: not in a document, a probe artifact or a limitation, and `run_evidence.sh` sends every probe's stderr to `/dev/null` so it never reached one | measured, and **taken**. The knob is settable from Python (`FabricRouterConfig.max_packet_payload_size_bytes`), and at 8192 B every collective arm at every shape is faster — the shipped `stack_sum` at the decode tile and the shipped `all_reduce` at the prefill chunk both by about 8%, up to 18% at the larger shapes, and a few tenths of a percent at the layer, which is what a category that is 3–9% of the window predicts. Now `DEFAULT_FABRIC_PACKET_BYTES = 8192` with `fabric_router_config()`, applied by the suite through `device_params`, by every probe, and by `bench.py`; the build default is kept as a measured arm in `probe_ccl.txt` (`CCLPKT`) and in the bench table (§2.1, §3 conclusion 5) |
| P2 — README §7 and the header table quoted watcher census counts that no longer matched the committed summary, and the two documents disagreed with each other | all five sites refreshed from `census_summary.txt`, and the guard gap closed: `audit_figures.py` now normalises space-separated thousands (a space-formatted line count was invisible to a `\d{4,}` scan, and so was every byte count in both documents) and checks the dump and detail-line counts against the label they sit next to in the artifact. Round 6 found the first version of that check capturing bare digits, so it could not fail; it now asserts the capture is a phrase and matches value-and-label on one artifact line |
| P2 — §16's round-4 EP resolution claimed the table carried "both points … ratios at the conservative one"; `table_moepar` emitted only the mean | it now emits both the mean local count and the point at or above the expected maximum the barrier waits for, labelled, and the ratios in §2.5 and §3 are restated at the conservative one (2.3x / 3.3x, not 2.4x / 3.4x) |
| P2 — work log quoted the expert-partition PCC as 0.9935, which is the pre-round-4 sampled value, and `audit_figures.py` carried a whitelist entry whose stated justification ("rounded from its two logged values") was false | the work log now quotes what the suite prints — every one of the 256 experts at ≥ 0.992281 against its own weights and ≤ 0.004249 against the neighbours a block-boundary error would produce — and the whitelist entry is deleted, so the value stays checked |
| P2 — README §8's limitation list was numbered `1,2,3,4,9,5,6,7,8,9` in source, so rendered markdown renumbered it and every cross-reference to limitations 5–8 pointed one item off, including the shipped module's docstring | renumbered in source order; the three cross-references re-checked |
| smaller: `test_ccl_modes_agree`'s PCC quote (stale after round 4 doubled its parametrization), the `stack_sum` local-reduce share, "identical to two decimal places" (contradicted at decode batch 32 by the table above it), two different `shared_down` ranges for one claim, a "table above" that pointed at the wrong table, `gdn_in` shipping at a grid that is not this sweep's local winner | each corrected, made qualitative where a re-sweep would move it, or — for `gdn_in` — stated and justified rather than left to the table |
| hard-check gap — `make_tables.py::splice` computed `found` and never read it, so deleting a table's markers and hand-writing the table passed silently | `UNPLACED-TABLE` is now reported, which is the drift the markers exist to prevent |
| coverage deltas against the single-chip suite that `capability_note` covered with "everything else is name-for-name" (the mesh full-context case at -3 rather than -1, ragged positions at batch 4 with a contiguous mapping) | both stated explicitly in `context_contract.json`, with where the missing coverage is picked up |

### Round 6 — `more-work-needed`

The first round to find a **P1**, and it was against round 5's own fix: the packet-size decision had
been made on measurements that structurally excluded the collective the warning was about.

| finding | resolution |
|---|---|
| P1 — the shipped 8192 B packet produced 864 warnings asking for 4352 B, and the knob had been measured only on `bfloat16` operands. The layer's *second* collective carries `bfloat8_b` (1088 B tile pages, ideal 4352 B), the README's page arithmetic was wrong (4096 B, and 4 x 4096 is not 8192), and the block-float collective is exactly where the stage's unexplained 15x anomaly lives | `probe_ccl.py --dtype bfloat8_b` added and swept at both packet sizes. The result settles it: bf16 prefers 8192 B at every shape (about 8% on both shipped arms, up to 18% at the larger ones) and **block-float is indifferent** — every arm at every shape within 1% either way, inside the probe's repeatability. One fabric setting serves both dtypes, so the warning cannot be driven to zero; it is now a limitation (§8 item 6) rather than an omission, with `logs/probe_warnings.txt` carrying the distinct warning text from every probe so the next class is visible in an artifact. The layer A/B now runs **three builds per arm**: decode is repeatably 2 us/step faster at 8192 B, prefill is a wash. The anomaly does not move with packet size, which removes the candidate round 6 raised and leaves §12b's barrier explanation where it was |
| P2 — the watcher stack-headroom figure was wrong in both documents, and the round-5 guard that was supposed to catch it **could not fail**: its `LABELLED` patterns captured bare digits, so the check asked whether "20" appeared anywhere in 31 artifacts | figures corrected from the artifact; the check now matches value *and* label on one artifact line, and asserts that a pattern never captures a bare number so this class of vacuous guard cannot be added again |
| P2 — the EP-vs-alternatives ratios in two documents did not follow from the artifact: 2.3x/3.3x against a measured 2.06x/2.91x, and round 5 had restated them by hand from the same rows | generated. `table_moepar_ratios` computes them from `probe_expert_parallel.txt`; three rounds of hand-derivation produced three wrong pairs, which is enough |
| P2 — work log §14 said "no hang, no ARC/ERISC fault, no device reset was needed at any point" and "`tt-triage.py` was therefore not needed", contradicting README limitation 5, the committed triage capture and §16's own round-3 row | §14 rewritten to record the `all_gather_matmul_async` wedge, the `tt-triage` capture, the `tt-smi -r`, and the false-recovery lesson (an open/close check passed while the fabric was still wedged) |
| P2 — "every collective arm at every shape is faster" and "never faster on any measured row" were contradicted by 4 of 72 rows | both restated to what the artifact shows, including the block-float exclusion |
| smaller: a `114 us` collective row (97), category deltas quoted as 4.3/6.9 points (4.4/4.5 and 6.6/9.8), `2.69%` against a table saying 2.70%, and a `gdn_in` sentence added in round 5 whose two figures both disagreed with the table beside it | each corrected or replaced by a pointer to the generated column |
| hard-check gap — `run_evidence.sh` discarded every probe's stderr, which is why the packet warning never reached an artifact | the first fix was wrong: the runtime prints these through its own logger, not to the probes' stderr, so collecting stderr produced an empty file. `logs/warning_census.py` censuses the **suite and watcher logs**, where they actually appear, into `logs/warning_census.txt` — one line per distinct class with counts, `--check` verifies it against the logs, and step 7 regenerates it. Three classes exist and README §7 classifies all three |
| hard-check gap — the audit's own self-test puts 1- and 2-decimal figures at a 0.23 / 0.12 false-positive rate, which is where every wrong prose figure in this round sat | stated in README §9, so a reader does not over-trust a pass: byte counts are pinned, ratios are only as good as being generated |

### Checkpoint

Local checkpoints only; nothing is pushed. `.agents/skills/tt-device-usage/SKILL.md` and
`.agents/fast-models-fast-feedback.md` are dirty in the worktree but are **not** stage-owned and are
deliberately excluded from every commit.

| repo | branch | round | commit |
|---|---|---|---|
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 0 | `914163b92b1` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 1 | `e3915a376f0`, SHA record `ca6674393a1` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 2 | `80563b120e4` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 3 | `9d889b01682` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 4 | `30de0e0e553` |
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | 5 | `87a23776f85` |
