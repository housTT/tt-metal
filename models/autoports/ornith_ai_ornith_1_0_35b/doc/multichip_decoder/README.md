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
| Suite | **137 passed, 3 skipped** (`logs/pytest_full_suite.txt.gz`) |
| PCC vs the single-chip TTNN baseline | 34 values, min **0.999892**, bar 0.999 |
| PCC vs the float32 HF golden | 90 values, min **0.999851**, bar 0.995 |
| Warmed 2048-token prefill | **3.51x** linear_attention, **3.29x** full_attention (§5.2's generated table has the exact figures and the efficiencies) |
| Warmed traced decode | **1.69x** linear_attention, **1.65x** full_attention (§5.2, same source) |
| Advertised context | **262144, unchanged**; per-device layer footprint falls 60–75% |
| Advertised batch | **32, unchanged**, exercised on the mesh at 1 / 4 / 13 / 32, aligned and not |
| Watcher | 0 fatal-class matches over 54 dumps (`watcher/census_summary.txt`) |

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
  (`test_batched_decode_ragged_positions`), and chunked-prefill continuation on a chunk boundary with
  a non-aligned tail (`test_prefill_continuation`; `start_pos` alignment is an inherited restriction,
  §4.4).
* Decode and prefill batch up to 32, exercised **on the mesh** at 1, 4, 13 and 32, at a tile-aligned
  `seq_len` (192) and a non-aligned one (130) (`test_batched_prefill_decode_pcc`). This is not
  inherited from the single-chip stage: TP=4 changes the per-device head counts that every
  batch-sensitive decode op is bounded by. (An earlier version of this bullet also claimed the case
  exercised the `CCL_COMPACT_ROWS` fold in prefill; it does not — the fold is decode-only, §4.5 — but
  it is still the only batched non-aligned prefill coverage the suite has.)
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
2. **`FABRIC_1D_RING`.** Two separate questions, which rounds 0-3 of this stage ran together and
   review round 3 separated: which *fabric config* is set before `ttnn.open_mesh_device`, and which
   `topology` argument the collectives are called with under it. The fabric config is what actually
   selects how a collective traverses the mesh, so it needs its own process; `probe_ccl.py --fabric
   line` is that process, and its rows carry the `CCLFAB` tag.

   **Fabric config, traced us per collective** (`FABRIC_1D_RING` against `FABRIC_1D`):

<!-- TABLE:fabric -->
| shape | `all_reduce` ring fabric | `all_reduce` line fabric | `stack_sum` ring fabric | `stack_sum` line fabric |
|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.11 | 22.47 | 15.84 | 18.12 |
| 64 rows | 25.94 | 26.65 | 22.70 | 26.89 |
| 128 rows | 31.04 | 32.59 | 43.69 | 59.81 |
| 512 rows | 69.92 | 88.46 | 108.23 | 152.36 |
| decode batch 32 (1024 rows) | 121.73 | 159.25 | 194.00 | 287.32 |
| prefill 2048 | 179.91 | 221.06 | 373.16 | 555.47 |
<!-- /TABLE:fabric -->

   Read per spelling, not per shape. For `all_reduce` the two fabrics are close below 512 rows and
   the ring pulls away above it. For **`stack_sum` — the spelling `CCL_MODE="auto"` actually picks at
   the batch-1 decode tile** — the ring is ahead everywhere, including by about 14% at that tile and
   more as rows grow. No row in the whole line-fabric set falls the other way for either spelling, so
   the decode critical path depends on the ring fabric materially rather than marginally: this is a
   load-bearing configuration line, not a default worth inheriting by accident. Round 7 found the
   earlier "tie to within a few tenths" reading taken from the `all_reduce` rows alone.

   The `ccl` table in §5.5's neighbourhood (`logs/probe_ccl.txt`, `CCL` rows) is the *other*
   question: `Topology.Ring` against `Topology.Linear` as the op's argument, both under the ring
   fabric. Ring wins there at every shape in the traced rows, which are the ones a decode step pays.
   Several *eager* rows go the other way — they are dominated by per-call dispatch rather than by the
   fabric — so the artifact's `eager` column is not evidence for either and is not used as any.

   `ttnn.Topology.Ring` is passed to the collectives that still accept it — `all_reduce` and
   `reduce_scatter`. `ttnn.all_gather` marks both `topology` and `num_links` deprecated and
   **ignored** (`all_gather_nanobind.cpp`), so the shipped decode collective takes the ring from the
   fabric config alone.
3. **Fabric packet payload: 8192 B, not the 4352 B build default.** The layer's two collectives
   carry **different dtypes** — the token mixer's is `bfloat16` and the MoE's is `bfloat8_b`, the
   inherited routed-expert activation dtype — so their tile pages are 2048 B and 1088 B, and
   `ccl_common.cpp:60-61` computes a different ideal packet for each and warns from `:64-70`, which is the `ccl_common.cpp:70` tag the census records: `min(15232/page, 4) * page` gives
   8192 B for the first and 4352 B for the second. One fabric setting has to serve both, so the
   runtime warns about whichever one it is not: at the build default it asked for 8192 (864 warnings
   in a suite log, which review round 5 found unclassified), and at the shipped 8192 it asks for 4352
   on the block-float pages instead. **The warning cannot be driven to zero**; the question is which
   setting is faster, and that is measured on both dtypes:

<!-- TABLE:packet -->
| shape | arm | bf16 8192 B (shipped) | bf16 4352 B | bfp8 8192 B (shipped) | bfp8 4352 B |
|---|---|---|---|---|---|
| decode tile | `stack_sum` | 15.84 | 17.17 | 14.82 | 14.77 |
| decode tile | `all_reduce` | 22.11 | 22.08 | 21.45 | 21.45 |
| decode batch 32 | `stack_sum` | 194.00 | 233.29 | 153.79 | 154.25 |
| decode batch 32 | `all_reduce` | 121.73 | 135.16 | 86.47 | 85.83 |
| prefill 2048 | `stack_sum` | 373.16 | 447.46 | 299.17 | 299.93 |
| prefill 2048 | `all_reduce` | 179.91 | 196.81 | 152.91 | 152.81 |
<!-- /TABLE:packet -->

   The bf16 collective is where the setting matters and the block-float one is a coin-flip. Counted
   over all 72 traced rows of each, rather than characterised:

<!-- TABLE:packet_census -->
| operand | rows | faster at 8192 B | slower at 8192 B | best gain | worst loss |
|---|---|---|---|---|---|
| bf16 | 72 | 65 | 6 | 18.4% | -0.7% |
| bfloat8_b | 72 | 36 | 33 | 2.7% | -2.2% |
<!-- /TABLE:packet_census -->

   So bf16 gains up to 18% and gives up under a percent on the handful of rows that fall the other way, while
   block-float is **net-neutral** — it splits almost evenly and its extremes are ±2–3% in both
   directions, with no shape favouring either size consistently. The 4352 B the runtime asks for on
   the block-float pages therefore buys nothing, and 8192 B is taken on the bf16 rows. Rounds 5, 6 and
   7 each found a *sentence* about this comparison overstated while the rows were fine, which is why
   the sentence is now a generated census.

   At the layer the difference is small, as a category that is 3–9% of the window implies. Round 6
   pointed out that one build per arm cannot resolve it, so the pair runs **three builds each**:

<!-- TABLE:packet_layer -->
| layer kind | phase | arm | builds | best | spread |
|---|---|---|---|---|---|
| full_attention | decode (traced) | 4352 B (build default) | 3 | 0.502 ms | 0.000 ms |
| full_attention | decode (traced) | 8192 B (shipped) | 3 | 0.500 ms | 0.004 ms |
| full_attention | prefill 2048 | 4352 B (build default) | 3 | 28.48 ms | 0.02 ms |
| full_attention | prefill 2048 | 8192 B (shipped) | 3 | 28.38 ms | 0.30 ms |
| linear_attention | decode (traced) | 4352 B (build default) | 3 | 0.612 ms | 0.000 ms |
| linear_attention | decode (traced) | 8192 B (shipped) | 3 | 0.610 ms | 0.000 ms |
| linear_attention | prefill 2048 | 4352 B (build default) | 3 | 29.36 ms | 0.57 ms |
| linear_attention | prefill 2048 | 8192 B (shipped) | 3 | 29.06 ms | 0.17 ms |
<!-- /TABLE:packet_layer -->

   Decode's best-of-three is 2 us a step faster at 8192 B on both layer kinds; the arms are cleanly
   separated on linear_attention and overlap by one build on full_attention (the generated table's
   `spread` column shows which). Prefill is a
   wash: the two arms' three-build ranges overlap on both layer kinds, which is what the sweep-to-
   sweep prefill spread predicts. So the layer-level claim is "faster at decode, indistinguishable at
   prefill, never slower", and the isolated collective rows above are where the size of the effect is
   actually visible:

   It is a *fabric* setting, applied before `ttnn.open_mesh_device` by whoever opens the mesh, so the
   module exposes it as `DEFAULT_FABRIC_PACKET_BYTES` / `fabric_router_config()`, the suite passes it
   through `device_params["fabric_router_config"]`, and every probe and benchmark here sets it the
   same way. `ab_single_vs_multichip.txt` carries the layer-level arms; a caller that forgets it gets
   a correct but slightly slower layer.
4. **Per-device DRAM is not the constraint.** 31.75 GiB allocatable per chip against a 0.31 GiB
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
  latency-critical path, to save part of one. (Their cost is inferred from the shapes rather than
  measured — the stats all-gather is a narrow tensor this stage never built. It is not what the
  rejection rests on: §5.10 measures the *consumer* side of a sharded residual directly, and that
  alone is net worse. The unmeasured norm cost only adds to it.)
* The MoE router's top-8 is a decision over all 256 experts. Under a sharded residual the router
  input is a quarter of the hidden dimension on each device, so the routing matmul needs the full
  width gathered back — reintroducing exactly the collective the sharded residual removed, at the
  point in the layer where it cannot be overlapped.
* `logs/probe_ccl.txt` measures the sharded-residual family directly: `rs_only_*` is the
  reduce-scatter half alone. At the decode tile it is about two thirds of the full all-reduce — see
  the `rs_only Ring` and `all_reduce Ring` columns of §2.2's generated table, which is the entire
  budget a sharded residual has to pay for itself out of, twice a layer. The distributed-norm pair
  plus the router gather costs more than that, and the same table's `rs_ag` column shows the round
  trip back to a replicated tensor is the **same number** as the stable all-reduce — within a couple of
  tenths of a microsecond at every shape, i.e. inside the probe's own repeatability —
  because the stable all-reduce lowers to exactly that pair.

**Every collective spelling, at every shape the layer produces, traced microseconds** (the same rows
work log §3 tabulates; `Ring`/`Linear` here are the ops' `topology` argument, all under the ring
fabric):

<!-- TABLE:ccl -->
| shape | `all_reduce` Ring | `rs_ag` Ring | `rs_only` Ring | `all_reduce` Linear | `stack_sum` | `async` |
|---|---|---|---|---|---|---|
| decode (batch 1, 32 rows) | 22.11 | 22.10 | 14.44 | 26.49 | 15.84 | 33.45 |
| 64 rows | 25.94 | 25.95 | 16.42 | 30.77 | 22.70 | 38.61 |
| 96 rows | 27.68 | 27.67 | 16.73 | 34.09 | 29.29 | 43.17 |
| 128 rows | 31.04 | 31.03 | 18.44 | 37.99 | 43.69 | 48.01 |
| 256 rows | 42.55 | 42.54 | 24.62 | 53.16 | 76.60 | 65.97 |
| 512 rows | 69.92 | 70.06 | 37.33 | 91.98 | 108.23 | 103.35 |
| decode batch 32 (1024 rows) | 121.73 | 121.90 | 63.30 | 156.54 | 194.00 | 177.84 |
| prefill 2048 | 179.91 | 180.07 | 102.35 | 250.68 | 373.16 | 325.00 |
<!-- /TABLE:ccl -->

So the layer pays **exactly two collectives**, one after the token mixer and one after the MoE, and
`test_collectives_per_forward` pins that count and their positions by intercepting the CCL entry
points. Nothing else crosses the fabric — not attention, not the router, not the norms. (The spy
covers `ttnn.all_reduce`, `all_gather` and `reduce_scatter`, which is every collective this module
calls; it is a guard against this layer growing a third one, not a proof that no other TTNN entry
point could move data. The four profiler captures are that proof: their only fabric ops are the two
per layer.)

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
* the k/v projection rows are duplicated across each sharing pair — 256 extra weight columns of
  2048 rows at bfloat8_b in `attn_in`, i.e. 557 056 B per layer per device
  (`projection_weights` `vs_ideal = 1.077`);
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
| Sharded (reduce-scatter) residual | needs distributed RMSNorm (2 extra collectives) **and** a router-input gather; the reduce-scatter half saves about a third of one collective and the additions cost more than two | `logs/probe_ccl.txt` `rs_only_*` vs `all_reduce_*`; §2.2 |
| Line fabric (`FABRIC_1D`) | slower on every traced row of both spellings; the shipped `stack_sum` at the batch-1 decode tile is about 14% behind and the gap grows with rows (§2.1) | `logs/probe_ccl.txt` `CCLFAB` rows |
| `Topology.Linear` as the collectives' argument, under the ring fabric | slower at **every** measured traced shape | `logs/probe_ccl.txt` `CCL` rows |
| The fabric's build-default 4352 B packet payload | §2.1's generated census counts every traced row of both operand dtypes rather than characterising them: bf16 favours 8192 B on the large majority, by up to 18%, giving up under a percent on the handful that fall the other way; block-float splits almost evenly with ±2–3% extremes and no consistent direction. So 8192 B is taken on the strength of the bf16 rows and costs nothing measurable on the block-float ones (§2.1) | `logs/probe_ccl.txt` `CCLPKT` rows; `ab_single_vs_multichip.txt` `multichip-build-default-packet` |
| `ttnn.experimental.all_reduce_async` (DRAM and L1 operands) | correct here, and **1.5-1.9x slower than the shipped arm at every measured shape** | `logs/probe_ccl.txt`, `all_reduce_async` / `all_reduce_async_l1` rows |
| Fused `matmul_reduce_scatter_async` at the row-parallel boundary | **slower than the unfused equivalent at every shape** with the replicated residual — §5.10's generated table has all three. The fused op also needs its matmul confined to a sub-grid so the collective's workers get free cores, which is part of why | `logs/probe_fused_ccl.txt` `o_proj` rows |
| A **sharded** residual, with the fused producer and a gathered consumer | the fused producer alone (`fused_mm_rs_only`) does beat the shipped arm at 2 of the 3 shapes, and the consumer pays it back and more: at the `attn_in` boundary a gathered input costs about a third more than the replicated one at both prefill and batch 32. Net worse at every shape, at one of the two column-parallel consumers | `logs/probe_fused_ccl.txt` both boundaries |
| Fused `all_gather_matmul_async` at the column-parallel boundary | **hung the mesh.** `AllGatherMatmulAsyncDeviceOperation` on `[1,1,32,512] x [1,1,2048,2560]` sat on all four devices and 40 cores with no progress and had to be killed; devices recovered after a `tt-smi -r`. Exact op-contract blocker, not a slow arm | `doc/multichip_decoder/triage/`, and the arm is off by default in `probe_fused_ccl.py` |
| Explicit `reduce_scatter` + `all_gather` | the same number as `ttnn.all_reduce` at every shape, within the probe's repeatability — the stable all-reduce lowers to exactly that pair | `logs/probe_ccl.txt` `rs_ag_*`; `ab_layer_knobs.txt` `ccl rs_ag` |
| Sharding the MoE **intermediate** (512-wide) instead of expert parallelism | keeps 8 `sparse_matmul` loop iterations and cuts `Nt` from 32 to 8 tiles — removes parallelism the op is already short of; the `[1, E, tokens, 2048]` down output does not shrink at all. about 2.1x slower than EP at decode and 2.5x at prefill (§3's generated ratio table, at the conservative basis; each arm at **its own** representative active count, which is what the layer runs) | `logs/probe_expert_parallel.txt` `tp` vs `ep` rows |
| No expert parallelism at all (all 256 experts on every device) | about 2.9x slower than EP at decode (§3's generated ratio table). This is the `single` arm; it is still *gate-selected*, so it prices EP, not density | `logs/probe_expert_parallel.txt` `single` rows |
| Replicating all 256 experts on every device | pointless here: EP already fits with 102x DRAM headroom, and replication would quadruple expert weight traffic for zero parallelism gain | §6 |
| `ttnn.gather` for the local-expert narrowing | 57–58 us/step slower at the layer than the one-hot selection matmul, and both are bit-equal | `logs/ab_layer_knobs.txt` `routing` arms; `test_routing_select_modes_agree` |
| Inheriting the single-chip decode matmul geometry | 23–24 us/step slower on linear_attention, 14 us/step on full_attention | `logs/ab_layer_knobs.txt` `geometry` arms |
| Inheriting the single-chip routed sparse-matmul core rule | **4.0–4.6 ms/layer** slower on 2048-token prefill, and 16–71 us/step slower at decode batch 4 and above | `logs/ab_layer_knobs.txt` `sparse` arms and `logs/probe_decode_batch.txt` `SPARSEB` rows; `logs/probe_sparse_matmul_local.txt`; §5.7 |
| Casting the MoE's block-float output to bf16 before its collective | costs a few microseconds on every decode step (3–4 us, §5.5's generated table) and moves prefill by nothing, despite the profiler attributing ~1500 us to that collective — see §5.8, where the null result doubles as the control for that anomaly | `logs/ab_layer_knobs.txt` `cast` arms |

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

**Why expert parallelism and not a sharded expert intermediate.** `logs/probe_expert_parallel.py`
times the inherited `_routed_experts` chain — the packed gate/up `sparse_matmul`, the unpacking
slices, the SwiGLU and score multiplies, the down `sparse_matmul` and the expert reduction — under
all three arrangements, each at the active-expert count it actually runs at (traced microseconds):

<!-- TABLE:moepar -->
| phase | arm | experts/device | active/device | us |
|---|---|---|---|---|
| decode | unsharded, still gate-selected | 256 | 8 | 665.74 |
| decode | **expert parallelism** (shipped) (mean local) | 64 | 2 | 198.10 |
| decode | **expert parallelism** (shipped) (>= the expected maximum, 3.512) | 64 | 4 | 204.92 |
| decode | intermediate sharded 4 ways | 256 | 8 | 471.95 |
| prefill | unsharded, still gate-selected | 256 | 162 | 1637.17 |
| prefill | **expert parallelism** (shipped) | 64 | 41 | 513.29 |
| prefill | intermediate sharded 4 ways | 256 | 162 | 1298.64 |
<!-- /TABLE:moepar -->

The ratios those rows produce, computed from the artifact rather than transcribed — rounds 4, 5 and 6
each found a hand-derived version of them wrong:

<!-- TABLE:moepar_ratios -->
| comparison | basis | ratio |
|---|---|---|
| EP vs intermediate-sharded | decode, at the expected maximum (conservative) | **2.30x** |
| EP vs unsharded, gate-selected | decode, at the expected maximum (conservative) | **3.25x** |
| EP vs intermediate-sharded | decode, at the mean local count | **2.38x** |
| EP vs unsharded, gate-selected | decode, at the mean local count | **3.36x** |
| EP vs intermediate-sharded | prefill | **2.53x** |
| EP vs unsharded, gate-selected | prefill | **3.19x** |
<!-- /TABLE:moepar_ratios -->

The EP arm is priced at two points because the two answer different questions: the mean local count
is what an average step runs, and the expected maximum over the four devices (3.512) is what the
collective barrier waits for. The ratios above and in §2.5 are quoted at the conservative one — review
round 4 asked for arms priced on one basis and round 5 found only the mean in the table.

The active counts differ between arms because expert parallelism is precisely what changes them: a
32-token group makes `32 * 8 = 256` draws globally, of which a quarter reach any one device, so the
expected local union is ~41 of 64 against ~162 of 256 for the arms that hold every expert. Holding
the count fixed across arms would price a workload neither arrangement has; review round 2 corrected
this stage's arithmetic for that division and round 4 found the EP arm of this probe still sweeping
the undivided count, which is why 63 remains in the sweep next to the corrected 41.

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

<!-- TABLE:pcc_baseline -->
| test | values | minimum |
|---|---|---|
| `test_prefill_matches_single_chip` | 18 | 0.999913 |
| `test_decode_matches_single_chip` | 16 | 0.999892 |
| **total** | **34** | **0.999892** |
<!-- /TABLE:pcc_baseline -->

(`test_prefill_matches_single_chip` is 9 lengths x 2 layer kinds; `test_decode_matches_single_chip`
is 4 steps x 2 prefill lengths x 2 kinds. Both counts and both minima are counted out of the
committed suite log by `logs/make_tables.py`, not transcribed.)

Lengths: `1, 7, 32, 128, 129, 250, 2048, 2049, 3000`. Prefill lengths for decode: 130 (decode writes
cross a 64-token page boundary) and 2048 (past an internal prefill chunk boundary).

### 4.2 Against the float32 HF golden

Bar `PCC_BAR = 0.995`, inherited from the functional stage and not lowered. Every value the suite
prints against the golden, counted by the test that printed it:

<!-- TABLE:pcc_inventory -->
| test | values | minimum |
|---|---|---|
| `test_batched_prefill_decode_pcc` | 32 | 0.999900 |
| `test_prefill_pcc` | 20 | 0.999851 |
| `test_decode_pcc` | 16 | 0.999923 |
| `test_traced_decode_pcc` | 6 | 0.999872 |
| `test_batched_decode_ragged_positions` | 4 | 0.999909 |
| `test_long_context_pcc` | 4 | 0.999866 |
| `test_unaligned_max_context` | 4 | 0.999869 |
| `test_permuted_page_table` | 2 | 0.999903 |
| `test_prefill_continuation` | 2 | 0.999911 |
| **total** | **90** | **0.999851** |
<!-- /TABLE:pcc_inventory -->

This breakdown is generated. Review round 1 found the hand-written version's total wrong, the fix
recounted it once by script, and round 3's two new assertions in `test_unaligned_max_context` then
made it wrong again — a two-digit integer that no figure audit can see. `logs/make_tables.py`
carries the per-test patterns, so adding an assertion updates the table.

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
| `test_ccl_modes_agree` | `all_reduce` / `rs_ag` / `stack_sum` agree with the shipped `auto` on **both sides of the crossover**, which means both *phases*: prefill pads every chunk to `PREFILL_ALIGN` = 128 rows, so no prefill shape can reach the `stack_sum` regime, and decode's 32-row tile is where the layer actually ships it. The test re-derives which spelling `auto` picked from `_physical_rows` and asserts the decode case really is `stack_sum`, so it cannot quietly stop covering the crossover again |
| `test_routing_select_modes_agree` | `gather` and `select_matmul` give identical layer output |
| `test_decode_runs_the_multichip_program_configs` | the re-swept geometry actually reaches `attn_in`/`gdn_in`/`shared_down`, and its `in0_block_w` stays compatible with the residual norm's shard carry |
| `test_chunk_size_invariance` | a 6000-token prefill run with internal chunk 2048 and 1024 agrees to PCC **1.000000** — the control for the long-prefill path, where the HF golden is intractable |
| `test_sparse_cores_match_the_local_sweep` | the routed sparse matmuls run on the core counts §5.7's sweep selected, at every advertised decode batch (8/8 at batch 1, 16/16 at 2, 32/32 from 4 up) and at prefill (32/32, where the bound saturates) |

### 4.4 Paged KV cache on the target mesh

* `test_permuted_page_table` — a shuffled block mapping, prefill PCC 0.999914 / decode 0.999903.
* `test_batched_prefill_decode_pcc[1,4,13,32]` x `seq_len ∈ {192, 130}` — disjoint per-user block
  spans, up to the advertised batch bound, at a tile-aligned and a non-aligned prefill length.
* `test_batched_decode_ragged_positions` — 4 users at **distinct** absolute positions
  (37, 130, 200, 64) over disjoint per-user block spans, each compared to its own HF golden. The
  block mapping here is contiguous per user; the shuffled-mapping case is `test_permuted_page_table`.
* `test_prefill_continuation` — a chunked prefill resumed on a chunk boundary and **ending** at a
  non-aligned length (128 then 130 tokens), checked against the HF golden *and* against a single
  258-token TTNN prefill in the same process. `start_pos` itself must be a multiple of `chunk_size`:
  that is an inherited restriction of `OptimizedDecoder.prefill_forward`, this stage neither adds to
  it nor relaxes it, `test_prefill_continuation_rejects_unaligned_start` pins it as a clean
  `ValueError`, and `doc/context_contract.json` records it. Review round 4 found this bullet
  claiming the opposite of what the API permits.
* The page table is **replicated** and indexes blocks; the head axis is what is sharded, so the page
  table contract is byte-for-byte the single-chip one.

### 4.5 Non-aligned lengths after sharding

This is the property multichip padding most easily breaks, so it is asserted at three levels:

* **public prefill lengths** — `1, 7, 129, 250, 2049, 3000` all compared to the single-chip baseline
  (§4.1). None is a multiple of the 32-token tile, the 64-token page, the 128-token physical
  alignment, or the 2048-token internal chunk.
* **the row fold** — `CCL_COMPACT_ROWS` is a **decode-only** path: a prefill operand is already
  `PREFILL_ALIGN`-padded, so its physical row count equals `align_up(b*t, 32)` and the fold's guard
  is false at every prefill shape. Round 8's correctness audit found an earlier claim that a batched
  non-aligned prefill exercised it; the decode batches (§5.9) are where it is covered.
* **`max_context`** — `test_unaligned_max_context` builds with `max_context=5000`, prefills 5000
  tokens and decodes at the last slot, both checked against the HF golden (it asserted shape, finiteness
  and variance only until review round 3, which pointed out the golden is tractable at that length).
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

<!-- TABLE:bench -->
| layer kind | phase | single-chip | 1x4 replication control | multichip | speedup | efficiency |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.76 ms | 101.95 ms | **28.96 ms** | **3.514x** | 87.8% |
| linear_attention | decode (traced) | 1.030 ms | 1.032 ms | **0.610 ms** | **1.689x** | 42.2% |
| full_attention | prefill 2048 | 95.37 ms | 95.99 ms | **28.99 ms** | **3.290x** | 82.2% |
| full_attention | decode (traced) | 0.827 ms | 0.827 ms | **0.500 ms** | **1.654x** | 41.3% |
<!-- /TABLE:bench -->

The replication control is within +0.3% of the single-chip arm on every row, so none of the speedup
is a mesh-dispatch artifact.

Both claims are gated inside the suite, not only reported by the bench harness:
`test_multichip_beats_single_chip_traced_decode` fails below `DECODE_SPEEDUP_BAR = 1.4x` (measured
1.65–1.69x) and `test_perf_prefill` fails above `PREFILL_MS_BAR = 48 ms` (measured 28–29 ms against a
95–102 ms single-chip baseline). Review round 4 found the first asserting only `> 1.0` and the second
asserting nothing at all; the bars are set well clear of run-to-run noise, so they fail a lost
parallelisation rather than a slow afternoon.

The full_attention prefill row is the one that moves most between sweeps; the `ab_layer_knobs.txt`
prefill columns show the same spread on every arm, which is how it is known to be build-to-build
rather than a property of any one choice. The figure quoted is this sweep's; the spread is build-to-build
and is visible in the `ab_layer_knobs.txt` prefill columns, where every arm covers a similar range.

### 5.3 Why decode efficiency is 41–42% and prefill is 82–88%

Prefill is compute-bound and scales close to linearly. Decode at batch 1 is latency-bound: each
device's share of the work shrinks 4x but the per-op launch cost, the fixed collective latency and
the serial op count do not. The profiler says exactly where the remaining time goes.

### 5.4 `tt-perf-report`

Four captures (`tracy/{linear,full}_attention/{prefill,decode}_perf_report.*` — per-op table, CSV and
roofline summary, plus a stacked-by-op-code CSV and PNG; everything over the repo's 500 KB limit or
matching its blanket `*.csv` ignore is committed gzipped). Prefill and decode are captured in
**separate** runs so each Tracy capture holds one signposted window and one device session, and the
routed matmul rows are modelled with `--active-experts` (4 per device at batch-1 decode, 41 for a
32-token prefill group).

**Top of the stack, share of merged 4-device time:**

<!-- TABLE:perf -->
| | linear_attention decode | full_attention decode | linear_attention prefill | full_attention prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 17.74% | 21.61% | 75.06% | 73.99% |
| `TopK` (router) | 9.01% | 9.46% | 0.17% | 0.18% |
| dense `Matmul` (all in0 layouts) | 13.13% | 9.40% | 1.10% | 1.07% |
| **collectives (`AllGather` / `ReduceScatter`)** | **6.49%** | **9.09%** | **2.94%** | **5.73%** |
| all data movement (`DM` category) | 8.18% | 10.92% | 3.21% | 5.73% |
| DRAM roofline (modeled ops) | 6.5% (33 GB/s) | 6.0% (31 GB/s) | 27.1% (139 GB/s) | 27.3% (140 GB/s) |
<!-- /TABLE:perf -->

`--active-experts` — the count `tt-perf-report` needs to model a `sparse_matmul` row at all — is the
**measured per-device, per-group** active count: 4 at batch-1 decode, and 41 for a 32-token prefill
group. 41 is `64 * (1 - (1 - 1/64)^(32*top_k/tp))` = 40.6, and
`test_gate_selected_experts_not_dense` measures 44 (linear) and 39–40 (full) for exactly that
quantity. Review round 2 found this at 63 — the same formula with the draws *not* divided by `tp`,
so every global draw was credited to every device — which inflated the modelled DRAM bytes and FLOPs
of the rows that are 74–75% of the prefill window by about 50%. The roofline figures above are at the
corrected input, and are **not** comparable to any roofline this stage quoted before round 2.

**The whole window, by `Op Category`,** with the single-chip stage's own share of the same phase in
brackets — the control for whether a category is this stage's doing:

<!-- TABLE:category -->
| `Op Category` | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
| `Compute` | 58.3% (72.5%) | 54.6% (71.5%) | 88.1% (92.1%) | 88.0% (94.1%) |
| `TM` (layout) | 22.3% (17.9%) | 19.6% (15.1%) | 4.7% (4.5%) | 4.2% (4.2%) |
| `DM` (data movement) | 8.2% (1.6%) | 10.9% (1.1%) | 3.2% (0.3%) | 5.7% (0.0%) |
| `Other` | 11.3% (8.1%) | 14.9% (12.3%) | 4.0% (3.0%) | 2.1% (1.7%) |
<!-- /TABLE:category -->

The op-code rows above are the ones this stage's decisions reach; the category table is what they sit
in, and it is here because review round 4 pointed out that six op-code rows summing to about 55% of
the decode window were being presented as "the profiler says exactly where the time goes" while
`TM` — layout — was a fifth of that window and appeared nowhere.

`TM` is **inherited, not introduced**: the category table above has this stage's decode share against
the single-chip decoder's on the same phase and the same layer kinds, a few points apart, and the two
are within a few tenths of each other at prefill.
The rows behind it:

<!-- TABLE:tm -->
| op | linear decode | full decode |
|---|---|---|
| `Slice` | 5.31% | 5.57% |
| `UntilizeWithUnpadding` | 4.18% | 3.99% |
| `FillPad` | 2.79% | 2.93% |
| `ReshapeView` | 2.69% | 0.46% |
| `Transpose` | 0.98% | 1.67% |
| `TilizeWithValPadding` | 1.18% | 1.26% |
| `Untilize` | 1.12% | 1.17% |
| `Permute` | 1.04% | 1.09% |
<!-- /TABLE:tm -->

`Slice` is the inherited MoE chain unpacking the packed gate/up product and slicing the fused
`attn_in`/`gdn_in` output; `UntilizeWithUnpadding`, `FillPad`, `TilizeWithValPadding` and `Untilize`
are the inherited paged-cache and norm boundaries. The one row this stage moves is `ReshapeView`
(the `ReshapeView` row of the table above), which includes `CCL_COMPACT_ROWS`' fold around the collective — and §5.9
measures that fold as a **net win** of 85–88 us a step at batch 32, so it pays for its own layout
cost several times over. On the whole category the multichip-minus-single-chip delta is about four
points at decode on both layer kinds, against a `DM` delta of six to ten points that is the
collectives themselves — both readable off the table above, and neither large enough to make layout
this stage's problem rather than the one it inherited.

Two consequences are recorded rather than acted on: the decode window is 20% layout work in the
single-chip stage already, so it is that stage's ledger and not this one's; and it is the reason the
six op-code rows do not sum to the window. Neither changes a decision here.

Findings this drove:

* **Communication is the collectives row of the table above — under 10% of decode and under 6% of
  prefill.** (That row counts the fabric ops.
  `stack_sum`'s local `ttnn.sum` — `FastReduceNC`, 2 ops a step — adds about a point at decode, so
  the all-in decode figure is about a point higher; it is compute, not fabric, and it is the price of
  the spelling that wins at the decode tile.) That is an upper bound on the
  cost of the whole parallelisation — §5.8 shows part of the prefill figure is barrier wait, not data
  movement — and it is small enough that no further collective-shape work would change the headline.
  In decode the two collectives are `AllGather` because `CCL_MODE="auto"` picks `stack_sum` at the
  batch-1 tile (§5.5); in prefill they are `ReduceScatter` + `AllGather`, which is what
  `ttnn.all_reduce` lowers to.
* **The routed experts dominate, and are where this stage's largest win came from.** 74–75% of
  prefill and 18–22% of decode. Every `SparseMatmul` row in all four captures is marked `SLOW`
  (257/257 rows per prefill capture, 128/128 per decode capture). The `1x1 output subblock` advisory
  is a **different** and narrower set, and round 3 found it attributed to the wrong ops here: of the
  257 sparse prefill rows, 64 carry it, 64 carry `output subblock 1x2 look good` and the rest carry
  no subblock advice at all, while **none** of the 128 sparse decode rows carries it — the 1x1
  advisories in the decode captures are on the small *dense* projections, where `per_core_N = 1` at
  the swept core count forces a 1x1 subblock by construction. Following the `SLOW` flag is what led
  to §5.7, which took 4.0–4.6 ms a layer off prefill. What remains sits at a geometry the sweeps
  confirm is the fastest available — both sweeps cover `out_block_w`/`sub_w` and core count — so it
  is the op's own advisory rather than an unexplored candidate.
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

<!-- TABLE:ablayer -->
| knob | arm | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|---|
| `ccl` | **`auto`** (shipped) | 0.610–0.611 | 0.500–0.501 | 29.11–29.31 | 28.86–29.63 |
| `ccl` | `stack_sum` | 0.610–0.611 | 0.500–0.501 | 29.28–30.23 | 28.40–29.56 |
| `ccl` | `all_reduce` | 0.621 | 0.508–0.509 | 28.99–29.10 | 28.32–28.67 |
| `ccl` | `rs_ag` | 0.621 | 0.508 | 29.27–30.09 | 28.25–29.60 |
| `geometry` | **multichip-retuned** (shipped) | 0.610–0.611 | 0.500–0.501 | 28.85–29.38 | 28.35–29.60 |
| `geometry` | single-chip-inherited | 0.635 | 0.516–0.517 | 29.39–30.20 | 28.28–28.94 |
| `routing` | **`select_matmul`** (shipped) | 0.611 | 0.500–0.501 | 28.97–29.83 | 28.36–28.76 |
| `routing` | `gather` | 0.668–0.669 | 0.559–0.560 | 29.36–30.66 | 28.89–30.00 |
| `sparse` | **tp-rescaled** (shipped) | 0.610–0.611 | 0.500–0.501 | 29.08–29.36 | 28.21–29.49 |
| `sparse` | single-chip-inherited | 0.610–0.611 | 0.500–0.501 | 33.17–33.24 | 32.32–33.17 |
| `cast` | **block-float** (shipped) | 0.611 | 0.500–0.501 | 28.91–30.35 | 28.43–29.05 |
| `cast` | `bf16` | 0.614 | 0.504–0.505 | 29.07–29.29 | 28.34–28.70 |
<!-- /TABLE:ablayer -->

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

<!-- TABLE:dense -->
| role | inherited (realised) | us | local winner | us | delta | winner spread | repeatability | shipped |
|---|---|---|---|---|---|---|---|---|
| `attn_in` | (32,2) → 33/2 | 30.77 | 110/8 | 17.77 | 13.00 | 0.13 | 0.26 | **retuned (110, 8)**, 17.77 |
| `gdn_in` | (110,2) → 110/2 | 48.84 | 33/8 | 20.36 | 28.48 | 0.04 | 0.19 | **retuned (110, 8)**, 20.52 |
| `shared_down` | (48,16) → 55/4 | 9.24 | 4/4 | 8.62 | 0.62 | 0.19 | 1.93 | **retuned (8, 4)**, 8.67 |
| `expert_select` | new role | — | 8/8 | 8.29 | — | 0.72 | 1.89 | **(8, 8)**, 8.29 |
| `o_proj` | (16,16) → 22/16 | 9.36 | 22/8 | 9.26 | 0.10 | 0.33 | 1.41 | inherited |
| `gdn_out` | (24,8) → 33/8 | 9.54 | 22/8 | 9.31 | 0.23 | 0.07 | 0.11 | inherited |
| `shared_in` | (32,32) → 33/32 | 8.73 | 33/32 | 8.73 | 0.00 | 0.49 | 0.19 | inherited |
| `router` | (32,32) → 33/32 | 9.07 | 66/32 | 8.87 | 0.20 | 0.45 | 0.10 | inherited |
<!-- /TABLE:dense -->

`repeatability` is this probe's own noise floor for that role, read out of the same file: several
distinct core *targets* collapse onto the same realised grid (16 and 24 both land on 22 and 33), so
the file contains independent repeat measurements of identical configs, and `repeatability` is the
widest disagreement between such repeats. A delta smaller than it is not a result.

Two retuned rows ship at a grid that is not this sweep's local winner, and both are deliberate.
`shared_down` is limitation 3 — an op-level win the layer cannot resolve. `gdn_in` ships at the same
`(110, 8)` as `attn_in` rather than at whichever grid this sweep's ladder puts first: the two are a
fraction of a microsecond apart — read the `us`, `delta` and `repeatability` columns of the row above,
where the gap is at or inside the noise floor — and worth a few hundredths of a percent of a decode
step, and the winning grid for this role has moved between sweeps while `(110, 8)` has been within a
few tenths of the winner in all of them. One grid for both in-projections is also what lets the
residual norm's shard carry reach them on the same core set (§5.6's `in0_block_w` bound). Review
round 5 asked for it to be stated rather than left to the table.

All four inherited rows sit inside both the winner's own spread and the probe's repeatability in this
sweep, and keep the inherited entry. Across the many sweeps this stage ran, their deltas moved
between 0.00 and 0.56 us with the winning core count wandering between 8, 22, 33 and 88 and no
consistent sign, which is why the `repeatability` column exists at all: judging these rows against a
single sweep's spread is what put `shared_down` on the wrong side of the line in round 1.

The retuned rows are outside that band by a wide margin. `attn_in` and `gdn_in` are the two wide
in-projections — the inherited cap of 2 is the *worst* legal value for the local shape, costing 1.7x
and 2.4x on the op — and together they move the layer by 24 us/step on linear_attention and 14
us/step on full_attention (§5.5 `geometry` arm). `shared_down` is retuned because TP=4 cuts its `K`
from 512 to 128 — 4 tiles — and a 55-core grid for a 4-tile `K` is launch overhead rather than
parallelism. It ships at **8** cores, not the 4 an earlier round chose: 4 measured 8.44–8.62 us on
three sweeps and about 10 us on another, while 8 has never read above 9 us in any sweep, so 8 is the
stable choice and is never behind. At the layer the whole `geometry` arm moves by ≤0.002 ms
for this row, inside the build-to-build spread (see §8).

`in0_block_w` is additionally bounded above by the residual norm's per-core shard width where the
shard is carried into the projection: the 2048-wide norm over 8 cores gives 8 tiles per core, so 8 is
the largest cap that keeps `mcast_in0`'s `block_w % in0_block_w == 0` legal. Caps of 16 and 32 do
build against an interleaved `in0` and are measured — `probe_dense_matmul.txt`'s `attn_in` rows read
progressively slower at caps 16 and 32 — so the shipped value is the measured winner as well as the largest legal
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
decode batch **and at prefill** so the table above cannot drift from the code again; round 3 found
the prefill row derivation-only (the spy used to be installed after `prefill_forward` returned) and
that `probe_sparse_matmul_local.py`'s `# SHIPPED` header still encoded the round-1 assumed-active
basis. The header now prints the bound explicitly, one line per group row count, and the measured
prefill grid is 32/32 on both layer kinds.

`logs/probe_sparse_matmul_local.py` re-runs the single-chip sweep's candidate ladder at the
per-device operating points (`E = 64`, `active ∈ {4, 8, 16, 32, 41, 63}`) under the same BFP4/LoFi
policy. Best microseconds per realised core count:

<!-- TABLE:sparse_ladder -->
| active | 4 cores | 8 cores | 16 cores | 32 cores | 64 cores | winner |
|---|---|---|---|---|---|---|
| 4, `down` | — | **52.3** | 59.6 | 66.7 | 85.4 | 8 |
| 4, `gate_up` | 82.7 | **62.5** | 64.2 | 70.1 | — | 8 |
| 8, `down` | — | 62.2 | **61.8** | 68.9 | 88.9 | 16 |
| 8, `gate_up` | 122.5 | 81.8 | **73.2** | 78.0 | — | 16 |
| 16, `down` | — | 74.4 | **65.2** | 72.4 | 92.4 | 16 |
| 16, `gate_up` | 180.6 | 109.5 | **86.7** | 90.0 | — | 16 |
| 32, `down` | — | 120.0 | 83.8 | **80.3** | 102.3 | 32 |
| 32, `gate_up` | 329.4 | 192.7 | 126.3 | **120.1** | — | 32 |
| 41, `down` | — | 146.3 | 96.1 | **85.9** | 107.5 | 32 |
| 41, `gate_up` | 414.6 | 240.2 | 150.2 | **139.3** | — | 32 |
| 63, `down` | — | 236.3 | 144.8 | **105.1** | 125.0 | 32 |
| 63, `gate_up` | 691.7 | 398.4 | 234.8 | **200.2** | — | 32 |
<!-- /TABLE:sparse_ladder -->

The ladder's winner tracks the shipped rule: 8 cores at the batch-1 decode point (measured active 4),
16 in the middle, 32 from active 32 up — which includes the corrected prefill point of 41 and the
over-counted 63 the earlier calibration used, so the prefill geometry decision is the same either way.

The two `—` cells are structural, not gaps in the sweep: `gate_up` produces `Nt = 32` output tiles so
64 cores cannot be realised, and `down`'s 4-core point is below `SPARSE_MIN_CORES = 8`, which the
shipped rule clamps to and therefore never asks for. Every core count the rule can select is
measured at every operating point it can select it at.

Both effects are then measured at the layer rather than argued from the ladder:

* **prefill** — the `sparse` rows of §5.5's generated table: about 4 ms a layer on both kinds,
  and the difference between roughly a 3.0x and a 3.5x prefill speedup.
* **decode** — `probe_decode_batch.txt` `SPARSEB` rows, three builds per arm per batch, generated
  into §5.9's table: a tie at batch 1 by construction, and tens of microseconds a step in the
  shipped rule's favour at every batch above it, on both layer kinds. Round 2 predicted this range
  might be *adverse*, because the isolated ladder
  favours 8 cores at low active counts; the layer says otherwise at every batch except
  linear_attention's batch 2, where the arms tie to within a microsecond. That is why the layer A/B
  decides and the ladder only shapes the candidate.

### 5.8 The block-float collective — an anomaly, and its control

The prefill profile shows the layer's two collectives at wildly different cost for the same logical
shape: `tracy/full_attention/prefill_perf_report.txt` has the BF16 reduce-scatter at ~100 us on 20
cores and the BFP8 one **an order of magnitude above it** on 12 cores — about 5% of the whole prefill window on its own, and
the largest non-sparse item in it. (The stacked report's single `ReduceScatter` row is the two
summed, which is why it reads higher.) The only
difference between them is the operand dtype: the MoE half produces `bfloat8_b` under the inherited
precision policy.

That reads as an obvious win, and it is not one. `CCL_CAST_BLOCKFLOAT` casts the operand up to
`bfloat16` before the collective; the `cast` arm of `ab_layer_knobs.txt` measures it at the layer.
Casting up **costs a few microseconds on every decode step — 3 us on linear_attention and 4 on full_attention in the table above — and moves warmed prefill by nothing** — the arms'
three-build ranges overlap on both layer kinds. It is therefore off.

The null result is the useful part: it is the control that classifies the anomaly. Removing the
block-float operand changes the block-float row's cost and not the layer's, which means the
profiler's ~1500 us is not data movement this layer pays. The remaining explanation is the
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

<!-- TABLE:decode_batch -->
| batch | mixer physical rows | off | on | delta |
|---|---|---|---|---|
| 1 | 32 | 0.500 / 0.611 | 0.500 / 0.610 | +0 / +1 us |
| 2 | 64 | 0.578 / 0.704 | 0.575 / 0.703 | +3 / +1 us |
| 4 | 128 | 0.634 / 0.816 | 0.639 / 0.820 | **-5 / -4 us** |
| 8 | 256 | 0.787 / 1.002 | 0.783 / 1.000 | +4 / +2 us |
| 13 | 416 | 1.619 / 2.061 | 1.608 / 2.051 | +11 / +10 us |
| 16 | 512 | 1.827 / 2.271 | 1.796 / 2.243 | +31 / +28 us |
| 32 | 1024 | 3.018 / 3.945 | 2.932 / 3.863 | **+86 / +82 us** |
<!-- /TABLE:decode_batch -->

(full_attention / linear_attention; positive delta = the fold is faster.) At batch 1 the guard skips
the fold entirely, because `[1, 1, dim]` is already one tile row. It ships always-on above that; the
single 4 us regression at batch 4 is explained and quantified in §8.

The same probe carries the `SPARSEB` rows §5.7 uses, so both round-1/round-2 findings are measured on
one harness at one set of batches — the sparse core rule's decode effect, at every advertised batch
(full_attention / linear_attention ms per step; positive delta = the shipped rule is faster):

<!-- TABLE:sparse_batch -->
| batch | inherited rule | shipped rule | delta |
|---|---|---|---|
| 1 | 0.500 / 0.611 | 0.500 / 0.611 | +0 / +0 us |
| 2 | 0.608 / 0.702 | 0.575 / 0.703 | +33 / -1 us |
| 4 | 0.689 / 0.849 | 0.639 / 0.820 | +50 / +29 us |
| 8 | 0.806 / 1.015 | 0.783 / 1.000 | +23 / +15 us |
| 13 | 1.646 / 2.078 | 1.607 / 2.051 | +39 / +27 us |
| 16 | 1.845 / 2.281 | 1.796 / 2.243 | +49 / +38 us |
| 32 | 3.003 / 3.928 | 2.933 / 3.862 | +70 / +66 us |
<!-- /TABLE:sparse_batch -->

### 5.10 The fused matmul+CCL family

`logs/probe_fused_ccl.py` prices the two ops `$multichip` requires in the topology table and rounds
0-2 of this stage skipped: `matmul_reduce_scatter_async` at the row-parallel `o_proj` boundary and
`all_gather_matmul_async` at the column-parallel `attn_in` one, at the real per-device shapes, traced.
The artifact's `# ARM` header block records each arm's residual layout in and out, where the
collective sits relative to the matmul, the CCL dtype, and the persistent buffers the async ops need.

Traced microseconds:

<!-- TABLE:fused -->
| boundary | arm | decode | decode batch 32 | prefill 2048 |
|---|---|---|---|---|
| `o_proj` | matmul + `all_reduce` | 40.78 | 185.41 | 293.02 |
| `o_proj` | matmul + `stack_sum` (shipped at the decode tile) | 36.41 | 254.64 | 487.79 |
| `o_proj` | **fused** matmul+reduce-scatter, then all-gather | 49.37 | 201.68 | 328.31 |
| `o_proj` | **fused**, no gather — the sharded-residual bound | 42.78 | 149.37 | 263.44 |
| `attn_in` | matmul on a replicated input (shipped: no collective) | 43.55 | 146.70 | 258.95 |
| `attn_in` | all-gather + matmul — what a sharded residual would pay | 50.77 | 204.45 | 335.59 |
<!-- /TABLE:fused -->

Three readings, all of which keep the shipped contract:

* fusing the reduce-scatter into the row-parallel matmul is **slower than the unfused pair at every
  shape** once the residual has to come back replicated;
* the fused producer *without* the gather — the sharded-residual lower bound — does beat the shipped
  arm at the two larger shapes, and the consumer row underneath it gives that back and more: an
  `attn_in` that must all-gather its input costs materially more than one reading an already
  replicated residual. That is the same conclusion §2.2 reaches from the unfused rows, now measured
  with the fused producer in hand rather than argued;
* `all_gather_matmul_async` **hung the mesh** at the decode shape and is recorded as an exact
  op-contract blocker with a `tt-triage` capture (§8, `triage/`), not as a slow arm. It is off by
  default in the probe so a repeat cannot take a whole sweep with it.

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
1.589 is dominated by two deliberately replicated terms — the RoPE tables (67 633 152 B, replicated
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
over the state-, trace- and collective-critical subset: **50 passed, 3 skipped**
(`logs/watcher_pytest.txt.gz`). `watcher/census.py` partitions all 59 027 log lines into disjoint
buckets — an unknown line kind lands in `UNCLASSIFIED` and trips an assert rather than hiding in a
catch-all — and reports **0 fatal-class matches** over 54 dumps, minimum stack headroom 1332 bytes
free over 20 detail lines (`watcher/census_summary.txt`). The stack figure is narrower than the
fatal-class one and is reported as such: the summary's `stack reporting cores: 1` says those detail
lines come from a single core, so it is a spot check on stack depth, not a survey. The fatal-class
result covers every dump.

**Runtime warnings are censused, not ignored.** `logs/warning_census.py` reads the committed suite
and watcher logs and writes one line per distinct warning class with its count
(`logs/warning_census.txt`). Three classes appear in every sweep and a fourth intermittently, all classified:

* `Fabric packet size 8192 B is suboptimal for transporting 1088 B pages. Configure 4352 B` — the
  block-float half of the packet trade-off, measured and taken deliberately (§2.1, limitation 6). It
  is the one warning on this stage's own critical path, and it cannot be driven to zero because the
  layer's two collectives want different packet sizes.
* `Unable to set process priority to 0, error code: -1` — the runtime asking for a priority this
  container does not grant. Environmental, identical on the single-chip stage.
* `Unknown motherboard 'B850M-C' … falling back to bus_id as tray_id` — host discovery on a board
  the topology table does not list. Cosmetic: this stage opens one 1x4 mesh and never uses tray ids.
* `AICLK settled at ... within 5% of the requested ...` — the ARC clock landing a fraction of a
  percent below the request. Environmental, at most once per session, and it does not appear in every
  sweep's census; when it does it applies equally to the single-chip baseline arm measured in the
  same sweep, so it cannot bias a speedup.

Review rounds 5 and 6 each turned on a warning that was sitting unread in a committed log, which is
why the classes are now an artifact that a diff will show.

`TT_METAL_WATCHER_DISABLE_ETH=1` is a **hard tool limit here, not a choice**. With watcher
instrumenting the ACTIVE_ETH cores, the 1D-fabric ERISC program grows to 28 656 B against a 25 600 B
ACTIVE_ETH kernel config buffer on Blackhole, so every test in the subset fails at `mesh_device`
setup before any model code runs:

```
TT_FATAL: Program size (28656) too large for kernel config buffer (25600) on ACTIVE_ETH (assert.hpp:104)
```

The control is re-run from the committed bytes on every sweep and preserved verbatim as
`logs/watcher_pytest_eth_enabled.txt`. **It is itself unstable, which is part of the finding**: some
runs error every selected test in ~15 s with zero tests executed, others take a `Fatal Python error:
Segmentation fault` inside `open_mesh_device` after the first few `TT_FATAL`s — i.e. once an ERISC
program has failed to load, the mesh is not reliably reopenable in that process. The committed run is
the second kind; `run_evidence.sh` detects it and says so rather than swallowing it under `|| true`.
Both outcomes demonstrate the same hard limit and neither is a shipped configuration
(`TT_METAL_WATCHER_DISABLE_ETH=1` ships). No environment knob grows that buffer, so the choice is
watcher coverage on the 110 Tensix worker cores per chip or none at all. The uninstrumented cores are the
fabric routers, which this stage does not author: they are stock `ttnn` 1D-fabric kernels, identical
to what any `ttnn` CCL user runs. Every op this stage does author runs on the covered Tensix cores.

---

## 8. Known limitations

1. **Watcher does not cover the ACTIVE_ETH cores on this configuration.** Hard tool limit, evidence
   and reasoning in §7.
2. **Decode parallel efficiency is 41–42%, not ~100%.** Batch-1 decode is launch- and latency-bound
   (about 6% of the DRAM roofline — §5.4's table has both layer kinds), so dividing the work by four
   does not divide the time by four. The collectives are under a tenth of the window, so this is not a
   communication problem and no
   collective-shape change would fix it. Prefill, which is compute-bound, reaches 82–88%.
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
5. **The fused matmul+CCL family is rejected, and one of its two ops hangs.** `probe_fused_ccl.txt`
   has the measurements (§2.5). `all_gather_matmul_async` is off by default in that probe because it
   wedged the mesh and needed a `tt-smi -r`; the probe runs one process per shape under `timeout` so
   that a repeat cannot take a whole sweep with it.
6. **The fabric packet size cannot satisfy both collectives, so the runtime always warns.** The
   token mixer's collective carries bf16 (2048 B pages, ideal 8192 B) and the MoE's carries
   `bfloat8_b` (1088 B pages, ideal 4352 B); one fabric setting serves both, so `ccl_common.cpp:64-70`
   emits a suboptimal-packet warning for whichever dtype it is not — `logs/warning_census.txt` carries
   the per-sweep count for the shipped setting, in the hundreds, and the build default produced a
   comparable count pointing the other way. The shipped 8192 B is chosen on measurement: §2.1's
   generated census counts every traced row of both dtypes, and 8192 B wins the bf16 comparison
   decisively while the block-float one splits almost evenly. The warnings on the block-float
   dispatches are therefore cosmetic on this layer, and `logs/warning_census.txt` carries every
   distinct warning class with its count so the next one is visible in an artifact rather than only in
   a log nobody greps.
7. **The kv cache is halved, not quartered.** `n_kv_heads = 2 < tp = 4`. Deliberate (§2.4): it buys
   zero attention traffic across the fabric. A head_dim split would quarter the cache and put a
   cross-device reduction on the decode critical path.
8. **`ttnn.experimental.all_reduce_async` is available and slower.** Rounds 0-3 of this stage
   recorded it as refusing Blackhole DRAM input outright; review round 3 pointed out that the op's
   Blackhole guard is DRAM-specific, and retrying the arm showed the rejection had never been
   earned at all — it was being called with one barrier semaphore where the op asserts two, and
   without the `cluster_axis` it requires, so it never reached any Blackhole check. Called correctly
   it is correct on both DRAM and L1 operands and costs 1.5-1.9x the shipped arm (`probe_ccl.txt`).
   It is rejected on that measurement. The limitation this entry used to claim did not exist.
9. **The mechanism behind a collective's device time is not pinned, only bounded.** §5.8
   establishes what it is *not* (operand dtype, data volume) with a measured control, and leaves the
   barrier/imbalance explanation as the remaining candidate rather than a demonstrated one. Pinning
   it needs per-device op timelines from the profiled run. It has two instances, not one: the prefill
   pair §5.8 discusses, and — found by review round 4 — the shipped decode `stack_sum` all-gather,
   which costs 33.74 us/op on full_attention against 21.80 on linear_attention for the same shape while
   the two captures' L1-operand rows agree to within 4%. Same signature, ~1.5% of the decode window.
   Nothing in the stage's conclusions rests on either: the collective share is quoted as an upper
   bound throughout.
10. **Long-context PCC is validated to 8000 tokens, not 262144.** Inherited from the earlier stages
   and unchanged by this one: the eager HF full-attention reference materialises a `[seq, seq]` score
   matrix and is not tractable on host beyond that. Beyond 8000 the mesh path is cross-checked
   against *itself* under a different internal decomposition (`test_chunk_size_invariance`, PCC
   1.000000 at 6000 tokens), and the full advertised context is validated for shape, finiteness and
   non-degeneracy at 262144 and 262141.
11. **One mesh shape.** `DEFAULT_MESH_SHAPE = (1, 4)` targets this host's hardware, as the stage goal
   directs. `local_decoder_config` is written against a general `tp` and validates divisibility, and
   `test_local_config_rejects_an_indivisible_mesh` covers the rejection path, but no other mesh shape
   is measured or claimed.

---

## 9. Exact artifacts

```
doc/multichip_decoder/
├── README.md                                  this file
├── work_log.md                                what was done, in order, with measurements
├── audit_figures.py                           every quoted figure must exist in an artifact
├── triage/                                    tt-triage capture of the all_gather_matmul_async hang
├── logs/
│   ├── run_evidence.sh                        regenerates everything below, in order
│   ├── make_tables.py                         generates every table in these two documents
│   ├── audit_selftest.txt                     what an audit_figures.py pass is worth, per class
│   ├── warning_census.py / .txt               every distinct runtime warning class, with counts
│   ├── source_stamp.json                      behavioural fingerprint of the code the sweep measured
│   ├── pytest_full_suite.txt.gz               137 passed, 3 skipped
│   ├── bench.py                               3-arm warmed prefill / traced decode harness
│   ├── ab_single_vs_multichip.txt             §5.2 — the speedup table's raw rows
│   ├── ab_layer_knobs.py / .txt               §5.5 — ccl / geometry / routing / sparse / cast arms
│   ├── probe_ccl.py / .txt                    §2.2 — every collective spelling x shape x topology
│   │                                          (`CCL` rows), and ring vs line **fabric** (`CCLFAB`)
│   ├── probe_fused_ccl.py / .txt              §2.5 — the fused matmul+CCL family, both boundaries
│   ├── probe_dense_matmul.py / .txt           §5.6 — decode matmul geometry ladder, per-device shapes
│   ├── probe_sparse_matmul_local.py / .txt    §5.7 — routed sparse geometry at E=64, active 4..63
│   ├── probe_expert_parallel.py / .txt        §2.5, §3 — EP vs intermediate-TP vs unsharded
│   ├── probe_footprint_local.py / .txt        §6 — measured per-device footprint, term by term
│   ├── probe_decode_batch.py / .txt           §5.7/§5.9 — collective shapes, and the fold and
│   │                                          sparse-core A/Bs at batch 1..32
│   ├── watcher_pytest.txt.gz                  §7 — 50 passed, 3 skipped, watcher on
│   └── watcher_pytest_eth_enabled.txt         §7 — the ACTIVE_ETH kernel-buffer failure, preserved
├── tracy/
│   ├── run_profiling.sh                       4 captures, prefill and decode in separate runs
│   ├── linear_attention/prefill_perf_report.{txt,csv.gz,summary.txt,console.txt}
│   ├── linear_attention/decode_perf_report.{txt.gz,csv.gz,summary.txt.gz,console.txt}
│   ├── linear_attention/{prefill,decode}_perf_report_stacked.{csv.gz,png}
│   ├── full_attention/…                       same set
│   └── */{prefill,decode}_tracy_run.txt       the capture's own console log
└── watcher/
    ├── census.py                              disjoint line-kind partition + fatal-class check
    ├── census_summary.txt                     0 fatal-class matches, 54 dumps
    └── watcher_log.txt.gz                     the raw watcher log
```

Freshness is checked against `logs/source_stamp.json` rather than against mtimes: the sweep records a
hash of each source's comment- and docstring-stripped AST, so a later *documentation* edit to
`multichip_decoder.py` or the test file does not read as a stale sweep, while any change to the code
does. Without the stamp the check falls back to mtime ordering.

`audit_figures.py` is the guard the three preceding stages carry, ported in review round 3: every
decimal and every 4+ digit integer in this README, the work log, the context contract and every probe
script must appear in a committed artifact as a *labelled* measurement, or be declared as an
expression over sourced values. `logs/run_evidence.sh` runs it last. `logs/audit_selftest.txt` records
what a pass is worth — the false-positive rate per figure class, measured by asking the same question
about arbitrary values:

<!-- TABLE:selftest -->
| figure class | trials | coincidental matches | rate |
|---|---|---|---|
| `1-decimal-us` | 2000 | 501 | **0.251** |
| `2-decimal` | 2000 | 301 | **0.150** |
| `3-decimal-ms` | 2000 | 47 | **0.024** |
| `6-decimal-pcc` | 2000 | 148 | **0.074** |
| `byte-count` | 2000 | 0 | **0.000** |
<!-- /TABLE:selftest -->

Read it before trusting a pass: byte counts are pinned hard, while 1- and 2-decimal figures can
survive the audit by coincidence roughly a fifth and a seventh of the time. That is why every table here is **generated** rather
than checked, and why review rounds 4, 5 and 6 each found a wrong hand-written ratio next to a correct
generated table. The remaining hand-written figures are the ones prose needs to argue with; the
numbers a decision rests on live in the tables.

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
