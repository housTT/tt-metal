# Ornith-1.0-35B — optimized decoder, work log

Stage: optimize the fused decoder (`tt/fused_decoder.py`) for per-device performance on one
Blackhole `p300c`, preserving its prefill/decode semantics, paged KV-cache behaviour, determinism,
non-aligned sequence support and advertised context.

Deliverable: [`tt/optimized_decoder.py`](../../tt/optimized_decoder.py),
[`tests/test_optimized_decoder.py`](../../tests/test_optimized_decoder.py),
[`README.md`](README.md).

Everything below is in the order it was done, with the measurement that decided each step. Rejected
candidates are in §4 with the number that rejected them.

---

## 1. Starting point and method

The optimized decoder starts as a **verbatim copy** of the fused decoder, which is this stage's
correctness floor and its "before" column. The copy was checked to reproduce the fused decoder's
traced decode before any change was made:

```
BENCH impl=optimized policy=fused-parity layer=3 (full_attention) decode(traced) iters=32 wall/iter=1.830 ms
# fused decoder, same harness, same process:                                     wall/iter=1.829 ms
```

That parity mode is not a historical artefact — `PrecisionPolicy` keeps it as a named policy
(`"fused-parity"`), so any candidate can be re-measured against the fused dtypes on demand.

Every latency number in this stage comes from
[`logs/bench.py`](logs/bench.py): same process, same device, same real `ornith-ai/Ornith-1.0-35B`
weights, same inputs, two warm-up prefills before a measured one, and 32 warmed `execute_trace`
replays for decode. Decode is **always** measured traced; no eager decode number appears anywhere in
this stage's evidence.

Op-level candidate sweeps use six standalone probes that build only the op under test at the
layer's real shapes, so a geometry sweep does not have to pay for a whole layer:
[`logs/probe_sparse_matmul.py`](logs/probe_sparse_matmul.py),
[`logs/probe_dense_matmul.py`](logs/probe_dense_matmul.py),
[`logs/probe_prefill_matmul.py`](logs/probe_prefill_matmul.py),
[`logs/probe_prefill_sdpa.py`](logs/probe_prefill_sdpa.py),
[`logs/probe_decode_micro.py`](logs/probe_decode_micro.py) and
[`logs/probe_projection_dtype.py`](logs/probe_projection_dtype.py); round 24 found
`probe_prefill_sdpa.py` missing from this list, which round 15 added and §4.19 records.
[`logs/probe_footprint.py`](logs/probe_footprint.py) is a seventh `probe_*.py` of a different kind — it
measures L1 footprint rather than latency, so it is not an op-level candidate sweep. Their outputs are committed next
to them as `.txt`. A probe is a *screen*, never an acceptance gate: two of this stage's rejections
(§4.5) are candidates a standalone probe called faster and the layer's HF-golden PCC called wrong,
because the probe's reference is the same op on the same inputs.

[`logs/run_evidence.sh`](logs/run_evidence.sh) regenerates every artifact in this directory in
order, as separate device runs, with watcher and the profiler never sharing a process.
[`logs/make_readme.py`](logs/make_readme.py) then fills every numeric block of the README from those
artifacts, and `--check` fails if the README and the artifacts disagree — so no figure in the README
is transcribed by hand.

---

## 2. Operation-topology audit of the measured path

Done first, before any knob tuning, from the fused stage's committed
`tracy/<kind>/decode_perf_report.csv` plus a fresh capture of the copied decoder. Per traced decode
step, `full_attention`, sorted by device time — this is the table the whole stage was planned from:

<!-- generated:topology-audit-full -->
| Rank | Op code | µs/step | What it is | Candidate | Action |
| --- | --- | --- | --- | --- | --- |
| 1 | `SparseMatmul active=?/256 x 32 x 2048 x 1024` | 368.0 | packed routed-expert gate/up | BFP4 weights + LoFi; core/block geometry; L1 output; fewer active experts | **all four taken** (§3.1, §3.2, §3.3, §3.4) |
| 2 | `SparseMatmul active=?/256 x 32 x 512 x 2048` | 343.8 | routed-expert down | same | **all four taken** |
| 3 | `UnaryDeviceOperation` | 270.1 | ~99 % `UnaryOpType::FILL` — `sparse_matmul` zeroing its 256-expert-wide output | move the output to L1; halve its dtype | **taken** (§3.2, §3.3) |
| 4 | `BinaryNgDeviceOperation` | 139.8 | SwiGLU multiply + router-score multiply, both over the 256-expert axis | L1 + BFP8 intermediates | **taken** (§3.2, §3.3) |
| 5 | `SliceDeviceOperation` | 109.3 | mostly the two slices that unpack the packed gate/up output | split the pair instead; L1 + BFP8 | **packed kept** (§4.2), L1+BFP8 taken |
| 6 | `MatmulDeviceOperation 32 x 2048 x 9216` | 95.6 | packed attention in-projection | explicit config; BFP8/BFP4 weights | **BFP8 + explicit config taken**, BFP4 measured (§4.6) |
| 7 | `DeepseekMoEFastReduceNC` | 92.0 | expert-axis reduction over the 256-wide down output | L1 + BFP8 input | **taken** |
| 8 | `MatmulDeviceOperation 32 x 4096 x 2048` | 73.9 | `o_proj` — flagged `SLOW`, 23.2 % of DRAM bandwidth | explicit decode program config; DRAM-sharded | **explicit 1D config taken**, DRAM-sharded measured and rejected (§4.1) |
| 9 | `LayerNormDeviceOperation` | 51.0 | four RMSNorms; the two residual ones run on **one core** | width-sharded L1 + `LayerNormShardedMultiCoreProgramConfig` | **taken** (§3.6) |
| 10 | `TopKDeviceOperation` | 48.3 | router top-8 over 256 experts, single core | pad to the multi-core width; replace the gate op | **both measured and rejected** (§4.3, §4.4) |
| 11 | `MatmulDeviceOperation 32 x 2048 x 1056` | 31.3 | the shared expert's packed gate/up projection | explicit config; BFP8 weights | **explicit config + BFP8 taken** (§3.5); it also shares the MoE's L1 bound |
| 12 | `MatmulDeviceOperation 32 x 2048 x 256` | 25.5 | router projection, 8 cores, 9.4 % of DRAM bandwidth | explicit config | **taken** (§3.5) |
| 13 | `UntilizeWithUnpadding` | 23.3 | the untilize half of three composite calls: the router scatter, the GQA head expansion and the `topk` index readback | no tile-native form at these shapes | **itemised, not removed** — README §6 splits it per call |
| 14 | `NLPCreateQKVHeadsDecodeDeviceOperation` | 19.3 | the dedicated QKV head split | inherited from the fused stage | unchanged; falls back to the functional spelling above 32 users |
| 15 | `SdpaDecodeDeviceOperation` | 17.6 | paged flash-decode | reduced cache dtype; program config sweep | **BFP8 cache taken** (§3.7), config swept (§4.5) |
| 16 | `FillPadDeviceOperation` | 14.8 | the MoE pads a 1-row decode activation to a 32-row tile. It used to also widen K and V to a tile for `paged_fused_update_cache` | **drop the cache-write pads** — the op takes its head count from the *cache*, so its writer kernel never reads the rows those pads zeroed | **taken, in two halves**: V's pad went with its reshard in review round 27 (§4.23) and K's in round 28 (§4.24), each measured at the layer. This row asserted the opposite for twenty-six rounds — "both pads are what the ops require of their inputs … irreducible at this layer" — which is the defect class §4.23 names. What remains is the MoE tile pad, and *that* one is load-bearing: §3.4's masking depends on the padding rows being exactly zero. Review round 16 pointed out this op had no disposition anywhere, having fallen into the remainder row |
|  | *the other 17 op codes, each under 14.8 µs/step* | 91.0 | — | — | — |
<!-- /generated:topology-audit-full -->

`linear_attention` shares the whole MoE and the norms with the table above; what differs is the
mixer, and its own top items (from the same capture set, per traced step) are:

<!-- generated:topology-audit-linear -->
| Op code | µs/step | What it is | Action |
| --- | --- | --- | --- |
| `MatmulDeviceOperation 32 x 2048 x 12352` | 128.5 | packed DeltaNet in-projection | BFP8 + explicit decode config (§3.5) |
| `MatmulDeviceOperation 32 x 4096 x 2048` | 80.9 | `out_proj` | explicit decode config (§3.5) + a bfloat16 activation (§4.11) |
| 3 x `MatmulDeviceOperation b={32} 32 x 128 x 128` | 44.2 | the float32 recurrent-state matmuls: decay read, delta outer product, output read | operands moved to L1 — 19 µs (§3.9 item 2); fidelity swept and rejected |
| `ReshapeView` + `Permute` (**all launches of both codes**) | 44.2 | the head-major relayout of the conv output is one `ReshapeView` and one `Permute` of these (30.9 µs/step together); the rest of this figure is three other `ReshapeView` launches | inherited from the fused stage, which already reduced it from three round trips to one |
| `TilizeWithValPadding` + `Concat` + `UntilizeWithUnpadding` (**all launches of all three codes**) | 36.2 | `repeat_interleave`'s GQA head expansion is one consecutive run of these three and costs 12.8 µs/step of this figure; the remainder is the router scatter's and `topk` readback's untilizes, which share the op codes | output moved to L1 (§3.9 item 5); the once-per-step op-to-op stall in front of it is the largest single gap in README §7's generated itemisation |
| `TernaryDeviceOperation` | 23.6 | the `addcmul` conv-tap accumulation | inherited; the fused stage measured `addcmul` against `mac` and kept it |
| *the other 20 op codes — 10 of them the shared MoE and norm rows the table above ranks (largest: `SparseMatmulDeviceOperation active=?/256 x 32 x 2048 x 1024` at 367.2), the rest below 23.6 µs/step* | 1629.8 | — | — |
<!-- /generated:topology-audit-linear -->

Structural observations from the same read, which drove §3.1 and §3.4:

* **Repeated same-input matmuls**: already packed by the fused stage — one `attn_in` for Q/K/V/gate,
  one `gdn_in` for the four DeltaNet projections, one `shared_in` for the shared expert's
  gate/up/router, one packed sparse gate/up. Nothing left to pack; the open question was whether
  packing still *wins* under BFP4/LoFi, which §4.2 answers.
* **Reshard / layout conversions**: 6 per `full_attention` decode step at the start of this stage, 1 per
  `linear_attention` one. This audit originally recorded them as "all required by an op contract" with
  "no avoidable ones existed to remove" — and that was the single most expensive sentence in the
  document. Review round 27 removed two of those six (§4.23) and round 28 removed a pad beside them
  (§4.24), in both cases because the op's own validation did not require what this line asserted. What
  the stage *adds* is the sharded-norm boundary, and it pays for it (§3.6). The current budgets are in
  README §6's generated table, taken from the gate's own constants rather than restated here: the pair
  went stale three times, and the deltas restated in this bullet went stale with them.
  (This line said "8–12" until review round 13, which is the *total* rather than the delta and disagreed
  with README §6 and §3.6, both of which had it right.)
* **Host fallback**: none in the measured path, inherited and re-asserted.
* **The 256-expert-wide intermediate chain** — fill → slice → slice → SwiGLU → score-multiply →
  fill → reduce — is 38 % of the window, on a tensor where **8 of 256** expert slots
  are non-zero. Every one of those ops was in `ttnn.DRAM_MEMORY_CONFIG`. That is the single largest
  finding of the audit and §3.2 is its fix.
* **The decode router routes the tile padding.** A batch-1 decode step runs the MoE on a 32-row
  tile with 1 real row. The 31 padding rows are exactly zero, so their router logits are exactly
  zero, `topk` returns a full set of experts for each, and the sparsity mask unions them in: **16
  active experts where the model asks for 8**. §3.4.

---

## 3. What was changed, in order, with the measurement

Cumulative traced decode, both layer kinds, `logs/bench.py`, batch 1:

| # | Change | `full_attention` ms | `linear_attention` ms |
| --- | --- | --- | --- |
| 0 | fused decoder (baseline) | 1.830 | 1.987 |
| 1 | precision policy: BFP4 experts + LoFi, BFP8 projections + HiFi2, BFP8 KV cache | 1.691 | 1.909 |
| 2 | padded-row routing mask (§3.4) | 1.624 | 1.842 |
| 3 | expert intermediates in L1 (§3.2) | 1.297 | 1.515 |
| 4 | BFP8 expert activations (§3.3) | 1.253 | 1.473 |
| 5 | sparse-matmul geometry, decode-shaped only | 0.976 | 1.198 |
| 6 | sparse-matmul geometry, active-expert-aware (§3.1) | 0.983 | 1.205 |
| 7 | dense decode program configs (§3.5) | 0.875 | 1.111 |
| 8 | width-sharded decode RMSNorms (§3.6) | 0.855 | 1.093 |
| 9 | explicit paged flash-decode program config kept, `k_chunk` pinned to the page block (§3.8) | 0.859 | 1.094 |
| 10 | residual norms hand their output to L1 rather than DRAM (`tt-perf-report` advice) | 0.858 | 1.093 |
| 11 | dense **prefill** projections take explicit 2D configs (§4.10) — prefill only | 0.858 | 1.093 |
| 12 | recurrent-state matmul operands in L1 (review round 1, §3.9 item 2) | 0.857 | 1.074 |
| 13 | GQA `repeat_interleave` output in L1 (§3.9 item 5) | 0.857 | 1.071 |
| 14 | gated-DeltaNet output activation bfloat16 (§4.11) | 0.858 | 1.071 |
| 15 | routed gate/up `in0_block_w` follows the active-expert bound (review round 6, §4.15) | 0.849 | 1.061 |
| 16 | two float32 promotions folded into the producing multiply (review round 14, §4.19) | 0.849 | 1.044 |
| 17 | the router's zero scatter-target hoisted out of the trace (review round 15, §4.19) | 0.846 | 1.038 |
| 18 | the token-mixer norm's shard carried into the in-projection (review round 25, §4.21) | *README §5.2* | *README §5.2* |

Row 18 carries no number of its own: it is the level that ships, and the ladder's rule is that the shipped level is README §5.2's generated table, always. Every earlier row is an intermediate revision that no longer exists to re-measure, which is why those rows keep their figures; the last row would simply go stale on the next sweep. `logs/ab_sharded_norm_in0.txt` has the paired arms this row is the win from.

Rows 16 and 17 are `linear_attention`-weighted for the same reason: the folded promotions are in the
recurrent-state path, which only that kind runs, and the hoisted target is shared by both but is a larger
share of the longer step. The shipped level is README §5.2's generated table, always — this ladder records
the path, and its rows are measurements of intermediate revisions that no artifact of the shipped code can
contain.

Warmed 2048-token prefill over the same steps: 243.44 → 96.89 ms (`full_attention`) and
257.73 → 102.94 ms (`linear_attention`) by step 15, and lower again after rounds 14 and 15 moved the routed
`in0` into L1 and made the prefill SDPA chunk reachable — README §5.2 has the shipped figures. Step 5 is
the one that matters for prefill and it went the wrong way first — see §3.1.

The shipped default is re-measured end to end after every change landed, and README §5.2's headline
table is **generated** from that measurement
([`logs/ab_fused_vs_optimized.txt`](logs/ab_fused_vs_optimized.txt)) rather than transcribed here, so
this log does not carry a second copy of it to go stale — roughly two and a half times on prefill for both
layer kinds and about two on traced decode, with the exact figures in that table. `test_optimized_beats_fused_traced_decode` gates the decode direction in
one process, in the delivered suite.

Row 0's `linear_attention` figure is the **fused stage's own committed number**, measured in that
stage's harness, quoted so this column starts where the previous stage left off. Re-measured in this
stage's harness it is 2.06-2.07 ms, a few percent slower — harness and run-to-run spread. Every "before"
figure the README quotes is this stage's own re-measurement, not row 0; against the fused stage's
published figure the `linear_attention` decode speedup would read slightly lower than README §5.2's.

The per-step figures in this column are the running total from one harness during development, and they
are the one group of numbers in this stage that **no artifact can contain**: each row measures an
intermediate revision of the code that no longer exists, so a re-run cannot reproduce it and
`audit_figures.py` exempts them by name, as the development ladder, only in this file. Where a step's own
A/B was captured as an artifact it is linked from the section that describes it (§3.6, §3.9, §4.10,
§4.11); the *shipped* numbers are always README §5.2's generated table. If you want to know what the
decoder does today, read README §5.2, not this column.

### 3.1 Sparse-matmul geometry is a function of the *active* expert count

The fused stage picked the largest core count that divides `Nt` — 32 cores for the packed gate/up
(`Nt` = 32) and 64 for the down projection (`Nt` = 64) — which pins `per_core_N` to 1 and therefore
the output block and subblock to 1×1.

`probe_sparse_matmul.py` sweeps core count (via `per_core_N`), grid shape, the whole `in0_block_w`
divisor ladder of `Kt`, output block/subblock width and output placement, under the selected
BFP4/LoFi policy, at four active-expert counts. The winner moves with the active count:

**README §5.4 has the table, generated** from the probe and from the geometry the suite log records the
layer building — the winner, the fused rule's candidate and the shipped choice at each of the four
active-expert counts, with the shipped row's gap against the winner where there is one. It is not
duplicated here: this log used to carry a second copy of it and review rounds 3 and 4 both found that
copy describing an older run of the probe.

What the sweep establishes, in words: the winner **moves with the active count**. At 8 active experts
(batch-1 decode) both roles want 8 cores with a wide output block, and the fused rule's 32/64-core
`per_core_N` 1 geometry is ~40 % slower. At 162 active experts (a 32-token prefill group) the ordering
reverses and the 8-core geometry is roughly 4× slower than the 32-core one — which is exactly what step
5 in the table above shows, decode falling 20 % and prefill rising 71 % when the decode geometry was
applied unconditionally.

The shipped rule is therefore an active-expert-aware target:
`cores = clamp(active_bound / k, 8, 32)` with `k` = 2 for gate/up and 4 for down, where
`active_bound = min(num_experts, real_rows * num_experts_per_tok)` is known exactly from the token
count. It reproduces the measured winner at all four sweep points. `in0_block_w` takes the largest legal
divisor of `Kt` up to a cap; `down` improves monotonically with it and then flattens, so its cap is the
whole tiled `K`, while **gate/up's cap follows the realised core count** — the narrow decode geometry wants
a smaller inner block and the wide one wants the whole `K`, by margins several times the measured spread in
both directions (§4.15).

`in0_block_w = 2` never appears: the shipped values are **32 or 64** for gate/up depending on the realised
core count (§4.15) and **16** for down, the whole tiled K, and the sweep shows the full ladder below them.

### 3.2 Every routed-expert intermediate moves to L1

The fused MoE wrote the packed gate/up output, the SwiGLU product, the scored activation and the
down output to `ttnn.DRAM_MEMORY_CONFIG`. Each is a `num_experts`-wide tensor — 16.8 MB and 33.6 MB
at bfloat16 — so the zero-fill, two slices, two elementwise passes and the expert reduction were all
DRAM round trips. They are small *per call* (one 32-row group), so they fit L1 comfortably; the only
reason they were in DRAM is that the fused stage never had a reason to move them.

`ttnn.L1_MEMORY_CONFIG` on all four: **1.624 → 1.297 ms** decode (`full_attention`) and
**1.842 → 1.515 ms** (`linear_attention`) — the single largest change in the stage — and 20 % off
prefill at the same time. `UnaryDeviceOperation` (the fill) fell 270 → 92 µs/step and
`DeepseekMoEFastReduceNC` 92 → 23 µs/step.

This is also what the `$optimize` guidance calls out directly: an explicit `DRAM_MEMORY_CONFIG` on a
decode intermediate is a performance smell.

### 3.3 BFP8 expert activations

With the intermediates in L1 the remaining cost of the chain is their width. Setting the routed
`sparse_matmul` output dtype to `bfloat8_b` halves every byte in it: rows 3→4 of §3's ladder on decode,
and about 4 % off prefill on both kinds, with no measurable PCC change on the `full_attention` prefill
screen (README §4.2's generated policy-sweep table times the reverse direction,
`expert_act_dtype=bfloat16`, against the shipped policy). The expert weights are already BFP4,
so this is the activation side of the same tensor group.

### 3.4 The decode router must not route the tile padding

`OptimizedMoE._active_expert_mask` reduces the routing vector over the group's **real** rows only.
One `ttnn.slice` on a `[1, 1, 32, 256]` tensor; the padded rows' expert *outputs* were already zero
(their activation row is zero, so every expert block they produce is zero and the score multiply
keeps it zero), so this changes which experts run and never the result.

`test_padded_rows_do_not_route` pins both halves and carries its own liveness control — it runs the
whole-tile reduction as well and asserts it activates strictly more experts, so the test cannot pass
if the masking is removed:

```
padded-row routing layer=3 (full_attention) batch=1: active experts masked=8  whole-tile=16 (bound 8)
padded-row routing layer=0 (linear_attention) batch=1: active experts masked=8  whole-tile=16 (bound 8)
padded-row routing layer=3 (full_attention) batch=4: active experts masked=31 whole-tile=38 (bound 32)
padded-row routing layer=0 (linear_attention) batch=4: active experts masked=30 whole-tile=37 (bound 32)
```

Worth 1.691 → 1.624 ms on its own, and it compounds with §3.1 because the sparse geometry is chosen
from the same bound.

### 3.5 Explicit decode program configs for every dense projection

Every dense matmul in a decode step is skinny — one tile of rows against a large weight — and ttnn's
heuristic sizes the grid from the output width, which is the wrong axis for these. `probe_dense_matmul.py`
compares three families per role at the real shapes under each role's own weight dtype and fidelity:

**README §5.4 has the table, generated** — all seven roles, ttnn's heuristic against the best
DRAM-sharded candidate against the shipped 1D `mcast_in0` config, with the shipped geometry read out of
the suite log. Not duplicated here, for the same reason as §3.1's.

What it establishes: the explicit 1D `mcast_in0` config wins on **every** role, ttnn's heuristic is
1.1–2.9× off it, and the DRAM-sharded family loses on all seven even measured without the activation
reshard it would additionally need — the op pins its compute grid to the 8 DRAM banks, and 8 wide-shard
cores cannot beat a large mcast grid at one tile of M. End to end this step is rows 6→7 of §3's ladder.

The `router` row keeps bfloat16 weights, HiFi4 and float32 accumulation — expert *selection* is a
discrete decision and this stage did not touch it — so it was swept separately under its own
settings rather than inheriting the BFP8/HiFi2 sweep.

### 3.6 Width-sharded decode RMSNorms

`ttnn.rms_norm` parallelises over rows, and a decode activation is one tile of rows, so the
interleaved form the fused stage used puts the whole 2048-wide norm on **one core**. Width-sharding
input and output over 8 cores with an explicit `LayerNormShardedMultiCoreProgramConfig` takes about a
third off it — README §5.5's generated knob table has both times, from `probe_decode_micro.py`'s `NORM`
rows, which also sweep 4, 16, 32 and 64 cores: 16 and up get progressively worse as the per-core block
shrinks, and 4 is inside the run-to-run spread of 8. At the *layer* none of 4/8/16/32 can be told apart at
all — §4.13 has the A/B and what it does and does not support — so 8 ships for the reason stated there and
nowhere else: it is the shard count every other piece of norm evidence in this stage was measured at. Review
round 11 found this sentence giving a second, differently-worded reason, which is how a constant ends up with
two justifications and no measurement.

Each sharded norm pays one `to_memory_config` in, and — where its consumer cannot take the shard — one
`sharded_to_interleaved` out, about 3 µs for a larger saving on the norm itself; README §5.5's generated knob
table carries both times, from the probe. Net effect on the layer: rows 7→8 of §3's ladder.

The `sharded_to_interleaved` on the **token-mixer** norm is gone as of review round 25: the 1D `mcast_in0`
projection consumes a width-sharded `in0` directly, so the shard is carried straight into `attn_in`/`gdn_in`.
Until that round this section asserted the opposite — that `mcast_in0` "needs an interleaved `in0` back" — and
that assertion closed the sharded-residual family `$optimize` OPT-003 makes mandatory. §4.21 records what the op
actually validates, why the probe could not contradict the claim, and the whole-layer A/B.
`test_no_layout_churn_in_measured_forward` budgets those conversions exactly (README §6's generated
table has the current pair, and rounds 25 and 27 each lowered it; per decode
step, from 1 and 6) and itemises each one.

Whether the narrow 256-wide Q/K head-dim norms should also shard was measured both ways, twice each
([`logs/ab_norm_shard_width.txt`](logs/ab_norm_shard_width.txt)): sharding them is 2–3 µs *better*
on `full_attention` and identical on `linear_attention`, so the simpler contract — every
decode-shaped norm shards — is also the faster one.

### 3.7 KV cache

BFP8/`bfloat8_b`, from the fused decoder's bfloat16, with the prefill fill tensors explicitly cast
to the cache dtype (`_cache_fill_tensor`) and the decode `paged_update_cache` inputs left bfloat16,
which is what that op accepts. Decode latency is unchanged at this stage's 8192-token test context
(0.855 both ways — the cache read is not a material fraction of a single-layer decode step), and it
is kept for **capacity**: it halves the per-token KV bytes, which is what
[`../context_contract.json`](../context_contract.json) advertises the maximum context from. PCC is
unchanged on the prefill screen and across the whole delivered ladder.

---

### 3.8 Paged flash-decode program config

The fused decoder already passed an explicit `SDPAProgramConfig`; this stage checked what that is
worth and what else the axis holds
([`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt), `SDPA` rows, four grids × five chunk
pairs at an 8192-token context under the shipped BFP8 paged cache, at the layer's own compute-kernel
contract — which is to pass none, and which round 8 found this sweep violating):

The ladder is in the artifact — 25 rows per section: the op's own default, four grids (`8x8`, `11x10`, `8x4`,
`4x8`) against five chunk pairs (`q32 k64`, `q32 k128`, `q32 k32`, `q0 k0`, `q0 k64`), three
`max_cores_per_head_batch` arms at the shipped grid and chunk pair, and one labelled arm carrying the rejected
HiFi2/fp32-acc compute-kernel config so its cost stays visible — and README §5.5's generated knob table quotes
the rows the decisions rest on. Round 11 added the `max_cores_per_head_batch` arms, because the stage had
called this config swept with one of its four fields defaulted; round 24 found this paragraph still describing
the pre-round-11 ladder and still omitting that field from the findings below.

Four findings, all kept as evidence:

* the **op default is more than an order of magnitude slower** than any explicit config here, which is
  why the explicit one stays;
* **`max_cores_per_head_batch` saturates at the default.** Its ttnn default of 16 gives `16 * B * kv_heads`
  active cores, and the three arms measured at the shipped grid and chunk pair show the axis flat above it and
  costly below: 8 is materially slower than 32, while 64 is inside the arms' own spread of 32. So the field is
  left at its default deliberately, on measurement, rather than by omission — which is how it stood before
  round 11 asked for the arms. README §5.5's generated knob table carries the three times.
* a `k_chunk_size` **larger than the 64-token paged block size is wrong**, not merely risky. The
  isolated op cannot see it — the probe's reference is the op default on the same page table — but
  the layer's decode PCC against the HF golden collapses to 0.02292–0.90512 at the paged contexts the
  delivered tests use. So the ~10 % the k128 row promises is rejected on correctness, and
  `k_chunk_size` is now pinned to `page_block_size` in code rather than to the literal 64.

* the **grid is not a latency axis at all**, which took taking it to find out. `8x4` leads the shipped `8x8`
  in both sections of [`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt) — README §5.5's generated knob
  table prints the pair and the gap, which is the fourth time this paragraph has had to stop quoting them by
  hand: rounds 5, 8, 9 and 13 each found the transcription wrong or stale, the last of them quoting one section's
  8x4 time against the other section's 8x8 time and understating the spread the gap has to clear (one of those
  8x4 rows carries a spread comparable to the gap being claimed). Identical PCC to six decimals — and unlike the k-chunk it appeared to cost nothing: no invariant, no correctness question, and a
  dead heat at the layer ([`logs/ab_sdpa_decode_grid.txt`](logs/ab_sdpa_decode_grid.txt); README §5.1's
  generated table carries every build of both arms, and the difference between the arms is smaller than one
  arm's own span, SDPA being ~2 % of a step). Review round 9 was right that nothing recorded which end of the
  axis shipped, so it was taken — and **the suite rejected it**:

  ```
  TT_FATAL @ sdpa_decode_program_factory.cpp:191: num_cores_available >= B
  test_decode_batch_above_head_split_limit[40-full_attention]  FAILED
  test_decode_batch_above_head_split_limit[56-full_attention]  FAILED
  ```

  Flash-decode assigns **at least one core per batch row**, so a 32-core grid silently caps decode at batch 32
  and the supported batch-40 and batch-56 cases die inside the op. The grid is not tuning; it is the largest
  decode batch the layer can serve, and 8x8's 64 cores are chosen to cover the 56 the suite exercises. The
  ~1 µs goes unclaimed for that reason, which is a better answer than round 9's finding asked for and a worse
  one than the sweep suggested. `11x10` satisfies the bound too and is slower in the same probe, so 8x8 is also
  the fastest grid that is *legal*, which is what the call site now says — README §5.5's generated knob table
  prints every grid's time, so this paragraph does not carry a second copy to go stale.

  The guard is a test, not a comment: `test_decode_runs_the_tuned_program_configs` asserts
  `grid cores >= LARGEST_SUPPORTED_DECODE_BATCH` rather than the literal `8x8`, so re-deriving "8x4 is a
  microsecond faster" from the probe fails the suite instead of shipping. Nothing had asserted any field of
  this config before round 9 — including `k_chunk_size == page_block_size`, whose violation is silent — which
  is why an op-level sweep could argue against a capability bound for two rounds without contradiction.

  The same A/B run also rejected the routed `down` orientation candidate (§4.14), so the harness is not a
  rubber stamp: one file, two candidates measured, both rejected, for two different reasons.

One further thing the same experiment caught: passing the *prefill* SDPA's compute-kernel config
(HiFi2, `fp32_dest_acc_en=True`) to the decode op collapses decode PCC to 0.33 on a BFP8 paged
cache. The decode call therefore deliberately passes **no** compute-kernel config, which is what the
fused decoder did and what every PCC number here is measured with; the call site says so.

### 3.9 What review round 1 changed

An independent `$stage-review` pass found five things this stage had wrong or unmeasured. All five
are fixed above rather than argued with, and two of them were real performance wins:

1. **The L1-size probe could never succeed.**
   `getattr(mesh_device, "l1_size_per_core", lambda: 1 << 20)()` — `ttnn.MeshDevice` has no such
   attribute, so every L1 budget in this file ran against the 1 MiB fallback, 67 % of Blackhole's
   real 1 532 032 B. Consequence: the 2D prefill config was silently `None` on the two widest
   projections (`attn_in`, `gdn_in`) while §4.10 claimed it was on, and the expert L1 budget was
   38.5 MiB instead of 58.9 MiB. Fixed to `ttnn.get_max_worker_l1_unreserved_size()`, the
   circular-buffer estimate replaced with the 2D factory's own arithmetic (calibrated against the one
   geometry known to fail), and `test_prefill_runs_the_tuned_program_configs` added so nothing can go
   back to `None` unnoticed.
2. **The `linear_attention` recurrent-state matmuls had three open advice items and no sweep.**
   Swept (`probe_decode_micro.py --section state`): the fidelity advice is worth <= 0.3 us per matmul
   and is rejected, but *"place input 0 in L1"* is worth **1.093 -> 1.074 ms** — 19 us, the largest
   single win of the round. The audit in §2 was `full_attention` only, which is how a 45 us/step
   group in the *slower* layer kind went unranked; README §5.5 now covers both reports.
3. **`nnz` was rejected on an inherited argument.** Measured instead; it wedged the device. §4.8.
4. **The `gdn_out` probe used a bfloat16 in0 where the layer had float32.** §4.11.
5. **The largest traced op-to-op stall was unclassified.** Attributed to the tilize inside
   `repeat_interleave`'s GQA expansion; ~3 us recovered by placing it in L1, the rest named in the
   accounting.

---

## 4. Candidates measured and rejected

### 4.1 DRAM-sharded decode matmuls — rejected on measurement

`tech_reports/LLMs/llms.md` prescribes `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for
small-M large-weight decode matmuls, and `tt-perf-report` flagged `o_proj` as `SLOW` at 23.2 % of
DRAM bandwidth, so this was mandatory to try. It was built properly — weight DRAM width-sharded over
the 8 DRAM banks, activation and output L1 width-sharded on a matching grid, `in0_block_w` swept over
the legal divisors of the per-core K — and it **loses on every role**, even measured with the
activation reshard outside the timed region:

README §5.4's generated dense-decode table carries the per-role pair (`best DRAM-sharded` against
`shipped 1D mcast_in0`) and its closing line states how many roles the family loses on, computed from
the probe rather than asserted — so this rejection cannot survive a re-run that reverses it.

The reason is structural rather than a tuning miss: the op pins its compute grid to the DRAM banks,
so 8 wide-shard cores compete with a 22–110-core multicast grid on shapes this skinny.
`models/demos/blackhole/qwen36/tt/tp_common.py` records the same conclusion for Blackhole decode
matmuls ("small grids beat the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode
matmuls"). It is kept in the probe so the comparison is re-runnable, not deleted.

### 4.2 Splitting the packed gate/up pair — rejected on measurement (OPT-010)

Under the new BFP4/LoFi policy and the tuned 8-core geometry, the packed `N = 2·I` sparse matmul plus
its two unpacking slices and the fused-SiLU multiply beats the separate `N = I` pair plus the same
multiply by about a quarter — README §5.5's generated knob table has both times, from
`probe_decode_micro.py`'s `SPLIT` rows, which also record that the two produce identical output
(PCC 1.000000). Packed stays. The split candidate loses because halving `N` halves
the usable output block: at `N = 512` the same 8-core grid gives `per_core_N` 2 instead of 4, and
two launches pay the per-expert loop twice.

### 4.3 Padding the router logits to the multi-core `topk` width — rejected on measurement

`ttnn.topk` is single-core on a 256-wide reduced dim and is one of the ten largest items in the
window (README §7's generated limitation list has its per-step cost). Its multi-core path needs a
power-of-two width of at least 8192, and the `$optimize` LM-head guidance recommends padding to reach a
fast TopK path, so the whole ladder was measured — 256, 512, 1024, 2048, 4096, 8192, 16384 — in
[`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt)'s `TOPK` rows, with README §5.5 quoting
the two endpoints.

Padding with `-inf` is *correct* — the probe records that the returned indices stay in range at every
width — but the multi-core path only becomes legal at 8192, every intermediate width is worse than the
one below it, and even 8192 is roughly 3× the single-core 256 call. Rejected with the numbers; the
single-core `topk` remains a named limitation.

### 4.4 Replacing `topk → softmax → scatter` with a threshold rewrite — rejected on measurement

The scatter internally untilizes its input, index and source and retilizes the result, ~34 µs/step.
`topk → ge(kth) → where(-inf) → softmax(256)` produces the *bit-identical* dense vector (max
absolute difference 0.000e+00 over the probe's inputs) without any of that, so it was re-measured
under this stage's regime rather than inherited from the fused stage's rejection. Still slower — README
§5.5's generated knob table has both times for the whole chain, from `probe_decode_micro.py`'s `GATE`
rows. Rejected again, now with this stage's own measurement.

`ttnn.experimental.deepseek.moe.generalized_moe_gate` fuses the whole gate into one kernel but is
bfloat16-only; the fused stage measured bfloat16 routing logits agreeing with float32 on only
99.8 % / 95.5 % of top-8 sets, and a changed expert set is a model-visible change rather than a
precision one, so it stays rejected on that ground.

### 4.5 SDPA decode program config — swept, two faster candidates rejected on layer correctness

§3.8 has the sweep. Two candidates are faster in isolation and wrong in the layer, and the numbers
that reject them are committed in
[`logs/ab_sdpa_decode_contract.txt`](logs/ab_sdpa_decode_contract.txt): `k_chunk_size` 128 (~10 %
faster standalone) drops the layer's decode PCC to 0.02292-0.90512, and passing the prefill compute-kernel
config to the decode op drops it to 0.33. Both failures are invisible to a standalone probe, whose
reference is the same op on the same page table. The shipped config keeps `k_chunk_size` pinned to
`page_block_size` and passes no compute-kernel config. `SdpaDecode` is 17 µs/step, 2 % of the window.

Review round 7 raised a **third** candidate — `q_chunk_size = 0, k_chunk_size = 0`, ~8 % faster in
isolation on all four grids, which OPT-002 names as the usual first paged decode candidate. It was measured
rather than argued, and the result corrected the review's own inference: it is **correct** at the layer (the
whole 109-case suite passes, and every case candidates A and B fail passes with worst PCC 0.999865), so the
correctness argument that rejects those two does not reject this.

Round 8 then asked the two questions that settle it, and both are now measured rather than asserted:

* **Is there a variant that keeps the page-block invariant?** No. The probe now sweeps `q_chunk=0` with
  `k_chunk=page_block` too, and it measures *identically* to the shipped `q_chunk=32` arm. The isolation win
  is entirely k-chunk-side, so there is no safe way to take it.
* **Is the layer-level gap real or noise?** Noise — and this answer has now been wrong in both directions.
  [`logs/ab_decode_harness.txt`](logs/ab_decode_harness.txt) times both arms with three fresh builds each, and
  README §5.1's generated table prints them with each arm's span beside it. The two arms have now swapped order
  between consecutive sweeps by a microsecond or two — round 8 read one direction out of that file, round 10
  read the other, and round 11's sweep swapped it back — which is the finding: at this harness's resolution the
  ordering is not a property of the configuration. That is why the figures live in a generated block now and why
  the rejection below rests on the invariant rather than on either ordering. The advantage is real but exists
  only in the isolated op
  ([`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt), where it is several microseconds ahead of the
  shipped arm at the op). Worth stating plainly:
  my first instinct here was right and I talked myself out of it on one run of a harness whose own repeats
  disagree by more than the effect.

Rejected on the trade *and* on the measurement, therefore, where round 8 had it resting on the trade alone:
there is nothing to trade away at the layer, and taking it would still mean replacing a
checkable invariant with trust in the op's internal chunk choice, where candidate A is the proof that an
oversized k-chunk here is *silently wrong* rather than an error. The 109-case pass bounds that risk without
eliminating it — those contexts are a subset and the failure mode is a wrong answer, not a crash. Both arms
and the reasoning are in `ab_sdpa_decode_contract.txt`.

### 4.6 BFP4 dense projection weights — measured, decision recorded in §5 of the README

`proj_dtype = bfloat4_b` (packed attention in-projection, `o_proj`, packed DeltaNet in-projection,
`out_proj`) is **faster** on traced decode at unchanged prefill; README §4.2's generated table has both
arms and §4.3 states the saving, computed from the same artifact.

OPT-007 requires that trial on real weights and requires the accept/reject decision to rest on
model-visible correctness, so it was decided on the **same HF-golden ladder the delivered suite
runs** — ten prefill lengths from 1 to 3000 including the non-aligned ones, plus four decode steps,
both arms in one process ([`logs/probe_projection_dtype.txt`](logs/probe_projection_dtype.txt)):

**README §4.3 has the table, generated** — the worst PCC of each arm on each layer kind, which case it
came from, the margin above the bar, the error ratio and what the candidate buys on traced decode, all
computed from the probe. Not duplicated here: the PCCs move in the fifth decimal between runs, so a
transcribed copy of them goes stale on every re-run, which is what review round 4 found.

The shape of the decision: every BFP4 row clears the bar, so this is not a pass/fail rejection — it is
an order-of-magnitude increase in layer error, leaving about a third of the headroom, in **one** layer of
a 40-layer stack, bought for a low-single-digit percentage of one traced decode step and nothing at all
in prefill. The routed-expert BFP4 step this stage *did* take is the opposite trade: a much smaller
error increase for a much larger share of decode.

Rejected on that comparison, and shipped as `POLICIES["bfp4-projections"]` so the candidate stays
one flag away for `$datatype-sweep`, which owns the accuracy/performance frontier.

### 4.7 Other precision candidates

Every row changes exactly one field of the selected policy and is measured against the shipped
default in the same harness ([`logs/ab_precision_policy.txt`](logs/ab_precision_policy.txt)). The
`pcc` column is the 2048-token real-weight prefill screen that runs in the same command; the
`linear_attention` screen input is dominated by DeltaNet state accumulation, so its absolute value
is not comparable to `full_attention`'s — what matters is the delta against the same input's
baseline. README §4.2's generated table carries the screen value for every arm, from that artifact.

The table is **generated** into README §4.2 from that artifact by
[`logs/make_readme.py`](logs/make_readme.py) rather than transcribed here, so it cannot drift from
the run; the decisions it records are:

* `proj_dtype=bfloat4_b` — faster, rejected on the real-weight ladder (§4.6);
* `proj_fidelity=LoFi` — no gain over HiFi2 at strictly less precision;
* `expert_fidelity=HiFi2` and `expert_gate_up_dtype=bfloat8_b` — both slower than the selected
  BFP4+LoFi, which is what backs the choice of that pair rather than an asserted percentage;
* `shared_dtype=bfloat4_b`+LoFi — inside the run-to-run spread, so the higher precision is free;
* `expert_down_dtype=bfloat8_b` — slower *and* 1.9x the weight bytes;
* `kv_cache_dtype=bfloat16` — identical latency, twice the cache bytes, so BFP8 is kept for capacity;
* `expert_act_dtype=bfloat16` — much slower; the `num_experts`-wide intermediate chain doubles and
  stops fitting the L1 budget at the larger expert-group sizes.

### 4.8 Static `nnz` for the routed sparse matmuls — tried, **hung the device**, rejected

`ttnn.sparse_matmul(..., nnz=N)` selects a faster fixed-count path worth 13-16 % on other models, and
after §3.4 the batch-1 decode mask holds exactly `num_experts_per_tok` experts. Review round 1 was
right that the inherited reason for skipping it — "a bfloat16 scattered softmax can flush to zero, so
the count is not exact" — is a property of the *mask construction*, fixable by deriving the mask from
the `topk` indices rather than from the weights, and that the factory's own comment says the sender
validates `count_nonzero(sparsity)` against `nnz` on device and "fail[s] loudly instead of
deadlocking" ([tt-metal #45943](https://github.com/tenstorrent/tt-metal/issues/45943)). So it was
measured rather than argued about.

At Ornith's shapes that validation does **not** hold. `probe_sparse_matmul.py --nnz --active 8
--role gate_up` builds a sparsity tensor with exactly 8 non-zeros and passes `nnz=8` — an exact
match, no flush path, float32 sparsity — and the **first candidate hung the device**. No row was
printed; `tt-triage.py` caught it mid-op:

```
dump_running_operations: 1,UnaryDeviceOperation, Tensor[0] logical_shape [1,1,1,256,32,1024] BFLOAT8_B, L1
                         2/4/6, SparseMatmulDeviceOperation
dump_callstacks:         device 3, brisc 16-2: #0 process_stall () cq_prefetch.cpp:1806
check_broken_components: device 3, 10 functional_workers + 2 erisc halted
```

Triage output is committed at [`triage/tt-triage.txt`](triage/tt-triage.txt) and
[`triage/triage-summary.txt`](triage/triage-summary.txt).

Recovery, per `$tt-device-usage`:

| step | result |
| --- | --- |
| kill the stale probe process | done, nothing else was running |
| `timeout 60 tt-smi -ls --local` | 8 Blackhole boards visible |
| `timeout 180 tt-smi -r` | `Resetting all PCI devices: [0, 1, 2, 3]` → re-initialised |
| `timeout 60 tt-smi -ls --local` | 8 boards, one reset sufficed |
| 1×1 mesh open/close smoke | `MESH_SMOKE_OK` |

No further profiler or watcher collection was run while the card was unhealthy, and every artifact in
this stage was preserved. This is infrastructure evidence, not a model result — but it *is* the
model-shape-specific blocker the skill asks for: `nnz` stays inferred at runtime, and the reason is
now a reproduced hang on this decoder's own sparse-matmul shapes rather than an inherited argument.
`probe_sparse_matmul.py` keeps the `--nnz` flag, documented as dangerous and deliberately excluded
from `run_evidence.sh`, so the finding is reproducible by anyone willing to reset the card.

### 4.9 The expert-major gathering path — assessed, rejected with a precise blocker

`ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` is the in-tree op that wants the
expert-major layout `doc/fused_decoder/README.md` §8 item 1 identified as the way to remove the
`num_experts`-wide intermediates. It is rejected on three specific grounds, not on effort:

* it is a **prefill** op that launches one device program per local expert — 256 per MoE call here,
  against this decoder's 2 — so it cannot serve a decode step at all;
* it consumes a *dispatched* token buffer with per-expert counts and region offsets, which this
  decoder does not build; adding it is a routing-algorithm change, not an op swap;
* its own documented accuracy target is **PCC >= 0.97** against the PyTorch reference (its nanobind
  docstring; the in-tree DS-V3 cases land at ~0.98 with LoFi), which is below this stage's inherited
  **0.995** layer bar.

### 4.10 Explicit 2D program configs for the dense prefill projections — taken

`tt-perf-report` flags `in0_block_w=1 is small` on every dense prefill row: ttnn's heuristic already
picks the 2D family and fills the 11x10 grid, but leaves the inner block at one tile. An explicit
`MatmulMultiCoreReuseMultiCastProgramConfig` with the largest inner block the `in1` circular buffer
holds wins on all six of them. **README §5.4 carries the table and it is generated** — from the probe's
`in0=DRAM` arm and from the configs the suite log records the layer building — because review round 4
found the hand-written version quoting a superseded run of the probe in every row, including one
geometry the artifact called unbuildable while the layer ran it (§6, round 4 P1).
[`logs/probe_prefill_matmul.txt`](logs/probe_prefill_matmul.txt) is the sweep: three grids x the whole
`in0_block_w` ladder x both `in0` placements, min of three repeats with the spread reported.

Two limits are encoded rather than discovered at runtime. `in0_block_w` 16 is best for the
narrow-output roles but fails to build for the two wide ones (`attn_in` at `per_core_N` 27, `gdn_in`
at 36) with "statically allocated circular buffers ... clash with L1 buffers", which
`PREFILL_MATMUL_IN1_TILE_BUDGET` bounds; and at a large prefill batch the fixed *output* block alone
exceeds L1 (`per_core_M` 13 x `per_core_N` 27 at batch 32), so `_prefill_2d_matmul_config` models the
three circular buffers with the 2D factory's own arithmetic and hands those shapes back to ttnn's
heuristic. That is an explicit size check, not a silent fallback, and it is gated by
`test_prefill_runs_the_tuned_program_configs` — which exists because review round 1 found the check
running against a 1 MiB L1 constant that turned the config off on `attn_in` and `gdn_in` (§3.9).

The whole dense group is well under one percent of the prefill window — the window is overwhelmingly
routed-expert `sparse_matmul` — so this step is worth a fraction of a percent of prefill end to end,
which is what it measures. README §5.4 states both shares, generated from the committed prefill
reports.

### 4.11 The gated-DeltaNet output activation dtype — taken

`chunk_gated_delta_rule` returns float32, so `merged * silu(z)` defaulted to a float32 activation and
`gdn_out` ran `HiFi2 FP32 x BFP8 => FP32` at 29 us decode / 222 us prefill where the identically
shaped `o_proj` runs `BF16 x BFP8 => BF16` at 25 / 169. Naming `dtype=ttnn.bfloat16` on the multiply
costs no extra op and fixes both: `linear_attention` decode 1.075 -> 1.071 ms, prefill
102.42 -> 102.30 ([`logs/ab_gdn_out_activation.txt`](logs/ab_gdn_out_activation.txt)). It also closes
the OPT-014 gap review round 1 found — the `gdn_out` geometry sweep had used a bfloat16 in0 while the
layer used float32, so the sweep and the shipped row now agree on dtype.

### 4.12 Prefill alignment padding is still routed — measured as immaterial

`_block` hands the MoE a `valid_tokens` count only when it actually padded, which in prefill never
happens: `tokens = batch * phys` and `phys` is already a multiple of the 128-token physical
alignment. So a `seq_len=7` prefill runs four 32-row expert groups of which three are pure padding.
It is left alone deliberately: the padding rows all carry identical zero logits, so each all-padding
group's union is the *same* 8 experts rather than 8 more per group, the waste is bounded by roughly
one extra 8-expert group per 32 padded rows, and it is **absent from every measured window** — the
2048-token prefill every §5 figure comes from has no padding at all. Recorded rather than left
silent.

### 4.13 Not applicable to this stage

* **Collectives / fused CCL+matmul / persistent CCL buffers.** This is a single-device 1×1 mesh
  decoder; there is no collective in the measured path. Multi-device topology is the next stage's.
* **LM head and sampling.** A decoder layer has neither. The terminal path is owned by the
  full-model stage.
* **`ttnn.sparse_matmul` `is_input_a_sparse`.** Already set on the down projection by the fused
  stage, and kept — the down projection consumes the expert-major activation.

### 4.14 The sparse-matmul grid *orientation* — measured, rejected on arithmetic

Review round 5 pointed out that `_sparse_matmul_config` fills one grid axis first (a column: 1×8, 2×8,
4×8) while the probe measures both rectangles of each core count, and that the other orientation is faster
at several points. It also pointed out that the code comment claimed the column form "beat the row form by
2-10 % at every geometry measured", which the sweep does not say. Both were true.

The probe now reports a `spread=` per row (min of three repeats, max−min), because a 1–3 % claim is not a
claim without one. The spread is small — typically well under a microsecond — so the gaps are real, and
README §5.4's generated table prints every one of them. What they are:

<!-- generated:orientation-ladder -->
| point | role | shipped (column) | other (row) | verdict |
| --- | --- | --- | --- | --- |
| 8 active — the tuned batch-1 decode target | gate/up | **153.4 µs** | 171.8 µs | **column** wins by 18.4 µs, beyond the ±0.5 µs spread |
| 8 active | down | **152.8 µs** | 172.1 µs | **column** wins by 19.3 µs, beyond the ±0.3 µs spread |
| 162 active — a 32-token prefill group | gate/up | **568.3 µs** | 578.9 µs | **column** wins by 10.6 µs, beyond the ±2.7 µs spread |
| 162 active | down | 346.1 µs | **341.8 µs** | **row** wins by 4.3 µs, beyond the ±0.2 µs spread |
| 64 active — decode batch 8, **not tuned** | gate/up | **385.4 µs** | 387.5 µs | **column** wins by 2.1 µs, beyond the ±0.5 µs spread |
| 64 active | down | 284.8 µs | **277.6 µs** | **row** wins by 7.2 µs, beyond the ±1.6 µs spread |
<!-- /generated:orientation-ladder -->

One row wants the row rectangle beyond its spread — `down` at the prefill group — and it is a geometry the
shipped code really builds, under a key with no competing point: `down` reaches 32 cores only in prefill,
because the largest *tuned* decode batch, 8, gives a 64-expert bound that realises 16 — an untuned decode batch
of 32 or more saturates the bound at 256 and does reach 32 cores, which review round 12 corrected here. Review round 9 asked for exactly
that rule, and it is a one-line rule: `("down", 32) -> row`.

**So it was implemented and measured end to end, and it loses.**
[`logs/ab_sdpa_decode_grid.txt`](logs/ab_sdpa_decode_grid.txt) alternates the two arms build-by-build with
three timed builds each, discarding each arm's first:

README §5.1's generated table has every timed build of both arms.

Every timed build of the column arm beats every timed build of the row arm, on both layer kinds, by close to a
millisecond — more than the whole op-level gap, in the opposite direction. The op-level advantage does not
merely fail to survive — it
**reverses**, and into a share of prefill an order of magnitude larger than the 64 expert groups of a
2048-token prefill can account for from that op alone. The mechanism is not measured here and so is not
claimed; the plausible reading is that the isolated probe holds `in0` still while the layer feeds `down` from
the preceding `gate_up`'s output, and the two rectangles do not place that input the same way.

Rejected, and now for a better reason than round 5's arithmetic: not "the gain is too small to be worth a
special case" but "the gain is not there at the layer, and the layer is what ships". The generalisation worth
keeping is the one this row cost three review rounds to learn — **an op-level microsecond is a hypothesis, not
a result** — and this stage now settles every geometry that reaches a document with a whole-layer A/B.

The 64-active `gate_up` row is a smaller lesson in the same direction. Round 9's review read it, correctly against the artifact
it had, as the *row* winning; every re-measurement since has had the *column* ahead. What has not held is the
margin: this row has moved back and forth across its own spread boundary from one sweep to the next, without a
line of shipped code changing — the ladder above prints whichever verdict the current capture supports, and it
is not the same one it printed a sweep ago. That is why the generated table in README §5.4 prints each
row's own spread, why the inside-vs-beyond test is made at the artifact's printed precision, and why neither
this paragraph nor the shipped comment quotes a magnitude for it any more — round 22 found both of them
describing it as sub-microsecond noise against a table that by then called it decisive. The direction is what
the shipped rule turns on, and the direction is the part that has held.

For the record, the earlier states of this paragraph: round 5 found it claiming a universal column win, which
the sweep never said; round 8 found the prefill `down` sign inverted; round 9 found round 8's correction
inverted the *other* way, so the paragraph asserted a win the artifact contradicted. README §5.4's generated
table read the artifact correctly in all three rounds. That is the argument for generating tables — and, since
prose beside a correct table can still be wrong, for `audit_figures.py` checking the source comment and this
file against the same artifact.

### 4.19 Three candidates review round 14 raised: two taken, one illegal

Round 14 was the first round in five to find shipped-code work rather than documentation defects, and it found it
by reading the *suppressed* half of the profiler's output. All three are recorded here with what they measured.

**The routed gate/up `in0` in L1 — taken, and the reason it was invisible.** `tt-perf-report` cannot model a
`sparse_matmul` row whose `nnz` is `std::nullopt`, and its advice generator *early-returns* on such a row, so
every committed report carried no `Bound`, no DRAM %, no FLOPs % and **no advice at all** on the two routed
projections — 31 % of the decode window and ~82 % of prefill — while printing a warning that said exactly that.
Passing `--active-experts` (8 for a batch-1 decode step, 162 for a 32-token prefill group: the expected distinct
union of 256 draws from 256 experts, which is the figure `probe_sparse_matmul.py` already tuned at) populates
both utilisations, classifies the rows `SLOW`, and raises three advice items on them. Two were already answered
elsewhere — the 1x1 output subblock is §4.14's territory and the HiFi2/HiFi4 suggestion is §4.2's — and the
third, "place input 0 in L1", was genuinely untried on that row. It costs no extra op: the per-group
`ttnn.slice` that produces `in0` names L1 instead of inheriting DRAM, ~131 KB for one group. Paired A/B, two
repeats per arm, and README §5.5's generated advice row prints the outcome:


**A shipped policy that could not run, found by shipping the SDPA change.** Raising the prefill SDPA chunk made
phase 3 of the sweep die on its `kv_cache_dtype=bfloat16` arm with
`TT_THROW: Statically allocated circular buffers ... grow to ... beyond max L1` — a bfloat16 cache doubles what K and V cost per chunk, so 256 does not fit where 256 with a BFP8 cache
does. `PREFILL_SDPA_CHUNK` is a *legality* table for that reason, and getting its key
right took two more rounds. Round 14 keyed it on `(kv_cache_dtype, sdpa_fp32_acc)`; round 15 found that **dead** —
every shipped policy sets `sdpa_fp32_acc=True`, both keys carried `False`, so every lookup missed, every policy
silently took the conservative fallback, and three documents claimed otherwise. It is keyed on the **policy name**
now, so a miss is a new policy rather than a field nobody checked, with a separate ceiling for a cache wider than
BFP8 — and that ceiling reads the **attached cache's** dtype rather than the policy's, because
`allocate_kv_cache(dtype=...)` and `attach_kv_cache` are a supported route that changes one without the other
(round 16). A too-large chunk is not slow; it is a throw at program construction, which is why all three facts are
asserted per policy by `test_every_shipped_policy_prefills_at_the_shipped_chunk` rather than left to a
build-and-run test that a legal fallback would always pass.

Chasing that key turned up a **pre-existing** defect the same error was hiding. `POLICIES["fused-parity"]`, the
policy README §4.1 describes as the fused decoder's exact dtypes, threw the same way on HEAD, before any round-14
change, and **no test in the stage ran it** — the suite only ever built `DEFAULT_POLICY`, and §4.2's sweep applies
`--set` overrides to the optimized policy rather than selecting a policy object. The cause is in
`_prefill_2d_matmul_config`: its L1 model hardcoded `tile_bytes_in1 = 1.0625 * TILE * TILE`, BFP8's bytes per
element, so under bfloat16 weights it under-predicted the `in1` circular buffers by nearly
double, declared the program legal, and left the throw to program construction. A model that only holds for one dtype silently mis-sizes every
other policy. It takes the policy's dtype now, and `fused-parity` prefills 2048 tokens at almost
exactly the *fused decoder's own* prefill time — the `POLICYPREFILL` rows of
[`logs/ab_routed_in0.txt`](logs/ab_routed_in0.txt) against README §5.2's generated before/after table — which is
the check that says the parity policy really does reproduce those dtypes rather than merely claiming to. `test_every_shipped_policy_prefills_at_the_shipped_chunk` gates all three
policies on both layer kinds now, so a policy cannot go back to existing only on paper.

**The prefill SDPA program config — swept and raised, closing README §9 item 7.** That item disclosed the config
as the one knob with no probe behind it, and round 14 correctly called that deferred work rather than a
limitation. [`logs/probe_prefill_sdpa.txt`](logs/probe_prefill_sdpa.txt) sweeps eighteen arms at the shipped
2048-token chunk — square pairs from 32 to 512, asymmetric pairs around the winner, two narrower grids, and the
bfloat16-cache case — and the shape of the result is: the op default and 32 are the two worst arms, each doubling
of the chunk up to 256 roughly halves the time, 512 does not build at *any* `k_chunk` pairing, the narrower grids
lose, and decoupling `k_chunk` from `q_chunk` buys nothing beyond the spread. `q_chunk` is the axis;
`PREFILL_SDPA_CHUNK` takes the winner, still clamped by the resume-offset divisibility rule and the physical
length, so a prefill resuming at a 128-token boundary still gets 128. The layer effect is in README §5.2's
generated table — the SDPA is `full_attention`-only, so `linear_attention` is unchanged — and the prefill
correctness cases (PCC, chunk-size invariance, continuation) pass at the new tiling.

A peer agent optimising the same op for a different model was measuring its occupancy at the same time, and the
two shapes together are worth recording. `sdpa_program_factory` spreads `B * NQH * q_num_chunks` chunk-pairs over
the grid, so a *larger* `q_chunk` means fewer pairs and lower occupancy: at their 6-head shape the inherited 128 fills under half the grid and 256
only about a fifth of it, and they measured 256 substantially slower — occupancy-bound, and their inherited value
was already optimal. (Their figures are theirs, measured on their model, so they are described here rather than
quoted as if this stage's artifacts contained them.) At this stage's 16 heads the same formula saturates the grid at every chunk below 256 and still puts 256
at 58 %, yet 256 is about twice as *fast* as 64. Both are consistent: occupancy binds until the grid
fills, and past that only per-core efficiency moves. The rule that survives both shapes is "raise occupancy until
the grid fills, then raise the chunk"; the rule that would have hurt either of us is "smaller `q_chunk` is
better", which is what an occupancy model alone suggests.

**Two typecast folds taken, one rejected as illegal.** Round 14 pointed out that `TypecastDeviceOperation` then
topped the `linear_attention` dispatch gap — the folds below are part of why it no longer does; README §7's
generated table ranks it fourth against the current capture — and that three of its seven
launches looked foldable into the op that produces them — the transformation §4.11 already applied for a measured win. In isolation all three fold cheaply - roughly a third off the `zeros_like` pair
and a fifth off each `multiply` pair, measured op-side before shipping either. Shipped: the two `multiply` folds
in the recurrent-state path, and the layer effect is in README §5.2's generated table, where traced
`linear_attention` decode drops by about twenty microseconds while `full_attention` is unchanged - those sites
are `linear_attention`-only, which is the check that the change did what it claims.
Rejected in that spelling: the router's `zeros_like(dtype=...)` is **illegal inside a trace region** — with a
`dtype` argument the op materialises its result with a host write, and `test_perf_decode_traced` dies on
`TT_FATAL: Writes are not supported during trace capture`.

Round 15 then pointed out, correctly, that this rejected a *spelling* rather than the candidate: the error proves
the zero operand cannot be **created** inside the trace, not that it must be created per step. `ttnn.scatter` is
out-of-place and decode replays a fixed shape, so the target is allocated once at build time and reused —
`_router_zeros_for`, the same persistent-tensor pattern this file already uses for the RoPE tables, `batch_idxs`
and `pos_ramp`. Both ops leave the step, and §3's ladder rows 16 and 17 carry the layer effect. A first API error
is not a rejection; this is the round that made that stick. Worth stating: the op-level figure said
the fold was the *largest* of the three wins, and it is the one that cannot ship — an isolated op measurement
cannot see a trace-region contract.

### 4.15 The routed gate/up `in0_block_w` — a phase-aware cap, taken

This is the one change a review round found by *measurement* rather than by reading. Review round 6 forced
README §5.4's generated sparse table to pin the shipped geometry — core count, orientation, output
placement **and `in0_block_w`**, the last read out of the suite log rather than assumed — and the table
immediately stopped calling the shipped batch-1 decode row "the measured winner". It is a couple of percent
behind, several times the measured spread.

The cause was a single divisor rule: `gate_up_in0_block_w = _largest_divisor_at_most(dim // TILE, 64)`, one
value for every call. The sweep says the two tuned points want *opposite* values, and not by a little
([`logs/probe_sparse_matmul.txt`](logs/probe_sparse_matmul.txt), at each point's own shipped core count,
orientation and L1 output):

| point | 32-tile inner block | whole tiled `K` (64) | winner |
| --- | --- | --- | --- |
| 8 active experts — batch-1 decode | faster by ~2 % | — | 32 |
| ~162 active — a 32-token prefill group | — | faster by ~10 % | 64 |

So one cap has to lose one of the two windows, and the shipped one was losing decode. `in0_block_w` is now
keyed off the same active-expert bound that already chooses the core count
(`SPARSE_GATE_UP_IN0_BLOCK_W`), which costs nothing and pays neither. Measured end to end, same harness,
same weights and one process: about 7 us off a traced decode step on **both** layer kinds, with prefill
unchanged — [`logs/ab_gate_up_in0_block_w.txt`](logs/ab_gate_up_in0_block_w.txt) has both arms, and it
measures prefill too, because a decode win that cost prefill would not be a win. README §5.2's generated
table carries the shipped level. `test_decode_runs_the_tuned_program_configs` and
`test_prefill_runs_the_tuned_program_configs` both assert the geometry each phase now builds, so the two
values cannot silently collapse back into one.

Worth stating plainly: rounds 2 through 5 fixed figures and found no performance, and this round found close to a
percent of decode — because the table was finally required to agree with the geometry the layer actually runs. That is
the argument for mechanical agreement over careful proofreading, in one data point. (Round 1 is not in that
count: §3.9 credits it with two real wins, the state L1 placement and the `repeat_interleave` tilize. Round 24
found this sentence and the one in §6's round-6 entry both writing it as though no earlier round had produced
performance.)

### 4.21 Carrying the residual norm's shard into the in-projection — taken, and the claim that blocked it

This is the one optimization in this stage that was closed by an assertion rather than by a measurement, and it
is worth recording how, because no amount of proofreading would have found it.

§3.6 and README §10 both said the 1D `mcast_in0` projection needs an interleaved `in0`, and that the
DRAM-sharded family "is the only one that consumes a width-sharded `in0`". On that basis the sharded-residual
family — which `$optimize` OPT-003 makes mandatory when a norm output is interleaved before a matmul — was
closed, and the layer paid a `sharded_to_interleaved` after every residual norm.

**The op says otherwise.** `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` validates a
sharded `in0` for `mcast_in0` explicitly, requiring WIDTH_SHARDED, ROW_MAJOR, `fuse_batch`,
`per_core_M == shard_shape[0] / tile_h`, and `(shard_shape[1] / tile_w) % in0_block_w == 0`. The decode residual
norm already produces exactly that shape, and `attn_in`/`gdn_in` ship an `in0_block_w` that divides the norm's
per-core shard width. The conversion was pure overhead.

**Where the false claim came from.** Not from a measurement — from the *probe*. Every `mcast1d` row in
`probe_dense_matmul.py` created its activation once, DRAM-interleaved, and reused it for every arm; only the
`dram_sharded` family ever resharded. So the artifact could not contain a counterexample, and the claim read as
though the sweep supported it. The probe now has a `mcast1d_sharded_in0` family, and the sharded-`in0` rows are
in `probe_dense_matmul.txt` alongside the interleaved ones.

**Taken, on a whole-layer A/B.** `logs/ab_sharded_norm_in0.txt` alternates the arms build-by-build, three timed
builds each, discarding each arm's first: every timed build of the shard-carried arm beats every timed build of
the interleaved-between arm, on both layer kinds. `_shard_feeds_projection` re-derives each of the op's
conditions rather than assuming them, so a role that does not qualify keeps the interleaved path, and
`test_decode_norm_shard_reaches_the_in_projection` asserts the projection really receives a width-sharded `in0`
— the test fails on the pre-round-25 path, which was checked rather than assumed.

**Not taken for the MoE norm.** Its consumer is `shared_in`, whose tuned `in0_block_w` is wider than the norm's
per-core shard, so the shard is not expressible there without narrowing the inner block — which
`probe_dense_matmul.txt` measures as slower. That is a measurement, not an assertion, which is the distinction
this section exists to make.

### 4.22 Two items review round 26 found by following round 25's thread

Round 25's finding was an optimization closed by an assertion rather than a measurement. Round 26 looked for
more of that class and found two, both downstream of the round-25 change itself.

**The decode output projection's `in0` placement — taken.** `tt-perf-report` raises "If possible place input 0
in L1" on the decode `o_proj` row, on every one of its launches. README §5.5 said the item was raised nowhere in
decode and attributed the one raised row to the head-dim norms, which is a different tensor. The cause is the
same shape as the routed `in0` item round 14 found: `_attention_output`'s gated multiply named no placement, so
it inherited DRAM from the paged flash-decode attention, whose output must be in DRAM. `linear_attention`'s
identically shaped `gdn_out` already ran from L1 only because *its* producer happens to be an L1 op — which is
what made the asymmetry visible. Taken, because it costs no extra op and is never slower at the layer; the
honest measurement is that the layer effect sits inside the build-to-build band
(`logs/ab_attn_out_in0.txt`), so this is an advice item cleared rather than a win claimed.

**The two sharded-`in0` in-projections' geometry — retuned.** Round 25 moved `attn_in` and `gdn_in` onto a
width-sharded `in0`, but README §5.4 kept selecting and ranking them in the DRAM-interleaved probe family. So
the table printed a time the shipped geometry does not have, ranked the shipped row against candidates the
layer no longer builds, and — the part that mattered — left both roles carrying an `in0_block_w` cap that had
been chosen under a placement they had stopped running. The sharded family prefers the opposite end of that
ladder. `make_readme.py` now picks the family per role, mirroring `_shard_feeds_projection`, and the corrected
table put both rows behind a candidate at `in0_block_w` 2.

Measured at the layer rather than adopted from the op rows (`logs/ab_dense_in0_block_w.txt`): every timed build
of the retuned geometry beats every timed build of the pre-round-26 one, on both layer kinds. `attn_in` moves
to 32 cores at `in0_block_w` 2, `gdn_in` keeps its core count and takes `in0_block_w` 2.

That A/B also has a lesson of its own. Its first committed run compared the shipped geometry against *itself*:
the "before" arm read `DECODE_MATMUL_GEOMETRY`, which by then held the adopted candidate, so both arms printed
the same number and the file looked like a null result. The before-arm geometry is written literally now, and
the script asserts that its after-arm really is what the module ships.

### 4.23 The decode V shard, thrown away and rebuilt — taken

Review round 27 kept pulling the thread rounds 25 and 26 had opened, and found the largest instance of it.

`nlp_create_qkv_heads_decode` emits V as HEIGHT_SHARDED L1 with shard `[32, head_dim]` on the first `batch`
cores. `_kv_update_memory_configs` then built *exactly that config* — so the layer converted V to DRAM
interleaved, zero-padded its kv-head dimension, and converted it back, once per decode step, to arrive at the
layout it already had. Three places called those conversions mandatory: §2 said the six `full_attention` decode
conversions were "all required by an op contract … No avoidable ones existed to remove", the topology audit
called the pad "irreducible at this layer", and README §6 and the churn test itemised them as contract-driven.

None of that was measured. `paged_fused_update_cache` reads the head count from the **cache**, not the input,
so the logical 2-versus-32 kv-head difference the pad existed to fix is never observed; its input rules are only
that the tensor be sharded, ROW_MAJOR, not width-sharded, with shard width equal to the last padded dimension
and a height its shard height divides. The head split's output satisfies every one of them as produced.

One thing had to move: the cache write's two grids are swapped, so **V** takes the first `batch` cores — the
range the head split emits it on — and K is resharded onto the second. The fused update requires only that its
two inputs be disjoint, so moving K costs nothing while moving V would cost the reshard this removes. Q and K
still interleave: both feed the norm and rope chain, and only V reaches the cache write untouched.

This paragraph originally named a second thing, `overlap_qk_coregrid=False`, and review round 28 found it
inert — the op's wrapper forces that flag to `True` for a non-sharded input, and this `qkv` is L1 interleaved.
Round 28 corrected the source comment and the A/B docstring and recorded the correction in §4.24, but not this
paragraph; review round 29 found it still here. Two of three places is how a correction becomes a
contradiction.

Measured at the layer (`logs/ab_v_shard_passthrough.txt`): every timed build of the passthrough beats every
timed build of the rebuild, by close to a percent of the step. The `full_attention` decode layout-conversion
budget falls from 13 to 11, and `test_decode_v_reaches_the_cache_on_the_head_split_shard` asserts both the
memory config the cache write receives and the op's disjoint-cores rule, re-derived rather than assumed.

**The pattern across rounds 25, 26 and 27 is worth naming.** Three optimizations were closed by a sentence
about what a TTNN op requires, and all three sentences were wrong in the same direction: they described the
shape the layer happened to be passing rather than the shape the op accepts. Two of them were reinforced by a
probe that only ever built the shape the claim asserted, so the artifact could not contradict it. The figure
audit cannot see this class — it checks numbers against artifacts, and these were prose about an API. What
catches it is reading the op's validation, which is now what §4.21, §4.22 and this section each cite.

### 4.24 The K cache-write pad — the same win, left on the table one round

Review round 28 found this by reading round 27's own change, and the observation is sharper than the fix.

Round 27 removed the interleave-pad-reshard round trip for V. K kept paying its pad. The shipped tree
therefore contained its own counter-example: **V reached the same `paged_fused_update_cache` call with its
kv-head dimension unpadded, and the full suite passed.** Both inputs could not both require the pad, and
the document beside them said both did.

The op reads its head count from the **cache**, not the input, and its writer kernel advances one row per
head for exactly that many heads. Rows past the real kv heads are never read, so they never needed
zeroing; none of the device op's validation constrains the input's head dimension, and the shard rules are
all on the padded shape, which a tile-layout tensor already satisfies. Measured at the layer
(`logs/ab_kv_pad_free_write.txt`), removing it is worth about half a percent of the step.

Two smaller things came out of the same round. `test_no_layout_churn_in_measured_forward` now counts
`pad` launches: a dead pad was invisible to every test, because the churn gate watched resharding and
relayout but not padding. And the pad that remains is now correctly described - the MoE group's tile pad
**is** load-bearing, because §3.4's masking depends on those rows being exactly zero, which is precisely
the property the cache-write pads did not need.

Round 28 also corrected a claim round 27 had written about its own change. `overlap_qk_coregrid=False`
was presented - in the source comment, in §4.23 and in the A/B's docstring - as what put K on a range
disjoint from V. It is inert: `nlp_create_qkv_heads_decode`'s wrapper forces that flag to `True` whenever
its input is not sharded, and this `qkv` is L1 interleaved. Q, K and V all come off the split on one
range, and the disjointness the fused write requires comes entirely from the two explicit cache-write
grids. The argument has been dropped. It is worth noting what this means about the class: the stage
wrote a *new* false op-contract claim in the very commit that removed three old ones.

---

## 5. Hardware

One incident, caused deliberately by the static-`nnz` experiment in §4.8 and fully recovered with a
single `tt-smi -r`; the table there records the failure signature, the commands, the reset and the
mesh smoke. `tt-smi -ls --local` showed all 8 Blackhole boards before the first run of this stage and
after the last. Watcher and profiler runs were kept in strictly separate processes throughout, and no
vLLM or serving process was started at any point.

---

## 6. Review rounds and checkpoint

Every `$stage-review` pass that ran against this stage, each by a fresh subagent, is recorded below — one entry per round, in order, and the list is the count. There is deliberately no total in this sentence any more. Round 16 corrected it from "two" and left it one short; round 17 caught that; round 23 found it still reading "twenty" with the narrative stopping at round 20 and rounds 21 and 22 unrecorded. A count is a figure like any other, this one is spelled in words so no gate sees it, and it is stale the moment another round runs — which is exactly the defect these rounds keep finding elsewhere. What the list says is worth stating plainly: it is how much of this stage's content came from being checked rather than from being written.

**Round 1** returned `more-work-needed` with five items: a device-capability query that could never
succeed (so every L1 budget ran against a 1 MiB fallback and the 2D prefill config was silently off
on the two widest projections), the `linear_attention` recurrent-state matmuls left unswept with
three open advice items, a `full_attention`-only accounting with an unclassified in-trace
stall, a `nnz` rejection resting on an inherited argument, and a `gdn_out` row that misdescribed the
shipped geometry and had been swept at the wrong activation dtype. §3.9 records what each one
changed; two of them were real performance wins and one of them (`nnz`) wedged the device when
measured properly, which is now the blocker of record.

**Round 2** returned `more-work-needed` with five more, all documentation-fidelity or small unclosed
items, all fixed:

* the `nnz` rationale in `../context_contract.json` still carried round 1's disproved reason — rewritten
  to the reproduced hang and pointed at `triage/`;
* `place input 0 in L1` was still open on the shared expert's down projection while the README's
  hand-written advice table attributed the count to the rows already fixed. The advice table is now
  **generated** from the committed reports (an item with no recorded action renders as
  *unclassified*, and an item this stage closed moves to a second table), and the shared expert's
  SwiGLU product moved to L1;
* the `in0_block_w` advice on the state matmuls had been closed with the claim that the op family
  takes no such field. It does — the batched non-mcast `MatmulMultiCoreReuseProgramConfig` — so the
  three matmuls now carry explicit program configs: `in0_block_w` 2 for the reads and 1 for the
  `transpose_a` outer product, where `Kt` is a single tile and 2/4 are rejected by the op. README
  §5.5's generated advice row carries both times, read out of the probe (§3.5 has the win);
* README §6's flat "no tilize/untilize appears in either measured path" was contradicted by this
  stage's own tables — restated to separate what the *model* dispatches from what three composite
  ops lower to, with their per-step cost;
* the work log's own closing figures were a pre-fix run — that paragraph now defers to README §5.2's
  generated table, and the difference between the fused stage's published baseline and this stage's
  re-measurement of it is stated (§3).

Round 2's other concerns were addressed in the same pass: the `linear_attention` roofline now counts
the recurrent state the way the other kind counts its KV read, the "same-process" method claim is
corrected to name the test that actually is one, README §5.4 says its core counts are program grids, the
buffer-side L1 budget absorbs the ~6.5 % gap between `get_max_worker_l1_unreserved_size()` and the
allocator's bank size, and the state-L1 win got its own `ab_*.txt`.

**Round 3** returned `more-work-needed` with three items, all closed:

* **P1, artifact plumbing rather than the model.** The repo's `.gitignore` has a blanket `*.csv` and
  `run_profiling.sh` only gzipped files over the 500 KB hook limit, so
  `full_attention/decode_perf_report.csv` (477 600 B) was never committed — `make_readme.py --check`
  passed against the working tree while the *committed* tree could regenerate neither README §5.5 nor
  §7, and `perf_accounting.py` quietly wrote a half accounting. Every report CSV is now gzipped
  unconditionally, `perf_accounting.py` raises instead of writing a partial summary, and
  `run_evidence.sh` ends by re-running all three generators against `git archive HEAD`.
* the generated advice table now covers **all four** committed reports rather than the two decode
  ones. That surfaced `place input 0 in L1` still open on three prefill rows; measured and rejected
  with numbers (an L1 `in0` is *slower* on `attn_in` and `gdn_in`, and worth ~3 µs on `shared_in` —
  a rounding error of the prefill window), `probe_prefill_matmul.txt`;
* hand-written µs figures in this log, the README and the code docstrings were re-derived from the
  committed probe artifacts, which a later `run_evidence.sh` re-run had left behind — NORM, the state
  outer product, SPLIT, GATE, the fused sparse geometry and two DRAM-sharded rows. No decision
  direction changed. **Round 4 then found this fix incomplete** (eight figures still wrong), which is
  what finally replaced "re-derive them by hand" with a mechanical gate; see round 4 below.

Round 3's other concerns were closed in the same pass: the state program configs now have a test gate
(they go through `ttnn.matmul`, which the dense-config spy cannot see), the shared expert's L1
placement gained the per-call-token guard the routed chain has, `_delta_rule_step`'s `dram`-named-but-L1
local is renamed with its docstring corrected, the five unqualified `work_log.md §N` references that
meant the *fused* stage are qualified, and README §9 item 5 lists all three batch thresholds including
the recurrent-state config's batch-3 bound.

**Round 4** returned `more-work-needed` with three items. Two were the same defect class as round
3's third item, which is the point: three consecutive rounds had closed on "the figures are
re-derived" and the figures were still not derived. This round replaced the promise with a gate.

* **P1 — the dense-prefill search table quoted a superseded run of its own probe, in every row.**
  Round 3's fix to `probe_prefill_matmul.py` added an `in0=L1` arm to measure the `place input 0 in L1`
  advice, and held that 8 MB L1 copy *resident across both arms*. On an 11×10 grid that is ~76 KB per
  core taken out of the same L1 the matmul's circular buffers are charged against, so the `in0=DRAM`
  rows were measured under an L1 pressure the shipped layer does not have. It slowed several rows and
  made `gdn_in` at `in0_block_w` 8 — the geometry `PREFILL_MATMUL_IN1_TILE_BUDGET` selects and the
  layer runs, asserted by `test_prefill_runs_the_tuned_program_configs` — report as failing to build.
  The artifact therefore contradicted the running code, and the prefill arm of the L1 advice was
  rejected against a baseline that did not exist. Fixed by running the two arms as separate passes,
  with the L1 copy allocated only during its own pass; the clean `in0=DRAM` arm confirms the shipped
  `in0_block_w` is the measured winner for **all six** roles. §4.10 has the corrected table, which is
  now generated (below) rather than transcribed.
* **P2 — eight figures still contradicted the artifact they cited**, three of them inside
  `make_readme.py`'s `ADVICE_ACTIONS` prose, where `--check` could never see them: that table is
  hand-written text spliced into a *generated* block, so the generator regenerated the block from the
  same stale prose and agreed with itself every time. Also caught: the module docstring's own A/B
  table, which disagreed with `ab_fused_vs_optimized.txt` in five of eight cells and which no round
  had flagged.
* **P2 — no figure-audit gate, although the preceding stage had built one for this exact failure mode
  and this stage's code still referenced it.** `doc/fused_decoder/audit_figures.py` existed;
  this stage had not carried it.

What changed, and why these three cannot recur rather than merely being fixed:

1. **`audit_figures.py` is ported** and runs last in `run_evidence.sh`, in-tree *and* against
   `git archive HEAD`. It asserts every decimal and every multi-digit integer quoted in the README,
   this log, the capability contract, `CLASSIFICATION.md`, `make_readme.py` **and the implementation
   and test sources** appears in a committed artifact; evaluates the declared derived figures and
   requires their operands to be sourced too; refuses a suite log with no pytest summary line; and
   re-runs all three generators.
2. **`logs/source_manifest.txt`** records the sha256 of the implementation, the tests and the
   suite conftest *before* the first device run, and the audit re-hashes them. Round 4's hard-check gap
   was six probe artifacts predating a source edit with nothing to prove the edit was inert; now a
   source edit after the evidence run fails the gate.
3. **No run-varying timing is quoted in `tt/optimized_decoder.py` or the tests any more.** They carry
   the decision, the shipped configuration and the artifact rows to read. This is the structural
   reason the docstrings kept going stale: a code comment has no generator and no reviewer re-reads it
   after a re-run.
4. **Three more README blocks are generated**: the dense-prefill search table (§4.10 / README §5.4),
   the op-to-op gap itemisation (README §7) and every microsecond figure inside the advice actions
   (README §5.5). The prefill table's last column re-checks each shipped row against the whole sweep,
   so "it is the measured winner" is asserted on regeneration instead of claimed in prose, and its
   config column is read out of the suite log — i.e. out of a run of the shipped code.
5. **The capability contract's footprint terms are measured, not modelled.**
   `logs/probe_footprint.py` walks every device tensor of a built layer at its allocated padded size
   under both policies. The hand-modelled figures were a few hundred kilobytes low on the worst-case layer: they
   omitted the RMSNorm gains and the RoPE transformation matrix, counted the batch-1 conv state at its
   logical rather than its padded size, and over-counted the projection weights. The direction of the
   claim is unchanged (63 % less on a `full_attention` layer, 69 % on a `linear_attention` one, 40×
   DRAM headroom) and the corrected total is *larger*, which is the conservative direction.

Round 4's other concerns were closed in the same pass: the last three unqualified section references
in the tests now name `doc/fused_decoder/`, README §7's typecast-gap count is generated rather than
counted by hand (there were three, not two), the prefill `in0=L1` rejection now also measures what the
enabling DRAM→L1 copy costs — 12–50 µs per role against a 2–17 µs saving, so it is a net loss on every
role and the rejection no longer rests on "it is only a few microseconds" — and the prefill probe rows
carry a measured `spread=` so the "inside the run-to-run spread" arguments are checkable.

Six further defects surfaced *while* fixing the above, none of them in any review round's list. They are
recorded here because each one is evidence that the gate was the right fix rather than the numbers:

1. **The module docstring's own before/after table** disagreed with `ab_fused_vs_optimized.txt` in five
   of its eight cells. No round had flagged it; the audit flags it, and it is gone — the docstring now
   states the ratios and points at the generated table.
2. **`probe_sparse_matmul.txt` was internally inconsistent**: 472 of its 1888 rows carried no `active=`
   field, because that section had been produced by an earlier revision of the probe than the other
   three. One `run_evidence.sh` pass now produces all four sections from one script, and the generated
   sparse table reports any row it has to exclude.
3. **README §10's checklist claimed the recurrent-state matmuls "take no `in0_block_w` at the API"** —
   the exact claim review round 2 disproved and this stage then fixed. The row had never been updated.
4. **README §5.4 said the shipped sparse rule "reproduces the measured winner at all four points"**,
   and its own regenerated table showed gate/up at 8 active experts landing 2–3 % behind. The rule
   reproduces the winning *core count* everywhere; `in0_block_w` comes from a separate divisor rule and
   is 2–3 % off at that one point, a fraction of a percent of a traced step. Now stated, with the gap
   computed from the probe.
5. **The first attempt at deriving the prefill window composition was wrong by two orders of
   magnitude**: `op_device_time` matched op codes by substring, and `MatmulDeviceOperation` is a
   substring of `SparseMatmulDeviceOperation`, so the *dense* projections were credited with 82 % of a
   window they are well under one percent of. Matching is by prefix now, and the function says why.
6. **The op-to-op gap itemisation was reading the report wrong.** Its rows are one op *per replay*, not
   one aggregated row per op, so the first version reported 32 copies of each gap and a per-step total
   32× too large. Grouped by op code and divided by the replay count, the itemisation reconciles with
   README §7's dispatch gap — and it shows the float32 gate-promotion typecasts are the *largest* line item of
   the `linear_attention` window, not the "two 6–8 µs gaps" the prose claimed.

**Round 5** returned `more-work-needed` with seven items. It confirmed the round-4 structure holds
mechanically — all four generators reproduce byte-identically in-tree and from `git archive HEAD`, and all
six self-reported defects check out — and then made the finding that matters most in this stage's history:
**the audit installed to stop figure drift was measurably permissive for exactly the figure class that had
drifted.** The reviewer measured it, which is the right way to argue about a check: with the four
multi-megabyte per-op `tt-perf-report` CSVs in the sourcing pool, `sourced()` returned true for **794 of
2000** arbitrary one-decimal values and **1581 of 2000** arbitrary three-decimal millisecond values. A
per-op report is thousands of durations; almost any plausible microsecond figure appears in it somewhere.
Two live wrong figures had cleared the gate on exactly that route.

What changed in the gate:

1. **Sourcing is now a labelled-token test, not a substring test.** `measured_tokens` extracts every value
   an artifact prints *as a measurement* — `us=`, `wall/iter=… ms`, `pcc=`, `bytes=`, a JSON key, a census
   count, a table cell — and membership is an exact string match, so a two-decimal figure is no longer
   "sourced" by a three-decimal measurement that merely starts with the same digits. That prefix match is
   precisely how a wrong prefill percentage had survived.
2. **The per-op reports are out of the sourcing pool** (still checked for existence and freshness). They
   reach the documents only through a generated block, whose agreement with them `make_readme.py --check`
   establishes separately. Re-measuring the reviewer's experiment against the new rule: three-decimal
   millisecond values, one-decimal microsecond values and byte counts all fall sharply — the current rates
   are in [`logs/audit_selftest.txt`](logs/audit_selftest.txt), regenerated by `run_evidence.sh` and diffed
   by the audit, rather than quoted here where they would go stale. The one-decimal class is the residual
   weakness and it is
   inherent — the probes genuinely print hundreds of distinct one-decimal microsecond values — so the
   docstring states the number rather than implying the check is airtight.
3. **A figure is checked against the artifact its own paragraph cites**, when it cites one. Most of this
   stage's prose names its log, so most figures are now checked against one file of a few kilobytes. A
   failure reads `MISCITED` rather than `UNSOURCED` when the value exists elsewhere in the pool, because
   citing the wrong log and inventing a number are different defects.
4. Smaller gate fixes, each of which had let something through: truncation-tolerance restricted to the
   5+-decimal PCC class, since at four decimals it registered the two-decimal prefix of every
   three-decimal timing and so re-created the prefix hole it was meant to close; `HISTORICAL` no longer
   applies to code
   comments, which had immediately hidden a live wrong claim in a generator; the integer pass scoped to
   documents, since an integer in code is a shape; space-separated thousands read as one number;
   scientific notation, mathematical exponents and rule identifiers (`OPT-013`) no longer read as figures;
   the contract's scoped section dumped with `ensure_ascii=False`, which had been turning a section
   reference into a phantom figure; both harness scripts added to the audited set; and
   `check_generators` now diffs the summaries it rewrites instead of trusting their exit code.
   `audit_figures.py --selftest` now measures the gate's own false-positive rate per figure class into
   [`logs/audit_selftest.txt`](logs/audit_selftest.txt), so a later change that weakens the matching rules
   shows up as a committed number moving rather than as nothing at all.

The other six findings, all fixed:

* **The generated dense-decode table attributed times measured on the wrong grid, and in three rows the
  wrong output placement.** The lookup matched `(in0_block_w, per_core_N)` on the stated theory that
  `per_core_N` pins the grid; it does not, because for a narrow output every core target from 24 to 110
  gives `per_core_N` 1. It now matches the *realised* grid (`11 × ceil(target/11)`) and the shipped L1
  output. This was round 4's own defect class inside round 4's fix.
* **The generated sparse table checked one of its eight rows** and printed the literal "as measured" for
  the other seven, four of which are 1–3 % behind the other rectangle of the same core count. Every row is
  checked now — core count, `in0_block_w`, `per_core_N`, placement *and grid orientation* — and the gap is
  printed when the shipped choice is not the winner. See §4.14 for what that gap turned out to be.
* **The `Output subblock 1x1 is small` action rendered "against unmeasured"** and claimed a rejection for
  a second row whose alternative the sweep never contains. Fixed, and the `router` row now says *not
  expressible* — `Nt` is 8, so `per_core_N ≥ 2` needs ≤ 4 cores and the ladder starts at 8 — instead of
  implying a measurement.
* **README §4.3's BFP4 headline quoted a superseded A/B** that contradicted the generated table three
  lines below it. Both figures were in the audit's own "found wrong, must not come back" list and passed
  anyway, on the substring route. Now generated.
* **"No run-varying timing is quoted in the implementation or its tests" was false** — six absolute
  figures survived round 4's sweep, and three of them disagreed with each other about a single
  measurement (the norm win, quoted as ~12, ~9 and 8.0 µs in three places, with one citing an artifact
  that does not contain that arm at all). Removed; the claim is restated as *no absolute* timing, and the
  audit checks every decimal in both files on every run.
* **README §7's generated limitation list was `full_attention`-only and unlabelled**, under a per-kind
  table — a partial regression of round 2's finding, in generated form. It is per-kind now.

One hard-check gap round 5 listed is closed by a new test rather than by prose: **nothing asserted the
*prefill* sparse-matmul geometry**, which is ~81 % of the prefill window's device time. Only the decode
configs were logged and gated, so README §5.4's prefill sparse rows were recomputed from the layer's rules
rather than read from a run of it. `test_prefill_runs_the_tuned_program_configs` now spies
`ttnn.sparse_matmul` too and asserts that both routed calls of a 2048-token chunk carry a program config,
write their `num_experts`-wide output to L1, and run on a **wide** grid — a prefill group activates most of
the 256 experts, and applying the 8-core decode geometry here was measured at roughly 4x slower (§3.1). It
logs 128 calls resolving to 2 distinct configs (`4-8`, `in0_block_w` 64 and 16), which is what the
generated table's ~162-active rows describe.

One more defect surfaced while closing round 5, and it is a *pipeline ordering* bug rather than a figure:
`run_evidence.sh` hashed the sources **before** the repo's pre-commit hooks had normalised them. The hooks
rewrite trailing whitespace, end-of-file newlines and `black` formatting at commit time, so the bytes the
evidence measured were not the bytes that got committed, and `audit_figures.py` correctly refused the
evidence as having been produced by a different revision — after a full two-hour sweep. The script now runs
the hooks over the three source files as phase 0a, before writing the manifest, so a run hashes what will
actually ship and is idempotent under commit. Relatedly, `check_freshness` now defers to the manifest's
sha256 rather than to mtimes: a hash comparison strictly dominates a timestamp, it works inside a
`git archive` extraction where every file carries the commit time, and it does not fire on a
content-preserving rewrite. mtime ordering is reported only when the hashes actually disagree, where it
usefully says which artifacts fall on the wrong side of the edit.

A related correction to the audit itself: four of its own `DERIVED` entries were expressions over
*hardcoded millisecond operands*, so the first re-measurement under them made every one of those operands
vanish from the artifacts — this file's disease, reproduced inside the file meant to cure it. The headline
speedup ratios are computed from `ab_fused_vs_optimized.txt` at audit time now (`derived_from_artifacts`),
at the precisions a document might quote, so they cannot go stale.

Round 5's other concerns closed in the same pass: the garbled `work_log.md` fragment round 3's edit left in
`_gdn_out`'s comment; the checklist's claim that both routed matmuls run `BF16 x BFP4` (the down projection
reads the BFP8 expert activation, which is the policy working, and the row now says so); the router's
prefill config being unswept, now stated as such where the table is introduced; the contract's footprint
note listing three of the four modelling errors; the `SLOW` row count read as exhaustive when it is a
classifier threshold that flaps between replays; and the note in `run_evidence.sh` that the `git archive`
run proves reproduction rather than freshness, because the archive stamps every file with the commit time.

**Round 6** returned `more-work-needed` with seven items. Two matter more than the rest, and one of them is a
*performance* finding rather than a documentation one — the first since round 1, which §3.9 credits with two.

* **P1 — the audit exempted every markdown-bold figure, which is how nearly all of them are written.** The
  exponent exemption added in round 5 (`\*\*\s*-?[\d.]+`, for `head_k_dim ** -0.5`) also matched a markdown
  **bold opener**, so it stripped the `**` *and the number* behind it. The reviewer proved it by injecting
  a fabricated bold microsecond figure and a bold `1.07 %` — round 5's own P1 value — into the README and
  watching the audit pass.
  A second consequence: the `LABELLED` rules ran on the post-strip text, so all three of them matched
  nothing at all across all 17 documents. The exemption now needs whitespace *after* the `**`, which
  markdown bold never has; the injection is caught; and the labelled pass runs on the pre-strip text and
  **reports how many claims it matched**, because a pass that silently matches nothing is not coverage. It
  legitimately matches zero today — every such claim is inside a generated block — and the number says so.
* **P1 — the generated sparse table did not pin the shipped geometry, and fixing it exposed a real
  sub-percent decode win.** The lookup filtered on the core count and output placement only, took the minimum over
  everything else, and so reported the shipped 8-active gate/up row as "the measured winner" at an
  `in0_block_w` the layer does not run. With the geometry read out of the suite log — a run of the shipped
  code — the row is 2.2 % *behind*, beyond the measured spread. That is not a documentation defect: the
  shipped `in0_block_w` came from a single divisor rule, and the two tuned points want opposite values
  (§4.15). Making the cap follow the active-expert bound, exactly as the core count already does, is worth
  about 7 us of a traced decode step on both layer kinds at unchanged prefill
  ([`logs/ab_gate_up_in0_block_w.txt`](logs/ab_gate_up_in0_block_w.txt)). Five review rounds of table-fixing found no performance; the sixth found this because
  the table was finally forced to agree with the running layer.

Three more findings were *claimed fixes from round 5 that never landed in the file*. Each had been written
inside a multi-edit script that hit a late assertion and exited before writing, so the earlier edits in the
same script were silently discarded while the transcript recorded success: README §7's limitation list was
still `full_attention`-only, `check_generators` still trusted an exit code instead of diffing the summaries
it rewrites, and the two harness scripts were still outside the audited document set. All three are in
place now and verified by re-reading the file rather than by trusting the edit. **The process lesson is
recorded here deliberately**: a claim that a fix landed is worth nothing without re-reading the artifact,
and this stage produced three of them in one round.

The remaining findings:

* the audit's own `--selftest` artifact was **in the evidence pool it measures**, so the experiment was
  self-referential and never converged — three consecutive runs produced three different files — and
  `run_evidence.sh` never regenerated it. It is out of the pool, called by the sweep, diffed by
  `check_generators`, and reproducible (verified by running it twice and diffing). Its rates are no longer
  quoted in prose, where they had drifted into three mutually inconsistent statements of one measurement.
* the selftest now also measures **two-digit integers**, which `ALLOWED_INT` exempts wholesale: that row
  reads 1.0 by construction, which is the point — the size of the exemption belongs in the artifact rather
  than in a reader's head.
* `HISTORICAL` still exempted integers in `.py` documents; scoped to `work_log.md` like the decimal pass.
* the norm-win figure was still quoted as "~12 µs" in the work log and "~9" in the README against the
  generated table's own value; both now defer to the table.
* the last absolute timing in the implementation (a "23 us" op-to-op stall) is gone, and three comment
  residues from in-place edits — a duplicated half-line, two sentences jammed together, an un-reflowed
  line — are fixed.
* `make_readme.py`'s `realised()` mirrored the layer's core-target rule wrongly below 11 cores (latent: no
  shipped role uses a sub-11-core grid), and nothing checked that its *other* mirrored constants still
  match the implementation. `audit_figures.check_mirrored_constants` now parses them out of the source with
  `ast` — no ttnn import — and I verified it catches an induced drift.

**The sharded-norm core count (round 6's P2-6) was measured and the shipped value kept, for a corrected
reason.** The reviewer was right that the isolated `NORM` rows contradicted the documented monotonicity
claim, that the A/B cited in its defence varies a *different* knob, and that 4 and 32 cores had never been
measured whole-layer. Both gaps are closed: the micro-probe's `NORM`, `TOPK`, `GATE`, `SPLIT` and `SDPA`
sections now report a measured `spread=` like the matmul probes (round 5 had fixed only those), and with
repeats the isolated ladder turns out to be *monotonic* — 4 fastest, then 8, 16, 32, 64 — so the earlier
non-monotonicity was single-shot noise, which is exactly what a spread exists to reveal. The new
whole-layer A/B ([`logs/ab_norm_shard_cores.txt`](logs/ab_norm_shard_cores.txt)) then shows all of 4/8/16/32
landing inside the layer harness's own run-to-run band, so it does not rank them. In the committed run the four
arms span barely more than a microsecond on either layer kind — a couple of
microseconds, with 8 nominally first on both. The *previous* run of the same file put 8 last on
`linear_attention`, which is the point: this artifact resolves nothing, and any ranking read out of it is a
reading of that run's noise. Review round 11 found this paragraph, the source comment and the A/B
script's docstring all claiming 8 was "marginally best on both layer kinds", which is the artifact read
backwards — and it was the replacement for the claim round 6 found wrong, which makes it the third round on
this one constant.

What the evidence supports is narrower: **the knob does not matter at the layer**, and 8 ships because it is
the shard count every other piece of norm evidence in this stage was measured at (`ab_norm_shard_width.txt`,
the §3 development ladder), not because it is fastest. That is now the single stated reason in all three
places; §3.6 previously gave a second, inconsistent one.

One loose end, stated rather than glossed: the boundary conversions are the obvious reason the op-level ladder
does not transfer, but nothing here establishes the sign at 4 cores either way — the earlier version of this
paragraph asserted the layer was "a microsecond slower there", and the run that was written from had it faster.
The honest summary is that the isolated ladder does not predict the layer at this knob and
the layer does not resolve the arms, so neither number is a fact about the hardware. It is not investigated
further because every arm sits inside the harness's own spread, so nothing measurable rides on the
explanation — but that also means no ranking may be quoted from this artifact, which is the mistake rounds 6
and 11 both caught here.

**Round 7** returned `more-work-needed` with eight items. Three were **defects in round 6's own fix**, which
is the pattern to notice: every round that changes a generator or the gate introduces a new way for the same
class of error to appear, and only an independent pass finds it.

* **P1 — the generated sparse table reported an `in0_block_w` the layer does not build on three of its four
  gate/up rows, including the prefill geometry.** The generator read the shipped cap from the suite log's
  *decode* line only, justified by a docstring saying "`in0_block_w` is a function of `K` alone" — true
  before round 6 and made false **by** round 6, whose entire point was to make the gate/up cap depend on the
  realised core count as well. So the table put the decode cap on the prefill row and understated shipped
  prefill sparse performance by ~10 % on the largest op in that window. It now reads both the decode and
  prefill lines (the new prefill test logs the latter) and picks the phase from the realised core count. Two
  secondary bugs surfaced while fixing it: the prefill marker was matched with a trailing colon that the log
  line does not have, so every wide-geometry row silently rendered "no probe row at the shipped geometry".
* **P2 — the new discriminator keyed off the *target* core count instead of the realised one**, and at
  exactly one batch size that mattered: 3 real rows give a bound of 24 and a target of 12, which is above
  `SPARSE_MIN_CORES`, while the grid actually built is still 8 cores — so batch-3 decode got the 8-core /
  wide-block pair that round 6 had just removed. The cap is derived after the `Nt` divisor reduction now,
  via a shared `_sparse_n_tiles` helper so the layer and the generator cannot disagree about it.
* **P1 — the round-6 replacement for the bold-figure exemption still stripped a figure after a bold
  *closer*** (`is **taken** 777.7 µs`), which is ordinary prose. That was the third failed attempt to
  separate a spaced mathematical exponent from a markdown bold marker by whitespace rule, and there is no
  such rule: a bold closer followed by a figure looks exactly like `x ** -0.5`. The spaced exponents are
  exempted **by value** now (there are two, both the DeltaNet key scale), the tight form requires the minus
  sign so `x**888.8` no longer leaks, and the scientific-notation exemption was narrowed to the exponent so
  a hand-written mantissa is still checked. Verified against the full injection set: bold opener, bold
  closer, tight form, table cell and plain are all caught.
* **P2 — nothing gated the new constant.** `check_mirrored_constants` covered the core-count rules but not
  `SPARSE_GATE_UP_IN0_BLOCK_W` — the very assumption whose drift caused the P1 above — and neither config
  test asserted the sparse `in0_block_w`, so setting the cap table to a single value passed the whole suite.
  Both tests assert it per phase now, derived from the layer's own rules rather than from a literal, plus an
  explicit assertion that the two phases' caps *differ*; I verified a collapsed table fails them. The
  constant is in the mirror check, and an induced drift is caught.
* **P1 — three documents still stated the pre-round-6 single-cap rule**, including two stacked stale comment
  blocks in the implementation left by the round-6 edit itself. All rewritten to the shipped phase-aware
  rule; the stacked blocks are one block.

The remaining items:

* README §5.5's sharded-norm knob row cited `ab_norm_shard_width.txt` as showing the conversions to be the
  smaller cost, and **both of that A/B's arms are sharded** — it varies which norms shard, not whether they
  do. This is round 6's "the A/B cited in its defence varies a different knob", recurring one row over. The
  row now cites `ab_norm_shard_cores.txt` for the core count and names the development-ladder row as the only
  layer-level sharded-vs-interleaved comparison there is.
* `ab_norm_shard_cores.txt` was in `EXEMPT_FROM_FRESHNESS` under a rationale ("not regenerable, needs a
  variant of the implementation") that does not describe it — it swaps a class attribute at runtime and the
  sweep regenerates it. Removed from the exempt set, and README §5.1's disclosure of that set corrected.
* the last "~9 µs" norm figure, in a code comment where the integer pass does not look; removed.
* `ab_norm_shard_cores.py`'s own docstring still asserted the non-monotonicity its artifact had disproved.

**The SDPA `q_chunk=0, k_chunk=0` candidate was measured, and the review's own reasoning about it was
wrong.** Round 7 observed it measured on all four grids and dispositioned nowhere, and inferred from its
`pcc_vs_default` fingerprint that it would collapse like the `k_chunk 128` candidate. It does not: with only
that config changed, the **whole 109-case suite passes**, and every case candidates A and B fail passes with
worst PCC 0.999865. So the correctness argument that rejects those two does not apply here. What rejects it
is the layer, where the isolation win almost entirely disappears because `SdpaDecode` is 2 % of the window —
§4.5 has the measured pair, and round 8 corrected the "inside the spread" framing this sentence used. Recorded as candidate C in
`ab_sdpa_decode_contract.txt` with both arms, rejected on *no layer-level gain* rather than on correctness —
and `k_chunk_size` stays pinned to `page_block_size` because candidate A is the proof that exceeding the
page block is silently wrong, and delegating the choice to the op makes that invariant uncheckable.

**Round 8** returned `more-work-needed` with six items. Two were substantive measurement problems, two were
defects inside earlier rounds' fixes, and one corrected a claim I had made without measuring it.

* **P1 — every SDPA row the decode program-config decision rests on was measured under a compute-kernel
  config the layer deliberately does not use: the exact HiFi2 + fp32-dest-accumulate config recorded two
  sections above as collapsing decode PCC.** The probe passed it to *every* arm, including the
  `default(None)` reference, so "the op default is an order of magnitude slower", the grid ranking and the
  chunk ranking were all measured against something the layer never builds. This is the same class as round
  7's P1 — a table reporting a configuration that is not shipped — one artifact over. The sweep runs at the
  shipped contract now (no compute-kernel config), the rejected config is kept as one extra labelled arm so
  its cost stays visible, and §3.8's conclusions are restated from the corrected rows. The direction survives
  — the op default is still more than an order of magnitude slower — but the numbers moved, and the rejected
  config turns out to be *slower* as well as wrong.
* **P1 — the §4.14 orientation figures had the `down` sign inverted at the prefill point**, and the
  rejection argument in that section rested on the inverted value ("the only tuned point where it loses").
  The artifact has the shipped column orientation winning **both** prefill rows; README §5.4's *generated*
  table said so all along, and the prose beside it disagreed. That is the strongest argument yet for
  generating tables: the round-5 replacement text for a claim the artifact contradicted itself contradicted
  the artifact, in a way `audit_figures.py` cannot see by construction — it checks whether a figure exists,
  not whether it supports the sentence around it.
* **P2 — `make_readme.block_sparse_search` still keyed the phase off the *target* core count**, which is the
  defect round 7 fixed in the layer, re-introduced in round 7's own generator fix. It agrees with the layer
  at the four active points the probe measures and diverges at 24, 40, 48, 72 and 96 — so it was correct
  today and would have silently mis-keyed any point added later. The generator now mirrors the layer's
  `Nt` reduction, and `check_mirrored_constants` actually reads `make_readme.py` instead of only claiming to.
* **P2 — `check_freshness` was inert in every passing tree.** Round 5's fix made the sha256 manifest
  authoritative by returning early whenever it matched, which is the passing case — so the mtime pass never
  ran, and round 7's "removed from the exempt set so the freshness rule applies" bought nothing. The two
  checks catch different things: the manifest proves the artifacts came from these bytes, but a sweep that
  dies part-way leaves a *matching* manifest beside artifacts from the previous revision, and only mtime
  ordering sees that. The pass runs unconditionally now, skips the uniform-timestamp signature of a
  `git archive` extraction, and downgrades a newer-source-with-matching-hash to an advisory **note** rather
  than a failure. Verified in all three states. The exempt set is also now defined mechanically — exempt iff
  no phase of `run_evidence.sh` writes the file — and a new check enforces that, because the previous
  rationale described two files it did not exempt.
* **P2 — the SDPA candidate's rejection rested on a spread nothing measured**, and the separable variant its
  own rationale pointed at had never been tried. Both are now measured, and both matter: `q_chunk=0` with
  `k_chunk` still pinned to the page block measures **identically** to the shipped arm, so the isolation win
  is entirely k-chunk-side and there is no safe way to take it; and at the layer, with three repeats per arm,
  the candidate looked reproducibly a microsecond or two faster rather than tied in that round's artifact -
  which round 10 then found reversed in the next run, so §4.5 now rests on the op-level gap and the layer-level
  dead heat rather than on either direction of a sub-spread difference. My "inside the spread" wording
  was simply wrong, and measuring the spread is what showed it. The rejection now rests on the trade — a
  fraction of a percent of one window against replacing a checkable invariant with trust in an op's internal
  chunk choice, where the failure mode is a silently wrong answer — which is a defensible reason where "it's
  a tie" was not.
* **P2 — the traced-decode figure differs ~2 % between harnesses, and round 8's explanation was wrong.**
  The review attributed it to the 88 MB trace region two A/B harnesses reserve.
  [`logs/ab_decode_harness.txt`](logs/ab_decode_harness.txt) measures that directly and rules it out: the
  same harness reports the same figure with the region reserved and with it zero. What is left is the
  multi-build process — those harnesses build several decoders in one device session to compare arms back to
  back. README §5.1 now says which harness each number belongs to and why the difference does not affect any
  A/B (every arm in a file shares its harness).

Round 8's smaller concerns closed in the same pass: the advice table truncated op codes without an ellipsis,
printing shapes that do not exist; the two search tables used different spread rules, and the looser one
(widest spread over a whole role sweep) could hide a real sub-microsecond gap — both are per-row now, which
immediately surfaced `gdn_in` sitting 0.5 µs behind another target, recorded with the alternative named and
not taken — the generated table states the gap as a share of that op, and it is under one percent of one op
of a decode step; the "two largest ops in both windows" claim, where the routed `down` is
actually third on `linear_attention`; the unswept **prefill** SDPA config, now a named limitation with its
measured share of the window rather than an omission; and the benign log noise a reader meets in the
artifacts (`nanobind` teardown leak lines, `tt-perf-report`'s "Unclassified operation" warnings for this
model's dedicated ops, and the `conv1d` capability probe's `TT_FATAL` bursts), now disclosed in README §1.

**Round 9** returned `more-work-needed` with five items. Two were shipped-code findings — the first in three
rounds — and the pattern in them is worth naming: **both were geometries where the op-level sweep and the
shipped value disagreed and no document said so.** Round 9 is also the round where measuring a reviewer's
own suggested fix reversed it.

* **P1 — the routed `down` grid orientation, and a documentation sign that had now been wrong in two
  directions.** Round 8 corrected the prefill `down` orientation row; round 9 found the correction had gone
  *past* the artifact, so `work_log.md` §4.14, `README.md` §5.4's prose and the shipped source comment all
  claimed the column form wins a row where the probe of the day had it losing by several times that row's
  spread — while README §5.4's generated table printed the gap, the spread and the word "beyond" two lines
  above the prose that denied it.
  Round 9's required next step was to correct the signs and then either take the row rule or re-derive the
  rejection. It was taken: `("down", 32) -> row` is a one-line rule, and `down` reaches 32 cores only in
  prefill, so the key that made round 5's "not monotonic in either" objection right has exactly one shipped
  point under it. Then it was measured end to end, and **it loses** — close to a millisecond of prefill on both
  layer kinds, every timed build, the op-level gap reversing rather than shrinking
  ([`logs/ab_sdpa_decode_grid.txt`](logs/ab_sdpa_decode_grid.txt)). Reverted, with §4.14 rewritten around the
  layer measurement. The rejection is now on the ground that the candidate is slower, which is a fact, rather
  than on arithmetic about whether a gain is worth a special case, which was a judgement — and which was
  computed from an inverted sign twice.

* **P2 — the decode SDPA grid was the swept loser, with nothing recording it.** True, and the fix taught the
  stage something it had had backwards for eight rounds. `8x4` leads the shipped `8x8` by 0.8–1.0 µs at spreads
  of 0.1–0.3 and identical PCC, and it costs none of the invariant the `q0/k0` candidate was rejected for, so
  it was taken. The layer A/B called it a dead heat. **The suite then failed it**: flash-decode assigns one core
  per batch row (`TT_FATAL(num_cores_available >= B)`, `sdpa_decode_program_factory.cpp:191`), so the 32-core
  grid caps decode at batch 32 and the supported batch-40 and batch-56 cases die inside the op. The grid is not
  a latency knob — it is the largest decode batch the layer can serve, 8x8's 64 cores cover the 56 the suite
  exercises, and `11x10` clears the bound but is slower, so 8x8 is the fastest *legal* grid. Reverted, with
  §3.8 rewritten around the constraint instead of the microsecond.

  Two things are worth keeping from that. The reviewer's finding was correct and its suggested remedy was
  wrong, which is only visible because the remedy was implemented and run rather than argued about — the same
  shape as the `down` orientation above. And nothing had ever asserted *any* field of this config, which is how
  an op-level sweep could argue against a capability bound for two rounds without contradiction; a new test now
  pins the servable-batch relation (`grid cores >= LARGEST_SUPPORTED_DECODE_BATCH`, not the literal `8x8`, so
  re-deriving the "free win" from the probe fails the suite) together with `k_chunk_size == page_block_size`,
  the field whose violation is silent. Round 9 also found §3.8 still quoting "17 rows, four chunk pairs" after
  round 8's own addition made it 22 and five.

* **P2 — the checkpoint has 40 layers, not 48**, and the wrong count was the denominator of the compounding
  argument that rejects the BFP4 projection candidate, which spelled the count out in words
  and bet the decision on it. Five places: README twice plus the prose inside a *generated* block, `work_log.md`, and the
  shipped source comment. The audit was structurally blind to it — two-digit integers are exempt wholesale, so
  a wrong `48` is unsourceable-but-allowed, and a numeral written in words is not a number at all. Fixing five
  strings is not the fix; the class is closed instead:
  [`logs/model_facts.py`](logs/model_facts.py) writes the checkpoint's own shape constants as a labelled
  artifact, `make_readme.py` reads the layer count from it rather than typing it, and
  `audit_figures.check_model_facts` asserts every document's layer/expert/head claims against it, spelled-out
  numerals included. Verified by injection: each of the five original phrasings fails the audit now.

* **P2 — `check_freshness` still could not fail.** Round 5 made it return early when the source manifest
  matched; round 8 made it *run* but downgrade a stale artifact to an advisory note when the manifest
  matched — and the manifest matches in every tree that passes, so the gate's power was still exactly zero,
  and the die-part-way scenario its own docstring names is a manifest-matching one. A stale artifact is a
  problem now, with no escape: the manifest says the sources were not edited since they were hashed, which is
  a different claim from "this artifact came out of the current sweep". The cost — touching a source without
  changing its bytes fails until the sweep is re-run — is the correct way round, and it fired on this very
  round's source edits before the sweep re-ran.

* **P2 — `check_mirrored_constants` could not report a drift.** It re-bound `problems = []` *after* the
  `make_readme.py` comparisons, so every append in them raised `UnboundLocalError` — round 8 asked for that
  function to really read the generator, and it read it and then crashed instead of reporting. Fixed, verified
  by injecting a wrong mirror (`MIRRORED-RULE-DRIFT`, not a traceback), and the orientation mirror the
  docstring claimed is now actually checked: the generator's `shipped_grid` and the implementation must agree
  on which axis fills first.

Round 9's smaller concerns closed in the same pass: `ab_decode_harness.py` counted its discard per *layer*
rather than per arm, so the shipped arm reported one repeat fewer than the candidate while three files claimed
an equal count, and the arms ran in blocks rather than alternating — the same multi-build drift that file
exists to characterise; both search tables decided inside-vs-beyond-the-spread in binary floats, so
`388.2 - 387.4` printed as "beyond the ±0.8 µs spread", and the comparison is made at the artifact's own
printed precision now; the dense table named a faster alternative's *time* without its geometry, which is
half of what "the alternative is named" should mean; and `check_freshness_exemptions` had an empty second loop
enforcing nothing, so only one direction of its stated biconditional was checked.

Two things round 9 raised that are recorded rather than changed. The `q0/k0` SDPA candidate is correct and
reproducibly 1–2 µs faster at the layer, and is rejected on the invariant-verifiability trade in §4.5 — so the
goal's "beat the best correct candidate" is met on every axis except that one, by ~0.1 %, deliberately and in
writing. And `logs/commit_record.txt` necessarily records the SHA of the commit *before* the one that records
it; the file says so.

**Round 10** returned `more-work-needed` with four items, all of them documentation or gate integrity: no
shipped configuration, PCC, capability or headline number changed. It also confirmed round 9's two reversals
from the artifacts, and independently re-derived the suite, watcher, dtype/fidelity, footprint and headline
claims, and both geometry rules against every config the suite log records.

* **P2 — the generated sparse table's shipped cell named a geometry the layer does not build.** The lookup
  matched the probe on (cores, `in0_block_w`, `per_core_N`) and took `min()`, but the probe sweeps `out_block_w`
  and `sub_w` as independent axes while the layer derives both from `per_core_N`. At the 32-active `down` point
  three rows share the key and `min()` took the fastest of them, so the table reported the `sub_w=2` row's time
  where the layer runs `sub_w=8` ([`logs/probe_sparse_matmul.txt`](logs/probe_sparse_matmul.txt) carries all
  three; README §5.4's shipped column now names the one the layer builds, and the gap it prints changed
  accordingly). **Fifth consecutive round that this one lookup was under-constrained by exactly one axis** — and the
  argument that a generated table cannot be wrong only holds if its key is the whole rule. Every swept axis is
  pinned now, and `audit_figures.check_sparse_block_rule` asserts the rule against the layer's own record: the
  suite log prints every sparse program config the layer built, so `out_block_w == per_core_N` and
  `out_subblock_w == largest divisor of per_core_N at most 8` are checked on measured behaviour rather than on
  the implementation's text. Verified by injecting a drifted triple into the log.
* **P2 — `census.py --check` could not fail.** Round 9 added the flag because the script accepted and ignored
  it; round 10 found the fix comparing the file against itself, because the write happened one line above the
  comparison. Replacing `census_summary.txt` with the word CORRUPTED and running `--check` reported success and
  silently rewrote the file. **Fourth "gate that cannot fail" in this stage** — `check_freshness` twice, the
  ignored flag, and now this — and the pattern in all four is identical: the check ran after the thing it
  checked had already been overwritten or excused. It compares before writing now, and fails on corruption.
* **P2 — two README claims contradicted by their own cited artifact.** §5.1 said the harness's repeats "agree to
  the last digit", which was the stated basis for treating a one-microsecond layer gap as real; three fresh
  builds of the same arm actually differ by more than ten microseconds. And §9 credited the SDPA `q0/k0`
  candidate with being "reproducibly a microsecond or two faster at the layer" when the current artifact has it
  level or behind on both trace-region settings. Both corrected, and §4.5's rejection is *stronger* for it: the
  candidate is faster only as an isolated op, so there is nothing to trade away at the layer. The honest summary
  of that sub-thread is that my original "inside the spread" reading was right, round 8 talked me out of it on
  one run of a harness whose own repeats disagree by more than the effect, and round 10 put it back.
* **P2 — five wrong figures in the operation-topology audit, and one in §6.** The §2 tables are the
  pre-optimization read the whole stage was planned from, and three rows carried this stage's *post*-change
  values (understating its own largest dense win by ~42 µs), one took a row from the other layer kind, and §6's
  `repeat_interleave` cost was roughly doubled by summing op codes across a step instead of the consecutive ops
  of that one call. All six were two-digit integers, which `ALLOWED_INT` exempts from sourcing wholesale — the
  live consequence of the hole README §1 discloses. Transcribing corrections would leave the class open, so
  both §2 tables and §6's two composite chains are **generated blocks** now: the topology tables from the fused
  stage's committed capture (the baseline, by definition), the chains summed positionally from this stage's.
  `make_readme.py --check` covers `work_log.md` as well as `README.md` from this round on.

Two of round 10's concerns are recorded rather than changed, one because it is right and one because it is not.

The contract's batch note claimed "any batch up to the 110-core grid is legal", which is true of the
`paged_update_cache` shard it was describing and false of the path as a whole: the same flash-decode bound
§3.8 documents caps `full_attention` decode at 64 page-table rows on the shipped 8x8 grid. The bound is
inherited unchanged from the functional and fused decoders and sits above the 56 the suite exercises, so no
capability is reduced here, but the note now states it.

The suggestion to add a decode batch in 5..8 was declined here on the grounds that `per_core_M` is
`ceil(batch / 32)`, making batches 1-32 one config class. **That was wrong, and review round 11 caught it.**
`ceil(batch / 32)` is true only of the three MoE roles, whose activation the block reshapes to
`[1, 1, padded_tokens, dim]`; the four token-mixer roles (`attn_in`, `o_proj`, `gdn_in`, `gdn_out`) are called
on `[batch, 1, dim]`, so `_physical_rows` returns `32 * batch` and **`per_core_M == batch`**. Simulating
`_ProjectionConfigs.get` properly gives **eight** distinct tuned signatures over batches 1-8 — batch 5 is where
`o_proj`'s `in0_block_w` drops from 16 to 8, because `DECODE_MATMUL_IN0_TILE_BUDGET // 5` is 12 — and batch **9**
upward builds no tuned mixer config at all, since `m_tiles > DECODE_MATMUL_MAX_M_TILES`. The three MoE roles are
the other convention and stay tuned at every supported batch. So the suite's 13/32/40/56 cases were never
covering a second class of tuned mixer config; they were covering the fallback. (This paragraph said "13 upward"
until review round 12 drove the real selector and found the boundary one batch above the band, not five.)

Two things follow, and both are done rather than argued. The coverage is real, so batches **5 and 8** are added
to `test_batched_prefill_decode_pcc` (PCC at both, both layer kinds) and
`test_decode_runs_the_tuned_program_configs` is parametrised over batches 1 and 5 — which immediately failed,
because its routed-`in0_block_w` assertion had been written against batch-1 literals, so a batch it had never
run at was a batch it could not have checked. It is keyed on the rule now. And batch 5 does build different
geometry: `grid=2-8 in0_block_w=64 per_core_N=2` for the routed gate/up against batch 1's
`grid=1-8 in0_block_w=32 per_core_N=4`.

The general lesson is the one this round is mostly about: I declined coverage on an analysis I had run but not
validated against the code path, and the wrong arithmetic was in a *simulation I wrote to check the claim*.
Simulating the selector by calling the inner helper directly skipped the two caps the real selector applies and
the row convention the caller uses. A simulation that does not go through the shipped entry point is a
hypothesis about the shipped entry point.

**Round 11** returned `more-work-needed` with four items. It verified all four round-10 fixes, could not defeat
any gate across eighteen injections, and re-derived the suite, watcher, dtype/fidelity, footprint, headline and
both geometry rules independently — and then found the first **shipped-coverage** finding in three rounds, in
the disposition round 10 had written for itself.

* **P2 — the round-10 refusal to add a decode batch in 5..8 rested on arithmetic the code contradicts.**
  Corrected in §6 above: `per_core_M == batch` for the four token-mixer roles, so batches 1-8 are **eight**
  distinct tuned config classes and 13 upward build none. Batches 5 and 8 are covered now, and parametrising the
  config test immediately exposed that its routed-`in0_block_w` assertion was written against batch-1 literals.
  Both fixed. The failure worth naming is not the missing coverage but the *simulation*: I checked the claim by
  calling the inner config helper directly, which skipped the caps the real selector applies and the activation
  convention its caller uses, and then trusted the result enough to decline work with it.
* **P2 — "8 cores is marginally best on both layer kinds" is the artifact read backwards.**
  [`logs/ab_norm_shard_cores.txt`](logs/ab_norm_shard_cores.txt) puts every arm inside the layer harness's own
  run-to-run band, and in the run round 11 read, 8 was the *slowest* of the four on `linear_attention` — while in
  the run committed here it is nominally the fastest, which is the same fact stated twice. The claim
  sat in three places, and §3.6 gave a second, differently-worded reason for the same constant. All four now say
  the one thing the evidence supports: the knob does not matter at the layer, and 8 ships because it is the
  shard count the rest of the stage's norm evidence was measured at. Third round on this one constant, and the
  second time a *ranking* was quoted from an artifact that does not rank anything.
* **P2 — README §8 misreported the stack-watermark coverage.** The generated sentence rendered the census's
  count of *detail lines* (five, one per RISC of the one core that reported) as "5 dump(s)", when the log holds
  one summary in one dump of sixty. `census.py` now emits a labelled `dumps:` count and separates the per-dump
  timestamp lines from the banners, so the generator can say "in 1 of its 60 dumps, across 5 RISC processors",
  and the tightest 1332-byte figure is stated as the single sample it is. Two bucket labels that oversold what
  they counted were renamed at the same time. The audit could not have caught this: `5` is a single digit.
* **P2 — README §5.3 printed an all-codes total over a truncated body.** The table lists the top 16 op codes by
  the larger of their two per-step costs and then a total summed over all of them, hiding tens of microseconds of
  each window — nearly a tenth of `full_attention`'s — including that kind's second and third
  largest attention-side ops, because the ranking is shared between the columns while the two kinds' op sets are
  not. §7's parallel table had always disclosed its remainder; §5.3 does now, per column, and the columns
  reconcile to the total.

Round 11's other concerns, all acted on: the sentence claiming §5.4's remaining gaps are "orientation, not the
inner block" is qualified, because at 32 active / `down` about half the gap is `out_subblock_w` — the layer's
largest-legal-subblock rule is measurably not optimal there, at an untuned batch; README §9 item 5 now lists
**four** batch thresholds, the fourth being that the tuned 2D prefill configs apply at batch 1 only; §8's two
artifact links pointed at uncompressed names that do not exist; `run_evidence.sh`'s committed-tree block now
runs both summary generators with `--check` instead of regenerating them there; and the contract's hand-modelled
footprint error is stated qualitatively rather than as a figure whose only support was an audit allowlist entry
for a superseded value.

One of those concerns was a genuine unswept knob, and it is swept now.
**`SDPAProgramConfig.max_cores_per_head_batch`** defaults to 16, and flash-decode activates
`max_cores_per_head_batch * batch * kv_heads` cores — 32 of the shipped grid's 64 at batch 1. So that field, not
the grid, is what sets SDPA's parallelism here, and it is the mechanical reason the 8x4 and 8x8 arms tie: both
activate the same 32 cores. The stage had called this config swept with one of its four fields defaulted.
Swept now, as generated rows of README §5.5's knob table: raising it to 32 or 64 changes nothing beyond the
spread, and halving it to 8 costs tens of microseconds. The default is right, the ~1 % of decode round 11
bounded as possible upside is not there, and one more "why" in §3.8 is now measured rather than asserted.

Adding those arms also caught a latent generator defect of the class rounds 5-10 kept finding: `best()` matches a
*subset* of a row's fields, so the new arms — which carry an extra labelled field and are faster — satisfied
every key the "shipped arm" lookup used, and would have quietly become the shipped figure in three README rows.
`best()` takes an `absent=` list now, and the plain SDPA arms declare what they must not carry.

**Round 12** returned `more-work-needed` with five items, four of them inside round 11's own fixes — which is
the seventh consecutive round where the previous round's fix contained the next round's defect, and worth
stating plainly rather than treating as bad luck: every one of those fixes replaced a *transcribed* figure with
a *derived* one, and the derivations were what needed reviewing.

* **P1 — README §5.4 named the batch-5 `o_proj` geometry as shipped.** Round 11 parametrised the config test
  over batches 1 and 5, which added a second `decode dense matmuls` line to the suite log; `shipped_configs`
  took the **last** line, so the table described batch 5's `in0_block_w` 8 and then reported the batch-1 winner
  as a gap the tuned target does not have. The row reads "the measured winner" again. Both suite-log lookups
  are pinned to `batch=1` explicitly now — the test logs the batch — and the assignment is `setdefault`, so
  neither order nor a log written before the pin can decide it. Note what made this findable: round 11's own
  asymmetry, where the sparse lookup `break`s on the first line and the dense one kept the last.
* **P2 — "13 upward build none" was wrong in both directions.** Driving the real selector: the four token-mixer
  roles stop at **batch 9**, and the three MoE roles never stop, because their activation is reshaped to one
  tile of rows. So batches 9-12 were described as tuned when the mixer roles are not, and 13+ as untuned when
  the MoE roles still are. Corrected in README §9 item 5 and above. The class is closed rather than the
  instance: `test_documented_batch_thresholds` now asserts the boundary against the built layer, keyed on
  `DECODE_MATMUL_MAX_M_TILES` rather than on a literal, for both layer kinds. Every threshold in that item is a
  one- or two-digit integer, which the figure audit exempts from sourcing wholesale — this was the one class of
  claim in the stage with nothing checking it, and it had produced a live wrong claim.
* **P2 — work_log §2's generated audit tables had a rank order contradicting their own times**, because the
  `Rank` column came from the hand-written list order while the times came from the capture, and two rows were
  labelled as single *calls* while summing op codes across the step. Rows are ordered by the generated time
  now; the two multi-code rows say "all launches" and README §6's generated block carries the per-call figures;
  and the three op codes above the table's smallest row that were silently missing are rows, with a disclosed
  remainder line for everything below — the same shape §5.3 and §7 already had.
* **P2 — README §8's watcher sentence rendered the stack detail-line count as a processor count** and asserted
  one reporting core where the log has two. Round 11 had rendered the same count as a *dump* count; the lesson
  is that one number was being reused for three different questions. `census.py` emits `stack summaries`,
  `stack processors per summary` and `stack reporting cores` as counted facts now, and the sentence reads them.
* **P2 — §9 item 8 claimed no measured candidate is faster at the layer**, which the committed harness artifact
  contradicts in the candidate's favour on every paired build of this sweep. Restated as "not *reliably*
  faster": the arms have now swapped order between sweeps in both directions, inside each arm's own span. The
  rejection is unchanged and rests on the `k_chunk == page_block_size` invariant.

Also corrected: the claim, in three places including the shipped source comment, that 64 active experts is
"the largest supported decode batch". It is the largest *tuned* one — an untuned decode batch of 32 or more
saturates the active bound at 256 and does reach 32 `down` cores. The orientation rule was rejected on a
whole-layer A/B, so nothing rides on it, but it was used to argue the rule had no competing point.

**Round 13** returned `more-work-needed` with six items and no model-correctness defect. Two were inside round
12's own fix, and the rest were claims that had been true once, or true of one table, and were never re-derived.

* **P2 — §2's `linear_attention` remainder row was arithmetically impossible**: "the other 20 op codes, each
  under 23.6 µs/step" summing more than a millisecond. The generator computed the floor as the smallest listed row and asserted
  everything unlisted was below it — true of the `full_attention` table, which is a top-N, and false of the
  mixer-only table, whose remainder is dominated by the shared MoE rows the table above ranks. The claim is now
  made only when it holds and names the shared rows when it does not.
* **P2 — round 12's "all launches" relabelling landed on the wrong row.** It went on the single-code
  `UntilizeWithUnpadding` row; the two *multi-code* rows — the ones the fix was for — were untouched, so §2 still
  sized `repeat_interleave`'s head expansion at the step-wide total of three shared op codes rather than the
  call, and §6 recorded the defect as closed. Both rows now say what they sum and carry the per-call figure,
  computed from the capture rather than typed: the first draft of that label hardcoded the derived sum and the
  figure audit refused it, which is the check working.
* **P2 — the context contract said "prefill keeps ttnn's 2D heuristic".** The stage ships explicit 2D configs on
  every dense prefill role, beating the heuristic on all six, asserted by a test. The contract is the file the
  next stage reads for the shipped configuration, and `audit_figures.py` checks its figures, not its prose.
* **P2 — the layout-conversion delta was wrong in the work log and in the test file** ("8-12 new conversions"):
  the budgets go 1 → 5 and 6 → 14, so the stage adds **4 and 8**, which README §6 and §3.6 both had right. Two
  two-digit integers, which `ALLOWED_INT` exempts wholesale.
* **P2 — README §5.5's SDPA lookups matched a superset.** `q_chunk` was unconstrained, so the `q_chunk=0` arms —
  the candidate §9 item 8 rejects — satisfied the "shipped" and "8x4" lookups and were one re-measurement away
  from becoming the shipped figure in three cells. `min()` happened to land on the right row. Pinned, and this is
  the seventh time this stage has found an under-constrained lookup: the lesson is that `best(**where)` matching a
  *subset* is the wrong default for a table about one specific configuration.
* **P2 — §3.8 quoted an 8x4/8x8 pair matching neither section of its probe**, pairing one section's 8x4 time with
  the other's 8x8 and understating the spread the gap must clear (one of those rows carries a 0.9 µs spread
  against a 0.8 µs gap). Fourth round on this paragraph, so it no longer transcribes the pair at all — README
  §5.5's generated knob table prints it.

Round 13's other concerns are closed too: the two op-code counts in one document (25/28 bare in §5.3 against
31/33 shape-qualified in §7) now say why they differ; §7's roofline sentence quoted the `full_attention` fraction
unlabelled; a cross-reference pointed at §9 item 8 instead of item 9; and the same sentence claimed
`CLASSIFICATION.md` classifies `TT_FATAL` lines in the watcher log, where that log has none — they are in the
pytest console logs only. The k128 candidate's PCC range was quoting a row belonging to the *other* candidate,
and is now the right range at a precision the audit can check.

Two gates got stronger, both from round 13's hard-check list. `check_generators` was invoking the two summary
generators in **write** mode, so the audit repaired the drift it reported and only a first run on a fresh clone
could see it; both are `--check` now, with the byte snapshot kept as proof that read-only wrote nothing. And
`HISTORICAL` — the allowance that lets a superseded figure be quoted where a document records that it was wrong —
was file-level while its own rationale was section-level, so a superseded value could be used as a live claim
anywhere in the work log. It is scoped to §3's development ladder and §6's review rounds now. That fix needed
two attempts: the first computed each paragraph's section from a string the section headings had already been
stripped out of, so it silently never matched, which is the same class as the gates rounds 5, 8, 10 and 12
found — a check that cannot fire — one layer up. The strippers run per paragraph now, and both directions of the
boundary are verified.

A test also got stronger: `test_decode_runs_the_tuned_program_configs` claimed in a comment to assert the
batch-dependent dense geometry and asserted only `in0_block_w >= 2`, which any batch passes. It now asserts
**both** activation conventions — `per_core_M == batch` for the four token-mixer roles, one tile for the three
MoE roles, and the in0 budget divided by the tile rows — which is exactly the distinction rounds 10 to 13 kept
getting wrong in prose.

**Round 14** returned `more-work-needed` with three items and was the first round in five to find *shipped-code*
work rather than documentation defects — by reading the half of `tt-perf-report`'s output the stage had been
suppressing. All three are in §4.19: `--active-experts` was never passed, so advice on the two routed
`sparse_matmul` rows (31 % of decode, ~82 % of prefill) was dropped by the tool's own early return, and one item on
the largest op of the prefill window had never been read; the prefill SDPA config was unswept; and three
`Typecast` launches looked foldable. Outcome: the routed `in0` moved to L1 (~0.6 ms of prefill on both kinds), two
folds shipped, one fold rejected as illegal inside a trace region, and — chasing the SDPA chunk's legality —
`POLICIES["fused-parity"]` turned out to be unable to prefill *at all*, on HEAD as well, because
`_prefill_2d_matmul_config`'s L1 model hardcoded BFP8's bytes per weight element. No test had ever run that policy.

**Round 15** returned `more-work-needed` with three items, and its P1 was the sharpest finding of the stage:
round 14's prefill-SDPA change **was never in the binary**. The table was keyed on a tuple whose second element no
policy matched, so every lookup missed and every policy took the conservative fallback while three documents
claimed the winner shipped. The negative control had been sitting in the artifacts — `full_attention` prefill
improved *less* than `linear_attention` across round 14, though only `full_attention` has an SDPA. Round 15 also
refused round 14's rejection of the router's zeros target (a first API error is not a rejection) and was right: the
target is hoisted out of the trace now. Both are in §4.19. The lasting change is a habit rather than a constant —
where a lookup selects a shipped configuration, the *resolved* value is asserted per policy, because a table whose
fallback is legal fails nothing.

**Round 16** returned `more-work-needed` with three items, all narrow: the wide-cache clamp keyed on the policy
field rather than on the attached cache, so `allocate_kv_cache(dtype=...)` — a documented, supported route — could
still resolve an illegal chunk; the `PCC_BAR` comment quoted the *fused* stage's worst-case PCC as this stage's and
called this "the fusing stage"; and §4.19 above still described round 15's dead tuple key as the shipped design
while this section had no entries for the two rounds that found shipped-code work. All three are fixed, and the
first now has the test it needed: a decoder built under the shipped policy but handed a bfloat16 cache must resolve
a legal chunk *and* prefill a full chunk with it.

**Round 17** returned `more-work-needed` with one item and two concerns. `DECODE_MATMUL_GEOMETRY` shipped
`shared_in` at a target of 80 — an 88-core grid for a projection whose `Nt` is 33 tiles, so 55 of those cores
never receive an output tile — and round 17 read the probe as showing the 33-core grid several microseconds
faster, beyond that row's spread, while the table's docstring claimed every entry was the measured winner.

Changed to 32, and then **round 18 found the justification wrong**, which is worth recording as carefully as the
change itself. Re-derived from the committed artifact, the `shared_in` ladder at the shipped `in0_block_w` is flat
within noise across every target from 24 to 110 — and the slower band round 17 cited belongs to the
`in0_block_w=8` rows, which are flat across core count as well. There was no core-count split to be on the wrong
side of. The layer A/B had already said as much: four alternating timed runs moved traced decode by nothing
measurable on either kind.

So the entry stays at 32 for a structural reason rather than a measured one — it names exactly `Nt` cores instead
of 88 — and both the code comment and README §5.4 now say that. Two rounds spent on a constant that does not move
the layer is not wasted if the outcome is that the file no longer contains a measurement claim its own artifact
refutes; that claim is the thing this stage exists to prevent.

The concern worth acting on immediately was mine, from round 15. `_router_zeros_for` kept **one** persistent
scatter target and replaced it when the shape changed. The routing-logits shape is stable for every batch up to 32
and changes at the supported 40 and 56, so a decode trace captured at one of those shapes and replayed after the
other would have scattered into a freed buffer — silently wrong routing weights, not an error, and no test would
have caught it because every trace site in the suite uses one batch. It is keyed by shape and never freed now.
That is a hazard this stage introduced two rounds ago while removing two ops, found by reading rather than by
failing, which is the argument for the review rounds continuing past the point where the gates are green.

**Round 18** returned `more-work-needed` with three items and said plainly that closing them would pass the
stage. All three came from round 17's own change, and the first is the one that matters: the justification
round 17 wrote for retargeting `shared_in` was **refuted by the artifact it cited**. Re-derived from the committed
probe, that role's core ladder is flat within noise at the shipped `in0_block_w`, and the slower band round 17 read
as a core-count split belongs to a different `in0_block_w` arm entirely. §4.15 above records the correction; the
entry stays at 32 on the structural ground that it names exactly `Nt` cores, and the code comment says so instead
of claiming a win.

Two smaller ones with the same shape: README quoted the pre-round-17 core count two lines below a generated table
that printed the new one, and §5.4 claimed `test_decode_runs_the_tuned_program_configs` asserted the core count,
`per_core_N` and output subblock when it asserted neither the grid nor `per_core_N`. The second is now true rather
than narrowed: the test derives each role's target from `DECODE_MATMUL_GEOMETRY`, reproduces the layer's own
axis-filling rule, and asserts the realised grid and that `per_core_N` covers every output tile. Writing it caught
a defect in itself immediately — `o_proj` and `gdn_out` are both 4096x2048 with *different* targets, so a
shape-keyed map is wrong and the key needs the layer kind — which is a fair advertisement for asserting a rule
rather than a literal.

Also corrected: two magnitude words that had drifted from their artifacts ("tens of microseconds" for a ~7 µs
whole-layer A/B, and "a couple of percent" for a figure that is a fraction of one), and the second of the two
`in1_bytes` fallbacks, which round 17 fixed in one lookup and left at bfloat16's 2.0 in the other — the direction
that under-models and lets program construction throw.

**Round 19** returned `more-work-needed` with three items, all documentation fidelity, and all of one kind:
a magnitude word that was true of an older artifact and never re-derived after the artifact changed.

The prefill-SDPA ratio is the clearest. "Nearly three times slower" was *correct* when round 14 wrote it — the
probe then measured bfloat16 K/V with no compute-kernel config. Round 15 rewrote that probe to the shipped
contract, the 64-arm's time dropped by a third, and the ratio has been about two ever since, through four rounds
including round 18's explicit sweep for exactly this kind of drift. It is stated as measured now, and the
implementation's docstring carries no ratio at all — a ratio moves on every re-run, and README §1 already says
this file quotes no run-varying absolute.

That invariant is the second item, and it is the sharper one: round 18's own replacement text for the `shared_in`
comment reintroduced two absolute microsecond bands into the implementation, one round after the round convened
to remove exactly that. Both are gone; the entry's argument is structural and needs no figures. A stated
invariant is only worth what the next edit respects, and this stage has now broken this one twice and repaired
it twice.

Third, README §5.4 claimed the shipped row's output block/subblock were asserted. Round 18 closed two thirds of
that sentence (the grid and `per_core_N`) and left the last third overclaiming. Rather than narrow the sentence,
the assertion now exists: the output subblock follows deterministically from `per_core_N`, the M tiles and the
dest-register budget, so the test derives it and asserts it, and the sentence is true as written.

Also corrected: `SPARSE_CORES_PER_ACTIVE`'s docstring said the 8-core form is ~4x slower at the prefill group
where the artifact has always said about twice; and the `shared_in` entry's layer A/B, which was run in the loop
but never committed as an artifact, no longer reads as though a committed measurement backs it.

**Round 20** returned `more-work-needed` with a single item: README §9 item 7 still carried the pre-round-15
prefill-SDPA ratio, one round after round 19 recorded that sentence as fixed. It was not a new drift — it was
round 19's edit never landing. The script that made it batched two replacements and asserted on the second; the
assertion failed, `write_text` was never reached, and the message before it had already claimed success. That is
the same multi-edit-with-assert failure this stage hit early on, and the discipline it taught (apply one edit,
then re-read the file to confirm) is the discipline that was not followed. The sentence now states the measured
ratio, names which arms are like-for-like, and says why the `bf16-kv` rows are not.

Two smaller corrections in the same pass, both claims the artifact contradicts. The orientation comment said the
row form leads `gate_up` at 64 active, "the opposite sign" from the prefill point — at the `in0_block_w` the layer
actually builds there, the column leads, the *same* sign; the row only leads at inner blocks the wide phase never
selects, so it was never a shipped comparison. And the `down` gap at the prefill group was restated as about a
percent — which round 21 then found was itself wrong, the ladder printing half of one. That correction is the
origin of the rule the next two rounds generalised: this comment states directions, not magnitudes.

**Round 21** returned `clean-pass` — the first pass verdict of the stage — with no required work, recording four
errata below its finding bar. Two were closed anyway, both cases where a document contradicted the stage's own
generated artifact: the round-20 `down` restatement above, and README §8's generated watcher sentence, which
composed two census counts into a total the log does not have ("five RISC processors on each of two cores"
implies thirty samples where there are fifteen; the cores are distinct *across* the summaries, not a second
axis). Rounds 11 and 12 had each rewritten that sentence and each left a composition error, so the generator now
asserts the product rather than trusting the phrasing.

The larger outcome was a rule, not a fix. `check_orientation_claims` was added to verify the orientation
comment's magnitudes against the generated ladder — and two consecutive sweeps then falsified three more bullets
with no shipped code changing. A magnitude in a comment has no generator, so it is stale as soon as the sweep is
re-run. The bullets now state directions, which are stable, and cite the ladder for figures; the check refuses a
magnitude in a tagged bullet outright. §4.14 records the row that taught it.

**Round 22** returned `more-work-needed` with two items, both documentation contradicted by committed artifacts.
README §7 named the float32 gate-promotion `Typecast` launches the largest `linear_attention` op-to-op line
item; the generated table directly beneath it ranks `TilizeWithValPadding` first and `Typecast` fourth. The
claim was true of a capture superseded several times over, and it pointed a reader at a small share of the
dispatch gap instead of the much larger one above it. It survived every round to that point because the sentence
sat *inside* a generated block while being hard-coded, so `make_readme.py --check` regenerated it from itself
and could never see the drift. The sentence is computed from the same `perf_summary.json` rows the table is
built from now, the hand-written bullet is reordered to match, and `check_top_line_item` matches any such claim
against the capture's actual top row — it found a third stale instance in §4.19 on its first run.

The second item was the 64-active `gate_up` row described as sub-microsecond noise while the ladder called it
decisive; §4.14 covers it. Round 22 also noted that the round-21 commit had swept in a shared skill file outside
this stage's declared scope; it was already modified when the stage began, so it was never this stage's to
commit, and it was reverted to the uncommitted state the stage inherited.

**Round 23** returned `more-work-needed` with two bookkeeping items, both in this stage's own review and
checkpoint records rather than in its measurements. This §6 claimed every round was recorded below while
stopping at round 20, and `logs/commit_record.txt` carried a `head` five commits stale alongside a sentence
claiming the audit checked it, which nothing did. Both are corrected above and in that file, and
`check_commit_record` now verifies the recorded list and `head` against `git log` instead of asserting that
something else does.

**Round 24** returned `more-work-needed` with three required items and a set of the same class beside them,
and it is the round that showed the "we no longer hand-maintain figures" claim was not yet true. The three:
README §5.4 said the `router` row's `out_subblock` alternative was measured and slower when the probe's core
ladder never expresses it — the same conflation round 12 corrected in §5.5 and only there; §3.8 described a
pre-round-11 SDPA ladder, with the wrong row count and with `max_cores_per_head_batch` missing from its
findings, 13 rounds after round 11 required that field be swept; and §6 itself, rewritten one round earlier,
had attached round 20's `check_source_magnitude_words` paragraph to the round 23 entry and twice called a
round-6 result the first performance finding of the stage when §3.9 credits round 1 with two.

Beside them, six more statements this document contradicts: a probe list missing `probe_prefill_sdpa.py`, a
prefill role count of five where six are swept, a shared-expert row calling a within-spread result "no gain",
an A/B claim scoped to all builds where only the prefill builds compare, two dangling `§4.20` references, and
an `in0_block_w` enumeration in the shipped comment that had drifted. None changed a decision; all were
wrong. They are fixed together rather than one round at a time, and the lesson is recorded in the paragraph
below rather than as a claim that the class is now shut.

**Round 25** returned `more-work-needed` with one required item, and it is the most valuable finding this stage
received. §4.21 has it in full: the sharded-residual family, which `$optimize` OPT-003 makes mandatory, had been
closed by an assertion that `mcast_in0` requires an interleaved `in0` — false against the op's own validation,
and inferred from a probe that only ever passed interleaved activations. Taking it removed a conversion per
decode step on both layer kinds and carried the `linear_attention` step across 2x for the first time.

**Round 26** returned `more-work-needed` with three items, two of them the same class round 25 exposed and both
downstream of round 25's own change. §4.22 has them: the decode output projection's `in0` was still raising a
`tt-perf-report` item that §5.5 claimed decode raised nowhere, and the two roles round 25 moved onto a sharded
`in0` were still being ranked — and tuned — in the interleaved probe family. The third was the layout-churn
comment's itemisation, which round 25 left summing to one more conversion than the budget it justifies.

Round 26 also asked for a narrower statement of the flash-decode Q constraint, on the grounds that "cannot move"
had just cost the stage a family: the op accepts a height-sharded Q in L1 and refuses only a *non-sharded* Q
outside DRAM, so the blocker is the interleaved form the intervening rotary and concat ops produce, not L1.

**Round 27** returned `more-work-needed` with three items and produced the largest of the three
op-contract wins. §4.23 has it: the decode V shard was thrown away and rebuilt every step to reach the
layout it already had, and three documents called those conversions op-contract requirements. Round 27 also
found README §6's churn table and two work-log sentences still carrying the pre-round-25 budgets — that
table is generated from the gate's own constants now — and the decode-search table ranking two roles
against shard-core counts the layer cannot build.

**Round 28** returned `more-work-needed` with four items, and the first is the sharpest comment this stage
has received: the *same* optimization round 27 took for V had been left on the table for K, and the shipped
tree already contained the counter-example, since V reached the same op unpadded and the whole suite
passed. §4.24 has it. Round 28 also found that `overlap_qk_coregrid=False` — which round 27's own comment,
work log and A/B docstring all presented as what put K on a disjoint range — is **inert**: the op's wrapper
forces that flag to `True` for a non-sharded input, and the layer's `qkv` is L1 interleaved. Disjointness
came entirely from the two explicit cache-write grids. The argument is gone and the claim corrected.

Its other two items were this §6 missing a round-27 entry, and §2's reshard bullet still asserting that
the baseline's conversions were "all required by an op contract" with "no avoidable ones existed to
remove" — the sentence rounds 27 and 28 had just spent two wins disproving. Round 28 also settled the
`rotary_embedding_hf` native-decode candidate that round 27 had left open: expressible in principle, but
the head-dim RMSNorm between the head split and the rope refuses a height-sharded input *and* output for
its sharded program config, so Q and K cannot arrive at the rope still sharded; and costing the
conversions the native spelling would need against the four transposes it would remove makes it a wash
inside the harness band. Recorded as closed on the op line rather than left open.

**Round 29** returned `more-work-needed` with three items, all documentation, and it is the round that
closed the op-contract class. It re-derived every remaining rejection rationale in this stage against the op
sources — the sharded-layernorm refusal, the rope width rule, the SDPA-decode core bound, `topk`'s
multi-core width, the DRAM-sharded worker/bank equality, the four `chunk_gated_delta_rule` claims, the
host-only conv weight preparation, the unified-MoE program count and PCC target — and found all of them
earned. No fifth instance of the class exists in this tree.

What it found instead was the *reporting* tail of rounds 25-28: corrections claimed in more places than they
were applied. §4.24 said the inert-flag claim had been corrected "in the source comment, in §4.23 and in the
A/B's docstring"; two of those three were. A comment round 28 wrote to replace a wrong one called the
two-launch cache-write branch unreachable, when `test_decode_batch_above_head_split_limit[56]` exists to
cover it. And `doc/context_contract.json` still described every decode norm as returning to DRAM — the
behaviour §4.21 removed four rounds earlier — in the one stage-owned file the goal contract names by name.

That tail is worth recording as its own lesson. Four rounds of real optimization findings were absorbed by
editing the code and the sections that discussed it, and each time a further copy of the old claim survived
somewhere the diff did not reach. The figure audit cannot see any of it: every one was prose about an API or
a cross-reference between two sections.

**Round 30** returned `more-work-needed` with three items, all of the cross-document class round 29 named,
and said plainly that closing them leaves nothing else. Its first is the stage's own recurring defect in its
last hiding place: README §5.5's `shared_in` cell was keyed to the 88-core grid **round 17 removed**, so §5.5
and §5.4 printed different shipped times for the same role (9.9 against 10.0), and the `router` lookup was
pinned to the widest `in0_block_w` rather than the shipped cap (9.0 against 9.1). The generator reads
`DECODE_MATMUL_GEOMETRY` now, and `check_generator_geometry_literals` refuses a return to a literal.

Its second item taught something about what a test can and cannot do. README §5.4 claimed
`test_decode_runs_the_tuned_program_configs` asserts the shipped `in0_block_w`; the dense loop only bounded
it, so a role's cap could revert to a value round 26 measured against and every assertion would pass. The
loop now pins the *rule* exactly. But the obvious next step — pinning the value — does not work, and this was
checked rather than assumed: the expectation is derived from the same table the cap lives in, so reverting
`attn_in`'s cap to 8 moves the expectation with it and the assertion still passes. A cap's value is a
measurement, not a rule, and what checks it is README §5.4's generated ranking on every regeneration. Both
the test comment and §5.4 now say that rather than claiming a gate that does not exist.

Its third was the test module's own docstring, which attributed four optimization-contract assertions to
`test_optimized_path_is_used` after they had been split out into three other tests — the bullet that answers
the goal contract's "tests exercise the optimized path, not a functional fallback".

**Across rounds 18-24, one class.** Nearly every finding in those rounds was prose restating a measured value
that later drifted, and each round's response tightened a gate rather than only fixing the sentence. Round 20
added `check_source_magnitude_words`: `audit_figures.py` matches numerals, so a ratio spelled in words is
invisible to it, and every one of rounds 18-20's findings was such a phrase, true of an older artifact. That
gate refuses a spelled-out performance ratio in the three source files — where README §1 already promises no
run-varying figure lives, because no sweep regenerates a comment — and it caught an instance on its first run,
a phrase written during round 19's own fix. Round 21 added `check_orientation_claims` and then, when verifying
magnitudes proved unwinnable, made it refuse them outright. Round 22 added `check_top_line_item` after a
hard-coded superlative inside a generated block went stale through many captures. Round 23 added
`check_commit_record`. Documents may still carry figures; they sit beside their artifacts and are regenerated
with them. What remains ungated, and is recorded here rather than claimed closed, is prose that states a count,
a ranking or a superlative in words — round 24 found six such statements wrong at once, which is why this
paragraph does not claim the class is finished.

Checkpoint: [`logs/commit_record.txt`](logs/commit_record.txt), which also records the exact command
that proves the committed tree reproduces every generator and passes the figure audit. Local commits
only; nothing is pushed.
