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

Op-level candidate sweeps use five standalone probes that build only the op under test at the
layer's real shapes, so a geometry sweep does not have to pay for a whole layer:
[`logs/probe_sparse_matmul.py`](logs/probe_sparse_matmul.py),
[`logs/probe_dense_matmul.py`](logs/probe_dense_matmul.py),
[`logs/probe_prefill_matmul.py`](logs/probe_prefill_matmul.py),
[`logs/probe_decode_micro.py`](logs/probe_decode_micro.py) and
[`logs/probe_projection_dtype.py`](logs/probe_projection_dtype.py). Their outputs are committed next
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
| 5 | `MatmulDeviceOperation 32 x 4096 x 2048` | 73.9 | `o_proj` — flagged `SLOW`, 23.2 % of DRAM bandwidth | explicit decode program config; DRAM-sharded | **explicit 1D config taken**, DRAM-sharded measured and rejected (§4.1) |
| 6 | `SliceDeviceOperation` | 109.3 | mostly the two slices that unpack the packed gate/up output | split the pair instead; L1 + BFP8 | **packed kept** (§4.2), L1+BFP8 taken |
| 7 | `MatmulDeviceOperation 32 x 2048 x 9216` | 95.6 | packed attention in-projection | explicit config; BFP8/BFP4 weights | **BFP8 + explicit config taken**, BFP4 measured (§4.6) |
| 8 | `TopKDeviceOperation` | 48.3 | router top-8 over 256 experts, single core | pad to the multi-core width; replace the gate op | **both measured and rejected** (§4.3, §4.4) |
| 9 | `LayerNormDeviceOperation` | 51.0 | four RMSNorms; the two residual ones run on **one core** | width-sharded L1 + `LayerNormShardedMultiCoreProgramConfig` | **taken** (§3.6) |
| 10 | `MatmulDeviceOperation 32 x 2048 x 256` | 25.5 | router projection, 8 cores, 9.4 % of DRAM bandwidth | explicit config | **taken** (§3.5) |
| 11 | `DeepseekMoEFastReduceNC` | 92.0 | expert-axis reduction over the 256-wide down output | L1 + BFP8 input | **taken** |
| 12 | `SdpaDecodeDeviceOperation` | 17.6 | paged flash-decode | reduced cache dtype; program config sweep | **BFP8 cache taken** (§3.7), config swept (§4.5) |
<!-- /generated:topology-audit-full -->

`linear_attention` shares the whole MoE and the norms with the table above; what differs is the
mixer, and its own top items (from the same capture set, per traced step) are:

<!-- generated:topology-audit-linear -->
| Op code | µs/step | What it is | Action |
| --- | --- | --- | --- |
| `MatmulDeviceOperation 32 x 2048 x 12352` | 128.5 | packed DeltaNet in-projection | BFP8 + explicit decode config (§3.5) |
| `MatmulDeviceOperation 32 x 4096 x 2048` | 80.9 | `out_proj` | explicit decode config (§3.5) + a bfloat16 activation (§4.11) |
| 3 x `MatmulDeviceOperation b={32} 32 x 128 x 128` | 44.2 | the float32 recurrent-state matmuls: decay read, delta outer product, output read | operands moved to L1 — 19 µs (§3.9 item 2); fidelity swept and rejected |
| `ReshapeViewDeviceOperation` + `PermuteDeviceOperation` | 44.2 | the one-shot head-major relayout of the conv output | inherited from the fused stage, which already reduced it from three round trips to one |
| `TilizeWithValPadding` + `Concat` + `UntilizeWithUnpadding` | 36.2 | `repeat_interleave`'s GQA head expansion | output moved to L1 (§3.9 item 5); the once-per-step op-to-op stall in front of it is the largest single gap in README §7's generated itemisation |
| `TernaryDeviceOperation` | 23.6 | the `addcmul` conv-tap accumulation | inherited; the fused stage measured `addcmul` against `mac` and kept it |
<!-- /generated:topology-audit-linear -->

Structural observations from the same read, which drove §3.1 and §3.4:

* **Repeated same-input matmuls**: already packed by the fused stage — one `attn_in` for Q/K/V/gate,
  one `gdn_in` for the four DeltaNet projections, one `shared_in` for the shared expert's
  gate/up/router, one packed sparse gate/up. Nothing left to pack; the open question was whether
  packing still *wins* under BFP4/LoFi, which §4.2 answers.
* **Reshard / layout conversions**: 6 per `full_attention` decode step, 1 per `linear_attention`
  one, all required by an op contract (§6 of the fused README). No avoidable ones existed to remove;
  this stage *adds* 8–12, all of them the sharded-norm boundary, and pays for them (§3.6).
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
| 15 | routed gate/up `in0_block_w` follows the active-expert bound (review round 6, §4.15) | **0.849** | **1.061** |

Warmed 2048-token prefill over the same steps: 243.44 → 96.89 ms (`full_attention`) and
257.73 → 102.94 ms (`linear_attention`). Step 5 is the one that matters for prefill and it went the
wrong way first — see §3.1.

The shipped default is re-measured end to end after every change landed, and README §5.2's headline
table is **generated** from that measurement
([`logs/ab_fused_vs_optimized.txt`](logs/ab_fused_vs_optimized.txt)) rather than transcribed here, so
this log does not carry a second copy of it to go stale: ~2.5x prefill on both layer kinds and
~1.9-2.2x on traced decode. `test_optimized_beats_fused_traced_decode` gates the decode direction in
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
shrinks, and 4 is inside the run-to-run spread of 8. 8 is shipped because it is the width the
whole-layer A/B in `ab_norm_shard_width.txt` was run at.

The 1D `mcast_in0` projection matmul that consumes the result needs an interleaved `in0` back, so
each sharded norm pays one `to_memory_config` in and one `sharded_to_interleaved` out — about 3 µs
for a larger saving on the norm itself — README §5.5's generated knob table carries both times, from
the probe. Net effect on the layer: rows 7→8 of §3's ladder.
`test_no_layout_churn_in_measured_forward` budgets those conversions exactly (5 and 14 per decode
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

The ladder is in the artifact — 22 rows per section, four grids (`8x8`, `11x10`, `8x4`, `4x8`) against five
chunk pairs (`q32 k64`, `q32 k128`, `q32 k32`, `q0 k0`, `q0 k64`), plus one labelled arm carrying the rejected
HiFi2/fp32-acc compute-kernel config so its cost stays visible — and README §5.5's generated knob table quotes
the rows the decisions rest on.

Three findings, all kept as evidence:

* the **op default is more than an order of magnitude slower** than any explicit config here, which is
  why the explicit one stays;
* a `k_chunk_size` **larger than the 64-token paged block size is wrong**, not merely risky. The
  isolated op cannot see it — the probe's reference is the op default on the same page table — but
  the layer's decode PCC against the HF golden collapses to 0.02–0.92 at the paged contexts the
  delivered tests use. So the ~10 % the k128 row promises is rejected on correctness, and
  `k_chunk_size` is now pinned to `page_block_size` in code rather than to the literal 64.

* the **grid is not a latency axis at all**, which took taking it to find out. `8x4` leads the shipped `8x8`
  in both sections of [`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt) — 60.0 vs 61.0 µs and
  60.1 vs 61.1 µs, at spreads of 0.1–0.2 and identical PCC to six decimals — and unlike the k-chunk it appeared to cost nothing: no invariant, no correctness question, and a
  dead heat at the layer ([`logs/ab_sdpa_decode_grid.txt`](logs/ab_sdpa_decode_grid.txt): the shipped arm's
  three builds are 0.888 / 0.893 / 0.889 ms and the candidate's 0.907 / 0.889 / 0.889 ms, i.e. the arms overlap
  and the spread within one arm exceeds the difference between them, SDPA being ~2 % of a step). Review round 9
  was right that nothing recorded which end of the axis shipped, so it was taken — and **the suite rejected
  it**:

  ```
  TT_FATAL @ sdpa_decode_program_factory.cpp:191: num_cores_available >= B
  test_decode_batch_above_head_split_limit[40-full_attention]  FAILED
  test_decode_batch_above_head_split_limit[56-full_attention]  FAILED
  ```

  Flash-decode assigns **at least one core per batch row**, so a 32-core grid silently caps decode at batch 32
  and the supported batch-40 and batch-56 cases die inside the op. The grid is not tuning; it is the largest
  decode batch the layer can serve, and 8x8's 64 cores are chosen to cover the 56 the suite exercises. The
  ~1 µs goes unclaimed for that reason, which is a better answer than round 9's finding asked for and a worse
  one than the sweep suggested. `11x10` satisfies the bound too and is slower in the same probe (62.4 / 62.5
  µs), so 8x8 is also the fastest grid that is *legal*, which is what the call site now says.

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
faster standalone) drops the layer's decode PCC to 0.02-0.92, and passing the prefill compute-kernel
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
  with no trace region reports 0.876 / 0.877 / 0.875 ms for the candidate against 0.871 / 0.880 / 0.870 ms for
  the shipped arm; with a reserved region, 0.875 / 0.883 / 0.875 against 0.883 / 0.870 / 0.873. The candidate is
  not ahead on either, and every difference between the arms is smaller than the span of one arm's own three
  builds. Round 8 corrected my original "inside the spread" wording to "reproducibly a microsecond or two
  faster" on the strength of that round's artifact; round 10 found the next run saying the opposite, and it is
  the current artifact that counts. The advantage is real but exists only in the isolated op
  ([`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt), 56.5 against 61.0 µs). Worth stating plainly:
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

| point | role | shipped (column) | other (row) | verdict |
| --- | --- | --- | --- | --- |
| 8 active — **the tuned batch-1 decode target** | gate/up | **219.3 µs** | 241.1 µs | column wins, ~10 % |
| 8 active | down | **214.9 µs** | 239.4 µs | column wins, ~10 % |
| ~162 active — a 32-token prefill group | gate/up | **570.0 µs** | 579.0 µs | column wins, beyond both spreads |
| ~162 active | down | 346.3 µs | **342.4 µs** | *row* wins, beyond both spreads (0.8, 0.1) |
| 64 active — decode batch 8, **not tuned** (README §9 item 5) | gate/up | **385.6 µs** | 387.7 µs | column wins, just outside the spreads |
| 64 active | down | 291.3 µs | **290.6 µs** | row wins, inside the 1.0 µs spread |

One row wants the row rectangle beyond its spread — `down` at the prefill group — and it is a geometry the
shipped code really builds, under a key with no competing point: `down` reaches 32 cores only in prefill,
because 64 active experts, the largest supported decode batch, realises 16. Review round 9 asked for exactly
that rule, and it is a one-line rule: `("down", 32) -> row`.

**So it was implemented and measured end to end, and it loses.**
[`logs/ab_sdpa_decode_grid.txt`](logs/ab_sdpa_decode_grid.txt) alternates the two arms build-by-build with
three timed builds each, discarding each arm's first:

| layer kind | column (shipped) | row (candidate) | verdict |
| --- | --- | --- | --- |
| `full_attention` prefill | 96.191 / 96.328 / 96.417 ms | 97.059 / 97.087 / 97.334 ms | column faster, no overlap |
| `linear_attention` prefill | 102.120 / 102.204 / 102.626 ms | 102.999 / 103.000 / 103.026 ms | column faster, no overlap |
| traced decode, both kinds | 0.880 / 1.080 ms | 0.880 / 1.081 ms | unchanged; decode never builds this grid |

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
it had, as the row winning by under a microsecond; re-measured from these bytes the *column* leads, by about as
much. A sub-microsecond op gap at a spread of the same order is not a fact about the hardware, which is why the
generated table in README §5.4 prints each row's own spread and why the inside-vs-beyond test is now made at
the artifact's printed precision.

For the record, the earlier states of this paragraph: round 5 found it claiming a universal column win, which
the sweep never said; round 8 found the prefill `down` sign inverted; round 9 found round 8's correction
inverted the *other* way, so the paragraph asserted a win the artifact contradicted. README §5.4's generated
table read the artifact correctly in all three rounds. That is the argument for generating tables — and, since
prose beside a correct table can still be wrong, for `audit_figures.py` checking the source comment and this
file against the same artifact.

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

Worth stating plainly: five rounds of fixing figures found no performance, and this round found close to a
percent of decode — because the table was finally required to agree with the geometry the layer actually runs. That is
the argument for mechanical agreement over careful proofreading, in one data point.

---

## 5. Hardware

One incident, caused deliberately by the static-`nnz` experiment in §4.8 and fully recovered with a
single `tt-smi -r`; the table there records the failure signature, the commands, the reset and the
mesh smoke. `tt-smi -ls --local` showed all 8 Blackhole boards before the first run of this stage and
after the last. Watcher and profiler runs were kept in strictly separate processes throughout, and no
vLLM or serving process was started at any point.

---

## 6. Review rounds and checkpoint

Two independent `$stage-review` passes ran against this stage.

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
   under both policies. The hand-modelled figures were 285 700 B low on the worst-case layer: they
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

**Round 6** returned `more-work-needed` with seven items. Two matter more than the rest, and one of them is
the first *performance* finding a review round has produced for this stage.

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
landing within a few microseconds, with 8 marginally best on both layer kinds. 8 ships because the layer
measurement says so, not because of the claim that was there.

One loose end, stated rather than glossed: the boundary conversions are the obvious reason the op-level
ladder does not transfer, but they do not explain the *sign* at 4 cores — fewer shards should be cheaper on
both the norm and the conversions, and the layer is nonetheless a microsecond slower there. So the honest
summary is that the isolated ladder does not predict the layer at this knob, and the layer is what decides.
It is not investigated further because every arm is within a few microseconds of every other, so nothing
measurable rides on the explanation.

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
  three rows share the key and `min()` took the fastest, so the table reported 237.4 µs where the layer runs
  241.5 ([`logs/probe_sparse_matmul.txt`](logs/probe_sparse_matmul.txt), the `sub_w=2` and `sub_w=8` rows of that
  geometry). **Fifth consecutive round that this one lookup was under-constrained by exactly one axis** — and the
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

The suggestion to add a decode batch in 5..8, on the grounds that those exercise the tuned dense configs at
`per_core_M` 5-8, does not hold: `per_core_M` is `ceil(batch / 32)`, so it is **1 for every batch from 1 to 32
and 2 for 40 and 56** — two classes, both already tested, plus 13 for a batch with no rectangular factor pair.
Simulating all seven dense roles across batches 1-8, 13, 32, 40 and 56 gives exactly two distinct config
signatures, so a batch-8 case would add a second copy of what batch 4 already covers. Recorded here rather than
implemented, because a test that cannot distinguish anything is not coverage.

Checkpoint: [`logs/commit_record.txt`](logs/commit_record.txt), which also records the exact command
that proves the committed tree reproduces every generator and passes the figure audit. Local commits
only; nothing is pushed.
