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

| Rank | Op code | µs/step | What it is | Candidate | Action |
| --- | --- | --- | --- | --- | --- |
| 1 | `SparseMatmul active=?/256 x 32 x 2048 x 1024` | 368.0 | packed routed-expert gate/up | BFP4 weights + LoFi; core/block geometry; L1 output; fewer active experts | **all four taken** (§3.1, §3.2, §3.3, §3.4) |
| 2 | `SparseMatmul active=?/256 x 32 x 512 x 2048` | 343.8 | routed-expert down | same | **all four taken** |
| 3 | `UnaryDeviceOperation` | 270.1 | ~99 % `UnaryOpType::FILL` — `sparse_matmul` zeroing its 256-expert-wide output | move the output to L1; halve its dtype | **taken** (§3.2, §3.3) |
| 4 | `BinaryNgDeviceOperation` | 139.8 | SwiGLU multiply + router-score multiply, both over the 256-expert axis | L1 + BFP8 intermediates | **taken** (§3.2, §3.3) |
| 5 | `MatmulDeviceOperation 32 x 4096 x 2048` | 74 | `o_proj` — flagged `SLOW`, 23.2 % of DRAM bandwidth | explicit decode program config; DRAM-sharded | **explicit 1D config taken**, DRAM-sharded measured and rejected (§4.1) |
| 6 | `SliceDeviceOperation` | 109.3 | mostly the two slices that unpack the packed gate/up output | split the pair instead; L1 + BFP8 | **packed kept** (§4.2), L1+BFP8 taken |
| 7 | `MatmulDeviceOperation 32 x 2048 x 9216` | 56 | packed attention in-projection | explicit config; BFP8/BFP4 weights | **BFP8 + explicit config taken**, BFP4 measured (§4.6) |
| 8 | `TopKDeviceOperation` | 48 | router top-8 over 256 experts, single core | pad to the multi-core width; replace the gate op | **both measured and rejected** (§4.3, §4.4) |
| 9 | `LayerNormDeviceOperation` | 51 | four RMSNorms; the two residual ones run on **one core** | width-sharded L1 + `LayerNormShardedMultiCoreProgramConfig` | **taken** (§3.6) |
| 10 | `MatmulDeviceOperation 32 x 2048 x 256` | 25 | router projection, 8 cores, 9.4 % of DRAM bandwidth | explicit config | **taken** (§3.5) |
| 11 | `DeepseekMoEFastReduceNC` | 92 | expert-axis reduction over the 256-wide down output | L1 + BFP8 input | **taken** |
| 12 | `SdpaDecodeDeviceOperation` | 18 | paged flash-decode | reduced cache dtype; program config sweep | **BFP8 cache taken** (§3.7), config swept (§4.5) |

`linear_attention` shares the whole MoE and the norms with the table above; what differs is the
mixer, and its own top items (from the same capture set, per traced step) are:

| Op code | µs/step | What it is | Action |
| --- | --- | --- | --- |
| `MatmulDeviceOperation 32 x 2048 x 12352` | 128.5 | packed DeltaNet in-projection | BFP8 + explicit decode config (§3.5) |
| `MatmulDeviceOperation 32 x 4096 x 2048` | 74 | `out_proj` | explicit decode config (§3.5) + a bfloat16 activation (§4.11) |
| 3 x `MatmulDeviceOperation b={32} 32 x 128 x 128` | 45 | the float32 recurrent-state matmuls: decay read, delta outer product, output read | operands moved to L1 — 19 µs (§3.9 item 2); fidelity swept and rejected |
| `ReshapeViewDeviceOperation` + `PermuteDeviceOperation` | 39 | the one-shot head-major relayout of the conv output | inherited from the fused stage, which already reduced it from three round trips to one |
| `TilizeWithValPadding` + `Concat` + `UntilizeWithUnpadding` | 24 | `repeat_interleave`'s GQA head expansion | output moved to L1 (§3.9 item 5); the once-per-step op-to-op stall in front of it is the largest single gap in README §7's generated itemisation |
| `TernaryDeviceOperation` | 17 | the `addcmul` conv-tap accumulation | inherited; the fused stage measured `addcmul` against `mac` and kept it |

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
  fill → reduce — is 613 µs/step, 38 % of the window, on a tensor where **8 of 256** expert slots
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
| 14 | gated-DeltaNet output activation bfloat16 (§4.11) | **0.858** | **1.071** |

Warmed 2048-token prefill over the same steps: 243.44 → 96.89 ms (`full_attention`) and
257.73 → 102.94 ms (`linear_attention`). Step 5 is the one that matters for prefill and it went the
wrong way first — see §3.1.

The shipped default is re-measured end to end after every change landed, and README §5.2's headline
table is **generated** from that measurement
([`logs/ab_fused_vs_optimized.txt`](logs/ab_fused_vs_optimized.txt)) rather than transcribed here, so
this log does not carry a second copy of it to go stale: ~2.5x prefill on both layer kinds and
~1.9-2.1x on traced decode. `test_optimized_beats_fused_traced_decode` gates the decode direction in
one process, in the delivered suite.

Row 0's `linear_attention` figure is the **fused stage's own committed number**, measured in that
stage's harness, quoted so this column starts where the previous stage left off. Re-measured in this
stage's harness it is 2.06-2.07 ms, about 4 % slower — harness and run-to-run spread. Every "before"
figure the README quotes is this stage's own re-measurement, not row 0; against the fused stage's
published figure the `linear_attention` decode speedup would read ~1.86x instead of ~1.93x.

The per-step figures in this column are the running total from one harness during development, and they
are the one group of numbers in this stage that **no artifact can contain**: each row measures an
intermediate revision of the code that no longer exists, so a re-run cannot reproduce it and
`audit_figures.py` exempts them by name, as the development ladder, only in this file. Where a step's own
A/B was captured as an artifact it is linked from the section that describes it (§3.6, §3.9, §4.10,
§4.11); the *shipped* numbers are always README §5.2's generated table. If you want to know what the
decoder does today, read §5.2, not this column.

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
count. It reproduces the measured winner at all four sweep points. `in0_block_w` takes the largest
legal divisor of `Kt` (64 and 16); both roles improve monotonically with it and then flatten, and
the one place a smaller value edged ahead — gate/up at the 8-core decode geometry, 32 over 64 — is a
2 % gap inside the run-to-run spread.

`in0_block_w = 2` never appears: the shipped values are **64** (gate/up, the whole tiled K) and
**16** (down, the whole tiled K), and the sweep shows the full ladder below them.

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
for a 12 µs saving. Net **0.875 → 0.855** and **1.111 → 1.093 ms**.
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
([`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt), `SDPA` rows, four grids × four chunk
pairs at an 8192-token context under the shipped BFP8 paged cache):

The ladder is in the artifact — 17 rows, four grids (`8x8`, `11x10`, `8x4`, `4x8`) against four chunk
pairs (`q32 k64`, `q32 k128`, `q32 k32`, `q0 k0`) — and README §5.5's generated knob table quotes the
three rows the decisions rest on: the op default, the shipped `8x8 / k64`, and the faster `k128`.

Two findings, both kept as evidence:

* the **op default is more than an order of magnitude slower** than any explicit config here, which is
  why the explicit one stays;
* a `k_chunk_size` **larger than the 64-token paged block size is wrong**, not merely risky. The
  isolated op cannot see it — the probe's reference is the op default on the same page table — but
  the layer's decode PCC against the HF golden collapses to 0.02–0.91 at the paged contexts the
  delivered tests use. So the ~10 % the k128 row promises is rejected on correctness, and
  `k_chunk_size` is now pinned to `page_block_size` in code rather than to the literal 64.

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
   group in the *slower* layer kind went unranked; §5.5 of the README now covers both reports.
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
faster standalone) drops the layer's decode PCC to 0.02-0.91, and passing the prefill compute-kernel
config to the decode op drops it to 0.33. Both failures are invisible to a standalone probe, whose
reference is the same op on the same page table. The shipped config keeps `k_chunk_size` pinned to
`page_block_size` and passes no compute-kernel config. `SdpaDecode` is 17 µs/step, 2 % of the window.

### 4.6 BFP4 dense projection weights — measured, decision recorded in §5 of the README

`proj_dtype = bfloat4_b` (packed attention in-projection, `o_proj`, packed DeltaNet in-projection,
`out_proj`) is **faster**: 0.858 → 0.838 ms and 1.093 → 1.073 ms traced decode, prefill unchanged.

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
a 48-layer stack, bought for a low-single-digit percentage of one traced decode step and nothing at all
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
baseline, and the fused decoder scores 0.994650 on it.

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
expert-major layout the fused stage's §8 item 1 identified as the way to remove the
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
  generated table, and the 1.987-vs-2.065 fused-baseline difference is stated.

Round 2's other concerns were addressed in the same pass: the `linear_attention` roofline now counts
the recurrent state the way the other kind counts its KV read, the "same-process" method claim is
corrected to name the test that actually is one, §5.4 says its core counts are program grids, the
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
  0.006 % of the prefill window), `probe_prefill_matmul.txt`;
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
   is 2–3 % off at that one point, worth ~0.4 % of a traced step. Now stated, with the gap computed.
5. **The first attempt at deriving the prefill window composition was wrong by two orders of
   magnitude**: `op_device_time` matched op codes by substring, and `MatmulDeviceOperation` is a
   substring of `SparseMatmulDeviceOperation`, so the *dense* projections were credited with 82 % of a
   window they are 0.8 % of. Matching is by prefix now, and the function says why.
6. **The op-to-op gap itemisation was reading the report wrong.** Its rows are one op *per replay*, not
   one aggregated row per op, so the first version reported 32 copies of each gap and a per-step total
   32× too large. Grouped by op code and divided by the replay count, the itemisation reconciles with
   §7's dispatch gap — and it shows the float32 gate-promotion typecasts are the *largest* line item of
   the `linear_attention` window, not the "two 6–8 µs gaps" the prose claimed.

Checkpoint: [`logs/commit_record.txt`](logs/commit_record.txt). Local commits only; nothing is pushed.
