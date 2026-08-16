# Ornith-1.0-35B — optimized multichip decoder: work log

What was done, in order, with the command and the measurement for each step. The narrative document
is [`README.md`](README.md); this is the ledger, including the things that did not work.

Every table between `<!-- TABLE:... -->` markers is generated from a committed artifact by
[`logs/make_tables.py`](logs/make_tables.py). No latency figure in either document is typed by hand.

---

## 1. Starting point

Stage 4 (`doc/multichip_decoder/`) delivered a 4-chip Blackhole ring decoder layer at

| | linear_attention | full_attention |
|---|---|---|
| warmed 2048-token prefill | 29.06 ms | 28.79 ms |
| warmed traced decode, batch 1 | 0.610 ms | 0.500 ms |

and, as it turned out, a cross-device divergence in its decode collective that no stage had seen
(§11).

reproduced at the start of this stage with the inherited harness before anything was changed, and
again at the end as the committed `before` arm of `logs/ab_before_after.txt`:

```
python models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/bench.py \
  --impl multichip --mesh 1x4 --layers 0,3 --weights real --tag before-optimized-multichip
```

**Decode reproduces exactly and prefill does not, which is expected and is why decode is the headline.**
The committed `before` arm reproduces stage 4's decode to within one unit of the harness's
resolution — see the `before` column of §4.4's generated table against the pair above — and
its prefill differs by a few tenths of a millisecond from stage 4's delivered 29.06 / 28.79. Any spread in either
direction here is the build-to-build prefill spread stage 4 documented and that this stage's own `ablayer`
prefill columns show on every arm (a range of well over a millisecond across arms that cannot affect prefill at all). Decode has
almost no such spread: every `ablayer` decode cell but one is stable to 0.001 ms. The exception is
`router_fidelity / hifi4-inherited`, `linear_attention`, build 0, which reads 0.727 against 0.565 for
the other two builds of the same arm — harness noise, and the artifact contains its own control:
eighteen further builds of that identical shipped configuration, across the `router`, `policy`,
`geometry`, `sparse`, `cast` and `residual` arms, all read 0.565-0.566. The arm is judged on
min-of-three, so the tie it reports is unaffected. Found by review round 7. No prefill claim in this stage
rests on a difference smaller than that spread.

The goal for this stage is an **in-place** optimization of that layer: same file
(`tt/multichip_decoder.py`), same suite (`tests/test_multichip_decoder.py`), same public contract.
That makes "before" a knob rather than a git revision — see §8.1 for how the before/after arm is
produced in one binary.

The stage inherits a large amount of already-earned evidence. Stage 4 swept collective spelling,
fabric config, topology argument, fabric packet size, dense decode matmul geometry, routed sparse
geometry, expert-parallel versus intermediate-sharded MoE, the local-expert narrowing, the
row-compaction fold and the fused matmul+CCL family. This stage does not re-derive those; it
re-**verifies** the ones a new default could move (§6) and spends its own search where stage 4 did
not look.

---

## 2. Operation-topology audit

Read the code and the stage-4 decode profile together before touching a knob. Merged 4-device device
time, `full_attention` decode, batch 1, 32 traced replays
(`doc/multichip_decoder/tracy/full_attention/decode_perf_report*`), 511 us/step.

| # | operation group | role | before, us/step | repeated same-input matmuls? | collective? | reshard / layout | candidate | action |
|---|---|---|---|---|---|---|---|---|
| 1 | `LayerNorm` (attn_norm) | residual norm | 4.9 | – | – | width-shard carried into the projection | already sharded, grid swept in stage 3 | keep |
| 2 | `attn_in` / `gdn_in` matmul | packed in-projection | 16.3 | **already packed** — q,k,v,gate (or qkv,z,a,b) in one matmul | – | width-sharded in0 | OPT-001 satisfied by the fused stage | keep |
| 3 | head creation + RoPE + cache write | attention plumbing | 20.5 | – | – | `Transpose`, `Concat`, `FillPad` | dedicated ops already used | keep |
| 4 | `SdpaDecode` | flash decode | 14.8 | – | – | – | explicit program config swept in stage 3 | keep |
| 5 | `o_proj` / `gdn_out` matmul | row-parallel out-projection | 9.0 | – | produces a partial | – | fused matmul+reduce-scatter | rejected on stage-4 measurement, §6.3 |
| 6 | **collective 1** (`AllGather`+`sum`) | close the row-parallel sum | 13.0 | – | **yes**, bf16 | – | `ttnn.all_reduce`; barrier-semaphore `all_gather_async`; persistent buffers | **`ttnn.all_reduce` taken, §11** — the deprecated gather diverges across devices; `all_reduce` is the cheaper of the two correct spellings at 3 us/step, the async gather rejected at 14 us/step behind it, persistent buffers rejected (§4.1b) |
| 7 | `LayerNorm` (ff_norm) + pad | MoE input | 11.0 | – | – | `FillPad` to the 32-row tile | – | keep |
| 8 | `shared_in` matmul | packed shared expert | 7.0 | **already packed** — gate,up,router column | – | – | – | keep |
| 9 | shared SwiGLU + `shared_down` + gate | shared expert tail | 12.0 | – | – | 3 slices | – | keep |
| 10 | `router` matmul | 256 logits | 8.0 | – | – | – | – | keep |
| 11 | **`TopK`** | global top-8 of 256 | **48.4** | – | – | – | fused `generalized_moe_gate` | **taken, §4** |
| 12 | `Softmax` over the kept 8 | routing weights | 6.6 | – | – | – | folded into the same kernel | **taken, §4** |
| 13 | `Scatter` chain | dense 256-wide routing vector | ~34 | – | – | 2-3 untilize + tilize | ROW_MAJOR operands | **taken, §4.3** |
| 14 | `expert_select` matmul | narrow 256 → local 64 | 3.0 | – | – | – | local index tensor in the gate | measured, rejected, §4.5 |
| 15 | sparsity mask | which local experts run | ~5 | – | – | `UntilizeWithUnpadding` to ROW_MAJOR | – | keep |
| 16 | `SparseMatmul` gate_up (packed) | routed experts | 67.0 | **already packed** | – | – | geometry swept in stage 4 | keep, re-verified §6.4 |
| 17 | SwiGLU + score multiply | routed expert tail | 20.0 | – | – | 2 slices, 1 permute | – | keep |
| 18 | `SparseMatmul` down | routed experts | 45.0 | – | – | – | geometry swept in stage 4 | keep, re-verified §6.4 |
| 19 | expert reduction | sum over local experts | 7.3 | – | – | – | – | keep |
| 20 | **collective 2** (`AllGather`+`sum`) | close routed + shared partial | 33.7 | – | **yes**, bfloat8_b | – | cast to bf16; `ttnn.all_reduce`; async gather | cast rejected §6.2; **`ttnn.all_reduce` taken, §11**, same decision as row 6 |
| 21 | trim + residual add | layer output | 4.0 | – | – | `Slice`, `ReshapeView` | – | keep |

Three conclusions drove the stage:

1. **No unpacked repeated same-input projection remains.** Every group that shares an activation —
   q/k/v/gate, the gated-DeltaNet in-projection, the shared expert's gate/up/router column, the
   routed experts' gate/up — is already one matmul, packed at load time by the fused stage. OPT-001
   and OPT-010 are inherited-satisfied, and the perf report confirms it: one `attn_in` row, not
   three.
2. **Exactly two collectives per layer, both on the smallest tensor either half can be reduced on,
   and both required by the replicated residual contract.** There is nothing to remove; §5 and §6.2
   are about how they are spelled, not how many there are.
3. **The router is the one large op no stage had attacked.** `TopK` alone is 9.5 % of the decode
   window on **one core**, and the whole routing block is ~17 %. The optimized stage had rejected the
   fused alternative as "bfloat16-only" and, in its own words, *never timed it*. That is the search
   this stage ran.

---

## 3. What the router chain actually costs

`logs/probe_gate.py`, on the target mesh at the real decode shapes. Untraced, so every row carries
the eager dispatch floor; the ranking that decides anything is the traced whole-layer A/B in §6.
These rows are for **decomposition**, not for selection.

<!-- TABLE:gate -->
| arm | rows / valid | us (untraced, min of 3x30) | PCC vs shipped | expert-set agreement |
|---|---|---|---|---|
| `shipped` | 32 / 1 | 113.1 | 1.000000 | 1.000 |
| `topk_bf16` | 32 / 1 | 115.7 | 1.000000 | 1.000 |
| `topk_unsorted` | 32 / 1 | 115.8 | 1.000000 | 1.000 |
| `shipped_local` | 32 / 1 | 140.2 | 1.000000 | 1.000 |
| `threshold_local` | 32 / 1 | 298.3 | 0.907802 | 0.000 |
| `threshold_local_noramp` | 32 / 1 | 266.7 | 0.907802 | 0.000 |
| `fused_gate` | 32 / 1 | 202.8 | 0.999992 | 1.000 |
| `fused_gate_valid` | 32 / 1 | 171.1 | 0.999992 | 1.000 |
| `shipped` | 32 / 32 | 116.3 | 1.000000 | 1.000 |
| `topk_bf16` | 32 / 32 | 115.4 | 0.999189 | 0.969 |
| `topk_unsorted` | 32 / 32 | 120.3 | 1.000000 | 1.000 |
| `shipped_local` | 32 / 32 | 147.9 | 1.000000 | 1.000 |
| `threshold_local` | 32 / 32 | 310.2 | 0.971480 | 0.656 |
| `threshold_local_noramp` | 32 / 32 | 279.2 | 0.971480 | 0.656 |
| `fused_gate` | 32 / 32 | 205.5 | 0.999986 | 1.000 |
| `fused_gate_valid` | 32 / 32 | 203.1 | 0.999986 | 1.000 |
<!-- /TABLE:gate -->

Op by op, generated from the same artifact (`GATEPART` rows; the file has two blocks, `rows=32
valid=1` — the batch-1 decode shape — and `rows=32 valid=32`):

<!-- TABLE:gateparts -->
| part | rows / valid | us (untraced, min of 3x30) |
|---|---|---|
| `topk_fp32` | 32 / 1 | 52.3 |
| `topk_bf16` | 32 / 1 | 49.5 |
| `softmax_k` | 32 / 1 | 11.9 |
| `scatter` | 32 / 1 | 49.1 |
| `fused_gate:convert_in` | 32 / 1 | 26.4 |
| `fused_gate:gate_op` | 32 / 1 | 11.1 |
| `fused_gate:convert_out+scatter` | 32 / 1 | 134.3 |
| `fused_gate_valid:convert_in` | 32 / 1 | 21.9 |
| `fused_gate_valid:gate_op` | 32 / 1 | 11.2 |
| `fused_gate_valid:convert_out+scatter` | 32 / 1 | 115.5 |
| `topk_fp32` | 32 / 32 | 52.4 |
| `topk_bf16` | 32 / 32 | 49.4 |
| `softmax_k` | 32 / 32 | 11.2 |
| `scatter` | 32 / 32 | 47.7 |
| `fused_gate:convert_in` | 32 / 32 | 27.5 |
| `fused_gate:gate_op` | 32 / 32 | 11.2 |
| `fused_gate:convert_out+scatter` | 32 / 32 | 141.3 |
| `fused_gate_valid:convert_in` | 32 / 32 | 28.3 |
| `fused_gate_valid:gate_op` | 32 / 32 | 12.4 |
| `fused_gate_valid:convert_out+scatter` | 32 / 32 | 142.0 |
<!-- /TABLE:gateparts -->

Reading it:

* the shipped chain is `topk` + `softmax` + `scatter`, and `topk` is about half of it;
* `topk` on bfloat16 logits is barely cheaper than on float32 — the dtype is not where the cost is;
* `sorted=False` changes nothing, so the ordering is not where it is either (the `GATE` table above);
* **the fused kernel is an order of magnitude cheaper than the two ops it replaces.**

The rest of §4 is about the ~100 us of conversion that stood between that ~11 us and the layer.

### 3.1 Two-stage top-k: measured and dropped

Top-8 of 256 is exactly the top-8 of the union of the per-chunk top-8s, so an eight-way chunked
`topk(k=8, width=32)` followed by `topk(k=8, width=64)` is exact. It is only worth a relayout if the
op's cost is close to linear in the searched width, which is what the optimized stage's ladder
(256 → 4096) suggested. At the low end it is not:

<!-- TABLE:topkw -->
| searched width | `ttnn.topk(k=8)` us |
|---|---|
| 32 | 48.6 |
| 64 | 37.1 |
| 128 | 36.0 |
| 256 | 59.2 |
<!-- /TABLE:topkw -->

Both blocks of the artifact say the same thing: width 32 and width 64 each cost most of what width
256 costs — a floor, not a slope. The two stages together cost more than the single
256-wide call before paying for the tile-to-face relayout the chunking needs. Dropped on that
measurement, and it is stated as the sum of two stages exceeding one rather than as a ratio, because
the `width=32` row is the noisiest in the artifact.

### 3.2 The threshold rewrite: an exact op-contract blocker

The other way to avoid the scatter is to never build the dense vector: take the 8th largest logit
`kth` and the softmax denominator from `topk`'s **values** (indices unused), narrow the logits to
this device's 64 experts with the same one-hot `expert_select` matmul, and emit
`(l >= kth) * exp(l - max) / Z` directly at 64 wide. A strictly decreasing `ramp` added before both
the `topk` and the comparison makes `>=` select exactly `k` even when logits tie — which matters,
because the tile-padding rows' logits are all exactly zero.

It does not work, for a reason that is a property of the op contract rather than of the idea. The
narrowing matmul **rounds its float32 activation to bfloat16** regardless of the weight dtype or the
compute-kernel config:

```
CHK sel DataType.FLOAT32  maxdiff 0.0019478797912597656
CHK sel DataType.BFLOAT16 maxdiff 0.0019478797912597656
```

so `kth` (taken from the unrounded logits) and the narrowed logits disagree by up to ~2e-3, and the
expert whose logit *is* the threshold is dropped whenever rounding takes it below. Measured
`setagree=0.000` on the valid row. Making both sides bfloat16 fixes the comparison and breaks the
ramp: at |logit| ~ 2 the bfloat16 ulp is ~1.6e-2, so a ramp small enough to be inert is invisible and
the ties it exists to break come back — and in bfloat16 near-ties are common, not rare. Rejected with
that blocker recorded rather than on the first wrong number.

---

## 4. The fused router gate

### 4.1 What it is

`ttnn.experimental.deepseek.moe.generalized_moe_gate` scores the router logits, selects the top-`k`
and normalizes the selected scores with a softmax, in one kernel, over a 16x16 expert face held in
the top-left face of a 32x32 tile, height-sharded one token per core. Ornith's router is
`softmax(topk(logits))` with no score-correction bias, 256 experts and top-8, which is exactly
`enable_sigmoid=False, output_softmax=True, topk=8, scaling_factor=1.0` with a zero bias.

The four tensors it needs — bias, input indices, output scores, output indices — are built once in
`MultichipMoE.prepare_decode_gate`, called from `MultichipDecoder.allocate_state`. They are
**preallocated output buffers**: the op writes into them and returns them, so a decode step allocates
nothing for the gate. `routing_weights` does a dict lookup and never a build, because
`prepare_decode_gate` uploads from host and a decode forward may be under trace capture; a row count
that was not prepared keeps the `topk` chain, which is slower and correct.

Prefill keeps the `topk` chain deliberately. The op is one token per core, so a 2048-token chunk
would need 19 sequential calls, and `TopK` is under a fifth of a percent of the prefill window.
`test_decode_runs_the_fused_router_gate` pins both halves of that contract in the runtime path:
exactly one gate call and zero `ttnn.topk` calls in a decode forward, and the reverse in a prefill
forward.

### 4.2 It is a precision change, and it is measured as one

The kernel reads bfloat16 logits where the chain read float32. Every earlier stage defended float32
router logits on the ground that expert *selection* is discrete — a rounding change swaps an expert
rather than perturbing a value — and the optimized stage rejected this op on that ground without
timing it. So the change is gated on the quantity that actually matters, on real checkpoint weights:

`tests/test_multichip_decoder.py::test_router_modes_agree` runs four decode steps after a 128-token
prefill under both modes and compares the per-token **selected-expert set** taken from the device's
own routing vector, plus layer output PCC against the float32 HF golden:

```
router modes layer=0 candidate=fused_gate step=0: topk-vs-golden PCC 0.999987,
    fused_gate-vs-golden PCC 0.999987, fused_gate-vs-topk PCC 1.000000, expert sets equal
router modes layer=3 candidate=fused_gate step=7: topk-vs-golden PCC 0.999895,
    fused_gate-vs-golden PCC 0.999896, fused_gate-vs-topk PCC 0.999999, expert sets equal
...
router modes layer=0 candidate=fused_gate: identical expert set on 8/8 decode steps
router modes layer=3 candidate=fused_gate: identical expert set on 8/8 decode steps
```

Identical expert sets on all 8 steps for both layer kinds, and golden PCC matching the float32
chain's to five or six decimal places on every step (0.999735-0.999987 on `linear_attention`,
0.999890-0.999939 on `full_attention`).

The adversarial sample is in `logs/probe_gate.txt`, on random-normal logits at 32 real rows rather
than checkpoint decode logits: the fused gate agrees with the float32 chain on 32/32 token sets,
while `ttnn.topk` on **bfloat16** logits agrees on only 31/32 (`setagree=0.969`, PCC 0.999189). So
bfloat16 logits *can* move a selection, and on this evidence the fused kernel's ranking tracks
float32 more closely than a bfloat16 `topk` does. The mechanism is not established - only the
measurement - so the claim made here is the narrow one: 16 real-weight decode steps and 32
random-logit tokens, all with identical selected sets.

### 4.3 The layout trap, and what fixed it

The first working version was **10 us/step faster**, not 50. A decode profile of it said why: the
gate op cost 2 us on 32 cores, and the conversion around it had added

* 2 x `ReshapeViewDeviceOperation` at **13.8 us each** — the `[rows, 1, k] -> [1, 1, rows, k]`
  reshape the scatter's operand contract needs, which in TILE layout gathers `rows` one-row tiles
  into one tile;
* 3 x `UntilizeWithUnpadding` inside `ttnn.scatter`, which converts every non-ROW_MAJOR operand
  itself (`scatter.cpp:164/193`).

Both are the same mistake: handing a row-major-shaped problem to tiled tensors. Taking the gate's two
outputs to ROW_MAJOR **before** the slice makes the reshape a metadata view and lets the scatter run
on its native layout. That single change moved the layer from 0.600/0.490 ms to 0.560/0.452 ms.

The scatter base was then also made a persistent ROW_MAJOR buffer, so the op no longer untilizes a
256-wide base on the way in and re-tilizes on the way out; that one is worth 1-2 us, at the edge of
the harness's resolution, and is kept because it removes two conversions rather than because the
number is large.

### 4.4 Result

<!-- TABLE:bench -->
| layer kind | phase | single-chip baseline | before (multichip stage) | after (this stage) | delta | speedup vs 1 chip |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.73 ms | 28.98 ms | **29.56 ms** | +2.0% | 3.441x |
| linear_attention | decode (traced) | 1.030 ms | 0.610 ms | **0.564 ms** | -7.5% | 1.826x |
| full_attention | prefill 2048 | 95.40 ms | 28.42 ms | **28.41 ms** | -0.0% | 3.358x |
| full_attention | decode (traced) | 0.827 ms | 0.500 ms | **0.453 ms** | -9.4% | 1.826x |
<!-- /TABLE:bench -->

### 4.5 `fused_gate_local`: measured and rejected

The gate returns, for each selected slot, whatever `input_indices_tensor` holds at that expert
position. Feeding it a **device-local** index — `e - d*64` for this device's experts and a dump
column for the rest — makes the scatter land straight in the 64-wide local block, which deletes the
`expert_select` one-hot matmul (3 us) and shrinks the scatter base from 256 to 65. It is exact:
`test_router_modes_agree[fused_gate_local]` gets identical expert sets on 8/8 steps for both layer
kinds and the same golden PCC.

It is also 1-2 us **slower**, on three builds per arm with a 0.001 ms spread — the `router` rows of
§6's generated table.

The slice that drops the dump column costs what the removed matmul and the narrower base save. Kept
in the code as a measured alternative, not shipped.

---

## 5. Persistent / preallocated collective buffers (OPT-009)

Stage 4 swept collective *spelling* and the fabric arguments but shipped the deprecated
`ttnn.all_gather`, which allocates its output every step, and never tried the thing OPT-009 asks for
by name. `logs/probe_ccl_persistent.py` measures `ttnn.experimental.all_gather_async` with an
explicit `persistent_output_buffer` and the `chunks_per_sync` / `num_workers_per_link` /
`num_buffers_per_channel` knobs, at both operand dtypes, traced with 8 copies per trace so the
per-replay dispatch is amortised the same way stage 4's `probe_ccl.txt` amortises it.

The first call was refused — `all_gather_async_device_operation.cpp:57: semaphore.size() == 2`,
"Default implementation requires 2 semaphores". That is a call-shape error, not a property of the op,
and it is the same mistake stage 4 found itself making with `all_reduce_async`; the arm passes two
semaphores and runs.

<!-- TABLE:cclpers -->
| dtype | shape | arm | chunks_per_sync | workers/link | buffers/channel | traced us | PCC |
|---|---|---|---|---|---|---|---|
| bfloat16 | decode | `ag_stack_sum_shipped` | None | None | None | 15.83 | 0.999998 |
| bfloat16 | decode | `ag_async` | None | None | None | 21.31 | 0.999998 |
| bfloat16 | decode | `ag_async_persist` | None | None | None | 18.87 | 0.999998 |
| bfloat16 | decode | `ag_async_persist_tuned` | 2 | None | None | 18.87 | 0.999998 |
| bfloat16 | decode | `ag_async_persist_tuned` | 8 | None | None | 18.90 | 0.999998 |
| bfloat16 | decode | `ag_async_persist_tuned` | None | 2 | None | 18.87 | 0.999998 |
| bfloat16 | decode | `ag_async_persist_tuned` | None | None | 2 | 18.68 | 0.999998 |
| bfloat16 | decode | `ag_async_persist_tuned` | 8 | 2 | 2 | 18.67 | 0.999998 |
| bfloat16 | rows64 | `ag_stack_sum_shipped` | None | None | None | 22.69 | 0.999998 |
| bfloat16 | rows64 | `ag_async` | None | None | None | 28.16 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist` | None | None | None | 25.45 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist_tuned` | 2 | None | None | 25.56 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist_tuned` | 8 | None | None | 25.48 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist_tuned` | None | 2 | None | 25.49 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist_tuned` | None | None | 2 | 24.30 | 0.999998 |
| bfloat16 | rows64 | `ag_async_persist_tuned` | 8 | 2 | 2 | 24.35 | 0.999998 |
| bfloat16 | decode_b32 | `ag_stack_sum_shipped` | None | None | None | 193.96 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async` | None | None | None | 225.59 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist` | None | None | None | 222.41 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist_tuned` | 2 | None | None | 236.24 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist_tuned` | 8 | None | None | 224.14 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist_tuned` | None | 2 | None | 272.81 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist_tuned` | None | None | 2 | 222.69 | 0.999998 |
| bfloat16 | decode_b32 | `ag_async_persist_tuned` | 8 | 2 | 2 | 269.80 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_stack_sum_shipped` | None | None | None | 374.24 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async` | None | None | None | 438.56 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist` | None | None | None | 435.89 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist_tuned` | 2 | None | None | 461.71 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist_tuned` | 8 | None | None | 441.82 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist_tuned` | None | 2 | None | 536.45 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist_tuned` | None | None | 2 | 436.07 | 0.999998 |
| bfloat16 | prefill_2048 | `ag_async_persist_tuned` | 8 | 2 | 2 | 532.71 | 0.999998 |
| bfloat8_b | decode | `ag_stack_sum_shipped` | None | None | None | 14.83 | 0.999953 |
| bfloat8_b | decode | `ag_async` | None | None | None | 19.88 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist` | None | None | None | 18.57 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist_tuned` | 2 | None | None | 18.59 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist_tuned` | 8 | None | None | 18.57 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist_tuned` | None | 2 | None | 18.58 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist_tuned` | None | None | 2 | 18.48 | 0.999953 |
| bfloat8_b | decode | `ag_async_persist_tuned` | 8 | 2 | 2 | 18.47 | 0.999953 |
| bfloat8_b | rows64 | `ag_stack_sum_shipped` | None | None | None | 21.70 | 0.999954 |
| bfloat8_b | rows64 | `ag_async` | None | None | None | 24.92 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist` | None | None | None | 22.54 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist_tuned` | 2 | None | None | 21.67 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist_tuned` | 8 | None | None | 22.47 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist_tuned` | None | 2 | None | 22.47 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist_tuned` | None | None | 2 | 21.25 | 0.999954 |
| bfloat8_b | rows64 | `ag_async_persist_tuned` | 8 | 2 | 2 | 21.26 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_stack_sum_shipped` | None | None | None | 153.66 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async` | None | None | None | 141.58 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist` | None | None | None | 138.08 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist_tuned` | 2 | None | None | 192.15 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist_tuned` | 8 | None | None | 146.63 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist_tuned` | None | 2 | None | 184.02 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist_tuned` | None | None | 2 | 135.98 | 0.999954 |
| bfloat8_b | decode_b32 | `ag_async_persist_tuned` | 8 | 2 | 2 | 182.55 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_stack_sum_shipped` | None | None | None | 298.99 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async` | None | None | None | 265.19 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist` | None | None | None | 261.62 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist_tuned` | 2 | None | None | 373.91 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist_tuned` | 8 | None | None | 284.83 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist_tuned` | None | 2 | None | 353.43 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist_tuned` | None | None | 2 | 260.00 | 0.999954 |
| bfloat8_b | prefill_2048 | `ag_async_persist_tuned` | 8 | 2 | 2 | 359.50 | 0.999954 |
<!-- /TABLE:cclpers -->

At the decode tile — the only shape the layer sends down this path, since `CCL_MODE="auto"` takes
`all_reduce` above 64 physical rows — the shipped deprecated op wins on both dtypes, and persistence
helps the async family without closing the gap:

<!-- TABLE:cclpersdecode -->
| dtype | `ttnn.all_gather` (deprecated) | `all_gather_async` | + persistent buffer | + best tuned |
|---|---|---|---|---|
| bfloat16 | **15.83 us** | 21.31 | 18.87 | 18.67 |
| bfloat8_b | **14.83 us** | 19.88 | 18.57 | 18.47 |
<!-- /TABLE:cclpersdecode -->

Persistence is worth ~11 % *within* the async family and the async family is ~18-25 % behind the
shipped op, so the persistent-buffer candidate is rejected on measurement. At the larger shapes the
async+persistent arm does beat `ag_stack_sum` on bfloat8_b — read the `decode_b32` and
`prefill_2048` rows of the generated table above — but those shapes take `ttnn.all_reduce` in the
layer, which stage 4 measured at 86.47 and 152.91 us for the same operands
(`doc/multichip_decoder/logs/probe_ccl.txt`, `CCLBF8 ... trace` rows), so the async arm is not
competitive there either.

Recorded per OPT-009 for each material decode CCL row: payload dtype bfloat16 (mixer) and bfloat8_b
(MoE); input and output memory config DRAM interleaved; CCL dim 0 onto a new leading axis; topology
Ring under `FABRIC_1D_RING`; `chunks_per_sync`, `num_workers_per_link` and `num_buffers_per_channel`
swept over `{2, 8} x {2} x {2}` on the persistent arm; persistent output buffer **available and
measured slower**; persistent intermediate buffer not exposed by this op.

---

## 6. Re-verification under the new default

A knob settled against a 500 us decode window has not necessarily been settled against a 450 us one.
`logs/ab_layer_knobs.py` re-runs every stage-4 knob under this stage's default, three builds per arm
in one process on one device with the same real weights.

<!-- TABLE:ablayer -->
| knob | arm | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|---|
| `router` | fused_gate | 0.565–0.567 | 0.453 | 29.04–29.17 | 28.43–28.46 |
| `router` | fused_gate_local | 0.567 | 0.455 | 29.04–29.33 | 28.24–28.44 |
| `router` | topk | 0.620–0.621 | 0.508–0.509 | 29.08–29.68 | 28.53–29.04 |
| `collective` | all_reduce | 0.565–0.566 | 0.453 | 28.88–29.38 | 28.19–28.49 |
| `collective` | stack_sum_async | 0.579–0.580 | 0.467 | 28.91–29.12 | 28.13–28.76 |
| `collective` | stack_sum-deprecated | 0.562 | 0.450–0.451 | 29.04–30.23 | 28.49–29.11 |
| `policy` | optimized | 0.565–0.566 | 0.453 | 29.04–29.61 | 28.35–29.29 |
| `policy` | bfp4-projections | 0.565–0.566 | 0.450–0.451 | 28.82–28.83 | 28.56–29.13 |
| `geometry` | multichip-retuned | 0.565–0.566 | 0.453 | 29.08–29.11 | 28.29–28.79 |
| `geometry` | single-chip-inherited | 0.583–0.584 | 0.465 | 28.93–29.48 | 28.16–28.55 |
| `sparse` | tp-rescaled | 0.565–0.566 | 0.453 | 28.91–30.72 | 28.26–28.91 |
| `sparse` | single-chip-inherited | 0.565–0.566 | 0.453–0.454 | 33.10–33.52 | 32.32–33.10 |
| `cast` | block-float | 0.565–0.566 | 0.453 | 28.92–29.01 | 28.09–30.24 |
| `cast` | bf16 | 0.569 | 0.456–0.457 | 29.12–29.47 | 28.25–28.59 |
| `router_fidelity` | hifi4-inherited | 0.565 | 0.453 | 28.91–29.65 | 28.12–28.42 |
| `router_fidelity` | hifi2 | 0.565–0.566 | 0.453 | 28.95–29.10 | 28.40–28.99 |
| `state_fidelity` | hifi4-inherited | 0.565–0.566 | - | 29.02–29.56 | - |
| `state_fidelity` | hifi2 | 0.565 | - | 28.95–29.53 | - |
| `residual` | l1 | 0.565–0.566 | 0.453–0.454 | 29.10–29.25 | 28.11–28.42 |
| `residual` | dram-interleaved-inherited | 0.565–0.566 | 0.453 | 28.96–29.36 | 28.53–29.01 |
<!-- /TABLE:ablayer -->

### 6.1 `collective` — the crossover is void, not moved

**The crossover is void, not moved.** It existed because the deprecated `ttnn.all_gather` spelling of
`stack_sum` was 3 us/step faster than `all_reduce` at the decode tile — and that spelling is the one
that diverges (§11). Of the two correct candidates, `all_reduce` is 14 us/step ahead of
`all_gather_async`, so this stage ships `CCL_MODE="all_reduce"` at every shape and keeps `"auto"` as
a measurement arm. Review round 2 caught an earlier version of this stage shipping the async gather
and this section still claiming, from the pre-fix numbers, that `auto` beat `all_reduce`.

### 6.2 `cast` — the block-float collective anomaly survives the new default

Stage 4 found the MoE's bfloat8_b collective costing an order of magnitude more than the mixer's
bfloat16 one at the same logical shape in the prefill profile, and found that casting the operand up
to bfloat16 costs 3-4 us/step and moves prefill by nothing — i.e. the cost is not data movement.
Re-measured here: the cast still costs 3 us/step on both kinds and still moves prefill by nothing.
The classification stands: what the collective's device time is absorbing is device skew at the
barrier, not payload.

### 6.3 The fused matmul+CCL family and the sharded residual

Not re-run on hardware; stage 4's `probe_fused_ccl.txt` measured both boundaries at the real
per-device shapes, traced, and this stage changes neither the projections nor the residual contract
those rows describe. Its conclusions are inherited explicitly rather than silently:

* fused `matmul_reduce_scatter_async` at the row-parallel `o_proj` boundary is slower than the
  unfused pair at every shape once the residual has to come back replicated (49.37 vs 40.78 us at the
  decode tile);
* the fused producer **without** the gather — the sharded-residual lower bound — does beat the
  shipped arm at the two larger shapes (42.78 vs 40.78 at decode, but 149.37 vs 185.41 at batch 32
  and 263.44 vs 293.02 at prefill), and the consumer pays it back and more: an `attn_in` that must
  all-gather its input costs 50.77 vs 43.55 us at the decode tile and 204.45 vs 146.70 at batch 32.
  That is the stack-compatible measurement the sharded-residual family needs — the *next consuming
  boundary* was measured, not just an immediate restore — and it is net worse at every shape;
* `all_gather_matmul_async` hung the mesh at the decode shape and is an exact op-contract blocker
  with a `tt-triage` capture in `doc/multichip_decoder/triage/`.

§9 of this log writes the residual contract down explicitly so full-model bringup inherits it as a
contract rather than rediscovering it.

### 6.4 `residual` — the largest untried lever, measured, and a tie

Review round 1 named the elementwise/layout families as the largest thing this stage's audit had
marked `keep` with no candidate: the decode window is about a fifth layout and a tenth to a sixth
`BinaryNg` — the generated share table in README §5.2 — with rows marked `in0:dram_interleaved` on
tensors that are one 32-row tile. OPT-003 says the same thing from
the other direction — a decode residual should not be in DRAM for convenience.

`DECODE_RESIDUAL_MEMORY` puts both residual adds in L1. Measured at the layer, three builds per arm:
it is a **tie** on both layer kinds, inside the 0.001 ms spread. The reading is the one §5.4's
accounting already gives: the step is launch-bound at ~7 % of the DRAM roofline, so where a 4 KiB
tensor lives does not matter — what matters is how many ops there are, and this candidate changes
none. The inherited spelling ships; the knob and its row stay so the next stage inherits the
measurement rather than the question.

This also bounds the rest of that family. The individual `BinaryNg` and `Slice` rows in the captures
are 2-5 us each and there are 10-16 of them; removing any one of them is worth less than the 0.001 ms
this arm could not detect, and removing several needs op-level fusion that TTNN does not offer for
this graph today. That is the shape of README limitation 1.

### 6.5 `router_fidelity` and `state_fidelity` — two advisories, tried, both ties

`tt-perf-report` advises HiFi2 on the two float32-fidelity rows in the decode window: the router
matmul (8 us/step, both layer kinds) and the DeltaNet recurrent-state matmuls (~5 us x 2 per step,
`linear_attention` only). Both were implemented as knobs and measured at the layer rather than
answered in prose, because the goal contract does not allow an applicable advisory to be handed off:

* `ROUTER_DECODE_FIDELITY` — the advisory has a real point here, since this stage's fused gate rounds
  the logits to bfloat16 anyway, so HiFi4 with float32 accumulation is buying precision the consumer
  discards. Measured: a tie on both layer kinds.
* `state_fidelity` (a `PrecisionPolicy` field, so the arm passes a modified policy rather than a new
  knob) — measured: a tie.

Both keep the inherited HiFi4. A tie is not a reason to weaken a discrete routing decision or an fp32
recurrent state, and §5.4's accounting says why there was nothing to win: the step is launch-bound at
~7 % of the DRAM roofline, so halving the math on a 5-8 us row does not show up.

The `state_fidelity` advisory is additionally **mis-fired** and is reported as such: its text is
"HiFi2 is sufficient for BFP8 multiplication", on a row whose dtypes are `FP32 x FP32 => FP32`.

### 6.6 A guard, so the next reviewer does not have to find this by hand

Review round 2 found this stage shipping a collective 14 us/step slower than an arm it had itself
measured, because the crossover moved and the prose did not. `logs/run_evidence.sh` now ends the `ab`
step with a check that prints, for every knob and layer kind, which arm was fastest and whether that
is the shipped one — with ties (inside the harness's 0.0015 ms resolution) and the two arms that are
*expected* to be faster than what ships (the diverging collective, and the BFP4 policy rejected on
accuracy) labelled as such, so an unexplained flag is visible rather than buried in 100 rows.

### 6.7 `geometry` and `sparse` — both still earn their place

The multichip-retuned dense decode geometry is about 18 us/step ahead on `linear_attention` and
12-13 us ahead on `full_attention`; the tp-rescaled routed-sparse core rule is 4-5 ms/layer ahead at prefill and a
tie at batch-1 decode by construction. Both unchanged in direction from stage 4 and both re-measured
here rather than assumed.

---

## 7. Precision and fidelity

### 7.1 Dense projections: BFP4 re-measured in the multichip topology (OPT-007)

The optimized stage measured a BFP4 dense-projection policy on real weights, found every PCC row
clear of the 0.995 bar, found it ~2 % faster, and rejected it on a compounding judgement — one layer
of forty spending most of its margin — shipping it as `POLICIES["bfp4-projections"]` for
`$datatype-sweep`. OPT-007 asks for that decision to be re-taken on the topology the stage actually
ships, because the per-device projections here are four times narrower.

Re-measured (`policy` arm above): a **tie** on `linear_attention` and about **−3 us, −0.7 %** on
`full_attention`, prefill unchanged within spread. TP=4 has
made the candidate *less* attractive than it was on one chip: the projections it shrinks are already
a quarter of their single-chip width, so the DRAM they save is a smaller share of a decode step that
is launch-bound rather than bandwidth-bound. The single-chip stage's accuracy ledger is unchanged by
sharding — the same weights are quantised the same way — so the trade is now a 31x increase in layer
error for 0 to 0.7 % of a decode step. Rejected again, on this stage's own numbers, and still
available as `POLICIES["bfp4-projections"]`.

### 7.2 Router logits: bfloat16, with the selection evidence in §4.2

This is the one precision change the stage takes. It is a change to the *stored* logits, not to the
router matmul, which keeps HiFi4 with float32 accumulation.

### 7.3 Everything else inherited

KV cache bfloat8_b, routed expert weights bfloat4_b/LoFi, dense projections bfloat8_b/HiFi2, router
weight bfloat16/HiFi4/fp32-acc, norms and DeltaNet state unchanged. The activation/CCL dtype question
is §6.2's `cast` arm, measured again here.

### 7.4 The dtype policy is verified in the measured rows, not in a constant (OPT-013)

The decode capture in `tracy/` shows, for the dominant matmuls, `HiFi2 BF16 x BFP8 => BF16` on
`attn_in`, `o_proj`, `shared_in` and `shared_down`; `HiFi4 BF16 x BF16 => BF16` on the router (the
`=> BF16` is this stage's change, and it is visible in the row); and `LoFi BF16 x BFP4 => BFP8` on the routed
gate/up `SparseMatmul` with `LoFi BFP8 x BFP4 => BFP8` on the routed down one, which consumes the
block-float expert activation. The claimed policy is the measured policy.

---

## 8. Evidence runs

### 8.1 Before/after, same harness, one binary

An in-place optimization has no "previous file" to benchmark, so `logs/bench.py` gains three
arguments over the copy it was taken from: `--router-mode`, `--ccl-mode` and `--auto-stack-sum`,
which set the corresponding `multichip_decoder` module knobs before the layer is built. The `before`
arm passes **all three** (`--router-mode topk --ccl-mode auto --auto-stack-sum stack_sum`), because
this stage changed two things and restoring one of them would not be stage 4's path; the default
reproduces this stage's. An earlier version of the sweep restored only the router and produced a
"before" of 0.625 / 0.515 ms, which is neither stage's path — it was this stage's collective under
stage 4's router. Both arms therefore share the build, the process shape, the
weights and the warmup. The single-chip `OptimizedDecoder` on a 1x1 mesh is measured in the same
sweep as the parallel-efficiency reference. Raw rows: `logs/ab_before_after.txt`.

### 8.2 Correctness

`logs/pytest_full_suite.txt.gz` — the whole `tests/test_multichip_decoder.py` on the final default
path. Prefill and decode PCC against the float32 HF golden for both layer kinds, the single-chip
comparison tests, paged cache, permuted page tables, ragged per-user positions, batches 1/4/13/32,
non-aligned lengths, the full advertised 262144-token context, trace replay, determinism, stress and
the no-host-fallback guard. Sixteen of the 153 are new in this stage, against the multichip stage's 137:
`test_router_modes_agree` (2 layer kinds x 2 candidate router modes),
`test_decode_runs_the_fused_router_gate` (2) and
`test_fused_router_gate_covers_every_supported_batch` (2 x 5 batches). `test_ccl_modes_agree` keeps
its count: it gained the `stack_sum_async` arm and lost the `all_reduce` one, which is now the
reference rather than an arm.

### 8.3 Profiler

`tracy/{linear,full}_attention/{prefill,decode}_perf_report.*`, four captures, prefill and decode in
separate runs, `--active-experts` at the measured per-device per-group count (4 at batch-1 decode, 41
for a 32-token prefill group), advice enabled. Regenerated by `tracy/run_profiling.sh`.

### 8.4 Watcher

`watcher/watcher_pytest.txt.gz`, `TT_METAL_WATCHER=10`, a separate run from every profiler run, with
`TT_METAL_WATCHER_DISABLE_ETH=1` for the reason stage 4 recorded: watcher's ACTIVE_ETH kernel config
buffer overflows on this 4-chip configuration. Every worker-core assert is armed.

---

## 9. The inter-layer residual layout contract

Written down here so full-model bringup preserves it instead of rediscovering it.

```
between two Ornith decoder layers on the 1x4 Blackhole ring
--------------------------------------------------------------------------------
tensor            hidden / residual stream
logical shape     [batch, seq, 2048]           (rank 3; decode is seq == 1)
dtype             bfloat16
layout            TILE
memory config     DRAM interleaved
mesh distribution REPLICATED - bitwise identical on all four devices
collectives at    NONE. Zero gather, reshard, all-reduce or all-gather between layers.
  the boundary
```

* **Both collectives are inside the layer**, immediately before each residual add, on the
  residual-shaped `[b, t, 2048]` tensor — the smallest tensor either half can be reduced on. A layer
  therefore consumes a replicated residual and produces a replicated residual, and a stack of them
  needs no boundary conversion at all.
* `tests/test_multichip_decoder.py::test_output_is_identical_on_every_device` asserts **bitwise**
  equality of the output across the mesh, so "replicated" is a checked property and not a convention.
* **Do not insert a layer-to-layer collective to simplify a wrapper.** There is nothing to insert:
  the contract is already collective-free at the boundary.
* **Do not convert the boundary to a sharded or fractured residual without re-running
  `doc/multichip_decoder/logs/probe_fused_ccl.py`.** That family was measured at both boundaries
  (§6.3): the reduce-scatter producer alone is faster at the larger shapes, and the column-parallel
  consumer that would then have to all-gather its input costs more than the producer saves, at every
  measured shape. The replicated residual is also what makes both RMSNorms exact and local (no
  distributed-norm statistics all-gather) and what makes the full-width activation available to
  expert parallelism without a gather.
* Everything else crossing the boundary is unchanged from the multichip stage: the page table is
  replicated and indexes blocks, `current_pos` and `rot_idxs` are replicated ROW_MAJOR index
  tensors, the paged KV cache is sharded on the kv-head axis with device `d` owning global kv head
  `d // 2`, and the DeltaNet recurrent and conv states are sharded on the value-head axis.

---

## 10. Rejected, with the number that rejected it

| candidate | why it was plausible | verdict | evidence |
|---|---|---|---|
| Two-stage chunked `topk` | top-8 of 256 = top-8 of the per-chunk top-8s, exactly | rejected: the `TOPKW` ladder (work log §3.1) shows width 32 and width 64 each costing most of what width 256 costs — the op has a floor, not a slope, so two stages sum to more than one, and the chunking needs a relayout on top | `logs/probe_gate.txt` `TOPKW` rows |
| Threshold rewrite at 64 wide (no scatter) | the dense 256-wide vector exists only to be narrowed | rejected on an exact op-contract blocker: the narrowing matmul rounds its float32 activation to bfloat16 (`maxdiff 1.9e-3`), so the threshold and the narrowed logits disagree at the 8th expert; in bfloat16 the tie-break ramp is below the ulp | §3.2 |
| `fused_gate_local` | deletes the `expert_select` matmul and shrinks the scatter base 256 → 65 | rejected: 1 us slower on both layer kinds, three builds each, 0.001 ms spread. Exact — identical expert sets 8/8 steps | `ab_layer_knobs.txt` `router` rows; `test_router_modes_agree[fused_gate_local]` |
| `all_gather_async` + `persistent_output_buffer` (OPT-009) | the shipped decode collective is the deprecated non-persistent op | rejected: the best tuned persistent arm loses to the shipped op at the decode tile on both operand dtypes — see the generated table in section 5 — after sweeping `chunks_per_sync`/`num_workers_per_link`/`num_buffers_per_channel` | `logs/probe_ccl_persistent.txt` |
| `POLICIES["bfp4-projections"]` (OPT-007) | faster on one chip, every PCC row above the bar | rejected again: a tie on `linear_attention` and −0.7 % on `full_attention` in this topology, against a 31x increase in single-layer error. Still shipped as a named policy | `ab_layer_knobs.txt` `policy` rows; `doc/optimized_decoder/logs/probe_projection_dtype.txt` |
| `CCL_CAST_BLOCKFLOAT` | the bfloat8_b collective's device time is an order of magnitude above the bf16 one at the same shape | rejected again under the new default: 3 us/step worse, prefill unchanged | `ab_layer_knobs.txt` `cast` rows |
| `CCL_MODE="auto"` (stack-sum crossover) at decode | 3 us/step faster at the batch-1 tile with the deprecated gather | **the fast spelling is the one that diverges** (§11). Of the two correct spellings `all_reduce` is 14 us/step ahead, so `all_reduce` ships and `auto` is a measurement arm | `ab_layer_knobs.txt` `collective` rows; `probe_replay_divergence_ab.txt`; `probe_replay_divergence_allreduce.txt` |
| `all_gather_async` + barrier semaphore as the shipped decode collective | correct, and the natural fix for a semaphore-free op | rejected: correct (0/600) but 14 us/step behind `ttnn.all_reduce`, which is also 0/600 | `ab_layer_knobs.txt` `collective` rows |
| HiFi2 for the DeltaNet recurrent-state matmuls | `tt-perf-report` advises it on every `linear_attention` decode capture, on ~5 us x 2 per step | rejected: a **tie** (0.565-0.566 both arms). Separately the advisory mis-fires — it names BFP8 on an `FP32 x FP32` row — and that is reported as a `tt-perf-report` improvement candidate | `ab_layer_knobs.txt` `state_fidelity` rows |
| HiFi2 for the decode router matmul | `tt-perf-report` advises it, and the fused gate rounds the logits to bfloat16 anyway, so HiFi4 + fp32 accumulate buys precision the consumer discards | rejected: a **tie** on both layer kinds at a 0.001 ms spread. Kept at the inherited HiFi4 because a tie is not a reason to weaken a discrete decision | `ab_layer_knobs.txt` `router_fidelity` rows |
| `DECODE_RESIDUAL_MEMORY = L1` (OPT-003) | a tenth to a sixth of the decode window is `BinaryNg` and about a fifth is layout (README §5.2's generated share table), some of it on DRAM-interleaved one-tile tensors | rejected: a **tie** on both layer kinds. The step is launch-bound; where a 4 KiB tensor lives does not change the op count | `ab_layer_knobs.txt` `residual` rows |
| Single-chip decode matmul geometry | fewer tables to maintain | rejected again: about 18 / 12-13 us/step worse | `ab_layer_knobs.txt` `geometry` rows |
| Single-chip routed-sparse core rule | fewer rules to maintain | rejected again: 4-5 ms/layer worse at prefill | `ab_layer_knobs.txt` `sparse` rows |
| Fused matmul+CCL at either boundary; sharded/fractured residual | fewer ops, fewer bytes | inherited rejection, measured at both boundaries including the consuming one | `doc/multichip_decoder/logs/probe_fused_ccl.txt`; §6.3 |

---

## 11. The cross-device divergence: found, reproduced, fixed

This started as an anomaly and ended as a bug fix, so it is written in that order.

### 11.1 The observation

In one full-suite run of the optimized default,
`test_traced_replay_does_not_leak[full_attention]` failed its cross-device bitwise equality
assertion after 128 traced decode replays: device 2's output differed from device 0's in the tail of
the hidden dimension (-0.0413 vs -0.0366, 0.2676 vs 0.2451, -0.4355 vs -0.3828) while the leading
elements were identical. It then passed twice in isolation and twice more in full-suite runs.

**The full log of that run was lost.** `logs/run_evidence.sh` wrote the suite to a fixed filename and
a later, passing run of the same script overwrote it before it was copied aside. That is a process
failure of this stage, recorded as one: what survives is
[`logs/traced_replay_divergence_failure.txt`](logs/traced_replay_divergence_failure.txt) — the
failing assertion's own output and the run's summary line — and the script now copies a failing log
to `logs/pytest_full_suite_FAILED.txt` before anything can rewrite it. Review round 1 also caught an
earlier version of this section citing `logs/pytest_full_suite_first_run.txt.gz` as "the failing run,
preserved"; that file was a different, *passing* run and has been removed rather than relabelled.

### 11.2 The first control was too weak, and said so

The first version of `logs/probe_replay_divergence.py` ran `--layers 3` only, in the back-to-back
non-blocking burst pattern only, and found nothing in 30 x 256 replays per router mode. Review round
1 pointed out that this covered one of the two layer kinds and one of the two replay patterns, and
that the failing test synchronizes after **every** replay, which is different pressure on the fabric.

Both gaps closed (`--sync-every-replay`, and both layer kinds by default), the control reproduced the
failure on the next run. It has since been re-run at the same 600-round scale as each arm of the A/B
below — review round 4 caught a 240-round version reading zero, which at this failure rate it can do
by chance — and the current artifact is unambiguous: the deprecated collective diverges at a
similar rate under the **`topk` router** — the path the multichip stage shipped — as under this
stage's. That settles attribution: the hazard is inherited, not introduced by the router change. Both
counts are rows of the generated table in §11.4.

### 11.3 The mechanism, bounded by construction

Both layer kinds run the same `_all_reduce`, and at the batch-1 decode tile that is
`ttnn.all_gather(dim=0)` onto a new leading axis plus a local `ttnn.sum`. Every device sums the same
four gathered blocks in the same order, so a cross-device difference **after** that sum can only mean
the gather delivered different bytes. The magnitudes settle what kind of difference: the worst values in the generated table below are
many orders of magnitude larger than any activation for the worst round, and small but non-zero for the rest, which is stale or unwritten peer data rather than numerics. `ttnn.all_gather` is the deprecated spelling: it takes no semaphore, and it documents that
it ignores both `topology` and `num_links`.

### 11.4 The fix, and the A/B that earned it

Two candidates, both of which give the deprecated op the synchronization it lacks:

* `AUTO_STACK_SUM_MODE = "stack_sum_async"` — the same decomposition with
  `ttnn.experimental.all_gather_async`, given two persistent gather semaphores (the op's
  `all_gather_async_device_operation.cpp:57` asserts exactly two) and a persistent **barrier**
  semaphore, built by `MultichipDecoder._ccl_semaphores` from `allocate_state` and held for the
  layer's lifetime;
* `CCL_MODE = "all_reduce"` — drop the crossover and take the maintained op at every shape.

`logs/probe_replay_divergence_ab.txt`, 150 rounds per (collective x replay pattern x layer kind),
600 rounds per spelling:

<!-- TABLE:divergence -->
| collective | replay pattern | layer kind | rounds | rounds with a cross-device difference | worst \|diff\| |
|---|---|---|---|---|---|
| `ttnn.all_reduce` (**shipped**) | back-to-back burst | full_attention | 150 | **0** | 0 |
| `ttnn.all_reduce` (**shipped**) | back-to-back burst | linear_attention | 150 | **0** | 0 |
| `ttnn.all_reduce` (**shipped**) | synchronize per replay | full_attention | 150 | **0** | 0 |
| `ttnn.all_reduce` (**shipped**) | synchronize per replay | linear_attention | 150 | **0** | 0 |
| `ttnn.all_gather`, under the multichip stage's **`topk` router** | back-to-back burst | full_attention | 150 | **2** | 1.435e+07 |
| `ttnn.all_gather`, under the multichip stage's **`topk` router** | back-to-back burst | linear_attention | 150 | **0** | 0 |
| `ttnn.all_gather`, under the multichip stage's **`topk` router** | synchronize per replay | full_attention | 150 | **1** | 9.688e-01 |
| `ttnn.all_gather`, under the multichip stage's **`topk` router** | synchronize per replay | linear_attention | 150 | **3** | 1.087e+00 |
| `ttnn.all_gather` (deprecated; multichip stage's decode default) | back-to-back burst | full_attention | 150 | **2** | 1.992e-01 |
| `ttnn.all_gather` (deprecated; multichip stage's decode default) | back-to-back burst | linear_attention | 150 | **1** | 5.469e-02 |
| `ttnn.all_gather` (deprecated; multichip stage's decode default) | synchronize per replay | full_attention | 150 | **3** | 9.688e-01 |
| `ttnn.all_gather` (deprecated; multichip stage's decode default) | synchronize per replay | linear_attention | 150 | **4** | 1.989e+19 |
| `all_gather_async` + barrier semaphore | back-to-back burst | full_attention | 150 | **0** | 0 |
| `all_gather_async` + barrier semaphore | back-to-back burst | linear_attention | 150 | **0** | 0 |
| `all_gather_async` + barrier semaphore | synchronize per replay | full_attention | 150 | **0** | 0 |
| `all_gather_async` + barrier semaphore | synchronize per replay | linear_attention | 150 | **0** | 0 |
| | | | | | |
| **`ttnn.all_gather` (deprecated; multichip stage's decode default)** | **all** | **all** | **600** | **10** (1.7 %) | |
| **`ttnn.all_gather`, under the multichip stage's **`topk` router**** | **all** | **all** | **600** | **6** (1.0 %) | |
| **`all_gather_async` + barrier semaphore** | **all** | **all** | **600** | **0** (0.0 %) | |
| **`ttnn.all_reduce` (**shipped**)** | **all** | **all** | **600** | **0** (0.0 %) | |
<!-- /TABLE:divergence -->

Of order 1 % of rounds diverge on the deprecated spelling, under both routers, against **zero in 1200**
across the two synchronized candidates. Both layer kinds, both replay patterns. A one-sided Fisher
exact test on the pooled counts in that table gives **p < 1e-4**.

### 11.5 Which fix ships, and what it costs

Both candidates are clean, so the choice is latency, and the `collective` arm of §6 settles it:

| decode collective | correctness (600 rounds) | linear decode | full decode |
|---|---|---|---|
| `ttnn.all_gather` + local sum (multichip stage) | **diverges on both routers** | 0.562 | 0.450 |
| `all_gather_async` + barrier semaphore + local sum | 0 / 600 | 0.579-0.580 | 0.467 |
| **`ttnn.all_reduce` (shipped)** | 0 / 600 | **0.565-0.566** | **0.453** |

(latency columns from §6's generated table, divergence counts from §11.4's)

`all_reduce` costs **3-4 us/step** against the op that diverges, and is 14 us/step ahead of the other
clean candidate. This stage first shipped the async gather — the obvious fix for a semaphore-free op —
and review round 2 pointed out that under the new default `all_reduce` had become the faster arm and
that this log still claimed the opposite from pre-fix numbers. The table above is the re-taken
decision, and it is a much cheaper one: 3 us/step rather than 17 for the same property.

The multichip stage's crossover is therefore **void rather than moved**. It existed only because the
deprecated spelling was 3 us/step faster below 64 rows; with that spelling withdrawn there is nothing
below the crossover that beats the stable op. `"auto"` and both stack-sum spellings stay as
measurement arms, and `test_collectives_per_forward` asserts the shipped one.

### 11.6 What is still open

The fix is verified by rate, not by root cause. What has been established is that the deprecated op
diverges at of order 1 % of rounds under sustained traced replay, over two routers, and that two
explicitly synchronized ops do not (0 in 1200), with a one-sided Fisher exact p below 1e-4. The
mechanism argument is independent of the statistics: the deprecated op takes no semaphore and
documents that it ignores `topology` and `num_links`, and the worst failure magnitude in the generated
table is stale or unwritten peer data rather than a rounding difference. What has **not** been
established is the exact failing mechanism inside `ttnn.all_gather` — that needs per-device op
timelines or a watcher-instrumented reproduction of a single failing replay, and the failure rate is
too low for the watcher runs this stage can afford (the watcher subset ran 98 tests clean). This is a
ttnn issue worth reporting upstream with `logs/probe_replay_divergence.py` attached: it is a small,
self-contained reproducer for a data-visible hazard in a shipped op.

Three consequences for later stages: the full model must not silently fall back to `"stack_sum"`
(`test_collectives_per_forward` asserts `CCL_MODE == "all_reduce"`); `test_traced_replay_does_not_leak`
— the test that caught this — should stay exactly as it is; and if a later stage wants the stack-sum
decomposition back for a shape where it wins, it must bring `logs/probe_replay_divergence.py` evidence
for the spelling it intends to use.

## 12. Review and commits

`$stage-review` ran seven rounds, each with a fresh independent subagent, and the last returned
`clean-pass` with no required work. What the earlier rounds actually changed is worth recording,
because most of it was not cosmetic:

| round | what it found | what changed |
|---|---|---|
| 1 | the divergence control covered one layer kind and one replay pattern while claiming both; the preserved "failing run" artifact contained no failure | the control was widened — and then **reproduced the divergence**, which is how the collective bug was found at all |
| 2 | the shipped collective was 14 us/step slower than an arm the stage had itself measured; OPT-009's persistent buffers were priced from the wrong row | the collective decision was re-taken: `ttnn.all_reduce` replaced the async gather, cutting the correctness fix's price from 17 to 3-4 us/step |
| 3 | the accounting declared a dispatch gap of zero that the same CSV's gap column contradicted | §5.4 rewritten as a four-term reconciliation including the gap column |
| 4 | the attribution control had drifted to 0/240 and could not support "inherited, not introduced" | re-run at 600 rounds, where it reproduces |
| 5 | four numeric claims outside the generated tables were wrong at once | the accounting, router PCC, gate parts and share tables became generated; the contract's performance block became derived |
| 6 | one `1e+26` survived; the prose guard silently scanned nothing for the source file; `_free_unless_aliased` was passed the wrong survivor for one of two tensors | run-varying figures removed from prose entirely, guard path fixed and made a hard failure, alias guard corrected |
| 7 | `clean-pass` | three sentences softened, the fastest-arm census committed as an artifact |

The through-line: every round but the last found a figure in prose that no run produced. That is why
the stage ends with two committed guards (`make_tables.py --check`, `check_prose_figures.py`), a
staleness check and a fastest-arm check, all wired into `run_evidence.sh` — the discipline is
mechanical now rather than editorial.

### Commits

| repo | branch | SHA | what |
|---|---|---|---|
| `tt-metal` | `agentic-research/hous/ornith-1.0-35B` | `4127fa65916` | the whole stage: `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, `doc/context_contract.json`, `doc/optimized_multichip_decoder/` |

Not pushed, as autonomous bringup requires. Two paths in the worktree are **not** stage-owned and are
deliberately excluded from that commit: `.agents/skills/tt-device-usage/SKILL.md` (modified) and
`.agents/fast-models-fast-feedback.md` (untracked), both of which predate this stage.
