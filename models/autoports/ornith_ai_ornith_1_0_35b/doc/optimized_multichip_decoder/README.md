# Ornith-1.0-35B — optimized multichip decoder (TTNN, 4-chip Blackhole ring)

An **in-place** optimization pass over the multichip decoder layer delivered by
`doc/multichip_decoder/`. Same file (`tt/multichip_decoder.py`), same suite
(`tests/test_multichip_decoder.py`), same public contract, same target mesh: four Blackhole `p300c`
chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4 for every dense tensor and EP=4 for
the 256 routed experts.

Two headlines.

**Performance:** warmed traced batch-1 decode falls by **7-8 %** on `linear_attention` and **9-10 %**
on `full_attention` — the §1 table has the run's exact numbers — prefill is unchanged, and no capability, contract
or correctness bar moves. That net is two opposite moves, both this stage's: the fused router gate
takes about 55 us/step off (§5.1's `router` arm), and the collective fix below puts some of it back.

The two do not add up unless the interaction is stated, so it is. The collective change costs
**3-4 us/step** measured under this stage's router (§5.1's `collective` arm, `all_reduce` against
`stack_sum-deprecated`) and about **10 us/step** measured under the inherited one (the `router topk`
arm runs the shipped collective, while the `before` row of the §1 table is `topk` plus the deprecated
gather). The two knobs interact: the router change removes an op that ran adjacent to the collective,
and the collective's cost depends on what it waits behind. Both contexts are stated because neither
alone reconciles the headline.

**Correctness:** chasing one flaky suite failure turned up a real, reproducible **cross-device
divergence in the inherited decode collective** — the deprecated `ttnn.all_gather`, which takes no
semaphore, produced a different result on one device from the others in about 1 % of sustained
traced-replay rounds — under this stage's router **and** under the router the multichip stage
shipped, so it is an inherited hazard, on both layer kinds. §4.1's generated table has the counts.
Two correct alternatives were then measured, both **0 of 600**: `ttnn.experimental.all_gather_async` with
two gather semaphores and a barrier semaphore, and the stable `ttnn.all_reduce`. **`all_reduce` is
what ships** — it is the faster of the two and it costs only 3 us/step against the op that diverges
(§4.1).

The chronological ledger — including the candidates that lost and the number that beat them — is
[`work_log.md`](work_log.md). Every table between `<!-- TABLE:... -->` markers in either document is
generated from a committed artifact by [`logs/make_tables.py`](logs/make_tables.py).

---

## 1. Result

<!-- TABLE:bench -->
| layer kind | phase | single-chip baseline | before (multichip stage) | after (this stage) | delta | speedup vs 1 chip |
|---|---|---|---|---|---|---|
| linear_attention | prefill 2048 | 101.73 ms | 28.98 ms | **29.56 ms** | +2.0% | 3.441x |
| linear_attention | decode (traced) | 1.030 ms | 0.610 ms | **0.564 ms** | -7.5% | 1.826x |
| full_attention | prefill 2048 | 95.40 ms | 28.42 ms | **28.41 ms** | -0.0% | 3.358x |
| full_attention | decode (traced) | 0.827 ms | 0.500 ms | **0.453 ms** | -9.4% | 1.826x |
<!-- /TABLE:bench -->

`before` is stage 4's shipped path and `after` is this stage's, both produced by one binary in one
sweep: `logs/bench.py --router-mode topk` reproduces the inherited decode router and the default
reproduces this stage's, so the two arms share the build, the weights, the warmup and the harness
(work log §8.1). The single-chip column is `OptimizedDecoder` on a `1x1` mesh, which is what the
speedup and the parallel efficiency are measured against.

Parallel efficiency at batch-1 decode rises from 41-42 % to **45.7 %** (`linear_attention`) and
**45.6 %** (`full_attention`), computed from the same three columns and re-derived into
`doc/context_contract.json` by `logs/make_tables.py`; without the correctness fix in §2.0 it would be
0.2-0.3 of a point higher, and
that 3 us/step difference is the price of §4.1. Prefill stays at 84-87 %; this stage found nothing to
win there and says so (§5.3).

Both claims stay gated inside the suite rather than only reported here:
`test_multichip_beats_single_chip_traced_decode` fails below `DECODE_SPEEDUP_BAR = 1.4x` (the table
above measures well over 1.8x on both layer kinds) and `test_perf_prefill` fails above `PREFILL_MS_BAR = 48 ms`.

---

## 2. What changed

One correctness change, one performance change, one layout change that makes the second pay.

### 2.0 The decode collective (correctness)

`CCL_MODE = "all_reduce"`. The multichip stage spelled the two per-layer collectives with a crossover:
`ttnn.all_reduce` above 64 physical rows and, below it — which is every batch-1 decode step —
`ttnn.all_gather` onto a new leading axis plus a local `ttnn.sum`, because that was 3 us/step faster
at the decode tile.

`ttnn.all_gather` is the deprecated op. It takes neither a semaphore nor a topology argument, and
§4.1 shows it producing a different result on one device from the others under sustained traced
replay, at of order 1 % of rounds and with the worst rounds far outside the range of any activation in the layer and the rest small but
non-zero — and a correct all-reduce is bitwise identical on every device by construction, so any
non-zero difference is a fault. With that spelling removed, the crossover
has nothing left to select below 64 rows that is faster than the stable op, so this stage takes
`ttnn.all_reduce` at every shape. It is clean over 600 rounds, and it costs **3 us/step** against the
op that diverges — a much better trade than the other clean candidate,
`ttnn.experimental.all_gather_async` with a barrier semaphore, which is correct but 14 us/step slower
than `all_reduce` at this shape. `"auto"` and both stack-sum spellings are kept as measurement arms.

### 2.1 The router gate (performance)

`ROUTER_MODE = "fused_gate"`. At **decode**, the router's `ttnn.topk(k=8)` over 256 experts plus the
`ttnn.softmax` over the kept 8 are replaced by a single
`ttnn.experimental.deepseek.moe.generalized_moe_gate` call: score, top-k and softmax-over-selected in
one kernel, over a 16x16 expert face, one token per core, writing into **preallocated** output
buffers built once in `MultichipDecoder.allocate_state`.

Why it was worth looking: in stage 4's decode profile `TopKDeviceOperation` is **48.4 us/step on one
core** — 9.5 % of the window, the largest single non-sparse op — and the whole routing block is
~17 %. The optimized stage had rejected this op as "bfloat16-only" and, in its own words, *never
timed it*. Timed, the kernel costs about **2 us on 32 cores** in the decode capture, against the 55 us of device
time in the two ops it replaces; the untraced probe's op-by-op breakdown is the `gateparts` table in
[`work_log.md` §3](work_log.md#3-what-the-router-chain-actually-costs).

Prefill deliberately keeps the `topk` chain: the op is one token per core, a 2048-token chunk would
need 19 sequential calls, and `TopK` is a fifth of a percent of the prefill window or less (§5.2).
`test_decode_runs_the_fused_router_gate` pins both halves in the runtime path.

### 2.2 It is a router **precision** change, and it is gated on selection, not on PCC alone

The kernel reads bfloat16 logits where the chain read float32. Three earlier stages defended float32
router logits on the ground that expert selection is a discrete decision — a rounding change swaps an
expert rather than perturbing a value — so the change is gated on exactly that quantity, on real
checkpoint weights, in `tests/test_multichip_decoder.py::test_router_modes_agree`:

<!-- TABLE:routerpcc -->
| layer kind | candidate | identical top-8 set | `topk` vs HF golden | candidate vs HF golden | candidate vs `topk` |
|---|---|---|---|---|---|
| full_attention | `fused_gate` | **8/8** steps | 0.999888 – 0.999938 | 0.999887 – 0.999939 | 0.999999 – 1.000000 |
| full_attention | `fused_gate_local` | **8/8** steps | 0.999888 – 0.999938 | 0.999887 – 0.999939 | 0.999999 – 1.000000 |
| linear_attention | `fused_gate` | **8/8** steps | 0.999731 – 0.999987 | 0.999731 – 0.999987 | 1.000000 – 1.000000 |
| linear_attention | `fused_gate_local` | **8/8** steps | 0.999731 – 0.999987 | 0.999731 – 0.999987 | 1.000000 – 1.000000 |
<!-- /TABLE:routerpcc -->

The routing decision is identical on every measured step, for both candidate fused modes and both
layer kinds, and the golden PCC matches the float32 chain's to five or six decimal places. The matmul that produces the logits keeps HiFi4
and float32 accumulation; only the stored logits change dtype.

Two further samples, from `logs/probe_gate.txt` on random-normal logits rather than checkpoint ones,
which is the adversarial direction for a selection question:

* at 32 real rows the fused gate agrees with the float32 chain on **32/32** token sets (PCC 0.999986);
* the same 32 rows put through `ttnn.topk` on **bfloat16** logits agree on only **31/32** (PCC
  0.999189). So bfloat16 logits *can* move a selection — one token in 32 on random-normal logits —
  and on this evidence the fused kernel's ranking tracks the float32 chain more closely than a
  bfloat16 `topk` does. The mechanism is not established here, only the measurement, so the claim
  this stage makes is the narrow one: 16 real-weight decode steps and 32 random-logit tokens, all
  with identical selected sets.

### 2.3 The layout that makes it pay

The first working version of the fused path was worth **10 us/step, not 50**. Its decode profile
showed the gate op at 2 us and the conversion around it adding two `ReshapeViewDeviceOperation` rows
at **13.8 us each** plus three `UntilizeWithUnpadding` rows inside `ttnn.scatter`. Both are the same
mistake — a row-major-shaped problem handed to tiled tensors:

* the `[rows, 1, k] -> [1, 1, rows, k]` reshape the scatter's operand contract needs is a metadata
  **view** in ROW_MAJOR and a `rows`-tile gather in TILE;
* `ttnn.scatter` converts every non-ROW_MAJOR operand to ROW_MAJOR itself (`scatter.cpp:164/193`) and
  converts the result back to the base's layout (`:233`), so tiled operands buy conversions the
  caller can simply not create.

Taking the gate's two outputs to ROW_MAJOR before the slice moved the layer from 0.600/0.490 ms to
0.560/0.452 ms; keeping the scatter base as a persistent ROW_MAJOR buffer took it to 0.559/0.450 ms.
(All three pairs predate the collective fix of §2.0, which then added 3 us/step to reach the shipped
0.565/0.453 of §1; the `router` and `collective` arms of §5.1 separate the two.) (That second step is worth 1-2 us, at the edge of the harness's resolution; it
is kept because it removes two layout conversions, not because the number is large.) This is the
stage's real finding: the fused kernel was the easy part.

**Provenance note.** Only the final pair is in a committed artifact (`logs/ab_before_after.txt`); the
two intermediate pairs are readings taken while iterating and are quoted as such. They are the
`bench.py` numbers for intermediate working states that were not kept, so there is nothing to commit
for them, and no conclusion rests on them — the conclusion rests on the `router` arm of §5.1, which
is committed.

---

## 3. Operation-topology audit

The full table — 21 operation groups with their before-cost, whether they repeat a same-input matmul,
whether they carry a collective, what layout conversions they pay, the candidate considered and the
action taken — is [`work_log.md` §2](work_log.md#2-operation-topology-audit). Its three conclusions:

1. **No unpacked repeated same-input projection remains.** q/k/v/gate, the gated-DeltaNet
   in-projection, the shared expert's gate/up/router column and the routed experts' gate/up are each
   already one matmul, packed at load time by the fused stage. OPT-001 and OPT-010 are
   inherited-satisfied and the perf report confirms it in the rows.
2. **Exactly two collectives per layer**, both on the residual-shaped `[b, t, 2048]` tensor — the
   smallest either half can be reduced on — and both required by the replicated residual contract.
   There is no collective to remove; §4 is about how they are spelled.
3. **The router was the one large op no stage had attacked**, and it is where this stage's win came
   from.

---

## 4. Multi-device families, measured as families

### 4.1 The decode collective: a correctness A/B, then a price

`logs/probe_replay_divergence.py` reproduces `test_traced_replay_does_not_leak`'s cross-device
bitwise check at pressure: many rounds of many traced replays, both layer kinds, both replay patterns
(a back-to-back non-blocking burst, and the synchronize-after-every-replay pattern the suite test
uses), reporting the worst per-round cross-device difference rather than a boolean. 150 rounds per
cell, 600 rounds per collective spelling:

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

Every device sums the same four gathered blocks in the same order, so a cross-device difference after
that sum means the **gather** delivered different bytes. The magnitudes settle what kind of failure it is. A correct all-reduce is bitwise identical on every
device by construction, so any non-zero value is a fault; the worst round in the table is many orders of magnitude
larger than any activation in the layer, and the rest are small but non-zero. That is stale or unwritten peer data, not a rounding difference.
The deprecated op has no semaphore to prevent that. Both alternatives do — `all_gather_async` takes
two gather semaphores plus a barrier semaphore, and `ttnn.all_reduce` is the maintained op — and both
are clean.

So the decision is between the two clean arms, and the `collective` arm of §5.1 settles it at the
layer:

| decode collective | correctness (600 rounds) | linear decode | full decode | shipped |
|---|---|---|---|---|
| `ttnn.all_gather` + local sum (multichip stage) | **diverges**, on both routers | 0.562 | 0.450 | no |
| `all_gather_async` + barrier semaphore + local sum | 0 / 600 | 0.579–0.580 | 0.467 | no |
| `ttnn.all_reduce` | 0 / 600 | **0.565–0.566** | **0.453** | **yes** |

(latencies from §5.1's `collective` arm, divergence counts from the table above)

`all_reduce` costs 3-4 us/step against the diverging op and is 14 us/step ahead of the other clean
candidate. An earlier iteration of this stage shipped the async gather, before the `all_reduce` arm
was measured under the new default; review round 2 caught that, and this table is the re-taken
decision. A decode step that is occasionally wrong on one device is not a faster decode step, but
3 us/step is a much smaller price than the 17 the async gather would have charged for the same
property.

### 4.1b Persistent / preallocated collective buffers (OPT-009)

The shipped decode collective is `ttnn.all_reduce` (§4.1), which **exposes no persistent-buffer
argument** — it is the maintained non-async op and allocates its own intermediates. So OPT-009's
question here is really about the async family, the only one that takes the argument, and that family
was measured in full: `logs/probe_ccl_persistent.py` runs
`ttnn.experimental.all_gather_async` with and without an explicit `persistent_output_buffer` and over
its `chunks_per_sync` / `num_workers_per_link` / `num_buffers_per_channel` knobs, at both operand
dtypes, traced.

<!-- TABLE:cclpersdecode -->
| dtype | `ttnn.all_gather` (deprecated) | `all_gather_async` | + persistent buffer | + best tuned |
|---|---|---|---|---|
| bfloat16 | **15.83 us** | 21.31 | 18.87 | 18.67 |
| bfloat8_b | **14.83 us** | 19.88 | 18.57 | 18.47 |
<!-- /TABLE:cclpersdecode -->

Reading, at the decode tile:

* persistence is worth ~11 % *within* the async family (see the table's bfloat16 row), so the
  argument does what OPT-009 expects it to do;
* the whole async family is nonetheless ~18-25 % behind the deprecated `ttnn.all_gather` on latency,
  and the layer A/B in §4.1 puts even the best of it 14 us/step behind `ttnn.all_reduce`;
* so the persistent-buffer candidate is **rejected on measurement, twice over** — it loses on latency
  and the op it belongs to loses the correctness-plus-latency comparison that actually chose the
  shipped collective.

The first call to the async op was refused for passing one semaphore where the op asserts two
(`all_gather_async_device_operation.cpp:57`). That is a call-shape bug, not a property of the op, and
it was fixed before the arm was judged — the same mistake stage 4 found itself making with
`all_reduce_async`.

Per-row record for the shipped decode CCL, as OPT-009 asks: payload dtype bfloat16 (token mixer) and
bfloat8_b (MoE); input and output memory config DRAM interleaved; `ttnn.all_reduce` over the whole
1x4 mesh with `topology=Ring` under `FABRIC_1D_RING` and `num_links=2`; no persistent output or
intermediate buffer, because the op takes neither; `chunks_per_sync` / `num_workers_per_link` /
`num_buffers_per_channel` not exposed by this op and swept on the family that does expose them. Full
sweep, including the larger shapes: [`logs/probe_ccl_persistent.txt`](logs/probe_ccl_persistent.txt),
work log §5.

### 4.2 Residual layout, collective placement, fused CCL+matmul

Stage 4 measured this family at both material boundaries at the real per-device shapes, traced,
including the **consuming** boundary rather than an immediate restore: a fused
matmul+reduce-scatter producer without the gather does beat the shipped arm at the two larger shapes,
and the column-parallel `attn_in` consumer that would then have to all-gather its input costs more
than the producer saves at every shape. This stage changes neither the projections nor the residual
contract those rows describe, so it inherits that rejection explicitly (work log §6.3) and writes the
contract down (§7) instead of re-deriving it.

### 4.3 Activation / CCL dtype

`CCL_CAST_BLOCKFLOAT` re-measured under the new default: casting the MoE's bfloat8_b collective
operand up to bfloat16 still costs 3 us/step on both layer kinds and still moves prefill by nothing,
which keeps stage 4's classification — the block-float collective's device time is a barrier
absorbing device skew, not payload.

### 4.4 Fidelity advisories on the two float32 rows

`tt-perf-report` advises HiFi2 on the router matmul and on the DeltaNet recurrent-state matmuls,
many times per capture. Both were implemented as knobs and A/B'd at the layer rather than argued
about (§5.1, `router_fidelity` and `state_fidelity` arms), and both are **ties** inside the harness's
0.001 ms resolution. The inherited HiFi4 ships for both: a tie is not a reason to weaken a discrete
routing decision or an fp32 recurrent state. The knobs and their rows stay so the next stage inherits
the measurement rather than the advisory.

### 4.5 Decode residual memory config (OPT-003)

OPT-003 says a decode residual should not be sitting in DRAM because it is convenient, and the decode
captures put a tenth to a sixth of the window in `BinaryNg` and about a fifth in layout (§5.2's
generated share table) with several rows marked
`in0:dram_interleaved` on tensors that are one 32-row tile. `DECODE_RESIDUAL_MEMORY` puts both
residual adds in L1; the `residual` arm of §5.1 measures it.

It is a **tie** — inside the 0.001 ms spread on both layer kinds. The step is launch-bound, and
moving a 4 KiB tensor's home does not change how many ops there are. The inherited spelling ships,
because a tie is not a reason to change anything, and the knob stays with its measurement so the next
stage does not re-run the experiment blind.

Scope note: this arm moves the residual adds to L1 **interleaved**. OPT-003's named candidate is a
width-sharded L1 residual carried through both norms with a sharded norm program config, and that
part is inherited rather than re-derived — stage 3 selected the sharded-norm grid
(`doc/optimized_decoder/logs/ab_norm_shard_cores.txt`) and this stage's captures show both norms as
`LayerNormDeviceOperation (in0:width_sharded)`. What was untested was the residual *add* boundary
between them, and that is what this arm answers.

### 4.6 Projection packing, geometry, sparse cores

All three re-verified under the new default rather than assumed: the multichip-retuned dense decode
geometry is 18 / 12-13 us/step ahead of the single-chip table, and the tp-rescaled routed-sparse core
rule is 4-5 ms/layer ahead at prefill. Table in §5.1.

---

## 5. Performance

### 5.1 Whole-layer A/B, `logs/ab_layer_knobs.txt`

Each arm built fresh in the same process on the same device with the same real weights, three builds
per arm so the spread sits next to the difference. All values in ms.

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

### 5.2 `tt-perf-report`

Four captures (`tracy/{linear,full}_attention/{prefill,decode}_perf_report.*` — advice-enabled table,
machine-readable CSV, roofline summary and a stacked-by-op-code CSV/PNG). Prefill and decode are
captured in separate runs so each holds one signposted window and one device session, and the routed
matmul rows are modelled with `--active-experts` at the measured per-device per-group count (4 at
batch-1 decode, 41 for a 32-token prefill group).

Merged 4-device device time in each window, so the shares below have an absolute scale:

<!-- TABLE:window -->
| capture | merged device time in the window | replays | per step / call |
|---|---|---|---|
| linear_attention decode | 15232 us | 32 | **476.0 us/step** |
| linear_attention prefill | 30204 us | 1 | **30204.4 us/call** |
| full_attention decode | 14458 us | 32 | **451.8 us/step** |
| full_attention prefill | 29973 us | 1 | **29973.2 us/call** |
<!-- /TABLE:window -->

Share of that window, by op group:

<!-- TABLE:perf -->
| | linear decode | full decode | linear prefill | full prefill |
|---|---|---|---|---|
| `SparseMatmul` (routed experts) | 19.99% | 24.46% | 74.96% | 73.97% |
| `GeneralizedMoeGate` (fused router) | 0.50% | 0.53% | 0.00% | 0.00% |
| `TopK` (router, prefill only) | 0.00% | 0.00% | 0.17% | 0.18% |
| dense `Matmul` (all in0 layouts) | 14.82% | 10.67% | 1.10% | 1.07% |
| **collectives (`AllGather` / `AllGatherAsync` / `ReduceScatter`)** | 10.25% | 13.32% | 3.04% | 5.75% |
| all data movement (`DM` category) | 12.50% | 15.75% | 3.31% | 5.75% |
| all layout (`TM` category) | 21.49% | 18.36% | 4.73% | 4.23% |
<!-- /TABLE:perf -->

What it says:

* **`TopKDeviceOperation` is gone from the decode captures.** `GeneralizedMoeGateDeviceOperation`
  replaces it at 2 us on 32 cores. `TopK` remains in the prefill captures, deliberately (§2.1).
* **The routed experts still dominate**, and they sit at the geometry stage 4's isolated ladder
  (`doc/multichip_decoder/logs/probe_sparse_matmul_local.txt`) picked as the fastest available at
  this active-expert count: at `active=4`, 8 cores with `in0_block_w=32` and `out_block_w=4` measured
  62.5 us against 70.1 for the best 32-core candidate. The `SLOW` flag on those rows is the op's own
  advisory, not an unexplored candidate.
* **The collectives row is now `ReduceScatter` + `AllGather`** — what `ttnn.all_reduce` lowers to —
  and it grows in both share and absolute device time. Both halves of that comparison, and the two
  groups §4.5's OPT-003 arm is about, derived the same way from the two stages' stacked CSVs:

<!-- TABLE:shares -->
| group | linear decode, stage 4 | linear decode, this stage | full decode, stage 4 | full decode, this stage |
|---|---|---|---|---|
| collectives (`AllGather` / `AllGatherAsync` / `ReduceScatter`) | 35.0 us/step (6.5 %) | 48.8 us/step (10.2 %) | 46.4 us/step (9.1 %) | 60.2 us/step (13.3 %) |
| `BinaryNg` (elementwise) | 79.0 us/step (14.7 %) | 78.4 us/step (16.5 %) | 49.1 us/step (9.6 %) | 49.1 us/step (10.9 %) |
| `TM` category (layout) | 119.8 us/step (22.3 %) | 102.3 us/step (21.5 %) | 100.0 us/step (19.6 %) | 83.0 us/step (18.4 %) |
<!-- /TABLE:shares -->

  Part of the share rise is the denominator — the window is about 60 us shorter — but the absolute
  time rises too, by much more than the 3-4 us/step the change costs end to end (§4.1). That gap is
  the same skew/barrier effect §4.3 and stage 4's §5.8 both describe: this collective absorbs
  whatever the four devices did not finish together, so its device time is an upper bound on
  communication cost, not a measurement of it.
* **Decode is launch- and latency-bound, not bandwidth-bound** — see the accounting in §5.4. This is
  why a 55 us op removal is worth 10 % of the window while a dtype change that halves weight bytes
  (§6.1) is worth 0.7 %.

### 5.2b `tt-perf-report` advice, item by item

`$optimize` asks for every actionable recommendation to be tried or rejected with a reason, so the
advice multiset of this stage's four captures is compared against stage 4's rather than left implicit.

| advisory | where it fires now | status |
|---|---|---|
| `place input 0 in L1` | **gone from decode.** Stage 4's decode captures carried 32 of these; this stage's carry none. It still fires 5x in each *prefill* capture, unchanged from stage 4 and out of this decode-focused change's path. | resolved at decode as a side effect of the ROW_MAJOR gate/scatter path (§2.3) — the operands it flagged now live in L1 |
| `Output subblock 1x1 is small` on the dense decode projections — `Matmul 32 x 2048 x 2560` (`attn_in`) on `full_attention`, `32 x 2048 x 3136` (`gdn_in`) on `linear_attention`, plus `shared_in`, the router and `expert_select` — and, in the **prefill** captures, on the dominant `active=41` routed `SparseMatmul` | all four captures | **inherited and answered.** `per_core_N = 1` is forced by the winning point of stage 4's 9-target x 6-cap ladder (`doc/multichip_decoder/logs/probe_dense_matmul.txt`): at 110 realised cores the 80 output tiles give one tile per core. A larger subblock needs fewer cores, which that ladder measured as slower. Re-verified at the layer here by the `geometry` arm of §5.1 |
| `Try a DRAM-sharded program config` on `Matmul 32 x 2048 x 3136` (`gdn_in`, the largest `linear_attention` dense row) | the `linear_attention` decode capture only; it does not fire on `full_attention` | **inherited and answered, with one caveat stated.** `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` pins its compute grid to the 8 DRAM banks, and the optimized stage measured it losing to an explicit 1D `mcast_in0` config on **every** dense decode role even without counting its activation-reshard cost (`doc/context_contract.json` → `optimized_decoder.named_limitations[2]`). The caveat: that measurement is at single-chip widths, and no DRAM-sharded candidate exists at the per-device multichip widths in any artifact. What argues against re-running it here is the row itself — it reports about two thirds of DRAM bandwidth, and a DRAM-sharded rewrite is for rows that are DRAM-bound *and under-utilised* |
| `in0_block_w = 2` on the float32 DeltaNet state matmuls | `linear_attention` decode | **inherited and answered.** `doc/optimized_decoder/logs/probe_decode_micro.txt` `STATE` rows: every `in0_block_w > 1` candidate on the `outer` role fails the op's own `TT_FATAL`, and the `read` role is flat across the legal values |
| `SLOW` on essentially every `Matmul` and `SparseMatmul` row | all four captures | **inherited and answered.** The flag is the op's own bandwidth heuristic on a step that is launch-bound at ~7 % of the DRAM roofline (§5.4), so it fires on nearly everything. Each dominant role's geometry is the winner of a measured ladder: the routed sparse matmuls from stage 4's `probe_sparse_matmul_local.txt` at this active-expert count (which covers `out_block_w`/`sub_w` and core count, at prefill's `active=41` as well as decode's), the dense projections from `probe_dense_matmul.txt`, both re-verified at the layer by §5.1's `sparse` and `geometry` arms. The prefill answer expires if a later stage changes the MoE group size |
| `Output subblock 1x1 is small` also on `Matmul 32 x 2048 x 256` (router) and `32 x 256 x 64` (`expert_select`) | both decode captures | **same answer as the two rows above**, and the same ladder: both are 8-core and 2-core roles at `per_core_N = 1` by construction, and the 9-target x 6-cap sweep in `doc/multichip_decoder/logs/probe_dense_matmul.txt` covers them. `expert_select` is 3 us/step and is deleted entirely by the `fused_gate_local` arm, which was measured and lost on other grounds (work log §4.5) |
| `HiFi2 may also work and has 2x the throughput of HiFi4` on the router matmul, and on `expert_select` | both decode captures | **tried this stage, and it is a tie.** The advisory has a point the stage agrees with — the fused gate rounds the logits to bfloat16, so HiFi4 + float32 accumulate is buying precision the consumer discards — so `ROUTER_DECODE_FIDELITY` was implemented and A/B'd (`router_fidelity` arm, §5.1): 0.565 vs 0.565 and 0.453 vs 0.453-0.454. The inherited HiFi4 is kept because a tie is not a reason to weaken a discrete decision |
| `HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4` on the DeltaNet state matmuls | `linear_attention` decode only | **tried this stage, and it is a tie** (`state_fidelity` arm, §5.1: 0.565-0.566 both arms), because ~5 us x 2 per step is a larger row than the router advisory that was tried. The inherited HiFi4 is kept. Separately, the advisory **mis-fires** and is reported as a `tt-perf-report` improvement candidate: those rows are `HiFi4 FP32 x FP32 => FP32`, so the sentence names a dtype the row does not use |
| `Use HiFi2 or HiFi4 with BF16 activations for improved accuracy` on the routed gate/up `SparseMatmul`, and `If your matmuls are not FLOP-bound use HiFi4 with BF16 activations for full accuracy` on the BFP8 dense rows | all four captures | **rejected by construction**: both ask for *higher* fidelity than the selected policy, which is the direction `$datatype-sweep` owns and the opposite of what this launch-bound step needs. The policy they question is the inherited one, validated on real-weight PCC by the suite |
| `High Op-to-Op Gap` | both decode captures; far more rows on `linear_attention` than on `full_attention` | **inherited, and it is the profiler's own dispatch overhead** — the identical advisory fires on stage 4's captures. §5.4 reconciles it: the same op durations un-profiled leave a 1.3 us residual on `full_attention`, which 72 us of real gap would not allow |

Two things changed in this stage's captures: one advisory class disappeared (`place input 0 in L1`), and one — the router HiFi2 item — was tried rather than inherited. Nothing else is new.

### 5.3 Prefill is unchanged, deliberately

Prefill is about three quarters routed `SparseMatmul` and a fraction of a percent `TopK` (§5.2). The stage's change is decode-only by
construction, and the `sparse`, `geometry`, `collective` and `cast` arms of §5.1 re-confirm that stage 4's
prefill choices are still the right ones. No prefill claim is made in either direction. The §1 table's two prefill rows
move by a few tenths of a percent to a couple of percent between regenerations of this stage's own
evidence, in both directions and on both layer kinds, which is smaller than the spread §5.1's
`ablayer` prefill columns show across arms that cannot touch prefill at all (over a millisecond).
Both inside the build-to-build spread visible in the `ablayer` prefill columns, where arms that
cannot touch prefill at all span more than a millisecond.

### 5.4 Performance accounting

Four numbers, `full_attention`, batch 1, one warmed traced decode step, with `linear_attention` in
brackets:

<!-- TABLE:accounting -->
| term | full_attention | linear_attention | source |
|---|---|---|---|
| device time | 451.8 us/step | 476.0 us/step | merged `PERF_DECODE` window / 32 replays |
| op-to-op gap, profiled | 72.2 us/step | 122.6 us/step | same CSV's `Op-to-Op Gap` column, excluding the 2 / 3 gaps above 100 us (the window boundaries) |
| device + gap | 524.0 us/step | 598.6 us/step | the two rows above |
| end-to-end, profiled | 500 us/step | 587 us/step | the same capture's own wall clock |
| end-to-end, un-profiled | 453 us/step | 564 us/step | `logs/ab_before_after.txt`, `after` arm |
| residual, un-profiled | 1.2 us/step | 88.0 us/step | un-profiled end-to-end minus device time |
<!-- /TABLE:accounting -->

plus the theoretical roofline, **~33 us/step**: 16.9 MB moved per device per step / 512 GB/s. Weights
at their stored dtypes — `attn_in` 5.57 MB and `o_proj` 2.23 MB (bfloat8_b), `shared_in` 0.63 MB and
`shared_down` 0.28 MB (bfloat8_b), router 1.05 MB (bfloat16), 4 active routed experts x 1.77 MB
(bfloat4_b) — plus 0.07 MB of paged KV at this context. The 512 GB/s peak is `tt-perf-report`'s own
basis: it reports the 237 GB/s `o_proj` row as 46.3 % of DRAM.

Reconciling them, all figures from the generated table above:

* **Inside the profiled run the terms close.** Device + gap runs 2-5 % *over* the profiled wall clock,
  which is the merged report over-counting by construction: it attributes one representative device
  per op, while gaps on the other three overlap.
* **Most of that gap is the profiler, not the path.** The same op durations un-profiled leave a
  ~1 us residual on `full_attention` — which 72 us of real per-step gap would not allow. Tracy
  instruments host dispatch, and that is what the gap column is mostly measuring.
  `tt-perf-report`'s own `High Op-to-Op Gap` advisory fires on these captures for the same reason it
  fires on stage 4's.
* **`linear_attention` keeps an 88 us residual and `full_attention` does not.** Both layer kinds
  replay one captured trace through the same mechanism with a similar op count (98 against 83), so a
  generic per-op launch cost does not explain a term that appears on one and not the other. It is
  **inherited and unchanged in kind** — stage 4's same pair is 538.1 us of device time against a
  610 us wall, a 72 us residual — and it is the largest single unexplained term in the stage. Two
  candidates remain and neither can be settled from the committed artifacts: per-device skew that
  the merged report cannot show (it keeps one device per op), and a real per-op gap specific to the
  float32 DeltaNet recurrent-state chain. Settling it needs per-device op timelines, the same open
  item stage 4 recorded as its limitation 9. Named, not dismissed — and **not** claimed to be zero.
* **The roofline fraction is ~7 %, and the reason is op count rather than bandwidth**: 83 device ops
  per step on `full_attention` and 98 on `linear_attention`, at a few microseconds each, on a step
  whose largest activation is one 32-row tile. That is why this stage's win came from deleting a
  48 us op rather than from moving fewer bytes, and why the BFP4 projection policy — which halves
  the weight bytes of four matmuls — is worth 0 to 0.7 % (§6.1).

No `perf_summary.json` is written: `$optimize` asks for it when optimizing a complete model or a
serving path, and this stage is a single decoder layer with no LM head, sampling or token feedback in
scope.

### 5.5 Batch

Batch-1 single-user latency is the optimized target, and larger batches are preserved rather than
traded: the suite runs prefill+decode PCC at batches 1, 4, 13 and 32 against the HF golden, ragged
per-user positions at batch 4, and the advertised bound of 32 unchanged. The fused gate is one token
per core and the layer's decode row count is `align_up(batch, 32)`, so it covers every batch up to
110 rows; above that `routing_weights` falls back to the `topk` chain, which is correct and slower.

---

## 6. Precision

### 6.1 BFP4 dense projections, re-measured in this topology (OPT-007)

The optimized stage measured `POLICIES["bfp4-projections"]` on real weights, found every PCC row
clear of the 0.995 bar, found it ~2 % faster on one chip, and rejected it on a compounding judgement,
shipping it as a named policy for `$datatype-sweep`. OPT-007 asks for that decision to be re-taken on
the topology actually shipped, because TP=4 makes the per-device projections four times narrower.

Re-measured (`policy` arm in §5.1): a **tie** on `linear_attention` (0.565 both arms) and **−3 us,
−0.7 %** on `full_attention` (0.453-0.454 → 0.450-0.451). Sharding has made the candidate less attractive, not more — the projections it
shrinks are already quartered, and the step is launch-bound. Same decision, now on this stage's own
numbers, with the policy still available by name.

### 6.2 The policy is verified in the measured rows (OPT-013)

The decode capture shows `HiFi2 BF16 x BFP8 => BF16` on `attn_in`, `o_proj`, `shared_in` and
`shared_down`; `HiFi4 BF16 x BF16 => BF16` on the router — the `=> BF16` output is this stage's
change and it is visible in the row; and `LoFi BF16 x BFP4 => BFP8` on the routed gate/up
`SparseMatmul` and `LoFi BFP8 x BFP4 => BFP8` on the routed down one, which consumes the block-float
expert activation. Claimed policy equals measured policy.

Everything else is inherited unchanged: bfloat8_b paged KV cache, bfloat4_b/LoFi routed experts,
bfloat8_b/HiFi2 dense projections, bfloat16/HiFi4/float32-accumulate router weight, bfloat16 norms
and float32 DeltaNet state.

---

## 7. The inter-layer residual layout contract

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

**The final path has no inter-layer collective at all.** Both of the layer's collectives sit *inside*
it, immediately before each residual add, so a layer consumes a replicated residual and produces a
replicated residual and a stack of them needs no boundary conversion.
`test_output_is_identical_on_every_device` asserts bitwise equality across the mesh, so "replicated"
is a checked property.

Full-model bringup should preserve this rather than rediscover it, and in particular should **not**
convert the boundary to a sharded or fractured residual without re-running
`doc/multichip_decoder/logs/probe_fused_ccl.py`: that family was measured at both boundaries and is
net worse at every shape once the consuming column-parallel projection has to all-gather its input.
The replicated residual is also what keeps both RMSNorms exact and local — no distributed-norm
statistics all-gather — and what makes the full-width activation available to expert parallelism.
Work log §9 has the rest of the boundary (page table, positions, cache and state distribution).

---

## 8. Known limitations

1. **Batch-1 decode reaches ~7 % of the DRAM roofline, and the gap is op count.** 83 device ops per
   step on `full_attention` and 98 on `linear_attention`, at a few microseconds each; the largest
   activation is one 32-row tile. Every large op in the
   window has been swept (routed sparse geometry, dense geometry, SDPA config, collective spelling)
   and the remaining cost is per-op launch. Closing it needs op-level fusion that does not exist in
   TTNN today for this graph — the two candidates that would have helped are recorded as measured
   rejections rather than as untried ideas (work log §10).
2. **Watcher does not cover the ACTIVE_ETH cores on this configuration.** Inherited hard tool limit;
   the watcher run sets `TT_METAL_WATCHER_DISABLE_ETH=1` and every worker-core assert stays armed.
3. **The cross-device divergence is fixed by rate plus a mechanism argument, not by root cause.**
   §4.1's generated table shows the deprecated `ttnn.all_gather` diverging in of order 1 % of sustained
   traced-replay rounds under this stage's router and, at a similar rate, under the multichip
   stage's — so the hazard is inherited, not introduced here — against 0 of 600 for each of the two
   synchronized candidates, including the shipped `ttnn.all_reduce`. A one-sided Fisher exact test on
   the pooled counts (see the table) gives p < 1e-4, so on this sample the difference is not chance.
   What is **not** established is the exact mechanism inside `ttnn.all_gather`;
   that needs per-device op timelines or a watcher-instrumented single failing replay, and at the
   per-round rate in §4.1's table that is beyond what this stage's watcher budget can catch. Worth reporting upstream —
   `logs/probe_replay_divergence.py` is a small self-contained reproducer for a data-visible hazard
   in a shipped op. Ledger: work log §11.

4. **`generalized_moe_gate` is decode-only here.** One token per core makes it the wrong shape for a
   2048-token prefill chunk (19 sequential calls), and `TopK` is a fraction of a percent of that
   window (§5.2), so prefill
   keeps the float32 chain. Not a blocker; a note for anyone who expects one router implementation.
5. **`ttnn.topk` has a floor, not a slope, below width 256.** The `TOPKW` ladder in
   `logs/probe_gate.txt` — generated as a table in work log §3.1 — shows width 32 and width 64 each
   costing most of what width 256 costs, which is what kills the exact two-stage chunked decomposition — the two stages
   sum to more than the single 256-wide call before the relayout is paid for. Worth reporting
   upstream: an op whose 32-wide call costs most of what its 256-wide call costs is not paying for
   the search.
6. **The one-hot narrowing matmul rounds float32 activations to bfloat16** regardless of weight dtype
   or compute-kernel config (measured `maxdiff 1.9e-3`), which is what blocks the threshold rewrite
   in work log §3.2. Also worth reporting upstream.
7. **One mesh shape.** `DEFAULT_MESH_SHAPE = (1, 4)` targets this host, as the goal directs.
8. **Long-context PCC is validated to 8000 tokens**, inherited and unchanged: the eager HF
   full-attention reference is not tractable on host beyond that. Above it the mesh path is
   cross-checked against itself under a different internal chunking, and the full advertised 262144
   is validated for shape, finiteness and non-degeneracy.

---

## 9. Exact artifacts

```
doc/optimized_multichip_decoder/
├── README.md                                  this file
├── work_log.md                                what was done, in order, with measurements
├── logs/
│   ├── run_evidence.sh                        regenerates everything below, in order
│   ├── make_tables.py                         generates every table in these two documents
│   ├── check_prose_figures.py                 and this checks every latency quoted OUTSIDE a table
│   ├── bench.py                               warmed prefill / traced decode, with --router-mode
│   ├── ab_before_after.txt                    §1 — before / after / single-chip, one sweep
│   ├── ab_layer_knobs.py / .txt               §5.1 — every knob, three builds per arm
│   ├── ab_best_arms.txt                       the fastest arm of each knob, and whether it ships
│   ├── probe_gate.py / .txt                   work log §3 — every router-gate spelling, plus the
│   │                                          ttnn.topk width ladder below 256
│   ├── probe_ccl_persistent.py / .txt         §4.1 — OPT-009 persistent-buffer collectives
│   ├── probe_replay_divergence.py / .txt      §4.1 — the attribution control (deprecated op pinned)
│   ├── probe_replay_divergence_ab.txt         §4.1 — the deprecated-vs-async A/B, 600 rounds each
│   ├── probe_replay_divergence_allreduce.txt  §4.1 — the shipped `ttnn.all_reduce`, 600 rounds
│   ├── traced_replay_divergence_failure.txt   §8 limitation 3 — what survives of the failing run
│   ├── bench_{before,after,single_chip}.json  the §1 rows as JSON, written by bench.py
│   └── pytest_full_suite.txt.gz               the shipped suite on the final default path
├── tracy/
│   ├── run_profiling.sh                       4 captures, prefill and decode in separate runs
│   ├── linear_attention/                      {prefill,decode}_perf_report.{txt,csv.gz,summary,...}
│   └── full_attention/                        same
└── watcher/
    ├── watcher_pytest.txt.gz                  TT_METAL_WATCHER=10, separate run
    └── watcher_error_count.txt                count of watcher error/assert/hang lines
```

Reproduce everything:

```bash
bash models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_multichip_decoder/logs/run_evidence.sh
```

Commit SHAs for this stage are recorded at the end of [`work_log.md`](work_log.md).
