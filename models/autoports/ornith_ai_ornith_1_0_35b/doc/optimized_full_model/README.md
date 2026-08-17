# Ornith-1.0-35B — optimized full model (TTNN, 4-chip Blackhole ring)

An **in-place** optimization pass over the full model delivered by [`doc/full_model/`](../full_model/).
Same two files (`tt/model.py`, `tt/generator.py`), same 40-layer decoder stack
(`tt/multichip_decoder.py`, still not a line changed), same suite file, same mesh: four Blackhole
`p300c` chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4 dense and EP=4 over the 256
routed experts. No shared repo code changed either — the sampler's grouped local top-k was already
opt-in from the previous stage; this stage only stops hard-coding its group count.

---

## 1. Before and after

Warmed, batch 1, the vLLM primary single-user profile (prompt 128 / generate 128), real checkpoint
weights, all 40 layers, **nine repeats per arm measured in one session**:
`logs/bench_full_model.py --arm inherited --repeats 9` → [`perf_summary_before.json`](perf_summary_before.json)
and `logs/bench_full_model.py --repeats 9` → [`perf_summary.json`](perf_summary.json). The `inherited`
arm rebuilds the pre-optimization path from constructor knobs — untuned interleaved LM head, unsharded
terminal norm, no sampler-friendly vocabulary alignment, serial readback loop — and reproduces the previous
stage's `doc/full_model/perf_summary.json` to **0.012 % on `traced_logits_only_decode`**, **0.002 % on
`traced_decode_plus_sampling_no_readback`** and **0.043 % on `token_out_decode`** (the row that carries
the host loop, and the one this arm's own nine repeats show varying by 0.28 %), which is what makes it a
valid baseline. Nine repeats because TTFT on this host has a spread far wider than the effect and a
three-sample figure cannot be compared with a nine-sample one.

| figure | inherited (n=9) | optimized (n=9) | delta |
|---|---|---|---|
| **token-out decode** | 23.879 ms/token — **41.88 t/s/u** | **23.300 ms/token — 42.92 t/s/u** | **−0.580 ms, +2.4 %** |
| traced logits-only decode (model trace alone) | 22.265 ms/token | **22.210 ms/token** | −0.055 ms |
| model trace + sampling trace, no readback | 23.434 ms/token | **23.345 ms/token** | −0.089 ms |
| decode run-to-run spread over the nine repeats | 23.879–23.947 (0.28 %) | **23.300–23.304 (0.018 %)** | — |
| **warmed TTFT** | 133.1 min / **139.1 median** / 150.1 max | 133.4 min / **140.1 median** / 146.1 max | **not a metric this stage moves — §6** |
| traced teacher-forcing decode (`run_teacher_forcing`) | 37.01 t/s/u (archive) | 38.18 t/s/u | not attributable to this stage — see below |
| layer-stack lower bound (unchanged, inherited) | 21.450 ms/token — 46.62 t/s/u | 21.450 | — |

**Decode is the result.** It is reproducible to 0.018 % across the nine optimized repeats — against
0.28 % for the inherited arm, because the serial loop's per-token host stall is itself variable — and
the same build's serial arm (`serial_token_out_decode`, 23.864 ms/token / 41.90 t/s/u) reproduces the
inherited number, so the 0.564 ms the pipelined loop wins is a same-build difference rather than a
cross-run one.

**TTFT is not a metric this stage can claim to move, in either direction.** Three nine-repeat pairs on
the same code have now come out at **+2.7, −7.2 and +1.0 ms** on the median. No sign is explicable by
device work: the only device-side change TTFT sees is the
LM head reading 384 more columns per device, which is **2.4 µs** at 353 GB/s, plus one
`interleaved_to_sharded` on a single 32-row block. §6 has both distributions and the one component that
*is* stable and attributable — the untraced first-token sampler, +0.36 ms here and +1.08 ms in the
earlier pair, always positive and always the eager group split.

**Teacher forcing** keeps the serial loop **by construction** (§2) — `next_input` decides the next token
input on the host, so there is nothing for the pipelined loop to overlap, and
`test_teacher_forcing_keeps_the_serial_loop` asserts exactly that. The 38.18 t/s/u in
`readiness_teacher.json` is therefore not attributable to this stage's change, and the harness's own
window additionally carries a per-token torch callback and a host token write. It is context, not a
result.

Where the 0.580 ms of decode came from, and what is left:

| term | inherited | optimized | what changed |
|---|---|---|---|
| decoder-layer stack (30 × 0.564 + 10 × 0.453) | 21.450 | 21.450 | nothing. Inherited, and required to be |
| embedding + final norm + LM head + device `plus_one` | 0.815 | **0.760** | tuned LM-head matmul + width-sharded terminal norm (§3) |
| sampling trace | 1.169 | **1.135** | grouped local top-k regrouped from 20 to 32 (§4) |
| host synchronize + caller token readback | 0.446 | **−0.045** | pipelined readback (§2) |
| **token-out total** | **23.879** | **23.300** | |

`sync_and_readback_ms` is negative because it is a residual: the token-out step is now *at* the
replay-plus-sampling throughput, so the caller's readback is entirely hidden rather than merely
cheap. `steady_state_counters` reports `decode_syncs: 0` against `decode_calls: 127`.

**8.6 % over the bare layer-stack lower bound, and ~0 % over lower bound plus terminal work** — all
of the gap is named: 0.760 ms of terminal arithmetic plus 1.135 ms of sampling, with the loop
contributing nothing.

Accuracy against the AIME24 chat-template reference the full-model stage generated
([`readiness_aime24_chat.meta.json`](../../readiness_aime24_chat.meta.json) is its provenance), both
bars cleared with maximum margin:

| gate | top-1 | top-5 | top-100 | bar |
|---|---|---|---|---|
| `run_prefill_check` | 0.940 (was 0.950) | **1.000** | **1.000** | top-5 ≥ 0.98, top-100 = 1.00 |
| `run_teacher_forcing` (traced decode) | 0.970 (was 0.940) | **1.000** | **1.000** | top-5 ≥ 0.98, top-100 = 1.00 |

Top-1 moves by one token down and three tokens up out of 100 — the two directions are the tell that
this is near-tie churn in the last bits of the logits (1536 more vocabulary columns, a tuned matmul, a
sharded norm), not a regression: top-5 and top-100 are both exactly 1.000 before and after, and the
free-running text is unchanged in character (§5).

Performance accounting, from the same run (`perf_summary.json::performance_accounting`):

| term | value |
|---|---|
| roofline estimate | **1.464 ms/token** (749,907,328 B per device per token / 512.3 GB/s) |
| end-to-end decode | 23.300 ms/token |
| fraction of roofline achieved | **6.3 %** |
| device-time decode | `null` — *not measurable for the 40-layer stack*, see limitation 2 |

The roofline is summed from the live device tensors, so it is per-device by construction: every entry
of each layer's weight dicts, the routed experts scaled to the profiled per-device active fraction
(**4 of the 64 experts each device owns**, not 8 of 256 — this is an EP=4 model), the LM-head shard,
the final norm and one embedding row. It is deliberately smaller than `footprint.json`'s measured
resident set, which counts the whole paged KV cache, all 64 local experts and the RoPE tables, none of
which one decode step reads. 6.3 % is low and the reason is not bandwidth: the step is
**launch-bound**, ~100 device ops per layer at one tile of M. The decoder stage measured ~7 % for a
single layer in isolation and named the same cause; this stage inherits it and does not have the
decoder's policy freedom to attack it.

---

## 2. The host stall: 0.564 ms/step that was not work at all

The inherited loop replayed both traces non-blocking and then **blocked the host on
`ttnn.synchronize_device` every token** before reading the sampled token. That is not device work; it
is device idle time, and the reason it can be removed is a property the full-model stage already
built: the steady-state free-running loop has **no host→device dependency**. The sampled token
reaches the next replay through `tt_out_tok` on device, and `current_pos`/`rot_idxs` advance with
`ttnn.plus_one` inside the captured graph, so step *N+1* does not need to know token *N*.

`OrnithGenerator` now enqueues the readback instead of waiting for it:

```python
self._decode_step_traced(); self._sample_traced()   # step N
in_flight = self._read_tokens_async()               # tok.cpu(blocking=False) + record_event, cq0
        ... the loop enqueues step N+1 ...
predicted = int(self._finish_read(pending)[0])      # event_synchronize, then to_torch
```

Correctness comes from the queue order, not from luck: the read is enqueued on the same command queue
**between** step *N*'s sampling and step *N+1*'s replay, and the queue is in order, so it observes
exactly step *N*'s token even though step *N+1* has already been submitted. The host's wait then
overlaps device work that is already running. This is the same
`cpu(blocking=False)` + `ttnn.record_event` + `ttnn.event_synchronize` idiom
`models/tt_transformers/tt/generator.py::read_decode_output(async_read=True)` uses.

* `test_the_pipelined_readback_agrees_with_the_serial_loop` runs the same prompt through both loops
  and asserts the token lists are **identical**, that the serial loop synchronizes once per token and
  the pipelined loop **never**, and that `token_refreshes`/`page_table_refreshes` stay 0 with
  `position_refreshes` at 1.
* Teacher forcing keeps the serial loop **by construction** — `next_input` decides step *N+1*'s token
  input on the host from token *N*, so there is nothing to overlap.
  `test_teacher_forcing_keeps_the_serial_loop` pins that (`perf["pipelined_readback"] is False`,
  `decode_syncs == decode_calls`). It is why §1 keeps teacher-forcing and token-out decode as separate
  rows and why only the latter moves.
* `pipelined_readback=False` is a constructor knob, so the arm stays measurable rather than becoming
  folklore — and it is what §1's `inherited` arm and the same-build `serial_token_out_decode` row
  (23.864 ms/token / 41.90 t/s/u) both use.

**What the one-token lookahead costs on an EOS stop, precisely.** The loop sees EOS in token *N* only
after step *N+1* has been enqueued, so that step **executes**: it consumes the EOS token as its input,
writes one paged-KV entry at the next position and advances `current_pos`/`rot_idxs` by one. Its
sampled token is discarded and the returned list is unchanged, but device state after an
EOS-terminated `generate` is **one position ahead of the returned tokens**. That is harmless here
because every `generate` resets before it prefills and the low-level API is not pipelined, and it is
why `read_waits == decode_calls` holds on a run to length — which is what the test asserts, with
`stop_on_eos=False` — rather than on an EOS-terminated one, where one extra replay has no matching
wait. `generate` synchronizes after the loop to retire it.

---

## 3. The terminal path

Two ladders, both one process per arm on the reduced two-layer variant so a build costs ~15 s instead
of ~200 s — the rows are absolute milliseconds, so a difference there is the same difference on 40
layers.

* [`logs/ab_terminal_table.md`](logs/ab_terminal_table.md) — 19 arms crossing matmul spelling, core
  count, terminal-norm sharding, vocabulary alignment and weight dtype. Driver `logs/ab_terminal.sh`,
  raw output `logs/ab_terminal.txt.gz`, table generated by `logs/make_ab_table.py`.
* [`logs/ab_terminal_kblock_table.md`](logs/ab_terminal_kblock_table.md) — 11 further arms crossing
  **`in0_block_w` × terminal-norm shard grid × math fidelity** on the shipped dtype policy, which is
  the sweep `$optimize` requires for the largest decode-time consumer. Raw output
  `logs/ab_terminal_kblock.txt`.

| arm | model trace | sampling trace | token-out (pipelined) |
|---|---|---|---|
| inherited: bare `ttnn.linear`, unsharded norm | 1.475 | 1.181 | 2.657 |
| the same with the terminal norm width-sharded | **1.650** | 1.181 | 2.857 |
| `mcast1d` 110 cores, **unsharded** norm | 1.480 | 1.181 | 2.668 |
| `mcast1d` 110 cores, **sharded** norm | **1.434** | 1.181 | 2.646 |
| `mcast1d` 88 cores, **unsharded** norm | 1.480 | 1.181 | 2.667 |
| `mcast1d` 64 cores, **sharded** norm | 1.477 | 1.181 | 2.688 |
| `dram_sharded` 64 cores, sharded norm | 1.553 | 1.133 | 2.704 |
| **shipped: `mcast1d` 110 + sharded norm + vocab align 32** | **1.435** | **1.121** | **2.583** |
| (`lm_head_dtype=bfloat4_b` on the shipped arm) | 1.351 | 1.122 | 2.487 |

(The 88-core arm was measured with the norm unsharded and the 64-core arm with it sharded, so those two
are not directly comparable to each other; each is comparable to the 110-core arm in its own family,
and 110 wins both — 1.434 against 1.477 sharded, 1.480 against 1.480 unsharded.)

### 3.1 Sharding the terminal norm is a regression *unless its consumer wants the layout*

The inherited `ttnn.rms_norm` had no program config and no memory config, and the decode capture shows
it on a **single core at ~20 µs** while the decoder's own in-layer norms run on 8 cores at 6 µs. The
obvious fix — width-shard it with a `LayerNormShardedMultiCoreProgramConfig` — makes the model trace
**0.175 ms slower** (1.650 vs 1.475), because a bare `ttnn.linear` cannot consume a width-sharded
activation and the layout is undone again. So `terminal_norm_sharded=None` resolves to *"shard iff the
head was given a program config"*, and in the shipped configuration the norm is 6 µs on 8 cores — a
14 µs/step saving visible directly in the profile's `LayerNorm` rows (the DRAM-interleaved
terminal-norm row disappears; the width-sharded family gains 6 µs/replay).

### 3.2 `tt-perf-report`'s own LM-head advice was tried and loses

The report says, on the LM-head row: *"Try a DRAM-sharded program config
(MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) to improve throughput further"*. It was built
following `models/common/modules/lm_head/lm_head_1d.py` — DRAM width-sharded weights over the DRAM
banks, L1 width-sharded activation, L1 width-sharded output, `sharded_to_interleaved` back to the
sampler's logits tensor — and it is **slower**: 1.553 against 1.435 ms. Two costs it cannot avoid: the
per-device vocabulary must be padded to a multiple of `32 * cores` (2.3 % more LM-head bytes to read
at 64 cores) and the sharded output must be converted back, because the sampling trace is captured
against an interleaved DRAM logits tensor. Below 64 cores it does not build at all — at 32 and 16
cores `per_core_N` reaches 61 and 122 tiles and the op fails with *"Statically allocated circular
buffers on core range [0-0 - 7-9] grow to 2220416 B which is beyond max L1 size of 1572864 B"*.
Rejected **with measurement and an exact L1 blocker**, not with a first API error.

`lm_head_program="dram_sharded"` remains a selectable arm, and it now **refuses an illegal core
count** rather than silently degrading: a DRAM-sharded matmul gives every compute core a whole number
of K tiles, so `lm_head_cores` must divide `dim/32 = 64`, and the shipped default of 110 does not.
Before this check, asking for that pair produced the untuned bare `ttnn.linear` *and* still paid the
spelling's vocabulary padding, with no error.

### 3.3 What wins: the decoder's own decode geometry — and its K block is shard-bound, not capped

`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `mcast_in0=True`, `fuse_batch=True`, over the whole
**11x10** worker grid, reading the width-sharded L1 activation the norm now produces:
`in0_block_w=8`, `per_core_M=1`, `per_core_N=18`, output subblock `1x6`. 110 cores is the top of the
ladder (88 and 64 measure 1.477–1.480).

`in0_block_w=8` is **not** an arbitrary cap, and the bound is worth stating exactly because it is
coupled to the norm: with `mcast_in0` and a width-sharded `in0` the matmul blocks the inner dimension
out of what each core *holds*, so it validates `in0_shard_tiles % in0_block_w == 0` as well as
`k_tiles % in0_block_w == 0`. The terminal norm shards `dim` over 8 cores, so each core holds
`2048/32/8 = 8` tiles and 8 is the maximum **for that grid**. Reaching 16, 32 or 64 means a *narrower*
norm grid, which is the OPT-011 trade — so it was measured rather than assumed
([`logs/ab_terminal_kblock_table.md`](logs/ab_terminal_kblock_table.md)):

| terminal norm cores | `in0_block_w` | model trace | verdict |
|---|---|---|---|
| **8** | **8** | **1.434** | **shipped** |
| 8 | 4 | 1.442 | slower |
| 4 | 8 | 1.437 | slower |
| 4 | 16 | 1.476 | slower — the narrower norm costs more than the wider K block buys |
| 2 | 16 | 1.463 | slower |
| 2 | 32 | *does not build* | `Statically allocated circular buffers in program 466 clash with L1 buffers on core range [0-0 - 10-8]. L1 buffer allocated at 1404928 and static circular buffer region ends at 1532864` |
| 1 | 64 | *does not build* | same assert, `L1 buffer allocated at 1273856` |

And math fidelity was swept for this row on its own rather than inherited from the dense-projection
group, because it is the largest full-model-only decode op:

| fidelity | model trace | verdict |
|---|---|---|
| **HiFi2 (shipped, the dense group's)** | **1.434** | kept |
| HiFi4 | 1.437 | tie — the row is DRAM-bound, so extra fidelity is nearly free but buys nothing here |
| LoFi | 1.466 | **slower.** Lower fidelity does not help a bandwidth-bound row, and it would cost logit precision |

So the terminal matmul's geometry and fidelity are the measured winners of **11 arms** (listed in
`work_log.md` §3.4) spanning
`in0_block_w ∈ {4,8,16,32,64}`, `norm cores ∈ {1,2,4,8}` and `{LoFi,HiFi2,HiFi4}`, with an exact L1
blocker for the two geometries that cannot be built. It is not a full 60-cell cross and does not need
to be: `in0_block_w` must divide both the tiled K and the norm's per-core shard, so most cells are
illegal, and `work_log.md` §3.4 lists the arms that were actually run. Two honest caveats:

* **norm grids wider than 8 cores are legal and were not measured.** `_terminal_norm_cfg` accepts 16
  and 32 (both divide `dim/32 = 64` and give a legal rectangle), which would force `in0_block_w` down
  to 4 and 2. Given that `in0_block_w=4` on the 8-core grid already measures only 0.008 ms worse, a
  narrower K block on a wider norm is very unlikely to win, but the cross above is the `≤8` half of
  the legal range and is stated as such;
* **the selection margins are at the ladder's resolution.** The choice was made on *model trace*,
  where the shipped arm leads by 0.008 ms over `in0_block_w=4` and 0.003 ms over HiFi4. The same
  table's **token-out pipelined** column — the delivered metric — slightly *favours* the rejected
  arms (`k4-n8` 2.580, `k8-n4` 2.583, `k8-n8-hifi4` 2.582 against the shipped 2.586), and the first
  ladder's repeat pairs show 0.001–0.003 ms of run-to-run spread. Any of these four is within 0.03 %
  of a decode step, so the row is reported as a near-tie rather than as a clear win. The ladder's
  `final_norm` column moves the *wrong* way for the shipped arm too (0.021 → 0.036 ms) because it is an
  eager, dispatch-dominated measurement of a 6 µs op; §3.1's claim rests on the profiler's `LayerNorm`
  rows instead, where the 1-core 20.4 µs DRAM-interleaved row disappears and the total falls
  262.5 → 205.8 µs/window. `test_the_lm_head_runs_the_tuned_program_config` asserts the config
class, `mcast_in0`, `fuse_batch`, `in0_block_w`, `per_core_M` and the 11x10 grid in the runtime path,
plus that the terminal norm is width-sharded.

The gain is small and the report says why: the LM head is **374 µs at 353 GB/s = 69.0 % of the DRAM
roofline on 109 cores**, essentially where it started (370 µs / 355 GB/s / 108 cores). It was already
DRAM-bound, and reading a 2048 x 62464 bfloat8_b weight per token is the floor. What the tuned
spelling actually buys is the norm, without paying a conversion to get there.

### 3.4 The LM-head weight dtype was tried on real weights and rejected

`lm_head_dtype=bfloat4_b` is the fastest arm in the whole ladder: **1.351 ms** of model trace against
1.435, i.e. 0.084 ms/token, 0.36 % of a decode step. It is rejected on **real-checkpoint accuracy**,
not on a synthetic PCC (`logs/readiness_bfp4_head.txt`, `readiness_prefill_bfp4head.json`,
`readiness_teacher_bfp4head.json`):

Both arms measured on the **same tree**, through the same two gates:

| gate | bfloat8_b (shipped) | bfloat4_b |
|---|---|---|
| `run_prefill_check` top-1 | **0.940** | 0.920 |
| `run_teacher_forcing` top-1 | **0.970** | 0.940 |
| top-5 / top-100, both gates | 1.000 / 1.000 | 1.000 / 1.000 |

Two and three points of top-1 agreement with the HF reference, on the one tensor that decides the
emitted token, for 0.36 % of a step. Both arms still clear the stage bars, so this is a genuine
frontier point rather than a failure — which is exactly `$datatype-sweep`'s job, and it is handed over
with the measurement rather than with an opinion. Note the size of the gap is comparable to the ±1–3
token near-tie churn §1 describes, which is a second reason the decision here is "keep the decoder's
dense-projection dtype, as the goal contract requires" rather than "bfp4 is 2 points worse".

---

## 4. Sampling: the group count is derived from the shard, and it is at the joint optimum

The full-model stage measured `ttnn.topk` as linear in the **reduced width** and independent of every
other dimension, and made the local top-k grouped: one `W`-wide reduction becomes a `W/g`-wide one
over `g` rows plus a `max_top_k * g`-wide one over the group winners. It chose `g = 20` as the best
divisor of 1940 (= 62080/32).

That is right for that width and wrong as a constant. The reduction cost is `W/g + max_top_k*g` width
units, minimised near `g = sqrt(W/max_top_k) ≈ 44` — but 1940 = 2²·5·97, whose divisors jump from 20
straight to 97 and skip the whole neighbourhood. `g = 20` costs 3744 units.

So this stage pads the vocabulary **for the factorisation, not for the matmul**: aligning the
per-device width to 32 tiles makes it 62464 = 32 x 1952, and 1952 = 2⁵·61 has 32 as a divisor, for
**2976 units**. `OrnithModel.best_topk_groups` computes it from whatever width the build produced and
`topk_num_groups="auto"` is the default; the rule reproduces the inherited 20 for the unpadded 62080
shard, which is how it is tested. It minimises the **whole** sampler cost — both terms of §4.1's
measured model, not the reduction alone. For this build both objectives land on 32 (reduction-only
would *tie* 32 with 61 and the loop keeps 32), so the shipped value does not depend on the change; what
the joint objective buys is that a future build which raised `vocab_align_tiles` to 1980 tiles would
get 33 rather than the `g = 44` candidate §4.2 rejects. `TOPK_US_PER_WIDTH_UNIT` and
`TOPK_GROUP_MACHINERY_US_PER_GROUP` in `tt/model.py` are that model's two generated coefficients.

Measured: the sampling trace goes **1.181 → 1.121 ms** on the probe (1.121 in three arms and 1.122 in
four more) and **1.169 → 1.135 ms** on the delivered 40-layer model, same sweep, and `TopKDeviceOperation` falls from **29.98 %
to 24.76 %** of the reduced decode window. The capture confirms the reduction cost model exactly:
stage 1 is 369 µs at width 1952 on 32 cores and stage 2 is 195 µs at width 1024 on **1** core — 0.189
and 0.190 µs per width unit, i.e. linear in width and indifferent to cores, and 2976 x 0.19 = 565 µs
against the 564 µs measured.

### 4.1 The grouping is not free, and that is what bounds `g`

Every number in this subsection is generated from the two committed stacked Tracy reports by
`logs/make_sampler_cost_model.py` into
[`logs/sampler_cost_model.md`](logs/sampler_cost_model.md), summed **per op code across every
memory-layout variant** so no row can mix a DRAM-only figure with a DRAM-plus-L1 one. The 20 → 32
group move is **not** a pure top-k win:

| op code | before (µs/window) | after (µs/window) | Δ per replay |
|---|---|---|---|
| `TopKDeviceOperation` | 2,833.2 | 2,255.3 | **−144.5** |
| `SliceDeviceOperation` | 350.1 | 496.2 | **+36.5** |
| `ConcatDeviceOperation` | 109.1 | 206.3 | **+24.3** |
| `GatherDeviceOperation` (stage-2 index recovery) | 150.7 | 172.6 | **+5.5** |
| `BinaryNg` *(reported, not fitted)* | 689.5 | 704.9 | +3.8 |
| **sampler net** | | | **−78.2** |
| **whole window** | 9,449.9 | 9,109.5 | **−85.1** |

Two things follow. The grouping machinery — `g` × `ttnn.slice`, one `ttnn.concat` over `g` inputs, and
the index-recovery `gather` — costs **5.52 µs/replay per group**, so ~**177 µs/replay** at `g = 32`.
And the whole-window device delta (−85.1 µs/replay) matches the **−88.8 µs/token** wall delta on the
40-layer model (`traced_decode_plus_sampling_no_readback`, 23.434 → 23.345 ms): **device time and wall
time move roughly 1:1** here, not at the 0.42 ratio a naive "60 µs of wall for 144 µs of top-k" reading
would suggest.

With both terms measured the sampler's device cost is

```
device(g) ~= 0.188 × (W/g + 32g)  +  5.52 × g  +  const
```

(`0.188 * (W/g + 32g)` and the `5.52 µs/replay per group` slope are both printed by
`logs/make_sampler_cost_model.py`; `logs/check_prose_figures.py` asserts this section against it.)

minimised at **`g = 31.9`** for `W = 62464`, so the shipped `g = 32` is the **joint optimum**, not
merely the best legal divisor. The model reproducing the move it was solved from is arithmetic rather
than validation; what checks the reduction coefficient independently is the two `TopK` rows of the
after report on their own — 369 µs at width 1952 and 195 µs at width 1024, i.e. 0.189 and 0.190 µs per
width unit against the 0.188 the fit solves.

### 4.2 Why there is no third reduction stage, and what the real next lever is

The same model rejects both remaining reduction candidates, with the grouping term counted:

| candidate | reduction | machinery | extra ops | net |
|---|---|---|---|---|
| pad to 1980 tiles, `g = 44` (+896 columns/device) | −24 µs | **+66 µs** | +4 µs of LM head | **+46 µs worse** |
| three-stage, `g = 99` / `h = 11` on 1980 tiles (1280 units) | −319 µs | **+431 µs** | +43 µs second index gather | **+155 µs worse** |

The sign does not depend on the exact split: the machinery term is linear in the group count while the
reduction term is only `W/g`, so every legal `(g, h)` with `g` above 32 loses.

So the remaining sampler lever is **not** a third stage: it is the machinery itself. ~177 µs/replay at
`g = 32` is about a fifth of the sampler's ~880 µs/replay of device time, and all of it is op-launch
overhead for what is
arithmetically a free reinterpretation — `[1,1,32,W]` viewed as `[1,32,32,W/32]` is exactly what stage
1 wants, with every group starting on a tile boundary. Replacing 32 slices plus a concat with one
reshape is a change to shared `TTSampling._local_topk_grouped` and to its index recovery; it is named
here with its size (~0.18 ms/token, ~0.75 % of a decode step, at the measured 1:1 device-to-wall
ratio) rather than left implicit, and it is the first thing to try in the sampler next.

### 4.3 The padding is masked, not merely padded

1536 extra columns produce 1536 real logits, and a zero-padded LM-head column is *not* a mask — a 0.0
logit beats every negative real one. `TTSampling` builds an additive invalid-vocab tail mask from
`(vocab_size, padded_vocab_size)` and applies it before the local top-k (`tt_sampling.py:958`, ahead
of `_local_topk` at `:997`); the mask is device-sharded so only the device owning the tail pays it.
`test_the_padded_vocabulary_is_masked_not_merely_padded` asserts the mask exists **and** that a real
generation never emits an id `>= vocab_size`.

### 4.4 Greedy is unchanged and still semantically greedy split sampling

`SamplingParams(temperature=0.0, top_k=1, top_p=1.0)` runs the same captured graph every other mode
runs — local top-32 per vocabulary shard, gather the 4 x 32 candidates, `ttnn.sampling` with `k=1` —
not a generic sampled `top_k=32` stand-in. `allow_force_argmax` stays off: it would all-gather the
full 249856-wide logits and run a global `ttnn.argmax`, which is precisely the shape the goal forbids,
and the sampler is 4.9 % of the step so there is nothing to buy. The sampler's own gather still uses
`ttnn.experimental.all_gather_async` through `OrnithSamplingCCL` rather than the deprecated
`ttnn.all_gather` the decoder stage removed for cross-device divergence.

Two smaller sampler items were examined and deliberately left:

* `ManualSeedDeviceOperation`, 18 µs/step, is dead work for greedy — but `SamplingGenerator` keys
  traces by `(penalties, logprobs, force_argmax)` and **not** by `k`, so a trace captured without the
  seed would be replayed for a sampled request. 0.08 % of a step is not worth silently non-random
  sampling.
* a persistent output buffer for the sampler's **values** gather remains blocked by the op-contract
  conflict the full-model stage found: `TTSampling` deallocates
  `topk_values_gathered_bf16_interleaved` at the end of every call, and on a DRAM sampling memory
  config that tensor *is* the gather's output, so a caller-supplied buffer is freed by the first call.
  The **indices** gather's output is not deallocated, so that blocker does not cover it — but both
  gathers together are `AllGatherAsync` at 20.3 µs/replay, so OPT-009's ~11 % is 1–2 µs and it was not
  pursued. Stated precisely because the previous phrasing claimed the blocker for both.

---

## 5. Behaviour is unchanged

Every gate the full-model stage set was re-run on the optimized path.
[`logs/run_evidence_status.txt`](logs/run_evidence_status.txt) is the per-step status file for one run of
the committed `logs/run_evidence.sh`: **14 steps**, of which steps 6 and 7 are the two nine-repeat bench
arms. **Three steps failed, all in one class and all for the same host reason, and it is not a model
result:** `readiness_autoregressive`, `readiness_autoregressive_chat` and `readiness_qualitative` are the
only steps that load a full 35B HuggingFace reference on the CPU, and the host had 51–55 GiB of
`MemAvailable` against a ~70 GiB requirement, so all three were OOM-killed inside the HF load with no
traceback. A standalone retry on an otherwise idle host died identically, so it is a persistent host
condition rather than a transient one. [`logs/host_memory_event.txt`](logs/host_memory_event.txt) has the
`/proc/meminfo` numbers, the evidence that it is the OOM killer, and the reasoning below.

Their artifacts are intact from the **immediately preceding complete sweep** of the same script on the
same delivered code (the driver writes them only on success, so nothing was truncated), and what they
measure — free-running generated text and the shared qualitative suite — is behaviour on a path that has
not changed since. Everything else in the final sweep passed, including both bench arms, every probe, the
degeneracy gate over those same completions, and the 48-case fast suite.

[`logs/check_prose_figures.py`](logs/check_prose_figures.py) is a separate gate, deliberately not a step
of the sweep: it asserts every figure in this file and in `work_log.md` against the artifact it names,
plus the existence of every referenced path, and it can only pass *after* the documents are refreshed
from a sweep's output. Its committed output is
[`logs/check_prose_figures.txt`](logs/check_prose_figures.txt).

* **Suite** — 48 fast cases pass (42 inherited + 6 new for this stage) and all 5 long cases pass:
  `logs/pytest_full_model.txt.gz`, `logs/pytest_long.txt`. The long set includes the 5003-token
  non-aligned prompt through the complete stack and the batch-32 prefill/decode case.
* **Watcher** — the fast suite under `TT_METAL_WATCHER=10`, in its own run, with
  `TT_METAL_WATCHER_DISABLE_ETH=1` inherited from the decoder stage's limitation: **48 passed, exit 0,
  0 watcher error/assert/hang lines** in either the pytest log or `generated/watcher/watcher.log`
  ([`watcher/watcher_error_count.txt`](watcher/watcher_error_count.txt), generated by
  `logs/watcher_report.sh`). That file also counts **bare** device asserts, which never say "watcher"
  and which the narrow pattern would miss: three lines match, all listed in the file and all benign —
  one expected `allocator.cpp:123` active-trace warning and two occurrences of a passing test's *name*.
  Re-run after the final code change.
* **Free-running generation** — 128 tokens, greedy, HF control alongside. On the **chat-template**
  prompt the TT text reproduces the control's structure and content, 42 of 128 tokens identical at the
  same positions. On the **raw continuation** prompt only 11 of 128 positions match, which is expected
  rather than a defect: greedy trajectories separate permanently after the first near-tie, and this is
  a creative continuation with many. Both TT completions are fluent, on-topic and non-repetitive
  (`adjacent_duplication 0.0`, trigram-loop 0.028 and 0.092), and `$qualitative-check` labels the raw
  prompt continuation *stress* coverage for an instruct model rather than the quality verdict.
* **Degeneracy gate** — `check_degenerate_output.py --missing-artifacts critical --scope
  autoregressive` is clean, `findings: []` ([`degenerate_report.json`](degenerate_report.json)).
* **`$qualitative-check` shared suite** — all six `vllm_prompts.txt` prompts, rendered with
  `tokenizer.apply_chat_template(add_generation_prompt=True)` because the checkpoint has a chat
  template, with an HF control generated the same way
  ([`readiness_qualitative.json`](readiness_qualitative.json)). Every TT completion is coherent,
  on-topic and structurally matched to its control. The one oddity is inherited and controlled:
  prompt 3's completion opens `Here's a thinking thinking sequence` — and **the HF control for the
  same prompt in the same run opens with the identical phrase**, so the doubled word is the
  checkpoint's, not the port's.
* **Multi-request** — `logs/probe_multi_prompt.txt` drives six prompts of different lengths twice
  through one generator; the full-model stage's §5.1 corruption guard still holds.
* **Slot contract at batch 4** — [`batch_slots.json`](batch_slots.json) is a passing *negative
  control*, not a null result: `logs/probe_batch_slots.py` shows slot 0's tokens are byte-identical
  whether its three neighbours hold the same prompt or three different ones (so batch-4 divergence is
  matmul/MoE geometry, not cross-request leakage), and its `batch4_generate_no_merge` arm patches
  `_merge_prefill_state_into_slot` out to show what a *real* state defect looks like — the **first**
  decoded token changes. Both arms reproduce the full-model stage's findings exactly.

---

## 6. TTFT: decomposed, and dominated by host drift

Same script, nine repeats per arm, back to back
([`perf_summary_before.json`](perf_summary_before.json) then [`perf_summary.json`](perf_summary.json),
steps 6 and 7 of `logs/run_evidence.sh`):

| arm | TTFT samples (ms, sorted) | min | median | max |
|---|---|---|---|---|
| inherited | 133.1 134.4 134.7 137.0 **139.1** 140.7 147.5 149.8 150.1 | 133.1 | **139.1** | 150.1 |
| optimized | 133.4 135.7 138.3 139.1 **140.1** 140.5 141.9 145.2 146.1 | 133.4 | **140.1** | 146.1 |

Read that as **no measurable change**. Three nine-repeat pairs on the same code have come out at +2.7,
−7.2 and +1.0 ms on the median — the committed one is the last — and no sign has a mechanism: the only device-side change TTFT sees is the LM head reading 384 more columns per device,
which is 384 × 2048 × 1.0625 B / 353 GB/s = **2.4 µs**, plus one `interleaved_to_sharded` on a single
32-row block. Both arms' distributions span ~15 ms and both have a long upper tail. TTFT on this host is
host-drift-dominated at the ±7 ms level, and the honest statement is that this stage does not move it.
This is also why §1 reports the *decode* figure as the result: its nine repeats agree to 0.018 %.

What *is* stable and attributable is the breakdown, because it is measured inside one process per arm:

| term | inherited | optimized | delta |
|---|---|---|---|
| page-table row upload | 0.053 | 0.052 | −0.001 |
| prefill (embedding + 40 layers + last-row norm/LM head) | 127.507 | 128.395 | +0.888 |
| **first-token sampling, untraced** | **3.048** | **3.350** | **+0.303** |
| total | 130.607 | 131.797 | **+1.189** |

The prefill term is flat, which is what the 2.4 µs bound predicts. The first-token term is the one that
moves, and it is the flip side of §4: that token is sampled **eagerly**, so the vocabulary alignment
that makes the *traced* sampler faster (20 → 32 groups, −0.034 ms/token) makes the *eager* one slower —
12 more `ttnn.slice` launches and a wider `ttnn.concat`, once per request. It measured +0.303 ms here and +0.36 / +1.08 ms in the two earlier pairs; it is always positive, and at this profile's 128 generated tokens it is
paid once against the 0.564 ms × 127 = 72 ms the pipelined loop saves over the same window. **The fix is limitation 6 and it is blocked**: sampling
the first token through the captured trace would make the term ~1.1 ms in both arms, and it wedged the
mesh (§9).

### Where the rest of TTFT is

`logs/probe_prefill.py` → [`prefill_profile.json`](prefill_profile.json) walks warmed TTFT over a length
ladder and fits it two ways, both computed by the probe:

| prompt tokens | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| warmed TTFT (ms) | 131.0 | 176.8 | 253.8 | 426.1 |

| fit | slope | intercept | share of a 128-token TTFT |
|---|---|---|---|
| two-point secant through 128 and 1024 | 0.329 ms/token | **88.9 ms** | **67.8 %** |
| least squares over all four points | 0.327 ms/token | **89.8 ms** | **68.5 %** |

The local 128→256 slope is 0.358 ms/token. So **68 % of a 128-token TTFT is length-independent**,
which is what 40 layers of *eager* dispatch looks like; the secant is the conservative end of that range
and the least-squares intercept is the smaller prize. Either way the term dominates.

One hypothesis was tested and refuted along the way: the eager path emits a per-layer,
per-expert-group `logger.debug` line, and a host log call inside a measured window is host work. But
`prefill_profile.json::debug_logging_cost_ms` reads **+3.3 / −0.4 / −10.0 / +1.4 ms** across
128/256/512/1024 in the committed run, against −4.6 / +9.1 / −2.4 / +0.2 in an earlier one — the
magnitudes do not repeat and the sign flips both within and between runs, which is what noise looks
like. Not it.

**Capturing a prefill trace is the optimization that intercept implies, and it is not taken here**,
for a specific reason rather than a preference: the prefill program set is keyed by the **logical**
prompt length, not the physical block length — the `ttnn.slice` offsets, the MoE valid-token count and
the `conv1d` length are all compile-time constants — so "one trace per prefill shape" means one trace
per distinct prompt length, unbounded, unless the decoder layer's prefill masking contract changes to
take a physical block plus a validity mask. That contract is decoder-owned and this stage's goal is to
preserve it. The ceiling (91–98 ms of a ~134 ms TTFT) is recorded so the next stage can decide with a
number.

### 6.1 What this stage *does* fix on the prefill side

`doc/full_model/`'s limitation 3 said a newly seen prompt length costs one trace re-capture that
"lands in *neither* reported metric", and that pre-compiling a bucket of lengths "is left to the
optimized-full-model stage". Both halves are now done.

**Measured** — `perf_summary.json::cold_prompt_length_cost`, for a fresh non-aligned length 135:

| | value |
|---|---|
| first request, wall clock | **0.548 s** |
| of which reported as TTFT | 296 ms |
| **hidden from both published metrics** | **252 ms** |
| trace re-captures | 1 |
| warmed TTFT at that length afterwards | 176 ms (135 pads to a 256-token physical block) |

(The inherited arm measures the same thing at 0.550 s / 305 ms / 246 ms — this is a property of the
guard, not of the optimization.)

**Fixed** — `OrnithGenerator.warmup([lengths])` pays it at startup and returns the per-length seconds
and the re-capture count. `test_warmup_removes_the_cold_length_recapture` drives a length nothing has
compiled (87 — not a multiple of the tile, the page or the 128-token alignment) and asserts that a
later request at that length does not re-capture.

---

## 7. What is preserved

### 7.1 The decoder stack's contract

`test_the_decoder_policy_is_carried_through_unchanged` still pins all of it from the full model's
side, and the decode capture confirms claimed policy equals measured policy: `HiFi2 BF16 x BFP8 =>
BF16` on the dense projections, `LoFi BF16 x BFP4 => BFP8` on the routed gate/up,
`GeneralizedMoeGate` at 2 µs on 32 cores with no `TopK` in the router chain at decode.

| item | value |
|---|---|
| mesh / parallelism | `1x4` Blackhole `p300c` ring, `FABRIC_1D_RING`; TP=4 dense, EP=4 over 256 experts |
| routed experts | bfloat4_b weights, LoFi, bfloat8_b activations |
| dense projections + shared expert | bfloat8_b, HiFi2, packer-L1 accumulate |
| router | bfloat16 weight, HiFi4, float32 accumulate; `ROUTER_MODE="fused_gate"` at decode |
| paged KV cache | **bfloat8_b**, 64-token blocks, bfloat16 `paged_update_cache` inputs |
| collectives | `CCL_MODE="all_reduce"`, exactly two per layer, both inside the layer |
| inter-layer residual | `[b, s, 2048]` bfloat16 TILE **DRAM-interleaved, replicated**, no collective at the boundary |

Nothing switched to a rejected policy or to a replicated stream: the residual contract, the collective
spelling, the router mode and every dtype are the decoder stage's own selections. The LM head takes
the dense projection group's dtype and fidelity (bfloat8_b / HiFi2) as it did before — §3.3 and §3.4
are the trials that confirm those choices rather than changes to them.

### 7.2 Capability

[`doc/context_contract.json`](../context_contract.json) is recomputed for this stage into its own
`optimized_full_model` block (the `full_model` block is left as the previous stage's record), from
measured allocator views rather than from a model:

| | |
|---|---|
| advertised context | **262144. No reduction.** |
| batch bound | **32**, unchanged (`ttnn.sampling` asserts `1 <= num_users <= 32`) |
| weights + embedding + LM head | 6,221,414,400 B per device (+835,584 B for the 1536 padded LM-head columns) |
| paged KV at 262144 + per-batch state | 1,802,859,008 B per device |
| decode + sampling traces and sampler tables | 7,998,464 B per device |
| **total resident** | **8,032,271,872 B (7.48 GiB)** of 31.65 GiB allocatable |
| **free for activations** | **24.16 GiB** |

Measured, not projected: [`long_prompt.json`](long_prompt.json) walks non-aligned prompts through the
**optimized** public path with the full 262144-token cache allocated —

| prompt | 5003 | 8191 | 16381 | 32749 | 65521 | 131071 | **262143** |
|---|---|---|---|---|---|---|---|
| prefill | 3.22 s | 3.46 s | 6.99 s | 14.30 s | 30.26 s | 67.84 s | **163.85 s** |
| tokens/s | 1552 | 2370 | 2345 | 2290 | 2165 | 1932 | **1600** |
| DRAM free after | 24.16 GiB | 24.16 | 24.16 | 24.15 | 24.15 | 24.15 | **24.14 GiB** |

Every row returns finite logits and a valid in-vocabulary sampled token. Non-aligned prompt length is
still a logical input at every scale: `test_prefill_accepts_any_logical_prompt_length` covers 1, 7,
31, 33, 63, 129, 250, 1000, 2049 and 3000, and `test_full_stack_non_aligned_long_prompt` runs 5003
through the complete stack.

### 7.3 The serving-ready generator contract

Explicit cache/page-table/position/prompt-length/batch state, mixed-length prompts, fixed slots and
inactive rows all still work, through the same low-level `prefill_forward`/`decode_forward` pair —
`test_batched_prefill_and_decode_with_mixed_prompt_lengths`, `test_batch_32_prefill_and_decode`,
`test_a_caller_can_own_the_cache`, `test_a_page_table_too_narrow_for_the_position_is_rejected`,
`test_a_changed_page_table_is_copied_exactly_once`. The pipelined loop is confined to the high-level
`generate` free-running path; the low-level API a serving adapter drives is untouched, which is
deliberate — a scheduler that builds step *N+1* from host request state must not have step *N*'s token
read overlapped out from under it.

---

## 8. Runtime fallback audit

| path | host work in the steady state | evidence |
|---|---|---|
| model decode (`ttnn_decode_forward`) | none. Device tensors in and out, positions advanced on device | `test_greedy_decode_has_no_host_fallback` |
| sampling | none. Traced, `tt_out_tok` into the decode token buffer | `test_split_sampling_feeds_the_token_back_on_device` |
| token feedback | none. No readback-and-rewrite | same test |
| **synchronization** | **none per token. `decode_syncs: 0` over 127 steps.** One synchronize per *request*, after the loop, to retire whatever the last iteration left in flight | `perf_summary.json::steady_state_counters` |
| caller readback | one enqueued 32-entry read per step, waited on **behind** the next replay. `token_readbacks: 128` for 127 steps — the extra one is the first token after prefill | `test_the_pipelined_readback_agrees_with_the_serial_loop` |
| page table | copied only when it changes | `test_a_changed_page_table_is_copied_exactly_once` |
| positions / RoPE | one host write per request, at the boundary | counters: `position_refreshes: 1` |
| token input | never written in free-running decode | `token_refreshes: 0` |
| prefill | host token upload per chunk and a page-row upload; inside TTFT, not in the decode loop | §6 |
| teacher forcing | one synchronize and one readback per token, and a host token write when the forced token differs — **by construction** | `test_teacher_forcing_keeps_the_serial_loop` |
| first token after prefill | untraced sampler, 3.350 ms, inside TTFT, and 0.30 ms of that is this stage's cost (§6). The traced alternative wedged the mesh — §9 | `triage/` |
| host sampling | `sampling_mode="host"` is an explicit compatibility mode, never the measured path | `test_host_sampling_compatibility_mode_agrees_with_device_sampling` |

The whole `doc/full_model/README.md` §9 table still applies for everything this stage did not touch,
including cache-ownership stickiness and the trace-safety guard.

---

## 9. Rejected, with the evidence

| candidate | measured / observed | decision |
|---|---|---|
| DRAM-sharded LM head (`tt-perf-report`'s own advice) | 1.553 vs 1.435 ms model trace at 64 cores; **does not build** at 32/16 cores — circular buffers grow to 2,220,416 B against 1,572,864 B of L1 | rejected on measurement **and** an exact L1 blocker (§3.2) |
| width-sharded terminal norm with the untuned head | 1.650 vs 1.475 ms | rejected; sharded only where the head consumes it (§3.1) |
| larger LM-head `in0_block_w` (16 / 32 / 64) | 16 needs a 4-core norm and measures 1.476 vs 1.434; 32 and 64 need a 2-/1-core norm and **do not build** (exact L1 clash quoted in §3.3) | rejected on measurement and an exact L1 blocker |
| smaller LM-head `in0_block_w` (4) | 1.442 vs 1.434 | rejected on measurement |
| LoFi / HiFi4 for the LM head | 1.466 / 1.437 vs 1.434 | rejected on measurement; the row is DRAM-bound (§3.3) |
| `lm_head_dtype=bfloat4_b` | fastest arm (1.351 ms) but real-weight top-1 0.920/0.940 against 0.940/0.970 | rejected here, handed to `$datatype-sweep` with the numbers (§3.4) |
| LM head on 88 / 64 cores | 1.480 / 1.477 vs 1.434 | rejected; 110 is the top of the ladder |
| further vocabulary padding (1980 tiles, `g=44`, +896 columns/device) | −24 µs of reduction against **+66 µs of grouping machinery** and +4 µs of LM head → **+46 µs/replay worse** | rejected on the measured cost model (§4.1–4.2); `g=32` is the joint optimum |
| three-stage grouped top-k (`g=99`, `h=11`) | −319 µs of reduction against **+431 µs of grouping machinery** and +43 µs for a second single-core index gather → **+155 µs/replay worse** | rejected on the measured cost model (§4.2), which also names the real next lever |
| dropping `ttnn.manual_seed` for greedy | 18 µs/step, but sampling traces are not keyed by `k` | rejected: would make a sampled request non-random (§4.4) |
| merging the sampling trace into the model trace | would remove one `execute_trace` and a ~43 µs inter-trace gap | rejected: `SamplingGenerator` keys sampling traces by mode, so merging would force a model-trace re-capture on every mode switch, and the canonical split-sampling contract is two cooperating traces |
| **traced first-token sampling** | would turn 3.350 ms of TTFT into ~1.1 ms and remove §6's one attributable TTFT cost | **rejected — it hung the mesh.** Copying prefill logits into `self._trace_logits` writes a **trace-region** buffer from outside a replay; `tt-triage` found all four devices stuck on one `ReshapeViewDeviceOperation` with kernel `.text` mismatches ([`triage/`](triage/)). Same hazard `SamplingGenerator.capture_trace(skip_precompile=True)` exists to avoid. Reverted, with the finding left in the code |
| traced prefill | would remove 91–98 ms of a ~134 ms TTFT | not taken: the prefill program set is keyed by *logical* prompt length, so it needs a decoder-owned contract change (§6) |
| force-argmax greedy | would all-gather 249856-wide logits + global `ttnn.argmax` | rejected by construction, as before (§4.4) |

---

## 10. `tt-perf-report`

[`tracy/`](tracy/), regenerated by `tracy/run_profiling.sh` on the final code, for the **reduced
profiling variant** (one real `linear_attention` layer, one real `full_attention` layer, real weights,
real cache/page-table shapes, the real terminal path and the real traced decode) — the shape
`$full-model` and `$optimize` both ask for, because a 40-layer capture is ~3300 device ops per step
and overruns the profiler.

Op shares of the signposted decode window, before → after. **Both directions are shown**, because the
top-k saving was partly spent on the grouping that produced it (§4.1):

| op group | before | after | |
|---|---|---|---|
| `TopKDeviceOperation` (sampler stages 1+2) | 29.98 % | **24.76 %** | |
| `MatmulDeviceOperation` (LM head + dense projections) | 20.69 % | 21.63 % | |
| routed `SparseMatmul` (gate/up + down) | 9.10 % | 9.56 % | |
| `BinaryNg` | 7.30 % | 7.74 % | |
| `SliceDeviceOperation` | 3.70 % | **5.45 %** | ← the 32-way grouping
| `ReduceScatter` + `AllGather` (`ttnn.all_reduce`'s two per layer) | 5.09 % | 5.24 % | |
| `ConcatDeviceOperation` | 1.15 % | **2.27 %** | ← the same
| `LayerNorm` | 2.78 % | **2.26 %** | ← the terminal norm off its single core
| `GatherDeviceOperation` (stage-2 index recovery) | 1.60 % | 1.89 % | |
| `AllGatherAsync` (the sampler's candidate gather, through the shim) | 0.96 % | 0.89 % | |

Whole-window device time falls 9449.9 → 9109.5 µs, i.e. **−85.1 µs/replay**, against an **−88.8
µs/token** wall delta on the 40-layer model — the ~1:1 ratio §4.1 uses. Every one of those figures,
including the wall delta, is generated into
[`logs/sampler_cost_model.md`](logs/sampler_cost_model.md) from the two stacked CSVs and the two perf
summaries. The *before* capture is the full-model stage's committed one rather than a re-take, and the
same generated file prints the op families this stage does not touch so that choice is checkable:
`SparseMatmul` 859.70 → 870.68 µs, `ManualSeed` 73.43 → 73.93, `Untilize` 77.65 → 77.66. Read the shares with the denominator in
mind: the reduced variant has 2 layers, not 40, so its terminal path is ~20x over-represented. The
same `TopK` rows are ~2.4 % and the same LM head ~1.6 % of the delivered 40-layer step.

Key rows and the advice, all of it addressed:

| row | measurement | advice, and what was done |
|---|---|---|
| LM head `32 x 2048 x 62464` | 374 µs, 109 cores, 353 GB/s, **69.0 % DRAM**, `HiFi2 BF16 x BFP8 => BF16` | *"try a DRAM-sharded program config"* — **tried, slower, §3.2**. *"use HiFi4 for full accuracy"* — HiFi4 measured, a tie, and it asks for higher fidelity than the selected policy (§3.3) |
| terminal `LayerNorm` | **6 µs on 8 cores** (was ~20 µs on 1) | fixed this stage, §3.1 |
| `TopK` stage 1 / stage 2 | 369 µs @ width 1952 on 32 cores / 195 µs @ width 1024 on 1 core | 0.189 / 0.190 µs per width unit — the cost model in §4 |
| grouping `Slice` + `Concat` + `Gather` | 5.52 µs/replay per group → ~177 µs/replay at `g = 32` (fitted; the raw op total, 219, also contains the decoder's own slices) | named as the next sampler lever with its size, §4.2 |
| dense decode projections (`32 x 2048 x 3136` etc.) | 19 µs; the four committed rows read 337–345 GB/s, 65.7–67.4 % DRAM | inherited geometry, swept by the decoder stage; `Output subblock 1x1 is small` is answered there (`per_core_N = 1` is the winning point of a 9-target x 6-cap ladder) |
| routed `SparseMatmul active=4/64` | 65 µs / 44 µs mean over the window (first rows quote 70 / 46), `LoFi BF16 x BFP4 => BFP8` | inherited; `in0_block_w=32` / `16` and the subblocks are the winners of two prior sweeps, and the op's parallelism cap is a named limitation |
| op-to-op gaps | *"could save 261 µs (2.4 %)"* | the window is already traced; the gaps are the model→sampling trace boundary and profiler instrumentation. Merging the traces is rejected in §9 |

**The prefill capture failed**, twice, on the profiler's post-processing rather than on device:
`Device data missing: Op 352259 not present in cpp_device_perf_report.csv` at the shipped
`--op-support-count`, and unbounded post-processing (killed at 38 minutes, ~2 GB RSS and growing)
above it. [`tracy/prefill_capture_failure.txt`](tracy/prefill_capture_failure.txt) records both
signatures. This is exactly `doc/full_model/README.md` limitation 2 for this model, in both of its
shapes; prefill is also the one phase this stage did not change, and
`doc/full_model/tracy/prefill_perf_report.txt` is the committed prefill table for the same reduced
variant. The prefill conclusions here come from wall-clock measurement instead (§6).

---

## 11. Known limitations

1. **Warmed TTFT is host-drift-dominated at the ±7 ms level, so this stage cannot claim to move it**,
   and 69–68 % of it is eager-dispatch overhead. Three nine-repeat same-code pairs
   came out at +2.7, −7.2 and +1.0 ms on the median (§6); the one attributable component is +0.303 ms of eager first-token sampling. The
   length-independent share is 67.8 % by a two-point secant
   (intercept 88.9 ms) or 68.5 % by
   four-point least squares (89.8 ms), both computed by the probe into
   `prefill_profile.json`. Traced prefill is the fix and it is blocked on a decoder-owned contract (§6).
   The cold-length half of the problem *is* fixed and measured (§6.1).
2. **The 40-layer device-time decode is not measurable.** Full-stack profiling is forbidden by both
   skills and impossible in practice here, and the reduced two-layer capture's per-replay device time
   is inflated enough by the profiler that scaling it exceeds the un-profiled 40-layer wall clock. The
   accounting in §1 is therefore roofline + wall clock, with `decode_ms_per_token_device` explicitly
   `null` and the reason recorded in `perf_summary.json`.
3. **The prefill `tt-perf-report` capture is unavailable** (§10), inherited.
4. **The decode step sits at 6.3 % of the DRAM roofline** because it is launch-bound, not
   bandwidth-bound: ~100 device ops per layer at one tile of M, and `ttnn.sparse_matmul`'s parallelism
   is capped by its output tile count. Both are inherited and named; neither is attackable without the
   decoder's policy freedom.
5. **The sampler's grouping machinery is ~177 µs/replay of pure op-launch overhead** (5.52 µs/replay
   per group, fitted in `logs/sampler_cost_model.md`) — about **20 %** of the sampler's ~880 µs/replay
   of device time — and the arithmetically-free reshape that would replace it is a shared-`TTSampling`
   change, quantified but not made (§4.2). It is the largest remaining named item in the measured path,
   at ~0.75 % of a step, and removing it would also shrink §6's eager first-token cost.
6. **The first token after a prefill is sampled untraced**, 3.350 ms inside TTFT, of which 0.303 ms is
   this stage's own cost (§6). The traced version wedged the mesh and is documented in §9 rather than
   retried.
7. **`logs/probe_bisect.py --order after` can now wedge the mesh**, where the full-model stage
   recorded it returning token 0. It is the deliberate reproducer for the post-capture compilation
   hazard, it is not in the delivered path (the generator's guard is), and it has been removed from
   `logs/run_evidence.sh` with the reason next to it. `work_log.md` §5.2 has the triage. No upstream
   issue has been filed for the escalation.
8. **An EOS-terminated `generate` leaves device state one position ahead of the returned tokens**,
   because the lookahead step executes before EOS is seen (§2). Harmless — every `generate` resets
   before it prefills, and `test_the_pipelined_loop_stops_on_eos_and_leaves_state_one_position_ahead`
   asserts both halves (the loop stops, exactly one speculative replay happens, and the following
   request still reproduces) — but it is a real behavioural difference from the serial loop.
9. **Non-greedy sampling is still wired and smoke-tested, not measured**, and `LogProbsCalculator`
   still does not support a 1x4 mesh. Inherited, unchanged.
10. **`models/common/tests/test_sampling.py::test_log_probs_calculation` still fails on this mesh**,
    before and after, for the same 8-or-32-device reason. Inherited, unrelated.
11. **Watcher runs with `TT_METAL_WATCHER_DISABLE_ETH=1`**, so ACTIVE_ETH cores are not covered on
    this configuration. Inherited from the decoder stage; every worker-core assert stays armed.
12. **One mesh shape.** `DEFAULT_MESH_SHAPE = (1, 4)`.
13. **An inactive row's RoPE index still advances while its position does not.** Inherited limitation
    7 of the full-model stage, unchanged and still harmless.

---

## 12. Exact artifacts

```
models/autoports/ornith_ai_ornith_1_0_35b/
├── tt/model.py                     terminal path: LM-head program configs and K-block/fidelity
│                                   knobs, sharded final norm, vocabulary alignment, best_topk_groups
├── tt/generator.py                 pipelined readback, warmup(), the reverted-trace-write finding
├── tests/test_full_model.py        48 fast cases (6 new) + 5 long
└── doc/
    ├── context_contract.json       recomputed into its own `optimized_full_model` block
    └── optimized_full_model/
        ├── README.md                       this file
        ├── work_log.md                     what was done in order, including two hangs
        ├── perf_summary.json               §1, §6; includes `performance_accounting`
        ├── perf_summary_before.json        the inherited path re-measured on this checkout
        ├── prefill_profile.json            §6, the TTFT ladder and the logging A/B
        ├── footprint.json                  §7.2, measured at the advertised context
        ├── footprint_batch32.json          §7.2, the 40-layer build at the batch bound
        ├── long_prompt.json                §7.2, non-aligned prompts up to 262143
        ├── batch_slots.json                the batch-4 slot contract
        ├── readiness_{prefill,teacher}.json               §1 accuracy
        ├── readiness_{prefill,teacher}_bfp4head.json      §3.4, the rejected dtype arm
        ├── readiness_autoregressive{,_chat}.json          §5
        ├── readiness_qualitative.json                     §5, the shared suite + HF control
        ├── degenerate_report.json                         §5
        ├── logs/
        │   ├── run_evidence.sh             regenerates everything behavioural, in order
        │   ├── run_evidence_status.txt     14 steps; 3 HF-reference steps OOM-killed, §5
        │   ├── host_memory_event.txt        that host condition, with /proc/meminfo
        │   ├── check_prose_figures.py,
        │   │   check_prose_figures.txt      the figure gate and its passing output (67 rows, 19 literals)
        │   ├── watcher_report.sh            §5, generates watcher/watcher_error_count.txt
        │   ├── ab_terminal.{py,sh}         §3, the 19-arm terminal ladder
        │   ├── ab_terminal_table.md,
        │   │   make_ab_table.py            that ladder's table and its generator
        │   ├── ab_terminal_kblock.txt,
        │   │   ab_terminal_kblock_table.md §3.3, the in0_block_w x norm-grid x fidelity cross
        │   ├── sampler_cost_model.md,
        │   │   make_sampler_cost_model.py   §4.1-4.2, generated from the two stacked Tracy reports
        │   ├── bench_full_model.py         §1, §6, the performance accounting
        │   ├── probe_prefill.py            §6
        │   ├── probe_terminal.py           the terminal-cost breakdown
        │   ├── probe_footprint.py, probe_long_prompt.py, update_context_contract.py   §7.2
        │   ├── probe_multi_prompt.py, probe_bisect.py, probe_batch_slots.py
        │   ├── run_readiness.py, run_watcher.sh
        │   └── *.txt                       each script's committed output
        ├── tracy/                          §10, run_profiling.sh + the decode report
        ├── watcher/                        §5
        └── triage/                         §9, the two hangs this stage caused
```

Reproduce, in this order. The profiler, the watcher and the two ladders each need a device session of
their own, so they are separate commands:

```bash
R=models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_full_model
bash $R/logs/run_evidence.sh                # readiness, qualitative, bench, probes, fast suite
bash $R/logs/ab_terminal.sh                 # separate run: the terminal-path ladder
python $R/logs/make_ab_table.py             # its table
python $R/logs/make_sampler_cost_model.py    # the sampler cost model in §4.1-4.2
python $R/logs/bench_full_model.py --arm inherited --repeats 9 --output $R/perf_summary_before.json
python $R/logs/bench_full_model.py --repeats 9              # the optimized arm, same session
python $R/logs/check_prose_figures.py        # every figure in this file, against its artifact
                                            # (run AFTER refreshing the docs from the sweep)
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -m long -q
python $R/logs/probe_long_prompt.py --budget-s 2400
python $R/logs/probe_footprint.py --cache-context 262144 --batch 1  --output $R/footprint.json
python $R/logs/probe_footprint.py --cache-context 8192   --batch 32 --output $R/footprint_batch32.json
python $R/logs/update_context_contract.py
python $R/logs/run_readiness.py --check prefill teacher --lm-head-dtype bfp4 --json-suffix _bfp4head
bash $R/tracy/run_profiling.sh              # separate run: profiler
bash $R/logs/run_watcher.sh                 # separate run: watcher
```

The `in0_block_w` × norm-grid × fidelity cross in §3.3 is the arm list in `work_log.md` §3.4, driven
through `logs/ab_terminal.py` with `--terminal-norm-cores`, `--lm-head-in0-block-w` and
`--lm-head-fidelity`.

Commit SHAs are at the end of [`work_log.md`](work_log.md).
