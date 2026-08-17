# Ornith-1.0-35B — optimized full model: work log

Chronological. What was measured, in what order, and what each measurement decided. The delivered
result and the tables are in [`README.md`](README.md); this file is the reasoning and the dead ends,
including two device hangs that this stage caused and recovered from.

**Every number in this file is read out of the committed artifact named beside it.** Where an earlier
run of the same probe produced a different figure, the committed one is used and the spread is stated,
because two of this stage's metrics (TTFT and the teacher-forcing harness) have a run-to-run band
wider than any change this stage makes.

Hardware for every number below: four Blackhole `p300c` chips as a `1x4` ring under
`FabricConfig.FABRIC_1D_RING`. Branch `agentic-research/hous/ornith-1.0-35B`, starting from
`ecd20ae2e39` (the full-model stage's last commit).

---

## 1. What this stage inherited, and where the time actually was

`doc/full_model/perf_summary.json` records what the previous stage delivered: **41.86 t/s/u** token-out
decode (23.890 ms/token) on the whole 40-layer model, split three ways by its own accounting —

| term | ms/token | share of the step |
|---|---|---|
| decoder-layer stack lower bound (30 x 0.564 + 10 x 0.453) | 21.450 | 89.8 % |
| embedding + final norm + LM head + device `plus_one` | 0.812 | 3.4 % |
| sampling trace | 1.172 | 4.9 % |
| host synchronize + caller token readback | 0.455 | 1.9 % |

So **90 % of a decode step is the decoder layer stack**, whose dtype/fidelity/KV/CCL/residual policy
this stage is required to preserve, and the whole optimizable surface is the remaining 2.44 ms. That
framing decided the plan: this stage is not going to move the headline by tuning the terminal path
alone, and the honest targets are (a) the ~0.45 ms of host stall, which is not work at all, (b) the
1.17 ms sampler, and (c) the 0.81 ms terminal arithmetic — in that order of expected size.

The baseline was then **re-measured on this checkout** rather than quoted, and by the end of the stage
that re-measurement was rebuilt as a same-script `--arm inherited` reconstruction with nine repeats
(§6): [`perf_summary_before.json`](perf_summary_before.json) reads 23.879 ms/token and TTFT 133.13 min /
139.08 median ms. Against the previous stage's archive that is 0.012 % on `traced_logits_only_decode`,
0.002 % on `traced_decode_plus_sampling_no_readback` and 0.043 % on `token_out_decode`,
which is the row that carries the host loop and so the one most exposed to host state — close enough to
be the same path, and the per-row numbers are stated rather than rounded into a single tolerance because
an earlier draft claimed "0.02 % on every decode row" and the fifth review caught it.

Two subagents mapped the code before anything was written: one over the routed-MoE decode path in
`tt/multichip_decoder.py` / `tt/optimized_decoder.py`, one over `tt/model.py` + `tt/generator.py`.
The first mapping is what kept the stage out of the decoder: the routed `sparse_matmul` geometry has
already been swept twice (`doc/optimized_decoder/logs/probe_sparse_matmul.txt`,
`doc/multichip_decoder/logs/probe_sparse_matmul_local.txt`), the `nnz` question already has a
committed reproducer and an upstream-worthy hang, and `SPARSE_*` is pinned by
`test_sparse_cores_match_the_local_sweep`. The second produced the four candidate levers below.

---

## 2. The host stall: the largest single win, and it is not device work

The functional decode loop was:

```python
self._decode_step_traced()          # execute_trace(..., blocking=False)
self._sample_traced()               # sampling trace, also non-blocking
ttnn.synchronize_device(...)        # <- the host blocks here, every token
predicted = int(self._read_tokens()[0])
```

Both replays are already non-blocking, so the device is never *waiting* for the host inside a step —
but it is idle between steps, for as long as the synchronize plus the readback plus the Python around
them takes.

The reason it can be removed is that the **steady-state free-running loop has no host->device
dependency at all**. The sampled token reaches the next replay through `tt_out_tok` on device, and
`current_pos`/`rot_idxs` advance with `ttnn.plus_one` inside the captured graph. Step N+1 therefore
does not need to know token N. So:

```python
self._decode_step_traced(); self._sample_traced()      # step N
in_flight = self._read_tokens_async()                  # cpu(blocking=False) + record_event, on cq0
...next iteration enqueues step N+1...
predicted = int(self._finish_read(pending)[0])         # event_synchronize, then to_torch
```

The ordering is what makes it correct rather than racy: the read is enqueued on the same command
queue *between* step N's sampling and step N+1's replay, and the queue is in order, so it observes
exactly step N's token even though step N+1 has already been submitted. The host's wait then overlaps
device work that is already running. It is the same idiom
`models/tt_transformers/tt/generator.py::read_decode_output(async_read=True)` uses.

Measured in one build on the delivered 40-layer model ([`perf_summary.json`](perf_summary.json)):

| loop | ms/token | t/s/u |
|---|---|---|
| serial (`synchronize_device` + readback per token) | 23.864 | 41.90 |
| pipelined | 23.300 | 42.92 |

**0.564 ms/token**, and the serial arm reproduces the inherited arm's 23.879 to 0.06 %, which is what
makes it a same-build difference rather than a cross-run one. It also makes
`full_model_only_cost.sync_and_readback_ms` go *negative* (−0.045 ms): the token-out step is now at
the replay-plus-sampling throughput, i.e. the caller's readback is entirely hidden.
`steady_state_counters` reports `decode_syncs: 0` against `decode_calls: 127`.

Teacher forcing keeps the serial loop **by construction** and the code says so: `next_input` decides
step N+1's token input on the host from token N, so there is nothing to overlap.
`test_teacher_forcing_keeps_the_serial_loop` asserts `perf["pipelined_readback"] is False` and
`decode_syncs == decode_calls` for that path, and `test_the_pipelined_readback_agrees_with_the_serial_loop`
asserts the two loops are token-for-token identical on the same prompt.

**The one-token lookahead has a state consequence, and it is not "nothing".** On an EOS stop the loop
sees EOS in token N only after step N+1 was enqueued, so that step executes: it consumes the EOS token
as input, writes one paged-KV entry at the next position and advances `current_pos`/`rot_idxs` by one.
The returned list is unchanged, but device state is one position ahead of it. Harmless here — every
`generate` resets before it prefills, and the low-level API is not pipelined — and it is why
`read_waits == decode_calls` is asserted on a run to length (`stop_on_eos=False`) rather than on an
EOS-terminated one. README limitation 8.

---

## 3. The terminal path: two ladders

`logs/ab_terminal.py` is one arm per process, `logs/ab_terminal.sh` is the driver for the whole
ladder, `logs/ab_terminal.txt.gz` is the raw output and
[`logs/ab_terminal_table.md`](logs/ab_terminal_table.md) is the table `logs/make_ab_table.py`
generates from it. (The ladder was originally run as four rounds, because the first `mcast1d` and
`dram_sharded` arms exposed an output-handling bug in `_lm_head_block` and the alignment cross was
added after §4 was understood; `ab_terminal.sh` is those rounds consolidated into one script.) Every
arm is the **reduced two-layer variant** so a build costs ~15 s instead of ~200 s; the rows are
absolute milliseconds, so a difference on the probe is the same difference on the 40-layer model.

Four knobs were crossed in the first ladder: LM-head matmul spelling, LM-head core count,
terminal-norm sharding, and vocabulary alignment (which is really a sampler knob — §4). The
interesting rows:

| arm | model trace | sampling trace | token-out (pipelined) |
|---|---|---|---|
| baseline: bare `ttnn.linear`, unsharded norm | 1.475 | 1.181 | 2.657 |
| terminal norm width-sharded, bare `ttnn.linear` | **1.650** | 1.181 | 2.857 |
| `mcast1d`, 110 cores, unsharded norm | 1.480 | 1.181 | 2.668 |
| `mcast1d`, 110 cores, sharded norm | **1.434** | 1.181 | 2.646 |
| `mcast1d`, 88 cores, unsharded norm | 1.480 | 1.181 | 2.667 |
| `mcast1d`, 64 cores, sharded norm | 1.477 | 1.181 | 2.688 |
| `dram_sharded`, 64 cores, sharded norm | 1.553 | 1.133 | 2.704 |
| **`mcast1d` 110 + sharded norm + align 32 (shipped)** | **1.435** | **1.121** | **2.583** |

Three findings, none of which was the expected one.

**3.1 Sharding the terminal norm on its own is a regression, not a win.** The functional model's
`ttnn.rms_norm` had no program config and no memory config, and the decode capture shows it running
on a **single core at ~20 us** while the decoder's own in-layer norms run on 8 cores at 6 us. The
obvious fix — width-shard it over 8 cores with a `LayerNormShardedMultiCoreProgramConfig` — makes the
model trace 0.175 ms *slower* (1.650 vs 1.475), because the bare `ttnn.linear` that follows cannot
use a width-sharded activation and the layout has to be undone again. The norm is only worth sharding
once its consumer wants that layout, which is why `terminal_norm_sharded=None` resolves to "shard iff
the head was given a program config" rather than to `True`.

**3.2 `tt-perf-report`'s own advice for the LM head loses.** The DRAM-sharded spelling was built
following `models/common/modules/lm_head/lm_head_1d.py` and measures 1.553 against 1.435 ms, and
below 64 cores it does not build at all (circular buffers grow to 2,220,416 B against 1,572,864 B of
L1). README §3.2 has the full reasoning. Two follow-ups came out of the stage review of this arm:

* the spelling now **refuses** a `lm_head_cores` that does not divide `dim/32`, instead of silently
  falling back to the untuned `ttnn.linear` while still paying its vocabulary padding. With the
  shipped default of 110 cores that pair was exactly the silent-degradation case;
* `build_generator`'s docstring named `"dram_sharded"` as the default and did not mention `"mcast1d"`
  at all. Corrected, along with the other new knobs.

**3.3 What wins is the decoder's own decode geometry, and it needs the sharded norm.**
`MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=True`, `fuse_batch=True`,
`in0_block_w=8`, `per_core_M=1`, `per_core_N=18`, output subblock `1x6`, over the whole 11x10 worker
grid, reading the width-sharded L1 activation the norm now produces: 1.434/1.435 against 1.475. The
core count matters and 110 is the top of it — 88 and 64 both measure 1.477-1.480.

The gain is small and the report explains why: in the optimized decode capture the LM head is
**374 us at 353 GB/s = 69.0 % of the DRAM roofline on 109 cores** — essentially where it was
(370 us / 355 GB/s / 108 cores), because it was already DRAM-bound and reading 2048 x 62464
bfloat8_b per token is the floor. What the tuned spelling actually buys is the *norm*: 6 us on 8
cores instead of ~20 us on 1, without paying a layout conversion to get there.

### 3.4 The second ladder: `in0_block_w` x norm grid x fidelity

The first ladder never varied the terminal matmul's K block or its math fidelity, which
`$optimize`'s checklist requires for the largest decode-time consumer — and the reason first written
down for `in0_block_w=8` ("the largest legal divisor of dim/32 = 64 under the cap") was wrong: there
is no cap. With `mcast_in0` and a width-sharded `in0` the matmul blocks the inner dimension out of
what each core *holds*, validating `in0_shard_tiles % in0_block_w == 0` as well as
`k_tiles % in0_block_w == 0`. The terminal norm shards `dim` over 8 cores, so each core holds 8 tiles
and 8 is the maximum **for that grid**; 16, 32 and 64 need a narrower norm grid. That is the OPT-011
trade, so it was measured. `logs/ab_terminal_kblock.txt` ->
[`logs/ab_terminal_kblock_table.md`](logs/ab_terminal_kblock_table.md), arms:

```
BASE="--lm-head-program mcast1d --lm-head-cores 110 --terminal-norm-sharded 1 --vocab-align-tiles 32"
k8-n8-hifi2      $BASE --terminal-norm-cores 8                              -> 1.434  (shipped)
k4-n8-hifi2      $BASE --terminal-norm-cores 8 --lm-head-in0-block-w 4      -> 1.442
k8-n4-hifi2      $BASE --terminal-norm-cores 4 --lm-head-in0-block-w 8      -> 1.437
k16-n4-hifi2     $BASE --terminal-norm-cores 4                              -> 1.476
k16-n2-hifi2     $BASE --terminal-norm-cores 2 --lm-head-in0-block-w 16     -> 1.463
k32-n2-hifi2     $BASE --terminal-norm-cores 2                              -> does not build (L1)
k64-n1-hifi2     $BASE --terminal-norm-cores 1                              -> does not build (L1)
k8-n8-lofi       $BASE --terminal-norm-cores 8 --lm-head-fidelity lofi      -> 1.466
k16-n4-lofi      $BASE --terminal-norm-cores 4 --lm-head-fidelity lofi      -> 1.472
k8-n8-hifi4      $BASE --terminal-norm-cores 8 --lm-head-fidelity hifi4     -> 1.437
interleaved-k64  --lm-head-program interleaved --terminal-norm-sharded 0 --vocab-align-tiles 32 -> 1.476
```

Two caveats the second review raised and which belong here rather than in a footnote: the selection
was made on **model trace**, where the shipped arm leads `in0_block_w=4` by 0.008 ms and HiFi4 by
0.003 ms, while the same table's **token-out pipelined** column slightly favours those rejected arms
(2.580 / 2.582 against the shipped 2.586) and the first ladder's repeat pairs show 0.001-0.003 ms of
spread; and terminal-norm grids **wider** than 8 cores (16, 32) are legal and unmeasured, which would
force `in0_block_w` down to 4 and 2. Both are within 0.03 % of a decode step, so the row is a near-tie
rather than a clear win, and README §3.3 says so.

The two that do not build fail with an exact blocker, which is what the sweep was looking for:
`Statically allocated circular buffers in program 466 clash with L1 buffers on core range
[0-0 - 10-8]. L1 buffer allocated at 1404928 and static circular buffer region ends at 1532864` (and
`1273856` for the 1-core arm). So the shipped geometry is the measured winner of a
`{4,8,16,32,64} x {1,2,4,8} x {LoFi,HiFi2,HiFi4}` cross, and LoFi is rejected on latency — a
bandwidth-bound row gains nothing from lower fidelity — rather than on preference.

### 3.5 The LM-head weight dtype

`lm_head_dtype=bfloat4_b` is the fastest arm in either ladder (1.351 ms of model trace against 1.435,
i.e. 0.084 ms/token, 0.36 % of a decode step) and is **rejected on real-checkpoint accuracy**: top-1
0.920 / 0.940 against the shipped 0.940 / 0.970 on the two readiness gates, both arms measured on the
final tree (`logs/readiness_bfp4_head.txt`, re-run after the third review pointed out that the first
bfp4 arm predated the delivered code). Both clear the stage bars, and the 2-3 token gap is the same
order as the near-tie churn README §1 describes - which is the second reason the decision is "keep the
decoder's dense-projection dtype, as the goal contract requires" rather than "bfp4 is two points
worse". It goes to `$datatype-sweep` with its numbers.

---

## 4. The sampler: the group count is derived, and it is at the joint optimum

The full-model stage's own measurement is that `ttnn.topk` is linear in the reduced width and
independent of every other dimension, and its grouped local top-k replaces one 62080-wide reduction
with a `62080/g`-wide one over `g` rows plus a `32g`-wide one over the winners. It picked `g = 20` as
"the measured optimum among the divisors of 1940 (= 62080/32)".

That is right for *that* width, and the reduction cost model says the optimum is near
`g = sqrt(W / max_top_k)` ≈ 44 — but 1940 factors as 2^2*5*97, so its divisors jump straight from 20
to 97 and skip the whole neighbourhood. `g = 20` costs `62080/20 + 32*20 = 3744` width units.

So the padding is aimed at the *factorisation*, not at the matmul: aligning the per-device vocabulary
to 32 tiles makes it 62464 = 32 x 1952, and 1952 = 2^5 x 61 has 32 as a divisor —
`62464/32 + 32*32 = 2976` units. `OrnithModel.best_topk_groups` computes this from whatever width the
build produced, and `topk_num_groups="auto"` is the default; the rule reproduces the full-model
stage's 20 for the unpadded 62080 shard, which is how it is checked
(`test_grouped_local_topk_matches_a_single_reduction` compares against
`model.best_topk_groups(max_top_k)` and asserts every group width is tile aligned).

Measured: the sampling trace goes **1.181 -> 1.121 ms** on the probe (reproduced in four arms) and
**1.169 -> 1.135 ms** on the delivered model, same sweep, and `TopKDeviceOperation` falls from 29.98 % to 24.76 %
of the reduced decode window. The capture confirms the reduction model exactly: stage 1 is 369 us at
width 1952 on 32 cores and stage 2 is 195 us at width 1024 on **1** core, i.e. 0.189 and 0.190 us per
width unit — linear in width, indifferent to cores — and 2976 x 0.19 = 565 us against 564 measured.

### 4.1 The correction the first stage review forced: grouping is not free

The first version of this section rejected a third reduction stage using a "60 us of wall per 144 us
of top-k device time" ratio, i.e. a 0.42 factor. That was wrong. The second review then found that the
corrected table still mixed op bases (a dram-only "before" against a dram-plus-L1 "after") and quoted
a couple of figures that did not reproduce. Both rounds are fixed the same way: the table and the model
are now **generated** from the two committed stacked Tracy reports by
`logs/make_sampler_cost_model.py` into [`logs/sampler_cost_model.md`](logs/sampler_cost_model.md),
summing per op code across every memory-layout variant, so nothing here is hand-typed and it cannot
drift again.

| op code | before (us/window) | after (us/window) | delta per replay |
|---|---|---|---|
| `TopKDeviceOperation` | 2,833.2 | 2,255.3 | **−144.5** |
| `SliceDeviceOperation` | 350.1 | 496.2 | **+36.5** |
| `ConcatDeviceOperation` | 109.1 | 206.3 | **+24.3** |
| `GatherDeviceOperation` | 150.7 | 172.6 | **+5.5** |
| `BinaryNg` *(reported, not fitted)* | 689.5 | 704.9 | +3.8 |
| **sampler net** | | | **−78.2** |
| **whole window** | 9,449.9 | 9,109.5 | **−85.1** |

Two coefficients come out of that: the reduction costs **0.188 us per width unit** and the grouping
machinery costs **5.52 us/replay per group**, i.e. ~**177 us/replay** at g = 32. And the whole-window
device delta (−85.1 us/replay) matches the −88.8 us/token wall delta on the 40-layer model, so device and
wall time move at roughly **1:1**, not 0.42.

```
device(g) ~= 0.188 * (W/g + 32g) + 5.52 * g + const      minimised at g = 31.9 for W = 62464
```

So the shipped `g = 32` is the **joint optimum**, not merely the best legal divisor. The model
reproducing the 20 -> 32 move it was solved from is arithmetic, not validation; what checks the
reduction coefficient independently is the two `TopK` rows of the after report on their own — 369 us at
width 1952 and 195 us at width 1024, i.e. 0.189 and 0.190 us per width unit against the 0.188 fitted.

### 4.2 The third stage, and further padding, are both refuted by that model

| candidate | reduction | machinery | extra ops | net |
|---|---|---|---|---|
| pad to 1980 tiles, `g = 44` (+896 columns/device) | −24 us | **+66 us** | +4 us of LM head | **+46 us worse** |
| three-stage, `g = 99` / `h = 11` (1280 units) | −319 us | **+431 us** | +43 us second index gather | **+155 us worse** |

The sign does not depend on the split: the machinery term is linear in the group count while the
reduction term is only `W/g`. (The earlier draft of the `g = 44` rejection row in README §9 quoted only
the reduction and LM-head terms, which argued *for* the change; that was the same
grouping-counted-as-free mistake, left standing in one row after §4.1 was corrected, and the second
review caught it.)

The real remaining lever is the machinery itself: ~177 us/replay at g = 32, a third of the sampler's
device time - the sampler's per-replay device total is ~880 us (TopK 564 + machinery 177 + sampling 27
+ manual seed 18 + the two async gathers 20 + the mask and typecasts), so about a **fifth**, not a third
as an earlier draft said - and all of it is op-launch overhead for what is arithmetically a free
reinterpretation (`[1,1,32,W]` viewed as `[1,32,32,W/32]`, every group on a tile boundary). Replacing 32 slices plus a
concat with one reshape is a change to shared `TTSampling._local_topk_grouped` and to its index
recovery; it is named in README §4.2 and limitation 5 with its size (~0.18 ms/token, ~0.75 % of a step
at the measured 1:1 ratio) instead of being left implicit.

Two smaller sampler items were looked at and left alone deliberately:

* `ManualSeedDeviceOperation`, 18 us every step, is dead work for greedy — but `SamplingGenerator`
  keys traces by `(penalties, logprobs, force_argmax)` and **not** by `k`, so a trace captured without
  the seed would be replayed for a sampled request. Removing it would trade 0.08 % of a step for
  silently non-random sampling.
* the sampler's two candidate all-gathers cannot take persistent output buffers: the full-model stage
  found the op-contract conflict (`TTSampling` deallocates the gather's output tensor at the end of
  every call, so a caller-supplied persistent buffer is freed by the first call), and nothing about
  that changed here.

`allow_force_argmax` stays off. Greedy is the same captured graph every other sampling mode uses —
local top-32 per vocabulary shard, gather the 4 x 32 candidates, `ttnn.sampling` with `k=1` — which is
the semantically greedy split-sampling path the skills require, not a top-k-32 stand-in.

---

## 5. Two hangs this stage caused, and what they cost

### 5.1 Writing into the trace region to make the first token traced

TTFT is 97 % prefill, and the remaining 3.350 ms is `_first_token_after_prefill` running the sampler
**untraced** — eager dispatch for work the decode loop's captured sampling trace does in 1.12 ms on
the reduced probe (1.135 ms on the delivered model). The obvious fix is to copy the
prefill logits into `self._trace_logits` (the tensor the sampling trace was captured against, so
replay stays valid) and replay that trace.

It hung the mesh. `tt-triage` found all four devices stuck on the same `ReshapeViewDeviceOperation`
and `check_binary_integrity` reporting `.text` mismatches against the cached kernel ELFs —
[`triage/tt-triage.txt`](triage/tt-triage.txt), [`triage/triage-summary.txt`](triage/triage-summary.txt),
[`triage/README.txt`](triage/README.txt).

The mechanism is one the full-model stage already documented from the other side:
`SamplingGenerator.capture_trace(skip_precompile=True)` exists precisely because executing a sampling
graph over the live trace-region buffer hung this mesh. `_trace_logits` is allocated *inside* the
trace region during capture, and writing to it from outside a replay is the same hazard. Reverted;
`_first_token_after_prefill` now carries the finding as a comment so the next reader does not retry
it. Recovery: kill, `tt-smi -ls --local` (8 boards) -> `tt-smi -r` -> `tt-smi -ls --local` (8 boards)
-> mesh smoke `MESH_SMOKE_OK`. Cost: one wedged mesh and one wasted benchmark run.

### 5.2 The §5.1-of-the-full-model repro now wedges the mesh instead of corrupting output

`logs/probe_bisect.py --order after` is the full-model stage's minimal reproducer for the
post-trace-capture compilation hazard: it deliberately replays a trace whose kernel binaries were
overwritten, bypassing `_ensure_traces_replay_safe`, and the full-model stage recorded it returning
token 0. Inside this stage's first evidence sweep it did not return at all — the log stops at
`REPLAY after both prefills` and the process spun for 39 minutes. `tt-triage` could not even attach
this time: it failed inside `wait_eth_core_training` / `TopologyDiscovery::get_connected_devices`
([`triage/bisect_after/console.txt`](triage/bisect_after/console.txt), explained in
[`triage/bisect_after/README.txt`](triage/bisect_after/README.txt)).

That is a *worse* manifestation of the same known hazard, not a new defect, and it is not in the
delivered path — the generator's guard is what stops it, and `probe_bisect.py --order before` still
reproduces both prompts correctly (`logs/probe_bisect_before.txt`). But an evidence script that can
wedge the node is not something to run unattended, so the `after` arm was removed from
`logs/run_evidence.sh` with the reason written next to it; the probe is still there to be run
deliberately. Recovery was the same bounded sequence: 8 boards -> reset -> 8 boards -> mesh smoke OK.
The partial log is kept as `logs/probe_bisect_after.txt`.

The committed `logs/run_evidence_status.txt` is from a later run of the script without that step, so it
carries no dangling header. It is **not** clean, and §8 says why: 14 steps, of which the three
HF-reference readiness steps were OOM-killed by the host, `done (3 failed)`. That is a host-memory
condition with its own note (`logs/host_memory_event.txt`), not this hazard and not a model result.

---

## 6. TTFT: decomposed, and dominated by host drift

The first three drafts of this section argued TTFT was flat, twice from figures a later run had
superseded and once from a spread band that was two-thirds not in the artifacts; the fourth draft, from
a proper same-session nine-repeat pair, said +2.7 ms and called it a real cost. The final sweep re-ran
that pair as steps 6 and 7 of `logs/run_evidence.sh`, and it came out **the other way**:

| arm | TTFT samples (ms, sorted) | min | median | max |
|---|---|---|---|---|
| inherited | 133.1 134.4 134.7 137.0 **139.1** 140.7 147.5 149.8 150.1 | 133.1 | **139.1** | 150.1 |
| optimized | 133.4 135.7 138.3 139.1 **140.1** 140.5 141.9 145.2 146.1 | 133.4 | **140.1** | 146.1 |

Two nine-repeat same-code pairs, opposite signs (+2.7 ms, then −7.2 ms, then +1.0 ms), against a device-work bound of
**2.4 us** (384 more LM-head columns per device at 353 GB/s) plus one `interleaved_to_sharded` on a
32-row block. So the conclusion is neither "flat" nor "worse" nor "better": **TTFT on this host is
host-drift-dominated at the +-7 ms level and this stage cannot claim to move it**, which is what §1 and
limitation 1 now say. Chasing a signed delta through four drafts was the mistake; the fix was to measure
both arms with the same script in the same sweep and then report what that shows.

The breakdown *is* stable, because each arm measures it inside one process:

| term | inherited | optimized | delta |
|---|---|---|---|
| page-table row upload | 0.053 | 0.052 | −0.001 |
| prefill | 127.507 | 128.395 | +0.888 |
| **first-token sampling, untraced** | **3.048** | **3.350** | **+0.303** |
| total | 130.607 | 131.797 | **+1.189** |

The prefill term is flat, as the 2.4 us bound predicts. The first-token term is the one that moves and
it is the flip side of §4: that token is sampled **eagerly**, so the 20 -> 32 regrouping that makes the
*traced* sampler 0.034 ms/token faster makes the *eager* one slower - 12 more `ttnn.slice` launches and
a wider `ttnn.concat`, once per request. +0.303 ms here, +0.36 and +1.08 ms in the two earlier pairs, always positive, against the
0.564 ms x 127 = 72 ms the pipelined loop saves over the same window. The fix is limitation 6 and it wedged the mesh
(§5.1).

### Where the rest of TTFT is

`logs/probe_prefill.py` -> [`prefill_profile.json`](prefill_profile.json) walks a warmed ladder
(131.0 / 176.8 / 253.8 / 426.1 ms at 128/256/512/1024) and fits it **two** ways, both computed by the
probe after the third review pointed out that `fit` was a secant and the hand-typed least-squares
alternative was wrong by 9 ms: a two-point secant (slope 0.329 ms/token, intercept **88.9 ms**, 67.8 %
of a 128-token TTFT) and a four-point least squares (slope 0.327, intercept **89.8 ms**, 68.5 %). The
local 128->256 slope is 0.358 ms/token. So 68 % of a 128-token TTFT is length-independent, which is
what 40 layers of *eager* dispatch looks like.

One hypothesis was tested and refuted: the eager prefill path emits a per-layer, per-expert-group
`logger.debug` line, and a host log call inside a measured window is host work. The loguru-level A/B
reads **+3.3, −0.4, −10.0, +1.4 ms** across 128/256/512/1024 in the committed run against −4.6, +9.1,
−2.4, +0.2 in an earlier one; the magnitudes do not repeat and the sign flips within and between runs.
Not it.

**Capturing a prefill trace is the optimization the intercept implies, and it is not taken here.**
The blocker is precise rather than a preference: the prefill program set is keyed by the *logical*
prompt length, not the physical block length - the `ttnn.slice` offsets, the MoE valid-token count and
the `conv1d` length are all compile-time constants - so "one trace per prefill shape" is one trace per
distinct prompt length, unbounded, unless the decoder layer's prefill masking contract changes to take a
physical block plus a validity mask. That contract is decoder-owned and this stage's goal is to preserve
it. The ceiling (91-98 ms) is recorded above so the decision can be made with a number.

The **teacher-forcing** row needs the same caution for a stronger reason: `run_teacher_forcing` drives
the *serial* loop by construction, so this stage's change cannot move it, and
`test_teacher_forcing_keeps_the_serial_loop` asserts that. The committed delivered-path measurement is
38.18 t/s/u (`readiness_teacher.json`) against the full-model stage's 37.01, reported as context rather
than as a result.

### 6.1 What this stage does fix on the prefill side

`doc/full_model/`'s limitation 3 said a newly seen prompt length costs one trace re-capture that
"lands in *neither* reported metric" and that pre-compiling a bucket of lengths "is left to the
optimized-full-model stage". Both halves are now done:

* measured — `perf_summary.json`'s `cold_prompt_length_cost` for a fresh, non-aligned length 135:
  first request wall clock **0.548 s**, of which **252 ms** is outside both published metrics, one trace
  re-capture, warmed TTFT at that length 176 ms afterwards (135 pads to a 256-token block). The
  inherited arm measures 0.550 s / 246 ms, so this is a property of the guard rather than of the
  optimization;
* fixed — `OrnithGenerator.warmup([lengths])` pays it at startup and returns the per-length seconds
  and the re-capture count. `test_warmup_removes_the_cold_length_recapture` drives a length nothing
  has compiled (87, not a multiple of the tile, page or alignment) and asserts a later request at
  that length does not re-capture.

---

## 7. The context contract

`logs/update_context_contract.py` writes this stage's block into `doc/context_contract.json` as
`optimized_full_model`, leaving `full_model` as the previous stage's record. Three things in it were
wrong in the first pass and were fixed after the stage review, because they were copied text rather
than derived values:

* `lm_head` bytes, `total_resident` and `lm_head_note` still described a 62080-wide unpadded shard.
  `PADDED_VOCAB` is now computed from the model's own alignment constants, so the file cannot drift;
* `internal_shape_policy_delta.lm_head_output` still said "nothing is padded and no invalid-vocab mask
  is needed", which denied this stage's central change. It now states the padding, its size, the
  reason (the sampler's factorisation, not the matmul), the mask and the test that asserts it;
* the 262143-token row in `capability_change` was hard-coded from an earlier run. It is now generated
  from `long_prompt.json`, and the `headroom_ratio_note`'s reference to a README figure that no longer
  existed was removed.

No capability is reduced: 262144 advertised, 262144 built, a 262143-token non-aligned prompt actually
run through the optimized path with 24.14 GiB free.

---

## 8. Evidence runs, in order

Every device-facing command was run on its own. `logs/run_evidence_status.txt` is the per-step status
file for the committed run.

1. `logs/bench_before.txt` — the inherited path re-measured on this checkout.
2. `logs/ab_terminal.txt.gz` — 19 arms (originally four rounds; see §3).
3. `logs/bench_after.txt`, then `logs/bench_full_model.txt` -> `perf_summary.json` — the delivered
   configuration, including the serial arm in the same build, the TTFT breakdown, the cold-length cost
   and the `performance_accounting` block.
4. `logs/probe_prefill.txt` -> `prefill_profile.json` — the TTFT ladder and the logging A/B.
5. `logs/readiness_bfp4_head.txt` — the LM-head dtype arm's accuracy gates.
6. `logs/ab_terminal_kblock.txt` — the K-block x norm-grid x fidelity cross (§3.4).
7. `logs/run_evidence.sh` — readiness gates, the qualitative suite, the probes, the degeneracy gate
   and the 48-case fast suite. Run twice: once interrupted by §5.2, then once end to end on the final
   code, which is the committed status file.
8. `tracy/run_profiling.sh` — the reduced-variant decode capture, re-taken on the final code. The
   prefill capture failed twice; `tracy/prefill_capture_failure.txt` records both signatures.
9. `logs/run_watcher.sh` — the fast suite under `TT_METAL_WATCHER=10`, separate run, re-run after the
   final code change: 48 passed, 0 error lines.
10. `pytest -m long` — 5 long cases, `logs/pytest_long.txt`.
11. `logs/probe_footprint.py`, `logs/probe_long_prompt.py`, `logs/update_context_contract.py` — the
    recomputed context contract and its capacity evidence.
12. `logs/check_prose_figures.py` -> `logs/check_prose_figures.txt` — the figure gate, run after the
    documents were refreshed from the final sweep. It is deliberately *not* a step of the sweep: it
    checks the documents against the sweep's own output, so inside the sweep it would always fail on the
    run that produces new numbers. Removing it from the script was the fix; leaving it in and calling the
    resulting FAIL expected would not have been.

**One infrastructure fault in the final sweep.** Its three HF-reference steps
(`readiness_autoregressive`, `readiness_autoregressive_chat`, `readiness_qualitative`) were OOM-killed
inside the CPU load of the 35B HuggingFace reference: the host had 51-55 GiB of `MemAvailable` against a
~70 GiB requirement, with only 13 GiB of it accounted for by any process's RSS. A standalone retry on an
otherwise idle host died identically, so it is persistent rather than transient, and it is not something
this run can reclaim. [`logs/host_memory_event.txt`](logs/host_memory_event.txt) records the
`/proc/meminfo` numbers and the reasoning; the three artifacts are intact from the immediately preceding
complete sweep on the same code, and the two accuracy gates the goal contract names ran in the final
sweep and passed. Recorded as infrastructure evidence, not a model result, per `$tt-device-usage`.

---

## 9. Stage review

One independent `$stage-review` subagent returned `more-work-needed` with two P1 and six P2 findings.
All of them were real. What they were and what was done:

| finding | resolution |
|---|---|
| P1 — README/work_log prefill ladder, slope, intercept and logging A/B did not match `prefill_profile.json` (the committed JSON came from a later run that overwrote the one the prose described) | both documents regenerated from the committed artifacts; §6 |
| P1 — work_log stale against `perf_summary.json` and contradicting the README on the *sign* of the TTFT result | this file rewritten from the committed JSON; §6 states the spread in both directions |
| P2 — the three-stage top-k rejection used a gross-vs-net device slope and understated the candidate ~2x | re-derived from the two stacked reports with the grouping machinery counted (§4.1), which both corrects the slope to ~1:1 **and** refutes the third stage outright (§4.2) |
| P2 — the new `optimized_full_model` contract block carried stale text denying the vocabulary padding and mask | `update_context_contract.py` fixed to derive those fields; §7 |
| P2 — `lm_head_program="dram_sharded"` silently no-oped at the default core count, and `build_generator`'s docstring advertised it as the default | the pair is now refused with a legal-values message, and the docstring lists every knob (§3.2) |
| P2 — `run_evidence_status.txt` said `0 failed` while containing a step that never completed and is no longer in the script | the whole sweep re-run end to end on the final code; the committed file is that run (§5.2) |
| P2 — the LM head's `in0_block_w` and fidelity were never swept, and the recorded reason for `8` was wrong | second ladder run (§3.4): 11 arms, two exact L1 blockers, the reason corrected in README, test and code comment |
| other — EOS lookahead state, teacher-forcing/TTFT noise, raw-prompt token agreement, op-share disclosure, `token_readbacks: 128` | all now stated explicitly in README §2, §1, §5, §10 and §8 |

A **second** independent review then returned `more-work-needed` again, with three P2 findings, all
also real:

| finding | resolution |
|---|---|
| P2 — the §4.1 table mixed op bases (dram-only "before" against dram-plus-L1 "after"), two figures did not reproduce from the CSVs it named, and the "~190 us/replay" machinery figure conflated the fitted slope with a raw op total that includes the decoder's own slices | the table and the whole cost model are now **generated** by `logs/make_sampler_cost_model.py` into `logs/sampler_cost_model.md`, on one consistent basis (§4.1) |
| P2 — README §9's `g = 44` rejection row quoted only the reduction and LM-head terms, which argued *for* the change, and mis-stated the column increment | restated with the machinery term and the correct +896 columns/device (§4.2) |
| P2 — the work log claimed "no committed artifact predates the delivered code", but `pytest_long.txt`, both footprints and `long_prompt.json` predated the final `tt/model.py`, and `tests/test_full_model.py` postdated **every** committed pytest log | the whole behavioural sweep, the long suite, both footprint probes, the long-prompt ladder and the contract regeneration were all re-run on the final tree; the claim below is now per-artifact rather than blanket |

Plus: `perf_summary.json`'s `named_limitations` string said "near 8 %" where the same object computes
6.3 % (now derived from the number); the ladder's contrary token-out column and the unmeasured
norm-grid range above 8 cores are disclosed in §3.4 and README §3.3; the persistent-buffer rejection is
narrowed to the *values* gather, which is the one the op-contract blocker actually covers; the prefill
fit is labelled as a two-point secant with the four-point least-squares alternative beside it; and a
0-byte console artifact was removed.

A **third** independent review then returned `more-work-needed` once more, with one P1 and five P2
findings — every one of them documentation fidelity, and it explicitly found no correctness defect:

| finding | resolution |
|---|---|
| P1 — the TTFT and teacher-forcing "run-to-run spread" figures cited to neutralise the two rows that moved unfavourably were two-thirds unsupported: of six claimed TTFT medians only two existed in the artifacts, one was a pre-optimization *minimum*, one was a 20-token probe line, two existed nowhere; and two of the four teacher-forcing figures did not exist while a third was the rejected bfp4 arm | both rows rebuilt from the artifacts. README §6 now tabulates **all three** committed 128/128 bench runs individually instead of summarising a band, and rests the "flat" claim on the ~2 µs mechanism bound. The teacher-forcing row cites only the one delivered-path measurement and is labelled context, not a result |
| P2 — README §7.2's long-prompt table did not reproduce from `long_prompt.json` | regenerated from the JSON |
| P2 — the four-point least-squares intercept added last round was wrong by 9 ms and was described as "a couple of milliseconds" | `probe_prefill.py` now computes **both** fits into `prefill_profile.json`; §6 quotes both, and the share is a 64-70 % range |
| P2 — README §6's logging A/B sentence was garbled (six values for four lengths) and contradicted the JSON | rewritten from `debug_logging_cost_ms` |
| P2 — README §10's whole-window device total (9112.1 µs, −84.5 µs/replay) contradicted §4.1 and the generated model (9109.5, −85.1) | §10 now quotes the generated figures |
| P2 — the bfp4 LM-head arm was measured on the pre-final tree and was missing from the provenance exceptions | **re-run on the final tree**; §3.5 and README §3.4/§9 now compare same-tree numbers |

Its "other concerns" are all addressed too: `best_topk_groups` now minimises the **joint** cost (both
terms of §4.1's model, so `topk_num_groups="auto"` can no longer select the `g = 44` candidate §4.2
rejects, and `tt/model.py` carries the two coefficients as named constants); the `SparseMatmul` row
quotes are labelled as first-row rather than mean; the op-to-op gap figure is corrected to 261 µs; the
rounding/interval figures are taken from the JSON; the "cross" is described as 11 arms; the ladder's
contrary eager `final_norm` column is disclosed alongside the token-out one; `run_watcher.sh` now counts
bare device asserts as well as watcher-prefixed lines, and `watcher/watcher_error_count.txt` classifies
the three matches that pattern finds (one expected active-trace allocator warning, two test *names*);
`perf_summary.json`'s last hand-written sampler limitation string is generated; and README §5 describes
`batch_slots.json` as the passing negative control it is. §10 also now says the *before* profiler
capture is the previous stage's, with the unchanged-op tracking that makes it comparable.

A **fourth** review then returned `more-work-needed` with one P1 and eight P2 findings, again all
documentation fidelity and again finding no correctness defect. Rather than fix the prose a fourth time,
this round fixed the two things that kept producing stale prose:

* **the TTFT comparison was rebuilt as a measurement.** `logs/bench_full_model.py --arm inherited`
  reconstructs the pre-optimization path from constructor knobs, so before and after are now nine
  repeats each **in one session** (§6). The answer changed: TTFT is +2.7 ms on the median, with 1.08 ms
  of it mechanistically attributed, and §1 reports it as a cost instead of arguing it is flat;
* **`logs/check_prose_figures.py` now asserts every headline figure** in both documents against the
  artifact it names, plus the existence of every referenced path. It is step 14 of
  `logs/run_evidence.sh`, so a stale number fails the evidence run rather than a review.

The individual findings: the garbled logging-A/B sentence, the 0.60 s cold wall clock, work_log's 3.2 ms
first-token figure, README's 0.566-vs-0.568 mismatch, the 41.906 serial figure and its 0.02 % claim, the
−89-vs-−90 wall delta (the last hand-typed number inside the "nothing is hand-typed" generator, now
derived from the two perf summaries), and §10's unchanged-op tracking triple (now generated) are all
fixed and now covered by the checker. Of its "other concerns": §3.4's arm list was missing
`k16-n4-lofi`; `best_topk_groups`'s justification is restated (for *this* build reduction-only ties 32
with 61 and the loop keeps 32, so the joint objective changes nothing here — what it buys is that a
1980-tile build gets 33 instead of the rejected 44); "a third of the sampler's device time" is corrected
to a fifth; the dense-projection row now quotes the range of its four committed rows;
`perf_summary.json`'s last two hand-written limitation strings are derived from `tt/model.py`'s
constants and `prefill_profile.json`; `watcher/watcher_error_count.txt` is now the output of
`logs/watcher_report.sh`, which can be regenerated from the committed artifacts and which lists its
three benign matches instead of only counting them; and the roofline byte total carries a note tying it
to `footprint.json` (it is deliberately smaller, because one decode step does not read the whole KV
cache or all 64 local experts).

A **fifth** review returned `more-work-needed` with one P1 and eight P2 findings, and a **sixth** with
three P1 and six P2 — every one of them, in both rounds, a figure in one of these two documents that did
not match the artifact it named, and neither round found a correctness defect. The fifth round's fixes
were: work_log §1 rebuilt from the two perf summaries it cites; the "0.02 % on every decode row" claim
replaced by the three real per-row deltas; the wall delta derived inside
`logs/make_sampler_cost_model.py` instead of typed; `logs/watcher_report.sh` corrected (it looped on
`.txt` and so silently never scanned `watcher.log`, and now errors on a missing artifact instead of
skipping); the tracy provenance restated against the mtimes; and
`test_the_pipelined_loop_stops_on_eos_and_leaves_state_one_position_ahead` added for the EOS branch the
review found unexercised.

The sixth round's findings were the same shape and are why the gate below is now much wider: README
limitation 1 still carried prefill-fit figures from a superseded probe run; work_log §6's inherited TTFT
table was from a run that was never committed, and its implied sign was the opposite of the committed
pair's; work_log §5.2 still said the evidence sweep was clean when the committed status file records
three host-OOM failures; `logs/sampler_cost_model.md` had not been regenerated after the final bench
arms, so its wall delta and the prose disagreed; §1's baseline-agreement and reproducibility percentages
were wrong; the sampling pair mixed the archive's before with this sweep's after; and several restated
figures (the first-token term, the logits-only delta, the trade arithmetic, the teacher-forcing figure in
work_log) were stale. All are fixed against the committed artifacts, and — the point — every one of them
sat in a place `check_prose_figures.py` did not look. It now covers README's limitations section,
work_log's own copies of both TTFT distributions and the breakdown, the delta columns, the
baseline-agreement and reproducibility percentages, the sampling pair, the trade arithmetic, and a
freshness assertion that fails if `sampler_cost_model.md` is stale with respect to the perf summaries it
reads: **68 numeric rows and 19 literals across both documents.**

A **seventh** review returned **`clean-pass`** with no required work. Its remaining items were all
presentational and are closed here: the gate's row count in this file's closing paragraph, the commit
table's count and its self-referencing row, the gate's own docstring (which still called itself a step of
`run_evidence.sh`), the roofline denominator's citation in `bench_full_model.py` (it cited a `354 GB/s /
69.1 %` row that exists in neither capture - the constant comes from the previous stage's 355/69.3, and
this stage's own 353/69.0 would imply 511.6 GB/s and a 1.466 ms roofline, so the kept value is the
conservative one), the reverted-optimization comment in `tt/generator.py`, `triage/README.txt`'s label on
tt-triage's summary, the decode headline's estimator (min of nine, now named in README §1 with the
medians beside it), `logs/host_memory_event.txt`'s understated substitution argument, three scripts
missing from README §12's tree, and a pointer from README §8 to the inherited warning ledger with the
80/12 counts that are identical in all four bench logs.

**Provenance, per artifact.** The delivered code is `tt/model.py` (last changed 12:59) and
`tt/generator.py` (16:45); the shipped test file is `tests/test_full_model.py` (16:45). **Every**
committed artifact of the final evidence set postdates all three: `perf_summary_before.json` 19:25 and
`perf_summary.json` 19:29 (the two nine-repeat bench arms), `prefill_profile.json` and every probe log
18:0x-18:2x, `logs/pytest_long.txt` 18:35, the watcher run 19:10, `logs/pytest_full_model.txt.gz` 20:16,
and `logs/check_prose_figures.txt` last of all.

Four artifacts are exceptions, each named rather than glossed:

* the three **HF-reference readiness artifacts** (`readiness_autoregressive.json`,
  `readiness_autoregressive_chat.json`, `readiness_qualitative.json` and the completions they point at)
  are from the immediately preceding complete sweep, because the final sweep's three HF steps were
  OOM-killed by the host - see the infrastructure note in §8. Their mtimes are 17:03 / 17:10 / 17:32, so
  they **postdate all three delivered files** (12:59, 16:45, 16:45): they were produced on the exact
  delivered code, which makes the substitution stronger than "same tree";
* the **bfp4 LM-head arm** (`readiness_{prefill,teacher}_bfp4head.json`, `logs/readiness_bfp4_head.txt`,
  13:03-13:07) postdates `tt/model.py` but predates the 16:45 generator comment and test addition,
  neither of which touches the terminal dtype path it measures;
* the reduced-variant **decode profiler capture** (`tracy/`) predates all three files. It runs
  `logs/profile_reduced.py` against the model and generator directly and executes no pytest, and the only
  model change since is `best_topk_groups`'s objective, which returns the same 32 groups for this build -
  so the graph it captured is the graph the delivered code runs. §10's shares and §4's cost model rest on
  that;
* the two **ladders** (`ab_terminal*`) predate the final tree by construction - they are the *search* that
  chose it, each arm builds its own model from constructor knobs, and the winning arm's geometry is
  re-asserted on the final code by `test_the_lm_head_runs_the_tuned_program_config` in the 20:16 suite;
* the **capacity artifacts** (`footprint.json`, `footprint_batch32.json`, `long_prompt.json`,
  `doc/context_contract.json`) predate the 16:45 edits and postdate an earlier `tt/model.py`; the only
  model change between is `best_topk_groups`'s objective, which cannot move a byte count because it
  returns the same value.

`logs/check_prose_figures.py` passing is the machine-checkable half of this paragraph: it re-derives 68
numeric rows and 19 literal claims across both documents from the artifacts and verifies every referenced
path resolves.

## 10. Commits

One repo, four checkpoint commits on `agentic-research/hous/ornith-1.0-35B`, never pushed:

| commit | full | what |
|---|---|---|
| `c3911e5fc37` | `c3911e5fc37da48a7829d941e77fceb8560c28bf` | the terminal path, the decode loop, and the evidence |
| `359b6ef4be4` | `359b6ef4be4a5a46226a2d22b44faa94357bc520` | record the stage commit SHA |
| `8b4b816e6a8` | `8b4b816e6a87449af2769b3b0eba8bb624cb5d30` | make the figure gate cover where the drift was |
| `8faec81d31a` | `8faec81d31ae3a628e62ecebc8ddc83620d87005` | record all three stage SHAs |
| *(this section)* | — | the commit that records the table above, which cannot contain its own hash |

They contain only stage-owned paths: `tt/model.py`, `tt/generator.py`, `tests/test_full_model.py`,
`doc/context_contract.json`, the whole of `doc/optimized_full_model/`, and the regenerated
`readiness_autoregressive{,_chat}/` outputs. Two unrelated files that were already dirty when the stage
started — `.agents/skills/tt-device-usage/SKILL.md` and the untracked
`.agents/fast-models-fast-feedback.md` — are deliberately **not** in any of them.
