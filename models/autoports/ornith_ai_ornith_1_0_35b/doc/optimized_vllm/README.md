# Ornith-1.0-35B — optimized vLLM serving (TTNN, 4-chip Blackhole ring)

The [vLLM integration](../vllm_integration/) stage's serving path, optimized in place on the same
hardware: four Blackhole `p300c` chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4 dense
and EP=4 over the 256 routed experts, the datatype sweep's selected policy **`C06-proj-bfp4-lofi`**,
paged KV cache **bfloat8_b**, **262144-token** advertised and served context. The model, the precision
policy, the served context and the captured decode graph are unchanged; three pieces of avoidable
serving work are not.

`before` is commit `81c01e1360a`, the tree the vLLM-integration stage closed on. `after` is this tree.
Every *performance* before/after pair was measured here, in this stage, on a `git stash`-ed before tree
with the same runner and the same server flags. Three **before** rows in §4 are the exception and say so
there: the `max_num_seqs=1` sampling suite, the qualitative prompts and the adapter-suite count come
from the integration stage's committed artifacts, because re-running them on the before tree would have
told us nothing the byte-level comparisons already do.

**What this stage found, in one paragraph.** Warmed single-user decode was already on the model's own
traced token-out floor and could not be improved from the serving layer (§1, §2). The one real serving
defect was that a request could compile prefill programs while the decode traces were live, which costs
the compile *and* forces a decode-trace re-capture inside the request. That class is now compiled at
start-up: **every before-tree server that served a prompt in a new prefill block logs a re-capture of
102–271 programs — four of the five did — and no after server logs one above 19 at `max_num_seqs=1` or
above 64 at 32** (§3.2 has the full census, including the fifth before server, which served only short
prompts and so never opened a new block). What the move is worth in wall clock depends on whether
tt-metal has to *build* those kernels or can load them from its on-disk cache: **10.0–14.3 s of TTFT per
new prefill block when it must build them, 0.12–0.29 s when it need not**. It is honest to call this a
**relocation** rather than a saving — on a cold machine roughly the same kernel-build time is paid
either way, at start-up instead of inside user requests, and §3.2 puts a measured bound on both sides —
plus one genuine removal: the re-capture stall for every length the warm-up covers, worth 1.77 ms/token
on the primary benchmark's first request (§1).

---

## 1. Headline: primary single-user serving performance

**Workload: 128-token prompt, 128 generated tokens, 1 request, `--max-concurrency 1`, greedy
(`--temperature 0.0`), `ignore_eos`, `--max-num-seqs 1`, mesh `(1, 4)`, `--max-model-len 262144`,
decode trace on, on-device sampling (`sample_on_device_mode: all`), async scheduling on.** Identical on
both sides. Raw [`vllm_result.json`](../../readiness_vllm/vllm_result.json), normalized
[`vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json), machine-readable before/after in
[`before_after.json`](before_after.json).

### Warm — the steady state

| metric (128/128/1, `max_num_seqs=1`) | **before** | **after** |
|---|---|---|
| **TTFT** P50 / P99 | 153.918 / 153.918 ms | **157.735 / 157.735 ms** |
| **TPOT** mean / P99 | 23.144 / 23.144 ms | **23.155 / 23.155 ms** |
| **ITL** P50 / P99 | 23.119 / 25.033 ms | **23.129 / 24.480 ms** |
| **decode t/s/u** (`1000 / mean_tpot_ms`) | 43.207 | **43.188** |
| aggregate output throughput | 41.377 tok/s | **41.308 tok/s** |
| requests completed | 1/1, 128/128 tokens | **1/1, 128/128 tokens** |

**The warm numbers are unchanged, and that is a finding rather than a gap.** Warmed serving decode was
already on the model's own floor before this stage began: the datatype sweep's traced token-out decode
at the same 128/128/1 shape, batch 1, nine warm repeats, measured **23.1647 ms/token** without vLLM in
the loop ([`post_selection_token_out.json`](../datatype_sweep/post_selection_token_out.json)); this
stage re-measured serving at 23.144 ms before and 23.155 after. Serving is *below* the standalone
generator by 0.01–0.02 ms/token, i.e. by less than either harness's repeat spread, so there is no
serving-specific decode overhead to remove. (One disclosure on that pair: the standalone floor was
measured with `cache_context 8192` / 128 blocks and serving runs the advertised 262144 / 4096 blocks,
so the two allocations differ; the comparison is used for its ~0.02 ms scale, not as an exact
equality.) §2 re-derives the same conclusion from this stage's own measurements.

TTFT moved by +3.8 ms, which is inside its own spread: the two warm `after` runs **at
`max_num_seqs=1`** are 148.767 and 157.735 ms, a 9.0 ms band that brackets the single before run's
153.918 ms. Two runs is a weak spread and it is quoted as one. Per `$optimize` the headline is the **final default run**: the last `max_num_seqs=1` server this
stage launched, no flags beyond the table, which also ran the full sampling suite and the qualitative
suite ([`logs/server_final_b1_vllm.log.gz`](logs/server_final_b1_vllm.log.gz)).

### Cold — the first request at a prompt length, on the same server

| metric (128/128/1, first request at this length) | **before** | **after** (final run) | **after** (earlier run) |
|---|---|---|---|
| TTFT P50 | 266.866 ms | **220.324 ms** | 253.783 ms |
| **TPOT** mean | 24.916 ms | **23.132 ms** | **23.175 ms** |
| **decode t/s/u** | 40.134 | **43.230** | **43.150** |
| ITL P50 / P99 | 23.053 / 26.376 ms | 23.089 / 25.973 ms | 23.066 / 24.213 ms |
| aggregate output throughput | 37.300 tok/s | **40.527 tok/s** | **40.034 tok/s** |

The TPOT half of this row is the robust one and it reproduces on both `after` runs: `vllm bench serve`
excludes TTFT from TPOT, so the ~1.77 ms/token the `before` cold column carries over 127 intervals is
**one ~225 ms stall a few tokens into the stream** — the decode-trace re-capture that request forced.
After this stage the benchmark's own prompt length is compiled at start-up, the server logs **0
re-captures across the whole benchmark**, and the cold column's decode rate is the warm one. The TTFT
half is quoted with both runs because it is not: 220.3 and 253.8 ms against 266.9, i.e. 13–47 ms
depending on the run, and the two `after` runs had different server histories behind them.

---

## 2. Why the warm numbers could not move

Measured in this stage, before any change:

| | ms/token | source |
|---|---|---|
| decoder layer stack (30 `linear_attention` @ 0.564 + 10 `full_attention` @ 0.453) | 21.45 | [`post_selection_token_out.json`](../datatype_sweep/post_selection_token_out.json) |
| \+ final norm, LM head, logits movement | 21.965 | same (`traced_logits_only_decode`) |
| \+ sampling trace (`tt_out_tok` feedback) | 23.157 | same (`traced_decode_plus_sampling_no_readback`) |
| \+ readback | **23.1647** | same (`token_out_decode`) — the generator floor |
| **served, before this stage** | **23.144** | [`before/vllm_result_b1_warm.json`](before/vllm_result_b1_warm.json) |
| **served, after** | **23.155** | [`readiness_vllm/vllm_result.json`](../../readiness_vllm/vllm_result.json) |
| roofline (bytes moved / aggregate DRAM bandwidth) | 1.111 | same artifact, `performance_accounting` |

92.6 % of the step is the decoder layer stack, which this stage does not own; the sampler is 5.1 % and
was already the full-model split-sampling path on its measured-best grouped-top-k geometry, so the
`$optimize` rule "if sampler work dominates token-out decode, fix the generator LM-head/sampling
contract" does not fire. The same holds at `max_num_seqs=32`: serving TPOT 137.738 ms against 137.96
ms/token for the identical traced serving step driven directly, without vLLM
([`before/batch_occupancy.json`](before/batch_occupancy.json), 1 active row — with the allocation caveat
in §6 limitation 3). There is no orchestration overhead left at either batch size, which is what sends
this stage at the parts of serving that are not the steady state.

---

## 3. The defect this stage fixed: compiling prefill programs while the decode traces are live

### 3.1 The mechanism, and what it costs

The prefill path pads every internal block up to a multiple of `PREFILL_ALIGN` (128) and chunks at 2048,
so there are exactly **16 physical block shapes**. Two other things still reach programs as compile-time
arguments — the *logical* length, and the *chunk offset* of a multi-chunk prompt (§3.3 B) — so "16"
bounds the block class, not every prefill program. The first two classes were counted directly, on the
reduced two-layer target, by
[`logs/probe_prefill_program_keys.py`](logs/probe_prefill_program_keys.py) →
[`candidates/prefill_program_keys.json`](candidates/prefill_program_keys.json):

| prompt length | physical block | programs compiled | re-captures |
|---|---|---|---|
| 64 (first in block 128) | 128 | **105** | 1 |
| 100 | 128 | 7 | 1 |
| 128 | 128 | 3 | 1 |
| 129 (first in block 256) | 256 | **101** | 1 |
| 200 / 211 | 256 | 8 / 7 | 1 |
| 256 | 256 | 3 | 1 |
| 257 (first in block 384) | 384 | **114** | 1 |
| 300 | 384 | 8 | 1 |

A compile forces a re-capture because tt-metal hands the trace's intermediates back to the allocator at
`end_trace_capture` while the captured commands still write those addresses; a program's kernel binaries
are such a buffer, so the next replay overwrites them and every prefill that reuses those programs then
executes corrupt code, permanently. That is the vLLM-integration stage's finding, isolated by
`doc/full_model/logs/probe_bisect.py` and implemented as
[`generator.py::_ensure_traces_replay_safe`](../../tt/generator.py). The re-capture costs ~230 ms and
lands on the first decode step after the prefill. The pre-optimization warm-up compiled **one** length
(64), so every other physical block was compiled by a real request.

**What a first request at a new length costs.** Probe
[`logs/probe_new_length_cost.py`](logs/probe_new_length_cost.py): the same request twice, back to back,
on one server, prompts sent as explicit token-id lists so the logical length is exact; 24 output tokens,
greedy, `ignore_eos`, streamed, `--max-num-seqs 1` on **every** arm. "Excess" is `first TTFT − repeat
TTFT` on the same server, which removes the length's own prefill cost. Joined in
[`new_length_cost_before_after.json`](new_length_cost_before_after.json).

| prompt length | physical block | **cold JIT cache, before** | warm JIT cache, before | warm JIT cache, after |
|---|---|---|---|---|
| 211 | 256 (new) | **9987.5 ms** | 233.3 ms | 130.0 ms |
| 347 | 384 (new) | **10750.8 ms** | 152.9 ms | 139.2 ms |
| 613 | 640 (new) | **10859.9 ms** | 163.0 ms | 123.3 ms |
| 900 | 1024 (new) | **12891.0 ms** | 255.4 ms | 139.4 ms |
| 2000 | 2048 (new) | **14261.9 ms** | 293.3 ms | 150.7 ms |
| 300 | 384 (**already built** by the 347 row) | 1030.0 ms | 32.6 ms | 91.3 ms |
| 3000 | 2048 + 1024 (**already built**) | 2247.0 ms | 101.0 ms | 93.5 ms |

The "cold JIT cache" arm ran the **before tree** with `TT_METAL_CACHE` pointed at an empty directory, so
tt-metal had to build those kernels rather than load them; that is the state of a freshly deployed
machine. The two "warm JIT" arms ran the before tree and this tree against the default on-disk cache,
which by then already held every one of these shapes.

Read together the table says three things:

1. **When the kernels must be built, a first request at a new physical block costs 10.0–14.3 s of TTFT.**
   That is the cost this stage moves to start-up.
2. **When they need not be built, the block class costs 0.12–0.29 s and the two trees are
   indistinguishable** — the five new-block rows are 152.9–293.3 ms before against 123.3–150.7 ms after,
   with the ordering reversing row to row. This stage does not claim a warm-cache win, and the
   single-sample-per-arm probe is not precise enough to support one.
3. **The residual is what the after tree still pays**, visible as the two rows whose block was already
   built: 1.03 s (300, logical class only) and 2.25 s (3000, logical class plus a new chunk offset)
   cold, and 0.03–0.10 s warm (32.6 and 101.0 ms before, 91.3 and 93.5 ms after). §3.3 enumerates all
   three residual classes and what each would take.

### 3.2 Change 1 — the serving warm-up compiles every physical prefill block

`prefill_warmup_lengths()` returns all 16 multiples of `PREFILL_ALIGN` up to `prefill_chunk`, longest
first, and `warmup_model_prefill` runs one prompt at each — in vLLM's **phase 1**, before
`warmup_model_decode` captures the decode traces, so none of it can force a re-capture. A prompt of any
length is full 2048-chunks plus one of those tails, so the set is complete **for block shapes** — it is
not complete for the two other keys §3.3 measures, the logical length and the chunk offset.
`ORNITH_VLLM_PREFILL_WARMUP` overrides it: `all` (default), `min` (the old single 64-token pass), or an
explicit comma-separated list.

**Proof it landed, at the mechanism.** Every re-capture every server logged, bucketed by how many
programs had been compiled since the capture
([`candidates/recapture_classes.json`](candidates/recapture_classes.json)) — all five before-tree
servers, not only the four that show the class:

| server | tree | programs per re-capture | block class present |
|---|---|---|---|
| `max_num_seqs=1`, first run | before | 3, **102**, **115**, **138** | **yes** |
| `max_num_seqs=32`, first run | before | 3, 7, 13, 64, **115**, **175**, **271** | **yes** |
| `max_num_seqs=1`, re-run | before | 8, 13, **102**, **115**, **138**, **175**, **271** | **yes** |
| `max_num_seqs=1`, cold JIT | before | 8, 13, **102**, **115**, **138**, **175**, **271** | **yes** |
| `max_num_seqs=32`, re-run | before | 7×8, 19, 20, 32, 56 | no — it served only the sampling suite, whose prompts all land in the 128 block the old warm-up already compiled at length 64 |
| `max_num_seqs=1`, first run | after | 2, 3, 6, 7×16, 8×2, 9×3, 10×2, 11, 12, 19 | **no** |
| `max_num_seqs=32` | after | 3×2, 6, 7×14, 9×2, 10, 11, 19, 56, 64 | **no** |
| `max_num_seqs=1`, final | after | 7×11, 11 | **no** |

The fifth before row is the honest qualifier on the claim: the block class only appears when a server is
actually asked for a prompt in a block it has not compiled. Every before server that was, showed it;
no after server shows it at all. The remaining after buckets are the two classes §3.3 measures.

`serving_counters` records the work done: `prefill_warmup_lengths: 16`, `prefill_warmup_programs:
2887–2891`
([`after/vllm_serving_capability_exit_final_b1_server.json`](after/vllm_serving_capability_exit_final_b1_server.json)).
(The `0 trace re-capture(s)` line every after warm-up prints is *not* evidence for anything — every
before warm-up prints it too.) `tests/test_generator_vllm.py::test_a_warmed_physical_block_compiles_nothing_and_never_re_captures`
pins it on device: a warmed block compiles **0** programs and forces **0** re-captures.

**Cost: server start-up, and it is not small on a cold machine.** Measured, not estimated:

| | before | after |
|---|---|---|
| prefill warm-up wall time, `max_num_seqs=1`, warm kernel cache | **0.835 s** (03:03:02.225 → 03:03:03.060) | **11.7–14.9 s** |
| `init engine (profile, create kv cache, warmup model)`, `max_num_seqs=1` | **11.02 s** | **22.21 s** (final run) / 25.25 s (first run) |
| `init engine`, `max_num_seqs=32` | **60.61 s** | **72.04 s** |
| program cache entries when warm-up finishes | **402** (`max_num_seqs=1`) / 448 (32) | **3098** (`max_num_seqs=1`) / 3144 (32) |
| programs the prefill warm-up itself compiled | not counted on that tree (the counter is this stage's) | **2887** (`max_num_seqs=1`) / 2891 (32) |

Those after figures are **warm-kernel-cache** numbers: the final after server logs `JIT cache stats:
2324/2324 hits (100.0%)` and the first (the 14.9 s one) `2533/2543 hits (99.6%)`. The cold-cache cost is bounded from this stage's own cold-JIT before
server, which built the same 402-entry program set in **73.74 s** (07:12:33.615 → 07:13:47.354) where
the warm before server took **2.255 s** (06:47:10.152 → 06:47:12.407) — about **0.18 s per program
actually built**. At that rate the after tree's 2887 warm-up programs are on the order of **8–9 minutes**
of first-boot start-up on a machine whose tt-metal kernel cache is empty, against ~12 s once it is not.

That is the number this trade has to be judged on, so it is stated plainly:

* it is paid **once per machine**, not once per server — the kernel cache is on disk and every later
  server on that host warms in ~12 s;
* on a cold machine the same kernels get built either way. The change decides *where*: at start-up, or
  inside the first user request at each block (10.0–14.3 s of TTFT, §3.1) plus a re-capture;
* but a deployment that only ever sees two or three block shapes builds **more** total kernels under
  this change than it otherwise would. `ORNITH_VLLM_PREFILL_WARMUP` accepts an explicit list precisely
  for that case, and `min` restores the old single-length behaviour;
* the reduced two-layer target's own 16-block warm-up was observed at **65.4 s, 6.3 s and 147.8 s**
  across the three adapter builds of one suite run
  ([`logs/warmup_cold_jit_cache_reduced_target.txt`](logs/warmup_cold_jit_cache_reduced_target.txt)) —
  a spread that is itself the point: that run's cache state was not controlled, so none of the three is
  the cold-cache cost, and the 8–9 minute figure above comes from the per-program derivation rather than
  from any of them. Measuring the 16-block warm-up directly on the 40-layer model with an empty cache was
  attempted twice and OOM-killed both times (§7).

### 3.3 What is left: three measured classes, none of them the block class

**A. The logical-length class (3–13 programs).** A request at a logical length no earlier request used
compiles 3 (when the length is a multiple of 128) to 8–13 (otherwise) programs and pays one re-capture:
1.03–2.25 s cold, 0.03–0.10 s warm, plus a ~230 ms stall on the first decode step. It is also the
**largest** residual bucket the after servers log: the 56-program re-capture at 05:27:37.775 on the
`max_num_seqs=32` server follows a 21-request prefill carrying **eight** new logical lengths
(18, 19, 20, 21, 23, 24, 26, 27), and that same server logs a single new non-aligned length at 7
programs fourteen separate times — 8 × 7 = 56. The before tree reproduces it with the identical eight
lengths (`server_beforetree_b32_vllm.log.gz`, 07:05:59.257 → 07:06:03.076). An earlier draft of §3.3 C
attributed that bucket to the slot remap; review round 4 corrected it from the log ordering. Those programs are
keyed by the logical length itself. `set_program_cache_misses_allowed(False)` names the first one
([`candidates/prefill_program_keys_named.json`](candidates/prefill_program_keys_named.json)):

```
Device operation "EmbeddingsDeviceOperation": program cache miss occurred, but cache misses are forbidden
```

The rest of the class is the tail pad and trim (`ttnn.pad` / `ttnn.slice` between `logical` and `phys`),
the last-token slice `[0, logical-1, 0] → [1, logical, dim]`, the DeltaNet gate ramp comparison against
`float(logical_len)`, and the conv tail slice at `logical_len`.

*Why they were not removed — corrected in review round 3.* An earlier draft of this section claimed
`ttnn.slice` has no tensor-valued form in this build. **That was wrong**, and the correction matters
because it was the stage's only stated op-contract blocker. `slice_nanobind.cpp` binds an *overload 1*
taking `starts`/`ends` as device `ttnn.Tensor`s plus `slice_dim`/`num_devices`, `slice.cpp` routes it to
`ttnn::prim::slice(..., use_tensor_args=true, ...)` with dummy bound shapes, and
`SliceDeviceOperation::compute_program_hash` then hashes only those dummies, `slice_dim`, `num_devices`
and the input/output specs — so the bounds really are runtime values. `ttnn.slice.__doc__` documents
only the `List[int]` overload, which is how the earlier draft got it wrong;
`$tt-enable-tracing` had it right.

What the overload does **not** do is express an arbitrary window: `compute_output_specs` divides
`slice_dim` into `num_devices` equal parts, so it takes an even split at a runtime offset, and
`validate_on_program_cache_miss` still requires a TILE input's output height to be tile-aligned. Pulling
the single last-token row (`[0, logical-1, 0] → [1, logical, dim]`, output height 1) is therefore not a
direct substitution; the adapted form would take the tile-aligned 32-row window that contains the last
token (`slice_dim=1, num_devices=phys/32`, which is keyed by the *physical* block and not by the logical
length) and then select the row inside it. That is a **named, unverified candidate**, not a blocker — it
was found from the source in review round 3, after the host had stopped being able to run this model
(§7), so it has not been tried on device.

The other members of the class need their own dispositions and did not get them here: `ttnn.embedding`
on the unpadded `[1, logical]` token row, the tail pad/trim, the DeltaNet gate ramp comparison against
`float(logical_len)`, and the conv tail slice at `logical_len`. The cheap half-measure — padding the
token row to the physical block before the embedding — trades one program for another, because the
embedding output then has to come back to the logical length. Removing the class properly still means
making the prefill's masking boundary physical rather than logical, and threading a physical-length
contract through `ttnn_prefill_forward` and every layer's `prefill_forward`.

**B. The chunk-offset class (6 programs per new chunk offset).** The warm-up's 16 prompts all start at
position 0, so they compile only the `chunk_start_idx == 0` variants. A prompt longer than one
`prefill_chunk` (2048) presents further chunks at `start_pos` 2048, 4096, … and the page-table slice
each chunk takes is keyed by its absolute position, so a chunk index the server has not seen still
compiles. Enumerated after the shipped warm-up, on the reduced target
([`candidates/prefill_chunk_offsets.json`](candidates/prefill_chunk_offsets.json)):

| prompt length | chunk start positions | programs compiled | re-captures |
|---|---|---|---|
| 2049 | 0, 2048 | 10 | 1 |
| 3000 | 0, 2048 | 13 | 1 |
| 4097 | 0, 2048, **4096** | **6** | 1 |
| 6145 | 0, 2048, 4096, **6144** | **6** | 1 |
| 8193 | …, **8192** | **6** | 1 |
| 10241 | …, **10240** | **6** | 1 |
| 10300 | no new offset | 8 (the logical class) | 1 |

So a new chunk offset is **6 programs**, an order of magnitude below the 102–271 of the block class, and
the offsets are cumulative — one long prompt compiles every offset it spans, and the advertised 262144
context has 128 of them.

The 19-program re-capture on an after `max_num_seqs=1` server after `prefill: 1 request(s), lengths
[9000]` (05:08:48.100 → 05:08:51.852) is **two** of these plus one of §3.3 A's, not four of these: that
server had already served 3000, 2049 and 4097, so offsets 0/2048/4096 were compiled and the 9000-token
prompt's five chunks introduce only 6144 and 8192 — 2 × 6 = 12 — while its 808-token tail is a new
logical length, worth the class's usual 7. 12 + 7 = 19. It is **not** a multi-slot or slot-remap effect;
an earlier draft of this README said it was, and the draft after that got the decomposition wrong.

**C. The slot-remap class (4–8 programs per permutation width).** `remap_state_slots` compiles per
permutation width. Measured after the shipped warm-up on an 8-slot reduced target
([`candidates/slot_program_keys.json`](candidates/slot_program_keys.json)): a remap of width 2 compiles
4 programs, width 4 compiles 4, width 8 compiles 8. Prefilling into a *slot* compiles nothing —
after slot 0 has been prefilled, slots 1–7 each compile **0** — so the class is the remap, not the
multi-request prefill, which is what an earlier draft of this README said.

The one bucket this class explains on a real server is the **64**, and it is reproduced on both trees:
the after `max_num_seqs=32` server logs it at 05:16:55.507, 3 ms after its first width-32 remap, with
only an already-served length 100 in the prefill before it — and the *before* `max_num_seqs=32` server
logs the identical 64 at 03:14:17.215, 2 ms after its own first width-32 remap, after the same
`[128] → [100] → 31×[100]` sequence. The **56** bucket is *not* this class; it is the logical-length
class, re-attributed in §3.3 A after review round 4 read the log ordering.

The counts fit **~2 programs per row index the server has not moved before**, which is the reading the
logs support and the 8-slot probe agrees with: widths 2/4/8 compile 4/4/8 (rows 0–1, then 2–3, then 4–7),
a width-10 remap compiles 20, and a *first* width-32 remap compiles 64. The before-tree re-run's first
width-32 remap compiles only **32**, because widths 10 and 16 had already moved half its rows — which is
why "per width" is the wrong axis and the earlier draft's extrapolation from an 8-slot probe was
fragile. It also makes the fix smaller than §6 assumed: one full-width rotation at warm-up covers every
row index, bounded by 2 × `max_batch_size`.

**B and C are both closable the same way the block class was**, and neither was landed here. C is three
lines in the warm-up (one permutation of each width over still-zeroed state) and B is a bounded prefill
of the longest prompt a deployment expects; both were written and then reverted, because the only
machine that can run this model ran out of RAM before either could be validated on the served
40-layer configuration (§7). They are named, measured follow-up work in
[`work_log.md` §6](work_log.md#6-what-was-not-done-and-what-it-would-take) with their fix designs, not
closed items.

### 3.4 Change 2 — unchanged sampling parameters are not re-pushed every token

`SamplingGenerator.apply_decode_state` rebuilds four host tensors (`k`, `p`, `temperature`, and the
greedy tie-break column) and copies each to the device. The adapter called it on **every** device-sampled
decode step. A serving batch's parameters change when vLLM changes the batch, not once per token, so
`_apply_decode_sampling` now keeps the formatted parameter row it last pushed and skips the push while it
is identical. The skip is dropped on `reset_batch`, on any prefill (`_apply_prefill_sampling` writes the
same tensors), and on a serving reset; seeds and RNG counters are still advanced every step, because they
are per-token state rather than parameters.

**Proof it landed and is safe:** **7684 of 8093** device-sampled steps skipped the push on the final
`max_num_seqs=1` server (94.9 %) and **6129 of 6388** on the `max_num_seqs=32` server (95.9 %). The
canonical sampling suite is unchanged on both configurations (§4), and
`test_unchanged_sampling_parameters_are_not_re_pushed_every_token` pins both halves on device: four
identical greedy steps push nothing, and a single changed row pushes again immediately.

**What it is worth:** nothing measurable at the headline, and that is reported rather than claimed away.
The copies are four `[32]`-wide tensors on the same in-order queue as a 23 ms decode replay, and warm
TPOT moved by −0.011/+0.011 ms across runs whose own spread is ±0.012 ms. It is kept because it removes
four avoidable `ttnn.from_torch` calls per token from the measured runtime path, which the `$optimize`
final audit asks to be absent.

### 3.5 Change 3 — a page-table-only refresh builds only the page table

`OrnithGenerator._refresh_page_table_only` asked `prepare_decode_inputs_host` for all four host tensors
and used one. It now passes `page_table_only=True`. Same status as §3.4: correct-by-construction removal
of avoidable work (three `ttnn.from_torch` per page-table refresh — 51 of them on the final
`max_num_seqs=1` server, 56 at 32), not a latency claim. Pinned by
`test_a_page_table_only_refresh_builds_only_the_page_table`.

---

## 4. Serving-path gates, before and after

| gate | before | after | artifact |
|---|---|---|---|
| canonical sampling suite, `--sampling-profile full`, `max_num_seqs=1` | 65 passed / 7 failed / 1 skipped | **65 / 7 / 1 — identical set, by name** | [`after/sampling_tests_final_b1.log.gz`](after/sampling_tests_final_b1.log.gz) |
| canonical sampling suite, `max_num_seqs=32` | 54 / 18 / 1 (this stage's before-tree run) | 53 / 19 / 1, then **54 / 18 / 1** on a repeat | [`before/sampling_tests_beforetree_b32.log.gz`](before/sampling_tests_beforetree_b32.log.gz), [`after/sampling_tests_b32.log.gz`](after/sampling_tests_b32.log.gz), [`…_repeat.log.gz`](after/sampling_tests_b32_repeat.log.gz) |
| qualitative, greedy | 6 prompts | **byte-identical to the before tree on all 6** | [`after/qualitative_control_vs_previous_stage.json`](after/qualitative_control_vs_previous_stage.json) |
| `check_degenerate_output.py --scope vllm` | pass | **pass**, "No degenerate output detected" | [`logs/check_degenerate_output.txt`](logs/check_degenerate_output.txt) |
| `check_context_contract.py --stage vllm --require-contract` | pass | **pass**, target 262144 = supported 262144 | [`logs/check_context_contract.txt`](logs/check_context_contract.txt) |
| adapter suite `tests/test_generator_vllm.py` | 22 passed | **26 passed** (4 new cases) | [`logs/pytest_generator_vllm.txt.gz`](logs/pytest_generator_vllm.txt.gz) |
| non-aligned prompt lengths 1, 3, 17, 65, 130, 257, 999, 2049, 4097 | all length-preserved | **all length-preserved**, both configurations | [`after/serving_requests_b1.json`](after/serving_requests_b1.json), [`after/serving_requests_b32.json`](after/serving_requests_b32.json) |
| 32 concurrent requests | all completed | **all completed** | same |
| served `max_model_len` | 262144 | **262144** | same |

**Output quality: the control, and the prompt-format decision.** `$qualitative-check` requires both, and
neither is a prose verdict here. The checkpoint **has** a chat template (`Qwen2Tokenizer`,
`chat_template_present: true`), so the shared runner suite — which posts the bare prompt to
`/v1/completions` — is continuation stress coverage, not the quality verdict; the chat-rendered verdict
against HF and full-model controls is the vLLM-integration stage's
[`qualitative_chat.json`](../vllm_integration/qualitative_chat.json). It still describes this tree,
because **this stage's committed serving completions are byte-identical to that run's on all six greedy
prompts** — the strongest available control, recorded with the comparison script's output in
[`after/qualitative_control_vs_previous_stage.json`](after/qualitative_control_vs_previous_stage.json).
The seedless *sampled* completions differ, as they must across servers. One pre-existing observation
carried forward rather than re-discovered: the greedy Fibonacci completion repeats a code block four
times (`trigram_loop_fraction` 0.1008, under the degenerate-output threshold); it is byte-identical to
the before tree, and it is a raw-completion-prompt artefact of a chat checkpoint, not a serving fault.

**The 7 failures at `max_num_seqs=1` are the same seven, by name**, and are a harness assumption rather
than a model answer: the three `test_different_*_penalties` slice their request list by `max_batch_size`
and are left with one request before asserting on ≥2 distinct outputs, the three
`test_*_penalty_mixed_batch` are left with **none** (two report `Got 0 unique results out of 0.`, the
third raises `IndexError` on the empty list), and `test_uniform_noseed_varied` asserts on variety across
rows there is only one of.

**The 18↔19 movement at `max_num_seqs=32` is run-to-run, measured three ways.** Two consecutive full
runs against the *same after server* on the *same code* gave 19 and 18 failures, differing by
`test_topk[15]` / `test_topk[19]` / `test_frequency_penalty_mixed_batch`. And a **before-tree** run on
this machine, made specifically to control it, gave 18 — including
`test_specific_seed_reproducible[42]`, which the integration stage's older log had passing. So `[42]`
fails on the before tree too and is not this stage's doing. The class (identical greedy requests are not
bit-reproducible above padded decode batch 4) is the open decoder-stage defect the integration stage
measured and handed on; membership within it moves between runs. `serving_requests.json` shows the same
on both sides, and on two of its rows rather than one: `single_user_determinism.identical` and
`null_block_containment.identical` are both `true` at `max_num_seqs=1` and both `false` at 32, before
and after alike (the integration stage's `batch32/serving_requests_max_num_seqs_32.json` is the control).

---

## 5. Secondary: CI serving-burst profile (vLLM-nightly shape)

**Workload: 100-token prompts, 100 output tokens, 32 requests, no `--max-concurrency`, greedy,
`ignore_eos`, `--max-num-seqs 32`.** Not the headline decode number — burst admission and the padded
decode batch dominate its TPOT. Raw
[`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json), normalized
[`vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json).

| metric (100/100/32) | before run 1 | before run 2 | after run 1 | **after run 2** |
|---|---|---|---|---|
| requests completed | 32/32 | 32/32 | 32/32 | **32/32**, 3200/3200 tokens |
| TTFT P50 / P99 | 5912.0 / 5913.3 ms | 4868.0 / 4869.2 ms | 5238.8 / 5239.9 ms | **4859.3 / 4860.7 ms** |
| TPOT mean / P99 | 143.14 / 180.17 ms | 146.65 / 177.84 ms | 150.03 / 182.80 ms | **146.92 / 178.06 ms** |
| ITL P50 / P99 | 141.46 / 147.64 ms | 141.40 / 269.08 ms | 141.40 / 376.59 ms | **141.75 / 269.06 ms** |
| aggregate output throughput | 160.72 tok/s | 166.31 tok/s | 160.49 tok/s | **166.15 tok/s** |
| request throughput | 1.607 req/s | 1.663 req/s | 1.605 req/s | **1.661 req/s** |
| elapsed | 19.91 s | 19.24 s | 19.94 s | 19.26 s |

Run 1 is the first burst on a fresh server and run 2 the repeat, on both sides. Warm against warm: TTFT
P50 4868.0 → 4859.3 ms, TPOT 146.65 → 146.92 ms, throughput 166.31 → 166.15 tok/s. **Unchanged**, as
expected — the burst's 100-token prompts land in physical block 128, which the *old* warm-up already
compiled at length 64, so this workload never paid the cost §3 removes. It is capacity and
vLLM-nightly-parity evidence, and proof the change costs the burst nothing.

The 128/128/1 single-user profile on a `--max-num-seqs 32` server is the other secondary row: 137.738 →
138.130 ms TPOT warm (7.260 → 7.240 t/s/u), also unchanged. Single-user latency and 32-user capacity
remain two deployments — §6 limitation 3.

---

## 6. Serving contract, re-verified on this tree

| | |
|---|---|
| measured path | real vLLM TT-plugin serving through [`tt/generator_vllm.py`](../../tt/generator_vllm.py), class `TTQwen3_5MoeForConditionalGeneration`, registered in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models` |
| async decode | `decode_forward(..., read_from_device=False)` returns device tensors; `read_decode_output(..., async_read=True)` is `cpu(blocking=False)` + `ttnn.record_event` and nothing else; `process_decode_output_host` is host formatting. `supports_async_decode=True`. **`async_reads` == `decode_calls`** on every server: 8955/8955 (`max_num_seqs=1`), 7139/7139 (32) |
| trace replay | `submit_serving_decode` replays the model trace **and** the sampling trace with `ttnn.execute_trace(..., blocking=False)`; `decode_syncs` is **0** across 8955 and 7139 decode steps |
| persistent trace inputs | token / current-position / RoPE / page-table / KV-cache / sampler tensors are the buffers the capture bound. Steady state copies nothing: **7633 of 8955** decode steps (`max_num_seqs=1`) and **6073 of 7139** (32) were `no_refresh_steps`; page-table-only refreshes were 51 and 56 |
| stale-input coverage | changed token/current-position and changed *and* unchanged page tables, on device: `test_the_steady_state_decode_copies_nothing_to_the_device`, `test_a_stale_host_pair_does_not_override_the_device`, `test_only_a_changed_page_table_is_copied` |
| on-device sampling | `sample_on_device_mode: all`, enforced by the runner. No host greedy/top-1 argmax, no full-logits readback, no eager sampling on the measured path; `force_argmax` stays disabled (**0** "Forcing argmax sampling" lines in every after server log), so greedy uses the full-model split-sampling path — local grouped top-k per vocab shard, gather of candidates, `tt_out_tok` written straight into the persistent decode token buffer |
| sampler cost | 1.192 ms of the 23.155 ms step (5.1 %), measured standalone by the datatype sweep, not by profiling the server. It does not dominate token-out decode, so the LM-head/sampling contract was not reopened |
| context contract | served `max_model_len` **262144** = `doc/context_contract.json`; benchmark and eval contexts unchanged; nothing lowered |
| non-aligned lengths | preserved and re-tested (§4). Nothing in this stage rounds, buckets or pads a *request's* length — the 16 warmed lengths are compile targets, not a request bucket set |
| batch capability | `max_num_seqs` 1 and 32 both served, 32 concurrent requests completed, 181 slot remaps exercised on the 32-slot server |
| profiler | **none**. No Tracy, no `tt-perf-report`, no `TT_METAL_DEVICE_PROFILER`, no `ttnn.ReadDeviceProfiler`, no serving-adapter profile — by instruction. Device-op evidence is the earlier non-serving profiles, still current because no device graph changed |
| watcher | not run in this stage, deliberately: `$vllm-integration` asks for watcher notes only when relevant to serving stability, this stage changed no device graph, kernel, memory config or CCL, and the optimized-decoder and full-model stages' watcher-clean runs still describe the graph that executes. Recorded here rather than left implicit |

### Limitations

1. **A first request at a new *logical* prompt length still compiles 3–13 programs and re-captures**:
   1.03–2.25 s when the kernels must be built, 0.03–0.10 s when they need not, plus a ~230 ms stall.
   This is the **largest** residual class — it is the 56-program bucket on a 32-slot server (§3.3 A) —
   and it is an **open candidate, not a closed-off blocker**: an earlier draft called `ttnn.slice`'s
   `List[int]` bounds an exact op contract and review round 3 refuted that from the source (§3.3 A).
   The adapted form and the other members of the class are named there.
2. **A new chunk offset (6 programs) and a slot remap that moves a row index it has not moved before
   (~2 programs per such row, 64 at a first width-32 remap) still compile and re-capture.** Both are
   measured and enumerated (§3.3 B and C), both are logged on the before tree as well as the after one,
   and both have a fix design that was written and reverted unvalidated when the host ran out of RAM
   (§7). They are smaller than limitation 1, which owns the largest residual bucket; a follow-up stage
   should weigh all three together.
3. **Server start-up costs 11.2–14.2 s more on a warm kernel cache, and on the order of 8–9 minutes on
   a cold one** (§3.2 derives the bound: ~0.18 s per program actually built × 2887). It is paid once per
   machine, and on a cold machine it is mostly a relocation of build time that would otherwise land in
   user requests — but a deployment that only ever uses two or three block shapes does build more total
   kernels than before. `ORNITH_VLLM_PREFILL_WARMUP` takes an explicit list, and `min` restores the old
   single-length behaviour.
4. **A single user on a `--max-num-seqs 32` server pays the 32-row decode step**, 138.1 ms/token against
   23.2 at `max_num_seqs=1`. This stage measured *why*, because the obvious theory was wrong — idle rows
   activating extra routed experts. [`before/batch_occupancy.json`](before/batch_occupancy.json) and
   [`before/batch_occupancy_distinct.json`](before/batch_occupancy_distinct.json) drive the real 40-layer
   traced serving step at batch 32 with 1, 2, 4, 8, 16 and 32 active rows, once with every row carrying
   the same token (routed-expert union ≈ 8) and once with distinct tokens (union up to 256): **137.96 →
   143.23 ms/token**, and the two token modes agree to within 0.29 ms at every occupancy (largest gap 0.288 ms at 8 active rows). The expert union
   costs nothing here; the batch penalty is per-row DeltaNet and attention work. Masking idle rows out of
   the routing sparsity was therefore refuted before it was implemented (work log §5.1). Caveat on those
   probes: they allocate for an 8192-token context (`[32, 128]` page table) rather than the served
   262144 (`[32, 4096]`), so they are used for the *shape* of the occupancy curve and its ~0.2 ms scale.
5. **Reproducibility above padded decode batch 4** is unchanged and remains an open decoder-stage defect
   handed on by the vLLM-integration stage; §4 shows this stage neither improved nor worsened it.
6. **Prefix caching is off** and not claimed. **Text only.** **`max_num_seqs` is capped at 32** by the
   sampler. **On-device log-probs need 8 or 32 devices**, so log-prob requests take the plugin's host
   sampler. All four are the integration stage's limitations, unchanged.

---

## 7. How this was run, and one host incident

Each server launch was preceded by `tt-smi -r`, and a `1x4` mesh open/close smoke was run at every
stage boundary and after every recovery — **six** reset records and **four** mesh-smoke records in
[`logs/device_reset_*.txt`](logs/) and [`logs/mesh_smoke_*.txt`](logs/); the resets issued inside a
launch command were not separately captured to a file. There were **13** launch attempts: eight servers
came up and are the ones every number here was measured on, and five were OOM-killed during weight load
([`logs/host_oom_kills.txt`](logs/host_oom_kills.txt) names all five by pid). Nine of the thirteen left a
console log in `logs/`; two of the OOM-killed retries overwrote the same `server_final2_b1.txt` and two
left only the kernel's record, which is why the OOM artifact rather than the log count is the authority
on how many there were. No profiler env var was set at any
point. The `before` arms were produced by `git stash`-ing this stage's source files, so the before tree
is literally `81c01e1360a`'s code and not an approximation of it.

```bash
# the server (identical on both sides; --max-num-seqs 32 for the CI-burst side)
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b \
  --hf-model ornith-ai/Ornith-1.0-35B \
  --mesh-device "(1, 4)" \
  --max-num-seqs 1 \
  --max-model-len 262144 \
  --server-timeout 2400 \
  --port 8100 \
  --tt-config '{"trace_region_size": 200000000, "l1_small_size": 24576, "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192}'

# the cold-JIT arm is the same command with an empty kernel cache
TT_METAL_CACHE=/home/ttuser/.cache/ornith-coldjit-before python -m models.common.readiness_check.run_vllm_server --stages serve ...

# the checks, attached to it
python -m models.common.readiness_check.run_vllm_server --stages sampling   ... --max-num-seqs 1 --sampling-profile full --server-url http://localhost:8100
python -m models.common.readiness_check.run_vllm_server --stages qualitative ... --max-num-seqs 1  --server-url http://localhost:8100
python -m models.common.readiness_check.run_vllm_server --stages benchmark   ... --max-num-seqs 1 --no-benchmark-ci-serving --server-url http://localhost:8100   # run twice: cold + warm
python -m models.common.readiness_check.run_vllm_server --stages benchmark   ... --max-num-seqs 32 --server-url http://localhost:8100                            # primary@32 + the CI burst

# stage-owned probes and checks
python .../doc/optimized_vllm/logs/probe_new_length_cost.py      --url http://localhost:8100 --lengths 211,347,613 --output .../new_length_cost.json
python .../doc/optimized_vllm/logs/probe_batch_occupancy.py      --token-mode distinct --output .../batch_occupancy_distinct.json      # no server; drives the generator
python .../doc/optimized_vllm/logs/probe_prefill_program_keys.py --name-them --lengths 64,100,128,200,211,222 --output .../prefill_program_keys_named.json
python .../doc/optimized_vllm/logs/probe_prefill_program_keys.py --warm-blocks --context 16384 --lengths 2049,3000,4097,6145,8193,10241,10300 --output .../prefill_chunk_offsets.json
python .../doc/optimized_vllm/logs/probe_slot_program_keys.py    --output .../slot_program_keys.json
python .../doc/vllm_integration/logs/probe_serving_requests.py   --url http://localhost:8100 --server-label "..."
python models/common/readiness_check/check_degenerate_output.py --hf-model ornith-ai/Ornith-1.0-35B --missing-artifacts critical --scope vllm
python .agents/scripts/check_context_contract.py --model-dir models/autoports/ornith_ai_ornith_1_0_35b --hf-model ornith-ai/Ornith-1.0-35B --stage vllm --require-contract
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -q
```

**The host incident, and what it cost this stage.** After the cold-JIT `before` arm completed, the host
ran out of RAM. The kernel OOM-killed **five** `VLLM::EngineCor` processes and, later, **three** reduced-target
pytest processes ([`logs/host_oom_kills.txt`](logs/host_oom_kills.txt), captured from `dmesg -T` and
`journalctl -k`; `anon-rss` between 0.7 and 7.7 GB). The devices stayed healthy throughout — `tt-smi -r`
returned 0 with all eight board rows and the mesh smoke printed `MESH_SMOKE_OK` after every attempt — so
this is a host-memory condition, not a device or model one; the machine reports ~232 GB "used" against
~4 GB of process RSS and ~21 GB accounted anywhere in `/proc/meminfo`, and it predates this stage
(`free` showed 227 GB used before its first command).

Recovery followed `$tt-device-usage`: no live device process was confirmed, then the stale
`/dev/shm` state left by killed tt-metal processes was cleared — **4015 files** (the count the cleanup log records), almost
all of them `sm_segment.*` segments, plus the `TT_UMD_LOCK.*` set, one of which was still held by a dead
OOM victim and had blocked a mesh open with `Waiting for lock 'CHIP_IN_USE_0_PCIe' … held by thread TID: 3663767` — the
page cache was dropped, and the `tenstorrent` module was reloaded at `refcnt 0`. The mesh smoke passed
after the cleanup (and again after a second cleanup when a pytest process was killed the same way);
none of it returned enough RAM to load a 35B model. A host reboot is the remaining step and belongs to
the operator: it would kill the multigoal runner and five unrelated `tt-studio` service containers.
[`logs/shm_lock_cleanup.txt`](logs/shm_lock_cleanup.txt),
[`logs/driver_reload_recovery.txt`](logs/driver_reload_recovery.txt),
[`logs/device_reset_after_coldjit.txt`](logs/device_reset_after_coldjit.txt),
[`logs/mesh_smoke_after_cleanup.txt`](logs/mesh_smoke_after_cleanup.txt).

**What that cost, named rather than estimated away.** Three measurements and two code changes:

* the **cold-JIT `after` arm** of §3.1 — composed instead from the same cold run's already-built-block
  rows (1.03 and 2.25 s), which is the class the after tree still pays, and labelled as composed;
* a **chat-rendered qualitative rerun** on this tree — covered instead by the byte-identity control (§4);
* a **re-run of the full adapter suite** after the round-2 edits;
* the **chunk-offset warm-up** and the **slot-remap warm-up** (§3.3 B and C) were written and then
  **reverted**, because a serving-path change that cannot be validated on the served 40-layer
  configuration is not a change worth shipping. The code in this tree is exactly the code the committed
  26/26 adapter-suite run ([`logs/pytest_generator_vllm.txt.gz`](logs/pytest_generator_vllm.txt.gz))
  and all three full serving evidence runs exercised; only docstrings were corrected afterwards. The
  two abandoned attempts are recorded with their measurements in
  [`work_log.md` §6](work_log.md#6-what-was-not-done-and-what-it-would-take).

Everything the goal contract requires was measured before the host degraded, on servers whose logs are
committed.

---

## 8. Artifacts

`readiness_vllm/` is last-writer-wins, so what is committed there is stated explicitly: `server.log`,
`sampling_tests.log`, `vllm_qualitative_outputs.json`, `vllm_benchmark.*` and `vllm_result.json` are the
**final `max_num_seqs=1` after server**, and `vllm_ci_serving_*` are the **`max_num_seqs=32` after server,
run 2**. Byte-identical copies of each live in [`after/`](after/) under names that say which server they
came from; `vllm_serving_capability_exit_*_server.json` names the server whose exit wrote it rather than
using a bare `_final` suffix. Two provenance details worth stating rather than leaving to inference:
`readiness_vllm/vllm_serving_capability.json` is the **final** b1 server's *warm-up-time* report (written
at 05:53:55, all runtime counters zero, byte-identical to the first server's warm-up copy in `after/`)
while `…_final.json` is that server's exit report; and
[`logs/check_degenerate_output.txt`](logs/check_degenerate_output.txt) was **re-run in review round 4**
against the committed artifacts, because the earlier copy had measured the *first* b1 server's
qualitative outputs (identical greedy completions, different seedless sampled ones) and was then
overwritten by the final server. The committed log and
`readiness_vllm/vllm_qualitative_outputs.json` now share a sha
(`52471ba4…`, equal to [`after/vllm_qualitative_outputs_final_b1.json`](after/vllm_qualitative_outputs_final_b1.json)).

| directory | what is in it |
|---|---|
| [`before/`](before/) | every `before` measurement made in this stage, all on the stashed `81c01e1360a` tree: both primary benchmark passes at `max_num_seqs` 1 and 32, both CI-burst passes, the `max_num_seqs=32` sampling suite, the new-length probes in three cache/config regimes, and the two batch-occupancy probes |
| [`after/`](after/) | the same set on this tree, plus the sampling logs, qualitative outputs and their control, the serving-request probes, and the three exit capability reports |
| [`candidates/`](candidates/) | the prefill program-key counts, the named cache-miss run, the per-server re-capture class census, the chunk-offset enumeration and the slot/remap enumeration |
| [`logs/`](logs/) | every console log, the four stage-owned probe sources, the reset / mesh-smoke / cleanup / driver-reload / OOM records, and the adapter test-suite log (plus the OOM-killed re-run attempt, kept as `…_oom_killed_attempt.txt.gz`) |
| [`before_after.json`](before_after.json) | the §1/§5 tables, machine-readable |
| [`new_length_cost_before_after.json`](new_length_cost_before_after.json) | the §3.1 table, machine-readable, with each arm's tree and cache state |
| [`perf_summary.json`](perf_summary.json) | the `$optimize` performance-accounting shape, device-time fields `null` with the no-profiler reason named |
| [`work_log.md`](work_log.md) | the stage narrative: what was tried, what was refuted, what was rejected and why |
