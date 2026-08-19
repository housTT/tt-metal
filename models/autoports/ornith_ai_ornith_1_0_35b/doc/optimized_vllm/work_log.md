# Optimized vLLM serving — work log

Stage: optimized-vLLM for `ornith-ai/Ornith-1.0-35B` on a 4-chip Blackhole `p300c` ring (`1x4`,
`FABRIC_1D_RING`). Skills: `$vllm-integration`, `$optimize`, `$tt-device-usage`, `$tt-enable-tracing`,
`$qualitative-check`, `$stage-review`. Results are in [`README.md`](README.md); this file is what was
tried, in order, including the things that turned out to be wrong — this stage's first framing of its
own headline among them (§5.5).

---

## 1. What the stage started from

The [vLLM-integration stage](../vllm_integration/) closed with a serving path that already satisfied most
of this stage's contract: async decode split, non-blocking `ttnn.execute_trace`, persistent trace inputs
refreshed only on scheduler state changes, `sample_on_device_mode=all` on the full-model split-sampling
path, vLLM-owned KV cache, the full 262144-token context, and non-aligned prompt-length support. Its
README claimed warmed serving decode was already at the model's own token-out floor.

So the first job was not to optimize. It was to find out whether that claim holds on this tree, and if
it does, where the remaining serving cost actually is. Two consequences:

* every "before" number here was re-measured in this stage. Where a before/after pair matters, the
  before arm was produced by `git stash`-ing this stage's four source files, so it is literally
  `81c01e1360a`'s code running on this machine, not the integration stage's committed numbers;
* the search started with measurement, not with a change.

## 2. The audit: where the serving step's time goes

**Warmed steady state, `max_num_seqs=1`.** Re-measured before touching anything: TPOT 23.144 ms, ITL P50
23.119 ms ([`before/vllm_result_b1_warm.json`](before/vllm_result_b1_warm.json)). The datatype sweep's
standalone traced token-out decode at the same shape is 23.1647 ms/token over nine warm repeats.
**Serving is 0.02 ms/token below the standalone generator** — inside both harnesses' repeat spread.
There is no serving-specific decode overhead at batch 1. The decomposition, from the same artifact:
21.45 ms decoder layer stack, 0.52 ms terminal (final norm + LM head + logits), 1.19 ms sampling trace,
0.008 ms readback. None of it is orchestration. One disclosure: that baseline was measured with
`cache_context 8192` / 128 blocks and serving runs 262144 / 4096, so the two allocations differ; the
comparison carries its ~0.02 ms scale, not an exact equality.

**Warmed steady state, `max_num_seqs=32`.** Serving TPOT 137.738 ms
([`before/vllm_result_b32_run2.json`](before/vllm_result_b32_run2.json)) against 137.96 ms/token for the
same traced serving step driven directly, without vLLM
([`before/batch_occupancy.json`](before/batch_occupancy.json), 1 active row; same allocation caveat).
Serving overhead is again ~0, and the 6× penalty against batch 1 is the model's.

That closes the `$optimize` question "is vLLM decode slower than the generator's traced decode" — it is
not, at either batch — and moves the search to the parts of serving that are not the steady state.

## 3. The defect: compiling prefill programs while the decode traces are live

The integration stage recorded "+77 ms TTFT and one ~220 ms stall" for a first request at a new prompt
length, measured at length 128 on a server that had already compiled 64. Measured at lengths a
deployment would actually see, on the unmodified tree
([`logs/probe_new_length_cost.py`](logs/probe_new_length_cost.py)), the first request cost between 0.37
and 8.6 s of extra TTFT, and the server log named the trigger per request:
`re-capturing the decode traces: 102 program(s) were compiled after capture` (211), `115` (347), `138`
(613).

`$autofix` was not needed: the mechanism was already documented by the integration stage
(`_ensure_traces_replay_safe`, a real corruption rather than a precaution), the logs named the trigger,
and the open question was quantitative — *which* programs, keyed by *what*. That is a probe, not a
repair loop.

[`logs/probe_prefill_program_keys.py`](logs/probe_prefill_program_keys.py) answered it by driving the
generator's serving primitives at a series of lengths and counting `num_program_cache_entries()` deltas
and `trace_recaptures` ([`candidates/prefill_program_keys.json`](candidates/prefill_program_keys.json)):
a length that opens a new **physical** block (a multiple of `PREFILL_ALIGN` = 128) compiles 101–114
programs; another length inside an already-compiled block compiles 3 (aligned) to 8 on that reduced
target, and up to 13 on a real server. There are exactly
16 physical blocks, because prefill pads every internal block to a multiple of 128 and chunks at 2048.

The expensive class is finite and enumerable, so it can be compiled at start-up, before the decode
traces exist.

## 4. The three changes

### 4.1 `prefill_warmup_lengths()` and a 16-length `warmup_model_prefill` (the one that matters)

vLLM's warm-up phase 1 now runs one prompt at each of the 16 physical block lengths, longest first
(largest activation footprint allocated when the heap is least fragmented), before phase 2 captures the
decode traces. Any prompt is full 2048-chunks plus one of those tails, so the set is complete **for
block shapes** — not for the two other keys §6 measures, the logical length and the chunk offset. `ORNITH_VLLM_PREFILL_WARMUP` overrides it (`all` / `min` / an explicit list); `min` is
the pre-optimization behaviour, and the test suite pins two lengths because its device fixture rebuilds
an adapter per test.

The mechanism-level result is the claim this stage stands behind, stated with its qualifier:
[`candidates/recapture_classes.json`](candidates/recapture_classes.json) buckets every re-capture every
server logged by how many programs had been compiled since the capture. **Every before-tree server that
was asked for a prompt in a block it had not compiled shows the block class (102, 115, 138, 175, 271
programs) — four of the five; the fifth served only the sampling suite, whose prompts all land in the
128 block the old warm-up already compiled at length 64. No after server shows the class at all**; its
largest buckets are 19 at `max_num_seqs=1` and 64 at 32, which §6 identifies as the chunk-offset and
slot-remap classes.

### 4.2 The sampling-parameter push is skipped while nothing changes

`SamplingGenerator.apply_decode_state` builds four host tensors and copies each to the device; the
adapter called it on every device-sampled decode step. `_apply_decode_sampling` now compares the
formatted parameter row against the last one pushed and skips when they match, dropping the key on
`reset_batch`, on any prefill, and on a serving reset. Seeds and RNG counters are still advanced every
step.

Measured effect on the headline: none — four `[32]`-wide copies against a 23 ms replay on the same
in-order queue. Kept because `$optimize`'s final audit asks for no unnecessary `from_torch` in the
measured runtime path and this removed four per token, and because the counters make it auditable:
7684/8093 device-sampled steps skipped on the final `max_num_seqs=1` server, 6129/6388 at 32.

### 4.3 `page_table_only=True`

`_refresh_page_table_only` built four host tensors and used one. Same status as 4.2.

Four new tests pin all three behaviours plus the default warm-up set:
`tests/test_generator_vllm.py` 22 → 26 cases, all passing.

## 5. What was tried and refuted

### 5.1 "Idle rows in a padded serving batch activate extra routed experts" — refuted before implementing

The obvious explanation for the 6× penalty a single user pays on a `--max-num-seqs 32` server was the
MoE routing sparsity: the MoE reduces the router's dense scores over the whole 32-row tile to build its
`sparse_matmul` sparsity, `valid_tokens` restricts that reduction to real rows, and `valid_tokens` is a
Python slice bound baked into the captured trace — so at `max_num_seqs=32` it is 32 and every slot's
scores enter the union, including slots whose `current_pos` is the `-1` idle sentinel. The proposed fix
was a device-side row mask from `current_pos >= 0` folded into the dense scores inside the traced graph,
legal because `ttnn.sparse_matmul` infers its non-zero count at runtime.

The prediction that makes it testable: if the union is what costs, driving the batch-32 traced serving
step with 1 active row and 31 idle rows carrying **distinct** stale tokens (union up to 256) must be
materially slower than the same step with every row carrying the **same** token (union ≈ 8).

[`logs/probe_batch_occupancy.py`](logs/probe_batch_occupancy.py) ran exactly that on the real 40-layer
model, through `submit_serving_decode` plus the pipelined async read, three repeats per arm:

| active rows | shared token (union ≈ 8) | distinct tokens (union ≤ 256) |
|---|---|---|
| 1 | 137.964 ms | 137.976 ms |
| 2 | 138.145 | 138.121 |
| 4 | 138.388 | 138.561 |
| 8 | 139.139 | 139.427 |
| 16 | 141.020 | 141.066 |
| 32 | 143.229 | 143.190 |

The two token modes agree to within 0.29 ms at every occupancy (largest gap 0.288 ms at 8 active rows) and the whole 1→32 range spans 5.3 ms out of
138. **The routed-expert union costs essentially nothing at a 32-row group**, so the mask would have
bought nothing; the batch penalty is per-row DeltaNet recurrent-state and paged-attention work, real for
real rows and wasted only for idle ones. The change was not written.

### 5.2 "The 18 → 19 sampling failures at `max_num_seqs=32` are a regression" — refuted, three ways

The first after run of the canonical suite at 32 reported 19 failures against the integration stage's
18. A second full run against the **same server** on the **same code** gave 18, differing by
`test_topk[15]` / `test_topk[19]` / `test_frequency_penalty_mixed_batch`. Membership moves run to run.
A stage review then pointed out that one member — `test_specific_seed_reproducible[42]` — failed on both
after runs and passed in the integration stage's older log, which the after-vs-after control does not
cover. So a **before-tree** run was made on this machine specifically to control it
([`before/sampling_tests_beforetree_b32.log.gz`](before/sampling_tests_beforetree_b32.log.gz)): 18
failures, `[42]` among them. It fails on the before tree too. At `max_num_seqs=1` the failure set is
identical by name on both trees, 7/65/1.

### 5.3 "Pad the prefill tail to a coarser bucket so fewer program sets exist" — rejected with a reason

The alternative to compiling 16 blocks at start-up is to make requests share fewer of them by rounding
the tail chunk up at request time. Rejected: `PREFILL_ALIGN` is already 128, so a coarser bucket means
real extra prefill work on **every** request forever — a 211-token prompt rounded to 512 instead of the
current 256 doubles its tail compute — in exchange for a one-off that start-up absorbs. It also trades
warm TTFT, the headline metric, for cold TTFT. The warm-up changes what no request computes; the 16
warmed lengths are compile targets, not a request bucket set.

### 5.4 "Move the re-capture into `prefill_forward` so it lands in TTFT instead of ITL" — rejected

It would move ~230 ms from the P99 ITL row to the TTFT row without removing a millisecond of work.

### 5.5 This stage's own first headline was confounded — corrected

The first version of §2 of the README claimed the first-request penalty fell from 0.37–8.64 s to
0.09–0.15 s, a factor of 4–70×. Two things were wrong with it, and both were found by re-measuring
rather than by argument:

* the *before* long rows had been taken on a `--max-num-seqs 32` server and the *after* rows on a
  `max_num_seqs=1` server, so part of every ITL delta in that table was just the 138 ms vs 23 ms decode
  step. A stage review caught it;
* more seriously, the two arms were in different **tt-metal JIT cache** states. The before arm was the
  first time those prefill shapes had ever been built on this machine; by the time the after arm ran,
  the before arm had built them, so "compiling a program" meant loading a cached kernel. Re-running the
  before tree against the now-warm cache gave 33–293 ms of excess — no worse than the after tree.

The fix was to control the variable rather than to soften the wording: `TT_METAL_CACHE` was pointed at
an empty directory and the before tree re-measured cold
([`before/new_length_cost_coldjit_*.json`](before/)), and both trees were also measured warm at
`max_num_seqs=1`. The corrected §3.1 reports both regimes: **10.0–14.3 s per new physical block when the
kernels must be built, 0.12–0.29 s when they need not (the residual already-built rows are 0.03–0.10 s),
and no measurable warm-cache difference between the trees.** The claim this stage stands behind is the mechanism-level one in 4.1, which no cache state
can confound.

The matching cold-JIT *after* arm could not be measured: the host ran out of RAM (§7) and OOM-killed the
two launches attempted for it, and three more after that. It is composed instead from the same cold-JIT run's rows whose physical
block was already built (1.03 s and 2.25 s), which is exactly the class the after tree still pays, and
labelled as composed rather than measured.

## 6. What was not done, and what it would take

Three compile classes survive the 16-block warm-up. All three are measured, and **none of them is
closed off by a ttnn contract** — the round-3 review refuted the one this stage had claimed (A below).
All three are open work that this host can no longer validate (§7).

**A. The logical-length class — the largest residual.** A request at a logical length no earlier request
used compiles 3–13 programs and pays one re-capture (1.03–2.25 s cold, 0.03–0.10 s warm). It is also the
biggest bucket the after servers log: the 56-program re-capture on the `max_num_seqs=32` server follows a
21-request prefill carrying eight new logical lengths, and that server prices a single new non-aligned
length at 7 programs fourteen times over — 8 × 7 = 56. Review round 4 re-attributed that bucket here from
C, by log ordering; the before tree reproduces it with the identical eight lengths. `set_program_cache_misses_allowed(False)` names the first offender —
`EmbeddingsDeviceOperation`, because `ttnn.embedding` runs on the *unpadded* `[1, logical]` token row —
and the rest is the tail pad/trim, the last-token slice at `[0, logical-1, 0]`, the DeltaNet gate ramp
comparison against `float(logical_len)`, and the conv tail slice at `logical_len`. Two things were
checked before leaving it:

* the cheap half-measure — padding the token row to the physical block before `ttnn.embedding` — trades
  one program for another, because the embedding output then has to be sliced back to the logical
  length. Not implemented for that reason, not because it failed;
* **the claim that `ttnn.slice` has no tensor-valued form was wrong**, and review round 3 refuted it from
  the source. `slice_nanobind.cpp` binds an overload taking `starts`/`ends` as device tensors plus
  `slice_dim`/`num_devices`; `slice.cpp` routes it to `ttnn::prim::slice(..., use_tensor_args=true, ...)`
  with dummy bound shapes, and `SliceDeviceOperation::compute_program_hash` hashes only those dummies,
  `slice_dim`, `num_devices` and the input/output specs — the bounds are runtime values.
  `ttnn.slice.__doc__` documents only the `List[int]` overload, which is how this stage got it wrong;
  `$tt-enable-tracing` line 83 had it right and was overruled on weaker evidence. That is the worst kind
  of mistake this review process exists to catch: an "exact op-contract blocker" that closed off the
  stage's largest remaining per-request cost and was refutable without any hardware.

What the overload does *not* give is an arbitrary window: `compute_output_specs` splits `slice_dim` into
`num_devices` equal parts and `validate_on_program_cache_miss` still wants a TILE input's output height
tile-aligned, so the single last-token row (output height 1) is not a direct substitution. The adapted
form is the tile-aligned 32-row window containing the last token (`slice_dim=1, num_devices=phys/32`,
keyed by the physical block rather than the logical length) plus a within-tile row selection. It is
recorded as a **named, unverified candidate**: it was found after the host stopped being able to run this
model (§7), so it has not been tried. The other members of the class — `ttnn.embedding` on the unpadded
row, the tail pad/trim, the ramp comparison, the conv tail — still need their own dispositions.

**B. The chunk-offset class — measured, fix written, reverted.** The warm-up's prompts all start at
position 0, so a prompt longer than one 2048-token chunk still compiles the `start_pos` variants of its
later chunks: **6 programs per new offset**, enumerated in
[`candidates/prefill_chunk_offsets.json`](candidates/prefill_chunk_offsets.json). It is **two** of the
19-program re-capture an after `max_num_seqs=1` server logs after a 9000-token prompt, not four: that
server had already served 3000/2049/4097, so only offsets 6144 and 8192 were new (2 × 6 = 12) and the
remaining 7 is A's class paying for the 808-token tail. Correcting that decomposition took two review
rounds, after the bucket was first attributed to multi-slot prefill entirely. The fix is a bounded
long-prompt pass in the warm-up: one prompt of the longest length a deployment expects covers every
offset it spans, cumulatively. It was not landed — covering the advertised 262144 context means
prefilling 128 chunks at start-up, so the sensible form is an env-named bound, and there was no way to
validate it on the served configuration once the host degraded.

**C. The slot-remap class — measured, fix written, reverted.** `remap_state_slots` compiles per
permutation *width*: 4 programs at widths 2 and 4, 8 at width 8
([`candidates/slot_program_keys.json`](candidates/slot_program_keys.json)). The same probe rules out the
alternative explanation: after slot 0 has been prefilled, prefilling into slots 1–7 compiles **0**
programs each, so the class is the remap and not the multi-request prefill. This class explains the **64** bucket on a `max_num_seqs=32` server — it follows the first width-32
remap by 3 ms with no new logical length pending, on the after tree (05:16:55.507) *and* on the before
tree (03:14:17.215) — and not the 56, which review round 4 re-attributed to A. The counts fit ~2 programs
per row index not moved before: 4/4/8 at widths 2/4/8 on the probe, 20 at width 10, 64 at a first width
32, and only 32 on the before-tree re-run whose earlier widths had already moved half its rows. That
also makes the fix smaller than "one permutation per width": a single full-width rotation covers every
row index, bounded by 2 × `max_batch_size`. The fix is three lines in the
warm-up — one permutation of each width over still-zeroed state, before capture, at most
`max_batch_size - 1` of them — and it was written together with a device assertion that a warmed width
compiles nothing. Both were reverted: the only machine that runs this model could no longer start a
40-layer server or finish the adapter suite (§7), and a serving-path change that cannot be validated on
the served configuration is not worth shipping. The tree therefore carries exactly the code the
committed 26/26 suite run and all three serving evidence runs exercised.

**D. Two unused host tensors, on the standalone path only.** `generator.py::_write_tokens` builds two
host tensors it drops and `_write_positions` one — the same waste §4.3 removed from the page-table
refresh. **Neither is on the serving path**: both are reached only from `generate()` (directly, or via
`_first_token_after_prefill`, whose one call site is inside `generate`), and serving uses
`stage_serving_decode_inputs` / `submit_serving_decode`. Two earlier drafts of this row got that wrong,
first attributing both to the serving path and then only `_write_tokens`; review round 4 traced the call
sites. A `want=(...)` generalisation of `prepare_decode_inputs_host` was
written and reverted with B and C, for the same reason.

**E. Batch-bucketed decode traces.** The only remaining lever on the single-user-on-32-slots penalty is
capturing decode traces at several batch sizes and replaying the smallest that covers the active rows —
138 → ~23 ms for one active user, by §5.1's numbers. The model already switches state packs
(`_use_pack`), but the DeltaNet recurrent state lives *per pack*, so a request decoding in the batch-1
trace would need its per-slot state moved between packs whenever occupancy changes, and the sampler,
token buffer and page table are all `max_batch_size`-shaped. Not attempted: it is a model-level
capability change, it does not touch either required workload, and it would put a currently-clean
serving path at risk.

**F. The fabric packet size the runtime advises — already decided, by an earlier stage.** `Fabric
packet size 8192 B is suboptimal for transporting 1088 B pages. Configure 4352 B packet size to maximize
throughput` fires 68 times in the final after server's log, in 17 groups of four: 16 inside the warm-up
prefills and **one inside the eager decode-path compile** (05:53:54.323, between `warm-up: compiling the
eager decode path` at 54.063 and the next decode marker at 54.633). The before tree shows the same shape
(4 of its 20 at 03:03:03.207). An earlier draft of this row said every occurrence was in the prefill
path; that is wrong, and the decode graph does transport 1088 B pages over the fabric.

It is nevertheless not this stage's call to make, and the rejection is earned by measurement this stage
should have cited from the start: `doc/multichip_decoder/README.md` §5's alternatives table records a
generated 4352-vs-8192 CCL census over every traced row of both operand dtypes — "bf16 favours 8192 B on
the large majority, by up to 18%… block-float splits almost evenly with ±2–3% extremes and no consistent
direction" — and takes 8192 deliberately; `doc/full_model/work_log.md`'s advisory table classifies the
warning as inherited from that decision. `fabric_router_max_packet_bytes` is also part of the
`--tt-config` every before and after number here was measured under, so re-opening it needs a fresh
before/after pair of servers the host can no longer start.

## 7. Device usage and the host incident

`$tt-device-usage` throughout: one hardware-facing command at a time, every server shut down before the
next, and `tt-smi -r` before each launch with a `1x4` mesh open/close smoke at every stage boundary and
after every recovery (six reset records, four mesh-smoke records). There were **13** launch attempts:
eight servers came up and carry every number here, and five were OOM-killed during weight load — README
§7 reconciles that against [`logs/host_oom_kills.txt`](logs/host_oom_kills.txt), which is the authority
because two of the killed retries overwrote one console log and two left only the kernel's record.

**No profiler evidence was collected or attempted**, per `$vllm-integration`, `$optimize` and
`$tt-enable-tracing`: no Tracy, no `tt-perf-report`, no `TT_METAL_DEVICE_PROFILER`, no serving-adapter
profile, no `ttnn.ReadDeviceProfiler`. The device-op split this stage quotes comes from the
optimized-full-model and datatype-sweep stages' non-serving profiles, still current because no device
graph changed. `perf_summary.json` carries `null` device-time fields with that reason named. **Watcher**
was also not run, deliberately and for the same reason: this stage changed no kernel, memory config,
CCL or captured graph, so the optimized-decoder and full-model stages' watcher-clean runs still describe
what executes. Recorded here rather than left implicit.

**The incident.** After the cold-JIT before arm, the host ran out of RAM. The kernel OOM-killed **five**
`VLLM::EngineCor` processes and, later, **three** reduced-target pytest processes, captured from `dmesg -T`
and `journalctl -k` in [`logs/host_oom_kills.txt`](logs/host_oom_kills.txt). The pressure reached outside
this stage's own processes — the same artifact records the kernel killing a desktop session's `pipewire`,
`dbus-daemon`, `systemd` and `(sd-pam)` between 08:04:37 and 08:04:49:

```
Aug 19 07:28:44 … Killed process 3652454 (VLLM::EngineCor) total-vm:160958068kB, anon-rss:6548368kB
Aug 19 08:11:57 … Killed process 3683387 (VLLM::EngineCor) total-vm:161264768kB, anon-rss:7727712kB
Aug 19 09:13:33 … Killed process 3752007 (python)          total-vm:149225912kB, anon-rss:723144kB
```

The devices were healthy at every point — `tt-smi -r` returned 0 with all eight board rows, the mesh
smoke printed `MESH_SMOKE_OK` — so this is host memory, not a device or model fault. The host reports
~232 GB "used" against ~4 GB of total process RSS and ~21 GB accounted anywhere in `/proc/meminfo`
(AnonPages 1.5, Cached 1.3, Slab 0.6, Hugetlb 16), a condition that predates this stage: `free` showed
227 GB used before its first command ran.

Recovery followed the skill. No live device process was confirmed, then the stale `/dev/shm` state left
by killed tt-metal processes was cleared — **4015 files** (the count the cleanup log records), almost
all of them `sm_segment.*` segments, plus the
`TT_UMD_LOCK.*` set, one of which was still held by a dead OOM victim and had blocked a mesh open with
`Waiting for lock 'CHIP_IN_USE_0_PCIe' … held by thread TID: 3663767`. The page cache was dropped and
the `tenstorrent` module was reloaded at `refcnt 0`. The mesh smoke passed after the cleanup, and again
after a second cleanup when a pytest process was killed the same way; none of it returned enough RAM to
load a 35B model. A host reboot is the remaining step and belongs to the operator, not to this stage: it
would kill the multigoal runner and five unrelated `tt-studio` service containers.

What that cost is listed in README §7 and in §6 above: the cold-JIT after arm, a chat-rendered
qualitative rerun, a re-run of the full adapter suite, and two warm-up fixes that were written and then
reverted rather than shipped unvalidated.

One process-hygiene note from earlier in the stage: the first attempt to run the adapter suite with the
new warm-up timed out, because the device fixture builds a fresh mesh — and a fresh adapter — per test,
and 16 compiles per rebuild exceeded the per-test timeout. Fixed by pinning the suite to two blocks
(`ORNITH_VLLM_PREFILL_WARMUP=256,128` at module import), with the shipped default asserted by a
host-only test. A second hazard bit once and is worth recording: waiting on `Application startup
complete` in `readiness_vllm/server.log` matches the *previous* server's log, because the runner
truncates that file at launch rather than before the wait. Every wait after that polls `/health`.

## 8. Files changed

| file | change |
|---|---|
| [`tt/generator_vllm.py`](../../tt/generator_vllm.py) | `prefill_warmup_lengths()` and the 16-length `warmup_model_prefill`; `ENV_PREFILL_WARMUP`; the `_apply_decode_sampling` push cache and its invalidations; four new `serving_counters` |
| [`tt/generator.py`](../../tt/generator.py) | `_host_decode_inputs(..., page_table_only=...)`, used by `_refresh_page_table_only` |
| [`tt/model.py`](../../tt/model.py) | `prepare_decode_inputs_host(..., page_table_only=False)` |
| [`tests/test_generator_vllm.py`](../../tests/test_generator_vllm.py) | 4 new cases; the suite's warm-up pin |
| `doc/optimized_vllm/` | this stage's evidence |
| `readiness_vllm/` | refreshed by this stage's final after servers (provenance in README §8) |

No change to the model's device graph, the precision policy, the KV-cache dtype, the served context, the
page-table contract, or the sampling math.

## 9. Stage review and commits

### Round 1

Verdict `more-work-needed`, three P2 items and nine other concerns. What each became:

| finding | resolution |
|---|---|
| P2: §2's new-length table mixed a `max_num_seqs=32` before server with a `max_num_seqs=1` after server on 4 of 7 rows | **fixed by re-measurement, and it uncovered a second confound.** The whole before arm was re-run at `max_num_seqs=1` on the stashed before tree, and the JIT-cache confound behind it was controlled with `TT_METAL_CACHE`. §3.1 now reports three arms with their tree and cache state named; work log §5.5 records the correction |
| P2: no `$qualitative-check` evidence — chat model judged from raw completions, no prompt-format metadata, no control | **fixed.** [`after/qualitative_control_vs_previous_stage.json`](after/qualitative_control_vs_previous_stage.json) records the prompt-format decision, why the raw-completion suite is stress coverage rather than the verdict, which artifact carries the chat-rendered verdict, and the control: all six greedy completions byte-identical to the before tree. README §4 replaces "reasonable" with it |
| P2: `4.6 s`, `10.91 s`, `25.25 s` in §3.2 unsupported or taken from the previous stage | **fixed.** The table now carries the measured 0.835 s prefill warm-up and 11.02 s / 60.61 s `init engine` from this stage's own before servers, against 22.21 s / 72.04 s after |
| the residual compile class is understated outside `max_num_seqs=1` (64- and 56-program re-captures) | **fixed.** Named as its own class in README §3.3 and work log §6, with the before-tree control showing it is not a regression, and quantified for every server in [`candidates/recapture_classes.json`](candidates/recapture_classes.json) |
| headline cold TTFT quoted the better of two after runs | **fixed.** §1 quotes both, and says which half of the row is robust |
| §1's "run-to-run spread" mixed configurations | **fixed.** Only `max_num_seqs=1` runs |
| `0.6–8.9 s` vs `0.37–8.64 s` inconsistency | **fixed**, and both superseded by the controlled arms |
| generator-floor comparison has an undisclosed allocation mismatch | **fixed.** Disclosed in README §1 and §2 |
| `test_specific_seed_reproducible[42]` had no before-tree control | **fixed by measurement**, §5.2 |
| new tests do not cover the push-cache invalidations | acknowledged as a coverage gap; the reviewer traced every writer and reset and found no path that can skip a push it should have made, and `_prefilled_rows.any()` forces a full refresh after any prefill independently |
| ambiguous `_final` capability-report naming | **fixed.** `vllm_serving_capability_exit_{first_b1,final_b1,b32}_server.json` |
| uncommitted non-stage-owned dirty state | kept out of the checkpoint commit |
| watcher not recorded | **fixed.** README §6 and work log §7 record why it was not run |

### Round 2

Verdict `more-work-needed` again, four P2 items and ten other concerns. Round 2's own findings and what
each became:

| finding | resolution |
|---|---|
| P2: the `max_num_seqs=32` residual class is misattributed to multi-slot prefill, and "the set is complete for prompts of every length" is false — a multi-chunk prompt compiles `start_pos` variants the warm-up never reaches | **fixed by measurement.** Two new probes: [`candidates/prefill_chunk_offsets.json`](candidates/prefill_chunk_offsets.json) enumerates the chunk-offset class (6 programs per new offset) and [`candidates/slot_program_keys.json`](candidates/slot_program_keys.json) shows prefilling into slots 1–7 compiles **0** programs while a remap of width 2/4/8 compiles 4/4/8 — so the 19 bucket is chunk offsets and neither bucket is multi-slot prefill. README §3.3 and §6 above are rewritten around the measurements, and the adapter docstring no longer claims completeness. **Round 4 refined this again**: the 56 is the *logical-length* class (8 new lengths x 7 programs) and only the 64 is the remap |
| P2: the headline "every before-tree server logs 102–271" is falsified by the stage's own census, and §3.2's table omits the disconfirming row | **fixed.** The claim is now conditional on the server having been asked for a prompt in an uncompiled block, all five before rows are in the §3.2 table with the fifth's reason, and work log §4.1 no longer says "all four" |
| P2: the benefit is quoted cold and the cost warm | **fixed.** §3.2 derives the cold-cache start-up bound from this stage's own cold-JIT before server (73.74 s for the 402-entry set against 2.255 s warm ⇒ ~0.18 s per program built ⇒ ~8–9 minutes for 2887), says the after figures are 100 %-cache-hit numbers, reframes the change as a *relocation* of unavoidable build time rather than a saving, and states the case where it is a net loss (a deployment that only ever uses two or three block shapes) |
| P2: the shipped docstring still quotes the superseded 0.6–1.1 s figure | **fixed.** `prefill_warmup_lengths` now quotes 10.0–14.3 s cold / 0.12–0.26 s warm with the cache regime named |
| the TTFT-spread sentence double-counts a run | **fixed.** Two runs, called two |
| "`0 trace re-capture(s)` at warm-up" is non-discriminating | **fixed.** Said so explicitly — every before warm-up prints it too |
| "integration numbers used only as context" contradicted by §4 | **fixed.** The three before rows that do come from the integration stage are named in §1 |
| dmesg quotes have no committed artifact; five kills, not four; 4015 files, not 3989 | **fixed.** [`logs/host_oom_kills.txt`](logs/host_oom_kills.txt) commits both `dmesg -T` and `journalctl -k`; the counts are corrected in both documents |
| "every server launch was preceded by reset + mesh smoke" is not backed for all launches | **fixed.** README §7 states what is actually recorded: six reset records and five mesh smokes for eleven launch attempts |
| `after/vllm_qualitative_outputs.json` is bare-named | **fixed.** Renamed `…_first_b1_server.json` |
| `_write_tokens` / `_write_positions` build unused host tensors | written, reverted unvalidated, recorded as §6 D |
| the fabric packet-size advisory was not considered | recorded as §6 F with the reason it was not tried |
| push-cache invalidation tests | written, reverted unvalidated with the rest; the reviewer's own trace of every writer and reset stands as the argument, and the gap is recorded rather than papered over |

The honest summary of round 2 is that it found the stage's central claim over-stated in three separate
ways — a false universal quantifier, a benefit measured in one cache regime against a cost measured in
another, and a compile class attributed to the wrong mechanism — and that fixing all three made the
result smaller and more defensible: a **relocation** of unavoidable kernel-build time out of request
latency, plus one measured removal (the re-capture stall at every warmed length), with three residual
classes enumerated rather than implied.

### Round 3

Verdict `more-work-needed`. One P1 and three P2s, all documentation defects rather than code defects,
and the P1 is the most important finding of the whole review chain:

| finding | resolution |
|---|---|
| **P1: `ttnn.slice` *does* expose a tensor-valued form** — the stage's only stated op-contract blocker for the residual logical-length class is false in this checkout | **retracted everywhere.** The reviewer read `slice_nanobind.cpp` (overload 1 takes `starts`/`ends` as device tensors plus `slice_dim`/`num_devices`), `slice.cpp` (routes to `use_tensor_args=true` with dummy bound shapes) and `SliceDeviceOperation::compute_program_hash` (hashes only the dummies plus specs). This stage had checked `ttnn.slice.__doc__`, which documents one overload, and used that to overrule `$tt-enable-tracing` line 83, which was right. README §3.3 A, this section, `perf_summary.json` and the shipped docstring now say so, describe the overload's real constraints (even split of `slice_dim` into `num_devices`; TILE output must stay tile-aligned) and classify the adapted 32-row-window form as a **named, unverified candidate** rather than as a blocker |
| P2: the fabric packet-size dismissal claims every advisory is in the prefill path; the logs show one group inside the eager decode-path compile — and the rejection that *is* earned, by `doc/multichip_decoder`, was never cited | **fixed**, §6 F. The placement claim is corrected against the log timestamps and the prior measured 4352-vs-8192 census is cited |
| P2: the shipped docstring's warm-cache range (0.12–0.26 s) matches no artifact | **fixed.** It now quotes 0.03–0.29 s, the range in `new_length_cost_before_after.json`, and cites the file |
| P2: README §7 says five mesh-smoke records; there are four | **fixed** |
| `candidates/recapture_classes.json`'s `what` field still carried the superseded attribution | **fixed** |
| the 3989 `sm_segment.*` breakdown appears in no artifact | **fixed.** Both documents now quote only the 4015 the cleanup log records |
| `logs/check_degenerate_output.txt` names a path renamed after it ran | **disclosed** in §8 with the byte-identity note |
| `_write_positions` has no serving call site | **fixed**, §6 D |
| the 56/64 remap buckets at 32 slots are extrapolated from an 8-slot measurement | **disclosed** as extrapolated, README §3.3 C |
| `readiness_vllm/vllm_serving_capability.json` is the *first* b1 server's warm-up report | **disclosed** in README §8 |
| the disposition of the modified `.agents/skills/tt-device-usage/SKILL.md` was never stated | **stated**: it was already modified in the working tree when this stage began — it is not this stage's edit — and it is excluded from the checkpoint commit along with the untracked `.agents/fast-models-fast-feedback.md` |

The reviewer also corroborated the round-2 revert claim independently, by loguru line numbers: every
`generator_vllm.py` call site the committed 26/26 suite log and the three after-server logs report is
either identical to the current file or offset by exactly the lines the docstring edits add (+11 after
round 2, +13 after round 3's further corrections), and the OOM-killed intermediate run reports a line
number the reverted edit would have produced.
The residual gap they name is real and is stated here rather than argued away: after round 2's edits only
the 10 host-only cases were re-run against the shipped bytes
([`logs/pytest_host_only_final.txt`](logs/pytest_host_only_final.txt)); the 16 device cases last ran
against them at 06:14, before the docstring-only edits. A full re-run was attempted twice more and was
OOM-killed both times, the second time at the very first device case
([`logs/pytest_generator_vllm_reverify.txt.gz`](logs/pytest_generator_vllm_reverify.txt.gz),
`PYTEST_EXIT=137`, and the matching kernel line in
[`logs/host_oom_kills.txt`](logs/host_oom_kills.txt)). Every edit after 06:14 is inside a Python
docstring, which has no runtime effect, and the reviewer's line-number forensics is independent
corroboration — but the direct re-run is missing and the host is why.

### Round 4

Verdict `more-work-needed`. One P1 and four P2s, all documentation or attribution, and two of them are
places where a round-3 fix was applied incompletely or replaced one wrong number with another:

| finding | resolution |
|---|---|
| **P1: the round-3 `ttnn.slice` retraction was not applied everywhere** — README §6 limitation 1 and the shipped `test_a_warmed_physical_block_compiles_nothing_and_never_re_captures` docstring still called it an exact blocker, while §9 claimed "retracted everywhere" | **fixed in both places.** Limitation 1 now points at §3.3 A's "open candidate", names the class as the largest residual, and the test docstring says the same |
| **P2: the 56-program bucket is the logical-length class, not the slot remap** — the stated evidence ("each follows a `remap_state_slots` line") is false | **fixed by re-reading the log.** The 56 at 05:27:37.775 follows a 21-request prefill carrying eight new logical lengths, on a server that prices a single new non-aligned length at 7 programs fourteen times (8 × 7 = 56); the before tree reproduces it with the same eight lengths. The 64 does follow the first width-32 remap with no new length pending, so that half stands. §3.3 A and C, §6 A and C, `candidates/recapture_classes.json` and `perf_summary.json` are all re-attributed. **This is the third round in which a compile class was attributed to the wrong mechanism**, which is worth stating plainly rather than burying |
| P2: the warm-cache ranges are computed over the wrong row sets and the documents disagree | **fixed.** Block class warm = 0.12–0.29 s (the five new-block rows); residual class warm = 0.03–0.10 s (the two already-built rows). The shipped docstring, §3.1, §3.3 A, §6 and `perf_summary.json` now agree, each over its own rows |
| P2: `logs/check_degenerate_output.txt` measured the *first* b1 server's outputs, which the final server then overwrote | **fixed by re-running it** — it is host-only and needs no device. The committed log and `readiness_vllm/vllm_qualitative_outputs.json` now share sha `52471ba4…`, and §8's provenance sentence is corrected |
| P2: §6 D still misplaced the unused-host-tensor waste | **fixed.** Neither `_write_tokens` nor `_write_positions` is on the serving path; both are reached only from `generate()` |
| three pytest OOM kills, not two | fixed in both documents |
| the "+11 lines" forensic offset is now +13 | fixed |
| "within 0.15 ms at every occupancy" is 0.288 ms | fixed |
| `readiness_vllm/vllm_serving_capability.json` is the **final** b1 server's warm-up report, not the first's | fixed |
| §3.2's "the set is complete for prompts of every length" survived round 2 in the README | fixed — scoped to block shapes |
| §3.3 C "present identically on the before tree" — 56 yes, 64 no | fixed |
| §4 does not note that `test_topk[32]` fails on both after b32 runs and passed on the before-tree control | noted below |

On the last point: `test_topk[32]` fails on both after `max_num_seqs=32` runs and passed on the
before-tree control, while `test_topk[15]` and `[19]` move the other way between the same runs. Both
directions sit inside the disclosed reproducibility class — three runs on this machine give 19, 18 and 18
failures with membership churning in both directions, and the class is bit-reproducibility of identical
greedy requests above padded decode batch 4, which this stage did not touch — but the asymmetry belongs
in the record rather than only in the aggregate.

Round 4 also independently confirmed the round-2 revert claim in a way this stage could not: `git fsck`
recovered the stash commit `06dd1ea31bd`, taken 24 minutes after the 26/26 suite run, and diffing the
working tree against it shows `tt/generator.py`, `tt/model.py` and `tests/test_generator_vllm.py`
byte-identical and `tt/generator_vllm.py` differing only in the two docstrings.

### Round 5

Verdict `more-work-needed`. Two P1s and three P2s, again all documentation, and again two of them are
places where a *previous round's fix* introduced a new error:

| finding | resolution |
|---|---|
| **P1: README §3.3 C's before-tree control for the 64-program bucket is false, and the round-4 edit that introduced it withdrew a true statement** — the 64 *is* reproduced on the before tree (`server_before_b32_vllm.log.gz` 03:14:17.215), and the before-tree re-run logs 33 width-32 remaps rather than "never condensing 32 rows" | **fixed, and it produced a better mechanism.** §3.3 C now cites both trees' 64 and reads the counts as **~2 programs per row index not moved before** — 4/4/8 at widths 2/4/8 on the probe, 20 at width 10, 64 at a first width-32 remap, and 32 on the re-run whose earlier widths had already moved half its rows. That is a stronger reading than "per width" and it shrinks the fix to one full-width rotation at warm-up |
| **P1: the `ttnn.slice` retraction was still not applied to work log §6's lead-in** ("the first is blocked by a ttnn contract") | **fixed.** All three residual classes are now described as open and host-blocked |
| P2: the 19-program bucket is mis-decomposed — 4 new offsets would be 24, not 19 | **fixed by re-reading the log.** That server had already served 3000/2049/4097, so the 9000-token prompt introduces only offsets 6144 and 8192 (2 × 6 = 12) plus one new 808-token tail length (7). 12 + 7 = 19, in README §3.3 B, work log §6 B, `recapture_classes.json` and `perf_summary.json` |
| P2: four corrections §9 recorded as landed were never applied to the work log (the "complete for every prompt length" sentence, the 0.15 ms occupancy gap, the 3989 file count, the 0.03–0.29 s block-class range) | **all four applied.** They had been fixed in the README only |
| P2: server-launch bookkeeping disagrees across documents and undercounts | **fixed.** There were **13** launch attempts: eight came up, five were OOM-killed. Nine left a console log; two retries overwrote the same file and two left only the kernel's record, so `host_oom_kills.txt` is the authority. README §7 says so |
| the shipped docstring said 3–8 programs where the docs said 3–13 | fixed — both now say 3 aligned, 7–13 non-aligned, with the reduced-target and real-server figures distinguished |
| "those after figures are **all** warm-kernel-cache numbers" — the 14.9 s one logs 99.6 %, not 100 % | fixed |
| work log §3 quoted "0.6 to 8.6 s" for an arm whose lowest row is 0.37 s | fixed |
| the OOM pressure also killed desktop-session processes | added to README §7 |

Five rounds is a lot, and the pattern is worth naming rather than leaving in the table: every round after
the first found a claim that was *directionally* right and *specifically* wrong — a universal quantifier
that had an exception, a benefit and a cost measured in different regimes, a bucket attributed to the
wrong one of three mechanisms, then the same bucket mis-decomposed within the right mechanism, then a
control that had been true and was edited into being false. None of them changed what the stage shipped;
all of them changed what the stage *claimed*. The measurements were stable throughout — every latency,
counter and census figure re-derived identically in all five rounds — and it was the prose over them that
kept needing correction.

### Round 6

**`clean-pass`**, with six editorial items the reviewer asked to be folded into the closing commit
rather than into a seventh round. All six are applied: the README's opening paragraph now quotes the
block class's warm cost over the block rows (0.12–0.29 s) like every other place; limitation 2 no longer
calls the 64-program remap bucket extrapolated, because round 5 showed it logged on both trees; the
launch/OOM bookkeeping is reconciled to 13 attempts / 8 up / 5 OOM-killed in the work log and
`perf_summary.json` as well as the README; the desktop-session processes the OOM killer also took are
recorded; a duplicated sentence in §3.2 is gone; and the reduced-target cold warm-up artifact's full
spread (65.4, 6.3 and 147.8 s across one suite run's three adapter builds) is quoted where the 65.4 was,
which makes plainer why none of the three is the cold-cache cost. §4 also now names
`null_block_containment` beside `single_user_determinism`, since both move the same way at
`max_num_seqs=32` and both do so on the before tree too.

The reviewer closed the one gap this stage could not: an AST comparison against the recovered stash
`06dd1ea31bd` — docstrings stripped, `ast.dump` compared — shows `tt/generator.py` and `tt/model.py`
byte-identical and `tt/generator_vllm.py` and `tests/test_generator_vllm.py` identical in structure. The
shipped runtime code is provably the code the 26/26 device suite exercised.

### Commits

| repo | branch | commit | contents |
|---|---|---|---|
| `tt-metal` (`/home/ttuser/dev/ornith/tt-metal`) | `agentic-research/hous/ornith-1.0-35B` | **`bc170c90240`** | the three adapter/generator/model changes, four new adapter test cases, the refreshed `readiness_vllm/` artifacts and all of `doc/optimized_vllm/` |
| `vllm` (`/home/ttuser/dev/ornith/vllm`) | `dev` | *none* | this stage changed nothing in the plugin or the fork; it still stands at `5380fd4`, the vLLM-integration stage's last commit |

Nothing was pushed. The commit excludes the two dirty paths that are not this stage's:
`.agents/skills/tt-device-usage/SKILL.md` (already modified in the working tree when this stage began)
and the untracked `.agents/fast-models-fast-feedback.md`.

One thing the commit did change that is worth recording, because it moved shipped code after the last
device run: the repo's `pre-commit` hooks reformatted `tt/model.py`, `tt/generator.py` and
`tests/test_generator_vllm.py` (black split the `None if page_table_only else ttnn.from_torch(...)`
expressions across lines; `end-of-file-fixer` added trailing newlines to the JSON and log artifacts).
The reformatting was checked to be semantics-preserving the same way review round 6 checked the revert —
`ast.dump` of each file before and after the hooks is identical — and the host-only half of the adapter
suite was re-run against the committed bytes (10 passed,
[`logs/pytest_host_only_postcommit.txt`](logs/pytest_host_only_postcommit.txt)). The 16 device cases
still date from 06:14; the host cannot run them (§7).
