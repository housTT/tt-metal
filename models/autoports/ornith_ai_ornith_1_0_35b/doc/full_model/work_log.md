# Full model — work log

Chronological. [`README.md`](README.md) is the result; this is how it was reached, including the
things that did not work and the three defects that had to be root-caused before anything could be
believed - the third of which (§7.1) the suite did not catch and a stage review did.

Environment for everything below: `tt-metal` on branch `agentic-research/hous/ornith-1.0-35B`,
4 × Blackhole `p300c` on one host (`tt-smi -ls --local` shows all four), UMD 0.9.9, KMD 2.10.0,
`transformers` 5.12.1, HF snapshot `ornith-ai/Ornith-1.0-35B` at revision
`5df2ed3f675c7beaa490328cc70bb573b65fb660`. Mesh opened as `1x4` with
`FabricConfig.FABRIC_1D_RING`, `fabric_router_config(8192)`, `l1_small_size=24576` and
`trace_region_size=200000000` (`tt/model.py::open_ornith_mesh`).

---

## 1. Reading before writing

Read, in order: `$full-model`, `$tt-device-usage`, `$tt-enable-tracing`, `$autofix`,
`$stage-review`, `$qualitative-check`; then `doc/optimized_multichip_decoder/README.md` and
`work_log.md`, `doc/context_contract.json`, the `MultichipDecoder` / `OptimizedDecoder` public
contract, `models/common/readiness_check/contract.py` and its four runners, and **both** common
sampling implementations (README §4.1 is the comparison that came out of it).

Decisions taken before any code:

* **stack per chunk, not per layer.** `MultichipDecoder.prefill_forward` requires
  `start_pos % chunk_size == 0`, so the model can either run the whole prompt through layer 0 and
  then layer 1, or run chunk 0 through all 40 layers and then chunk 1. The second bounds the
  inter-layer activation at one chunk (`[1, 2048, 2048]`, 8 MB) instead of the whole prompt (1 GiB
  at the advertised context) and is arithmetically identical, because each layer still sees its
  chunks in order.
* **prefill at batch 1, one user at a time.** A `linear_attention` layer's DeltaNet state is per-row
  and recurrent, so a batch of mixed-length prompts cannot be right-padded into one call (the pad
  tokens would advance the short users' state) and cannot be left-padded either (`full_attention`
  RoPE positions are absolute). Per-user prefill is what the paged/recurrent split allows. It is
  also *faster*: `ttnn.conv1d`'s prepared weights are per batch and its coverage shrinks as the
  batch grows — at batch 1 every prefill block length gets a conv program, at batch 32 none do and
  the whole prefill runs the FIR fallback.
* **replicated embedding.** The decoder stage's inter-layer contract is a replicated residual with
  no collective at the boundary; a vocabulary- or hidden-sharded embedding would need a gather to
  produce it. 1 GiB per device against 24 GiB of headroom (README §3).
* **column-parallel LM head.** No partial sums, no collective, and the per-device vocabulary shard
  is exactly what the split sampler consumes.
* **decode token buffer is `[1, 1, 1, 32]`.** `ttnn.sampling` emits one token per sampler row and
  wants a rank-4 preallocated output, so the same buffer is the sampler's output *and* the decode
  graph's token input — which is what makes token feedback device-side. The graph slices the first
  `max_batch_size` entries and reshapes to `[batch, 1]`, which `ttnn.embedding` turns directly into
  the `[batch, 1, dim]` the decoder's decode path wants.

API shapes were checked on device before committing to them
(`ttnn.embedding([b,1]) -> [b,1,dim]` TILE, `ttnn.rms_norm` on rank 3, `ttnn.plus_one` on both
position tensors), not assumed.

## 2. First light on the reduced probe

`logs/smoke.py` builds one real `linear_attention` layer (0) and one real `full_attention` layer (3)
with the real terminal path and a deliberately non-aligned 37-token prompt. Build 15 s; the whole
probe runs in about 40 s, which is what made the rest of this log affordable.

First failure, immediately: `TT_FATAL: Input Tensor is not allocated` inside
`ttnn.experimental.all_gather_async`, from the sampler's second call.
`/tmp/repro_sampling_trace.py` — a 40-line script with a fake one-matmul "model" — reproduced it in
seconds. Cause: the CCL shim was handing `TTSampling` a **persistent output buffer**, and
`TTSampling.forward` ends with `ttnn.deallocate(topk_values_gathered_bf16_interleaved)`, which *is*
that buffer. Fixed by dropping `buffer_key` from the hook's signature — both samplers probe it with
`inspect.signature` and only pass what it accepts. Recorded as a rejection in README §4.4 rather
than as a bug, because the buffer was a deliberate OPT-009-motivated attempt.

After that the reduced probe ran clean: prefill at 37 tokens, traced token-out decode, teacher
forcing, EOS handling, counters at `token_refreshes=0`.

## 3. The sampler cost 86 % of a token-out step

`logs/probe_terminal.py` split a warmed token-out step. This is the *original* measurement, the one
that set the direction; the committed `logs/probe_terminal_single.txt` is the same probe re-run at
the end against the delivered source and agrees to within run-to-run noise (1.474 / 11.820 / 0.391 /
0.076 / 13.719):

```
model trace replay                        1.468 ms
sampling trace replay                    11.820 ms     <-- 86 %
final norm + LM head (eager)              0.391 ms
token readback only                       0.062 ms
token-out step                           13.698 ms
```

`$full-model` says in terms: *"If sampler ops dominate token-out decode, fix the LM-head/sampling
contract before completing."* So that is what happened next.

`/tmp/probe_topk*.py` (committed as `logs/probe_topk.txt`) measured `ttnn.topk(k=32)` directly. The
result is the whole story: **cost is linear in the reduced width and independent of everything
else** — rows, leading dims and core count all do nothing. README §4.3 has the ladder. At 248320
tokens over four devices the shard is 62080 wide and one reduction costs 9.86 ms.

Rejected before the fix, each measured:

* `pad_logits_to_power_of_2` (62080 → 65536): **worse**, 10.87 ms. The pad is free (0.03 ms); the
  wider reduction is not;
* a shallow 2- or 4-way split: 5.96 / 3.08 ms — better than 9.86 but far off the balance point,
  because stage 1's width still dominates;
* `stable=False`: 9.86 vs 11.62 ms, but `stable=True` is what upstream asks for and the tie-break
  workaround depends on the ordering, so this was not taken;
* sub-core grids: 32, 64 and 110 cores are all 9.86 ms, and below 32 the op refuses the split
  (`topk_utils.cpp:93: split_size != 0`).

The fix follows from the measurement rather than from a guess: present the shard as `groups` rows.
`args.topk_num_groups` was added to `models/common/sampling/tt_sampling.py`, **default 1** so no
existing caller changes, and the local top-k becomes a two-stage exact reduction. Two things had to
be got right, and the first version got both wrong:

* `ttnn.topk`'s `indices_tensor` is a **preallocated output workspace, not a value source** — it
  returns *positions*, not the buffer's contents. A one-line sanity check (values `[3,1,2,0]`
  repeated, `indices_tensor = arange+1000`) returned `[0,4,8,12]`, which settled it. So the
  group offset has to be added explicitly, and stage 2's positions have to be mapped back through
  `ttnn.gather`;
* the grouping must be `[1, B, G, per]`, not `[1, G, B, per]`. Only the first is the flat-order
  reshape of `[1, 1, B, W]`; the second silently interleaves rows and groups. Caught by comparing
  indices against `torch.topk` on **well-separated** values — on random bfloat16 the comparison is
  dominated by tie-breaking and hides the bug.

Group count swept over the divisors of 1940 that keep the group width tile-aligned; 20 wins at
0.96 ms (README §4.3's table, and `logs/probe_topk.txt` checks every row for exactness against
`torch.topk` in the same run). End to end the sampling trace replay went **11.820 → 1.181 ms** and
the reduced token-out step **13.719 → 2.845 ms** (`logs/probe_terminal_single.txt` vs
`logs/probe_terminal_grouped.txt`, the two lines named in README §4.3's table), with byte-identical
greedy tokens. On the 40-layer model the sampling stage costs 1.17 ms (`perf_summary.json`); that is
a different measurement from the probe's 1.181 and the two are quoted separately.

## 4. The 40-layer stack: a hang, then two silent corruptions

### 4.1 Hang at the sampling trace capture

The first 40-layer run hung in `ttnn.synchronize_device` inside `SamplingGenerator.capture_trace`.
Per `$tt-device-usage`, triage before killing:
`tools/tt-triage.py --llm-output` → [`triage/`](triage/) — every check `pass`, no hung operation, no
watcher assert, ARC healthy on all four devices. A `gdb -p ... thread apply all bt` put the main
thread in `FDMeshCommandQueue::wait_for_outstanding_reads`, i.e. device work submitted and never
completing. Killed, `tt-smi -r`, `tt-smi -ls --local` showed all four boards, mesh smoke OK.

Two changes, both defensible on their own:

* **warm the sampler before capturing the model trace**, then capture with
  `skip_precompile=True`. `SamplingGenerator.capture_trace` otherwise precompiles by *executing* a
  full sampling graph over the model trace's live output buffer while a captured trace already
  exists. Warming first is also what `$tt-enable-tracing`'s program-cache rule asks for;
* **cycle the CCL semaphores.** The shim returned one fixed handle set, and one sampling call issues
  two gathers back to back. The shared `TT_CCL` double-buffers precisely for that, so the shim now
  subclasses it.

The 40-layer model then ran end to end: build 181 s, TTFT 140 ms, 23.9 ms/token,
`token_refreshes=0`.

### 4.2 Prefill died on the 40-layer stack only

`run_prefill_check` on the full stack: *"Statically allocated circular buffers in program 502 clash
with L1 buffers ... L1 buffer allocated at 1196032 and static circular buffer region ends at
1430912"*, inside `chunked_scaled_dot_product_attention`. The two-layer probe never saw it, which is
the clue: something persistent in L1 scales with the layer count.

First hypothesis — the embedding output. Naming `ttnn.DRAM_MEMORY_CONFIG` explicitly did **not**
fix the clash: same addresses, same error. Refuted as *the* cause, and kept for a different reason
than the one first written down: reading `embedding_device_operation.cpp` afterwards,
`ttnn.embedding` does not default to L1 at all — `output_mem_config.value_or(
input_tensor_arg.memory_config())` means it inherits the **indices** tensor's memory config, so the
residual's placement would follow whatever a caller built its token tensor in. Pinning it is right
because the decoder stage's inter-layer contract says DRAM interleaved, not because the residual was
in L1. (README §5.2 says the same; the earlier wording in both files claimed an L1 default and was
corrected in review 7.)

Second hypothesis, arithmetic first this time. `MultichipMoE.prepare_decode_gate` allocates five
persistent **L1** tensors per layer; four are HEIGHT_SHARDED with a 32×32 shard per core, i.e. 8 KiB
of L1 per core per layer. 40 × 8 KiB = 327,680 B, and 1,572,864 − 327,680 = 1,245,184, which is
within 48 KiB (l1-small plus the interleaved scatter bases) of the 1,196,032 in the error. Their
contents depend only on the expert count, the top-k and the mesh — never on the layer — so
`OrnithModel._share_fused_gate_buffers` keeps one set for the whole stack. Safe because layers run
sequentially in the traced graph as well as eagerly. Prefill then passed, and the decoder stage's
measured 256-token prefill SDPA chunk was preserved rather than given up.

Result: `run_prefill_check` 0.950 / 1.000 / 1.000, `run_teacher_forcing` 0.940 / 1.000 / 1.000.

### 4.3 The one that mattered: request 2 onwards was garbage

The shared qualitative suite was the first thing to drive six prompts through one generator. Prompt
0 was perfect; prompts 1–5 emitted token 0 (`!`) then gibberish, Chinese date spam, single-token
collapse. Deterministic — a rerun with the HF control skipped produced byte-identical garbage, so
not a race.

This is where `$autofix`'s "verify or refute each hypothesis" discipline paid, because the obvious
hypotheses were all wrong:

| hypothesis | experiment | verdict |
|---|---|---|
| `reset()` does not clear the DeltaNet/conv/KV state | `logs/probe_multi_prompt.py` printed each state's abs-max before/after reset and after prefill, four prompts | **refuted** — all zero after reset, and all four prompts, including a repeat of the first, were perfect |
| it needs a long first generation | same probe at `--gen-len 128` | **refuted** — still perfect |
| it is the HF model being loaded and freed in the same process first | qualitative rerun with `--skip-hf` | **refuted** — identical garbage |
| `ttnn.scatter` mutates the shared fused-gate base | instrumented `_gate_zeros` abs-max across requests | **refuted** — stays exactly zero |
| it is a race that a device synchronize hides | added a `sync` arm to the probe | **refuted** — the `sync` arm is as broken as the `plain` one |

What the probe arms *did* establish is that the corruption is **permanent** — once it starts, a bare
prefill of the original prompt is wrong too — and that repeating a single prompt never triggers it.
Permanent plus prompt-length-dependent points at something compiled, not at state.

`logs/probe_bisect.py` then settled it in one binary by printing
`mesh.num_program_cache_entries()` at every step and re-checking a bare prefill after each. README
§5.1 has the two columns; the summary is that compiling the two prompt lengths **before** capture
leaves both prefills stable across replays and compiling the same 382 programs **after** capture
destroys both after a single replay.

The mechanism is documented in tt-metal itself:
`tt_metal/impl/allocator/allocator.cpp` warns *"Allocating device buffers is unsafe due to the
existence of an active trace. These buffers may be corrupted once a trace is executed"*, and
`mesh_device.cpp` shows `end_mesh_trace` marking allocations unsafe with no protection of the
addresses the trace writes. A program's kernel binaries are such a buffer, and unlike an activation
they live in the program cache forever.

Fix: `OrnithGenerator._ensure_traces_replay_safe`, called after prefill and before the first replay
in both `generate` and `decode_forward`. One integer comparison; a re-capture only when a request
actually introduced a program, and capture records without executing so the prefill state survives
it. The same guard covers a caller-owned KV cache attached after capture.

After the fix, all six qualitative prompts are coherent and a second pass through the same six
reproduces the first exactly, with zero re-captures on the second pass.

Two regression tests pin it: `test_requests_of_different_prompt_lengths_do_not_corrupt_each_other`
and `test_traces_are_recaptured_when_a_new_program_is_compiled`.

**This is the finding to carry into vLLM.** A server sees a new prompt length on nearly every
request, and would have hit this on request two.

### 4.4 Two smaller bugs the suite caught

* **one-token prefill freed the tensor it was about to use.** For `logical == 1` the
  last-position slice covers the whole tensor and `ttnn.slice` returns an alias, so
  `ttnn.deallocate(hidden)` left the LM head reading a deallocated tensor
  (`TT_THROW ... Tensor is not allocated`). Hit by prompt length 1 *and* 2049 (whose second chunk is
  one token). Fixed with the decoder's own `_slice_owned` ownership discipline.
* **the batch-1 prefill pack was freed by the batch-B allocation.**
  `MultichipDecoder.allocate_state` frees the conv1d weights and paged-fill row indices it finds on
  the layer before building new ones, so capturing the batch-1 pack and then allocating batch 4
  handed `ttnn.conv1d` a deallocated weight. Fixed by detaching each pack from the layer right after
  capturing it.

## 5. Evidence

`logs/run_evidence.sh` regenerates everything in order; `tracy/run_profiling.sh` and
`logs/run_watcher.sh` are deliberately separate runs (profiler and watcher must not share one).

The AIME24 chat-template reference was generated **fresh**; there was no prior reference under this
autoport directory to prove a match against, and `readiness_aime24_chat.meta.json` records the
provenance a later stage would need to prove one. Two shared-code fixes were needed to generate it
at all, both of which would silently produce a meaningless reference otherwise:

* `models/common/readiness_check/generate.py` called
  `tokenizer.apply_chat_template(..., tokenize=True)` and iterated the result. In `transformers` 5
  that returns a `BatchEncoding`, which iterates over its **keys**, so the code did
  `int("input_ids")`. Now normalised for both shapes;
* `AutoModelForCausalLM` resolves `qwen3_5_moe` to `Qwen3_5MoeForCausalLM`, which expects
  `model.layers.*` while this checkpoint stores `model.language_model.layers.*` — every weight would
  be reported missing and the "reference" would be a randomly initialised model, with no error. A
  shared helper (`models/common/readiness_check/hf_model.py`) now prefers the architecture the
  checkpoint declares in `config.architectures`, which for an ordinary text checkpoint *is* what the
  auto class resolves to. Both `generate.py` and `run_autoregressive.py` use it.

## 6. Profiling

`tt-perf-report` for the reduced variant took five attempts, and the failures are worth recording
because they bound what the artifact can be:

1. pytest node + `--op-support-count 50000` → `AssertionError: Device data missing: Op 401411 not
   present in cpp_device_perf_report.csv for device 3`;
2. `--op-support-count 2000000` → the post-processing consumed 14.7 GB and ran for over an hour
   without finishing;
3. `--op-support-count 500000` → segfault in `close_mesh_device` during pytest fixture teardown,
   with the model's device tensors still referenced by the test frame;
4. releasing the traces before teardown → same segfault. Switched to a standalone script
   (`logs/profile_reduced.py`) that releases the generator and collects before closing the mesh,
   which the profiler survives;
5. `prefill_chunk=256` for the profiling build (setup-only: sixteen `ttnn.conv1d` probe runs per
   layer become two, and no program in either signposted window changes), no prefill in the decode
   phase, `ttnn.ReadDeviceProfiler` between stages, and a **4-replay** window → both reports.

Four replays is enough for op shares, which is what the report is for; the latency numbers come from
un-profiled wall clock. Recorded as limitation 2.

## 7. Suite

`tests/test_full_model.py`: 42 fast cases (reduced model) + 5 `long` ones (two all-layer, two
profiling, one batch-32). The fast suite's full pass is
[`logs/pytest_full_model.txt.gz`](logs/pytest_full_model.txt.gz) (`-m "not long"`) and the `long`
suite's is [`logs/pytest_long.txt`](logs/pytest_long.txt); the two together are every case in the
file.

Coverage: readiness-contract conformance, the advertised context/batch/vocab contract, the carried
decoder policy, the sampler CCL shim, the replicated DRAM-interleaved inter-layer residual,
ten logical prompt lengths from 1 to 3000, all-logits vs last-position agreement, the split-sampling
token-feedback/position-coherence/page-table contract, host-fallback freedom, greedy determinism,
the multi-request corruption regression, the trace-recapture counter, prefill logit reproducibility,
the host-sampling compatibility mode, teacher-forcing callback counts, batch-4 mixed prompts with an
inactive row, batch-1/batch-4 slot agreement, reset semantics, caller-owned cache, the grouped
top-k's exactness, and page-table validation on both the row and the blocks-per-user axis.

### 7.1 What the three stage reviews changed

Each review was treated as work, not as commentary. Review 1: the `long` suite had never been run
and the shared-sampler A/B had no committed log; both were run. Review 2: `continue_from_state` was
dead code (declared on the model, never threaded through the generator) and silently wrong above
batch 1 — threaded through, made to raise, and covered by
`test_chunked_prefill_continuation_matches_a_single_call`; plus a non-idempotent contract-note
append, a wrong speedup ratio, and a TTFT attribution. Review 3, three code changes and a probe:

* **the trace-safety baseline was re-taken after post-capture work.** `_ensure_decode_trace` used to
  capture, then call `model.reset_state()`, then set `_program_cache_at_capture` again. The state
  wipe is itself a device graph (a pack apply per batch, 40 layers' state reset, one `ttnn.multiply`
  per paged K/V tensor), so any program it compiled landed in exactly the post-capture window §4.3 is
  about — and the re-baseline then hid it from `_ensure_traces_replay_safe` permanently. The wipe now
  runs **before** capture (capture records without executing, so the state is just as clean at the
  first replay) and the baseline is the one `_capture_traces` sets. Nothing was observed failing;
  this closes a hole in the mitigation, not a live defect;
* **a page table too narrow for the position it is asked to serve is now rejected**, on both public
  entry points, instead of letting the paged SDPA kernel read past the end of a row.
  `test_a_page_table_too_narrow_for_the_position_is_rejected` pins it;
* **the split-sampling test now asserts all four devices sampled the same token**, not just that
  device 0 produced something. The sampler's all-gather (§4.4) is what makes them agree, and the
  generator's readback only ever looks at device 0, so a divergent shard would have been invisible;
* **`logs/probe_long_prompt.py`**: the longest non-aligned prompt through the full stack was 5003
  against an advertised 262144. The probe walks the ladder to **262143** — one token short of the
  advertised context — and it runs, which turns README §3's capacity arithmetic into a measurement.

Review 6 found the last one, in the same family and on the last public path that still had it:
**`generate()` at batch > 1 decoded from a zeroed recurrent state.** `generate` prefills through
`prefill_forward_single`, which writes the batch-1 state pack; the captured decode trace is bound to
the batch-B pack; and the merge that copies one into the other (`_merge_prefill_state_into_slot`)
was called only from the *batched* `prefill_forward`. At batch 1 the two packs are the same object,
which is why nothing showed. `OrnithModel.prefill_request_into_slot` now does the reset → prefill →
merge → restore sequence for a single request and both `generate` and `_generate_eager` go through
it; `logs/probe_batch_slots.py` measures the before and after (first decoded token 169222 → 240560,
matching batch 1) and separates it from the batch-geometry divergence that starts one token later.
The rest of review 6: three misquoted README figures (38.21, 3.64, 2.42), the shared-sampler A/B
re-run against the delivered `tt_sampling.py`, trace capture no longer marking the model's state
live (it records, it does not write), `prefill_forward` and `decode_forward` resolving a
caller-supplied page table through one shared function so they cannot disagree, `generate`'s `reset`
keyword documented, and a device-free test for the `topk_num_groups`/`pad_logits_to_power_of_2`
refusal.

Review 5 closed the same defect's last hole: the liveness flag review 4 introduced lived on the
*generator* and `generate()` writes prompt state through `model.prefill_forward_single`, so it never
set the flag — meaning `teardown()` followed by a low-level `decode_forward` would still have wiped a
live prompt silently, and `generate(enable_trace=False)` followed by `generate(enable_trace=True)`
raised because the reset that would have cleared the state ran *after* the capture check. The flag
now lives on `OrnithModel` where every write path is, `generate` resets before it captures,
`test_the_eager_debug_path_does_not_poison_the_traced_one` covers the previously untested eager path,
and `sample_on_device=True` in host sampling mode is refused instead of failing inside the sampler.
The per-slot merge masks — the one long-lived buffer still allocated after trace capture, covered
only incidentally by `_merge_rows` compiling programs — are now preallocated in `allocate_state`.
The rest of review 5: the watcher count, the `transformers` version in this file's environment
block, the freshness exception list (four historical artifacts named), two derived figures presented
as quoted, a limitation for batched prefill's slot ordering, and `topk_num_groups` +
`pad_logits_to_power_of_2` now refusing to combine instead of silently honouring one.

Review 4 found the one that mattered most, and it was hiding behind the review-3 fix rather than
being caused by it — **lazy trace capture threw the prompt away** (README §5.3). Capture
warm-compiles a real decode step and then wipes the DeltaNet state and the whole paged KV cache;
because capture was lazy and first ran inside `decode_forward`, the low-level pair
`prefill_forward` → `decode_forward` on a fresh generator decoded from an *empty* cache and returned
a fluent token that had never seen the prompt. `generate()` was safe (it captures first) and so was
every probe (they all call `_ensure_decode_trace()` up front), which is exactly why nothing caught
it: the batch-4, batch-32 and prompt-ladder assertions checked `isfinite` and position advance, both
of which an empty cache satisfies. `prefill_forward` now captures before it writes,
`_ensure_decode_trace` **raises** if it is entered with live prompt state instead of wiping it, and
`test_low_level_prefill_then_decode_sees_the_prompt` compares the low-level pair's second token
against the high-level path's. The test was checked for teeth the only way that means anything:
reverting the fix (and disabling the guard) makes it fail with `assert 102909 == 10980` — the
high-level path's token against the one a wiped cache produces
([`logs/probe_lazy_capture_ab.txt`](logs/probe_lazy_capture_ab.txt), the console output of that
one-off run; it is not in `run_evidence.sh` because reproducing it means shipping the defect). The rest of review 4: four misquoted probe figures re-quoted line by
line, `logs/pytest_batch32.txt` (a superseded pre-fix log) deleted, the evidence-freshness exception
list corrected, the bench's `cache_context=8192` named in README §1, the grouped top-k's per-group
tie-break bound written down in §4.3, the page-table check moved onto the *physical* prefill extent
and its message corrected to name the 32-block stick rule, a `headroom_ratio` definition,
`stage 4`/`stage 5` label drift, and the AIME24 metadata's dtype string.

## 8. Repo changes outside the autoport directory

Four files, all additive or default-off. They are the whole of this stage's footprint outside the
autoport directory: the worktree also carries `.agents/skills/tt-device-usage/SKILL.md` (modified)
and `.agents/fast-models-fast-feedback.md` (untracked), which are agent-process notes that predate
this stage, are not stage-owned, and are excluded from the checkpoint commits in §10.

| file | change | why it is safe |
|---|---|---|
| `models/common/sampling/tt_sampling.py` | opt-in `args.topk_num_groups` grouped local top-k; the existing single-reduction path extracted into `_local_topk` unchanged | default 1 keeps every existing caller byte-for-byte identical. `models/common/tests/test_sampling.py` was run **with** the change ([`logs/pytest_common_sampling_with_change.txt`](logs/pytest_common_sampling_with_change.txt)) and **without** it, the change stashed ([`logs/pytest_common_sampling_baseline.txt`](logs/pytest_common_sampling_baseline.txt)): identical results, one pre-existing failure in both (`test_log_probs_calculation`, which needs an 8- or 32-device mesh) |
| `models/common/readiness_check/hf_model.py` | new: resolve the HF reference class from `config.architectures` | for an ordinary text checkpoint the declared architecture *is* what `AutoModelForCausalLM` resolves to, and the `hasattr(cls, "generate")` filter keeps a non-generative declared class from being picked. `test_the_hf_reference_class_resolver_picks_the_checkpoints_own_architecture` pins both the wrapped-multimodal case and the empty-`architectures` fallback. No other checkpoint is present on this host, so there is no cross-model regression evidence — recorded as residual risk |
| `models/common/readiness_check/generate.py` | use that helper; normalise `apply_chat_template`'s `BatchEncoding` | both fix silent wrong-reference bugs |
| `models/common/readiness_check/run_autoregressive.py` | use that helper | same |

## 9. Final evidence, all from the delivered code

The whole evidence set was regenerated **after the last source change**, so every artifact in this
directory comes from the delivered `tt/model.py`, `tt/generator.py` and `tests/test_full_model.py` —
the behavioural artifacts are all from it. What is older, and why: `tracy/` (a profiler capture of the reduced
variant, whose op shares no post-review change moves — README limitation 2 bounds what it claims),
and `logs/probe_topk.txt` (synthetic `ttnn.topk` shapes on an open mesh, no model code at all).
The shared-sampler A/B below used to be a third exception; review 6 pointed out that
`tt_sampling.py` had been edited after it ran, so both arms were re-run against the delivered file
and are current. Four more files are historical by nature and are not evidence about the delivered code:
`triage/` (the tt-triage capture taken *during* the §4.1 hang), `defect_qualitative_before_fix.json`
(the recorded §5.1 symptom, from the pre-fix code — that is the point of it),
`logs/followup_status.txt` (the status file of an earlier pass, superseded by
`logs/final_status.txt`), `readiness_aime24_chat.refpt` (HF-side; no TT code participates in it) and
`logs/probe_lazy_capture_ab.txt` (the console output of the one-off *pre-fix* run behind README §5.3;
by construction it cannot come from the delivered source, and the file says so in its own header).
Driver: `logs/run_evidence.sh` inside `logs/final_status.txt`, every step `ok`
([`logs/run_evidence_status.txt`](logs/run_evidence_status.txt)).

| step | result |
|---|---|
| `run_prefill_check` | top-1 0.950, top-5 **1.000**, top-100 **1.000** |
| `run_teacher_forcing` (traced decode) | top-1 0.940, top-5 **1.000**, top-100 **1.000**; decode 37.01 t/s/u (runner window) / 41.63 (generator loop). The runner reports TTFT 1126.3 ms; the generator logs **439.5 ms** for the same first token, so ~687 ms is runner-side per-entry setup outside `generate`, and 439.5 ms is the cold-length TTFT that limitation 3 says the warmed 138 ms does not cover. Both runner figures are host-sensitive: across the sweeps they ranged 37.0-38.4 t/s/u and 817-1126 ms while the generator's own loop stayed within 0.6 % of the free-running benchmark |
| `run_autoregressive` (raw continuation prompt) | 128 tokens, coherent, `adjacent_duplication` 0.0000 |
| `run_autoregressive` (chat-template prompt) | 128 tokens, coherent, 42/128 tokens identical to the HF control (first divergence at token 15) |
| qualitative suite (6 prompts, HF control + TT) | every TT completion coherent and structurally matched to its control |
| `bench_full_model` | TTFT 135.7/138.1/144.4 ms (min/median/max — the host-sensitive figure; see README §1), token-out 23.89 ms/token = 41.86 t/s/u, 1 trace re-capture, `token_refreshes=0` |
| `probe_multi_prompt --arms plain,plain` | six prompts twice, all coherent, second pass reproduces the first, 0 re-captures on the second pass |
| `probe_bisect --order after / before` | the §5.1 repro, both arms as documented |
| `probe_terminal --topk-groups 1 / 20` | 13.719 → 2.845 ms token-out on the reduced probe, byte-identical greedy tokens |
| `check_degenerate_output.py --missing-artifacts critical` | *No degenerate output detected* |
| `pytest -m "not long"` | **42 passed** in 1475 s ([`logs/pytest_full_model.txt.gz`](logs/pytest_full_model.txt.gz)) |
| `pytest -m long` | **5 passed** in 436 s ([`logs/pytest_long.txt`](logs/pytest_long.txt)) — the two all-layer cases (a coherent 64-token chat completion; a 5003-token non-aligned prompt through the complete stack), the two reduced profiling cases, and batch 32 |
| `probe_batch_slots.py` | slot 0 at batch 4: the prompt token and the first decoded token equal batch 1's; **removing the prefill→slot merge changes the first decoded token** (240560 → 169222), which is the defect review 6 found; slot 0 is byte-identical with identical and with different neighbours, so the later greedy divergence from batch 1 is batch geometry, not leakage ([`batch_slots.json`](batch_slots.json)) |
| `probe_long_prompt.py` | non-aligned 5003 → **262143** tokens through the public path on the full stack at the full advertised cache: every length prefills with finite logits and a valid sampled token, 262143 in 167.3 s with 24.15 GiB DRAM still free ([`long_prompt.json`](long_prompt.json)) |

The two pytest runs collect 47 items each (42 selected + 5 deselected, and the reverse), so the two
logs together are every case in the file.

Earlier passes, kept for the record: the first evidence run (33 fast cases) and a second pass after
review 1 (`logs/followup_status.txt`) that added the `long` suite and the shared-sampler A/B, and a
third (`logs/final_status.txt`, since overwritten by the sweep above) that re-ran the four headline
artifacts after review 2. The shared-sampler A/B is the one thing not repeated in the final sweep
because it is not affected by this stage's model code:

| step | result |
|---|---|
| `models/common/tests/test_sampling.py` **with** the delivered shared-sampler change | 15 collected: 7 passed, 7 skipped, 1 failed (`test_log_probs_calculation`, `AttributeError: 'NoneType' object has no attribute 'dtype'`) |
| the same suite with the change **reverted** (`git checkout -- tt_sampling.py`, then restored) | 15 collected: 7 passed, 7 skipped, 1 failed — *the identical failure on the identical test*, which is what makes it pre-existing rather than caused here. `LogProbsCalculator` needs an 8- or 32-device mesh; this is a 1x4. (An earlier pair of runs collected 32 items on this same file and reported 24 passed; the collection count depends on what the fixtures discover at import time. Both arms of *this* pair were run back to back in the same environment, which is the comparison that matters) |

(Both arms exit non-zero because of the pre-existing failure. The point of the pair is that the two
runs are the same, not that either is green.)

Separate runs, as `$tt-device-usage` requires:

* `tracy/run_profiling.sh` → the two `tt-perf-report` captures (§6 above for the five attempts it
  took, and README limitation 2 for what the artifact is bounded to);
* `logs/run_watcher.sh` → **42 passed** under `TT_METAL_WATCHER=10`, **0** watcher
  error/assert/hang lines in either the pytest log or `generated/watcher/watcher.log`
  ([`watcher/watcher_error_count.txt`](watcher/watcher_error_count.txt)). tt-metal truncates
  `generated/watcher/watcher.log` on every mesh open and the suite opens the mesh once per test, so
  the committed copy of that file is the last session only; the committed **pytest** log is what
  covers all 42 cases, and watcher reports to stderr as well as to its own file;
* `logs/probe_footprint.py` at `(batch 1, context 262144)` and `(batch 32, context 8192)` → the
  measured capacity rows in `doc/context_contract.json`;
* `pytest ::test_batch_32_prefill_and_decode` → the advertised batch bound, end to end.

### 9.1 Anomalies seen and classified

| observed | classification |
|---|---|
| `TT_THROW: Statically allocated circular buffers ... grow to 1684416/2470848 B which is beyond max L1 size of 1572864 B`, **50** lines in the fast-suite log at **batch 4** (5 batch-4 builds x the 10 block lengths `ttnn.conv1d` refuses; the same log carries 45 batch-1 builds with none) | **expected and caught, same mechanism as the row below.** They come from `_prepare_conv1d_weights_local` probing one `ttnn.conv1d` program per prefill block length: at batch 1 all 16 lengths are accepted, at batch 4 six are, and the refusals are caught and degraded to the FIR path. Worth naming separately because §5.2 is a *real* circular-buffer-vs-L1 defect and this message looks like it — the difference is that this one is a probe's refusal at setup, on a batch prefill never runs at (`prefill_forward` uses the batch-1 pack) |
| `TT_FATAL: Out of Memory ... L1 buffer` lines during a batch-32 build | **expected and caught.** `_prepare_conv1d_weights_local` probes one `ttnn.conv1d` program per prefill block length and degrades to the FIR path when the op refuses; the decoder stage documented that coverage shrinks to zero at batch 32. Not raised, not a failure, and prefill runs at batch 1 anyway |
| `Allocating device buffers is unsafe due to the existence of an active trace` warning | **root-caused**, §4.3. It is the warning for the defect that guard now prevents; it still fires because allocation after capture is legal and normal, and every such buffer in the delivered path is short-lived |
| `ttnn::tilize: Using input shard spec for output tensor because the legacy sharded optimized program factory is being used` | inherited from the decoder stage's captures, unchanged in kind, and outside this stage's path |
| `Fabric packet size 8192 B is suboptimal for transporting 1088 B pages` | inherited; the decoder stage measured 8192 as the shipped value at its own shapes (its README §4.1b) |
| `models/common/tests/test_sampling.py::test_log_probs_calculation` fails | **pre-existing**, verified by stashing this stage's `tt_sampling.py` change and re-running: `LogProbsCalculator` supports 8- and 32-device meshes and this is a 1x4 |
| `run_teacher_forcing` reports 37.01 t/s/u where the generator's own loop reports 41.63 | **explained, not dismissed.** The runner times between its own per-token `next_input` callbacks, so its window includes that callback and the host token write on the ~6 % of steps where the forced token differs from the sampled one. Both figures are reported with their boundaries named (README §1, §7) |
| qualitative prompt 3's completion opens `Here's a thinking thinking sequence` | **controlled by the HF control.** The HF reference generated for the same prompt in the same run opens with the *identical* phrase (`readiness_qualitative.json`, `hf[2].completion`), so the doubled word is the checkpoint's own output, not a port artifact. No other completion repeats |
| free-running raw-continuation output diverges from HF at token 11 | **expected and controlled.** A creative continuation with near-tied logits; the chat-template control diverges much later (42/128 identical) and teacher forcing gives top-5 1.000, so the token-level agreement is a property of greedy decoding on this prompt, not of the port |

## 10. Stage review and commit SHAs

`$stage-review` ran **eight** times, each as a fresh independent subagent given the goal contract,
the skill paths and the artifact roots, and each read-only. Passes 1-7 returned `more-work-needed`;
§7.1 records what each one found and what was changed for it. Pass 8 returned **`clean-pass`** with
no required work, having re-derived every headline figure from its artifact, re-checked the two
review-7 documentation items, read all four autoregressive completions and all six qualitative
HF/TT pairs, and confirmed `tt/multichip_decoder.py` is byte-identical to `HEAD`. Its remaining
"other concerns" were applied before the commit below: the batch-4 conv1d refusal count (50, not
~30), the caller-owned cache being sticky (`attach_kv_cache` has no inverse), what `generate()` at
batch > 1 actually does with slots 1..B-1, and adding `logs/probe_lazy_capture_ab.txt` to §9's
historical-artifact list.

Three of the eight passes found real defects rather than documentation problems - the dead
`continue_from_state`, lazy trace capture wiping the prompt (§5.3), and `generate()` at batch > 1
decoding from an unmerged state - and none of the three would have failed a check that existed
before the review that found it.

Commits are local only; nothing was pushed. Repo `tt-metal`, branch
`agentic-research/hous/ornith-1.0-35B`:

| SHA | what |
|---|---|
| `895e84b7343` | [autoports] Ornith-1.0-35B full model: the model, the generator and the suite |
| `97ce88d3c78` | [autoports] Ornith-1.0-35B full model: evidence, docs and the recomputed context contract |
| `ac81fb6686c` | [autoports] Ornith-1.0-35B full model: regenerate the evidence from the formatted source |
| *(this section)* | the review record and these SHAs |

Stage-owned files only. The worktree also carries `.agents/skills/tt-device-usage/SKILL.md`
(modified) and `.agents/fast-models-fast-feedback.md` (untracked); both are agent-process notes that
predate this stage and are deliberately left out of every commit above.
