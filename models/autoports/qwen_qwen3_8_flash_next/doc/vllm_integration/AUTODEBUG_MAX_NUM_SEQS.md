# AutoDebug: true multi-active vLLM serving with a physical batch-one trace

Date: 2026-08-28

Scope: source-only diagnosis. No TT command, server, or hardware test was run, and no implementation source was changed.

> Historical design report. Its `MAX_NUM_SEQS = 1` statements describe the
> pre-remediation source. The recommended physical-B1 virtual-slot design was
> subsequently implemented and full-model serving validated two simultaneously
> active requests with generation-guarded release/cancellation. The final
> serving contract and evidence are in `README.md`, `work_log.md`, and
> `../../readiness_vllm/max_num_seqs_limit.json`; no larger active width is
> claimed as hardware-validated.

## Headline finding

`MAX_NUM_SEQS = 1` is a real software limitation, not a physical-capacity result. The smallest viable route to true active `max_num_seqs > 1` is a **virtual-slot scheduler microbatch over the existing physical-B1 segmented token-out trace**, first proving two active requests. It should keep one model/weight instance and the existing B1 trace, store request-local model-owned state in B virtual device slots, and sequentially restore one slot, replay the canonical trace and sampler, then commit that slot for every active request in one vLLM decode step.

This is not queued admission: both requests remain active in vLLM, own independent QSA pages, GDN/PLE state, token/position/sampler state, and each produces one output in the same engine step. It is physical-B1 execution of a logical-B batch.

The superficially smaller dual-mode proposal, “trace B1 and use eager B>1 in the same model,” is contradicted by current construction-time state. `Qwen38FullModel.max_batch` sizes the token, position, page-table and sampling buffers as well as every layer’s recurrent/conv/PLE state (`tt/model.py:309-344, 773-818`). The shared DRAM/L1 decode-state workspace is created only for `max_batch == 1` (`tt/model.py:344`; `tt/multichip_decoder.py:557-684, 973-990`), and both token-out decode and full-stack trace capture reject `max_batch != 1` (`tt/model.py:1618-1619, 1703-1704`). A model constructed at B=2 can run the existing eager path, but cannot retain the measured B1 trace; a model constructed at B=1 cannot hold request 2’s recurrence. A second full model would duplicate TT-resident weights and cache/runtime resources and has no capacity evidence.

The virtual-slot design is therefore recommended for the B=2 proof. It is still substantial work: current batch-one canonical recurrence must gain device-resident backing banks plus traced restore/commit boundaries. It is smaller and more falsifiable than generalizing all 48 segmented layer traces to batch B.

## Established facts

1. The adapter hard-codes and enforces one active sequence at three independent points: `MAX_NUM_SEQS = 1`, initializer validation/build `max_batch=1`, and `get_max_tokens_all_users()` (`tt/generator_vllm.py:29-30, 115-147`). It also accepts only prefill slot `[0]` (`:196-198`).
2. The adapter has one global `_decode_started` bit (`:84, 208, 233-248`), while the generator has one mutable `state` and one `_serving_device_feedback_current` bit (`tt/generator.py:157-166`). A second prefill calls `allocate_batch_state()`, releases any trace, replaces that single state, and resets its cohort (`:169-188, 287-320`). These are cohort-global, not request-local semantics.
3. Serving decode rejects a non-identity `slot_remap` (`tt/generator.py:513-518`). That discards the shared runner’s existing request-to-device-state movement contract.
4. The plugin already has most logical-slot lifecycle machinery. It owns `_req_state_slot` (`vllm_tt_plugin/model_runner.py:220-223`), retains slots for unscheduled but still-running requests, releases only finished or preempted requests (`:611-701, 817-839`), allocates safe prefill slots (`:841-882`), and constructs/commits a decode remap (`:884-931`). `TTModelInput` already carries `slot_remap` and `prefill_empty_slots` (`vllm_tt_plugin/model_input.py:116-132`).
5. The plugin pads decode to configured capacity and sends one row of tokens, positions, block tables and sampling parameters per logical request (`model_runner.py:1006-1027, 1126-1174`). Async completion snapshots request identities/states and applies sampled tokens only after completion; this is the correct anti-stale boundary to retain (`async_decode.py`, `SubmittedStepContext`, `CompletedDecodeStep`, and `TTAsyncDecodeController.apply_completed_decode_step`).
6. vLLM’s QSA KV ownership is already compatible with logical multi-active requests: the runner builds per-request page-table rows, and the model consumes a uniform QSA page table against the exact cache object returned by `allocate_vllm_attention_cache()`. Multi-active work must select the appropriate row for each physical replay; it must not allocate another KV cache.
7. Model-owned state is the missing virtualization boundary. There are 36 GDN recurrent/conv states, PLE convolution state on declared layers, PLE two-token history keyed by request ID, persistent token and current-position buffers, sampling parameters/seeds, and trace-lifetime tensors. The batch-one workspace already hydrates DRAM-canonical GDN/PLE state to fixed L1 addresses and commits it back (`tt/multichip_decoder.py:557-684`). That is the natural seam for virtual backing banks.
8. Host expert slots are model-wide immutable-weight cache entries and should remain shared across users. PLE history is already keyed by request ID and must remain request-isolated. Neither store should be copied as part of a slot swap.
9. The existing full-model eager B=32/context=4096 result proves a batch-shaped eager implementation exists; it does not prove vLLM request lifecycle, 262144-context cache capacity for several simultaneous users, or coexistence with the batch-one trace.

## Recommended implementation: physical-B1 virtual-slot microbatch

### Runtime contract

Advertise an initial logical capacity of two. Keep `Qwen38FullModel.max_batch == 1` for the measured trace. Add a separate `max_active_slots`/`virtual_slot_capacity` used only by serving state banks. A vLLM step with rows `[r0, r1]` carries stable slot IDs `[s0, s1]` and request IDs `[q0, q1]`. For each row, in row order:

1. Restore virtual slot `s`’s GDN/PLE convolution backing state, token, current position, sampling seed/parameters, and one QSA page-table row into the physical-B1 trace-bound buffers.
2. Read the compact physical token only for the declared PLE n-gram lookup when async device feedback makes the host token stale; call PLE with the real request ID.
3. Replay the existing ingress, 48 segmented layers, terminal projection, canonical `Sampling1D`, direct device token feedback and position-increment trace.
4. Commit the mutated physical GDN/PLE state, sampled token, position and RNG state to virtual slot `s`.
5. Retain the compact sampled token as row `r` of one logical batch output. No full logits, host argmax, Python token writeback, or separate sampler is introduced.

The B=1 case must bypass virtual bank copies and run the existing path byte-for-byte. This is how the primary TPOT path is preserved.

### Model and generator changes

- `tt/multichip_decoder.py`
  - Extend `MultichipDecodeStateWorkspace` with model-owned DRAM virtual banks for each layer’s recurrent, three GDN conv and optional nine PLE conv tensors. Do not enlarge the fixed B1 L1 workspace.
  - Add explicit `restore_slot(slot)` and `commit_slot(slot)` operations around the existing hydrate/compute/commit trace. Prefer fixed-address TT copy traces or one indexed device copy primitive so swaps stay on device and their TT work is traced around a declared boundary.
  - Keep expert cache shared. Do not bank expert weights or route cache entries.
  - Add allocation/release and metrics: virtual capacity, active slots, restore/commit count and bytes/time, but no hidden host state fallback.
- `tt/model.py`
  - Separate physical trace batch (`1`) from serving virtual capacity (`B`). Allocate virtual token, position, page-table/sampling state and layer banks without changing trace-bound tensor shapes.
  - Replace cohort-wide `Qwen38BatchState` semantics with a serving slot manager. It must track `slot -> request_id`, generation, active flag, prompt length, device-state validity and page-table fingerprint. Generation prevents a canceled slot’s late async completion from mutating a reused slot.
  - Add `prefill_virtual_slot(...)`, `decode_virtual_slots(...)`, `release_virtual_slot(...)`, and `reset_virtual_slot(...)`. Prefill is performed one physical row at a time and commits recurrence/token/position to the assigned bank; it must not reset other live slots or release the B1 trace merely because another request prefills.
  - Preserve vLLM cache identity. Copy/select only the supplied row of vLLM’s page table into the trace-bound B1 table.
  - Make PLE reset specific to the released/reused request, never every request in a logical cohort. Cross-request IDs must be passed to every PLE layer.
  - Keep batch-one `decode_token_out_traced()` unchanged; add the logical-B orchestration outside it.
- `tt/generator.py`
  - Move `_serving_device_feedback_current`, sampling signature/RNG and reset generation from generator-global fields to virtual-slot state.
  - Add a canonical `decode_virtual_slots` loop that delegates each row to `decode_token_out_traced()`; assemble compact token outputs without host sampling or token feedback.
  - Accept stable slot IDs rather than gathering/copying every state bank on a row condense. Mixed per-request top-k/top-p/temperature/seeds must select canonical device sampler parameters per slot. Greedy and random trace variants may be cached separately if the sampler op differs, but both must call the same `Qwen38FullModel.sample_logits`/`Sampling1D` path.
- `tt/generator_vllm.py`
  - Replace `MAX_NUM_SEQS = 1` with initial logical capacity `2`, while building the generator with physical trace batch one plus `virtual_slot_capacity=2`.
  - Remove slot `[0]`, non-identity-remap and global `_decode_started` restrictions.
  - Forward request IDs, stable slot IDs, unpadded row count and reset generations into canonical generator methods. Keep this file a protocol translator only.
  - Do not add an adapter loop that implements sampling, logits readback, token feedback, KV ownership, or PLE history itself.

### Plugin ABI changes

The current plugin’s `slot_remap` means “physically gather all state so request rows become slots.” That is unnecessarily expensive for virtual banks. Add a capability such as `supports_virtual_state_slots`, then:

- Add `request_ids` and `state_slot_ids` to `TTModelInput`. Build them from `row_req_ids` and `_req_state_slot` in `_prepare_model_inputs()`.
- For virtual-slot models, do not mutate `_req_state_slot` into row order in `_decode_state_slot_remap()`. Pass the stable slot vector for active rows. Existing models keep current remap behavior.
- Forward these fields from `submit_prefill()` and decode submission. `prefill_empty_slots` remains the allocation destination.
- Forward `unpadded_batch_size`; padded tail rows must never restore, execute, sample, advance PLE history, or emit outputs.
- Include `(req_id, slot, generation)` in `SubmittedStepContext`. Before applying a deferred result, verify the same request still owns the same generation. A finished/canceled/preempted request releases its slot only after pending submissions are drained or invalidated.
- On preemption, invalidate model-owned virtual state because re-prefill rebuilds it, matching current runner comments. On ordinary unscheduling, retain it. On cancellation/finish, reset PLE history and ownership but avoid zeroing shared expert cache.

### Page-table and reset semantics

- Prefill row `r` writes QSA K/V/indexer state through vLLM block IDs in `page_table[r]`; it commits only model-owned recurrence to `state_slot_ids[r]`.
- Decode row `r` restores `state_slot_ids[r]`, supplies `page_table[r:r+1]`, and saves back to that same slot.
- A changed page table is row-local. Page-table fingerprints/copy counters must be per virtual slot; one request’s allocation growth must not force reset of another request.
- `reset_batch` becomes a layout/submission synchronization signal, not permission to wipe all model state. Slot generation/request ownership decides reset.
- Reuse after cancel must call `reset_virtual_slot`, zero recurrence/conv/token/position/sampler state, and reset only the old PLE request history before assigning the new request.

### Sampling and async implications

- Preserve `supports_async_decode=True`. One logical step may enqueue B sequential trace replays, then return a single `_ServingDecodeOutput` that reads B compact tokens after the last event.
- The runner’s deferred completion must snapshot request/slot generations. A cancellation after submit may drop that row’s result, but must not let it update a new occupant.
- For steady device sampling, token feedback remains on TT within each slot. Slot switching restores the TT token bank; the host never writes the sampled token back.
- The existing compact token read for PLE remains declared host work. It must use the restored slot’s token and actual request ID.
- Per-request sampling parameters and seeds must survive row reorder/unscheduling. Do not retain the current generator-global sampling signature/RNG as the authority.
- Explicit host-sampling compatibility may remain for unsupported penalties/logprobs, but performance proof must use the canonical traced on-device path.

## Alternative architectures

### 1. Dual traced-B1 plus eager multi-active

Feasibility: lower than virtual microbatch without a major state-shape refactor.

Exact issue: a single current model cannot simultaneously be constructed with max batch one for the trace and max batch two for eager recurrence/sampler buffers. Two model instances require duplicated weights/runtime and unproven device capacity. Dynamically rebuilding a model or releasing/recreating traces on every cohort transition destroys B1 performance and live request state.

It becomes viable only after decoupling physical trace batch from virtual state capacity—which is effectively the recommended virtual-slot work—or after proving two resident full models fit. CPU tests would cover mode switching and state migration; TT proof would need alternating B1 traced/B2 eager requests with no trace recapture in steady B1, no state contamination, and memory headroom evidence.

### 2. Per-user B1 traces

Feasibility: lowest.

Each trace captures fixed token/position/page-table, layer canonical state, front/back crossing, terminal and sampling addresses. Independent users therefore require independent trace-bound tensors and layer canonical state or explicit swapping. The shared L1 workspace serializes layer replay, but does not provide separate per-user recurrence. Holding multiple complete trace sets also consumes trace-region and retained-buffer memory. Cancellation must release exactly one trace set without invalidating another, and async routing must select the correct set. This duplicates most of the virtual-bank problem plus trace memory, so it is not the first implementation.

### 3. Generalized segmented batch trace

Feasibility: best long-term throughput, highest implementation risk.

Construct the whole model at fixed B=2 (then 4/8/16/32), enlarge trace shapes, remove batch-one guards, extend `MultichipDecodeStateWorkspace`, and generalize host expert service from one row’s ten routes to B rows with correct per-user weights and aggregation. PLE staging/history and canonical sampling already have batch-shaped APIs but need cancellation/padding tests. Slot remap must gather every model-owned recurrent/conv/sampler state consistently. This is the right optimization after microbatch correctness, not the smallest completion fix.

TT proof must compare every active row against independent B1 controls, exercise different positions/pages/routes/sampling policies, and show traced B2 performance without fallback.

### 4. Scheduler microbatch with physical-B1 virtual slots

Feasibility: highest and recommended for the B=2 proof.

It reuses the already-correct B1 segmented model and runner lifecycle, pays serial B1 compute plus device state-swap cost, and avoids batch-shape changes in the 48-layer trace. It provides real active semantics even though throughput will initially be approximately serial. The main risk is state-bank copy volume and trace/address correctness, which is directly measurable at B=2 before increasing capacity.

## Focused CPU tests before TT

Add model-free tests using fake state banks/outputs:

1. Two requests prefill into distinct slots; interleaved decode preserves independent token, position, page table, sampler seed and PLE history.
2. Row condense/reorder sends stable `state_slot_ids` and does not copy or relabel virtual state.
3. An unscheduled running request retains ownership; preemption invalidates it; finish/cancel resets only that request.
4. Cancel-after-submit plus immediate slot reuse: stale completion is discarded by `(req_id, slot, generation)` and cannot append a token to the new request.
5. Page-table changes for request A never change request B’s row/fingerprint/counters.
6. Padding rows never execute, sample, increment position, lookup PLE, or emit tokens.
7. Mixed greedy/random parameters follow request identity across reorder; seeds advance exactly once per generated token.
8. PLE histories are request-keyed and reset on finish/reuse; expert cache entries remain shared and resident.
9. B=1 dispatch calls the pre-existing trace method directly with no virtual restore/commit.
10. Adapter tests assert no host argmax/top-k fallback, full-logit read, Python token feedback, hidden KV allocation, or adapter-owned sampling loop.
11. Plugin async tests cover two pending rows, cancellation, preemption, reused request ID, and completion reordering.
12. Capacity ladder configuration rejects values above allocated virtual slots and never reports queued requests as active capacity.

Minimum plugin test surfaces: `tests/test_input_batch.py`/state-slot tests, async decode tests, model-runner input/submission tests, and cancellation/preemption tests. Minimum model test surfaces: generator vLLM protocol tests, serving-state CPU fakes, host PLE isolation/reset, sampling identity/reorder, and lifecycle metrics.

## Required TT/vLLM proof ladder

1. **B1 regression:** existing primary command and sampling profile; prove the same trace replay counters and no measurable avoidable TPOT regression. Verify no virtual state-copy counters increment.
2. **B2 direct model runner:** two distinct prompts, unequal/non-aligned lengths and different page tables. Compare each token stream against two independent deterministic B1 runs. Require stale-token/current-position/page-table checks for both rows.
3. **B2 lifecycle:** concurrent requests with one cancellation, one unscheduled interval, slot reuse, preemption/re-prefill and cross-request PLE isolation. Inject expert upload failure and PLE upload failure; verify ownership remains recoverable.
4. **B2 sampling:** greedy/greedy and greedy/random within device-supported top-k; seeds and output attribution remain request-local. Unsupported sampling uses only explicit optional host compatibility.
5. **B2 async:** `supports_async_decode=True`, two outputs per engine step, deferred completion generation guard, no stale result after cancellation.
6. **B2 server:** advertise `--max-num-seqs 2`; prove two simultaneous requests are active, not merely queued, from scheduler/slot/trace metrics. Run qualitative and a two-concurrency benchmark.
7. **Capacity ladder:** try 4, 8, 16 and 32 virtual slots. At each rung run the same isolation suite and measure device state-bank memory, restore/commit bytes/time, KV block capacity, host expert/PLE metrics and TPOT. Stop only at observed allocation or latency/resource failure and record the largest proven value.
8. **Cleanup:** after every failure/cancel/server shutdown, no vLLM/EngineCore process or device holder remains.

For the B2 gate, required metrics are logical active requests, physical trace batch, virtual capacity/occupancy, slot restores/commits and bytes/time, per-request page-table copies/skips, trace replays (expected two per two-row decode step), expert hit/miss/eviction/H2D/stall, PLE lookup/history/reset/H2D, compact token reads, host-sampling calls and prohibited-fallback counters.

## Ranked hypotheses by feasibility

1. **Physical-B1 virtual-slot scheduler microbatch:** most feasible; reuses the measured trace and current runner slot lifecycle. The key unknown is TT cost/correctness of device state banks and fixed-address restore/commit.
2. **Generalized segmented B trace:** likely best eventual throughput, but requires the broadest kernel/host expert route generalization.
3. **Dual B1 trace/B>1 eager:** not independently small in one current model; it first needs the same physical/logical state decoupling or a second-model capacity proof.
4. **Per-user B1 traces:** duplicates trace-bound state and memory while still needing request-local recurrence; least attractive.

These are feasibility hypotheses, not runtime evidence. Only the source contradictions and existing ownership/lifecycle behavior above are established.

## AutoDebug runner note

The skill runner was attempted twice. The default fresh Codex run failed before repository access because nested bubblewrap could not create the loopback namespace. The skill-supported Claude fallback produced no report or source output and was terminated before any file was written. The findings above were then independently derived and checked against the cited source; no hardware was touched.
