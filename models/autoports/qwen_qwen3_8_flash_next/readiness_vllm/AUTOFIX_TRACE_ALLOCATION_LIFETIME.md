# AutoFix: B2 Trace-Allocation Lifetime

Date: 2026-08-29

## Outcome

The final B2 tracker failure had two observed root causes, and source review
found one additional cache-lifecycle blind spot. All three are fixed:

1. Later prefills replaced construction-time GDN/PLE state tensors with new
   device buffers while decode traces were live. Prefill now copies updates
   into the fixed state buffers and frees only the transient producer output.
2. A previously unseen prompt or scheduler-release workload can add persistent
   TTNN program-cache buffers while an older decode trace is live. Virtual
   prefill now compares the exact mesh program-cache entry count across the
   whole public admission, starting before `released_state_slots` and ending
   after prefill/sampling/output handling. If it grows, the old traces are
   released before the method returns or any later decode replay; the next
   decode safely recaptures. Cache-hit admissions keep the existing trace.
3. A failed first enqueue can leave a cached MeshWorkload whose kernel-binary
   initialization is lazy, so a later cache hit can allocate without changing
   entry count. Any partial prefill/decode/release exception permanently marks
   cache initialization uncertain for that generator lifetime. Later prefills
   then release a live trace before eager work. Request-slot poison may clear
   after lifecycle cleanup, but this allocation-safety flag does not.

Both final tracker gates pass with unfiltered program-cache tracking. The
reduced real-layer sequence (GDN, GDN+PLE, QSA; logical prompt lengths
`1,63,1,63`) completed with deterministic repeated outputs, program-cache
entries `[697,737,737,737]`, two trace replays, and exactly one
`prefill_trace_invalidations` event. The exact final B2 server lifecycle also
passed: 150 replay tokens, one safe prefill invalidation/recapture, active B2
overlap and cancellation, deterministic follow-ups, zero tracker errors or
warnings, and clean process/device release.

## Original evidence and classification

- Final serving configuration: `readiness_vllm/server.log.gz` uses physical-B1
  decode with virtual capacity two, `max_model_len=262144`, `block_size=64`,
  async scheduling, decode-only trace, and all-device sampling.
- The coarse allocator warning is at line 735 of that compressed log.
- Exact final-config tracker reproduction:
  `readiness_vllm/final_b2_trace_tracker/server.log` line 1594 reports 75 live
  buffers before replay during a follow-up request after 129 successful
  replays.
- 73 buffers were request-persistent state created by prefill: 36 GDN
  convolution states, 36 GDN recurrent states, and one PLE convolution state.
- The remaining two were program-cache buffers. Their contexts are preserved
  at lines 1646 and 1697 of the tracker log: `UntilizeDeviceOperation` and
  shape-specialized `UntilizeWithUnpaddingDeviceOperation`.
- After fixed-address state updates, the alternating-length diagnostic showed
  program-cache entries `697 -> 737 -> 737 -> 737`: exactly one unseen
  length-63 workload introduced 40 entries, and subsequent identical shapes
  were cache hits.
- Source review then identified a count-delta blind spot: a failed first
  enqueue can leave a cached MeshWorkload whose kernel binaries are initialized
  lazily on a later hit without adding another cache entry. It also found that
  scheduler slot-release copies originally ran before the prefill snapshot.
  The persistent uncertainty fallback and full public-boundary guard close
  both cases.

The allocator warning alone cannot identify the allocation because it is a
coarse once-per-host-thread warning. The tracker evidence is definitive: the
first failure contained persistent state and program-cache buffers; after the
fixed-state patch, the remaining delta was entirely program cache.

## Proven fixes

### Fixed request-state ownership

`FunctionalDecoder._update_prefill_state()` enqueues
`ttnn.copy(update, persistent)` and then address-aware `_free(update,
persistent)`. The persistent tensor retains its construction-time object,
buffer address, TensorSpec, layout, memory config, and dtype. All three
functional and all four fused state-update sites use this helper.

This also removes one clone-internal untilize workload. The separate
shape-specialized `to_layout(..., ROW_MAJOR)` workload remains valid and is
handled by the program-cache lifecycle guard.

### Exact program-cache delta guard

`Qwen38Generator._snapshot_live_trace_program_cache()` records the mesh's
actual `num_program_cache_entries()` before scheduler slot releases at the
public virtual-prefill boundary. `_finish_live_trace_prefill()` runs only after
release processing and every row has completed prefill, sampling/output
disposal, and `finish_virtual_prefill()`. Standalone scheduler release calls
use the same guard. No decode trace replay occurs inside either synchronous
region; the shared model runner serializes these model API calls.

If the exact count changed while the old trace is still live, all decode traces
are released before returning. The next decode captures against the same
stable physical-B1 input/state buffers. If the count did not change, the trace
is retained. The exception path releases a still-live trace regardless of the
count and marks initialization uncertain. Because `MeshWorkload::load_binaries`
can lazily allocate on a cache hit after a failed first enqueue, the persistent
uncertainty flag makes every later prefill pre-release a live trace. This is a
rare failure-mode fallback; healthy cache-hit serving remains unchanged.

The lifecycle is reported as
`virtual_slot_metrics.prefill_trace_invalidations`. It is an admission-time
event, not per-token work. The primary single-user path has no old trace during
its first prefill, and steady decode remains the canonical traced token-out
path.

## Rejected hypotheses and controls

- `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` was not used as a fix. It hides
  exactly the persistent allocations implicated by the tracker and does not
  change their lifetime.
- Logical prompt-length or hand-maintained signature sets were rejected. TTNN
  program identity includes the full workload configuration, and length alone
  cannot safely predict cache hits.
- Unconditional release on every admission was rejected because exact
  cache-hit admissions can safely retain the trace. The runtime delta observes
  the complete workload set without per-token overhead.
- Marking an entire prefill corruptible was rejected because that would also
  suppress tracking of persistent program-cache or request-state allocations.

## Tests and artifacts

- `readiness_vllm/autofix_trace_fixed_state_host.xml`: 2/2 pass for fixed
  object/address retention and all functional/fused call sites.
- `readiness_vllm/autofix_trace_program_delta_host.xml`: 4/4 pass for cache-hit
  retention, cache-growth invalidation, release-copy growth inside the guard,
  failed-enqueue/no-count-growth invalidation, and persistent uncertainty.
- Full host adapter suite: 33/33 pass (`tests/test_generator_vllm.py`).
- Readiness metric derivation suite: 6/6 pass; the lifecycle delta now includes
  `prefill_trace_invalidations`.
- `readiness_vllm/autofix_trace_fixed_state_tt.xml`: deterministic completion
  equality for repeated logical lengths and the observed program-cache sequence
  `[697, 737, 737, 737]`.
- `readiness_vllm/autofix_trace_program_delta_tt.xml`: unfiltered tracker run
  passes 1/1 with deterministic repeated outputs, program-cache entries
  `[697,737,737,737]`, `trace_replays == 2`, and
  `prefill_trace_invalidations == 1`.
- Targeted TT assertion:
  `tests/test_vllm_virtual_slots_tt.py::test_reused_prefill_programs_alternating_one_and_sixty_three_tokens`
  requires that exact lifecycle.
- `readiness_vllm/trace_allocation_tracker_b2_audit.json`: exact final-config
  verdict `pass`, with source SHA-256 values matching the frozen
  `generator.py`, `functional_decoder.py`, and `fused_decoder.py` used by the
  server.
- `readiness_vllm/final_b2_trace_tracker_fixed/server.log.gz`: zero generic
  active-trace allocation warnings, zero unsafe-live-allocation errors, zero
  corruption-warning blocks, zero runtime errors, and zero tracebacks.
- `readiness_vllm/final_b2_trace_tracker_fixed/host_serving_lifecycle.json`:
  active two-stream overlap, accepted cancellation, survivor completion,
  identical follow-ups, 150 trace replays, zero model-only replays, 8 bank
  commits, 5 restores, 1 reset, 1 prefill trace invalidation, 0 stale
  rejections, and no active/valid slots or PLE history remaining.
- `readiness_vllm/final_b2_trace_tracker_fixed/process_cleanup_audit.json`:
  pass with zero matching vLLM/EngineCore processes, zero holders on devices 0
  and 1, and healthy DRAM status on both devices.

Tracker environment used for the targeted TT gate:

```text
TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_TRACEBACKS=1
TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=unset
```

No tracker skip option is part of the gate.

## Exact final B2 proof

The passing audit used `max_num_seqs=2`, physical decode batch one, virtual
capacity two, `max_model_len=262144`, `block_size=64`, async scheduling,
decode-only trace, all-device sampling, the selected final TT configuration,
and no tracker skip flag. Its workload covered a cold baseline, two
simultaneously active 128-token streams, peer cancellation with survivor
completion, two deterministic 12-token follow-ups, and a final sentinel.

Server command (with the tracker environment above exported before launch):

```text
python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py --model-dir models/autoports/qwen_qwen3_8_flash_next --hf-model Qwen/Qwen3.8-Flash-Next --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/final_b2_trace_tracker_fixed --stages serve --mesh-device P300 --port 8019 --max-num-seqs 2 --sampling-profile full --block-size 64 --max-model-len 262144 --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' --additional-server-args='--async-scheduling'
```

The changed prefill program set caused exactly one lifecycle invalidation. The
old trace was released before return; the next decode recaptured, and all later
unfiltered tracker replay checks found no younger live buffers. The server then
shut down without leaving a vLLM/EngineCore process or device holder. This is
current-source, exact-B2 proof and supersedes the historical max-one tracker
audit and the pre-fix 75-buffer reproduction.
