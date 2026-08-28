# AutoDebug trace allocation report

Source-only diagnosis for the full-model trace-lifecycle warning flagged by stage review. This pass did not use TT hardware, did not spawn subagents, and did not edit implementation files.

## Problem statement

Two accepted full-model artifacts contain the same Metal allocator warning:

```text
Allocating device buffers is potentially unsafe due to the existence of an active trace. These buffers may be corrupted once a trace is executed if they are not released before then. Use the trace allocation tracker to verify.
```

Observed locations:

- `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_split_trace_watcher_fixed.log`: warning at approximately line 87 after ANSI stripping, followed by `1 passed, 1 warning in 9.85s`. This runs `test_full_model.py::test_reduced_real_weight_split_greedy_trace_contract` with `TT_VISIBLE_DEVICES=0,1`, `TT_METAL_WATCHER=2`, and `TT_METAL_WATCHER_DISABLE_ETH=1`.
- `models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_watcher_fixed.log`: warning at approximately line 497 after ANSI stripping, followed by `1 passed, 1 warning in 87.56s`. This runs `test_full_model.py::test_full_48_layer_token_out_trace_smoke` with `RUN_QWEN38_FULL_MODEL=1`, `TT_METAL_WATCHER=5`, and `TT_METAL_WATCHER_DISABLE_ETH=1`.

The warning appears amid repeated deprecated `all_gather` warnings and is emitted from `tt_metal/impl/allocator/allocator.cpp:130`. The full-48 run still reports `traced=True`, `trace_replays: 1`, and the expected host-copy/readback counters, so this is not a functional crash. It is a trace safety proof gap.

## What the allocator warning means

The warning is not, by itself, proof of corruption. It means a normal device buffer allocation occurred while at least one trace was active.

Relevant source behavior:

- `tt_metal/distributed/mesh_device.cpp`: a trace becomes active after `end_mesh_trace` registers it with the trace allocation tracker, and remains active until `release_mesh_trace` unregisters it.
- `tt_metal/impl/allocator/trace_allocation_tracker.cpp`: if trace allocation tracking is disabled, registering an active trace sets `allocations_unsafe_ = true`; any later non-trace buffer allocation logs the generic warning.
- `tt_metal/impl/allocator/allocator.cpp`: `AllocatorImpl::verify_safe_allocation()` logs this warning once per host thread when `allocations_unsafe_` is true.
- `ttnn/ttnn/trace_allocation_config.py` and `ttnn/ttnn/__init__.py`: with `TT_METAL_TRACE_ALLOC_TRACKING=1`, Python wraps `ttnn.execute_trace` and calls `UnsafeAllocationTracker(device).verify_before_replay(trace_id)` before replay.
- `ttnn/ttnn/unsafe_allocation_tracker.py`: verification raises `RuntimeError("Found N device buffer(s) still alive before trace replay...")` only for tracked younger allocations that are still alive and not marked corruptible.

Therefore, the accepted logs only prove that at least one allocation occurred while a trace was live. They do not prove whether any unsafe allocation survived until an older trace replay. Stage review is correct to require allocation-tracker evidence because the tracker was not enabled in either accepted artifact.

## Trace lifecycle in the full-model path

The relevant full-model path is in `models/autoports/qwen_qwen3_8_flash_next/tt/model.py`.

Persistent decode inputs are allocated before capture:

- `Qwen38FullModel._allocate_persistent_decode_inputs()` allocates stable device buffers for `decode_token_input`, `decode_current_pos`, `decode_page_table`, and sampling parameters.
- `reset_batch_state()` copies prompt state into those existing stable buffers and rejects a different batch state once traces are ready.
- `sample_logits()` writes sampled token output into `state.token_input` via `tt_out_tok=state.token_input`.

`capture_decode_traces()` then captures multiple traces that are intentionally live together:

1. Warm the trace programs with `_warm_trace_programs()`.
2. Capture ingress and leave `ingress_trace_id` active.
3. Mark `ingress_trace_output` corruptible and execute ingress once.
4. Capture each `HostBackedSegmentedDecodeTrace` layer while ingress and previous layer traces remain active.
5. Capture terminal projection, mark `trace_logits` corruptible, and execute terminal once.
6. Capture sampling and position traces.
7. Leave all traces active for steady-state decode replay.

`_replay_decode_traces()` then executes ingress, each layer trace, terminal, sampling, and position in order. `release_decode_traces()` releases position/sampling/terminal first, then layer traces in reverse order, then ingress.

This design necessarily creates younger buffers while older traces are active. It relies on either:

- releasing temporary younger buffers before any older trace can replay, or
- marking retained younger outputs as corruptible because the replay order regenerates them before consumers read them.

## Most likely allocation source

The most likely first explicit Python-level allocation that triggers the warning is the GDN decode-state snapshot clone inside `HostBackedSegmentedDecodeTrace.capture()`.

Evidence:

- The reduced failing log runs `test_reduced_real_weight_split_greedy_trace_contract`, whose `layer_indices=(0,)`. That test includes layer 0 only and therefore excludes layer 1 PLE as the primary cause of the shared warning.
- In `Qwen38FullModel.capture_decode_traces()`, ingress is captured and ended before the first layer capture starts. From this point onward, ingress is an active trace.
- In `models/autoports/qwen_qwen3_8_flash_next/tt/multichip_decoder.py`, `HostBackedSegmentedDecodeTrace.capture()` calls `_snapshot_decode_state()` for linear-attention/GDN layers.
- `_snapshot_decode_state()` uses `ttnn.clone(...)` to allocate DRAM snapshots of canonical GDN/PLE state.
- Those clone allocations occur after ingress is active, so the allocator warning is expected even if the snapshots are later freed safely.
- `_release_state_snapshots()` is called before `HostBackedSegmentedDecodeTrace.capture()` returns, so these snapshot buffers should not still be alive at the next ingress replay if cleanup succeeds.

Source-only confidence: high for “first warning is caused by an allocation after ingress capture starts”; medium-high that the first explicit source allocation is the GDN snapshot clone. Exact buffer IDs and stack traces require the tracker-enabled rerun.

## Other allocations that can occur under active traces

Ranked hypotheses:

1. GDN state snapshot clones during layer-0 trace capture
   - Source: `HostBackedSegmentedDecodeTrace._snapshot_decode_state()` via `ttnn.clone(...)`.
   - Why likely: layer 0 is present in both warning logs, and reduced layer-0-only coverage rules out PLE as the shared first cause.
   - Risk assessment: likely controlled if snapshots are deallocated before any older ingress trace replay. A tracker pass should prove this by finding no live unsafe snapshot buffers at replay.

2. Retained layer front/back outputs and terminal logits
   - Source: `HostBackedSegmentedDecodeTrace.capture()` retains `HostDecodeFront` and `output`; `Qwen38FullModel.capture_decode_traces()` retains `trace_logits`.
   - Why plausible: these buffers are allocated while older traces are live and intentionally survive into replay.
   - Existing mitigation: the layer front fields are marked with `_mark_front_corruptible()`, back output is marked with `ttnn.mark_corruptible(output)`, and terminal logits are marked with `ttnn.mark_corruptible(self.trace_logits)`.
   - Risk assessment: controlled if every retained younger output is regenerated by its own trace before any consumer reads it. This is exactly what the allocation tracker should verify.

3. Program-cache allocation under an active trace
   - Source candidates: lazy kernels for `ttnn.clone`, terminal projection, sampling, position increment, or host-backed service operations.
   - Why less likely: full-model capture freezes program-cache misses with `device.set_program_cache_misses_allowed(False)`, and each layer capture also requires `programs_prepared=True`. Prior multichip work documented that missing BF16 clone warmup could appear as an unsafe allocation, then adopted a two-phase warm-before-capture protocol.
   - Risk assessment: if tracker context reports `program_cache:*`, the warmup is incomplete and should be fixed rather than suppressed.

4. PLE/expert host-service uploads during full-48 replay
   - Source candidates: `QwenDeviceExpertCache.ensure_indexed()` and `PLEDeviceStaging.upload_decode()`.
   - Why lower rank: the reduced layer-0-only test has the same warning without PLE. Source inspection shows host TTNN tensors are created with no device and copied into existing device staging/slot buffers.
   - Risk assessment: still worth checking in the full-48 tracker run, because hidden device allocations in this path would only appear when PLE layers are included.

5. Sampling or position trace temporaries that survive replay
   - Source candidates: `Sampling1D.decode_forward(...)` and `ttnn.plus_one(..., output_tensor=state.position_input)`.
   - Why lower rank: sampling output is written into the preallocated `state.token_input`; position increment uses the preallocated position tensor.
   - Risk assessment: tracker should catch any unexpected retained device output from these traces.

## Existing related evidence

Prior multichip-decoder work already investigated similar multi-live trace allocation safety:

- `models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/work_log.md` documents a two-live-trace allocation tracker pass and the accepted stack protocol:
  1. warm every distinct capture signature before any trace exists,
  2. capture in model order with `programs_prepared=True`,
  3. mark only front crossings, PLE staging, and final output as corruptible,
  4. serialize capture/replay/release and release back before front.
- `models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/SEGMENTED_TRACE_AUTOFIX.md` records that canonical state snapshots are ordinary DRAM clones that are restored/freed before older replay and are not marked corruptible.
- `models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_shared_workspace_segmented_trace_stack` exercises two retained layer traces, verifies state stability, and was documented as passing with tracker evidence.

That evidence is useful but not sufficient for the full-model stage gate. The full-model path adds the ingress trace, terminal trace, sampling trace, position trace, token feedback, and page-table update behavior. The full-model accepted artifacts must therefore have their own allocation-tracker logs.

## Verified verdict

The source-only report correctly left this unresolved. The required runtime
experiments were subsequently run with `TT_METAL_TRACE_ALLOC_TRACKING=1`,
tracebacks/referrer depth 12, and watcher:

- `reduced_split_trace_alloc_tracker.xml`: 1/1 passed in 10.091 s;
- `full48_tokenout_trace_alloc_tracker.xml`: 1/1 passed in 124.733 s.

Neither run raised the tracker's unsafe-live-allocation `RuntimeError`; both
completed capture, replay, and clean device teardown. The warning class is
therefore verified controlled for the complete delivered path: snapshot and
other younger buffers are freed or explicitly marked safe before any older
trace replay. No source change, tracker suppression, program-cache skip, or
trace downgrade was needed.

## Smallest verify/refute experiment

Run the reduced layer-0-only full-model trace test first. This is the shortest reproduction that contains the shared warning and excludes PLE.

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_TRACEBACKS=1
export TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12
export TT_METAL_WATCHER=2
export TT_METAL_WATCHER_DISABLE_ETH=1
pytest -q --tt-arch blackhole --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_reduced_real_weight_split_greedy_trace_contract \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_split_trace_alloc_tracker.xml \
  2>&1 | tee models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_split_trace_alloc_tracker.log
```

Expected outcomes:

- Pass: the warning class is controlled for the reduced full-model path. The likely GDN snapshot allocations were freed before replay, and retained outputs were correctly marked corruptible.
- Fail with `Found N device buffer(s) still alive before trace replay`: use the reported allocation contexts/referrers as the fix target. If the context is `ttnn.clone` and referrers include state snapshots, inspect `_release_state_snapshots()` and lifetime ordering first. If the context is a layer front/back or terminal output, inspect missing `ttnn.mark_corruptible(...)`. If the context is `program_cache:*`, add the missing warmup instead of suppressing the tracker.

If the reduced run passes, run the full-48 token-output smoke with the same tracker evidence:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_TRACEBACKS=1
export TT_METAL_TRACE_ALLOC_REFERRER_DEPTH=12
export TT_METAL_WATCHER=5
export TT_METAL_WATCHER_DISABLE_ETH=1
export RUN_QWEN38_FULL_MODEL=1
pytest -q --tt-arch blackhole --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_48_layer_token_out_trace_smoke \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_trace_alloc_tracker.xml \
  2>&1 | tee models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_trace_alloc_tracker.log
```

Expected outcomes:

- Pass: stage-review P1 should be closed by attaching both tracker logs/XMLs and noting that the generic allocator warning is expected when tracking is disabled in a multi-live-trace capture protocol.
- Fail only in full-48: prioritize PLE/expert service and sampling/position contexts, because reduced layer-0 already cleared ingress/GDN/front/back/terminal basics.

Important environment detail: `TT_METAL_TRACE_ALLOC_TRACKING` and `TT_METAL_TRACE_ALLOC_TRACEBACKS` are read at Python import time, so they must be exported before starting `pytest`.

Diagnostic-only knob:

- `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` can separate program-cache noise from model tensor lifetimes if, and only if, the tracker failure points to `program_cache:*`. It should not be used for the final gate artifact because program-cache allocations under active traces indicate incomplete warmup.

## Minimal fix direction if tracker fails

Do not make a broad trace rewrite. Use the tracker failure context:

- Snapshot clone survives: release it earlier or fix a leaked Python reference; do not mark canonical state snapshots corruptible unless replay ordering proves they are regenerated before use.
- Retained crossing/output survives unmarked: add the narrow `ttnn.mark_corruptible(...)` at the boundary that is regenerated before consumption.
- Program cache survives: add the missing warmup before any trace is captured; keep `device.set_program_cache_misses_allowed(False)` as the guard.
- Replay-time host service allocates a device buffer: convert it to persistent preallocated staging or host-only construction plus copy into existing buffers.

The final acceptance evidence should be tracker-enabled logs for both the reduced trace-contract test and the full-48 token-output smoke.
