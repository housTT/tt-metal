# AutoFix: decode output collection

## Starting evidence

`AUTODEBUG_output_collection.md` identified per-token `_read_async` transfers and
`_finish_read` event waits in free-running device generation. The initial
investigator inspected source only and prepared `probe_output_history.py`.
The selected BF16 HiFi4 LM head, decoder policy, sampler semantics, and native
1x4 TP4 mesh on four Blackhole chips on P300c boards remain the required contract.

## Hypothesis experiment

- Hypothesis: a replicated UINT32 RM history `[128,1,1,32]`, INT32 RM DRAM
  cursor `[1]`, and existing UINT32 RM feedback `[1,1,1,32]` support exact repeated
  trace-side append using `indexed_fill`, `copy` back, and `plus_one(cursor)`.
- Experiment executed by the parent hardware owner on 2026-09-05:

  ```bash
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.probe_output_history \
    --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/output_history_probe_v1.json
  ```

- Result: **verified**. Five windows of 1, 7, 128, 33, and 33 replays passed exact
  UINT32 comparison on all four ranks, including values above 65535 and 2^31,
  wraparound near 2^32, sparse fixed slots, changed capture inputs, unwritten
  sentinel rows, repeated resets, and persistent allocation identities. Each
  window submitted nonblocking replays, with zero loop host reads/writes/waits,
  then one final mesh history read and one event wait. Four program-cache
  entries remained stable.
- Evidence: `output_history_probe_v1.json`,
  `logs/output_history_probe_v1.log`,
  `logs/output_history_probe_v1.provenance.json`, and the corresponding source
  snapshot archive. The process returned zero. Watcher ETH monitoring was
  explicitly disabled; no profiler was enabled in this experiment.
- Timing context: the 128-step component window completed in 34.76 ms, including
  probe-only token increment and final mesh transfer, with watcher enabled.
  This establishes bounded component cost for the experiment; it is not a
  measured full-model performance improvement.

## Candidate implementation after verification

Only `tt/generator.py` received implementation edits from this investigator.
The parent's existing `decode_forward(read_from_device=...)`, `replay_decode`,
and `build_generator` norm configuration changes are preserved.

- Allocate fixed history and DRAM cursor at generator construction for device
  sampling. The history consumes 16 KiB per rank; the append uses a 16 KiB
  temporary that is deallocated within the captured operation sequence. The
  cursor uses its aligned DRAM allocation.
- Warm the append sequence before capture and reset its cursor after warmup.
  The append helper only reads feedback and mutates history/cursor; it does not
  run another sampler, change token feedback, or touch seed/penalty/cache state.
- Capture a plain sampler trace and a sampling-plus-history variant, both bound
  to the model trace's persistent logits and feedback allocations. `_replay`
  selects one sampler variant after the model replay, so every decode step
  still submits exactly two traces. The collector's temporary is freed after
  the recorded copy and cursor increment, before capture ends.
- `_capture` releases/recreates both sampler variants and the model trace.
  Existing program-cache-change recapture and sampling-mode reconfiguration
  therefore refresh all trace dependencies. `teardown` releases both variants;
  no collector temporary is retained on the generator.
- Free-running device generation records **decode outputs only**, in windows
  of up to 128 steps. It reads one rank's replicated history at each window
  boundary and once after the final partial window, selects active user lanes,
  and appends them to the existing first-token prediction. An explicit capacity
  check prevents an extra collector replay before a boundary reset.
- The first prefill token retains the original `_read_tokens` boundary. TTFT
  remains measured through caller-visible first-token readback. This first read,
  initial history-cursor reset, and position/page setup precede decode timing
  and its counter snapshot. There is no eager post-capture append allocation.
- Teacher forcing, host-sampling callbacks, EOS trimming, and low-level APIs use
  their previous plain sampler path and visible outputs. `replay_decode` never
  appends history and cannot overrun the collector even after prior generation.
- Counters add `history_replays`, `history_index_refreshes`, and
  `history_readbacks`. The final transfer also increments existing `readbacks`
  and `read_waits`, preserving honest end-to-end decode timing.

Expected free-generation decode counters (excluding first-token read/setup):

| Requested new tokens | Decode/history replays | History reads / final waits | Cursor refreshes inside timed loop |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |
| 128 | 127 | 1 | 0 |
| 129 | 128 | 1 | 0 |
| 260 | 259 | 3 | 2 |

The request still performs its original first-token read. A long request reads
history every 128 decode steps, so the claim is **no per-token decode readback**,
not zero host reads or one transfer for an arbitrarily long entire request.

## Verification and remaining work

Source-only checks executed by the investigator:

```bash
python3 -m py_compile models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py
pre-commit run --files models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py -q
```

All applicable pre-commit checks passed; all 16 host-contract tests passed. The
tests load the generator AST without importing TTNN or accessing hardware.
No C++/CMake changes, dependencies, or build were needed.

The parent is running `generation_contract.py` under watcher to compare this
collector against a same-token per-step-readback control at 8, 128, and 260
tokens, alternate greedy/sampled/greedy requests, and verify the unchanged
public device-output API. Full-layer parity, penalty-enabled paths, mixed
lengths/inactive slots, trace recapture, and warmed full-model measurements remain
the parent's integration gates. The component result alone does not prove those
gates or complete the optimized-full-model stage.

Status at investigator handoff: **component hypothesis verified; integration
candidate implemented and host checks passing; hardware integration validation
owned by parent**.

## Verified allocation-lifetime failure and focused fix

`generation_contract_v2` was rejected before its first model replay by the
enabled allocation tracker: `Found 1 device buffer(s) still alive`, buffer 8461.
This was an integration lifetime failure; no decode parity result was produced.
The parent reran the unchanged source with allocation tracebacks enabled.
`logs/history_tracker_diagnostic_v1.log:50` identifies the same buffer, its
allocation stack `_capture -> _append_output_history -> indexed_fill`, and its
sole live reference `gen._history_scratch`, shape `[128,1,1,32]`. The diagnostic
exited 1 and closed devices. This proves the exact offending allocation.

Source mechanism:

- `MeshDeviceImpl::end_mesh_trace` registers completed traces. The allocator's
  `record_allocation_if_unsafe` assigns a newly allocated buffer to every already
  active trace's unsafe set. Thus scratch allocated during the later history
  sampler capture is unsafe relative to the older model/plain-sampler traces.
  Retaining it as `_history_scratch` makes the rejection reproducible.
- The standalone component probe had only one trace and allocated its scratch
  while that trace was still being captured. It therefore did not exercise
  the older-live-trace lifetime boundary exposed by generator integration.
- For the exact history shape and dim 0, `indexed_fill`'s generic geometry has
  128 slices and `outer_count=inner_count=1`. Its worker split covers all 128
  slices. Every reader iteration selects either the new tokens or the persistent
  history row, and the generic writer writes `page_id=my_slice` unconditionally.
  Every one of the temporary's 128 rows is rewritten before `copy` consumes it.
  No append invocation reads a previous value from this temporary.
- Standard traced op chains release intermediates after their final consumer;
  this model already explicitly deallocates decoder intermediates during
  capture. `test_trace_allocation_tracking_acknowledgments_and_lifetime` also
  verifies that deallocation retires that specific buffer from unsafe tracking.

Fix after exact attribution: `_append_output_history` now deallocates `updated`
after recording `indexed_fill`, `copy`, and `plus_one`. Remove the retained
`_history_scratch` attribute and its capture/teardown ownership; warmup uses the
same bounded lifetime. Persistent history/cursor/feedback allocations remain
unchanged. No `mark_corruptible`, allocation suppression scope, tracker bypass,
or tracker configuration change was added.

The same pre-commit checks and all 16 host-contract tests passed after the fix.
The parent owns the `generation_contract_v4` hardware rerun with tracking kept
enabled. **The lifetime cause is proven and the narrow fix is implemented;
device correctness and performance verification remain pending that rerun.**

## Subsequent shared-generator lifecycle fix

The full qualitative run later reached its seventh request and exceeded the
100000000-byte trace reservation. `AUTODEBUG_trace_lifecycle.md` records a fresh
source diagnosis, failing/passing host experiment, and reduced hardware proof.
The runner intentionally overrides public `teardown` while reusing a generator;
the collector refactor's internal call to that public method therefore leaked
superseded traces. Internal release now uses `_release_traces`, while public
`teardown` delegates to it. All three internal callers were corrected.

The combined host suite passes 19 tests. `trace_lifecycle_v2` passed eight
recaptures with exact tokens, constant 1,638,400 TRACE bytes/device, and zero
TRACE bytes after final cleanup. Watcher and allocation tracking were enabled;
the mesh closed. This resolves the focused lifecycle failure; the parent owns
the original full qualitative rerun and remaining stage gates.

## Parent integration verification

`generation_contract_quick_v1.json` passes greedy/sampled/greedy eight-token
exact parity against explicit per-token readback controls, then public device-output
API parity. Worker watcher and allocation tracking remain enabled. The exact
scratch-lifetime fix therefore closes the original tracker failure. Longer
window-boundary verification follows separately.

The interrupted long instrumented v3 was progressing before live triage; the
triage tool halts Blackhole cores by design. This was recovered with a bounded
reset, four-chip listing and configured mesh open/close (`MESH_SMOKE_OK`,
`logs/mesh_smoke_after_triage_v2.log`). See `AUTOTRIAGE_history.md` and work log.


## Final stage closure

The earlier investigation status above is preserved as historical evidence.
The selected implementation and completed final gates are recorded in the
[stage report](README.md), [runtime audit](runtime_audit.md),
[final full32 watcher control](prefill_integration_full32_v2/summary.json),
[long exact replay controls](prefill_integration_long_v2.json), and
[final profiling report](tracy/README.md). Earlier pending experiments are not
claims that these final gates remain unrun; rejected hypotheses and failed
receipts remain preserved. Independent stage review owns the final verdict.
