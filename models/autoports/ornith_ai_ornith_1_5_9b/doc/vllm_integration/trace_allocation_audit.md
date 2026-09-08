# Trace allocation audit

Date: 2026-09-08 UTC. Source-only audit prompted by allocator warnings in the
serving logs. No TTNN import, hardware command, profiler, watcher, or inference
was run by this audit. Only the standalone adapter probe gained an optional
guard requirement and JSON metadata; production runtime source is unchanged.

## Findings

The native mechanism already checks every public `ttnn.execute_trace` call when
`TT_METAL_TRACE_ALLOC_TRACKING=1` was set **before importing TTNN**:

1. `ttnn/ttnn/trace_allocation_config.py` captures the process-start setting.
2. `ttnn/ttnn/__init__.py` selects a wrapper which invokes
   `UnsafeAllocationTracker(device).verify_before_replay(trace_id)` before the
   native replay call.
3. `ttnn/ttnn/unsafe_allocation_tracker.py` runs `gc.collect()`, then queries
   `ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(device, trace_id)`.
   A nonempty result raises `RuntimeError` before replay. Optional traceback
   diagnostics are controlled separately by `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`.
4. `tt_metal/impl/allocator/trace_allocation_tracker.cpp` tracks allocation IDs
   against each active trace, intersects them with currently allocated buffers,
   and retires freed IDs. Program-cache buffers are included unless explicitly
   disabled with `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1`.

The native query returns **buffer IDs and allocation contexts**, not physical
address ranges or the trace's address-write footprint. Its accounting is
conservative and temporal: a live ordinary buffer allocated after capture is
flagged for an older trace. Consequently, an allocator warning alone does not
prove a live overlap, and a tracker rejection requires inspecting the named
buffer and ownership. A successful guarded run establishes that no tracked
live unsafe allocation survived at its replay boundaries; it is not a general
physical address-overlap measurement.

The native unit tests in
`tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py` exercise
per-trace accounting, allocations between captures, retirement after GC, and
rejection of still-live buffers. They also cover explicit corruptible-buffer
acknowledgments. This probe adds no acknowledgments and does not suppress any
tracking category.

## Previous mechanisms and evidence

The supplied 35B predecessor's `tt/generator.py::_ensure_traces_replay_safe`
recaptures when the program-cache count or attached KV-cache identity changes.
Its docstring identifies `doc/full_model/logs/probe_bisect.py` as the historical
two-prefill-length corruption reproducer; that script is not present in the
available predecessor snapshot, so its contents were not inspected. No native
tracker-specific probe was found in that snapshot.

The current 9B model has earlier native-tracker evidence instead:
`doc/full_model/work_log.md` records `cache_contract_v1` with tracking enabled,
and `doc/full_model/AUTOFIX_norm_row_order.md` records tracked
`trace_contract.py`/batch-contract invocations. These are historical evidence
for earlier code stages, not validation of the new serving adapter.

Current `tt/generator.py::_ensure_replay_safe` recaptures after a program-cache
count change. This targets persistent new program binaries but does not itself
enumerate arbitrary live buffers created without program-cache growth. Native
accounting therefore supplies a distinct, stronger diagnostic for the adapter
request boundaries.

## Minimal probe addition

`tests/adapter_serving_device_probe.py` now accepts
`--require-trace-allocation-tracking`. It checks the actual
`ttnn.TRACE_ALLOC_TRACKING` import-time value before opening the mesh, rejects
program-cache skipping, and records the settings and mechanism in JSON. The
existing native wrapper checks **all actual model and sampler replays**,
including warmup/prefill paths, without a custom wrapper, fake trace list,
manually injected allocations, or runtime source change.

Run the existing reduced adapter probe in a separate supervising hardware job
with `TT_METAL_TRACE_ALLOC_TRACKING=1`,
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, and the new flag. Start without
tracebacks; if it rejects a buffer whose ownership is unclear, a subsequent
bounded diagnostic can enable traceback collection. Neither invocation needs
watcher or profiling. The optional flag leaves the default probe behavior
unchanged.

Tracking performs a full GC before each replay. The existing
`doc/optimized_full_model/AUTOTRIAGE_history.md` explains the resulting host
submission overhead even with `blocking=False`. This diagnostic must therefore
not supply performance measurements or replace the normal deferred-read
equivalence run. Its purpose is allocation ownership, and any failure must be
resolved or localized before claiming that ownership safe.

Static validation passed: AST parsing and every applicable pre-commit hook,
recorded in `adapter_probe_tracker_static_checks.log`. The authoring subagent
performed source checks only; the separate supervising result follows.

## Supervising hardware result

The supervising lane ran the guarded adapter probe with tracking1 and
program-cache skipping0. `adapter_device_tracker_v3.json`/`.log` record all
four cases passed and cleanup completed, exit0. The import-time native flag is
true. No tracked live unsafe allocation survived actual model/sampler replay
boundaries in these cases. This closes the observed allocator-warning audit
for the exercised reduced-adapter boundaries; it supplies no performance number
or blanket full-server guarantee. The generic allocator advisory is retained:
it warns about possible lifetimes, while the native guard checks the live
allocation IDs at replay. Neither output directly measures physical overlap.

The final source was rerun after the host-history and request-seed repairs in
`adapter_device_tracker_release.json` / `.log`: all four cases passed with
tracking enabled, program-cache allocations included, no acknowledgments,
and clean device closure at 2026-09-08 14:16:41 UTC (exit0). Its source hashes
match `full_b32_verified.command.json`. Performance servers leave this
diagnostic disabled.

The final startup-only repair also passed the native guard with program-cache
allocations included: `b1_startup_after_tracker.json` exercises actual warmup and
first admission; `adapter_device_tracker_startup_final.json` repeats the stale-token,
current-position, page-table and remap cases. Both close devices cleanly and
record the final adapter hash881b38abae6ff278d3f66212e2ab4d8a462ce4ebd42153f65c5497b1d97ad324.
