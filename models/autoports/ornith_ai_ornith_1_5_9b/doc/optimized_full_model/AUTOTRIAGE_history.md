# AUTOTRIAGE: history generation contract v3

## Diagnosis

The reported `plus_one` hang was not a continuous device-kernel hang: before
triage, the watcher repeatedly showed completed kernels and later host operation
IDs through 203 seconds. The strongest source explanation for the original
high-CPU delay is allocation tracking's unconditional full Python garbage
collection before **each** trace replay. That timing attribution remains a
hypothesis until a host stack or measured GC time confirms it.

The new freeze after the first capture is explained directly by this checkout's
triage implementation: it intentionally leaves Blackhole/Wormhole RISC cores
halted. `tools/triage/triage.py:918–972` patches `cont()` and
`continue_without_debug()` to no-ops to avoid a documented HALT/read/CONTINUE
hardware bug. The post-capture CCL/dispatch waits are therefore not evidence of
an output-history implementation bug. Waiting cannot normally recover cores
that this tool intentionally does not resume.

## Triage Evidence

Workload: PID 91123, native TP4 on four Blackhole chips on P300c boards, real
layers 0 and 3, context 2048. Exact launch and source/binary hashes are in
`logs/generation_contract_v3.provenance.json`; source snapshot is in the paired
`.sources.json.gz`. Command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.generation_contract \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/generation_contract_v3.json
```

Both captures completed with exit 0. Commands ran from the repository root;
`STAGE` below denotes `models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model`:

```bash
timeout 180 python_env/bin/python tools/tt-triage.py --llm-output \
  --llm-output-path "$STAGE/triage/history_v3_tt-triage.txt" \
  --triage-summary-path "$STAGE/triage/history_v3_triage-summary.txt"
timeout 120 python_env/bin/python tools/tt-triage.py --llm-output \
  --run=dump_callstacks --run=dump_running_operations --run=check_eth_status \
  --llm-output-path "$STAGE/triage/history_v3_focused_tt-triage.txt" \
  --triage-summary-path "$STAGE/triage/history_v3_focused_summary.txt"
```

Console output is retained in `triage/history_v3_capture.log` and
`triage/history_v3_focused_capture.log`; final writes were respectively
2026-09-05 18:52:58.729 UTC and 18:54:43.468 UTC. No dependencies were installed.

Direct observations:

- Before capture, PID 91123 consumed approximately one CPU core. At 201 seconds
  process age it was running; its main thread later stopped at 207.50 CPU seconds
  and waited in a futex while thread 91176 consumed CPU. Stack tools (`gdb`,
  `py-spy`, `perf`, `strace`, `pstack`) were unavailable; `/proc/.../stack` access
  was denied. This is scheduling evidence, not a resolved native/Python stack.
- The watcher records host IDs 777 / 710 / 742 repeatedly changing through
  203.191 seconds. `plus_one` kernel 644 at host ID 777 has `rmsg:D1D|bnt`,
  `smsg:DDDD`, and `GW/W` waypoints: the kernel completed and the worker waits
  for more work. At 163.065 seconds the last ID is 710, at 173.096 it is 777,
  at 193.159 it is 710, and at 203.191 it is 742. A static interpretation of
  kernel name 644 is directly refuted. The extracted timeline and watcher
  snapshot are `triage/history_v3_watcher_timeline.json` and
  `triage/history_v3_watcher.log`.
- The initial triage operation mesh is idle on all four chips and its running
  operation table is empty. No model-worker callstack identifies a history
  kernel wait. Ethernet links are up, heartbeats true, retrain counts zero;
  ARC heartbeats are approximately 10/s; DRAM has no recorded corrected or
  uncorrected errors.
- The full report contains binary-integrity mismatches on device 0, unused ETH
  software/hardware NoC-counter mismatches, and many cores reported as having
  escaped the tool's halt. Some dispatch variables cannot be read because the
  core is not halted. The summary file says script `pass` even where the full
  report has `fail`; it is not a clean hardware verdict. The live changing
  snapshot and halt behavior prevent attributing these anomalies to model code.
- Starting at watcher 213.222 seconds, device 2 core (0,0) remains at
  `D1T`/host ID 742 while device 3 core (0,0) remains at `tt_fabric_mux` 250,
  host ID 595, `NWID`. This stable state begins during the first capture.
  `T` means replay-trace signal, not a plus-one kernel assert.
- The focused capture identifies device 3 `AllGatherAsyncDeviceOperation`
  ID 595, trace 3, BF16 local input `[1,1,1,1024]`, after tilize ID 594.
  Four writers at logical (1,0), (2,0), (4,0), (5,0) wait in
  `minimal_default_writer.cpp:282`, `noc_semaphore_wait_min` (`NSMW`). Their
  readers wait at `minimal_default_reader.cpp:127`, `cb_reserve_back` (`CRBW`).
  Mux cores (0,0), (3,0) are in ordinary forwarding/connection checks; `NWID`
  alone does not establish a blocked NoC write. Device 2 dispatch waits in
  `cq_dispatch.cpp:1120` and prefetch waits for dispatcher pages. Devices 0/1
  dispatch waits for new host work. Device 2 operation labels refer to sampler
  trace 4 and cannot establish a model defect in the halted snapshot.

## Source Evidence

### Host scheduling contract

`ttnn/ttnn/__init__.py:154–162` replaces `execute_trace` when
`TT_METAL_TRACE_ALLOC_TRACKING=1`, calling
`UnsafeAllocationTracker.verify_before_replay` before enqueueing.
`ttnn/ttnn/unsafe_allocation_tracker.py:77` unconditionally calls `gc.collect()`
for every verification, even when no unsafe allocation remains. Thus `blocking=False`
still has synchronous host work before submission. Turning off only traceback
capture does not remove this GC.

`tt/generator.py:443–446` submits the model trace and one sampler trace per
decode step. The five paired collector/control requests require
`2 * ((8-1)+(128-1)+(260-1)+(128-1)+(8-1)) = 1054` decode steps and therefore
2108 full GC calls. The final device-output API check adds one `decode_forward`,
six `replay_decode` steps, and seven control steps: total 1068 decode steps and
2136 full GC calls if the entire harness completes. This is the intended count,
not proof that the interrupted process completed that many steps.

Python traceback formatting occurs at allocation boundaries in
`ttnn/ttnn/decorators.py:532–559`; it may add startup/recapture cost. The repeated
full GC path is a more direct explanation for steady-state CPU activity. Actual
per-GC latency was not measured in the target process.

### Post-capture wait ledger

| Resource or transition | Producer | Consumer / required contract | Observed state |
| --- | --- | --- | --- |
| Ring entry barrier | Each eligible same-direction remote writer sends one multicast atomic increment | Every writer waits for its configured `barrier_target_count`, then resets its local semaphore | Device 3 writers waiting at line 282 |
| Output CB pages | Reader reserves a packet, fills local input tiles, then pushes a packet | Writer must pass barrier and eventually pop pages for capacity to return | Device 3 reader reserve is downstream of blocked writer |
| Mux forwarding | Worker connections and packet slots feed mux | Mux forwards available data while checking connections | Active forwarding loop; no proven send-credit defect |
| Worker completion/replay reset | Worker firmware processes GO/replay and notifies dispatch | Dispatch waits for its expected completion counter; prefetch needs returned dispatch pages | Device 2/3 dispatch and prefetch backpressure |

The ring writer protocol is present in
`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp:235–284`.
The reader reserve/read/push protocol is at the adjacent
`minimal_default_reader.cpp:120–140`. Numeric barrier values were not captured;
no producer-count mismatch is claimed. The trace replay firmware transition is
`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:398–425`: it clears the replay signal and
notifies dispatch. A core intentionally halted during this transition cannot
fulfill its ordinary producer obligation.

The causal boundary is the triage tool: `_patch_risc_debug()` is called by every
`_init_ttexalens` path and explicitly never continues affected cores.
`tools/triage/check_broken_components.py:98–115` further confirms that cores are
expected to remain halted at tool completion. Do not add CCL acknowledgements,
change its barrier counts, or modify the history collector based on these
post-capture waiters.

## Downstream Effects

The initial host work leaves workers idle between completed traces and makes a
silent, long contract test look stalled. The later diagnostic halt stops
producers and creates real dispatch/CCL/host-completion waits. These are two
different conditions. Neither proves a `plus_one`, `indexed_fill`, or history
copy defect. The prior retained-scratch failure was separately diagnosed and
already fixed with `ttnn.deallocate(updated)` inside `_append_output_history`;
that fix is present and is not proposed again here.

## Proposed Fix

No implementation change is justified by this evidence. Before rerunning, the
hardware owner should preserve evidence, stop only the task-owned workload, and
use the bounded reset/health/mesh-smoke sequence from `$tt-device-usage`.
Do not manually continue halted cores against the tool's explicit hardware
workaround. The investigator handed hardware ownership back after the focused
capture; it did not kill processes, reset devices, clear locks, or open new work.

For the next contract run, add flushed per-case progress and a timed Python
stack dump before considering another destructive live device capture. Time
GC with `gc.callbacks` or a host stack to verify the overhead hypothesis while
keeping the tracker enabled. A shorter watcher/tracker contract plus the full
long parity contract and separate uninstrumented performance evidence can
separate safety checks from benchmark overhead without suppressing tracking
inside implementation code. Preserve all original parity and sampling checks.

## Uncertainty

The exact original host CPU stack and per-GC timings are unavailable. The live
run was interrupted by diagnostic core halts before a final parity report;
it has no passing correctness or performance result. The original delay is
consistent with repeated GC and demonstrable forward progress, while the
post-capture no-resume mechanism is source-proven. Recovery and a fresh contract
run remain owned by the parent; this report does not claim they ran.
