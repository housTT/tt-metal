# AutoDebug: coexisting outer-product traces

Source-only investigation, 2026-09-04. No device access, TTNN import, or implementation edits. Applied AutoFix, AutoDebug, and the trace-debugging guidance. The five relevant current Python files match their hashes in `logs/reuse_outer_matched_v1.provenance.json`.

## Finding

**The leading hypothesis is that the first trace overwrites persistent allocations belonging to the second decoder.** The allocation-order hazard is established by source; an actual overlapping address/corrupted buffer has not yet been measured. Do not attribute the failure to the reuse kernel before running the controls below.

`logs/reuse_outer_matched_v1.log` records eager comparison passing and final stress PCC **0.25064918168434985**, against 0.995. It also records the allocator's live-trace allocation warning at **23:15:36.174**, between the two decoder setup/prefill sequences. No replay output was checked before the final window, so this does **not** establish that the failure begins at replay 64. The post-stress one-replay checks were never reached.

Reproduction recorded in provenance, with `OMP_NUM_THREADS=8`, `ORNITH_WEIGHTS=real`, and the existing task environment:

```bash
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py -k reuse_outer_matched_timing -x -v -s
```

## Ranked hypotheses and controls

### 1. Persistent buffers allocated after an older trace alias its intermediates — high confidence

Evidence:

- `tests/test_fusion_equivalence.py:132–148` completely constructs, warms, and captures runtime A before constructing decoder B. `handles` keeps B's weights/state/input alive when A replays at lines 155–162.
- `tests/test_functional_decoder.py:167–179` allocates the model weights and recurrent/convolution state. `tt/fused_decoder.py:67–74` explicitly uploads weights to device DRAM; `tests/test_fusion_equivalence.py:136,140` also allocates B's token and position tensors after A's capture.
- `tt_metal/impl/allocator/allocator.hpp:174–175` explicitly states that trace-used memory is not tracked by the allocator once capture completes. `allocator.cpp:118–143` warns about this condition but continues allocating. `tt_metal/distributed/mesh_device.cpp:1505–1508` registers the completed trace for subsequent allocation accounting.
- `tt/fused_decoder.py:617–637` and `tests/transpose_fusion_candidates.py:45–72` explicitly deallocate intermediate tensors. Those addresses remain in recorded commands while becoming available to later allocations.
- `tests/ttnn/unit_tests/base_functionality/test_single_device_trace.py:165–198` tests this exact ordering: an allocation between captures is unsafe for trace A, but safe for later trace B.

Prediction: replaying A can change B's token, weights, or state even before B executes. Restoring B's recurrent/convolution buffers cannot repair corrupted weights or its token. Eager agreement before any A replay is compatible with this mechanism.

Smallest controls, run separately:

1. Replace B's class with **the identical `FusedDecoder`**, retaining the original allocation/capture order. Check each output after one replay before long timing. Failure removes native transpose as a necessary cause. Passing would not exclude an allocation-sensitive overlap.
2. Snapshot B's token and small persistent weights (for example convolution taps/norm weights), and optionally recurrent/convolution state, to host. Execute **only A once**, then compare B's buffers exactly. Any change to an unconsumed B tensor proves cross-trace corruption directly. Record addresses/IDs for changed tensors.
3. Change only setup ordering: construct **both** decoders, upload all persistent inputs, run both prefills and eager warmups, take host state snapshots, and retain all handles **before either capture**. Then capture both and run the original 64-replay/15-measured-window checks unchanged. Check one-replay eager equality first, and preserve final state checks. A pass alongside control 1 or 2 failure verifies the harness repair.

Optional diagnostic: start a separate run with `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1`; query `ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh_device, trace)` before replay. This tracker is conservative temporal accounting, not proof of physical address overlap. Later captured output/scratch and dynamic trace storage may also be flagged; inspect ownership instead of broadly acknowledging every allocation as safe.

Both capture inputs are already retained at `test_fusion_equivalence.py:148`; retaining them again does not repair their allocation timing. Keeping Python references to intermediates also cannot defeat the explicit `ttnn.deallocate` calls. Serializing the complete capture/replay/release lifecycle is a useful control, but changes the intended matched timing arrangement.

### 2. Divergence specific to longer recurrent stress or reuse execution — possible, currently weaker

The previous `reuse_outer_whole_v1.log` reports seven passing pair/core tests, including 32-replay stress PCC **0.9999118369528625**. That paired harness releases its first trace before building the next decoder (`test_fusion_equivalence.py:239–257`). The new test changes both coexistence and replay length, so prior success does not prove correctness at 64 replays.

If hypothesis 1 is refuted or repaired and failure remains, run one decoder/trace at a time for 1, 2, 4, 8, 16, 32, and 64 replays from the same saved state, releasing the trace before creating the next decoder. Compare each trace against its own eager N-step execution, then compare runtime and candidate output/state. Allocate everything required by each trace before capture. First divergence only in candidate traced execution implicates replay/kernel behavior; divergence also in candidate eager execution implicates arithmetic/recurrence instead. Localize recurrent state and the outer product before modifying dtype or program configuration.

### 3. Restore, input lifetime, or asynchronous ordering defect — low support

`_snapshot_state` deliberately returns host copies (`test_functional_decoder.py:783–796`). `_restore_state` constructs host-only tensors and copies into existing device state (`799–811`); it does not allocate replacement device state. Both variants update recurrent and convolution state in place (`tt/fused_decoder.py:586–588,622–635`; candidate `49–70`). Both traces use CQ 0, with synchronization around each timed window (`test_fusion_equivalence.py:157–162`). No missing restore buffer or asynchronous overlap is evident here.

If needed after the first two controls, compare restore readback exactly against the host snapshot and log state buffer addresses before/after restore. A blocking-per-replay control isolates queue behavior, but its timing is not comparable to the intended asynchronous window measurement.

## Disposition

Report precedes any fix. The first control should isolate harness behavior using two identical runtime decoders, followed by allocation-before-capture ordering. Keep the original correctness gates and candidate implementation unchanged until evidence identifies the failing boundary. No performance conclusion follows from the failed matched run.
