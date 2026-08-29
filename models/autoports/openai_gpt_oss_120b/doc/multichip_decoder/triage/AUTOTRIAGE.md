# AUTOTRIAGE

## Diagnosis

- The most-supported cause is the acceptance-test harness switching from work on a fabric-enabled `(1,4)` parent mesh to work on an overlapping `(1,1)` child mesh without the required parent `quiesce_devices()` phase barrier. The first observable stop is therefore a dispatch/device stall on physical device 3, the sole device in the baseline submesh, during the optimized baseline prefill. It is not evidence of a tensor-parallel attention or expert collective bug: no multichip forward had run.
- Ranked hypotheses:
  1. **High:** missing parent/overlapping-submesh quiescence in the test harness.
  2. **Medium-low:** an independent device-3/UMD fault. Triage could no longer read device 3, but the fault appeared on exactly the device exposed to the unsafe overlapping phase switch; the other links, heartbeats, and ARC checks were healthy.
  3. **Low:** a single-chip optimized MoE/gather kernel bug. The same baseline passed on a standalone `(1,1)` mesh, and Inspector records programs after the RM gather through the final slice; the capture does not place a core in the gather kernel.
  4. **Ruled out by this capture:** TP attention/expert CCL protocol or geometry. Only construction ran on the parent.

## Triage Evidence

- Inspector records mesh 0 as `(1,4)` over physical devices `[3,2,1,0]`, then mesh 1 as its `(1,1)` child over device `[3]`. Thus the unreadable device is precisely the baseline device, not an arbitrary peer.
- Parent workloads 0--8 target all four coordinates. The next workload, 9, targets only mesh 1, followed by the rest of the single-device optimized graph. This matches the source order: construct TP=4, then construct/run the child-mesh baseline.
- The host timeout was at `ttnn.synchronize_device(baseline_mesh)` after `baseline.prefill_forward`. Triage then timed out reading device 3 at a TENSIX register. Device 3 consequently has no call stack; an exact worker kernel/PC cannot be recovered from this capture.
- `dump_op_mesh` labels all devices idle, `dump_running_operations`, CB, binary-integrity, core-magic, lightweight-assert, and watcher-ring checks pass. On readable devices 0--2, `cq_dispatch` is waiting for upstream pages and fabric ERISCs are in their service loops. Ethernet links are up with zero retrains and live heartbeats; ARC is responsive.
- The ERISC NoC-counter warnings and router call stacks are broad fabric state after the timeout, not a proved originating send. Likewise, the RM gather compilation line is not a stop-site: Inspector records later typecast, untilize, reshape, post-combine-reduce, and slice programs as compiled/committed. `Committed` proves enqueue/binary state, not device completion.

## Source Evidence

- `tests/test_multichip_decoder.py:210-230` creates an overlapping `(1,1)` child, constructs `MultichipDecoder` on the parent, and immediately constructs `OptimizedDecoder` on the child. There is no parent quiescence between those phases. Lines 273--290 then execute the child prefill before the first parent forward. Later the test alternates child and parent trace capture/replay again, also without parent phase barriers.
- The MeshDevice API contract in `tt_metal/api/tt-metalium/mesh_device.hpp:296-305` explicitly requires `quiesce_devices()` as the barrier between phases using overlapping mesh views. The implementation in `tt_metal/distributed/mesh_device.cpp:1763-1785` drains child and parent queues and resets shared launch-message/GO-mailbox state. The focused runtime test in `tests/tt_metal/distributed/test_mesh_workload.cpp:425-442` quiesces the parent at every child/parent transition.
- The common fixture already has the correct topology rule for TP=2 on a four-device host: `models/common/tests/conftest.py:187-203,255-274` opens the full system parent and returns a `(1,2)` submesh. A direct two-device fabric open is not the repair.

## Downstream Effects

- Device 3 becoming unreadable prevents triage from identifying its worker stop-site. Dispatch waits on devices 0--2, fabric-router call stacks/counter warnings, and teardown failures are fanout after the baseline-side stall. They do not implicate `multichip_decoder.py` collectives.
- The evidence does not support changing an RM gather kernel, a fabric route, CCL counts, tensor-plan geometry, or TP decoder code.

## Proposed Fix

- Smallest discriminating isolation test: call `mesh_device.quiesce_devices()` immediately after TP construction and before constructing or enqueueing anything on `baseline_mesh`; quiesce the parent again before switching back to the TP mesh. If the baseline prefill then completes, it confirms the overlapping-mesh phase contract as the cause. A complete same-process test would need such a parent barrier at every child/parent transition, including trace capture, replay, timing, and release.
- Smallest robust acceptance-harness fix: run `OptimizedDecoder` in a separate standalone `(1,1)`, fabric-disabled process and save a host artifact keyed by checkpoint revision, layer, policy, seeds, shapes, and page table. Store baseline prefill/decode outputs, the compared K/V cache block, and warmed decode latency. Close that process/mesh before the TP test loads the artifact. The TP=2 target must continue through the common fixture as a `(1,2)` child of the opened four-device parent; TP=4 uses the full parent. This removes overlapping live model/trace state and needs no `multichip_decoder.py` change.

## Uncertainty

- Process separation is the safest fix, but this capture does not prove it is the only legal fix: the runtime explicitly supports overlapping mesh views when parent quiescence separates phases. The proposed one-barrier isolation run distinguishes those cases.
- Because device 3 was unreadable, no exact optimized-op kernel deadlock is proved. If a quiesced or standalone `(1,1)` baseline still hangs, capture device 3 before teardown and compare the same baseline on another physical device; only then investigate the resolved worker kernel.
- This diagnosis is read-only and no hardware command was run.
