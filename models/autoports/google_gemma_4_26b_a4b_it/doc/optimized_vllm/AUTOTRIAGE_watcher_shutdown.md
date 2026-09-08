# AUTOTRIAGE: watcher failure at fabric shutdown

## Diagnosis

The combined TP1/TP2/TP4 watcher process failed a real fabric-router
kernel-return postcondition: a subordinate active ERISC observed a nonzero
write/atomic NoC packet-tag register after its router kernel returned. The
three adapter test bodies passed, but the process aborted with exit 134 during
final cleanup; this is **not a complete watcher pass**.

The source contains a concrete candidate cause: router teardown drains
transactions and resets software counters but does not restore the sticky
packet tags required by the firmware wrapper. The coordinating agent's
isolated TP4 rerun after reset also failed final teardown, so prior-profile
topology transitions are **not required** for the cleanup failure. Its watcher
detached before the later heartbeat timeout, so the second run does not capture
the tag assertion directly. Together, the first run's assertion and the absent
router tag reset justify an isolated repair experiment at the router handoff
boundary, preserving assertions. They do not yet prove that this candidate fix
resolves every observed heartbeat failure.

## Triage evidence

All artifact paths below are relative to this directory.

- `trace_reuse_probe/final_watcher.log:897` reports three passes in 283.90
  seconds. The three JSON files under `trace_reuse_probe/final_watcher/` record
  `verdict: pass`. Their scope is the adapter assertions and comparisons before
  process cleanup; they do not certify the overall process lifetime.
- At 20:18:27.554 UTC on 2026-09-08, the next watcher poll reports device 0,
  active Ethernet logical core `(0,9)`, virtual core `(29,25)`, processor
  `subordinate_erisc`, assert class
  `DebugAssertNCriscNOCPacketTagClearedTripped`. Both reported current kernels
  are `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`.
- Despite the message's wording, “before starting the next kernel,” source
  places this assertion **after the current `kernel_main()` returns**. It is a
  handoff postcondition, not evidence that an additional model kernel started.
- The watcher used interval 10 seconds, `NOINLINE=1`, and disabled features
  exactly `WAYPOINT`; the JSON records the same settings. Assertions and NoC
  sanitization remained enabled. Last-waypoint values `X` are expected under
  this coverage choice and cannot locate the device instruction.
- The previous poll at 20:18:17 checked all devices without a reported error.
  Detection after pytest's summary does not establish that the assertion
  originally tripped after that summary: polling adds up to one interval of
  latency. The source nevertheless locates the failing contract at router
  return, which is consistent with mesh/fabric shutdown.
- `triage/watcher_shutdown/watcher.log.xz` (read with `xz -dc`) preserves earlier TP4 watcher state.
  The most recent retained row for device 0/core `(29,25)` shows
  `rmsg:H0G|EE`, subordinate message `G`, and kernel IDs `14|18`. The corresponding
  entries in `kernel_names.txt` are router kernels; `kernel_elf_paths.txt`
  identifies ID 14's `active_erisc.elf` and ID 18's
  `subordinate_active_erisc.elf`. These are pre-failure snapshots, not register
  contents at the asserting instruction.
- The focused tt-triage capture failed with exit 1.
  `triage/watcher_shutdown/capture.log` says inspector RPC/log acquisition
  failed because `/tmp/tt-metal/inspector` did not exist. Therefore no live
  call stacks, exact command-buffer index, packet-tag value, or peer-core state
  at the assert was captured. Do not invent that missing ledger detail.

The log also records actual same-process configuration transitions:
`DISABLED` → `FABRIC_2D` at 20:15:35.714 and `FABRIC_2D` → `FABRIC_1D_RING` at
20:16:56.987. The final error is detected approximately 90 seconds after the
last transition, following successful TP4 model execution. Timing alone
cannot label it a harmless transition artifact.

## Source evidence and state ledger

| Resource/boundary | Producer or owner | Consumer/required state | Observed or inferred gap |
| --- | --- | --- | --- |
| Sticky NoC packet tag | `noc_async_write_one_packet_with_trid()` in `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2454` calls `ncrisc_noc_fast_write<..., true, ...>()`; Blackhole `noc_nonblocking_api.h:555` writes `NOC_PACKET_TAG_TRANSACTION_ID(trid)`. Router packet transmission uses these APIs. | Later implicit-TRID writes inherit that command buffer's tag unless it is explicitly changed. | Nonzero tags can remain after a transaction completes. Completion is not a tag reset. |
| Tracked router writes | Router receiver channels allocate/reuse TRIDs in `fabric_erisc_router_transaction_id_tracker.hpp`. Its `all_buffer_slot_transactions_acked()` at line 109 iterates the TRID range and waits on local/downstream NoCs. | Router teardown must wait for all issued traffic before changing shared state. | Existing wait is present; do not propose adding it as a missing fix. Exact last TRID is not captured. |
| TRID barrier | `dataflow_api.h:2603` polls `ncrisc_noc_nonposted_write_with_transaction_id_flushed()`. | Ack counters establish completion. | This function does not zero packet-tag registers. |
| Router return | `fabric_erisc_router.cpp:2879` synchronizes local ERISCs, waits tracked writes, resets counters, sends final termination notification, executes write/atomic barriers, synchronizes again, and marks `TERMINATED`. `kernel_main()` calls this at line 3800 before returning. | Firmware wrapper expects a safe handoff after return. | Neither this teardown nor the subsequent return path clears packet tags. |
| Firmware postcondition | `tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc:42` checks four transaction-completion predicates after `kernel_main()`. Line 51 then checks `ncrisc_noc_packet_tags_cleared(NOC_INDEX)`. | Blackhole `noc_nonblocking_api.h:255` requires `NCRISC_WR_CMD_BUF`, `NCRISC_WR_REG_CMD_BUF`, and `NCRISC_AT_CMD_BUF` packet tags all equal zero. | The retained error proves at least one of those three tag reads was nonzero for the subordinate's selected NoC. It does not identify which one. |
| Neighboring kernel precedent | `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:272` closes its connection, performs `noc_async_full_barrier()`, then `noc_clear_packet_tags(noc_index)` before reporting termination. | Restore neutral packet-tag state after all owned traffic drains. | The router lacks an analogous explicit restoration. This is supporting source evidence, not proof that copying the mux's NoC ownership choice is sufficient. |

`active_erisck.cc` checks pending reads, nonposted writes, nonposted atomics,
and posted writes before checking tags. The reported tag class therefore
supports a stale-register handoff problem more directly than a missing generic
traffic barrier. Do not infer that every fabric route/NoC is idle from those
four predicates alone: they check the wrapper's selected NoC and the kernel's
counter conventions.

The fixture also performs explicit cleanup: `conftest.py:670` yields the mesh,
then closes submeshes, closes the owning mesh, and resets fabric configuration.
The model adapter's `teardown()` runs in the test's `finally`. This is a direct
pytest fixture path; the vLLM worker's earlier missing `shutdown()` hook is not
the cleanup mechanism under test here.

## Downstream effects

Watcher maps the assertion enum to the message in
`tt_metal/impl/debug/debug_helpers.hpp:248`. Its device reader throws on the
tripped state at `watcher_device_reader.cpp:866`; the watcher thread's uncaught
exception produces process abort. The ERISC assertion handler
(`tt_metal/hw/inc/api/debug/assert.h:118`) writes its launch message done and
disables the ERISC app, then the subordinate spins because it has no direct
base-firmware exit. A later board heartbeat problem can follow this state; it
does not replace the packet-tag assertion as the first observed fault.

## Smallest verify/refute experiment

The coordinating agent owns recovery and device execution. After preserving
this incident and resetting/listing the devices, run only the TP4 node in a
fresh process. Keep the same allocator checks, watcher interval, noinline
setting, and waypoint-only omission. Preserve a separate output directory.
For example, from the repository root:

```bash
env GEMMA4_VLLM_TRACE_REUSE_PROBE=1 \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_WATCHER_DISABLE_WAYPOINT=1 \
  GEMMA4_VLLM_TRACE_REUSE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/trace_reuse_probe/isolated_tp4_watcher \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_trace_reuse.py::test_adapter_request_trace_reuse[blackhole-p150x4]'
```

Keep other environment settings identical to the failing run and ensure no
additional watcher-disable variables are inherited. Do not disable assertions,
NoC validation, or Ethernet watcher instrumentation. The command is proposed
here, not executed by this investigator.

Require the complete process to exit 0, a clean close/detach sequence, and no
post-summary fatal marker. A JSON pass or pytest pass line alone is inadequate.

- **Same tag assertion in isolated TP4:** earlier profile transitions are not
  required. Investigate and fix the router's final packet-tag ownership/reset
  boundary, with the assertion still enabled. Capture the relevant tag/NoC
  owner before modifying source if possible.
- **Isolated TP4 exits cleanly:** the combined-process sequence is implicated,
  but one pass does not prove absence of a race. Next isolate TP2 → TP4 with
  the same watcher settings to localize the lifecycle interaction. Do not
  dismiss the original process failure or patch router code solely from the
  standalone pass.

## Proposed fix if reproduced and remaining uncertainty

If isolation reproduces the packet-tag postcondition failure, the smallest
candidate is to restore the router-owned write/atomic command-buffer tags to
zero **after final traffic drains and the peer ERISC is synchronized, before
returning to the firmware wrapper**. Verify the selected NoC and command-buffer
ownership for each ERISC first; blindly clearing both NoCs could disturb a
still-running peer. Existing helpers already provide explicit packet-tag
clearing. Adding a barrier alone does not reset the sticky register, and
removing the assertion would hide the violated contract.

No source edit is made in this report. Exact bad tag values and peer ownership
at the failure were not captured. A transition-independent **cleanup failure**
is now reproduced, as detailed below; direct packet-tag-state confirmation in
that second run is unavailable. Any eventual C++/kernel fix must be compiled under the repository
build instructions, retain the 26,624-byte ACTIVE_ETH size gate, and pass the
isolated and original combined watcher process lifetimes before being called
verified. No performance claim is made.

## Isolated TP4 result and watcher shutdown ordering

The coordinating agent reset/listed devices and completed its ring-mesh smoke,
all exit 0, then ran TP4 only with the same source and watcher settings.
`trace_reuse_probe/isolated_watcher_tp4.log:233` reports one functional pass in
32.35 seconds. Watcher stops at 20:22:28 UTC. At 20:22:48.226, device 0/core
`(29,25)` times out waiting for base-firmware heartbeat during
`RiscFirmwareInitializer::teardown()`; the process self-aborts with exit 134.
The coordinating agent reports no remaining live process. This is a failed
isolated process lifetime, not a clean standalone pass.

The source explains why a stopped watcher can change the visible failure:

1. `FabricFirmwareInitializer::teardown()` at
   `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:405` sends
   termination signals to the master routers. This function does not wait for
   their final firmware-wrapper postconditions before returning.
2. `MetalContext::teardown()` at
   `tt_metal/impl/context/metal_context.cpp:354` detaches and destroys watcher
   **before** calling the RISC firmware teardown at line 359.
3. `WatcherServer::Impl::detach_devices()` at
   `tt_metal/impl/debug/watcher_server.cpp:146` requests stop and joins the
   thread. It performs no mandatory final assert dump. The poll loop at
   line 606 wakes on the stop request and exits without another poll.
4. RISC firmware teardown then reaches
   `assert_active_ethernet_cores_to_reset()` →
   `return_to_base_firmware_and_wait_for_heartbeat()`. A subordinate that has
   already taken the tag assertion can remain in the assertion handler's
   terminal loop. Whether the main ERISC can return to base firmware in that
   state depends on the local handoff protocol. The later heartbeat timeout
   is consistent with that failed handoff, but the second run does not retain
   the assertion mailbox needed to prove this exact intermediate state.

Therefore the difference between “watcher caught packet-tag assert” and
“watcher stopped, then heartbeat timeout” is plausibly detection timing, not
evidence that the first assertion was a harmless multi-profile artifact. A
single future exit-0 run would also need care: shutdown can stop polling before
a final device postcondition becomes visible.

Proceed with one focused candidate only: after the router's final tracked-write
drain, ordinary write/atomic barriers, and last dual-ERISC synchronization,
restore its owned packet-tag registers before `kernel_main()` returns. Determine
the compiled per-ERISC NoC/command-buffer ownership from the retained router
specializations; preserve the wrapper assertion so it verifies the repaired
contract. Do not clear tags while a peer might still program them, and do not
weaken watcher coverage or change polling to mask the error. The existing mux
barrier-then-tag-clear sequence is a concrete precedent for this boundary.

Validation must include the full isolated TP4 process lifetime, the original
combined TP1/TP2/TP4 run, and immediate same-profile reopen without reset. If
the candidate does not fix the final heartbeat timeout, retain the original
assertion finding but refute its sufficiency as the sole cleanup cause. A
bounded model-free ring open/close control can then separate traffic-dependent
tag state from empty-fabric lifecycle behavior; a clean empty-mesh control
alone would not refute a tag left by real router traffic.

## Retained failing specialization: owned-NoC scope

The saved `kernel_elf_paths.txt` maps failing subordinate kernel ID 18 to
router specialization `12773088658352305445` beneath cache root
`/home/hous/.cache/tt-metal-cache/5507830578873079285/kernels/fabric_erisc_router/`.
Its retained `named_ct_arg_map_generated.h` records:

| Compile-time argument | Value |
| --- | ---: |
| `MY_ERISC_ID` | 1 |
| `NUM_ACTIVE_ERISCS` | 2 |
| `IS_RECEIVER_CHANNEL_0_SERVICED` | 1 |
| `IS_SENDER_CHANNEL_0_SERVICED` / `_1_` | 0 / 0 |
| `RX_CH_0_LOCAL_WRITE_NOC_ID` / `RX_CH_0_FWD_NOC_ID` | 1 / 1 |
| `RX_CH_0_LOCAL_WRITE_CMD_BUF_ID` / `RX_CH_0_FWD_DATA_CMD_BUF_ID` | 0 / 2 |
| `RX_CH_0_FWD_SYNC_CMD_BUF_ID` | 1 |
| `IS_2D_FABRIC` / `UDM_MODE` | 0 / 0 |

`tt_metal/fabric/erisc_datamover_builder.cpp:203` configures ERISC0 for NoC0
and ERISC1 for NoC1. The single-ERISC forced-NoC1 override is a separate path.
`edm_fabric_utils.hpp:13` sets local-chip writes to NoC1;
`fabric_erisc_router_ct_args.hpp:604` derives downstream NoC from the retained
forwarding argument, also NoC1 here. The Blackhole worker handshake uses
`noc_index` at line 606. Thus for this observed subordinate, clearing
`noc_clear_packet_tags(noc_index)` after the final drain/synchronization covers
its own NoC1 handoff, including local and forwarding write command buffers,
while avoiding an all-NoCs clear that would also touch its peer's NoC0.

The bounded candidate is therefore one router-kernel cleanup call to
`noc_clear_packet_tags(noc_index)` after the last dual-ERISC synchronization and
write/atomic barriers, before kernel return. Keep all firmware assertions.
Expected verification: the isolated TP4 workload still satisfies every adapter
check, its unchanged packet-tag postcondition passes, the process closes
without heartbeat timeout and exits 0, and the original combined run plus
immediate reopen also pass. Other fabric specializations still require review
and testing; this ledger proves the proposed scope for the retained failing
specialization rather than claiming every configuration uses only one NoC.

## Resolution: router cleanup verified on 2026-09-08

This update supersedes the earlier pending repair/verification status while
preserving the investigation chronology. The bounded router fix now clears
`noc_clear_packet_tags(noc_index)` after final barriers and local-ERISC
synchronization, before `TERMINATED`. All assertions and sanitizers remain.
[kernel_cleanup_experiment.md](kernel_cleanup_experiment.md) retains the exact
three-line diff, ownership review, and full verification ledger.

- Isolated TP4 now exits 0 after one pass in 36.96 s and complete device close;
  [log](trace_reuse_probe/router_fix_tp4.log). The previous isolated run passed
  its body but aborted with a heartbeat timeout and exit 134.
- Combined TP1/TP2/TP4 now exits 0 after three passes in 101.48 s, with UMD
  close and cluster destruction complete at 20:31:02.819 UTC;
  [log](trace_reuse_probe/final_router_fix.log),
  [JUnit](trace_reuse_probe/final_router_fix.xml). Per-profile JSONs retain
  allocation tracking, watcher interval 10, noinline and waypoint-only omission.
- The next separate process opens/closes a 2x2 `FABRIC_2D` parent with a 1x2
  model submesh **without reset**, returns exit 0, and records
  `MESH_REOPEN_WITHOUT_RESET_OK`; [log](trace_reuse_probe/router_fix_reopen.log).

The coordinating agent observed all process exit codes. Runtime JIT and kernel
lint pass. The mandated repository build wrapper was attempted but Docker
socket access is denied; host compilation remains unverified for that reason.
These results verify the reproduced router cleanup repair and reopen boundary,
not every firmware failure or the separate real-server `TTWorker.shutdown()`
lifecycle proof, which remains pending.
