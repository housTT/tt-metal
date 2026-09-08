# Router packet-tag cleanup experiment

## Hypothesis and evidence

This is the isolated AutoFix candidate from
[AUTOTRIAGE_watcher_shutdown.md](AUTOTRIAGE_watcher_shutdown.md). The combined
TP1/TP2/TP4 run passed three functional test bodies but aborted with exit 134
after a subordinate ERISC reported `DebugAssertNCriscNOCPacketTagClearedTripped`.
An isolated TP4 run also passed its test body but aborted at shutdown with a
heartbeat timeout after watcher stopped. These are failed process lifetimes.

**Verified source gap:** router teardown waits for transaction completion but
does not clear sticky packet tags. Firmware checks those tags after
`kernel_main()` returns. Completion counters do not reset the command-buffer
registers. This explains the violated handoff contract; whether restoring it
also fixes every heartbeat timeout remains a hardware hypothesis.

## Minimal candidate

Only [fabric_erisc_router.cpp](../../../../../tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp)
is changed in implementation:

```diff
     if constexpr (NUM_ACTIVE_ERISCS > 1) {
         wait_for_other_local_erisc();
     }
+    // Transaction completion leaves sticky tags; restore this ERISC's NoC
+    // before returning to the firmware wrapper.
+    noc_clear_packet_tags(noc_index);
     if constexpr (IS_TEARDOWN_MASTER()) {
         *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;
```

This is the last local-ERISC synchronization in `teardown()`, after tracked
write completion, counter initialization, final termination notification, and
ordinary write/atomic barriers. Subsequent kernel code only disables its L1
data cache and emits the existing final waypoint. Assertions, sanitizers,
barriers, synchronization, termination logic, and linker size limits remain
unchanged. No test that merely mirrors this call was added.

## Ownership and architecture review

- The retained failing specialization has `MY_ERISC_ID=1`, two active ERISCs,
  receiver local-write/forwarding NoC IDs both 1, and command-buffer IDs 0/2/1.
  The [diagnosis ledger](AUTOTRIAGE_watcher_shutdown.md#retained-failing-specialization-owned-noc-scope)
  links the actual generated header and kernel identity.
- [FabricRiscConfig](../../../../../tt_metal/fabric/erisc_datamover_builder.cpp)
  selects NoC0 for ERISC0 and NoC1 for ERISC1; its single-ERISC forced-NoC1
  path is separate. Clearing `noc_index` covers the observed subordinate's
  handoff without clearing its peer's NoC. Dynamic-NoC mode is already rejected
  by `teardown()` static assertions.
- [Blackhole helper](../../../../../tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h)
  zeros every command-buffer packet tag on the selected NoC; the wrapper's
  postcondition reads write, register-write, and atomic command-buffer tags.
  [Wormhole](../../../../../tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h)
  provides the same helper. Both Quasar API variants provide no-op definitions,
  so no new architecture-specific symbol is introduced.
- [tt_fabric_mux.cpp](../../../../../tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp)
  already uses full barrier → `noc_clear_packet_tags(noc_index)` → termination.
  This supports the cleanup boundary, not a claim that every router
  specialization has identical cross-NoC ownership. Broader architecture and
  fabric-mode runtime coverage is not supplied by this bounded experiment.

## Checks and hardware verification

Source inspections performed:

```bash
sed -n '2850,2975p' tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
rg -n 'noc_clear_packet_tags|ncrisc_noc_clear_packet_tags|ncrisc_noc_packet_tags_cleared' tt_metal/hw/inc tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp
git diff --check -- tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
git diff -- tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
```

The scoped whitespace check passed and the reviewed implementation diff is
three added lines. The investigator ran no build, hardware command, TTNN
import, or pytest; the coordinating agent performed the following checks.

| Check | Retained result |
| --- | --- |
| Mandatory `.github/scripts/copilot-build.sh` | Attempted; exit 1. [Wrapper log](kernel_build.log) reports Docker access unavailable; [environment check](kernel_build_environment.log) identifies permission denied on `/var/run/docker.sock`. Repository compilation remains **unverified because of the environment blocker**. |
| Kernel lint | [kernel_lint.log](kernel_lint.log) passes applicable pre-commit checks, including clang-format and Metalium include validation. |
| Device JIT and ring smoke | Coordinator reports runtime JIT compilation and ring open/close pass. [Fixed-kernel smoke log](triage/isolated_watcher_tp4/mesh_smoke_fixed_kernel.log) is retained. No size-gate bypass was introduced; the 26,624-byte ACTIVE_ETH limit remains enforced. Runtime JIT does not substitute for the blocked repository build. |
| Isolated TP4 watcher | **Full process exit 0**, reported by the coordinator. [Log](trace_reuse_probe/router_fix_tp4.log), [JUnit](trace_reuse_probe/router_fix_tp4.xml) and [functional JSON](trace_reuse_probe/router_fix_tp4/trace_reuse_tp4.json) show one pass in 36.96 s. Watcher stops at 20:28:46.711 UTC, UMD close completes at 20:28:47.304, and cluster destruction completes at 20:28:47.305 with no later fatal marker. |
| Combined TP1/TP2/TP4 watcher | **Full process exit 0**, reported by the coordinator. [Log](trace_reuse_probe/final_router_fix.log) and [JUnit](trace_reuse_probe/final_router_fix.xml) record **3 passed in 101.48 s**; [profile JSONs](trace_reuse_probe/final_router_fix/) retain assertion and allocation results. UMD close and cluster destruction complete at 20:31:02.819 UTC without post-summary failure. |
| Immediate separate-process reopen without reset | **Exit 0**, reported by the coordinator. The next operation opens a 2x2 `FABRIC_2D` physical parent and 1x2 model submesh, then closes both. [Reopen log](trace_reuse_probe/router_fix_reopen.log) records `MESH_REOPEN_WITHOUT_RESET_OK` and UMD/cluster closure at 20:31:38.422 UTC. This also exercises the transition from the final TP4 ring run to the TP2 parent/submesh configuration. |

The isolated before/after result verifies the fix at the reproduced TP4
cleanup boundary: unchanged functional assertions passed in both runs, while
the earlier run ended with a 20-second heartbeat timeout and exit 134 and the
patched run completed shutdown and exited 0. All watcher assertions and NoC
sanitizers remain enabled with the existing noinline and waypoint-only settings.

Current verdict: **router cleanup repair verified for isolated TP4, combined
TP1/TP2/TP4 lifetime, and immediate parent/submesh reopen without reset**.
Repository build verification remains environment-blocked; device JIT and
applicable lint passed. The combined JSONs confirm watcher interval 10,
noinline, allocation tracking, and only waypoint breadcrumbs disabled.

This closes the router-kernel experiment. The explicit `TTWorker.shutdown()`
fix has separate ownership tests. Its later real vLLM server shutdown/reopen
proof is recorded in [cleanup_evidence.json](cleanup_evidence.json) and
[worker_cleanup_experiment.md](worker_cleanup_experiment.md); it is not inferred
from these pytest-fixture runs. No performance improvement or universal
firmware root-cause claim is made.
