# AutoDebug: watcher Ethernet program size

## Diagnosis and verified mesh configuration

Watcher instrumentation makes the TP2 fabric startup program exceed the fixed
26,624-byte ACTIVE_ETH kernel-config allocation. This failure occurs while
finalizing the fabric program during mesh creation, before the candidate model
adapter runs. It is distinct from an execution-time NoC/assertion failure.

The smallest tested configuration that passes the parent-mesh smoke is:

```bash
TT_METAL_WATCHER=10
TT_METAL_WATCHER_NOINLINE=1
TT_METAL_WATCHER_DISABLE_WAYPOINT=1
```

Keep assertions and NoC sanitization enabled on **both Ethernet and Tensix**.
This configuration removes progress-waypoint stores only. The coordinating
agent verified a fresh full 2x2 `FABRIC_2D` mesh open/close: exit 0,
`MESH_SMOKE_OK`, disabled features exactly `WAYPOINT`, and watcher checks on all
four devices. It therefore fits the allocation for that parent-mesh program;
the exact new byte count was not logged. Full TP2/TP4 adapter watcher
regressions remain pending the separate stable-page-state repair. No
implementation change is proposed for this program-size failure.

## Evidence

| Artifact under this directory | Direct observation |
| --- | --- |
| `trace_reuse_probe/watcher.log:139` | TP1 watcher regression passes. |
| `trace_reuse_probe/watcher.log:140` | TP2 fixture changes to `FABRIC_2D`, then begins fabric initialization. |
| `trace_reuse_probe/watcher.log:152` | Program size 30,880 exceeds 26,624: overflow 4,256 bytes. TP2 fails in fixture setup before model execution. |
| `trace_reuse_probe/watcher_noinline_mesh.log:21` | The separate mesh smoke initializes watcher with disabled features `None`. |
| `trace_reuse_probe/watcher_noinline_mesh.log:27` | With the coordinating agent's `NOINLINE=1` experiment, size is 26,784: 4,096 bytes smaller, but still 160 bytes over. Stack is `finalize_program_offsets` → `configure_fabric` → fabric initializer → mesh creation. |
| `trace_reuse_probe/watcher_noinline_mesh.log:64` | Heartbeat timeout appears approximately 20 seconds later during teardown of the failed open. It is downstream of the program-size exception. |
| `trace_reuse_probe/watcher_noinline_no_waypoint_mesh.log:21` | Disabled features are exactly `WAYPOINT`; watcher checks devices 0, 1, 2, and 3. |
| `trace_reuse_probe/watcher_noinline_no_waypoint_mesh.log:28` | `FABRIC_2D` initialization completes at 20:10:36.855 UTC, then `MESH_SMOKE_OK`. UMD close and cluster destruction complete at 20:10:40.303 UTC. The coordinating agent records exit 0. |

The failed noinline-only smoke subsequently aborted itself during teardown
(exit 134 before its 120-second timeout), according to the coordinating agent;
it was not forcibly killed. The next bounded list/reset/list sequence returned
exit 0 for each command and recovered all four devices. Logs are preserved in
`triage/watcher_size/{list_before,reset,list_after}.log`. No locks were cleared.

These are existing hardware-run artifacts read by the investigator. This
investigator ran no hardware command, imported no TTNN module, and changed no
implementation file.

## Source contracts

- `tt_metal/impl/program/program.cpp:2903` finalizes runtime arguments,
  semaphores, circular/dataflow-buffer metadata, and kernel binaries, then at
  line 2993 enforces total `state.offset <= get_ringbuffer_size(...)`. Thus the
  error's program size includes aligned program configuration and binaries;
  it is not necessarily the size of a single ELF text section.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:233` defines
  `MEM_ERISC_KERNEL_CONFIG_SIZE` as `26 * 1024`. Changing trace region, context
  length, model batch, or model tensor precision does not enlarge this limit.
  Keep the fixed allocation and guard unchanged.
- `tt_metal/llrt/rtoptions.cpp:1242` accepts `TT_METAL_WATCHER_NOINLINE`;
  lines 1322 and 1854 accept the waypoint-disable flag.
  `tt_metal/jit_build/build.cpp:260` turns these settings into
  `WATCHER_ENABLED`, `WATCHER_NOINLINE`, and `WATCHER_DISABLE_WAYPOINT` compiler
  definitions.
- `tt_metal/hw/inc/internal/risc_attribs.h:43` removes the `FORCE_INLINE`
  expansion when watcher and `WATCHER_NOINLINE` are defined. It does not disable
  validation; it also does not force every ordinary C++ `inline` function to
  become out-of-line. No separate supported NoC-sanitizer-specific noinline
  environment flag was found in the runtime options.
- `tt_metal/hw/inc/api/debug/waypoint.h:16` gates only waypoint stores:
  `WAYPOINT(x)` expands to nothing when `WATCHER_DISABLE_WAYPOINT` is present.
  Fabric router sites include
  `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:3751`, 3754,
  3798, and 3815; included dataflow APIs have additional waypoint sites around
  NoC operations and barriers. Removing these stores is a plausible way to
  save the small remaining allocation excess. Exact savings depend on the
  compiler and selected fabric specialization.
- Assertion instrumentation is independently gated in
  `tt_metal/hw/inc/api/debug/assert.h:17` by `WATCHER_ENABLED`, absence of
  `WATCHER_DISABLE_ASSERT`, and absence of `FORCE_WATCHER_OFF`. No waypoint
  condition appears in that gate.
- NoC sanitizer instrumentation is independently gated in
  `tt_metal/hw/inc/internal/debug/sanitize.h:21` by watcher enabled, absence of
  `WATCHER_DISABLE_NOC_SANITIZE`, absence of `FORCE_WATCHER_OFF`, and a
  non-TRISC compilation target. Ethernet/dataflow NoC checks therefore remain
  compiled under the proposed flags. Waypoint removal does not disable their
  address/alignment checks or error reporting.

## Focused verification and coverage

The coordinating agent completed recovery and the fresh-process 2x2
`FABRIC_2D` parent-mesh smoke with the waypoint flag. That startup gate passes.
Next rerun the TP2 adapter regression, including its actual 1x2 submesh fixture,
with the identical flags, then TP4 under its own fabric configuration. Preserve
the watcher logs independently of profiler collection. These full adapter
regressions remain necessary: the parent-mesh smoke does not establish model
execution coverage or prove that every later model kernel fits.

Before that launch, **unset**, rather than setting to `0`, any inherited
`TT_METAL_WATCHER_DISABLE_ASSERT`, `TT_METAL_WATCHER_DISABLE_NOC_SANITIZE`,
`TT_METAL_WATCHER_DISABLE_SANITIZE_NOC` (accepted alternate environment name),
`TT_METAL_WATCHER_DISABLE_ETH`, or `TT_METAL_WATCHER_DISABLE_DISPATCH`.
`rtoptions.cpp:1854` treats feature-disable variable presence as enabled
disablement even when its value is `0`. Ensure the log reports no unexpected
disabled features. Waypoint changes participate in the watcher cache hash at
`rtoptions.cpp:2284`, so no cache deletion is needed for this experiment.

Do not use `scripts/run_safe_pytest.sh --dev` unchanged when requiring this exact
coverage: lines 217–219 deliberately set `NOINLINE=1`, `DISABLE_ASSERT=1`, and
`DISABLE_DISPATCH=1`. That wrapper uses lightweight/LLK ebreak assertions with
a different diagnostic contract. A direct pytest invocation under the explicit
watcher environment avoids silently changing the requested coverage.

The sole coverage loss of the three-flag candidate is **waypoint progress
breadcrumbs across all instrumented kernels**, including last-wait-site hints.
Watcher assertions, normal NoC sanitization, CB sanitization where implemented,
ring-buffer records, and Ethernet link checks remain configured as before.
This does not enable opt-in linked-transaction checking, which the original
watcher configuration did not enable either.

If a later program still overflows, retain the observed size before selecting another
configuration. Disabling stack-usage tracking cannot address active-ERISC size:
`tt_metal/hw/inc/internal/debug/stack_usage.h:13` already excludes
`COMPILE_FOR_ERISC`. Disabling CB sanitization removes useful Tensix coverage
while the sanitizer's CB checks likewise exclude ERISC. No measured reason is
available to remove either feature here. Changing poll interval or skipping
host log output also does not remove the instrumented program's code.

The documented fallback for an oversized Ethernet kernel is
`TT_METAL_WATCHER_DISABLE_ETH=1`, optionally retaining `NOINLINE=1`, in
`docs/source/tt-metalium/tools/watcher.rst:63`. Its implementation in
`tt_metal/impl/host_api/tt_metal.cpp:1461` injects `FORCE_WATCHER_OFF` into
Ethernet kernels. This is a **scoped fallback, not equivalent coverage**:
Tensix assertions and NoC checks remain, but Ethernet-kernel watcher assertions,
NoC checks, waypoints, and other instrumentation are compiled out. If used,
report the multi-chip run as Tensix-watcher coverage with Ethernet excluded;
do not call it a full Ethernet/fabric watcher pass. First test waypoint removal,
which preserves the requested checks.

## Status

Source diagnosis and parent-mesh size experiment complete. The three-flag
configuration preserves assertions and NoC checks and passes full 2x2
`FABRIC_2D` mesh initialization/close, with only waypoint breadcrumbs disabled.
Full TP2/TP4 adapter watcher coverage remains pending. Documentation-only
changes require no C++ build. No model correctness result or performance
improvement is inferred from the mesh smoke.

## Resolution: full adapter watcher coverage completed

This later update supersedes the pending TP2/TP4 status above. The exact
three-flag watcher configuration successfully compiled and ran the full
TP1/TP2/TP4 adapter regression with allocation tracking. That execution exposed
a separate real router-return packet-tag defect; see
[AUTOTRIAGE_watcher_shutdown.md](AUTOTRIAGE_watcher_shutdown.md). After the
minimal owned-NoC cleanup repair, isolated TP4 and the combined three-profile
process both complete with exit 0, including final UMD/cluster cleanup.

The final [combined log](trace_reuse_probe/final_router_fix.log) and
[JUnit](trace_reuse_probe/final_router_fix.xml) record **3 passed in 101.48 s**;
the [profile artifacts](trace_reuse_probe/final_router_fix/) record watcher
interval 10, noinline, allocation tracking, and only waypoint breadcrumbs
disabled. Assertions and NoC checks were preserved. The immediate separate
2x2 `FABRIC_2D` parent/1x2 submesh open-close also exits 0 without reset;
[reopen log](trace_reuse_probe/router_fix_reopen.log).

[kernel_cleanup_experiment.md](kernel_cleanup_experiment.md) records the
successful runtime JIT, lint, and lifecycle checks, plus the required repository
build attempt blocked by Docker socket permissions. This closes the watcher
coverage/size experiment without removing Ethernet assertions or increasing
the 26,624-byte ACTIVE_ETH limit. Real vLLM server shutdown verification remains
separate from these fixture-based runs.
