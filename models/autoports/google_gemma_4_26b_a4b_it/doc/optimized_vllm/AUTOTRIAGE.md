# AUTOTRIAGE

## Diagnosis

The first optimized-vLLM P150 baseline failed while returning device 0's
active Ethernet core 29-25 to base firmware during mesh initialization. Base
firmware did not advance its heartbeat within the 20-second startup timeout.
This is a pre-model initialization failure; its underlying firmware/state cause
is not established. A bounded board reset recovered the failure boundary:
the 1x1 mesh smoke passed, and the unchanged server retry opened its mesh and
constructed `TTModelRunner` at 19:29:08 UTC on 2026-09-08. The retry started its
API server at 19:30:00 UTC and subsequently returned HTTP 200 for `/health` and
completion requests.

No implementation fix is justified by the available evidence. Continue the
unchanged serving baseline. No additional reset, source patch, tracing change,
or firmware upgrade is warranted while the resumed run remains healthy.

This report is the initial AutoFix/AutoTriage investigator pass. The investigator
read source and artifacts and wrote this report only; the coordinating agent
exclusively executed hardware recovery and the retry.

## Triage Evidence

- The original command and traceback are preserved in
  `triage/failed_runner.log` and `triage/failed_server.log`. The runner launched
  `python_env/bin/python -m vllm.entrypoints.openai.api_server` for
  `google/gemma-4-26B-A4B-it`, logical mesh alias `N150` (1x1), block size 64,
  maximum sequences 32, maximum context 50,624, trace region 220,000,000 bytes,
  device sampling `all`, async scheduling, and the Gemma 4 tool/reasoning
  parsers. The alias selects a shape; the discovered architecture is Blackhole.
- At 19:25:58, UMD discovered local devices 0–3 and firmware bundle 19.13.1.
  At 19:26:19, mesh open failed with `return_to_base_firmware_and_wait_for_heartbeat`
  in the stack. The diagnostic reports heartbeat start/end `0xdcba2600`,
  changed `false`, port status `0x1`, RX link up `0x1`, train status `0x2`,
  PCS status `0x1`, postcode `0xc0dea000`, and RISC soft reset `0x47000`.
  These are register observations, not proof of a specific firmware fault.
- `triage/capture.log` proves only that AutoTriage could not acquire inspector
  data: `/tmp/tt-metal/inspector` did not exist. `triage/tt-triage.txt` is empty.
  There are no captured device call stacks, running model ops, CB counters,
  semaphore values, or routing ledgers from this incident. Missing inspector
  logs are not evidence that model execution began or that a source bug exists.
- The runner detected `EngineCore failed to start`, then needed SIGKILL after
  SIGTERM did not finish shutdown. That cleanup symptom follows the mesh-open
  exception and does not identify its cause.
- Earlier evidence in `../vllm_integration/work_log.md` (recovery audit) and
  `../../readiness_vllm/hardware_recovery_20260908.json` describes the same
  device/core heartbeat signature during a model-free 2x2 mesh smoke at
  17:47:07 UTC, followed by successful reset, mesh smoke, and all three serving
  profiles. This supports the recovery experiment, not a claim that every
  occurrence has an identical root cause.

## Source Evidence

Paths below are repository-relative except the explicitly named vLLM checkout.

| Boundary | Source and contract | Interpretation |
| --- | --- | --- |
| Model construction | `/home/hous/dev/vllm/vllm/v1/executor/uniproc_executor.py:47` calls `init_device()` before `load_model()`. The plugin's `src/vllm_tt_plugin/worker.py:130` opens the mesh before constructing `TTModelRunner`. | The failed attempt never constructed the model runner or loaded model tensors. Current-attempt model kernels, KV updates, and trace capture cannot be the initiating operation. |
| Firmware startup | `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:215` calls `reset_cores()` before `initialize_and_launch_firmware()`. At line 377, two-ERISC reset waits for base-firmware heartbeat before asserting reset on subordinate RISCs. | The stack identifies the startup return-to-base-firmware handshake, before fresh firmware launch/model execution. |
| Producer/consumer handshake | `tt_metal/llrt/llrt.cpp:577` samples the base-firmware heartbeat. While unchanged, the host repeatedly clears the Metal Ethernet run flag and rereads the heartbeat. Line 623's helper writes `aerisc_run_flag` and issues an L1 barrier. `tt_metal/llrt/llrt.hpp:117` supplies a 20,000 ms timeout. | Host requests Metal firmware to stop; base firmware must resume and produce a changing heartbeat; host consumes that evidence before proceeding. The exception proves this handshake did not complete. It does not show why the producer failed to advance. |
| Existing global exit request | `risc_firmware_initializer.cpp:433` already sets `exit_erisc_kernel=1` on active Ethernet launch slots on all discovered chips, with an L1 barrier, for the applicable Blackhole non-cooperative fabric path. | Do not propose adding an all-chip exit request or repeated run-flag clear as a missing fix: both mechanisms already exist. Their successful consumption in this incident is unobserved. |
| Active-core scope | `tt_metal/fabric/control_plane.cpp:2251` selects Blackhole active Ethernet cores from cluster active-channel metadata, including some cores without links. Plugin `worker.py:725` maps `N150` and `P150` to the same 1x1 shape. | A single-chip model mesh can legitimately encounter active-Ethernet startup. The alias and the four discovered PCI devices do not prove an erroneous multi-chip model configuration. |
| Failed triage acquisition | `tools/triage/inspector_data.py:176` first tries inspector RPC, then falls back to serialized logs and raises if the log directory does not exist. | The capture failed before producing device-state diagnosis. Re-running the same failed capture after process exit cannot reconstruct the missing state. |

The timeout's firmware-minimum text is unconditional on that failure branch;
it does not compare the installed version. The logged 19.13.1 is above the
message's 18.10.0 minimum. This incident provides no evidence for an outdated
firmware diagnosis. Likewise, link-up status alone does not prove base firmware
is live. No model CB/semaphore or fabric-route ledger can be constructed from
the absent triage data, and none is needed to locate the observed host boundary.

## Downstream Effects

Engine startup failure, readiness failure, and forced process cleanup follow
the mesh-open exception. The earlier mesh-topology transition is a plausible
state trigger but is not proven by this capture. A prior workload could have
left stale state; the evidence does not establish whether that was a runtime,
firmware, teardown, or hardware-origin event. The motherboard-discovery warning
also appears in the successful mesh smoke and retry, so it is not sufficient
to explain this failure.

## Proposed Fix

No source changes. The smallest verify/refute experiments are:

| Hypothesis | Experiment | Result and verdict |
| --- | --- | --- |
| Recoverable residual Ethernet/firmware state blocks startup. | After coordinating-agent owner cleanup, `timeout 180 tt-smi -r`, `timeout 60 tt-smi -ls --local`, then a 1x1 `ttnn.open_mesh_device(..., trace_region_size=0)` and close. | Main agent reports reset/list/smoke exit 0. `triage/reset_1.log` records reset of devices 0–3; `list_after_1.log` lists all four P300C devices; `mesh_smoke_1.log` records `MESH_SMOKE_OK` and clean driver close at 19:28:28 UTC. Recovery of this boundary is verified; the original state trigger remains uncertain. |
| Original server configuration deterministically causes the heartbeat failure. | Retry the unchanged P150 baseline after the passing smoke, retaining the failed logs separately. | The new `../../readiness_vllm/optimized_vllm/before/P150/server.log` records successful 1x1 mesh creation and `TTModelRunner` construction at 19:29:08 UTC, API startup at 19:30:00 UTC, and HTTP 200 health/completion responses. Deterministic failure at the observed boundary is refuted. Complete serving gates remain the main agent's ongoing work. |
| Current model or trace implementation caused this startup failure. | Check executor ordering and the exception stack; confirm unchanged retry crosses the same point. | Refuted as the initiating operation in this attempt: mesh creation failed before runner/model construction. This does not exclude effects left by a previous process. |
| A persistent firmware/runtime defect or physical fault survives reset. | Observe the reset mesh smoke and unchanged retry; only investigate further if the same signature recurs. | Not supported as a persistent blocker: both opens now succeed. No reason to alter timeout values, bypass heartbeat checks, change fabric topology, or upgrade firmware from this evidence. |

If the exact failure recurs, preserve the failing command, owner/process state,
mesh transition sequence, and register dump before cleanup. A short,
model-free reproduction that alternates the actually failing mesh shapes would
then separate lifecycle sensitivity from serving/model behavior. Capture live
inspector/triage evidence where initialization gets far enough to provide it;
otherwise retain the pre-model stack and hardware diagnostics. Escalate through
the documented device-recovery path only if reset/list/mesh-open recovery fails.
Do not add a broad topology stress experiment while the baseline is healthy.

## Uncertainty

This report establishes the first observable failing contract and its recovery,
not the base-firmware root cause. No live device-state triage was captured.
Passing mesh open, server health, and completion responses establish startup
recovery; they do not substitute for correctness or performance evidence.
No performance improvement is claimed. This documentation-only investigation
requires no C++ build and made no implementation edits.

## Follow-up: repeated failure after serving shutdown

This appendix updates the earlier recovery-only conclusion. After two more
occurrences, a concrete missing worker-lifecycle hook merits an isolated
experiment. The logs refute the outer readiness runner's 15-second timeout as
the cause of these successful baseline shutdowns. They do **not** prove that
the worker explicitly closed its mesh and terminated its fabric routers.

### New evidence and close markers

Evidence is now under `../../readiness_vllm/<profile>/optimized_vllm/before/`;
older launch logs and manifests retain their original output-path spelling.

| Event | Retained evidence | Observation |
| --- | --- | --- |
| TP2 baseline shutdown | P150x2 `launch.log:10` and `server.log:1086` | API PID 1727355 receives shutdown at 19:33:33 UTC. The launcher reports `Terminated cleanly.`; there is no outer SIGKILL fallback message. |
| TP2 host driver close | P150x2 `server.log:1137` | UMD close starts 19:33:33.789 and completes 19:33:34.413; cluster destructor completes, followed by API application shutdown. |
| Next TP4 startup fails | `triage/tp2_to_tp4/failed_run/server.log` | Device 0/core 29-25 again fails the startup heartbeat handshake. A subsequent destructor failure at 19:34:34.185 is downstream cleanup of that failed open. Reset plus 1x4 smoke and the unchanged TP4 baseline then pass, as recorded in the adjacent recovery logs. |
| TP4 baseline shutdown | P150x4 `launch.log:10` and `server.log:884` | API PID 1729433 receives shutdown at 19:38:45 UTC. Again, `Terminated cleanly.` and no outer SIGKILL fallback message. The manifest records readiness-runner exit 0. |
| TP4 host driver close | P150x4 `server.log:935` | UMD close starts 19:38:45.980 and completes 19:38:46.600; cluster destructor completes, followed by API application shutdown. |
| Next 1x4 fixture setup fails | `triage/post_tp4_serving/failed_probe.log` | The next pytest fails in mesh-open setup, before the candidate adapter executes; setup lasts 20.61 seconds. Its subsequent destructor exception at 19:46:18.658 is downstream. This occurrence does not require a change from a 1x2 to a 1x4 logical mesh. |

The coordinating agent reports no live workload owners before each reset.
Both successful baseline logs contain nanobind leaked-instance/type reports,
including TT tensors, before driver closure. These support investigation of
late object destruction but do not establish which destructor was skipped.

`Closing user mode device drivers`, `Closing devices in cluster completed`,
and `Cluster destructor completed` are host driver/cluster markers.
`Application shutdown complete` is an HTTP application marker. None explicitly
acknowledges `TTWorker` mesh cleanup or router termination. The current C++ mesh
close marker is only `log_trace` (`mesh_device.cpp:962`); normal fabric teardown
has no INFO completion marker. Its absence from these INFO logs cannot prove
that it was skipped.

### Verified source-side lifecycle gap

The following paths are in `/home/hous/dev/vllm/` unless marked tt-metal:

1. `vllm/v1/engine/core.py:1041` calls `engine_core.shutdown()` from the
   process entrypoint's `finally`. `EngineCore.shutdown()` at line 579 calls
   `self.model_executor.shutdown()`.
2. `vllm/v1/executor/uniproc_executor.py:135` calls the driver worker wrapper's
   `shutdown()`. `vllm/v1/worker/worker_base.py:206` forwards it to the actual
   worker's `shutdown()`.
3. `plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py:94` defines
   `TTWorker(WorkerBase)`. It does **not** override `shutdown()`, so this explicit
   lifecycle call resolves to `WorkerBase.shutdown()` at
   `vllm/v1/worker/worker_base.py:170`, whose entire executable body is `return`.
4. TT device cleanup exists only in `TTWorker.__del__()` at plugin
   `worker.py:498`: delete `model_runner`, then call the helper
   `close_mesh_device(self.mesh_device, get_tt_config(self.vllm_config))` and
   delete `mesh_device`. It relies on object destruction to reach this code.
   Moreover, one `suppress(AttributeError)` covers both deleting the runner and
   closing the mesh: if the runner attribute is missing, that exception skips
   the subsequent mesh close even when a mesh attribute exists.
5. The helper at plugin `worker.py:838` already implements the required TP2
   ownership order: close child submeshes, close the model mesh, close its saved
   physical parent, then reset the configured fabric. Reuse this helper.
   Directly closing only the model submesh would omit the parent owning the
   physical devices.

The missing explicit cleanup hook is verified by source. Its causal connection
to the heartbeat recurrence remains a hypothesis until a focused serving
shutdown/reopen comparison passes. An apparently clean process exit can still
reach C++/UMD global destruction without running Python worker cleanup in the
intended ownership order.

In tt-metal, `MeshDeviceImpl::close_impl()` (`tt_metal/distributed/mesh_device.cpp:959`)
releases the scoped physical-device owner; its destructor at line 191 calls
`DeviceManager::close_devices()`. That function
(`tt_metal/impl/device/device_manager.cpp:680`) tears down dispatch then fabric
before closing devices. `FabricFirmwareInitializer::teardown()` at
`tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:349` writes the
termination signal to each device's master fabric router; `post_teardown()` at
line 421 resets fabric configuration. This is the concrete cleanup path that
the explicit worker hook must reach while the model and mesh owners are valid.

### Shutdown deadlines and bounded repair experiment

- `doc/optimized_vllm/run_profiles.py:174` sends SIGTERM to the readiness
  runner, waits 60 seconds, and kills its process group only on expiry.
  `models/common/readiness_check/run_vllm_server.py:758` sends SIGTERM to the
  API process and waits 15 seconds. The two `Terminated cleanly.` messages
  establish that this inner readiness timeout did not expire. That wording
  means the API process returned from `wait()`; it does not validate its exit
  code or worker cleanup.
- There is also a separate **five-second EngineCore deadline**:
  `vllm/v1/engine/core_client.py:384` closes its process manager;
  `vllm/v1/engine/utils.py:223` runs its finalizer; `vllm/v1/utils.py:302` sends
  SIGTERM, joins for five seconds, then calls `kill_process_tree()` on any live
  process. The current logs do not retain the EngineCore exit code or a kill
  marker. The observed UMD closure finishes roughly 1–2 seconds after API
  shutdown starts, so there is no positive evidence that this deadline fired.
  Increasing the readiness runner timeout alone would not change this deadline.

The smallest candidate repair belongs in the TT plugin worker: implement an
explicit, idempotent `shutdown()` that releases the model runner and calls the
existing mesh/parent/fabric close helper, and have `__del__()` use the same
cleanup as a best-effort fallback. Treat missing runner and missing mesh
independently; support partially initialized workers and non-device DP ranks.
After successful close, a repeated `shutdown()` or later destructor must not
close a mesh or parent twice. Explicit shutdown must surface cleanup failure;
do not silently report success after a failed close. Add concise begin/end
markers around the explicit cleanup so future logs distinguish completion
from host-driver destruction.

Before retaining that repair, verify the hook with a host-only ownership test:
the actual worker `shutdown()` must close the expected mesh exactly once,
including when `model_runner` is absent; repeat shutdown and exercise the
destructor fallback. Test physical parent ownership through the existing helper
tests rather than replacing that helper. Then the main agent should run one
serving baseline, shut it down normally, confirm explicit cleanup completion,
and immediately open/close the same profile **without reset**. Follow with the
previous TP2-to-TP4 transition if needed. If the heartbeat failure persists
after proven explicit cleanup, this hook gap does not explain the recurrence;
retain that distinction and return to firmware lifecycle evidence. No change
to router kernels, heartbeat timeout, or generic process deadlines is supported
by this bounded investigation.

No implementation changes or hardware commands were made by this investigator
in this follow-up. The candidate adapter was not reached by the failed fixture,
and these failures supply no evidence against its trace-reuse implementation.
