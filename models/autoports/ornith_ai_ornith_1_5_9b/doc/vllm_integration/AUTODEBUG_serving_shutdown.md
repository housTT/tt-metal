# AutoDebug: serving shutdown leaves Ethernet unready

Date: 2026-09-08. Fresh source-only investigation under AutoFix. No device commands, TTNN imports, live serving, resets, watcher, or profiler ran in this investigation.

## Evidence and verified defect

`reduced_v2.server.log` and `reduced_v3.server.log` reach C++ UMD driver close on exit. The healthy v3 server finishes five completions and reports zero running requests before shutdown. The next direct generator probe, `serving_contract_device_v2.log`, fails at mesh initialization before model code: device 0 active Ethernet core 29-25 has unchanged heartbeat, postcode `0xc0dea000`, and the 20-second firmware-return wait times out. Earlier `penalty_remap_broadcast.log` has the same signature after v2. Bounded reset restores mesh; sequential direct probes using explicit `close_ornith_mesh` close/reopen successfully.

The pinned vLLM engine executes `EngineCore.shutdown` (`vllm/v1/engine/core.py:579`) -> `UniProcExecutor.shutdown` (`vllm/v1/executor/uniproc_executor.py:135`) -> `WorkerWrapperBase.shutdown` (`vllm/v1/worker/worker_base.py:206`) -> `TTWorker.shutdown`. **TTWorker does not implement shutdown.** Its inherited `WorkerBase.shutdown` (`worker_base.py:170`) returns immediately. Mesh and fabric teardown live only in `TTWorker.__del__` (`plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py:498`). Python finalization does not provide a deterministic lifetime boundary for this resource cleanup.

This is a verified lifecycle defect. It explains why driver-close logging alone is insufficient: native `MetalContext` registers its own process-exit destruction handler (`tt_metal/impl/context/metal_context.cpp:498`), which can produce UMD-close lines independently of the plugin's explicit mesh close/fabric reset. The actual firmware consequence remains a hardware hypothesis pending the A/B check below; the source-only probe does not prove the exact firmware failure mechanism.

The plugin close helper also unconditionally invokes `ttnn.ReadDeviceProfiler`. Remove that implicit profiling from serving shutdown; profiler evidence belongs in explicit separate tooling.

## Focused verify/refute experiment

A new host-only test compiles the actual worker/base/wrapper lifecycle methods from source, using inert resource stubs. It keeps the worker alive across the explicit vLLM shutdown call and checks trace/model teardown before mesh close, repeated shutdown, partial initialization, teardown failure, and close-helper ordering without any profiler API.

Command (from tt-metal):

```bash
USER=hous ../state/serving-env/bin/python -m pytest -q -c /dev/null ../vllm/plugins/vllm-tt-plugin/tests/test_worker_shutdown.py
```

Before-fix artifact: `worker_shutdown_before.log`: **3 failed, 1 passed**. The explicit lifecycle call produces no close events; model teardown is never invoked; the close helper calls the absent profiler stub. These failures verify the no-op call-chain defect before editing runtime code.

## Smallest proposed repair and hardware validation

Implement an explicit, idempotent `TTWorker.shutdown`: call an optional adapter teardown while the mesh is live, release the runner, close submeshes and parent mesh, reset fabric; make `__del__` a best-effort fallback to that same method. Close the mesh in a finally block even if adapter teardown fails. Log explicit close completion for subsequent evidence. Remove the automatic profiler call from the close helper. Preserve the existing fabric/router configuration change.

Parent hardware A/B: after its current recovered direct probe, launch the same reduced `[0,3]` server with max-num-seqs 4 and native 262144 context; repeat the five successful requests; stop through the shared runner; verify explicit worker mesh/fabric close completion; audit no live server/EngineCore processes; immediately run `open_ornith_mesh`/`close_ornith_mesh` without reset. Repeat one server/open cycle if the first passes. A successful fresh open refutes the need for routine post-serving resets and verifies the lifecycle fix in the original failing sequence. Failure requires fresh diagnosis; do not call the hardware issue fixed solely from host tests.

## Secondary hypothesis, not yet a fix

The shared runner signals the entire private process group and ultimately SIGKILLs remaining members after the API launcher exits. A launcher exit could precede EngineCore cleanup. Existing v3 timestamps show native close finished before API shutdown completed, making interruption less likely for this observed run. If explicit shutdown is cut off in the next run, measure live private-group members and allow a bounded group drain before force-kill. Do not combine this speculative runner change with the first A/B test.

## Source repair verification

The proposed repair is now applied to the plugin worker lifecycle and close helper. No fabric setup behavior was changed by this repair. The same host-only test command with `-p no:cacheprovider` reports **4 passed** in `worker_shutdown_after.log`. Tests exercise the real source methods, explicit wrapper dispatch, one-time teardown/close ordering, partially initialized workers, closing despite adapter teardown failure, and submesh/parent/fabric-reset order with no profiler entry point available. Tests and new lifecycle code were formatted; unrelated formatting changes were reverted to keep scope narrow. Hardware causality remains pending the parent's serving-stop/reopen sequence.

## Hardware verification: first serving-stop/reopen cycle passes

Parent executed the focused original-sequence check after applying the explicit shutdown fix. The reduced `[0,3]` model served with native `max_model_len=262144`, `max_num_seqs=4`, async scheduling, traced device sampling, P150x4, and the same ring/8192-byte fabric configuration. Five original completion requests plus six sampling transitions passed before stopping via the shared runner. This is lifecycle evidence using a reduced model, not full-model quality/performance evidence.

Exact server argv, TT config, layer selection, host-sampling compatibility setting, and successful runner return code 0 are preserved in `reduced_v4_async.command.json`. `reduced_v4_async.server.log:106` records `Closing TT worker model and mesh`; line 108 records `TT worker mesh closed and fabric reset`, both at 12:58:42 UTC. The subsequent direct `open_ornith_mesh(trace_region_size=0)` / `close_ornith_mesh(mesh)` probe exited 0 and printed `POST_SERVER_MESH_SMOKE_OK` in `reduced_v4_post_server_mesh.log:23`. Parent performed no reset between stopping this server and the fresh open.

Verdict: **the verified no-op worker shutdown defect is fixed, and the first hardware reproduction cycle now passes without a reset.** The prior repeatable post-serving initialization failure no longer occurs in this cycle. A second independent server-stop/reopen check is scheduled for the next all-layer server shutdown; do not describe it as completed until its artifacts exist. The source investigator read the cited logs and command record; all hardware actions were serialized and executed by the parent.

## Hardware verification: second cycle passes after all-layer serving

The previously scheduled second cycle is now complete. `full_b32_v1.command.json` records all layers (`layers: null`), native `max_model_len=262144`, `max_num_seqs=32`, P150x4, async scheduling, the same ring/8192-byte fabric config, and runner return code 0. `full_b32_v1.server.log:90` confirms layer31 loaded. The shutdown markers at lines1807-1808 show explicit worker close and completed mesh/fabric reset at13:24:43 UTC; native driver close completes at13:24:44.365.

Without a reset between serving shutdown and the next command, the parent's subsequent exact sampler probe successfully initializes fabric on all four devices at13:25:27.124 (`presence_sampler_exact.log:21`), runs its device work, and completes driver close at13:25:32.963 (line226). The following standalone all-layer run independently initializes four-device fabric at13:26:24.310 (`standalone_haiku_512_v1.log:22`), loads layer31 at13:28:12.213 (line55), and completes driver close at13:28:24.990 (line99). These logs verify fresh mesh initialization after the all-layer server and another consecutive device run; their sampling/qualitative conclusions belong to their separate stage checks.

Final shutdown verdict: **fixed with source regression and two successful serving-stop/reopen cycles, including all-layer B32 serving.** The original repeated post-serving Ethernet initialization timeout is not reproduced in either repaired cycle; no routine reset was needed. This documentation update only read existing artifacts. The parent serialized all hardware activity and confirmed the absence of an intervening reset. No source, device, or process change was made by the investigator during this follow-up.
