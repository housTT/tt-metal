# AutoFix: sharded residual eager corruption

Starting report: [AUTODEBUG_sharded_trace.md](AUTODEBUG_sharded_trace.md).
Four Blackhole chips on physical P300c boards, 1x4 ring. Python-only repair;
no C++ or build changes.

The verified failure boundary is the two-link asynchronous collective path.
The model now defaults async AG/RS to one link through `MeshConfig.async_links`;
native all-reduce retains independent `links=2`. This is a measured model-side
configuration workaround, not a diagnosis or repair of the underlying CCL kernel.

All experiments used the stage `record_run.py` wrapper. Exact commands, source
archives/hashes, environment, UTC times, and return codes are in the named
`logs/<name>.provenance.json`; console output is `logs/<name>.log.gz`.
Environment: `TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
OMP_NUM_THREADS=8`.

## Hypothesis experiments

| Experiment artifact name | Intervention | Result and interpretation |
| --- | --- | --- |
| `sharded_norm_localize_bf16` | Actual recorded token, actual gamma, exact decode norm shape, two-link async gather | Local norm CPU PCC 0.9999985373; stats gather exact all ranks. Activation gather corrupts ranks 1/3, finite extrema up to 1.58e29. First corrupt boundary verified. |
| `sharded_norm_localize_dram` | Same chain in DRAM | Same ranks corrupt; L1-only mechanism refuted. |
| `sharded_norm_localize_pad` | Expose padded height through reshape | Host validation rejects unequal logical volumes; no numerical conclusion. Mesh closes cleanly. |
| `sharded_norm_localize_link1` | Gather links 2 to 1 only | All source and gathered values finite; both gathers byte-exact on every rank. |
| `sharded_norm_prefill_link1` | Same one-link chain at 128 prefill rows | Both gathers byte-exact; CPU norm PCC 0.9999982164. |
| `sharded_link1_gather_control` | Original full probe, test subclass modifies only AG links; RS remains 2 | Still corrupt eager/replay outputs; gather-only change insufficient. |
| `sharded_all_link1_control` | Same subclass additionally changes RS links 2 to 1 | Original full probe passes exact eager/restored-repeat/restored-trace and baseline PCC; differing intervention is RS link count. |

No precision change, host norm substitution, lifetime change, or semaphore-reset
change was needed. Norm/statistics are BF16, compute HiFi4 with FP32 destination
accumulation throughout. The temporary subclass is preserved in source archives;
only the durable exact-shape `tests/multichip_norm_diagnostic.py` remains.
It checks CPU norm PCC >=0.995 and both gathers against host concatenation.

## Fixed implementation verification

Exact commands after the environment prefix:

```bash
python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py sharded_async_link1_fixed timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --residual sharded --collective async
python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py sharded_async_link1_fixed_2048 timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --length 2048 --residual sharded --collective async
```

Commands run from the repository root.

| Length | Prefill PCC vs optimized single-chip | Decode PCC | Restored eager/repeat/trace exact | TP4 traced decode ms |
| --- | --- | --- | --- | --- |
| 128 | 0.9999407592 | 0.9999546865 | yes | 0.52156 |
| 2048 | 0.9999220236 | 0.9999799026 | yes | 0.52133 |

These repair measurements do not beat the parent's replicated native candidate
(~0.411 ms at 2048). They establish correctness of this residual alternative.

Watcher verification and recovery are recorded below once complete. The first
`sharded_async_link1_fixed_watcher` attempt failed before TP4 construction:
`ACTIVE_ETH` program size 29072 exceeds kernel config buffer 26624. This is an
instrumented mesh-open limit, not a model assertion. The single-chip phase passed.

Evidence correction: the historical probe overwrote `trace_mismatch.pt` on
`sharded_link1_gather_control`, its last failing trace comparison. That tensor
is now `sharded_link1_gather_control_trace_mismatch.pt`. Original 747-infinity
counts in AutoDebug were measured before this overwrite; the original tensor
no longer exists. AutoDebug now explicitly identifies that distinction.

## Watcher and recovery

`sharded_async_link1_fixed_watcher_noinline` retries the full original command
with `TT_METAL_WATCHER=1` and `env TT_METAL_WATCHER_NOINLINE=1`. Every watcher
feature remains enabled. It passes all model checks: exact eager/repeat/trace,
output PCC matching the 128-row table, and local-state PCCs
`[0.9994818138, 0.9999681908, 0.9999563483, 0.9999748932]` from the parent's
extended probe. No watcher assertion occurred. Raw watcher output is preserved
as `logs/sharded_async_link1_noinline.watcher.log`.

**The watcher command did not exit cleanly.** After results and watcher-thread
shutdown, MetalContext teardown waited 20 seconds for active Ethernet core
28-25 to resume base firmware, then aborted with return code -6. This does not
invalidate the completed model comparisons, but is a real instrumentation/
firmware teardown limitation. Do not call this an unqualified passing watcher
command or use its instrumented latency as performance evidence.

The earlier size failure was followed by serialized successful
`reset_after_watcher_size`, `list_after_watcher_size` (all four devices), and
`sharded_norm_fixed_regression` (mesh open, exact gathers and norm PCC, clean
close). After the late teardown failure, `reset_after_watcher_teardown` was
started only after the failed process ended. No stale test process or UMD lock
was removed; no hardware workloads ran concurrently.

Python syntax checks (`python -m py_compile` on implementation and diagnostic)
and Black formatting of the diagnostic completed. No build was required.

Final recovery succeeded: `reset_after_watcher_teardown` and
`list_after_watcher_teardown` both exited 0, all four chips were visible, and
`mesh_smoke_after_watcher_teardown` opened/closed the ring with `MESH_SMOKE_OK`.

The scoped watcher control `sharded_async_link1_fixed_watcher_noeth` then
**exited 0** with `TT_METAL_WATCHER=10` and
`env TT_METAL_WATCHER_DISABLE_ETH=1`. It passed exact eager/repeat/trace,
output PCCs 0.9999407592 / 0.9999546865 and all four local-state PCCs above.
Only Ethernet instrumentation was disabled; worker-core instrumentation
remained active. Raw output is `logs/sharded_async_link1_noeth.watcher.log`.
The limitation is Ethernet watcher instrumentation/teardown, not an exempted
model accuracy check. All devices closed cleanly and no job remained running.

## Final status

Fixed model-side async collective configuration, independently verified by
focused norm/gather comparison and uninstrumented 128/2048-token original
probes. Scoped watcher control exits cleanly. Underlying two-link async CCL
kernel cause remains outside this Python-only repair. The faster replicated
native design and all unrelated model code remain unchanged by this fix.
