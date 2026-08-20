# Independent stage review 2

Verdict: **more-work-needed**.

The independent reviewer found that the then-current linear batch-one headline path did not enter the DRAM-sharded MLP because the implementation tested the logical row count against a physical 32-row condition. It also found that projection-sharding and the promised 8/16/32/64 geometry family had not been exercised, and that the performance section did not yet reconcile same-run host, kernel, and inter-op time or include mandatory KV/state traffic in its Blackhole byte floor.

Required remediation:

1. Make batch-one traced linear decode exercise the selected optimized MLP path and assert its counters in the profiler test.
2. Measure legal input/output projection sharding and all material MLP core-count candidates, retaining first errors only as intermediate evidence.
3. Recollect exact-source correctness, Watcher, profiler, repeated fused/optimized end-to-end controls, host/device/gap reconciliation, and a required-traffic Blackhole floor.
4. Explain the post-test nanobind diagnostics using teardown and JUnit evidence, refresh manifests, and request a fresh review.

These findings are historical and are not overwritten by remediation. Evidence that addresses them is recorded in `candidates/dram_sharded_mlp/geometry/`, `candidates/dram_sharded_projections/`, `correctness/projection_selected/`, and `final_selected_v2/`.
