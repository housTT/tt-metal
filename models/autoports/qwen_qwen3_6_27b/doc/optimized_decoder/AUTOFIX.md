# AutoFix record

The first independent review returned `more-work-needed`; `AUTODEBUG.md` formed five testable hypotheses. Each was isolated before integration.

| Hypothesis | Experiment | Result / action |
|---|---|---|
| H1: a mixed full MLP policy beats all-BFP8 while meeting PCC | Real-weight 2x2x2 gate/up/down BFP4/BFP8 sweep, then trace each passing frontier | BFP4/BFP4/BFP8 passes at 0.996678/0.996589 and profiles at 1445.846/1405.906 us; selected for prefill. All-BFP4 fails at 0.994122. |
| H2: packed same-input gate/up can reduce traffic | Real-weight correctness, repeated E2E, and profiler comparisons for both kinds | Rejected: one extra slice/op and slower device time (linear 4682.935/1794.579 vs 4681.596/1791.024; full 1449.091/1408.727 vs 1445.846/1405.906). |
| H3: coherent width sharding plus DRAM-sharded decode weights wins | Iterated through sharded-A requirement, tile-height-only DRAM program, duplicate prefill/decode weights, legal 32-core layouts, PCC, stress, and profiler | Proven and selected. The later batch-one and projection repair produces exact-source 4394.522/1324.744 us linear and 1275.460/1147.166 us full. Required boundary joins are retained; no sharded public output escapes. |
| H6: batch-one must enter the physical 32-row sharded program | Changed the logical condition from exactly 32 to at most 32, retained physical shard height 32, ran non-aligned/batch-32 correctness, and asserted optimized counters in the perf node | Proven. Linear batch-one decode drops to 1324.744 us and the counter assertion prevents recurrence. |
| H7: projection sharding and material core families can improve the selected path | Implemented input/output/both DRAM projection candidates and 8/16/32/64 MLP core variants; continued past first API errors | Linear input is physically invalid at 2.42 MiB L1, output wins; full both wins. Only 32 MLP cores are legal and fast. Exact failures and wins are retained. |
| H4: issue #50475 GDN opportunities remain | Inspected binding/source contract, profiled sequence adapter, then applied installed recurrent row/state geometry | Sequence adapter rejected at 344 ops / 6482.619 us; its kernel row is 141.937 us but inverse/layout traffic dominates. Recurrent 4x4 geometry passes and reduces linear decode from 1791.024 to 1661.192 us before MLP sharding. |
| H5: performance claims need same-run E2E and Blackhole roofline | Five warmed fused/optimized samples, exact-source profiler host/kernel/gap triplets, and a 512-GB/s-per-chip required-byte model | Added median E2E wins, same-run reconciliation, and 500.288/553.096/731.664-us floors including mandatory recurrent/KV traffic. |

## Integration failures caught by broad gates

The first sharded whole-suite run found traced full PCC 0.994987 and a batch-32 L1 circular-buffer collision. These were not dismissed:

1. Full decode-only gate/up weights changed from mixed BFP4 to BFP8; trace PCC became 0.998004.
2. The final residual add now writes DRAM-interleaved output instead of retaining L1-sharded outputs; batch-32 minimum per-user PCC is 0.997867 and no L1 collision remains.
3. Requested matmul output shards were aligned to Blackhole's native first-32-worker grid while RMSNorm retained its required rectangular 8x4 grid; final profiler logs contain no memory-config mismatch warnings.

Final exact correctness, Watcher, profiler, and review artifacts are referenced by `README.md` and `work_log.md`.
