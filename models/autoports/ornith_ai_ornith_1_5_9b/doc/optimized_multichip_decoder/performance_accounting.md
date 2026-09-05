# TP4 performance accounting

Complete paired before/after accounting; required raw files, logs, provenance and rank accounting verified

Four Blackhole chips on physical P300c boards; native 1x4 ring. All timing is from the same profiled run as its raw ops. Profiler overhead is retained.

Ranks have independent clocks. Ranges below are min–max across ranks, never sums. Endpoint spans divide the measured firmware cycle range by an independently inferred per-rank cycles/ns scale. Firmware durations are not added to kernel times.

| Label / kind / mode | Host wall µs/iter | Rank kernel µs/iter | Rank interior gaps µs/iter | Rank endpoint span µs/iter | Span − kernel − gaps µs/iter | Ops/rank/iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before / linear_attention / prefill | 4254.113 | 3240.275–3265.833 | 711.325–724.289 | 3962.560–3982.009 | 0.807–0.840 | 49.000 |
| before / linear_attention / decode | 401.492 | 300.397–302.153 | 80.641–82.890 | 382.915–384.126 | 0.121–0.126 | 62.000 |
| before / full_attention / prefill | 3523.598 | 2520.122–2583.012 | 591.725–607.630 | 3112.675–3187.025 | 0.802–0.828 | 38.000 |
| before / full_attention / decode | 337.783 | 266.755–268.655 | 49.888–50.183 | 317.010–318.687 | 0.116–0.124 | 47.000 |
| after / linear_attention / prefill | 4243.108 | 3215.104–3226.755 | 735.883–741.706 | 3955.759–3969.243 | 0.782–0.833 | 49.000 |
| after / linear_attention / decode | 385.308 | 286.832–288.344 | 79.100–81.021 | 367.351–368.621 | 0.120–0.122 | 59.000 |
| after / full_attention / prefill | 3438.701 | 2541.976–2574.415 | 584.350–588.579 | 3130.223–3163.701 | 0.777–0.834 | 38.000 |
| after / full_attention / decode | 314.762 | 254.142–254.852 | 48.478–48.871 | 302.940–303.530 | 0.115–0.128 | 46.000 |

| Kind / mode | Before host µs/iter | After host µs/iter | Host latency reduction | Max independent rank-span reduction |
| --- | ---: | ---: | ---: | ---: |
| linear_attention / prefill | 4254.113 | 4243.108 | 0.26% | 0.32% |
| linear_attention / decode | 401.492 | 385.308 | 4.03% | 4.04% |
| full_attention / prefill | 3523.598 | 3438.701 | 2.41% | 0.73% |
| full_attention / decode | 337.783 | 314.762 | 6.82% | 4.76% |

Only the first rank gap begins before the signpost and is excluded; its original value is retained in JSON/CSV. Every interior gap is preserved, including signed gaps and boundaries between trace replays.

PM BANDWIDTH is a modeled time in nanoseconds, not measured traffic or bandwidth. Values ≤1 ns are excluded: the generic model initializes bandwidth/compute/ideal time to 1 ns (ttnn/api/ttnn/operation.hpp). Coverage and exclusions are explicit in JSON. These partial modeled totals are not a full-layer roofline.

| Label / kind / mode | Valid bandwidth-model ops/rank/iter | Modeled bandwidth µs/iter | Kernel time covered by model µs/iter |
| --- | ---: | ---: | ---: |
| before / linear_attention / prefill | 25.000 | 444.179 | 1833.777–1844.228 |
| before / linear_attention / decode | 6.000 | 99.657 | 117.587–123.094 |
| before / full_attention / prefill | 20.000 | 353.804 | 1765.392–1818.243 |
| before / full_attention / decode | 7.000 | 95.380 | 118.856–129.566 |
| after / linear_attention / prefill | 25.000 | 459.917 | 1800.063–1818.801 |
| after / linear_attention / decode | 4.000 | 99.658 | 92.547–93.353 |
| after / full_attention / prefill | 20.000 | 353.804 | 1784.522–1813.441 |
| after / full_attention / decode | 6.000 | 95.380 | 97.383–98.103 |

## Optimistic decode storage floor, paired with the same run

This separate model reads each active projection weight once, including BFP tile exponent bytes, plus the minimum tiled K/V cache at 2049 logical tokens. Actual profiled weight dtype/shape and BFP8 cache metadata must match the final policy: 30,818,304 projection bytes/rank for linear attention; 34,734,080 for full attention, plus 65×8×1088×2 = 1,131,520 KV bytes/rank. Packed GDN and gate/up count their consumed packed weights once; unused persistent copies are excluded.

The denominator is an explicit 512 GB/s per-chip model assumption (decimal GB), taken from the installed tt-perf-report Blackhole architecture model and the prior-stage PERF_ANALYSIS.md. It is not a measured P300c peak. Official [P150 specifications](https://docs.tenstorrent.com/aibs/blackhole/) list 512 GB/s; the [QuietBox P300c specifications](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html) do not establish a measured per-chip DRAM value for this runner. TP4 aggregate modeled bytes divided by 4×512 GB/s gives the same floor, without adding times across chips.

Recurrent-state traffic, cache writes, activation movement, collectives, chunk over-read/repeated tile reads, compute, and host/synchronization overhead are omitted. These optimistic modeled floors are not attainable end-to-end targets. Fractions below use the same profiled host wall and rank spans, including profiler overhead and all interior gaps.

| Label / kind | Projection + KV bytes/rank | Modeled floor µs | Same-run host µs | Floor / host | Same-run rank span µs | Floor / rank span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before / linear_attention | 30,818,304 + 0 | 60.192 | 401.492 | 14.99% | 382.915–384.126 | 15.67–15.72% |
| before / full_attention | 34,734,080 + 1,131,520 | 70.050 | 337.783 | 20.74% | 317.010–318.687 | 21.98–22.10% |
| after / linear_attention | 30,818,304 + 0 | 60.192 | 385.308 | 15.62% | 367.351–368.621 | 16.33–16.39% |
| after / full_attention | 34,734,080 + 1,131,520 | 70.050 | 314.762 | 22.25% | 302.940–303.530 | 23.08–23.12% |

JSON also retains every measured norm/SDPA tensor's actual logical/padded shape, dtype, layout, memory location, complete attributes, call counts and per-rank kernel timing under `norm_sdpa_runtime_contracts`.

## Projection runtime metadata

Actual activation/weight/output dtype, fidelity, program family, K block, output tiles per core, and explicit DRAM reader count follow. A dash means the raw configuration does not expose that field. JSON retains complete program/memory configurations and per-rank model coverage. Its reported_core_count is the profiler's program worker-core count, not the input shard-core count (e.g. 110 or 80 program workers can use a 32-core input shard). Roles are inferred from local K/N and gate/up order.

| Label / kind / mode | Role | A / B / output dtype | Fidelity | Program; K block; M×N; readers | Rank kernel µs/iter |
| --- | --- | --- | --- | --- | ---: |
| before / linear_attention / prefill | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 162.802–164.012 |
| before / linear_attention / prefill | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.159–155.856 |
| before / linear_attention / prefill | gdn_out | FLOAT32 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 99.012–102.894 |
| before / linear_attention / prefill | gdn_packed | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×6; — | 111.070–112.969 |
| before / linear_attention / prefill | gdn_z_epilogue | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×3; — | 94.512–97.565 |
| before / linear_attention / prefill | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 155.787–156.915 |
| before / linear_attention / decode | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 6; 1×4; — | 25.143–25.476 |
| before / linear_attention / decode | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×3; — | 25.340–27.931 |
| before / linear_attention / decode | gdn_out | FLOAT32 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×4; — | 9.590–9.777 |
| before / linear_attention / decode | gdn_packed | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 32; 1×3; — | 19.044–19.984 |
| before / linear_attention / decode | gdn_z_epilogue | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 32; 1×1; — | 12.474–12.505 |
| before / linear_attention / decode | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×3; — | 25.679–27.608 |
| before / full_attention / prefill | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 161.444–164.227 |
| before / full_attention / prefill | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.301–155.939 |
| before / full_attention / prefill | o_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 91.505–93.390 |
| before / full_attention / prefill | qkvg | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×8; — | 136.379–137.876 |
| before / full_attention / prefill | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 155.519–156.475 |
| before / full_attention / decode | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 6; 1×4; — | 25.158–25.594 |
| before / full_attention / decode | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×3; — | 25.350–27.625 |
| before / full_attention / decode | o_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×4; — | 9.736–9.848 |
| before / full_attention / decode | qkvg | BFLOAT16 / BFLOAT8_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 4; 1×3; 1 | 29.044–35.139 |
| before / full_attention / decode | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×3; — | 25.442–27.703 |
| after / linear_attention / prefill | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 163.026–163.799 |
| after / linear_attention / prefill | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.255–155.276 |
| after / linear_attention / prefill | gdn_all | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.195–155.293 |
| after / linear_attention / prefill | gdn_out | FLOAT32 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 101.136–102.959 |
| after / linear_attention / prefill | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 155.273–156.353 |
| after / linear_attention / decode | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 6; 1×16; 2 | 20.483–20.741 |
| after / linear_attention / decode | gate_up | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 4; 1×6; 3 | 34.373–34.627 |
| after / linear_attention / decode | gdn_all | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×4; — | 27.993–28.587 |
| after / linear_attention / decode | gdn_out | FLOAT32 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×4; — | 9.569–9.742 |
| after / full_attention / prefill | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 162.155–163.054 |
| after / full_attention / prefill | gate_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.569–156.114 |
| after / full_attention / prefill | o_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×12; — | 92.728–94.287 |
| after / full_attention / prefill | qkvg | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×8; — | 136.950–137.720 |
| after / full_attention / prefill | up_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastProgramConfig; 16; 7×9; — | 154.875–156.959 |
| after / full_attention / decode | down_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 6; 1×16; 2 | 20.306–20.391 |
| after / full_attention / decode | gate_up | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 4; 1×6; 3 | 34.596–35.280 |
| after / full_attention / decode | o_proj | BFLOAT16 / BFLOAT4_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCast1DProgramConfig; 8; 1×4; — | 9.701–9.857 |
| after / full_attention / decode | qkvg | BFLOAT16 / BFLOAT8_B / BFLOAT16 | LoFi | MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig; 4; 1×3; 1 | 28.975–29.070 |

## Material operation counts

Counts below are per rank per iteration. The table includes every changed opcode count, plus opcodes contributing at least 2% of a rank's kernel total in either run; JSON retains all opcodes.

### linear_attention / prefill

| Opcode | before | after |
| --- | ---: | ---: |
| AllGatherDeviceOperation | 2.000 | 2.000 |
| BinaryNgDeviceOperation | 6.000 | 6.000 |
| ChunkGdnPrepOperation | 1.000 | 1.000 |
| ChunkGdnScanOperation | 1.000 | 1.000 |
| LayerNormDeviceOperation | 2.000 | 2.000 |
| MatmulDeviceOperation | 6.000 | 5.000 |
| QkvCausalConv1dSiluOperation | 1.000 | 1.000 |
| ReduceScatterDeviceOperation | 2.000 | 2.000 |
| SliceDeviceOperation | 7.000 | 8.000 |

### linear_attention / decode

| Opcode | before | after |
| --- | ---: | ---: |
| AllGatherDeviceOperation | 2.000 | 2.000 |
| BinaryNgDeviceOperation | 12.000 | 12.000 |
| LayerNormDeviceOperation | 4.000 | 4.000 |
| MatmulDeviceOperation | 9.000 | 7.000 |
| ReduceScatterMinimalDirectDeviceOperation | 2.000 | 2.000 |
| ReshapeViewDeviceOperation | 4.000 | 4.000 |
| ReshardDeviceOperation | 1.000 | 0.000 |
| ShardedToInterleavedDeviceOperation | 4.000 | 2.000 |
| SliceDeviceOperation | 7.000 | 10.000 |
| TernaryDeviceOperation | 3.000 | 3.000 |
| UnaryDeviceOperation | 5.000 | 4.000 |

### full_attention / prefill

| Opcode | before | after |
| --- | ---: | ---: |
| AllGatherDeviceOperation | 2.000 | 2.000 |
| BinaryNgDeviceOperation | 4.000 | 4.000 |
| LayerNormDeviceOperation | 4.000 | 4.000 |
| MatmulDeviceOperation | 5.000 | 5.000 |
| ReduceScatterDeviceOperation | 2.000 | 2.000 |
| SDPAOperation | 1.000 | 1.000 |
| SliceDeviceOperation | 10.000 | 10.000 |

### full_attention / decode

| Opcode | before | after |
| --- | ---: | ---: |
| AllGatherDeviceOperation | 2.000 | 2.000 |
| BinaryNgDeviceOperation | 4.000 | 4.000 |
| LayerNormDeviceOperation | 4.000 | 4.000 |
| MatmulDeviceOperation | 5.000 | 4.000 |
| ReduceScatterMinimalDirectDeviceOperation | 2.000 | 2.000 |
| ReshardDeviceOperation | 4.000 | 3.000 |
| SdpaDecodeDeviceOperation | 1.000 | 1.000 |
| ShardedToInterleavedDeviceOperation | 3.000 | 2.000 |
| SliceDeviceOperation | 6.000 | 8.000 |
| TilizeWithValPaddingDeviceOperation | 2.000 | 2.000 |
