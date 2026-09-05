# Selected BFP4/LoFi head geometry results

**C32/K4/R2 is the focused component winner:** 0.436340 ms for the head and
0.449184 ms for normalization plus head, versus paired C64/K1/R2 medians
0.814679 and 0.827854 ms. Terminal latency falls 45.74% in this component
experiment. This result advances a geometry to the parent’s full-model gates;
it does not select the complete precision policy or establish end-to-end speed.

All 27 legal points in the existing C16/C32/C64, legal-K, R1/R2/R3 family
are resolved: **19 device passes, one controlled allocation rejection, and
seven source exclusions**. The 21 device runs include a serial retry of the
allocation failure. Raw receipts, exact commands, source snapshots and hashes
remain immutable; [machine-readable results](head_geometry_results.json) map
every point to its evidence.

The parent ran all device jobs serially on four Blackhole chips on physical
P300c boards. Both sides use selected BFP4 weights, LoFi, BF16 input/output,
FP32 destination and packer L1 accumulation, fixed 8x4 final norm, the same
frozen real hidden, and 221952 persistent L1 bytes/bank. Each measurement is
the median of three alternating paired rounds of 64 trace replays. Head
timings include resharding and any trimming; terminal timings also include
normalization. R3 pads each logical 32768-column chunk to 33024 and trims
the two chunks independently.

## All device-tested points

| Cores | K block | Readers | Head ms | Terminal ms | Logits vs K1 baseline | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16 | 1 | 1 | — | — | No candidate output | L1 rejection; serial confirmed |
| 16 | 1 | 2 | 0.807867 | 0.821123 | Exact | Pass |
| 16 | 1 | 3 | 0.632467 | 0.645190 | Exact | Pass |
| 16 | 2 | 2 | 0.550391 | 0.562993 | Max error 0.125 | Pass |
| 16 | 2 | 3 | 0.526195 | 0.538651 | Max error 0.125 | Pass |
| 16 | 4 | 2 | 0.551170 | 0.563877 | Max error 0.125 | Pass |
| 16 | 4 | 3 | 0.525743 | 0.538512 | Max error 0.125 | Pass |
| 16 | 8 | 3 | 0.530804 | 0.543965 | Max error 0.125 | Pass |
| 32 | 1 | 1 | 1.533071 | 1.546364 | Exact | Pass |
| 32 | 1 | 2 | 0.807061 | 0.820463 | Exact | Pass |
| 32 | 1 | 3 | 0.629951 | 0.642569 | Exact | Pass |
| 32 | 2 | 2 | 0.454298 | 0.467314 | Max error 0.125 | Pass |
| 32 | 2 | 3 | 0.456793 | 0.468807 | Max error 0.125 | Pass |
| 32 | 4 | 2 | 0.436340 | 0.449184 | Max error 0.125 | Focused winner |
| 32 | 4 | 3 | 0.456094 | 0.468461 | Max error 0.125 | Pass |
| 64 | 1 | 1 | 1.541124 | 1.554063 | Exact | Pass |
| 64 | 1 | 2 | 0.814468 | 0.827510 | Exact | Pass |
| 64 | 1 | 3 | 0.638593 | 0.651705 | Exact | Pass |
| 64 | 2 | 2 | 0.460285 | 0.472656 | Max error 0.125 | Pass |
| 64 | 2 | 3 | 0.447134 | 0.460149 | Max error 0.125 | Pass |

All 19 passes have finite logits, exact eager repetition, exact trace versus
eager output, and exact output after every timed replay batch. Head and
terminal outputs agree exactly. The baseline top-1 token 39102 stays top-1
and in top-5 on this frozen row; this is not a full-model accuracy percentage.
All complete 248320-logit hashes agree within each K group across cores and
readers. K1 is bit-identical to the baseline. K2, K4 and K8 each change logits
and produce different hashes from each other; their full-model numerical and
qualitative gates remain necessary.

| K block | Successful geometries | Max absolute error | Mean absolute error | Boundary max error | PCC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.000000 | 0.000000000 | 0.000000000 | 1.000000000 |
| 2 | 6 | 0.125000 | 0.000519841 | 0.000976562 | 0.999999762 |
| 4 | 4 | 0.125000 | 0.000534778 | 0.000976562 | 0.999998510 |
| 8 | 1 | 0.125000 | 0.000543745 | 0.007812500 | 0.999998033 |

## Source exclusions and measured allocation rejection

The [AutoDebug arithmetic](AUTODEBUG_head_geometry.md) uses BFP4 tiles of
576 bytes including exponents, BF16 input/output tiles of 2048 bytes, FP32
intermediate tiles of 4096 bytes, and the measured static base of 111616.
The necessary optimistic frontiers are 1350912 bytes after persistent state
and 1342720 bytes after the frozen normalized tensor. These seven points
were excluded before any device execution:

| Cores | K block | Readers | Static end bytes | Exact violated bound |
| ---: | ---: | ---: | ---: | --- |
| 16 | 2 | 1 | 1348608 | Resident plus norm frontier 1342720 |
| 16 | 4 | 1 | 1799168 | Physical L1 1572864 |
| 16 | 8 | 1 | 2700288 | Physical L1 1572864 |
| 16 | 8 | 2 | 1422336 | Resident frontier 1350912 |
| 32 | 2 | 1 | 1348608 | Resident plus norm frontier 1342720 |
| 32 | 4 | 1 | 1799168 | Physical L1 1572864 |
| 64 | 2 | 1 | 1348608 | Resident plus norm frontier 1342720 |

C16/K1/R1 passed the necessary source bound but failed the live allocation
check. Both paired and serial-trace controls produce static end 1123328
versus frontier 1023232. The latter equals normalized address 1301760 minus
16384 input bytes and two simultaneously live 131072-byte common-head output
shards. Releasing the baseline trace does not change the failure. Both runs
close normally. [AutoFix](AUTOFIX_head_geometry.md) and the hashed
[rejection classifications](geometry_rejections.json) preserve this exact
implementation/lifetime blocker without claiming a universal matmul limit.

## Handoff

The geometry investigation is complete for this fixed precision, shape and
common-head family. It makes no production edit. C32/K4/R2 proceeds to the
parent’s complete precision/geometry comparisons, full32 teacher forcing,
bounded qualitative review, watcher/trace validation, final default-path
reproduction, native-context execution and end-to-end performance gates.
The final stage README and selected artifact own the complete model decision.
