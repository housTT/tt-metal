# Frozen-source tt-perf-report summary

Source digest: `e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`

| Window | Ops | Device time | Op-to-op gaps | Sampling | LM-head rows | DRAM roofline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer 0 GDN decode + endpoints | 195 | 3.213 ms | 125.283 ms | 0.491 ms | 0.895 ms | 27.2%, 139 GB/s |
| Layer 1 PLE+GDN decode + endpoints | 265 | 3.632 ms | 153.876 ms | 0.491 ms | 0.899 ms | 25.8%, 132 GB/s |
| Layer 3 QSA decode + endpoints | 285 | 4.574 ms | 143.112 ms | 0.492 ms | 0.897 ms | 18.1%, 92 GB/s |
| Warm layer-0 prefill + endpoints | 201 | 5.736 ms | 6.368 ms | n/a | included | 17.9%, 92 GB/s |

The decode rows are representative full-model windows, not an additive
full-stack latency estimate. Reduced captures retain lazy exact host packing
to fit the profiler window, so their host gaps are diagnostic. Full-stack
prepacked token-out is measured separately at 231.490 ms/token.

Sampling is the split-sampler greedy trace: approximately 0.401 ms local-
vocabulary all-gather, 0.039 ms untilize/unpad, and 0.051 ms device argmax.
`TopKDeviceOperation` in the token-out reports is the exact expert router, not
terminal sampling. All four LM-head rows receive DRAM-sharded advice; the
focused exact DRAM s5/c40 A/B is correct but 36.24% slower, so the selected
interleaved BFP8/HiFi2 endpoint remains justified.
