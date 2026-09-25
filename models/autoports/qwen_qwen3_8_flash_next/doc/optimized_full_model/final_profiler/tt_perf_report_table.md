# Selected-policy tt-perf-report table

| Window | Ops | Device time (us) | Host gaps (us) | Sampling (us) |
| --- | ---: | ---: | ---: | ---: |
| layer0 GDN token-out | 195 | 3216.2155 | 87497.747 | 488.3635 |
| layer1 PLE+GDN token-out | 265 | 3632.1865 | 111426.290 | 490.0120 |
| layer3 QSA token-out | 285 | 4570.0085 | 97662.964 | 489.8210 |
| layer0 warmed prefill | 201 | 5718.4280 | 6386.355 | — |

Reports were produced from isolated selected BFP8/HiFi2 endpoint captures with
advice enabled. The large host gaps are the exact segmented route/cache/DMA
boundary measured in the completed token timeline. The layer-0 report reaches
139 GB/s and 27.1% of roofline overall. Sampling includes vocabulary all-gather
and device argmax and is not dominant. See `../profiler_provenance.txt` for the
commands, capture directories, hashes, and rejection rationale.
