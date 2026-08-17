| arm | lm head program | lm head cores | lm head dtype | terminal norm sharded | vocab align tiles | padded vocab size | topk groups | model trace | sampling trace | token out serial | token out pipelined | final norm | final norm head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | interleaved | 32 | - | no | - | 248320 | - | 1.475 | 1.181 | 2.833 | 2.657 | 0.021 | 0.390 |
| norm-sharded | interleaved | 32 | - | yes | - | 248320 | - | 1.650 | 1.181 | 3.018 | 2.857 | 0.032 | 0.596 |
| bfp4-head-interleaved | interleaved | 32 | bf4 | yes | - | 248320 | - | 1.610 | 1.182 | 3.000 | 2.803 | 0.026 | 0.542 |
| mcast1d-c64 | mcast1d | 64 | - | yes | - | 248320 | - | 1.477 | 1.181 | 2.837 | 2.688 | 0.026 | 0.426 |
| mcast1d-c110 | mcast1d | 110 | - | yes | - | 248320 | - | 1.434 | 1.181 | 2.806 | 2.642 | 0.029 | 0.382 |
| dram-sharded-c64 | dram_sharded | 64 | - | yes | - | 253952 | - | 1.553 | 1.133 | 2.898 | 2.704 | 0.045 | 0.493 |
| bfp4-head-plain | interleaved | 32 | bf4 | no | - | 248320 | - | 1.366 | 1.181 | 2.711 | 2.542 | 0.021 | 0.283 |
| groups-auto | interleaved | 32 | - | no | - | 248320 | - | 1.475 | 1.181 | 2.823 | 2.657 | 0.021 | 0.390 |
| align32 | interleaved | 32 | - | no | 32 | 249856 | 32 | 1.476 | 1.121 | 2.781 | 2.597 | 0.021 | 0.392 |
| mcast1d-c110-align32 | mcast1d | 110 | - | no | 32 | 249856 | 32 | 1.480 | 1.122 | 2.791 | 2.606 | 0.021 | 0.396 |
| mcast1d-c110-plain | mcast1d | 110 | - | no | 1 | 248320 | 20 | 1.480 | 1.181 | 2.823 | 2.668 | 0.021 | 0.396 |
| mcast1d-c110-bfp4 | mcast1d | 110 | bf4 | no | 1 | 248320 | 20 | 1.345 | 1.182 | 2.756 | 2.561 | 0.022 | 0.294 |
| mcast1d-c88 | mcast1d | 88 | - | no | 1 | 248320 | 20 | 1.480 | 1.181 | 2.840 | 2.667 | 0.021 | 0.397 |
| mcast1d-c110-align32-b | mcast1d | 110 | - | no | 32 | 249856 | 32 | 1.481 | 1.122 | 2.795 | 2.606 | 0.021 | 0.396 |
| mcast1d-c110-nsh | mcast1d | 110 | - | yes | 1 | 248320 | 20 | 1.434 | 1.181 | 2.830 | 2.646 | 0.027 | 0.382 |
| mcast1d-c110-nsh-align32 | mcast1d | 110 | - | yes | 32 | 249856 | 32 | 1.435 | 1.121 | 2.781 | 2.583 | 0.036 | 0.383 |
| interleaved-nsh-align32 | interleaved | 32 | - | yes | 32 | 249856 | 32 | 1.624 | 1.121 | 2.986 | 2.772 | 0.026 | 0.571 |
| mcast1d-c110-nsh-align32-bfp4 | mcast1d | 110 | bf4 | yes | 32 | 249856 | 32 | 1.351 | 1.122 | 2.655 | 2.487 | 0.027 | 0.283 |
| align32-repeat | interleaved | 32 | - | no | 32 | 249856 | 32 | 1.476 | 1.122 | 2.798 | 2.594 | 0.021 | 0.392 |
