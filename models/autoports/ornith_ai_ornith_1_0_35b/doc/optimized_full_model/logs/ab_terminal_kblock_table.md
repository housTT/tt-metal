| arm | norm cores | in0 block w | per core N | out subblock | fidelity | model trace | sampling trace | token out pipelined | final norm | final norm head |
|---|---|---|---|---|---|---|---|---|---|---|
| k8-n8-hifi2 | 8 | 8 | 18 | 1x6 | HiFi2 | 1.434 | 1.121 | 2.586 | 0.028 | 0.383 |
| k16-n4-hifi2 | 4 | 16 | 18 | 1x6 | HiFi2 | 1.476 | 1.122 | 2.596 | 0.025 | 0.392 |
| k4-n8-hifi2 | 8 | 4 | 18 | 1x6 | HiFi2 | 1.442 | 1.121 | 2.580 | 0.025 | 0.380 |
| k8-n4-hifi2 | 4 | 8 | 18 | 1x6 | HiFi2 | 1.437 | 1.122 | 2.583 | 0.034 | 0.384 |
| k16-n2-hifi2 | 2 | 16 | 18 | 1x6 | HiFi2 | 1.463 | 1.121 | 2.602 | 0.031 | 0.396 |
| k8-n8-lofi | 8 | 8 | 18 | 1x6 | LoFi | 1.466 | 1.122 | 2.593 | 0.035 | 0.380 |
| k16-n4-lofi | 4 | 16 | 18 | 1x6 | LoFi | 1.472 | 1.122 | 2.610 | 0.025 | 0.387 |
| k8-n8-hifi4 | 8 | 8 | 18 | 1x6 | HiFi4 | 1.437 | 1.122 | 2.582 | 0.029 | 0.385 |
| interleaved-k64 | 8 | - | - | - | HiFi2 | 1.476 | 1.121 | 2.595 | 0.021 | 0.392 |

Two arms are absent because they do not build, which is the blocker the sweep was looking for:

| arm | terminal norm cores | in0_block_w it would use | exact failure |
|---|---|---|---|
| k32-n2-hifi2 | 2 | 32 | `Statically allocated circular buffers in program 466 clash with L1 buffers on core range [0-0 - 10-8]. L1 buffer allocated at 1404928 and static circular buffer region ends at 1532864` |
| k64-n1-hifi2 | 1 | 64 | same assert, `L1 buffer allocated at 1273856 and static circular buffer region ends at 1532864` |

