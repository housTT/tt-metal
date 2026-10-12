| Op family | x4 linear | x4 full | x2 linear | x2 full | p150 linear | p150 full |
|---|---:|---:|---:|---:|---:|---:|
| All ops (ops per step) | 210.1 (39) | 253.6 (53) | 246.6 (39) | 278.3 (53) | 350.4 (38) | 366.3 (52) |
| Matmul | 44.8 | 42.1 | 62.1 | 56.3 | 111.8 | 95.3 |
| Generic ops (GDN, expert kernels) | 57.4 | 16.6 | 73.4 | 27.8 | 107.3 | 48.2 |
| Layout (reshape, typecast, slice, concat, copy, transpose, tilize, untilize, permute) | 18.4 | 29.7 | 18.6 | 31.8 | 20.7 | 35.9 |
| Eltwise binary / unary | 13.5 | 20.9 | 13.5 | 18.5 | 25.7 | 32.4 |
| SDPA decode |  | 40.1 |  | 39.2 |  | 47.0 |
| Norms | 11.9 | 22.8 | 12.6 | 23.6 | 12.6 | 24.2 |
| CCL | 17.2 | 17.4 | 13.0 | 13.4 |  |  |
| Expert down sparse_matmul | 13.4 | 13.5 | 20.5 | 20.4 | 30.6 | 30.6 |
| I2S / S2I | 10.2 | 12.2 | 9.5 | 11.5 | 12.2 | 14.1 |
| RoPE |  | 24.1 |  | 21.8 |  | 23.0 |
| GDN conv | 14.6 |  | 16.6 |  | 22.8 |  |
| KV update, head split and concat |  | 6.0 |  | 7.1 |  | 8.9 |
| MoE gate, top-k, fused reduce | 8.5 | 8.2 | 6.8 | 6.8 | 6.8 | 6.8 |
