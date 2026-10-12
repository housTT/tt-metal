| Op family | x4 linear | x4 full | x2 linear | x2 full | p150 linear | p150 full |
|---|---:|---:|---:|---:|---:|---:|
| All ops (ops per step) | 208.7 (38) | 251.3 (52) | 247.3 (38) | 279.0 (52) | 341.6 (35) | 358.3 (49) |
| Matmul | 47.3 | 44.2 | 66.0 | 60.3 | 113.6 | 97.5 |
| Generic ops (GDN, expert kernels) | 57.4 | 16.5 | 73.2 | 27.9 | 107.3 | 48.0 |
| Layout (reshape, typecast, slice, concat, copy, transpose, tilize, untilize, permute) | 18.3 | 29.7 | 18.4 | 31.8 | 20.6 | 36.1 |
| Eltwise binary / unary | 10.1 | 17.4 | 10.2 | 15.2 | 19.8 | 26.7 |
| SDPA decode |  | 40.1 |  | 39.2 |  | 46.9 |
| Norms | 11.9 | 22.9 | 12.6 | 23.7 | 12.7 | 24.3 |
| CCL | 17.4 | 17.5 | 13.2 | 13.0 |  |  |
| Expert down sparse_matmul | 13.0 | 13.0 | 21.0 | 21.1 | 30.0 | 30.0 |
| I2S / S2I | 10.4 | 12.2 | 9.6 | 11.4 | 8.6 | 10.6 |
| RoPE |  | 23.9 |  | 21.8 |  | 22.8 |
| GDN conv | 14.6 |  | 16.5 |  | 22.7 |  |
| KV update, head split and concat |  | 6.0 |  | 7.1 |  | 8.9 |
| MoE gate, top-k, fused reduce | 8.2 | 7.9 | 6.5 | 6.4 | 6.4 | 6.4 |
