| Op family | x4 linear | x4 full | x2 linear | x2 full | p150 linear | p150 full |
|---|---:|---:|---:|---:|---:|---:|
| All ops (ops per step) | 197.9 (33) | 248.2 (51) | 237.2 (33) | 274.7 (51) | 328.7 (30) | 352.6 (48) |
| Matmul | 47.1 | 44.8 | 66.1 | 60.2 | 113.6 | 97.7 |
| Generic ops (GDN, expert kernels) | 57.0 | 16.6 | 73.1 | 28.0 | 106.9 | 48.1 |
| Layout (reshape, typecast, slice, concat, copy, transpose, tilize, untilize, permute) | 12.6 | 29.7 | 15.9 | 31.8 | 14.1 | 35.9 |
| Eltwise binary / unary | 6.8 | 14.1 | 3.9 | 12.1 | 16.6 | 23.4 |
| SDPA decode |  | 40.0 |  | 39.2 |  | 47.0 |
| Norms | 11.8 | 22.9 | 12.4 | 23.7 | 12.3 | 24.2 |
| CCL | 17.1 | 17.4 | 13.5 | 13.0 |  |  |
| Expert down sparse_matmul | 12.0 | 12.0 | 19.8 | 19.9 | 28.0 | 27.7 |
| I2S / S2I | 10.1 | 12.0 | 9.4 | 11.4 | 8.2 | 10.6 |
| RoPE |  | 24.0 |  | 21.8 |  | 22.8 |
| GDN conv | 14.7 |  | 16.6 |  | 22.6 |  |
| KV update, head split and concat |  | 6.0 |  | 7.1 |  | 8.9 |
| MoE gate, top-k, fused reduce | 8.6 | 8.6 | 6.5 | 6.5 | 6.3 | 6.3 |
