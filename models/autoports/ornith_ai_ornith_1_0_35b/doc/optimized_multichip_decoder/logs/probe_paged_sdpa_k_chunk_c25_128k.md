# Ornith C25 long-context paged-SDPA K-tile probe

This is a rejected optimization candidate, not a new default. The selected C25
policy remains `q_chunk=128, k_chunk=128`.

`probe_paged_sdpa_k_chunk.py` measured a real-weight full-attention layer on the
production 1x4 mesh at the final 2,048-token chunk of a 131,072-token context.
It used the serving-only flexible device offset, the full page-table width,
BF8 paged K/V, selected C25 arithmetic, the canonical top-k-native MoE path,
and a control-candidate-control sequence with two warmups and five synchronized
samples per arm. Every output was finite. Raw samples are in
`probe_paged_sdpa_k_chunk_c25_topk_native_128k.json`. The corresponding run with
the sparse MoE fallback is retained in `probe_paged_sdpa_k_chunk_c25_128k.json`
and reaches the same conclusion.

| physical batch | Q chunk | K chunk | median layer wall | versus mean of controls |
|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 42.197 ms (control mean) | — |
| 1 | 128 | 256 | 41.903 ms | 0.70% faster |
| 4 | 128 | 128 | 121.090 ms (control mean) | — |
| 4 | 128 | 256 | 120.611 ms | 0.40% faster |

Only ten of Ornith's forty layers use full attention; the other thirty are
DeltaNet layers and do not read the paged K/V cache. Therefore even the larger
batch-1 layer delta projects to substantially less than one percent end to end,
below the prefill promotion bar. Keep BF8 K/V and C25's symmetric 128/128 tile;
the batch-throughput work should target scheduling and the MoE path instead.
