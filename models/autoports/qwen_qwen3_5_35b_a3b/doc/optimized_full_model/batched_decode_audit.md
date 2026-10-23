# Batched decode audit (source only, Oct 14)

Read-only audit by a subagent; claims cite source lines; not hardware evidence.

| Part | B=1 only because | Batch-capable reference |
|---|---|---|
| GDN recurrence | `tt/gdn_decode.py` custom kernel: one state, row 0 only | qwen36 `tt/gdn/tp.py` stacked `[B,Nv,Dk,Dv]` state |
| GDN conv | `qkv_causal_conv1d_silu` needs `[1,T,C]`, history `[1,3,C]` | qwen36 K-tap shift plus `mac` over `[1,B,C]` |
| GDN state store | per-slot Python tensor pairs, swapped by `_select_state` | qwen36 `write_slot`, `remap_slots` |
| Full attention | `_OptimizedAttention.forward` asserts batch 1; single-core KV update shards; `[:1]` slices | paged SDPA decode and update cache take `B` |
| MoE | `_PackedGateMoE` falls back for T > 1; expert gemv reads ids page 0 | `_LocalDispatch` prefill path takes T rows; qwen36 union-mask `sparse_matmul` |
| Norm, CCL | decode path chosen by `shape[-2] == 1` | buffers already hold 32 rows |

- Hardest part: GDN decode kernel and conv, both single-sequence.
- `fused_recurrent_gated_delta_rule` takes `[B,...]` but needs `B*HV` cores.
- B=32 MoE expert union about 163 of 256; B=4 about 31.
