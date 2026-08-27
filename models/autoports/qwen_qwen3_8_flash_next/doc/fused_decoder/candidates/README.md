# Fusing candidate evidence index

Executable candidates were applied one at a time against a staged correct
decoder. Full terminal transcripts contain the exact command, source diff and
SHA-256 when captured, PCC/timing or validator failure, and exit status. The
final bounded KDA/indexer entries link reversible patches or journal-recovered
diffs, source/test hashes, exact commands, and raw device artifacts; promoted
forms are revalidated by the final-source gate set.

## GDN and hyperconnection

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Fold GDN output sigmoid | `gdn_output_sigmoid_fold.log`: source `95113d962b88`, prefill PCC 0.98772216 | Reject: below 0.995. |
| Fold add+softplus | `gdn_add_softplus_fold.log`: source `04774faad0a7`, PCC 0.99719596/0.99811465 | Reject: material regression versus correct graph. |
| BF16 taps in FP32 MAC | `gdn_mixed_bf16_taps_fp32_mac.log`: source `edbf501f041d`, prefill PCC 0.19577280 | Reject. |
| Pack z into FP32 qkv/b/a plus typecast | `gdn_pack_z_fp32_typecast.log`: PCC passes; 19.528400/4.256352 and 22.940676/5.185828 ms | Reject: extra typecast did not win. |
| Pack decay bias into projection | `gdn_packed_projection_bias_fold.log`: PCC passes; 19.399399/4.233487 and 22.943831/5.174257 ms | Promote. |
| Fold recurrent q/k scales into RMSNorm weights | `gdn_rmsnorm_scale_weight_fold.log`: PCC passes; 19.494241/4.251980 and 22.965516/5.186544 ms | Promote. |
| KDA causal-conv+SiLU+QKV | `gdn_kda_qkv_causal_conv1d_silu.log` + `.journal.jsonl`: exact journal patches/raw output and reconstructed source identities for chunks 320/640/1280; retained 640 PCC 0.99842572/0.99703580, 17.0 ms prefill | Promote; final-source gates cover layers 0 and 1. |
| KDA sigmoid-gated RMSNorm | `gdn_kda_sigmoid_gated_rms_norm.log` + `.patch` + raw/XML artifacts: exact T=128 prefill form; valid retained-input L1 decode PCC 0.86030912/0.86216938 despite identical state | Reject after AutoFix: address/lifetime-sensitive TTNN composition failure; faster prefill cannot be delivered correctly. |
| Hyper factor-2 output-weight fold | `hyper_inject_output_weight_scale_fold.log`: all pass, but layer-0 decode PCC falls to 0.99700934 and timings are mixed | Reject. |
| Hyper scalar MAC | `hyper_inject_scalar_mac.log`: all layers pass | Promote; later final gates validate it. |

## MoE

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Four per-group expert-down calls | `per_group_sparse_moe_down.log`: source `8d946e695607`, all PCC pass; 22.810565/4.514022, 26.313628/5.460376, 48.715102/6.670980 ms | Superseded by one group-major call. |
| Fixed-320 indexed sparse MoE | `indexed_sparse_moe_fixed320.log`: PCC 0.99656379/0.99974972; 78.541340/14.560467 ms | Reject: correct but much slower. |
| Move routed/shared gates before down | `moe_pre_down_gate_placement.log`: all PCC pass; 17.981688/3.883235, 21.495932/4.818834, 43.892802/5.696528 ms | Promote. |
| Router top-k logits then selected softmax | `router_topk_logits_softmax.log`: all PCC pass; removes full-width softmax/sum/divide | Promote. |

## PLE

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Hidden-axis dot via batched matmul | `ple_batched_dot_matmul.log`: PCC passes; 32.126572/4.871731 ms | Reject: prefill regression. |
| Remove dead decode reshape | `ple_dead_decode_reshape.log`: PCC passes; 22.910294/5.173239 ms at that checkpoint | Promote. |
| Six split dilation-history tensors | `ple_split_decode_state.log`: PCC 0.99910438/0.99989325; 21.461261/4.320279 ms | Promote. |

## QSA structure, cache, and attention

| Candidate | Evidence | Decision |
| --- | --- | --- |
| SDPA K chunk 32 | `sdpa_k_chunk_32.log`: source `66c693c24089`, PCC 0.99656433/0.99974990; 45.580898/6.431233 ms | Reject versus prefill K=64. |
| SDPA K chunk 128 | `sdpa_k_chunk_128.log`: source `dde33f45010a`, PCC 0.99656421/0.99974990; 45.552516/6.422241 ms | Reject. |
| Packed gather + repeated 24-head K/V | `packed_gather_repeated_kv.log`: source `c498fb5d0220`; 246.420518/9.393573 ms | Reject. |
| Packed gather + native 24:2 GQA | `packed_gather_native_gqa.log`: source `a31a92469e17`; 154.843514/8.680890 ms | Reject: tiled 2-head padding remains. |
| Specialized decode QKV | `decode_specialized_qkv_reshard*.log` and reports: source `32a7b3805a8b`, PCC pass, 6186.090 us device | Promote, then retain V sharding farther. |
| Dedicated/specialized concat heads | `decode_specialized_concat_heads*.log` and reports: source `2991376220a9`, PCC pass, median decode 6.293552 ms | Superseded by complete inverse-TM/concat cancellation. |
| Inverse V/Q/head TM cancellation | `qsa_inverse_tm_elimination.log`: PCC pass, 45.330781/6.244270 ms at checkpoint | Promote. |
| Remove redundant gathered-valid mask | `redundant_gathered_valid_mask.log`: PCC pass | Promote. |
| Precompute static compressed addresses | `qsa_static_compressed_address_precompute.log`: PCC pass | Promote. |
| HF partial rotary | `qsa_rotary_embedding_hf.log`: PCC pass, 45.307061/6.219211 ms | Promote. |
| Dedicated decode SDPA | `qsa_dedicated_sdpa_decode.log`: first mask-width validator failure, adapted success PCC 0.99660957/0.99977440, 45.260523/6.051686 ms | Promote. |
| Small broadcast/TM sweep | `qsa_small_broadcast_tm_sweep.log`: PCC pass, 43.871195/5.680545 ms | Promote legal batch-one forms; batch fallbacks retained. |
| Fused K/V paged update | `qsa_paged_fused_kv_update.log`: PCC pass, 43.9/5.68 ms; three-run confirmation | Promote. |
| Maximal sharded Q/K/V pipeline | `qsa_sharded_decode_pipeline.log`: RMSNorm rejects splitter height sharding; legal V-retained form PCC passes | Promote legal V portion; Q/K bridge retained. |
| Cache static block RoPE | `qsa_cached_static_block_rope.log`: PCC pass, 43.877624/5.618471 ms | Promote. |
| Persistent compressed key cache | `qsa_persistent_compressed_key_cache.log`: initial trace fix, then PCC pass, 43.666625/5.540206 ms | Promote. |
| Dedicated indexer score | `qsa_indexer_score_dsa.log` + `.patch` + `.journal.jsonl`: exact reversible source, raw commands/output, PCC pass; median 43.660651/5.588385 ms | Reject: decode loses to 5.540206 ms. |

## Native FIR probe

`native_conv1d_probe.py` preserves the exact 10240-channel/T=128 adaptation.
`native_conv1d_probe_l1_full.log`, `native_conv1d_probe_dram_width8.log`,
`native_conv1d_probe_dram_width4.log`, and
`native_conv1d_probe_dram_width4_config_dram.log` show the physical blockers:
the halo requires 716800 bytes per bank with only 125952 contiguous bytes
free, and the kernel permits at most four width slices. The later KDA
causal-convolution op has a different legal contract and is the retained FIR.

## Binding-only blockers

Direct `sparse_sdpa`, `generalized_moe_gate`, `topk_router_gpt`, DeepSeek
unified routed-expert/reduce, KDA affine reduction, fused-QK rotary, and
minimal-matmul variants are documented with exact contract mismatches in
`../graph_inventory.md`. They require no device run when the binding cannot
represent the model. The final delivered decoder SHA-256 is recorded in
`../provenance_manifest.md`; all promoted paths are covered by the final
correctness, latency, watcher, stress, context, and Tracy artifacts.
