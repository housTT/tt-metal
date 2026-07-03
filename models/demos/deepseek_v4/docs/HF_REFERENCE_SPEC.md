<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4 (HF reference) — Correctness Oracle Spec for the TT-NN Port

Source: `transformers/models/deepseek_v4/modeling_deepseek_v4.py` (transformers 5.11, generated from `modular_deepseek_v4.py`) + `configuration_deepseek_v4.py`. Line numbers refer to `modeling_deepseek_v4.py`. This is the spec each TT-NN module must match at PCC ≥ 0.999 (module) / 0.99 (e2e).

> **This architecture is materially more novel than the original BRINGUP_PLAN assumed.** Eight+ V4-specific mechanisms, several with NO tt-metal prior art (mHC residual, CSA/HCA compressors, lightning indexer, grouped o_lora, attention sinks, hash routing). The `deepseek_v3` demos are a partial reference at best — the MLA shape differs (K==V shared MQA, no kv_lora_rank), and mHC/compression/indexer are entirely new.

## 0. Config (V4-Flash)
hidden=4096, layers=43, heads=64, kv_heads=1 (shared-KV MQA), head_dim=512, q_lora_rank=1024.
`qk_rope_head_dim` derived: `partial_rotary_factor=64/512` → 64. **No `kv_lora_rank`** (that was V3); KV compressed to a single 512-dim MQA head.
MoE: moe_intermediate=2048, n_routed_experts=256, top_k=6, n_shared=1, scoring_func=`sqrtsoftplus`, norm_topk_prob=True, routed_scaling_factor=1.5.
swiglu_limit=10.0, sliding_window=128, o_groups=8, o_lora_rank=1024.
Indexer: index_n_heads=64, index_head_dim=128, index_topk=512.
mHC: hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6.
rope_theta=10000 (main), compress_rope_theta=160000 (compress/YaRN).

### Layer schedule (config `__post_init__`)
- `layer_types[i]` ∈ {`sliding_attention`, `compressed_sparse_attention`(CSA,rate 4), `heavily_compressed_attention`(HCA,rate 128)}. Default: first 2 = HCA bootstrap, then interleave. (Probe shows Flash: layers 0-1 sliding, then alternating CSA/HCA.)
- `mlp_layer_types[i]` ∈ {`hash_moe` (first num_hash_layers=3), `moe` (rest)}. **Every layer is MoE + shared expert; there are NO pure-dense MLP layers.**

## 1. Attention — `DeepseekV4Attention` (751-869), MLA-style shared-KV MQA
Weights: `q_a_proj`(4096→1024), `q_a_norm`(RMSNorm 1024), `q_b_proj`(1024→64*512), `q_b_norm`(**Unweighted**RMSNorm per-head over 512), `kv_proj`(4096→512, single KV head), `kv_norm`(RMSNorm 512), `o_a_proj`(GroupedLinear §2), `o_b_proj`(8192→4096), `sinks`(Param[64]), optional `compressor`(§4). `scaling=512**-0.5`. `rope_layer_type`="main"(sliding) / "compress"(else).
forward:
1. `q_residual = q_a_norm(q_a_proj(h))` [B,S,1024] — **also feeds indexer**.
2. `q = q_b_norm(q_b_proj(q_residual).view(B,S,64,512).transpose→[B,64,S,512])`; `q = apply_rotary(q,cos,sin)`.
3. `kv = kv_norm(kv_proj(h)).view(B,S,1,512).transpose→[B,1,S,512]`; `kv = apply_rotary(kv,cos,sin)`. **K==V (same tensor).**
4. sliding cache update (keeps last sliding_window-1); if compressor: `ckv,block_bias=compressor(...)`, `kv=cat([kv,ckv],dim=2)`; extend mask with block_bias (else zero-pad).
5. `eager_attention_forward` (713-741): `repeat_kv` 1→64; `aw=(q@kᵀ)*scaling+mask`; **sink**: concat `sinks[1,64,1,1]` col onto logits, softmax(input dtype, minus rowmax), then **drop sink column** (`probs[...,:-1]`); `out=probs@V`.
- **Partial interleaved RoPE** (342-359): only trailing 64 chans rotated; `rotate_half` uses even/odd (`x1=x[0::2],x2=x[1::2]→stack(-x2,x1).flatten`); cos/sin size-32 `repeat_interleave(2)`→64; fp32.
- **Conjugate un-rotation** (864): after attn, `out = apply_rotary(out, cos, -sin)` on rope slice — removes absolute-pos rotation from V (K==V). MUST replicate.
6. Output (866-868): reshape [B,S,64*512]→[B,S,8,4096]; `o_a_proj`(grouped)→[B,S,8,1024]; flatten→[B,S,8192]; `o_b_proj`→[B,S,4096].

## 2. Grouped output `o_lora` — `DeepseekV4GroupedLinear` (303-332) [NEW]
Block-diagonal bmm: `w=weight.view(8,-1,4096).transpose(1,2)`→[8,4096,1024]; `x=x.reshape(-1,8,4096).transpose(0,1)`→[8,BS,4096]; `y=bmm(x,w).transpose(0,1)`→[BS,8,1024]. Group g = heads 8g..8g+7. Then dense `o_b_proj[4096,8192]`.

## 3. Lightning Indexer — `DeepseekV4Indexer`(462-584)+`Scorer`(446-459) [NEW]  (inside CSA only)
Builds compressed keys at head_dim=128 (same Ca/Cb windowing as CSA §4). `q=apply_rotary(q_b_proj(q_residual).view(B,S,64,128))`. Scorer: `softmax_scale=128**-0.5`, `weights_scaling=64**-0.5`; `scores=relu(q.float@ckv.floatᵀ)*softmax_scale` [B,S,64,T]; `weights=weights_proj(h)*weights_scaling` [B,S,64]; `index_scores=(scores*weights.unsqueeze(-1)).sum(dim=2)` [B,S,T]. Top-k: `top_k=min(512,T)`; `causal_threshold=(pos+1)//4`; mask future (≥threshold)→-inf; `topk.indices`; picked-but-invalid→sentinel -1. block_bias: scatter 0.0 at selected slots into a [-inf]-filled [B,1,S,T+1], drop sentinel col.
**Sliding∪sparse combine**: query attends its local sliding window (dense causal) PLUS ≤512 selected compressed entries; else -inf.

## 4. Per-layer KV compression — CSA/HCA compressors (171-301,362-443,587-698) [NEW]
`compress_rates={CSA:4, HCA:128}`. Every attn layer also runs sliding K=V branch; compressor adds long-range entries.
- **HCA** (rate 128, non-overlapping): weights kv_proj(4096→512),gate_proj(4096→512),position_bias[128,512],kv_norm(512),rope(compress). `chunk_kv.view(B,nwin,128,512)`; `chunk_gate=view+position_bias`; `compressed=kv_norm((chunk_kv*softmax(chunk_gate,dim=2,fp32)).sum(2))`; rope at `i*128+first_pos`; causal block_bias `w<(pos+1)//128`. No indexer.
- **CSA** (rate 4, overlapping, WITH indexer): kv_proj(4096→1024=2*512),gate_proj(4096→1024),position_bias[4,1024]. Output = two series Ca[:512]/Cb[512:]. `chunk.view(B,nwin,4,1024)`; build new_kv[B,nwin,8,512]: `[:,:,4:]=Cb`(current), `[:,1:,:4]=prev window Ca`, window0 first-half from cache overlap; `compressed=kv_norm((new_kv*softmax(new_gate,dim=2,fp32)).sum(2))`; rope at `i*4+first_pos`; then indexer(§3)+block_bias.
- Caches (171-301): sliding K=V (last window-1); per-name buffer_kv/gate, compressed_kv, entry_count; CSA adds overlap_kv/gate (Ca carry-over).
- compress rope = YaRN θ=160000, **attention_factor forced 1.0** (no mscale). main rope = default θ=10000.

## 5. MoE routing (1029-1078)
**Sinkhorn/hc_* are NOT here — they belong to mHC §10.**
- **TopKRouter** (`moe` layers, = noaux_tc): `weight[256,4096]`, `e_score_correction_bias[256]`, score_fn=sqrtsoftplus. `logits=linear(x,weight)`; `scores=sqrtsoftplus(logits)` (=`softplus(x).sqrt()`); `indices=topk(scores+bias, 6).indices` (**bias for SELECTION only**); `weights=scores.gather(indices)` (**un-biased scores**); `weights/=weights.sum+1e-20`; return `weights*1.5, indices`.
- **HashRouter** (first 3 `hash_moe`): same but `indices=tid2eid[input_ids]` frozen [vocab,6] table; no bias.

## 6. Experts + shared (970-1027), clamped SwiGLU
- Experts (3D weights): `gate_up_proj[256,4096,4096]`, `down_proj[256,4096,2048]`. Per expert: `_apply_gate(linear(x,gate_up[e]))` then `linear(.,down[e])*weight`.
- **`_apply_gate` clamp asymmetry**: `gate,up=chunk(2); gate=gate.clamp(max=10); up=up.clamp(-10,10); return silu(gate)*up`.
- Shared `DeepseekV4MLP` (intermediate=2048): same clamp.
- Block: `out = experts(flat,idx,w).view + shared(residual)`.

## 8. RMSNorm/Embed/LMHead/RoPE
RMSNorm eps=1e-6 standard; UnweightedRMSNorm (no weight) for q_b_norm & HC input norms. Embedding nn.Embedding(129280,4096). LM head Linear(4096,129280) bias=False; tie_word_embeddings=False but `_tied_weights_keys` maps lm_head→embed — verify vs checkpoint. RoPE per-type {main,compress}, interleaved partial (dim 64). YaRN only on compress, attention_factor=1.0.

## 9. FP8 quant
weight_block_size=[128,128], ue8m0 scales; dequant = `w_fp8.to(bf16)*scale` per 128×128 block. **Keep BF16** (auto-skip): compressor/indexer kv_proj,gate_proj + indexer scorer weights_proj. **Keep fp32**: attn_hc/ffn_hc/hc_head, sinks, position_bias, e_score_correction_bias, all norms.

## 10. Manifold-Constrained Hyper-Connections (mHC) [NEW, PERVASIVE — biggest port risk]
Residual is `hc_mult=4` parallel streams: `[B,S,4,4096]` throughout. Set in Model.forward: `embeds.unsqueeze(2).expand(-1,-1,4,-1)`; collapsed by HyperHead before final norm.
- **`DeepseekV4HyperConnection`** (872-948), two per layer (attn_hc, ffn_hc). Weights: `fn[24,16384]`, `base[24]`, `scale[3]`, unweighted input_norm. All fp32:
  `flat=input_norm(streams.flatten(2))`; `pre_w,post_w,comb_w=linear(flat,fn).split([4,4,16])`; `pre=sigmoid(pre_w*ps+pb)+eps` [B,S,4]; `post=2*sigmoid(post_w*psc+pob)` [B,S,4]; `comb=softmax(comb_w.view(4,4)*csc+cb,-1)+eps` then col-norm; **Sinkhorn 20 iters** (1 col-norm + 19×[row-norm; col-norm]); `collapsed=(pre.unsqueeze(-1)*streams).sum(2)` [B,S,4096].
- **Layer application** (1125-1149): `post,comb,collapsed=attn_hc(streams)`; `attn_out=self_attn(input_layernorm(collapsed))`; `streams = post.unsqueeze(-1)*attn_out.unsqueeze(-2) + matmul(comb.transpose(-1,-2), streams)`; then same with ffn_hc + mlp(post_attention_layernorm(collapsed)). `comb.T @ streams` = `Σ_j comb[j,k]*streams[j]` (comb non-symmetric — direction matters).
- **`DeepseekV4HyperHead`** (951-967): `hc_fn[4,16384]`,`hc_base[4]`,`hc_scale[1]`. `pre=sigmoid(linear(norm(x.flatten),hc_fn)*sc+b)+eps`; `return (pre.unsqueeze(-1)*x).sum(2)` [B,S,4096]. Then `norm` → lm_head.

## PCC gotcha checklist
1. Interleaved partial RoPE (even/odd, trailing 64, fp32). 2. Conjugate -sin un-rotation (864). 3. Attn sink in denom only, dropped from values. 4. K==V MQA broadcast 1→64. 5. Grouped o_a block-diag bmm + dense o_b. 6. CSA overlap Ca/Cb + carry state; HCA non-overlap. 7. Indexer Σ_h w·ReLU(q·k)·128^-.5, w×64^-.5, top512, (pos+1)//4 threshold, -1 sentinel scatter. 8. Router: bias for selection only, raw sqrtsoftplus for weights, /sum, ×1.5. 9. SwiGLU clamp asymmetry (gate max=10; up [-10,10]). 10. mHC 4-stream + Sinkhorn(20), comb.T direction, fp32. 11. YaRN compress attention_factor=1.0. 12. fp8 skip lists.
