# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC-validate the on-device MLA-v4 sliding decode (tt/mla_v4_device.py) against a pure-torch
reference mirroring tt/attention.py's math. Increment 1 of task #36 (traced all-device decode).

Uses a real sliding-attention layer module (randomized weights isolate the math from weight
loading). Compares (output, kv_new) for one new token attending a small past KV cache."""
import argparse

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import mla_v4_device as D


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rms(x, w, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def grouped_linear_ref(gflat, weight, g):
    ipg = weight.shape[1]
    rank = weight.shape[0] // g
    w = weight.view(g, rank, ipg).transpose(1, 2).contiguous()  # [g,ipg,rank]
    xg = gflat.reshape(-1, g, ipg).transpose(0, 1).contiguous()  # [g,N,ipg]
    y = torch.bmm(xg, w).transpose(0, 1)  # [N,g,rank]
    return y.reshape(*gflat.shape[:-1], g * rank)


def host_ref(new_ln, kv_cache, attn, cfg, cos, sin):
    """Pure torch, mirrors kv_cache_decode.mla_attention_decode (sliding, no compressor)."""
    B, _, H = new_ln.shape
    nh, hd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    scaling = attn.scaling
    new_ln = new_ln.float()
    kv_cache = kv_cache.float() if kv_cache is not None else None
    q_res = new_ln @ attn.q_a_proj.weight.data.float().t()
    q_res = rms(q_res, attn.q_a_norm.weight.data.float(), eps)
    q = (q_res @ attn.q_b_proj.weight.data.float().t()).view(B, 1, nh, hd).transpose(1, 2)  # [B,nh,1,hd]
    q = rms(q, None, eps)
    q = A.apply_rope(q, cos, sin)
    kv_new = new_ln @ attn.kv_proj.weight.data.float().t()
    kv_new = rms(kv_new, attn.kv_norm.weight.data.float(), eps).view(B, 1, 1, hd).transpose(1, 2)  # [B,1,1,hd]
    kv_new = A.apply_rope(kv_new, cos, sin)
    main_kv = kv_new if kv_cache is None else torch.cat([kv_cache, kv_new], dim=2)  # [B,1,Lkv,hd]
    Lkv = main_kv.shape[2]
    k = main_kv.expand(B, nh, Lkv, hd)
    v = k
    aw = torch.matmul(q.float(), k.float().transpose(2, 3)) * scaling  # [B,nh,1,Lkv]
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, nh, 1, 1).float()
    combined = torch.cat([aw, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)
    scores = probs[..., :-1]
    attn_out = torch.matmul(scores, v.float()).to(new_ln.dtype)  # [B,nh,1,hd]
    attn_out = A.apply_rope(attn_out, cos, -sin)
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, 1, nh * hd)
    grouped = grouped_linear_ref(attn_out, attn.o_a_proj.weight.data.float(), cfg.o_groups)
    output = grouped @ attn.o_b_proj.weight.data.float().t()
    return output, kv_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--past", type=int, default=5)  # past KV length
    args = ap.parse_args()
    from transformers import AutoConfig

    snap = RW.find_snapshot()
    cfg = AutoConfig.from_pretrained(snap)
    cfg.num_nextn_predict_layers = 0
    scfg = AutoConfig.from_pretrained(snap)
    scfg.num_hidden_layers = 5
    scfg.num_nextn_predict_layers = 0
    scfg.layer_types = scfg.layer_types[:5]
    scfg.mlp_layer_types = scfg.mlp_layer_types[:5]
    scratch = RW.build_scratch(scfg)

    # find a sliding-attention layer (compressor is None)
    sl = None
    for layer in scratch.model.layers:
        if getattr(layer.self_attn, "compressor", None) is None:
            sl = layer
            break
    assert sl is not None, "no sliding layer in scratch"
    attn = sl.self_attn
    H, nh, hd, rd = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim

    torch.manual_seed(0)
    # randomize weights (small) so the math is well-conditioned
    for p in attn.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)
    if hasattr(attn, "sinks"):
        attn.sinks.data = torch.randn_like(attn.sinks.data) * 0.5

    new_ln = torch.randn(1, 1, H) * 0.1
    kv_cache = torch.randn(1, 1, args.past, hd) * 0.1  # post-rope past K==V
    # rope cos/sin for the new position (shape [1,1,rd/2] as apply_rope expects)
    pos = args.past
    dummy = torch.zeros(1, pos + 1, H)
    cos_all, sin_all = scratch.model.rotary_emb(dummy, position_ids=torch.arange(pos + 1).unsqueeze(0), layer_type="main")
    cos = cos_all[:, pos : pos + 1, :]
    sin = sin_all[:, pos : pos + 1, :]

    ref_out, ref_kv = host_ref(new_ln, kv_cache, attn, cfg, cos, sin)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        W = D.MLAv4DeviceWeights(attn, cfg, dev)
        cos64 = cos.repeat_interleave(2, dim=-1).reshape(1, 1, 1, rd)
        sin64 = sin.repeat_interleave(2, dim=-1).reshape(1, 1, 1, rd)
        cos_d = D._dev(cos64, dev)
        sin_d = D._dev(sin64, dev)
        new_ln_d = D._dev(new_ln, dev)
        kv_cache_d = D._dev(kv_cache, dev)
        out_d, kv_d = D.mla_decode_device(new_ln_d, kv_cache_d, W, cos_d, sin_d, dev)
        out = ttnn.to_torch(out_d)
        kv_out = ttnn.to_torch(kv_d)[:, :, -1:, :]  # the new token's kv
        print(f"[PCC] output   {pcc(ref_out, out):.5f}  (shape ref {tuple(ref_out.shape)} dev {tuple(out.shape)})", flush=True)
        print(f"[PCC] kv_new   {pcc(ref_kv, kv_out):.5f}", flush=True)
        p = pcc(ref_out, out)
        print("MLA_V4_DEVICE_OK" if p > 0.97 else f"MLA_V4_DEVICE_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
