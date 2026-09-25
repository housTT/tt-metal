# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate the attention+compressor composition (increment 4 integration): on-device
hca_compress_device -> mla_decode_device(ckv=...) vs a host reference that concats the
compressed KV onto main K==V before the sink softmax. HCA layer (block_bias is None)."""
import argparse

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import compressors as C
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
    w = weight.view(g, rank, ipg).transpose(1, 2).contiguous()
    xg = gflat.reshape(-1, g, ipg).transpose(0, 1).contiguous()
    y = torch.bmm(xg, w).transpose(0, 1)
    return y.reshape(*gflat.shape[:-1], g * rank)


def host_ref(new_ln, kv_cache, ckv_ref, attn, cfg, cos, sin):
    B, _, H = new_ln.shape
    nh, hd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    scaling = attn.scaling
    new_ln = new_ln.float()
    q_res = rms(new_ln @ attn.q_a_proj.weight.data.float().t(), attn.q_a_norm.weight.data.float(), eps)
    q = (q_res @ attn.q_b_proj.weight.data.float().t()).view(B, 1, nh, hd).transpose(1, 2)
    q = A.apply_rope(rms(q, None, eps), cos, sin)
    kv_new = rms(new_ln @ attn.kv_proj.weight.data.float().t(), attn.kv_norm.weight.data.float(), eps).view(B, 1, 1, hd).transpose(1, 2)
    kv_new = A.apply_rope(kv_new, cos, sin)
    main_kv = torch.cat([kv_cache.float(), kv_new], dim=2)
    kv_full = torch.cat([main_kv, ckv_ref.float()], dim=2)  # concat compressed entries
    Lkv = kv_full.shape[2]
    k = kv_full.expand(B, nh, Lkv, hd)
    aw = torch.matmul(q.float(), k.transpose(2, 3)) * scaling
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, nh, 1, 1).float()
    combined = torch.cat([aw, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    scores = torch.softmax(combined, dim=-1)[..., :-1]
    ao = torch.matmul(scores, k).to(torch.float32)
    ao = A.apply_rope(ao, cos, -sin).transpose(1, 2).contiguous().reshape(B, 1, nh * hd)
    grouped = grouped_linear_ref(ao, attn.o_a_proj.weight.data.float(), cfg.o_groups)
    return grouped @ attn.o_b_proj.weight.data.float().t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--past", type=int, default=5)
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
    attn = None
    for layer in scratch.model.layers:
        c = getattr(layer.self_attn, "compressor", None)
        if c is not None and type(c).__name__.startswith("DeepseekV4HCA"):
            attn = layer.self_attn
            break
    assert attn is not None
    torch.manual_seed(0)
    for p in attn.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)
    attn.sinks.data = torch.randn_like(attn.sinks.data) * 0.5

    H, nh, hd, rd = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, cfg.qk_rope_head_dim
    hca = attn.compressor
    m = hca.compress_rate
    B, S = 1, args.seq
    hbuf = torch.randn(B, S, H) * 0.1  # buffer for compressor
    new_ln = hbuf[:, -1:, :]  # the new token's collapsed_ln is the last buffer row
    kv_cache = torch.randn(1, 1, args.past, hd) * 0.1
    nw = S // m
    pos = args.past

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ckv_ref = C.hca_compressor(hbuf, hca, cfg, dev)  # [B,1,nw,hd]
        dummy = torch.zeros(1, pos + 1, H)
        cos_all, sin_all = scratch.model.rotary_emb(dummy, position_ids=torch.arange(pos + 1).unsqueeze(0), layer_type="main")
        cos, sin = cos_all[:, pos : pos + 1, :], sin_all[:, pos : pos + 1, :]
        ref_out = host_ref(new_ln, kv_cache, ckv_ref, attn, cfg, cos, sin)

        W = D.MLAv4DeviceWeights(attn, cfg, dev)
        Wc = D.HCACompressorDevice(hca, cfg, dev, R=W.R)
        cpos = (torch.arange(nw) * m).unsqueeze(0)
        cco, csi = hca.rotary_emb(hbuf, position_ids=cpos, layer_type="compress")
        cos_c = D._dev(cco.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        sin_c = D._dev(csi.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        hbuf_d = D._dev(hbuf, dev)
        ckv_d = D.hca_compress_device(hbuf_d, Wc, cos_c, sin_c, dev)
        cosm = D._dev(cos.repeat_interleave(2, -1).reshape(1, 1, 1, rd), dev)
        sinm = D._dev(sin.repeat_interleave(2, -1).reshape(1, 1, 1, rd), dev)
        new_ln_d = D._dev(new_ln, dev)
        kv_cache_d = D._dev(kv_cache, dev)
        out_d, _ = D.mla_decode_device(new_ln_d, kv_cache_d, W, cosm, sinm, dev, ckv=ckv_d)
        out = ttnn.to_torch(out_d)
        p = pcc(ref_out, out)
        print(f"[PCC] attn+compressor output {p:.5f}", flush=True)
        print("ATTN_COMP_OK" if p > 0.97 else f"ATTN_COMP_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
