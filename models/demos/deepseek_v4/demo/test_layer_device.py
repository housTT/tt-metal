# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Capstone: validate the FULL assembled on-device decoder layer (mla_v4_device.decode_layer_device)
against the host path (M.hyperconnection + pure-torch attn⊕compressor + model.sparse_moe).
HCA+moe layer, single decode token. Increment 4a of task #36."""
import argparse

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import attention as A
from models.demos.deepseek_v4.tt import compressors as C
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt import model as MODEL
from models.demos.deepseek_v4.tt import modules as M


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rms(x, w, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def gl_ref(gflat, weight, g):
    ipg = weight.shape[1]
    rank = weight.shape[0] // g
    w = weight.view(g, rank, ipg).transpose(1, 2).contiguous()
    xg = gflat.reshape(-1, g, ipg).transpose(0, 1).contiguous()
    return torch.bmm(xg, w).transpose(0, 1).reshape(*gflat.shape[:-1], g * rank)


def host_attn(new_ln, kv_cache, ckv, attn, cfg, cos, sin):
    B, _, H = new_ln.shape
    nh, hd, eps = cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    new_ln = new_ln.float()
    q_res = rms(new_ln @ attn.q_a_proj.weight.data.float().t(), attn.q_a_norm.weight.data.float(), eps)
    q = (q_res @ attn.q_b_proj.weight.data.float().t()).view(B, 1, nh, hd).transpose(1, 2)
    q = A.apply_rope(rms(q, None, eps), cos, sin)
    kvn = rms(new_ln @ attn.kv_proj.weight.data.float().t(), attn.kv_norm.weight.data.float(), eps).view(B, 1, 1, hd).transpose(1, 2)
    kvn = A.apply_rope(kvn, cos, sin)
    kv_full = torch.cat([torch.cat([kv_cache.float(), kvn], dim=2), ckv.float()], dim=2)
    Lkv = kv_full.shape[2]
    k = kv_full.expand(B, nh, Lkv, hd)
    aw = torch.matmul(q.float(), k.transpose(2, 3)) * attn.scaling
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, nh, 1, 1).float()
    comb = torch.cat([aw, sinks], dim=-1)
    comb = comb - comb.max(dim=-1, keepdim=True).values
    sc = torch.softmax(comb, dim=-1)[..., :-1]
    ao = torch.matmul(sc, k).to(torch.float32)
    ao = A.apply_rope(ao, cos, -sin).transpose(1, 2).contiguous().reshape(B, 1, nh * hd)
    grp = gl_ref(ao, attn.o_a_proj.weight.data.float(), cfg.o_groups)
    return (grp @ attn.o_b_proj.weight.data.float().t()), kvn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--past", type=int, default=None)
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
    layer = None
    for L in scratch.model.layers:
        c = getattr(L.self_attn, "compressor", None)
        if c is not None and type(c).__name__.startswith("DeepseekV4HCA") and not getattr(L.mlp, "is_hash", False):
            layer = L
            break
    assert layer is not None, "no HCA+moe layer"
    torch.manual_seed(0)
    for p in layer.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)
    layer.self_attn.sinks.data = torch.randn_like(layer.self_attn.sinks.data) * 0.5

    HC, H, hd, rd = cfg.hc_mult, cfg.hidden_size, cfg.head_dim, cfg.qk_rope_head_dim
    m = layer.self_attn.compressor.compress_rate
    S = args.seq
    past = args.past if args.past is not None else S - 1
    streams = torch.randn(1, 1, HC, H) * 0.1
    hbuf_past = torch.randn(1, past, H) * 0.1  # S_past collapsed_ln rows
    kv_cache = torch.randn(1, 1, past, hd) * 0.1
    pos = past

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        attn = layer.self_attn
        # ---- host reference ----
        post, comb, collapsed = M.hyperconnection(streams, layer.attn_hc, dev)
        cln = rms(collapsed.float(), layer.input_layernorm.weight.data.float(), cfg.rms_norm_eps)
        hbuf_full = torch.cat([hbuf_past, cln], dim=1)
        ckv = C.hca_compressor(hbuf_full, attn.compressor, cfg, dev)
        dummy = torch.zeros(1, pos + 1, H)
        cos_all, sin_all = scratch.model.rotary_emb(dummy, position_ids=torch.arange(pos + 1).unsqueeze(0), layer_type="main")
        cosm, sinm = cos_all[:, pos : pos + 1, :], sin_all[:, pos : pos + 1, :]
        ao, _ = host_attn(cln, kv_cache, ckv, attn, cfg, cosm, sinm)
        st = torch.tensor(post).float().unsqueeze(-1) * ao.float().unsqueeze(-2) + torch.matmul(
            torch.tensor(comb).float().transpose(-1, -2), streams.float())
        post2, comb2, collapsed2 = M.hyperconnection(st, layer.ffn_hc, dev)
        cln2 = rms(collapsed2.float(), layer.post_attention_layernorm.weight.data.float(), cfg.rms_norm_eps)
        mlp_out = MODEL.sparse_moe(cln2, layer.mlp, torch.tensor([[7]]), cfg, dev)
        st2 = post2.float().unsqueeze(-1) * mlp_out.float().unsqueeze(-2) + torch.matmul(
            comb2.float().transpose(-1, -2), st.float())

        # ---- device ----
        LW = D.LayerDeviceWeights(layer, cfg, dev)
        nw = (pos + 1) // m
        cpos = (torch.arange(nw) * m).unsqueeze(0)
        cco, csi = attn.compressor.rotary_emb(dummy, position_ids=cpos, layer_type="compress")
        cos_c = D._dev(cco.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        sin_c = D._dev(csi.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        cos_md = D._dev(cosm.repeat_interleave(2, -1).reshape(1, 1, 1, rd), dev)
        sin_md = D._dev(sinm.repeat_interleave(2, -1).reshape(1, 1, 1, rd), dev)
        streams_d = D._dev(streams, dev)
        hbuf_d = D._dev(hbuf_past, dev)
        kv_d = D._dev(kv_cache, dev)
        out_d, _, _ = D.decode_layer_device(streams_d, hbuf_d, kv_d, LW, cos_md, sin_md, cos_c, sin_c, dev)
        out = ttnn.to_torch(out_d)
        p = pcc(st2, out)
        print(f"[PCC] full layer streams {p:.5f}  (shapes ref {tuple(st2.shape)} dev {tuple(out.shape)})", flush=True)
        print("LAYER_DEVICE_OK" if p > 0.97 else f"LAYER_DEVICE_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
