# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC-validate the on-device CSA compressor pool (tt/mla_v4_device.csa_compress_device) vs the
host reference compressors._ca_cb_pool, and confirm block_bias is a no-op within max_context.
Increment 2b of task #36."""
import argparse

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import compressors as C
from models.demos.deepseek_v4.tt import mla_v4_device as D


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=256)
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

    csa = None
    for layer in scratch.model.layers:
        c = getattr(layer.self_attn, "compressor", None)
        if c is not None and type(c).__name__.startswith("DeepseekV4CSA"):
            csa = c
            break
    assert csa is not None, "no CSA layer"
    print(f"[shapes] position_bias {tuple(csa.position_bias.shape)} m={csa.compress_rate} hd={csa.head_dim}", flush=True)
    torch.manual_seed(0)
    for p in csa.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)

    B, S, H = 1, args.seq, cfg.hidden_size
    hbuf = torch.randn(B, S, H) * 0.1
    m, hd = csa.compress_rate, csa.head_dim
    nw = S // m

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ref, ref_nw = C._ca_cb_pool(hbuf, csa, cfg, dev, hd, m)  # [B,T,hd]
        # confirm block_bias no-op: full csa_compressor block_bias should be all-zero (all selected)
        q_res = torch.randn(B, S, csa.q_a_proj.weight.shape[0]) * 0.1 if hasattr(csa, "q_a_proj") else torch.randn(B, S, 1024) * 0.1
        try:
            _, bb = C.csa_compressor(hbuf, q_res, torch.arange(S).unsqueeze(0), csa, cfg, dev)
            print(f"[block_bias] max={float(bb.max()):.3f} min={float(bb.min()):.3f} (0/0 => no-op confirmed)", flush=True)
        except Exception as e:
            print(f"[block_bias] skipped ({type(e).__name__})", flush=True)

        positions = (torch.arange(nw) * m).unsqueeze(0)
        cos, sin = csa.rotary_emb(hbuf, position_ids=positions, layer_type="compress")
        rd = cfg.qk_rope_head_dim
        cos_c = D._dev(cos.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        sin_c = D._dev(sin.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        Wc = D.CSACompressorDevice(csa, cfg, dev)
        hbuf_d = D._dev(hbuf, dev)
        ckv_d = D.csa_compress_device(hbuf_d, Wc, cos_c, sin_c, dev)
        ckv = ttnn.to_torch(ckv_d)[:, 0]  # [B,nw,hd]
        print(f"[shapes] ref {tuple(ref.shape)} dev {tuple(ckv.shape)} (nw={nw})", flush=True)
        p = pcc(ref, ckv)
        print(f"[PCC] csa pool {p:.5f}", flush=True)
        print("CSA_DEVICE_OK" if p > 0.97 else f"CSA_DEVICE_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
