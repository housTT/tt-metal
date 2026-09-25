# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC-validate the on-device HCA compressor (tt/mla_v4_device.hca_compress_device) against the
host reference tt/compressors.hca_compressor. Increment 2a of task #36."""
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

    hca = None
    for layer in scratch.model.layers:
        c = getattr(layer.self_attn, "compressor", None)
        if c is not None and type(c).__name__.startswith("DeepseekV4HCA"):
            hca = c
            break
    assert hca is not None, "no HCA layer"
    torch.manual_seed(0)
    for p in hca.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)

    B, S, H = 1, args.seq, cfg.hidden_size
    hbuf = torch.randn(B, S, H) * 0.1
    m, hd = hca.compress_rate, hca.head_dim
    nw = S // m

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        # host reference
        ref = C.hca_compressor(hbuf, hca, cfg, dev)  # [B,1,nw,hd]
        # device compress cos/sin at positions arange(nw)*m
        positions = (torch.arange(nw) * m).unsqueeze(0)  # [1,nw]
        cos, sin = hca.rotary_emb(hbuf, position_ids=positions, layer_type="compress")  # [1,nw,rd/2]
        rd = cfg.qk_rope_head_dim
        cos_c = D._dev(cos.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        sin_c = D._dev(sin.repeat_interleave(2, -1).reshape(1, 1, nw, rd), dev)
        Wc = D.HCACompressorDevice(hca, cfg, dev)
        hbuf_d = D._dev(hbuf, dev)
        ckv_d = D.hca_compress_device(hbuf_d, Wc, cos_c, sin_c, dev)
        ckv = ttnn.to_torch(ckv_d)
        print(f"[shapes] ref {tuple(ref.shape)} dev {tuple(ckv.shape)} (nw={nw})", flush=True)
        p = pcc(ref, ckv)
        print(f"[PCC] hca_compressor {p:.5f}", flush=True)
        print("HCA_DEVICE_OK" if p > 0.97 else f"HCA_DEVICE_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
