# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC-validate on-device mHC (tt/mla_v4_device.mhc_device) vs host modules.hyperconnection.
Increment 3 (mHC part) of task #36. Decode: S=1."""
import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt import modules as M


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
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
    hc = scratch.model.layers[0].attn_hc
    torch.manual_seed(0)
    for p in hc.parameters():
        p.data = (torch.randn_like(p.data) * 0.05).to(p.data.dtype)

    HC, H = cfg.hc_mult, cfg.hidden_size
    streams = torch.randn(1, 1, HC, H) * 0.1

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        post_r, comb_r, collapsed_r = M.hyperconnection(streams, hc, dev)
        W = D.MHCDevice(hc, cfg, dev)
        streams_d = D._dev(streams, dev)
        post_d, comb_d, collapsed_d = D.mhc_device(streams_d, W)
        post = ttnn.to_torch(post_d).reshape(post_r.shape)
        comb = ttnn.to_torch(comb_d).reshape(comb_r.shape)
        collapsed = ttnn.to_torch(collapsed_d).reshape(collapsed_r.shape)
        pp, pc, pcol = pcc(post_r, post), pcc(comb_r, comb), pcc(collapsed_r, collapsed)
        print(f"[PCC] post {pp:.5f}  comb {pc:.5f}  collapsed {pcol:.5f}", flush=True)
        ok = pp > 0.97 and pc > 0.97 and pcol > 0.97
        print("MHC_DEVICE_OK" if ok else f"MHC_DEVICE_LOWPCC post={pp:.3f} comb={pc:.3f} col={pcol:.3f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
