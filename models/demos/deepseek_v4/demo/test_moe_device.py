# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate on-device MoE (tt/mla_v4_device.moe_device, device fp32-accum) vs host
model.sparse_moe. Increment 3 MoE assembly. Single decode token."""
import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt import model as MODEL


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
    mlp = None
    for layer in scratch.model.layers:
        if not getattr(layer.mlp, "is_hash", False):
            mlp = layer.mlp
            break
    assert mlp is not None, "no moe (non-hash) layer"
    torch.manual_seed(0)
    for p in mlp.parameters():
        p.data = (torch.randn_like(p.data) * 0.02).to(p.data.dtype)

    H = cfg.hidden_size
    cln = torch.randn(1, 1, H) * 0.1
    input_ids = torch.tensor([[7]])  # unused for non-hash routing

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ref = MODEL.sparse_moe(cln, mlp, input_ids, cfg, dev)  # [1,1,H] host
        out_d = D.moe_device(None, cln, mlp, cfg, dev)
        out = ttnn.to_torch(out_d)
        p = pcc(ref, out)
        print(f"[PCC] moe_device vs sparse_moe {p:.5f}", flush=True)
        print("MOE_DEVICE_OK" if p > 0.97 else f"MOE_DEVICE_LOWPCC {p:.4f}", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
