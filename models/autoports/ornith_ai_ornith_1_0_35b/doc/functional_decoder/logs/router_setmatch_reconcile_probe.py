# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile the two router set-agreement figures in the evidence tree.

`router_precision_ab.txt` reports 99.8 % top-8 set agreement for the fp32-logit router;
`probe_moe_vs_hf.txt` reports 95.9 % at the same T and seed. The A/B reads the selected set from the
raw `ttnn.topk` **indices**; the MoE probe reads it from `torch.nonzero` of the **bf16 dense score
vector**. This probe measures the shipped `OrnithMoE.routing_weights` and reports the set both ways at
once, so the two numbers can be compared on one run of one code path.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/logs/router_setmatch_reconcile_probe.py
"""

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import moe as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig

mp = R.resolve_model_path()
cfg_hf = R.load_text_config(mp)
cfg = OrnithDecoderConfig.from_hf_config(cfg_hf)
sd = R.load_layer_state_dict(0, mp)
mlp_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}
with torch.device("meta"):
    ref = Qwen3_5MoeSparseMoeBlock(cfg_hf)
ref.to_empty(device="cpu")
ref.load_state_dict({k: v.float() for k, v in mlp_sd.items()}, strict=False, assign=True)
ref.eval()

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
w = M.load_moe_weights(mesh, cfg, sd)
moe = M.OrnithMoE(mesh, cfg, w)
gate_w = mlp_sd["gate.weight"].float()

for scale in (0.5, 0.6):
    torch.manual_seed(0)
    T = 512
    x = (torch.randn(1, T, cfg.dim) * scale).to(torch.bfloat16)
    logits = x.float().reshape(-1, cfg.dim) @ gate_w.T
    p = torch.softmax(logits, -1, dtype=torch.float32)
    tv, ti = torch.topk(p, 8, -1)
    tvn = tv / tv.sum(-1, keepdim=True)
    rd = torch.zeros(T, cfg.num_experts)
    rd.scatter_(1, ti, tvn)

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, T, cfg.dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dense = ttnn.to_torch(moe.routing_weights(x_tt)).reshape(T, cfg.num_experts).float()

    # (a) the set the scatter destination reports, i.e. what the sparsity mask consumes
    by_nonzero = sum(1 for m in range(T) if set(torch.nonzero(dense[m]).flatten().tolist()) == set(ti[m].tolist()))
    # (b) the set the router actually selected: the 8 largest entries, whether or not the smallest
    #     of them survived the bf16 round-trip as a non-zero
    by_topk = sum(1 for m in range(T) if set(torch.topk(dense[m], 8).indices.tolist()) == set(ti[m].tolist()))
    # how often bf16 flattened one or more of the eight selected weights to exactly zero
    fewer = sum(1 for m in range(T) if int((dense[m] != 0).sum()) < 8)
    dropped_mass = float(rd[dense == 0].sum() / rd.sum())
    relerr = float((dense - rd).abs().sum() / rd.abs().sum())
    print(
        f"RECONCILE scale={scale} T={T} setmatch_by_nonzero={by_nonzero}/{T} ({100*by_nonzero/T:.1f}%) "
        f"setmatch_by_topk={by_topk}/{T} ({100*by_topk/T:.1f}%) rows_with_fewer_than_8_nonzero={fewer} "
        f"HF_weight_mass_on_zeroed_entries={dropped_mass:.3g} score_L1_relerr={relerr:.4g}"
    )

ttnn.close_mesh_device(mesh)
print("RECONCILE DONE")
