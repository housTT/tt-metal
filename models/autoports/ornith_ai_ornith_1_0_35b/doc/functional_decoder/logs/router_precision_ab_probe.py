"""A/B: does computing the MoE router logits in float32 improve top-8 agreement with HF
on REAL Ornith weights? Reports set agreement, score L1 error and full-MoE PCC."""
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import moe as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


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
torch.manual_seed(0)
T = 512
x = (torch.randn(1, T, cfg.dim) * 0.5).to(torch.bfloat16)
with torch.no_grad():
    y_ref = ref(x.float())
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


def routing(logit_dtype):
    lg = ttnn.linear(x_tt, w.router, dtype=logit_dtype, compute_kernel_config=moe.dense_ckc)
    vals, idxs = ttnn.topk(lg, k=8, dim=-1, sorted=True)
    wts = ttnn.softmax(vals, dim=-1, numeric_stable=True, compute_kernel_config=moe.dense_ckc)
    zeros = ttnn.zeros([1, 1, T, cfg.num_experts], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    dense = ttnn.scatter(zeros, dim=-1, index=idxs, src=ttnn.typecast(wts, ttnn.bfloat16))
    return dense, ttnn.to_torch(idxs).reshape(T, 8).long()


for name, dt in (("bf16", ttnn.bfloat16), ("fp32", ttnn.float32)):
    try:
        dense, idx = routing(dt)
        setm = sum(1 for m in range(T) if set(idx[m].tolist()) == set(ti[m].tolist()))
        dt_t = ttnn.to_torch(dense).reshape(T, cfg.num_experts).float()
        relerr = float((dt_t - rd).abs().sum() / rd.abs().sum())
        y = moe._routed_experts(x_tt, dense, T)
        shared = moe._shared_expert(x_tt)
        total = ttnn.to_torch(ttnn.add(y, shared)).reshape(1, T, cfg.dim)
        print(
            f"AB logits={name}: setmatch={setm}/{T} ({100*setm/T:.1f}%) score_L1_relerr={relerr:.4g} moe_pcc={pcc(y_ref,total):.6f}"
        )
    except Exception as e:
        print(f"AB logits={name} FAIL {str(e)[:300]}")
ttnn.close_mesh_device(mesh)
print("AB DONE")
