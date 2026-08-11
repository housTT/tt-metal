import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import moe as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


mp = R.resolve_model_path()
cfg_hf = R.load_text_config(mp)
cfg = OrnithDecoderConfig.from_hf_config(cfg_hf)
sd = R.load_layer_state_dict(0, mp)
mlp_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}
with torch.device("meta"):
    ref = Qwen3_5MoeSparseMoeBlock(cfg_hf)
ref.to_empty(device="cpu")
missing, unexpected = ref.load_state_dict({k: v.float() for k, v in mlp_sd.items()}, strict=False, assign=True)
print("PROBE ref load missing", sorted(missing)[:4], "unexpected", sorted(unexpected)[:4])
ref.eval()

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
w = M.load_moe_weights(mesh, cfg, sd)
moe = M.OrnithMoE(mesh, cfg, w)
torch.manual_seed(0)
for T in (32, 128, 512, 1024):
    x = (torch.randn(1, T, cfg.dim) * 0.6).to(torch.bfloat16)
    with torch.no_grad():
        y_ref = ref(x.float())
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, T, cfg.dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        y = moe.forward(x_tt)
        yt = ttnn.to_torch(y).reshape(1, T, cfg.dim)
        # also compare expert selection
        dense = ttnn.to_torch(moe.routing_weights(x_tt)).reshape(T, cfg.num_experts).float()
        logits = x.float().reshape(-1, cfg.dim) @ mlp_sd["gate.weight"].float().T
        p = torch.softmax(logits, -1, dtype=torch.float32)
        tv, ti = torch.topk(p, 8, -1)
        tvn = tv / tv.sum(-1, keepdim=True)
        rd = torch.zeros(T, cfg.num_experts)
        rd.scatter_(1, ti, tvn)
        setmatch = sum(1 for m in range(T) if set(torch.nonzero(dense[m]).flatten().tolist()) == set(ti[m].tolist()))
        print(
            f"PROBE T={T} moe_pcc={pcc(yt,y_ref):.6f} routing_setmatch={setmatch}/{T} score_relerr={float((dense-rd).abs().sum()/rd.abs().sum()):.4g}"
        )
        ttnn.deallocate(y)
    except Exception as e:
        print(f"PROBE T={T} FAIL {str(e)[:600]}")
    ttnn.deallocate(x_tt)
ttnn.close_mesh_device(mesh)
print("PROBE DONE")
