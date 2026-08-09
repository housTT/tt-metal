"""Is the TTNN nilpotent-doubling inverse accurate for the attn0 real weights actually produce?

Captures the real ``attn0`` matrices from HF's ``torch_chunk_gated_delta_rule`` during a
real-weight prefill, then runs the layer's doubling product on them in TTNN and compares
against HF's exact forward substitution.
"""
import math
import torch
import ttnn
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

CAPTURE = {}
_orig = M.torch_chunk_gated_delta_rule


def patched(query, key, value, g, beta, chunk_size=64, **kw):
    # replicate the reference up to attn0 / attn, then delegate
    q2 = M.l2norm(query, dim=-1, eps=1e-6) if kw.get("use_qk_l2norm_in_kernel") else query
    k2 = M.l2norm(key, dim=-1, eps=1e-6) if kw.get("use_qk_l2norm_in_kernel") else key
    qq, kk, vv, bb, gg = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q2, k2, value, beta, g)]
    sl = kk.shape[2]
    pad = (chunk_size - sl % chunk_size) % chunk_size
    kk = torch.nn.functional.pad(kk, (0, 0, 0, pad))
    bb = torch.nn.functional.pad(bb, (0, pad))
    gg = torch.nn.functional.pad(gg, (0, pad))
    k_beta = kk * bb.unsqueeze(-1)
    kk_c = kk.reshape(kk.shape[0], kk.shape[1], -1, chunk_size, kk.shape[-1])
    kb_c = k_beta.reshape(k_beta.shape[0], k_beta.shape[1], -1, chunk_size, k_beta.shape[-1])
    gc = gg.reshape(gg.shape[0], gg.shape[1], -1, chunk_size).cumsum(dim=-1)
    decay_mask = ((gc.unsqueeze(-1) - gc.unsqueeze(-2)).tril().exp().float()).tril()
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    attn0 = -((kb_c @ kk_c.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    attn = attn0.clone()
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    CAPTURE.setdefault("attn0", attn0)
    CAPTURE.setdefault("inv_ref", attn)
    return _orig(query, key, value, g, beta, chunk_size=chunk_size, **kw)


M.torch_chunk_gated_delta_rule = patched
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()
lut = H.build_layer(mesh, H.LINEAR_LAYER_IDX, max_batch=1, max_seq_len=8192, real_weights=True)
lut.ref_layer.linear_attn.chunk_gated_delta_rule = patched
hidden = ref.synthetic_hidden_states(lut.config, 1, 256, stats, seed=0)
H.reference_prefill(lut, hidden, DynamicCache(config=lut.config))

a0 = CAPTURE["attn0"][0]        # [heads, nc, 64, 64]
ivr = CAPTURE["inv_ref"][0]
heads, nc, C, _ = a0.shape
print(f"attn0 shape={tuple(a0.shape)} absmax={float(a0.abs().max()):.4f}", flush=True)
print(f"inv_ref absmax={float(ivr.abs().max()):.6e}", flush=True)

flat = a0.reshape(heads * nc, 1, C, C).contiguous()
tt = ttnn.from_torch(flat, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
eye = ttnn.from_torch(torch.eye(C).reshape(1, 1, C, C), dtype=ttnn.float32,
                      layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
inv = ttnn.add(tt, eye)
power = tt
for _ in range(int(math.log2(C)) - 1):
    power = ttnn.matmul(power, power, dtype=ttnn.float32, compute_kernel_config=cfg)
    inv = ttnn.matmul(inv, ttnn.add(power, eye), dtype=ttnn.float32, compute_kernel_config=cfg)
got = ttnn.to_torch(inv).reshape(heads, nc, C, C).to(torch.float32)

err = (got - ivr).abs()
rel = err.amax(dim=(-1, -2)) / ivr.abs().amax(dim=(-1, -2)).clamp_min(1e-30)
print(f"TTNN inv absmax={float(got.abs().max()):.6e} max_abs_err={float(err.max()):.6e} "
      f"max_rel_err={float(rel.max()):.6e} mean_rel_err={float(rel.mean()):.6e}", flush=True)
worst = int(rel.flatten().argmax())
print(f"worst head/chunk rel err {float(rel.flatten()[worst]):.4e}", flush=True)

# torch control: the same doubling product in float64 and float32
for dt in (torch.float64, torch.float32):
    A = a0.to(dt)
    I = torch.eye(C, dtype=dt)
    invt = A + I
    P = A.clone()
    for _ in range(int(math.log2(C)) - 1):
        P = P @ P
        invt = invt @ (P + I)
    print(f"torch {dt} doubling max_abs_err vs forward-subst = "
          f"{float((invt.to(torch.float32) - ivr).abs().max()):.6e}", flush=True)
ttnn.close_mesh_device(mesh)
