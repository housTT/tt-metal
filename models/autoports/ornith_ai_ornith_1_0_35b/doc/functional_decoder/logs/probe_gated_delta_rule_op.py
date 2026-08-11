import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

torch.manual_seed(0)
Nk, Nv, Dk, Dv = 16, 32, 128, 128
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
const = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)


def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


for B, T in [(1, 32), (1, 128), (1, 1024), (2, 256)]:
    q = torch.randn(B, T, Nk, Dk) * 0.5
    k = torch.randn(B, T, Nk, Dk) * 0.5
    v = torch.randn(B, T, Nv, Dv) * 0.5
    g = -torch.rand(B, T, Nv) * 0.5
    beta = torch.rand(B, T, Nv)
    qe = q.repeat_interleave(Nv // Nk, dim=2)
    ke = k.repeat_interleave(Nv // Nk, dim=2)
    o_ref, s_ref = torch_chunk_gated_delta_rule(
        qe.clone(),
        ke.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        chunk_size=64,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    try:
        o, s = ttnn.transformer.chunk_gated_delta_rule(
            l2_norm_ttnn(tt(q)),
            l2_norm_ttnn(tt(k)),
            tt(v),
            tt(g, dtype=ttnn.float32),
            tt(beta, dtype=ttnn.float32),
            initial_state=None,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            use_qk_l2norm=False,
            eye=const[0],
            tril=const[1],
            ones=const[2],
            masks=const[3],
        )
        ot = ttnn.to_torch(ttnn.to_layout(o, ttnn.TILE_LAYOUT)).reshape(B, T, Nv, Dv)
        st = ttnn.to_torch(s).reshape(B, Nv, Dk, Dv)
        print(f"PROBE B={B} T={T} o_pcc={pcc(ot,o_ref):.6f} state_pcc={pcc(st,s_ref):.6f} oshape={list(o.shape)}")
    except Exception as e:
        print(f"PROBE B={B} T={T} FAIL {str(e)[:400]}")

# non-multiple-of-32 T check
for T in [96, 100]:
    B = 1
    q = torch.randn(B, T, Nk, Dk) * 0.5
    k = torch.randn(B, T, Nk, Dk) * 0.5
    v = torch.randn(B, T, Nv, Dv) * 0.5
    g = -torch.rand(B, T, Nv) * 0.5
    beta = torch.rand(B, T, Nv)
    try:
        o, s = ttnn.transformer.chunk_gated_delta_rule(
            l2_norm_ttnn(tt(q)),
            l2_norm_ttnn(tt(k)),
            tt(v),
            tt(g, dtype=ttnn.float32),
            tt(beta, dtype=ttnn.float32),
            initial_state=None,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            use_qk_l2norm=False,
            eye=const[0],
            tril=const[1],
            ones=const[2],
            masks=const[3],
        )
        print(f"PROBE T={T} ran, oshape={list(o.shape)}")
    except Exception as e:
        print(f"PROBE T={T} FAIL {str(e)[:200]}")

# initial-state carry test: two halves vs one
B, T = 1, 128
q = torch.randn(B, T, Nk, Dk) * 0.5
k = torch.randn(B, T, Nk, Dk) * 0.5
v = torch.randn(B, T, Nv, Dv) * 0.5
g = -torch.rand(B, T, Nv) * 0.5
beta = torch.rand(B, T, Nv)
qe = q.repeat_interleave(2, dim=2)
ke = k.repeat_interleave(2, dim=2)
o_ref, s_ref = torch_chunk_gated_delta_rule(
    qe.clone(),
    ke.clone(),
    v.clone(),
    g.clone(),
    beta.clone(),
    chunk_size=64,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
half = 64
o1, s1 = ttnn.transformer.chunk_gated_delta_rule(
    l2_norm_ttnn(tt(q[:, :half])),
    l2_norm_ttnn(tt(k[:, :half])),
    tt(v[:, :half]),
    tt(g[:, :half], dtype=ttnn.float32),
    tt(beta[:, :half], dtype=ttnn.float32),
    initial_state=None,
    output_final_state=True,
    chunk_size=_FUSED_CHUNK_SIZE,
    use_qk_l2norm=False,
    eye=const[0],
    tril=const[1],
    ones=const[2],
    masks=const[3],
)
o2, s2 = ttnn.transformer.chunk_gated_delta_rule(
    l2_norm_ttnn(tt(q[:, half:])),
    l2_norm_ttnn(tt(k[:, half:])),
    tt(v[:, half:]),
    tt(g[:, half:], dtype=ttnn.float32),
    tt(beta[:, half:], dtype=ttnn.float32),
    initial_state=s1,
    output_final_state=True,
    chunk_size=_FUSED_CHUNK_SIZE,
    use_qk_l2norm=False,
    eye=const[0],
    tril=const[1],
    ones=const[2],
    masks=const[3],
)
oc = torch.cat(
    [
        ttnn.to_torch(ttnn.to_layout(o1, ttnn.TILE_LAYOUT)).reshape(B, half, Nv, Dv),
        ttnn.to_torch(ttnn.to_layout(o2, ttnn.TILE_LAYOUT)).reshape(B, half, Nv, Dv),
    ],
    dim=1,
)
print(f"PROBE carry o_pcc={pcc(oc,o_ref):.6f} state_pcc={pcc(ttnn.to_torch(s2).reshape(B,Nv,Dk,Dv),s_ref):.6f}")

# recurrent decode step from that state
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)

q1 = torch.randn(B, 1, Nk, Dk) * 0.5
k1 = torch.randn(B, 1, Nk, Dk) * 0.5
v1 = torch.randn(B, 1, Nv, Dv) * 0.5
g1 = -torch.rand(B, 1, Nv) * 0.5
b1 = torch.rand(B, 1, Nv)
o_r, s_r = torch_recurrent_gated_delta_rule(
    q1.repeat_interleave(2, dim=2),
    k1.repeat_interleave(2, dim=2),
    v1,
    g=g1,
    beta=b1,
    initial_state=s_ref,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
qd = tt(q1.repeat_interleave(2, dim=2))
kd = tt(k1.repeat_interleave(2, dim=2))
od, sd = recurrent_gated_delta_rule_decode_ttnn(
    q=qd, k=kd, v=tt(v1), beta=tt(b1), g=tt(g1), initial_state=s2, device=mesh, high_precision=True
)
print(
    f"PROBE decode o_pcc={pcc(ttnn.to_torch(od).reshape(B,1,Nv,Dv), o_r):.6f} state_pcc={pcc(ttnn.to_torch(sd).reshape(B,Nv,Dk,Dv), s_r):.6f}"
)
ttnn.close_mesh_device(mesh)
print("PROBE DONE")
