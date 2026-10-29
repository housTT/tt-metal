"""Prefill matmuls on DRAM-width-sharded reduced-precision weights, SiLU fusion, pad/update-cache contracts."""
import json
import math
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.common.utility_functions import comp_pcc

REP = 4
REPLAYS = 3
OUT = "/tmp/qwen3_opt/exp3.json"
rows = []
state = weight_source.load_layer_state(0)
emb = weight_source.load_embedding()
tok = json.loads(
    open(
        "/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/functional_decoder/layer_inputs/tale_token_ids.json"
    ).read()
)["tokens"]
x = emb[torch.tensor(tok[:1024])].float()
ln_w = state["input_layernorm.weight"].float()


def rms(v, w, eps=1e-6):
    return w * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps))


xn = rms(x, ln_w)
W = {
    "qkv": torch.cat([state[f"self_attn.{n}_proj.weight"].float().T for n in ("q", "k", "v")], dim=-1),
    "gate": state["mlp.gate_proj.weight"].float().T,
    "down": state["mlp.down_proj.weight"].float().T,
}
ACT = {
    "qkv": xn,
    "gate": xn,
    "down": torch.nn.functional.silu(xn @ W["gate"]) * (xn @ state["mlp.up_proj.weight"].float().T),
}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
NB = mesh.dram_grid_size().x
DRAM_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NB - 1, 0))])


def dram_sharded_cfg(k, n):
    padded = math.ceil(n / (32 * NB)) * 32 * NB
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_GRID, (k, padded // NB), ttnn.ShardOrientation.ROW_MAJOR),
    )


def ckc(fid, fp32, l1acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )


def pcc(a, b):
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


def time_trace(fn):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REP):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    return best * 1e6 / REP


def record(row):
    print(json.dumps(row), flush=True)
    rows.append(row)


# 1. prefill minimal_matmul on DRAM-sharded vs interleaved weights, reduced precision
for role in [r for r in os.getenv("EXP3_ROLES", "qkv,gate,down").split(",") if r in W]:
    k, n = W[role].shape
    ref = ACT[role] @ W[role]
    x_t = ttnn.from_torch(
        ACT[role].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for dtype in os.getenv("EXP3_DTYPES", "bf16,bfp8,bfp4").split(","):
        for sharded in (False, True):
            w_t = ttnn.as_tensor(
                W[role].unsqueeze(0).unsqueeze(0),
                dtype=DT[dtype],
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=dram_sharded_cfg(k, n) if sharded else ttnn.DRAM_MEMORY_CONFIG,
            )
            fids = {
                "bf16": ["hifi4+t", "hifi2+t"],
                "bfp8": ["hifi2+t", "hifi2+f", "lofi+t"],
                "bfp4": ["lofi+t", "lofi+f"],
            }[dtype]
            for fidspec in fids:
                fid, fp32 = fidspec.split("+")
                cfg = ckc(fid, fp32 == "t")
                row = dict(
                    op="prefill_minimal_matmul",
                    role=role,
                    rows=1024,
                    dtype=dtype,
                    weight_sharded=sharded,
                    fidelity=fid,
                    fp32=fp32 == "t",
                )
                print("START", json.dumps(row), flush=True)
                try:
                    y = ttnn.experimental.minimal_matmul(
                        x_t, w_t, dtype=ttnn.bfloat16, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    row["pcc"] = pcc(ref, ttnn.to_torch(y)[0, 0])
                    y.deallocate(True)

                    def fn():
                        y = ttnn.experimental.minimal_matmul(
                            x_t,
                            w_t,
                            dtype=ttnn.bfloat16,
                            compute_kernel_config=cfg,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        y.deallocate(True)

                    row["us"] = time_trace(fn)
                except Exception as e:
                    row["error"] = str(e).splitlines()[0][:300]
                record(row)
            w_t.deallocate(True)
    x_t.deallocate(True)

# 2. pad front padding on row-major int32
try:
    p = ttnn.moreh_full(
        (1, 1), 37, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q = ttnn.pad(p, [(0, 0), (3, 2)], -1)
    q1 = ttnn.reshape(q, (6,))
    record(
        dict(
            op="pad_front_rowmajor_int32",
            result=ttnn.to_torch(q).tolist(),
            reshaped=list(q1.shape),
            same_buffer=q1.buffer_address() == q.buffer_address(),
        )
    )
except Exception as e:
    record(dict(op="pad_front_rowmajor_int32", error=str(e).splitlines()[0][:300]))
try:
    t = ttnn.from_torch(
        torch.arange(6, dtype=torch.int32).reshape(1, 6),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    r = ttnn.repeat(t, ttnn.Shape([5, 1]))
    record(dict(op="repeat_rowmajor_int32", shape=list(r.shape), row3=ttnn.to_torch(r)[3].tolist()))
except Exception as e:
    record(dict(op="repeat_rowmajor_int32", error=str(e).splitlines()[0][:300]))
try:
    a = ttnn.arange(37, 37 + 27, 1, device=mesh, dtype=ttnn.int32)
    record(
        dict(
            op="arange_int32", shape=list(a.shape), layout=str(a.layout), first=ttnn.to_torch(a).flatten()[:3].tolist()
        )
    )
except Exception as e:
    record(dict(op="arange_int32", error=str(e).splitlines()[0][:300]))

# 3. fused update cache with skipped users: n pseudo-users, only user i active
blocks = 4
kc = ttnn.from_torch(
    torch.zeros(blocks, 8, 32, 128).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
vc = ttnn.from_torch(
    torch.zeros(blocks, 8, 32, 128).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
n = 27
table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
k_tok = torch.randn(1, n, 8, 128).to(torch.bfloat16)
v_tok = torch.randn(1, n, 8, 128).to(torch.bfloat16)
k_cores = ttnn.num_cores_to_corerangeset(n, grid, True)
v_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(0, 3),
    n,
    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
    True,
)


def hcfg(cores):
    return ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


k_t = ttnn.from_torch(k_tok, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=hcfg(k_cores))
v_t = ttnn.from_torch(v_tok, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=hcfg(v_cores))
pt = ttnn.from_torch(
    table.repeat(n, 1),
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
positions = [5 + i for i in range(n)]
t0 = time.perf_counter()
for i in range(n):
    pos = ttnn.moreh_full(
        (1, 1),
        positions[i],
        mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_n = ttnn.reshape(ttnn.pad(pos, [(0, 0), (i, n - 1 - i)], -1), (n,))
    ttnn.experimental.paged_fused_update_cache(kc, k_t, vc, v_t, update_idxs_tensor=pos_n, page_table=pt)
    pos.deallocate(True)
    pos_n.deallocate(True)
ttnn.synchronize_device(mesh)
wall = time.perf_counter() - t0
kc_t = ttnn.to_torch(kc)
exp = torch.zeros(blocks, 8, 32, 128)
for i in range(n):
    p = positions[i]
    exp[table[0, p // 32], :, p % 32, :] = k_tok[0, i].float()
record(
    dict(
        op="fused_update_skip_users",
        n=n,
        wall_ms=wall * 1e3,
        key_equal=bool(torch.equal(exp.to(torch.bfloat16), kc_t)),
        untouched_rows_zero=bool((kc_t[table[0, 0], :, :5, :] == 0).all()),
    )
)

# 4. gate matmul with fused SiLU vs mul-with-silu (32-core width sharded, bfp4 lofi)
k, n_ = W["gate"].shape
x32 = ACT["gate"][:32]
cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
acfg = ttnn.create_sharded_memory_config(
    shape=(32, 4096 // 32),
    core_grid=cs,
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
x_t = ttnn.from_torch(
    x32.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=acfg,
)
wg = ttnn.as_tensor(
    W["gate"].unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=dram_sharded_cfg(k, n_),
)
wu = ttnn.as_tensor(
    state["mlp.up_proj.weight"].float().T.unsqueeze(0).unsqueeze(0),
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=dram_sharded_cfg(k, n_),
)
cfg = ckc("lofi", False, True)
ref = torch.nn.functional.silu(x32 @ W["gate"]) * (x32 @ state["mlp.up_proj.weight"].float().T)
for variant in ("mul_silu", "fused_silu_matmul"):
    pc_gate = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,
        per_core_M=1,
        per_core_N=12,
        fused_activation=(ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if variant == "fused_silu_matmul" else None),
        num_workers_per_dram_bank=1,
    )
    pc_up = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4, per_core_M=1, per_core_N=12, fused_activation=None, num_workers_per_dram_bank=1
    )

    def fn():
        g = ttnn.linear(
            x_t,
            wg,
            program_config=pc_gate,
            compute_kernel_config=cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        u = ttnn.linear(
            x_t,
            wu,
            program_config=pc_up,
            compute_kernel_config=cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if variant == "mul_silu":
            a = ttnn.mul(
                g,
                u,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=g.memory_config(),
            )
        else:
            a = ttnn.mul(g, u, dtype=ttnn.bfloat16, memory_config=g.memory_config())
        g.deallocate(True)
        u.deallocate(True)
        return a

    row = dict(op="gate_up_silu", variant=variant)
    try:
        a = fn()
        row["pcc"] = pcc(ref, ttnn.to_torch(a)[0, 0])
        a.deallocate(True)
        row["us"] = time_trace(lambda: fn().deallocate(True))
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:300]
    record(row)

# 5. concat heads with sub_core_grids 8x4
attn = ttnn.from_torch(
    torch.randn(1, 1, 32, 128).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=hcfg(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])),
)
try:
    c = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=32, sub_core_grids=cs)
    record(dict(op="concat_heads_subgrid", grid=str(c.memory_config().shard_spec.grid)))
except Exception as e:
    record(dict(op="concat_heads_subgrid", error=str(e).splitlines()[0][:300]))

json.dump(rows, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE", OUT)
