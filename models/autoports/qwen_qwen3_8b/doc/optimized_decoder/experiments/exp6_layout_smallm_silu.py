"""Layout equality probe, 32-row logical view, small-M prefill matmul kernels on DRAM-sharded BFP8 weights, SiLU-mul variants."""
import json
import math
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    dram_sharded_program_config,
    dram_sharded_weight_memory_config,
    same_layout,
    width_sharded_memory_config,
)
from models.common.utility_functions import comp_pcc

REP, REPLAYS = 4, 3
rows = []
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
NB = mesh.dram_grid_size().x


def record(r):
    print(json.dumps(r), flush=True)
    rows.append(r)


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


state = weight_source.load_layer_state(0)
wg = state["mlp.gate_proj.weight"].float().T
wd = state["mlp.down_proj.weight"].float().T

# A. layout equality and 32-row view
cfg = width_sharded_memory_config(4096, 32, grid)
x = ttnn.from_torch(
    torch.randn(1, 1, 1, 4096).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
xs = ttnn.to_memory_config(x, cfg)
w_o = ttnn.from_torch(
    torch.randn(1, 1, 4096, 4096),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=dram_sharded_weight_memory_config(4096, 4096, NB),
)
mm = ttnn.linear(
    xs,
    w_o,
    program_config=dram_sharded_program_config(DEFAULT_POLICY.o_proj, 4096, 4096, NB),
    dtype=ttnn.bfloat16,
    memory_config=cfg,
)
a, b = mm.memory_config(), cfg
record(
    dict(
        op="layout_eq",
        same_layout=same_layout(a, b),
        grid_eq=a.shard_spec.grid == b.shard_spec.grid,
        shape_eq=list(a.shard_spec.shape) == list(b.shard_spec.shape),
        orient_eq=a.shard_spec.orientation == b.shard_spec.orientation,
        layout_eq=a.memory_layout == b.memory_layout,
        buf_eq=a.buffer_type == b.buffer_type,
        grids=[str(a.shard_spec.grid), str(b.shard_spec.grid)],
        shapes=[list(a.shard_spec.shape), list(b.shard_spec.shape)],
        types=[str(type(a.shard_spec.shape)), str(type(b.shard_spec.shape))],
    )
)
try:
    v = ttnn.reshape(xs, (1, 1, 32, 4096), (1, 1, 32, 4096))
    record(
        dict(
            op="view32",
            shape=list(v.shape),
            padded=list(v.padded_shape),
            same_buffer=v.buffer_address() == xs.buffer_address(),
            sharded=v.is_sharded(),
        )
    )
    v1 = ttnn.reshape(v, (1, 1, 1, 4096), (1, 1, 32, 4096))
    record(dict(op="view1", shape=list(v1.shape), same_buffer=v1.buffer_address() == xs.buffer_address()))
except Exception as e:
    record(dict(op="view32", error=str(e).splitlines()[0][:300]))

# B. small-M prefill matmuls on DRAM-sharded BFP8 weights (gate 4096x12288, down 12288x4096)
ckc = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
for role, w in (("gate", wg), ("down", wd)):
    k, n = w.shape
    w_t = ttnn.from_torch(
        w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram_sharded_weight_memory_config(k, n, NB),
    )
    w_il = ttnn.from_torch(
        w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for m in (64, 128, 256):
        act = torch.randn(1, 1, m, k)
        ref = act[0, 0] @ w
        x_t = ttnn.from_torch(
            act.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cands = {
            "minimal_default_sharded_w": lambda: ttnn.experimental.minimal_matmul(
                x_t, w_t, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            "minimal_default_interleaved_w": lambda: ttnn.experimental.minimal_matmul(
                x_t, w_il, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            "linear_auto_sharded_w": lambda: ttnn.linear(
                x_t, w_t, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            "linear_auto_interleaved_w": lambda: ttnn.linear(
                x_t, w_il, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        }
        mt, kt, nt = m // 32, k // 32, n // 32
        for cols, rws in ((11, 10), (8, 8), (11, 4)):
            per_core_n = math.ceil(nt / (cols * rws))
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cols, rws),
                in0_block_w=4 if kt % 4 == 0 else 2,
                out_subblock_h=1,
                out_subblock_w=max(i for i in (4, 2, 1) if per_core_n % i == 0),
                per_core_M=mt,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            cands[f"linear_1d_{cols}x{rws}_sharded_w"] = lambda pc=pc: ttnn.linear(
                x_t,
                w_t,
                program_config=pc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cands[f"linear_1d_{cols}x{rws}_interleaved_w"] = lambda pc=pc: ttnn.linear(
                x_t,
                w_il,
                program_config=pc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for mk, nk in ((1, 4), (2, 4), (4, 8)):
            mc = ttnn.MinimalMatmulConfig(
                M_block_size=mk,
                K_block_size=4,
                N_block_size=nk,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            )
            cands[f"minimal_cfg_m{mk}_k4_n{nk}_sharded_w"] = lambda mc=mc: ttnn.experimental.minimal_matmul(
                x_t,
                w_t,
                config=mc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for name, fn in cands.items():
            r = dict(op="small_m_matmul", role=role, rows=m, variant=name)
            try:
                y = fn()
                r["pcc"] = pcc(ref, ttnn.to_torch(y)[0, 0])
                y.deallocate(True)
                r["us"] = time_trace(lambda: fn().deallocate(True))
            except Exception as e:
                r["error"] = str(e).splitlines()[0][:200]
            record(r)
        x_t.deallocate(True)
    w_t.deallocate(True)
    w_il.deallocate(True)

# C. SiLU-mul variants on 32-core width-sharded [32, 12288]
cfg2 = width_sharded_memory_config(12288, 32, grid)
g = ttnn.from_torch(
    torch.randn(1, 1, 32, 12288).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=cfg2,
)
u = ttnn.from_torch(
    torch.randn(1, 1, 32, 12288).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=cfg2,
)
ref = torch.nn.functional.silu(ttnn.to_torch(g).float()) * ttnn.to_torch(u).float()
variants = {
    "mul_silu_activation": lambda: ttnn.mul(
        g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16, memory_config=cfg2
    ),
    "silu_then_mul": lambda: (
        lambda s: (ttnn.mul(s, u, dtype=ttnn.bfloat16, memory_config=cfg2), s.deallocate(True))[0]
    )(ttnn.silu(g, memory_config=cfg2)),
    "mul_silu_legacy_false": lambda: ttnn.mul(
        g,
        u,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=ttnn.bfloat16,
        memory_config=cfg2,
        use_legacy=False,
    ),
    "mul_only": lambda: ttnn.mul(g, u, dtype=ttnn.bfloat16, memory_config=cfg2),
}
for name, fn in variants.items():
    r = dict(op="silu_mul", variant=name)
    try:
        y = fn()
        r["pcc"] = pcc(ref, ttnn.to_torch(y)[0, 0])
        y.deallocate(True)
        r["us"] = time_trace(lambda: fn().deallocate(True))
    except Exception as e:
        r["error"] = str(e).splitlines()[0][:200]
    record(r)
json.dump(rows, open("/tmp/qwen3_opt/exp6.json", "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
