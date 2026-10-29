"""Prefill projection candidates at 1024 and 4096 rows under the final policy: minimal_matmul (sharded and interleaved
weight), ttnn.linear auto, ttnn.linear 2D program configs on the interleaved copy, and in0-in-L1 variants."""
import json
import math
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import dram_sharded_weight_memory_config
from models.common.utility_functions import comp_pcc

REP, REPLAYS = 3, 3
rows = []
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
NB = mesh.dram_grid_size().x
state = weight_source.load_layer_state(0)
emb = weight_source.load_embedding()
tok = json.loads(
    open(
        "/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/functional_decoder/layer_inputs/tale_token_ids.json"
    ).read()
)["tokens"]
w_ln = state["input_layernorm.weight"].float()
ckc = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
W = {
    "qkv": (
        torch.cat([state[f"self_attn.{n}_proj.weight"].float().T for n in ("q", "k", "v")], dim=-1),
        ttnn.bfloat8_b,
    ),
    "o": (state["self_attn.o_proj.weight"].float().T, ttnn.bfloat8_b),
    "gate": (state["mlp.gate_proj.weight"].float().T, ttnn.bfloat8_b),
    "up": (state["mlp.up_proj.weight"].float().T, ttnn.bfloat4_b),
    "down": (state["mlp.down_proj.weight"].float().T, ttnn.bfloat8_b),
}


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


def subblock(per_m, per_n, cap=4):
    best = (1, 1)
    for h in range(1, cap + 1):
        for w in range(1, cap + 1):
            if h * w <= cap and per_m % h == 0 and per_n % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


for m in (1024, 4096):
    x = emb[torch.tensor(tok[:m])].float()
    xn = w_ln * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    acts = {"qkv": xn, "o": (xn @ state["self_attn.v_proj.weight"].float().T).repeat(1, 4), "gate": xn, "up": xn}
    acts["down"] = torch.nn.functional.silu(xn @ W["gate"][0]) * (xn @ W["up"][0])
    for role, (w, dtype) in W.items():
        k, n = w.shape
        act = acts[role]
        ref = act @ w
        w_sh = ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram_sharded_weight_memory_config(k, n, NB),
        )
        w_il = ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_t = ttnn.from_torch(
            act.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mt, kt, nt = m // 32, k // 32, n // 32
        cands = {
            "minimal_default_sharded_w": lambda: ttnn.experimental.minimal_matmul(
                x_t, w_sh, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            "minimal_default_interleaved_w": lambda: ttnn.experimental.minimal_matmul(
                x_t, w_il, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            "linear_auto_interleaved_w": lambda: ttnn.linear(
                x_t, w_il, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        }
        for gx, gy in ((grid.x, grid.y), (10, 8), (8, 8)):
            per_m = math.ceil(mt / gy)
            per_n = math.ceil(nt / gx)
            sh, sw = subblock(per_m, per_n)
            for b in (4, 8, 16, 32):
                if kt % b:
                    continue
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=b,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=per_m,
                    per_core_N=per_n,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                cands[f"linear_2d_{gx}x{gy}_b{b}_sub{sh}x{sw}"] = lambda pc=pc: ttnn.linear(
                    x_t,
                    w_il,
                    program_config=pc,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        x_l1 = None
        try:
            x_l1 = ttnn.to_memory_config(x_t, ttnn.L1_MEMORY_CONFIG)
            cands["minimal_default_sharded_w_in0_l1"] = lambda: ttnn.experimental.minimal_matmul(
                x_l1, w_sh, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            cands["linear_auto_interleaved_w_in0_l1"] = lambda: ttnn.linear(
                x_l1, w_il, dtype=ttnn.bfloat16, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        except Exception as e:
            rows.append(dict(role=role, rows=m, variant="in0_l1", error=str(e).splitlines()[0][:160]))
            print(json.dumps(rows[-1]), flush=True)
        for name, fn in cands.items():
            r = dict(role=role, rows=m, variant=name)
            try:
                y = fn()
                r["pcc"] = float(comp_pcc(ref, ttnn.to_torch(y)[0, 0].float(), 0.0)[1])
                y.deallocate(True)
                r["us"] = time_trace(lambda: fn().deallocate(True))
            except Exception as e:
                r["error"] = str(e).splitlines()[0][:160]
            print(json.dumps(r), flush=True)
            rows.append(r)
            json.dump(rows, open("/tmp/qwen3_opt/exp12.json", "w"), indent=1)
        for t in (w_sh, w_il, x_t, x_l1):
            if t is not None:
                t.deallocate(True)
ttnn.close_mesh_device(mesh)
print("DONE")
