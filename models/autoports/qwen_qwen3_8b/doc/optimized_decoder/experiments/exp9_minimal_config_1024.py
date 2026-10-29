"""MinimalMatmulConfig sweep at 1024 rows on the DRAM-sharded weights under the final policy (BFP8 LoFi, BFP4 LoFi up)."""
import json
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import weights as weight_source
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import dram_sharded_weight_memory_config
from models.common.utility_functions import comp_pcc

REP, REPLAYS = 4, 3
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
x = emb[torch.tensor(tok[:1024])].float()
w_ln = state["input_layernorm.weight"].float()
xn = w_ln * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
W = {
    "qkv": (
        torch.cat([state[f"self_attn.{n}_proj.weight"].float().T for n in ("q", "k", "v")], dim=-1),
        ttnn.bfloat8_b,
        xn,
    ),
    "o": (
        state["self_attn.o_proj.weight"].float().T,
        ttnn.bfloat8_b,
        (xn @ state["self_attn.v_proj.weight"].float().T).repeat(1, 4),
    ),
    "gate": (state["mlp.gate_proj.weight"].float().T, ttnn.bfloat8_b, xn),
    "up": (state["mlp.up_proj.weight"].float().T, ttnn.bfloat4_b, xn),
}
W["down"] = (
    state["mlp.down_proj.weight"].float().T,
    ttnn.bfloat8_b,
    torch.nn.functional.silu(xn @ W["gate"][0]) * (xn @ W["up"][0]),
)
ckc = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


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


for role, (w, dtype, act) in W.items():
    k, n = w.shape
    ref = act @ w
    w_t = ttnn.from_torch(
        w.unsqueeze(0).unsqueeze(0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram_sharded_weight_memory_config(k, n, NB),
    )
    x_t = ttnn.from_torch(
        act.unsqueeze(0).unsqueeze(0).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    configs = {"default": None}
    for mb in (2, 4, 8):
        for kb in (2, 4, 8):
            for nb in (2, 4, 8):
                for gname, g in (("11x10", (grid.x, grid.y)), ("8x8", (8, 8))):
                    configs[f"m{mb}_k{kb}_n{nb}_{gname}"] = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        compute_with_storage_grid_size=ttnn.CoreCoord(*g),
                    )
    for name, cfg in configs.items():
        r = dict(role=role, rows=1024, config=name)
        try:

            def fn():
                y = ttnn.experimental.minimal_matmul(
                    x_t,
                    w_t,
                    config=cfg,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                y.deallocate(True)

            y = ttnn.experimental.minimal_matmul(
                x_t,
                w_t,
                config=cfg,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            r["pcc"] = float(comp_pcc(ref, ttnn.to_torch(y)[0, 0].float(), 0.0)[1])
            y.deallocate(True)
            r["us"] = time_trace(fn)
        except Exception as e:
            r["error"] = str(e).splitlines()[0][:160]
        print(json.dumps(r), flush=True)
        rows.append(r)
        json.dump(rows, open("/tmp/qwen3_opt/exp9.json", "w"), indent=1)
    w_t.deallocate(True)
    x_t.deallocate(True)
ttnn.close_mesh_device(mesh)
print("DONE")
