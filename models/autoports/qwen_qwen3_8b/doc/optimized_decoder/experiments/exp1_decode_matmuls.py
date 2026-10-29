"""Decode matmul micro-benchmark for the Qwen3-8B optimized decoder (stage 3).

Real layer-0 weights, realistic activations, one Blackhole chip. Times each
candidate by capturing REP back-to-back ops in a trace and replaying it.
Writes JSON rows to /tmp/qwen3_opt/exp1_<tag>.json.
"""
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

TAG = os.getenv("EXP_TAG", "a")
ROLES = os.getenv("EXP_ROLES", "qkv,o,gate,down").split(",")
DTYPES = os.getenv("EXP_DTYPES", "bfp8,bfp4").split(",")
CORES = [int(c) for c in os.getenv("EXP_CORES", "8,16,32,64").split(",")]
FIDS = os.getenv("EXP_FIDS", "default").split(",")
READERS = [int(r) for r in os.getenv("EXP_READERS", "1").split(",")]
BLOCKS = os.getenv("EXP_BLOCKS", "max,half,2")
REP = int(os.getenv("EXP_REP", "10"))
REPLAYS = int(os.getenv("EXP_REPLAYS", "5"))
ONE_D = os.getenv("EXP_1D", "0") == "1"
OUT = f"/tmp/qwen3_opt/exp1_{TAG}.json"

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}

state = weight_source.load_layer_state(0)
emb = weight_source.load_embedding()
tokens = torch.tensor(
    json.loads(
        open(
            "/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/functional_decoder/layer_inputs/tale_token_ids.json"
        ).read()
    )["tokens"][100:132]
)
x = emb[tokens].float()  # [32, 4096]


def rms(v, w, eps=1e-6):
    return w.float() * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps))


xn = rms(x, state["input_layernorm.weight"])
wq, wk, wv = (state[f"self_attn.{n}_proj.weight"].float().T for n in ("q", "k", "v"))
W = {
    "qkv": torch.cat([wq, wk, wv], dim=-1),
    "o": state["self_attn.o_proj.weight"].float().T,
    "gate": state["mlp.gate_proj.weight"].float().T,
    "up": state["mlp.up_proj.weight"].float().T,
    "down": state["mlp.down_proj.weight"].float().T,
}
act_down = torch.nn.functional.silu(xn @ W["gate"]) * (xn @ W["up"])
act_o = (xn @ wv).repeat(1, 4)
ACT = {"qkv": xn, "o": act_o, "gate": xn, "up": xn, "down": act_down}

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
dram = mesh.dram_grid_size()
NB = dram.x
print("grid", grid, "dram banks", NB, flush=True)
DRAM_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NB - 1, 0))])


def dram_sharded_cfg(k, n):
    padded = math.ceil(n / (32 * NB)) * 32 * NB
    spec = ttnn.ShardSpec(DRAM_GRID, (k, padded // NB), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def core_set(n):
    if n <= grid.x:
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))])
    for w in (8, 11, 10):
        if n % w == 0 and n // w <= grid.y and w <= grid.x:
            return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, n // w - 1))])
    raise ValueError(n)


def act_cfg(k, n_cores):
    return ttnn.create_sharded_memory_config(
        shape=(32, k // n_cores),
        core_grid=core_set(n_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def ckc(fid, fp32, l1acc=True):
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


rows = []
weights_cache = {}


def weight(role, dtype, sharded):
    key = (role, dtype, sharded)
    if key not in weights_cache:
        print("LOAD", key, flush=True)
        w = W[role]
        k, n = w.shape
        cfg = dram_sharded_cfg(k, n) if sharded else ttnn.DRAM_MEMORY_CONFIG
        weights_cache[key] = ttnn.as_tensor(
            w.unsqueeze(0).unsqueeze(0),
            dtype=DT[dtype],
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=cfg,
        )
    return weights_cache[key]


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


for role in ROLES:
    k, n = W[role].shape
    kt, nt = k // 32, n // 32
    ref = ACT[role] @ W[role]
    for dtype in DTYPES:
        fids = ["lofi" if dtype == "bfp4" else "hifi2"] if FIDS == ["default"] else FIDS
        for n_cores in CORES:
            if kt % n_cores or nt % n_cores:
                continue
            kpc = kt // n_cores
            divs = divisors(kpc)
            if BLOCKS == "all":
                blocks = divs
            else:
                blocks = sorted(
                    {
                        {
                            "max": divs[-1],
                            "half": divs[-1] // 2 if divs[-1] // 2 in divs else divs[-1],
                            "2": 2 if 2 in divs else 1,
                        }[b]
                        for b in BLOCKS.split(",")
                    }
                )
            x_t = ttnn.from_torch(
                ACT[role].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=act_cfg(k, n_cores),
            )
            for fidspec in fids:
                fid, fp32 = (fidspec.split("+") + ["f"])[:2]
                fp32 = fp32 == "t"
                cfg = ckc(fid, fp32)
                for readers in READERS:
                    if (nt // NB) % readers:
                        continue
                    for b in blocks:
                        row = dict(
                            role=role,
                            dtype=dtype,
                            cores=n_cores,
                            in0_block_w=b,
                            per_core_N=nt // n_cores,
                            fidelity=fid,
                            fp32_acc=fp32,
                            readers=readers,
                            family="dram_sharded",
                        )
                        print("START", json.dumps(row), flush=True)
                        try:
                            wt = weight(role, dtype, True)
                            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                                in0_block_w=b,
                                per_core_M=1,
                                per_core_N=nt // n_cores,
                                fused_activation=None,
                                num_workers_per_dram_bank=readers,
                            )

                            def fn():
                                y = ttnn.linear(
                                    x_t,
                                    wt,
                                    program_config=pc,
                                    compute_kernel_config=cfg,
                                    dtype=ttnn.bfloat16,
                                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                )
                                y.deallocate(True)

                            y = ttnn.linear(
                                x_t,
                                wt,
                                program_config=pc,
                                compute_kernel_config=cfg,
                                dtype=ttnn.bfloat16,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            )
                            row["pcc"] = pcc(ref, ttnn.to_torch(y)[0, 0])
                            y.deallocate(True)
                            row["us"] = time_trace(fn)
                            row["gbps"] = (k * n * {"bf16": 2, "bfp8": 1.0625, "bfp4": 0.5625}[dtype]) / row["us"] / 1e3
                        except Exception as e:  # noqa
                            row["error"] = str(e).splitlines()[0][:300]
                        print(json.dumps(row), flush=True)
                        rows.append(row)
            x_t.deallocate(True)
        if ONE_D:
            fid, fp32 = (fids[0].split("+") + ["f"])[:2]
            fp32 = fp32 == "t"
            cfg = ckc(fid, fp32)
        if ONE_D:
            for cores_1d in (32, 44, 55, 64, 88, 110):
                cols = min(grid.x, cores_1d)
                rws = math.ceil(cores_1d / cols)
                if cols * rws > grid.x * grid.y:
                    continue
                per_core_n = math.ceil(nt / (cols * rws))
                for b in sorted({divisors(kt)[-1] if divisors(kt)[-1] <= 8 else 8, 4, 2}):
                    if kt % b:
                        continue
                    row = dict(
                        role=role,
                        dtype=dtype,
                        cores=cols * rws,
                        grid=f"{cols}x{rws}",
                        in0_block_w=b,
                        per_core_N=per_core_n,
                        fidelity=fid,
                        fp32_acc=fp32,
                        readers=0,
                        family="1d_mcast",
                    )
                    try:
                        wt = weight(role, dtype, False)
                        sub_w = max(i for i in range(1, (4 if fp32 else 8) + 1) if per_core_n % i == 0)
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(cols, rws),
                            in0_block_w=b,
                            out_subblock_h=1,
                            out_subblock_w=sub_w,
                            per_core_M=1,
                            per_core_N=per_core_n,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        )
                        x_il = ttnn.from_torch(
                            ACT[role].unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

                        def fn():
                            y = ttnn.linear(
                                x_il,
                                wt,
                                program_config=pc,
                                compute_kernel_config=cfg,
                                dtype=ttnn.bfloat16,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            )
                            y.deallocate(True)

                        y = ttnn.linear(
                            x_il,
                            wt,
                            program_config=pc,
                            compute_kernel_config=cfg,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        )
                        row["pcc"] = pcc(ref, ttnn.to_torch(y)[0, 0])
                        y.deallocate(True)
                        row["us"] = time_trace(fn)
                        x_il.deallocate(True)
                    except Exception as e:  # noqa
                        row["error"] = str(e).splitlines()[0][:300]
                    print(json.dumps(row), flush=True)
                    rows.append(row)
    json.dump(rows, open(OUT, "w"), indent=1)

json.dump(rows, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE", OUT)
