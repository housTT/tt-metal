"""Is there a *trace-capturable* way to widen a bfloat4_b tensor on device?

C19 (`doc/datatype_sweep/blocked/C19-expert-act-bfp4.json`) asks for a bfloat4_b routed-expert
output activation. `ttnn.experimental.deepseek_moe_fast_reduce_nc` accepts only BFLOAT16/BFLOAT8_B,
so the adaptation in `OptimizedMoE._routed_experts` widens with `ttnn.typecast` — and the model then
fails trace capture with "Writes are not supported during trace capture".

This probe runs each widening spelling twice: once outside trace capture (does it work at all?) and
once inside (is it capturable?). Shapes are the decode routed-expert `down` tensor of the shipped
1x4 build: [1, E_local=64, 32, 2048].

    python probe_bfp4_widen.py [candidate ...]
"""

import sys
import traceback

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh

E, T, H = 64, 32, 2048
SHAPE = (1, E, T, H)


def make_src(mesh, mem):
    torch.manual_seed(0)
    host = torch.randn(*SHAPE, dtype=torch.float32)
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


# Each candidate takes (src, mem, prealloc) and returns a bfloat8_b tensor.
def c_typecast(src, mem, prealloc):
    return ttnn.typecast(src, ttnn.bfloat8_b, memory_config=mem)


def c_typecast_prealloc(src, mem, prealloc):
    return ttnn.typecast(src, ttnn.bfloat8_b, output_tensor=prealloc)


def c_clone(src, mem, prealloc):
    return ttnn.clone(src, dtype=ttnn.bfloat8_b, memory_config=mem)


def c_copy(src, mem, prealloc):
    ttnn.copy(src, prealloc)
    return prealloc


def c_to_layout(src, mem, prealloc):
    return ttnn.to_layout(src, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=mem)


def c_mul_one(src, mem, prealloc):
    return ttnn.multiply(src, 1.0, dtype=ttnn.bfloat8_b, memory_config=mem)


def c_add_zero(src, mem, prealloc):
    return ttnn.add(src, 0.0, dtype=ttnn.bfloat8_b, memory_config=mem)


def c_mul_one_prealloc(src, mem, prealloc):
    ttnn.multiply(src, 1.0, dtype=ttnn.bfloat8_b, output_tensor=prealloc)
    return prealloc


def c_exp_typecast(src, mem, prealloc):
    return ttnn.experimental.typecast(src, ttnn.bfloat8_b, memory_config=mem)


CANDIDATES = {
    "typecast": c_typecast,
    "typecast_prealloc": c_typecast_prealloc,
    "clone": c_clone,
    "copy": c_copy,
    "to_layout": c_to_layout,
    "mul_one": c_mul_one,
    "add_zero": c_add_zero,
    "mul_one_prealloc": c_mul_one_prealloc,
    "exp_typecast": c_exp_typecast,
}


def short(exc):
    text = str(exc).replace("\n", " ")
    return text[:300]


def main():
    names = sys.argv[1:] or list(CANDIDATES)
    mesh = open_ornith_mesh(trace_region_size=64 << 20)
    results = []
    try:
        for mem_name, mem in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
            for name in names:
                fn = CANDIDATES[name]
                src = make_src(mesh, mem)
                prealloc = ttnn.from_torch(
                    torch.zeros(*SHAPE, dtype=torch.float32),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                # 1. eager: does the spelling exist and produce bfloat8_b at all?
                try:
                    out = fn(src, mem, prealloc)
                    ttnn.synchronize_device(mesh)
                    eager = f"ok dtype={out.dtype}"
                    if out is not prealloc:
                        ttnn.deallocate(out)
                except Exception as exc:  # noqa: BLE001
                    eager = f"FAIL {type(exc).__name__}: {short(exc)}"

                # 2. traced: is the same spelling capturable?
                if eager.startswith("ok"):
                    try:
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        try:
                            out = fn(src, mem, prealloc)
                        finally:
                            ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        ttnn.release_trace(mesh, tid)
                        traced = "CAPTURABLE"
                        if out is not prealloc:
                            ttnn.deallocate(out)
                    except Exception as exc:  # noqa: BLE001
                        traced = f"NOT-CAPTURABLE {type(exc).__name__}: {short(exc)}"
                else:
                    traced = "skipped (eager failed)"

                print(f"RESULT {mem_name:4s} {name:18s} eager={eager} | traced={traced}", flush=True)
                results.append((mem_name, name, eager, traced))
                ttnn.deallocate(src)
                ttnn.deallocate(prealloc)
                ttnn.synchronize_device(mesh)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        print("=== SUMMARY ===", flush=True)
        for row in results:
            print(" | ".join(row), flush=True)
        close_ornith_mesh(mesh)
        print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
