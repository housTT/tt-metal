"""Minimal repro: ttnn.to_device of a 53 MB BFP8 tensor loaded with ttnn.load_tensor never returns (host spins at 100 percent CPU).

Run with TT_VISIBLE_DEVICES=<chip> and a timeout. A 26 MB tensor from the same loader and a 53 MB tensor from
ttnn.from_torch both complete. Files were written by ttnn.as_tensor(..., cache_file_name=...) on the same build.
"""
import sys
import time

import torch

import ttnn

path = (
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3_opt/cache/l0_gate_bfp8_il_dtype_BFLOAT8_B_layout_TILE.tensorbin"
)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
t0 = time.time()
d = ttnn.from_torch(
    torch.randn(1, 1, 4096, 12288),
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
ttnn.synchronize_device(mesh)
print("STEP from_torch 53MB on device", round(time.time() - t0, 1), "s", flush=True)
d.deallocate(True)
t = ttnn.load_tensor(path)
print("STEP loaded", t.shape, t.dtype, flush=True)
t0 = time.time()
d = ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
ttnn.synchronize_device(mesh)
print("STEP cached tensor on device", round(time.time() - t0, 1), "s", flush=True)
ttnn.close_mesh_device(mesh)
