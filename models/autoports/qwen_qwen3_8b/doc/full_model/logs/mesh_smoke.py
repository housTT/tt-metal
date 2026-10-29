import time

import torch

import ttnn

t0 = time.time()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
print("opened", round(time.time() - t0, 1), flush=True)
x = ttnn.from_torch(torch.ones(1, 1, 32, 4096).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
y = ttnn.add(x, x)
print("add", float(ttnn.to_torch(y).float().mean()), round(time.time() - t0, 1), flush=True)
w = ttnn.from_torch(
    torch.randn(1, 1, 4096, 12288).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh
)
ttnn.synchronize_device(mesh)
print("53MB bfp8 upload", round(time.time() - t0, 1), flush=True)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK", round(time.time() - t0, 1), flush=True)
