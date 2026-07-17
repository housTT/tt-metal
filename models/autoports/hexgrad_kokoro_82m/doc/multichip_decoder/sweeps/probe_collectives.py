import sys as _s

_s.meta_path = [m for m in _s.meta_path if "editable" not in getattr(type(m), "__module__", "")]
_s.path = [p for p in _s.path if "model-bringup" not in p]
import time

import torch

import ttnn

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 4)), trace_region_size=90000000)
print("OPEN", tuple(md.shape))
topo = ttnn.Topology.Linear
N = 50


def bench(name, fn, *args):
    # warm
    for _ in range(3):
        r = fn(*args)
        ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(N):
        r = fn(*args)
    ttnn.synchronize_device(md)
    dt = (time.perf_counter() - t0) / N * 1e6
    print(f"{name:40s} {dt:8.1f} us  out={tuple(r.shape)}")
    return dt


H = 768
for S in (512, 128):
    # replicated [1,S,H], all_reduce across mesh
    xr = ttnn.from_torch(
        torch.randn(1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    bench(f"all_reduce[1,{S},{H}]", lambda t: ttnn.all_reduce(t, cluster_axis=1, num_links=1, topology=topo), xr)

# SP: reduce_scatter [1,512,768]->[1,128,768] on dim=1, then all_gather back
xs = ttnn.from_torch(
    torch.randn(1, 512, H),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
)
try:
    rs_out = bench(
        "reduce_scatter[1,512,768]->dim1",
        lambda t: ttnn.reduce_scatter(t, dim=1, cluster_axis=1, num_links=1, topology=topo),
        xs,
    )
except Exception as e:
    print("reduce_scatter ERR", repr(e)[:200])

# all_gather [1,128,768]->[1,512,768] dim=1 (input sharded on dim1)
xg = ttnn.from_torch(
    torch.randn(1, 512, H),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(1, 4)),
)
print("gather in per-dev", xg.shape)
try:
    bench("all_gather[1,128,768]->dim1", lambda t: ttnn.all_gather(t, dim=1, num_links=1, topology=topo), xg)
except Exception as e:
    print("all_gather ERR", repr(e)[:200])

ttnn.close_mesh_device(md)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("DONE")
