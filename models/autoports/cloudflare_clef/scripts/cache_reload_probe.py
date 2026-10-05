import os
import shutil
import sys
import time

import torch

import ttnn

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/hous/dev/clef/tt_cache/_cache_probe"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
parent = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=2, trace_region_size=0
)
sub = parent.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
x = torch.randn(1, 1, 256, 5120)


def step(name, mesh, dtype, layout, mapper):
    path = f"{OUT}/{name}"
    for phase in ("write", "reload"):
        t0 = time.time()
        t = ttnn.as_tensor(
            x,
            dtype=dtype,
            layout=layout,
            device=mesh,
            mesh_mapper=mapper(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=path,
        )
        ttnn.synchronize_device(mesh)
        print(f"[cache_probe] {name} {phase}: {round(time.time() - t0, 2)} s shape={list(t.shape)}", flush=True)
        ttnn.deallocate(t)


try:
    shard = lambda m: ttnn.ShardTensorToMesh(m, dim=3)
    rep = lambda m: ttnn.ReplicateTensorToMesh(m)
    order = os.environ.get("CACHE_PROBE_ORDER", "parent,sub").split(",")
    for which in order:
        mesh = parent if which == "parent" else sub
        step(f"{which}_shard_bfp8_tile", mesh, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard)
        step(f"{which}_rep_bf16_tile", mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT, rep)
        step(f"{which}_shard_bf16_rowmajor", mesh, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, shard)
    print("CACHE_PROBE_DONE", flush=True)
finally:
    ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
