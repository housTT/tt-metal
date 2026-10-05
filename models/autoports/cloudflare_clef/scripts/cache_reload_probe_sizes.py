import os
import sys
import time

import torch

import ttnn

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/hous/dev/clef/tt_cache/_cache_probe_sizes"
MODE = os.environ.get("CACHE_PROBE_MODE", "alts")
ROWS = int(os.environ.get("CACHE_PROBE_ROWS", "5120"))
COLS = int(os.environ.get("CACHE_PROBE_COLS", "8240"))
os.makedirs(OUT, exist_ok=True)


def say(msg):
    print(f"[cache_probe_sizes] {time.strftime('%H:%M:%S')} {msg}", flush=True)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
parent = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=2, trace_region_size=0
)
sub = parent.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
say(f"parent {list(parent.shape)} sub {list(sub.shape)} tensor [{ROWS}, {COLS}] bfp8 sharded dim 3")
x = torch.randn(1, 1, ROWS, COLS)


def write(name, mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT):
    path = f"{OUT}/{name}"
    full = f"{path}_dtype_{dtype.name}_layout_{layout.name}.tensorbin"
    if os.path.exists(full):
        os.remove(full)
    t0 = time.time()
    t = ttnn.as_tensor(
        x,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=path,
    )
    ttnn.synchronize_device(mesh)
    say(f"{name} as_tensor write: {time.time() - t0:.2f} s shape={list(t.shape)} file={os.path.getsize(full)} B")
    ttnn.deallocate(t)
    return path, full


def reload_as_tensor(name, path, mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT):
    say(f"{name} as_tensor reload: start")
    t0 = time.time()
    t = ttnn.as_tensor(
        x,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=path,
    )
    ttnn.synchronize_device(mesh)
    say(f"{name} as_tensor reload: {time.time() - t0:.2f} s shape={list(t.shape)}")
    ttnn.deallocate(t)


def reload_host_then_to_device(name, full, mesh):
    say(f"{name} load_tensor(host): start")
    t0 = time.time()
    h = ttnn.load_tensor(full)
    say(f"{name} load_tensor(host): {time.time() - t0:.2f} s shape={list(h.shape)} storage={h.storage_type()}")
    say(f"{name} to_device: start")
    t0 = time.time()
    d = ttnn.to_device(h, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh)
    say(f"{name} to_device: {time.time() - t0:.2f} s shape={list(d.shape)}")
    ttnn.deallocate(d)


try:
    if MODE == "alts":
        path_s, full_s = write("sub_shard", sub)
        reload_host_then_to_device("sub_shard", full_s, sub)
        path_p, full_p = write("parent_shard", parent)
        reload_as_tensor("parent_shard", path_p, parent)
        reload_host_then_to_device("parent_shard", full_p, parent)
    elif MODE == "spinner":
        path_s, full_s = write("sub_shard", sub)
        reload_as_tensor("sub_shard", path_s, sub)
    elif MODE == "parent_only":
        path_p, full_p = write("parent_shard", parent)
        reload_as_tensor("parent_shard", path_p, parent)
        reload_host_then_to_device("parent_shard", full_p, parent)
    elif MODE == "sub_only_alt":
        path_s, full_s = write("sub_shard", sub)
        reload_host_then_to_device("sub_shard", full_s, sub)
    say("CACHE_PROBE_SIZES_DONE")
finally:
    ttnn.synchronize_device(parent)
    for child in parent.get_submeshes():
        ttnn.close_mesh_device(child)
    ttnn.close_mesh_device(parent)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
