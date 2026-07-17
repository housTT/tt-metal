import sys as _s

_s.meta_path = [m for m in _s.meta_path if "editable" not in getattr(type(m), "__module__", "")]
_s.path = [p for p in _s.path if "model-bringup" not in p]
import time

import torch

import ttnn
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    default_topology,
    get_num_links,
    get_tt_ccl,
)

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 4)), trace_region_size=90000000)
print("OPEN", tuple(md.shape))
ccl = get_tt_ccl(md)
topo = default_topology(md)
nl = get_num_links(md)
print("topology", topo, "num_links", nl)


def RS(x, dim):
    return ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=dim,
        multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=nl,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topo,
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )


def AG(x, dim):
    return ttnn.experimental.all_gather_async(
        x,
        dim=dim,
        persistent_output_buffer=None,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=nl,
        topology=topo,
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
    )


# Correctness: seq-sharded reduce_scatter on dim=2 of [1,1,S,768]
S = 512
# per-device distinct data, replicate-mapped so we can check sum
torch.manual_seed(0)
base = torch.randn(4, 1, 1, S, 768)  # one slab per device
full = ttnn.from_torch(
    base.reshape(4, 1, S, 768),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 0), mesh_shape=(1, 4)),
)
print("full per-dev", full.shape)
try:
    rs = RS(full, dim=2)
    ttnn.synchronize_device(md)
    print("RS dim2 OK ->", rs.shape)  # expect [1,1,128,768] per dev
    ag = AG(rs, dim=2)
    ttnn.synchronize_device(md)
    print("AG dim2 OK ->", ag.shape)  # expect [1,1,512,768] per dev
    # verify AG(RS(x)) == sum over devices, replicated
    got = ttnn.to_torch(ag, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1]  # take dev0 copy
    exp = base.sum(dim=0)  # [1,1,S,768]
    pcc = torch.corrcoef(torch.stack([got.flatten().float(), exp.flatten().float()]))[0, 1]
    print("PCC AG(RS) vs sum:", float(pcc))
except Exception as e:
    print("dim2 ERR", repr(e)[:400])


# timing (untraced wall clock, warmed)
def bench(name, fn, *a):
    for _ in range(3):
        r = fn(*a)
        ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(50):
        r = fn(*a)
    ttnn.synchronize_device(md)
    print(f"{name:30s} {(time.perf_counter()-t0)/50*1e6:8.1f} us {tuple(r.shape)}")
    return r


xseq = ttnn.from_torch(
    torch.randn(1, 1, S, 768),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    mesh_mapper=ttnn.ReplicateTensorToMesh(md),
)
bench("RS[1,1,512,768]dim2", RS, xseq, 2)
xsh = ttnn.from_torch(
    torch.randn(1, 1, S, 768),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=md,
    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(1, 4)),
)
bench("AG[1,1,128,768]dim2", AG, xsh, 2)

ttnn.close_mesh_device(md)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("DONE")
