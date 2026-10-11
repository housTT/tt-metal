import json
import sys
import time

import torch
import ttnn

chips = int(sys.argv[1])
out = sys.argv[2]
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if chips == 4 else ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips), trace_region_size=64 << 20)
topology = ttnn.Topology.Ring if chips == 4 else ttnn.Topology.Linear
L1 = ttnn.L1_MEMORY_CONFIG
eps = 1e-6
torch.manual_seed(0)
partial_host = torch.randn(chips, 1, 32, 2048).to(torch.bfloat16)
weight_host = (1 + 0.1 * torch.randn(2048)).to(torch.bfloat16)
partial = ttnn.from_torch(partial_host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          memory_config=L1, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
weight_full = ttnn.from_torch(weight_host.reshape(1, 1, 1, 2048), device=mesh, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
weight_shard = ttnn.from_torch(weight_host.reshape(chips, 1, 1, 2048 // chips), device=mesh, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, math_approx_mode=False)


def replicated():
    full = ttnn.all_reduce(partial, cluster_axis=1, topology=topology, num_links=2, memory_config=L1)
    return ttnn.rms_norm(full, epsilon=eps, weight=weight_full, compute_kernel_config=compute, memory_config=L1)


def sharded(gather_output=True):
    local = ttnn.reduce_scatter(partial, dim=3, cluster_axis=1, topology=topology, num_links=2, memory_config=L1)
    stats = ttnn.rms_norm_pre_all_gather(local, compute_kernel_config=compute, dtype=ttnn.bfloat16)
    stats = ttnn.all_gather(stats, dim=3, cluster_axis=1, topology=topology, num_links=2, memory_config=L1)
    normed = ttnn.rms_norm_post_all_gather(local, stats, epsilon=eps, weight=ttnn.reshape(weight_shard, (1, 1, 1, 2048 // chips)),
                                           compute_kernel_config=compute, memory_config=L1)
    if not gather_output:
        return normed
    return ttnn.all_gather(normed, dim=3, cluster_axis=1, topology=topology, num_links=2, memory_config=L1)


def traced(fn, repeats=10):
    fn()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e6 / repeats)
    ttnn.release_trace(mesh, tid)
    samples.sort()
    return samples[len(samples) // 2]


summed = partial_host.float().sum(0)
expected = (summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + eps) * weight_host.float())
rows = {}
for name, fn in (("replicated_allreduce_localnorm", replicated), ("sharded_rs_distnorm_ag", sharded),
                 ("sharded_rs_distnorm_no_output_gather", lambda: sharded(False))):
    row = {}
    try:
        value = fn()
        if name != "sharded_rs_distnorm_no_output_gather":
            got = ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float().reshape(expected.shape)
            row["max_abs_error"] = float((got - expected).abs().max())
        row["trace_us"] = traced(fn)
    except Exception as error:
        row["error"] = str(error).splitlines()[0][:300]
    rows[name] = row
    print(name, json.dumps(row), flush=True)
json.dump(dict(chips=chips, rows=rows), open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
