import json
import sys
import time

import torch
import ttnn

chips = int(sys.argv[1])
fabric = sys.argv[2]
out = sys.argv[3]
ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips), trace_region_size=64 << 20)
results = {"chips": chips, "fabric": fabric, "device_ids": list(mesh.get_device_ids()), "rows": []}
topologies = {"Ring": ttnn.Topology.Ring, "Linear": ttnn.Topology.Linear}


def make(rows, seed):
    g = torch.Generator().manual_seed(seed)
    host = torch.randn((chips, rows, 2048), generator=g).to(torch.bfloat16)
    tensor = ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    return host, tensor


def check(host, result):
    expected = host.float().sum(0)
    worst = 0.0
    for shard in ttnn.get_device_tensors(result):
        value = ttnn.to_torch(shard).float().reshape(expected.shape)
        worst = max(worst, (value - expected).abs().max().item())
    return worst


def variants(topology, links):
    t = topologies[topology]

    def all_reduce(x):
        return ttnn.all_reduce(x, cluster_axis=1, topology=t, num_links=links, memory_config=x.memory_config())

    def rs_ag(x):
        y = ttnn.reduce_scatter(x, dim=2, cluster_axis=1, topology=t, num_links=links)
        return ttnn.all_gather(y, dim=2, cluster_axis=1, topology=t, num_links=links)

    return {"all_reduce": all_reduce, "rs_ag": rs_ag}


for rows, memory, repeats in ((1, ttnn.L1_MEMORY_CONFIG, 20), (32, ttnn.L1_MEMORY_CONFIG, 20),
                              (4096, ttnn.DRAM_MEMORY_CONFIG, 4)):
    for topology in ("Ring", "Linear"):
        for links in (1, 2):
            for name, fn in variants(topology, links).items():
                row = {"rows": rows, "topology": topology, "links": links, "variant": name}
                try:
                    host, x = make(rows, rows + links)
                    x = ttnn.to_memory_config(x, memory)
                    x = ttnn.reshape(x, (1, rows, 2048))
                    y = fn(x)
                    row["max_abs_error"] = check(host, y)
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(repeats):
                        y = fn(x)
                    ttnn.synchronize_device(mesh)
                    row["eager_us"] = (time.perf_counter() - t0) / repeats * 1e6
                    if rows <= 32:
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        z = x
                        for _ in range(repeats):
                            z = fn(z)
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        samples = []
                        for _ in range(30):
                            t0 = time.perf_counter()
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                            samples.append((time.perf_counter() - t0) * 1e6 / repeats)
                        ttnn.release_trace(mesh, tid)
                        samples.sort()
                        row["trace_us_per_op_median"] = samples[len(samples) // 2]
                except Exception as error:
                    row["error"] = str(error).splitlines()[0][:300]
                print(json.dumps(row), flush=True)
                results["rows"].append(row)
json.dump(results, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("DONE")
