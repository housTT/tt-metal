import json
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import open_mesh

tp, out = int(sys.argv[1]), sys.argv[2]
mesh = open_mesh(tp)
topology = ttnn.Topology.Ring if tp > 2 else ttnn.Topology.Linear
grid = mesh.compute_with_storage_grid_size()
cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
sem = lambda n: [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(n)]
sets = [dict(barrier=sem(2), rs=sem(3), ag=sem(2)) for _ in range(2)]
view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
state = {"i": 0}


def explicit(x):
    s = sets[state["i"]]
    state["i"] ^= 1
    return ttnn.experimental.all_reduce_async(x, cluster_axis=1, mesh_device=mesh, barrier_semaphores=s["barrier"],
                                              rs_global_semaphores=s["rs"], ag_global_semaphores=s["ag"],
                                              math_op=ttnn.ReduceType.Sum, memory_config=x.memory_config(),
                                              topology=topology, num_links=2)


def default(x):
    return ttnn.all_reduce(x, cluster_axis=1, topology=topology, num_links=2, memory_config=x.memory_config())


rows = []
for n, memory in ((32, ttnn.L1_MEMORY_CONFIG), (32, ttnn.DRAM_MEMORY_CONFIG), (1024, ttnn.DRAM_MEMORY_CONFIG),
                  (4096, ttnn.DRAM_MEMORY_CONFIG)):
    data = [torch.randn(1, 1, n, 2048).to(torch.bfloat16) for _ in range(tp)]
    x = ttnn.from_torch(torch.cat(data, dim=0), device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=memory,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    expected = sum(d.float() for d in data)
    for name, fn in (("default", default), ("explicit", explicit)):
        y = fn(x)
        got = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
        pcc = float(torch.corrcoef(torch.stack((got.flatten(), expected.flatten())))[0, 1])
        times = []
        for _ in range(3):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(50):
                y = fn(x)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - t0) / 50 * 1e6)
        traced = None
        if n == 32:
            fn(x)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(20):
                y = fn(x)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            samples = []
            for _ in range(15):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                samples.append((time.perf_counter() - t0) / 20 * 1e6)
            ttnn.release_trace(mesh, tid)
            traced = sorted(samples)[7]
        row = dict(rows=n, memory=str(memory.buffer_type), path=name, pcc=pcc, eager_us=min(times), trace_us=traced)
        print(json.dumps(row), flush=True)
        rows.append(row)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
