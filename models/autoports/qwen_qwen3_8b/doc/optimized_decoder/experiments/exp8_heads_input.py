"""Head-split input layout variants: run each under TT_METAL_WATCHER (separate processes) and time them.

argv[1]: variant in {subcoregrid, rect, l1_interleaved, dram_interleaved}; argv[2]: batch (1 or 32); env EXP8_TIME=1 to time.
"""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import _decode_height_sharded_memory_config
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import width_sharded_memory_config

variant, batch = sys.argv[1], int(sys.argv[2])
timing = os.getenv("EXP8_TIME") == "1"
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=32 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
native = width_sharded_memory_config(6144, 32, grid)
rect = ttnn.create_sharded_memory_config(
    shape=(32, 192),
    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))]),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
heads_cfg = _decode_height_sharded_memory_config(mesh, batch, 128)
src = torch.randn(1, 1, 32, 6144).bfloat16()
x32 = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=native)
x = ttnn.reshape(x32, (1, 1, batch, 6144), (1, 1, 32, 6144)) if batch != 32 else x32
ref = None


def split(t):
    return ttnn.experimental.nlp_create_qkv_heads_decode(t, num_heads=32, num_kv_heads=8, memory_config=heads_cfg)


def run():
    if variant == "subcoregrid":
        q, k, v = split(x)
    elif variant == "rect":
        t = ttnn.to_memory_config(x, rect)
        q, k, v = split(t)
        t.deallocate(True)
    elif variant == "l1_interleaved":
        t = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        q, k, v = split(t)
        t.deallocate(True)
    elif variant == "dram_interleaved":
        t = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = split(t)
        t.deallocate(True)
    else:
        raise ValueError(variant)
    return q, k, v


q, k, v = run()
out = [ttnn.to_torch(t) for t in (q, k, v)]
expect = src[0, 0, :batch].float()
q_ref = expect[:, : 32 * 128].reshape(batch, 32, 128)
k_ref = expect[:, 32 * 128 : 40 * 128].reshape(batch, 8, 128)
v_ref = expect[:, 40 * 128 :].reshape(batch, 8, 128)
ok = all(
    torch.equal(a.float(), b)
    for a, b in ((out[0][0, :, :32], q_ref), (out[1][0, :, :8], k_ref), (out[2][0, :, :8], v_ref))
)
row = dict(variant=variant, batch=batch, exact=ok)
for t in (q, k, v):
    t.deallocate(True)
if timing:

    def fn():
        for t in run():
            t.deallocate(True)

    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(10):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        best = min(best, time.perf_counter() - t0)
    row["us"] = best * 1e5
    ttnn.release_trace(mesh, tid)
else:
    for _ in range(3):
        for t in run():
            t.deallocate(True)
    ttnn.synchronize_device(mesh)
    time.sleep(12)
print("RESULT", json.dumps(row), flush=True)
ttnn.close_mesh_device(mesh)
