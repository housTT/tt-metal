"""One-hot position vector construction variants under the watcher. argv[1]: full_pad | arange_slice_pad | arange_pad_only"""
import sys
import time

import ttnn

variant = sys.argv[1]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
count, start = 27, 37
results = []
if variant == "full_pad":
    for i in range(count):
        one = ttnn.moreh_full(
            (1, 1),
            start + i,
            mesh,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        padded = ttnn.pad(one, [(0, 0), (i, count - 1 - i)], -1)
        results.append(ttnn.to_torch(ttnn.reshape(padded, (count,))))
elif variant == "arange_slice_pad":
    positions = ttnn.reshape(ttnn.arange(start, start + count, 1, device=mesh, dtype=ttnn.int32), (1, count))
    for i in range(count):
        one = ttnn.slice(positions, [0, i], [1, i + 1])
        padded = ttnn.pad(one, [(0, 0), (i, count - 1 - i)], -1)
        results.append(ttnn.to_torch(ttnn.reshape(padded, (count,))))
elif variant == "arange_pad_only":
    positions = ttnn.reshape(ttnn.arange(start, start + count, 1, device=mesh, dtype=ttnn.int32), (1, count))
    results.append(ttnn.to_torch(positions).flatten())
else:
    raise ValueError(variant)
ttnn.synchronize_device(mesh)
time.sleep(12)
expected_ok = (
    all(int(r.flatten()[i]) == start + i and (r.flatten() != -1).sum() == 1 for i, r in enumerate(results))
    if variant != "arange_pad_only"
    else list(results[0].tolist()) == list(range(start, start + count))
)
print("RESULT", variant, "ok" if expected_ok else "WRONG", flush=True)
ttnn.close_mesh_device(mesh)
