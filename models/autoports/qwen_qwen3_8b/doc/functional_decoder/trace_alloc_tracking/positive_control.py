"""Positive control for TT_METAL_TRACE_ALLOC_TRACKING=1: an allocation made while a trace is
alive must make ttnn.execute_trace raise. Run with the tracking environment set before import."""
import os

import torch

import ttnn

assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", "set TT_METAL_TRACE_ALLOC_TRACKING=1 before running"
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    y = ttnn.add(x, x)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = ttnn.add(x, x)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    print("clean replay after capture: ok", flush=True)
    late = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    try:
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        print("POSITIVE CONTROL FAILED: replay with a live post-capture allocation did not raise", flush=True)
    except RuntimeError as exc:
        print("positive control raised as expected:", str(exc).splitlines()[0][:160], flush=True)
    late.deallocate(True)
    ttnn.release_trace(mesh, tid)
finally:
    ttnn.close_mesh_device(mesh)
