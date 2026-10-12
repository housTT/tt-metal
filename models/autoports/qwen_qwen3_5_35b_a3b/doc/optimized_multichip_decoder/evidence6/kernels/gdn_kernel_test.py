import json
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tt.gdn_decode import gdn_decode

nk, nv, d = int(sys.argv[1]), int(sys.argv[2]), 128
out = sys.argv[3]
groups = nv // nk
torch.manual_seed(0)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=8 << 20, physical_device_ids=[3])


def put(x, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory)


def reference(q, k, v, z, decay_log, beta, state, weight):
    q, k, v, z, decay_log, beta, state, weight = (t.double() for t in (q, k, v, z, decay_log, beta, state, weight))
    q = q.reshape(nk, d)
    k = k.reshape(nk, d)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * d ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(groups, 0)
    k = k.repeat_interleave(groups, 0)
    v = v.reshape(nv, d)
    s = state.reshape(nv, d, d) * decay_log.reshape(nv, 1, 1).exp()
    stored = (s * k[:, :, None]).sum(1)
    delta = (v - stored) * beta.reshape(nv, 1)
    s = s + k[:, :, None] * delta[:, None, :]
    o = (s * q[:, :, None]).sum(1)
    raw = o.clone()
    unnormed = o * weight.reshape(1, d) * torch.nn.functional.silu(z.reshape(nv, d))
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6) * weight.reshape(1, d)
    gated = o * torch.nn.functional.silu(z.reshape(nv, d))
    return {0: gated.reshape(-1), 1: raw.reshape(-1), 2: unnormed.reshape(-1)}, s.reshape(1, nv, d, d)


rows = []
try:
    q = torch.randn(1, 1, 1, nk * d).to(torch.bfloat16)
    k = torch.randn(1, 1, 1, nk * d).to(torch.bfloat16)
    v = torch.randn(1, 1, 1, nv * d).to(torch.bfloat16)
    z = torch.randn(1, 1, 1, nv * d).to(torch.bfloat16)
    decay_log = -torch.rand(1, 1, 1, nv) * 2
    beta = torch.rand(1, 1, 1, nv).to(torch.bfloat16)
    state = torch.randn(1, nv, d, d) * 0.1
    weight = (torch.rand(1, 1, 1, d) + 0.5).to(torch.bfloat16)
    tq, tk, tv, tz = (put(t, ttnn.bfloat16) for t in (q, k, v, z))
    td, tb = put(decay_log, ttnn.float32), put(beta, ttnn.bfloat16)
    ts, tw = put(state, ttnn.float32), put(weight, ttnn.bfloat16)
    expected_all, expected_state = reference(q, k, v, z, decay_log, beta, state, weight)
    blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    expected = expected_all[0]
    y = gdn_decode(tq, tk, tv, tz, td, tb, ts, tw, groups=groups, blocks=blocks)
    got = ttnn.to_torch(y).double().reshape(-1)[: nv * d]
    got_state = ttnn.to_torch(ts).double()
    print("FIRST", expected[:6].tolist(), got[:6].tolist())

    def pcc(a, b):
        return float(torch.corrcoef(torch.stack((a.reshape(-1), b.reshape(-1))))[0, 1])

    row = dict(nk=nk, nv=nv, blocks=blocks, output_pcc=pcc(expected, got), output_max_rel=float(((expected.reshape(-1) - got).abs().max()) / expected.abs().max()),
               state_pcc=pcc(expected_state, got_state),
               state_max_abs=float((expected_state - got_state).abs().max()))
    print(json.dumps(row), flush=True)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(20):
        gdn_decode(tq, tk, tv, tz, td, tb, ts, tw, groups=groups, blocks=blocks)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(10):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) / 20 * 1e6)
    ttnn.release_trace(mesh, tid)
    row["trace_us"] = sorted(samples)[5]
    print(json.dumps(row), flush=True)
    rows.append(row)
    json.dump(rows, open(out, "w"), indent=1)
finally:
    ttnn.close_mesh_device(mesh)
