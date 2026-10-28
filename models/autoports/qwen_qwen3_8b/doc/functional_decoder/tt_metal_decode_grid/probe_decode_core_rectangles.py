import sys

import torch

import ttnn

sys.argv = ["x"]
from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as T
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as FD

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    fx = T._layer(0)
    dec = T._decoder(mesh, fx)
    cache = {}

    def users(batch):
        if batch in cache:
            return cache[batch]
        lens = [T.DECODE_LENGTH_CHOICES[(u * 7) % len(T.DECODE_LENGTH_CHOICES)] for u in range(batch)]
        hs = [fx.inputs(lens[u] + 2, seed=700 + u) for u in range(batch)]
        hf = []
        for u in range(batch):
            r = fx.reference()
            r.forward(hs[u][:, : lens[u]], 0)
            hf.append(r.forward(hs[u][:, lens[u] : lens[u] + 1], lens[u]))
        cache[batch] = (lens, hs, torch.cat(hf, dim=1))
        return cache[batch]

    orig = FD.decode_physical_batch
    for batch, x, y in (
        (14, 7, 2),
        (14, 2, 7),
        (16, 8, 2),
        (16, 4, 4),
        (16, 2, 8),
        (12, 6, 2),
        (12, 4, 3),
        (12, 3, 4),
        (24, 8, 3),
        (24, 6, 4),
        (32, 8, 4),
        (32, 4, 8),
        (8, 8, 1),
        (8, 4, 2),
    ):
        FD.decode_physical_batch = lambda b, gx, gy, max_batch=32, _r=(batch, x, y): _r
        lens, hs, hf = users(batch)
        blocks = T._blocks_for(max(lens) + 3)
        table, page_table = T._page_table(mesh, batch, blocks, seed=9)
        kv = T._kv_cache(mesh, batch * blocks)
        for u in range(batch):
            dec.prefill_forward(
                T._tt(mesh, hs[u][:, : lens[u]].unsqueeze(1)), page_table=page_table, kv_cache=kv, user_id=u
            )
        tokens = torch.cat([hs[u][:, lens[u] : lens[u] + 1] for u in range(batch)], dim=1).reshape(
            1, 1, batch, T.HIDDEN
        )
        xd = T._tt(mesh, tokens)
        cpos = T._tt(mesh, torch.tensor(lens, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        try:
            out = T._to_torch(dec.decode_forward(xd, current_pos=cpos, page_table=page_table, kv_cache=kv))
            pccs = [T._pcc(hf[:, u], out[0, 0, u]) for u in range(batch)]
            bad = [u for u in range(batch) if pccs[u] < 0.99]
            print(f"batch {batch} grid {x}x{y}: min pcc {min(pccs):.4f} bad users {bad}", flush=True)
        except Exception as e:
            print(f"batch {batch} grid {x}x{y}: EXCEPTION {str(e).splitlines()[0][:100]}", flush=True)
        for t in kv:
            t.deallocate(True)
    FD.decode_physical_batch = orig
finally:
    ttnn.close_mesh_device(mesh)
