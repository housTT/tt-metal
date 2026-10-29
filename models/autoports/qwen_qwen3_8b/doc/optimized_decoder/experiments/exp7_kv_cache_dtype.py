"""KV-cache dtype trial: BF16 versus BFP8 cache with the default optimized policy (real weights, HF PCC, traced decode time)."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import OptimizedDecoder

OUT = os.getenv("EXP7_OUT", "/tmp/qwen3_opt/exp7.json")
layers = [int(v) for v in os.getenv("EXP7_LAYERS", "0,17").split(",")]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
rows = []
for layer_idx in layers:
    fixture = base._layer(layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=layer_idx, mesh_device=mesh
    )
    seq_len = 1024
    hidden = fixture.inputs(seq_len + 2, seed=900)
    reference = fixture.reference()
    hf_prefill = reference.forward(hidden[:, :seq_len], 0)
    hf_decode = [reference.forward(hidden[:, seq_len + i : seq_len + i + 1], seq_len + i) for i in range(2)]
    for dtype_name, dtype in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        row = {"layer": layer_idx, "kv_cache_dtype": dtype_name}
        print("START", row, flush=True)
        try:
            blocks = base._blocks_for(seq_len + 2)
            table, page_table = base._page_table(mesh, 1, blocks, seed=19)
            kv_cache = (
                base._kv_cache(mesh, blocks)
                if dtype_name == "bf16"
                else OptimizedDecoder.allocate_kv_cache(
                    mesh, num_blocks=blocks, block_size=32, num_kv_heads=8, head_dim=128, dtype=dtype
                )
            )
            x = base._tt(mesh, hidden[:, :seq_len].unsqueeze(1))
            out = base._to_torch(decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=0))[:, 0]
            row["prefill_pcc"] = base._pcc(hf_prefill, out)
            row["prefill_delta_pcc"] = base._delta_pcc(hf_prefill, out, hidden[:, :seq_len])
            kc = base._to_torch(kv_cache[0])[table[0].long()].reshape(-1, 8, 128)[:seq_len]
            dec = []
            for i in range(2):
                position = seq_len + i
                xd = base._tt(mesh, hidden[:, position : position + 1].reshape(1, 1, 1, base.HIDDEN))
                cur = base._tt(
                    mesh, torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                od = base._to_torch(
                    decoder.decode_forward(xd, current_pos=cur, page_table=page_table, kv_cache=kv_cache)
                ).reshape(1, 1, base.HIDDEN)
                dec.append(
                    (base._pcc(hf_decode[i], od), base._delta_pcc(hf_decode[i], od, hidden[:, position : position + 1]))
                )
            row["decode_pcc"] = dec
            kv_cache[0].deallocate(True)
            kv_cache[1].deallocate(True)
            for batch, context in ((1, 1024), (32, 1024), (1, 8192), (32, 8192)):
                blocks = base._blocks_for(context + 2)
                tb, pt = base._page_table(mesh, batch, blocks, seed=23)
                cache = OptimizedDecoder.allocate_kv_cache(
                    mesh, num_blocks=batch * blocks, block_size=32, num_kv_heads=8, head_dim=128, dtype=dtype
                )
                xd = base._tt(
                    mesh, hidden[:, seq_len : seq_len + 1].reshape(1, 1, 1, base.HIDDEN).repeat(1, 1, batch, 1)
                )
                cur = base._tt(
                    mesh,
                    torch.full((batch,), context, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                decoder.decode_forward(xd, current_pos=cur, page_table=pt, kv_cache=cache)
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = decoder.decode_forward(xd, current_pos=cur, page_table=pt, kv_cache=cache)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    best = min(best, time.perf_counter() - t0)
                row[f"decode_trace_us_batch{batch}_ctx{context}"] = best * 1e6
                ttnn.release_trace(mesh, tid)
                cache[0].deallocate(True)
                cache[1].deallocate(True)
        except Exception as e:
            row["error"] = str(e).splitlines()[0][:300]
        print(json.dumps(row), flush=True)
        rows.append(row)
    for t in decoder.weights.__dict__.values():
        t.deallocate(True)
json.dump(rows, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
