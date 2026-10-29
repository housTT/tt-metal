"""Whole-layer policy sweep for the Qwen3-8B optimized decoder: real layer-0 weights, HF reference PCC,
warmed prefill (1024 tokens) and traced decode (batch 1 and 32) latency per OptimizedPolicy candidate."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from dataclasses import replace

from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import DEFAULT_POLICY
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import DecodeMatmulRole as R
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import OptimizedDecoder, OptimizedPolicy

OUT = os.getenv("EXP5_OUT", "/tmp/qwen3_opt/exp5.json")
POLICIES = {
    "bfp8_hifi2": OptimizedPolicy(
        R("bfp8", "hifi2", False, 32, 2, 3),
        R("bfp8", "hifi2", False, 32, 4, 2),
        R("bfp8", "hifi2", False, 32, 2, 3),
        R("bfp8", "hifi2", False, 32, 12, 2),
    ),
    "bfp8_lofi": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 6, 1),
    ),
    "attn8_mlp4": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp4", "lofi", False, 32, 12, 2),
    ),
    "attn8_mlp4_b2": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp4", "lofi", False, 32, 2, 3),
        R("bfp4", "lofi", False, 32, 12, 2),
    ),
    "attn8_mlp4_fp32": OptimizedPolicy(
        R("bfp8", "lofi", True, 32, 4, 1),
        R("bfp8", "lofi", True, 32, 4, 1),
        R("bfp4", "lofi", True, 32, 4, 3),
        R("bfp4", "lofi", True, 32, 12, 2),
    ),
    "attn8_gateup4_down8": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp8", "lofi", False, 32, 6, 1),
    ),
    "attn4_mlp4": OptimizedPolicy(
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp4", "lofi", False, 32, 4, 2),
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp4", "lofi", False, 32, 12, 2),
    ),
    "attn8_gate8_up4_down8": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 6, 1),
        up=R("bfp4", "lofi", False, 32, 4, 3),
    ),
    "attn8_gate4_up8_down8": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp8", "lofi", False, 32, 6, 1),
        up=R("bfp8", "lofi", False, 32, 4, 1),
    ),
    "attn8_gate8_up8_down4": OptimizedPolicy(
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp8", "lofi", False, 32, 4, 1),
        R("bfp4", "lofi", False, 32, 12, 2),
    ),
    "default": None,
    "default_prefill_lofi": "prefill_lofi",
    "attn8_hifi2_mlp4": OptimizedPolicy(
        R("bfp8", "hifi2", False, 32, 2, 3),
        R("bfp8", "hifi2", False, 32, 4, 2),
        R("bfp4", "lofi", False, 32, 4, 3),
        R("bfp4", "lofi", False, 32, 12, 2),
    ),
}
selected = os.getenv("EXP5_POLICIES")
names = selected.split(",") if selected else list(POLICIES)
layer_idx = int(os.getenv("EXP5_LAYER", "0"))

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
fixture = base._layer(layer_idx)
seq_len = 1024
hidden = fixture.inputs(seq_len + 2, seed=900)
reference = fixture.reference()
hf_prefill = reference.forward(hidden[:, :seq_len], 0)
hf_decode = [reference.forward(hidden[:, seq_len + i : seq_len + i + 1], seq_len + i) for i in range(2)]
rows = []
for name in names:
    policy = POLICIES[name]
    if policy is None:
        policy = DEFAULT_POLICY
    elif policy == "prefill_lofi":
        policy = replace(DEFAULT_POLICY, prefill_bfp8_fidelity="lofi")
    row = {"policy": name, "spec": policy.describe()}
    print("START", name, flush=True)
    try:
        t0 = time.time()
        decoder = OptimizedDecoder.from_state_dict(
            fixture.state, hf_config=fixture.config, layer_idx=layer_idx, mesh_device=mesh, policy=policy
        )
        row["construct_s"] = time.time() - t0
        row["weight_bytes_per_token"] = decoder.weight_bytes_per_token()
        blocks = base._blocks_for(seq_len + 2)
        table, page_table = base._page_table(mesh, 1, blocks, seed=19)
        kv_cache = base._kv_cache(mesh, blocks)
        x = base._tt(mesh, hidden[:, :seq_len].unsqueeze(1))
        out = decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=0)
        out_t = base._to_torch(out)[:, 0]
        row["prefill_pcc"] = base._pcc(hf_prefill, out_t)
        row["prefill_delta_pcc"] = base._delta_pcc(hf_prefill, out_t, hidden[:, :seq_len])
        times = []
        for _ in range(3):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            o = decoder.prefill_forward(x, page_table=page_table, kv_cache=kv_cache, user_id=0)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - t0) * 1e3)
            o.deallocate(True)
        row["prefill_host_ms_best"] = min(times)
        decode_pcc = []
        for i in range(2):
            position = seq_len + i
            xd = base._tt(mesh, hidden[:, position : position + 1].reshape(1, 1, 1, base.HIDDEN))
            cur = base._tt(
                mesh, torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            od = base._to_torch(
                decoder.decode_forward(xd, current_pos=cur, page_table=page_table, kv_cache=kv_cache)
            ).reshape(1, 1, base.HIDDEN)
            decode_pcc.append(
                (base._pcc(hf_decode[i], od), base._delta_pcc(hf_decode[i], od, hidden[:, position : position + 1]))
            )
        row["decode_pcc"] = decode_pcc
        for batch in (1, 32):
            xd = base._tt(mesh, hidden[:, seq_len : seq_len + 1].reshape(1, 1, 1, base.HIDDEN).repeat(1, 1, batch, 1))
            tb, pt = base._page_table(mesh, batch, blocks, seed=23)
            cache = base._kv_cache(mesh, batch * blocks)
            cur = base._tt(
                mesh, torch.full((batch,), seq_len, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
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
            row[f"decode_trace_us_batch{batch}"] = best * 1e6
            ttnn.release_trace(mesh, tid)
            cache[0].deallocate(True)
            cache[1].deallocate(True)
        for t in decoder.weights.__dict__.values():
            t.deallocate(True)
        decoder.ones_row.deallocate(True)
        kv_cache[0].deallocate(True)
        kv_cache[1].deallocate(True)
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:400]
    print(json.dumps(row), flush=True)
    rows.append(row)
    json.dump(rows, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
