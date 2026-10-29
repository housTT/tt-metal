"""Attribute the batch-32 short-prefix PCC drop (user 20, prefix 32, layer 0): cache dtype x weight policy."""
import json
import sys

import torch

import ttnn

sys.path.insert(0, "/home/hous/dev/tt-metal-wt-qwen3")
from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as base
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import ALL_BFP8_POLICY, DEFAULT_POLICY, OptimizedDecoder

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
fixture = base._layer(0)
batch = 32
lens = [base.DECODE_LENGTH_CHOICES[u % len(base.DECODE_LENGTH_CHOICES)] for u in range(batch)]
blocks = base._blocks_for(max(lens) + 2)
table, page_table = base._page_table(mesh, batch, blocks, seed=13)
hf_rows, tokens = [], []
for user in range(batch):
    hidden = fixture.inputs(lens[user] + 1, seed=820 + user)
    reference = fixture.reference()
    reference.forward(hidden[:, : lens[user]], 0)
    hf_rows.append(reference.forward(hidden[:, lens[user] :], lens[user])[0])
    tokens.append(hidden[:, lens[user] :])
hf_batch = torch.cat(hf_rows, dim=0)
rows = []
for pname, policy in (("default", DEFAULT_POLICY), ("all_bfp8", ALL_BFP8_POLICY)):
    decoder = OptimizedDecoder.from_state_dict(
        fixture.state, hf_config=fixture.config, layer_idx=0, mesh_device=mesh, policy=policy
    )
    for cname, cdtype in (("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)):
        kv_cache = OptimizedDecoder.allocate_kv_cache(
            mesh, num_blocks=batch * blocks, block_size=32, num_kv_heads=8, head_dim=128, dtype=cdtype
        )
        for user in range(batch):
            hidden = fixture.inputs(lens[user] + 1, seed=820 + user)
            decoder.prefill_forward(
                base._tt(mesh, hidden[:, : lens[user]].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user,
            )
        xd = base._tt(mesh, torch.cat(tokens, dim=1).reshape(1, 1, batch, base.HIDDEN))
        cur = base._tt(mesh, torch.tensor(lens, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = base._to_torch(decoder.decode_forward(xd, current_pos=cur, page_table=page_table, kv_cache=kv_cache))[
            0, 0
        ]
        pcc = [base._pcc(hf_batch[u], out[u]) for u in range(batch)]
        row = dict(
            policy=pname,
            cache=cname,
            min_pcc=min(pcc),
            argmin=pcc.index(min(pcc)),
            user20=pcc[20],
            below_995=[(u, round(pcc[u], 5), lens[u]) for u in range(batch) if pcc[u] < 0.995],
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        kv_cache[0].deallocate(True)
        kv_cache[1].deallocate(True)
    for t in decoder.weights.__dict__.values():
        t.deallocate(True)
    for t in decoder.prefill_weights.values():
        t.deallocate(True)
json.dump(rows, open("/tmp/qwen3_opt/exp10.json", "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
