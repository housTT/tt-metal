"""Capacity probe: the full model with the KV pool at Qwen3Generator.max_tokens_all_users (two users at the
advertised 40960-token context plus spare), slot 0 prefills 40959 tokens (non-aligned, full context minus the
generated token), slots 1 and 2 prefill 40960 and 200 tokens, then eight rows (two live, the rest padded) decode
for 4 steps. A pool of 3 x 1280 blocks does not fit (logs/capacity_probe_3x1280.log).
"""
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import Qwen3Generator, greedy_sampling_params

OUT = Path("models/autoports/qwen_qwen3_8b/doc/full_model/capacity_probe_budget.json")
from transformers import AutoConfig, AutoTokenizer

from models.autoports.qwen_qwen3_8b.tt.hf_weights import CheckpointReader
from models.autoports.qwen_qwen3_8b.tt.model import Qwen3Model

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)


def view():
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {k: int(getattr(v, k)) for k in dir(v) if not k.startswith("_") and isinstance(getattr(v, k), int)}


fresh = view()
reader = CheckpointReader()
cfg = AutoConfig.from_pretrained(str(reader.snapshot), local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(reader.snapshot), local_files_only=True)
t0 = time.time()
model = Qwen3Model(mesh, hf_config=cfg, reader=reader)
after_weights = view()
budget_tokens = Qwen3Generator.max_tokens_all_users(model, mesh_device=mesh)
budget = budget_tokens // 32
print("dram fresh", fresh, "after weights", after_weights, "budget blocks", budget, flush=True)
gen = Qwen3Generator(mesh, model, tokenizer, num_blocks=budget)
build_s = time.time() - t0
after_build = view()
tok = gen.tokenizer
import bz2

text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=False)
lengths = [40955, 40960, 200]
prompts = [ids[:40955], ids[1000 : 1000 + 40960], ids[5000:5200]]
for i, n in enumerate(lengths):
    assert len(prompts[i]) == n
gen.reset()
record = {
    "num_blocks": budget,
    "budget_tokens": budget_tokens,
    "build_s": build_s,
    "dram_view_fresh": fresh,
    "dram_view_after_weights": after_weights,
    "dram_view_after_build": after_build,
    "weight_bytes_computed": gen.model.weight_bytes(),
    "kv_bytes_per_token": gen.model.kv_bytes_per_token(),
    "rows": [],
}
first_tokens = []
for i, ids_i in enumerate(prompts):
    gen._ensure_blocks(i, len(ids_i) if len(ids_i) == 40960 else len(ids_i) + 1)
table = gen._host_page_table([0, 1, 2])
t0 = time.time()
tokens = torch.zeros(3, 40960, dtype=torch.int64)
for i, ids_i in enumerate(prompts):
    tokens[i, : len(ids_i)] = torch.tensor(ids_i)
first = gen.prefill_forward(
    tokens, page_table=table, kv_cache=gen.kv_cache, prompt_lens=lengths, sampling_params=greedy_sampling_params(3)
)
record["prefill_3_rows_s"] = time.time() - t0
record["first_tokens"] = first.tolist()
positions = torch.tensor(lengths, dtype=torch.int32)
live = [i for i in range(3) if lengths[i] < 40960]
steps = []
cur = first.clone()
pos = positions.clone()
for step in range(4):
    rows = torch.tensor([[int(cur[i])] if i in live else [0] for i in range(3)])
    p = torch.tensor([int(pos[i]) if i in live else -1 for i in range(3)], dtype=torch.int32)
    print(f"decode step {step} positions {p.tolist()}", flush=True)
    out = gen.decode_forward(
        rows, p, page_table=table, kv_cache=gen.kv_cache, sampling_params=greedy_sampling_params(3)
    )
    print(f"decode step {step} tokens {out.tolist()}", flush=True)
    steps.append(out.tolist())
    for i in live:
        cur[i] = out[i]
        pos[i] += 1
record["decode_steps_live_rows"] = {"live_rows": live, "tokens": steps}
record["decoded_text_row0"] = tok.decode([int(first[0])] + [s[0] for s in steps])
record["dram_view_after_probe"] = view()
record["counters"] = gen.counter_snapshot()
OUT.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=1), flush=True)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("CAPACITY_PROBE_OK", flush=True)
