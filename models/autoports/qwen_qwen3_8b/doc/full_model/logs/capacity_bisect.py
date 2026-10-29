"""Bisect the capacity probe: the same pool and rows, one stage at a time with a device sync and a print after each."""
import bz2
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import Qwen3Generator, greedy_sampling_params
from models.autoports.qwen_qwen3_8b.tt.hf_weights import CheckpointReader
from models.autoports.qwen_qwen3_8b.tt.model import Qwen3Model


def stage(name, started=[time.time()]):
    ttnn.synchronize_device(mesh)
    print(f"STAGE {name} ok at {time.time() - started[0]:.1f}s", flush=True)


mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
reader = CheckpointReader()
cfg = AutoConfig.from_pretrained(str(reader.snapshot), local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(reader.snapshot), local_files_only=True)
model = Qwen3Model(mesh, hf_config=cfg, reader=reader)
budget = Qwen3Generator.max_tokens_all_users(model, mesh_device=mesh) // 32
pool = int(sys.argv[1]) if len(sys.argv) > 1 else budget
print("pool blocks", pool, "budget", budget, flush=True)
gen = Qwen3Generator(mesh, model, tokenizer, num_blocks=pool)
stage("build")
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tokenizer.encode(text, add_special_tokens=False)
rows = [ids[:40959], ids[1000 : 1000 + 40960], ids[5000:5200]]
gen.reset()
for i, r in enumerate(rows):
    gen._ensure_blocks(i, len(r))
table = gen._host_page_table([0, 1, 2])
stage("tables")
tt_table = ttnn.from_torch(
    gen._redirect_table(table, [len(r) for r in rows]), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
)
hiddens = []
for i, r in enumerate(rows):
    h = model.prefill_hidden(r, start_pos=0, user_slot=i, page_table=tt_table, kv_cache=gen.kv_cache)
    stage(f"prefill row {i} len {len(r)}")
    piece = model.gather_rows(h, [len(r) - 1])
    h.deallocate(True)
    stage(f"gather row {i}")
    hiddens.append(piece)
tile = model.rows_to_tile(hiddens)
stage("rows_to_tile")
logits = model.logits_from_tile(tile)
stage("logits_from_tile")
bucket = gen.buckets[8]
gen._apply_sampling_params(greedy_sampling_params(3), 3)
gen._sample_eager(logits, bucket)
stage("sample_eager")
first = gen._read_tokens(bucket, 3)
print("first tokens", first.tolist(), flush=True)
positions = torch.tensor([40959, -1, 200], dtype=torch.int32)
out = gen.decode_forward(
    torch.tensor([[int(first[0])], [0], [int(first[2])]]),
    positions,
    page_table=table,
    kv_cache=gen.kv_cache,
    sampling_params=greedy_sampling_params(3),
)
stage("decode step 1 (capture + replay)")
print("decode tokens", out.tolist(), flush=True)
out = gen.decode_forward(
    torch.tensor([[int(out[0])], [0], [int(out[2])]]),
    positions + 1,
    page_table=table,
    kv_cache=gen.kv_cache,
    sampling_params=greedy_sampling_params(3),
)
stage("decode step 2")
print("decode tokens", out.tolist(), flush=True)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("BISECT_OK", flush=True)
