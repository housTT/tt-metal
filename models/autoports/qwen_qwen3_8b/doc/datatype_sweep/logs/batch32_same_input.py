"""Batch-32 same-input and short-prefix case for a KV-cache dtype candidate.

Usage: batch32_same_input.py <reference.refpt> <output.json> [<steps>]
Every one of 32 users gets the reference's short prompt; the batch decodes <steps> greedy tokens in the
32-row bucket and a solo arm decodes the same prompt alone. Per user: greedy tokens against the HF
greedy continuation in the reference (match count, first divergence) and bit-equality with the solo arm.
"""
import json
import os
import subprocess
import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params

ref_path, out_path = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
ref = torch.load(ref_path, map_location="cpu", weights_only=False)
entry = ref["entries"][0]
prompt = entry["prompt_tokens"][0].tolist()
hf_tokens = entry["generated_tokens"][0].tolist()[:steps]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
gen = build_generator("models/autoports/qwen_qwen3_8b", mesh)
batch = 32


def run(rows: int) -> list[list[int]]:
    gen.reset()
    tokens = torch.tensor([prompt] * rows, dtype=torch.int64)
    for i in range(rows):
        gen._ensure_blocks(i, len(prompt) + steps)
    table = gen._host_page_table(list(range(rows)))
    first = gen.prefill_forward(
        tokens,
        page_table=table,
        kv_cache=gen.kv_cache,
        prompt_lens=[len(prompt)] * rows,
        sampling_params=greedy_sampling_params(rows),
    )
    generated = [[int(first[i])] for i in range(rows)]
    positions = torch.full((rows,), len(prompt), dtype=torch.int32)
    for _ in range(1, steps):
        nxt = torch.tensor([[generated[i][-1]] for i in range(rows)])
        out = gen.decode_forward(
            nxt, positions, page_table=table, kv_cache=gen.kv_cache, sampling_params=greedy_sampling_params(rows)
        )
        for i in range(rows):
            generated[i].append(int(out[i]))
        positions = positions + 1
    return generated


t0 = time.time()
solo = run(1)[0]
batched = run(batch)
counters = gen.counter_snapshot()


def first_div(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)


per_user = [
    {
        "user": i,
        "hf_matches": sum(a == b for a, b in zip(batched[i], hf_tokens)),
        "first_divergence_vs_hf": first_div(batched[i], hf_tokens),
        "first_divergence_vs_solo": first_div(batched[i], solo),
        "equal_to_solo": batched[i] == solo,
    }
    for i in range(batch)
]
result = {
    "reference": ref_path,
    "prompt_tokens": len(prompt),
    "steps": steps,
    "precision_config_id": gen.precision_config["config_id"],
    "precision_config_path": str(gen.precision_config_path),
    "kv_cache_dtype": str(gen.kv_cache_dtype),
    "solo_hf_matches": sum(a == b for a, b in zip(solo, hf_tokens)),
    "solo_first_divergence_vs_hf": first_div(solo, hf_tokens),
    "solo_text": gen.tokenizer.decode(solo),
    "hf_text": gen.tokenizer.decode(hf_tokens),
    "min_hf_matches_over_users": min(u["hf_matches"] for u in per_user),
    "users_equal_to_solo": sum(u["equal_to_solo"] for u in per_user),
    "distinct_user_outputs": len({tuple(b) for b in batched}),
    "per_user": per_user,
    "batched_user0_text": gen.tokenizer.decode(batched[0]),
    "counters": counters,
    "seconds": round(time.time() - t0, 1),
    "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
}
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(
    f"solo hf matches {result['solo_hf_matches']}/{steps} (div {result['solo_first_divergence_vs_hf']}); "
    f"batched min hf matches {result['min_hf_matches_over_users']}, users equal to solo {result['users_equal_to_solo']}/32, "
    f"distinct outputs {result['distinct_user_outputs']}",
    flush=True,
)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("BATCH32_DONE", flush=True)
