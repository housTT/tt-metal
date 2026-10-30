#!/usr/bin/env python3
"""Drive the generator the way the plugin does and compare with the generator's own greedy loop.

Builds the generator without warmup, adopts a vLLM-sized cache (3584 blocks plus scratch), warms and
captures on it, then for each rendered qualitative prompt: ``prefill_forward`` with greedy sampling over a
vLLM-style page table (non-contiguous block ids allocated one block ahead), then ``decode_forward`` steps
with the deferred read split, the page table growing one block at a time, host tokens and positions as
the plugin sends them. Compares the 128 greedy tokens with the same-chip generator control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
CONTROL = Path(sys.argv[1])
OUT = Path(sys.argv[2])
NUM_BLOCKS = 3584
MAX_NEW = 128


def main() -> None:
    rendered = json.loads((MODEL_DIR / "doc/optimized_full_model/qualitative/rendered_prompts.json").read_text())
    control = {e["id"]: e["token_ids"] for e in json.loads((CONTROL / "tt_outputs.json").read_text())}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    gen = build_generator(str(MODEL_DIR), mesh, prepare=False, num_blocks=1)
    cache = gen.model.allocate_kv_cache(NUM_BLOCKS, dtype=gen.kv_cache_dtype)
    gen.adopt_cache(cache, scratch_block=NUM_BLOCKS)
    gen.prepare()
    gen.capture_decode_traces(cache, scratch_block=NUM_BLOCKS)
    gen.release_owned_cache()
    width = gen.page_table_width
    results = []
    next_block = 97
    for entry in rendered:
        ids = list(entry["token_ids"])
        blocks = []

        def ensure(tokens):
            nonlocal next_block
            while len(blocks) * 32 < tokens:
                blocks.append(next_block)
                next_block += 13

        def table():
            t = torch.zeros(1, width, dtype=torch.int32)
            t[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            return t

        ensure(len(ids) + 1)
        first = gen.prefill_forward(
            torch.tensor([ids], dtype=torch.int64),
            page_table=table(),
            kv_cache=cache,
            prompt_lens=[len(ids)],
            start_pos=torch.tensor([0]),
            sampling_params=greedy_sampling_params(1),
            scratch_block=NUM_BLOCKS,
        )
        out = [int(first[0])]
        pos = len(ids)
        for step in range(1, MAX_NEW):
            ensure(pos + 2)
            handle = gen.decode_forward(
                torch.tensor([[out[-1]]], dtype=torch.int32),
                torch.tensor([pos], dtype=torch.int32),
                page_table=table(),
                kv_cache=cache,
                enable_trace=True,
                read_from_device=False,
                sampling_params=greedy_sampling_params(1),
                reset_batch=(step == 1),
                scratch_block=NUM_BLOCKS,
            )
            handle, events = gen.read_decode_output(handle, async_read=True)
            for ev in events:
                ttnn.event_synchronize(ev)
            tok = gen.process_decode_output_host(handle, is_tokens=True)
            out.append(int(tok[0]))
            pos += 1
        ctrl = control[entry["id"]]
        n = min(len(out), len(ctrl))
        div = next((k for k in range(n) if out[k] != ctrl[k]), n)
        results.append(
            {"id": entry["id"], "first_divergence_vs_control": div, "identical": out[:n] == ctrl[:n], "tokens": out}
        )
        print(entry["id"], "first divergence vs control", div, flush=True)
    OUT.write_text(
        json.dumps({"num_blocks": NUM_BLOCKS, "results": results, "counters": gen.counter_snapshot()}, indent=2) + "\n"
    )
    print("SIM_DONE", all(r["identical"] for r in results), flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
