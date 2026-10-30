#!/usr/bin/env python3
"""Continuation prefill at the context end: starts whose bucket-rounded window would pass 40960.

For each (start, end) case the generator prefills the prefix fresh, continues at ``start`` and gathers the
last 64 logits; the reference is the same positions from a fresh prefill of the whole prompt through the
same row gather. PCC and top-1 agreement per case; then a decode capture, the same continuations through
the served entry (``prefill_forward``) and the program-cache count before and after.
"""

from __future__ import annotations

import bz2
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator, greedy_sampling_params

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
LAYERS = int(os.getenv("QWEN3_FULL_MODEL_LAYERS", "36"))
OUT = Path(sys.argv[1])
CASES = ((40000, 40959), (32, 39969), (40928, 40959), (36869, 40960), (40959, 40960))
SERVED_CASES = CASES + ((40100, 40959), (39000, 40959), (37000, 40959), (40500, 40960), (38000, 40933))
WINDOW = 64


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(
        "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        local_files_only=True,
    )
    text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
    book = tok.encode(text, add_special_tokens=False)
    assert len(book) >= 40960, len(book)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    kwargs = {} if LAYERS == 36 else {"num_layers": LAYERS}
    gen = build_generator(str(MODEL_DIR), mesh, **kwargs)
    record = {"layers": LAYERS, "cases": [], "status": "pass"}
    for start, end in CASES:
        ids = book[:end]
        positions = list(range(max(start, end - WINDOW), end))
        t0 = time.perf_counter()
        ref = gen.prefill_logits_continuation(ids, 0, positions=positions)[0]
        cont = gen.prefill_logits_continuation(ids, start, positions=positions)[0]
        pccs = [pcc(ref[i], cont[i]) for i in range(len(positions))]
        top1 = float((ref.argmax(-1) == cont.argmax(-1)).float().mean())
        case = {
            "start": start,
            "end": end,
            "lead_rows": gen.model.lead_rows(start, end - start),
            "positions": len(positions),
            "pcc_mean": sum(pccs) / len(pccs),
            "pcc_min": min(pccs),
            "top1": top1,
            "final_top1_equal": bool(ref[-1].argmax() == cont[-1].argmax()),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        record["cases"].append(case)
        print("A", json.dumps(case), flush=True)
        if case["pcc_min"] < 0.98 or case["top1"] < 0.98:
            record["status"] = "fail"
    gen.generate(book[:129], 4)
    before = mesh.num_program_cache_entries()
    served = []
    for start, end in SERVED_CASES:
        gen.reset()
        gen._ensure_blocks(0, end)
        table = gen._host_page_table([0])
        tokens = torch.tensor(book[:end], dtype=torch.int64).unsqueeze(0)
        t0 = time.perf_counter()
        out = gen.prefill_forward(
            tokens,
            page_table=table,
            kv_cache=gen.kv_cache,
            prompt_lens=[end],
            start_pos=torch.tensor([start]),
            sampling_params=greedy_sampling_params(1),
        )
        served.append(
            {
                "start": start,
                "end": end,
                "token": int(out[0]),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "program_cache": mesh.num_program_cache_entries(),
            }
        )
        print("B", json.dumps(served[-1]), flush=True)
    after = mesh.num_program_cache_entries()
    record["served_entry"] = {"program_cache_before": before, "program_cache_after": after, "runs": served}
    if after != before:
        record["status"] = "fail"
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print("STATUS", record["status"], OUT, flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
