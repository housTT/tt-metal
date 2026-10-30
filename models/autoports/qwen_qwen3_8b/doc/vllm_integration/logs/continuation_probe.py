#!/usr/bin/env python3
"""Continuation prefill probe: numerics at unaligned starts, then program-cache stability under live traces.

Phase A: full prefill logits of a natural-text prompt against continuation logits at several starts
(block-aligned, tile-unaligned, mid-prompt), PCC and top-1 per start.
Phase B: generate a few tokens so the decode and sampling traces are captured, record the program-cache
count, then run continuations at new starts and lengths (including a one-token prefix-cache-style repeat
and a window longer than one chunk) and generate again: the program-cache count must not change and the
allocation tracker (set in the environment) must stay silent.
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
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
LAYERS = int(os.getenv("QWEN3_FULL_MODEL_LAYERS", "2"))
OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else MODEL_DIR / "doc" / "vllm_integration" / f"continuation_probe_L{LAYERS}.json"
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(
        "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
        local_files_only=True,
    )
    text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
    book = tok.encode(text[:60000], add_special_tokens=False)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
    started = time.perf_counter()
    gen = build_generator(str(MODEL_DIR), mesh, num_layers=LAYERS)
    record = {"layers": LAYERS, "build_s": round(time.perf_counter() - started, 1), "phase_a": [], "phase_b": {}}
    ids = book[:600]
    ref = gen.prefill_logits(ids)[0]
    window = 160
    for start in (32, 37, 68, 100, 288, 303, 544, 599):
        positions = list(range(start, min(600, start + window)))
        cont = gen.prefill_logits_continuation(ids, start, positions=positions)[0]
        ref_rows = ref[start : start + len(positions)]
        pccs = [pcc(ref_rows[i], cont[i]) for i in range(len(positions))]
        top1 = float((ref_rows.argmax(-1) == cont.argmax(-1)).float().mean())
        record["phase_a"].append(
            {
                "start": start,
                "positions": len(positions),
                "pcc_mean": sum(pccs) / len(pccs),
                "pcc_min": min(pccs),
                "top1": top1,
            }
        )
        print(
            f"A start={start:4d} n={len(positions):3d} pcc_mean={sum(pccs)/len(pccs):.4f} pcc_min={min(pccs):.4f} top1={top1:.3f}",
            flush=True,
        )
    pre = gen.generate(book[:129], 8)
    caches_before = mesh.num_program_cache_entries()
    counters_before = gen.counter_snapshot()
    print(f"B captured: program cache {caches_before}, counters {counters_before}", flush=True)
    phase_b = []
    from models.autoports.qwen_qwen3_8b.tt.generator import greedy_sampling_params

    for start, length in (
        (45, 100),
        (77, 300),
        (150, 1),
        (289, 50),
        (992, 8),
        (1000, 200),
        (100, 4100),
        (4095, 2),
        (31, 33),
        (4192, 4200),
        (8200, 5000),
    ):
        end = start + length
        gen.reset()
        gen._ensure_blocks(0, end)
        gen._ensure_blocks(1, 64)
        table = gen._host_page_table([0, 1])
        tokens = torch.zeros(2, end, dtype=torch.int64)
        tokens[0, :end] = torch.tensor(book[:end])
        tokens[1, :64] = torch.tensor(book[100:164])
        t0 = time.perf_counter()
        gen.prefill_forward(
            tokens,
            page_table=table,
            kv_cache=gen.kv_cache,
            prompt_lens=[end, 64],
            start_pos=torch.tensor([start, 0]),
            sampling_params=greedy_sampling_params(2),
        )
        phase_b.append(
            {
                "start": start,
                "length": length,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "program_cache": mesh.num_program_cache_entries(),
            }
        )
        print(
            f"B start={start} len={length} program_cache={mesh.num_program_cache_entries()} ms={phase_b[-1]['ms']}",
            flush=True,
        )
    post = gen.generate(book[:129], 8)
    multi = gen.generate(book[:4200], 4)
    record["phase_b"] = {
        "program_cache_after_capture": caches_before,
        "program_cache_after_workload": mesh.num_program_cache_entries(),
        "runs": phase_b,
        "greedy_before_equals_after": pre == post,
        "counters_after": gen.counter_snapshot(),
        "prefill_4200_tokens_ok": len(multi) == 4,
    }
    record["status"] = (
        "pass"
        if record["phase_b"]["program_cache_after_workload"] == caches_before
        and all(r["pcc_min"] >= 0.98 and r["top1"] >= 0.98 for r in record["phase_a"])
        else "fail"
    )
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["phase_b"], indent=1))
    print("STATUS", record["status"], OUT, flush=True)
    gen.teardown()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
