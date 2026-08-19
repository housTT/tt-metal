# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the serving prefill's program set keyed by the decode **slot** it writes into?

A `--max-num-seqs 32` server logs decode-trace re-captures of 19-64 programs right after a
multi-request prefill and after a slot remap, which the 16-block warm-up does not remove because that
warm-up only ever prefills into slot 0. This probe isolates the key: it warms every physical block
(into slot 0, exactly as the shipped warm-up does), then prefills the *same* prompt into each slot in
turn and counts what each one compiles, then runs a slot remap and counts that.

    python .../doc/optimized_vllm/logs/probe_slot_program_keys.py --output .../candidates/slot_program_keys.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    OrnithModel,
    close_ornith_mesh,
    load_text_config,
    open_ornith_mesh,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import PREFILL_ALIGN, num_blocks_for_context

MODEL_DIR = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")] if args.layers else None
    mesh = open_ornith_mesh()
    report = {
        "layers": layers,
        "context": args.context,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "slots": [],
        "remaps": [],
    }
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path, mesh_device=mesh, hf_config=load_text_config(path), layer_indices=layers, max_context=args.context
        )
        blocks = num_blocks_for_context(args.context, model.page_block_size)
        kv_cache = model.allocate_kv_cache(1 + args.batch * blocks)
        table = torch.zeros(args.batch, blocks, dtype=torch.int32)
        for slot in range(args.batch):
            base = 1 + slot * blocks
            table[slot] = torch.arange(base, base + blocks, dtype=torch.int32)
        gen = OrnithGenerator(
            model,
            max_batch_size=args.batch,
            cache_context=args.context,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=table,
        )

        # The shipped warm-up: every physical block, into slot 0.
        warmed = list(range(model.prefill_chunk, 0, -PREFILL_ALIGN))
        before = mesh.num_program_cache_entries()
        for length in warmed:
            gen.reset()
            gen.prefill_requests_into_slots(
                torch.ones(1, length, dtype=torch.int32),
                [length],
                [0],
                page_table=table[0:1],
                kv_cache=kv_cache,
                sample_on_device=False,
                ensure_traces=False,
            )
        gen.reset()
        report["warmup_programs"] = mesh.num_program_cache_entries() - before
        gen.ensure_serving_traces()
        gen.ensure_sampling_trace()
        logger.info(f"warmed {len(warmed)} block(s), {report['warmup_programs']} program(s)")

        prompt = torch.randint(10, 90000, (1, args.prompt_len))
        for slot in range(args.batch):
            b = mesh.num_program_cache_entries()
            r = gen.trace_recaptures
            gen.prefill_requests_into_slots(
                prompt,
                [args.prompt_len],
                [slot],
                page_table=table[slot : slot + 1],
                kv_cache=kv_cache,
                sample_on_device=True,
            )
            row = {
                "slot": slot,
                "programs_compiled": mesh.num_program_cache_entries() - b,
                "trace_recaptures": gen.trace_recaptures - r,
            }
            report["slots"].append(row)
            logger.info(f"prefill into slot {slot}: +{row['programs_compiled']} program(s)")

        for width in (2, 4, args.batch):
            perm = torch.arange(args.batch)
            perm[:width] = torch.roll(perm[:width], 1)
            b = mesh.num_program_cache_entries()
            gen.remap_serving_slots(perm)
            row = {"remap_width": width, "programs_compiled": mesh.num_program_cache_entries() - b}
            report["remaps"].append(row)
            logger.info(f"remap of width {width}: +{row['programs_compiled']} program(s)")
    finally:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        logger.info(f"wrote {args.output}")
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
