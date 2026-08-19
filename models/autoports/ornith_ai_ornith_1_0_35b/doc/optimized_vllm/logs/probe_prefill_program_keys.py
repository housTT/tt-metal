# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which prefill programs are keyed by the *logical* prompt length, and which by the physical block.

Every internal prefill block is padded up to a multiple of ``PREFILL_ALIGN`` (128), so two prompts
whose lengths round to the same physical block share almost every program - but not all of them: a
handful of ops take the logical length as a compile-time argument (slice bounds, comparison
constants), so they compile again for each new logical length. Those few programs are what makes a
first request at a new length pay a decode-trace re-capture
(``OrnithGenerator._ensure_traces_replay_safe``).

This probe counts both classes and, with ``--name-them``, uses
``set_program_cache_misses_allowed(False)`` so the first offending op raises with its own name.

    python .../doc/optimized_vllm/logs/probe_prefill_program_keys.py --output .../prefill_program_keys.json
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
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

MODEL_DIR = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--lengths", default="64,100,128,129,200,211,256,257")
    ap.add_argument("--name-them", action="store_true", help="forbid cache misses on the last length")
    ap.add_argument(
        "--warm-blocks",
        action="store_true",
        help=(
            "Run the shipped warm-up set first - one prompt at every physical block length - so the "
            "measured lengths below are counted against a server that has already compiled them. This "
            "is what isolates the chunk-offset class of a multi-chunk prompt from the block-shape class."
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    lengths = [int(v) for v in args.lengths.split(",")]
    layers = [int(v) for v in args.layers.split(",")] if args.layers else None
    mesh = open_ornith_mesh()
    report = {"layers": layers, "context": args.context, "rows": []}
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path, mesh_device=mesh, hf_config=load_text_config(path), layer_indices=layers, max_context=args.context
        )
        blocks = num_blocks_for_context(args.context, model.page_block_size)
        kv_cache = model.allocate_kv_cache(1 + blocks)
        table = torch.arange(1, blocks + 1, dtype=torch.int32).reshape(1, blocks)
        gen = OrnithGenerator(
            model,
            max_batch_size=1,
            cache_context=args.context,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=table,
        )
        # Capture the decode traces once, before the measured prefills, exactly as serving does.
        if args.warm_blocks:
            from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import PREFILL_ALIGN

            warmed = list(range(model.prefill_chunk, 0, -PREFILL_ALIGN))
            report["warmed_block_lengths"] = warmed
            before_warm = mesh.num_program_cache_entries()
            for length in warmed:
                gen.reset()
                gen.prefill_requests_into_slots(
                    torch.ones(1, length, dtype=torch.int32),
                    [length],
                    [0],
                    page_table=table,
                    kv_cache=kv_cache,
                    sample_on_device=False,
                    ensure_traces=False,
                )
            gen.reset()
            report["warmup_programs"] = mesh.num_program_cache_entries() - before_warm
            logger.info(f"warmed {len(warmed)} block length(s), {report['warmup_programs']} program(s)")
        gen.ensure_serving_traces()
        gen.ensure_sampling_trace()
        report["cache_entries_at_capture"] = mesh.num_program_cache_entries()
        for index, length in enumerate(lengths):
            before = mesh.num_program_cache_entries()
            if args.name_them and index == len(lengths) - 1:
                mesh.set_program_cache_misses_allowed(False)
            try:
                recaptures = gen.trace_recaptures
                gen.reset()
                gen.prefill_requests_into_slots(
                    torch.randint(10, 90000, (1, length)),
                    [length],
                    [0],
                    page_table=table,
                    kv_cache=kv_cache,
                    sample_on_device=True,
                )
                # One decode step is what actually triggers the replay-safety re-capture.
                gen.stage_serving_decode_inputs(
                    torch.zeros(1, dtype=torch.int64),
                    torch.tensor([length], dtype=torch.int64),
                    table,
                    full_refresh=True,
                )
                gen.submit_serving_decode(sample_on_device=True)
                gen.read_tokens()
                error = None
            except Exception as exc:  # noqa: BLE001 - the message is the result
                error = f"{type(exc).__name__}: {exc}"
            after = mesh.num_program_cache_entries()
            chunk = int(model.prefill_chunk)
            row = {
                "logical_len": length,
                "chunks": (length + chunk - 1) // chunk,
                "chunk_start_positions": list(range(0, length, chunk)),
                "physical_block": min(chunk, ((length % chunk or chunk) + 127) // 128 * 128),
                "programs_compiled": after - before,
                "cache_entries": after,
                "trace_recaptures": gen.trace_recaptures - recaptures,
                "error": error,
            }
            report["rows"].append(row)
            logger.info(f"len {length:5d} -> phys {row['physical_block']:5d}: +{row['programs_compiled']} programs")
            if error:
                logger.warning(error)
    finally:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        logger.info(f"wrote {args.output}")
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
