# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What one *idle* serving slot costs, on a server sized for many.

A vLLM server launched with ``--max-num-seqs N`` captures **one** decode trace at batch ``N`` and
replays it whatever the occupancy is: a single user on a 32-slot server still pays a 32-row decode
step. This probe measures that step directly, without vLLM in the way, as a function of how many of
the 32 rows are active (``current_pos >= 0``); the idle rows carry the ``-1`` sentinel exactly as a
served idle slot does.

The loop is the serving loop: ``submit_serving_decode`` (model trace replay + sampling trace replay,
both non-blocking) plus the pipelined async read, so the number is directly comparable to the
serving TPOT the benchmark reports and to ``doc/datatype_sweep/post_selection_token_out.json``.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_vllm/logs/probe_batch_occupancy.py \
        --output .../doc/optimized_vllm/before/batch_occupancy.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
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


def _page_table(batch, blocks_per_user):
    """A vLLM-shaped table: slot u owns its own run of real blocks, 0 (the null block) elsewhere."""
    table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
    for user in range(batch):
        base = 1 + user * blocks_per_user
        table[user] = torch.arange(base, base + blocks_per_user, dtype=torch.int32)
    return table


def _time_steps(gen, steps):
    """Warmed, pipelined serving decode: replay + sampling replay + async read, no host feedback."""
    pending = None
    start = time.perf_counter()
    for _ in range(steps):
        gen.submit_serving_decode(sample_on_device=True)
        nxt = gen.read_output_async()
        if pending is not None:
            ttnn.event_synchronize(pending[1])
        pending = nxt
    ttnn.event_synchronize(pending[1])
    return (time.perf_counter() - start) / steps * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32, help="captured decode batch (the server's --max-num-seqs)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warm-steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--active", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--layers", type=str, default="", help="comma-separated layer indices (reduced target)")
    ap.add_argument(
        "--token-mode",
        choices=("shared", "distinct"),
        default="shared",
        help=(
            "What the 32 rows carry. `shared` gives every row the same id, so the routed-expert union "
            "is one token's worth however many rows are active; `distinct` gives every row its own id, "
            "which is what a real serving batch (and a real *idle* slot, holding whatever it last "
            "decoded) looks like."
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    actives = [int(v) for v in args.active.split(",") if v]
    batch = args.batch
    layer_indices = [int(v) for v in args.layers.split(",")] if args.layers else None

    mesh = open_ornith_mesh()
    report = {
        "batch": batch,
        "context": args.context,
        "steps": args.steps,
        "repeats": args.repeats,
        "layers": layer_indices,
        "loop": "submit_serving_decode(sample_on_device=True) + pipelined read_output_async",
        "token_mode": args.token_mode,
        "rows": [],
    }
    try:
        path = resolve_model_path()
        model = OrnithModel.from_pretrained(
            path,
            mesh_device=mesh,
            hf_config=load_text_config(path),
            layer_indices=layer_indices,
            max_context=args.context,
        )
        report["num_layers"] = len(model.layers)
        blocks_per_user = num_blocks_for_context(args.context, model.page_block_size)
        kv_cache = model.allocate_kv_cache(1 + batch * blocks_per_user)
        gen = OrnithGenerator(
            model,
            max_batch_size=batch,
            cache_context=args.context,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=torch.zeros(batch, blocks_per_user, dtype=torch.int32),
        )
        table = _page_table(batch, blocks_per_user)
        gen.ensure_serving_traces()
        gen.ensure_sampling_trace()
        report["moe_row_mask"] = bool(getattr(model, "decode_row_mask_enabled", False))

        for active in actives:
            if args.token_mode == "shared":
                tokens = torch.full((batch,), 1234, dtype=torch.int64)
            else:
                tokens = torch.arange(1000, 1000 + batch, dtype=torch.int64) * 37 % 100000
            positions = torch.full((batch,), -1, dtype=torch.int64)
            positions[:active] = 64
            gen.stage_serving_decode_inputs(tokens, positions, table, full_refresh=True)
            _time_steps(gen, args.warm_steps)
            runs = [_time_steps(gen, args.steps) for _ in range(args.repeats)]
            row = {
                "active_rows": active,
                "ms_per_token": min(runs),
                "ms_per_token_runs": runs,
                "t/s/u": 1000.0 / min(runs),
            }
            report["rows"].append(row)
            logger.info(f"active {active:2d}/{batch}: {row['ms_per_token']:.3f} ms/token  ({row['t/s/u']:.2f} t/s/u)")
    finally:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        logger.info(f"wrote {args.output}")
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
