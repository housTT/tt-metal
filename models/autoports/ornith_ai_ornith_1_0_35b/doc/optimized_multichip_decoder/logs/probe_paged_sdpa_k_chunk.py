# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B the paged-SDPA K tile at a real long-context serving offset.

This is deliberately a full, real-weight layer probe instead of a contiguous
short-context SDPA microbenchmark.  Ornith serving uses the flexible
``chunk_start_idx_tensor`` overload, a full-width page table, BF8 paged K/V, and
the selected C25 precision policy.  Those details determine the long-prefix
read geometry that this experiment is meant to isolate.

The cache prefix does not need meaningful values for a latency experiment: the
current chunk is filled normally and the earlier allocated pages are read just
as they are in serving.  The probe still checks that every measured output is
finite.  Run a control-candidate-control sequence to expose thermal drift::

    python .../probe_paged_sdpa_k_chunk.py \
        --batches 1,4 --k-chunks 128,256,128 \
        --start-pos 129024 --max-context 131072 --warmups 2 --iters 3 \
        --json /tmp/ornith-paged-sdpa-k.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.doc.optimized_multichip_decoder.logs.bench import (
    _state_dict,
    build,
    dev,
    first_shard,
    open_mesh,
)
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import PrefillChunkInputs
from models.autoports.ornith_ai_ornith_1_0_35b.tt.precision_config import load_selected_policy

FULL_ATTENTION_LAYER = 3
PREFILL_CHUNK = 2048
PAGE_BLOCK_SIZE = 64


def _csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _staged_inputs(mesh, full_page_table, *, batch: int, blocks_per_user: int, start_pos: int):
    block0 = start_pos // PAGE_BLOCK_SIZE
    blocks_per_chunk = PREFILL_CHUNK // PAGE_BLOCK_SIZE
    fill = torch.empty(batch, blocks_per_chunk, dtype=torch.int32)
    for user in range(batch):
        first_physical = user * blocks_per_user + block0
        fill[user] = torch.arange(first_physical, first_physical + blocks_per_chunk, dtype=torch.int32)
    positions = torch.arange(start_pos, start_pos + PREFILL_CHUNK, dtype=torch.int32).reshape(1, -1)
    chunk_start = torch.tensor([start_pos], dtype=torch.int32)
    return PrefillChunkInputs(
        full_page_table=full_page_table,
        fill_page_table=dev(mesh, fill, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        position_idxs=dev(mesh, positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        chunk_start_idx_tensor=dev(mesh, chunk_start, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        start_pos=start_pos,
        physical_len=PREFILL_CHUNK,
    )


def _measure(mesh, decoder, x, staged, *, start_pos: int, warmups: int, iters: int):
    # First execution pays any program compilation.  Warmups and samples are
    # synchronized independently so the reported wall time cannot hide queued work.
    out = decoder.prefill_forward(x, start_pos=start_pos, page_table=staged)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    for _ in range(warmups):
        out = decoder.prefill_forward(x, start_pos=start_pos, page_table=staged)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(out)

    samples_ms = []
    final_out = None
    for _ in range(iters):
        before = time.perf_counter()
        out = decoder.prefill_forward(x, start_pos=start_pos, page_table=staged)
        ttnn.synchronize_device(mesh)
        samples_ms.append((time.perf_counter() - before) * 1e3)
        if final_out is not None:
            ttnn.deallocate(final_out)
        final_out = out
    host = first_shard(mesh, final_out)
    finite = bool(torch.isfinite(host).all())
    ttnn.deallocate(final_out)
    return samples_ms, finite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=_csv_ints, default=[1, 4])
    parser.add_argument("--k-chunks", type=_csv_ints, default=[128, 256, 128])
    parser.add_argument("--start-pos", type=int, default=129024)
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.start_pos % PREFILL_CHUNK:
        parser.error(f"--start-pos must be a multiple of {PREFILL_CHUNK}")
    if args.start_pos + PREFILL_CHUNK > args.max_context:
        parser.error("the measured chunk extends beyond --max-context")
    if args.warmups < 0 or args.iters < 1:
        parser.error("--warmups must be >= 0 and --iters must be >= 1")
    for k_chunk in args.k_chunks:
        OD._parse_prefill_sdpa_k_chunk(str(k_chunk))

    cfg = R.load_text_config()
    state_dict = _state_dict(FULL_ATTENTION_LAYER, "real")
    policy = load_selected_policy().for_layer(FULL_ATTENTION_LAYER)
    mesh = open_mesh("1x4", fabric=True)
    rows = []
    try:
        for batch in args.batches:
            decoder, full_page_table = build(
                "multichip",
                mesh,
                cfg,
                state_dict,
                FULL_ATTENTION_LAYER,
                batch=batch,
                max_context=args.max_context,
                policy=policy,
            )
            blocks_per_user = int(full_page_table.shape[1])
            staged = _staged_inputs(
                mesh,
                full_page_table,
                batch=batch,
                blocks_per_user=blocks_per_user,
                start_pos=args.start_pos,
            )
            activations = (
                torch.randn(
                    batch,
                    PREFILL_CHUNK,
                    cfg.hidden_size,
                    generator=torch.Generator().manual_seed(4100 + batch),
                )
                * 0.5
            ).to(torch.bfloat16)
            x = dev(mesh, activations)
            try:
                for arm, k_chunk in enumerate(args.k_chunks):
                    OD.PREFILL_SDPA_K_CHUNK_OVERRIDE = k_chunk
                    resolved = decoder._prefill_sdpa_config(args.start_pos, PREFILL_CHUNK)
                    samples_ms, finite = _measure(
                        mesh,
                        decoder,
                        x,
                        staged,
                        start_pos=args.start_pos,
                        warmups=args.warmups,
                        iters=args.iters,
                    )
                    row = {
                        "arm": arm,
                        "batch": batch,
                        "start_pos": args.start_pos,
                        "max_context": args.max_context,
                        "q_chunk": resolved.q_chunk_size,
                        "k_chunk": resolved.k_chunk_size,
                        "samples_ms": samples_ms,
                        "median_ms": statistics.median(samples_ms),
                        "min_ms": min(samples_ms),
                        "max_ms": max(samples_ms),
                        "finite": finite,
                    }
                    rows.append(row)
                    print("PAGED_SDPA_K " + json.dumps(row, sort_keys=True), flush=True)
            finally:
                ttnn.deallocate(x)
                del decoder, full_page_table, staged
    finally:
        OD.PREFILL_SDPA_K_CHUNK_OVERRIDE = None
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
