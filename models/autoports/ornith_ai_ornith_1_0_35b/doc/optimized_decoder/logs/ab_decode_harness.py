# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two things review round 8 asked for that no artifact measured: the layer-level run-to-run spread, and
what a reserved trace region does to the traced-decode number.

**Why the spread matters.** Several decisions in this stage turn on differences of one or two microseconds
at the layer — most sharply the SDPA `q_chunk=0, k_chunk=0` candidate, whose rejection once rested on the two
arms being "inside the spread". Every op-level probe reports a measured ``spread=``; the layer-level harness
(``bench.py``) takes a single mean of 32 replays and reports none, so that argument rested on an asserted
spread — and once measured, it turned out not to hold. This measures it: the same build, timed repeatedly in one process.

**Why the trace region matters.** The stage's headline traced-decode figure comes from harnesses that open
the device with ``trace_region_size=0`` (``bench.py``, the suite), while two of its A/B harnesses reserve
88 MB and consistently report ~2 % slower for the identical configuration. A serving path must reserve a
trace region, so which number is representative is a real question rather than a curiosity.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_decode_harness.py

Every row is ``HARNESS arm=<name> trace_region=<bytes> run=<i> layer=<idx> (<kind>) decode(traced) …``.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
PREFILL_LEN = 130
LAYERS = {3: "full_attention", 0: "linear_attention"}
#: What `bench.py` and the suite use, and what two A/B harnesses use instead.
TRACE_REGIONS = {"none": 0, "reserved": 90112 * 1024}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build(mesh, cfg, sd, layer_idx):
    decoder = impl.OptimizedDecoder.from_state_dict(
        sd,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_context=CTX,
        policy=impl.POLICIES["optimized"],
    )
    blocks = impl.num_blocks_for_context(CTX)
    decoder.allocate_kv_cache(blocks)
    decoder.allocate_state(1)
    page_table = None
    if decoder.is_full_attention:
        page_table = dev(
            mesh,
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    gen = torch.Generator().manual_seed(11)
    x = (torch.randn(1, PREFILL_LEN, cfg.hidden_size, generator=gen) * 0.5).to(torch.bfloat16)
    ttnn.deallocate(decoder.prefill_forward(dev(mesh, x), page_table=page_table))
    return decoder, page_table


def time_traced_decode(mesh, decoder, page_table, iters):
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(1, 1, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([PREFILL_LEN], dtype=torch.int32)
    args = dict(
        current_pos=dev(mesh, pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        rot_idxs=dev(mesh, pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        page_table=page_table,
    )
    xd = dev(mesh, x)
    for _ in range(2):
        ttnn.deallocate(decoder.decode_forward(xd, **args))
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = decoder.decode_forward(xd, **args)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / iters * 1e3
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return elapsed


def sdpa_arms(decoder):
    """``{name: program_config}`` — the shipped config and the round-7/8 candidate."""
    grid = ttnn.CoreCoord(8, 8)
    return {
        "shipped-q32-kpage": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=32,
            k_chunk_size=decoder.page_block_size,
            exp_approx_mode=False,
        ),
        "candidate-q0-k0": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    cfg = R.load_text_config()
    print("# Layer-level run-to-run spread, and the effect of reserving a trace region.")
    print(f"# {args.runs} timed repeats per arm per layer kind, each a fresh build in the same process.")
    for region_name, region in TRACE_REGIONS.items():
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=region)
        try:
            for layer_idx, kind in LAYERS.items():
                sd = R.load_layer_state_dict(layer_idx)
                # `candidate-q0-k0` only differs on full_attention: the linear kind has no SDPA at all.
                arms = ("shipped-q32-kpage", "candidate-q0-k0") if kind == "full_attention" else ("shipped-q32-kpage",)
                #: Discards are counted **per arm**, and the arms alternate build-by-build. Round 9 found this
                #: loop discarding per *layer*, so the shipped arm reported one repeat fewer than the candidate
                #: while the header claimed an equal count, and the arms ran in blocks — which is the same
                #: multi-build-in-one-process effect this file exists to characterise, so a drift across the
                #: process could have appeared as an arm difference.
                seen = {arm: 0 for arm in arms}
                for run in range(1, args.runs + 2):
                    for arm in arms:
                        decoder, page_table = build(mesh, cfg, sd, layer_idx)
                        decoder.decode_sdpa_config = sdpa_arms(decoder)[arm]
                        ms = time_traced_decode(mesh, decoder, page_table, args.iters)
                        del decoder, page_table
                        seen[arm] += 1
                        if seen[arm] == 1:
                            continue  # discard: this arm's first build pays weight upload and compile
                        print(
                            f"HARNESS arm={arm} trace_region={region} run={run} layer={layer_idx} ({kind}) "
                            f"decode(traced) iters={args.iters} wall/iter={ms:.3f} ms",
                            flush=True,
                        )
                del sd
        finally:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
