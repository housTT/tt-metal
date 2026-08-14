# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer traced decode against ``OptimizedDecoder.NORM_SHARD_CORES``.

Why this exists: the isolated ``NORM`` rows of ``probe_decode_micro.txt`` rank the shard counts one way and
the layer ranks them another, because each sharded norm also pays a conversion whose cost scales with the
shard count. Review round 6 found the shipped choice of 8 defended by a monotonicity claim the artifact
contradicted and by an A/B that varies a different knob, so this measures the layer.

Two things about the protocol, both added by review round 32, both because this file was quoted as evidence
for a claim it did not support:

* **Arms alternate build-by-build** and each arm's first build is discarded. The arm-at-a-time form it used
  before could not separate a per-arm effect from build order.
* **Every arm carries ``pcc_vs_shipped``**, its replayed decode output against the shipped shard count's, on
  the same inputs in the same process. Round 32 read an ~8 µs-a-step *win* off this file for the 16-core arm;
  the arm was faster because it was silently computing the wrong thing (its layer PCC was far below the acceptance bar), which a
  latency-only artifact cannot show. ``_shard_feeds_projection`` now refuses that geometry, and with the
  guard in place the arm is correct and the gap is gone.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_norm_shard_cores.py

Every row is ``NORMCORES cores=<n> run=<i> layer=<idx> (<kind>) decode(traced) iters=<n> wall/iter=<ms> ms
pcc_vs_shipped=<pcc>``.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    num_blocks_for_context,
)

CTX = 8192
PREFILL_LEN = 130
LAYERS = {3: "full_attention", 0: "linear_attention"}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def time_traced_decode(mesh, decoder, page_table, iters):
    """Warmed traced decode, ms per step — the same shape as bench.py's decode arm."""
    import time

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
    # The replayed output, so the caller can check every arm is computing the same thing. Review round 32
    # found a 16-core arm reading ~8 us a step FASTER while its layer PCC was far below the bar: the matmul that consumes
    # the norm's shard takes only the *core count* from the shard spec and lays those cores out inside its own
    # rect, so a 2-D norm shard silently reads the wrong cores - cheaper, and wrong. A latency-only A/B cannot
    # tell that from an optimization, and this one had been quoted as evidence in five places for many rounds.
    replayed = ttnn.to_torch(out).float().flatten()
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return elapsed, replayed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--cores", default="4,8,16,32")
    args = ap.parse_args()
    arms = [int(c) for c in args.cores.split(",")]

    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=90112 * 1024)
    baseline = OptimizedDecoder.NORM_SHARD_CORES
    print(f"# Whole-layer traced decode against NORM_SHARD_CORES. Shipped default: {baseline}.")
    print(
        "# Arms alternate build-by-build and each arm's first build is discarded, the same protocol as "
        "ab_dense_in0_block_w.py."
    )
    print("# One process, one device, real weights.")
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            # Arms alternate build-by-build rather than running arm-at-a-time, and each arm's first build
            # is discarded. Review round 32 found the arm-at-a-time form reporting a real 8 us gap on
            # `full_attention` that five documents had been calling noise for many rounds - and the gap's
            # sign had flipped when round 25 changed the mechanism underneath it, with nobody re-reading the
            # artifact. Alternation is what the stage's later A/Bs use precisely so a per-arm result cannot
            # be an artefact of build order; this one predates them.
            seen = {c: 0 for c in arms}
            reference: dict = {}
            for run in range(1, args.runs + 2):
                for cores in arms:
                    OptimizedDecoder.NORM_SHARD_CORES = cores
                    seen[cores] += 1
                    warmup_arm = seen[cores] == 1
                    decoder = OptimizedDecoder.from_state_dict(
                        sd,
                        hf_config=cfg,
                        layer_idx=layer_idx,
                        mesh_device=mesh,
                        max_context=CTX,
                        policy=POLICIES["optimized"],
                    )
                    blocks = num_blocks_for_context(CTX)
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
                    ms, replayed = time_traced_decode(mesh, decoder, page_table, args.iters)
                    # Every arm is compared against the shipped shard count's own replayed output, on the same
                    # inputs in the same process. An arm that computes something else shows up here even when it
                    # is faster, which is the whole point.
                    if cores == baseline and reference.get(layer_idx) is None:
                        reference[layer_idx] = replayed
                    ref = reference.get(layer_idx)
                    # `n/a`, never 1.0, when there is no reference yet - a default of "perfect agreement" for an
                    # unmeasured comparison is the same shape of defect this artifact exists to prevent.
                    pcc = (
                        f"{float(torch.corrcoef(torch.stack([ref, replayed]))[0, 1]):.6f}"
                        if ref is not None and replayed.numel() == ref.numel()
                        else "n/a"
                    )
                    if not warmup_arm:
                        print(
                            f"NORMCORES cores={cores} run={run} layer={layer_idx} ({kind}) "
                            f"decode(traced) iters={args.iters} wall/iter={ms:.3f} ms pcc_vs_shipped={pcc}",
                            flush=True,
                        )
                    del decoder, page_table
            del sd
    finally:
        OptimizedDecoder.NORM_SHARD_CORES = baseline
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
