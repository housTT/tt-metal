# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B for the two geometry choices review round 9 found on the wrong side of their own sweep.

Both are pure geometry — a grid rectangle — so neither changes numerics beyond tile-order accumulation,
and both were measured only at the op. Round 9's finding was not that either is large; it was that the
shipped value was the *swept loser* with no document recording the gap. An op-level microsecond does not
have to survive at the layer, so this measures the layer:

**SDPA decode grid.** `probe_decode_micro.txt` puts `8x4` about a microsecond ahead of the shipped `8x8` in both
sections at the shipped chunk pair and compute-kernel contract, at identical PCC. Only `full_attention` has an
SDPA at all. This arm measures a dead heat, and the candidate is rejected for a reason no timing harness could
have found: flash-decode assigns one core per batch row (`TT_FATAL(num_cores_available >= B)`), so 32 cores cap
decode at batch 32 and the supported batch-40/56 cases die inside the op. The arm is kept because "the swept
winner buys nothing at the layer even before it breaks a capability" is the complete answer, and because a
future reader will find the same microsecond in the probe.

**Routed `down` grid orientation.** `probe_sparse_matmul.txt` puts the row rectangle `8x4` ahead of the
shipped column `4x8` by 1.6 us at the ~162-active prefill group, the only shipped geometry where `down`
reaches 32 cores. The routed `sparse_matmul` is >80 % of the prefill window, so this is a prefill arm;
decode never builds a 32-core `down` grid and is measured here to confirm that it is unaffected. This is
the arm that shows why an op-level row cannot be trusted to a shipped decision: it reverses at the layer.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_sdpa_decode_grid.py

Rows are ``ABGRID knob=<knob> arm=<name> run=<i> layer=<idx> (<kind>) <phase> …``. Arms alternate
build-by-build within a layer so a monotonic drift across the process cannot masquerade as an arm
difference, and each arm's first build is discarded (it pays weight upload and kernel compilation).
"""

from __future__ import annotations

import argparse
import math
import time
import types

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
#: Long enough that the prefill runs many 32-token expert groups, which is where the orientation lives.
PREFILL_LEN = 2048
DECODE_POS = PREFILL_LEN


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
    return decoder, page_table


#: The shipped `_sparse_matmul_config`, kept so the row arm can wrap rather than reimplement it.
_SHIPPED_SPARSE_CONFIG = impl._sparse_matmul_config


def _row_form_sparse_config(m, n, k, *, cores=impl.SPARSE_MIN_CORES, in0_block_w=None, grid=None):
    """`_sparse_matmul_config`, but the routed `down` shape at a 32-core target gets the row rectangle.

    The candidate is expressed as a wrapper rather than as a flag in the layer because the layer does not
    ship one: `grid.y` is the sole input to the orientation, so substituting a stand-in grid of height 4
    turns the 32-core column (4x8) into the row (8x4) and leaves every other geometry untouched. `down` is
    identified by its shape — it is the only routed role whose K is narrower than its N.
    """
    n_tiles = max(1, int(math.ceil(n / impl.TILE)))
    realised = impl._largest_divisor_at_most(n_tiles, max(1, cores))
    if k < n and realised == 32 and grid is not None:
        grid = types.SimpleNamespace(x=int(grid.x), y=4)
    return _SHIPPED_SPARSE_CONFIG(m, n, k, cores=cores, in0_block_w=in0_block_w, grid=grid)


def select_arm(knob: str, arm: str) -> None:
    """Select the arm *before* the build: the sparse configs are built lazily inside the forward and cached,
    so the only single point of control is the config function itself."""
    if knob == "down_orientation":
        impl._sparse_matmul_config = _SHIPPED_SPARSE_CONFIG if arm == "shipped-column-4x8" else _row_form_sparse_config


def apply_arm(decoder, knob: str, arm: str) -> None:
    """Mutate the built decoder to the requested arm. The shipped arm leaves it exactly as constructed."""
    if knob == "sdpa_grid" and arm == "candidate-8x4":
        decoder.decode_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32,
            k_chunk_size=decoder.page_block_size,
            exp_approx_mode=False,
        )


def prefill_ms(mesh, decoder, page_table, iters):
    gen = torch.Generator().manual_seed(11)
    x = (torch.randn(1, PREFILL_LEN, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    xd = dev(mesh, x)
    ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    return (time.time() - start) / iters * 1e3


def decode_ms(mesh, decoder, page_table, iters):
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(1, 1, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    pos = torch.tensor([DECODE_POS], dtype=torch.int32)
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


#: ``knob -> (arms, layers, phases)``. The shipped arm is first in every pair for readability, but the
#: run order alternates.
PLAN = {
    "sdpa_grid": (("shipped-8x8", "candidate-8x4"), {3: "full_attention"}, ("decode",)),
    "down_orientation": (
        ("shipped-column-4x8", "row-8x4"),
        {3: "full_attention", 0: "linear_attention"},
        ("prefill", "decode"),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4, help="builds per arm; the first is discarded")
    ap.add_argument("--iters", type=int, default=32)
    args = ap.parse_args()

    cfg = R.load_text_config()
    print("# End-to-end A/B for the two round-9 geometry candidates: SDPA decode grid, routed `down` grid")
    print("# orientation. Arms alternate build-by-build; each arm's first build is discarded.")
    print(f"# prefill_len={PREFILL_LEN} decode_pos={DECODE_POS} context={CTX} runs={args.runs} iters={args.iters}")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        for knob, (arms, layers, phases) in PLAN.items():
            for layer_idx, kind in layers.items():
                sd = R.load_layer_state_dict(layer_idx)
                seen = {arm: 0 for arm in arms}
                for run in range(1, args.runs + 1):
                    for arm in arms:
                        select_arm(knob, arm)
                        decoder, page_table = build(mesh, cfg, sd, layer_idx)
                        apply_arm(decoder, knob, arm)
                        rows = {}
                        if "prefill" in phases:
                            rows["prefill(warmed)"] = prefill_ms(mesh, decoder, page_table, 3)
                        if "decode" in phases:
                            rows["decode(traced)"] = decode_ms(mesh, decoder, page_table, args.iters)
                        grids = sorted(
                            str(c.compute_with_storage_grid_size) for c in decoder.moe._sparse_cfg_cache.values()
                        )
                        del decoder, page_table
                        seen[arm] += 1
                        if seen[arm] == 1:
                            continue  # discard: first build of this arm pays upload + compile
                        for phase, ms in rows.items():
                            print(
                                f"ABGRID knob={knob} arm={arm} run={run} layer={layer_idx} ({kind}) "
                                f"{phase} wall/iter={ms:.3f} ms sparse_grids={','.join(grids)}",
                                flush=True,
                            )
                del sd
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
