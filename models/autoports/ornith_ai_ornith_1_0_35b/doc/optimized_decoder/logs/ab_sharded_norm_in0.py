# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for carrying the residual norm's width shard into the in-projection.

Until review round 25 this stage rejected the whole sharded-residual family on an assertion about the
op contract: that the DRAM-sharded matmul family "is the only one that consumes a width-sharded `in0`;
`mcast_in0` requires interleaved". That is false, and the way it became false is worth recording,
because no amount of proofreading would have caught it. It was never measured — it was *inferred from
the probe*. Every `mcast1d` row in `probe_dense_matmul.py` handed the op a DRAM-interleaved activation,
so no row in the artifact could contradict the claim, and the claim then closed a family `$optimize`
OPT-003 makes mandatory.

`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` validates a sharded `in0` for
`mcast_in0` explicitly: WIDTH_SHARDED, ROW_MAJOR, `fuse_batch`, `per_core_M == shard_shape[0]/tile_h`,
and `(shard_shape[1]/tile_w) % in0_block_w == 0`. The decode residual norm already produces exactly
that, and `attn_in`/`gdn_in` ship `in0_block_w=8` against a per-core shard of 8 tiles, so the
`sharded_to_interleaved` between them was pure overhead. `_shard_feeds_projection` re-derives each
condition rather than assuming it, so a role that does not qualify keeps the interleaved path.

The MoE norm is *not* in this A/B: its consumer is `shared_in`, whose tuned `in0_block_w` is wider than
the norm's per-core shard, and lowering it to fit is measurably slower in `probe_dense_matmul.txt`.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_sharded_norm_in0.py

Rows are ``SHARDEDNORM arm=<name> run=<i> layer=<idx> (<kind>) decode(traced) wall/iter=<ms> ms``. Arms
alternate build-by-build and each arm's first build is discarded, the same protocol as
`ab_sdpa_decode_grid.py`.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
LAYERS = {3: "full_attention", 0: "linear_attention"}
DECODE_ITERS = 32
PREFILL_LEN = 128


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


def decode_ms(mesh, decoder, page_table):
    """Traced warmed decode, the same protocol `bench.py` uses for the headline number."""
    gen = torch.Generator().manual_seed(43)
    # Prefill first, so decode runs against a populated cache and state exactly as `bench.py` does.
    prefill = (torch.randn(1, PREFILL_LEN, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    ttnn.deallocate(decoder.prefill_forward(dev(mesh, prefill), page_table=page_table))
    x = (torch.randn(1, 1, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    xd = dev(mesh, x)
    step_pos = torch.tensor([PREFILL_LEN], dtype=torch.int32)
    pos = dev(mesh, step_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # uint32, which `ttnn.embedding` requires for the rotary index lookup.
    rot = dev(mesh, step_pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def step():
        return decoder.decode_forward(xd, current_pos=pos, rot_idxs=rot, page_table=page_table)

    ttnn.deallocate(step())
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = step()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    start = time.time()
    for _ in range(DECODE_ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = (time.time() - start) / DECODE_ITERS * 1e3
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="builds per arm; the first is discarded")
    args = ap.parse_args()

    cfg = R.load_text_config()
    print("# Whole-layer A/B: residual norm shard carried into the in-projection vs interleaved between them.")
    print(f"# context={CTX} decode_iters={DECODE_ITERS}; arms alternate build-by-build, first build per arm discarded.")
    print("# `interleaved-between` is the pre-round-25 shipped path; `shard-carried` is what ships now.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    shipped = impl.OptimizedDecoder._shard_feeds_projection
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            arms = {
                "shard-carried": shipped,
                # The pre-round-25 path, reproduced by refusing every shard rather than by editing the
                # layer: the `sharded_to_interleaved` then runs exactly as it used to.
                "interleaved-between": lambda self, role, shape, norm_cfg: False,
            }
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, predicate in arms.items():
                    impl.OptimizedDecoder._shard_feeds_projection = predicate
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    ms = decode_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"SHARDEDNORM arm={arm} run={run} layer={layer_idx} ({kind}) decode(traced) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.OptimizedDecoder._shard_feeds_projection = shipped
    finally:
        impl.OptimizedDecoder._shard_feeds_projection = shipped
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
