# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the `full_attention` output projection's `in0` placement at decode.

`tt-perf-report` raises "If possible place input 0 in L1" on the decode `o_proj` row. The stage took
that advice everywhere else in decode, and README §5.5 said the item was raised nowhere in decode - but
it was still raised here, on every one of that row's launches, and the action prose pointed at a
different tensor. Review round 26 found both.

The cause has the same shape as the routed `in0` item round 14 found: the producing op named no
placement. `_attention_output`'s gated multiply inherited DRAM from the paged flash-decode attention,
whose output has to be in DRAM; `linear_attention`'s identically shaped `gdn_out` already ran from L1
only because *its* producer happens to be an L1 op. So this costs no extra op either way, which is
exactly why it needs a layer measurement rather than an op one.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_attn_out_in0.py

Rows are ``ATTNOUTIN0 arm=<name> run=<i> layer=<idx> (<kind>) decode(traced) wall/iter=<ms> ms``. Arms
alternate build-by-build and each arm's first build is discarded, the same protocol as
`ab_sharded_norm_in0.py`. Only `full_attention` runs this projection.
"""


from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
LAYERS = {3: "full_attention"}
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
    print("# Whole-layer A/B: the decode output projection's `in0` in L1 vs DRAM.")
    print(f"# context={CTX} decode_iters={DECODE_ITERS}; arms alternate build-by-build, first build per arm discarded.")
    print("# `in0-DRAM` is the pre-round-26 shipped path; `shipped-in0-L1` is what ships now.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    shipped = impl.ATTN_OUT_IN0_MEMORY
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            arms = {"shipped-in0-L1": ttnn.L1_MEMORY_CONFIG, "in0-DRAM": ttnn.DRAM_MEMORY_CONFIG}
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, memory_config in arms.items():
                    impl.ATTN_OUT_IN0_MEMORY = memory_config
                    decoder, page_table = build(mesh, cfg, sd, layer_idx)
                    ms = decode_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"ATTNOUTIN0 arm={arm} run={run} layer={layer_idx} ({kind}) decode(traced) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.ATTN_OUT_IN0_MEMORY = shipped
    finally:
        impl.ATTN_OUT_IN0_MEMORY = shipped
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
