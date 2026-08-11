# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced-decode A/B between the functional and the fused decoder.

Both implementations are timed in the **same process, on the same device, with the same real
checkpoint weights and the same inputs**, so the before/after numbers are like-for-like.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/bench_ab.py \
        [--impl functional|fused|both] [--layers 0,3] [--prefill-len 2048] [--iters 32]
"""

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

CTX = 8192


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build(impl, mesh, cfg, sd, layer_idx, batch=1, moe_group_tokens=None, rope_mode=None):
    extra = {}
    if impl == "functional":
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import FunctionalDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import num_blocks_for_context
    else:
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import FusedDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import num_blocks_for_context

        if moe_group_tokens:
            extra["moe_group_tokens"] = moe_group_tokens
        if rope_mode:
            extra["rope_mode"] = rope_mode
    decoder = Cls.from_state_dict(sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX, **extra)
    blocks = num_blocks_for_context(CTX)
    decoder.allocate_kv_cache(blocks * batch)
    decoder.allocate_state(batch)
    page_table = None
    if decoder.is_full_attention:
        table = torch.arange(blocks * batch, dtype=torch.int32).reshape(batch, blocks)
        page_table = dev(mesh, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return decoder, page_table


def bench_prefill(mesh, decoder, page_table, cfg, seq_len, warmups=2):
    from tracy import signpost

    x = dev(
        mesh,
        (torch.randn(1, seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(71)) * 0.5).to(
            torch.bfloat16
        ),
    )
    for _ in range(warmups):
        ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
    ttnn.synchronize_device(mesh)
    signpost("PERF_PREFILL")
    start = time.time()
    out = decoder.prefill_forward(x, page_table=page_table)
    ttnn.synchronize_device(mesh)
    elapsed = time.time() - start
    signpost("PERF_PREFILL_END")
    ttnn.deallocate(out)
    ttnn.deallocate(x)
    return elapsed


def bench_decode(mesh, decoder, page_table, cfg, iters, prefill_len=128):
    from tracy import signpost

    ttnn.deallocate(
        decoder.prefill_forward(
            dev(
                mesh,
                (torch.randn(1, prefill_len, cfg.hidden_size, generator=torch.Generator().manual_seed(81)) * 0.5).to(
                    torch.bfloat16
                ),
            ),
            page_table=page_table,
        )
    )
    x_buf = dev(
        mesh, (torch.randn(1, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(82)) * 0.5).to(torch.bfloat16)
    )
    pos = torch.tensor([prefill_len], dtype=torch.int32)
    pos_buf = dev(mesh, pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    rot_buf = dev(mesh, pos.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=page_table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    trace_out = forward()
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(4):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    signpost("PERF_DECODE")
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = time.time() - start
    signpost("PERF_DECODE_END")
    finite = bool(torch.isfinite(ttnn.to_torch(trace_out).float()).all())
    ttnn.release_trace(mesh, trace_id)
    return elapsed / iters, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="both")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--prefill-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--moe-group-tokens", type=int, default=0)
    ap.add_argument("--rope-mode", default="")
    args = ap.parse_args()

    impls = ["functional", "fused"] if args.impl == "both" else [args.impl]
    layers = [int(v) for v in args.layers.split(",")]
    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        for layer_idx in layers:
            sd = R.load_layer_state_dict(layer_idx)
            for impl in impls:
                decoder, page_table = build(
                    impl, mesh, cfg, sd, layer_idx, moe_group_tokens=args.moe_group_tokens, rope_mode=args.rope_mode
                )
                kind = "full_attention" if decoder.is_full_attention else "linear_attention"
                if args.phase in ("prefill", "both"):
                    elapsed = bench_prefill(mesh, decoder, page_table, cfg, args.prefill_len)
                    print(
                        f"BENCH impl={impl} moe_group_tokens={args.moe_group_tokens or 'default'} "
                        f"rope_mode={args.rope_mode or 'default'} layer={layer_idx} "
                        f"({kind}) prefill seq_len={args.prefill_len} "
                        f"wall={elapsed * 1e3:.2f} ms tok/s={args.prefill_len / elapsed:.1f}",
                        flush=True,
                    )
                if args.phase in ("decode", "both"):
                    per_iter, finite = bench_decode(mesh, decoder, page_table, cfg, args.iters)
                    print(
                        f"BENCH impl={impl} moe_group_tokens={args.moe_group_tokens or 'default'} "
                        f"rope_mode={args.rope_mode or 'default'} layer={layer_idx} "
                        f"({kind}) decode(traced) iters={args.iters} "
                        f"wall/iter={per_iter * 1e3:.3f} ms steps/s={1 / per_iter:.1f} finite={finite}",
                        flush=True,
                    )
                del decoder, page_table
            del sd
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
