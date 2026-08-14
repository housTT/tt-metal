# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed-prefill / traced-decode benchmark for the multichip decoder and its single-chip baseline.

One script for every latency number in ``doc/multichip_decoder/``, so a before/after pair is always
the same harness, the same real checkpoint weights, the same inputs and the same warmup/iteration
counts. ``--impl optimized --mesh 1x1`` is the single-chip baseline arm (the previous stage's
decoder, unmodified, on a 1x1 mesh); ``--impl multichip --mesh 1x4`` is this stage's.

    python .../doc/multichip_decoder/logs/bench.py --impl optimized  --mesh 1x1 --layers 0,3
    python .../doc/multichip_decoder/logs/bench.py --impl multichip --mesh 1x4 --layers 0,3

``--impl optimized --mesh 1x4`` is the *replication control*: the single-chip decoder run on the
4-chip mesh with every weight replicated, i.e. four copies of the same single-chip work. It
separates "the mesh made dispatch slower" from "the parallelisation helped", which a 1x1-vs-1x4
comparison alone cannot.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

CTX = 8192

MESHES = {"1x1": (1, 1), "1x4": (1, 4), "2x2": (2, 2)}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def first_shard(mesh, tensor):
    """Device 0's copy of a mesh tensor, as a host tensor."""
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0))
    return whole[: int(tensor.shape[0])]


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.double().flatten()
    b = actual.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def build(impl, mesh, cfg, sd, layer_idx, batch=1, max_context=CTX, **extra):
    if impl == "multichip":
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import MultichipDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import num_blocks_for_context
    else:
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import OptimizedDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

    decoder = Cls.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=max_context, **extra
    )
    blocks = num_blocks_for_context(max_context)
    decoder.allocate_kv_cache(blocks * batch)
    decoder.allocate_state(batch)
    page_table = None
    if decoder.is_full_attention:
        table = torch.arange(blocks * batch, dtype=torch.int32).reshape(batch, blocks)
        page_table = dev(mesh, table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return decoder, page_table


def bench_prefill(mesh, decoder, page_table, cfg, seq_len, warmups=2, want_out=False):
    from tracy import signpost

    x_t = (torch.randn(1, seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(71)) * 0.5).to(
        torch.bfloat16
    )
    x = dev(mesh, x_t)
    for _ in range(warmups):
        ttnn.deallocate(decoder.prefill_forward(x, page_table=page_table))
    ttnn.synchronize_device(mesh)
    signpost("PERF_PREFILL")
    start = time.time()
    out = decoder.prefill_forward(x, page_table=page_table)
    ttnn.synchronize_device(mesh)
    elapsed = time.time() - start
    signpost("PERF_PREFILL_END")
    host = first_shard(mesh, out).float() if want_out else None
    ttnn.deallocate(out)
    ttnn.deallocate(x)
    return elapsed, x_t, host


def bench_decode(mesh, decoder, page_table, cfg, iters, prefill_len=128, warmups=4, want_out=False):
    from tracy import signpost

    prefill_t = (torch.randn(1, prefill_len, cfg.hidden_size, generator=torch.Generator().manual_seed(81)) * 0.5).to(
        torch.bfloat16
    )
    ttnn.deallocate(decoder.prefill_forward(dev(mesh, prefill_t), page_table=page_table))
    decode_t = (torch.randn(1, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(82)) * 0.5).to(
        torch.bfloat16
    )
    x_buf = dev(mesh, decode_t)
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
    for _ in range(warmups):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    signpost("PERF_DECODE")
    start = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    elapsed = time.time() - start
    signpost("PERF_DECODE_END")
    host = first_shard(mesh, trace_out).float()
    finite = bool(torch.isfinite(host).all())
    ttnn.release_trace(mesh, trace_id)
    return elapsed / iters, finite, prefill_t, decode_t, (host if want_out else None)


def open_mesh(name, fabric: bool):
    shape = MESHES[name]
    if fabric and shape != (1, 1):
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import DEFAULT_FABRIC_CONFIG

        ttnn.set_fabric_config(DEFAULT_FABRIC_CONFIG)
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=24576, trace_region_size=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="multichip", choices=["optimized", "multichip"])
    ap.add_argument("--mesh", default="1x4", choices=sorted(MESHES))
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--prefill-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--weights", default="real", choices=["real", "synthetic"])
    ap.add_argument("--pcc", action="store_true", help="also run the HF float32 golden and print prefill PCC")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")]
    cfg = R.load_text_config()
    mesh = open_mesh(args.mesh, fabric=args.impl == "multichip" or args.mesh != "1x1")
    rows = []
    try:
        for layer_idx in layers:
            sd = _state_dict(layer_idx, args.weights)
            decoder, page_table = build(args.impl, mesh, cfg, sd, layer_idx)
            kind = "full_attention" if decoder.is_full_attention else "linear_attention"
            label = (
                f"impl={args.impl} mesh={args.mesh} weights={args.weights} "
                f"tag={args.tag or '-'} layer={layer_idx} ({kind})"
            )
            row = {
                "impl": args.impl,
                "mesh": args.mesh,
                "weights": args.weights,
                "tag": args.tag,
                "layer": layer_idx,
                "kind": kind,
                "devices": mesh.get_num_devices(),
            }
            if args.phase in ("prefill", "both"):
                elapsed, x_t, host = bench_prefill(mesh, decoder, page_table, cfg, args.prefill_len, want_out=args.pcc)
                row["prefill_ms"] = elapsed * 1e3
                row["prefill_tok_s"] = args.prefill_len / elapsed
                msg = (
                    f"BENCH {label} prefill seq_len={args.prefill_len} "
                    f"wall={elapsed * 1e3:.2f} ms tok/s={args.prefill_len / elapsed:.1f}"
                )
                if args.pcc:
                    ref = R.build_reference_layer(cfg, layer_idx, sd)
                    golden, _ = R.reference_prefill(ref, cfg, x_t.float())
                    value = pcc(golden.squeeze(0), host.reshape(golden.squeeze(0).shape))
                    row["prefill_pcc"] = value
                    msg += f" pcc={value:.6f}"
                    del ref, golden
                print(msg, flush=True)
            if args.phase in ("decode", "both"):
                per_iter, finite, _, _, _ = bench_decode(mesh, decoder, page_table, cfg, args.iters)
                row["decode_ms"] = per_iter * 1e3
                row["decode_steps_s"] = 1 / per_iter
                row["decode_finite"] = finite
                print(
                    f"BENCH {label} decode(traced) iters={args.iters} "
                    f"wall/iter={per_iter * 1e3:.3f} ms steps/s={1 / per_iter:.1f} finite={finite}",
                    flush=True,
                )
            rows.append(row)
            del decoder, page_table, sd
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


def _state_dict(layer_idx: int, source: str):
    if source == "real":
        return R.load_layer_state_dict(layer_idx)
    import json as _json
    from pathlib import Path

    stats = Path(__file__).resolve().parents[2] / "functional_decoder" / f"weight_stats_layer{layer_idx}.json"
    with open(stats) as handle:
        return R.synthetic_state_dict(_json.load(handle)["tensors"], seed=1234 + layer_idx)


if __name__ == "__main__":
    main()
