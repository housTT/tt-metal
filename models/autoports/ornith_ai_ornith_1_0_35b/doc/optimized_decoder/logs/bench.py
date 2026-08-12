# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed-prefill / traced-decode benchmark and real-weight PCC probe for the optimized decoder.

Every candidate in ``doc/optimized_decoder/README.md`` is measured with this one script, so a
before/after pair is always same-process, same-device, same real checkpoint weights, same inputs
and same warmup/iteration counts. ``--impl fused`` builds the previous stage's decoder from
``tt/fused_decoder.py`` unchanged, which is what the "before" column of every table is.

    python .../doc/optimized_decoder/logs/bench.py \
        --impl optimized --layers 0,3 --phase decode --policy optimized \
        --set expert_gate_up_dtype=bfloat4_b,expert_fidelity=LoFi

``--set`` overrides individual :class:`PrecisionPolicy` fields on top of ``--policy``, so a
one-tensor-group sweep does not need a code edit. ``--pcc`` additionally runs the HF float32 golden for the same prefill input and prints layer-output
PCC, which makes a precision candidate's screening decision real-weight evidence rather than a
latency number. It is a *screen*: the accept/reject bar is the full delivered suite
(``tests/test_optimized_decoder.py``), which covers decode, paged-cache, traced-replay and
non-aligned PCC as well.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

CTX = 8192

DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def parse_overrides(text: str) -> dict:
    """``"expert_fidelity=LoFi,proj_dtype=bfloat8_b,expert_fp32_acc=true"`` -> kwargs."""
    out: dict = {}
    for item in filter(None, (piece.strip() for piece in text.split(","))):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if value in DTYPES:
            out[key] = DTYPES[value]
        elif value in FIDELITIES:
            out[key] = FIDELITIES[value]
        elif value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        elif value.lstrip("-").isdigit():
            out[key] = int(value)
        else:
            out[key] = value
    return out


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.double().flatten()
    b = actual.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def build(impl, mesh, cfg, sd, layer_idx, batch=1, policy=None, overrides=None, moe_group_tokens=None, **extra):
    kwargs = dict(extra)
    if impl == "functional":
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import FunctionalDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder import num_blocks_for_context
    elif impl == "fused":
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import FusedDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.fused_decoder import num_blocks_for_context
    else:
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import POLICIES
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import OptimizedDecoder as Cls
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

        pol = POLICIES[policy or "optimized"]
        if overrides:
            pol = pol.replace(**overrides)
        kwargs["policy"] = pol
    if moe_group_tokens and impl != "functional":
        kwargs["moe_group_tokens"] = moe_group_tokens
    decoder = Cls.from_state_dict(sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX, **kwargs)
    blocks = num_blocks_for_context(CTX)
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
    host = ttnn.to_torch(out).float() if want_out else None
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
    host = ttnn.to_torch(trace_out).float()
    finite = bool(torch.isfinite(host).all())
    ttnn.release_trace(mesh, trace_id)
    return elapsed / iters, finite, prefill_t, decode_t, (host if want_out else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="optimized", choices=["functional", "fused", "optimized"])
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--prefill-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--policy", default="optimized")
    ap.add_argument("--set", dest="overrides", default="")
    ap.add_argument("--moe-group-tokens", type=int, default=0)
    ap.add_argument("--pcc", action="store_true", help="also run the HF float32 golden and print prefill PCC")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    overrides = parse_overrides(args.overrides)
    layers = [int(v) for v in args.layers.split(",")]
    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    rows = []
    try:
        for layer_idx in layers:
            sd = R.load_layer_state_dict(layer_idx)
            decoder, page_table = build(
                args.impl,
                mesh,
                cfg,
                sd,
                layer_idx,
                policy=args.policy,
                overrides=overrides,
                moe_group_tokens=args.moe_group_tokens or None,
            )
            kind = "full_attention" if decoder.is_full_attention else "linear_attention"
            label = (
                f"impl={args.impl} policy={args.policy} set={args.overrides or '-'} "
                f"tag={args.tag or '-'} layer={layer_idx} ({kind})"
            )
            row = {
                "impl": args.impl,
                "policy": args.policy,
                "set": args.overrides,
                "tag": args.tag,
                "layer": layer_idx,
                "kind": kind,
            }
            if args.phase in ("prefill", "both"):
                elapsed, x_t, host = bench_prefill(mesh, decoder, page_table, cfg, args.prefill_len, want_out=args.pcc)
                row["prefill_ms"] = elapsed * 1e3
                row["prefill_tok_s"] = args.prefill_len / elapsed
                msg = f"BENCH {label} prefill seq_len={args.prefill_len} wall={elapsed * 1e3:.2f} ms tok/s={args.prefill_len / elapsed:.1f}"
                if args.pcc:
                    ref = R.build_reference_layer(cfg, layer_idx, sd)
                    golden, _ = R.reference_prefill(ref, cfg, x_t.float())
                    value = pcc(golden.squeeze(0), host.reshape(golden.squeeze(0).shape))
                    row["prefill_pcc"] = value
                    msg += f" pcc={value:.6f}"
                    del ref, golden
                print(msg, flush=True)
            if args.phase in ("decode", "both"):
                per_iter, finite, prefill_t, decode_t, host = bench_decode(
                    mesh, decoder, page_table, cfg, args.iters, want_out=args.pcc
                )
                row["decode_ms"] = per_iter * 1e3
                row["decode_steps_s"] = 1 / per_iter
                row["decode_finite"] = finite
                msg = (
                    f"BENCH {label} decode(traced) iters={args.iters} "
                    f"wall/iter={per_iter * 1e3:.3f} ms steps/s={1 / per_iter:.1f} finite={finite}"
                )
                print(msg, flush=True)
            rows.append(row)
            del decoder, page_table, sd
    finally:
        ttnn.close_mesh_device(mesh)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
