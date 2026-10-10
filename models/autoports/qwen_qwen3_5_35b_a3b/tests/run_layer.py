# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only real-checkpoint HF/TT comparison; host conversions are explicit boundaries."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

from ..tt.functional_decoder import FunctionalDecoder
from .hf_reference import decode_reference, prefill_reference
from .layer_weights import load_layer_state_dict, synthetic_layer_state_dict
from .runtime_audit import device_only


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, choices=[0, 3], required=True)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--checkpoint", type=Path, default=Path("../weights/Qwen3.5-35B-A3B"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument("--minimum-cache-pages", type=int, default=32)
    p.add_argument("--cache-guard-pages", type=int, default=3)
    p.add_argument("--sparse-reference", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--weights", choices=["real", "synthetic"], default="real")
    p.add_argument("--expert-precision", choices=["inherited", "bf16"], default="inherited")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.minimum_cache_pages < 0 or args.cache_guard_pages < 0:
        raise ValueError("Cache page counts must be nonnegative")
    if args.profile and (args.length, args.steps, args.batch, args.trace, args.audit) != (4096, 128, 1, True, False):
        raise ValueError("Profile requires length4096 steps128 batch1 --trace and no --audit")
    torch.manual_seed(1234)
    torch.set_num_threads(8)
    if args.weights == "real":
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    else:
        from transformers import Qwen3_5MoeConfig

        cfg = Qwen3_5MoeConfig.from_dict(
            json.loads(Path(__file__).with_name("target_config.json").read_text())
        ).text_config
    cfg._attn_implementation = "eager"
    stats_path = (
        Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / f"layer{args.layer}_weight_stats.json"
    )
    state = (
        load_layer_state_dict(args.checkpoint, args.layer)
        if args.weights == "real"
        else synthetic_layer_state_dict(stats_path)
    )
    prefix = f"model.language_model.layers.{args.layer}."
    local = {k.removeprefix(prefix): v for k, v in state.items()}
    with torch.device("meta"):
        reference = Qwen3_5MoeDecoderLayer(cfg, args.layer)
    reference.load_state_dict(local, assign=True, strict=True)
    reference.eval()
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    if args.length + args.steps > cfg.max_position_embeddings:
        raise ValueError("Requested workload exceeds advertised context")
    x = torch.randn(args.batch, args.length + args.steps, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
    expected_pf, cache, probes = prefill_reference(
        reference, cfg, rope, x[:, : args.length], args.layer, sparse=args.sparse_reference
    )
    expected_dec = decode_reference(reference, rope, cache, x[:, args.length :], args.length)
    print("HF_REFERENCE_READY", flush=True)
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def put(t, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), device=mesh, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    def get(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    def drain_profile():
        # Drain only between complete layer passes, never inside a timed window or capture.
        # MID_RUN_DUMP flushes these explicit reads; it does not schedule reads itself.
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)

    try:
        model = FunctionalDecoder.from_state_dict(
            state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, expert_precision=args.expert_precision
        )
        print("TT_LAYER_READY", flush=True)
        drain_profile()
        from contextlib import nullcontext

        audit = device_only if args.audit else nullcontext
        kw, dkw = {}, {}
        page_size = args.page_size
        if args.layer == 3:
            blocks_per_row = max(args.minimum_cache_pages, (args.length + args.steps + page_size - 1) // page_size)
            blocks = args.batch * blocks_per_row + args.cache_guard_pages
            table = torch.randperm(blocks - args.cache_guard_pages, dtype=torch.int32).reshape(
                args.batch, blocks_per_row
            )
            kc = ttnn.zeros(
                (blocks, cfg.num_key_value_heads, page_size, cfg.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            vc = ttnn.zeros_like(kc)
            model.bind_paged_kv_cache(kc, vc)
            pt = put(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            kw = dict(page_table=pt)
            dkw = dict(
                page_table=pt,
                position_tensor=put(
                    torch.full((args.batch,), args.length, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
        px = put(x[:, : args.length])
        with audit():
            pf_out = model.prefill_forward(px, **kw)
        pf = get(pf_out)
        pf_out.deallocate(True)
        if args.profile:
            from tracy import signpost

            model.reset_request_state(range(args.batch))
            ttnn.synchronize_device(mesh)
            drain_profile()
            signpost("PERF_PREFILL")
            measured = model.prefill_forward(px, **kw)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            drain_profile()
            pf = get(measured)
            measured.deallocate(True)
        if probes is not None:
            pf = pf[:, probes]
        print("PREFILL_PCC", pcc(expected_pf, pf), flush=True)
        dec, deterministic, trace_id = None, None, None
        if args.steps:
            dx = put(x[:, args.length : args.length + 1])
            # Host TTNN inputs are created before capture; copies only refresh stable inputs.
            dx_host = [
                ttnn.from_torch(
                    x[:, args.length + i : args.length + i + 1].contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                for i in range(args.steps)
            ]
            pos_host = [
                ttnn.from_torch(
                    torch.full((args.batch,), args.length + i, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for i in range(args.steps)
            ]
            saved = [(t, ttnn.clone(t)) for row in model._states[: args.batch] for t in row] if args.layer == 0 else []

            def restore():
                for tensor, snapshot in saved:
                    ttnn.copy(snapshot, tensor)

            with audit():
                eager_out = model.decode_forward(dx, **dkw)
            first = get(eager_out)
            eager_out.deallocate(True)
            drain_profile()
            if args.trace:
                from ttnn.tools import trace_allocation_tracker

                restore()
                ttnn.synchronize_device(mesh)
                mesh.set_program_cache_misses_allowed(False)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                with audit():
                    out = model.decode_forward(dx, **dkw)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                drain_profile()
                trace_id = str(tid)
                trace_allocation_tracker.acknowledge_corruptible(out)
                restore()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                first = get(out)
                drain_profile()
                restore()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                deterministic = torch.equal(first, get(out))
                assert deterministic
                drain_profile()
                restore()
                outputs = []
                if args.profile:
                    ttnn.synchronize_device(mesh)
                    drain_profile()
                    signpost("PERF_DECODE")
                for i in range(args.steps):
                    ttnn.copy_host_to_device_tensor(dx_host[i], dx)
                    if args.layer == 3:
                        ttnn.copy_host_to_device_tensor(pos_host[i], dkw["position_tensor"])
                    if args.profile:
                        signpost(f"PERF_DECODE_STEP_{i}")
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    if args.profile:
                        signpost(f"PERF_DECODE_STEP_{i}_END")
                    outputs.append(get(out))
                    drain_profile()
                if args.profile:
                    ttnn.synchronize_device(mesh)
                    signpost("PERF_DECODE_END")
                dec = torch.cat(outputs, dim=1)
                ttnn.release_trace(mesh, tid)
                mesh.set_program_cache_misses_allowed(True)
            else:
                outputs = [first]
                for i in range(1, args.steps):
                    ttnn.copy_host_to_device_tensor(dx_host[i], dx)
                    if args.layer == 3:
                        ttnn.copy_host_to_device_tensor(pos_host[i], dkw["position_tensor"])
                    with audit():
                        y = model.decode_forward(dx, **dkw)
                    outputs.append(get(y))
                    y.deallocate(True)
                dec = torch.cat(outputs, dim=1)
        decode_pcc_by_token = [pcc(expected_dec[:, i : i + 1], dec[:, i : i + 1]) for i in range(args.steps)]
        import hashlib

        source_root = Path(__file__).resolve().parents[1] / "tt"
        result = dict(
            layer_type=cfg.layer_types[args.layer],
            layer_idx=args.layer,
            weights=args.weights,
            implementation_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))
            },
            gdn_recurrence="native_exact_chunk_and_sfpu_fp32_token" if args.layer == 0 else None,
            expert_precision=args.expert_precision,
            profile=args.profile,
            length=args.length,
            batch=args.batch,
            decode_steps=args.steps,
            page_size=args.page_size,
            minimum_cache_pages=args.minimum_cache_pages,
            cache_guard_pages=args.cache_guard_pages,
            prefill_pcc=pcc(expected_pf, pf),
            prefill_pcc_min=min(pcc(expected_pf[r], pf[r]) for r in range(args.batch)),
            decode_pcc=pcc(expected_dec, dec) if args.steps else None,
            prefill_probes=probes,
            runtime_audit=args.audit,
            trace_id=trace_id,
            decode_pcc_min=(
                min(pcc(expected_dec[r, i], dec[r, i]) for r in range(args.batch) for i in range(args.steps))
                if args.steps
                else None
            ),
            decode_pcc_by_token=decode_pcc_by_token,
            traced=args.trace,
            repeated_identical=deterministic,
            threshold=0.995,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["prefill_pcc_min"] >= 0.995
        if args.steps:
            assert result["decode_pcc_min"] >= 0.995
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
