# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only real-checkpoint HF/TT comparison; host conversions are explicit boundaries."""

import argparse
import json
import time
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
    p.add_argument("--warm-prefill", action="store_true")
    p.add_argument("--decoder", choices=["functional", "fused", "optimized", "multichip"], default="functional")
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--optimization-json", help="Explicit experimental policy; omitted uses delivered defaults")
    p.add_argument("--input-activations", type=Path)
    p.add_argument("--fusions", help="Comma-separated candidates, or empty for the construction control")
    p.add_argument("--save-tensors", type=Path)
    p.add_argument("--compare-tensors", type=Path)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--no-kv-scratch", action="store_true")
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
    input_metadata = {"kind": "random_activation", "seed": 1234}
    if args.input_activations:
        recorded = torch.load(args.input_activations, weights_only=True)
        assert recorded["metadata"]["layer"] == args.layer
        x = recorded["activations"][:, : args.length + args.steps].contiguous()
        assert x.shape == (args.batch, args.length + args.steps, cfg.hidden_size)
        input_metadata = dict(recorded["metadata"], path=str(args.input_activations), kind="recorded_hf_activation")
    expected_pf, cache, probes = prefill_reference(
        reference, cfg, rope, x[:, : args.length], args.layer, sparse=args.sparse_reference
    )
    expected_dec = decode_reference(reference, rope, cache, x[:, args.length :], args.length)
    print("HF_REFERENCE_READY", flush=True)
    import ttnn

    if args.mesh != 1 and args.decoder != "multichip":
        raise ValueError("Only the multichip decoder opens a multi-device mesh")
    if args.mesh == 1:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, **(
            {"physical_device_ids": [int(__import__("os").environ["QDEV"])]} if "QDEV" in __import__("os").environ else {}))
    else:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if args.mesh == 4 else ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh), trace_region_size=0)
    replicas_identical = []

    def put(t, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), device=mesh, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    def get(t):
        shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]
        replicas_identical.append(all(torch.equal(shards[0], s) for s in shards[1:]))
        return shards[0]

    def drain_profile():
        # Drain only between complete layer passes, never inside a timed window or capture.
        # MID_RUN_DUMP flushes these explicit reads; it does not schedule reads itself.
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)

    try:
        decoder_class, fusion_kwargs = FunctionalDecoder, {}
        if args.decoder in ("fused", "optimized", "multichip"):
            from ..tt.fused_decoder import FusedDecoder

            decoder_class = FusedDecoder
            if args.fusions is not None:
                fusion_kwargs["fusions"] = tuple(filter(None, args.fusions.split(",")))
            if args.decoder in ("optimized", "multichip"):
                from ..tt.optimized_decoder import OptimizedDecoder

                decoder_class = OptimizedDecoder
                if args.decoder == "multichip":
                    from ..tt.multichip_decoder import MultichipDecoder

                    decoder_class = MultichipDecoder
                if args.optimization_json is not None:
                    text = args.optimization_json
                    if text.startswith("@"):
                        text = Path(text[1:]).read_text()
                    fusion_kwargs["optimization"] = json.loads(text)
        model = decoder_class.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            expert_precision=args.expert_precision,
            **fusion_kwargs,
        )
        print("TT_LAYER_READY", flush=True)
        drain_profile()
        from contextlib import nullcontext

        audit = device_only if args.audit else nullcontext
        kw, dkw = {}, {}
        page_size = args.page_size
        if cfg.layer_types[args.layer] == "full_attention":
            blocks_per_row = max(args.minimum_cache_pages, (args.length + args.steps + page_size - 1) // page_size)
            scratch = args.decoder in ("optimized", "multichip") and not args.no_kv_scratch
            blocks = args.batch * blocks_per_row + args.cache_guard_pages + int(scratch)
            table = torch.randperm(args.batch * blocks_per_row, dtype=torch.int32).reshape(
                args.batch, blocks_per_row
            )
            kc = ttnn.zeros(
                (blocks, model.config.n_kv_heads, page_size, cfg.head_dim),
                dtype=ttnn.bfloat8_b if getattr(getattr(model, "optimization", None), "kv_dtype", "bf16") == "bfp8" else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            vc = ttnn.zeros_like(kc)
            if scratch:
                model.bind_paged_kv_cache(kc, vc, scratch_block=blocks - 1)
            else:
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
        split = args.split
        if split:
            px_parts = (put(x[:, :split]), put(x[:, split : args.length]))

        def prefill(continuation_us=None):
            if not split:
                return model.prefill_forward(px, **kw)
            first_out = model.prefill_forward(px_parts[0], **kw)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter_ns()
            second_out = model.prefill_forward(px_parts[1], start_pos=split, **kw)
            ttnn.synchronize_device(mesh)
            if continuation_us is not None:
                continuation_us.append((time.perf_counter_ns() - started) / 1000)
            joined = ttnn.concat([first_out, second_out], dim=1)
            first_out.deallocate(True)
            second_out.deallocate(True)
            return joined

        continuation_us = []
        with audit():
            pf_out = prefill()
        pf = get(pf_out)
        pf_out.deallocate(True)
        warmed_prefill_host_us = None
        prefill_repeat_identical = None
        moe_pieces = None
        prefill_repeat_difference = None
        if args.profile or args.warm_prefill:
            from tracy import signpost

            model.reset_request_state(range(args.batch))
            ttnn.synchronize_device(mesh)
            drain_profile()
            if args.profile:
                signpost("PERF_PREFILL")
            model.moe_pieces_log = []
            started = time.perf_counter_ns()
            measured = prefill(continuation_us)
            ttnn.synchronize_device(mesh)
            warmed_prefill_host_us = (time.perf_counter_ns() - started) / 1000
            moe_pieces, model.moe_pieces_log = [list(p) for p in model.moe_pieces_log], None
            if args.profile:
                signpost("PERF_PREFILL_END")
            drain_profile()
            measured_host = get(measured)
            prefill_repeat_identical = bool(torch.equal(pf, measured_host))
            if not prefill_repeat_identical:
                delta = (pf.float() - measured_host.float()).abs()
                rows = torch.nonzero(delta.reshape(-1, delta.shape[-1]).amax(-1) > 0).flatten().tolist()
                def ordered(t):
                    bits = t.view(torch.int16).int()
                    return torch.where(bits < 0, -(bits & 0x7FFF), bits)

                ulp = (ordered(pf) - ordered(measured_host)).abs()
                prefill_repeat_difference = dict(elements=int((delta > 0).sum()), rows=len(rows),
                                                 first_rows=rows[:8], blocks_of_32=sorted({r // 32 for r in rows}),
                                                 max_abs=float(delta.max()), max_bf16_ulp=int(ulp.max()),
                                                 repeat_pcc=pcc(pf, measured_host))
            pf = measured_host
            measured.deallocate(True)
        if probes is not None:
            pf = pf[:, probes]
        print("PREFILL_PCC", pcc(expected_pf, pf), flush=True)
        dec, deterministic, trace_id = None, None, None
        replay_host_us = []
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
            saved = ([(t, ttnn.clone(t)) for row in model._states[: args.batch] for t in row]
                     if not cfg.layer_types[args.layer] == "full_attention" else [])

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
                try:
                    with audit():
                        out = model.decode_forward(dx, **dkw)
                except BaseException:
                    # An invalid candidate must not leave an active capture
                    # for close_mesh_device to wait on indefinitely.
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    ttnn.release_trace(mesh, tid)
                    raise
                else:
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
                    if cfg.layer_types[args.layer] == "full_attention":
                        ttnn.copy_host_to_device_tensor(pos_host[i], dkw["position_tensor"])
                    if args.profile:
                        signpost(f"PERF_DECODE_STEP_{i}")
                    started = time.perf_counter_ns()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    replay_host_us.append((time.perf_counter_ns() - started) / 1000)
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
                    if cfg.layer_types[args.layer] == "full_attention":
                        ttnn.copy_host_to_device_tensor(pos_host[i], dkw["position_tensor"])
                    with audit():
                        y = model.decode_forward(dx, **dkw)
                    outputs.append(get(y))
                    y.deallocate(True)
                dec = torch.cat(outputs, dim=1)
        decode_pcc_by_token = [pcc(expected_dec[:, i : i + 1], dec[:, i : i + 1]) for i in range(args.steps)]
        import hashlib
        from dataclasses import asdict

        def unwrap(module):
            if type(module).__name__ == "_AllReduceOutput":
                return object.__getattribute__(module, "reduced_module")
            return module

        source_root = Path(__file__).resolve().parents[1] / "tt"
        result = dict(
            device_ids=mesh.get_device_ids(),
            weight_memory=getattr(model,"weight_memory",None),
            input_metadata=input_metadata,
            optimization=asdict(model.optimization) if args.decoder in ("optimized", "multichip") else None,
            mesh_shape=list(mesh.shape),
            tensor_parallel=getattr(model, "tensor_parallel", None),
            merge_stats=getattr(model, "merge_stats", None),
            construction_memory=getattr(model, "construction_memory", None),
            output_replicas_identical=all(replicas_identical),
            decoder_class=type(model).__name__,
            runtime_components={
                "attention": type(unwrap(model.layer.attention)).__name__,
                "attention_reduce": type(model.layer.attention).__name__ == "_AllReduceOutput",
                "feed_forward_reduce": type(model.layer.feed_forward).__name__ == "_AllReduceOutput",
                "router": type(model.layer.feed_forward.module.router).__name__,
                "experts": type(model.layer.feed_forward.module.experts).__name__,
                "decode_experts": type(getattr(model.layer.feed_forward.module.experts, "original", None)).__name__,
                "shared": type(model.layer.feed_forward.module.shared).__name__,
            },
            fusions=getattr(model, "fusions", None),
            layer_type=cfg.layer_types[args.layer],
            layer_idx=args.layer,
            weights=args.weights,
            implementation_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))
            },
            gdn_recurrence=(
                "native_tf32_token"
                if "native_recurrence" in (getattr(model, "fusions", ()) or ())
                else "native_exact_chunk_and_sfpu_fp32_token"
            )
            if cfg.layer_types[args.layer] == "linear_attention"
            else None,
            expert_precision=args.expert_precision,
            profile=args.profile,
            warmed_prefill_host_us=warmed_prefill_host_us,
            split=split or None,
            kv_scratch_block=getattr(model, "kv_scratch_block", None),
            warmed_continuation_host_us=continuation_us[-1] if continuation_us else None,
            prefill_repeat_identical=prefill_repeat_identical,
            prefill_repeat_difference=prefill_repeat_difference,
            moe_pieces=moe_pieces,
            length=args.length,
            batch=args.batch,
            decode_steps=args.steps,
            page_size=args.page_size,
            minimum_cache_pages=args.minimum_cache_pages,
            cache_guard_pages=args.cache_guard_pages,
            prefill_pcc=pcc(expected_pf, pf),
            prefill_pcc_min=min(pcc(expected_pf[r], pf[r]) for r in range(args.batch)),
            prefill_token_pcc=(token_pcc := sorted(
                (pcc(expected_pf[r, t], pf[r, t]), r, t) for r in range(args.batch) for t in range(pf.shape[1])
            ))[:8],
            prefill_tokens_below_0995=sum(value < 0.995 for value, _, _ in token_pcc),
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
            traced_replay_host_us=replay_host_us,
        )
        if args.save_tensors:
            torch.save({"prefill": pf, "decode": dec}, args.save_tensors)
        if args.compare_tensors:
            control = torch.load(args.compare_tensors, weights_only=True)
            result["unfused_equivalence"] = result["control_equivalence"] = {
                "prefill_pcc": pcc(control["prefill"], pf),
                "decode_pcc_min": min(pcc(control["decode"][:, i], dec[:, i]) for i in range(args.steps))
                if args.steps
                else None,
                "source": str(args.compare_tensors),
                "prefill_max_abs": float((control["prefill"].float() - pf.float()).abs().max()),
                "decode_max_abs": float((control["decode"].float() - dec.float()).abs().max()) if args.steps else None,
                "prefill_identical": bool(torch.equal(control["prefill"], pf)),
                "decode_identical": bool(torch.equal(control["decode"], dec)) if args.steps else None,
            }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if args.compare_tensors:
            assert result["unfused_equivalence"]["prefill_pcc"] >= 0.995
            if args.steps:
                assert result["unfused_equivalence"]["decode_pcc_min"] >= 0.995
        assert result["prefill_pcc_min"] >= 0.995
        if args.steps:
            assert result["decode_pcc_min"] >= 0.995
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
