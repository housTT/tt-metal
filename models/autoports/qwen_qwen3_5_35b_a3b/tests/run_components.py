# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight layer-0 diagnosis with explicit intermediate device-to-host reads.

This is an eager diagnostic, never a performance, trace or device-only audit run.
Both implementations consume the entire prompt and every decode token. Instance
probes observe the actual TT layer path; they do not replace its computation.
"""

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("../weights/Qwen3.5-35B-A3B"))
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--selected",
        default=None,
        help="Comma-separated zero-based decode steps; defaults to all eight in positions-row mode",
    )
    parser.add_argument(
        "--positions-row",
        type=int,
        choices=(10, 18),
        default=None,
        help="Reproduce a run_positions batch-32/seed-1234 row; overrides length/steps",
    )
    parser.add_argument("--expert-precision", choices=("inherited", "bf16"), default="inherited")
    parser.add_argument(
        "--norm-policy",
        choices=("current", "inherited"),
        default="current",
        help="inherited recreates the original common RMSNorm as a diagnostic control",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.positions_row is not None:
        from .run_positions import prompt_lengths

        args.length = prompt_lengths(32, 1234)[args.positions_row]
        args.steps = 8
    defaults = "0,1,2,3,4,5,6,7" if args.positions_row is not None else "70,71,72,108,109,110"
    args.selected = sorted({int(value) for value in (args.selected or defaults).split(",")})
    if args.length < 1 or args.steps < 1 or any(i < 0 or i >= args.steps for i in args.selected):
        parser.error("length/steps must be positive and selected steps must lie in [0,steps)")
    return args


def positions_row_input(row, hidden_size):
    """Regenerate the complete run_positions default stream; retain one real row."""
    import torch

    from .run_positions import prompt_lengths

    generator = torch.Generator().manual_seed(1234)
    selected = None
    for index, length in enumerate(prompt_lengths(32, 1234)):
        # Match the source runner's shape, generator, BF16 draw and BF16 scale
        # in the same order. Skipping preceding draws would change this row.
        current = torch.randn(1, length + 8, hidden_size, dtype=torch.bfloat16, generator=generator) * 0.1
        if index == row:
            selected = current
    if selected is None:
        raise ValueError("Requested positions row is outside the original batch")
    return selected


def metrics(reference, actual):
    import torch

    a, b = reference.float().reshape(-1), actual.float().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Component volume differs: {reference.shape} versus {actual.shape}")
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "pcc": None, "max_abs": None, "relative_l2": None}
    denominator = torch.linalg.vector_norm(a)
    correlation = torch.corrcoef(torch.stack((a, b)))[0, 1]
    return {
        "finite": True,
        "pcc": float(correlation) if torch.isfinite(correlation) else None,
        "max_abs": float((a - b).abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(a - b) / denominator) if denominator else None,
    }


def route_summary(hf_record, tt_record, reference):
    """Use actual TT dense routing to recover selected IDs, including softmax ties."""
    import torch

    hf_logits, hf_weights, hf_ids = hf_record["router"]
    logits = tt_record["router_logits"].reshape(-1)
    dense = tt_record["router_dense"].reshape(-1)
    top_k = reference.mlp.gate.top_k
    if int(torch.count_nonzero(dense)) != top_k:
        raise AssertionError("Actual router output does not contain exactly top-k nonzero routes")
    tt_ids = torch.topk(dense, top_k).indices.tolist()
    expected_ids = hf_ids.reshape(-1).tolist()
    on_tt_logits, _, on_tt_ids = reference.mlp.gate(tt_record["ff_norm"])
    fp32_softmax_ids = logits.float().softmax(-1).topk(top_k).indices.tolist()

    def overlap(left, right):
        return len(set(left) & set(right))

    def boundary(logit_vector):
        top = logit_vector.float().reshape(-1).topk(top_k + 1)
        prob = logit_vector.float().reshape(-1).softmax(-1).topk(top_k + 1).values
        return {
            "logit_gap_8_9": float(top.values[top_k - 1] - top.values[top_k]),
            "probability_gap_8_9": float(prob[top_k - 1] - prob[top_k]),
        }

    def bf16_probability_control(logit_vector):
        # Isolate storing softmax probabilities in BF16 before selection.
        # Native SFPU error/tie ordering is separate; expose every boundary tie.
        probs = logit_vector.float().reshape(-1).softmax(-1).to(torch.bfloat16)
        values, indices = probs.topk(top_k + 1)
        cutoff = values[top_k - 1]
        chosen = indices[:top_k].tolist()
        return {
            "ids": chosen,
            "overlap_actual_tt": overlap(chosen, tt_ids),
            "boundary_tie_ids": torch.where(probs == cutoff)[0].tolist(),
            "probability_gap_8_9": float(values[top_k - 1] - values[top_k]),
            "experts_at_or_above_cutoff": int(torch.count_nonzero(probs >= cutoff)),
        }

    hf_dense = torch.zeros_like(dense)
    hf_dense.scatter_(0, hf_ids.reshape(-1), hf_weights.reshape(-1).to(dense.dtype))
    hf_input_tt_dense = tt_record["router_dense_on_hf_ff"].reshape(-1)
    hf_input_tt_ids = hf_input_tt_dense.topk(top_k).indices.tolist()
    return {
        "hf_ids": expected_ids,
        "tt_actual_ids": tt_ids,
        "tt_logits_fp32_softmax_ids": fp32_softmax_ids,
        "hf_router_on_tt_ff_ids": on_tt_ids.reshape(-1).tolist(),
        "tt_router_on_hf_ff_ids": hf_input_tt_ids,
        "overlap_hf_tt": overlap(expected_ids, tt_ids),
        "overlap_actual_tt_vs_fp32_softmax": overlap(tt_ids, fp32_softmax_ids),
        "overlap_same_tt_input": overlap(on_tt_ids.reshape(-1).tolist(), tt_ids),
        "overlap_same_hf_input": overlap(expected_ids, hf_input_tt_ids),
        "hf_boundary": boundary(hf_logits),
        "tt_boundary": boundary(logits),
        "hf_on_tt_input_boundary": boundary(on_tt_logits),
        "dense_weight_metrics": metrics(hf_dense, dense),
        "router_logits_same_tt_input": metrics(on_tt_logits, logits),
        "bf16_probability_controls": {
            "tt_logits": bf16_probability_control(logits),
            "hf_logits": bf16_probability_control(hf_logits),
            "hf_logits_on_tt_ff_input": bf16_probability_control(on_tt_logits),
        },
    }


def main():
    args = parse_args()
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [source_root / "tt" / name for name in ("functional_decoder.py", "functional_gdn.py")]
    source_paths.append(Path(__file__).resolve())
    if args.positions_row is not None:
        source_paths.append(Path(__file__).with_name("run_positions.py"))
    source_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    from ..tt.functional_decoder import FunctionalDecoder
    from .hf_reference import decode_reference, prefill_reference
    from .layer_weights import load_layer_state_dict

    torch.manual_seed(1234)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    if args.length + args.steps > cfg.max_position_embeddings:
        raise ValueError("Requested workload exceeds the target advertised context")
    state = load_layer_state_dict(args.checkpoint, 0)
    prefix = "model.language_model.layers.0."
    local = {key.removeprefix(prefix): value for key, value in state.items()}
    with torch.device("meta"):
        reference = Qwen3_5MoeDecoderLayer(cfg, 0)
    reference.load_state_dict(local, assign=True, strict=True)
    reference.eval()
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    x = (
        positions_row_input(args.positions_row, cfg.hidden_size)
        if args.positions_row is not None
        else torch.randn(1, args.length + args.steps, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
    )
    input_sha256 = hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    reproduction = (
        dict(
            source="run_positions.py",
            source_batch=32,
            source_seed=1234,
            source_row=args.positions_row,
            source_slot=31 - args.positions_row,
            source_decode_steps=8,
            isolated_batch=1,
            isolated_slot=0,
            input_sha256=input_sha256,
        )
        if args.positions_row is not None
        else None
    )
    hf_records, expected_decode = {}, []
    selected, cursor = set(args.selected), {"step": -1}
    hooks = []

    def hf_hook(name, save_input=False):
        def record(module, inputs, output):
            step = cursor["step"]
            if step not in selected:
                return
            entry = hf_records.setdefault(step, {})
            if save_input:
                entry["residual"] = inputs[0].detach().clone()
            entry[name] = (
                tuple(item.detach().clone() for item in output)
                if isinstance(output, tuple)
                else output.detach().clone()
            )

        return record

    with torch.no_grad():
        expected_prefill, cache, _ = prefill_reference(reference, cfg, rope, x[:, : args.length], 0)
        for module, name, save_input in (
            (reference.input_layernorm, "attn_norm", False),
            (reference.linear_attn, "gdn", False),
            (reference.post_attention_layernorm, "ff_norm", True),
            (reference.mlp.gate, "router", False),
            (reference.mlp, "moe", False),
        ):
            hooks.append(module.register_forward_hook(hf_hook(name, save_input)))
        try:
            for step in range(args.steps):
                cursor["step"] = step
                token = x[:, args.length + step : args.length + step + 1]
                output = decode_reference(reference, rope, cache, token, args.length + step)
                expected_decode.append(output)
                if step in selected:
                    hf_records[step]["output"] = output
        finally:
            for handle in hooks:
                handle.remove()
    print("HF_COMPONENT_REFERENCE_READY", flush=True)

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def put(tensor):
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def get(tensor):
        # Intentional diagnostic boundary, prohibited in measured production forward.
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).clone()

    tt_records, decode_metrics = {}, []
    cursor["step"] = -1

    def observe(name, tensor):
        step = cursor["step"]
        if step in selected:
            tt_records.setdefault(step, {})[name] = get(tensor)

    class NormProbe:
        def __init__(self, original, name):
            self.original, self.name = original, name

        def __call__(self, tensor, *positional, **kwargs):
            if self.name == "ff_norm":
                observe("residual", tensor)
            result = self.original(tensor, *positional, **kwargs)
            observe(self.name, result)
            return result

    def probe_forward(module, name):
        original = module.forward

        def forward(*positional, **kwargs):
            result = original(*positional, **kwargs)
            observe(name, result)
            return result

        module.forward = forward

    class RouterProbe:
        def __init__(self, original):
            self.original = original

        def __call__(self, tensor):
            if cursor["step"] in selected:
                # This extra stateless matmul mirrors Qwen36Router.__call__.
                # Actual routes below always come from the production router.
                logits = ttnn.linear(
                    tensor, self.original.proj_weight, compute_kernel_config=self.original.compute_kernel_config
                )
                observe("router_logits", logits)
                logits.deallocate(True)
            result = self.original(tensor)
            observe("router_dense", result)
            return result

    try:
        model = FunctionalDecoder.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh,
            max_batch_size=1,
            expert_precision=args.expert_precision,
        )
        if args.norm_policy == "inherited":
            from models.common.rmsnorm import RMSNorm

            for attribute, key in (("attention_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
                setattr(
                    model.layer,
                    attribute,
                    RMSNorm(
                        device=mesh,
                        dim=cfg.hidden_size,
                        state_dict=local,
                        weight_key=key,
                        weight_dtype=ttnn.bfloat16,
                        add_unit_offset=True,
                        eps=cfg.rms_norm_eps,
                    ),
                )
        px = put(x[:, : args.length])
        pf_out = model.prefill_forward(px)
        prefill_metrics = metrics(expected_prefill, get(pf_out))
        pf_out.deallocate(True)
        px.deallocate(True)
        print("TT_PREFILL_READY", json.dumps(prefill_metrics), flush=True)

        layer, moe = model.layer, model.layer.feed_forward.module
        router = moe.router
        if type(router).__name__ not in ("Qwen36Router", "_FunctionalRouter"):
            raise TypeError("Update the duplicate logits probe for the current router implementation before using it")
        layer.attention_norm = NormProbe(layer.attention_norm, "attn_norm")
        layer.ffn_norm = NormProbe(layer.ffn_norm, "ff_norm")
        probe_forward(layer.attention, "gdn")
        probe_forward(layer.feed_forward, "moe")
        moe.router = RouterProbe(router)

        for step in range(args.steps):
            cursor["step"] = step
            dx = put(x[:, args.length + step : args.length + step + 1])
            output = model.decode_forward(dx)
            host_output = get(output)
            decode_metrics.append(metrics(expected_decode[step], host_output))
            if step in selected:
                tt_records[step]["output"] = host_output
                # Same-HF-input router control is stateless and cannot advance GDN.
                hf_ff = put(hf_records[step]["ff_norm"].reshape(1, 1, 1, cfg.hidden_size))
                same_input_routing = router(hf_ff)
                tt_records[step]["router_dense_on_hf_ff"] = get(same_input_routing)
                same_input_routing.deallocate(True)
                hf_ff.deallocate(True)
                print("COMPONENT_STEP", step, decode_metrics[-1]["pcc"], flush=True)
            output.deallocate(True)
            dx.deallocate(True)
        cursor["step"] = -1
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)

    details = []
    with torch.no_grad():
        for step in args.selected:
            hf, tt = hf_records[step], tt_records[step]
            comparisons = {
                name: metrics(hf[name], tt[name])
                for name in ("attn_norm", "gdn", "residual", "ff_norm", "moe", "output")
            }
            comparisons["router_logits"] = metrics(hf["router"][0], tt["router_logits"])
            comparisons["ff_norm_same_tt_input"] = metrics(
                reference.post_attention_layernorm(tt["residual"]), tt["ff_norm"]
            )
            comparisons["moe_same_tt_input"] = metrics(reference.mlp(tt["ff_norm"]), tt["moe"])
            details.append(
                {
                    "decode_step_zero_based": step,
                    "absolute_position": args.length + step,
                    "components": comparisons,
                    "routes": route_summary(hf, tt, reference),
                }
            )

    minimums = {}
    for name in details[0]["components"]:
        candidates = [
            (row["decode_step_zero_based"], row["components"][name]["pcc"])
            for row in details
            if row["components"][name]["pcc"] is not None
        ]
        step, value = min(candidates, key=lambda entry: entry[1]) if candidates else (None, None)
        minimums[name] = {"pcc": value, "decode_step_zero_based": step}
    source_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    result = {
        "kind": "component_diagnostic",
        "layer_idx": 0,
        "layer_type": "linear_attention",
        "weights": "real",
        "length": args.length,
        "decode_steps": args.steps,
        "batch": 1,
        "positions_reproduction": reproduction,
        "input_sha256": input_sha256,
        "norm_policy": args.norm_policy,
        "expert_precision": args.expert_precision,
        "traced": False,
        "runtime_audit": False,
        "performance_measurement": False,
        "host_reads_between_components": True,
        "selected_steps_zero_based": args.selected,
        "source_sha256": source_before,
        "source_changed_during_run": source_before != source_after,
        "prefill": prefill_metrics,
        "decode_pcc_by_token": [entry["pcc"] for entry in decode_metrics],
        "decode_pcc_min": min((entry["pcc"] for entry in decode_metrics if entry["pcc"] is not None), default=None),
        "selected_component_minimums": minimums,
        "selected_steps": details,
        "notes": [
            "All prefill/decode tokens execute the real layer; only selected steps have intermediate host reads.",
            "Router logits are a duplicate stateless matmul; IDs come from actual production dense routing.",
            "No timings or tensors are persisted; this run cannot satisfy trace or fallback-audit gates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "COMPONENT_DIAGNOSTIC_WRITTEN",
        str(args.output),
        json.dumps({"decode_pcc_min": result["decode_pcc_min"], "component_minimums": minimums}),
        flush=True,
    )


if __name__ == "__main__":
    main()
