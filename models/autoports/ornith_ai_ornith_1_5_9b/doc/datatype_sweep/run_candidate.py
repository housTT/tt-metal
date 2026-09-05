# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model readiness measurements with runtime policy and replay evidence."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_5_9b.tt.precision import layer_precision, load_precision

ROOT = Path(__file__).resolve().parents[2]


def compute_summary(config):
    return {
        name: str(getattr(config, name))
        for name in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
    }


def runtime_summary(gen):
    model = gen.model
    model.lm_head.load_device_weights()
    assert all(
        a.buffer_address() == b.buffer_address() and a.dtype == b.dtype
        for a, b in zip(model.head_weights, model.lm_head.output_weights)
    )
    assert str(gen._inputs[0].dtype).split(".")[-1].lower() == model.precision["token_dtype"]
    assert str(gen.kv_cache.zero_hidden.dtype).split(".")[-1].lower() == model.precision["residual_dtype"]
    for index, layer in zip(model.layer_indices, gen.kv_cache.decode_layers):
        selected = layer_precision(model.precision, index)
        w, f = selected["weight_groups"], selected["compute_fidelities"]
        for role, compute in layer.projection_compute.items():
            group = (
                "mlp_down"
                if role == "down_proj"
                else "mlp_gate_up"
                if role in ("gate_proj", "up_proj", "gate_up")
                else "attention"
            )
            assert layer.w[role].dtype == getattr(ttnn, w[group]), (index, role)
            expected_decode = w["decode_qkvg"] if role == "qkvg" else w[group]
            if role in layer.decode_weights:
                assert layer.decode_weights[role].dtype == getattr(ttnn, expected_decode), (index, role)
            assert compute.math_fidelity == getattr(ttnn.MathFidelity, f[group]), (index, role)
            assert compute.fp32_dest_acc_en == model.precision["matmul_flags"]["projection_fp32_dest_acc_en"]
        assert layer.activation_dtype == getattr(ttnn, model.precision["activation_dtype"])
        if layer.is_full_attention:
            assert layer.k_cache.dtype == layer.v_cache.dtype == getattr(ttnn, model.precision["kv_cache_dtype"])
        else:
            assert layer.recurrent_state.dtype == getattr(ttnn, model.precision["recurrent_dtype"])
        assert layer.mesh_config.ccl_dtype == model.precision["ccl_dtype"]
    assert all(w.dtype == getattr(ttnn, model.precision["weight_groups"]["lm_head"]) for w in model.head_weights)
    assert model.head_compute.math_fidelity == getattr(
        ttnn.MathFidelity, model.precision["compute_fidelities"]["lm_head"]
    )
    geometry = model.precision["head_geometry"]
    assert model.head_columns == geometry["columns"]
    assert model.head_program.in0_block_w == geometry["in0_block_w"]
    assert model.head_program.num_workers_per_dram_bank == geometry["readers"]
    assert model.head_program.per_core_N == geometry["columns"] // 32 // geometry["cores"]
    assert model.head_input_memory.shard_spec.shape == [32, model.dim // geometry["cores"]]
    assert model.embed_weight.dtype == getattr(ttnn, model.precision["weight_groups"]["embedding"])
    assert model.norm_weight.dtype == getattr(ttnn, model.precision["weight_groups"]["norm"])
    return {
        "propagation_assertions_passed": True,
        "policy": model.precision,
        "token_dtype": str(gen._inputs[0].dtype),
        "head_weights": [str(w.dtype) for w in model.head_weights],
        "head_compute": compute_summary(model.head_compute),
        "head_program": str(model.head_program),
        "head_input_memory": str(model.head_input_memory),
        "embedding": str(model.embed_weight.dtype),
        "norm": str(model.norm_weight.dtype),
        "zero_hidden": str(gen.kv_cache.zero_hidden.dtype),
        "cache_context": gen.kv_cache.context,
        "model_max_context": model.max_context,
        "layers": [
            {
                "index": index,
                "kind": layer.kind,
                "weights": {k: str(v.dtype) for k, v in layer.w.items() if hasattr(v, "dtype")},
                "decode_weights": {k: str(v.dtype) for k, v in layer.decode_weights.items()},
                "compute": {k: compute_summary(v) for k, v in layer.projection_compute.items()},
                "activation_dtype": str(getattr(layer, "activation_dtype", ttnn.bfloat16)),
                "recurrent_dtype": str(layer.recurrent_state.dtype) if not layer.is_full_attention else None,
                "ccl_dtype": layer.mesh_config.ccl_dtype,
                "kv_dtype": str(layer.k_cache.dtype) if layer.is_full_attention else None,
            }
            for index, layer in zip(model.layer_indices, gen.kv_cache.decode_layers)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--head-block-w", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--teacher-repeats", type=int, default=1)
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    if args.teacher_repeats < 1:
        parser.error("teacher-repeats must be positive")
    policy = load_precision(args.config)
    result = dict(
        command=sys.argv,
        config_id=policy["config_id"],
        dtype_policy=policy,
        hardware="four Blackhole chips on two P300c boards",
        mesh=[1, 4],
        measurement_regime="full32 batch1 AIME24 chat100; traced teacher forcing with callback/logit readback",
        reference=str(ROOT / "readiness_aime24_chat.refpt"),
        pass_status="error",
    )
    mesh = open_ornith_mesh()
    gen = None
    teardown = None
    try:
        gen = build_generator(
            ROOT,
            mesh,
            precision_config=policy,
            cache_context=2048,
            layer_indices=[0, 3] if args.smoke else None,
            **({"lm_head_block_w": args.head_block_w} if args.head_block_w is not None else {}),
        )
        teardown = gen.teardown
        result["runtime"] = runtime_summary(gen)
        result["dtype_policy"] = gen.model.precision
        if args.smoke:
            first = gen.generate([100] * 131, 8, stop_on_eos=False)
            second = gen.generate([100] * 131, 8, stop_on_eos=False)
            assert first == second
            result.update(tokens=first, perf=gen.perf, pass_status="smoke_pass")
        else:
            gen.teardown = lambda: None
            modes = ["teacher_forcing"] if args.teacher_only else ["prefill_check", "teacher_forcing"]
            for mode in modes:
                module = importlib.import_module(f"models.common.readiness_check.run_{mode}")
                original = module._import_build_generator
                module._import_build_generator = lambda path: lambda **kwargs: gen
                try:
                    samples = []
                    for repeat in range(args.teacher_repeats if mode == "teacher_forcing" else 1):
                        gen.reset()
                        before = dict(gen.counters)
                        rows = getattr(module, f"run_{mode}")(
                            model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                        )
                        result[mode] = rows
                        result[mode + "_counters"] = {k: gen.counters[k] - before[k] for k in before}
                        if mode == "teacher_forcing":
                            result["generator_perf"] = gen.perf
                            result["runtime"]["logits_dtype"] = str(gen._logits.dtype)
                            assert gen._logits.dtype == getattr(ttnn, policy["logits_dtype"])
                            assert gen._logits.dtype == getattr(ttnn, policy["sampling_dtype"])
                            assert gen._model_trace is not None
                            assert result[mode + "_counters"]["model_replays"] >= 99
                            result["trace_verified"] = True
                        samples.append(dict(metrics=rows, counters=result[mode + "_counters"], perf=dict(gen.perf)))
                    result[mode + "_samples"] = samples
                    if mode == "teacher_forcing":
                        selected_sample = sorted(samples, key=lambda sample: sample["metrics"][0]["decode_t/s/u"])[
                            len(samples) // 2
                        ]
                        result[mode] = selected_sample["metrics"]
                        result[mode + "_counters"] = selected_sample["counters"]
                        result["generator_perf"] = selected_sample["perf"]
                        result["repeat_selection"] = "median traced teacher-forcing throughput; TTFT from same sample"
                finally:
                    module._import_build_generator = original
            result["pass_status"] = (
                "pass"
                if all(
                    r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1 and r["total"] == 100
                    for mode in modes
                    for sample in result[mode + "_samples"]
                    for r in sample["metrics"]
                )
                else "accuracy_fail"
            )
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if teardown is not None:
            teardown()
        close_ornith_mesh(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
