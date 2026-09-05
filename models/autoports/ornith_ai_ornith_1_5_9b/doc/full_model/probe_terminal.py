"""Real-weight terminal-only LM-head packing, accuracy, and trace experiment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.reference.hf_reference import CHECKPOINT_TEXT_PREFIX
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh


def metric(actual, expected):
    a, b = actual.float(), expected.float()
    error = (a - b).abs()
    return dict(
        finite=bool(torch.isfinite(a).all()),
        pcc=float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]),
        max_abs=float(error.max()),
        mean_abs=float(error.mean()),
        top1=float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        top5=float((a.topk(5).indices == b.argmax(-1)[:, None]).any(-1).float().mean()),
        top100=float((a.topk(100).indices == b.argmax(-1)[:, None]).any(-1).float().mean()),
    )


def describe(t):
    return dict(
        shape=list(t.shape),
        padded_shape=list(t.padded_shape),
        dtype=str(t.dtype),
        layout=str(t.layout),
        memory=str(t.memory_config()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--in0-block-w", type=int, required=True)
    parser.add_argument("--readers", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "bfloat8_b", "bfloat4_b"], default="bfloat16")
    parser.add_argument("--fidelity", choices=["HiFi4", "HiFi2", "LoFi"], default="HiFi4")
    parser.add_argument("--resident-l1-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = dict(
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, rows=[]
    )
    mesh = open_ornith_mesh()
    try:
        model = OrnithModel(
            None,
            mesh,
            layer_indices=[],
            max_context=2048,
            lm_head_columns=args.columns,
            lm_head_block_w=args.in0_block_w,
            lm_head_readers=args.readers,
            lm_head_dtype=getattr(ttnn, args.dtype),
            lm_head_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
        )
        if args.resident_l1_bytes:
            width = args.resident_l1_bytes // 64
            resident_mem = ttnn.create_sharded_memory_config(
                shape=(32, width),
                core_grid=ttnn.CoreGrid(x=8, y=4),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            resident = model.upload(torch.zeros(1, 1, 32, width * 32, dtype=torch.bfloat16), memory=resident_mem)
            report["resident_l1"] = describe(resident)
        weight_map = json.loads((model.model_path / "model.safetensors.index.json").read_text())["weight_map"]

        def read(key):
            with safe_open(str(model.model_path / weight_map[key]), framework="pt", device="cpu") as f:
                return f.get_tensor(key)

        head = read("lm_head.weight").float().T.contiguous()
        norm = read(CHECKPOINT_TEXT_PREFIX + "norm.weight").float() + 1
        report["weight"] = describe(model.head_weights[0])
        report["program"] = str(model.head_program)
        report["compute"] = str(model.head_compute)
        report["dram_grid"] = str(mesh.dram_grid_size())
        print("MODEL_READY", flush=True)
        for rows in [1, 32]:
            torch.manual_seed(1024 + rows)
            hidden = torch.randn(1, rows, model.dim, dtype=torch.bfloat16)
            x = model.upload(hidden)
            padded = torch.nn.functional.pad(hidden, (0, 0, 0, 32 - rows)).reshape(1, 1, 32, model.dim)
            normal_device = ttnn.rms_norm(
                model.upload(padded), weight=model.norm_weight, epsilon=model.hf_config.rms_norm_eps
            )
            normal = ttnn.to_torch(ttnn.get_device_tensors(normal_device)[0])[0, 0, :rows].float()
            linear_oracle = normal @ head
            hf_normal = (
                hidden.float()
                * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + model.hf_config.rms_norm_eps)
                * norm
            )[0]
            full_oracle = hf_normal @ head
            logits = model.terminal(x)
            actual = model.logits_to_host(logits, rows)
            item = dict(
                rows=rows,
                input=describe(x),
                normalized=describe(normal_device),
                output=describe(logits),
                linear=metric(actual, linear_oracle),
                full=metric(actual, full_oracle),
            )
            item["logits_sha256"] = hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()
            boundaries = sorted(set(i for i in range(args.columns, model.vocab_size, args.columns)))
            cols = torch.tensor([i + d for i in boundaries for d in [-1, 0, 1] if i + d < model.vocab_size])
            item["boundary_max_abs"] = float((actual[:, cols] - linear_oracle[:, cols]).abs().max())
            repeated = model.terminal(x)
            item["eager_exact_repeat"] = bool(torch.equal(actual, model.logits_to_host(repeated, rows)))
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = model.terminal(x)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            trace_host = model.logits_to_host(traced, rows)
            item["trace_exact_eager"] = bool(torch.equal(actual, trace_host))
            start = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            item["trace_ms"] = (time.perf_counter() - start) * 1000 / 32
            ttnn.release_trace(mesh, trace)
            report["rows"].append(item)
            print(json.dumps(item), flush=True)
            assert item["linear"]["finite"]
            if args.dtype == "bfloat16":
                assert item["linear"]["top5"] >= 0.98 and item["linear"]["top100"] == 1
            assert item["eager_exact_repeat"] and item["trace_exact_eager"]
        suffix = (
            ""
            if args.dtype == "bfloat16" and args.fidelity == "HiFi4" and not args.resident_l1_bytes
            else f"_{args.dtype}_{args.fidelity}_l1{args.resident_l1_bytes}"
        )
        path = (
            Path(__file__).parent
            / f"terminal_{args.columns}_block{args.in0_block_w}_readers{args.readers}{suffix}.json"
        )
        path = args.output or path
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("TERMINAL_PROBE_OK", path, flush=True)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
