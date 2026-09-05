# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real FP32 GDN boundary with the native full-stack L1 allocation footprint."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def host_all(tensor):
    return [ttnn.to_torch(local).float() for local in ttnn.get_device_tensors(tensor)]


def l1(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return {
        key: int(getattr(view, key))
        for key in ("total_bytes_allocated_per_bank", "largest_contiguous_bytes_free_per_bank")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"target_resident_bytes_per_bank": 221952, "candidates": []}
    mesh = open_ornith_mesh()
    captured = {}
    original = OptimizedDecoder._prefill_linear
    try:
        model = OrnithModel(None, mesh, layer_indices=[0, 3], max_context=4096)
        cache = model.allocate_cache(context=4096)
        table = model.page_table(cache)
        layer = cache.prefill_layers[0]
        layer.optimization = replace(layer.optimization, large_prefill_role_configs={})

        def capture(self, x, role, **kwargs):
            if role == "gdn_out":
                captured["input"] = ttnn.clone(x)
                captured["kwargs"] = kwargs.copy()
            result = original(self, x, role, **kwargs)
            if role == "gdn_out":
                captured["baseline"] = host_all(result)
            return result

        OptimizedDecoder._prefill_linear = capture
        logits = model.prefill_forward([[100] * 2048], page_table=table, kv_cache=cache, prompt_lens=[2048])
        ttnn.deallocate(logits)
        OptimizedDecoder._prefill_linear = original
        x = captured["input"]
        assert x.dtype == ttnn.float32 and tuple(x.shape) == (1, 2048, 1024), (x.dtype, x.shape)
        report["input"] = {"shape": list(x.shape), "dtype": str(x.dtype), "memory": str(x.memory_config())}
        report["weight"] = {"shape": list(layer.w["gdn_out"].shape), "dtype": str(layer.w["gdn_out"].dtype)}
        inputs, weights = host_all(x), host_all(layer.w["gdn_out"])
        oracles = [a @ b for a, b in zip(inputs, weights)]
        report["before_reservation"] = l1(mesh)
        extra = 221952 - report["before_reservation"]["total_bytes_allocated_per_bank"]
        assert extra > 0 and extra % 64 == 0, extra
        grid = mesh.compute_with_storage_grid_size()
        banks = grid.x * grid.y
        memory = ttnn.create_sharded_memory_config(
            shape=(1, extra // 2),
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        resident = model.upload(
            torch.zeros(banks, extra // 2, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, memory=memory
        )
        report["with_reservation"] = l1(mesh)
        assert report["with_reservation"]["total_bytes_allocated_per_bank"] == 221952, report
        try:
            original(layer, x, "gdn_out", **captured["kwargs"])
        except RuntimeError as exc:
            message = str(exc)
            assert "1422336" in message and "1318144" in message, message
            report["original_failure"] = message
            print("EXACT_ORIGINAL_COLLISION_VERIFIED", flush=True)
        else:
            raise AssertionError("Original program unexpectedly fits the full-stack footprint")
        for label, config in (("n6_k16", {"out_block_w": 6}), ("n12_k8", {"block_w": 8})):
            layer.optimization = replace(layer.optimization, large_prefill_role_configs={"gdn_out": config})
            output = layer._prefill_linear(x, "gdn_out", **captured["kwargs"])
            actuals = host_all(output)
            ranks = []
            for actual, baseline, oracle in zip(actuals, captured["baseline"], oracles):
                pcc = float(torch.corrcoef(torch.stack([actual.flatten(), oracle.flatten()]))[0, 1])
                relative = float(torch.linalg.vector_norm(actual - baseline) / torch.linalg.vector_norm(baseline))
                ranks.append(
                    {
                        "oracle_pcc": pcc,
                        "baseline_relative_l2": relative,
                        "baseline_exact": bool(torch.equal(actual, baseline)),
                    }
                )
            ttnn.deallocate(output)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(20):
                output = layer._prefill_linear(x, "gdn_out", **captured["kwargs"])
                ttnn.deallocate(output)
            ttnn.synchronize_device(mesh)
            item = {
                "label": label,
                "config": config,
                "warm_ms": (time.perf_counter() - start) * 1000 / 20,
                "ranks": ranks,
                "accepted": all(rank["oracle_pcc"] >= 0.99 and rank["baseline_relative_l2"] < 0.01 for rank in ranks),
            }
            report["candidates"].append(item)
            print("CANDIDATE_OK", item, flush=True)
        winner = min((item for item in report["candidates"] if item["accepted"]), key=lambda item: item["warm_ms"])
        report["selected"] = winner["label"]
        layer.optimization = replace(layer.optimization, large_prefill_role_configs={"gdn_out": winner["config"]})
        model.reset_cache(cache)
        logits = model.prefill_forward([[100] * 2048], page_table=table, kv_cache=cache, prompt_lens=[2048])
        assert torch.isfinite(model.logits_to_host(logits, 1)).all()
        ttnn.deallocate(logits)
        report["reduced_2048_prefill_full_residency"] = True
        report["pass"] = True
        print("CONTEXT_L1_PROBE_OK", flush=True)
    finally:
        OptimizedDecoder._prefill_linear = original
        close_ornith_mesh(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
