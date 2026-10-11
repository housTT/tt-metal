# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight heterogeneous request positions and state ownership under one trace.

Each row owns an independent prompt/cache and a permuted persistent request slot.
All reference results, inputs and metadata are prepared before trace capture.
This is correctness evidence; it deliberately records no performance numbers.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

BOUNDARY_LENGTHS = (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513)


def prompt_lengths(batch, seed):
    """Distinct valid lengths, including each required boundary at batch >=16."""
    selected = list(BOUNDARY_LENGTHS[:batch])
    candidates = [value for value in range(1, 1024) if value not in BOUNDARY_LENGTHS]
    selected.extend(random.Random(seed).sample(candidates, max(0, batch - len(selected))))
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("../weights/Qwen3.5-35B-A3B"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--no-kv-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--expert-precision", choices=("inherited", "bf16"), default="inherited")
    parser.add_argument("--decoder", choices=("functional", "fused", "optimized"), default="functional")
    parser.add_argument("--fusions")
    parser.add_argument("--optimization-json")
    parser.add_argument("--input-activations", type=Path)
    parser.add_argument("--window-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32:
        parser.error("batch must lie in [1,32]")
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.page_size < 32 or args.page_size % 32:
        parser.error("page-size must be a positive multiple of 32")
    return args


def pcc(expected, actual):
    import torch

    a, b = expected.float().flatten(), actual.float().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("PCC input shapes or finite-value checks failed")
    value = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
    if not (-1.00001 <= value <= 1.00001):
        raise AssertionError("PCC is not finite")
    return value


def main():
    args = parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        raise RuntimeError("Launch this required trace check with TT_METAL_TRACE_ALLOC_TRACKING=1")

    source_root = Path(__file__).resolve().parents[1]
    sources = sorted((source_root / "tt").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("hf_reference.py"),
        Path(__file__).with_name("runtime_audit.py"),
    ]
    source_before = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeTextRotaryEmbedding,
    )

    from ..tt.functional_decoder import FunctionalDecoder
    from .hf_reference import decode_reference, prefill_reference
    from .layer_weights import load_layer_state_dict
    from .runtime_audit import device_only

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    lengths = prompt_lengths(args.batch, args.seed)
    if max(lengths) + args.steps > cfg.max_position_embeddings:
        raise ValueError("Requested positions exceed the target advertised context")
    slots = tuple(reversed(range(32)))[: args.batch]
    state = load_layer_state_dict(args.checkpoint, args.layer)
    prefix = f"model.language_model.layers.{args.layer}."
    with torch.device("meta"):
        reference = Qwen3_5MoeDecoderLayer(cfg, args.layer)
    reference.load_state_dict(
        {key.removeprefix(prefix): value for key, value in state.items()}, assign=True, strict=True
    )
    reference.eval()
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)

    generator = torch.Generator().manual_seed(args.seed)
    recorded = None
    if args.input_activations:
        recorded = torch.load(args.input_activations, weights_only=True)
        assert recorded["metadata"]["layer"] == args.layer
        recorded = recorded["activations"][0]

    def window(row, width):
        start = (row * 131 + args.window_offset) % (recorded.shape[0] - width + 1)
        return recorded[None, start : start + width].contiguous()

    cases = [
        dict(
            row=row,
            slot=slots[row],
            length=length,
            x=(window(row, length + args.steps) if recorded is not None else
               torch.randn(1, length + args.steps, cfg.hidden_size, dtype=torch.bfloat16, generator=generator) * 0.1),
        )
        for row, length in enumerate(lengths)
    ]
    for case in cases:
        length, x = case["length"], case["x"]
        expected_prefill, cache, _ = prefill_reference(reference, cfg, rope, x[:, :length], args.layer)
        expected_decode = decode_reference(reference, rope, cache, x[:, length:], length)
        # Each reference cache is separate; no padded shared-batch reference can
        # accidentally assign one row's length to another row.
        case.update(expected_prefill=expected_prefill, expected_decode=expected_decode, hf_cache=cache)
    decode_inputs = [
        torch.cat([case["x"][:, case["length"] + step : case["length"] + step + 1] for case in cases], dim=0)
        for step in range(args.steps)
    ]
    position_inputs = [
        torch.tensor([length + step for length in lengths], dtype=torch.int32) for step in range(args.steps)
    ]
    pages_per_row = max(32, (max(lengths) + args.steps + args.page_size - 1) // args.page_size)
    owned_pages = args.batch * pages_per_row
    page_generator = torch.Generator().manual_seed(args.seed + 1)
    table = torch.randperm(owned_pages, dtype=torch.int32, generator=page_generator).reshape(args.batch, pages_per_row)
    table_hash = hashlib.sha256(table.numpy().tobytes()).hexdigest()
    print("HETEROGENEOUS_REFERENCES_READY", json.dumps({"lengths": lengths, "slots": slots}), flush=True)

    from ttnn.tools import trace_allocation_tracker

    import ttnn

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise RuntimeError("TTNN reports trace allocation tracking disabled despite the required environment")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, **(
        {"physical_device_ids": [int(__import__("os").environ["QDEV"])]} if "QDEV" in __import__("os").environ else {}))

    def put(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, *, device=True):
        placement = (
            dict(device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if device
            else {}
        )
        return ttnn.from_torch(tensor.contiguous(), dtype=dtype, layout=layout, **placement)

    def get(tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).clone()

    result = {
        "layer_idx": args.layer,
        "layer_type": cfg.layer_types[args.layer],
        "weights": "real",
        "batch": args.batch,
        "decode_steps": args.steps,
        "prompt_lengths": lengths,
        "request_slots": slots,
        "seed": args.seed,
        "input_activations": str(args.input_activations) if args.input_activations else "random_normal_0.1",
        "window_offset": args.window_offset,
        "expert_precision": args.expert_precision,
        "page_size": args.page_size,
        "initial_positions": lengths,
        "position_increment_per_step": 1,
        "position_tensor_consumed": args.layer == 3,
        "heterogeneous_lengths": len(set(lengths)) > 1,
        "runtime_audit": True,
        "trace_allocation_tracking": True,
        "traced": True,
        "performance_measurement": False,
        "captures": 0,
        "replays": 0,
        "teardown_releases": 0,
        "threshold": 0.995,
        "source_sha256": source_before,
        "prefill_rows": [],
        "decode_rows": [],
        "page_ownership": (
            {
                "owned_pages": owned_pages,
                "pages_per_row": pages_per_row,
                "physical_pages_unique": int(table.unique().numel()) == owned_pages,
                "page_table_sha256": table_hash,
                "guard_pages": 3,
                "guard_key_sentinel": 13.0,
                "guard_value_sentinel": -7.0,
            }
            if args.layer == 3
            else None
        ),
    }
    trace_id, trace_ended = None, False
    try:
        decoder_class, fusion_kwargs = FunctionalDecoder, {}
        if args.decoder in ("fused", "optimized"):
            from ..tt.fused_decoder import FusedDecoder

            decoder_class = FusedDecoder
            if args.fusions is not None:
                fusion_kwargs["fusions"] = tuple(filter(None, args.fusions.split(",")))
        if args.decoder == "optimized":
            from ..tt.optimized_decoder import OptimizedDecoder

            decoder_class = OptimizedDecoder
            if args.optimization_json is not None:
                fusion_kwargs["optimization"] = json.loads(args.optimization_json)
        model = decoder_class.from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            max_batch_size=32,
            expert_precision=args.expert_precision,
            **fusion_kwargs,
        )
        result.update(decoder_class=type(model).__name__, fusions=getattr(model, "fusions", None))
        result["optimization"] = __import__("dataclasses").asdict(model.optimization) if args.decoder=="optimized" else None
        # These stable buffers and every host update tensor are constructed
        # before the first capture. Replay only refreshes their contents.
        dx = put(decode_inputs[0])
        dx_host = [put(value, device=False) for value in decode_inputs]
        pos_host = [put(value, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=False) for value in position_inputs]
        decode_kwargs = dict(request_slots=slots)
        row_tables = None
        if args.layer == 3:
            cache_shape = (owned_pages + 4, cfg.num_key_value_heads, args.page_size, cfg.head_dim)
            key_storage = torch.zeros(cache_shape, dtype=torch.bfloat16)
            value_storage = torch.zeros_like(key_storage)
            key_storage[owned_pages:].fill_(13.0)
            value_storage[owned_pages:].fill_(-7.0)
            kv_dtype = (ttnn.bfloat8_b if getattr(getattr(model, "optimization", None), "kv_dtype", "bf16") == "bfp8"
                        else ttnn.bfloat16)
            key_cache, value_cache = put(key_storage, kv_dtype), put(value_storage, kv_dtype)
            if args.decoder == "optimized" and not args.no_kv_scratch:
                model.bind_paged_kv_cache(key_cache, value_cache, scratch_block=owned_pages + 3)
                result["kv_scratch_block"] = owned_pages + 3
            else:
                model.bind_paged_kv_cache(key_cache, value_cache)
            pt = put(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            pos = put(position_inputs[0], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            row_tables = [put(table[row : row + 1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for row in range(args.batch)]
            decode_kwargs.update(page_table=pt, position_tensor=pos)
        for case in cases:
            case["prefill_input"] = put(case["x"][:, : case["length"]])

        model.reset_request_state(slots)
        for case in cases:
            kwargs = dict(request_slots=(case["slot"],))
            if row_tables is not None:
                kwargs["page_table"] = row_tables[case["row"]]
            with device_only():
                output = model.prefill_forward(case["prefill_input"], **kwargs)
            value = pcc(case["expected_prefill"], get(output))
            result["prefill_rows"].append(
                {"row": case["row"], "slot": case["slot"], "length": case["length"], "pcc": value}
            )
            output.deallocate(True)
        result["prefill_pcc_min"] = min(row["pcc"] for row in result["prefill_rows"])
        print("HETEROGENEOUS_PREFILL_DONE", result["prefill_pcc_min"], flush=True)

        saved = (
            [(tensor, ttnn.clone(tensor)) for slot in slots for tensor in model._states[slot]]
            if args.layer == 0
            else []
        )

        def restore_state():
            for tensor, snapshot in saved:
                ttnn.copy(snapshot, tensor)

        with device_only():
            warmed = model.decode_forward(dx, **decode_kwargs)
        warmed_host = get(warmed)
        result["warm_decode_pcc_min"] = min(
            pcc(case["expected_decode"][:, :1], warmed_host[case["row"] : case["row"] + 1]) for case in cases
        )
        warmed.deallocate(True)
        # Compile copy-back too, before forbidding all cache misses. Full
        # attention needs no state restore: each probe rewrites the identical
        # first decode position, and causal reads exclude all future slots.
        restore_state()
        ttnn.synchronize_device(mesh)
        mesh.set_program_cache_misses_allowed(False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            trace_out = model.decode_forward(dx, **decode_kwargs)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        trace_ended = True
        result.update(captures=1, trace_id=str(trace_id), program_cache_misses_forbidden=True)
        # Only the final output allocation is acknowledged. It remains alive
        # for this trace's whole lifetime and replay intentionally overwrites it.
        trace_allocation_tracker.acknowledge_corruptible(trace_out)
        result["allocation_acknowledgments"] = ["exact final trace output only"]

        restore_state()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        result["replays"] += 1
        identical_first = get(trace_out)
        restore_state()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        result["replays"] += 1
        repeated = get(trace_out)
        result["repeated_identical_by_row"] = [
            bool(torch.equal(identical_first[row], repeated[row])) for row in range(args.batch)
        ]
        result["repeated_identical"] = all(result["repeated_identical_by_row"])
        result["warm_vs_replay_identical"] = bool(torch.equal(warmed_host, identical_first))
        restore_state()

        actual_decode = []
        for step in range(args.steps):
            ttnn.copy_host_to_device_tensor(dx_host[step], dx)
            if args.layer == 3:
                ttnn.copy_host_to_device_tensor(pos_host[step], pos)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            result["replays"] += 1
            actual_decode.append(get(trace_out))
        actual_decode = torch.cat(actual_decode, dim=1)
        for case in cases:
            row = case["row"]
            values = [
                pcc(case["expected_decode"][:, step : step + 1], actual_decode[row : row + 1, step : step + 1])
                for step in range(args.steps)
            ]
            result["decode_rows"].append(
                {
                    "row": row,
                    "slot": case["slot"],
                    "initial_position": case["length"],
                    "pcc_by_token": values,
                    "pcc_min": min(values),
                }
            )
        result["decode_pcc_min"] = min(row["pcc_min"] for row in result["decode_rows"])

        # All unknown eager paths, including guard-page slicing, occur after
        # release. No program-cache allocation is exempted while a trace lives.
        ttnn.release_trace(mesh, trace_id)
        trace_id, trace_ended = None, False
        result["teardown_releases"] = 1
        mesh.set_program_cache_misses_allowed(True)
        result["guards_clean"] = None
        if args.layer == 3:
            guard_keys = get(key_cache[owned_pages : owned_pages + 3])
            guard_values = get(value_cache[owned_pages : owned_pages + 3])
            result["guard_key_changed_elements"] = int(torch.count_nonzero(guard_keys != 13.0))
            result["guard_value_changed_elements"] = int(torch.count_nonzero(guard_values != -7.0))
            result["guards_clean"] = not (
                result["guard_key_changed_elements"] or result["guard_value_changed_elements"]
            )
        source_after = {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        }
        result["source_changed_during_run"] = source_before != source_after
        result["status"] = (
            "pass"
            if (
                result["prefill_pcc_min"] >= 0.995
                and result["decode_pcc_min"] >= 0.995
                and result["repeated_identical"]
                and result["warm_vs_replay_identical"]
                and result["guards_clean"] is not False
                and not result["source_changed_during_run"]
            )
            else "failed_gate"
        )
    except Exception as error:
        result["status"] = "execution_error"
        result["error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if trace_id is not None and trace_ended:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)

    print(
        "HETEROGENEOUS_POSITIONS",
        json.dumps(
            {
                key: result[key]
                for key in ("status", "prefill_pcc_min", "decode_pcc_min", "repeated_identical", "guards_clean")
            }
        ),
        flush=True,
    )
    assert result["status"] == "pass", f"Heterogeneous-position gate failed; see {args.output}"


if __name__ == "__main__":
    main()
