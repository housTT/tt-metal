# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Selected BFP4/LoFi common-head geometry probe; --plan has no TTNN imports."""

import argparse
import hashlib
import json
import math
import statistics
import time
import traceback
from dataclasses import replace
from pathlib import Path

DOC = Path(__file__).resolve().parent
LOGICAL_COLUMNS = 32768
RESIDENT_BYTES = 221952
NORM_BYTES = 8192
PHYSICAL_L1_BYTES = 1572864
STATIC_BASE_BYTES = 111616
WEIGHT_TILE_BYTES = 576
POLICY = DOC / "selected_precision_config.json"


def selected_policy(path):
    policy = json.loads(path.read_text())
    # Lock the material head contract while retaining the complete selected
    # policy in the receipt. Changes to the selected artifact cannot silently
    # turn this into another precision experiment.
    assert policy["weight_groups"]["lm_head"] == "bfloat4_b", policy
    assert policy["compute_fidelities"]["lm_head"] == "LoFi", policy
    for field in ("activation_dtype", "logits_dtype"):
        assert policy[field] == "bfloat16", policy
    assert policy["matmul_flags"]["head_fp32_dest_acc_en"] is True, policy
    assert policy["matmul_flags"]["packer_l1_acc"] is True, policy
    assert policy["matmul_flags"]["math_approx_mode"] is False, policy
    return policy


def geometry(cores, block, readers):
    assert cores in (16, 32, 64)
    assert readers in (1, 2, 3)
    assert block > 0 and (128 // cores) % block == 0
    physical = math.ceil(LOGICAL_COLUMNS / (32 * 8 * readers)) * (32 * 8 * readers)
    reader_tiles = physical // (32 * 8 * readers)
    # Factory's FP32-dest subblock optimization yields 128,64,44 for this scope.
    compute_tiles = math.ceil(reader_tiles / 4) * 4
    cb = dict(
        in0=2 * block * 2048,
        in1=3 * reader_tiles * block * WEIGHT_TILE_BYTES,
        output=compute_tiles * 2048,
        intermediate=compute_tiles * 4096,
    )
    return dict(
        cores=cores,
        grid=[8, cores // 8],
        input_shard=[32, 4096 // cores],
        in0_block_w=block,
        readers=readers,
        logical_columns=LOGICAL_COLUMNS,
        physical_columns=physical,
        weight_shard=[4096, physical // 8],
        reader_tiles=reader_tiles,
        compute_tiles=compute_tiles,
        per_core_N=math.ceil(physical / (32 * cores)),
        cb_bytes=cb,
        weight_dtype="bfloat4_b",
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        intermediate_dtype="float32",
        aligned_weight_tile_bytes=WEIGHT_TILE_BYTES,
        predicted_static_end=STATIC_BASE_BYTES + sum(cb.values()),
        resident_bytes_per_bank=RESIDENT_BYTES,
        normalized_bytes_per_bank=NORM_BYTES,
        optimistic_live_frontier=PHYSICAL_L1_BYTES - RESIDENT_BYTES - NORM_BYTES,
        source_feasible=STATIC_BASE_BYTES + sum(cb.values()) <= PHYSICAL_L1_BYTES - RESIDENT_BYTES - NORM_BYTES,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--in0-block-w", type=int, required=True)
    parser.add_argument("--readers", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision-config", type=Path, default=POLICY)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--serial-traces", action="store_true", help="Release each trace before capturing the next")
    args = parser.parse_args()
    policy = selected_policy(args.precision_config)
    candidate_geometry = geometry(args.cores, args.in0_block_w, args.readers)
    if args.plan:
        print(json.dumps(dict(candidate=candidate_geometry, baseline=geometry(64, 1, 2), precision=policy), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for a device run")
    if not candidate_geometry["source_feasible"]:
        parser.error("source-refuted: static CBs exceed physical L1 minus persistent residency and frozen final norm")
    assert args.repeats > 0 and args.rounds > 0

    import torch
    from safetensors import safe_open

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.reference.hf_reference import CHECKPOINT_TEXT_PREFIX
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.lm_head.lm_head_1d import LMHead1D

    report = dict(
        candidate=candidate_geometry,
        baseline=geometry(64, 1, 2),
        precision=policy,
        precision_config_path=str(args.precision_config.resolve()),
        precision_config_sha256=hashlib.sha256(args.precision_config.read_bytes()).hexdigest(),
        hardware="TP4, four Blackhole chips on P300c boards",
        normalization="Fixed 8x4 sharded final norm and padding for both geometries",
        serial_traces=args.serial_traces,
        paired_trials=[],
        probe_execution_pass=False,
        source_hashes={
            str(path.relative_to(DOC.parents[4])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                DOC.parents[1] / "tt/model.py",
                DOC.parents[1] / "tt/precision.py",
                DOC.parents[4] / "models/common/modules/lm_head/lm_head_1d.py",
                DOC.parents[4]
                / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp",
                DOC.parents[4] / "ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp",
                DOC.parents[4] / "ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp",
                DOC.parents[4] / "tt_metal/impl/data_format/tile.cpp",
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def memory_view(mesh):
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        return {
            name: int(getattr(view, name))
            for name in (
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
                "largest_contiguous_bytes_free_per_bank",
            )
        }

    def describe(tensor):
        return dict(
            shape=list(tensor.shape),
            padded_shape=list(tensor.padded_shape),
            dtype=str(tensor.dtype),
            memory=str(tensor.memory_config()),
            address=int(tensor.buffer_address()),
        )

    def metrics(actual, expected):
        error = (actual - expected).abs()
        boundaries = sorted(
            set(
                max(0, min(actual.shape[-1] - 1, b + offset))
                for b in range(LOGICAL_COLUMNS, actual.shape[-1], LOGICAL_COLUMNS)
                for offset in (-1, 0, 1)
            )
        )
        values, ids = actual[0].topk(10)
        return dict(
            finite=bool(torch.isfinite(actual).all()),
            exact_baseline=bool(torch.equal(actual, expected)),
            max_abs=float(error.max()),
            mean_abs=float(error.mean()),
            boundary_max_abs=float(error[:, boundaries].max()),
            pcc=float(torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1]),
            top1_equal=bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
            baseline_top1_in_top5=bool((actual.topk(5).indices == expected.argmax(-1)[:, None]).any(-1).all()),
            top10_ids=ids.tolist(),
            top10_logits=values.tolist(),
            scores_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
        )

    mesh = None
    traces = []
    try:
        report["phase"] = "open_mesh"
        save_report()
        mesh = open_ornith_mesh()
        model = OrnithModel(None, mesh, layer_indices=[], max_context=2048, precision_config=policy)
        assert model.head_columns == LOGICAL_COLUMNS and len(model.head_weights) == 2
        assert all(w.dtype == ttnn.bfloat4_b for w in model.head_weights)
        assert model.lm_head.config.lm_head_dtype == ttnn.bfloat16
        report["model_precision"] = model.precision
        report["model_path"] = str(model.model_path)
        # Freeze the geometry baseline explicitly; model defaults may later
        # select another geometry while these evidence probes remain useful.
        norm_memory = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(x=8, y=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        baseline_memory = ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        baseline_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=1,
            per_core_N=LOGICAL_COLUMNS // (32 * 64),
            num_workers_per_dram_bank=2,
            fused_activation=None,
        )
        baseline_head = LMHead1D.from_config(
            replace(
                model.lm_head.config,
                input_memcfg=baseline_memory,
                program_configs=[baseline_program] * 2,
            )
        )
        norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4), block_h=1, block_w=4, subblock_w=4, inplace=False
        )
        source = DOC.parent / "full_model/french_head_v1/hidden.pt"
        saved = torch.load(source, weights_only=True)
        hidden = saved["hidden"][0]
        assert tuple(hidden.shape) == (1, 1, 4096)
        assert all(torch.equal(hidden, other) for other in saved["hidden"])
        report["hidden_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        report["hidden_source"] = str(source)
        x = model.upload(hidden)
        report["hidden_tensor"] = describe(x)
        assert x.dtype == ttnn.bfloat16

        # Match the previous full-stack French probe, including all 24 recurrent
        # states and constants. Reserve before any terminal activation buffers.
        before = memory_view(mesh)
        extra = RESIDENT_BYTES - before["total_bytes_allocated_per_bank"]
        assert extra > 0 and extra % 64 == 0, extra
        grid = mesh.compute_with_storage_grid_size()
        reserve_memory = ttnn.create_sharded_memory_config(
            shape=(1, extra // 2),
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        resident = model.upload(
            torch.zeros(grid.x * grid.y, extra // 2, dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory=reserve_memory,
        )
        report["resident_allocation"] = describe(resident)
        report["resident_memory"] = memory_view(mesh)
        assert report["resident_memory"]["total_bytes_allocated_per_bank"] == RESIDENT_BYTES

        candidate_memory = ttnn.create_sharded_memory_config(
            shape=tuple(candidate_geometry["input_shard"]),
            core_grid=ttnn.CoreGrid(x=8, y=args.cores // 8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=args.in0_block_w,
            per_core_M=1,
            per_core_N=candidate_geometry["per_core_N"],
            num_workers_per_dram_bank=args.readers,
            fused_activation=None,
        )
        weights = model.head_weights
        physical = candidate_geometry["physical_columns"]
        if physical != LOGICAL_COLUMNS:
            report["phase"] = "pad_weights"
            save_report()
            dram = mesh.dram_grid_size()
            assert dram.x * dram.y == 8
            dram_grid = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))]
            )
            weight_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, [4096, physical // 8], ttnn.ShardOrientation.ROW_MAJOR),
            )
            index = json.loads((model.model_path / "model.safetensors.index.json").read_text())["weight_map"]
            key = (
                CHECKPOINT_TEXT_PREFIX + "embed_tokens.weight"
                if model.hf_config.tie_word_embeddings
                else "lm_head.weight"
            )
            weights = []
            with safe_open(str(model.model_path / index[key]), framework="pt", device="cpu") as handle:
                source_weight = handle.get_slice(key)
                for start in (0, LOGICAL_COLUMNS):
                    packed = torch.zeros(4096, physical * 4, dtype=torch.bfloat16)
                    for rank in range(4):
                        first = rank * 65536 + start
                        last = min(first + LOGICAL_COLUMNS, model.vocab_size)
                        if last > first:
                            packed[:, rank * physical : rank * physical + last - first] = source_weight[first:last, :].T
                    weights.append(model.upload(packed, dtype=ttnn.bfloat4_b, shard_dim=1, memory=weight_memory))
                    del packed

        candidate = LMHead1D.from_config(
            replace(
                baseline_head.config,
                output_weights=[LazyWeight(source=w, _value=w, dtype=ttnn.bfloat4_b) for w in weights],
                program_configs=[pc] * 2,
                input_memcfg=candidate_memory,
                weights_memcfgs=[w.memory_config() for w in weights],
            )
        )
        report["candidate_program"] = str(pc)
        report["compute"] = str(model.head_compute)
        report["weights"] = [describe(w) for w in weights]
        report["baseline_weights"] = [describe(w) for w in model.head_weights]
        report["baseline_program"] = str(baseline_program)
        assert all(w.dtype == ttnn.bfloat4_b for w in weights)

        def normalize():
            flat = ttnn.reshape(x, [1, 1, 1, 4096])
            sharded = ttnn.to_memory_config(flat, norm_memory)
            padded = ttnn.pad(
                sharded,
                [(0, 0), (0, 0), (0, 31), (0, 0)],
                0.0,
                memory_config=norm_memory,
            )
            norm = ttnn.rms_norm(
                padded,
                weight=model.norm_weight,
                epsilon=model.hf_config.rms_norm_eps,
                program_config=norm_program,
                memory_config=norm_memory,
            )
            assert norm.dtype == ttnn.bfloat16
            return norm, padded

        def project(normalized, is_candidate):
            mem = candidate_memory if is_candidate else baseline_memory
            head = candidate if is_candidate else baseline_head
            sharded = ttnn.to_memory_config(normalized, mem)
            output = head(sharded)
            if (args.cores if is_candidate else 64) != 32:
                ttnn.deallocate(sharded)
            if is_candidate and physical != LOGICAL_COLUMNS:
                chunks = [
                    ttnn.slice(
                        output,
                        [0, 0, 0, offset],
                        [1, 1, 32, offset + LOGICAL_COLUMNS],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for offset in (0, physical)
                ]
                trimmed = ttnn.concat(chunks, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(output)
                for chunk in chunks:
                    ttnn.deallocate(chunk)
                output = trimmed
            assert tuple(output.shape) == (1, 1, 32, 65536), output.shape
            assert output.dtype == ttnn.bfloat16
            return output

        baseline_terminal_scores = None
        for mode in ("head", "terminal"):
            normalized = None
            if mode == "head":
                normalized, padded = normalize()
                ttnn.deallocate(padded)
            if normalized is not None:
                report["normalized"] = describe(normalized)

            def forward(is_candidate):
                if mode == "head":
                    norm, padded = normalized, None
                else:
                    norm, padded = normalize()
                output = project(norm, is_candidate)
                if mode == "terminal":
                    ttnn.deallocate(norm)
                    ttnn.deallocate(padded)
                return output

            scores_by_name = {}
            output_by_name = {}
            trace_by_name = {}
            report[mode] = {}
            for name, is_candidate in (("baseline", False), ("candidate", True)):
                report["phase"] = f"{mode}_{name}_eager"
                report[mode][name] = {"memory_before": memory_view(mesh)}
                save_report()
                eager = forward(is_candidate)
                scores = model.logits_to_host(eager, 1)
                scores_by_name[name] = scores
                expected = scores_by_name["baseline"]
                result = metrics(scores, expected)
                assert result["finite"]
                result["output_tensor"] = describe(eager)
                repeat = forward(is_candidate)
                result["eager_exact_repeat"] = bool(torch.equal(scores, model.logits_to_host(repeat, 1)))
                assert result["eager_exact_repeat"]
                ttnn.deallocate(eager)
                ttnn.deallocate(repeat)
                ttnn.synchronize_device(mesh)
                report[mode][name].update(result)
                report["phase"] = f"{mode}_{name}_capture"
                save_report()
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = forward(is_candidate)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
                trace_by_name[name] = trace
                output_by_name[name] = traced
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                result["trace_exact_eager"] = bool(torch.equal(scores, model.logits_to_host(traced, 1)))
                assert result["trace_exact_eager"]
                report[mode][name].update(result)
                print("GEOMETRY_CHECK", mode, name, json.dumps(result), flush=True)

                if args.serial_traces:
                    # The allocation tracker must never see later candidate
                    # allocations when replaying the earlier baseline trace.
                    # This is a lifecycle check, not the alternating benchmark.
                    for trial in range(args.rounds):
                        start = time.perf_counter()
                        for _ in range(args.repeats):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        ms = (time.perf_counter() - start) * 1000 / args.repeats
                        report["paired_trials"].append(
                            dict(mode=mode, name=name, trial=trial, trace_ms=ms, replays=args.repeats)
                        )
                        assert torch.equal(scores, model.logits_to_host(traced, 1))
                    ttnn.release_trace(mesh, trace)
                    traces.remove(trace)
                    ttnn.deallocate(traced)

            report["phase"] = f"{mode}_timing"
            save_report()
            for trial in range(0 if args.serial_traces else args.rounds):
                order = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                for name in order:
                    ttnn.execute_trace(mesh, trace_by_name[name], cq_id=0, blocking=True)
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace_by_name[name], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    ms = (time.perf_counter() - start) * 1000 / args.repeats
                    item = dict(mode=mode, name=name, trial=trial, trace_ms=ms, replays=args.repeats)
                    report["paired_trials"].append(item)
                    print("GEOMETRY_TIMING", json.dumps(item), flush=True)
                    assert torch.equal(scores_by_name[name], model.logits_to_host(output_by_name[name], 1))
            for name in ("baseline", "candidate"):
                report[mode][name]["median_trace_ms"] = statistics.median(
                    t["trace_ms"] for t in report["paired_trials"] if t["mode"] == mode and t["name"] == name
                )
                if not args.serial_traces:
                    ttnn.release_trace(mesh, trace_by_name[name])
                    traces.remove(trace_by_name[name])
                    ttnn.deallocate(output_by_name[name])
            if normalized is not None:
                ttnn.deallocate(normalized)
                baseline_terminal_scores = scores_by_name["baseline"]
                candidate_head_scores = scores_by_name["candidate"]
            else:
                assert torch.equal(baseline_terminal_scores, scores_by_name["baseline"])
                assert torch.equal(candidate_head_scores, scores_by_name["candidate"])
            save_report()
        report["phase"] = "complete"
        report["probe_execution_pass"] = True
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if mesh is not None:
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            close_ornith_mesh(mesh)
        save_report()
    print("HEAD_GEOMETRY_OK", args.output, flush=True)


if __name__ == "__main__":
    main()
