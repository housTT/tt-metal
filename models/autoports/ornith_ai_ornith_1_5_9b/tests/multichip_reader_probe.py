# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real TP4 DRAM-reader regression and alternating traced timing windows."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.common.tensor_utils import compute_kernel_config_to_dict

from ..tt.multichip_decoder import TP, MultichipDecoder, _projection_weights, fabric_router_config, partition_state_dict
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations

ALIASES = {"gdn_z": "gdn_z_epilogue", "gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
ROW_ROLES = {"gdn_out", "o_proj", "down_proj"}


def role_value(value, role, default):
    """Accept a scalar or a JSON mapping of role names to values."""
    if isinstance(value, dict):
        return value.get(role, default)
    return default if value is None else value


def read_ranks(value):
    return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(value)]


def compare(reference, actual):
    assert len(reference) == len(actual) == TP, "expected four local projections"
    assert all(torch.isfinite(part).all() for part in actual), "non-finite projection output"
    values = [H.pcc(a, b) for a, b in zip(reference, actual)]
    assert min(values) >= H.PCC_BAR, f"all-rank PCC gate failed: {values}"
    return values


def shard_record(tensor):
    spec = tensor.memory_config().shard_spec
    return {
        "logical_shape": list(tensor.shape),
        "padded_shape": list(tensor.padded_shape),
        "dtype": str(tensor.dtype),
        "memory_config": str(tensor.memory_config()),
        "shard_shape": list(spec.shape),
        "storage_cores": spec.grid.num_cores(),
    }


def runtime_hashes():
    repo = Path(__file__).resolve().parents[4]
    paths = [
        "build/lib/_ttnncpp.so",
        "build/lib/libtt_metal.so",
        "ttnn/ttnn/_ttnn.so",
        "ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp",
        "ttnn/cpp/ttnn/operations/matmul/device/factory/"
        "matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp",
        "ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp",
    ]
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["default", "packed_gdn_all_modes"], default="default")
    parser.add_argument("--roles", nargs="+")
    parser.add_argument("--readers", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--cores", type=json.loads, help="integer or JSON role-to-core-count mapping")
    parser.add_argument("--block", type=json.loads, help="integer or JSON role-to-K-block mapping")
    parser.add_argument(
        "--per-core-n", type=json.loads, help="integer or JSON role-to-output-shard-width mapping, in tiles"
    )
    parser.add_argument("--windows", type=int, default=7, help="includes two warmup windows per reader")
    parser.add_argument("--iterations", type=int, default=64, help="trace replays in each timing window")
    parser.add_argument("--profile", action="store_true", help="emit separate warmed four-replay reader signposts")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite evidence: {args.output}")
    if args.windows < 3 or args.iterations < 1:
        parser.error("need at least three windows and one iteration")
    readers = sorted(set([1, *args.readers]))
    document = {
        "status": "running",
        "hardware": "four Blackhole chips on physical P300c boards",
        "mesh": [1, TP],
        "fabric": "FABRIC_1D_RING",
        "layer": args.layer,
        "variant": args.variant,
        "position": 2048,
        "logical_batch": 1,
        "padded_rows": 32,
        "requested_readers": args.readers,
        "effective_readers": readers,
        "pcc_threshold": H.PCC_BAR,
        "scope": "native local projection only; input sharding precedes timing; no row all-reduce",
        "activation_source": "recorded real layer input, then actual TP4 prefill and decode at position 2048",
        "weight_source": "raw pinned HF weights, partitioned before independent target-dtype uploads",
        "runtime_sha256": runtime_hashes(),
        "results": [],
        "window_order": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(document, indent=2) + "\n")

    save()
    mesh = None
    active_record = None
    try:
        torch.set_num_threads(8)
        state = H.layer_state_dict(args.layer, "real")
        decoder_type = MultichipDecoder
        if args.variant == "packed_gdn_all_modes":
            from .optimized_multichip_candidates import PackedGDNAllModes

            decoder_type = PackedGDNAllModes
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, TP), trace_region_size=32 * 1024**2, l1_small_size=24576)
        decoder = decoder_type.from_state_dict(
            state,
            hf_config=H.hf_config(),
            layer_idx=args.layer,
            mesh_device=mesh,
            max_context=4096,
        )
        decoder.allocate_state(1)
        decoder.allocate_kv_cache(64)
        table = H.to_device(
            mesh, torch.arange(64, dtype=torch.int32)[None], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        source = recorded_activations(args.layer)
        prefill_input = H.to_device(mesh, source[:, :2048])
        ttnn.deallocate(decoder.prefill_forward(prefill_input, page_table=table))
        ttnn.deallocate(prefill_input)
        captured = {}
        original = decoder._linear

        def collect(x, role, **kwargs):
            saved = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            if x.memory_config() == ttnn.DRAM_MEMORY_CONFIG:
                saved = ttnn.clone(saved)
            captured[role] = saved, kwargs.copy()
            return original(x, role, **kwargs)

        decoder._linear = collect
        token = H.to_device(mesh, source[:, 2048:2049])
        positions, rotations = H.decode_inputs(mesh, torch.tensor([2048]))
        try:
            ttnn.deallocate(decoder.decode_forward(token, current_pos=positions, rot_idxs=rotations, page_table=table))
        finally:
            decoder._linear = original
        ttnn.deallocate(token)
        local_hf_config = copy.deepcopy(decoder.global_hf_config)
        local_hf_config.num_attention_heads //= TP
        packed = [
            _projection_weights(partition_state_dict(state, decoder.global_hf_config, rank), local_hf_config)
            for rank in range(TP)
        ]
        if "gdn_all" in captured:
            for weights in packed:
                weights["gdn_all"] = torch.cat([weights["gdn_packed"], weights["gdn_z_epilogue"]], dim=1)
        if "gate_up" in captured:
            for weights in packed:
                weights["gate_up"] = torch.cat([weights["gate_proj"], weights["up_proj"]], dim=1)
        roles = [ALIASES.get(role, role) for role in args.roles] if args.roles else list(captured)
        roles = list(dict.fromkeys(roles))
        for role in roles:
            if role not in captured or role not in packed[0]:
                raise ValueError(f"role {role!r} unavailable in layer {args.layer}; captured: {list(captured)}")
        dram = mesh.dram_grid_size()
        banks = dram.x * dram.y
        dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])

        for role in roles:
            saved, kwargs = captured[role]
            parts = [weights[role] for weights in packed]
            k, n = parts[0].shape
            assert list(saved.shape) == [1, 1, k], f"unexpected logical input shape for {role}: {saved.shape}"
            defaults = decoder._role_config(role)
            cores = role_value(args.cores, role, defaults[0])
            block = role_value(args.block, role, defaults[1])
            if not isinstance(cores, int) or not isinstance(block, int) or cores < 1 or block < 1:
                raise ValueError(f"invalid cores/block for {role}: {cores}/{block}")
            if k % (32 * cores) or (k // (32 * cores)) % block:
                raise ValueError(f"illegal exact K sharding for {role}: K={k}, cores={cores}, block={block}")
            per_core_n = role_value(args.per_core_n, role, math.ceil(n / (32 * cores)))
            if not isinstance(per_core_n, int) or per_core_n < 1:
                raise ValueError(f"invalid output shard width for {role}: {per_core_n}")
            input_memory = decoder._width_memory(k, cores)
            working = ttnn.to_memory_config(saved, input_memory)
            weight_dtype = decoder.decode_weights.get(role, decoder.w[role]).dtype
            output_dtype = kwargs.get("dtype", ttnn.bfloat16)
            activation = kwargs.get("activation")
            if activation not in (None, "silu"):
                raise ValueError(f"unsupported captured fused activation: {activation}")
            compute = decoder.projection_compute[role]
            axis = 0 if role in ROW_ROLES else 1
            raw_mesh_weight = torch.cat(parts, dim=axis).contiguous()
            tile_bytes = ttnn.Tile([32, 32]).get_tile_size(weight_dtype)
            candidates = {}
            reader_one = None
            try:
                for reader in readers:
                    width = math.ceil(n / (32 * banks * reader)) * 32 * reader
                    active_banks = banks - (banks * (width // 32) - math.ceil(n / 32)) // (width // 32)
                    memory = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(dram_grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    record = {
                        "status": "constructing",
                        "layer": args.layer,
                        "role": role,
                        "local_shape": [1, k, n],
                        "readers": reader,
                        "cores": cores,
                        "block_w": block,
                        "per_core_M": 1,
                        "per_core_N": per_core_n,
                        "dram_banks": banks,
                        "active_dram_banks": active_banks,
                        "active_reader_cores": active_banks * reader,
                        "weight_shard_width": width,
                        "padded_dram_width": width * banks,
                        "padding_columns": width * banks - n,
                        "weight_tile_bytes": tile_bytes,
                        "per_bank_tile_row_bytes": width // 32 * tile_bytes,
                        "per_reader_tile_row_bytes": width // (32 * reader) * tile_bytes,
                        "physical_weight_bytes_per_chip": k // 32 * width // 32 * banks * tile_bytes,
                        "compute": compute_kernel_config_to_dict(compute),
                        "fused_activation": activation,
                        "input": shard_record(working),
                        "warmup_samples_us": [],
                        "samples_us": [],
                    }
                    if role == "gdn_all":
                        record["packing_fields"] = ["qkv", "a_padded", "b_padded", "z"]
                        record["packing_widths"] = [decoder.cfg.conv_dim, 32, 32, decoder.cfg.linear_v_dim]
                        assert sum(record["packing_widths"]) == n, "packed GDN field widths do not match raw weights"
                        assert activation is None, "packed GDN applies Z SiLU after projection, not to every field"
                    active_record = record
                    document["results"].append(record)
                    save()
                    weight = ttnn.from_torch(
                        raw_mesh_weight,
                        device=mesh,
                        dtype=weight_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=memory,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=axis),
                    )
                    candidate = {"weight": weight, "record": record, "trace": None, "output": None}
                    candidates[reader] = candidate
                    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block,
                        per_core_M=1,
                        per_core_N=record["per_core_N"],
                        num_workers_per_dram_bank=reader,
                        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None,
                    )
                    record["program_config"] = str(program)
                    record["program_reader_count"] = program.num_workers_per_dram_bank

                    def run(x, weight=weight, program=program):
                        return ttnn.linear(
                            x,
                            weight,
                            program_config=program,
                            compute_kernel_config=compute,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            dtype=output_dtype,
                        )

                    candidate["run"] = run
                    eager = run(working)
                    actual = read_ranks(eager)
                    record["output"] = shard_record(eager)
                    ttnn.deallocate(eager)
                    if reader == 1:
                        reader_one = actual
                    record["pcc_vs_reader1"] = compare(reader_one, actual)
                    record["weight"] = shard_record(weight)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    candidate["trace"] = trace
                    try:
                        candidate["output"] = run(working)
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    record["trace_exact"] = all(
                        torch.equal(a, b) for a, b in zip(actual, read_ranks(candidate["output"]))
                    )
                    assert record["trace_exact"], f"unchanged trace differs from eager: {role}, R{reader}"
                    candidate["eager_host"] = actual
                    record["status"] = "timing"
                    save()

                for window in range(args.windows):
                    order = readers if window % 2 == 0 else list(reversed(readers))
                    document["window_order"].append({"role": role, "window": window, "readers": order})
                    for reader in order:
                        candidate = candidates[reader]
                        active_record = candidate["record"]
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            ttnn.execute_trace(mesh, candidate["trace"], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        elapsed = (time.perf_counter() - start) * 1e6 / args.iterations
                        key = "warmup_samples_us" if window < 2 else "samples_us"
                        active_record[key].append(elapsed)
                        save()

                if args.profile:
                    from tracy import signpost

                    assert not os.environ.get("TT_METAL_WATCHER"), "Keep watcher and profiler separate"
                    ttnn.ReadDeviceProfiler(mesh)
                    for reader in readers:
                        signpost(f"PERF_READER_{role}_R{reader}")
                        start = time.perf_counter()
                        for _ in range(4):
                            ttnn.execute_trace(mesh, candidates[reader]["trace"], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        candidates[reader]["record"]["profile_host_us"] = (time.perf_counter() - start) * 1e6 / 4
                        signpost(f"PERF_READER_{role}_R{reader}_END")
                        ttnn.ReadDeviceProfiler(mesh)
                    save()

                # Rebind a cached program to a different live input allocation, then
                # change the captured input contents without changing its address.
                changed = ttnn.neg(working, memory_config=input_memory)
                changed_reference = None
                try:
                    assert any(
                        not torch.equal(a, b) for a, b in zip(read_ranks(working), read_ranks(changed))
                    ), "changed-input regression requires nonzero activations"
                    for reader in readers:
                        candidate = candidates[reader]
                        active_record = candidate["record"]
                        actual = read_ranks(candidate["output"])
                        assert all(torch.equal(a, b) for a, b in zip(actual, candidate["eager_host"]))
                        assert changed.buffer_address() != working.buffer_address(), "fresh input aliased trace input"
                        changed_eager = candidate["run"](changed)
                        try:
                            changed_host = read_ranks(changed_eager)
                            if reader == 1:
                                changed_reference = changed_host
                            active_record["fresh_input_pcc_vs_reader1"] = compare(changed_reference, changed_host)
                            active_record["fresh_output_address_distinct"] = (
                                changed_eager.buffer_address() != candidate["output"].buffer_address()
                            )
                            assert active_record["fresh_output_address_distinct"], "fresh output aliases traced output"
                        finally:
                            ttnn.deallocate(changed_eager)
                        ttnn.copy(changed, working)
                        ttnn.execute_trace(mesh, candidate["trace"], cq_id=0, blocking=True)
                        active_record["changed_input_trace_exact"] = all(
                            torch.equal(a, b) for a, b in zip(changed_host, read_ranks(candidate["output"]))
                        )
                        assert active_record[
                            "changed_input_trace_exact"
                        ], f"changed input trace differs: {role}, R{reader}"
                        active_record["trace_us"] = statistics.median(active_record["samples_us"])
                        active_record["status"] = "pass"
                        print("READER " + json.dumps(active_record), flush=True)
                        save()
                finally:
                    ttnn.deallocate(changed)
            finally:
                for candidate in candidates.values():
                    if candidate["trace"] is not None:
                        ttnn.release_trace(mesh, candidate["trace"])
                    if candidate["output"] is not None:
                        ttnn.deallocate(candidate["output"])
                    ttnn.deallocate(candidate["weight"])
                ttnn.deallocate(working)
        for saved, _ in captured.values():
            ttnn.deallocate(saved)
        document["status"] = "pass"
    except Exception as exc:
        document.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if active_record is not None:
            active_record.update(status="failed", error=document["error"])
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
