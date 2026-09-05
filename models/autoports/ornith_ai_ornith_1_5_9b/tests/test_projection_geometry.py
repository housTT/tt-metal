# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact real-input decode projection sweeps; whole-layer acceptance is separate."""

import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.tensor_utils import compute_kernel_config_to_dict

from . import test_functional_decoder as H
from .optimization_candidates import selected_candidate
from .test_optimization_experiments import recorded_activations

pytestmark = H.pytestmark


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_projection_geometry(mesh_device, monkeypatch, layer_idx):
    def drain_profile():
        if os.environ.get("ORNITH_READER_PROFILE") == "1" and os.environ.get("ORNITH_READER_DRAIN") == "1":
            ttnn.ReadDeviceProfiler(mesh_device)

    cls = selected_candidate()
    monkeypatch.setattr(H, "FunctionalDecoder", cls)
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
    source = recorded_activations(layer_idx)
    ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh_device, source[:, :2048]), page_table=table))
    captured = {}
    original = decoder._linear

    def collect(x, role, **kwargs):
        captured[role] = ttnn.to_torch(x).clone(), kwargs.copy()
        return original(x, role, **kwargs)

    decoder._linear = collect
    pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
    ttnn.deallocate(
        decoder.decode_forward(
            H.to_device(mesh_device, source[:, 2048:2049]), current_pos=pos, rot_idxs=rot, page_table=table
        )
    )
    decoder._linear = original
    drain_profile()
    dg = mesh_device.dram_grid_size()
    banks = dg.x * dg.y
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
    rows = []
    initial_config = decoder.optimization
    per_core_n_request = json.loads(os.environ.get("ORNITH_READER_PER_CORE_N", "0"))
    per_core_n_override = 0
    if per_core_n_request:
        program_factory = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig

        def reader_program(**kwargs):
            if per_core_n_override:
                kwargs["per_core_N"] = per_core_n_override
            return program_factory(**kwargs)

        monkeypatch.setattr(ttnn, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", reader_program)
    if "gate_proj" in captured and "gate_up" in decoder.w:
        captured["gate_up"] = captured["gate_proj"][0], {}
    if "qkvg" in captured:
        for name in ("separate_q", "separate_k", "separate_v", "separate_gate"):
            if name in decoder.w:
                captured[name] = captured["qkvg"][0], {}
    roles = os.environ.get("ORNITH_SWEEP_ROLES", ",".join(captured)).split(",")
    for role in roles:
        if role not in captured:
            continue
        per_core_n_override = (
            per_core_n_request.get(role, 0) if isinstance(per_core_n_request, dict) else per_core_n_request
        )
        host_input, extra = captured[role]
        x = H.to_device(
            mesh_device, host_input, dtype=ttnn.float32 if host_input.dtype == torch.float32 else ttnn.bfloat16
        )
        weight = ttnn.to_torch(decoder.w[role])
        k, n = weight.shape
        saved_decode_weight = decoder.decode_weights.pop(role, None)
        reference_tensor = original(x, role, **extra)
        reference = ttnn.to_torch(reference_tensor)
        ttnn.deallocate(reference_tensor)
        if saved_decode_weight is not None:
            decoder.decode_weights[role] = saved_decode_weight
        configurations = [
            (cores, block, reader)
            for cores in (8, 16, 32, 64, 4)
            for block in range(2, k // 32 // cores + 1)
            if (k // 32) % cores == 0 and (k // 32 // cores) % block == 0
            for reader in (1, 2, 3)
        ]
        if os.environ.get("ORNITH_SWEEP_EXTRA") == "1":
            configurations = [
                (cores, block, reader)
                for cores in (4, 8, 16, 32, 64, 12, 24, 48, 96)
                if k // 32 % cores == 0
                for block in range(1, k // 32 // cores + 1)
                if (cores in (12, 24, 48, 96) or block == 1) and (k // 32 // cores) % block == 0
                for reader in (1, 2, 3)
            ]
        if os.environ.get("ORNITH_READER_CONFIRM") == "1":
            role_config = initial_config.role_configs[role]
            configurations = [(role_config["cores"], role_config["block_w"], reader) for reader in (1, 2, 3, 3, 2, 1)]
        # Each trace is released before changing the next candidate's allocation.
        for candidate_index, (cores, block, reader) in enumerate(configurations):
            per_bank = math.ceil(n / (32 * banks * reader)) * reader
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, per_bank * 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            dw = ttnn.from_torch(
                weight.contiguous(),
                device=mesh_device,
                dtype=decoder.w[role].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            decoder.decode_weights[role] = dw
            decoder.optimization = replace(
                decoder.optimization, cores=cores, block_w=block, readers=reader, role_configs={}
            )
            record = {
                "layer": layer_idx,
                "role": role,
                "shape": [1, k, n],
                "dtype": str(dw.dtype),
                "compute": compute_kernel_config_to_dict(decoder.projection_compute[role]),
                "cores": cores,
                "input_shard_tiles": k // 32 // cores,
                "block_w": block,
                "readers": reader,
                "per_core_N": per_core_n_override or math.ceil(n / (32 * cores)),
                "per_bank_tiles": per_bank,
                "dram_banks": banks,
                "weight_tile_bytes": {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[dw.dtype],
                "per_reader_row_bytes": per_bank
                // reader
                * ({ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[dw.dtype]),
                "bandwidth_peak_GBs": 512,
                "activation_source": "real recorded layer input and on-device preceding ops",
            }
            trace = None
            try:
                out = original(x, role, **extra)
                actual = ttnn.to_torch(out)
                record["pcc_vs_interleaved"] = H.pcc(reference, actual)
                ttnn.deallocate(out)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                out = original(x, role, **extra)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                times = []
                profile_readers = os.environ.get("ORNITH_READER_PROFILE") == "1"
                window_replays = 4 if profile_readers else 64
                for repeat in range(7):
                    ttnn.synchronize_device(mesh_device)
                    start = time.perf_counter()
                    for _ in range(window_replays):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    if repeat >= 2:
                        times.append((time.perf_counter() - start) * 1e6 / window_replays)
                    drain_profile()
                if profile_readers:
                    from tracy import signpost

                    label = f"READER_L{layer_idx}_{role}_R{reader}_I{candidate_index}"
                    signpost(label)
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    signpost(label + "_END")
                    record["profile_signpost"] = label
                    drain_profile()
                record.update(
                    trace_us=statistics.median(times),
                    samples_us=times,
                    exact_eager_trace=torch.equal(actual, ttnn.to_torch(out)),
                )
                record["physical_GBs"] = (
                    math.ceil(k / 32)
                    * per_bank
                    * banks
                    * ({ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[dw.dtype])
                    / record["trace_us"]
                    / 1e3
                )
                record["logical_GBs"] = (
                    k
                    * n
                    * ({ttnn.bfloat4_b: 0.5, ttnn.bfloat8_b: 1, ttnn.bfloat16: 2}[dw.dtype])
                    / record["trace_us"]
                    / 1e3
                )
                record["status"] = (
                    "pass"
                    if record["exact_eager_trace"] and record["pcc_vs_interleaved"] >= H.PCC_BAR
                    else "correctness_failure"
                )
            except RuntimeError as exc:
                record.update(status="runtime_error", error=str(exc).split("backtrace:")[0])
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh_device, trace)
                    ttnn.deallocate(out)
            print("PROJECTION_GEOMETRY " + json.dumps(record), flush=True)
            rows.append(record)
            decoder.decode_weights.pop(role)
            ttnn.deallocate(dw)
            drain_profile()
    path = (
        Path(__file__).resolve().parents[1]
        / "doc/optimized_decoder"
        / os.environ.get("ORNITH_SWEEP_OUTPUT", f"geometry_layer{layer_idx}.json")
    )
    path.write_text(json.dumps(rows, indent=2) + "\n")
    if os.environ.get("ORNITH_READER_CONFIRM") == "1":
        assert set(roles) <= set(captured), "A requested role was not executed"
        assert len(rows) == 6 * len(roles), "Incomplete alternating reader coverage"
        assert all(row["status"] == "pass" for row in rows), [
            (row["role"], row["readers"], row["status"]) for row in rows if row["status"] != "pass"
        ]
