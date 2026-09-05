# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Restored eager, immediate replay, and stressed replay boundary checks."""
import json
import os

import pytest
import torch

import ttnn

from ..tt.fused_decoder import FusedDecoder
from . import test_functional_decoder as H
from .optimization_candidates import selected_candidate
from .test_optimization_experiments import recorded_activations

pytestmark = H.pytestmark


def differences(a, b):
    if isinstance(a, dict):
        return {key: differences(value, b[key]) for key, value in a.items()}
    if isinstance(a, list):
        return [differences(value, other) for value, other in zip(a, b)]
    return {"equal": torch.equal(a, b), "max_abs": (a.float() - b.float()).abs().max().item(), "pcc": H.pcc(a, b)}


def all_equal(record):
    if isinstance(record, list):
        return all(all_equal(value) for value in record)
    if "equal" in record:
        return record["equal"]
    return all(all_equal(value) for value in record.values())


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_restored_trace(mesh_device, monkeypatch, layer_idx):
    records = []
    for name, cls in (("fused", FusedDecoder), ("optimized", selected_candidate())):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
        recorded = recorded_activations(layer_idx)
        x = H.to_device(mesh_device, recorded[:, :2048])
        d = H.to_device(mesh_device, recorded[:, 2048:2049])
        pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
        for _ in range(7):
            decoder.reset_state()
            out = decoder.prefill_forward(x, page_table=table)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
        saved = H._snapshot_state(decoder)
        immutable = {
            "token": ttnn.to_torch(d),
            "attn_norm": ttnn.to_torch(decoder.w["attn_norm"]),
            "ff_norm": ttnn.to_torch(decoder.w["ff_norm"]),
        }

        def forward():
            return decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)

        def restore():
            H._restore_state(decoder, saved)
            assert all_equal(differences(saved, H._snapshot_state(decoder)))

        out = forward()
        eager1, state1 = ttnn.to_torch(out), H._snapshot_state(decoder)
        ttnn.deallocate(out)
        restore()
        out = forward()
        eager2, state2 = ttnn.to_torch(out), H._snapshot_state(decoder)
        ttnn.deallocate(out)
        record = {
            "name": name,
            "config": os.environ.get("ORNITH_OPT_CONFIG", "{}"),
            "E1_E2": differences(eager1, eager2),
            "state_E1_E2": differences(state1, state2),
        }
        restore()
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        try:
            for i in range(4):
                restore()
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                record[f"E2_T{i+1}"] = differences(eager2, ttnn.to_torch(out))
                record[f"state_E2_T{i+1}"] = differences(state2, H._snapshot_state(decoder))
            for _ in range(7):
                restore()
                for _ in range(32):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
            restore()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            record["E2_poststress"] = differences(eager2, ttnn.to_torch(out))
            record["state_E2_poststress"] = differences(state2, H._snapshot_state(decoder))
            record["immutable"] = differences(
                immutable,
                {
                    "token": ttnn.to_torch(d),
                    "attn_norm": ttnn.to_torch(decoder.w["attn_norm"]),
                    "ff_norm": ttnn.to_torch(decoder.w["ff_norm"]),
                },
            )
            print("TRACE_BOUNDARY " + json.dumps(record), flush=True)
            records.append(record)
        finally:
            ttnn.release_trace(mesh_device, trace)
    for record in records:
        assert all(all_equal(value) for key, value in record.items() if key not in ("name", "config")), record


def test_two_norms(mesh_device, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(H, "FunctionalDecoder", selected_candidate())
        decoder, _, _ = H.build_decoder(mesh_device, 0, "real")
    d = H.to_device(mesh_device, recorded_activations(0)[:, 2048:2049])
    original = ttnn.to_torch(d)

    def forward():
        outputs = []
        for role in ("attn_norm", "ff_norm"):
            out = decoder._norm(d, decoder.w[role])
            outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(out)
        return outputs

    eager = []
    for _ in range(3):
        outputs = forward()
        eager.append([ttnn.to_torch(out) for out in outputs])
        for out in outputs:
            ttnn.deallocate(out)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        record = {
            "E1_E2": differences(eager[0], eager[1]),
            "E2_E3": differences(eager[1], eager[2]),
            "E2_T1": differences(eager[1], [ttnn.to_torch(out) for out in outputs]),
            "input": differences(original, ttnn.to_torch(d)),
        }
        print("TWO_NORMS " + json.dumps(record), flush=True)
        assert all_equal(record)
    finally:
        ttnn.release_trace(mesh_device, trace)


def test_eager_boundaries(mesh_device, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(H, "FunctionalDecoder", selected_candidate())
        decoder, table, _ = H.build_decoder(mesh_device, 0, "real")
    recorded = recorded_activations(0)
    x, d = H.to_device(mesh_device, recorded[:, :2048]), H.to_device(mesh_device, recorded[:, 2048:2049])
    out = decoder.prefill_forward(x, page_table=table)
    ttnn.to_torch(out)
    ttnn.deallocate(out)
    saved = H._snapshot_state(decoder)
    observations = []
    norm, linear = decoder._norm, decoder._linear
    add = ttnn.add

    def observe_add(a, b, **kwargs):
        watch = list(a.shape) == [1, 1, 4096] and a.is_sharded()
        if watch:
            index = sum(key.startswith("add_a") for key in observations[-1])
            observations[-1][f"add_a{index}"] = ttnn.to_torch(a)
            observations[-1][f"add_b{index}"] = ttnn.to_torch(b)
            if index == 0:
                from pathlib import Path

                torch.save(
                    (observations[-1][f"add_a{index}"], observations[-1][f"add_b{index}"]),
                    Path(__file__).resolve().parents[1] / "doc/optimized_decoder/activations/autofix_add.pt",
                )
        out = add(a, b, **kwargs)
        if watch:
            observations[-1][f"add_out{index}"] = ttnn.to_torch(out)
        return out

    monkeypatch.setattr(ttnn, "add", observe_add)

    def observe_norm(x, weight):
        name = (
            "ff_norm"
            if weight is decoder.w["ff_norm"]
            else "attn_norm"
            if weight is decoder.w["attn_norm"]
            else "other_norm"
        )
        observations[-1][name + "_in"] = ttnn.to_torch(x)
        out = norm(x, weight)
        observations[-1][name + "_out"] = ttnn.to_torch(out)
        return out

    def observe_linear(x, role, **kwargs):
        if role in ("gate_proj", "up_proj", "down_proj", "gdn_out", "gdn_z_epilogue"):
            observations[-1][role + "_in"] = ttnn.to_torch(x)
        out = linear(x, role, **kwargs)
        if role in ("gate_proj", "up_proj", "down_proj", "gdn_out", "gdn_z_epilogue"):
            observations[-1][role + "_out"] = ttnn.to_torch(out)
        return out

    monkeypatch.setattr(decoder, "_norm", observe_norm)
    monkeypatch.setattr(decoder, "_linear", observe_linear)
    for _ in range(3):
        H._restore_state(decoder, saved)
        observations.append({})
        out = decoder.decode_forward(d, page_table=table)
        observations[-1]["out"] = ttnn.to_torch(out)
        ttnn.deallocate(out)
    record = {
        "E1_E2": differences(observations[0], observations[1]),
        "E2_E3": differences(observations[1], observations[2]),
    }
    print("EAGER_BOUNDARIES " + json.dumps(record), flush=True)
    assert all_equal(record)


@pytest.mark.parametrize("mode", ("native", "bf16", "fp32", "interleaved"))
def test_residual_add(mesh_device, monkeypatch, mode):
    from pathlib import Path

    with monkeypatch.context() as patch:
        patch.setattr(H, "FunctionalDecoder", selected_candidate())
        decoder, _, _ = H.build_decoder(mesh_device, 0, "real")
    host_a, host_b = torch.load(
        Path(__file__).resolve().parents[1] / "doc/optimized_decoder/activations/autofix_add.pt", weights_only=True
    )
    mem = decoder._width_memory(4096, 8)
    a, b = [
        ttnn.to_memory_config(
            H.to_device(mesh_device, host, dtype=ttnn.float32 if host.dtype == torch.float32 else ttnn.bfloat16), mem
        )
        for host in (host_a, host_b)
    ]

    def forward():
        aa, bb = a, b
        if mode == "bf16":
            aa, bb = [ttnn.typecast(t, ttnn.bfloat16) for t in (a, b)]
        elif mode == "fp32":
            aa, bb = [ttnn.typecast(t, ttnn.float32) for t in (a, b)]
        elif mode == "interleaved":
            aa, bb = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (a, b)]
        out = ttnn.add(aa, bb, memory_config=ttnn.DRAM_MEMORY_CONFIG if mode == "interleaved" else mem)
        if mode == "fp32":
            out = ttnn.typecast(out, ttnn.bfloat16)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

    eager = []
    for _ in range(3):
        out = forward()
        eager.append(ttnn.to_torch(out))
        ttnn.deallocate(out)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        replay = []
        for _ in range(4):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            replay.append(ttnn.to_torch(out))
        record = {
            "E1_E2": differences(eager[0], eager[1]),
            "E2_E3": differences(eager[1], eager[2]),
            "E2_T": differences([eager[1]] * 4, replay),
            "oracle": differences(host_a + host_b, eager[1]),
            "input": differences([host_a, host_b], [ttnn.to_torch(a), ttnn.to_torch(b)]),
        }
        print("RESIDUAL_ADD " + mode + " " + json.dumps(record), flush=True)
        assert all(all_equal(value) for key, value in record.items() if key != "oracle")
    finally:
        ttnn.release_trace(mesh_device, trace)
