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


def test_residual_add(mesh_device):
    """The GDN FP32 update must add deterministically to the BF16 residual.

    ORNITH_REPRO_MIXED_ADD=1 calls the faulty primitive directly for kernel
    investigation; the default exercises the decoder adaptation.
    """
    from ..tt.optimized_decoder import OptimizedDecoder

    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    decoder.device = mesh_device
    host_a = recorded_activations(0)[:, 2048:2049].to(torch.bfloat16)
    generator = torch.Generator().manual_seed(812)
    host_b = torch.randn(host_a.shape, generator=generator, dtype=torch.float32) * 0.02
    mem = decoder._width_memory(4096, 8)
    a, b = [
        ttnn.to_memory_config(H.to_device(mesh_device, host, dtype=dtype), mem)
        for host, dtype in ((host_a, ttnn.bfloat16), (host_b, ttnn.float32))
    ]
    raw = os.environ.get("ORNITH_REPRO_MIXED_ADD") == "1"

    def forward():
        out = ttnn.add(a, b, memory_config=mem) if raw else decoder._residual_add(a, b, mem)
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
            "oracle": differences((host_a.float() + host_b).to(torch.bfloat16), eager[1]),
            "input": differences([host_a, host_b], [ttnn.to_torch(a), ttnn.to_torch(b)]),
        }
        print("RESIDUAL_ADD " + json.dumps({"raw_primitive": raw, **record}), flush=True)
        assert all_equal(record)
    finally:
        ttnn.release_trace(mesh_device, trace)
