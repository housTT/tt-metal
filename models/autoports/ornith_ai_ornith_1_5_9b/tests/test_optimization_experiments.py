# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-harness fused/optimized comparisons using recorded checkpoint activations."""

import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import ttnn

from ..tt.fused_decoder import FusedDecoder
from . import test_functional_decoder as H
from .optimization_candidates import selected_candidate

pytestmark = H.pytestmark


def recorded_activations(layer_idx):
    return torch.load(
        Path(__file__).resolve().parents[1] / f"doc/optimized_decoder/activations/layer{layer_idx}.pt",
        weights_only=True,
    )


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_optimized_pair(mesh_device, monkeypatch, layer_idx):
    outputs, timings = {}, {}
    recorded = recorded_activations(layer_idx)
    prompt = recorded[:, :2048]
    token = recorded[:, 2048:2049]
    for name, cls in (("fused", FusedDecoder), ("optimized", selected_candidate())):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
        x = H.to_device(mesh_device, prompt)
        d = H.to_device(mesh_device, token)
        pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
        prefill_times = []
        for iteration in range(7):
            decoder.reset_state()
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            out = decoder.prefill_forward(x, page_table=table)
            ttnn.synchronize_device(mesh_device)
            duration = (time.perf_counter() - start) * 1000
            if iteration >= 2:
                prefill_times.append(duration)
            if iteration == 6:
                outputs[name] = {"prefill": ttnn.to_torch(out)}
            ttnn.deallocate(out)
        saved = H._snapshot_state(decoder)

        def forward():
            return decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)

        out = forward()
        outputs[name]["decode"] = ttnn.to_torch(out)
        ttnn.deallocate(out)
        H._restore_state(decoder, saved)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        decode_times = []
        for iteration in range(7):
            H._restore_state(decoder, saved)
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            duration = (time.perf_counter() - start) * 1000 / 32
            if iteration >= 2:
                decode_times.append(duration)
        outputs[name]["stress"] = ttnn.to_torch(out)
        H._restore_state(decoder, saved)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        outputs[name]["trace"] = ttnn.to_torch(out)
        outputs[name]["state"] = H._snapshot_state(decoder)
        if hasattr(decoder, "cache_permutation"):
            inverse = torch.argsort(torch.tensor(decoder.cache_permutation))
            outputs[name]["state"]["k"] = outputs[name]["state"]["k"][..., inverse]
        ttnn.release_trace(mesh_device, trace)
        timings[name] = {
            "prefill_ms": prefill_times,
            "decode_ms": decode_times,
            "prefill_median_ms": statistics.median(prefill_times),
            "decode_median_ms": statistics.median(decode_times),
        }
    correlations = {}
    for mode in ("prefill", "decode", "trace", "stress"):
        correlations[mode] = H.pcc(outputs["fused"][mode], outputs["optimized"][mode])
    for name in outputs:
        assert torch.equal(outputs[name]["decode"], outputs[name]["trace"])
    state_pcc = []
    for key, expected in outputs["fused"]["state"].items():
        actual = outputs["optimized"]["state"][key]
        if isinstance(expected, list):
            for a, b in zip(expected, actual):
                state_pcc.append(H.pcc(a, b))
        else:
            state_pcc.append(H.pcc(expected, actual))
    golden, decoded = H.run_reference(layer_idx, "real", prompt, decode_x=[token] * 32, decode_steps=32)
    hf_pcc = {
        "prefill": H.pcc(golden, outputs["optimized"]["prefill"]),
        "decode": H.pcc(decoded[0], outputs["optimized"]["decode"]),
    }
    if layer_idx == 0:
        hf_pcc["stress"] = H.pcc(decoded[-1], outputs["optimized"]["stress"])
    record = {
        "layer": layer_idx,
        "pcc": correlations,
        "hf_pcc": hf_pcc,
        "state_pcc_diagnostic": state_pcc,
        "timings": timings,
        "policy": os.environ.get("ORNITH_OPT_POLICY", "default"),
        "config": os.environ.get("ORNITH_OPT_CONFIG", "default"),
        "resolved_policy": asdict(decoder.policy),
        "resolved_config": asdict(decoder.optimization),
        "variant": os.environ.get("ORNITH_OPT_VARIANT", "default"),
        "activation_source": "recorded pinned-HF layer input",
    }
    print("OPTIMIZATION_PAIR " + json.dumps(record))
    assert min(hf_pcc.values()) >= H.PCC_BAR
