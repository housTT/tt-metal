# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed fused-layer performance windows for Tracy/tt-perf-report."""

from __future__ import annotations

import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import _paged_inputs, _upload
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder

LAYER_KINDS = (0, 1, 3)
DECODE_REPLAYS = int(os.environ.get("QWEN38_FUSED_PERF_DECODE_REPLAYS", "10"))


@pytest.mark.parametrize("layer_idx", LAYER_KINDS)
def test_warmed_prefill_and_traced_decode(mesh_device, layer_idx):
    mode = os.environ.get("QWEN38_FUSED_PERF_MODE", "both")
    if mode not in {"both", "prefill", "decode"}:
        raise ValueError(f"unsupported QWEN38_FUSED_PERF_MODE={mode}")
    config = H.target_config()
    max_seq_len = 4096 if layer_idx == 3 else 128
    layer = FusedDecoder.from_state_dict(
        None,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    assert type(layer) is FusedDecoder
    prefill_hidden = _upload(torch.zeros(1, 1, 128, 10240, dtype=torch.bfloat16), mesh_device)
    prefill_kwargs = {}
    if layer_idx == 1:
        prefill_kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, 128, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        cos, sin = H.rope_tables(max_seq_len)
        page_host = H.shuffled_page_table(max_seq_len)
        page, chunk_pages, rot = _paged_inputs(layer, mesh_device, page_host, cos, sin, 128)
        prefill_kwargs.update(page_table=page, page_tables_per_chunk=chunk_pages, rot_mats=rot)

    layer.prefill_forward(prefill_hidden, **prefill_kwargs)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = None
    if mode in {"both", "prefill"}:
        signpost(f"FUSED_PERF_PREFILL_L{layer_idx}")
        start = time.perf_counter()
        with H.ForbidHostFallback():
            layer.prefill_forward(prefill_hidden, **prefill_kwargs)
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        signpost(f"FUSED_PERF_PREFILL_L{layer_idx}_END")
    if mode == "prefill":
        print(f"FUSED_PERFEVIDENCE layer={layer_idx} prefill_seq=128 warmed_prefill_ms={prefill_ms:.6f}")
        return

    layer.prepare_decode_state()
    decode_hidden = _upload(torch.zeros(1, 1, 1, 10240, dtype=torch.bfloat16), mesh_device)
    current_pos = _upload(
        torch.tensor([127], dtype=torch.int32),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    decode_kwargs = {"current_pos": current_pos}
    if layer_idx == 1:
        decode_kwargs["ple_embeddings"] = _upload(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16), mesh_device)
    if layer_idx == 3:
        decode_kwargs.update(page_table=page, rot_mats=rot)

    layer.decode_forward(decode_hidden, **decode_kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    with H.ForbidHostFallback():
        traced_output = layer.decode_forward(decode_hidden, **decode_kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    if os.environ.get("QWEN38_FUSED_PERF_FLUSH_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
        ttnn.synchronize_device(mesh_device)

    signpost(f"FUSED_PERF_DECODE_L{layer_idx}")
    start = time.perf_counter()
    for _ in range(DECODE_REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    if os.environ.get("QWEN38_FUSED_PERF_FLUSH_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
        ttnn.synchronize_device(mesh_device)
    decode_ms = (time.perf_counter() - start) * 1000.0 / DECODE_REPLAYS
    signpost(f"FUSED_PERF_DECODE_L{layer_idx}_END")
    assert list(traced_output.shape) == [1, 1, 1, 10240]
    ttnn.release_trace(mesh_device, trace_id)
    prefill_field = "not-profiled" if prefill_ms is None else f"{prefill_ms:.6f}"
    print(
        f"FUSED_PERFEVIDENCE layer={layer_idx} prefill_seq=128 warmed_prefill_ms={prefill_field} "
        f"traced_decode_ms={decode_ms:.6f} decode_replays={DECODE_REPLAYS}"
    )
