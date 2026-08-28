# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model endpoint/trace windows for Tracy and tt-perf-report."""

from __future__ import annotations

import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder


@pytest.mark.skipif(os.getenv("RUN_QWEN38_FULL_MODEL_PROFILE") != "1", reason="explicit reduced profiler gate")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_full_model_profile_window(bh_1d_mesh_device, device_params):
    """Profile one real layer of every kind plus final norm/LM head/sampling."""

    mode = os.getenv("QWEN38_FULL_MODEL_PROFILE_MODE", "decode")
    if mode not in {"prefill", "decode"}:
        raise ValueError("QWEN38_FULL_MODEL_PROFILE_MODE must be prefill or decode")
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    layer_indices = tuple(int(value) for value in os.getenv("QWEN38_FULL_MODEL_PROFILE_LAYERS", "0").split(","))
    if not layer_indices or any(value not in (0, 1, 3) for value in layer_indices):
        raise ValueError("QWEN38_FULL_MODEL_PROFILE_LAYERS must select representative layers 0, 1, or 3")
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=layer_indices,
    )
    generator = Qwen38Generator(model, object())
    prompt = torch.tensor([[17]], dtype=torch.int64)
    try:
        state = generator.allocate_batch_state([1], request_ids=("full-profile-warm",))
        logits = model.prefill_forward(prompt, state=state)
        ttnn.synchronize_device(bh_1d_mesh_device)

        if mode == "prefill":
            ttnn.deallocate(logits)
            state = generator.allocate_batch_state([1], request_ids=("full-profile-prefill",))
            ttnn.ReadDeviceProfiler(bh_1d_mesh_device)
            signpost("FULL_MODEL_REDUCED_PREFILL")
            started = time.perf_counter()
            logits = model.prefill_forward(prompt, state=state)
            ttnn.synchronize_device(bh_1d_mesh_device)
            elapsed = time.perf_counter() - started
            ttnn.ReadDeviceProfiler(bh_1d_mesh_device)
            signpost("FULL_MODEL_REDUCED_PREFILL_END")
            ttnn.deallocate(logits)
            print({"full_model_reduced_profile": {"mode": mode, "host_seconds": elapsed}})
            return

        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        sampled = model.sample_logits(logits, state)
        input_shadow = model.sampled_tokens_to_torch(sampled, state)
        ttnn.deallocate(logits)
        _, sampled = model.decode_token_out_traced(state, input_shadow.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        input_shadow = model.sampled_tokens_to_torch(sampled, state)
        ttnn.ReadDeviceProfiler(bh_1d_mesh_device)

        signpost("FULL_MODEL_REDUCED_TOKEN_OUT")
        started = time.perf_counter()
        _, sampled = model.decode_token_out_traced(state, input_shadow.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        elapsed = time.perf_counter() - started
        ttnn.ReadDeviceProfiler(bh_1d_mesh_device)
        signpost("FULL_MODEL_REDUCED_TOKEN_OUT_END")
        input_shadow = model.sampled_tokens_to_torch(sampled, state)

        signpost("FULL_MODEL_REDUCED_SAMPLING")
        sampling_started = time.perf_counter()
        ttnn.execute_trace(bh_1d_mesh_device, model.sampling_trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(bh_1d_mesh_device)
        sampling_elapsed = time.perf_counter() - sampling_started
        ttnn.ReadDeviceProfiler(bh_1d_mesh_device)
        signpost("FULL_MODEL_REDUCED_SAMPLING_END")
        assert input_shadow.shape == (1,)
        print(
            {
                "full_model_reduced_profile": {
                    "mode": mode,
                    "token_out_host_seconds": elapsed,
                    "sampling_host_seconds": sampling_elapsed,
                    "audit": model.runtime_fallback_audit(state),
                }
            }
        )
    finally:
        generator.close()
