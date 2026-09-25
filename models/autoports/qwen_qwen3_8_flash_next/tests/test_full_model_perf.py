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

        sampling_elapsed = None
        if getattr(model, "sampling_trace_id", None) is not None:
            # The per-layer schedule samples in its own trace; the resident
            # stack schedule folds sampling into the suffix trace above.
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


@pytest.mark.skipif(os.getenv("RUN_QWEN38_PREFILL_SWEEP") != "1", reason="explicit long-prompt prefill sweep point")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_long_prompt_prefill_sweep_point(bh_1d_mesh_device, device_params):
    """One prefill sweep point: a real long prompt through vLLM-sized 1,024-token calls.

    The microchunk size is an import-time constant (``QWEN38_PREFILL_CHUNK``), so the
    driver script runs one process per grid point.  Records seconds per call, the model's
    per-microchunk timing, tokens/s, and the final-token top-5 for cross-config agreement.
    Env: QWEN38_SWEEP_ISL (16384), QWEN38_SWEEP_MAX_SEQ_LEN (262144), QWEN38_SWEEP_OUT (json path).
    """

    import json
    from pathlib import Path

    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    isl = int(os.getenv("QWEN38_SWEEP_ISL", "16384"))
    max_seq_len = int(os.getenv("QWEN38_SWEEP_MAX_SEQ_LEN", "262144"))
    call_tokens = int(os.getenv("QWEN38_SWEEP_CALL_TOKENS", "1024"))
    out_path = os.getenv("QWEN38_SWEEP_OUT")
    prompt = _real_text_prompt(isl)
    assert prompt.shape == (1, isl)

    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=max_seq_len,
    )
    generator = Qwen38Generator(model, tokenizer=None)
    pages = model._default_page_table_host.clone()
    calls = []
    try:
        # One unrecorded warm-up so program compiles are not in the numbers.
        warm = generator.prefill_forward(
            prompt[:, :call_tokens],
            page_table=pages,
            prompt_lens=[call_tokens],
            start_pos=[0],
            intermediate_prefill_mask=[False],
            request_ids=("sweep-warm",),
            read_from_device=False,
        )
        ttnn.deallocate(warm)
        ttnn.synchronize_device(bh_1d_mesh_device)

        started = time.perf_counter()
        logits = None
        for start in range(0, isl, call_tokens):
            end = min(isl, start + call_tokens)
            final = end == isl
            t0 = time.perf_counter()
            out = generator.prefill_forward(
                prompt,
                page_table=pages,
                prompt_lens=[end],
                start_pos=[start],
                intermediate_prefill_mask=[not final],
                request_ids=("sweep",),
                read_from_device=False,
            )
            if final:
                ttnn.synchronize_device(bh_1d_mesh_device)
                logits = out
            else:
                ttnn.deallocate(out)
            calls.append(
                {
                    "start": start,
                    "end": end,
                    "seconds": time.perf_counter() - t0,
                    "model_timing": dict(model.last_prefill_timing or {}),
                }
            )
        total = time.perf_counter() - started
        host = model.logits_to_torch(logits).float().reshape(-1)
        ttnn.deallocate(logits)
        top5 = torch.topk(host, 5).indices.tolist()
        result = {
            "isl": isl,
            "max_seq_len": max_seq_len,
            "call_tokens": call_tokens,
            "prefill_chunk": int(os.getenv("QWEN38_PREFILL_CHUNK", "128")),
            "moe_prefill_slabs": os.getenv("QWEN38_MOE_PREFILL_SLABS", "0"),
            "timing_sync": os.getenv("QWEN38_PREFILL_TIMING_SYNC", "0"),
            "seconds": total,
            "tokens_per_second": isl / total,
            "seconds_per_microchunk_mean": sum(c["model_timing"].get("seconds", 0.0) for c in calls)
            / max(1, sum(c["model_timing"].get("microchunks", 0) for c in calls)),
            "final_top5": top5,
            "calls": calls,
            "host_service": model.host_service_totals(),
        }
        print({"prefill_sweep_point": {k: v for k, v in result.items() if k != "calls"}})
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(result, indent=1))
    finally:
        generator.close()


def _real_text_prompt(isl: int) -> torch.Tensor:
    """Exactly ``isl`` tokens of real English prose (this port's own documentation)."""

    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(H.MODEL_SNAPSHOT))
    root = Path(__file__).resolve().parents[1] / "doc"
    pieces = []
    for path in sorted(root.rglob("*.md")):
        try:
            pieces.append(path.read_text(errors="replace"))
        except OSError:
            continue
    text = "\n\n".join(pieces)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    while ids.shape[-1] < isl:
        ids = torch.cat([ids, ids], dim=-1)
    return ids[:, :isl].to(torch.int64)
