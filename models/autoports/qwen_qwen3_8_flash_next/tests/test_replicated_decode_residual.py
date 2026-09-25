# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B: replicated decode residual against the fractured decode ABI.

The replicated layout keeps the full ``[1,1,1,10240]`` residual on every rank
so both hyper-connection mixers run without collectives.  It must reproduce
the fractured stack's logits (same weights, only the reduction order differs)
and must reduce the number of collectives per decode token.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder

LAYERS = tuple(int(v) for v in os.getenv("QWEN38_AB_LAYERS", "0,1,3").split(","))
PROMPT = torch.tensor([[17, 4051, 279, 1550, 6722, 315, 9625, 374]], dtype=torch.int64)
if os.getenv("QWEN38_AB_PROMPT", "short").startswith("aime"):
    # The 201-token AIME24 chat prompt used by the 48-layer gates (4 KV pages);
    # ``aime<N>`` repeats it N times for prefill timing of longer prompts.
    _ref = torch.load(
        os.path.join(os.path.dirname(__file__), "..", "doc", "full_model", "readiness_aime24_chat.refpt"),
        map_location="cpu",
        weights_only=False,
    )
    PROMPT = torch.as_tensor(_ref["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    _repeat = os.getenv("QWEN38_AB_PROMPT", "aime")[len("aime"):]
    if _repeat:
        PROMPT = PROMPT.repeat(1, int(_repeat))
DECODE_STEPS = int(os.getenv("QWEN38_AB_DECODE_STEPS", "4"))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _run(mesh_device, decode_residual: str, *, traced: bool, forced_tokens=None, moe_kernel: str = "dispatch"):
    """Run prefill + DECODE_STEPS decode steps.

    ``forced_tokens`` teacher-forces the token fed at each decode step so two
    arms see identical inputs; otherwise the arm feeds back its own argmax.
    """

    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=LAYERS,
        decode_residual=decode_residual,
        moe_kernel=moe_kernel,
    )
    generator = Qwen38Generator(model, object())
    logits_out = []
    tokens = []
    timing = None
    try:
        state = generator.allocate_batch_state([int(PROMPT.shape[-1])], request_ids=(f"ab-{decode_residual}",))
        prefill_started = time.perf_counter()
        logits = model.prefill_forward(PROMPT, state=state)
        ttnn.synchronize_device(mesh_device)
        prefill_seconds = time.perf_counter() - prefill_started
        save_logits = os.getenv("QWEN38_AB_SAVE_PREFILL_LOGITS")
        if save_logits:
            # Per-position prefill logits of a fresh state, for offline exactness
            # comparison between prefill configurations (e.g. microchunk sizes).
            all_state = generator.allocate_batch_state([int(PROMPT.shape[-1])], request_ids=(f"ab-all-{decode_residual}",))
            all_logits = model.prefill_forward(PROMPT, state=all_state, return_all_logits=True)
            torch.save(model.logits_to_torch(all_logits).to(torch.bfloat16), save_logits)
            ttnn.deallocate(all_logits)
            generator.release_batch_state(all_state) if hasattr(generator, "release_batch_state") else None
        # Second (warm) prefill of the same prompt into a fresh state for timing
        # without kernel compilation; its state is discarded.
        warm_state = generator.allocate_batch_state([int(PROMPT.shape[-1])], request_ids=(f"ab-warm-{decode_residual}",))
        warm_started = time.perf_counter()
        warm_logits = model.prefill_forward(PROMPT, state=warm_state)
        ttnn.synchronize_device(mesh_device)
        warm_prefill_seconds = time.perf_counter() - warm_started
        print({"prefill_seconds": prefill_seconds, "warm_prefill_seconds": warm_prefill_seconds, "prompt_len": int(PROMPT.shape[-1])})
        # Continue decode from the warm state (a committed state cannot be prefilled again).
        ttnn.deallocate(logits)
        logits = warm_logits
        state = warm_state
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        logits_out.append(model.logits_to_torch(logits)[0, 0, -1].clone())
        sampled = model.sample_logits(logits, state)
        token = model.sampled_tokens_to_torch(sampled, state)
        tokens.append(int(token[0]))
        ttnn.deallocate(logits)
        for step in range(DECODE_STEPS):
            if forced_tokens is not None:
                # Teacher forcing must reach the device token buffer that the
                # embedding reads, not only the host id used for PLE hashing.
                token = torch.tensor([forced_tokens[step]], dtype=torch.int64)
                model.copy_tokens(state, token)
            if traced:
                step_logits, sampled = model.decode_token_out_traced(state, token.reshape(1, 1))
            else:
                step_logits = model._decode_stack_eager(state, token.reshape(1, 1))
                sampled = model.sample_logits(step_logits, state)
            ttnn.synchronize_device(mesh_device)
            if step_logits is not None:
                logits_out.append(model.logits_to_torch(step_logits)[0, 0, -1].clone())
            token = model.sampled_tokens_to_torch(sampled, state)
            tokens.append(int(token[0]))
        if traced and model.trace_replays:
            # Steady-state replay timing: several replays back to back.
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                _, sampled = model.decode_token_out_traced(state, token.reshape(1, 1))
                ttnn.synchronize_device(mesh_device)
                samples.append(time.perf_counter() - started)
                token = model.sampled_tokens_to_torch(sampled, state)
            timing = sorted(samples)[len(samples) // 2]
        audit = model.runtime_fallback_audit(state)
    finally:
        generator.close()
    return logits_out, tokens, timing, audit


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_replicated_decode_matches_fractured(bh_1d_mesh_device, device_params, record_property):
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    traced = os.getenv("QWEN38_AB_TRACED", "1") == "1"
    fractured_logits, fractured_tokens, fractured_ms, _ = _run(bh_1d_mesh_device, "fractured", traced=traced)
    replicated_logits, replicated_tokens, replicated_ms, audit = _run(
        bh_1d_mesh_device, "replicated", traced=traced, forced_tokens=fractured_tokens
    )

    audit_text = repr(audit)
    assert "'decode_residual': 'replicated'" in audit_text, audit_text[:2000]
    assert len(fractured_logits) == len(replicated_logits)
    pccs = [_pcc(a, b) for a, b in zip(fractured_logits, replicated_logits)]
    top1 = [int(a.argmax()) == int(b.argmax()) for a, b in zip(fractured_logits, replicated_logits)]
    record_property("logit_pccs", pccs)
    record_property("top1_agreement", top1)
    record_property("fractured_tokens", fractured_tokens)
    record_property("replicated_tokens", replicated_tokens)
    record_property("fractured_step_ms", None if fractured_ms is None else fractured_ms * 1e3)
    record_property("replicated_step_ms", None if replicated_ms is None else replicated_ms * 1e3)
    print(
        {
            "replicated_decode_ab": {
                "layers": LAYERS,
                "logit_pccs": pccs,
                "top1_agreement": top1,
                "fractured_tokens": fractured_tokens,
                "replicated_tokens": replicated_tokens,
                "fractured_step_ms": None if fractured_ms is None else fractured_ms * 1e3,
                "replicated_step_ms": None if replicated_ms is None else replicated_ms * 1e3,
            }
        }
    )
    # The prefill logits are produced by the shared fractured prefill path and
    # must match exactly; decode logits differ only by reduction order.  The
    # replicated arm is teacher-forced with the fractured arm's tokens, so the
    # per-step PCC measures the two ABIs on identical inputs and state history.
    assert pccs[0] > 0.9999
    assert all(p > 0.98 for p in pccs[1:]), pccs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_sparse_bank_moe_matches_dispatch(bh_1d_mesh_device, device_params, record_property):
    """Dispatch-free sparse-bank MoE against the fabric dispatch/combine pipeline."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    traced = os.getenv("QWEN38_AB_TRACED", "1") == "1"
    base_logits, base_tokens, base_ms, _ = _run(bh_1d_mesh_device, "fractured", traced=traced)
    bank_logits, bank_tokens, bank_ms, audit = _run(
        bh_1d_mesh_device, "fractured", traced=traced, forced_tokens=base_tokens, moe_kernel="sparse_bank"
    )
    assert "'moe_kernel': 'sparse_bank'" in repr(audit)
    pccs = [_pcc(a, b) for a, b in zip(base_logits, bank_logits)]
    top1 = [int(a.argmax()) == int(b.argmax()) for a, b in zip(base_logits, bank_logits)]
    print(
        {
            "sparse_bank_moe_ab": {
                "layers": LAYERS,
                "logit_pccs": pccs,
                "top1_agreement": top1,
                "dispatch_tokens": base_tokens,
                "sparse_bank_tokens": bank_tokens,
                "dispatch_step_ms": None if base_ms is None else base_ms * 1e3,
                "sparse_bank_step_ms": None if bank_ms is None else bank_ms * 1e3,
            }
        }
    )
    record_property("logit_pccs", pccs)
    assert all(p > 0.98 for p in pccs), pccs


# Greedy tokens of the reduced (0, 1, 3) stack for PROMPT, recorded from the
# shipped fractured/dispatch build (tt-metal 60f1562e8ec dirty tree).  Any
# decode-path change must reproduce them; the step time is printed for A/B.
GOLDEN_TOKENS_013 = [99933, 106847, 220403, 180146, 44072, 159505, 91142, 234339, 76215, 56821, 77952, 8594, 239103]
# Re-recorded 2026-09-10 for the current defaults (sparse-bank MoE, dense QSA prefill,
# fused GDN prefill norm); the previous list ([99933, 106847, 176330, 171425, 236926])
# is reproduced with QWEN38_QSA_DENSE_PREFILL=0 QWEN38_GDN_PREFILL_FUSED_NORM=0.


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_stack_golden_tokens(bh_1d_mesh_device, device_params, record_property):
    """Regression guard: current env/config reproduces the recorded greedy tokens."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    assert LAYERS == (0, 1, 3), "golden tokens are recorded for layers (0, 1, 3)"
    if os.getenv("QWEN38_AB_PROMPT", "short") != "short":
        pytest.skip("golden tokens are recorded for the short prompt")
    decode_residual = os.getenv("QWEN38_DECODE_RESIDUAL", "fractured")
    moe_kernel = os.getenv("QWEN38_MOE_KERNEL", "dispatch")
    logits, tokens, step_ms, audit = _run(
        bh_1d_mesh_device, decode_residual, traced=True, moe_kernel=moe_kernel
    )
    print(
        {
            "reduced_stack_golden": {
                "decode_residual": decode_residual,
                "moe_kernel": moe_kernel,
                "tokens": tokens,
                "step_ms": None if step_ms is None else step_ms * 1e3,
                "env": {k: v for k, v in os.environ.items() if k.startswith("QWEN38_")},
            }
        }
    )
    record_property("tokens", tokens)
    record_property("step_ms", None if step_ms is None else step_ms * 1e3)
    assert tokens == GOLDEN_TOKENS_013[: len(tokens)], tokens


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_stack_token_dump(bh_1d_mesh_device, device_params):
    """Print the greedy tokens of the reduced stack for the configured prompt/env (bisection aid)."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    _, tokens, step_ms, _ = _run(
        bh_1d_mesh_device,
        os.getenv("QWEN38_DECODE_RESIDUAL", "fractured"),
        traced=True,
        moe_kernel=os.getenv("QWEN38_MOE_KERNEL", "dispatch"),
    )
    print({"reduced_stack_dump": {"prompt_len": int(PROMPT.shape[-1]), "tokens": tokens, "step_ms": None if step_ms is None else step_ms * 1e3}})
