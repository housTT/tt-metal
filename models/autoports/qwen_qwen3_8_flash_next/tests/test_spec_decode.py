# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lossless k=1 speculative decoding: two-row target step + MTP draft.

The speculative run must emit exactly the greedy tokens of the same kernels
(``force_reject=True`` runs the identical two-row graph but never accepts a
draft, so row 0 alone produces the reference sequence).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import REFERENCE, _device_params
from models.autoports.qwen_qwen3_8_flash_next.tests.test_replicated_decode_residual import GOLDEN_TOKENS_013
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.mtp import Qwen38MTPDraftHead
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.spec_decode import Qwen38SpeculativeDecoder

LAYERS = tuple(int(v) for v in os.getenv("QWEN38_SPEC_LAYERS", "0,1,3").split(","))
NEW_TOKENS = int(os.getenv("QWEN38_SPEC_TOKENS", "13"))
EVIDENCE = Path(os.getenv("QWEN38_EVIDENCE_DIR", str(Path(__file__).parents[1] / "doc/phase2_mtp")))
SHORT_PROMPT = torch.tensor([17, 4051, 279, 1550, 6722, 315, 9625, 374], dtype=torch.int64)


def _prompt() -> torch.Tensor:
    if os.getenv("QWEN38_SPEC_PROMPT", "short") == "aime":
        return torch.as_tensor(load_reference(REFERENCE)["prompt_tokens"], dtype=torch.int64).reshape(-1)
    return SHORT_PROMPT


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_speculative_decode_is_lossless(bh_1d_mesh_device, device_params, record_property):
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    prompt = _prompt()
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=2,
        max_seq_len=4096,
        layer_indices=None if LAYERS == tuple(range(48)) else LAYERS,
    )
    head = None
    report: dict[str, object] = {"layers": list(LAYERS), "prompt_tokens": int(prompt.numel()), "new_tokens": NEW_TOKENS}
    try:
        head = Qwen38MTPDraftHead(model)
        traced = os.getenv("QWEN38_SPEC_TRACED", "0") == "1"
        report["traced"] = traced
        reference = Qwen38SpeculativeDecoder(model, head, force_reject=True, traced=traced)
        started = time.perf_counter()
        plain_tokens = reference.generate(prompt, NEW_TOKENS)
        report["plain_seconds"] = time.perf_counter() - started
        report["plain_report"] = reference.report()

        if os.getenv("QWEN38_SPEC_REPEAT", "0") == "1":
            report["plain_tokens_repeat"] = Qwen38SpeculativeDecoder(model, head, force_reject=True, traced=traced).generate(prompt, NEW_TOKENS)
            report["plain_deterministic"] = report["plain_tokens_repeat"] == plain_tokens
        reference.release_traces()
        speculative = Qwen38SpeculativeDecoder(model, head, traced=traced)
        started = time.perf_counter()
        spec_tokens = speculative.generate(prompt, NEW_TOKENS)
        report["speculative_seconds"] = time.perf_counter() - started
        report["speculative_report"] = speculative.report()
        if os.getenv("QWEN38_SPEC_REPEAT", "0") == "1":
            report["speculative_tokens_repeat"] = Qwen38SpeculativeDecoder(model, head, traced=traced).generate(prompt, NEW_TOKENS)
            report["speculative_deterministic"] = report["speculative_tokens_repeat"] == spec_tokens
        report["plain_tokens"] = plain_tokens
        report["speculative_tokens"] = spec_tokens
        report["matches_plain"] = spec_tokens == plain_tokens
        if LAYERS == (0, 1, 3) and torch.equal(prompt, SHORT_PROMPT):
            report["matches_golden_013"] = plain_tokens[: len(GOLDEN_TOKENS_013)] == GOLDEN_TOKENS_013[: len(plain_tokens)]
        if os.getenv("QWEN38_SPEC_PROMPT", "short") == "aime" and LAYERS == tuple(range(48)):
            report_path = Path(__file__).parents[1] / "doc/phase3_prefill_a/aime24_autoregressive_100_report_final.json"
            if report_path.exists():
                tt_tokens = json.loads(report_path.read_text())["tt_tokens"]
                report["matches_free_run_report"] = spec_tokens == tt_tokens[: len(spec_tokens)]
        print({"speculative_decode": report})
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / "speculative_decode_lossless.json").write_text(json.dumps(report, indent=2))
        for key, value in report.items():
            if isinstance(value, (int, float, bool)):
                record_property(key, value)
        assert report["matches_plain"], report
    finally:
        for decoder in ("speculative", "reference"):
            obj = locals().get(decoder)
            if obj is not None:
                obj.release_traces()
        if head is not None:
            head.close()
        model.close(best_effort=True)
