# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MTP draft head: load ``mtp.*`` and measure draft acceptance against HF greedy tokens.

Teacher-forced on the AIME24 chat reference: the target runs eagerly on the HF
tokens, the MTP head drafts ``token_{i+2}`` from ``hidden_i`` and
``token_{i+1}``, and every draft is compared with the HF greedy token.  The
acceptance rate is the expected gain of lossless k=1 speculative decoding.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import load_reference, write_report
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import REFERENCE, _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.model import HC_COUNT, Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.mtp import Qwen38MTPDraftHead
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import RESIDUAL_SHARD_WIDTH, MultichipDecoder

DECODE_ROWS = int(os.getenv("QWEN38_MTP_DECODE_ROWS", "98"))
EVIDENCE = Path(os.getenv("QWEN38_EVIDENCE_DIR", str(Path(__file__).parents[1] / "doc/phase2_mtp")))


def _top1(model, logits) -> int:
    return int(model.logits_to_torch(logits)[0, -1].argmax())


def _slice_rows(residual, index: int):
    start = HC_COUNT * index
    return ttnn.slice(residual, [0, 0, start, 0], [1, 1, start + HC_COUNT, RESIDUAL_SHARD_WIDTH])


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_mtp_draft_acceptance(bh_1d_mesh_device, device_params, record_property):
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    ref = load_reference(REFERENCE)
    prompt = torch.as_tensor(ref["prompt_tokens"], dtype=torch.int64).reshape(1, -1)
    teacher = torch.as_tensor(ref["reference_tokens"], dtype=torch.int64).reshape(-1)
    n = int(prompt.shape[-1])
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    head = None
    residual_all = None
    report: dict[str, object] = {}
    try:
        started = time.perf_counter()
        head = Qwen38MTPDraftHead(model)
        report["mtp_load_seconds"] = time.perf_counter() - started
        report["mtp_layer_load_seconds"] = head.layer_load_seconds
        state = model.new_batch_state([n], request_ids=("mtp-draft",))
        residual_all = model.prefill_forward(prompt, state=state, return_residual=True)
        assert list(residual_all.shape) == [1, 1, HC_COUNT * n, RESIDUAL_SHARD_WIDTH], list(residual_all.shape)

        # Target prefill prediction (position n-1 -> teacher[0]).
        last = _slice_rows(residual_all, n - 1)
        prefill_logits = model.project_logits(last)
        ttnn.deallocate(last)
        prefill_top1 = _top1(model, prefill_logits)
        ttnn.deallocate(prefill_logits)
        report["prefill_top1_matches_hf"] = prefill_top1 == int(teacher[0])

        # Seed the MTP layer over the prompt: hidden_i + token_{i+1} at position i.
        # For i <= n-3 the draft is checked against prompt[i+2]; i = n-2 predicts
        # teacher[0]; i = n-1 (with teacher[0]) predicts teacher[1].
        prompt_hits = prompt_total = 0
        gen_hits = gen_total = 0
        gen_hits_given_target_correct = gen_total_given_target_correct = 0
        seed_started = time.perf_counter()
        full_sequence = torch.cat([prompt.reshape(-1), teacher])
        for i in range(n):
            rows = _slice_rows(residual_all, i)
            next_token = int(full_sequence[i + 1])
            model.copy_tokens(state, torch.tensor([next_token]))
            head.set_scratch_position(i)
            logits = head.draft_logits(rows, state.token_input, current_pos=head.scratch_pos, page_table=state.page_table)
            draft = _top1(model, logits)
            ttnn.deallocate(logits)
            ttnn.deallocate(rows)
            expected = int(full_sequence[i + 2])
            if i + 2 < n:
                prompt_total += 1
                prompt_hits += int(draft == expected)
            else:
                gen_total += 1
                gen_hits += int(draft == expected)
        report["mtp_prompt_seed_seconds"] = time.perf_counter() - seed_started
        ttnn.deallocate(residual_all)
        residual_all = None

        # Teacher-forced decode: target consumes teacher[t] at position n+t.
        target_hits = 0
        decode_rows = min(DECODE_ROWS, int(teacher.numel()) - 2)
        target_seconds = 0.0
        for t in range(decode_rows):
            token = int(teacher[t])
            model.copy_tokens(state, torch.tensor([token]))
            model._apply_qsa_variant(model._select_qsa_variant(state))
            ple_ids = torch.tensor([[token]], dtype=torch.int64)
            started = time.perf_counter()
            residual = model.decode_residual_eager(state, ple_ids)
            logits = model.project_logits(residual)
            target_top1 = _top1(model, logits)
            ttnn.deallocate(logits)
            target_seconds += time.perf_counter() - started
            target_correct = target_top1 == int(teacher[t + 1])
            target_hits += int(target_correct)
            # Draft token_{t+2} from hidden at position n+t and token teacher[t+1].
            model.copy_tokens(state, torch.tensor([int(teacher[t + 1])]))
            draft_logits = head.draft_logits(
                residual, state.token_input, current_pos=state.current_pos, page_table=state.page_table
            )
            draft = _top1(model, draft_logits)
            ttnn.deallocate(draft_logits)
            ttnn.deallocate(residual)
            hit = draft == int(teacher[t + 2])
            gen_total += 1
            gen_hits += int(hit)
            if target_correct:
                gen_total_given_target_correct += 1
                gen_hits_given_target_correct += int(hit)
            ttnn.plus_one(state.current_pos)
            model._advance_host_positions(state)

        report.update(
            prompt_tokens=n,
            decode_rows=decode_rows,
            target_top1_percent=100.0 * target_hits / max(decode_rows, 1),
            draft_prompt_acceptance_percent=100.0 * prompt_hits / max(prompt_total, 1),
            draft_prompt_total=prompt_total,
            draft_generation_acceptance_percent=100.0 * gen_hits / max(gen_total, 1),
            draft_generation_total=gen_total,
            draft_generation_acceptance_given_target_correct_percent=(
                100.0 * gen_hits_given_target_correct / max(gen_total_given_target_correct, 1)
            ),
            target_eager_seconds_per_token=target_seconds / max(decode_rows, 1),
            draft_eager_seconds_per_call=head.draft_seconds / max(head.draft_calls, 1),
            draft_calls=head.draft_calls,
        )
        acceptance = report["draft_generation_acceptance_percent"] / 100.0
        report["expected_tokens_per_step_k1"] = 1.0 + acceptance
        print({"mtp_draft_acceptance": report})
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        write_report(report, EVIDENCE / "mtp_draft_acceptance.json")
        for key, value in report.items():
            if isinstance(value, (int, float, bool)):
                record_property(key, value)
        assert report["prefill_top1_matches_hf"]
        assert report["draft_generation_acceptance_percent"] > 40.0, report
    finally:
        if residual_all is not None and residual_all.is_allocated():
            ttnn.deallocate(residual_all)
        if head is not None:
            head.close()
        model.close(best_effort=True)
