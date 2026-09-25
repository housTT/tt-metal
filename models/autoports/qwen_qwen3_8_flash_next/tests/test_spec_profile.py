# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracy window for the traced two-row speculative step (reduced stack)."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tests.test_full_model import _device_params
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel
from models.autoports.qwen_qwen3_8_flash_next.tt.mtp import Qwen38MTPDraftHead
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.spec_decode import Qwen38SpeculativeDecoder

LAYERS = tuple(int(v) for v in os.getenv("QWEN38_SPEC_LAYERS", "0,1,3").split(","))
PROMPT = torch.tensor([17, 4051, 279, 1550, 6722, 315, 9625, 374], dtype=torch.int64)


@pytest.mark.skipif(os.getenv("RUN_QWEN38_SPEC_PROFILE") != "1", reason="profiling run only")
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_spec_profile_window(bh_1d_mesh_device, device_params):
    del device_params
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=2,
        max_seq_len=4096,
        layer_indices=LAYERS,
    )
    head = None
    try:
        head = Qwen38MTPDraftHead(model)
        decoder = Qwen38SpeculativeDecoder(
            model,
            head,
            force_reject=os.getenv("QWEN38_SPEC_FORCE_REJECT", "1") == "1",
            traced=os.getenv("QWEN38_SPEC_PROFILE_TRACED", "1") == "1",
        )
        decoder.start(PROMPT)
        for _ in range(int(os.getenv("QWEN38_SPEC_PROFILE_STEPS", "4"))):
            decoder.step()
        ttnn.synchronize_device(bh_1d_mesh_device)
        print({"spec_profile": decoder.report()})
        decoder.release_traces()
    finally:
        if head is not None:
            head.close()
        model.close(best_effort=True)
