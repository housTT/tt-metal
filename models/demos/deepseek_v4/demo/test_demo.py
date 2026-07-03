# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pytest wrapper for the DeepSeek-V4-Flash TT-NN demo (mirrors other demos' test_demo.py).

    pytest models/demos/deepseek_v4/demo/test_demo.py
"""
import pytest

from models.demos.deepseek_v4.demo.demo import run_demo


@pytest.mark.parametrize(
    "prompt, max_new_tokens",
    [("The capital of France is", 8), ("Tenstorrent builds", 6)],
)
def test_deepseek_v4_demo(prompt, max_new_tokens):
    new_ids, text = run_demo(prompt, max_new_tokens=max_new_tokens, num_layers=2, device_id=0, verify=True)
    assert len(new_ids) == max_new_tokens
    assert isinstance(text, str) and len(text) > 0
