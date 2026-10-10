# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject experiment combinations that could silently lose state or routing."""

import pytest

from models.autoports.qwen_qwen3_5_35b_a3b.tt.fused_decoder import FusedDecoder


@pytest.mark.parametrize(
    "fusions,message",
    [
        (("gdn_inplace",), "gdn_inplace requires"),
        (("gdn_grouped", "gdn_inplace", "native_recurrence"), "gdn_inplace requires"),
        (("moe_gate", "compact_routing"), "compact_routing requires"),
        (("prefill_minimal", "prefill_split"), "Select one packed prefill adapter"),
        (("indexed_rope",), "indexed_rope requires"),
        (("indexed_rope", "rope", "qkv", "heads", "decode_heads"), "indexed_rope requires"),
    ],
)
def test_reject_unsafe_fusion_combination(fusions, message, expect_error):
    # Validation precedes checkpoint construction and device allocation.
    with expect_error(ValueError, message):
        FusedDecoder.from_state_dict(fusions=fusions)
