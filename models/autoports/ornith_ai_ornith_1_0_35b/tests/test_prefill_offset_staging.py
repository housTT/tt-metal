# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract tests for offset-stable serving-prefill metadata."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from models.autoports.ornith_ai_ornith_1_0_35b.tt import generator as G
from models.autoports.ornith_ai_ornith_1_0_35b.tt import model as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD


def _host_only_model():
    model = M.OrnithModel.__new__(M.OrnithModel)
    model.prefill_chunk = 2048
    model.page_block_size = 64
    model.max_context = 262144
    return model


def test_offset_chunk_metadata_rebases_page_ids_and_covers_physical_padding():
    model = _host_only_model()
    page_table = torch.arange(96, dtype=torch.int32).reshape(1, 96) + 1000

    positions, chunk_start, fill_table, physical_len = model._prefill_chunk_host_inputs(
        page_table,
        start_pos=2048,
        logical_len=130,
    )

    assert physical_len == 256
    assert positions.shape == (1, 256)
    assert positions[0, 0].item() == 2048
    assert positions[0, -1].item() == 2303
    assert chunk_start.tolist() == [2048]
    assert fill_table.shape == (1, 32), "the paged-fill program shape stays fixed for every tail"
    assert fill_table[0, :4].tolist() == page_table[0, 32:36].tolist()
    assert torch.count_nonzero(fill_table[0, 4:]).item() == 0


@pytest.mark.parametrize(
    ("start_pos", "logical_len", "blocks", "message"),
    [
        (128, 128, 96, "multiple of prefill_chunk"),
        (0, 0, 96, "1..2048"),
        (0, 2049, 96, "1..2048"),
        (4096, 2048, 64, "needs block"),
    ],
)
def test_offset_chunk_metadata_rejects_non_scheduler_or_short_table_inputs(
    start_pos, logical_len, blocks, message, expect_error
):
    model = _host_only_model()
    with expect_error(ValueError, message):
        model._prefill_chunk_host_inputs(
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            start_pos=start_pos,
            logical_len=logical_len,
        )


class _ShapeTensor:
    def __init__(self, *shape):
        self.shape = shape


def test_prefill_bundle_cannot_be_reused_for_a_different_offset_or_shape(expect_error):
    staged = OD.PrefillChunkInputs(
        full_page_table=_ShapeTensor(1, 64),
        fill_page_table=_ShapeTensor(1, 32),
        position_idxs=_ShapeTensor(1, 256),
        chunk_start_idx_tensor=_ShapeTensor(1),
        start_pos=2048,
        physical_len=256,
    )
    staged.validate_for(start_pos=2048, physical_len=256, batch=1, page_block_size=64)

    with expect_error(ValueError, "staged prefill metadata"):
        staged.validate_for(start_pos=4096, physical_len=256, batch=1, page_block_size=64)
    with expect_error(ValueError, "position row"):
        OD.PrefillChunkInputs(
            full_page_table=_ShapeTensor(1, 64),
            fill_page_table=_ShapeTensor(1, 32),
            position_idxs=_ShapeTensor(1, 128),
            chunk_start_idx_tensor=_ShapeTensor(1),
            start_pos=2048,
            physical_len=256,
        ).validate_for(start_pos=2048, physical_len=256, batch=1, page_block_size=64)


def _call_name(node):
    parts = []
    value = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def test_attention_keeps_scalar_fallback_and_uses_flexible_sdpa_for_staged_inputs():
    tree = ast.parse(textwrap.dedent(inspect.getsource(OD.OptimizedDecoder._attention_prefill)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "ttnn.transformer.chunked_scaled_dot_product_attention"
    ]
    assert len(calls) == 2

    flexible = [call for call in calls if {kw.arg for kw in call.keywords} >= {"chunk_start_idx_tensor"}]
    scalar = [call for call in calls if "chunk_start_idx_tensor" not in {kw.arg for kw in call.keywords}]
    assert len(flexible) == 1 and len(flexible[0].args) == 4
    assert len(scalar) == 1 and len(scalar[0].args) == 5

    rope_tree = ast.parse(textwrap.dedent(inspect.getsource(OD.OrnithFusedRope.indexed_forward)))
    rope_calls = [_call_name(node) for node in ast.walk(rope_tree) if isinstance(node, ast.Call)]
    assert rope_calls.count("ttnn.embedding") == 2
    assert "ttnn.slice" not in rope_calls


def test_serving_stages_one_bundle_for_one_scheduler_chunk(monkeypatch):
    calls = []
    marker = object()
    page_row = object()

    class _Model:
        def prepare_prefill_chunk_inputs(self, **kwargs):
            calls.append(("stage", kwargs))
            return marker

        def prefill_request_into_slot(self, tokens, **kwargs):
            calls.append(("prefill", tokens.clone(), kwargs))
            return torch.zeros(1, 1, 8)

    generator = G.OrnithGenerator.__new__(G.OrnithGenerator)
    generator.model = _Model()
    generator.max_batch_size = 4
    generator._resolve_page_table = lambda table, cache, caller: table
    generator._page_row_tensor = lambda row: page_row
    monkeypatch.setattr(G.ttnn, "deallocate", lambda tensor: calls.append(("deallocate", tensor)))

    table = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    output = generator.prefill_requests_into_slots(
        torch.arange(4096, dtype=torch.int32).reshape(1, 4096),
        [4096],
        [3],
        page_table=table,
        start_pos=[2048],
        ensure_traces=False,
    )

    assert output.shape == (1, 1, 8)
    stage = next(call for call in calls if call[0] == "stage")
    assert stage[1]["page_table"] is page_row
    assert torch.equal(stage[1]["host_page_table"], table)
    assert stage[1]["start_pos"] == 2048
    assert stage[1]["logical_len"] == 2048
    prefill = next(call for call in calls if call[0] == "prefill")
    assert prefill[1].shape == (1, 2048)
    assert prefill[2]["page_table"] is marker
    assert prefill[2]["continue_from_state"] is True
    assert calls.count(("deallocate", page_row)) == 1
