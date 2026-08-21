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
    model._prefill_fill_page_tables = {1: object(), 2: object(), 4: object()}
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


def test_prefill_bundle_validates_real_batch4_page_rows():
    staged = OD.PrefillChunkInputs(
        full_page_table=_ShapeTensor(4, 96),
        fill_page_table=_ShapeTensor(4, 32),
        position_idxs=_ShapeTensor(1, 2048),
        chunk_start_idx_tensor=_ShapeTensor(1),
        start_pos=4096,
        physical_len=2048,
    )

    staged.validate_for(start_pos=4096, physical_len=2048, batch=4, page_block_size=64)


def test_same_batch_state_packs_clone_only_mutable_device_tensors(monkeypatch):
    cloned = []
    batch_idxs = object()
    weight = object()
    recurrent = object()
    conv = object()
    canonical = {
        "batch_size": 4,
        "batch_idxs": batch_idxs,
        "recurrent_state": recurrent,
        "conv_state": [conv],
        "conv1d_weights": {2048: [weight]},
        "conv1d_lengths": [2048],
    }
    monkeypatch.setattr(M.ttnn, "zeros_like", lambda tensor: cloned.append(tensor) or ("clone", tensor))

    current = M.OrnithModel._clone_pack_mutable_state(canonical)

    assert current["batch_idxs"] is batch_idxs
    assert current["conv1d_weights"] is canonical["conv1d_weights"]
    assert current["recurrent_state"] == ("clone", recurrent)
    assert current["conv_state"] == [("clone", conv)]
    assert cloned == [recurrent, conv]


def test_batch1_state_row_is_a_borrowed_persistent_tensor_not_a_slice(monkeypatch):
    class _Buffer:
        shape = (1, 8, 128, 128)

    buffer = _Buffer()
    monkeypatch.setattr(M.ttnn, "slice", lambda *args, **kwargs: pytest.fail("B1 must not call slice"))

    row, owned = M.OrnithModel._state_row(M.OrnithModel.__new__(M.OrnithModel), buffer, 0, 1)

    assert row is buffer
    assert owned is False


def test_grouped_continuation_requires_exact_pack_size_lane_order_and_slots(expect_error):
    model = M.OrnithModel.__new__(M.OrnithModel)
    model._prefill_pack_slots = {
        (2, 0): None,
        (4, 0): (7, 2, 6, 1),
        (4, 1): (5, 0, 4, 3),
    }
    model.prefill_batching_runtime = {
        "fallback_invocations": 0,
        "fallback_reasons": {},
    }

    model._require_prefill_pack_binding(4, 0, (7, 2, 6, 1))
    model._require_prefill_pack_binding(4, 1, (5, 0, 4, 3))
    with expect_error(RuntimeError, "not continuation slots"):
        model._require_prefill_pack_binding(4, 0, (2, 7, 6, 1))
    # A preempted B4 wave cannot silently regroup its surviving first two rows as B2: that B2 pack
    # was never their authority and the intervening fixed-batch decode may have advanced its rows.
    with expect_error(RuntimeError, "bound to slots None"):
        model._require_prefill_pack_binding(2, 0, (7, 2))

    assert model.prefill_batching_runtime == {
        "fallback_invocations": 2,
        "fallback_reasons": {"continuation_pack_mismatch": 2},
    }


def test_new_pack_binding_invalidates_stale_authority_for_every_overlapping_slot():
    model = M.OrnithModel.__new__(M.OrnithModel)
    model._prefill_pack_slot = 7
    model._prefill_pack_slots = {
        (1, 0): (7,),
        (2, 0): (5, 0),
        (4, 0): (7, 2, 6, 1),
        (4, 1): (3, 4, 5, 0),
    }

    model._bind_prefill_pack(2, 0, (7, 5))

    assert model._prefill_pack_slots == {
        (1, 0): None,
        (2, 0): (7, 5),
        (4, 0): (None, 2, 6, 1),
        (4, 1): (3, 4, None, 0),
    }
    assert model._prefill_pack_slot is None


def _continuation_planner_model(authorities):
    model = M.OrnithModel.__new__(M.OrnithModel)
    model._prefill_pack_slot = None
    model._prefill_pack_slots = dict(authorities)
    model.prefill_batching_runtime = {
        "migrated_continuations": 0,
        "migrated_users": 0,
        "fallback_invocations": 0,
        "fallback_reasons": {},
    }
    return model


@pytest.mark.parametrize(
    ("survivors", "expected"),
    [
        ((7, 6, 1), [((0, 1), 2, 0), ((2,), 1, 0)]),
        ((7, 6), [((0, 1), 2, 0)]),
        ((6,), [((0,), 1, 0)]),
    ],
)
def test_batch4_shrink_plans_collision_free_b2_b1_migration(survivors, expected):
    model = _continuation_planner_model(
        {
            (1, 0): None,
            (1, 1): None,
            (1, 2): None,
            (1, 3): None,
            (2, 0): None,
            (2, 1): None,
            (4, 0): (7, 2, 6, 1),
            (4, 1): None,
        }
    )

    assert model.plan_prefill_continuation_groups(survivors) == expected
    assert model.prefill_batching_runtime["fallback_invocations"] == 0


def test_batch2_shrink_does_not_clobber_the_original_batch1_branch():
    model = _continuation_planner_model(
        {
            (1, 0): (3,),
            (1, 1): None,
            (2, 0): (7, 2),
            (4, 0): None,
        }
    )

    # Initial wave B2(7,2)+B1(3); slot 7 aborts. Request order is deliberately B1 then B2-survivor.
    assert model.plan_prefill_continuation_groups((3, 2)) == [
        ((0,), 1, 0),
        ((1,), 1, 1),
    ]
    assert model._prefill_pack_slots[(1, 0)] == (3,)
    assert model._prefill_pack_slots[(2, 0)] == (None, 2)


def test_same_count_reorder_keeps_original_pack_and_row_order():
    model = _continuation_planner_model(
        {
            (1, 0): None,
            (2, 0): None,
            (4, 0): (7, 2, 6, 1),
        }
    )

    assert model.plan_prefill_continuation_groups((6, 7, 1, 2)) == [
        ((1, 3, 0, 2), 4, 0),
    ]


def _record_migrations(model):
    copies = []
    uses = []

    def reset(batch=1, *, pack_index=0):
        model._prefill_pack_slots[(int(batch), int(pack_index))] = None

    model._reset_prefill_pack = reset
    model._copy_prefill_state_row = lambda source, source_row, target, target_row: copies.append(
        (source, source_row, target, target_row)
    )
    model._use_pack = lambda batch, *, purpose="decode", pack_index=0: uses.append((batch, purpose, pack_index))
    return copies, uses


def test_batch4_to_three_migrates_authoritative_rows_zero_two_three_before_rebinding():
    model = _continuation_planner_model(
        {
            (1, 0): None,
            (1, 1): None,
            (2, 0): None,
            (4, 0): (7, 2, 6, 1),
        }
    )
    survivors = (7, 6, 1)
    plan = model.plan_prefill_continuation_groups(survivors)
    copies, _ = _record_migrations(model)

    for users, batch, lane in plan:
        slots = tuple(survivors[user] for user in users)
        model.prepare_prefill_pack_continuation(batch, lane, slots)

    assert copies == [
        ((4, 0), 0, (2, 0), 0),
        ((4, 0), 2, (2, 0), 1),
        ((4, 0), 3, (1, 0), 0),
    ]
    assert model._prefill_pack_slots[(2, 0)] == (7, 6)
    assert model._prefill_pack_slots[(1, 0)] == (1,)
    assert model._prefill_pack_slots[(4, 0)] is None
    assert model.prefill_batching_runtime["migrated_continuations"] == 2
    assert model.prefill_batching_runtime["migrated_users"] == 3


def test_batch2_to_one_migration_uses_a_second_lane_while_original_batch1_continues():
    model = _continuation_planner_model(
        {
            (1, 0): (3,),
            (1, 1): None,
            (2, 0): (7, 2),
        }
    )
    survivors = (2, 3)
    plan = model.plan_prefill_continuation_groups(survivors)
    copies, _ = _record_migrations(model)

    for users, batch, lane in plan:
        slots = tuple(survivors[user] for user in users)
        model.prepare_prefill_pack_continuation(batch, lane, slots)

    assert copies == [((2, 0), 1, (1, 1), 0)]
    assert model._prefill_pack_slots[(1, 0)] == (3,)
    assert model._prefill_pack_slots[(1, 1)] == (2,)


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
        def plan_prefill_continuation_groups(self, slots):
            calls.append(("plan", tuple(slots)))
            return [((0,), 1, 0)]

        def prepare_prefill_pack_continuation(self, batch, pack_index, slots):
            calls.append(("prepare", batch, pack_index, tuple(slots)))

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
    generator._page_table_tensor = lambda row: page_row
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
    assert ("plan", (3,)) in calls
    assert ("prepare", 1, 0, (3,)) in calls
    assert calls.count(("deallocate", page_row)) == 1


def test_concurrency8_plans_two_distinct_batch4_pack_lanes():
    assert G.plan_synchronized_prefill_groups(8) == [(0, 4, 0), (4, 8, 1)]
    assert G.plan_synchronized_prefill_groups(7) == [(0, 4, 0), (4, 6, 0), (6, 7, 0)]
    assert G.plan_synchronized_prefill_groups(1) == [(0, 1, 0)]


def test_concurrency8_runs_two_grouped_model_calls_with_arbitrary_slots(monkeypatch):
    calls = []

    class _Model:
        def attach_kv_cache(self, cache):
            raise AssertionError("the test did not pass an alternate cache")

        def record_prefill_fallback(self, reason, invocations=1):
            calls.append(("fallback", reason, invocations))

        def prepare_prefill_chunk_inputs(self, **kwargs):
            return kwargs

        def prefill_forward_batched_into_slots(self, tokens, **kwargs):
            calls.append(
                (
                    "group",
                    tuple(kwargs["slots"]),
                    kwargs["pack_index"],
                    tuple(tokens.shape),
                )
            )
            return torch.full((tokens.shape[0], 1, 3), kwargs["pack_index"] + 1.0)

    generator = G.OrnithGenerator.__new__(G.OrnithGenerator)
    generator.model = _Model()
    generator.max_batch_size = 8
    generator._resolve_page_table = lambda table, cache, caller: table
    generator._page_table_tensor = lambda rows: object()
    monkeypatch.setattr(G.ttnn, "deallocate", lambda tensor: None)

    slots = [7, 2, 6, 1, 5, 0, 4, 3]
    output = generator.prefill_requests_into_slots(
        torch.arange(8 * 128, dtype=torch.int32).reshape(8, 128),
        [128] * 8,
        slots,
        page_table=torch.zeros(8, 64, dtype=torch.int32),
        start_pos=[0] * 8,
        ensure_traces=False,
    )

    assert output.shape == (8, 1, 3)
    assert [call for call in calls if call[0] == "group"] == [
        ("group", tuple(slots[:4]), 0, (4, 128)),
        ("group", tuple(slots[4:]), 1, (4, 128)),
    ]
    assert not [call for call in calls if call[0] == "fallback"]


def test_continuation_does_not_mix_b2_survivor_with_original_b1_authority(monkeypatch):
    calls = []

    class _Model:
        plan_prefill_continuation_groups = M.OrnithModel.plan_prefill_continuation_groups
        _prefill_sources_for_slots = M.OrnithModel._prefill_sources_for_slots
        record_prefill_fallback = M.OrnithModel.record_prefill_fallback

        def __init__(self):
            self._prefill_pack_slot = 3
            self._prefill_pack_slots = {
                (1, 0): (3,),
                (1, 1): None,
                (2, 0): (7, 2),
            }
            self.prefill_batching_runtime = {
                "fallback_invocations": 0,
                "fallback_reasons": {},
            }

        def prepare_prefill_pack_continuation(self, batch, pack_index, slots):
            calls.append(("prepare", batch, pack_index, tuple(slots)))

        def prepare_prefill_chunk_inputs(self, **kwargs):
            return kwargs

        def prefill_request_into_slot(self, tokens, **kwargs):
            calls.append(("b1", kwargs["slot"], kwargs["pack_index"]))
            return torch.full((1, 1, 3), float(kwargs["slot"]))

    generator = G.OrnithGenerator.__new__(G.OrnithGenerator)
    generator.model = _Model()
    generator.max_batch_size = 8
    generator._resolve_page_table = lambda table, cache, caller: table
    generator._page_table_tensor = lambda rows: object()
    monkeypatch.setattr(G.ttnn, "deallocate", lambda tensor: None)

    output = generator.prefill_requests_into_slots(
        torch.zeros(2, 2176, dtype=torch.int32),
        [2176, 2176],
        [2, 3],
        page_table=torch.zeros(2, 64, dtype=torch.int32),
        start_pos=[2048, 2048],
        ensure_traces=False,
    )

    assert calls == [
        ("prepare", 1, 1, (2,)),
        ("b1", 2, 1),
        ("prepare", 1, 0, (3,)),
        ("b1", 3, 0),
    ]
    assert output[:, 0, 0].tolist() == [2.0, 3.0]


def test_mixed_fresh_lengths_fall_back_to_the_unchanged_batch1_path(monkeypatch):
    calls = []

    class _Model:
        def record_prefill_fallback(self, reason, invocations=1):
            calls.append(("fallback", reason, invocations))

        def prepare_prefill_chunk_inputs(self, **kwargs):
            return kwargs

        def prefill_request_into_slot(self, tokens, **kwargs):
            calls.append(("serial", kwargs["slot"], tuple(tokens.shape)))
            return torch.zeros(1, 1, 3)

        def prefill_forward_batched_into_slots(self, *args, **kwargs):
            raise AssertionError("mixed geometry must not enter the grouped device path")

    generator = G.OrnithGenerator.__new__(G.OrnithGenerator)
    generator.model = _Model()
    generator.max_batch_size = 4
    generator._resolve_page_table = lambda table, cache, caller: table
    generator._page_table_tensor = lambda rows: object()
    monkeypatch.setattr(G.ttnn, "deallocate", lambda tensor: None)

    generator.prefill_requests_into_slots(
        torch.arange(256, dtype=torch.int32).reshape(2, 128),
        [64, 128],
        [3, 1],
        page_table=torch.zeros(2, 64, dtype=torch.int32),
        start_pos=[0, 0],
        ensure_traces=False,
    )

    assert calls == [
        ("fallback", "mixed_chunk_geometry", 2),
        ("serial", 3, (1, 64)),
        ("serial", 1, (1, 128)),
    ]


def test_grouped_device_sampling_selects_each_real_row_and_keeps_slot_callback(monkeypatch):
    callbacks = []
    samples = []
    generator = G.OrnithGenerator.__new__(G.OrnithGenerator)
    generator.mesh_device = object()
    generator._prefill_tokens = object()
    generator.sampling = type(
        "_Sampling",
        (),
        {"sample": lambda self, logits, **kwargs: samples.append((logits, kwargs))},
    )()
    monkeypatch.setattr(G.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(G.ttnn, "get_device_tensors", lambda tensor: [object()])
    monkeypatch.setattr(G.ttnn, "to_torch", lambda tensor: torch.tensor([11, 22, 33, 44]))

    token = generator._sample_prefill_row(
        object(),
        local_row=2,
        user=5,
        slot=7,
        before_sample=lambda user, slot: callbacks.append((user, slot)),
    )

    assert token == 33
    assert callbacks == [(5, 7)]
    assert len(samples) == 1 and samples[0][1]["tt_out_tok"] is generator._prefill_tokens
