# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect

from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import (
    Gemma4Generator,
    PrefillGroup,
    batched_prefill_variants,
    plan_prefill_groups,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import Gemma4ForCausalLM


def test_burst_rows_form_one_group_in_row_order():
    groups, singles = plan_prefill_groups([128] * 16, [0] * 16, enabled=True)
    assert singles == []
    assert groups == [PrefillGroup(rows=tuple(range(16)), segment_len=128, padded_batch=16)]


def test_row_count_pads_to_a_power_of_two_and_lengths_pad_to_a_bucket():
    groups, singles = plan_prefill_groups([5, 31, 33, 127, 129], [0] * 5, enabled=True)
    assert groups == [PrefillGroup(rows=(0, 1, 2, 3), segment_len=128, padded_batch=4)]
    assert singles == [4]
    groups, singles = plan_prefill_groups([5, 31, 33, 127, 129, 200], [0] * 6, enabled=True)
    assert groups == [
        PrefillGroup(rows=(0, 1, 2, 3), segment_len=128, padded_batch=4),
        PrefillGroup(rows=(4, 5), segment_len=256, padded_batch=2),
    ]
    assert singles == []


def test_single_bucket_rows_below_min_rows_stay_single():
    groups, singles = plan_prefill_groups([100, 300], [0, 0], enabled=True)
    assert groups == []
    assert singles == [0, 1]


def test_continuations_and_long_rows_keep_the_single_row_path():
    groups, singles = plan_prefill_groups([128, 128, 128, 2000], [0, 64, 0, 0], enabled=True)
    assert groups == [PrefillGroup(rows=(0, 2), segment_len=128, padded_batch=2)]
    assert singles == [1, 3]


def test_disabled_or_single_row_is_all_single():
    assert plan_prefill_groups([128, 128], [0, 0], enabled=False) == ([], [0, 1])
    assert plan_prefill_groups([128], [0], enabled=True) == ([], [0])


def test_groups_respect_the_token_budget():
    groups, singles = plan_prefill_groups([1000] * 8, [0] * 8, enabled=True, max_tokens=4096)
    assert singles == []
    assert [group.rows for group in groups] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert all(group.segment_len == 1024 and group.padded_batch == 4 for group in groups)


def test_variants_are_bounded_by_the_token_budget():
    variants = batched_prefill_variants(buckets=(128, 256, 512, 1024), max_tokens=4096, max_rows=32)
    assert (32, 128) in variants and (4, 1024) in variants
    assert (8, 1024) not in variants
    assert all(batch * segment <= 4096 for batch, segment in variants)
    assert len(variants) == 14


def test_generator_and_adapter_wire_the_batched_path():
    generator = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "plan_prefill_groups(" in generator
    assert "self._run_batched_prefill_group(" in generator
    assert "user_id=row" in generator
    adapter = inspect.getsource(Gemma4ForCausalLM.warmup_model_prefill)
    assert "prepare_batched_prefill" in adapter
