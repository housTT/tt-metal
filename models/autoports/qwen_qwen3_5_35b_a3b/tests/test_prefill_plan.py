# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the optimized prefill chunk planner invariants."""

from types import SimpleNamespace

import pytest

from ..tt import optimized_decoder as od

LENGTHS = [1, 2, 3, 5, 31, 32, 33, 63, 100, 127, 129, 511, 513, 1000, 1024, 1100, 1153, 2048, 2176, 3072, 4000,
           3952, 4095, 4096, 4097, 4098, 4100, 8191, 8193, 262143, 262144]


def planner(kind, page):
    model = od.OptimizedDecoder.__new__(od.OptimizedDecoder)
    model.optimization = od.OptimizationPolicy(**od.DEFAULT_POLICIES[kind])
    model.layer = SimpleNamespace(attention=SimpleNamespace(paged_kv_cache_key=SimpleNamespace(shape=(1, 1, page, 1))))
    model.config = SimpleNamespace(max_seq_len=262144)
    return model


@pytest.mark.parametrize("kind,full,page", [("linear_attention", False, 32), ("full_attention", True, 32),
                                            ("full_attention", True, 128)])
@pytest.mark.parametrize("start", [0, 128, 1000, 4096, 258176, 260000])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("tight", [False, True])
def test_plan_invariants(kind, full, page, start, length, tight):
    length = min(length, 262144 - start)
    model = planner(kind, page)
    block = page if full else 32
    capacity = None
    if full:
        capacity = -(-(start + length) // page) * page if tight else start + length + 4096
    groups = model._chunk_plan(full, start, length, capacity)
    flat = [entry for group in groups for entry in group]
    offset = 0
    for index, (chunk_offset, count, valid) in enumerate(flat):
        assert chunk_offset == offset
        assert 0 < valid <= count
        absolute = start + chunk_offset
        if count > 1:
            assert count in model._planner_buckets(full)
            assert count % block == 0 and absolute % block == 0 and absolute % min(count, 128) == 0
        assert absolute + count <= model.config.max_seq_len
        if valid < count:
            assert index == len(flat) - 1
            assert capacity is None or absolute + count <= capacity
        offset += valid
    assert offset == length
    for group in groups:
        assert sum(count for _, count, _ in group) <= max(model._planner_buckets(full))
        assert len(group) == 1 or all(count > 1 for _, count, _ in group)


@pytest.mark.parametrize("kind,full,length,expected", [
    ("full_attention", True, 2, [(1, 1), (1, 1)]),
    ("full_attention", True, 3, [(1, 1)] * 3),
    ("full_attention", True, 4, [(32, 4)]),
    ("full_attention", True, 5, [(32, 5)]),
    ("linear_attention", False, 2, [(1, 1), (1, 1)]),
    ("linear_attention", False, 3, [(32, 3)]),
    ("linear_attention", False, 4, [(32, 4)]),
    ("full_attention", True, 4100, [(4096, 4096)] + [(1, 1)] * 4),
    ("linear_attention", False, 4100, [(4096, 4096), (32, 4)]),
])
def test_short_plan_choice(kind, full, length, expected):
    model = planner(kind, 32)
    plan = [(count, valid) for group in model._chunk_plan(full, 0, length, None) for _, count, valid in group]
    assert plan == expected
