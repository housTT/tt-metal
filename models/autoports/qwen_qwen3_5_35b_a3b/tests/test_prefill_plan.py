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
        long_chunk = full and count == model.optimization.prefill_long_chunk
        if long_chunk:
            assert valid == count and absolute >= model.optimization.prefill_long_from
        if count > 1:
            assert count in model._planner_buckets(full) or long_chunk
            assert count % block == 0 and absolute % block == 0 and absolute % min(count, 128) == 0
        assert absolute + count <= model.config.max_seq_len
        if valid < count:
            assert index == len(flat) - 1
            assert capacity is None or absolute + count <= capacity
        offset += valid
    assert offset == length
    for group in groups:
        assert (sum(count for _, count, _ in group) <= max(model._planner_buckets(full))
                or (len(group) == 1 and group[0][1] == model.optimization.prefill_long_chunk))
        assert len(group) == 1 or all(count > 1 for _, count, _ in group)


@pytest.mark.parametrize("kind,full,length,expected", [
    ("full_attention", True, 2, [(1, 1), (1, 1)]),
    ("full_attention", True, 3, [(32, 3)]),
    ("full_attention", True, 4, [(32, 4)]),
    ("full_attention", True, 5, [(32, 5)]),
    ("linear_attention", False, 2, [(1, 1), (1, 1)]),
    ("linear_attention", False, 3, [(32, 3)]),
    ("linear_attention", False, 4, [(32, 4)]),
    ("full_attention", True, 4100, [(4096, 4096)] + [(1, 1)] * 4),
    ("linear_attention", False, 4100, [(4096, 4096)] + [(1, 1)] * 4),
    ("linear_attention", False, 4101, [(4096, 4096), (32, 5)]),
])
def test_short_plan_choice(kind, full, length, expected):
    model = planner(kind, 32)
    plan = [(count, valid) for group in model._chunk_plan(full, 0, length, None) for _, count, valid in group]
    assert plan == expected


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("kind,full,length,expected", [
    ("full_attention", True, 1, [(1, 1)]),
    ("full_attention", True, 2, [(32, 2)]),
    ("full_attention", True, 3, [(32, 3)]),
    ("linear_attention", False, 2, [(32, 2)]),
    ("full_attention", True, 4100, [(4096, 4096)] + [(1, 1)] * 4),
    ("linear_attention", False, 4100, [(4096, 4096)] + [(1, 1)] * 4),
])
def test_mesh_short_plan_choice(tp, kind, full, length, expected):
    from ..tt.multichip_decoder import mesh_policy

    model = planner(kind, 32)
    model.optimization = od.OptimizationPolicy(**mesh_policy(od.DEFAULT_POLICIES[kind], tp, kind))
    plan = [(count, valid) for group in model._chunk_plan(full, 0, length, None) for _, count, valid in group]
    assert plan == expected


@pytest.mark.parametrize("kind,full", [("linear_attention", False), ("full_attention", True)])
@pytest.mark.parametrize("tp,length,expected", [
    (4, 1300, [(2048, 1300)]),
    (2, 1300, [(2048, 1300)]),
    (4, 2600, [(4096, 2600)]),
    (2, 2600, [(4096, 2600)]),
    (4, 4103, [(4096, 4096), (32, 7)]),
])
def test_mesh_mid_plan_choice(kind, full, tp, length, expected):
    from ..tt.multichip_decoder import mesh_policy

    model = planner(kind, 32)
    model.optimization = od.OptimizationPolicy(**mesh_policy(od.DEFAULT_POLICIES[kind], tp, kind))
    plan = [(count, valid) for group in model._chunk_plan(full, 0, length, None) for _, count, valid in group]
    assert plan == expected


@pytest.mark.parametrize("start,length,long_from,expected", [
    (0, 16384, 0, [8192, 8192]),
    (0, 16384, 8192, [4096, 4096, 8192]),
    (0, 12288, 0, [8192, 4096]),
    (4096, 8192, 0, [8192]),
    (0, 8190, 0, [4096, 4096]),
])
def test_long_chunk_merge(start, length, long_from, expected):
    model = planner("full_attention", 32)
    policy = dict(od.DEFAULT_POLICIES["full_attention"], prefill_long_chunk=8192, prefill_long_from=long_from)
    model.optimization = od.OptimizationPolicy(**policy)
    groups = model._chunk_plan(True, start, length, None)
    counts = [count for group in groups for _, count, _ in group]
    assert counts == expected
    assert all(len(group) == 1 for group in groups if group[0][1] == 8192)
    assert sum(valid for group in groups for _, _, valid in group) == length


def test_long_chunk_moe_pieces():
    sizes = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    assert od._moe_pieces(8192, sizes, 2, 1024) == [4096, 4096]
    with pytest.raises(ValueError):
        od._moe_pieces(8192 + 32, sizes, 2, 1024)
