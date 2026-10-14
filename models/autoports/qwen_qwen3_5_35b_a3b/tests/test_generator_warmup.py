# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ..tt import generator as gen
from ..tt import optimized_decoder as od
from .test_prefill_plan import planner

CHUNK = gen.PREFILL_CHUNK


def _bucket(length):
    size = 128
    while size < length:
        size *= 2
    return size


def _items(model, full, start, length):
    if length == 1:
        return {(full, start, "single_token")}
    physical = _bucket(length)
    capacity = 262144 if full else None
    groups = model._chunk_plan(full, start, length, capacity)
    planned = max(offset + count for group in groups for offset, count, _ in group)
    assert sum(valid for group in groups for _, _, valid in group) == length
    assert planned >= physical
    rows = -(-max(planned, physical) // 32) * 32
    composition = tuple(tuple((count, valid == count) for _, count, valid in group) for group in groups)
    return {(full, start, physical, rows, composition)}


@pytest.mark.parametrize("kind", ["linear_attention", "full_attention"])
def test_warmup_covers_every_prefill_signature(kind):
    model = planner(kind, 32)
    model.optimization = od.OptimizationPolicy(**dict(od.DEFAULT_POLICIES[kind], prefill_bucket_plan=True))
    full = kind == "full_attention"
    needed, warmed = set(), set()
    for start, warm in ((0, gen.WARMUP_FIRST), (CHUNK, gen.WARMUP_CONTINUATION)):
        for length in range(1, CHUNK + 1):
            needed |= _items(model, full, start, length)
        for length in warm:
            warmed |= _items(model, full, start, length)
    assert needed <= warmed, sorted(needed - warmed)[:10]


def test_physical_buckets_match_model():
    from ..tt.model import Qwen35Model

    model = Qwen35Model.__new__(Qwen35Model)
    assert [model.physical_length(n) for n in (1, 128, 129, 300, 4097, 8192)] == [128, 128, 256, 512, 8192, 8192]
    assert all(b in gen.PREFILL_BUCKETS for b in {model.physical_length(n) for n in range(1, CHUNK + 1)})


def _chunks(start, length):
    out, pos, end = [], start, start + length
    while pos < end:
        size = min(CHUNK - pos % CHUNK, end - pos)
        out.append((pos, size))
        pos += size
    return out


@pytest.mark.parametrize("kind", ["linear_attention", "full_attention"])
def test_warmup_covers_aligned_continuation_starts(kind):
    model = planner(kind, 32)
    model.optimization = od.OptimizationPolicy(**dict(od.DEFAULT_POLICIES[kind], prefill_bucket_plan=True))
    full = kind == "full_attention"

    def keyed(start, length):
        return {(item[0], start > 0) + item[2:] for item in _items(model, full, start, length)}

    warmed = set()
    for start, warm in ((0, gen.WARMUP_FIRST), (CHUNK, gen.WARMUP_CONTINUATION)):
        for length in warm:
            warmed |= keyed(start, length)
    lengths = sorted(set(range(1, 600)) | set(range(600, 2 * CHUNK + 1, 37)) | {CHUNK - 1, CHUNK, CHUNK + 1})
    missing = []
    for start in (128, 256, 4096, 4224, 8064, 8320, 12288, 16256):
        for length in lengths:
            for pos, size in _chunks(start, length):
                missing.extend(sorted(keyed(pos, size) - warmed))
    assert not missing, missing[:10]


def test_unaligned_prefill_start_rejected():
    import torch

    from ..tt.model import Qwen35Model

    model = Qwen35Model.__new__(Qwen35Model)
    model.max_seq_len = 262144
    model.full_layers = []
    for start in (32, 64, 96, 8224):
        with pytest.raises(ValueError, match="multiple of 128"):
            model.prefill(torch.zeros(1, 4, dtype=torch.int64), start_pos=start, page_table=None, request_slots=(0,))
