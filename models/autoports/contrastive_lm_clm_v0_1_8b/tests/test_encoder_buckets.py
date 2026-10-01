# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import TtQwen3Encoder


@pytest.fixture
def expect_error():
    def _expect(exc_type, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except exc_type as exc:
            return exc
        raise AssertionError(f"{exc_type.__name__} was not raised")

    return _expect


def bare_encoder():
    return TtQwen3Encoder.__new__(TtQwen3Encoder)


def test_default_buckets_cover_the_context(monkeypatch):
    monkeypatch.delenv("CLM_TRACE_LENS", raising=False)
    enc = bare_encoder()
    assert enc._select_trace_lens(2048) == [128, 256, 512, 1024, 2048]
    assert enc._select_trace_lens(1024) == [128, 256, 512, 1024]
    assert enc._select_trace_lens(640) == [128, 256, 512, 640]


def test_env_override_and_validation(monkeypatch, expect_error):
    monkeypatch.setenv("CLM_TRACE_LENS", "1024,128,2048")
    enc = bare_encoder()
    assert enc._select_trace_lens(2048) == [128, 1024, 2048]
    monkeypatch.setenv("CLM_TRACE_LENS", "128,200")
    expect_error(ValueError, enc._select_trace_lens, 2048)
    monkeypatch.setenv("CLM_TRACE_LENS", "4096")
    assert enc._select_trace_lens(2048) == [2048]


def test_padded_len_picks_the_smallest_bucket():
    enc = bare_encoder()
    enc.trace_lens = [128, 256, 512, 1024, 2048]
    assert [enc.padded_len(n) for n in (1, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048)] == [
        128,
        128,
        256,
        256,
        512,
        512,
        1024,
        1024,
        2048,
        2048,
    ]
    enc.trace_lens = [128, 1024, 2048]
    assert enc.padded_len(129) == 1024
    assert enc.padded_len(4096) == 2048
