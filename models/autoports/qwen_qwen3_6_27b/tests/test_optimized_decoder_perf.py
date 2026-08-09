# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill / traced-decode performance runs for the **optimized** Qwen3.6-27B decoder.

Identical windows, signposts and iteration counts to
``tests/test_functional_decoder_perf.py`` and ``tests/test_fused_decoder_perf.py`` — only
``harness.DECODER_CLS`` differs — so the three tt-perf-report tables are directly comparable.
Run one at a time under the profiler; see ``doc/optimized_decoder/probes/run_perf.sh``.
"""

from __future__ import annotations

import pytest

from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tests import test_functional_decoder_perf as base
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

PERF_PREFILL_LEN = base.PERF_PREFILL_LEN
PERF_DECODE_POS = base.PERF_DECODE_POS
PERF_DECODE_ITERS = base.PERF_DECODE_ITERS


@pytest.fixture(autouse=True)
def _use_optimized_decoder():
    previous = H.DECODER_CLS
    H.DECODER_CLS = OptimizedDecoder
    yield
    H.DECODER_CLS = previous


test_perf_prefill = base.test_perf_prefill
test_perf_decode_traced = base.test_perf_decode_traced
