# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the complete prior decoder contract against FusedDecoder.

The autouse fixture binds the shared harness to the fused class, checks every
constructed instance, and rejects use of the functional block implementation.
"""

import pytest

from ..tt.functional_decoder import FunctionalDecoder
from ..tt.fused_decoder import FusedDecoder as RuntimeFusedDecoder
from . import test_functional_decoder as H
from .test_contract_extensions import (
    test_native_context_decode_oracle,
    test_trace_changed_page_table,
    test_trace_repeated_identical_state,
    test_unaligned_continuation_to_capacity,
)
from .test_functional_decoder import (
    test_batched_decode_ragged_positions,
    test_batched_prefill_decode_pcc,
    test_decode_pcc,
    test_determinism_repeated_inputs,
    test_forward_with_poisoned_free_pool,
    test_full_context_chunk_size_invariance,
    test_full_context_prefill_and_decode,
    test_long_context_pcc,
    test_no_host_fallback_in_forward,
    test_perf_decode_traced,
    test_perf_prefill,
    test_permuted_page_table,
    test_prefill_continuation,
    test_prefill_pcc,
    test_real_weights_pcc,
    test_synthetic_weights_pcc,
    test_traced_decode_pcc,
    test_unaligned_max_context,
)
from .test_fusion_equivalence import selected_candidate

__all__ = [
    "H",
    "test_prefill_pcc",
    "test_decode_pcc",
    "test_batched_prefill_decode_pcc",
    "test_batched_decode_ragged_positions",
    "test_permuted_page_table",
    "test_prefill_continuation",
    "test_unaligned_max_context",
    "test_determinism_repeated_inputs",
    "test_forward_with_poisoned_free_pool",
    "test_no_host_fallback_in_forward",
    "test_traced_decode_pcc",
    "test_real_weights_pcc",
    "test_synthetic_weights_pcc",
    "test_full_context_prefill_and_decode",
    "test_full_context_chunk_size_invariance",
    "test_long_context_pcc",
    "test_perf_prefill",
    "test_perf_decode_traced",
    "test_trace_changed_page_table",
    "test_trace_repeated_identical_state",
    "test_native_context_decode_oracle",
    "test_unaligned_continuation_to_capacity",
]


pytestmark = H.pytestmark


@pytest.fixture(autouse=True)
def fused_path(monkeypatch):
    FusedDecoder = selected_candidate()
    original_build = FusedDecoder.from_state_dict
    built = []

    def build(cls, *args, **kwargs):
        decoder = original_build(*args, **kwargs)
        assert type(decoder) is FusedDecoder
        built.append(decoder.kind)
        return decoder

    def forbidden(*args, **kwargs):
        raise AssertionError("functional block fallback in fused test")

    monkeypatch.setattr(H, "FunctionalDecoder", FusedDecoder)
    monkeypatch.setattr(FusedDecoder, "from_state_dict", classmethod(build))
    methods = (
        ("_block", "_attention_prefill", "_attention_decode", "_gdn_prefill", "_gdn_decode")
        if FusedDecoder is RuntimeFusedDecoder
        else ("_block",)
    )
    for method in methods:
        monkeypatch.setattr(FunctionalDecoder, method, forbidden)
    yield
    assert built, "test did not construct the fused decoder"
