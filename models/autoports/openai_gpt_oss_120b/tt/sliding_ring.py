# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bounded KV rings for the sliding-window attention layers.

A sliding layer attends to the last 128 positions, so it keeps only
SLIDING_RING_TOKENS positions per device slot: the paged-cache kernels take
``cache_position_modulo`` and address position ``p`` at ring slot
``p mod SLIDING_RING_TOKENS`` through the first ring blocks of the layer's page
table. The ring must span at least two decode K chunks (128 tokens each) so the
two chunks the window can touch never alias the same physical tiles.
"""

import os

PAGE_SIZE = 64
SLIDING_RING_TOKENS = 256
SLIDING_RING_BLOCKS = SLIDING_RING_TOKENS // PAGE_SIZE
ENV_SLIDING_RING = "GPT_OSS_120B_SLIDING_RING"


def sliding_ring_enabled() -> bool:
    return os.environ.get(ENV_SLIDING_RING, "1") != "0"


def ring_modulo_for_layer(layer_type: str) -> int | None:
    if layer_type == "sliding_attention" and sliding_ring_enabled():
        return SLIDING_RING_TOKENS
    return None
