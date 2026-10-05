# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY


class TtnnModernBertEmbeddings:
    """Token lookup followed by a weight-only LayerNorm; no positional table, RoPE carries position."""

    def __init__(self, parameters, config, device=None, policy=DEFAULT_POLICY):
        self.tok_embeddings = parameters["tok_embeddings"]
        self.norm = parameters["norm"]
        self.eps = config.norm_eps
        self.norm_config = policy.compute_config(device, "norm") if device is not None else None

    def __call__(self, input_ids):
        hidden = ttnn.embedding(input_ids, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)
        kw = {} if self.norm_config is None else {"compute_kernel_config": self.norm_config}
        normed = ttnn.layer_norm(hidden, weight=self.norm, epsilon=self.eps, **kw)
        ttnn.deallocate(hidden)
        return normed
