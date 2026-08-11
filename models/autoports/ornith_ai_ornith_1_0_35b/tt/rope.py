# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Partial-RoPE cos/sin tables for the Ornith-1.0-35B ``full_attention`` layers.

Ornith rotates only ``head_dim * partial_rotary_factor`` = 256 * 0.25 = 64 dims of each
head; the remaining 192 pass through. HF builds the table through the interleaved M-RoPE
path (``Qwen3_5MoeTextRotaryEmbedding``), which for a text-only request receives the same
position id in all three (t, h, w) grids, so the interleave selects identical frequency
rows and the result reduces exactly to 1-D RoPE over ``[T, 64]``. This module builds that
1-D table (the reduction is asserted against HF in
``tests/test_functional_decoder.py::test_rope_matches_hf``).

Runtime contract
----------------
* ``prefill_forward(start_pos, seq_len)`` slices the device-resident table — device ops only.
* ``decode_forward(rot_idxs)`` gathers rows with ``ttnn.embedding`` from a device index
  tensor — device ops only, and trace-safe because the index tensor is an input buffer.

Both return ``cos``/``sin`` shaped ``[1, T, rope_dim]`` (prefill) or ``[1, B, rope_dim]``
(decode), which is what ``FunctionalDecoder._apply_partial_rope`` consumes after reshaping to
``[1, 1, T, rope_dim]`` / ``[B, 1, 1, rope_dim]``.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def build_rope_tables(rope_dim: int, max_context: int, theta: float, attention_scaling: float = 1.0):
    """Host cos/sin tables ``[max_context, rope_dim]`` (float32).

    Mirrors ``Qwen3_5MoeTextRotaryEmbedding.compute_default_rope_parameters`` +
    ``forward`` for text-only position ids.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    positions = torch.arange(max_context, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_context, rope_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_context, rope_dim]
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


class OrnithRope(LightweightModule):
    """Device-resident partial-RoPE tables with prefill-slice and decode-gather lookups."""

    def __init__(self, mesh_device, config, max_context: int, table_context: int | None = None, dtype=ttnn.bfloat16):
        """``max_context`` is the logical bound callers may request.

        ``table_context`` (default ``max_context``) is how many rows are actually built. The
        decoder passes a value rounded up to its prefill alignment, because a prefill block's
        *physical* (padded) window can run past the last logical position — those extra rows only
        ever multiply zero-padded activations, but the table has to be long enough to slice them.
        """
        self.device = mesh_device
        self.rope_dim = config.rope_dim
        # Rows actually built. Deliberately >= the logical context: the decoder validates the
        # logical bound itself, this only has to cover a padded physical window.
        self.table_rows = int(table_context or max_context)

        cos, sin = build_rope_tables(self.rope_dim, self.table_rows, config.rope_theta)
        # ROW_MAJOR [1, 1, table_rows, rope_dim] is what ttnn.embedding consumes as a table
        # and what ttnn.slice can carve prefill windows out of.
        self.cos_table = ttnn.from_torch(
            cos.reshape(1, 1, self.table_rows, self.rope_dim),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_table = ttnn.from_torch(
            sin.reshape(1, 1, self.table_rows, self.rope_dim),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def prefill_forward(self, start_pos: int, seq_len: int):
        """cos/sin ``[1, seq_len, rope_dim]`` for absolute positions ``[start_pos, start_pos+seq_len)``."""
        end = start_pos + seq_len
        if end > self.table_rows:
            raise ValueError(f"rope window [{start_pos}, {end}) exceeds the {self.table_rows}-row table")
        out = []
        for table in (self.cos_table, self.sin_table):
            t = ttnn.slice(table, [0, 0, start_pos, 0], [1, 1, end, self.rope_dim])
            t = ttnn.reshape(t, [1, seq_len, self.rope_dim])
            out.append(ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return out[0], out[1]

    def decode_forward(self, rot_idxs):
        """cos/sin ``[1, batch, rope_dim]`` gathered at the per-user positions in ``rot_idxs``.

        ``rot_idxs`` is a ``[1, batch]`` uint32 device tensor. Keeping it a device tensor is
        what makes decode traceable: trace replay only rewrites its contents.
        """
        cos = ttnn.embedding(rot_idxs, self.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_table, layout=ttnn.TILE_LAYOUT)
        return cos, sin
