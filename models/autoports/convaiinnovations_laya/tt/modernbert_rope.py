# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_PORT, rotary_shard_config


def rope_cos_sin(head_dim: int, theta: float, seq_len: int):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class TtnnModernBertRotary:
    """One cos/sin cache pair per layer type for one sequence length; sharded round trip above a row threshold."""

    def __init__(
        self,
        config,
        device,
        seq_len,
        batch_size=1,
        dtype=ttnn.bfloat16,
        mesh_mapper=None,
        port=DEFAULT_PORT,
        attention_memory=None,
    ):
        self.seq_len = seq_len
        head_dim = config.hidden_size // config.num_attention_heads
        self._shard_mem = rotary_shard_config((batch_size, config.num_attention_heads, seq_len, head_dim), port)
        self._interleaved = attention_memory if attention_memory is not None else ttnn.DRAM_MEMORY_CONFIG
        self.thetas = {lt: config.rope_parameters[lt]["rope_theta"] for lt in set(config.layer_types)}
        kw = {"device": device}
        if mesh_mapper is not None:
            kw["mesh_mapper"] = mesh_mapper
        self.caches = {}
        for layer_type, theta in self.thetas.items():
            cos, sin = rope_cos_sin(head_dim, theta, seq_len)
            self.caches[layer_type] = tuple(
                ttnn.from_torch(t[None, None].contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, **kw)
                for t in (cos, sin)
            )

    @property
    def sharded(self) -> bool:
        return self._shard_mem is not None

    def __call__(self, tensor, layer_type):
        cos, sin = self.caches[layer_type]
        if self._shard_mem is None:
            return ttnn.experimental.rotary_embedding_hf(
                tensor, cos, sin, is_decode_mode=False, memory_config=self._interleaved
            )
        sh = ttnn.to_memory_config(tensor, self._shard_mem)
        out = ttnn.experimental.rotary_embedding_hf(sh, cos, sin, is_decode_mode=False, memory_config=self._shard_mem)
        ttnn.deallocate(sh)
        il = ttnn.to_memory_config(out, self._interleaved)
        ttnn.deallocate(out)
        return il

    def deallocate(self):
        for cos, sin in self.caches.values():
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        self.caches = {}
