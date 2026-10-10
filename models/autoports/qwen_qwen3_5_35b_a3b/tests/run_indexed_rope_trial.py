# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coherent partial-width indexed RoPE experiment with construction-only packing."""

from ..tt.fused_decoder import FusedDecoder, _FusedAttention
from . import run_layer


class _IndexedAttention(_FusedAttention):
    def _rope(self, x, cos, sin, decode):
        import ttnn

        if decode and not self.natural_layout:
            x = ttnn.transpose(x, 1, 2)
        output = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            cos,
            sin,
            self.rotation,
            0,
            0,
            rotary_dim=64,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.transpose(output, 1, 2) if decode else output


class _NaturalIndexedAttention(_IndexedAttention):
    """Same frozen forward, without the head-layout round trip before indexed RoPE."""

    def forward(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        residual=None,
    ):
        import ttnn
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
            _get_flexible_sdpa_program_config,
            _get_paged_sdpa_decode_program_config,
            _get_sdpa_compute_kernel_config,
            apply_rotary_pos_emb_ttnn,
        )

        batch, length = x.shape[0], x.shape[1]
        assert batch == 1 and self.use_paged_attention
        h, kv, d = self.config.num_heads, self.config.num_kv_heads, self.config.head_dim
        decode = length == 1
        ckc = self.compute_kernel_config_decode if decode else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
        heads_in_decode = decode and "decode_heads" in self.fusions
        if {"qkv", "heads", "decode_heads"}.intersection(self.fusions):
            packed = ttnn.linear(x, self.qkvg_weight, compute_kernel_config=ckc, memory_config=mc)
            qw, kw = h * d, kv * d
            gate = packed[..., qw + 2 * kw :]
            if heads_in_decode:
                query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
                    ttnn.reshape(packed[..., : qw + 2 * kw], (1, 1, 1, qw + 2 * kw)),
                    num_heads=h,
                    num_kv_heads=kv,
                    memory_config=self.cache_mem,
                )
                query, key, value = (ttnn.to_memory_config(t, mc) for t in (query, key, value))
            elif "heads" in self.fusions:
                query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                    packed[..., : qw + 2 * kw],
                    num_heads=h,
                    num_kv_heads=kv,
                    transpose_key=False,
                )
            else:
                query = ttnn.transpose(ttnn.reshape(packed[..., :qw], (1, length, h, d)), 1, 2)
                key = ttnn.transpose(ttnn.reshape(packed[..., qw : qw + kw], (1, length, kv, d)), 1, 2)
                value = ttnn.transpose(ttnn.reshape(packed[..., qw + kw : qw + 2 * kw], (1, length, kv, d)), 1, 2)
            query = ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps)
            key = ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps)
        else:
            qg = ttnn.linear(x, self.weights.q_proj, compute_kernel_config=ckc, memory_config=mc)
            query, gate = ttnn.chunk(ttnn.reshape(qg, (1, length, h, 2 * d)), 2, dim=-1)
            gate = ttnn.reshape(gate, (1, length, h * d))
            query = ttnn.transpose(ttnn.rms_norm(query, weight=self.weights.q_norm, epsilon=self.config.norm_eps), 1, 2)
            key = ttnn.linear(x, self.weights.k_proj, compute_kernel_config=ckc, memory_config=mc)
            key = ttnn.reshape(key, (1, length, kv, d))
            key = ttnn.transpose(ttnn.rms_norm(key, weight=self.weights.k_norm, epsilon=self.config.norm_eps), 1, 2)
            value = ttnn.linear(x, self.weights.v_proj, compute_kernel_config=ckc, memory_config=mc)
            value = ttnn.transpose(ttnn.reshape(value, (1, length, kv, d)), 1, 2)
        cos = ttnn.reshape(cos, (1, 1, length, cos.shape[-1]))
        sin = ttnn.reshape(sin, (1, 1, length, sin.shape[-1]))
        if "rope" in self.fusions:
            query, key = self._rope(query, cos, sin, decode), self._rope(key, cos, sin, decode)
        else:
            query, key = apply_rotary_pos_emb_ttnn(query, key, cos, sin)
            if decode and not heads_in_decode:
                query = ttnn.transpose(query, 1, 2)
                key = ttnn.transpose(key, 1, 2)
        if decode:
            key = ttnn.to_memory_config(key, self.cache_mem)
            value = ttnn.to_memory_config(
                value if heads_in_decode else ttnn.transpose(value, 1, 2),
                self.value_cache_mem if "cache" in self.fusions else self.cache_mem,
            )
            if "cache" in self.fusions:
                ttnn.experimental.paged_fused_update_cache(
                    self.paged_kv_cache_key,
                    key,
                    self.paged_kv_cache_value,
                    value,
                    update_idxs_tensor=position_tensor,
                    page_table=page_table,
                )
            else:
                for cache, tensor in ((self.paged_kv_cache_key, key), (self.paged_kv_cache_value, value)):
                    ttnn.experimental.paged_update_cache(
                        cache, tensor, update_idxs_tensor=position_tensor, page_table=page_table
                    )
            pc = (
                self._tile_decode_program_config
                if (page_table.shape[1] * self.paged_kv_cache_key.shape[2]) % 64
                else _get_paged_sdpa_decode_program_config(
                    self.device, self.paged_kv_cache_key.shape[0] * self.paged_kv_cache_key.shape[2]
                )
            )
            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG),
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                cur_pos_tensor=position_tensor,
                page_table_tensor=page_table,
                is_causal=True,
                scale=d**-0.5,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if "decode_concat" not in self.fusions:
                output = ttnn.transpose(output, 1, 2)
        else:
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_key, key, chunk_page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.paged_kv_cache_value, value, chunk_page_table, batch_idx=0)
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                query,
                self.paged_kv_cache_key,
                self.paged_kv_cache_value,
                page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                scale=d**-0.5,
                program_config=_get_flexible_sdpa_program_config(self.device),
                compute_kernel_config=_get_sdpa_compute_kernel_config(),
            )
        if decode and "decode_concat" in self.fusions:
            output = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(output, self.cache_mem), num_heads=h
            )
            output = ttnn.to_memory_config(output, mc)[:, :, :1, :]
            output = ttnn.reshape(output, (1, 1, h * d))
        else:
            output = ttnn.transformer.concatenate_heads(output)
        if "gate" in self.fusions:
            output = ttnn.multiply(gate, output, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        else:
            output = ttnn.multiply(output, ttnn.sigmoid(gate))
        if residual is not None:
            return self.residual_projection(output, residual, ckc, mc)
        return ttnn.linear(output, self.weights.o_proj, compute_kernel_config=ckc, memory_config=mc)


def main():
    import sys

    natural = "--indexed-native-layout" in sys.argv
    if natural:
        sys.argv.remove("--indexed-native-layout")
    original = FusedDecoder.from_state_dict.__func__

    def build(cls, *args, **kwargs):
        from dataclasses import replace

        import torch

        import ttnn

        kwargs["fusions"] = tuple(f for f in (kwargs.get("fusions") or cls.DEFAULT_FUSIONS) if f != "indexed_rope")
        model = original(cls, *args, **kwargs)
        assert model.config.is_full_attention_layer(model.layer_idx)
        mesh = kwargs["mesh_device"]

        def host(tensor):
            return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        def put(value, like):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=like.dtype,
                layout=like.layout,
                memory_config=like.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # Whole16-value exponent groups preserve the existing BFP8 coefficients.
        partial = torch.cat((torch.arange(16), torch.arange(32, 48), torch.arange(16, 32), torch.arange(48, 64)))
        perm = torch.cat((partial, torch.arange(64, 256)))
        attention_cls = _NaturalIndexedAttention if natural else _IndexedAttention
        attention = attention_cls.__new__(attention_cls)
        attention.__dict__.update(model.layer.attention.__dict__)
        attention.natural_layout = natural
        norm_weights = {}
        for name in ("q_norm", "k_norm"):
            old = getattr(attention.weights, name)
            norm_weights[name] = put(host(old)[..., perm], old)
        attention.weights = replace(attention.weights, **norm_weights)
        old = attention.qkvg_weight
        value = host(old)
        columns = torch.arange(value.shape[-1])
        for start, heads in ((0, 16), (4096, 2)):
            columns[start : start + heads * 256] = (start + torch.arange(heads)[:, None] * 256 + perm).reshape(-1)
        attention.qkvg_weight = put(value[..., columns], old)
        for name in ("_rope_cos", "_rope_sin"):
            old = getattr(model, name)
            setattr(model, name, put(host(old)[..., partial], old))
        rotation = torch.zeros(1, 1, 32, 32)
        rotation[..., torch.arange(16), torch.arange(16, 32)] = 1
        rotation[..., torch.arange(16, 32), torch.arange(16)] = -1
        attention.rotation = ttnn.from_torch(
            rotation,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        model.layer.attention = attention
        model.fusions = (*model.fusions, "indexed_rope_native_layout" if natural else "indexed_rope_trial")
        return model

    FusedDecoder.from_state_dict = classmethod(build)
    run_layer.main()


if __name__ == "__main__":
    main()
