# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functionally complete, single-device GPT-OSS 120B decoder layer.

The public contract is paged-only. ``prefill_forward`` consumes a device tensor
with logical shape ``[1, batch, sequence, 2880]`` and fills the supplied (or
owned) paged KV cache. ``decode_forward`` consumes
``[1, 1, batch, 2880]`` and a device-resident current-position tensor, making
the complete decode pass safe to capture and replay with a TTNN trace. Logical
sequence lengths need not be tile, page, or chunk aligned; TILE_LAYOUT padding
is an internal physical property and is not part of this API.

GPT-OSS alternates sliding-window and full-attention layers. Both kinds use the
same implementation; ``hf_config.layer_types[layer_idx]`` selects the attention
window. This functional stage intentionally requires a 1x1 mesh. P150x2 and
P150x4 parallelization belongs to the later multichip-decoder stage.

Host tensor conversion and HF-to-Meta Q/K swizzling occur only in
``from_state_dict``. The two forward methods and everything they invoke use
TTNN operations exclusively.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention import Attention, AttentionConfig
from models.demos.gpt_oss.tt.attention_configs import GPTOSSAttentionProgramConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts import ExpertConfig, Experts
from models.demos.gpt_oss.tt.rms_norm import RMSNorm
from models.demos.gpt_oss.tt.topk import TopKRouter
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig, rope_scaling_model_factory
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import RotarySetup

_SUPPORTED_LAYER_TYPES = {"sliding_attention", "full_attention"}


def _local_layer_state_dict(state_dict, layer_idx: int):
    """Return Meta-RoPE-formatted keys local to one HF decoder layer."""
    prefix = f"model.layers.{layer_idx}."
    if any(key.startswith(prefix) for key in state_dict):
        local = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    else:
        local = dict(state_dict)
    required = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "self_attn.o_proj.bias",
        "self_attn.sinks",
        "mlp.router.weight",
        "mlp.router.bias",
        "mlp.experts.gate_up_proj",
        "mlp.experts.gate_up_proj_bias",
        "mlp.experts.down_proj",
        "mlp.experts.down_proj_bias",
    }
    missing = sorted(required - local.keys())
    if missing:
        raise KeyError(f"Layer {layer_idx} state_dict is missing required GPT-OSS weights: {missing}")
    return convert_hf_qkv_to_meta_format(local, head_dim=64)


class _FunctionalMLP:
    """GPT-OSS router plus BF16 active experts for the functional PCC gate."""

    def __init__(self, mesh_device, hf_config, state_dict, ccl_manager, mesh_config, tensor_cache_path):
        self.router = TopKRouter(
            mesh_device,
            hf_config,
            substate(state_dict, "router"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
        )
        expert_config = ExpertConfig(
            intermediate_size=hf_config.intermediate_size,
            num_experts=hf_config.num_local_experts,
            hidden_size=hf_config.hidden_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            swiglu_limit=hf_config.swiglu_limit,
        )
        self.experts = Experts(
            mesh_device=mesh_device,
            config=expert_config,
            state_dict=substate(state_dict, "experts"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=GPTOSSProgramConfig(),
            weight_dtype=ttnn.bfloat16,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
        )

    def __call__(self, hidden_states, *, is_decode: bool):
        if is_decode and hidden_states.shape[-2] > 1:
            # The reusable active-expert implementation has a batch-one decode
            # kernel. Split the fixed decode batch on device and concatenate the
            # independent token-wise results; this remains trace-capture safe.
            user_hidden_states = ttnn.split(hidden_states, 1, dim=2)
            outputs = []
            for user_hidden in user_hidden_states:
                expert_indices, expert_weights = self.router(user_hidden, use_throughput_experts=False)
                expert_indices.deallocate(True)
                outputs.append(
                    self.experts(
                        user_hidden,
                        topk_expert_indices=None,
                        topk_expert_weights=expert_weights,
                        is_decode=True,
                    )
                )
            output = ttnn.concat(outputs, dim=2)
            for user_output in outputs:
                user_output.deallocate(True)
            return output

        expert_indices, expert_weights = self.router(hidden_states, use_throughput_experts=False)
        expert_indices.deallocate(True)
        return self.experts(
            hidden_states,
            topk_expert_indices=None,
            topk_expert_weights=expert_weights,
            is_decode=is_decode,
        )


class FunctionalDecoder(LightweightModule):
    """One GPT-OSS 120B decoder layer with paged prefill and traced decode."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context_length=None,
        page_size=64,
        tensor_cache_path=None,
    ):
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(
                "FunctionalDecoder is the single-device stage and requires mesh shape (1, 1); "
                "P150x2/P150x4 parallelization belongs to multichip-decoder"
            )
        if hf_config.hidden_size != 2880 or hf_config.head_dim != 64:
            raise ValueError("Expected the real openai/gpt-oss-120b hidden/head dimensions (2880, 64)")
        layer_type = hf_config.layer_types[layer_idx]
        if layer_type not in _SUPPORTED_LAYER_TYPES:
            raise ValueError(f"Unsupported GPT-OSS layer type {layer_type!r}")
        advertised_context = int(hf_config.max_position_embeddings)
        max_context_length = advertised_context if max_context_length is None else int(max_context_length)
        if not 1 <= max_context_length <= advertised_context:
            raise ValueError(f"max_context_length must be within [1, {advertised_context}], got {max_context_length}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        local_state = _local_layer_state_dict(state_dict, layer_idx)
        cache_root = str(Path(tensor_cache_path)) if tensor_cache_path is not None else None
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=1, ep=1, sp=1),
            prefill=ModeConfig(tp=1, ep=1, sp=1),
        )
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
        paged_attention_config = PagedAttentionConfig(
            block_size=page_size,
            max_num_blocks=max_batch_size * ((max_context_length + page_size - 1) // page_size),
        )
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=hf_config.sliding_window if layer_type == "sliding_attention" else None,
            max_seq_len=max_context_length,
            max_local_batch_size=max_batch_size,
            users_row_sharded=False,
        )
        rope_scaling_config = getattr(hf_config, "rope_scaling", None)
        rope_scaling = rope_scaling_model_factory(rope_scaling_config) if rope_scaling_config else None
        rope_theta = getattr(hf_config, "rope_theta", None) or getattr(hf_config, "default_theta", 150000.0)
        rope_setup = RotarySetup(
            device=mesh_device,
            batch_size=max_batch_size,
            head_dim=hf_config.head_dim,
            max_seq_len=max_context_length,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=ttnn.bfloat16,
        )
        attention = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(local_state, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=GPTOSSAttentionProgramConfig(),
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=rope_setup.get_both_trans_mats(),
            weight_dtype=ttnn.bfloat16,
            tensor_cache_path=get_cache_file_name(cache_root, "self_attn"),
        )
        return cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            page_size=page_size,
            rope_setup=rope_setup,
            input_layernorm=RMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "input_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "input_layernorm"),
                mesh_config=mesh_config,
            ),
            post_attention_layernorm=RMSNorm(
                mesh_device,
                hf_config,
                substate(local_state, "post_attention_layernorm"),
                tensor_cache_path=get_cache_file_name(cache_root, "post_attention_layernorm"),
                mesh_config=mesh_config,
            ),
            attention=attention,
            mlp=_FunctionalMLP(
                mesh_device,
                hf_config,
                substate(local_state, "mlp"),
                ccl_manager,
                mesh_config,
                get_cache_file_name(cache_root, "mlp"),
            ),
        )

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        layer_idx,
        layer_type,
        max_batch_size,
        max_context_length,
        page_size,
        rope_setup,
        input_layernorm,
        post_attention_layernorm,
        attention,
        mlp,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.max_batch_size = max_batch_size
        self.max_context_length = max_context_length
        self.page_size = page_size
        self.rope_setup = rope_setup
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = attention
        self.mlp = mlp

    @property
    def kv_cache(self):
        return self.self_attn.kv_cache

    def _forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache,
        is_decode,
        user_id,
        batch_size,
    ):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attention_out = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            position_idx=current_position,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
        )
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, attention_out, output_tensor=attention_out)
        residual.deallocate(True)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed, is_decode=is_decode)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        page_table,
        kv_cache=None,
        user_id=0,
        batch_size=1,
    ):
        if page_table is None:
            raise ValueError("FunctionalDecoder is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != batch_size:
            raise ValueError(
                "prefill requires [1, batch, sequence, hidden] input matching batch_size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1 or batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} is outside configured maximum {self.max_batch_size}")
        if hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                f"prefill hidden dimension must be {self.hf_config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        logical_tokens = hidden_states.shape[-2]
        if logical_tokens < 1 or logical_tokens > self.max_context_length:
            raise ValueError(
                f"prefill logical sequence must be within [1, {self.max_context_length}], got {logical_tokens}"
            )
        padded_tokens = ((logical_tokens + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        if len(position_embeddings) != 2 or any(rope.shape[-2] != logical_tokens for rope in position_embeddings):
            raise ValueError("prefill position_embeddings must be a cosine/sine pair of exact logical length")

        # The canonical attention and expert kernels consume their activations.
        # Clone/pad at the public boundary so callers retain a stable input
        # buffer that can be rewritten between trace replays.
        if padded_tokens == logical_tokens:
            working_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = position_embeddings
        else:
            padding = [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)]
            owned_hidden = ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_hidden = ttnn.pad(owned_hidden, padding, value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            working_rope = [
                ttnn.pad(
                    ttnn.clone(rope, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    [(0, 0), (0, 0), (0, padded_tokens - logical_tokens), (0, 0)],
                    value=0.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for rope in position_embeddings
            ]

        working_hidden = ttnn.reshape(
            working_hidden,
            (1, 1, batch_size * padded_tokens, self.hf_config.hidden_size),
        )
        output = self._forward(
            working_hidden,
            position_embeddings=working_rope,
            current_position=None,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=False,
            user_id=user_id,
            batch_size=batch_size,
        )
        if padded_tokens != logical_tokens:
            for rope in working_rope:
                rope.deallocate(True)
        output = ttnn.reshape(output, (1, batch_size, padded_tokens, self.hf_config.hidden_size))
        if padded_tokens != logical_tokens:
            output = ttnn.slice(
                output,
                starts=[0, 0, 0, 0],
                ends=[1, batch_size, logical_tokens, self.hf_config.hidden_size],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return output

    def decode_forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        current_position,
        page_table,
        kv_cache=None,
        batch_size=1,
    ):
        if page_table is None:
            raise ValueError("FunctionalDecoder is paged-only and requires page_table")
        if len(hidden_states.shape) != 4 or hidden_states.shape[0] != 1 or hidden_states.shape[1] != 1:
            raise ValueError(f"decode requires [1, 1, batch, hidden] input, got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-2] != batch_size or hidden_states.shape[-1] != self.hf_config.hidden_size:
            raise ValueError(
                "decode input must match batch_size and hidden size, "
                f"got {tuple(hidden_states.shape)} and batch_size={batch_size}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self.max_batch_size:
            raise ValueError(f"batch_size {batch_size} exceeds configured maximum {self.max_batch_size}")
        if current_position is None:
            raise ValueError("decode requires a device-resident current_position tensor")
        if current_position.shape[-1] < batch_size:
            raise ValueError("current_position is shorter than the decode batch")
        if len(position_embeddings) != 2 or any(rope.shape[1] < batch_size for rope in position_embeddings):
            raise ValueError("decode position_embeddings must be a cosine/sine pair covering the decode batch")
        if page_table.shape[-2] < batch_size:
            raise ValueError(f"page_table has {page_table.shape[-2]} rows for batch_size={batch_size}")
        return self._forward(
            ttnn.clone(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            position_embeddings=position_embeddings,
            current_position=current_position,
            page_table=page_table,
            kv_cache=self.kv_cache if kv_cache is None else kv_cache,
            is_decode=True,
            user_id=0,
            batch_size=batch_size,
        )

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
