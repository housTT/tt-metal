# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A functional, one-layer Qwen3.5-35B-A3B decoder contract.

This composes Blackhole Qwen36 attention/MoE with exact GDN and HF norms. TT
imports are lazy so configuration and checkpoint-contract tests remain CPU
only and cannot discover a device.
"""

import contextvars
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from models.common.lightweightmodule import LightweightModule

DECODE_BATCH = contextvars.ContextVar("qwen35_decode_batch", default=None)


def decode_rows(length: int) -> bool:
    batch = DECODE_BATCH.get()
    return length == 1 or (batch is not None and length == batch["rows"])


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(config: Any) -> Any:
    text = _field(config, "text_config")
    if text is not None:
        return text
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if get_text_config is not None else config


@dataclass(frozen=True)
class FunctionalDecoderConfig:
    """The exact text-decoder fields consumed by the reused TT layer."""

    dim: int
    norm_eps: float
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    attention_type_list: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    moe_num_experts: int
    moe_top_k: int
    moe_intermediate_size: int
    moe_shared_intermediate_size: int
    moe_norm_topk_prob: bool
    moe_decoder_sparse_step: int
    mlp_only_layers: frozenset[int]
    num_devices: int = 1

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "FunctionalDecoderConfig":
        c = _text_config(hf_config)
        layer_types = tuple(_field(c, "layer_types") or ())
        n_layers = int(_field(c, "num_hidden_layers"))
        if len(layer_types) != n_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries; expected {n_layers}")
        return cls(
            dim=int(_field(c, "hidden_size")),
            norm_eps=float(_field(c, "rms_norm_eps")),
            n_layers=n_layers,
            n_heads=int(_field(c, "num_attention_heads")),
            n_kv_heads=int(_field(c, "num_key_value_heads")),
            head_dim=int(_field(c, "head_dim")),
            max_seq_len=int(_field(c, "max_position_embeddings")),
            attention_type_list=layer_types,
            linear_num_key_heads=int(_field(c, "linear_num_key_heads")),
            linear_num_value_heads=int(_field(c, "linear_num_value_heads")),
            linear_key_head_dim=int(_field(c, "linear_key_head_dim")),
            linear_value_head_dim=int(_field(c, "linear_value_head_dim")),
            linear_conv_kernel_dim=int(_field(c, "linear_conv_kernel_dim")),
            moe_num_experts=int(_field(c, "num_experts", 0) or 0),
            moe_top_k=int(_field(c, "num_experts_per_tok", 0) or 0),
            moe_intermediate_size=int(_field(c, "moe_intermediate_size", 0) or 0),
            moe_shared_intermediate_size=int(_field(c, "shared_expert_intermediate_size", 0) or 0),
            moe_norm_topk_prob=bool(_field(c, "norm_topk_prob", True)),
            moe_decoder_sparse_step=int(_field(c, "decoder_sparse_step", 1) or 1),
            mlp_only_layers=frozenset(_field(c, "mlp_only_layers", ()) or ()),
        )

    @property
    def linear_q_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_k_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_v_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    def layer_kind(self, layer_idx: int) -> str:
        self._check_layer(layer_idx)
        return self.attention_type_list[layer_idx]

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_kind(layer_idx) == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.layer_kind(layer_idx) == "linear_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        self._check_layer(layer_idx)
        return (
            self.moe_num_experts > 0
            and layer_idx not in self.mlp_only_layers
            and (layer_idx + 1) % self.moe_decoder_sparse_step == 0
        )

    def _check_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} outside [0, {self.n_layers})")


class FunctionalDecoder(LightweightModule):
    """One real Qwen3.5 decoder layer, with explicit prefill/decode entrypoints.

    Stage 1 is intentionally single-device.  It covers both checkpoint layer
    kinds (linear attention at layer 0 and full attention at layer 3), including
    the real 256-expert MoE.  Tensor-parallel construction requires the Qwen36
    architecture-specific program/memory configuration and is a later stage.
    """

    PREFILL_MULTIPLE = 128

    def __init__(self, config: FunctionalDecoderConfig, layer_idx: int, layer: Any):
        config._check_layer(layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.layer = layer
        self.batched_decode = False

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        tensor_cache_path: Optional[Path] = None,
        max_batch_size: int = 32,
        expert_precision: str = "inherited",
        replicated_mesh: bool = False,
    ) -> "FunctionalDecoder":
        config = FunctionalDecoderConfig.from_hf_config(hf_config)
        config._check_layer(layer_idx)
        device_count = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        if device_count != 1 and not replicated_mesh:
            raise NotImplementedError(
                f"Stage-1 FunctionalDecoder supports one TT device; received {device_count} devices"
            )

        # The selective loader returns raw HF names; the reused implementation
        # consumes its internal layers.N.* scheme and split convolution taps.
        if f"layers.{layer_idx}.input_layernorm.weight" in state_dict:
            remapped = dict(state_dict)
        else:
            from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

            remapped = remap_qwen36_state_dict(dict(state_dict))

        required = [
            f"layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            f"layers.{layer_idx}.mlp.gate.weight",
            f"layers.{layer_idx}.mlp.experts.gate_up_proj",
            f"layers.{layer_idx}.mlp.experts.down_proj",
        ]
        branch = (
            "self_attn.q_proj.weight" if config.is_full_attention_layer(layer_idx) else "linear_attn.qkv_proj.weight"
        )
        required.append(f"layers.{layer_idx}.{branch}")
        missing = [key for key in required if key not in remapped]
        if missing:
            raise KeyError(f"layer {layer_idx} is missing required tensors: {missing}")

        from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer

        cache = Path(tensor_cache_path) if tensor_cache_path is not None else None
        if replicated_mesh and config.is_deltanet_layer(layer_idx):
            layer = _replicated_deltanet_layer(mesh_device, config, remapped, layer_idx, cache)
        else:
            layer = Qwen36DecoderLayer(
                mesh_device=mesh_device,
                args=config,
                state_dict=remapped,
                layer_num=layer_idx,
                tensor_cache_path=cache,
            )
        layer.attention_norm = _FunctionalRMSNorm(
            mesh_device, remapped[f"layers.{layer_idx}.input_layernorm.weight"], config.norm_eps
        )
        layer.ffn_norm = _FunctionalRMSNorm(
            mesh_device, remapped[f"layers.{layer_idx}.post_attention_layernorm.weight"], config.norm_eps
        )
        if config.is_deltanet_layer(layer_idx):
            from .functional_gdn import FunctionalGatedDeltaNet

            prefix = f"layers.{layer_idx}.linear_attn."
            layer.attention = FunctionalGatedDeltaNet.from_existing(
                layer.attention, state_dict={k[len(prefix) :]: v for k, v in remapped.items() if k.startswith(prefix)}
            )
        else:
            from .functional_attention import FunctionalGatedAttention

            layer.attention = FunctionalGatedAttention.from_existing(layer.attention)
        if expert_precision == "bf16":
            import ttnn
            from models.demos.blackhole.qwen36.tt.moe.weights import load_expert_weights

            prefix = f"layers.{layer_idx}.mlp.experts."
            expert_state = {k[len(prefix) :]: v for k, v in remapped.items() if k.startswith(prefix)}
            experts = layer.feed_forward.experts
            experts.weights.gate_up_proj.deallocate(True)
            experts.weights.down_proj.deallocate(True)
            experts.weights = load_expert_weights(
                mesh_device, experts.config, expert_state, gate_up_dtype=ttnn.bfloat16, down_dtype=ttnn.bfloat16
            )
        elif expert_precision != "inherited":
            raise ValueError("expert_precision must be bf16 or inherited")
        layer.feed_forward.router = _FunctionalRouter(layer.feed_forward.router, mesh_device)
        layer.feed_forward = _RankThreeMoE(layer.feed_forward)
        result = cls(config=config, layer_idx=layer_idx, layer=layer)
        result._prepare_request_buffers(mesh_device, hf_config, max_batch_size)
        return result

    def _prepare_request_buffers(self, mesh_device, hf_config, max_batch_size):
        """Setup-only host construction. All persistent buffers precede trace capture."""
        import torch

        import ttnn

        if not 1 <= max_batch_size <= 32:
            raise ValueError("max_batch_size must be between 1 and 32")
        self.max_batch_size = max_batch_size
        self._states = []
        # Runtime scalar add patches offsets without changing its program signature.
        # Keep index vectors resident: public passes never upload host selectors.
        self._index_offsets = {
            count: ttnn.from_torch(
                torch.arange(count, dtype=torch.float32)[None],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for count in (1, 2, 4, 8, 16, 32, 128, 512)
        }
        if self.config.is_deltanet_layer(self.layer_idx):
            a = self.layer.attention
            self._states.append((a.recurrent_state, a.fused_conv_state))
            for _ in range(1, max_batch_size):
                self._states.append((ttnn.zeros_like(a.recurrent_state), ttnn.zeros_like(a.fused_conv_state)))
        else:
            c = _text_config(hf_config)
            rp = _field(c, "rope_parameters")
            if rp["rope_type"] != "default":
                raise ValueError("This checkpoint uses default partial RoPE")
            width = int(self.config.head_dim * rp["partial_rotary_factor"])
            freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, width, 2).float() / width))
            phase = torch.arange(self.config.max_seq_len).float()[:, None] * freq[None, :]
            phase = torch.cat((phase, phase), dim=-1)

            def put(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                return ttnn.from_torch(
                    x,
                    device=mesh_device,
                    dtype=dtype,
                    layout=layout,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )

            self._rope_cos = put(phase.cos(), layout=ttnn.ROW_MAJOR_LAYOUT)
            self._rope_sin = put(phase.sin(), layout=ttnn.ROW_MAJOR_LAYOUT)

    def _select_state(self, slot):
        if self._states:
            self.layer.attention.recurrent_state, self.layer.attention.fused_conv_state = self._states[slot]

    def _slots(self, batch, request_slots):
        slots = tuple(range(batch)) if request_slots is None else tuple(request_slots)
        if len(slots) != batch or len(set(slots)) != batch:
            raise ValueError("Each batch row needs one distinct request slot")
        if any(s < 0 or s >= self.max_batch_size for s in slots):
            raise ValueError("request slot outside configured capacity")
        return slots

    def _indices(self, start, length):
        import ttnn

        indices = ttnn.typecast(ttnn.add(self._index_offsets[length], float(start)), ttnn.uint32)
        return ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)

    def _rotary(self, positions, length):
        import ttnn

        indices = ttnn.reshape(ttnn.typecast(positions, ttnn.uint32), (1, length))
        return tuple(
            ttnn.reshape(ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT), (1, length, table.shape[-1]))
            for table in (self._rope_cos, self._rope_sin)
        )

    @classmethod
    def physical_prefill_length(cls, logical_length: int) -> int:
        if logical_length <= 0:
            raise ValueError("logical_length must be positive")
        return ((logical_length + cls.PREFILL_MULTIPLE - 1) // cls.PREFILL_MULTIPLE) * cls.PREFILL_MULTIPLE

    def bind_paged_kv_cache(self, key_cache: Any, value_cache: Any) -> None:
        import ttnn

        if not self.config.is_full_attention_layer(self.layer_idx):
            raise TypeError("linear-attention layers use recurrent state, not a paged KV cache")
        if tuple(key_cache.shape) != tuple(value_cache.shape) or len(key_cache.shape) != 4:
            raise ValueError("K/V caches must share [pages,kv_heads,page_size,head_dim]")
        if key_cache.shape[1] != self.config.n_kv_heads or key_cache.shape[3] != self.config.head_dim:
            raise ValueError("K/V cache head geometry differs from the target config")
        if key_cache.shape[2] % 32 or key_cache.shape[2] < 32:
            raise ValueError("K/V page size must be a positive multiple of 32")
        if any(
            t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            for t in (key_cache, value_cache)
        ):
            raise ValueError("Functional K/V caches require BF16 tiled DRAM storage")
        self.layer.attention.set_paged_kv_cache(key_cache, value_cache)

    def reset_request_state(self, request_slots=None) -> None:
        """Reset fresh requests outside capture. Explicit continuations must not reset."""
        if self.config.is_full_attention_layer(self.layer_idx):
            self.layer.attention.reset_cache()
        else:
            slots = range(self.max_batch_size) if request_slots is None else request_slots
            for slot in slots:
                self._select_state(slot)
                self.layer.attention.reset_state_inplace()

    def prefill_forward(
        self,
        hidden_states: Any,
        *,
        valid_len: Optional[int] = None,
        page_table: Any = None,
        start_pos: int = 0,
        request_slots: Optional[Sequence[int]] = None,
    ) -> Any:
        """Consume [B,T,H] device activations; return all logical rows [B,T,H].

        start_pos is the absolute cached-prefix length; state is retained until an
        explicit reset. Full attention requires an owned page table [B,blocks].
        Bounded chunks and token-wise tails never write padding into live state.
        Partial text RoPE and chunk planning are owned by this decoder.
        """
        import ttnn

        if len(hidden_states.shape) != 3:
            raise ValueError("hidden_states must have shape [B,T,H]")
        batch, physical_len, hidden = hidden_states.shape
        length = physical_len if valid_len is None else int(valid_len)
        if not 0 < length <= physical_len or hidden != self.config.dim:
            raise ValueError("invalid logical length or hidden dimension")
        if start_pos < 0 or start_pos + length > self.config.max_seq_len:
            raise ValueError("prefill exceeds advertised context")
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("full attention requires bound paged KV and page_table")
        if full and (
            page_table.shape[0] != batch
            or page_table.shape[1] * self.layer.attention.paged_kv_cache_key.shape[2] < start_pos + length
        ):
            raise ValueError("page table does not cover the requested batch/context")
        read_page_table = page_table
        if full and page_table.shape[1] % 8:
            # Chunk SDPA reads 32-byte sticks. Extra entries are outside the
            # logical causal prefix; pad on device without changing ownership.
            read_page_table = ttnn.pad(page_table, ((0, 0), (0, -page_table.shape[1] % 8)), 0)
        flat_input = ttnn.reshape(ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT), (batch * physical_len, hidden))
        output = ttnn.zeros(
            (batch, physical_len, hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=hidden_states.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for row, slot in enumerate(self._slots(batch, request_slots)):
            self._select_state(slot)
            pt = page_table[row : row + 1, :] if full else None
            read_pt = read_page_table[row : row + 1, :] if full else None
            block = self.layer.attention.paged_kv_cache_key.shape[2] if full else 128
            offset = 0
            while offset < length:
                absolute = start_pos + offset
                buckets = (512, 128) if full else (128,)
                count = next(
                    (
                        n
                        for n in buckets
                        if n % block == 0 and absolute % block == 0 and absolute % 128 == 0 and length - offset >= n
                    ),
                    1,
                )
                x = ttnn.embedding(
                    self._indices(row * physical_len + offset, count), flat_input, layout=ttnn.TILE_LAYOUT
                )
                kw = {}
                if full:
                    pos = self._indices(absolute, count)
                    c, s = self._rotary(pos, count)
                    if count == 1:
                        kw = dict(
                            cos=c,
                            sin=s,
                            position_tensor=ttnn.reshape(ttnn.typecast(pos, ttnn.int32), (1,)),
                            page_table=pt,
                        )
                    else:
                        kw = dict(
                            cos=ttnn.unsqueeze(c, 1),
                            sin=ttnn.unsqueeze(s, 1),
                            page_table=read_pt,
                            chunk_page_table=ttnn.gather(pt, 1, self._indices(absolute // block, count // block)),
                            chunk_start_idx_tensor=ttnn.reshape(
                                ttnn.typecast(self._indices(absolute, 1), ttnn.int32), (1,)
                            ),
                        )
                y = self.layer.forward(x, mode="decode" if count == 1 else "prefill", chunk_size=128, **kw)
                ttnn.experimental.slice_write(
                    ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT),
                    output,
                    (row, offset, 0),
                    (row + 1, offset + count, hidden),
                    (1, 1, 1),
                )
                offset += count
        tiled = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        return ttnn.reshape(tiled, (batch, length, hidden), tiled.padded_shape)

    def decode_forward(
        self,
        hidden_states: Any,
        *,
        position_tensor: Any = None,
        page_table: Any = None,
        request_slots: Optional[Sequence[int]] = None,
        packed: bool = False,
    ) -> Any:
        if (
            len(hidden_states.shape) != 3
            or int(hidden_states.shape[0 if packed else 1]) != 1
            or hidden_states.shape[2] != self.config.dim
        ):
            raise ValueError("decode_forward requires exactly one token")
        import ttnn

        batch = hidden_states.shape[1 if packed else 0]
        rows = []
        full = self.config.is_full_attention_layer(self.layer_idx)
        if full and (position_tensor is None or page_table is None or not self.layer.attention.use_paged_attention):
            raise ValueError("paged decode requires device current positions and page table")
        if full and (page_table.shape[0] != batch or position_tensor.shape[0] != batch):
            raise ValueError("decode metadata must have one row/position per request")
        slots = self._slots(batch, request_slots)
        if self.batched_decode and len(slots) > 1:
            return self._decode_batched(hidden_states, slots, full, position_tensor, page_table, packed)
        if packed:
            raise ValueError("packed decode input requires batched decode with more than one row")
        for row, slot in enumerate(slots):
            self._select_state(slot)
            kw = {}
            if full:
                pos = position_tensor[row : row + 1]
                c, s = self._rotary(pos, 1)
                kw = dict(cos=c, sin=s, position_tensor=pos, page_table=page_table[row : row + 1, :])
            x = ttnn.to_layout(hidden_states[row : row + 1, :, :], ttnn.TILE_LAYOUT)
            rows.append(self.layer.forward(x, mode="decode", **kw))
        if len(rows) == 1:
            return rows[0]
        return ttnn.concat([ttnn.to_memory_config(r, ttnn.L1_MEMORY_CONFIG) if r.is_sharded() else r for r in rows], dim=0)

    def _decode_batched(self, hidden_states, slots, full, position_tensor, page_table, packed=False):
        import ttnn

        rows, hidden = len(slots), hidden_states.shape[2]
        kw = {}
        if full:
            cos, sin = self._rotary(position_tensor, rows)
            kw = dict(cos=cos, sin=sin, position_tensor=position_tensor, page_table=page_table)
        states = [self._states[slot] for slot in slots] if self._states else None
        token = DECODE_BATCH.set(dict(rows=rows, states=states))
        try:
            x = hidden_states if packed else ttnn.reshape(ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT),
                                                          (1, rows, hidden))
            y = self.layer.forward(x, mode="decode", **kw)
        finally:
            DECODE_BATCH.reset(token)
        if y.is_sharded():
            y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
        return y if packed else ttnn.reshape(y, (rows, 1, hidden))

    def forward(self, hidden_states: Any, *, mode: str, **kwargs: Any) -> Any:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


def _replicated_deltanet_layer(mesh_device, config, state_dict, layer_idx, cache):
    from types import SimpleNamespace

    from models.demos.blackhole.qwen36.tt.gdn import GDNConfig
    from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
    from models.demos.blackhole.qwen36.tt.moe import MoEConfig, Qwen36MoE
    from models.demos.blackhole.qwen36.utils.substate import substate

    layer = Qwen36DecoderLayer.__new__(Qwen36DecoderLayer)
    layer.layer_num, layer.device, layer.args, layer.tt_ccl = layer_idx, mesh_device, config, None
    layer.num_devices, layer.is_full_attention = 1, False
    layer.attention = SimpleNamespace(device=mesh_device, cfg=GDNConfig.from_args(config))
    layer.feed_forward = Qwen36MoE(mesh_device, MoEConfig.from_args(config), substate(state_dict, f"layers.{layer_idx}.mlp"),
                                   cache / f"layers.{layer_idx}" if cache is not None else None, args=config, tt_ccl=None)
    return layer


class _RankThreeMoE:
    """Bridge the single-device attention [B,T,H] and sparse MoE [1,1,S,H]."""

    def __init__(self, module):
        self.module = module

    def forward(self, x, mode="decode"):
        import ttnn

        batch, seq, hidden = x.shape
        y = self.module.forward(ttnn.reshape(x, (1, 1, batch * seq, hidden)), mode=mode)
        return ttnn.reshape(y, (batch, seq, hidden))


class _FunctionalRMSNorm:
    """HF computes both normalization and the zero-centered weight in FP32."""

    def __init__(self, mesh_device, weight, eps):
        import ttnn

        self.eps = eps
        self.weight = ttnn.from_torch(
            (1 + weight.float()).reshape(1, 1, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def __call__(self, x, *, mode=None, norm_config=None):
        import ttnn

        normalized = ttnn.rms_norm(
            ttnn.typecast(x, ttnn.float32),
            epsilon=self.eps,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.typecast(ttnn.multiply(normalized, self.weight), ttnn.bfloat16)


class _FunctionalRouter:
    """HF keeps softmax, top-k and renormalization FP32, then casts scores BF16."""

    def __init__(self, router, mesh_device):
        import ttnn

        self.__dict__.update(router.__dict__)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )

    def __call__(self, hidden_states):
        import ttnn

        logits = ttnn.linear(hidden_states, self.proj_weight, compute_kernel_config=self.compute_kernel_config)
        probs = ttnn.softmax(ttnn.typecast(logits, ttnn.float32), dim=-1)
        values, indices = ttnn.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            values = ttnn.div(values, ttnn.sum(values, dim=-1, keepdim=True))
        values = ttnn.typecast(values, ttnn.bfloat16)
        return ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=indices, src=values)
