# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for Qwen/Qwen3-8B on one Blackhole chip.

Qwen3-8B has one decoder layer kind: pre-norm residual block with full causal
grouped-query attention (32 query heads, 8 key/value heads, head_dim 128,
per-head RMSNorm on Q and K before RoPE, rotary theta 1e6) followed by a
SwiGLU MLP (hidden 4096, intermediate 12288). No biases, no sliding window.

Public contract (all tensors are device tensors on the 1x1 mesh):

Prefill: ``prefill_forward(hidden_states, *, page_table, kv_cache, user_id=0,
start_pos=0, seq_lens=None)``

* ``hidden_states``: ``[B, 1, S_max, 4096]`` bfloat16 TILE DRAM. Row ``b``
  holds ``seq_lens[b]`` valid new tokens at indices ``[0, seq_lens[b])``;
  trailing rows are ignored. ``S_max`` may be any positive length (no tile,
  page, or chunk alignment is required from the caller).
* ``start_pos``: int or per-row list. Absolute cache position of the first
  new token of each row. ``0`` is a fresh request. A nonzero value is an
  explicit continuation: the cache must already hold positions
  ``[0, start_pos)`` for that row's page-table entry.
* ``user_id``: int or per-row list. Page-table row used for each batch row.
* ``page_table``: ``[num_users, blocks_per_user]`` int32 ROW_MAJOR, absolute
  block ids into ``kv_cache``. It must cover
  ``round_up(start_pos + seq_len, 32)`` positions for every row.
* ``kv_cache``: ``(key_cache, value_cache)`` each
  ``[num_blocks, 8, block_size, 128]`` TILE DRAM in the cache dtype.
  ``block_size`` is a multiple of 32.
* Returns ``[B, 1, S_max, 4096]`` bfloat16 TILE DRAM. Row ``b`` is valid at
  ``[0, seq_lens[b])``; the rest is zero padding.

The layer processes each row in bounded physical chunks of
``prefill_chunk_size`` tokens. Padding is internal: a tile-unaligned
``start_pos`` is handled by front-padding the physical window to the previous
tile boundary, writing the unaligned leading cache rows token by token so
live prefix rows are never overwritten, and slicing the padding away before
returning. Every chunk attends to the whole cached prefix plus its own causal
window through ``chunked_scaled_dot_product_attention``.

Decode: ``decode_forward(hidden_states, *, current_pos, page_table, kv_cache)``

* ``hidden_states``: ``[1, 1, B, 4096]`` bfloat16 TILE DRAM, one token per
  user, ``1 <= B <= 32``.
* ``current_pos``: ``[B]`` int32 ROW_MAJOR DRAM. The absolute position of the
  token being decoded for each user. RoPE indices are derived from it on
  device, so one tensor drives cache update, attention length, and rotary.
* ``page_table``: ``[B, blocks_per_user]`` int32 ROW_MAJOR.
* Returns ``[1, 1, B, 4096]`` bfloat16 TILE DRAM.

``decode_forward`` is trace-safe: with stable ``hidden_states``,
``current_pos``, ``page_table`` and ``kv_cache`` allocations, a caller may
capture it once and replay it with ``ttnn.execute_trace`` after refreshing the
contents of those tensors.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE_SIZE = 32
PREFILL_CHUNK_SIZE = 4096
PREFILL_SDPA_MAX_CHUNK = 256
MODEL_TYPE = "qwen3"
MAX_DECODE_BATCH = 32


def _float32(value: float) -> float:
    """Round a Python float to the nearest float32 value.

    The SDPA bindings take a 32-bit ``scale`` and reject doubles that do not
    round-trip through float32, so ``head_dim ** -0.5`` must be rounded first.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class DecoderGeometry:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @property
    def qkv_width(self) -> int:
        return self.head_dim * (self.num_heads + 2 * self.num_kv_heads)

    @property
    def scale(self) -> float:
        return _float32(self.head_dim**-0.5)


def validate_hf_config(hf_config: Any, layer_idx: int) -> DecoderGeometry:
    """Derive the layer geometry from the HF config and reject unsupported variants."""
    model_type = getattr(hf_config, "model_type", None)
    if model_type != MODEL_TYPE:
        raise ValueError(f"expected model_type {MODEL_TYPE!r}, got {model_type!r}")
    num_layers = int(hf_config.num_hidden_layers)
    if not 0 <= layer_idx < num_layers:
        raise ValueError(f"layer_idx {layer_idx} outside [0, {num_layers})")
    if getattr(hf_config, "attention_bias", False):
        raise ValueError("Qwen3-8B attention_bias=False is the only supported value")
    if hf_config.hidden_act != "silu":
        raise ValueError(f"unsupported hidden_act {hf_config.hidden_act!r}")
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is not None and layer_types[layer_idx] != "full_attention":
        raise ValueError(f"layer {layer_idx} is {layer_types[layer_idx]!r}; only full_attention is supported")
    if getattr(hf_config, "sliding_window", None) is not None and getattr(hf_config, "use_sliding_window", False):
        raise ValueError("sliding-window attention is not supported")
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_type = rope_parameters.get("rope_type", "default")
        rope_theta = float(rope_parameters["rope_theta"])
    else:
        rope_type = "default"
        rope_theta = float(hf_config.rope_theta)
    if rope_type != "default":
        raise ValueError(f"unsupported rope_type {rope_type!r}")
    head_dim = int(getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads)
    num_heads = int(hf_config.num_attention_heads)
    num_kv_heads = int(hf_config.num_key_value_heads)
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
    if head_dim % 64 != 0:
        raise ValueError("rotary_embedding_hf requires head_dim to be a multiple of 64")
    return DecoderGeometry(
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(hf_config.intermediate_size),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rope_theta=rope_theta,
        max_position_embeddings=int(hf_config.max_position_embeddings),
    )


@dataclass(frozen=True)
class PrefillChunk:
    """One physical prefill chunk of a single row.

    ``phys_start`` is tile aligned and is the absolute position of the first
    physical row. ``lead`` physical rows precede the first valid token
    (nonzero only for a tile-unaligned ``start_pos``). ``phys_len`` is a
    multiple of 32. ``abs_start`` is the absolute position of the first valid
    token and ``logical_len`` the number of valid tokens in the chunk.
    """

    phys_start: int
    phys_len: int
    lead: int
    abs_start: int
    logical_len: int
    input_offset: int

    @property
    def abs_end(self) -> int:
        return self.abs_start + self.logical_len


def plan_prefill_window(start_pos: int, seq_len: int) -> tuple[int, int, int]:
    """Return ``(phys_base, lead, phys_total)`` for one prefill row."""
    if seq_len < 1:
        raise ValueError("prefill requires at least one logical token")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    lead = start_pos % TILE_SIZE
    phys_base = start_pos - lead
    phys_total = round_up(lead + seq_len, TILE_SIZE)
    return phys_base, lead, phys_total


def plan_prefill_chunks(start_pos: int, seq_len: int, chunk_size: int) -> list[PrefillChunk]:
    """Split one row's physical window into bounded tile-aligned chunks."""
    if chunk_size < TILE_SIZE or chunk_size % TILE_SIZE != 0:
        raise ValueError("chunk_size must be a positive multiple of 32")
    phys_base, _, phys_total = plan_prefill_window(start_pos, seq_len)
    chunks: list[PrefillChunk] = []
    offset = 0
    while offset < phys_total:
        phys_len = min(chunk_size, phys_total - offset)
        phys_start = phys_base + offset
        abs_lo = max(phys_start, start_pos)
        abs_hi = min(phys_start + phys_len, start_pos + seq_len)
        chunks.append(
            PrefillChunk(
                phys_start=phys_start,
                phys_len=phys_len,
                lead=abs_lo - phys_start,
                abs_start=abs_lo,
                logical_len=max(0, abs_hi - abs_lo),
                input_offset=offset,
            )
        )
        offset += phys_len
    return chunks


def plan_cache_writes(abs_start: int, abs_end: int, block_size: int) -> tuple[list[int], int | None]:
    """Return ``(token_positions, fill_start)`` for writing ``[abs_start, abs_end)``.

    ``token_positions`` are written one row at a time with ``paged_update_cache``
    because they share a cache block with live rows that precede ``abs_start``.
    ``fill_start`` is the first block-aligned position written with whole
    tiles by ``paged_fill_cache`` (or ``None`` if the range ends first).
    """
    if block_size < TILE_SIZE or block_size % TILE_SIZE != 0:
        raise ValueError("block_size must be a positive multiple of 32")
    if abs_end <= abs_start:
        return [], None
    first_block_boundary = round_up(abs_start, block_size)
    token_positions = list(range(abs_start, min(first_block_boundary, abs_end)))
    fill_start = first_block_boundary if first_block_boundary < abs_end else None
    return token_positions, fill_start


def prefill_sdpa_chunk_size(chunk_start: int, phys_len: int, kv_extent: int) -> int:
    """Largest SDPA chunk (<= 256) that divides ``chunk_start`` and keeps the padded K window inside the page table.

    The chunked SDPA kernel pads ``chunk_start + Sq`` up to a multiple of its chunk size
    and follows the page table that far. Entries past the row end are read from stale
    memory, and non-finite data there corrupts every query row of the chunk even though
    those keys are causally masked. The chunk is chosen so
    ``chunk_start + round_up(phys_len, chunk) <= kv_extent`` always holds.
    """
    for chunk in (256, 128, 64, 32):
        if chunk_start % chunk == 0 and chunk_start + round_up(phys_len, chunk) <= kv_extent:
            return chunk
    raise ValueError(f"page table extent {kv_extent} cannot hold chunk [{chunk_start}, {chunk_start + phys_len})")


def decode_sdpa_k_chunk_size(kv_extent: int) -> int:
    """Largest K chunk (<= 128) that divides the page-table extent so decode never reads past the row."""
    for chunk in (128, 64, 32):
        if kv_extent % chunk == 0:
            return chunk
    raise ValueError(f"page table extent {kv_extent} is not a multiple of 32")


@dataclass(frozen=True)
class DecoderWeights:
    qkv: ttnn.Tensor
    o_proj: ttnn.Tensor
    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor
    input_ln: ttnn.Tensor
    post_attn_ln: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor
    rope_cos: ttnn.Tensor
    rope_sin: ttnn.Tensor
    rope_cos_table: ttnn.Tensor
    rope_sin_table: ttnn.Tensor


def _strip_layer_prefix(state_dict: dict, layer_idx: int) -> dict:
    prefix = f"model.layers.{layer_idx}."
    prefixed = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if prefixed:
        return prefixed
    return dict(state_dict)


def _replicate_mapper(mesh_device: ttnn.MeshDevice):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def hf_rope_tables(geometry: DecoderGeometry):
    """Return HF rotate-half cos/sin tables ``[max_pos, head_dim]`` in float32."""
    import torch

    dim = geometry.head_dim
    inv_freq = 1.0 / (geometry.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    positions = torch.arange(geometry.max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class FunctionalDecoder(LightweightModule):
    """One Qwen3-8B decoder layer with paged prefill and traced paged decode."""

    def __init__(
        self,
        *,
        geometry: DecoderGeometry,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        weights: DecoderWeights,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
    ) -> None:
        if prefill_chunk_size % PREFILL_SDPA_MAX_CHUNK != 0 or prefill_chunk_size <= 0:
            raise ValueError(f"prefill_chunk_size must be a positive multiple of {PREFILL_SDPA_MAX_CHUNK}")
        self.geometry = geometry
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.weights = weights
        self.activation_dtype = activation_dtype
        self.prefill_chunk_size = prefill_chunk_size
        self.eps = geometry.rms_norm_eps
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.prefill_sdpa_grid = grid
        self.decode_sdpa_grid = ttnn.CoreCoord(min(8, grid.x), min(8, grid.y))
        self._decode_sdpa_program_configs: dict[int, ttnn.SDPAProgramConfig] = {}

    def _decode_sdpa_program_config(self, kv_extent: int) -> ttnn.SDPAProgramConfig:
        config = self._decode_sdpa_program_configs.get(kv_extent)
        if config is None:
            config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.decode_sdpa_grid,
                q_chunk_size=32,
                k_chunk_size=decode_sdpa_k_chunk_size(kv_extent),
                exp_approx_mode=False,
            )
            self._decode_sdpa_program_configs[kv_extent] = config
        return config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        **_: Any,
    ) -> "FunctionalDecoder":
        """Build the layer from canonical HF checkpoint keys.

        ``state_dict`` may use the full ``model.layers.{layer_idx}.`` prefix or
        the layer-local keys (``self_attn.q_proj.weight`` and so on).
        """
        import torch

        geometry = validate_hf_config(hf_config, layer_idx)
        local = _strip_layer_prefix(state_dict, layer_idx)
        required = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
        missing = [k for k in required if k not in local]
        if missing:
            raise KeyError(f"layer {layer_idx} state dict is missing {missing}")
        expected_shapes = {
            "self_attn.q_proj.weight": (geometry.num_heads * geometry.head_dim, geometry.hidden_size),
            "self_attn.k_proj.weight": (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
            "self_attn.v_proj.weight": (geometry.num_kv_heads * geometry.head_dim, geometry.hidden_size),
            "self_attn.o_proj.weight": (geometry.hidden_size, geometry.num_heads * geometry.head_dim),
            "self_attn.q_norm.weight": (geometry.head_dim,),
            "self_attn.k_norm.weight": (geometry.head_dim,),
            "mlp.gate_proj.weight": (geometry.intermediate_size, geometry.hidden_size),
            "mlp.up_proj.weight": (geometry.intermediate_size, geometry.hidden_size),
            "mlp.down_proj.weight": (geometry.hidden_size, geometry.intermediate_size),
            "input_layernorm.weight": (geometry.hidden_size,),
            "post_attention_layernorm.weight": (geometry.hidden_size,),
        }
        for key, shape in expected_shapes.items():
            if tuple(local[key].shape) != shape:
                raise ValueError(f"{key} has shape {tuple(local[key].shape)}, expected {shape}")

        cache_dir = Path(tensor_cache_path) if tensor_cache_path is not None else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        def as_tt(name: str, tensor: torch.Tensor, dtype: ttnn.DataType, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
            kwargs = {}
            if cache_dir is not None:
                kwargs["cache_file_name"] = str(cache_dir / f"layer{layer_idx}_{name}")
            return ttnn.as_tensor(
                tensor.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(mesh_device),
                **kwargs,
            )

        with torch.no_grad():
            qkv = torch.cat(
                [
                    local["self_attn.q_proj.weight"].to(torch.float32).T,
                    local["self_attn.k_proj.weight"].to(torch.float32).T,
                    local["self_attn.v_proj.weight"].to(torch.float32).T,
                ],
                dim=-1,
            )
            o_proj = local["self_attn.o_proj.weight"].to(torch.float32).T
            gate_proj = local["mlp.gate_proj.weight"].to(torch.float32).T
            up_proj = local["mlp.up_proj.weight"].to(torch.float32).T
            down_proj = local["mlp.down_proj.weight"].to(torch.float32).T
            norm_shape = (1, 1, 1, geometry.hidden_size)
            head_norm_shape = (1, 1, 1, geometry.head_dim)
            cos, sin = hf_rope_tables(geometry)
            weights = DecoderWeights(
                qkv=as_tt("qkv", qkv, weight_dtype),
                o_proj=as_tt("o_proj", o_proj, weight_dtype),
                gate_proj=as_tt("gate_proj", gate_proj, weight_dtype),
                up_proj=as_tt("up_proj", up_proj, weight_dtype),
                down_proj=as_tt("down_proj", down_proj, weight_dtype),
                input_ln=as_tt(
                    "input_ln", local["input_layernorm.weight"].to(torch.float32).reshape(norm_shape), ttnn.bfloat16
                ),
                post_attn_ln=as_tt(
                    "post_attn_ln",
                    local["post_attention_layernorm.weight"].to(torch.float32).reshape(norm_shape),
                    ttnn.bfloat16,
                ),
                q_norm=as_tt(
                    "q_norm", local["self_attn.q_norm.weight"].to(torch.float32).reshape(head_norm_shape), ttnn.bfloat16
                ),
                k_norm=as_tt(
                    "k_norm", local["self_attn.k_norm.weight"].to(torch.float32).reshape(head_norm_shape), ttnn.bfloat16
                ),
                rope_cos=as_tt("rope_cos", cos.unsqueeze(0).unsqueeze(0), ttnn.bfloat16),
                rope_sin=as_tt("rope_sin", sin.unsqueeze(0).unsqueeze(0), ttnn.bfloat16),
                rope_cos_table=as_tt("rope_cos_table", cos, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                rope_sin_table=as_tt("rope_sin_table", sin, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
            )
        return cls(
            geometry=geometry,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            activation_dtype=activation_dtype,
            prefill_chunk_size=prefill_chunk_size,
        )

    @staticmethod
    def allocate_kv_cache(
        mesh_device: ttnn.MeshDevice,
        *,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Allocate a zeroed paged K/V cache directly on device."""
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        return tuple(
            ttnn.moreh_full(
                shape, 0.0, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for _ in range(2)
        )

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int | Sequence[int] = 0,
        start_pos: int | Sequence[int] = 0,
        seq_lens: Sequence[int] | None = None,
    ) -> ttnn.Tensor:
        batch = hidden_states.shape[0]
        s_max = hidden_states.shape[-2]
        if batch < 1:
            raise ValueError("prefill requires at least one batch row")
        user_ids = [int(user_id)] * batch if isinstance(user_id, int) else [int(u) for u in user_id]
        starts = [int(start_pos)] * batch if isinstance(start_pos, int) else [int(s) for s in start_pos]
        lens = [int(s_max)] * batch if seq_lens is None else [int(n) for n in seq_lens]
        if not (len(user_ids) == len(starts) == len(lens) == batch):
            raise ValueError("user_id, start_pos and seq_lens must have one entry per batch row")
        for row in range(batch):
            if not 1 <= lens[row] <= s_max:
                raise ValueError(f"row {row}: seq_len {lens[row]} outside [1, {s_max}]")
            if starts[row] + lens[row] > self.geometry.max_position_embeddings:
                raise ValueError(
                    f"row {row}: start_pos {starts[row]} + seq_len {lens[row]} exceeds "
                    f"max_position_embeddings {self.geometry.max_position_embeddings}"
                )
            if not 0 <= user_ids[row] < page_table.shape[0]:
                raise ValueError(f"row {row}: user_id {user_ids[row]} outside page table rows {page_table.shape[0]}")
        if batch == 1:
            out = self._prefill_row(hidden_states, page_table, kv_cache, user_ids[0], starts[0], lens[0])
            return self._pad_rows(out, s_max)
        outputs = []
        for row in range(batch):
            hidden_row = ttnn.slice(
                hidden_states,
                [row, 0, 0, 0],
                [row + 1, 1, s_max, self.geometry.hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = self._prefill_row(hidden_row, page_table, kv_cache, user_ids[row], starts[row], lens[row])
            hidden_row.deallocate(True)
            outputs.append(self._pad_rows(out, s_max))
        result = ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for out in outputs:
            out.deallocate(True)
        return result

    def _pad_rows(self, out: ttnn.Tensor, s_max: int) -> ttnn.Tensor:
        if out.shape[-2] == s_max:
            return out
        padded = ttnn.pad(out, [(0, 0), (0, 0), (0, s_max - out.shape[-2]), (0, 0)], 0.0)
        if not _same_buffer(padded, out):
            out.deallocate(True)
        return padded

    def _prefill_row(
        self,
        hidden_row: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        start_pos: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        hidden_size = self.geometry.hidden_size
        phys_base, lead, phys_total = plan_prefill_window(start_pos, seq_len)
        block_size = kv_cache[0].shape[2]
        if page_table.shape[1] * block_size < phys_base + phys_total:
            raise ValueError(
                f"page table covers {page_table.shape[1] * block_size} positions, "
                f"prefill needs {phys_base + phys_total}"
            )
        x = hidden_row
        if x.shape[-2] != seq_len:
            x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, seq_len, hidden_size], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lead or phys_total != seq_len:
            x = self._pad_sequence(x, lead, phys_total - lead - seq_len)
        user_page_table = page_table
        owns_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            owns_table = True
        chunks = plan_prefill_chunks(start_pos, seq_len, self.prefill_chunk_size)
        outputs = []
        for chunk in chunks:
            if len(chunks) == 1:
                x_chunk = x
            else:
                x_chunk = ttnn.slice(
                    x,
                    [0, 0, chunk.input_offset, 0],
                    [1, 1, chunk.input_offset + chunk.phys_len, hidden_size],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            outputs.append(self._prefill_chunk(x_chunk, chunk, user_page_table, kv_cache))
            if x_chunk is not x:
                x_chunk.deallocate(True)
        if not _same_buffer(x, hidden_row):
            x.deallocate(True)
        if owns_table:
            user_page_table.deallocate(True)
        if len(outputs) == 1:
            out = outputs[0]
        else:
            out = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for o in outputs:
                o.deallocate(True)
        if lead or phys_total != seq_len:
            sliced = ttnn.slice(
                out,
                [0, 0, lead, 0],
                [1, 1, lead + seq_len, hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out.deallocate(True)
            out = sliced
        return out

    def _pad_sequence(self, x: ttnn.Tensor, front: int, back: int) -> ttnn.Tensor:
        """Zero-pad the sequence axis on device. TILE tensors need a row-major detour for front padding."""
        pad = [(0, 0), (0, 0), (front, back), (0, 0)]
        if front == 0:
            return ttnn.pad(x, pad, 0.0)
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        padded = ttnn.pad(rm, pad, 0.0)
        rm.deallocate(True)
        tiled = ttnn.to_layout(padded, ttnn.TILE_LAYOUT)
        padded.deallocate(True)
        return tiled

    def _prefill_chunk(
        self,
        x: ttnn.Tensor,
        chunk: PrefillChunk,
        user_page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        residual = x
        attn_in = self._rms_norm(x, self.weights.input_ln)
        attn_out = self._attention_prefill(attn_in, chunk, user_page_table, kv_cache)
        attn_in.deallocate(True)
        h = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)
        mlp_in = self._rms_norm(h, self.weights.post_attn_ln)
        mlp_out = self._mlp(mlp_in)
        mlp_in.deallocate(True)
        out = ttnn.add(h, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        mlp_out.deallocate(True)
        return out

    def _rms_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            dtype=self.activation_dtype,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = self._linear(x, self.weights.gate_proj)
        up = self._linear(x, self.weights.up_proj)
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate.deallocate(True)
        up.deallocate(True)
        out = self._linear(act, self.weights.down_proj)
        act.deallocate(True)
        return out

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        chunk: PrefillChunk,
        user_page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        g = self.geometry
        xqkv = self._linear(x, self.weights.qkv)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=g.num_heads,
            num_kv_heads=g.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)
        q_normed = self._rms_norm(q_heads, self.weights.q_norm)
        q_heads.deallocate(True)
        k_normed = self._rms_norm(k_heads, self.weights.k_norm)
        k_heads.deallocate(True)
        cos = ttnn.slice(
            self.weights.rope_cos,
            [0, 0, chunk.phys_start, 0],
            [1, 1, chunk.phys_start + chunk.phys_len, g.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.slice(
            self.weights.rope_sin,
            [0, 0, chunk.phys_start, 0],
            [1, 1, chunk.phys_start + chunk.phys_len, g.head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q_normed, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )
        q_normed.deallocate(True)
        k_rot = ttnn.experimental.rotary_embedding_hf(
            k_normed, cos, sin, is_decode_mode=False, compute_kernel_config=self.compute_config
        )
        k_normed.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)

        key_cache, value_cache = kv_cache
        self._write_prefill_cache(key_cache, value_cache, k_rot, v_heads, chunk, user_page_table)
        k_rot.deallocate(True)
        v_heads.deallocate(True)

        kv_extent = user_page_table.shape[1] * key_cache.shape[2]
        chunk_size = prefill_sdpa_chunk_size(chunk.phys_start, chunk.phys_len, kv_extent)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.prefill_sdpa_grid,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_rot,
            key_cache,
            value_cache,
            user_page_table,
            chunk_start_idx=chunk.phys_start,
            scale=g.scale,
            program_config=program_config,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        out = self._linear(concat, self.weights.o_proj)
        concat.deallocate(True)
        return out

    def _write_prefill_cache(
        self,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        k_rot: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        chunk: PrefillChunk,
        user_page_table: ttnn.Tensor,
    ) -> None:
        """Write the chunk's valid K/V rows at their absolute positions.

        Rows that share a cache block with live rows before ``abs_start`` are
        written one token at a time with ``paged_update_cache``. From the first
        block boundary at or after ``abs_start`` the remaining rows are copied
        as whole tiles with ``paged_fill_cache`` on a page-table view that
        starts at that block. Trailing tile padding beyond the chunk's valid
        end lands only in positions no reader consumes before they are
        rewritten.
        """
        g = self.geometry
        block_size = key_cache.shape[2]
        token_positions, fill_start = plan_cache_writes(chunk.abs_start, chunk.abs_end, block_size)
        cache_dtype = key_cache.dtype
        if token_positions:
            update_mem_config = _single_user_update_memory_config(g.head_dim)
            for position in token_positions:
                row = position - chunk.phys_start
                k_token = ttnn.slice(k_rot, [0, 0, row, 0], [1, g.num_kv_heads, row + 1, g.head_dim])
                v_token = ttnn.slice(v_heads, [0, 0, row, 0], [1, g.num_kv_heads, row + 1, g.head_dim])
                k_token = ttnn.transpose(k_token, 1, 2)
                v_token = ttnn.transpose(v_token, 1, 2)
                k_token = ttnn.to_memory_config(k_token, update_mem_config, dtype=k_token.dtype)
                v_token = ttnn.to_memory_config(v_token, update_mem_config, dtype=v_token.dtype)
                position_tensor = ttnn.moreh_full(
                    (1, 1),
                    position,
                    self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.experimental.paged_update_cache(
                    key_cache, k_token, update_idxs_tensor=position_tensor, page_table=user_page_table
                )
                ttnn.experimental.paged_update_cache(
                    value_cache, v_token, update_idxs_tensor=position_tensor, page_table=user_page_table
                )
                k_token.deallocate(True)
                v_token.deallocate(True)
                position_tensor.deallocate(True)
        if fill_start is None:
            return
        row0 = fill_start - chunk.phys_start
        k_fill = k_rot
        v_fill = v_heads
        owns_fill = False
        if row0 != 0:
            k_fill = ttnn.slice(k_rot, [0, 0, row0, 0], [1, g.num_kv_heads, chunk.phys_len, g.head_dim])
            v_fill = ttnn.slice(v_heads, [0, 0, row0, 0], [1, g.num_kv_heads, chunk.phys_len, g.head_dim])
            owns_fill = True
        if k_fill.dtype != cache_dtype:
            k_cast = ttnn.typecast(k_fill, cache_dtype)
            v_cast = ttnn.typecast(v_fill, cache_dtype)
            if owns_fill:
                k_fill.deallocate(True)
                v_fill.deallocate(True)
            k_fill, v_fill = k_cast, v_cast
            owns_fill = True
        first_block = fill_start // block_size
        fill_table = user_page_table
        owns_table = False
        if first_block != 0:
            fill_table = ttnn.slice(user_page_table, [0, first_block], [1, user_page_table.shape[1]])
            owns_table = True
        ttnn.experimental.paged_fill_cache(key_cache, k_fill, fill_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(value_cache, v_fill, fill_table, batch_idx=0)
        if owns_fill:
            k_fill.deallocate(True)
            v_fill.deallocate(True)
        if owns_table:
            fill_table.deallocate(True)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        batch = hidden_states.shape[-2]
        if not 1 <= batch <= MAX_DECODE_BATCH:
            raise ValueError(f"decode batch must be in [1, {MAX_DECODE_BATCH}], got {batch}")
        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(attn_in, current_pos, page_table, kv_cache)
        attn_in.deallocate(True)
        h = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)
        mlp_in = self._rms_norm(h, self.weights.post_attn_ln)
        mlp_out = self._mlp(mlp_in)
        mlp_in.deallocate(True)
        out = ttnn.add(h, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        mlp_out.deallocate(True)
        return out

    def _decode_rope(self, current_pos: ttnn.Tensor, batch: int, heads_mem_config) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        g = self.geometry
        idx = ttnn.reshape(current_pos, (1, batch))
        idx = ttnn.typecast(idx, ttnn.uint32)
        outs = []
        for table in (self.weights.rope_cos_table, self.weights.rope_sin_table):
            emb = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            emb = ttnn.reshape(emb, (1, 1, batch, g.head_dim))
            emb = ttnn.transpose(emb, 1, 2)
            sharded = ttnn.interleaved_to_sharded(emb, heads_mem_config)
            emb.deallocate(True)
            outs.append(sharded)
        idx.deallocate(True)
        return outs[0], outs[1]

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        g = self.geometry
        batch = x.shape[-2]
        xqkv = self._linear(x, self.weights.qkv)
        heads_mem_config = _decode_height_sharded_memory_config(self.mesh_device, batch, g.head_dim)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=g.num_heads,
            num_kv_heads=g.num_kv_heads,
            memory_config=heads_mem_config,
        )
        xqkv.deallocate(True)
        q_l1 = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        q_heads.deallocate(True)
        q_normed = ttnn.rms_norm(
            q_l1, epsilon=self.eps, weight=self.weights.q_norm, compute_kernel_config=self.compute_config
        )
        q_l1.deallocate(True)
        q_sh = ttnn.to_memory_config(q_normed, heads_mem_config, dtype=q_normed.dtype)
        q_normed.deallocate(True)
        k_l1 = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        k_heads.deallocate(True)
        k_normed = ttnn.rms_norm(
            k_l1, epsilon=self.eps, weight=self.weights.k_norm, compute_kernel_config=self.compute_config
        )
        k_l1.deallocate(True)
        k_sh = ttnn.to_memory_config(k_normed, heads_mem_config, dtype=k_normed.dtype)
        k_normed.deallocate(True)

        cos, sin = self._decode_rope(current_pos, batch, heads_mem_config)
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q_sh, cos, sin, is_decode_mode=True, compute_kernel_config=self.compute_config
        )
        q_sh.deallocate(True)
        k_rot = ttnn.experimental.rotary_embedding_hf(
            k_sh, cos, sin, is_decode_mode=True, compute_kernel_config=self.compute_config
        )
        k_sh.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)

        key_cache, value_cache = kv_cache
        ttnn.experimental.paged_update_cache(key_cache, k_rot, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(
            value_cache, v_heads, update_idxs_tensor=current_pos, page_table=page_table
        )
        k_rot.deallocate(True)
        v_heads.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=g.scale,
            program_config=self._decode_sdpa_program_config(page_table.shape[1] * key_cache.shape[2]),
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        attn_sh = ttnn.to_memory_config(attn, heads_mem_config, dtype=attn.dtype)
        attn.deallocate(True)
        concat = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=g.num_heads)
        attn_sh.deallocate(True)
        concat_dram = ttnn.sharded_to_interleaved(concat, ttnn.DRAM_MEMORY_CONFIG)
        concat.deallocate(True)
        out = self._linear(concat_dram, self.weights.o_proj)
        concat_dram.deallocate(True)
        if out.shape[-2] != batch:
            sliced = ttnn.slice(
                out,
                [0, 0, 0, 0],
                [1, 1, batch, g.hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out.deallocate(True)
            out = sliced
        return out


def _same_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    """True when two device tensors share one allocation, as ``ttnn.pad`` returns for tile-only padding."""
    return a is b or a.buffer_address() == b.buffer_address()


def _single_user_update_memory_config(head_dim: int) -> ttnn.MemoryConfig:
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, head_dim),
        core_grid=one_core,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _decode_height_sharded_memory_config(mesh_device: ttnn.MeshDevice, batch: int, width: int) -> ttnn.MemoryConfig:
    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = max(1, batch)
    grid_x = min(num_cores, grid.x)
    while num_cores % grid_x != 0 or num_cores // grid_x > grid.y:
        grid_x -= 1
    grid_y = num_cores // grid_x
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
