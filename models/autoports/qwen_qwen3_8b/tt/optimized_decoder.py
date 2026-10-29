# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for Qwen/Qwen3-8B on one Blackhole chip.

``OptimizedDecoder`` keeps the public contract, paged KV-cache behaviour,
chunk planning and padding rules of ``FusedDecoder`` and changes the weight
precision, the weight placement and the decode activation layout:

* Projection weights are width-sharded across the DRAM banks in the dtype
  given by ``OptimizedPolicy``; decode reads them with DRAM-sharded matmuls
  and prefill chunks longer than ``prefill_interleaved_copy_max_rows`` read
  them with ``minimal_matmul`` at the fidelity that matches the dtype (the
  residual add is a separate ``ttnn.add`` because the addcmul-fused kernel
  needs equal weight and residual tile sizes). With the interleaved copy of
  each weight present (``prefill_interleaved_copy_max_rows > 0``), chunks up
  to that many rows use ``ttnn.linear`` (auto config), chunks up to
  ``prefill_2d_max_rows`` use an explicit 2D program config for the QKV, O
  and down projections, and every other chunk runs ``minimal_matmul`` on the
  interleaved copy; the DRAM-sharded copy serves decode only.
* Decode activations stay width-sharded in L1 across the residual stream,
  both RMSNorms, the QKV projection, the attention output projection and the
  MLP. The layer accepts a DRAM-interleaved residual (the functional contract)
  or a residual already in ``decode_residual_memory_config`` (the layer-stack
  contract) and returns the residual in the layout it received.
* Decode RMSNorm uses the sharded multi-core program config; the Q/K/V head
  split reads the width-sharded projection output directly.
* A tile-unaligned continuation start writes its leading cache rows with one
  fused K/V cache update per token on a prepared pseudo-user tensor instead of
  thirteen ops per token.

Every geometry, dtype and fidelity choice is explicit in ``OptimizedPolicy``
so the evidence in ``doc/optimized_decoder`` can name the measured path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ttnn
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import (
    TILE_SIZE,
    DecoderGeometry,
    DecoderWeights,
    PrefillChunk,
    _decode_height_sharded_memory_config,
    _replicate_mapper,
    _same_buffer,
    _strip_layer_prefix,
    decode_physical_batch,
    hf_rope_tables,
    plan_cache_writes,
    validate_hf_config,
)
from models.autoports.qwen_qwen3_8b.tt.fused_decoder import (
    PREFILL_MINIMAL_MATMUL_MIN_ROWS,
    FusedDecoder,
    _decode_v_update_memory_config,
)

DTYPE_NAMES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FIDELITY_NAMES = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}
WEIGHT_BYTES_PER_ELEMENT = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}
MAX_LEAD_GROUP = 32


@dataclass(frozen=True)
class DecodeMatmulRole:
    """One DRAM-sharded decode projection: weight dtype, compute fidelity and block geometry.

    ``cores`` is the width-shard core count of the activation (and output); ``in0_block_w``
    must divide the per-core K tiles; ``readers`` is ``num_workers_per_dram_bank`` and must
    divide the per-bank weight width in tiles.
    """

    dtype: str
    fidelity: str
    fp32_acc: bool
    cores: int
    in0_block_w: int
    readers: int = 1
    fused_silu: bool = False


@dataclass(frozen=True)
class OptimizedPolicy:
    qkv: DecodeMatmulRole
    o_proj: DecodeMatmulRole
    gate_up: DecodeMatmulRole
    down: DecodeMatmulRole
    up: DecodeMatmulRole | None = None
    residual_cores: int = 32
    norm_subblock_w: int = 2
    norm_fidelity: str = "hifi4"
    sdpa_fidelity: str = "hifi4"
    prefill_fp32_acc: bool = True
    prefill_bfp8_fidelity: str = "lofi"
    prefill_interleaved_copy_max_rows: int = 256
    prefill_2d_max_rows: int = 1024
    prefill_2d_in0_block_w: int = 16

    def describe(self) -> dict:
        return asdict(self)

    def prefill_fidelity(self, dtype: str) -> str:
        return {"bf16": "hifi4", "bfp8": self.prefill_bfp8_fidelity, "bfp4": "lofi"}[dtype]

    @property
    def up_role(self) -> DecodeMatmulRole:
        return self.up if self.up is not None else self.gate_up


DEFAULT_POLICY = OptimizedPolicy(
    qkv=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    o_proj=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    gate_up=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    down=DecodeMatmulRole("bfp8", "lofi", False, 32, 6, 1),
    up=DecodeMatmulRole("bfp4", "lofi", False, 32, 4, 3),
)

ALL_BFP8_POLICY = OptimizedPolicy(
    qkv=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    o_proj=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    gate_up=DecodeMatmulRole("bfp8", "lofi", False, 32, 4, 1),
    down=DecodeMatmulRole("bfp8", "lofi", False, 32, 6, 1),
)


def width_sharded_core_grid(cores: int, grid: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    """The first ``cores`` cores of the compute grid in row-major order.

    This is the core set the DRAM-sharded matmul gives its output regardless of the memory config
    it is asked for (and the default output set of ``nlp_concat_heads_decode``), so using it for
    the residual stream keeps every decode op on one layout without reshards.
    """
    if cores > grid.x * grid.y:
        raise ValueError(f"{cores} cores do not fit a {grid.x}x{grid.y} grid")
    return ttnn.num_cores_to_corerangeset(cores, grid, True)


def width_sharded_memory_config(width: int, cores: int, grid: ttnn.CoreCoord) -> ttnn.MemoryConfig:
    if width % (TILE_SIZE * cores) != 0:
        raise ValueError(f"width {width} does not split into {cores} tile-aligned shards")
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width // cores),
        core_grid=width_sharded_core_grid(cores, grid),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def dram_sharded_weight_memory_config(k: int, n: int, dram_banks: int) -> ttnn.MemoryConfig:
    padded = math.ceil(n / (TILE_SIZE * dram_banks)) * TILE_SIZE * dram_banks
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_banks - 1, 0))])
    spec = ttnn.ShardSpec(grid, (k, padded // dram_banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def dram_sharded_program_config(role: DecodeMatmulRole, k: int, n: int, dram_banks: int):
    k_tiles, n_tiles = k // TILE_SIZE, n // TILE_SIZE
    if k_tiles % role.cores or n_tiles % role.cores:
        raise ValueError(f"{k}x{n} does not split over {role.cores} cores")
    if (k_tiles // role.cores) % role.in0_block_w:
        raise ValueError(f"in0_block_w {role.in0_block_w} does not divide {k_tiles // role.cores} K tiles per core")
    if (n_tiles // dram_banks) % role.readers:
        raise ValueError(f"{role.readers} readers do not divide {n_tiles // dram_banks} tiles per DRAM bank")
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=role.in0_block_w,
        per_core_M=1,
        per_core_N=n_tiles // role.cores,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if role.fused_silu else None,
        num_workers_per_dram_bank=role.readers,
    )


def lead_update_core_sets(count: int, grid: ttnn.CoreCoord) -> tuple[ttnn.CoreRangeSet, ttnn.CoreRangeSet]:
    """Two disjoint sets of ``count`` cores for the K and V pseudo-users of a lead-row cache write."""
    if 2 * count > grid.x * grid.y:
        raise ValueError(f"{count} lead rows need {2 * count} cores; the grid has {grid.x * grid.y}")
    k_rows = math.ceil(count / grid.x)
    k_set = ttnn.num_cores_to_corerangeset(count, grid, True)
    v_region = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, k_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    v_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(0, k_rows), count, v_region, True)
    return k_set, v_set


def same_layout(a: ttnn.MemoryConfig, b: ttnn.MemoryConfig) -> bool:
    """Equality on layout, buffer type and shard spec; ``MemoryConfig.__eq__`` also compares derived ND fields."""
    if a.memory_layout != b.memory_layout or a.buffer_type != b.buffer_type:
        return False
    sa, sb = a.shard_spec, b.shard_spec
    if sa is None or sb is None:
        return sa is None and sb is None
    return sa.grid == sb.grid and list(sa.shape) == list(sb.shape) and sa.orientation == sb.orientation


def _height_sharded_config(cores: ttnn.CoreRangeSet, width: int) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class OptimizedDecoder(FusedDecoder):
    """One Qwen3-8B decoder layer with reduced-precision DRAM-sharded weights and an L1-sharded decode path."""

    def __init__(
        self,
        *,
        geometry: DecoderGeometry,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        weights: DecoderWeights,
        policy: OptimizedPolicy = DEFAULT_POLICY,
        weight_dtypes: dict[str, str] | None = None,
        prefill_weights: dict[str, ttnn.Tensor] | None = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_chunk_size: int = 4096,
        prefill_minimal_matmul_min_rows: int = PREFILL_MINIMAL_MATMUL_MIN_ROWS,
    ) -> None:
        super().__init__(
            geometry=geometry,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            activation_dtype=activation_dtype,
            prefill_chunk_size=prefill_chunk_size,
            prefill_minimal_matmul_min_rows=prefill_minimal_matmul_min_rows,
        )
        self.policy = policy
        self.prefill_weights = dict(prefill_weights or {})
        self.weight_dtypes = weight_dtypes or {
            "qkv": policy.qkv.dtype,
            "o_proj": policy.o_proj.dtype,
            "gate_proj": policy.gate_up.dtype,
            "up_proj": policy.up_role.dtype,
            "down_proj": policy.down.dtype,
        }
        g = geometry
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = grid
        self.dram_banks = mesh_device.dram_grid_size().x
        arch = mesh_device.arch()

        def ckc(fidelity: str, fp32: bool, l1acc: bool) -> ttnn.DeviceComputeKernelConfig:
            return ttnn.init_device_compute_kernel_config(
                arch,
                math_fidelity=FIDELITY_NAMES[fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=l1acc,
            )

        self.norm_compute_config = ckc(policy.norm_fidelity, True, False)
        self.sdpa_compute_config = ckc(policy.sdpa_fidelity, True, False)
        self._prefill_compute = {}
        self._interleaved_copy = {}
        for name, weight in (
            ("qkv", weights.qkv),
            ("o_proj", weights.o_proj),
            ("gate_proj", weights.gate_proj),
            ("up_proj", weights.up_proj),
            ("down_proj", weights.down_proj),
        ):
            config = ckc(policy.prefill_fidelity(self.weight_dtypes[name]), policy.prefill_fp32_acc, False)
            self._prefill_compute[id(weight)] = config
            copy = self.prefill_weights.get(name)
            if copy is not None:
                self._prefill_compute[id(copy)] = config
                self._interleaved_copy[id(weight)] = copy
        self._prefill_2d_roles = {id(weights.qkv), id(weights.o_proj), id(weights.down_proj)}
        self.decode_compute = {
            "qkv": ckc(policy.qkv.fidelity, policy.qkv.fp32_acc, True),
            "o_proj": ckc(policy.o_proj.fidelity, policy.o_proj.fp32_acc, True),
            "gate_up": ckc(policy.gate_up.fidelity, policy.gate_up.fp32_acc, True),
            "up": ckc(policy.up_role.fidelity, policy.up_role.fp32_acc, True),
            "down": ckc(policy.down.fidelity, policy.down.fp32_acc, True),
        }
        self.decode_program_config = {
            "qkv": dram_sharded_program_config(policy.qkv, g.hidden_size, g.qkv_width, self.dram_banks),
            "o_proj": dram_sharded_program_config(
                policy.o_proj, g.num_heads * g.head_dim, g.hidden_size, self.dram_banks
            ),
            "gate_up": dram_sharded_program_config(policy.gate_up, g.hidden_size, g.intermediate_size, self.dram_banks),
            "up": dram_sharded_program_config(policy.up_role, g.hidden_size, g.intermediate_size, self.dram_banks),
            "down": dram_sharded_program_config(policy.down, g.intermediate_size, g.hidden_size, self.dram_banks),
        }
        self.residual_memory_config = width_sharded_memory_config(g.hidden_size, policy.residual_cores, grid)
        self.qkv_input_memory_config = width_sharded_memory_config(g.hidden_size, policy.qkv.cores, grid)
        self.o_input_memory_config = width_sharded_memory_config(g.num_heads * g.head_dim, policy.o_proj.cores, grid)
        self.o_input_core_grid = width_sharded_core_grid(policy.o_proj.cores, grid)
        self.gate_up_input_memory_config = width_sharded_memory_config(g.hidden_size, policy.gate_up.cores, grid)
        self.down_input_memory_config = width_sharded_memory_config(g.intermediate_size, policy.down.cores, grid)
        self.decode_output_memory_config = {
            "qkv": width_sharded_memory_config(g.qkv_width, policy.qkv.cores, grid),
            "o_proj": width_sharded_memory_config(g.hidden_size, policy.o_proj.cores, grid),
            "gate_up": width_sharded_memory_config(g.intermediate_size, policy.gate_up.cores, grid),
            "up": width_sharded_memory_config(g.intermediate_size, policy.up_role.cores, grid),
            "down": width_sharded_memory_config(g.hidden_size, policy.down.cores, grid),
        }
        if policy.up_role.cores != policy.gate_up.cores:
            raise ValueError("gate and up projections must share the activation core grid")
        residual_box = width_sharded_core_grid(policy.residual_cores, grid).bounding_box()
        block_w = g.hidden_size // TILE_SIZE // policy.residual_cores
        if block_w % policy.norm_subblock_w:
            raise ValueError(f"norm_subblock_w {policy.norm_subblock_w} does not divide block_w {block_w}")
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(residual_box.end.x + 1, residual_box.end.y + 1),
            subblock_w=policy.norm_subblock_w,
            block_h=1,
            block_w=block_w,
            inplace=False,
        )

    @property
    def decode_residual_memory_config(self) -> ttnn.MemoryConfig:
        """Layout a caller may use for decode residual input and output to skip the DRAM round trip."""
        return self.residual_memory_config

    def weight_bytes_resident(self) -> int:
        """DRAM bytes of all projection weights including the interleaved prefill copies."""
        return self.weight_bytes_per_token() * (2 if self._interleaved_copy else 1)

    def weight_bytes_per_token(self) -> int:
        g = self.geometry
        sizes = {
            "qkv": g.hidden_size * g.qkv_width,
            "o_proj": g.num_heads * g.head_dim * g.hidden_size,
            "gate_proj": g.hidden_size * g.intermediate_size,
            "up_proj": g.hidden_size * g.intermediate_size,
            "down_proj": g.intermediate_size * g.hidden_size,
        }
        return int(sum(sizes[n] * WEIGHT_BYTES_PER_ELEMENT[d] for n, d in self.weight_dtypes.items()))

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        policy: OptimizedPolicy = DEFAULT_POLICY,
        weight_dtypes: dict[str, str] | None = None,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        prefill_chunk_size: int = 4096,
        prefill_minimal_matmul_min_rows: int | None = None,
        rope_tables: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None,
        **_: Any,
    ) -> "OptimizedDecoder":
        """Build the layer from canonical HF keys with per-role weight dtypes and DRAM-sharded placement.

        ``rope_tables`` (``rope_cos, rope_sin, rope_cos_table, rope_sin_table`` of an already built layer)
        shares the position-independent RoPE tables instead of loading a 42 MB copy per layer.
        """
        import torch

        geometry = validate_hf_config(hf_config, layer_idx)
        local = _strip_layer_prefix(state_dict, layer_idx)
        dtypes = (
            dict(weight_dtypes)
            if weight_dtypes
            else {
                "qkv": policy.qkv.dtype,
                "o_proj": policy.o_proj.dtype,
                "gate_proj": policy.gate_up.dtype,
                "up_proj": policy.up_role.dtype,
                "down_proj": policy.down.dtype,
            }
        )
        cache_dir = Path(tensor_cache_path) if tensor_cache_path is not None else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        dram_banks = mesh_device.dram_grid_size().x

        def as_tt(name: str, tensor: torch.Tensor, dtype: ttnn.DataType, layout=ttnn.TILE_LAYOUT, memory_config=None):
            kwargs = {}
            if cache_dir is not None:
                kwargs["cache_file_name"] = str(cache_dir / f"layer{layer_idx}_{name}")
            return ttnn.as_tensor(
                tensor.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(mesh_device),
                **kwargs,
            )

        def projection(name: str, tensor: torch.Tensor) -> ttnn.Tensor:
            k, n = tensor.shape
            dtype_name = dtypes[name]
            return as_tt(
                f"{name}_{dtype_name}_ds{dram_banks}",
                tensor.unsqueeze(0).unsqueeze(0),
                DTYPE_NAMES[dtype_name],
                memory_config=dram_sharded_weight_memory_config(k, n, dram_banks),
            )

        prefill_copies: dict[str, ttnn.Tensor] = {}

        def interleaved_copy(name: str, tensor: torch.Tensor) -> None:
            if policy.prefill_interleaved_copy_max_rows > 0:
                prefill_copies[name] = as_tt(
                    f"{name}_{dtypes[name]}_il", tensor.unsqueeze(0).unsqueeze(0), DTYPE_NAMES[dtypes[name]]
                )

        with torch.no_grad():
            qkv = torch.cat([local[f"self_attn.{p}_proj.weight"].to(torch.float32).T for p in ("q", "k", "v")], dim=-1)
            norm_shape = (1, 1, 1, geometry.hidden_size)
            head_norm_shape = (1, 1, 1, geometry.head_dim)
            cos, sin = hf_rope_tables(geometry)
            projections = {
                "qkv": qkv,
                "o_proj": local["self_attn.o_proj.weight"].to(torch.float32).T,
                "gate_proj": local["mlp.gate_proj.weight"].to(torch.float32).T,
                "up_proj": local["mlp.up_proj.weight"].to(torch.float32).T,
                "down_proj": local["mlp.down_proj.weight"].to(torch.float32).T,
            }
            for name, tensor in projections.items():
                interleaved_copy(name, tensor)
            weights = DecoderWeights(
                qkv=projection("qkv", projections["qkv"]),
                o_proj=projection("o_proj", projections["o_proj"]),
                gate_proj=projection("gate_proj", projections["gate_proj"]),
                up_proj=projection("up_proj", projections["up_proj"]),
                down_proj=projection("down_proj", projections["down_proj"]),
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
                rope_cos=rope_tables[0]
                if rope_tables
                else as_tt("rope_cos", cos.unsqueeze(0).unsqueeze(0), ttnn.bfloat16),
                rope_sin=rope_tables[1]
                if rope_tables
                else as_tt("rope_sin", sin.unsqueeze(0).unsqueeze(0), ttnn.bfloat16),
                rope_cos_table=(
                    rope_tables[2]
                    if rope_tables
                    else as_tt("rope_cos_table", cos, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                ),
                rope_sin_table=(
                    rope_tables[3]
                    if rope_tables
                    else as_tt("rope_sin_table", sin, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                ),
            )
        kwargs = {}
        if prefill_minimal_matmul_min_rows is not None:
            kwargs["prefill_minimal_matmul_min_rows"] = prefill_minimal_matmul_min_rows
        return cls(
            geometry=geometry,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            policy=policy,
            weight_dtypes=dtypes,
            prefill_weights=prefill_copies,
            activation_dtype=activation_dtype,
            prefill_chunk_size=prefill_chunk_size,
            **kwargs,
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
        """Allocate a zeroed paged K/V cache on device in any cache dtype (``moreh_full`` only fills BF16)."""
        shape = (num_blocks, num_kv_heads, block_size, head_dim)
        caches = []
        for _ in range(2):
            zeros = ttnn.moreh_full(
                shape,
                0.0,
                mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if dtype == ttnn.bfloat16:
                caches.append(zeros)
                continue
            cast = ttnn.typecast(zeros, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            zeros.deallocate(True)
            caches.append(cast)
        return caches[0], caches[1]

    # ----------------------------------------------------------------- prefill

    def _prefill_compute_config(self, weight: ttnn.Tensor) -> ttnn.DeviceComputeKernelConfig:
        return self._prefill_compute[id(weight)]

    def _prefill_2d_program_config(self, rows: int, k: int, n: int):
        """2D matmul config on the full grid for a prefill chunk of ``rows`` rows (exp12 winners: in0_block_w 16)."""
        grid = self.grid
        per_core_m = -(-(rows // TILE_SIZE) // grid.y)
        per_core_n = -(-(n // TILE_SIZE) // grid.x)
        best = (1, 1)
        for h in range(1, 5):
            for w in range(1, 5):
                if h * w <= 4 and per_core_m % h == 0 and per_core_n % w == 0 and h * w > best[0] * best[1]:
                    best = (h, w)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=self.policy.prefill_2d_in0_block_w,
            out_subblock_h=best[0],
            out_subblock_w=best[1],
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill projection kernel by chunk rows (measured in work log 3.10, 8.2 and 8.8).

        With the interleaved copy present: chunks up to ``prefill_interleaved_copy_max_rows`` run
        ``ttnn.linear`` with the auto config; chunks up to ``prefill_2d_max_rows`` run the explicit 2D
        config for the roles where it won (QKV, O, down); everything else runs ``minimal_matmul`` on the
        interleaved copy. Without the copy every chunk runs ``minimal_matmul`` on the DRAM-sharded weight.
        """
        copy = self._interleaved_copy.get(id(weight))
        rows = x.shape[-2]
        if copy is not None:
            if rows <= self.policy.prefill_interleaved_copy_max_rows:
                return ttnn.linear(
                    x,
                    copy,
                    dtype=self.activation_dtype,
                    compute_kernel_config=self._prefill_compute_config(copy),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if rows <= self.policy.prefill_2d_max_rows and id(weight) in self._prefill_2d_roles:
                k, n = weight.shape[-2], weight.shape[-1]
                return ttnn.linear(
                    x,
                    copy,
                    program_config=self._prefill_2d_program_config(rows, k, n),
                    dtype=self.activation_dtype,
                    compute_kernel_config=self._prefill_compute_config(copy),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            weight = copy
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            dtype=self.activation_dtype,
            compute_kernel_config=self._prefill_compute_config(weight),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_matmul(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return self._linear(x, weight)

    def _prefill_matmul_residual(self, x: ttnn.Tensor, weight: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        """``residual + x @ weight``; the stage-2 addcmul-fused kernel needs a weight tile size equal to
        the residual's, which block-float weights do not satisfy (``minimal_matmul_program_factory.cpp:372``)."""
        y = self._linear(x, weight)
        out = ttnn.add(residual, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y.deallocate(True)
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
        """Lead rows: one fused K/V update per token over a prepared pseudo-user tensor; then the tile fill.

        The per-token one-hot position vector is a slice of one ``ttnn.arange`` row padded with -1;
        ``moreh_full`` of a ``[1, 1]`` int32 scalar trips the watcher (its writer sends a 64-byte NoC
        write for a 4-byte page).
        """
        g = self.geometry
        block_size = key_cache.shape[2]
        token_positions, fill_start = plan_cache_writes(chunk.abs_start, chunk.abs_end, block_size)
        for group_start in range(0, len(token_positions), MAX_LEAD_GROUP):
            positions = token_positions[group_start : group_start + MAX_LEAD_GROUP]
            count = len(positions)
            row0 = positions[0] - chunk.phys_start
            k_lead = ttnn.slice(k_rot, [0, 0, row0, 0], [1, g.num_kv_heads, row0 + count, g.head_dim])
            v_lead = ttnn.slice(v_heads, [0, 0, row0, 0], [1, g.num_kv_heads, row0 + count, g.head_dim])
            k_users = ttnn.transpose(k_lead, 1, 2)
            v_users = ttnn.transpose(v_lead, 1, 2)
            k_lead.deallocate(True)
            v_lead.deallocate(True)
            k_cores, v_cores = lead_update_core_sets(count, self.grid)
            k_sh = ttnn.to_memory_config(k_users, _height_sharded_config(k_cores, g.head_dim))
            v_sh = ttnn.to_memory_config(v_users, _height_sharded_config(v_cores, g.head_dim))
            k_users.deallocate(True)
            v_users.deallocate(True)
            table = ttnn.repeat(user_page_table, ttnn.Shape([count, 1])) if count > 1 else user_page_table
            position_row = ttnn.reshape(
                ttnn.arange(positions[0], positions[0] + count, 1, device=self.mesh_device, dtype=ttnn.int32),
                (1, count),
            )
            for index in range(count):
                one = ttnn.slice(position_row, [0, index], [1, index + 1])
                padded = ttnn.pad(one, [(0, 0), (index, count - 1 - index)], -1) if count > 1 else one
                update_idxs = ttnn.reshape(padded, (count,))
                ttnn.experimental.paged_fused_update_cache(
                    key_cache, k_sh, value_cache, v_sh, update_idxs_tensor=update_idxs, page_table=table
                )
                if not _same_buffer(padded, one):
                    padded.deallocate(True)
                one.deallocate(True)
            position_row.deallocate(True)
            if table is not user_page_table:
                table.deallocate(True)
            k_sh.deallocate(True)
            v_sh.deallocate(True)
        if fill_start is None:
            return
        self._fill_prefill_cache(key_cache, value_cache, k_rot, v_heads, chunk, user_page_table, fill_start)

    def _fill_prefill_cache(self, key_cache, value_cache, k_rot, v_heads, chunk, user_page_table, fill_start) -> None:
        g = self.geometry
        block_size = key_cache.shape[2]
        cache_dtype = key_cache.dtype
        row0 = fill_start - chunk.phys_start
        k_fill, v_fill, owns_fill = k_rot, v_heads, False
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
            k_fill, v_fill, owns_fill = k_cast, v_cast, True
        first_block = fill_start // block_size
        fill_table, owns_table = user_page_table, False
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

    # ------------------------------------------------------------------ decode

    @staticmethod
    def _to(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
        """Reshard ``x`` to ``memory_config`` only when it is not already there. Returns (tensor, owned)."""
        if same_layout(x.memory_config(), memory_config):
            return x, False
        return ttnn.to_memory_config(x, memory_config), True

    def _decode_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            program_config=self.norm_program_config,
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.norm_compute_config,
        )

    def _decode_projection(self, x: ttnn.Tensor, weight: ttnn.Tensor, role: str) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            program_config=self.decode_program_config[role],
            compute_kernel_config=self.decode_compute[role],
            dtype=self.activation_dtype,
            memory_config=self.decode_output_memory_config[role],
        )

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Decode one token per user.

        DRAM-interleaved ``hidden_states`` ``[1, 1, B, H]`` follow the functional contract: users are
        padded to the physical batch with zero rows, the activation is viewed as its full 32-row tile
        (rows past the physical batch are never read by a cross-row op), and the result is sliced back
        to ``B`` rows in DRAM. A residual already in ``decode_residual_memory_config`` (``[1, 1, 32, H]``,
        users in the first ``current_pos.shape[0]`` rows, which must be a physical batch) is the
        layer-stack contract and comes back in the same layout without any DRAM round trip. Padded
        users of that contract carry position ``-1`` (the cache update and SDPA skip them) and any
        page-table row; their RoPE index is clamped to 0 on device so the gather never reads out of
        bounds, and their rows of the result are unspecified.
        """
        hidden = self.geometry.hidden_size
        sharded_in = hidden_states.is_sharded()
        owned = []
        if sharded_in:
            users = current_pos.shape[0]
            physical, _, _ = decode_physical_batch(users, self.grid.x, self.grid.y)
            if not same_layout(hidden_states.memory_config(), self.residual_memory_config):
                raise ValueError("a sharded decode residual must use decode_residual_memory_config")
            if physical != users or hidden_states.shape[-2] != TILE_SIZE:
                raise ValueError(
                    f"a sharded decode residual carries {TILE_SIZE} rows and a physical batch of users; "
                    f"got {hidden_states.shape[-2]} rows and {users} users (physical {physical})"
                )
            x = hidden_states
            rope_pos = ttnn.relu(current_pos)
            owned.append(rope_pos)
        else:
            batch = hidden_states.shape[-2]
            physical, _, _ = decode_physical_batch(batch, self.grid.x, self.grid.y)
            padded = hidden_states
            if physical != batch:
                extra = physical - batch
                cleaned = ttnn.fill_implicit_tile_padding(hidden_states, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if not _same_buffer(cleaned, hidden_states):
                    owned.append(cleaned)
                padded = ttnn.pad(cleaned, [(0, 0), (0, 0), (0, extra), (0, 0)], 0.0)
                if not _same_buffer(padded, cleaned):
                    owned.append(padded)
                rope_pos = ttnn.pad(current_pos, [(0, extra)], 0)
                padded_pos = ttnn.pad(current_pos, [(0, extra)], -1)
                padded_table = ttnn.pad(page_table, [(0, extra), (0, 0)], 0)
                for tensor, source in ((rope_pos, current_pos), (padded_pos, current_pos), (padded_table, page_table)):
                    if not _same_buffer(tensor, source):
                        owned.append(tensor)
                current_pos = padded_pos
                page_table = padded_table
            else:
                rope_pos = current_pos
            x = ttnn.to_memory_config(padded, self.residual_memory_config)
            owned.append(x)
            if x.shape[-2] != TILE_SIZE:
                x = ttnn.reshape(x, (1, 1, TILE_SIZE, hidden), (1, 1, TILE_SIZE, hidden))
        attn_in = self._decode_norm(x, self.weights.input_ln)
        h = self._attention_decode(
            attn_in, current_pos, page_table, kv_cache, rope_pos=rope_pos, residual=x, users=physical
        )
        attn_in.deallocate(True)
        mlp_in = self._decode_norm(h, self.weights.post_attn_ln)
        out = self._mlp_decode(mlp_in, residual=h)
        mlp_in.deallocate(True)
        h.deallocate(True)
        if not sharded_in:
            interleaved = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
            out.deallocate(True)
            out = interleaved
            if out.shape[-2] != batch:
                sliced = ttnn.slice(out, [0, 0, 0, 0], [1, 1, batch, hidden], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out.deallocate(True)
                out = sliced
        for tensor in owned:
            tensor.deallocate(True)
        return out

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        rope_pos: ttnn.Tensor | None = None,
        residual: ttnn.Tensor | None = None,
        users: int | None = None,
    ) -> ttnn.Tensor:
        g = self.geometry
        batch = current_pos.shape[0] if users is None else users
        if rope_pos is None:
            rope_pos = current_pos
        qkv_in, owns_in = self._to(x, self.qkv_input_memory_config)
        xqkv = self._decode_projection(qkv_in, self.weights.qkv, "qkv")
        if owns_in:
            qkv_in.deallocate(True)
        heads_mem_config = _decode_height_sharded_memory_config(self.mesh_device, batch, g.head_dim)
        xqkv_l1 = ttnn.sharded_to_interleaved(xqkv, ttnn.L1_MEMORY_CONFIG)
        xqkv.deallocate(True)
        xqkv_users = xqkv_l1
        if xqkv_l1.shape[-2] != batch:
            xqkv_users = ttnn.reshape(xqkv_l1, (1, 1, batch, g.qkv_width), (1, 1, TILE_SIZE, g.qkv_width))
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_users,
            num_heads=g.num_heads,
            num_kv_heads=g.num_kv_heads,
            memory_config=heads_mem_config,
        )
        xqkv_l1.deallocate(True)
        q_sh = self._decode_head_norm(q_heads, self.weights.q_norm, heads_mem_config)
        k_sh = self._decode_head_norm(k_heads, self.weights.k_norm, heads_mem_config)

        cos, sin = self._decode_rope(rope_pos, batch, heads_mem_config)
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
        v_moved = ttnn.to_memory_config(v_heads, _decode_v_update_memory_config(self.mesh_device, batch, g.head_dim))
        v_heads.deallocate(True)
        ttnn.experimental.paged_fused_update_cache(
            key_cache, k_rot, value_cache, v_moved, update_idxs_tensor=current_pos, page_table=page_table
        )
        k_rot.deallocate(True)
        v_moved.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_rot,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=g.scale,
            program_config=self._decode_sdpa_program_config(page_table.shape[1] * key_cache.shape[2]),
            compute_kernel_config=self.sdpa_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_rot.deallocate(True)
        attn_sh = ttnn.to_memory_config(attn, heads_mem_config, dtype=attn.dtype)
        attn.deallocate(True)
        concat = ttnn.experimental.nlp_concat_heads_decode(
            attn_sh, num_heads=g.num_heads, sub_core_grids=self.o_input_core_grid
        )
        attn_sh.deallocate(True)
        o_in, owns_o_in = self._to(concat, self.o_input_memory_config)
        if owns_o_in:
            concat.deallocate(True)
        o = self._decode_projection(o_in, self.weights.o_proj, "o_proj")
        o_in.deallocate(True)
        o_res, owns_o = self._to(o, self.residual_memory_config)
        if owns_o:
            o.deallocate(True)
        if residual is None:
            return o_res
        out = ttnn.add(residual, o_res, memory_config=self.residual_memory_config)
        o_res.deallocate(True)
        return out

    def _mlp_decode(self, x: ttnn.Tensor, residual: ttnn.Tensor) -> ttnn.Tensor:
        mlp_in, owns_in = self._to(x, self.gate_up_input_memory_config)
        gate = self._decode_projection(mlp_in, self.weights.gate_proj, "gate_up")
        up = self._decode_projection(mlp_in, self.weights.up_proj, "up")
        if owns_in:
            mlp_in.deallocate(True)
        activations = [] if self.policy.gate_up.fused_silu else [ttnn.UnaryOpType.SILU]
        act = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=activations,
            dtype=self.activation_dtype,
            memory_config=gate.memory_config(),
        )
        gate.deallocate(True)
        up.deallocate(True)
        down_in, owns_down_in = self._to(act, self.down_input_memory_config)
        if owns_down_in:
            act.deallocate(True)
        down = self._decode_projection(down_in, self.weights.down_proj, "down")
        down_in.deallocate(True)
        down_res, owns_down = self._to(down, self.residual_memory_config)
        if owns_down:
            down.deallocate(True)
        out = ttnn.add(residual, down_res, memory_config=self.residual_memory_config)
        down_res.deallocate(True)
        return out
