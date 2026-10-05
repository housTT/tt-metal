# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Sequence, Tuple

import ttnn

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

WEIGHTS_DTYPE = ttnn.bfloat16
ACTIVATIONS_DTYPE = ttnn.bfloat16
MASK_DTYPE = ttnn.bfloat16
MASK_NEG = -1e30

TILE = 32
DEST_TILES_FULL_SYNC = 16

ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
SEQ_BUCKETS = (512,)
SEQ_BUCKETS_SIBLING = (512, 1024)
ROW_BUCKETS_AT_1024 = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PrecisionPolicy:
    """Numerics of every matmul, SDPA and norm in the encoder, the head layers and the scorer."""

    name: str
    linear_dtype: object = ttnn.bfloat8_b
    act_dtype: object = ttnn.bfloat16
    fidelity: object = ttnn.MathFidelity.HiFi3
    gelu_approx: bool = True
    mlp_fidelity: Optional[object] = None
    head_policy: str = "same"
    scorer_fp32_out: bool = True
    scorer_gelu_approx: bool = False
    fp32_dest_acc_en: bool = True
    packer_l1_acc: bool = True
    dst_full_sync_en: bool = False
    residual_dtype: object = ttnn.bfloat16

    @property
    def residual_fp32(self) -> bool:
        return self.residual_dtype == ttnn.float32

    @property
    def mlp_math_fidelity(self):
        return self.fidelity if self.mlp_fidelity is None else self.mlp_fidelity

    @property
    def head_linear_dtype(self):
        return ttnn.bfloat16 if self.head_policy == "bf16_hifi4" else self.linear_dtype

    @property
    def head_fidelity(self):
        return ttnn.MathFidelity.HiFi4 if self.head_policy == "bf16_hifi4" else self.fidelity

    @property
    def dest_tiles(self) -> int:
        tiles = DEST_TILES_FULL_SYNC
        if not self.dst_full_sync_en:
            tiles //= 2
        if self.fp32_dest_acc_en:
            tiles //= 2
        return tiles

    def _config(self, arch, fidelity, approx=False):
        return ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=fidelity,
            math_approx_mode=approx,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
            dst_full_sync_en=self.dst_full_sync_en,
        )

    def compute_config(self, device, group: str = "attn"):
        arch = device.arch()
        if group == "attn":
            return self._config(arch, self.fidelity)
        if group == "mlp":
            return self._config(arch, self.mlp_math_fidelity)
        if group in ("head", "scorer"):
            return self._config(arch, self.head_fidelity)
        if group == "norm":
            return ttnn.init_device_compute_kernel_config(
                arch,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                dst_full_sync_en=False,
            )
        raise ValueError(f"unknown compute group {group!r}")

    def gelu_activation(self, fuse: bool = True):
        if not fuse:
            return None
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, int(self.gelu_approx))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "linear_dtype": str(self.linear_dtype),
            "act_dtype": str(self.act_dtype),
            "fidelity": str(self.fidelity),
            "mlp_fidelity": str(self.mlp_math_fidelity),
            "gelu": "tanh" if self.gelu_approx else "erf",
            "head_policy": self.head_policy,
            "head_linear_dtype": str(self.head_linear_dtype),
            "head_fidelity": str(self.head_fidelity),
            "scorer_fp32_out": self.scorer_fp32_out,
            "scorer_gelu": "tanh" if self.scorer_gelu_approx else "erf",
            "fp32_dest_acc_en": self.fp32_dest_acc_en,
            "packer_l1_acc": self.packer_l1_acc,
            "dst_full_sync_en": self.dst_full_sync_en,
            "dest_tiles": self.dest_tiles,
            "residual_dtype": str(self.residual_dtype),
        }


POLICIES = {
    "bf16_hifi4": PrecisionPolicy(
        "bf16_hifi4", linear_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4, gelu_approx=False
    ),
    "bf8w_hifi3": PrecisionPolicy("bf8w_hifi3"),
    "bf8w_hifi2": PrecisionPolicy("bf8w_hifi2", fidelity=ttnn.MathFidelity.HiFi2),
    "bf8w_lofi_mlp": PrecisionPolicy("bf8w_lofi_mlp", mlp_fidelity=ttnn.MathFidelity.LoFi),
    "bf8_act": PrecisionPolicy("bf8_act", act_dtype=ttnn.bfloat8_b),
    "bf8w_hifi3_head_bf16": PrecisionPolicy("bf8w_hifi3_head_bf16", head_policy="bf16_hifi4"),
    "bf8w_hifi3_fp32res": PrecisionPolicy("bf8w_hifi3_fp32res", residual_dtype=ttnn.float32),
    "bf8w_hifi4": PrecisionPolicy("bf8w_hifi4", fidelity=ttnn.MathFidelity.HiFi4),
    "bf8w_hifi3_erf": PrecisionPolicy("bf8w_hifi3_erf", gelu_approx=False),
    "bf16w_hifi3": PrecisionPolicy("bf16w_hifi3", linear_dtype=ttnn.bfloat16),
    "bf16_hifi4_fp32res": PrecisionPolicy(
        "bf16_hifi4_fp32res", linear_dtype=ttnn.bfloat16, fidelity=ttnn.MathFidelity.HiFi4, gelu_approx=False, residual_dtype=ttnn.float32
    ),
}
DEFAULT_POLICY_NAME = "bf8w_hifi3_erf"
DEFAULT_POLICY = POLICIES[DEFAULT_POLICY_NAME]


def policy_from_name(name: Optional[str]) -> PrecisionPolicy:
    if name is None or name == "":
        return DEFAULT_POLICY
    if name not in POLICIES:
        raise KeyError(f"unknown precision policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name]


@dataclass(frozen=True)
class PortConfig:
    """Placement and program-config choices for one device; every field is an A/B lever of stage 3."""

    down_grid: str = "minimal_11x10"
    wo_minimal_min_rows: int = 4096
    qkv_mode: str = "minimal_11x10"
    geglu_plan: str = "sharded"
    intermediate_pad: int = 2816
    interleaved_pad: int = 2816
    mlp_grid: Tuple[int, int] = (11, 8)
    wo_program_config: bool = False
    gelu_separate: bool = False
    shard_grid: Tuple[int, int] = (8, 8)
    shard_min_tiles_per_core: int = 12
    shard_max_rows: int = 2048
    shard_mlp_norm: bool = True
    resident_residual: bool = True
    l1_attention_max_rows: int = 4096
    l1_chain_small_chunk_rows: int = 4096
    rotary_shard_min_rows: int = 24576
    rotary_shard_max_bytes_per_core: int = 0
    sdpa_grid: str = "8x8"
    sdpa_q_chunk: Optional[int] = None
    sdpa_k_chunk: Optional[int] = None
    sdpa_large_chunk_rows: int = 768
    qkv_out_block_h_max: int = 8
    minimal_block: Tuple[int, int, int] = (8, 8, 8)
    minimal_grid: Tuple[int, int] = (11, 10)

    def with_(self, **kw):
        return replace(self, **kw)

    def describe(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


DEFAULT_PORT = PortConfig()


def expected_tensor_count(config) -> int:
    return 6 * config.num_hidden_layers + 2


def padded_intermediate(intermediate_size: int, pad: int) -> int:
    if pad <= 0:
        return intermediate_size
    if pad < intermediate_size or pad % TILE:
        raise ValueError(f"intermediate pad {pad} must be a tile multiple >= {intermediate_size}")
    return pad


def pick_bucket(n: int, buckets: Sequence[int]) -> int:
    if n <= 0:
        raise ValueError(f"need a positive size, got {n}")
    for b in sorted(buckets):
        if b >= n:
            return b
    raise ValueError(f"size {n} exceeds the largest bucket {max(buckets)}")


def grid_size(device):
    g = device.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def select_down_projection_grid(grid_x: int, grid_y: int):
    if grid_x < 8 or grid_y < 8:
        return None
    return ttnn.CoreGrid(y=8, x=8)


def down_projection_core_grid(device, port: PortConfig = DEFAULT_PORT):
    if port.down_grid == "auto":
        return None
    gx, gy = grid_size(device)
    return select_down_projection_grid(gx, gy)


STAGE1_PORT = PortConfig(
    down_grid="8x8",
    qkv_mode="mcast_8x8",
    interleaved_pad=0,
    mlp_grid=(8, 8),
    l1_attention_max_rows=2048,
    rotary_shard_max_bytes_per_core=512 * 1024,
    sdpa_grid="full",
)


def minimal_matmul_config(device, port: PortConfig = DEFAULT_PORT):
    gx, gy = grid_size(device)
    mx, my = port.minimal_grid
    if gx < mx or gy < my:
        mx, my = min(gx, mx), min(gy, my)
    m, k, n = port.minimal_block
    return ttnn.MinimalMatmulConfig(
        M_block_size=m,
        K_block_size=k,
        N_block_size=n,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(mx, my),
    )


def largest_divisor_at_most(n: int, cap: int) -> int:
    for v in range(min(n, cap), 0, -1):
        if n % v == 0:
            return v
    return 1


def qkv_in0_block_w(hidden_size: int) -> int:
    k_tiles = hidden_size // TILE
    for w in (8, 4, 2, 1):
        if k_tiles % w == 0:
            return w
    return 1


def qkv_matmul_program_config(device, batch_size, seq_len, hidden_size, policy=DEFAULT_POLICY, port=DEFAULT_PORT):
    if port.qkv_mode != "mcast_8x8":
        return None
    gx, gy = grid_size(device)
    if gx < 8 or gy < 8:
        return None
    rows = batch_size * seq_len
    if rows % TILE or (3 * hidden_size) % TILE:
        return None
    m_tiles, n_tiles = rows // TILE, (3 * hidden_size) // TILE
    if m_tiles % 8 or n_tiles % 8:
        return None
    per_core_m, per_core_n = m_tiles // 8, n_tiles // 8
    in0_block_w = qkv_in0_block_w(hidden_size)
    out_subblock_w = largest_divisor_at_most(per_core_n, policy.dest_tiles)
    out_block_h = largest_divisor_at_most(per_core_m, port.qkv_out_block_h_max)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


SDPA_MEASURED_SEQ_LENS = (256, 512, 768, 1024)


def sdpa_program_config(device, seq_len, rows, port: PortConfig = DEFAULT_PORT):
    if seq_len not in SDPA_MEASURED_SEQ_LENS:
        return None
    chunk = 256 if rows >= port.sdpa_large_chunk_rows else 128
    if port.l1_chain_small_chunk_rows <= rows <= port.l1_attention_max_rows:
        chunk = 128
    q_chunk = port.sdpa_q_chunk or min(chunk, seq_len)
    k_chunk = port.sdpa_k_chunk or min(chunk, seq_len)
    gx, gy = grid_size(device)
    if port.sdpa_grid == "8x8":
        gx, gy = min(gx, 8), min(gy, 8)
    grid = ttnn.CoreCoord(gx, gy)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


def attention_interleaved(rows: int, port: PortConfig = DEFAULT_PORT):
    if rows <= port.l1_attention_max_rows:
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def rotary_shard_config(shape, port: PortConfig = DEFAULT_PORT):
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    head_dim = int(shape[-1])
    tile_rows = rows // TILE
    grid = next(((8, y) for y in (8, 6, 4, 2) if tile_rows % (8 * y) == 0), None)
    if grid is None or rows < port.rotary_shard_min_rows:
        return None
    per_core_bytes = rows // (grid[0] * grid[1]) * head_dim * 2
    if per_core_bytes > port.rotary_shard_max_bytes_per_core:
        return None
    return ttnn.create_sharded_memory_config(
        shape=(1, rows, head_dim),
        core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


class MlpShardPlan(NamedTuple):
    hidden_memory: object
    intermediate_memory: object
    act_matmul: object
    gate_matmul: object
    down_matmul: object
    norm: object
    width: int


def _sharded_matmul(m_t, in_width, out_width, grid, fused_activation, dest_tiles):
    gx, gy = grid
    k_t, n_t = in_width // TILE, out_width // TILE
    per_core_n = n_t // gx
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=k_t // gx,
        out_subblock_h=1,
        out_subblock_w=largest_divisor_at_most(per_core_n, dest_tiles),
        per_core_M=m_t // gy,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def mlp_shard_plan(
    device, batch_size, seq_len, hidden_size, intermediate_size, policy=DEFAULT_POLICY, port=DEFAULT_PORT
):
    if port.geglu_plan != "sharded":
        return None
    gx, gy = port.shard_grid
    dx, dy = grid_size(device)
    if dx < gx or dy < gy:
        return None
    rows = batch_size * seq_len
    width = padded_intermediate(intermediate_size, port.intermediate_pad)
    if rows % TILE or hidden_size % TILE or width % TILE:
        return None
    m_t, d_t, i_t = rows // TILE, hidden_size // TILE, width // TILE
    if m_t % gy or d_t % gx or i_t % gx:
        return None
    if (m_t * d_t) / (gx * gy) < port.shard_min_tiles_per_core:
        return None
    if rows > port.shard_max_rows:
        return None

    def block(w):
        return ttnn.create_sharded_memory_config(
            shape=(1, rows, w),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    block_w = d_t // gx
    norm = None
    if port.shard_mlp_norm:
        norm = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=largest_divisor_at_most(block_w, 4),
            block_h=m_t // gy,
            block_w=block_w,
            inplace=False,
        )
    dest = policy.dest_tiles
    return MlpShardPlan(
        hidden_memory=block(hidden_size),
        intermediate_memory=block(width),
        act_matmul=_sharded_matmul(m_t, hidden_size, width, (gx, gy), policy.gelu_activation(not port.gelu_separate), dest),
        gate_matmul=_sharded_matmul(m_t, hidden_size, width, (gx, gy), None, dest),
        down_matmul=_sharded_matmul(m_t, width, hidden_size, (gx, gy), None, dest),
        norm=norm,
        width=width,
    )


def mlp_up_projection_program_config(
    device, batch_size, seq_len, hidden_size, width, fuse_gelu, policy=DEFAULT_POLICY, port=DEFAULT_PORT
):
    gx, gy = port.mlp_grid
    dx, dy = grid_size(device)
    if dx < gx or dy < gy:
        return None
    rows = batch_size * seq_len
    if rows % TILE or width % TILE or hidden_size % TILE:
        return None
    m_tiles, n_tiles, k_tiles = rows // TILE, width // TILE, hidden_size // TILE
    if m_tiles % gy or n_tiles % gx:
        return None
    per_core_m, per_core_n = m_tiles // gy, n_tiles // gx
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=largest_divisor_at_most(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=largest_divisor_at_most(per_core_n, policy.dest_tiles),
        out_block_h=largest_divisor_at_most(per_core_m, port.qkv_out_block_h_max),
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=policy.gelu_activation(fuse_gelu),
    )


def down_projection_program_config(device, batch_size, seq_len, in_width, out_width, policy=DEFAULT_POLICY, port=DEFAULT_PORT):
    gx, gy = 8, 8
    dx, dy = grid_size(device)
    if dx < gx or dy < gy:
        return None
    rows = batch_size * seq_len
    if rows % TILE or in_width % TILE or out_width % TILE:
        return None
    m_tiles, n_tiles, k_tiles = rows // TILE, out_width // TILE, in_width // TILE
    if m_tiles % gy or n_tiles % gx:
        return None
    per_core_m, per_core_n = m_tiles // gy, n_tiles // gx
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=largest_divisor_at_most(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=largest_divisor_at_most(per_core_n, policy.dest_tiles),
        out_block_h=largest_divisor_at_most(per_core_m, port.qkv_out_block_h_max),
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


class BucketPlan(NamedTuple):
    batch_size: int
    seq_len: int
    rows: int
    attention_memory: object
    down_core_grid: object
    qkv_program_config: object
    minimal_config: object
    sdpa_program_config: object
    mlp_shard: object
    mlp_width: int
    wo_program_config: object = None
    mlp_down_program_config: object = None
    qkv_minimal: bool = False
    wo_minimal: bool = False


def bucket_plan(device, config, batch_size, seq_len, policy=DEFAULT_POLICY, port=DEFAULT_PORT) -> BucketPlan:
    rows = batch_size * seq_len
    shard = mlp_shard_plan(device, batch_size, seq_len, config.hidden_size, config.intermediate_size, policy, port)
    width = shard.width if shard is not None else padded_intermediate(config.intermediate_size, port.interleaved_pad)
    return BucketPlan(
        batch_size=batch_size,
        seq_len=seq_len,
        rows=rows,
        attention_memory=attention_interleaved(rows, port),
        down_core_grid=down_projection_core_grid(device, port),
        qkv_program_config=qkv_matmul_program_config(device, batch_size, seq_len, config.hidden_size, policy, port),
        minimal_config=minimal_matmul_config(device, port)
        if (port.qkv_mode == "minimal_11x10" or (port.down_grid == "minimal_11x10" and rows >= port.wo_minimal_min_rows))
        else None,
        sdpa_program_config=sdpa_program_config(device, seq_len, rows, port),
        mlp_shard=shard,
        mlp_width=width,
        wo_program_config=down_projection_program_config(device, batch_size, seq_len, config.hidden_size, config.hidden_size, policy, port)
        if port.wo_program_config
        else None,
        mlp_down_program_config=down_projection_program_config(device, batch_size, seq_len, width, config.hidden_size, policy, port)
        if (port.wo_program_config and shard is None)
        else None,
        qkv_minimal=port.qkv_mode == "minimal_11x10",
        wo_minimal=port.down_grid == "minimal_11x10" and rows >= port.wo_minimal_min_rows,
    )


def describe_plan(plan: BucketPlan) -> dict:
    def pc(x):
        return None if x is None else repr(x)

    return {
        "batch_size": plan.batch_size,
        "seq_len": plan.seq_len,
        "rows": plan.rows,
        "attention_memory": "L1" if plan.attention_memory == ttnn.L1_MEMORY_CONFIG else "DRAM",
        "down_core_grid": pc(plan.down_core_grid),
        "qkv_program_config": pc(plan.qkv_program_config),
        "minimal_config": pc(plan.minimal_config),
        "sdpa_program_config": pc(plan.sdpa_program_config),
        "mlp_sharded": plan.mlp_shard is not None,
        "mlp_width": plan.mlp_width,
        "wo_program_config": pc(plan.wo_program_config),
        "mlp_down_program_config": pc(plan.mlp_down_program_config),
        "qkv_minimal": plan.qkv_minimal,
        "wo_minimal": plan.wo_minimal,
    }
