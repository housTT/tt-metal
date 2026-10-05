# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.autoports.convaiinnovations_laya.tt import model_config as mc


class _Device:
    def __init__(self, x=11, y=10):
        self._grid = type("G", (), {"x": x, "y": y})()

    def compute_with_storage_grid_size(self):
        return self._grid

    def arch(self):
        return ttnn.device.Arch.BLACKHOLE


P150 = _Device(11, 10)
N300 = _Device(8, 8)
HIDDEN, INTER, LAYERS = 1024, 2624, 28


class _Config:
    hidden_size = HIDDEN
    intermediate_size = INTER
    num_hidden_layers = LAYERS
    num_attention_heads = 16


def test_policy_table_matches_the_plan():
    assert set(mc.POLICIES) == {
        "bf16_hifi4",
        "bf8w_hifi3",
        "bf8w_hifi2",
        "bf8w_lofi_mlp",
        "bf8_act",
        "bf8w_hifi3_head_bf16",
        "bf8w_hifi3_fp32res",
        "bf16_hifi4_fp32res",
        "bf8w_hifi4",
        "bf16w_hifi3",
        "bf8w_hifi3_erf",
    }
    assert mc.POLICIES["bf8w_hifi3_fp32res"].residual_fp32 and not mc.DEFAULT_POLICY.residual_fp32
    p = mc.DEFAULT_POLICY
    assert p.name == "bf8w_hifi3_erf"
    assert p.linear_dtype == ttnn.bfloat8_b and p.act_dtype == ttnn.bfloat16
    assert p.fidelity == ttnn.MathFidelity.HiFi3 and not p.gelu_approx and p.scorer_fp32_out
    assert mc.POLICIES["bf8w_hifi3"].gelu_approx
    assert mc.POLICIES["bf16_hifi4"].linear_dtype == ttnn.bfloat16
    assert mc.POLICIES["bf16_hifi4"].gelu_approx is False
    assert mc.POLICIES["bf8w_lofi_mlp"].mlp_math_fidelity == ttnn.MathFidelity.LoFi
    assert mc.POLICIES["bf8w_lofi_mlp"].fidelity == ttnn.MathFidelity.HiFi3
    assert mc.POLICIES["bf8_act"].act_dtype == ttnn.bfloat8_b
    hb = mc.POLICIES["bf8w_hifi3_head_bf16"]
    assert hb.head_linear_dtype == ttnn.bfloat16 and hb.head_fidelity == ttnn.MathFidelity.HiFi4
    assert hb.linear_dtype == ttnn.bfloat8_b


@pytest.mark.parametrize("name", sorted(mc.POLICIES))
def test_compute_configs_come_from_the_device_arch(name):
    p = mc.POLICIES[name]
    for group in ("attn", "mlp", "head", "scorer", "norm"):
        cc = p.compute_config(P150, group)
        assert cc.fp32_dest_acc_en is True
        assert cc.math_approx_mode is False
    assert p.compute_config(P150, "attn").math_fidelity == p.fidelity
    assert p.compute_config(P150, "mlp").math_fidelity == p.mlp_math_fidelity
    assert p.compute_config(P150, "norm").math_fidelity == ttnn.MathFidelity.HiFi4
    assert p.dest_tiles == 4


def test_dest_cap_follows_sync_and_accumulation():
    assert mc.PrecisionPolicy("x").dest_tiles == 4
    assert mc.PrecisionPolicy("x", dst_full_sync_en=True).dest_tiles == 8
    assert mc.PrecisionPolicy("x", fp32_dest_acc_en=False).dest_tiles == 8
    assert mc.PrecisionPolicy("x", fp32_dest_acc_en=False, dst_full_sync_en=True).dest_tiles == 16


def test_unknown_policy_name_is_rejected():
    with pytest.raises(KeyError):
        mc.policy_from_name("bf4_everything")
    assert mc.policy_from_name(None) is mc.DEFAULT_POLICY


def test_tensor_count_is_170():
    assert mc.expected_tensor_count(_Config) == 170


def test_shipped_port_defaults_match_the_stage_3_decision():
    d = mc.DEFAULT_PORT
    assert d.qkv_mode == "minimal_11x10" and d.down_grid == "minimal_11x10" and d.wo_minimal_min_rows == 4096
    assert d.interleaved_pad == 2816 and d.mlp_grid == (11, 8) and d.geglu_plan == "sharded" and d.intermediate_pad == 2816
    assert d.sdpa_grid == "8x8" and d.rotary_shard_max_bytes_per_core == 0 and d.l1_attention_max_rows == 4096
    assert mc.sdpa_program_config(P150, 512, 4096).q_chunk_size == 128
    assert mc.sdpa_program_config(P150, 512, 8192).q_chunk_size == 256
    assert mc.sdpa_program_config(P150, 512, 2048).q_chunk_size == 256
    assert mc.attention_interleaved(4096) == ttnn.L1_MEMORY_CONFIG and mc.attention_interleaved(8192) == ttnn.DRAM_MEMORY_CONFIG
    p1 = mc.bucket_plan(P150, _Config, 1, 512)
    assert p1.qkv_minimal and not p1.wo_minimal and p1.minimal_config is not None and p1.mlp_width == 2816
    p2 = mc.bucket_plan(P150, _Config, 2, 512)
    assert p2.mlp_shard is not None and not p2.wo_minimal
    p8 = mc.bucket_plan(P150, _Config, 8, 512)
    assert p8.qkv_minimal and p8.wo_minimal and p8.mlp_shard is None and p8.mlp_width == 2816
    assert p8.attention_memory == ttnn.L1_MEMORY_CONFIG and p8.sdpa_program_config.q_chunk_size == 128
    assert (p8.sdpa_program_config.compute_with_storage_grid_size.x, p8.sdpa_program_config.compute_with_storage_grid_size.y) == (8, 8)
    assert mc.rotary_shard_config((8, 16, 512, 64)) is None
    s1 = mc.STAGE1_PORT
    assert s1.qkv_mode == "mcast_8x8" and s1.down_grid == "8x8" and s1.interleaved_pad == 0 and s1.sdpa_grid == "full"


def test_down_projection_grid_policy():
    assert (mc.select_down_projection_grid(11, 10).x, mc.select_down_projection_grid(11, 10).y) == (8, 8)
    assert (mc.select_down_projection_grid(8, 8).x, mc.select_down_projection_grid(8, 8).y) == (8, 8)
    for gx, gy in ((7, 8), (8, 7), (4, 8), (1, 1)):
        assert mc.select_down_projection_grid(gx, gy) is None
    assert mc.down_projection_core_grid(P150, mc.DEFAULT_PORT.with_(down_grid="auto")) is None
    assert mc.down_projection_core_grid(P150) is not None


def test_minimal_matmul_alternative_targets_the_full_p150_grid():
    cfg = mc.minimal_matmul_config(P150)
    g = cfg.compute_with_storage_grid_size
    assert (g.x, g.y) == (11, 10)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (8, 8, 8)
    g2 = mc.minimal_matmul_config(N300).compute_with_storage_grid_size
    assert (g2.x, g2.y) == (8, 8)


@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 10, 16, 32, 50, 64])
@pytest.mark.parametrize("seq_len", [512, 1024])
def test_qkv_program_config_covers_every_bucket(batch, seq_len):
    cfg = mc.qkv_matmul_program_config(P150, batch, seq_len, HIDDEN, port=mc.STAGE1_PORT)
    assert cfg is not None, "the largest matmul must not fall back to ttnn's choice at any bucket"
    assert cfg.in0_block_w == 8
    assert cfg.per_core_M == batch * seq_len // 32 // 8
    assert cfg.per_core_N == 3 * HIDDEN // 32 // 8 == 12
    assert cfg.out_subblock_h == 1 and cfg.out_subblock_w == 4
    assert cfg.out_subblock_h * cfg.out_subblock_w <= mc.DEFAULT_POLICY.dest_tiles
    assert cfg.per_core_M % cfg.out_block_h == 0 and cfg.out_block_h <= 8
    assert cfg.out_block_w == cfg.per_core_N


def test_qkv_program_config_declines_small_grid_and_odd_shapes():
    assert mc.qkv_matmul_program_config(_Device(7, 8), 1, 512, HIDDEN, port=mc.STAGE1_PORT) is None
    assert mc.qkv_matmul_program_config(P150, 1, 300, HIDDEN, port=mc.STAGE1_PORT) is None
    assert mc.qkv_matmul_program_config(P150, 1, 512, HIDDEN, port=mc.DEFAULT_PORT.with_(qkv_mode="auto")) is None
    assert mc.qkv_matmul_program_config(P150, 1, 512, HIDDEN) is None


def test_intermediate_padding_arithmetic():
    assert mc.padded_intermediate(INTER, 0) == INTER
    assert mc.padded_intermediate(INTER, 2816) == 2816
    assert mc.padded_intermediate(INTER, 3072) == 3072
    assert 2816 // 32 == 88 and 88 % 8 == 0
    assert 3072 // 32 == 96 and 96 % 8 == 0
    assert (INTER // 32) % 8 != 0
    with pytest.raises(ValueError):
        mc.padded_intermediate(INTER, 2600)
    with pytest.raises(ValueError):
        mc.padded_intermediate(INTER, 2700)


@pytest.mark.parametrize("pad,per_core_n,subblock_w,down_in0", [(2816, 11, 1, 11), (3072, 12, 4, 12)])
def test_geglu_shard_plan_geometry(pad, per_core_n, subblock_w, down_in0):
    port = mc.DEFAULT_PORT.with_(intermediate_pad=pad)
    plan = mc.mlp_shard_plan(P150, 4, 512, HIDDEN, INTER, port=port)
    assert plan is not None and plan.width == pad
    for cfg in (plan.act_matmul, plan.gate_matmul, plan.down_matmul):
        g = cfg.compute_with_storage_grid_size
        assert (g.x, g.y) == (8, 8)
        assert cfg.per_core_M == 4 * 512 // 32 // 8
        assert cfg.out_subblock_h * cfg.out_subblock_w <= 4
    assert plan.act_matmul.per_core_N == per_core_n and plan.act_matmul.out_subblock_w == subblock_w
    assert plan.act_matmul.in0_block_w == 4
    assert plan.down_matmul.in0_block_w == down_in0 and plan.down_matmul.per_core_N == 4
    assert plan.act_matmul.fused_activation is not None
    assert plan.gate_matmul.fused_activation is None and plan.down_matmul.fused_activation is None
    assert "params=[0]" in repr(plan.act_matmul.fused_activation)
    assert "params=[1]" in repr(mc.mlp_shard_plan(P150, 4, 512, HIDDEN, INTER, policy=mc.POLICIES["bf8w_hifi3"], port=port).act_matmul.fused_activation)
    assert plan.norm is not None and plan.norm.block_w == 4


def test_geglu_shard_plan_thresholds():
    assert mc.mlp_shard_plan(P150, 1, 512, HIDDEN, INTER) is None
    assert mc.mlp_shard_plan(P150, 2, 512, HIDDEN, INTER) is not None
    assert mc.mlp_shard_plan(P150, 4, 512, HIDDEN, INTER) is not None
    assert mc.mlp_shard_plan(P150, 8, 512, HIDDEN, INTER) is None
    assert mc.mlp_shard_plan(P150, 8, 512, HIDDEN, INTER, port=mc.DEFAULT_PORT.with_(shard_max_rows=4096)) is not None
    assert mc.mlp_shard_plan(P150, 4, 512, HIDDEN, INTER, port=mc.DEFAULT_PORT.with_(geglu_plan="interleaved")) is None
    assert mc.mlp_shard_plan(_Device(4, 8), 4, 512, HIDDEN, INTER) is None


def test_interleaved_up_projection_config_needs_a_padded_width():
    assert mc.mlp_up_projection_program_config(P150, 8, 512, HIDDEN, INTER, True) is None
    g8 = mc.DEFAULT_PORT.with_(mlp_grid=(8, 8))
    cfg = mc.mlp_up_projection_program_config(P150, 8, 512, HIDDEN, 2816, True, port=g8)
    assert cfg is not None and cfg.per_core_N == 11 and cfg.out_subblock_w == 1 and cfg.out_block_h == 8
    assert mc.mlp_up_projection_program_config(P150, 8, 512, HIDDEN, 2816, False, port=g8).fused_activation is None
    shipped = mc.mlp_up_projection_program_config(P150, 8, 512, HIDDEN, 2816, True)
    assert shipped.per_core_N == 8 and shipped.out_subblock_w == 4 and shipped.compute_with_storage_grid_size.x == 11
    wide = mc.mlp_up_projection_program_config(P150, 64, 512, HIDDEN, 2816, True, port=mc.DEFAULT_PORT.with_(mlp_grid=(11, 8)))
    assert wide is not None and wide.per_core_N == 8 and wide.per_core_M == 128 and wide.out_block_h == 8 and wide.out_subblock_w == 4
    assert mc.mlp_up_projection_program_config(P150, 1, 512, HIDDEN, 3072, True, port=mc.DEFAULT_PORT.with_(mlp_grid=(11, 8))) is None
    down = mc.down_projection_program_config(P150, 64, 512, 2816, HIDDEN)
    assert down.per_core_N == 4 and down.in0_block_w == 8 and down.per_core_M == 128 and down.out_block_h == 8
    plan = mc.bucket_plan(P150, _Config, 64, 512, port=mc.DEFAULT_PORT.with_(interleaved_pad=2816, wo_program_config=True))
    assert plan.mlp_width == 2816 and plan.wo_program_config is not None and plan.mlp_down_program_config is not None
    assert mc.bucket_plan(P150, _Config, 64, 512).wo_program_config is None
    assert mc.bucket_plan(P150, _Config, 64, 512, port=mc.STAGE1_PORT).mlp_width == INTER


@pytest.mark.parametrize("seq_len,rows,chunk", [(512, 512, 128), (512, 1024, 256), (1024, 1024, 256), (256, 256, 128)])
def test_sdpa_chunk_table(seq_len, rows, chunk):
    cfg = mc.sdpa_program_config(P150, seq_len, rows, mc.STAGE1_PORT)
    assert cfg.q_chunk_size == chunk and cfg.k_chunk_size == chunk
    g = cfg.compute_with_storage_grid_size
    assert (g.x, g.y) == (11, 10)
    shipped = mc.sdpa_program_config(P150, seq_len, rows)
    assert (shipped.compute_with_storage_grid_size.x, shipped.compute_with_storage_grid_size.y) == (8, 8) and shipped.q_chunk_size == chunk
    assert mc.sdpa_program_config(P150, 300, 300) is None
    small = mc.sdpa_program_config(P150, seq_len, rows, mc.DEFAULT_PORT.with_(sdpa_grid="8x8", sdpa_q_chunk=64))
    assert (small.compute_with_storage_grid_size.x, small.q_chunk_size) == (8, 64)


def test_attention_memory_by_row_count():
    assert mc.attention_interleaved(2048, mc.STAGE1_PORT) == ttnn.L1_MEMORY_CONFIG
    assert mc.attention_interleaved(4096, mc.STAGE1_PORT) == ttnn.DRAM_MEMORY_CONFIG
    assert mc.attention_interleaved(8192, mc.DEFAULT_PORT.with_(l1_attention_max_rows=8192)) == ttnn.L1_MEMORY_CONFIG
    assert mc.attention_interleaved(512, mc.DEFAULT_PORT.with_(l1_attention_max_rows=0)) == ttnn.DRAM_MEMORY_CONFIG


def test_rotary_shard_bounds():
    s1 = mc.STAGE1_PORT
    assert mc.rotary_shard_config((1, 16, 512, 64), s1) is None
    assert mc.rotary_shard_config((8, 16, 512, 64), s1) is not None
    assert mc.rotary_shard_config((16, 16, 512, 64), s1) is not None
    assert mc.rotary_shard_config((64, 16, 512, 64), s1) is None
    assert mc.rotary_shard_config((64, 16, 512, 64), s1.with_(rotary_shard_max_bytes_per_core=2**21)) is not None
    assert mc.rotary_shard_config((8, 16, 512, 64)) is None


def test_bucket_selection():
    assert mc.pick_bucket(1, mc.ROW_BUCKETS) == 1
    assert mc.pick_bucket(5, mc.ROW_BUCKETS) == 8
    assert mc.pick_bucket(5, (1, 2, 4, 5, 8, 10, 50, 64)) == 5
    assert mc.pick_bucket(50, mc.ROW_BUCKETS) == 64
    assert mc.pick_bucket(64, mc.ROW_BUCKETS) == 64
    assert mc.pick_bucket(300, mc.SEQ_BUCKETS) == 512
    assert mc.pick_bucket(513, mc.SEQ_BUCKETS_SIBLING) == 1024
    with pytest.raises(ValueError):
        mc.pick_bucket(65, mc.ROW_BUCKETS)
    with pytest.raises(ValueError):
        mc.pick_bucket(0, mc.ROW_BUCKETS)
    with pytest.raises(ValueError):
        mc.pick_bucket(1025, mc.SEQ_BUCKETS_SIBLING)


def test_bucket_plan_summary_is_serialisable():
    plan = mc.bucket_plan(P150, _Config, 8, 512, port=mc.STAGE1_PORT)
    d = mc.describe_plan(plan)
    assert d["attention_memory"] == "DRAM" and d["mlp_sharded"] is False and d["mlp_width"] == INTER
    assert mc.describe_plan(mc.bucket_plan(P150, _Config, 4, 512))["attention_memory"] == "L1"
    assert mc.describe_plan(mc.bucket_plan(P150, _Config, 8, 512))["mlp_width"] == 2816
    plan2 = mc.bucket_plan(P150, _Config, 2, 512)
    assert plan2.mlp_shard is not None and plan2.mlp_width == 2816
    plan64 = mc.bucket_plan(P150, _Config, 64, 512)
    assert mc.describe_plan(plan64)["attention_memory"] == "DRAM"
    assert mc.bucket_plan(P150, _Config, 8, 512, port=mc.DEFAULT_PORT.with_(qkv_mode="minimal_11x10")).minimal_config is not None
