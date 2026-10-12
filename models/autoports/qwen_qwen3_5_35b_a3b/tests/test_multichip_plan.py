# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from ..tt import multichip_decoder as mc
from ..tt.functional_decoder import FunctionalDecoderConfig

LAYERS = ("linear_attention", "linear_attention", "linear_attention", "full_attention")


def config(**overrides):
    values = dict(
        dim=64,
        norm_eps=1e-6,
        n_layers=4,
        n_heads=16,
        n_kv_heads=2,
        head_dim=8,
        max_seq_len=1024,
        attention_type_list=LAYERS,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        moe_num_experts=16,
        moe_top_k=4,
        moe_intermediate_size=32,
        moe_shared_intermediate_size=32,
        moe_norm_topk_prob=True,
        moe_decoder_sparse_step=1,
        mlp_only_layers=frozenset(),
    )
    values.update(overrides)
    return FunctionalDecoderConfig(**values)


def state(cfg, layer):
    g = torch.Generator().manual_seed(layer)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    p = f"layers.{layer}."
    h, kv, d, hidden = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.dim
    out = {p + "input_layernorm.weight": rand(hidden), p + "post_attention_layernorm.weight": rand(hidden),
           p + "mlp.gate.weight": rand(cfg.moe_num_experts, hidden),
           p + "mlp.shared_expert_gate.weight": rand(1, hidden),
           p + "mlp.experts.gate_up_proj": rand(cfg.moe_num_experts, 2 * cfg.moe_intermediate_size, hidden),
           p + "mlp.experts.down_proj": rand(cfg.moe_num_experts, hidden, cfg.moe_intermediate_size),
           p + "mlp.shared_expert.gate_proj.weight": rand(cfg.moe_shared_intermediate_size, hidden),
           p + "mlp.shared_expert.up_proj.weight": rand(cfg.moe_shared_intermediate_size, hidden),
           p + "mlp.shared_expert.down_proj.weight": rand(hidden, cfg.moe_shared_intermediate_size)}
    if cfg.is_full_attention_layer(layer):
        out.update({p + "self_attn.q_proj.weight": rand(h * 2 * d, hidden), p + "self_attn.k_proj.weight": rand(kv * d, hidden),
                    p + "self_attn.v_proj.weight": rand(kv * d, hidden), p + "self_attn.o_proj.weight": rand(hidden, h * d),
                    p + "self_attn.q_norm.weight": rand(d), p + "self_attn.k_norm.weight": rand(d)})
    else:
        qk, v, nv = cfg.linear_q_dim, cfg.linear_v_dim, cfg.linear_num_value_heads
        out.update({p + "linear_attn.qkv_proj.weight": rand(2 * qk + v, hidden),
                    p + "linear_attn.in_proj_a.weight": rand(nv, hidden), p + "linear_attn.in_proj_b.weight": rand(nv, hidden),
                    p + "linear_attn.in_proj_z.weight": rand(v, hidden), p + "linear_attn.out_proj.weight": rand(hidden, v),
                    p + "linear_attn.dt_bias": rand(nv), p + "linear_attn.A_log": rand(nv),
                    p + "linear_attn.norm.weight": rand(cfg.linear_value_head_dim),
                    p + "linear_attn.q_conv.weight": rand(qk, 1, 4), p + "linear_attn.k_conv.weight": rand(qk, 1, 4),
                    p + "linear_attn.v_conv.weight": rand(v, 1, 4)})
    return out


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("layer", [0, 3])
def test_tensor_parallel_slices_reassemble(tp, layer):
    cfg = config()
    full = state(cfg, layer)
    parts = [mc.tensor_parallel_state(full, layer, tp, rank, cfg) for rank in range(tp)]
    p = f"layers.{layer}."
    for name in ("input_layernorm.weight", "mlp.gate.weight", "mlp.shared_expert_gate.weight"):
        assert all(torch.equal(part[p + name], full[p + name]) for part in parts)
    cat = lambda name, dim: torch.cat([part[p + name] for part in parts], dim=dim)
    assert torch.equal(cat("mlp.shared_expert.gate_proj.weight", 0), full[p + "mlp.shared_expert.gate_proj.weight"])
    assert torch.equal(cat("mlp.shared_expert.down_proj.weight", 1), full[p + "mlp.shared_expert.down_proj.weight"])
    assert torch.equal(cat("mlp.experts.down_proj", 2), full[p + "mlp.experts.down_proj"])
    width = cfg.moe_intermediate_size // tp
    gate = torch.cat([part[p + "mlp.experts.gate_up_proj"][:, :width] for part in parts], dim=1)
    up = torch.cat([part[p + "mlp.experts.gate_up_proj"][:, width:] for part in parts], dim=1)
    assert torch.equal(torch.cat((gate, up), dim=1), full[p + "mlp.experts.gate_up_proj"])
    if cfg.is_full_attention_layer(layer):
        h, d = cfg.n_heads, cfg.head_dim
        q = torch.cat([part[p + "self_attn.q_proj.weight"].reshape(h // tp, 2 * d, -1) for part in parts])
        assert torch.equal(q.reshape(-1, cfg.dim), full[p + "self_attn.q_proj.weight"])
        assert torch.equal(cat("self_attn.o_proj.weight", 1), full[p + "self_attn.o_proj.weight"])
        for rank, part in enumerate(parts):
            group = rank * h // tp // (h // cfg.n_kv_heads)
            expected = full[p + "self_attn.k_proj.weight"].reshape(cfg.n_kv_heads, d, -1)[group]
            assert torch.equal(part[p + "self_attn.k_proj.weight"], expected)
    else:
        qk, v = cfg.linear_q_dim, cfg.linear_v_dim
        rows = [part[p + "linear_attn.qkv_proj.weight"] for part in parts]
        q = torch.cat([r[: qk // tp] for r in rows])
        k = torch.cat([r[qk // tp : 2 * qk // tp] for r in rows])
        vv = torch.cat([r[2 * qk // tp :] for r in rows])
        assert torch.equal(torch.cat((q, k, vv)), full[p + "linear_attn.qkv_proj.weight"])
        assert torch.equal(cat("linear_attn.out_proj.weight", 1), full[p + "linear_attn.out_proj.weight"])
        assert torch.equal(cat("linear_attn.A_log", 0), full[p + "linear_attn.A_log"])
        heads_per_key = cfg.linear_num_value_heads // cfg.linear_num_key_heads
        for rank in range(tp):
            value_heads = range(rank * cfg.linear_num_value_heads // tp, (rank + 1) * cfg.linear_num_value_heads // tp)
            key_heads = {head // heads_per_key for head in value_heads}
            assert key_heads == set(range(rank * cfg.linear_num_key_heads // tp, (rank + 1) * cfg.linear_num_key_heads // tp))


@pytest.mark.parametrize("tp", [2, 4])
def test_expert_parallel_owners(tp):
    cfg = config()
    full = state(cfg, 3)
    seen = []
    for rank in range(tp):
        gate_up, down, offset = mc.expert_parallel_state(full, 3, tp, rank, cfg)
        experts = cfg.moe_num_experts // tp
        assert offset == rank * experts and gate_up.shape[0] == down.shape[0] == experts
        seen.append(gate_up)
    assert torch.equal(torch.cat(seen), full["layers.3.mlp.experts.gate_up_proj"])


def test_unknown_tensor_is_rejected():
    cfg = config()
    with pytest.raises(KeyError):
        mc.tensor_parallel_state({"layers.0.unexpected.weight": torch.zeros(1)}, 0, 2, 0, cfg)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_mesh_policy(tp):
    policy = mc.mesh_policy(mc.DEFAULT_POLICIES["full_attention"], tp)
    assert policy["dispatch_weights"] == ("stacked" if tp == 1 else "expert_parallel")
    assert policy["projections"]["attention_out"] == mc.DEFAULT_POLICIES["full_attention"]["projections"]["attention_out"]
    if tp > 1:
        shared = policy["projections"]["shared_down"]
        assert not shared["dram"] and (512 // tp // 32) % shared["k_block"] == 0
        assert policy["shared_decode_minimal_config"]["K_block_size"] == 16
        assert "shared_minimal_config" not in mc.MESH_POLICIES[tp]


@pytest.mark.parametrize("tp,kind,n", [(2, "full_attention", 4608), (4, "full_attention", 2560),
                                       (4, "linear_attention", 3136)])
def test_mesh_policy_attention_in(tp, kind, n):
    policy = mc.mesh_policy(mc.DEFAULT_POLICIES[kind], tp, kind)
    options = policy["projections"]["attention_in"]
    gx, gy = options["decode_grid"]
    assert options["per_n"] * gx * gy >= n // 32 > options["per_n"] * (gx * gy - 1)
    assert 64 % options["k_block"] == 0
    assert options["dtype"] == mc.DEFAULT_POLICIES[kind]["projections"]["attention_in"]["dtype"]
    assert "prefill_minimal" in options


@pytest.mark.parametrize("rows,limit,expected", [(2176, 1, [4096]), (2176, 2, [2048, 128]), (3072, 2, [2048, 1024]),
                                                 (3968, 2, [4096]), (3584, 2, [4096]), (4096, 2, [4096]),
                                                 (96, 1, [128]), (96, 2, [64, 32]), (1152, 2, [1024, 128]),
                                                 (1312, 2, [2048]), (2432, 2, [2048, 512]), (2624, 2, [2048, 1024])])
def test_moe_pieces(rows, limit, expected):
    from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import _moe_pieces

    assert _moe_pieces(rows, [32, 64, 128, 256, 512, 1024, 2048, 4096], limit) == expected


@pytest.mark.parametrize("rows,expected", [(2176, [2048, 128]), (2432, [2048, 512]), (2624, [4096]), (3072, [4096]),
                                           (1152, [1024, 128])])
def test_moe_pieces_merge(rows, expected):
    from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import _moe_pieces

    assert _moe_pieces(rows, [32, 64, 128, 256, 512, 1024, 2048, 4096], 2, 1024) == expected


def test_mesh_policy_moe_pieces():
    assert mc.mesh_policy(mc.DEFAULT_POLICIES["full_attention"], 1).get("prefill_moe_pieces", 1) == 1
    assert mc.mesh_policy(mc.DEFAULT_POLICIES["full_attention"], 2)["prefill_moe_pieces"] == 2
    assert mc.mesh_policy(mc.DEFAULT_POLICIES["linear_attention"], 4)["prefill_moe_pieces"] == 2
    assert mc.mesh_policy(mc.DEFAULT_POLICIES["linear_attention"], 4)["prefill_moe_merge_min"] == 1024
    assert mc.mesh_policy(mc.DEFAULT_POLICIES["full_attention"], 1).get("prefill_moe_merge_min", 0) == 0


@pytest.mark.parametrize("routes,rows", [(32768, 32), (16384, 16), (8192, 8), (1024, 1), (800, 1), (10400, 10)])
def test_scatter_rows(routes, rows):
    from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import _scatter_rows

    chosen = _scatter_rows(routes, 32)
    assert chosen == rows and routes % chosen == 0 and (chosen == 1 or routes // chosen >= 1024)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("kind", ["full_attention", "linear_attention"])
def test_stage5_decode_policy(tp, kind):
    policy = mc.mesh_policy(mc.DEFAULT_POLICIES[kind], tp, kind)
    assert policy["expert_decode_kernel"] and policy["expert_decode_shared"]
    assert policy["router_cached_slices"] and policy["decode_norm_output"] == "l1"
    assert not policy["decode_input_fill"] and not policy.get("decode_norm_fill", False)
    assert policy.get("residual_sharded_decode", False) == (tp > 1)
    assert policy["dispatch_scatter_rows"] == 32
    assert policy["prefill_norm"] == "bf16" and policy["router_topk_prefill"]
    assert policy["gate_pack_fidelity"] == "HiFi2" and policy["projections"]["router"]["fidelity"] == "HiFi2"
    addcmul = policy["shared_addcmul_config"]
    assert addcmul["grid"] == [8, 2] and (512 // tp // 32) % addcmul["K_block_size"] == 0
    assert (2048 // 32 // addcmul["grid"][0]) % addcmul["N_block_size"] == 0
    if kind == "full_attention":
        sdpa = policy["sdpa"]
        assert (sdpa["k_chunk_size"], sdpa["grid"]) == (512, [11, 10])
        assert (sdpa["prefill"]["q_chunk_size"], sdpa["prefill"]["k_chunk_size"]) == (128, 256)
        assert (policy["prefill_long_chunk"], policy["prefill_long_from"]) == (8192, 8192)
        long_sdpa = policy["sdpa_prefill_long"]
        assert (long_sdpa["q_chunk_size"], long_sdpa["k_chunk_size"]) == ((64, 256) if tp == 4 else (128, 512))
    else:
        assert policy["gdn_decode_kernel"] and policy["gdn_decode_gates"] and not policy["conv_decode_fill"]
        assert policy["gdn_flat_qk"] and policy["gdn_gate_mixed"] and policy["gdn_decode_direct"]
        assert policy["conv_shift"] == "gate" and policy["conv_state_rm"]


def test_local_hf_config():
    text = SimpleNamespace(num_attention_heads=16, num_key_value_heads=2, linear_num_key_heads=16,
                           linear_num_value_heads=32, moe_intermediate_size=512, shared_expert_intermediate_size=512,
                           num_experts=256, get_text_config=lambda: text)
    local = mc.local_hf_config(SimpleNamespace(text_config=text), 4).text_config
    assert (local.num_attention_heads, local.num_key_value_heads, local.linear_num_key_heads) == (4, 1, 4)
    assert (local.linear_num_value_heads, local.moe_intermediate_size, local.shared_expert_intermediate_size) == (8, 128, 128)
    assert local.num_experts == 256 and text.num_attention_heads == 16
