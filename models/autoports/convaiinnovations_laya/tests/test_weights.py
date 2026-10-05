# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.autoports.convaiinnovations_laya.tt import weights as W

HEAD_DIM = 64


def test_key_map_is_complete_and_exact(parts, laya_config):
    W.check_encoder_keys(parts["encoder"], laya_config)
    W.check_head_keys(parts, laya_config)
    assert len(parts["encoder"]) == 170
    assert "layers.0.attn_norm.weight" not in parts["encoder"]
    assert "layers.1.attn_norm.weight" in parts["encoder"]
    assert len(parts["head"]) == 24 and len(parts["scorer"]) == 6 and len(parts["act_head"]) == 4
    assert tuple(parts["type_emb"].shape) == (3, 1024)
    assert tuple(parts["temperature"].shape) == (3,)


def test_key_map_rejects_missing_and_extra(parts, laya_config):
    enc = dict(parts["encoder"])
    enc.pop("layers.27.mlp.Wo.weight")
    with pytest.raises(ValueError, match="missing"):
        W.check_encoder_keys(enc, laya_config)
    enc = dict(parts["encoder"])
    enc["layers.0.attn_norm.weight"] = torch.ones(1024)
    with pytest.raises(ValueError, match="extra"):
        W.check_encoder_keys(enc, laya_config)


def test_q_scale_fold_is_exact(parts):
    w = parts["encoder"]["layers.3.attn.Wqkv.weight"]
    f = W.fold_q_scale(w, HEAD_DIM)
    assert torch.equal(f[:1024], w[:1024] / 8)
    assert torch.equal(f[1024:], w[1024:])
    assert not torch.equal(f, w)
    hw = parts["head"]["layers.1.self_attn.in_proj_weight"]
    hf = W.fold_q_scale(hw, HEAD_DIM)
    assert torch.equal(hf[:1024], hw[:1024] / 8) and torch.equal(hf[1024:], hw[1024:])


def test_head_bias_fold_touches_only_the_q_third(parts):
    b = parts["head"]["layers.0.self_attn.in_proj_bias"]
    fb = W.fold_q_scale_bias(b, HEAD_DIM)
    assert torch.equal(fb[:1024], b[:1024] / 8)
    assert torch.equal(fb[1024:2048], b[1024:2048])
    assert torch.equal(fb[2048:], b[2048:])


def test_folded_attention_matches_unfolded_math(parts):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 1024)
    w = parts["head"]["layers.0.self_attn.in_proj_weight"]
    b = parts["head"]["layers.0.self_attn.in_proj_bias"]
    q, k, v = (x @ w.T + b).chunk(3, dim=-1)
    ref = torch.softmax((q.view(1, 8, 16, 64).transpose(1, 2) @ k.view(1, 8, 16, 64).transpose(1, 2).transpose(-1, -2)) / 8, -1)
    fq, fk, fv = (x @ W.fold_q_scale(w, HEAD_DIM).T + W.fold_q_scale_bias(b, HEAD_DIM)).chunk(3, dim=-1)
    got = torch.softmax(fq.view(1, 8, 16, 64).transpose(1, 2) @ fk.view(1, 8, 16, 64).transpose(1, 2).transpose(-1, -2), -1)
    assert torch.allclose(ref, got, atol=1e-6)


def test_geglu_split_and_padding(parts, laya_config):
    wi = parts["encoder"]["layers.16.mlp.Wi.weight"]
    act, gate = W.split_wi(wi, laya_config.intermediate_size)
    assert tuple(act.shape) == (2624, 1024) and tuple(gate.shape) == (2624, 1024)
    assert torch.equal(act, wi[:2624]) and torch.equal(gate, wi[2624:])
    pa = W.pad_up_projection(act, 2816)
    pg = W.pad_up_projection(gate, 3072)
    wo = parts["encoder"]["layers.16.mlp.Wo.weight"]
    po = W.pad_down_projection(wo, 2816)
    assert tuple(pa.shape) == (2816, 1024) and tuple(pg.shape) == (3072, 1024) and tuple(po.shape) == (1024, 2816)
    assert torch.equal(pa[:2624], act) and float(pa[2624:].abs().max()) == 0.0
    assert torch.equal(pg[:2624], gate) and float(pg[2624:].abs().max()) == 0.0
    assert torch.equal(po[:, :2624], wo) and float(po[:, 2624:].abs().max()) == 0.0
    assert W.pad_up_projection(act, 2624) is act


def test_padded_geglu_is_exact_in_torch(parts, laya_config):
    torch.manual_seed(1)
    x = torch.randn(4, 1024)
    wi = parts["encoder"]["layers.0.mlp.Wi.weight"]
    wo = parts["encoder"]["layers.0.mlp.Wo.weight"]
    act, gate = W.split_wi(wi, laya_config.intermediate_size)
    ref = (torch.nn.functional.gelu(x @ act.T) * (x @ gate.T)) @ wo.T
    pa, pg, po = W.pad_up_projection(act, 2816), W.pad_up_projection(gate, 2816), W.pad_down_projection(wo, 2816)
    got = (torch.nn.functional.gelu(x @ pa.T) * (x @ pg.T)) @ po.T
    assert float((ref - got).abs().max()) == 0.0


def test_source_can_be_a_full_state_dict_or_a_module(state_dict, parts, torch_encoder):
    assert set(W._as_encoder_state_dict(state_dict)) == set(parts["encoder"])
    assert set(W._as_encoder_state_dict(torch_encoder)) == set(parts["encoder"])
    assert set(W._as_encoder_state_dict(parts["encoder"])) == set(parts["encoder"])
