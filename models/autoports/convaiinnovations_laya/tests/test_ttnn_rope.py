# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.reference.modernbert import ModernBertRotaryEmbedding, apply_rotary_pos_emb
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import pcc
from models.autoports.convaiinnovations_laya.tt.model_config import ACTIVATIONS_DTYPE, FULL_ATTENTION, SLIDING_ATTENTION
from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary, rope_cos_sin

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
ROPE_PCC = 0.999


def _real_qk(torch_encoder, laya_config, seq_len, batch_size):
    b = encoder_inputs(batch_size=batch_size, seq_len=seq_len, fill=batch_size > 1)
    with torch.no_grad():
        hidden = torch_encoder.embeddings(b["input_ids"])
        attn = torch_encoder.layers[0].attn
        B, S, _ = hidden.shape
        qkv = attn.Wqkv(hidden).view(B, S, 3, attn.num_heads, attn.head_dim)
        q, k, _ = qkv.unbind(dim=-3)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()


def _torch_rope(config, layer_type, q, k, seq_len):
    hd = config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters[layer_type]["rope_theta"]
    cos, sin = ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)
    return apply_rotary_pos_emb(q, k, cos, sin)


def test_thetas_are_distinct(laya_config):
    full = laya_config.rope_parameters[FULL_ATTENTION]["rope_theta"]
    sliding = laya_config.rope_parameters[SLIDING_ATTENTION]["rope_theta"]
    assert full == 160000.0 and sliding == 10000.0


def test_host_cache_matches_reference_generator(laya_config):
    hd = 64
    for lt in (FULL_ATTENTION, SLIDING_ATTENTION):
        theta = laya_config.rope_parameters[lt]["rope_theta"]
        cos_r, sin_r = ModernBertRotaryEmbedding(hd, theta)(torch.arange(1024).unsqueeze(0), torch.float32)
        cos, sin = rope_cos_sin(hd, theta, 1024)
        assert torch.allclose(cos, cos_r[0], atol=1e-6) and torch.allclose(sin, sin_r[0], atol=1e-6)


@pytest.mark.parametrize("layer_type", [FULL_ATTENTION, SLIDING_ATTENTION])
@pytest.mark.parametrize("seq_len,batch_size", [(512, 1), (512, 8), (1024, 1), (1024, 2)])
def test_ttnn_rope_matches_reference(device, torch_encoder, laya_config, pcc_log, layer_type, seq_len, batch_size):
    q, k = _real_qk(torch_encoder, laya_config, seq_len, batch_size)
    q_ref, k_ref = _torch_rope(laya_config, layer_type, q, k, seq_len)
    rotary = TtnnModernBertRotary(laya_config, device, seq_len, batch_size=batch_size)
    tt_q = ttnn.from_torch(q, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    oq, ok = rotary(tt_q, layer_type), rotary(tt_k, layer_type)
    got_q = ttnn.to_torch(oq).float().reshape(q_ref.shape)
    got_k = ttnn.to_torch(ok).float().reshape(k_ref.shape)
    for t in (tt_q, tt_k, oq, ok):
        ttnn.deallocate(t)
    rotary.deallocate()
    pq, pk = pcc(q_ref, got_q), pcc(k_ref, got_k)
    record(pcc_log, test="rope", layer_type=layer_type, seq=seq_len, batch=batch_size, sharded=rotary.sharded, pcc_q=pq, pcc_k=pk)
    assert pq >= ROPE_PCC and pk >= ROPE_PCC, f"rope PCC q={pq:.8f} k={pk:.8f}"


def test_negative_control_wrong_theta(device, torch_encoder, laya_config, pcc_log):
    seq_len = 512
    q, k = _real_qk(torch_encoder, laya_config, seq_len, 1)
    q_ref, _ = _torch_rope(laya_config, FULL_ATTENTION, q, k, seq_len)
    rotary = TtnnModernBertRotary(laya_config, device, seq_len)
    tt_q = ttnn.from_torch(q, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    out = rotary(tt_q, SLIDING_ATTENTION)
    got = ttnn.to_torch(out).float().reshape(q_ref.shape)
    ttnn.deallocate(tt_q)
    ttnn.deallocate(out)
    rotary.deallocate()
    p = pcc(q_ref, got)
    record(pcc_log, test="NC rope wrong theta", pcc=p, threshold=ROPE_PCC)
    assert p < ROPE_PCC


def test_negative_control_no_rope(torch_encoder, laya_config, pcc_log):
    q, k = _real_qk(torch_encoder, laya_config, 512, 1)
    q_ref, _ = _torch_rope(laya_config, FULL_ATTENTION, q, k, 512)
    p = pcc(q_ref, q)
    record(pcc_log, test="NC rope not applied", pcc=p, threshold=ROPE_PCC)
    assert p < ROPE_PCC
