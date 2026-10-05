# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import pcc
from models.autoports.convaiinnovations_laya.tt.model_config import (
    ACTIVATIONS_DTYPE,
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    bucket_plan,
)
from models.autoports.convaiinnovations_laya.tt.modernbert_attention import TtnnModernBertAttention
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, build_masks_host, deallocate_masks
from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})


def _random_attention(batch, seq_len, seed=3):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(40, seq_len + 1, (batch,), generator=g)
    lengths[0] = seq_len
    att = torch.zeros(batch, seq_len, dtype=torch.long)
    for i, n in enumerate(lengths.tolist()):
        att[i, :n] = 1
    return att, lengths


@pytest.mark.parametrize("batch,seq_len", [(1, 512), (4, 512), (2, 1024)])
def test_device_masks_equal_host_builder(device, laya_config, pcc_log, batch, seq_len):
    att, lengths = _random_attention(batch, seq_len)
    host = build_masks_host(laya_config, att, seq_len)
    builder = TtnnMaskBuilder(laya_config, device, seq_len, batch)
    pad_row = builder.allocate_pad_row()
    ttnn.copy_host_to_device_tensor(builder.pad_row_tensor(att), pad_row)
    masks = builder.build(pad_row)
    for lt in (SLIDING_ATTENTION, FULL_ATTENTION):
        m = masks[lt]
        assert m.memory_config().buffer_type == ttnn.BufferType.DRAM
        assert m.layout == ttnn.TILE_LAYOUT
        assert tuple(m.shape) == (batch, 1, seq_len, seq_len)
        got = ttnn.to_torch(m).float()
        want = host[lt].float()
        assert torch.isfinite(got).all()
        same_zero = torch.equal(got == 0, want == 0)
        same_neg = torch.equal(got < -1e29, want < -1e29)
        assert same_zero and same_neg, f"{lt}: device mask pattern differs from the host builder"
    record(pcc_log, test="device masks equal host", batch=batch, seq=seq_len, lengths=lengths.tolist(), ok=True)
    deallocate_masks(masks)
    ttnn.deallocate(pad_row)
    builder.deallocate()


def test_rewriting_pad_row_changes_sdpa_output(device, parts, laya_config, torch_encoder, pcc_log):
    seq_len, batch = 512, 2
    b = encoder_inputs(batch_size=batch, seq_len=seq_len, fill=True)
    params = prepare_weights(parts["encoder"], laya_config, device, layers=[0])
    plan = bucket_plan(device, laya_config, batch, seq_len)
    rotary = TtnnModernBertRotary(laya_config, device, seq_len, batch_size=batch, attention_memory=plan.attention_memory)
    attn = TtnnModernBertAttention(params["layers"][0]["attn"], laya_config, FULL_ATTENTION, plan, device)
    with torch.no_grad():
        x = torch_encoder.embeddings(b["input_ids"])
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    builder = TtnnMaskBuilder(laya_config, device, seq_len, batch)
    pad_row = builder.allocate_pad_row()
    outs = []
    for att in (b["attention_mask"], torch.ones_like(b["attention_mask"])):
        ttnn.copy_host_to_device_tensor(builder.pad_row_tensor(att), pad_row)
        masks = builder.build(pad_row)
        out = attn(tt_x, rotary, masks[FULL_ATTENTION])
        outs.append(ttnn.to_torch(out).float())
        ttnn.deallocate(out)
        deallocate_masks(masks)
    real = b["attention_mask"][0] == 1
    p = pcc(outs[0][0][real], outs[1][0][real])
    record(pcc_log, test="pad_row rewrite changes SDPA output", pcc_padded_vs_unpadded=p)
    assert p < 0.999, "rewriting pad_row had no effect on the padded row's attention"
    unpadded_row = pcc(outs[0][1], outs[1][1])
    assert unpadded_row > 0.9999, "the row without padding must be unaffected by the pad mask of the other row"
    ttnn.deallocate(tt_x)
    ttnn.deallocate(pad_row)
    builder.deallocate()
    rotary.deallocate()
    deallocate_weights(params)
