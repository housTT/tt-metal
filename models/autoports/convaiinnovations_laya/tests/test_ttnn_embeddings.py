# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import outlier_report, pcc
from models.autoports.convaiinnovations_laya.tt.modernbert_embeddings import TtnnModernBertEmbeddings
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
EMBEDDINGS_PCC = 0.999


@pytest.fixture(scope="module")
def tt_params(module_device, parts, laya_config):
    device = module_device
    params = prepare_weights(parts["encoder"], laya_config, device, layers=[])
    yield params
    deallocate_weights(params)


def _ids(ids, device):
    return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_ttnn_embeddings_match_reference(device, tt_params, torch_encoder, laya_config, pcc_log, batch_size):
    b = encoder_inputs(batch_size=batch_size, seq_len=512)
    with torch.no_grad():
        expected = torch_encoder.embeddings(b["input_ids"])
    module = TtnnModernBertEmbeddings(tt_params["embeddings"], laya_config, device)
    out = module(_ids(b["input_ids"], device))
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    ttnn.deallocate(out)
    p = pcc(expected, got)
    record(pcc_log, test="embeddings", batch=batch_size, seq=512, pcc=p, outliers=outlier_report(expected))
    assert p >= EMBEDDINGS_PCC, f"embeddings PCC {p:.8f} < {EMBEDDINGS_PCC}"


def test_negative_control_missing_layernorm(device, tt_params, torch_encoder, pcc_log):
    b = encoder_inputs(batch_size=1, seq_len=512)
    with torch.no_grad():
        expected = torch_encoder.embeddings(b["input_ids"])
    raw = ttnn.embedding(_ids(b["input_ids"], device), tt_params["embeddings"]["tok_embeddings"], layout=ttnn.TILE_LAYOUT)
    got = ttnn.to_torch(raw).float().reshape(expected.shape)
    ttnn.deallocate(raw)
    p = pcc(expected, got)
    record(pcc_log, test="NC embeddings without layernorm", pcc=p, threshold=EMBEDDINGS_PCC)
    assert p < EMBEDDINGS_PCC, "dropping the LayerNorm did not change the output"


def test_weight_prep_rejects_wrong_tensor_count(parts, laya_config):
    enc = dict(parts["encoder"])
    enc["extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="extra"):
        prepare_weights(enc, laya_config, None, layers=[])
