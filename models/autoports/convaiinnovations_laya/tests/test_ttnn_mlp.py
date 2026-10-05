# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import max_abs_err, outlier_report, pcc
from models.autoports.convaiinnovations_laya.tt.model_config import ACTIVATIONS_DTYPE, DEFAULT_PORT, bucket_plan
from models.autoports.convaiinnovations_laya.tt.modernbert_mlp import TtnnModernBertMLP
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
MLP_PCC = 0.999
LAYERS = [0, 16]


@pytest.fixture(scope="module")
def tt_params(module_device, parts, laya_config):
    device = module_device
    params = prepare_weights(parts["encoder"], laya_config, device, layers=LAYERS, intermediate_pads=(None, 2816, 3072))
    yield params
    deallocate_weights(params)


@pytest.fixture(scope="module")
def mlp_inputs(torch_encoder):
    out = {}
    for batch in (1, 2):
        b = encoder_inputs(batch_size=batch, seq_len=512, fill=batch > 1)
        with torch.no_grad():
            _, hidden = torch_encoder(b["input_ids"], b["attention_mask"], output_hidden_states=True)
            out[batch] = {i: torch_encoder.layers[i].mlp_norm(hidden[i]) for i in LAYERS}
    return out


def _run(device, laya_config, tt_params, layer_idx, x, port):
    batch, seq_len, _ = x.shape
    plan = bucket_plan(device, laya_config, batch, seq_len, port=port)
    module = TtnnModernBertMLP(tt_params["layers"][layer_idx]["mlp"], laya_config, plan, device, port=port)
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    if plan.mlp_shard is not None:
        sh = ttnn.to_memory_config(tt_x, plan.mlp_shard.hidden_memory)
        ttnn.deallocate(tt_x)
        tt_x = sh
    out = module(tt_x)
    got = ttnn.to_torch(out).float().reshape(x.shape)
    ttnn.deallocate(out)
    ttnn.deallocate(tt_x)
    return got, plan


@pytest.mark.parametrize("layer_idx", LAYERS)
@pytest.mark.parametrize("batch", [1, 2])
def test_ttnn_mlp_matches_reference(device, tt_params, torch_encoder, laya_config, mlp_inputs, pcc_log, layer_idx, batch):
    x = mlp_inputs[batch][layer_idx]
    with torch.no_grad():
        expected = torch_encoder.layers[layer_idx].mlp(x)
    got, plan = _run(device, laya_config, tt_params, layer_idx, x, DEFAULT_PORT)
    p = pcc(expected, got)
    record(
        pcc_log,
        test="mlp",
        layer=layer_idx,
        batch=batch,
        seq=512,
        sharded=plan.mlp_shard is not None,
        width=plan.mlp_width,
        pcc=p,
        max_abs_err=max_abs_err(expected, got),
        input_outliers=outlier_report(x),
    )
    assert p >= MLP_PCC, f"MLP layer {layer_idx} batch {batch} PCC {p:.8f} < {MLP_PCC}"


def test_padded_widths_agree(device, tt_params, torch_encoder, laya_config, mlp_inputs, pcc_log):
    x = mlp_inputs[2][16]
    with torch.no_grad():
        expected = torch_encoder.layers[16].mlp(x)
    results = {}
    for name, port in (
        ("interleaved_2624", DEFAULT_PORT.with_(geglu_plan="interleaved")),
        ("sharded_2816", DEFAULT_PORT),
        ("sharded_3072", DEFAULT_PORT.with_(intermediate_pad=3072)),
    ):
        got, plan = _run(device, laya_config, tt_params, 16, x, port)
        results[name] = got
        record(pcc_log, test="mlp width variant", variant=name, width=plan.mlp_width, pcc=pcc(expected, got), max_abs_err=max_abs_err(expected, got))
    d1 = max_abs_err(results["sharded_2816"], results["sharded_3072"])
    d2 = max_abs_err(results["interleaved_2624"], results["sharded_2816"])
    scale = float(expected.abs().max())
    record(pcc_log, test="mlp padding exactness", max_abs_2816_vs_3072=d1, max_abs_interleaved_vs_2816=d2, output_scale=scale)
    assert d1 <= 1e-3 * scale, "the two padded widths must agree (same kernel, zero blocks)"
    assert pcc(results["interleaved_2624"], results["sharded_2816"]) > 0.9999


def test_negative_control_swapped_gate(device, tt_params, torch_encoder, mlp_inputs, pcc_log):
    x = mlp_inputs[1][0]
    with torch.no_grad():
        expected = torch_encoder.layers[0].mlp(x)
    w = tt_params["layers"][0]["mlp"][2624]
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    act_half = ttnn.linear(tt_x, w["Wi_act"])
    gate_half = ttnn.linear(tt_x, w["Wi_gate"])
    swapped = ttnn.mul(ttnn.gelu(gate_half, fast_and_approximate_mode=False), act_half)
    out = ttnn.linear(swapped, w["Wo"])
    got = ttnn.to_torch(out).float().reshape(expected.shape)
    for t in (tt_x, act_half, gate_half, swapped, out):
        ttnn.deallocate(t)
    p = pcc(expected, got)
    record(pcc_log, test="NC geglu gate swapped", pcc=p, threshold=MLP_PCC)
    assert p < MLP_PCC, "swapping the GeGLU gate did not change the output"
