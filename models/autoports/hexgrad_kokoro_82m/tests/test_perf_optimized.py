# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill + traced warmed decode performance harness for the Kokoro-82M
OPTIMIZED decoder (plbert / ALBERT encoder). Mirrors tests/test_perf.py but uses
OptimizedDecoder. Run under Tracy, one measured window per invocation:

  python -m tracy -r -p -v -m pytest \
    models/autoports/hexgrad_kokoro_82m/tests/test_perf_optimized.py -k prefill
  python -m tracy -r -p -v -m pytest \
    models/autoports/hexgrad_kokoro_82m/tests/test_perf_optimized.py -k decode

Then feed the newest ops_perf_results CSV to tt-perf-report with the matching
signposts (PERF_PREFILL / PERF_DECODE).
"""
import json
import os
import random

import pytest
import torch
import tracy

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy

MODEL_ID = "hexgrad/Kokoro-82M"
PERF_SEQ_LEN = int(os.environ.get("KOKORO_PERF_SEQ_LEN", "512"))
PERF_ITERS = int(os.environ.get("KOKORO_PERF_ITERS", "20"))
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "


def _policy():
    # Optional override for sweeps: KOKORO_POLICY='attn_weight=bf16,mlp_weight=bf16,...'
    p = PrecisionPolicy()
    ov = os.environ.get("KOKORO_POLICY", "")
    for kv in [x for x in ov.split(",") if x]:
        k, v = kv.split("=")
        setattr(p, k, v if not v.isdigit() else int(v))
    return p


def _build():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    pool = [cfg["vocab"][c] for c in _REP_SYMBOLS if c in cfg["vocab"]]
    rng = random.Random(0)
    body = [rng.choice(pool) for _ in range(PERF_SEQ_LEN - 2)]
    ids = torch.tensor([[0, *body, 0]], dtype=torch.long)
    return ac, sd, ids


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_perf_prefill(device):
    ac, sd, ids = _build()
    dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=device, policy=_policy())
    prep = OptimizedDecoder.prepare_inputs(ids, device)

    def run():
        return dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )

    out = run()
    ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tracy.signpost("PERF_PREFILL")
    for _ in range(PERF_ITERS):
        out = run()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    tracy.signpost("PERF_PREFILL_END")


def test_perf_decode(device):
    ac, sd, ids = _build()
    dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=device, policy=_policy())
    prep = OptimizedDecoder.prepare_inputs(ids, device)

    dec.decode_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    ttnn.synchronize_device(device)
    record = dec._traces[(prep["batch"], prep["padded_seq_len"], prep["attention_mask"] is not None)]

    tracy.signpost("PERF_DECODE")
    for _ in range(PERF_ITERS):
        ttnn.execute_trace(device, record["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    tracy.signpost("PERF_DECODE_END")
    dec.release_traces()
