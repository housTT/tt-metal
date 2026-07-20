# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy perf harness for the Kokoro-82M OPTIMIZED MULTICHIP decoder (TP=4 on a (1,4) mesh).

One measured window per invocation:

  python -m tracy -r -p -v -m pytest \
    models/demos/audio/kokoro/tests/test_perf_multichip.py -k decode
  python -m tracy -r -p -v -m pytest \
    models/demos/audio/kokoro/tests/test_perf_multichip.py -k prefill

Then feed the newest ops_perf_results CSV to tt-perf-report with the matching
signposts (PERF_PREFILL / PERF_DECODE). Ring fabric (physical 4-ring).
"""
import json
import os
import random

import pytest
import torch
import tracy

import ttnn
from models.demos.audio.kokoro.tt.optimized_multichip_decoder import OptimizedMultichipDecoder as MultichipDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
PERF_SEQ_LEN = int(os.environ.get("KOKORO_PERF_SEQ_LEN", "512"))
PERF_ITERS = int(os.environ.get("KOKORO_PERF_ITERS", "20"))
MESH_SHAPE = (1, 4)
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "


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
def mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    yield dev
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_perf_prefill(mesh):
    ac, sd, ids = _build()
    dec = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
    prep = dec.prepare_inputs(ids)

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
    ttnn.synchronize_device(mesh)
    tracy.signpost("PERF_PREFILL")
    for _ in range(PERF_ITERS):
        out = run()
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    tracy.signpost("PERF_PREFILL_END")


def test_perf_decode(mesh):
    ac, sd, ids = _build()
    dec = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
    prep = dec.prepare_inputs(ids)
    dec.decode_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    ttnn.synchronize_device(mesh)
    record = dec._traces[(prep["batch"], prep["padded_seq_len"], prep["attention_mask"] is not None)]

    tracy.signpost("PERF_DECODE")
    for _ in range(PERF_ITERS):
        ttnn.execute_trace(mesh, record["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    tracy.signpost("PERF_DECODE_END")
    dec.release_traces()
