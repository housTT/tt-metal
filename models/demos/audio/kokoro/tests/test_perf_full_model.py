# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tracy perf harness for the Kokoro-82M FULL-MODEL token-out traced decode.

Profiles the token-out path (encode -> reconstruction readout -> on-device
argmax) so the perf report shows whether sampler work (ArgMax) dominates. Uses a
REDUCED 2-layer model (Kokoro's only layer kind is the weight-tied AlbertLayer)
with real shapes/dtypes/sharding/trace, per $full-model (do not Tracy the full
all-layer stack).

  python -m tracy -r -p -v -m pytest \
    models/demos/audio/kokoro/tests/test_perf_full_model.py -k tokenout

Signposts PERF_TOKENOUT / PERF_TOKENOUT_END.
"""
import json
import os
import random

import pytest
import torch
import tracy

import ttnn
from models.demos.audio.kokoro.tt.model import KokoroModel

MODEL_ID = "hexgrad/Kokoro-82M"
PERF_SEQ_LEN = int(os.environ.get("KOKORO_PERF_SEQ_LEN", "128"))
PERF_ITERS = int(os.environ.get("KOKORO_PERF_ITERS", "20"))
PERF_LAYERS = int(os.environ.get("KOKORO_PERF_LAYERS", "2"))
MESH_SHAPE = (1, 4)
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "


def _build(mesh):
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    plbert = dict(cfg["plbert"])
    plbert["num_hidden_layers"] = PERF_LAYERS  # reduced profiling variant (tied layer)
    ac = AlbertConfig(vocab_size=cfg["n_token"], **plbert)
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    pool = [cfg["vocab"][c] for c in _REP_SYMBOLS if c in cfg["vocab"]]
    rng = random.Random(0)
    body = [rng.choice(pool) for _ in range(PERF_SEQ_LEN - 2)]
    ids = torch.tensor([[0, *body, 0]], dtype=torch.long)
    model = KokoroModel.from_state_dict(sd, hf_config=ac, vocab=cfg["vocab"], mesh_device=mesh)
    return model, ids


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    yield dev
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_perf_tokenout(mesh):
    model, ids = _build(mesh)
    prep = model.prepare_inputs(ids)
    rec = model.capture_out_trace(prep, want_logits=False)  # warm + capture
    ttnn.synchronize_device(mesh)
    tracy.signpost("PERF_TOKENOUT")
    for _ in range(PERF_ITERS):
        ttnn.execute_trace(mesh, rec["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    tracy.signpost("PERF_TOKENOUT_END")
    model.release()
