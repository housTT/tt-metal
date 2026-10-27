# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Regenerate layer_inputs/layer17_tale_1024.pt (gitignored, 8.2 MB): the real hidden state entering
layer 17 for the first 1024 tale-of-two-cities tokens, from HF layers 0..16 in float32 (about 20 s on CPU).
Run from the checkout root with the stage environment. Also refreshes the layer_inputs statistics in
real_weight_stats.json."""
import bz2
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_8b.tests.reference import HFLayerReference, load_config
from models.autoports.qwen_qwen3_8b.tests.weights import MODEL_ID, STATS_PATH, load_embedding, load_layer_state

cfg = load_config()
tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt").read()
ids = tok(text[:20000], return_tensors="pt").input_ids[:, :1024]
print("tokens", ids.shape)
emb = load_embedding()
hidden = emb[ids[0]].unsqueeze(0).to(torch.float32)
out_dir = Path("models/autoports/qwen_qwen3_8b/doc/functional_decoder/layer_inputs")
out_dir.mkdir(parents=True, exist_ok=True)
stats = json.loads(STATS_PATH.read_text())
stats.setdefault("layer_inputs", {})


def rec(layer_idx, h, store=True):
    v = h[0]
    entry = {
        "source": "models/tt_transformers/tests/tale-of-two-cities.txt.bz2 first 20000 characters, first 1024 tokens, HF layers 0..layer_idx-1 in float32",
        "tokens": int(v.shape[0]),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "rms": float(v.pow(2).mean().sqrt()),
        "max_abs": float(v.abs().max()),
    }
    if store:
        entry["artifact"] = f"layer_inputs/layer{layer_idx}_tale_1024.pt"
    else:
        entry[
            "note"
        ] = "layer-0 test inputs are real token ids (layer_inputs/tale_token_ids.json) through the real embedding; the hidden-state capture is not stored"
    stats["layer_inputs"][str(layer_idx)] = entry
    if store:
        torch.save(
            {"token_ids": ids[0].clone(), "hidden": h[0].to(torch.bfloat16).clone()},
            out_dir / f"layer{layer_idx}_tale_1024.pt",
        )


rec(0, hidden, store=False)
t = time.time()
for layer_idx in range(17):
    ref = HFLayerReference(cfg, layer_idx, load_layer_state(layer_idx))
    hidden = ref.forward(hidden, 0)
    del ref
    print(
        "layer",
        layer_idx,
        "done",
        round(time.time() - t, 1),
        "s rms",
        float(hidden[0].pow(2).mean().sqrt()),
        flush=True,
    )
rec(17, hidden)
STATS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
print("done", stats["layer_inputs"])
