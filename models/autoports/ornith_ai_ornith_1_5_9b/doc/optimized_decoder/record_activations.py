# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Record actual checkpoint activations for decoder precision experiments (CPU only)."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from models.autoports.ornith_ai_ornith_1_5_9b.reference import hf_reference as R

root = Path(__file__).resolve().parents[2]
output = root / "doc/optimized_decoder/activations"
output.mkdir(exist_ok=True)
model = R.resolve_model_path()
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
text = """Explain why the sky appears blue and how rainbows form. Write a Python function to merge two sorted lists. The function should preserve duplicate values and handle empty inputs. Calculate the sum of the first twenty positive integers and explain the formula. A traveler visits a library in a quiet town and discovers an old map. Describe what happens next. Compare the roles of memory and attention in learning a new language. Translate this sentence into French: the garden is beautiful in spring. In a database transaction, atomicity ensures that all changes succeed together. Discuss a practical example and provide SQL statements.
"""
ids = tokenizer.encode(text * 40, add_special_tokens=False)[:2112]
key = R.CHECKPOINT_TEXT_PREFIX + "embed_tokens.weight"
weight_map = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(model / weight_map[key]), framework="pt") as f:
    weight = f.get_slice(key)
    rows = {i: weight[i : i + 1] for i in set(ids)}
x = torch.cat([rows[i] for i in ids]).unsqueeze(0).float()
metadata = {
    "model": R.HF_MODEL_ID,
    "revision": R.HF_REVISION,
    "token_ids": ids,
    "text": text,
    "method": "real tokenizer -> real embeddings -> pinned HF layers 0..2 in FP32",
    "files": {},
}
cfg = R.load_text_config()
for layer_idx in range(4):
    if layer_idx in (0, 3):
        path = output / f"layer{layer_idx}.pt"
        torch.save(x.to(torch.bfloat16), path)
        metadata["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "shape": list(x.shape)}
    if layer_idx < 3:
        layer = R.build_reference_layer(cfg, layer_idx, R.load_layer_state_dict(layer_idx))
        with torch.no_grad():
            x, _ = R.reference_prefill(layer, cfg, x)
        del layer
        print("RECORDED_PREFIX_LAYER", layer_idx, flush=True)
(output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("ACTIVATIONS_RECORDED", flush=True)
