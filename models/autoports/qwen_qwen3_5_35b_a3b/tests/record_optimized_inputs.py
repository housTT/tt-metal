# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record HF embedding/layer activations for layer-only precision experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

from .hf_reference import prefill_reference
from .layer_weights import load_layer_state_dict


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("../weights/Qwen3.5-35B-A3B"))
    p.add_argument("--source", type=Path, default=Path("tech_reports/LLMs/llms.md"))
    p.add_argument("--length", type=int, default=4352)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    text = a.source.read_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < a.length:
        raise ValueError("Source text is too short")
    ids = torch.tensor(tokens[:a.length], dtype=torch.long).unsqueeze(0)
    index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(str(a.checkpoint / index[name]), framework="pt", device="cpu") as f:
        x = torch.nn.functional.embedding(ids, f.get_tensor(name))
    cfg = AutoConfig.from_pretrained(a.checkpoint, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
    metadata = dict(checkpoint_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
                    source=str(a.source), source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                    tokens=a.length, purpose="Layer-only real activation fixture; no full-model implementation or generation")
    for i in range(4):
        if i in (0, 3):
            torch.save({"activations":x.contiguous(), "token_ids":ids, "metadata":dict(metadata,layer=i)},
                       a.directory / f"layer{i}.pt")
            print("RECORDED", i, x.shape, flush=True)
        if i == 3:
            break
        state = load_layer_state_dict(a.checkpoint, i)
        prefix = f"model.language_model.layers.{i}."
        with torch.device("meta"):
            layer = Qwen3_5MoeDecoderLayer(cfg, i)
        layer.load_state_dict({k.removeprefix(prefix):v for k,v in state.items()}, assign=True, strict=True)
        layer.eval()
        x, _, _ = prefill_reference(layer, cfg, rope, x, i)
        del layer, state
    (a.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
