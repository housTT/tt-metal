# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 norm and unquantized LM-head oracle on the exact saved TT hidden."""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

doc = Path(__file__).resolve().parent
root = Path("/home/hous/dev/ornith-1.5-9b/upstream")
weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def read(key):
    with safe_open(str(root / weight_map[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


hidden = torch.load(doc / "french_head_v1/hidden.pt", weights_only=True)["hidden"][0].float()
norm_key = next(k for k in weight_map if k.endswith("model.norm.weight") or k == "model.language_model.norm.weight")
norm = read(norm_key).float() + 1
config = json.loads((root / "config.json").read_text())
config = config.get("text_config", config)
normal = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + config["rms_norm_eps"]) * norm
scores = (normal @ read("lm_head.weight").float().T).flatten()
values, indices = scores.topk(10)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
report = dict(
    norm_key=norm_key,
    hidden_shape=list(hidden.shape),
    top10_ids=indices.tolist(),
    top10_logits=values.tolist(),
    top10_text=[tokenizer.decode([i]) for i in indices.tolist()],
    inform_rank=int((scores > scores[39102]).sum()) + 1,
    ttnn_imported="ttnn" in sys.modules,
)
assert not report["ttnn_imported"]
(doc / "french_head_v1/cpu_fp32_oracle.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
