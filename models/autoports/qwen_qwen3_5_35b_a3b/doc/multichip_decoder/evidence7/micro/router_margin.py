import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict

torch.set_num_threads(16)
layer, length = int(sys.argv[1]), 4096
tokens = [int(t) for t in sys.argv[2].split(",")]
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
state = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in load_layer_state_dict(ck, layer).items()}
with torch.device("meta"):
    ref = Qwen3_5MoeDecoderLayer(cfg, layer)
ref.load_state_dict(state, assign=True, strict=True)
ref.eval()
captured = {}
ref.mlp.gate.register_forward_hook(lambda m, i, o: captured.__setitem__("logits", (o[0] if isinstance(o, tuple) else o).detach().float()))
x = torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer{layer}.pt", weights_only=True)["activations"]
prefill_reference(ref, cfg, rope := Qwen3_5MoeTextRotaryEmbedding(cfg), x[:, :length].contiguous(), layer)
logits = captured["logits"].reshape(-1, captured["logits"].shape[-1])
probs = torch.softmax(logits, -1)
gaps = (torch.topk(logits, 9).values[:, 7] - torch.topk(logits, 9).values[:, 8])
result_extra = {"exact_tie_tokens": int((gaps == 0).sum()), "tokens_total": int(gaps.numel())}
print("EXTRA", result_extra)
result = {"layer": layer, "length": length, "gap_definition": "8th minus 9th router logit (BF16 HF reference)",
          "median_gap_all_tokens": float(gaps.median()), "fraction_tokens_gap_below_1e-2": float((gaps < 1e-2).float().mean()),
          "tokens": {}}
for t in tokens:
    top = torch.topk(logits[t], 10)
    result["tokens"][t] = {"gap_8_9": float(gaps[t]), "gap_rank_percentile": float((gaps < gaps[t]).float().mean()),
                           "top10_ids": top.indices.tolist(), "top10_logits": [round(v, 5) for v in top.values.tolist()],
                           "prob_8th": float(probs[t, top.indices[7]]), "prob_9th": float(probs[t, top.indices[8]])}
print(json.dumps(result, indent=1))
json.dump(result, open(f"router_margin_layer{layer}.json", "w"), indent=1)
