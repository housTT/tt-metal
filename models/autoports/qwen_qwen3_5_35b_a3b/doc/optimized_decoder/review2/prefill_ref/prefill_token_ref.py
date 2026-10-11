import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
torch.set_num_threads(16)
layer, length, tokens = int(sys.argv[1]), int(sys.argv[2]), [int(t) for t in sys.argv[3].split(",")]
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
sd = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in load_layer_state_dict(ck, layer).items()}
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
x = torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer{layer}.pt", weights_only=True)["activations"][:, :length].contiguous()
res = {}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    with torch.device("meta"):
        m = Qwen3_5MoeDecoderLayer(cfg, layer)
    m.load_state_dict({k: v.to(dt) for k, v in sd.items()}, assign=True, strict=True)
    m.eval()
    res[name] = prefill_reference(m, cfg, rope, x.to(dt), layer)[0].float()
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
by_tok = [pcc(res["bf16"][0, t], res["fp32"][0, t]) for t in range(length)]
out = dict(layer=layer, length=length, selected={t: by_tok[t] for t in tokens},
           below_0995=[(t, round(p, 5)) for t, p in enumerate(by_tok) if p < 0.995], min=min(by_tok), argmin=by_tok.index(min(by_tok)))
print(json.dumps(out), flush=True)
json.dump(dict(out, by_token=by_tok), open(f"prefill_token_ref_l{layer}_{length}.json", "w"), indent=1)
