import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
layer, length, steps, tokens = 3, 1000, 8, [684, 592]
torch.manual_seed(1234)
torch.set_num_threads(8)
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
state = load_layer_state_dict(ck, layer)
local = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in state.items()}
x = torch.randn(1, length + steps, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
logits = {}
def build(dt, name):
    with torch.device("meta"):
        m = Qwen3_5MoeDecoderLayer(cfg, layer)
    m.load_state_dict({k: v.to(dt) for k, v in local.items()}, assign=True, strict=True)
    m.eval()
    m.mlp.gate.register_forward_hook(lambda mod, i, o: logits.__setitem__(name, (o[0] if isinstance(o, tuple) else o).detach().float()))
    return m
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
out = {}
res = {}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    res[name] = prefill_reference(build(dt, name), cfg, rope, x[:, :length].to(dt), layer)[0].float()
for t in tokens:
    entry = {"bf16_vs_fp32": pcc(res["bf16"][0, t], res["fp32"][0, t])}
    for name in ("bf16", "fp32"):
        lg = logits[name].reshape(-1, logits[name].shape[-1])[t]
        top = torch.topk(lg, 10)
        entry[f"{name}_top10_ids"] = top.indices.tolist()
        entry[f"{name}_top10_logits"] = [round(v, 5) for v in top.values.tolist()]
    out[t] = entry
    print(t, entry, flush=True)
by_tok = [pcc(res["bf16"][0, t], res["fp32"][0, t]) for t in range(length)]
print("bf16 vs fp32 min", min(by_tok), "argmin", by_tok.index(min(by_tok)), "below 0.995", [(t, round(p, 5)) for t, p in enumerate(by_tok) if p < 0.995], flush=True)
json.dump(dict(out, below_0995=[(t, p) for t, p in enumerate(by_tok) if p < 0.995]), open("token684.json", "w"), indent=1)
