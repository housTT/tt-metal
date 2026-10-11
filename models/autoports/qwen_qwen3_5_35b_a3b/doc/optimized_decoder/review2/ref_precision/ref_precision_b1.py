import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference, decode_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
torch.set_num_threads(12)
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
layer = 0
sd = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in load_layer_state_dict(ck, layer).items()}
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
x = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer0.pt", weights_only=True)["activations"][:, :4096 + 128].contiguous()
res = {}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    with torch.device("meta"):
        m = Qwen3_5MoeDecoderLayer(cfg, layer)
    m.load_state_dict({k: v.to(dt) for k, v in sd.items()}, assign=True, strict=True)
    m.eval()
    xx = x.to(dt)
    pf, cache, _ = prefill_reference(m, cfg, rope, xx[:, :4096], layer)
    res[name] = (pf.float(), decode_reference(m, rope, cache, xx[:, 4096:], 4096).float())
    print("done", name, flush=True)
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
by_tok = [pcc(res["bf16"][1][:, i], res["fp32"][1][:, i]) for i in range(128)]
out = dict(prefill=pcc(res["bf16"][0], res["fp32"][0]), decode_min=min(by_tok), argmin=by_tok.index(min(by_tok)),
           token32=by_tok[32], token21=by_tok[21], token102=by_tok[102], below_0999=[(i, round(p, 5)) for i, p in enumerate(by_tok) if p < 0.999])
print(json.dumps(out), flush=True)
json.dump(dict(out, by_token=by_tok), open("ref_precision_b1.json", "w"), indent=1)
