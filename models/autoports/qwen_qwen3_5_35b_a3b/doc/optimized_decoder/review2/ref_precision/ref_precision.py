import json, sys, copy
import torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference, decode_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict

torch.set_num_threads(8)
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
E = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
layer = 0
state = load_layer_state_dict(ck, layer)
prefix = f"model.language_model.layers.{layer}."
sd = {k.removeprefix(prefix): v for k, v in state.items()}
def build(dtype):
    with torch.device("meta"):
        m = Qwen3_5MoeDecoderLayer(cfg, layer)
    m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, assign=True, strict=True)
    return m.eval()
ref16, ref32 = build(torch.bfloat16), build(torch.float32)
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
rec = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer0.pt", weights_only=True)["activations"][0]

def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

cases = [("final_evidence6/real_inputs/positions_0.json", 0, 17),
         ("review1/p1/default_realpos_517.json", 517, 7),
         ("review1/p1/default_realpos_1531.json", 1531, 1)]
out = []
for path, offset, row in cases:
    d = json.load(open(E / path))
    steps = d["decode_steps"]
    length = d["prompt_lengths"][row]
    width = length + steps
    start = (row * 131 + offset) % (rec.shape[0] - width + 1)
    x = rec[None, start:start + width].contiguous()
    res = {}
    for name, m, dt in (("bf16", ref16, torch.bfloat16), ("fp32", ref32, torch.float32)):
        xx = x.to(dt)
        _, cache, _ = prefill_reference(m, cfg, rope, xx[:, :length], layer)
        res[name] = decode_reference(m, rope, cache, xx[:, length:], length).float()
    per_tok_ref = [pcc(res["bf16"][:, i], res["fp32"][:, i]) for i in range(steps)]
    tt_tok = d["decode_rows"][row]["pcc_by_token"]
    rec_row = dict(path=path, row=row, length=length, start=start,
                   hf_bf16_vs_fp32_by_token=per_tok_ref, tt_vs_hf_bf16_by_token=tt_tok)
    print(json.dumps(rec_row), flush=True)
    out.append(rec_row)
json.dump(out, open(sys.argv[1], "w"), indent=1)
