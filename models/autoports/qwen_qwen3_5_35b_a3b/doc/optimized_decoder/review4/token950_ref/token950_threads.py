import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
torch.manual_seed(1234)
torch.set_num_threads(8)
layer = 0
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
state = load_layer_state_dict(ck, layer)
local = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in state.items()}
with torch.device("meta"):
    ref = Qwen3_5MoeDecoderLayer(cfg, layer)
ref.load_state_dict(local, assign=True, strict=True)
ref.eval()
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
x = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer0.pt", weights_only=True)["activations"]
tt = {L: torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review4/token950/tt_{L}.pt")["prefill"][0] for L in (4096, 4097, 2600)}
routes = {}
gate = ref.mlp.gate
def hook(module, inputs, output):
    logits = output[0] if isinstance(output, tuple) else output
    routes["logits"] = logits.detach().float()
handle = gate.register_forward_hook(hook)
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
out = {}
rows = {}
for L in (4096, 4097, 2600):
    pf = prefill_reference(ref, cfg, rope, x[:, :L].contiguous(), layer)[0].float()
    rows[L] = pf[0, 950]
    logits = routes.get("logits")
    entry = {"tt_vs_ref_token950": pcc(tt[L][950], pf[0, 950])}
    if logits is not None and logits.shape[-2] >= 951:
        lg = logits.reshape(-1, logits.shape[-1])[950]
        top = torch.topk(lg, 10)
        entry["router_top10_ids"] = top.indices.tolist()
        entry["router_top10_logits"] = [round(v, 5) for v in top.values.tolist()]
    out[L] = entry
    print(L, entry, flush=True)
print("ref token950 cross-length", {L: pcc(rows[4096], rows[L]) for L in rows}, flush=True)
json.dump(out, open("token950_threads8.json", "w"), indent=1)
