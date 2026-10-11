import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
layer, length, steps = 3, 1000, 8
torch.manual_seed(1234)
torch.set_num_threads(8)
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
local = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in load_layer_state_dict(ck, layer).items()}
x = torch.randn(1, length + steps, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
with torch.device("meta"):
    ref = Qwen3_5MoeDecoderLayer(cfg, layer)
ref.load_state_dict(local, assign=True, strict=True)
ref.eval()
tt = torch.load("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/optimized_decoder/review5/token684_tt/tt_1000.pt")["prefill"][0]
force = {}
gate = ref.mlp.gate
import torch.nn.functional as F
def forced_forward(hidden_states):
    hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
    router_logits = F.linear(hidden_states, gate.weight)
    for token, (expert, delta) in force.items():
        router_logits[token, expert] = router_logits[token, expert] + delta
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, gate.top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    if force:
        print("token684 experts", sorted(router_indices[684].tolist()), flush=True)
    return router_logits, router_top_value, router_indices
gate.forward = forced_forward
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
out = {}
for name, f in (("as_is", {}), ("force_70", {684: (70, 0.0078125)}), ("force_10", {684: (10, 0.0078125)}), ("force_92", {684: (92, 0.0078125)})):
    force.clear(); force.update(f)
    pf = prefill_reference(ref, cfg, rope, x[:, :length], layer)[0].float()
    out[name] = {"tt_vs_ref_token684": pcc(tt[684], pf[0, 684]), "tt_vs_ref_token592": pcc(tt[592], pf[0, 592]),
                 "tt_vs_ref_all": pcc(tt[:length], pf[0])}
    print(name, out[name], flush=True)
json.dump(out, open("token684_variants.json", "w"), indent=1)
