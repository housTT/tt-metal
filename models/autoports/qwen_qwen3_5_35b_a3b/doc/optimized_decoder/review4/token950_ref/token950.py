import json, sys, torch
from pathlib import Path
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding
sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
torch.set_num_threads(16)
layer, tokens = 0, [950, 1720]
lengths = [int(v) for v in sys.argv[1].split(",")]
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
sd = {k.removeprefix(f"model.language_model.layers.{layer}."): v for k, v in load_layer_state_dict(ck, layer).items()}
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
x = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer0.pt", weights_only=True)["activations"]
def build(dt):
    with torch.device("meta"):
        m = Qwen3_5MoeDecoderLayer(cfg, layer)
    m.load_state_dict({k: v.to(dt) for k, v in sd.items()}, assign=True, strict=True)
    return m.eval()
models = {"bf16": build(torch.bfloat16), "fp32": build(torch.float32)}
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
out = {}
for L in lengths:
    res = {n: prefill_reference(m, cfg, rope, x[:, :L].to(dt), layer)[0].float()
           for (n, m), dt in zip(models.items(), (torch.bfloat16, torch.float32))}
    out[L] = {"bf16_vs_fp32": {t: pcc(res["bf16"][0, t], res["fp32"][0, t]) for t in tokens},
              "bf16_row": {t: res["bf16"][0, t].tolist() for t in tokens}}
    print(L, out[L]["bf16_vs_fp32"], flush=True)
rows = {L: torch.tensor(out[L]["bf16_row"][950]) for L in lengths}
ref = rows[lengths[0]]
cross = {L: pcc(ref, rows[L]) for L in lengths}
print("bf16 token950 row PCC vs length", lengths[0], cross, flush=True)
json.dump({"bf16_vs_fp32": {L: out[L]["bf16_vs_fp32"] for L in lengths}, "bf16_token950_cross_length_pcc": cross},
          open(f"token950_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
