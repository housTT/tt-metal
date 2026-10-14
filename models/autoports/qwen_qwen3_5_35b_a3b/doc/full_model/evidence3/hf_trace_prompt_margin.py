import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B"
TRACE = "/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence3/m4/split_sampling_trace.json"
tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, local_files_only=True).eval()
tt = json.load(open(TRACE))["tokens_greedy"]
ids = tok(tok.apply_chat_template([{"role": "user", "content": "Name three prime numbers."}], tokenize=False,
                                  add_generation_prompt=True), add_special_tokens=False).input_ids
with torch.no_grad():
    logits = model(torch.tensor([ids + tt]), use_cache=False).logits[0, len(ids) - 1 : len(ids) + len(tt) - 1].float()
rows = []
for step, (row, t) in enumerate(zip(logits, tt)):
    top = row.topk(32)
    probs = (top.values / 0.8).softmax(-1)
    keep = torch.cumsum(probs, -1) - probs < 0.95
    keep[0] = True
    kept = probs[keep] / probs[keep].sum()
    hit = (top.indices[keep] == t).nonzero()
    rows.append(dict(step=step, token=t, text=tok.decode([t]), hf_argmax=int(row.argmax()),
                     p_sampled_equals_tt=float(kept[hit[0, 0]]) if len(hit) else 0.0))
result = dict(tokens=tt, steps=rows, p_all_equal=float(torch.tensor([r["p_sampled_equals_tt"] for r in rows]).prod()))
json.dump(result, open(sys.argv[1], "w"), indent=1)
print(json.dumps(result))
