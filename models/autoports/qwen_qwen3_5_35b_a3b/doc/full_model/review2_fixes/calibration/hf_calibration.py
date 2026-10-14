import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B"
REF = "/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/readiness_aime24_chat.refpt"
STORY = "/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/m2/autoregressive/autoregressive_meta.json"
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, local_files_only=True).eval()
entry = torch.load(REF, weights_only=False)["entries"][0]
seq = torch.cat([entry["prompt_tokens"][0], entry["generated_tokens"][0]]).long()
prompt_len = entry["prompt_tokens"].shape[1]
with torch.no_grad():
    logits = model(seq[None], use_cache=False).logits[0].float()
torch.save(dict(seq=seq, prompt_len=prompt_len, logits=logits[prompt_len - 1 : -1].clone()), f"{out}/hf_aime24_logits.pt")
meta = json.load(open(STORY))
prompt = meta["prompt_token_ids"]
tt = meta["tt"]["token_ids"]
with torch.no_grad():
    story = model(torch.tensor([prompt + tt]), use_cache=False).logits[0, len(prompt) - 1 : -1].float()
probs = story.softmax(-1)
rows = []
for i, t in enumerate(tt):
    rows.append(dict(step=i, token=t, text=tok.decode([t]), hf_rank=int((story[i] > story[i, t]).sum()),
                     hf_prob=float(probs[i, t]), hf_top=tok.decode([int(story[i].argmax())]),
                     hf_top_prob=float(probs[i].max())))
json.dump(dict(text=tok.decode(tt), steps=rows), open(f"{out}/hf_story_m2_teacher_forced.json", "w"), indent=1)
print("done", flush=True)
