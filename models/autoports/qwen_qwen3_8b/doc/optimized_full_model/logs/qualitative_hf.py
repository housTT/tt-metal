"""HF control for the shared qualitative prompt suite, rendered with the HF chat template (greedy, 128 tokens)."""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
PROMPTS = Path("/home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/vllm_prompts.txt")
OUT = Path("models/autoports/qwen_qwen3_8b/doc/optimized_full_model/qualitative")
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 128

tok = AutoTokenizer.from_pretrained(MODEL)
prompts = [p.strip() for p in PROMPTS.read_text().split("\n\n") if p.strip()]
rendered = []
for i, text in enumerate(prompts):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(chat, add_special_tokens=False)
    rendered.append({"id": i, "prompt": text, "rendered": chat, "token_ids": ids, "num_tokens": len(ids)})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "qualitative_prompt_format.json").write_text(
    json.dumps(
        {
            "hf_model": MODEL,
            "checkpoint_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_class": type(tok).__name__,
            "chat_template_present": bool(tok.chat_template),
            "prompt_mode": "chat",
            "rendering": "tokenizer.apply_chat_template([{role: user, content}], add_generation_prompt=True, tokenize=False) then encode(add_special_tokens=False)",
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW, "stop": "eos_token_id (151645, 151643)"},
            "prompt_source": str(PROMPTS),
            "note": "Qwen3 chat template emits a <think> block by default; both HF and TT outputs keep it.",
        },
        indent=2,
    )
    + "\n"
)
(OUT / "rendered_prompts.json").write_text(json.dumps(rendered, indent=2) + "\n")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
results = []
for entry in rendered:
    ids = torch.tensor([entry["token_ids"]])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
    gen = out[0, ids.shape[1] :].tolist()
    results.append(
        {
            "id": entry["id"],
            "prompt": entry["prompt"],
            "token_ids": gen,
            "num_tokens": len(gen),
            "text": tok.decode(gen, skip_special_tokens=False),
            "seconds": time.time() - t0,
        }
    )
    print(f"prompt {entry['id']} done {len(gen)} tokens in {time.time() - t0:.0f}s", flush=True)
    (OUT / "hf_outputs.json").write_text(json.dumps(results, indent=2) + "\n")
print("HF_QUALITATIVE_DONE", flush=True)
