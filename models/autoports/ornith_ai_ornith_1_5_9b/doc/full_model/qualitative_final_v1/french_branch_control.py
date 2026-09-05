"""Pinned CPU HF control before the TT French register-label anomaly."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path("/home/hous/dev/ornith-1.5-9b/upstream")
META = ROOT / "prompt_4/autoregressive_meta.json"
assert os.environ["HF_HUB_OFFLINE"] == "1"
torch.set_num_threads(8)
started = time.monotonic()
meta = json.loads(META.read_text())
tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": meta["prompt_text"]}], tokenize=False, add_generation_prompt=True
)
assert rendered == meta["rendered_prompt"]
prompt = tokenizer.encode(rendered, add_special_tokens=False)
assert prompt == meta["prompt_token_ids"]
tt = meta["tt"]["token_ids"]
hf = meta["hf"]["token_ids"]
branch = 34
assert tokenizer.decode(tt[:branch]).endswith('"Bonjour" (')
assert tokenizer.decode(tt[branch : branch + 2]) == "informal"
model, loading = AutoModelForCausalLM.from_pretrained(SNAPSHOT, local_files_only=True, output_loading_info=True)
model = model.eval().to(torch.device("cpu"))
assert not any(loading.values()), loading
inputs = torch.tensor([prompt + tt], dtype=torch.long)
with torch.inference_mode():
    scores = model(inputs).logits[0, len(prompt) - 1 : len(prompt) + len(tt) - 1].float()
    topv, topi = torch.topk(scores, 100, dim=-1)
    ranks = (scores > scores.gather(1, torch.tensor(tt)[:, None])).sum(dim=-1) + 1
    prefix = torch.tensor([prompt + tt[:branch]], dtype=torch.long)
    generated = model.generate(
        prefix,
        attention_mask=torch.ones_like(prefix),
        do_sample=False,
        max_new_tokens=96,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
continuation = generated[0, prefix.shape[1] :].tolist()
first_diff = next((i for i, (a, b) in enumerate(zip(tt, hf)) if a != b), None)
rows = []
for index in sorted(set([first_diff, 31, 32, 33, 34, 35, 36, 43, 44])):
    if index is None:
        continue
    rows.append(
        {
            "generated_index": index,
            "absolute_logit_position": len(prompt) - 1 + index,
            "tt_token_id": tt[index],
            "tt_token_text": tokenizer.decode([tt[index]]),
            "hf_conditional_rank": int(ranks[index]),
            "hf_top10_token_ids": topi[index, :10].tolist(),
            "hf_top10_text": [tokenizer.decode([i]) for i in topi[index, :10].tolist()],
            "hf_top10_logits": topv[index, :10].tolist(),
        }
    )
report = {
    "canonical_hf_model_id": "ornith-ai/Ornith-1.5-9B",
    "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
    "snapshot": str(SNAPSHOT),
    "device": "cpu",
    "model_class": type(model).__name__,
    "model_dtype": str(model.dtype),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "loading_diagnostics": {k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()},
    "command": "HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 PYTHONPATH=. python_env/bin/python "
    + str(Path(__file__).relative_to(Path.cwd())),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_metadata_sha256": hashlib.sha256(META.read_bytes()).hexdigest(),
    "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    "prompt_token_ids": prompt,
    "tt_prefix_token_ids": tt[:branch],
    "tt_prefix_text": tokenizer.decode(tt[:branch]),
    "first_hf_tt_divergence_index": first_diff,
    "conditional_rows": rows,
    "conditional_full_tt_top1": float((ranks <= 1).float().mean()),
    "conditional_full_tt_top5": float((ranks <= 5).float().mean()),
    "conditional_full_tt_top100": float((ranks <= 100).float().mean()),
    "hf_continuation_ids": continuation,
    "hf_continuation": tokenizer.decode(continuation, skip_special_tokens=False),
    "hf_reproduces_informal_label_without_forcing_it": tokenizer.decode(continuation).startswith("informal)"),
    "wall_seconds": time.monotonic() - started,
    "ttnn_imported": "ttnn" in sys.modules,
}
assert not report["ttnn_imported"]
(ROOT / "french_branch_control.json").write_text(json.dumps(report, indent=2) + "\n")
(ROOT / "french_branch_hf_continuation.txt").write_text(report["hf_continuation"])
print(json.dumps(report, indent=2), flush=True)
