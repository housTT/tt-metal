"""Natural-text teacher-forced agreement with HF: TT prefill logits of the first L book tokens against HF fp32 logits.

Usage: natural_text_check.py <output.json> [<L> [<first position>]]
The generator is built through build_generator (QWEN3_PRECISION_CONFIG or the default selected config). Over
positions [first, L) it records top-1 and top-5 agreement with HF (HF top-1 inside the TT top-5), mean and minimum
PCC, and the HF margin quantiles. Every position is a real next token of the book, not an HF-generated one, so the
sample is larger and harder than the readiness references.
"""
import bz2
import json
import os
import subprocess
import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

out_path = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
first = int(sys.argv[3]) if len(sys.argv) > 3 else 32
t0 = time.time()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
gen = build_generator(
    "models/autoports/qwen_qwen3_8b",
    mesh,
    decode_buckets=(1,),
    warmup_prefill_lengths=(32,),
    warmup_chunk_starts=(),
    warmup_multi_row=False,
)
tok = gen.tokenizer
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=False)[:L]
gen.reset()
tt = gen.prefill_logits(ids)[0, first:L].float()
config_id = gen.precision_config["config_id"]
config_path = str(gen.precision_config_path)
gen.teardown()
ttnn.close_mesh_device(mesh)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float32, local_files_only=True).eval()
with torch.no_grad():
    hf = model(torch.tensor([ids])).logits[0, first:L].float()
hf_top1 = hf.argmax(-1)
tt_top1 = tt.argmax(-1)
tt_top5 = tt.topk(5, dim=-1).indices
top1 = (tt_top1 == hf_top1).float()
top5 = (tt_top5 == hf_top1[:, None]).any(-1).float()
next_tokens = torch.tensor(ids[first + 1 : L] + [-1])
tt_c = tt - tt.mean(-1, keepdim=True)
hf_c = hf - hf.mean(-1, keepdim=True)
pcc = (tt_c * hf_c).sum(-1) / (tt_c.norm(dim=-1) * hf_c.norm(dim=-1) + 1e-12)
hf_margin = hf.topk(2, dim=-1).values
hf_margin = hf_margin[:, 0] - hf_margin[:, 1]
result = {
    "config_id": config_id,
    "config_path": config_path,
    "book_tokens": L,
    "positions": [first, L],
    "num_positions": int(L - first),
    "top1_agreement_with_hf": float(top1.mean()),
    "top5_agreement_with_hf": float(top5.mean()),
    "hf_top1_equals_next_book_token": float((hf_top1[:-1] == next_tokens[:-1]).float().mean()),
    "tt_top1_equals_next_book_token": float((tt_top1[:-1] == next_tokens[:-1]).float().mean()),
    "pcc_mean": float(pcc.mean()),
    "pcc_min": float(pcc.min()),
    "pcc_quantiles": {str(q): float(pcc.quantile(q)) for q in (0.01, 0.05, 0.1, 0.5)},
    "hf_margin_quantiles": {str(q): float(hf_margin.quantile(q)) for q in (0.05, 0.1, 0.25, 0.5)},
    "top1_agreement_where_hf_margin_above_1": float(top1[hf_margin > 1.0].mean()),
    "top1_agreement_where_hf_margin_above_3": float(top1[hf_margin > 3.0].mean()),
    "disagreements_by_hf_margin_bucket": {
        "<0.5": int(((top1 == 0) & (hf_margin < 0.5)).sum()),
        "0.5-1": int(((top1 == 0) & (hf_margin >= 0.5) & (hf_margin < 1.0)).sum()),
        "1-3": int(((top1 == 0) & (hf_margin >= 1.0) & (hf_margin < 3.0)).sum()),
        ">=3": int(((top1 == 0) & (hf_margin >= 3.0)).sum()),
    },
    "worst_pcc_positions": [
        {
            "position": int(first + i),
            "pcc": float(pcc[i]),
            "hf_margin": float(hf_margin[i]),
            "top1_equal": bool(top1[i]),
        }
        for i in pcc.argsort()[:5].tolist()
    ],
    "seconds": round(time.time() - t0, 1),
    "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
}
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(
    f"{config_id}: top1 {result['top1_agreement_with_hf']:.4f} top5 {result['top5_agreement_with_hf']:.4f} pcc mean {result['pcc_mean']:.4f} min {result['pcc_min']:.4f} "
    f"(HF predicts the next book token at {result['hf_top1_equals_next_book_token']:.3f}, TT at {result['tt_top1_equals_next_book_token']:.3f}); disagreements by HF margin {result['disagreements_by_hf_margin_bucket']}",
    flush=True,
)
print("NATURAL_TEXT_DONE", flush=True)
