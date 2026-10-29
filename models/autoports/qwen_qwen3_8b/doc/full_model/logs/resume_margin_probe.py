"""Why do the prefix_cache_hit rows flip top-1? Compare both TT paths against HF logits over the window."""
import bz2
import json
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

import os
from models.autoports.qwen_qwen3_8b.tt.model import TerminalPolicy
pairs = [(int(a), int(b)) for a, b in (p.split(":") for p in sys.argv[1:])] or [(2560, 2528)]
dtype = os.environ.get("QWEN3_PROBE_LOGITS_DTYPE", "fp32")
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
gen = build_generator("models/autoports/qwen_qwen3_8b", mesh, decode_buckets=(1,), warmup_prefill_lengths=(32,), terminal_policy=TerminalPolicy(prefill_logits_dtype=dtype))
tok = gen.tokenizer
text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read()
all_ids = tok.encode(text, add_special_tokens=False)
tt = {}
for L, S in pairs:
    ids = all_ids[:L]
    positions = list(range(S, L))
    gen.reset()
    ref = gen.prefill_logits_continuation(ids, 0, positions=positions)[0]
    gen.reset()
    cand = gen.prefill_logits_continuation(ids, S, positions=positions)[0]
    tt[(L, S)] = (ref, cand, positions)
gen.teardown()
ttnn.close_mesh_device(mesh)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.float32).eval()
for (L, S), (ref, cand, positions) in tt.items():
  ids = all_ids[:L]
  with torch.no_grad():
    hf = model(torch.tensor([ids])).logits[0, S:L].float()
  OUT = Path(f"models/autoports/qwen_qwen3_8b/doc/full_model/resume_margin_probe_L{L}_s{S}_{dtype}.json")

  def top1(x):
    return x.argmax(-1)
  def margin(x):
    v = x.topk(2, dim=-1).values
    return (v[:, 0] - v[:, 1])
  rt, ct, ht = top1(ref), top1(cand), top1(hf)
  rec = {
    "L": L, "start": S, "window": len(positions), "logits_dtype": dtype,
    "top1_ref_vs_cand": float((rt == ct).float().mean()),
    "top1_ref_vs_hf": float((rt == ht).float().mean()),
    "top1_cand_vs_hf": float((ct == ht).float().mean()),
    "flipped_positions": [],
}
  for i in range(len(positions)):
    if rt[i] != ct[i]:
        rec["flipped_positions"].append({
            "position": positions[i],
            "ref_top1": int(rt[i]), "cand_top1": int(ct[i]), "hf_top1": int(ht[i]),
            "ref_margin": float(margin(ref[i : i + 1])[0]), "cand_margin": float(margin(cand[i : i + 1])[0]), "hf_margin": float(margin(hf[i : i + 1])[0]),
            "hf_logit_gap_between_the_two": float(hf[i, int(rt[i])] - hf[i, int(ct[i])]),
        })
  def pcc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))
  per_pos = []
  for i in range(len(positions)):
    per_pos.append({"position": positions[i], "pcc_ref_vs_cand": pcc(ref[i], cand[i]), "pcc_ref_vs_hf": pcc(ref[i], hf[i]), "pcc_cand_vs_hf": pcc(cand[i], hf[i]), "ref_top1": int(rt[i]), "cand_top1": int(ct[i]), "hf_top1": int(ht[i]), "hf_margin": float(margin(hf[i:i+1])[0]), "max_abs_diff_ref_cand": float((ref[i]-cand[i]).abs().max()), "max_abs_diff_ref_hf": float((ref[i]-hf[i]).abs().max()), "max_abs_diff_cand_hf": float((cand[i]-hf[i]).abs().max())})
  worst = sorted(per_pos, key=lambda r: r["pcc_ref_vs_cand"])[:5]
  rec["worst_positions_ref_vs_cand"] = worst
  rec["pcc_ref_vs_cand_min"] = min(r["pcc_ref_vs_cand"] for r in per_pos)
  rec["pcc_ref_vs_hf_min"] = min(r["pcc_ref_vs_hf"] for r in per_pos)
  rec["pcc_cand_vs_hf_min"] = min(r["pcc_cand_vs_hf"] for r in per_pos)
  rec["pcc_ref_vs_hf_mean"] = sum(r["pcc_ref_vs_hf"] for r in per_pos) / len(per_pos)
  rec["pcc_cand_vs_hf_mean"] = sum(r["pcc_cand_vs_hf"] for r in per_pos) / len(per_pos)
  rec["hf_margin_quantiles"] = {q: float(margin(hf).quantile(q)) for q in (0.05, 0.1, 0.25, 0.5)}
  rec["positions_with_hf_margin_below_0p5"] = int((margin(hf) < 0.5).sum())
  rec["ref_margin_quantiles"] = {q: float(margin(ref).quantile(q)) for q in (0.05, 0.1, 0.25, 0.5)}
  OUT.write_text(json.dumps(rec, indent=2) + "\n")
  print(json.dumps(rec, indent=1), flush=True)
print("MARGIN_PROBE_OK")
