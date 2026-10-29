"""Compare the stage-7 TT qualitative outputs with the HF control and the stage-6 TT outputs; write verdict.json."""
import json
import re
import subprocess
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8b/doc")
cur = root / "optimized_full_model" / "qualitative"
prev = root / "full_model" / "qualitative"
tt = json.loads((cur / "tt_outputs.json").read_text())
hf = json.loads((cur / "hf_outputs.json").read_text())
old = json.loads((prev / "tt_outputs.json").read_text())
counters = json.loads((cur / "tt_counters.json").read_text())
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def dup_fraction(ids):
    return sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1]) / max(1, len(ids) - 1)


rows = []
for t, h, o in zip(tt, hf, old):
    text = t["text"]
    non_ascii = sum(1 for c in text if ord(c) > 127) / max(1, len(text))
    leaked = re.findall(r"<\|[a-z_]+\|>", text)
    rows.append(
        {
            "id": t["id"],
            "prompt": t["prompt"],
            "hf_tokens": h["num_tokens"],
            "tt_tokens": t["num_tokens"],
            "first_divergence_index_vs_hf": first_divergence(t["token_ids"], h["token_ids"]),
            "first_divergence_index_vs_stage6_tt": first_divergence(t["token_ids"], o["token_ids"]),
            "stage6_first_divergence_index_vs_hf": first_divergence(o["token_ids"], h["token_ids"]),
            "tt_adjacent_duplicate_fraction": round(dup_fraction(t["token_ids"]), 4),
            "non_ascii_fraction": round(non_ascii, 4),
            "control_tokens_in_text": leaked,
            "seconds": round(t["seconds"], 2),
            "tt_snippet": text[:160],
            "hf_snippet": h["text"][:160],
        }
    )
verdict = {
    "prompt_mode": "chat (HF chat template, add_generation_prompt=True), prompt-format metadata in qualitative_prompt_format.json",
    "control": "HF Qwen/Qwen3-8B greedy on CPU, same rendered token ids (hf_outputs.json, stage-6 control reused: same model id, tokenizer, prompts and rendering)",
    "previous_stage_control": "stage-6 TT outputs (doc/full_model/qualitative/tt_outputs.json)",
    "tt": f"Qwen3Generator.generate greedy, traced split sampling, 128 tokens, chip 2, commit {sha} (tt_outputs.json, tt_counters.json)",
    "program_cache_entries": counters.get("program_cache_entries_after_capture"),
    "seeded_eager_sampling_steps": counters.get("seeded_eager_sampling_steps"),
    "rows": rows,
}
(cur / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
for r in rows:
    print(
        r["id"],
        "div_hf",
        r["first_divergence_index_vs_hf"],
        "(stage6",
        r["stage6_first_divergence_index_vs_hf"],
        ") div_stage6",
        r["first_divergence_index_vs_stage6_tt"],
        "dup",
        r["tt_adjacent_duplicate_fraction"],
        "nonascii",
        r["non_ascii_fraction"],
        r["control_tokens_in_text"],
    )
    print("   TT:", r["tt_snippet"][:140].replace("\n", " "))
