"""Compare the stage-8 TT qualitative outputs with the HF control and the stage-7 TT outputs; write verdict.json."""
import json
import re
from pathlib import Path

root = Path("models/autoports/qwen_qwen3_8b/doc")
cur = root / "datatype_sweep" / "qualitative"
prev = root / "optimized_full_model" / "qualitative"
tt = json.loads((cur / "tt_outputs.json").read_text())
hf = json.loads((cur / "hf_outputs.json").read_text())
old = json.loads((prev / "tt_outputs.json").read_text())
counters = json.loads((cur / "tt_counters.json").read_text())
prov = json.loads((cur / "tt_provenance.json").read_text())


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def dup_fraction(ids):
    return sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1]) / max(1, len(ids) - 1)


def trigram_loop(ids):
    seen = {}
    for i in range(len(ids) - 2):
        key = tuple(ids[i : i + 3])
        seen[key] = seen.get(key, 0) + 1
    repeats = sum(v - 1 for v in seen.values())
    return repeats / max(1, len(ids) - 2)


rows = []
for t, h, o in zip(tt, hf, old):
    assert t["id"] == h["id"] == o["id"]
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
            "first_divergence_index_vs_stage7_tt": first_divergence(t["token_ids"], o["token_ids"]),
            "stage7_first_divergence_index_vs_hf": first_divergence(o["token_ids"], h["token_ids"]),
            "tt_adjacent_duplicate_fraction": round(dup_fraction(t["token_ids"]), 4),
            "tt_trigram_repeat_fraction": round(trigram_loop(t["token_ids"]), 4),
            "non_ascii_fraction": round(non_ascii, 4),
            "control_tokens_in_text": leaked,
            "think_block": text.startswith("<think>"),
            "seconds": round(t["seconds"], 2),
            "tt_snippet": text[:200],
            "hf_snippet": h["text"][:200],
            "stage7_snippet": o["text"][:200],
        }
    )
verdict = {
    "prompt_mode": "chat (HF chat template, add_generation_prompt=True), prompt-format metadata in qualitative_prompt_format.json",
    "control": "HF Qwen/Qwen3-8B greedy on CPU, same rendered token ids (hf_outputs.json, stage-6 control reused: same model id, revision, tokenizer, prompts and rendering)",
    "previous_stage_control": "stage-7 TT outputs on the baseline policy (doc/optimized_full_model/qualitative/tt_outputs.json)",
    "tt": f"Qwen3Generator.generate greedy, traced split sampling, {prov['max_new_tokens']} tokens, chip {prov['visible_devices']}, commit {prov['checkout_git_sha']}, precision config {prov['precision_config_id']} ({prov['precision_config_path']})",
    "program_cache_entries": counters.get("program_cache_entries_after_capture"),
    "seeded_eager_sampling_steps": counters.get("seeded_eager_sampling_steps"),
    "eager_decode_steps": counters.get("eager_decode_steps"),
    "rows": rows,
}
(cur / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
for r in rows:
    print(
        f"{r['id']}: tokens {r['tt_tokens']}, div vs HF {r['first_divergence_index_vs_hf']} (stage 7: {r['stage7_first_divergence_index_vs_hf']}), "
        f"div vs stage-7 TT {r['first_divergence_index_vs_stage7_tt']}, dup {r['tt_adjacent_duplicate_fraction']}, "
        f"trigram {r['tt_trigram_repeat_fraction']}, non-ascii {r['non_ascii_fraction']}, leaked {r['control_tokens_in_text']}, think {r['think_block']}"
    )
    print("   TT:", r["tt_snippet"].replace("\n", " ")[:160])
    print("   HF:", r["hf_snippet"].replace("\n", " ")[:160])
