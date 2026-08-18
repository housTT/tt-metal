# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Three-way comparison of the shared qualitative suite: HF, the pre-sweep policy, the selected one.

`$qualitative-check` asks for prompt-format metadata, the rendered prompts, and an HF control. The HF
column here is the optimized full-model stage's, not a fresh run: the HF reference is a torch model
that knows nothing about the TTNN precision policy, so it is the *same* control for both arms, and a
fresh 35B CPU reference cannot be loaded on this host (see ``host_memory.md``). What this stage
actually has to answer - "does the selected precision config change the visible text?" - is answered
by the two TT columns, which are the same suite, the same rendered prompts and the same greedy
decode, differing only in the precision policy.

    python .../doc/datatype_sweep/logs/compare_qualitative.py
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

ROOT = Path("models/autoports/ornith_ai_ornith_1_0_35b")
SWEEP = ROOT / "doc" / "datatype_sweep"
SELECTED = SWEEP / "readiness_qualitative.json"
BASELINE = SWEEP / "readiness_qualitative_baseline.json"
HF_ARCHIVE = ROOT / "doc" / "optimized_full_model" / "readiness_qualitative.json"


def _degenerate(text: str) -> dict:
    """The same shape of degeneracy check the optimized full-model stage's gate applies."""
    words = [w.lower() for w in text.split()]
    doubled = sum(1 for a, b in zip(words, words[1:]) if a == b)
    trigrams = [" ".join(words[i : i + 3]) for i in range(max(len(words) - 2, 0))]
    repeated = 1 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
    return {
        "words": len(words),
        "immediate_word_doubling_rate": doubled / max(len(words) - 1, 1),
        "repeated_trigram_rate": repeated,
        "non_ascii_rate": sum(1 for c in text if ord(c) > 127) / max(len(text), 1),
        "empty": not text.strip(),
    }


def main():
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    archive = json.loads(HF_ARCHIVE.read_text(encoding="utf-8"))

    hf_by_prompt = {row["prompt"]: row["completion"] for row in archive["hf"]}
    base_by_prompt = {row["prompt"]: row["completion"] for row in baseline["tt"]}

    # The two TT arms must have seen byte-identical rendered prompts, or the comparison is not one.
    rendered_selected = {r["prompt"]: r["token_ids"] for r in selected["prompts"]}
    rendered_baseline = {r["prompt"]: r["token_ids"] for r in baseline["prompts"]}
    rendered_archive = {r["prompt"]: r["token_ids"] for r in archive["prompts"]}
    assert (
        rendered_selected == rendered_baseline == rendered_archive
    ), "the three arms did not see the same rendered prompts, so their completions are not comparable"

    rows = []
    for row in selected["tt"]:
        prompt = row["prompt"]
        text = row["completion"]
        base = base_by_prompt.get(prompt, "")
        hf = hf_by_prompt.get(prompt, "")
        matcher = difflib.SequenceMatcher(None, base.split(), text.split())
        common_prefix = 0
        for a, b in zip(base.split(), text.split()):
            if a != b:
                break
            common_prefix += 1
        rows.append(
            {
                "prompt": prompt,
                "selected_completion": text,
                "baseline_policy_completion": base,
                "hf_completion": hf,
                "selected_vs_baseline_word_similarity": matcher.ratio(),
                "selected_vs_baseline_identical": base == text,
                "identical_leading_words": common_prefix,
                "selected_degeneracy": _degenerate(text),
                "baseline_degeneracy": _degenerate(base),
                "hf_degeneracy": _degenerate(hf) if hf else None,
            }
        )

    summary = {
        "prompt_format": selected["prompt_format"],
        "arms": {
            "hf": {
                "source": str(HF_ARCHIVE),
                "why_archived": "the HF reference is a torch model with no dependence on the TTNN "
                "precision policy, so it is the same control for both TT arms; a fresh 35B CPU "
                "reference needs ~70 GiB and this host has ~50 GiB available (doc/datatype_sweep/"
                "host_memory.md), the same persistent condition doc/optimized_full_model/logs/"
                "host_memory_event.txt records",
            },
            "baseline_policy": {"source": str(BASELINE), "policy": "optimized (the pre-sweep decoder-stage policy)"},
            "selected": {"source": str(SELECTED), "policy": "the selected precision config, taken by default"},
        },
        "n_prompts": len(rows),
        "mean_selected_vs_baseline_word_similarity": sum(r["selected_vs_baseline_word_similarity"] for r in rows)
        / max(len(rows), 1),
        "worst_selected_vs_baseline_word_similarity": min(
            (r["selected_vs_baseline_word_similarity"] for r in rows), default=None
        ),
        "any_degenerate": any(
            r["selected_degeneracy"]["empty"]
            or r["selected_degeneracy"]["immediate_word_doubling_rate"] > 0.10
            or r["selected_degeneracy"]["repeated_trigram_rate"] > 0.50
            or r["selected_degeneracy"]["non_ascii_rate"] > 0.20
            for r in rows
        ),
        "rows": rows,
    }
    out = SWEEP / "qualitative_comparison.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    lines = [
        "| prompt | selected vs baseline-policy word similarity | identical leading words | doubling | repeated trigrams |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["selected_degeneracy"]
        lines.append(
            f"| {r['prompt'][:44]}… | {r['selected_vs_baseline_word_similarity']:.3f} | "
            f"{r['identical_leading_words']} | {d['immediate_word_doubling_rate']:.3f} | "
            f"{d['repeated_trigram_rate']:.3f} |"
        )
    table = "\n".join(lines)
    (SWEEP / "qualitative_comparison.md").write_text(
        "# Qualitative suite: selected precision config against the pre-sweep policy\n\n"
        f"{len(rows)} prompts, chat template, greedy, {selected['prompt_format']['generation']['max_new_tokens']} "
        "new tokens. Both TT arms saw byte-identical rendered prompts (asserted).\n\n" + table + "\n",
        encoding="utf-8",
    )
    print(table)
    print("\nmean similarity:", round(summary["mean_selected_vs_baseline_word_similarity"], 4))
    print("any degenerate:", summary["any_degenerate"])


if __name__ == "__main__":
    main()
