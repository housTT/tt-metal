"""Render three public benchmarks into labelled SystemOne request JSONL for Cloudflare/clef.

Outputs under /home/hous/dev/clef/evals/:
  arc_challenge_test.jsonl        allenai/ai2_arc, config ARC-Challenge, split test (1,172)
  banking77_test.jsonl            PolyAI/banking77, split test (3,080)
  newyorker_matching_test.jsonl   jmhessel/newyorker_caption_contest, config matching, split test (528)
  <name>_sample100.jsonl          a 100-item sample of each, stratified by _label, random.Random(0)
  images/newyorker/<instance_id>.png  the test cartoons
  MANIFEST.json                   counts, dataset revisions, fingerprints, rendering constants

Each line is a SystemOne request body {"id", "model": "clef", "state", "questions"} plus
"_label" (question id -> gold option id) and "_source". One fixed rendering: ARC uses the
original choice labels as option ids (A-E or 1-4 as the item gives them); BANKING77 uses
the 77 intent names as option ids with the description equal to the name with underscores
replaced by spaces; New Yorker uses A-E over the five captions with the image attached as
{"images": [absolute png path]}. No option is ever dropped and no prompt is tuned.

Usage:
  python render_public_evals.py [--out /home/hous/dev/clef/evals] [--skip-images]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import write_jsonl

ARC_INSTRUCTIONS = "Which answer is correct?"
BANKING_INSTRUCTIONS = "Which banking intent does the customer have?"
NEWYORKER_STATE = "A New Yorker cartoon. Pick the caption that was written for it."
NEWYORKER_INSTRUCTIONS = "Which caption was written for this cartoon?"
LETTERS = ("A", "B", "C", "D", "E")
SAMPLE_SIZE = 100


def hub_sha(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception as error:
        return f"unavailable: {type(error).__name__}"


def dataset_meta(dataset, repo: str) -> dict:
    info = dataset.info
    return {
        "repo": repo,
        "config": info.config_name,
        "split": str(dataset.split),
        "builder": info.builder_name,
        "version": str(info.version),
        "download_checksums": sorted((info.download_checksums or {}).keys()),
        "fingerprint": dataset._fingerprint,
        "hub_main_sha_at_render": hub_sha(repo),
        "num_rows": len(dataset),
    }


def render_arc(dataset) -> list[dict]:
    rows = []
    for index, item in enumerate(dataset):
        labels = list(item["choices"]["label"])
        texts = list(item["choices"]["text"])
        if item["answerKey"] not in labels:
            raise RuntimeError(f"{item['id']}: answerKey {item['answerKey']!r} not in {labels}")
        rows.append(
            {
                "id": item["id"],
                "model": "clef",
                "state": item["question"],
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": ARC_INSTRUCTIONS,
                        "criteria": dict(zip(labels, texts)),
                    }
                },
                "_label": {"answer": item["answerKey"]},
                "_source": "allenai/ai2_arc ARC-Challenge test",
                "_source_row": index,
            }
        )
    return rows


def render_banking(dataset) -> list[dict]:
    names = dataset.features["label"].names
    criteria = {name: name.replace("_", " ") for name in names}
    rows = []
    for index, item in enumerate(dataset):
        rows.append(
            {
                "id": f"banking77/test/{index}",
                "model": "clef",
                "state": item["text"],
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": BANKING_INSTRUCTIONS,
                        "criteria": dict(criteria),
                    }
                },
                "_label": {"intent": names[item["label"]]},
                "_source": "PolyAI/banking77 test",
                "_source_row": index,
            }
        )
    return rows


def render_newyorker(dataset, image_dir: Path, save_images: bool) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(dataset):
        png = image_dir / f"{item['instance_id']}.png"
        if save_images:
            item["image"].convert("RGB").save(png, format="PNG")
        captions = list(item["caption_choices"])
        if len(captions) != len(LETTERS):
            raise RuntimeError(f"{item['instance_id']}: expected 5 captions, got {len(captions)}")
        if item["label"] not in LETTERS:
            raise RuntimeError(f"{item['instance_id']}: label {item['label']!r}")
        rows.append(
            {
                "id": item["instance_id"],
                "model": "clef",
                "state": NEWYORKER_STATE,
                "images": [str(png)],
                "questions": {
                    "caption": {
                        "type": "choice",
                        "instructions": NEWYORKER_INSTRUCTIONS,
                        "criteria": dict(zip(LETTERS, captions)),
                    }
                },
                "_label": {"caption": item["label"]},
                "_source": "jmhessel/newyorker_caption_contest matching test",
                "_source_row": index,
                "_contest_number": item["contest_number"],
            }
        )
    return rows


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = collections.OrderedDict()
    for row in rows:
        key = json.dumps(row["_label"], sort_keys=True)
        groups.setdefault(key, []).append(row)
    total = len(rows)
    quota = {}
    remainders = []
    for key, members in groups.items():
        exact = size * len(members) / total
        quota[key] = int(exact)
        remainders.append((exact - int(exact), key))
    if len(groups) <= size:
        for key in groups:
            quota[key] = max(quota[key], 1)
    assigned = sum(quota.values())
    remainders.sort(key=lambda pair: (-pair[0], pair[1]))
    cursor = 0
    while assigned < size and cursor < len(remainders):
        key = remainders[cursor][1]
        if quota[key] < len(groups[key]):
            quota[key] += 1
            assigned += 1
        cursor += 1
    while assigned > size:
        key = max(quota, key=lambda k: (quota[k], k))
        quota[key] -= 1
        assigned -= 1
    sample = []
    for key, members in groups.items():
        sample.extend(rng.sample(members, min(quota[key], len(members))))
    rng.shuffle(sample)
    if len(sample) != size:
        raise RuntimeError(f"sample has {len(sample)} rows, wanted {size}")
    return sample


def label_counts(rows: list[dict]) -> dict:
    counter = collections.Counter(next(iter(row["_label"].values())) for row in rows)
    return dict(sorted(counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/hous/dev/clef/evals")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    manifest = {"seed": 0, "sample_size": SAMPLE_SIZE, "datasets": {}}
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_rows = render_arc(arc)
    banking = load_dataset("PolyAI/banking77", split="test", trust_remote_code=True)
    banking_rows = render_banking(banking)
    newyorker = load_dataset("jmhessel/newyorker_caption_contest", "matching", split="test")
    newyorker_rows = render_newyorker(newyorker, out / "images" / "newyorker", not args.skip_images)

    for name, dataset, repo, rows, instructions in (
        ("arc_challenge_test", arc, "allenai/ai2_arc", arc_rows, ARC_INSTRUCTIONS),
        ("banking77_test", banking, "PolyAI/banking77", banking_rows, BANKING_INSTRUCTIONS),
        (
            "newyorker_matching_test",
            newyorker,
            "jmhessel/newyorker_caption_contest",
            newyorker_rows,
            NEWYORKER_INSTRUCTIONS,
        ),
    ):
        full = out / f"{name}.jsonl"
        sample = out / f"{name}_sample100.jsonl"
        sample_rows = stratified_sample(rows, SAMPLE_SIZE, 0)
        write_jsonl(full, rows)
        write_jsonl(sample, sample_rows)
        option_counts = collections.Counter(len(next(iter(r["questions"].values()))["criteria"]) for r in rows)
        manifest["datasets"][name] = {
            **dataset_meta(dataset, repo),
            "rendered_rows": len(rows),
            "sample_rows": len(sample_rows),
            "instructions": instructions,
            "options_per_item": dict(sorted(option_counts.items())),
            "label_counts_full": label_counts(rows),
            "label_counts_sample": label_counts(sample_rows),
            "full_path": str(full),
            "sample_path": str(sample),
        }
        print(name, len(rows), "rows ->", full)
        print(name, len(sample_rows), "sample rows ->", sample)
    manifest["datasets"]["newyorker_matching_test"]["state"] = NEWYORKER_STATE
    manifest["datasets"]["newyorker_matching_test"]["image_dir"] = str(out / "images" / "newyorker")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("manifest ->", out / "MANIFEST.json")


if __name__ == "__main__":
    main()
