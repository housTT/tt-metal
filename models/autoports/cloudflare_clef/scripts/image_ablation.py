"""Check that the image changes the answer of the New Yorker reference records on CPU.

For every record of records_image.jsonl this runs the model twice, once with the image
and once with the same request minus "images", and writes both probability vectors,
the processor's image_grid_thw, the number of image tokens, and the max absolute
probability difference. Output: one JSON file.

Usage:
  python image_ablation.py --input records_image.jsonl --output image_ablation.json [--snapshot DIR] [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import add_snapshot_to_path, read_jsonl, snapshot_dir
from cpu_reference import answer_with_probs, build_request, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    snapshot = snapshot_dir(args.snapshot)
    add_snapshot_to_path(snapshot)
    import joint_schema_model as jsm

    model, processor = jsm.load_release_model(snapshot, device="cpu", dtype=torch.bfloat16)
    results = []
    for row in read_jsonl(args.input):
        with_image = build_request(row)
        encoded = jsm.encode_record(processor.tokenizer, with_image, processor=processor)
        grid = encoded.media["image_grid_thw"].tolist() if encoded.media else None
        image_tokens = sum(int(t) for t in encoded.media["mm_token_type_ids"]) if encoded.media else 0
        started = time.perf_counter()
        _, probs_image = answer_with_probs(model, processor, with_image, jsm)
        image_seconds = time.perf_counter() - started
        text_only = {key: value for key, value in with_image.items() if key != "images"}
        started = time.perf_counter()
        _, probs_text = answer_with_probs(model, processor, text_only, jsm)
        text_seconds = time.perf_counter() - started
        question_id = next(iter(probs_image))
        diff = max(abs(probs_image[question_id][k] - probs_text[question_id][k]) for k in probs_image[question_id])
        gold = row.get("_label", {}).get(question_id)
        pick_image = max(probs_image[question_id], key=probs_image[question_id].__getitem__)
        pick_text = max(probs_text[question_id], key=probs_text[question_id].__getitem__)
        result = {
            "id": row["id"],
            "image_size": with_image["images"][0].size,
            "image_grid_thw": grid,
            "image_tokens": image_tokens,
            "input_tokens_with_image": len(encoded.input_ids),
            "gold": gold,
            "with_image": {"probs": probs_image[question_id], "choice": pick_image, "seconds": round(image_seconds, 2)},
            "text_only": {"probs": probs_text[question_id], "choice": pick_text, "seconds": round(text_seconds, 2)},
            "max_abs_prob_diff": round(diff, 4),
        }
        results.append(result)
        log(
            f"{row['id']} grid={grid} image_tokens={image_tokens} gold={gold} image={pick_image} text={pick_text} max|dp|={diff:.3f}"
        )
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    log(f"IMAGE_ABLATION_DONE -> {args.output}")


if __name__ == "__main__":
    main()
