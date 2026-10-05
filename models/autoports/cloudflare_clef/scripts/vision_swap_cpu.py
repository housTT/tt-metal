"""Run the CPU bf16 release backbone and head with the DEVICE vision tower rows.

The reverse of `image_parity.py --vision-source reference`: the Qwen3_5Model
`get_image_features` of the release backbone is wrapped so that its pooler output is
replaced by the device tower rows that `image_parity.py --dump-vision-rows DIR` saved
(DIR/<record_id>.pt, float32 [N, 5120]); the rest of the model (embedding, 64 layers, head)
is the author's CPU bf16 code. The output rows are in the CPU reference format, so
parity_compare.py against ref_image_bf16.jsonl isolates the effect of the device tower
alone, and against the TT rows it isolates the device backbone alone.

Usage (host):
  python vision_swap_cpu.py --rows DIR --input records.jsonl --output out.jsonl [--threads 8] [--snapshot DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import add_snapshot_to_path, read_jsonl, snapshot_dir
from cpu_reference import log, run_records


def install_hook(backbone, rows_dir, current):
    import torch

    text_model = backbone.model
    original = text_model.get_image_features

    def get_image_features(pixel_values, image_grid_thw=None, **kwargs):
        output = original(pixel_values, image_grid_thw, **kwargs)
        rows = torch.load(rows_dir / f"{current['id']}.pt").to(torch.bfloat16)
        split_sizes = (image_grid_thw.prod(-1) // text_model.visual.spatial_merge_size**2).tolist()
        assert int(rows.shape[0]) == sum(split_sizes), (rows.shape, split_sizes)
        hf = torch.cat(list(output.pooler_output), dim=0).float()
        current["pcc"] = pcc(hf, rows.float())
        output.pooler_output = torch.split(rows, split_sizes)
        return output

    text_model.get_image_features = get_image_features


def pcc(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return round(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)), 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    snapshot = snapshot_dir(args.snapshot)
    add_snapshot_to_path(snapshot)
    import joint_schema_model as jsm

    started = time.perf_counter()
    model, processor = jsm.load_release_model(snapshot, device="cpu", dtype=torch.bfloat16)
    log(f"model loaded in {time.perf_counter() - started:.1f}s")
    current = {"id": None, "pcc": None}
    install_hook(model.language_model, Path(args.rows), current)
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    original_answer = jsm.encode_record

    def encode_record(tokenizer, record, **kwargs):
        current["id"] = record.get("id")
        return original_answer(tokenizer, record, **kwargs)

    jsm.encode_record = encode_record
    seconds = run_records(model, processor, jsm, rows, Path(args.output))
    log(f"SUMMARY records={len(rows)} ok={len(seconds)} output={args.output}")
    log("VISION_SWAP_CPU_DONE")


if __name__ == "__main__":
    main()
