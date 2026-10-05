"""Collect every measurement of one image record into one JSON for the tower-or-backbone classification.

For the record id the script computes, on the host, the PCC of every saved device tower row file
(--rows name=path, float32 or bf16 [N, 5120]) against the HF CPU bf16 reference rows
(vision_ref/<record>.pt, key "merged") and against the HF tower rebuilt in float32 on the same
pixel_values (the bf16-versus-fp32 floor of the reference itself), and the pairwise PCC between the
row files (process-to-process reproducibility). It then reads the probability row of the record
from every --probs name=path JSONL file (CPU references, TT runs, swap runs) and prints the
distribution, the argmax, the margin and the max |dp| against the first --probs file (the
reference). Nothing here touches a device.

Usage (host):
  python sensitive_record_report.py --record 7b7ef38338fffa28b9106027c882c1b4 \
      --rows accuracy=/path/a.pt --rows upstream=/path/b.pt \
      --probs ref_bf16=/path/ref.jsonl --probs tt_accuracy=/path/tt.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl

VISION_REF_DIR = Path("/home/hous/dev/clef/reports/reference/vision_ref")


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return round(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)), 6)


def pair(value):
    name, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected name=path, got {value!r}")
    return name, Path(path)


def hf_fp32_rows(reference):
    import torch

    from models.autoports.cloudflare_clef.tt import vision as clef_vision

    visual = clef_vision.load_hf_visual(dtype=torch.float32)
    with torch.inference_mode():
        out = visual(reference["pixel_values"].float(), grid_thw=reference["image_grid_thw"])
    return out.pooler_output if hasattr(out, "pooler_output") else out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--rows", type=pair, action="append", default=[])
    parser.add_argument("--probs", type=pair, action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-fp32-tower", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(8)
    report = dict(record=args.record, rows={}, probs={}, pairwise_rows_pcc={})
    reference = torch.load(VISION_REF_DIR / f"{args.record}.pt")
    merged_ref = reference["merged"].float()
    report["grid"] = reference["image_grid_thw"].tolist()
    report["n_tokens"] = int(merged_ref.shape[0])
    fp32 = None
    if not args.no_fp32_tower:
        t0 = time.perf_counter()
        fp32 = hf_fp32_rows(reference).float()
        report["hf_fp32_tower_seconds"] = round(time.perf_counter() - t0, 1)
        report["hf_bf16_vs_fp32_pcc"] = pcc(merged_ref, fp32)
    loaded = {}
    for name, path in args.rows:
        if not path.exists():
            report["rows"][name] = dict(missing=str(path))
            continue
        rows = torch.load(path)
        rows = rows if isinstance(rows, torch.Tensor) else rows["merged"]
        rows = rows.float()
        loaded[name] = rows
        entry = dict(path=str(path), shape=list(rows.shape), pcc_vs_hf_bf16=pcc(merged_ref, rows))
        if fp32 is not None:
            entry["pcc_vs_hf_fp32"] = pcc(fp32, rows)
        entry["max_abs_vs_hf_bf16"] = float((rows - merged_ref).abs().max())
        entry["std"] = float(rows.std())
        report["rows"][name] = entry
    names = list(loaded)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            report["pairwise_rows_pcc"][f"{a}|{b}"] = dict(
                pcc=pcc(loaded[a], loaded[b]), max_abs=float((loaded[a] - loaded[b]).abs().max())
            )
    reference_probs = None
    for name, path in args.probs:
        if not path.exists():
            report["probs"][name] = dict(missing=str(path))
            continue
        row = next((r for r in read_jsonl(path) if r.get("id") == args.record), None)
        if row is None or "probs" not in row:
            report["probs"][name] = dict(missing_record=str(path))
            continue
        entry = {}
        for qid, dist in row["probs"].items():
            top = sorted(dist.values(), reverse=True)
            q = dict(
                probs={o: round(p, 4) for o, p in dist.items()},
                argmax=max(dist, key=dist.__getitem__),
                margin=round(top[0] - top[1], 4),
            )
            if reference_probs is not None:
                q["max_dp_vs_reference"] = round(max(abs(dist[o] - reference_probs[qid][o]) for o in dist), 4)
            entry[qid] = q
        if reference_probs is None:
            reference_probs = row["probs"]
            report["probs_reference"] = name
        entry["_path"] = str(path)
        entry["_vision"] = (
            {k: row["vision"].get(k) for k in ("grid", "device_vs_reference_pcc")} if row.get("vision") else None
        )
        report["probs"][name] = entry
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
