"""Run the image and video reference records through the TT engine and compare with the CPU reference.

For each set the script runs ClefEngine.probs_for_request (full path, and the cached path for the
sets listed in --cached), writes candidate rows in the CPU reference format, then runs
parity_compare.py against the CPU bf16 reference with the stage 2 bars (max |dp| 0.10, flip margin
0.05). Per record it also keeps the vision tower time against the image token count, the device
time and the end-to-end latency. The summary JSON carries the engine load timings and the DRAM
free bytes after the text weights, after the vision tower and after the snapshot slots.

--vision-source reference replaces the device tower output of every record with the HF CPU
bf16 tower rows saved by scripts/vision_reference.py (vision_ref/<record_id>.pt, key "merged"),
so the backbone and the head run on the device with the reference image features; the
difference against the plain run attributes the parity deltas between the tower and the
backbone. --dump-vision-rows DIR saves the device tower rows of every record as
DIR/<record_id>.pt (float32 [N, 5120]) for the reverse experiment on the CPU
(scripts/vision_swap_cpu.py).

Usage (device, through devrun):
  python image_parity.py [--sets reference,dev16,video] [--cached reference,video] [--n-layers N]
                         [--out-dir /home/hous/dev/clef/reports] [--tag stage2]
                         [--vision-source device|reference] [--dump-vision-rows DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, write_jsonl

REFERENCE_DIR = Path("/home/hous/dev/clef/reports/reference")
VISION_REF_DIR = REFERENCE_DIR / "vision_ref"
SETS = {
    "reference": (REFERENCE_DIR / "records_image.jsonl", REFERENCE_DIR / "ref_image_bf16.jsonl"),
    "dev16": (REFERENCE_DIR / "dev16_image.jsonl", REFERENCE_DIR / "dev16_image.ref_bf16.jsonl"),
    "video": (REFERENCE_DIR / "records_video.jsonl", REFERENCE_DIR / "ref_video_bf16.jsonl"),
    "two_image": (REFERENCE_DIR / "records_two_image.jsonl", REFERENCE_DIR / "ref_two_image_bf16.jsonl"),
}
PARITY_COMPARE = Path(__file__).resolve().parent / "parity_compare.py"
MAX_DP_BAR = 0.10
FLIP_MARGIN = 0.05


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def compare(reference, candidate, name, out_stem):
    cmd = [
        sys.executable,
        str(PARITY_COMPARE),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--max-dp-bar",
        str(MAX_DP_BAR),
        "--margin",
        str(FLIP_MARGIN),
        "--no-margin-flips",
        "--name",
        name,
        "--out-json",
        f"{out_stem}.json",
        "--out-md",
        f"{out_stem}.md",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log(f"parity_compare rc={proc.returncode}\n{proc.stdout}{proc.stderr[-2000:]}")
    summary = json.loads(Path(f"{out_stem}.json").read_text())
    return dict(rc=proc.returncode, overall=summary["overall"], command=" ".join(cmd), json=f"{out_stem}.json")


def cross_max_dp(rows_a, rows_b):
    worst = 0.0
    for a, b in zip(rows_a, rows_b):
        if "error" in a or "error" in b:
            continue
        for qid, dist in a["probs"].items():
            worst = max(worst, max(abs(dist[o] - b["probs"][qid][o]) for o in dist))
    return worst


def install_vision_hooks(engine, source, dump_dir):
    import torch

    import ttnn

    current = {"id": None}
    original_probs = engine.probs_for_request
    original_vision = engine._vision_request

    def probs_for_request(record, *args, **kwargs):
        current["id"] = record.get("id")
        return original_probs(record, *args, **kwargs)

    def vision_request(media):
        request = original_vision(media)
        if request is None:
            return None
        rows = engine.vision.rows_to_torch(request.tokens)
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            torch.save(rows, dump_dir / f"{current['id']}.pt")
        if source == "reference":
            reference = torch.load(VISION_REF_DIR / f"{current['id']}.pt")["merged"]
            assert tuple(reference.shape) == tuple(rows.shape), (reference.shape, rows.shape)
            request.release()
            request.tokens = ttnn.from_torch(
                reference.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=engine.mesh,
                mesh_mapper=ttnn.ShardTensor2dMesh(engine.mesh, dims=(None, -1), mesh_shape=engine.args.cluster_shape),
            )
            engine.last_vision["source"] = "reference"
            engine.last_vision["device_vs_reference_pcc"] = pcc(reference, rows)
        return request

    engine.probs_for_request = probs_for_request
    engine._vision_request = vision_request


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return round(float((a * b).sum() / (a.norm() * b.norm() + 1e-8)), 6)


def per_record(rows):
    out = {}
    for row in rows:
        if "error" in row:
            out[row["id"]] = dict(error=row["error"])
            continue
        vision = row.get("vision") or {}
        out[row["id"]] = dict(
            input_tokens=row["input_tokens"],
            media_rows=vision.get("n_rows"),
            media_patches=vision.get("n_patches"),
            padded_rows=(vision.get("tower") or {}).get("rows"),
            tower_pcc_vs_reference=vision.get("device_vs_reference_pcc"),
            vision_s=row["timing"].get("vision_s"),
            device_s=row["timing"]["device_s"],
            encode_s=row["timing"]["encode_s"],
            head_s=row["timing"]["head_s"],
            seconds=row["seconds"],
            cache_hit=row.get("cache_hit"),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="reference,dev16,video")
    parser.add_argument("--cached", default="reference,video")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--out-dir", default="/home/hous/dev/clef/reports")
    parser.add_argument("--tag", default="stage2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vision-source", default="device", choices=("device", "reference"))
    parser.add_argument("--dump-vision-rows", default=None)
    args = parser.parse_args()
    from models.autoports.cloudflare_clef.scripts import reference_rows
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, mesh_dram_free_bytes, tp2_mesh

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.tag}_image_parity.json"
    sets = [name for name in args.sets.split(",") if name]
    cached_sets = {name for name in args.cached.split(",") if name}
    summary = dict(sets={}, gates={})
    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, n_layers=args.n_layers)
        if args.vision_source != "device" or args.dump_vision_rows:
            install_vision_hooks(
                engine, args.vision_source, Path(args.dump_vision_rows) if args.dump_vision_rows else None
            )
        summary["vision_source"] = args.vision_source
        summary["dump_vision_rows"] = args.dump_vision_rows
        summary["engine"] = dict(
            n_layers=engine.args.n_layers,
            precision=engine.precision,
            timings=engine.timings,
            vision=engine.vision.describe() if engine.vision is not None else None,
            dram_free_after_text_weights_bytes=engine.dram_free_after_text_weights,
            dram_free_after_weights_bytes=engine.dram_free_after_weights,
            dram_free_after_slots_bytes=engine.dram_free_after_slots,
            snapshot_slots=engine.snapshot_slots,
            submesh_chips=list(mesh.get_device_ids()),
        )
        log(f"engine: {json.dumps(summary['engine'], default=str)}")
        for name in sets:
            records_path, reference_path = SETS[name]
            if not reference_path.exists():
                log(f"{name}: missing CPU reference {reference_path}; skipped")
                summary["sets"][name] = dict(skipped=f"missing reference {reference_path}")
                continue
            rows = read_jsonl(records_path)
            if args.limit:
                rows = rows[: args.limit]
            entry = dict(records=str(records_path), reference=str(reference_path), n=len(rows), modes={})
            outputs = {}
            modes = ["full"] + (["cached"] if name in cached_sets else [])
            for mode in modes:
                t0 = time.perf_counter()
                results = reference_rows.run_rows(engine, rows, mode=mode, slot=0 if mode == "full" else 3)
                candidate = out_dir / f"{args.tag}_tt_{name}_{mode}.jsonl"
                write_jsonl(candidate, results)
                outputs[mode] = results
                errors = [r["id"] for r in results if "error" in r]
                parity = compare(
                    reference_path,
                    candidate,
                    f"TT {name} {mode} vs CPU bf16",
                    out_dir / f"{args.tag}_parity_{name}_{mode}",
                )
                entry["modes"][mode] = dict(
                    candidate=str(candidate),
                    seconds_total=round(time.perf_counter() - t0, 1),
                    errors=errors,
                    parity=parity,
                    records=per_record(results),
                )
                summary["gates"][f"{name}_{mode}"] = dict(rc=parity["rc"], errors=len(errors), **parity["overall"])
            if "cached" in outputs:
                entry["cached_vs_full_max_dp"] = round(cross_max_dp(outputs["full"], outputs["cached"]), 6)
            summary["sets"][name] = entry
            summary_path.write_text(json.dumps(summary, indent=2, default=str))
            log(f"{name}: {json.dumps({m: summary['gates'][f'{name}_{m}'] for m in modes}, default=str)}")
        summary["dram_free_at_end_bytes"] = mesh_dram_free_bytes(mesh)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"IMAGE_PARITY_DONE summary={summary_path}")
    failed = [k for k, v in summary["gates"].items() if v["rc"] != 0 or v["errors"]]
    log(f"gates failed: {failed}" if failed else "all gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
