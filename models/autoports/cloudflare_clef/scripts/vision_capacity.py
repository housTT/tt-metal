"""Measure the largest image grid the vision tower and the full engine accept.

The script builds synthetic PIL images whose processor grid gives a chosen number of
patches (tokens = patches / 4 after the 2 x 2 merge), opens the 64-layer engine with the
4 snapshot slots the server uses, and for every size runs the tower alone (first and warm
call), then the full forward and the cached forward of a one-question record through
ClefEngine.probs_for_request. Per size it records pass or fail, the exception text and
the last traceback lines on a failure, the tower seconds, the padded rows and the window,
the device and end-to-end seconds, the cached-versus-full max |dp|, and the DRAM free
bytes per device before and after each step. The run stops at the first failure unless
--keep-going is given, because a failed device op leaves the mesh in an unknown state.

--vision-max-seq-len overrides the engine module constant VISION_MAX_SEQ_LEN before the
engine is built (the engine file itself is not changed), so the next limit above the
default 4096 can be measured in a second run.

Usage (device, through devrun):
  python vision_capacity.py [--patches 1024,2048,4096,6144,8192] [--step 2048] [--max-patches 16384]
                            [--vision-max-seq-len N] [--n-layers N] [--slots 4] [--tag stage2r]
                            [--out-dir /home/hous/dev/clef/reports] [--tower-only] [--keep-going]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PATCH = 16
MERGE = 2


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def grid_for(patches):
    best = None
    for h in range(2, int(math.sqrt(patches)) + 1, 2):
        if patches % h == 0 and (patches // h) % 2 == 0:
            best = h
    if best is None:
        raise ValueError(f"no even grid for {patches} patches")
    return best, patches // best


def synthetic_image(h_patches, w_patches, seed):
    import numpy as np
    from PIL import Image

    height = h_patches * PATCH
    width = w_patches * PATCH
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (255 * xx / max(width - 1, 1)),
            (255 * yy / max(height - 1, 1)),
            (127 + 127 * np.sin(xx / 23.0) * np.cos(yy / 17.0)),
        ],
        axis=-1,
    )
    noise = rng.integers(0, 48, size=base.shape)
    array = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def record_for(patches, image):
    return {
        "id": f"synthetic_{patches}",
        "model": "clef",
        "state": "A synthetic test image made of a colour gradient with noise.",
        "images": [image],
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "What does the image show?",
                "criteria": {"A": "A gradient pattern", "B": "A photograph of people", "C": "A page of text"},
            }
        },
    }


def failure(error):
    lines = traceback.format_exc().strip().splitlines()
    return dict(error=f"{type(error).__name__}: {str(error)[:1500]}", traceback_tail=lines[-12:])


def max_dp(a, b):
    return max(abs(a[qid][o] - b[qid][o]) for qid in a for o in a[qid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", default="1024,2048,4096,6144,8192")
    parser.add_argument("--step", type=int, default=2048)
    parser.add_argument("--max-patches", type=int, default=16384)
    parser.add_argument("--vision-max-seq-len", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--tag", default="stage2r")
    parser.add_argument("--out-dir", default="/home/hous/dev/clef/reports")
    parser.add_argument("--tower-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    import torch

    from models.autoports.cloudflare_clef.tt import encode as clef_encode
    from models.autoports.cloudflare_clef.tt import engine as engine_module
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, mesh_dram_free_bytes, tp2_mesh

    if args.vision_max_seq_len is not None:
        engine_module.VISION_MAX_SEQ_LEN = args.vision_max_seq_len
    sizes = [int(v) for v in args.patches.split(",") if v]
    while sizes[-1] + args.step <= args.max_patches:
        sizes.append(sizes[-1] + args.step)
    out_path = Path(args.out_dir) / f"{args.tag}_vision_capacity.json"
    summary = dict(
        vision_max_seq_len=engine_module.VISION_MAX_SEQ_LEN,
        sizes=sizes,
        slots_requested=args.slots,
        sizes_run={},
        first_failure=None,
        largest_pass_tower=None,
        largest_pass_full=None,
    )

    def save():
        out_path.write_text(json.dumps(summary, indent=2, default=str))

    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, n_layers=args.n_layers, snapshot_slots=args.slots)
        summary["engine"] = dict(
            n_layers=engine.args.n_layers,
            max_len=engine.max_len,
            snapshot_slots=engine.snapshot_slots,
            timings=engine.timings,
            vision=engine.vision.describe(),
            vision_args_max_seq_len=engine.vision.vision_args.max_seq_len,
            dram_free_after_text_weights_bytes=engine.dram_free_after_text_weights,
            dram_free_after_weights_bytes=engine.dram_free_after_weights,
            dram_free_after_slots_bytes=engine.dram_free_after_slots,
        )
        save()
        log(f"engine: {json.dumps(summary['engine'], default=str)}")
        for patches in sizes:
            h, w = grid_for(patches)
            image = synthetic_image(h, w, seed=patches)
            record = record_for(patches, image)
            entry = dict(patches=patches, grid_expected=[1, h, w], pixels=[image.width, image.height], steps={})
            summary["sizes_run"][str(patches)] = entry
            try:
                encoded = clef_encode.encode(engine.tokenizer, record, processor=engine.processor)
            except Exception as error:
                entry["steps"]["encode"] = dict(passed=False, **failure(error))
                entry["tower_passed"] = entry["full_passed"] = False
                summary["first_failure"] = summary["first_failure"] or dict(
                    patches=patches, step="encode", **entry["steps"]["encode"]
                )
                save()
                log(f"FIRST_FAILURE {patches} encode: {entry['steps']['encode']['error']}")
                if not args.keep_going:
                    break
                continue
            media = encoded.media
            grid = torch.as_tensor(media["image_grid_thw"]).reshape(-1, 3)
            entry["grid"] = grid.tolist()
            entry["n_tokens"] = int(grid.prod(dim=1).sum()) // (MERGE * MERGE)
            entry["input_tokens"] = len(encoded.input_ids)
            log(
                f"{patches} patches: grid {entry['grid']}, {entry['n_tokens']} tokens, request {entry['input_tokens']} tokens"
            )
            ok = True
            for step_name in ("tower_first", "tower_warm", "full", "cached"):
                if args.tower_only and step_name in ("full", "cached"):
                    break
                step = dict(dram_free_before_bytes=mesh_dram_free_bytes(mesh))
                entry["steps"][step_name] = step
                t0 = time.perf_counter()
                try:
                    if step_name.startswith("tower"):
                        rows = engine.vision.image_features_torch(media["pixel_values"], grid)
                        step["seconds"] = round(time.perf_counter() - t0, 3)
                        step["rows"] = list(rows.shape)
                        step["finite"] = bool(torch.isfinite(rows).all())
                        step["padded_rows"] = engine.vision.last_run.get("rows")
                        step["window"] = engine.vision.last_run.get("window")
                        step["tower_seconds"] = round(engine.vision.last_run.get("seconds", 0.0), 3)
                        del rows
                    elif step_name == "full":
                        result = engine.probs_for_request(record, mode="full", slot=0)
                        step["seconds"] = result["seconds"]
                        step["timing"] = result["timing"]
                        step["probs"] = result["probs"]
                        step["vision"] = result["vision"]
                        entry["full_probs"] = result["probs"]
                    else:
                        result = engine.probs_for_request(record, mode="cached", slot=args.slots - 1)
                        step["seconds"] = result["seconds"]
                        step["timing"] = result["timing"]
                        step["cache_hit"] = result["cache_hit"]
                        step["cached_vs_full_max_dp"] = max_dp(entry["full_probs"], result["probs"])
                    step["passed"] = True
                except Exception as error:
                    step["seconds"] = round(time.perf_counter() - t0, 3)
                    step["passed"] = False
                    step.update(failure(error))
                    ok = False
                step["dram_free_after_bytes"] = mesh_dram_free_bytes(mesh)
                log(
                    f"{patches} {step_name}: {json.dumps({k: v for k, v in step.items() if k not in ('probs', 'vision')}, default=str)}"
                )
                save()
                if not ok:
                    break
            tower_ok = entry["steps"].get("tower_first", {}).get("passed") and entry["steps"].get("tower_warm", {}).get(
                "passed"
            )
            full_ok = entry["steps"].get("full", {}).get("passed") and entry["steps"].get("cached", {}).get("passed")
            entry["tower_passed"] = bool(tower_ok)
            entry["full_passed"] = bool(full_ok)
            if tower_ok:
                summary["largest_pass_tower"] = patches
            if full_ok:
                summary["largest_pass_full"] = patches
            if not ok:
                failed_step = next(name for name, step in entry["steps"].items() if not step.get("passed"))
                summary["first_failure"] = dict(
                    patches=patches,
                    step=failed_step,
                    **{k: entry["steps"][failed_step][k] for k in ("error", "traceback_tail")},
                )
                save()
                log(f"FIRST_FAILURE {patches} {failed_step}: {entry['steps'][failed_step]['error']}")
                if not args.keep_going:
                    break
            save()
        summary["dram_free_at_end_bytes"] = mesh_dram_free_bytes(mesh)
    save()
    log(
        f"VISION_CAPACITY_DONE largest_pass_tower={summary['largest_pass_tower']} largest_pass_full={summary['largest_pass_full']} summary={out_path}"
    )


if __name__ == "__main__":
    main()
