from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, snapshot_dir

DEFAULT_OUT = "/home/hous/dev/clef/reports/reference/vision_ref"
DEFAULT_TAPS = "0,12,23,26"
DEFAULT_VIDEO_FROM = "2a7ddcfe4724ee1403a6291d21347162"


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} UTC {message}", flush=True)


def load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def run_visual(visual, pixel_values, grid_thw, taps):
    import torch

    captured = {}
    hooks = [
        visual.blocks[index].register_forward_hook(
            lambda module, inputs, output, index=index: captured.__setitem__(index, output.detach().float().clone())
        )
        for index in taps
    ]
    started = time.perf_counter()
    with torch.inference_mode():
        result = visual(pixel_values.to(visual.dtype), grid_thw=grid_thw)
    seconds = time.perf_counter() - started
    for hook in hooks:
        hook.remove()
    return result.pooler_output.float().clone(), captured, seconds


def reference_for_record(row, processor, visual, taps, image_token_id, out_dir):
    import torch

    from models.autoports.cloudflare_clef.tt import encode as clef_encode

    request = {key: row[key] for key in ("id", "model", "state", "questions", "images") if key in row}
    request.setdefault("model", "clef")
    image_paths = list(request["images"])
    request["images"] = [load_image(path) for path in image_paths]
    encoded = clef_encode.encode(processor.tokenizer, request, processor=processor)
    pixel_values = encoded.media["pixel_values"]
    grid_thw = encoded.media["image_grid_thw"]
    placeholders = sum(1 for token in encoded.input_ids if token == image_token_id)
    merged, blocks, seconds = run_visual(visual, pixel_values, grid_thw, taps)
    n_tokens = int(merged.shape[0])
    if placeholders != n_tokens:
        raise RuntimeError(f"{row['id']}: {placeholders} image placeholders but {n_tokens} merged rows")
    expected_tokens = int(grid_thw.prod(dim=-1).sum().item()) // 4
    if expected_tokens != n_tokens:
        raise RuntimeError(f"{row['id']}: grid gives {expected_tokens} tokens but merged has {n_tokens}")
    payload = {
        "id": row["id"],
        "source": row.get("_source"),
        "images": image_paths,
        "image_size": [list(image.size) for image in request["images"]],
        "pixel_values": pixel_values.detach().clone(),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "image_grid_thw": grid_thw.detach().clone(),
        "n_patches": int(pixel_values.shape[0]),
        "n_tokens": n_tokens,
        "placeholders": placeholders,
        "input_tokens": len(encoded.input_ids),
        "media_token_offset": encoded.media.get("token_offset"),
        "merged": merged,
        "blocks": blocks,
        "seconds": round(seconds, 3),
    }
    target = out_dir / f"{row['id']}.pt"
    torch.save(payload, target)
    summary = {key: value for key, value in payload.items() if key not in ("pixel_values", "merged", "blocks")}
    summary["file"] = str(target)
    summary["block_shapes"] = {str(index): list(tensor.shape) for index, tensor in blocks.items()}
    summary["merged_shape"] = list(merged.shape)
    return summary


def video_reference(row, processor, visual, taps, frames, out_dir):
    import torch

    image = load_image(row["images"][0])
    frame_list = [image.rotate(step, fillcolor=(255, 255, 255)) for step in range(frames)]
    outputs = processor.video_processor(videos=[frame_list], do_sample_frames=False, return_tensors="pt")
    pixel_values = outputs["pixel_values_videos"]
    grid_thw = outputs["video_grid_thw"]
    merged, blocks, seconds = run_visual(visual, pixel_values, grid_thw, taps)
    payload = {
        "id": f"video_{row['id']}",
        "source_record": row["id"],
        "images": list(row["images"]),
        "frames": frames,
        "frame_rule": "frame k is the image rotated by k degrees with a white fill",
        "pixel_values_videos": pixel_values.detach().clone(),
        "pixel_values_shape": list(pixel_values.shape),
        "video_grid_thw": grid_thw.detach().clone(),
        "n_patches": int(pixel_values.shape[0]),
        "n_tokens": int(merged.shape[0]),
        "merged": merged,
        "blocks": blocks,
        "seconds": round(seconds, 3),
    }
    target = out_dir / f"video_{row['id']}.pt"
    torch.save(payload, target)
    summary = {k: v for k, v in payload.items() if k not in ("pixel_values_videos", "merged", "blocks")}
    summary["file"] = str(target)
    summary["video_grid_thw"] = grid_thw.tolist()
    summary["merged_shape"] = list(merged.shape)
    summary["block_shapes"] = {str(index): list(tensor.shape) for index, tensor in blocks.items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--taps", default=DEFAULT_TAPS)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--video-from", default=DEFAULT_VIDEO_FROM)
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    snapshot = snapshot_dir(args.snapshot)
    os.environ.setdefault("CLEF_MODEL", str(snapshot))
    os.environ.setdefault("HF_MODEL", str(snapshot))
    from transformers import AutoConfig

    from models.autoports.cloudflare_clef.tt import encode as clef_encode
    from models.autoports.cloudflare_clef.tt.vision import load_hf_visual, read_vision_state_dict

    taps = tuple(int(t) for t in args.taps.split(",") if t != "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(str(snapshot))
    image_token_id = int(config.image_token_id)

    started = time.perf_counter()
    processor = clef_encode.load_processor(str(snapshot))
    processor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    state_dict = read_vision_state_dict(str(snapshot))
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    visual = load_hf_visual(str(snapshot), state_dict=state_dict, attn_implementation=args.attn)
    build_seconds = time.perf_counter() - started
    inv_freq_dtype = str(visual.rotary_pos_emb.inv_freq.dtype)
    log(
        f"visual ready: {len(state_dict)} tensors read in {read_seconds:.1f}s, module built and loaded in "
        f"{build_seconds:.1f}s, dtype {visual.dtype}, attn {visual.config._attn_implementation}, inv_freq {inv_freq_dtype}"
    )

    index = {
        "snapshot": str(snapshot),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "threads": args.threads,
        "visual_dtype": str(visual.dtype),
        "attn_implementation": visual.config._attn_implementation,
        "inv_freq_dtype": inv_freq_dtype,
        "taps": list(taps),
        "processor_seconds": round(processor_seconds, 2),
        "state_dict_read_seconds": round(read_seconds, 2),
        "visual_build_seconds": round(build_seconds, 2),
        "image_token_id": image_token_id,
        "records": [],
        "video": None,
        "command": " ".join(sys.argv),
    }
    index_path = out_dir / "index.json"
    total = 0.0
    video_row = None
    for records_path in args.records:
        rows = read_jsonl(records_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        for row in rows:
            summary = reference_for_record(row, processor, visual, taps, image_token_id, out_dir)
            summary["records_file"] = str(records_path)
            index["records"].append(summary)
            total += summary["seconds"]
            log(
                f"{row['id']} grid={summary['image_grid_thw'].tolist()} patches={summary['n_patches']} "
                f"tokens={summary['n_tokens']} placeholders={summary['placeholders']} seconds={summary['seconds']}"
            )
            if row["id"] == args.video_from:
                video_row = row
            index_path.write_text(json.dumps(index, indent=2, default=str) + "\n")
    if video_row is not None and args.video_frames > 0:
        index["video"] = video_reference(video_row, processor, visual, taps, args.video_frames, out_dir)
        log(
            f"video from {video_row['id']}: frames={args.video_frames} grid={index['video']['video_grid_thw']} "
            f"patches={index['video']['n_patches']} tokens={index['video']['n_tokens']} seconds={index['video']['seconds']}"
        )
    index["total_forward_seconds"] = round(total, 2)
    index_path.write_text(json.dumps(index, indent=2, default=str) + "\n")
    log(f"VISION_REFERENCE_DONE records={len(index['records'])} forward_seconds={total:.1f} -> {index_path}")


if __name__ == "__main__":
    main()
