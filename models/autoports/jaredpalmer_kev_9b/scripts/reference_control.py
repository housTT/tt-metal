import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

REF = Path("/home/hous/dev/kev/reports/reference")
OUT = Path("/home/hous/dev/kev/reports")
MODES = ("merged", "base")
DEVICE_PARAMS = dict(l1_small_size=24576, num_command_queues=2, trace_region_size=0)
WORST = 5


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


def args_cls_for(mode):
    if mode == "merged":
        from models.autoports.jaredpalmer_kev_9b.tt.loader import KevModelArgs

        return KevModelArgs
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    return Qwen36ModelArgs


def score_rows(engine, head, rows, ref_hidden, ref_records):
    out = []
    for row in rows:
        ids = torch.tensor(row["ids"], dtype=torch.long).unsqueeze(0)
        positions = list(row["opt_positions"]) + [row["decide_position"]]
        labels = [f"opt{i}" for i in range(len(row["opt_positions"]))] + ["decide"]
        t0 = time.perf_counter()
        tt = engine.prefill_hidden(ids, positions)
        dt = time.perf_counter() - t0
        ref = ref_hidden[row["row_key"]].float()
        pos_pcc = [pcc(ref[j], tt[j]) for j in range(len(positions))]
        probs = head.probs(tt[-1], tt[:-1]).float()
        rq = ref_records[str(row["record"])]["questions"][row["qid"]]["probabilities"]
        rp = torch.tensor([rq[k] for k in row["keys"]], dtype=torch.float32)
        dp = (probs - rp).abs()
        rec = {
            "row_key": row["row_key"],
            "type": row["type"],
            "row_tokens": row["row_tokens"],
            "positions": positions,
            "position_labels": labels,
            "position_pcc": pos_pcc,
            "min_pcc": min(pos_pcc),
            "argmax_tt": int(probs.argmax()),
            "argmax_ref": int(rp.argmax()),
            "argmax_agree": bool(probs.argmax() == rp.argmax()),
            "max_dp": dp.max().item(),
            "mean_entry_dp": dp.mean().item(),
            "probs_tt": probs.tolist(),
            "probs_ref": rp.tolist(),
            "seconds": dt,
        }
        out.append(rec)
        logger.info(
            f"row {rec['row_key']} {rec['type']} T={rec['row_tokens']} n={len(positions)} {dt:.2f}s "
            f"min_pcc={rec['min_pcc']:.6f} argmax tt={rec['argmax_tt']} ref={rec['argmax_ref']} max_dp={rec['max_dp']:.6f}"
        )
    return out


def summarize_rows(rows):
    mins = [r["min_pcc"] for r in rows]
    max_dps = [r["max_dp"] for r in rows]
    worst = sorted(rows, key=lambda r: r["min_pcc"])[:WORST]
    return {
        "rows": len(rows),
        "min_pcc": min(mins),
        "median_row_min_pcc": statistics.median(mins),
        "max_row_min_pcc": max(mins),
        "rows_below_0_99": sum(m < 0.99 for m in mins),
        "rows_below_0_97": sum(m < 0.97 for m in mins),
        "argmax_agree": sum(r["argmax_agree"] for r in rows),
        "argmax_flips": [r["row_key"] for r in rows if not r["argmax_agree"]],
        "max_dp": max(max_dps),
        "max_dp_row": rows[max_dps.index(max(max_dps))]["row_key"],
        "mean_dp_row_max": statistics.mean(max_dps),
        "mean_dp_entries": statistics.mean(r["mean_entry_dp"] for r in rows),
        "worst_rows": [
            {
                "row_key": r["row_key"],
                "type": r["type"],
                "row_tokens": r["row_tokens"],
                "min_pcc": r["min_pcc"],
                "per_position": dict(zip(r["position_labels"], r["position_pcc"])),
            }
            for r in worst
        ],
    }


def run(mode, n_layers, device_id):
    import ttnn
    from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine
    from models.autoports.jaredpalmer_kev_9b.tt.head import PointerHead

    rows = json.load(open(REF / "rows.json"))["rows"]
    ref_hidden = torch.load(REF / "hidden_fp32.pt")
    ref_records = json.load(open(REF / "probs_fp32.json"))["records"]
    head = PointerHead(os.environ["KEV_RUN"])
    device = ttnn.open_device(device_id=device_id, **DEVICE_PARAMS)
    device.enable_program_cache()
    try:
        t0 = time.perf_counter()
        engine = KevEngine(device, args_cls=args_cls_for(mode), max_state_len=8192, n_layers=n_layers)
        t_load = time.perf_counter() - t0
        args = engine.args
        info = {
            "mode": mode,
            "args_cls": type(args).__name__,
            "n_layers": args.n_layers,
            "weight_cache_path": str(args.weight_cache_path()),
            "adapter_sha8": getattr(args, "adapter_sha8", None),
            "hf_model": os.environ.get("HF_MODEL"),
            "kev_run": os.environ.get("KEV_RUN"),
            "device_id": device_id,
            "device_params": DEVICE_PARAMS,
            "engine_load_seconds": t_load,
        }
        logger.info(f"engine {info}")
        results = score_rows(engine, head, rows, ref_hidden, ref_records)
        summary = summarize_rows(results)
        logger.info(f"{mode}: {json.dumps({k: v for k, v in summary.items() if k != 'worst_rows'})}")
        for w in summary["worst_rows"]:
            logger.info(f"{mode} worst {w['row_key']} min_pcc={w['min_pcc']:.6f} per_position={w['per_position']}")
        path = OUT / f"stage1_control_{mode}.json"
        json.dump({"info": info, "summary": summary, "rows": results}, open(path, "w"), indent=1)
        logger.info(f"wrote {path}")
    finally:
        ttnn.close_device(device)


def table(runs):
    lines = [
        "| weights | args class | min row PCC | median row PCC | rows < 0.99 | argmax agree | max dp | mean dp (row max) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        s, i = runs[mode]["summary"], runs[mode]["info"]
        lines.append(
            f"| {mode} | {i['args_cls']} | {s['min_pcc']:.4f} | {s['median_row_min_pcc']:.4f} | {s['rows_below_0_99']}/{s['rows']} "
            f"| {s['argmax_agree']}/{s['rows']} | {s['max_dp']:.4f} | {s['mean_dp_row_max']:.4f} |"
        )
    lines.append("")
    for mode in MODES:
        lines.append(f"{mode}: 5 worst rows, per-position hidden PCC (options in order, then decide)")
        for w in runs[mode]["summary"]["worst_rows"]:
            per = " ".join(f"{k}={v:.4f}" for k, v in w["per_position"].items())
            lines.append(f"- {w['row_key']} ({w['type']}, T={w['row_tokens']}): {per}")
        lines.append("")
    return "\n".join(lines)


def summarize():
    runs = {}
    for mode in MODES:
        path = OUT / f"stage1_control_{mode}.json"
        runs[mode] = json.load(open(path))
    merged_rows = {r["row_key"]: r for r in runs["merged"]["rows"]}
    base_rows = {r["row_key"]: r for r in runs["base"]["rows"]}
    per_row = [
        {
            "row_key": k,
            "type": merged_rows[k]["type"],
            "row_tokens": merged_rows[k]["row_tokens"],
            "merged_min_pcc": merged_rows[k]["min_pcc"],
            "base_min_pcc": base_rows[k]["min_pcc"],
            "merged_argmax_agree": merged_rows[k]["argmax_agree"],
            "base_argmax_agree": base_rows[k]["argmax_agree"],
            "merged_max_dp": merged_rows[k]["max_dp"],
            "base_max_dp": base_rows[k]["max_dp"],
        }
        for k in merged_rows
    ]
    md = table(runs)
    out = {
        "description": "29 reference rows through KevEngine.prefill_hidden (full row, 32 layers) against the HF fp32 kev reference; merged = LoRA merged into the base weights, base = unmerged Qwen3.5-9B-Base through the same device path and the same pointer head",
        "reference": {
            "rows": str(REF / "rows.json"),
            "hidden": str(REF / "hidden_fp32.pt"),
            "probs": str(REF / "probs_fp32.json"),
        },
        "runs": {mode: {"info": runs[mode]["info"], "summary": runs[mode]["summary"]} for mode in MODES},
        "per_row": per_row,
        "markdown": md,
    }
    path = OUT / "stage1_control.json"
    json.dump(out, open(path, "w"), indent=1)
    print(md)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES + ("summarize",), required=True)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--device-id", type=int, default=0)
    a = p.parse_args()
    if a.mode == "summarize":
        summarize()
    else:
        run(a.mode, a.n_layers, a.device_id)


if __name__ == "__main__":
    main()
