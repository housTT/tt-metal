"""Run Cloudflare/clef records through the TT engine and write rows in the CPU reference format.

Each output row carries "id", the SystemOne response fields ("model", "answers", "usage"),
"probs" (question id -> option id -> probability, unrounded, encoded option order),
"input_tokens", "seconds", the engine "mode" ("full" or "cached"), "cache_hit", and "_label"
when the input row has one. A record that raises writes an "error" row. The file is the
candidate input of parity_compare.py.

Usage (device, through devrun):
  python reference_rows.py --records R.jsonl --output OUT.jsonl [--mode full|cached] [--n-layers N] [--limit N]
  --records/--output may repeat (pairs in order); one engine serves every pair. --mode may be a
  comma list (full,cached): each mode then writes OUT with _<mode> inserted before .jsonl.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_paths import read_jsonl, write_jsonl

REQUEST_KEYS = ("id", "model", "state", "questions", "images", "videos", "media_kwargs")


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), message, flush=True)


def request_of(row):
    request = {key: row[key] for key in REQUEST_KEYS if key in row}
    request.setdefault("model", "clef")
    return request


def run_rows(engine, rows, mode="full", slot=0):
    out = []
    for index, row in enumerate(rows):
        record_id = row.get("id", f"row{index}")
        result = {"id": record_id}
        if "_label" in row:
            result["_label"] = row["_label"]
        started = time.perf_counter()
        try:
            result.update(engine.probs_for_request(request_of(row), mode=mode, slot=slot))
            log(
                f"{index + 1}/{len(rows)} {record_id} tokens={result['input_tokens']} {result['seconds']:.2f}s "
                f"mode={mode} hit={result['cache_hit']} device={result['timing']['device_s']:.2f}s"
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["seconds"] = round(time.perf_counter() - started, 3)
            log(f"{index + 1}/{len(rows)} {record_id} FAILED: {result['error']}")
        out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-state-len", type=int, default=16384)
    args = parser.parse_args()
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine, tp2_mesh

    modes = args.mode.split(",")
    for mode in modes:
        if mode not in ("full", "cached"):
            raise SystemExit(f"--mode {mode!r}; expected full or cached")
    if len(args.records) != len(args.output):
        raise SystemExit("--records and --output must be given the same number of times")
    with tp2_mesh() as mesh:
        engine = ClefEngine(mesh, max_state_len=args.max_state_len, n_layers=args.n_layers)
        for records_path, output in zip(args.records, args.output):
            rows = read_jsonl(records_path)
            if args.limit:
                rows = rows[: args.limit]
            for mode in modes:
                path = output if len(modes) == 1 else output.replace(".jsonl", f"_{mode}.jsonl")
                results = run_rows(engine, rows, mode=mode, slot=0 if mode == "full" else 1)
                write_jsonl(path, results)
                seconds = [r["seconds"] for r in results if "error" not in r]
                log(
                    f"SUMMARY records={len(rows)} ok={len(seconds)} failed={len(rows) - len(seconds)} mode={mode} "
                    f"median_s={sorted(seconds)[len(seconds) // 2] if seconds else None} "
                    f"max_s={max(seconds) if seconds else None} output={path}"
                )
    log("REFERENCE_ROWS_DONE")


if __name__ == "__main__":
    main()
