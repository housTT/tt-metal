# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Collect every PCC the shipped runs recorded into one file.

The suite prints ``PCCEVIDENCE {...}`` for each measurement it makes; the draw-sensitivity
probe prints ``DRAWS {...}`` with one PCC per decode-token seed.  Both end up here so a
reviewer can find the worst number this stage produced without re-reading four logs.

    python collect_pcc_evidence.py          # print the summary
    python collect_pcc_evidence.py --write  # rewrite pcc_evidence.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Fixed gate logs, plus *every* draw-sensitivity log by glob - a hard-coded list of those went
#: stale the moment a new candidate was swept, which is exactly what happened in review round 6.
FIXED_LOGS = ("suite_main.log", "long_context.log", "long_context_bfp8_control.log",
              "watcher_run.log")


def log_names() -> list:
    draws = sorted(p.name for p in (ROOT / "logs").glob("probe_draws_*.log"))
    return list(FIXED_LOGS) + draws


def records() -> tuple[list, list]:
    out, sources = [], []
    for name in log_names():
        path = ROOT / "logs" / name
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        found = 0
        for match in re.finditer(r"^PCCEVIDENCE (\{.*\})$", text, re.M):
            row = json.loads(match.group(1))
            if isinstance(row.get("value"), (int, float)) and not isinstance(row["value"], bool):
                row["source_log"] = f"logs/{path.name}"
                out.append(row)
                found += 1
        for match in re.finditer(r"^DRAWS (\{.*\})$", text, re.M):
            row = json.loads(match.group(1))
            for key, value in row.items():
                if not key.startswith("decode_pcc_seed"):
                    continue
                out.append({"kind": row["kind"], "metric": "real_weight_decode_pcc_by_draw",
                            "candidate": row["candidate"], "seq_len": row["seq_len"],
                            "seed": int(key[len("decode_pcc_seed"):]), "value": value,
                            "source_log": f"logs/{path.name}"})
                found += 1
        sources.append(f"logs/{path.name}:{found}")
    return out, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows, sources = records()
    worst = min(rows, key=lambda r: r["value"])
    doc = {"sources": sources, "num_records": len(rows), "min_pcc": worst["value"],
           "min_pcc_record": worst, "records": rows}
    if args.write:
        json.dump(doc, open(ROOT / "pcc_evidence.json", "w"), indent=1)
        open(ROOT / "pcc_evidence.json", "a").write("\n")
    print(json.dumps({k: doc[k] for k in ("sources", "num_records", "min_pcc", "min_pcc_record")},
                     indent=1))
    # The worst *shipped-default* real-weight number, which is the one the reports quote.
    real = [r for r in rows if r["metric"].startswith("real_weight")
            and r.get("candidate", "default") == "default"]
    if real:
        low = min(real, key=lambda r: r["value"])
        print("worst real-weight (default policy):", json.dumps(low))


if __name__ == "__main__":
    main()
