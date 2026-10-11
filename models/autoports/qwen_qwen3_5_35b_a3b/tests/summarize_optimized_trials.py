# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build a compact, reproducible index of every measured optimization trial."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("*/commands.json")):
        for trial in json.loads(path.read_text()):
            if "policy" not in trial:
                continue
            cmd = trial["command"]
            def option(name, default=None):
                return cmd[cmd.index(name)+1] if name in cmd else default
            row = dict(matrix=path.parent.name, name=trial["name"],
                       layer=int(option("--layer")), input_tokens=int(option("--length")),
                       decode_steps=int(option("--steps")), batch=1, concurrency=1,
                       input_source="recorded_hf" if "--input-activations" in cmd else "random",
                       exit_code=trial["exit_code"], prefill_pcc=trial.get("prefill_pcc"),
                       decode_pcc=trial.get("decode_pcc"),
                       warmed_prefill_host_us=trial.get("prefill_host_us"),
                       traced_decode_host_us=trial.get("decode_host_us"),
                       policy=json.dumps(trial["policy"],sort_keys=True,separators=(",",":")),
                       evidence=str(path.relative_to(args.directory)),failure="")
            if trial["exit_code"]:
                log = path.parent/(trial["name"]+".log")
                lines = log.read_text(errors="replace").splitlines() if log.exists() else []
                row["failure"] = next((line for line in reversed(lines)
                    if line.startswith(("RuntimeError:","AssertionError:","ValueError:","TypeError:","AttributeError:"))),
                    "See trial log")[:500]
            rows.append(row)
    if not rows:
        raise ValueError("No trial artifacts found")
    with (args.directory/"trial_results.csv").open("w",newline="") as output:
        writer = csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} trials, {sum(r['exit_code']!=0 for r in rows)} failed attempts")


if __name__ == "__main__":
    main()
