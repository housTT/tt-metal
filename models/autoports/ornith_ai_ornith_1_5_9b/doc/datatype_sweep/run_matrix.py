# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialize immutable candidate runs; stop immediately on runtime failure."""

import argparse
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent
PREFIX = "models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--suffix", default="v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in args.configs:
        label = f"{name}_{'smoke_' if args.smoke else ''}{args.suffix}"
        command = [
            sys.executable,
            "-m",
            PREFIX + "record_run",
            label,
            sys.executable,
            "-m",
            PREFIX + "run_candidate",
            "--config",
            str(DOC / "configs" / f"{name}.json"),
            "--output",
            str(DOC / f"{label}.json"),
        ]
        if args.smoke:
            command.append("--smoke")
        print(label, flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
