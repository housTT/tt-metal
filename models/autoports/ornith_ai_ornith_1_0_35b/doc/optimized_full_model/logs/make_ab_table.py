# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turn logs/ab_terminal.txt(.gz)'s ARM_JSON lines into logs/ab_terminal_table.md (README §3)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

LOGS = Path(__file__).resolve().parent
COLS = [
    "arm", "lm_head_program", "lm_head_cores", "lm_head_dtype", "terminal_norm_sharded",
    "vocab_align_tiles", "padded_vocab_size", "topk_groups", "model_trace", "sampling_trace",
    "token_out_serial", "token_out_pipelined", "final_norm", "final_norm_head",
]


def lines():
    plain, gz = LOGS / "ab_terminal.txt", LOGS / "ab_terminal.txt.gz"
    if plain.exists():
        return plain.read_text().splitlines()
    return gzip.decompress(gz.read_bytes()).decode().splitlines()


def cell(row, col):
    value = row.get(col)
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return (
        str(value).replace("DataType.BFLOAT", "bf").replace("_B", "").replace("True", "yes").replace("False", "no")
    )


def main():
    rows = [json.loads(line[len("ARM_JSON "):]) for line in lines() if line.startswith("ARM_JSON ")]
    out = ["| " + " | ".join(c.replace("_", " ") for c in COLS) + " |", "|" + "|".join("---" for _ in COLS) + "|"]
    out += ["| " + " | ".join(cell(r, c) for c in COLS) + " |" for r in rows]
    (LOGS / "ab_terminal_table.md").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} arm(s) -> {LOGS / 'ab_terminal_table.md'}")


if __name__ == "__main__":
    main()
