"""Shared path helpers for the Cloudflare/clef host-only scripts.

snapshot_dir() returns the Hugging Face snapshot directory of Cloudflare/clef. It takes
the value from --snapshot when given, else parses the STAGE0_WEIGHTS_OK line of
/home/hous/dev/clef/logs/stage0_weights.log, else asks huggingface_hub for the local
copy of the pinned revision. read_jsonl() and write_jsonl() are the one record per line
helpers used by every script.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
WEIGHTS_LOG = Path("/home/hous/dev/clef/logs/stage0_weights.log")


def snapshot_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if WEIGHTS_LOG.exists():
        match = re.search(r"STAGE0_WEIGHTS_OK\s+(\S+)", WEIGHTS_LOG.read_text(errors="replace"))
        if match:
            return Path(match.group(1))
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("Cloudflare/clef", revision=CLEF_REVISION, local_files_only=True))


def add_snapshot_to_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
