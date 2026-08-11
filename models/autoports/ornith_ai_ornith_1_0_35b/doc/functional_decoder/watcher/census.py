# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the line-kind census and fatal-class check in CLASSIFICATION.md.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/functional_decoder/watcher/census.py

The buckets are disjoint prefix/substring rules and must sum to the file's line count, so no line
kind can go unclassified.
"""

import collections
import re
from pathlib import Path

LOG = Path(__file__).resolve().parent / "watcher_log.txt"

FATAL = re.compile(
    r"watcher.*(error|fatal|assert)|out of bounds|overflow|sanitiz|corrupt|unexpected|"
    r"invalid (noc|address|coord)|hang|deadlock|NOC_ERR|CB_ERR",
    re.IGNORECASE,
)


#: The legend block watcher prints once per device open. Listed explicitly so that an unexpected
#: line kind lands in "UNCLASSIFIED" and trips the assert instead of hiding in a catch-all.
LEGEND_PREFIXES = (
    "Legend:",
    "\tComma separated list specifies",
    "\tBRISC is main processor",
    "\tI=initialization sequence",
    "\tW=wait",
    "\tR=run",
    "\tD=done",
    "\tX=host written value",
    "\tA single character status",
    "\t\tNRW is",
    "\t\tNWD is",
    "\trmsg(BRISC host run message)",
    "\tsmsg(subordinate run message)",
    "\tk_ids: kernel IDs per processor",
)


def classify(line: str) -> str:
    if re.match(r"^Device \d", line):
        return "per-core status dump header"
    if line.startswith("k_ids:"):
        return "kernel-id continuation"
    if line.startswith("k_id["):
        return "kernel id -> source map"
    if re.match(r"^(At [0-9.]+s|Dump #)", line):
        return "dump banner"
    if "Stack usage summary" in line:
        return "stack usage summary"
    if "highest stack usage" in line:
        return "stack usage detail"
    if line.startswith("-----"):
        return "separator"
    if not line.strip():
        return "blank"
    if line.startswith(LEGEND_PREFIXES):
        return "legend row"
    return "UNCLASSIFIED"


def main():
    lines = LOG.read_text(errors="replace").splitlines()
    census = collections.Counter(classify(line) for line in lines)
    assert sum(census.values()) == len(lines), "buckets must partition the file"
    assert "UNCLASSIFIED" not in census, f"unexpected watcher line kind: {census['UNCLASSIFIED']} lines"
    width = max(len(k) for k in census)
    out = [f"{count:7d}  {kind:<{width}}" for kind, count in census.most_common()]
    out.append(f"{len(lines):7d}  TOTAL")

    headroom = [int(m) for m in re.findall(r"(\d+) bytes free", "\n".join(lines))]
    if headroom:
        out.append(f"minimum stack headroom: {min(headroom)} bytes free over {len(headroom)} detail lines")
    else:
        # Watcher only emits a stack-usage summary for dumps where firmware had recorded a watermark,
        # so a log can legitimately contain none. Say so rather than crashing on min([]) — silence
        # here would read as "no overflow" when it actually means "not measured".
        out.append("stack headroom: not reported in this log (no 'bytes free' lines)")

    fatal = [line for line in lines if FATAL.search(line)]
    out.append(f"fatal-class matches: {len(fatal)}")

    # Committed next to the log so the counts CLASSIFICATION.md quotes are a generated artifact
    # rather than a transcription, and so audit_figures.py can trace them.
    text = "\n".join(out) + "\n"
    (LOG.parent / "census_summary.txt").write_text(text)
    print(text, end="")
    assert not fatal, fatal[:5]


if __name__ == "__main__":
    main()
