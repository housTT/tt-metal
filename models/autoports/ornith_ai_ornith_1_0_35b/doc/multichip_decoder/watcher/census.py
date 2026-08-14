# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the line-kind census and fatal-class check in CLASSIFICATION.md.

Copied unchanged from the optimized stage's ``watcher/census.py`` except for this note and the
docstring path: it resolves its log relative to its own directory, so the same code classifies this
stage's log. The bucket rules are the same because watcher's line kinds are the same; a **new** line
kind (an Ethernet-related one this stage could produce that the single-chip stage could not) lands in
``UNCLASSIFIED`` and trips the assert rather than hiding in a catch-all.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/watcher/census.py

The buckets are disjoint prefix/substring rules and must sum to the file's line count, so no line
kind can go unclassified.
"""

import collections
import gzip
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
#: The log is committed gzipped: it is ~4 MB and the repo blocks files over 500 KB.
LOG = _HERE / "watcher_log.txt"
if not LOG.is_file():
    LOG = _HERE / "watcher_log.txt.gz"

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


#: The core a stack-usage detail line belongs to, e.g. "on core 14-3".
CORE = re.compile(r"on core \d+-\d+")


def classify(line: str) -> str:
    if re.match(r"^Device \d", line):
        # One line per core per dump, carrying that core's status string - not one header per dump. Review
        # round 11 pointed out the old label ("per-core status dump header") read as the latter.
        return "per-core status row"
    if line.startswith("k_ids:"):
        return "kernel-id continuation"
    if line.startswith("k_id["):
        return "kernel id -> source map"
    if re.match(r"^Dump #", line):
        return "dump banner"
    if re.match(r"^At [0-9.]+s", line):
        # A per-dump *timestamp* line, several per dump; bucketed with the banners before round 11, which
        # made the banner count read as a dump count. They are separated so `dump banner` is the dump count.
        return "dump timestamp"
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
    raw = gzip.decompress(LOG.read_bytes()) if LOG.suffix == ".gz" else LOG.read_bytes()
    # rstrip: the repo's trailing-whitespace pre-commit hook rewrites the committed log, and several
    # of watcher's line kinds end in spaces. Bucketing the stripped line makes the census reproduce
    # the same numbers before and after that hook runs.
    lines = [line.rstrip() for line in raw.decode(errors="replace").splitlines()]
    census = collections.Counter(classify(line) for line in lines)
    assert sum(census.values()) == len(lines), "buckets must partition the file"
    assert "UNCLASSIFIED" not in census, f"unexpected watcher line kind: {census['UNCLASSIFIED']} lines"
    # No right-padding on the label: this file is committed, and the repo's trailing-whitespace
    # pre-commit hook would rewrite every padded line on every commit.
    out = [f"{count:7d}  {kind}" for kind, count in census.most_common()]
    out.append(f"{len(lines):7d}  TOTAL")

    headroom = [int(m) for m in re.findall(r"(\d+) bytes free", "\n".join(lines))]
    if headroom:
        out.append(f"minimum stack headroom: {min(headroom)} bytes free over {len(headroom)} detail lines")
    else:
        # Watcher only emits a stack-usage summary for dumps where firmware had recorded a watermark,
        # so a log can legitimately contain none. Say so rather than crashing on min([]) — silence
        # here would read as "no overflow" when it actually means "not measured".
        out.append("stack headroom: not reported in this log (no 'bytes free' lines)")

    # Three labelled counts README §8 needs, because every one of them has been inferred wrongly from another:
    # round 11 found the sentence quoting a detail-line count as a dump count, and round 12 found the
    # replacement quoting the same count as a processor count and asserting one reporting core where the log
    # has two. Each is now counted directly and named.
    out.append(f"dumps: {sum(1 for line in lines if re.match(r'^Dump #[0-9]+ at', line))}")
    summaries = [line for line in lines if "Stack usage summary" in line]
    details = [line for line in lines if "highest stack usage" in line]
    out.append(f"stack summaries: {len(summaries)}")
    out.append(f"stack processors per summary: {len(details) // len(summaries) if summaries else 0}")
    out.append(f"stack reporting cores: {len({m.group(0) for m in (CORE.search(d) for d in details) if m})}")

    fatal = [line for line in lines if FATAL.search(line)]
    out.append(f"fatal-class matches: {len(fatal)}")

    # Committed next to the log so the counts CLASSIFICATION.md quotes are a generated artifact
    # rather than a transcription, and so audit_figures.py can trace them.
    text = "\n".join(out) + "\n"
    unknown = [a for a in sys.argv[1:] if a != "--check"]
    if unknown:
        raise SystemExit(f"unknown argument(s): {unknown}; this script takes only --check")
    # `--check` verifies the committed summary; anything else regenerates it. The comparison happens
    # **before** any write, which is the whole content of the flag: round 9 added `--check` because the
    # script accepted and ignored it, and round 10 found that fix comparing the file against itself - it
    # wrote `census_summary.txt` unconditionally on the line above, so `--check` reported success on a
    # summary that had been replaced by the word CORRUPTED. Fourth "gate that cannot fail" in this stage
    # (check_freshness twice, the ignored flag, this), and the pattern in all four is the same: the check
    # ran after the thing it was checking had already been overwritten or excused.
    summary = LOG.parent / "census_summary.txt"
    if "--check" in sys.argv[1:]:
        if not summary.is_file() or summary.read_text() != text:
            print("census_summary.txt does not match what this script produces")
            raise SystemExit(1)
        print("census_summary.txt matches the artifacts")
    else:
        # Committed next to the log so the counts CLASSIFICATION.md quotes are a generated artifact
        # rather than a transcription, and so audit_figures.py can trace them.
        summary.write_text(text)
        print(text, end="")
    assert not fatal, fatal[:5]


if __name__ == "__main__":
    main()
