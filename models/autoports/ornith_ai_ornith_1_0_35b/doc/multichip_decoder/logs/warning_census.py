# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every distinct runtime warning the committed suite and watcher logs contain, with counts.

Review round 5 found 864 identical `Fabric packet size ... is suboptimal` warnings on this stage's
own collectives, classified in no document, probe artifact or limitation; round 6 found the same
count still there, pointing the other way, after the setting changed. Both times the warning was
sitting in a committed log that nobody greps.

So the sweep produces this census: one line per distinct warning text, with how many times it
occurred and which log it came from. A new warning class showing up in a diff is the point — it is
the cheapest possible guard against the failure mode that produced two review findings in a row.

Warnings are matched on the runtime's own `| warning |` / `| WARNING |` log field rather than on the
word appearing anywhere, and the timestamp, log level and source-location suffix are stripped so that
two occurrences of the same warning collapse to one line.

    python .../doc/multichip_decoder/logs/warning_census.py            # writes logs/warning_census.txt
    python .../doc/multichip_decoder/logs/warning_census.py --check    # non-zero if it would change
"""

from __future__ import annotations

import collections
import gzip
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Logs to scan. Each is read as `foo` or `foo.gz`, whichever exists.
SOURCES = ["pytest_full_suite.txt", "watcher_pytest.txt"]

OUT = HERE / "warning_census.txt"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: `2026-08-15 09:09:10.268 | warning  |  Metal | <text> (file.cpp:70)`
WARNING = re.compile(r"\|\s*(?:warning|WARNING)\s*\|\s*(?P<where>[^|]*)\|\s*(?P<text>.*)")
#: Numbers inside a warning vary per call site (sizes, ids); the *class* is what a census tracks.
DIGITS = re.compile(r"(?<![\w.])\d+(?![\w.])")


def read(name: str) -> str:
    path = HERE / name
    if path.is_file():
        return path.read_text(errors="replace")
    gz = HERE / (name + ".gz")
    if gz.is_file():
        return gzip.open(gz, "rt", errors="replace").read()
    return ""


def census() -> str:
    lines = ["# distinct runtime warnings in the committed logs, by class", "# count log text"]
    total = 0
    for name in SOURCES:
        body = read(name)
        if not body:
            continue
        seen: collections.Counter = collections.Counter()
        exemplar: dict = {}
        for raw in body.splitlines():
            match = WARNING.search(ANSI.sub("", raw))
            if not match:
                continue
            text = match.group("text").strip()
            key = (match.group("where").strip(), DIGITS.sub("N", text))
            seen[key] += 1
            exemplar.setdefault(key, text)
        total += sum(seen.values())
        for key, count in seen.most_common():
            lines.append(f"WARN {count:6d} {name} {exemplar[key]}")
    lines.append(f"# {total} warning lines over {len(SOURCES)} logs")
    return "\n".join(lines) + "\n"


def main() -> int:
    body = census()
    if "--check" in sys.argv[1:]:
        current = OUT.read_text() if OUT.is_file() else ""
        if current != body:
            print(f"STALE  {OUT.name} does not match the committed logs")
            return 1
        print(f"{OUT.name} matches the committed logs")
        return 0
    OUT.write_text(body)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
