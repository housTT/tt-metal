# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every latency quoted in the prose must exist in an artifact.

``make_tables.py --check`` owns the numbers inside ``<!-- TABLE: -->`` markers. Nothing owned the
numbers outside them, and three review rounds of this stage were spent on exactly that: prose
figures left behind by a re-run of the probe that generated them, sitting next to a generated table
that disagreed.

This scans README.md and work_log.md for anything shaped like a latency (``12.3 us``, ``0.565 ms``)
outside the table markers and fails if it cannot find that literal in

* **this stage's own** committed artifacts (``logs/*.txt``, ``logs/*.json``, gzipped logs, Tracy
  tables) — deliberately not the earlier stages', because a substring match over three stages of
  profiler CSV will match almost any three-digit number and the guard then proves nothing;
* the generated tables of these same two documents, which ``make_tables.py --check`` owns and where
  derived per-step figures legitimately live;
* :data:`INHERITED_FIGURES`, an explicit allowlist naming the earlier-stage artifact each inherited
  figure comes from.

It is deliberately literal, and its limits are worth stating so nobody over-trusts a green run: it
proves a *decimal* figure carrying a time unit was produced by some run of this stage, not that it
was the right run, not that it was interpreted correctly, and not that integer or unit-less figures
("3 us/step", "1.3 %", a diverged-round count) are right. Those belong in generated tables — which is
where this stage moved them after review round 5 found four of them wrong at once.

    python .../logs/check_prose_figures.py
"""

from __future__ import annotations

import glob
import gzip
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent
MODEL = DOC.parent.parent
#: Everything scanned. Review round 5 found all four of its numeric defects outside README/work_log:
#: two in the context contract and two in the implementation's own docstrings, which the first
#: version of this guard did not look at.
DOCUMENTS = ("README.md", "work_log.md")
EXTRA_SOURCES = (
    Path("..") / "context_contract.json",
    Path("..") / ".." / "tt" / "multichip_decoder.py",
)
#: Figures this stage legitimately quotes from an EARLIER stage's artifacts. Listed explicitly, with
#: the file they come from, rather than by widening the haystack to three stages' worth of CSV —
#: review round 4 pointed out that a bare substring match over megabytes of profiler output will
#: match almost any three-digit number, so the guard passed on figures no run of this stage produced.
INHERITED_FIGURES = {
    "538.1 us": "doc/multichip_decoder/tracy/linear_attention/decode_perf_report_stacked.csv.gz (sum/32)",
    "40.78 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "49.37 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "42.78 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "185.41 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "149.37 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "293.02 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "263.44 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "50.77 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "43.55 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "204.45 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "146.70 us": "doc/multichip_decoder/logs/probe_fused_ccl.txt",
    "86.47 us": "doc/multichip_decoder/logs/probe_ccl.txt (CCLBF8 decode_b32 all_reduce_ring trace)",
    "152.91 us": "doc/multichip_decoder/logs/probe_ccl.txt (CCLBF8 prefill_2048 all_reduce_ring trace)",
    "62.5 us": "doc/multichip_decoder/logs/probe_sparse_matmul_local.txt (SPARSELBEST gate_up)",
    "70.1 us": "doc/multichip_decoder/logs/probe_sparse_matmul_local.txt",
    "48.4 us": "doc/multichip_decoder/tracy/full_attention/decode_perf_report_stacked.csv.gz (TopK sum/32)",
    "13.8 us": "this stage's own first-iteration profile, quoted in README section 2.3 as history",
}
LATENCY = re.compile(r"\b(\d+\.\d{1,3})\s*(ms|us|µs)\b")
TABLE_SPAN = re.compile(r"<!-- TABLE:.*?-->.*?<!-- /TABLE:.*?-->", re.S)


def _text(path: Path) -> str:
    try:
        if path.suffix == ".gz":
            return gzip.decompress(path.read_bytes()).decode(errors="ignore")
        return path.read_text(errors="ignore")
    except Exception:  # noqa: BLE001 - an unreadable artifact simply contributes nothing
        return ""


def haystack() -> str:
    parts = []
    root = MODEL / "doc" / DOC.name
    # Deliberately NOT the Tracy per-op CSVs: a substring match over megabytes of profiler output
    # will hit almost any three-digit number, which is what made the first version of this guard
    # pass on figures no run produced. The human-readable tables and the probe logs are enough.
    for pattern in (
        "logs/*.txt",
        "logs/*.json",
        "logs/*.txt.gz",
        "tracy/*/*_perf_report.txt",
        "tracy/*/*_perf_report.txt.gz",
        "tracy/*/*summary*",
        "tracy/*/*_tracy_run.txt",
    ):
        parts.extend(_text(Path(p)) for p in glob.glob(str(root / pattern)))
    # The generated tables of this stage's own documents: derived per-step figures live there and are
    # checked by make_tables.py --check.
    for name in DOCUMENTS:
        parts.extend(TABLE_SPAN.findall((DOC / name).read_text()))
    return "\n".join(parts)


def main() -> int:
    hay = haystack()
    missing: dict[str, set[str]] = {}
    scanned = [(name, TABLE_SPAN.sub("", (DOC / name).read_text())) for name in DOCUMENTS]
    for extra in EXTRA_SOURCES:
        path = (DOC / extra).resolve()
        # A missing entry is a hard failure, not a skip. Review round 6 found the first version of
        # this list carrying one `..` too many, so it silently scanned nothing for the source file
        # whose docstrings round 5 had just found wrong.
        if not path.exists():
            print(f"EXTRA_SOURCES entry does not resolve: {extra} -> {path}")
            return 1
        scanned.append((str(extra), path.read_text(errors="ignore")))
    for name, prose in scanned:
        for match in LATENCY.finditer(prose):
            value = match.group(1)
            if match.group(0) in INHERITED_FIGURES:
                continue
            spellings = {value, value.rstrip("0").rstrip("."), value + "0", value.lstrip("0")}
            if any(s and s in hay for s in spellings):
                continue
            missing.setdefault(name, set()).add(match.group(0))
    for name, values in sorted(missing.items()):
        print(f"{name}: quoted but in no artifact: {sorted(values)}")
    if missing:
        return 1
    print("prose-figure check: every quoted latency appears in a committed artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
