# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract every logged metric from a fused-decoder pytest log into a flat summary.

Keeps the README tables and the raw log in sync — the tables are transcribed from
``pcc_summary.txt``, which is generated from the same log that is committed.

Usage::

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/summarise_pcc.py \
        models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/pytest_full_suite.txt
"""

import re
import sys
from pathlib import Path

# Every INFO line the test module logs is a metric line; keep them all rather than an allow-list,
# so a new test's evidence cannot silently miss the summary.
KEEP = re.compile(r"test_fused_decoder:(?P<test>[a-z_]+):\d+ - (?P<msg>.*)$")

# The decoder itself logs one measurement: which prefill block lengths ttnn.conv1d actually accepted
# at the allocated batch. That is a property of the device, not of a test, and it changes with the
# batch, so it is kept too - deduplicated, because allocate_state runs once per decoder built.
DEVICE_METRIC = re.compile(r"fused_decoder:allocate_state:\d+ - (?P<msg>.*conv1d accepted.*)$")


def main(log_path: Path):
    lines, seen_device = [], set()
    for raw in log_path.read_text(errors="replace").splitlines():
        m = KEEP.search(raw)
        if m:
            lines.append(f"{m.group('test')}: {m.group('msg')}")
            continue
        m = DEVICE_METRIC.search(raw)
        if m and m.group("msg") not in seen_device:
            seen_device.add(m.group("msg"))
            lines.append(f"conv1d_coverage: {m.group('msg')}")
    summary = log_path.with_name("pcc_summary.txt")
    tail = [ln for ln in log_path.read_text(errors="replace").splitlines() if re.match(r"=+ .*(passed|failed)", ln)]
    summary.write_text("\n".join(lines + [""] + tail) + "\n")
    print(f"wrote {summary} ({len(lines)} metric lines)")
    for ln in tail:
        print(ln)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
