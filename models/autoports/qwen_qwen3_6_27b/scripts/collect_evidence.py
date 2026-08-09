# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turn functional-decoder pytest run logs into ``doc/functional_decoder/pcc_evidence.json``.

The tests emit one ``PCCEVIDENCE {...}`` line per measured quantity (see
:func:`..tests.harness.record`), so the numbers behind every ``PASSED`` stay in the run log
and can be collected without re-running anything on hardware::

    python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \\
        models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/logs/*.log

``logs/*.log`` is deliberately a *flat* glob. ``logs/controls/`` holds runs that are supposed to
fail - the reverted-build control for the SDPA decode fix - and must never be folded into the
stage's evidence; keeping them one directory down is what stops the glob from picking them up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tests.harness import EVIDENCE_PREFIX

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / "pcc_evidence.json"


def parse_log(path: Path) -> list[dict]:
    records = []
    for raw in path.read_text(errors="replace").splitlines():
        index = raw.find(EVIDENCE_PREFIX)
        if index < 0:
            continue
        try:
            records.append(json.loads(raw[index + len(EVIDENCE_PREFIX) :]))
        except json.JSONDecodeError:
            continue
    return records


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="pytest run logs to scan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    by_key: dict[str, dict] = {}
    sources: list[str] = []
    for log in args.logs:
        found = parse_log(log)
        sources.append(f"{log}:{len(found)}")
        for record in found:
            key = json.dumps({k: v for k, v in record.items() if k != "value"}, sort_keys=True)
            record["source_log"] = str(log)
            # The same measurement can appear in several logs (e.g. a test that also runs under
            # watcher). Keep the *worst* numeric value rather than whichever log was listed last,
            # so the reported minimum cannot be improved by reordering the arguments. Non-numeric
            # values (the determinism booleans) fall back to last-wins, which is order-stable
            # because they are all True.
            previous = by_key.get(key)
            if previous is not None and _is_number(previous["value"]) and _is_number(record["value"]):
                record = min(previous, record, key=lambda r: r["value"])
            by_key[key] = record

    records = [by_key[k] for k in sorted(by_key)]
    numeric = [r for r in records if _is_number(r["value"])]
    # Scale ratios are a different quantity from PCC and are bounded on both sides, so they must
    # not be folded into the PCC minimum - a scale of 0.995 is excellent, a PCC of 0.995 is the
    # bar. Split them by metric name.
    pcc_records = [r for r in numeric if not r["metric"].endswith("_scale")]
    scale_records = [r for r in numeric if r["metric"].endswith("_scale")]
    payload = {
        "sources": sources,
        "num_records": len(records),
        "num_pcc_records": len(pcc_records),
        "num_scale_records": len(scale_records),
        "min_pcc": min((r["value"] for r in pcc_records), default=None),
        "min_pcc_record": min(pcc_records, key=lambda r: r["value"], default=None),
        "scale_range": (
            [min(r["value"] for r in scale_records), max(r["value"] for r in scale_records)] if scale_records else None
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(
        f"wrote {args.out}: {len(records)} records "
        f"({len(pcc_records)} PCC, {len(scale_records)} scale), min PCC {payload['min_pcc']}"
    )


if __name__ == "__main__":
    main()
