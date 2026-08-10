# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Write ``doc/fused_decoder/watcher/WATCHER_AUDIT.md`` from the committed watcher artifacts.

Every quantity in that document — line count, dump count, category histogram, selected tests,
pass/deselect counts, wall time, and the offender grep — is read out of
``watcher/generated/watcher/watcher.log`` and ``logs/watcher_run.log`` here rather than typed.
A stage review caught the hand-written version describing an earlier run than the committed log;
generating it is the fix, and ``tests/test_fused_decoder_docs.py::test_watcher_audit_matches_its_artifacts``
is the gate that keeps it that way.

Reads only committed artifacts; opens no device.

    python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/make_watcher_audit.py
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]
WATCHER = DOC / "watcher"
LOG = WATCHER / "generated" / "watcher" / "watcher.log"
RUN = DOC / "logs" / "watcher_run.log"

OFFENDER = re.compile(
    r"fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected",
    re.IGNORECASE,
)
SELECTOR = (
    "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or "
    "test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable"
)


def read_log() -> str:
    """The watcher log, whether it is committed verbatim or gzipped."""
    if LOG.exists():
        return LOG.read_text(errors="replace")
    return gzip.decompress(LOG.with_suffix(".log.gz").read_bytes()).decode(errors="replace")


def main() -> None:
    lines = read_log().splitlines()
    dumps = sum(1 for line in lines if line.startswith("Dump"))
    offenders = [line for line in lines if OFFENDER.search(line) and "highest stack usage" not in line]
    if offenders:
        raise SystemExit(f"watcher log is NOT clean; {len(offenders)} offender lines, first: {offenders[0][:120]}")

    histogram: dict[str, int] = {}
    for line in lines:
        token = line.split(" ")[0] if line else ""
        histogram[token] = histogram.get(token, 0) + 1
    top = sorted(histogram.items(), key=lambda item: -item[1])[:6]

    run_text = RUN.read_text(errors="replace")
    passed, deselected, seconds = re.findall(r"(\d+) passed, (\d+) deselected, \d+ warnings in ([\d.]+)s", run_text)[-1]
    selected = sorted(set(re.findall(r"::(test_[A-Za-z_0-9]+\[[A-Za-z_0-9-]+\])", run_text)))
    selected += sorted(set(re.findall(r"::(test_bfloat8_kv_cache)\b", run_text)))

    histogram_block = "\n".join(f"{count:7d} {token}" for token, count in top)
    selected_block = "\n".join(f"  {name}" for name in selected)

    WATCHER.mkdir(parents=True, exist_ok=True)
    (WATCHER / "WATCHER_AUDIT.md").write_text(
        f"""# Watcher audit — Qwen3.6-27B fused decoder

*Generated from the committed artifacts by `../probes/make_watcher_audit.py`; every quantity
below is read out of them, not typed.*

Hardware: one Blackhole chip (device 2) of a p300c board — see `../README.md`. Run against the
final fused code on branch `agentic-research/hous/qwen3.6-27b-v2`, i.e. with
`ttnn.transformer.chunk_gated_delta_rule` on the `linear_attention` prefill path, the bfloat16
causal-conv FIR with its SiLU folded into the last tap, the batch-major decode conv tap buffers,
the width-sharded decode RMS norms, the group-reduction gated norm and the explicit recurrence
core grids.

Command, from the repo root with `ttenv.sh` sourced:

```bash
export ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
export TT_METAL_LOGS_PATH=$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \\
       TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \\
    -k "{SELECTOR}" \\
    -v -s
```

Result: **{passed} passed, {deselected} deselected** in {seconds} s. Selected tests:

```
{selected_block}
```

That is both layer kinds through paged prefill at 2049, paged decode, trace capture and replay
at batch 1 *and* batch 4, six repeated prefill+decode cycles, the BFP8 KV-cache path, and the two
alternate page block sizes through prefill *and* decode. Run log: `../logs/watcher_run.log`.

Watcher log: `generated/watcher/watcher.log` ({len(lines)} lines, {dumps} `Dump` header/footer
lines). `watcher.log` and `kernel_names.txt` are committed **gzipped** because each exceeds this
repo's 500 KB per-file commit limit; `kernel_elf_paths.txt` is under it and is committed
verbatim. The `generated/inspector/` tree the run also emits is not stage evidence and
is not committed.

## Clean-run audit

```
$ grep -niE 'fatal|assert|corrupt|sanitiz|out of bounds|overflow|invalid|exception|error|fault|hang|unexpected' \\
      generated/watcher/watcher.log | grep -v 'highest stack usage'
(no matches — grep -c returns 0)
```

Line categories present, all normal watcher bookkeeping:

```
$ awk '{{print $1}}' generated/watcher/watcher.log | sort | uniq -c | sort -rn | head -6
{histogram_block}
```

`Dump` lines delimit the periodic watcher dumps; `Device` / `k_id` / `k_ids` lines are the
per-core waypoint and active-kernel-id bookkeeping every dump emits, and the `BRISC` / `-----`
lines are the stack-usage summary block. A watcher stack overflow, NOC sanitisation failure, CB
out-of-bounds transaction or L1 overflow would each be an explicit error line; there are none.

The check is also a test: `tests/test_fused_decoder_docs.py::test_watcher_log_is_clean` re-runs
the grep over the committed log, and `::test_watcher_audit_matches_its_artifacts` re-derives the
line count, the dump count, the histogram and the pass/deselect counts above, so this audit
cannot silently describe a different run than the one committed next to it.

Watcher and the device profiler were kept in separate runs, as `$tt-device-usage` requires: the
eight Tracy runs under `../tracy/` were launched separately with no `TT_METAL_WATCHER` set.
"""
    )
    print(f"wrote {WATCHER / 'WATCHER_AUDIT.md'}: {len(lines)} lines, {dumps} dumps, {passed} passed")


if __name__ == "__main__":
    main()
