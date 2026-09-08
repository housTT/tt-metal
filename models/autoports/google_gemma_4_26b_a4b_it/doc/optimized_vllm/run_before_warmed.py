#!/usr/bin/env python3
"""Measure the original adapter, then restore the selected source in finally."""

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[4]
ADAPTER = ROOT / "models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py"
BASE_COMMIT = "6eb0427423392d7c6a7f87a511be892b8bf677ae"
SELECTED_SHA256 = "649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325"


def main():
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2).close()
    except urllib.error.URLError:
        pass
    else:
        raise RuntimeError("Stop the serving driver before changing adapter source")
    selected = ADAPTER.read_bytes()
    assert hashlib.sha256(selected).hexdigest() == SELECTED_SHA256, "Unexpected source; preserve current edits"
    baseline = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{ADAPTER.relative_to(ROOT)}"], cwd=ROOT)
    backup = DOC / "source_swap_backup.py"
    with backup.open("xb") as handle:
        handle.write(selected)
    ledger = {
        "baseline_commit": BASE_COMMIT,
        "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        "selected_sha256": SELECTED_SHA256,
        "backup_file": str(backup),
        "adapter_restored": False,
        "cleanup_code": "The same fixed router and worker are used on both sides of the warmed comparison.",
    }
    ledger_path = DOC / "baseline_source_swap.json"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    try:
        ADAPTER.write_bytes(baseline)
        result = subprocess.run(
            [str(ROOT / "python_env/bin/python"), str(DOC / "run_profiles.py"), "--phase", "before_warmed"],
            cwd=ROOT,
        )
        ledger["driver_exit_code"] = result.returncode
        result.check_returncode()
    finally:
        ADAPTER.write_bytes(backup.read_bytes())
        assert ADAPTER.read_bytes() == selected
        ledger["adapter_restored"] = True
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        backup.unlink()


if __name__ == "__main__":
    main()
