"""Record source and environment provenance for a serialized stage command."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    doc = Path(__file__).resolve().parent
    model = doc.parents[1]
    for name, value in os.environ.items():
        if ("PROFILER" in name or name.startswith("TRACY")) and value not in ("", "0"):
            raise RuntimeError(f"Serving-stage commands must not inherit {name}")
    manifest = {
        "argv": command,
        "started_unix": time.time(),
        "environment": {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("TT_METAL", "ORNITH", "HF_HUB", "OMP_NUM"))
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (model / "tt").glob("*.py")
        },
    }
    output = doc / f"{args.label}.provenance.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    with (doc / f"{args.label}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    manifest["returncode"] = result.returncode
    manifest["elapsed_s"] = time.time() - manifest["started_unix"]
    manifest["log_sha256"] = hashlib.sha256((doc / f"{args.label}.log").read_bytes()).hexdigest()
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.label}: exit {result.returncode}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
