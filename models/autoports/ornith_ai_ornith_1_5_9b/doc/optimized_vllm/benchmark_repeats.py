"""Archive one warmup and three identical shared-runner serving benchmarks."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    args = parser.parse_args()
    doc = Path(__file__).resolve().parent
    model = doc.parents[1]
    root = doc / args.label
    root.mkdir(exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "models.common.readiness_check.run_vllm_server",
        "--stages",
        "benchmark",
        "--server-url",
        "http://localhost:8000",
        "--model-dir",
        str(model),
        "--hf-model",
        "/home/hous/dev/ornith-1.5-9b/upstream",
        "--mesh-device",
        "P150x4",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        "262144",
        "--additional-benchmark-args=--save-detailed",
    ]
    if args.max_num_seqs == 1:
        command.append("--no-benchmark-ci-serving")
    for name in ("warmup", "repeat1", "repeat2", "repeat3"):
        target = root / name
        target.mkdir()
        with (target / "runner.log").open("w") as log:
            result = subprocess.run(command, env=dict(os.environ, USER="hous"), stdout=log, stderr=subprocess.STDOUT)
        manifest = {"argv": command, "returncode": result.returncode, "warmup": name == "warmup", "artifacts": {}}
        for path in (model / "readiness_vllm").glob("vllm_*"):
            if "benchmark" in path.name or "result" in path.name:
                if args.max_num_seqs == 1 and "ci_serving" in path.name:
                    continue
                shutil.copyfile(path, target / path.name)
                manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{args.label}/{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
