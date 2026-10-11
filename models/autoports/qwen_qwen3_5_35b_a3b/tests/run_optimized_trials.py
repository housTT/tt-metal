# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized optimized-layer candidates; never hide failed measurements."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest():
    files = sorted(ROOT.glob("tt/*.py")) + sorted(ROOT.glob("tests/*.py"))
    result = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    repo = ROOT.parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True).stdout
    result["git_head"] = git("rev-parse", "HEAD").decode().strip()
    result["git_diff_outside_autoport"] = hashlib.sha256(
        git("diff", "HEAD", "--", ".", ":(exclude)models/autoports")).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--input-activations", type=Path)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--decoder", choices=("optimized", "multichip"), default="optimized")
    p.add_argument("--mesh", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--save-tensors", action="store_true")
    p.add_argument("--compare-tensors", type=Path)
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()
    a.directory.mkdir(parents=True, exist_ok=False)
    frozen = source_manifest()
    subprocess.run(["tar", "-czf", str(a.directory / "source_snapshot.tar.gz"), "-C", str(ROOT), "tt", "tests"],
                   check=True)
    inputs = {}
    if a.input_activations:
        inputs[str(a.input_activations)] = hashlib.sha256(a.input_activations.read_bytes()).hexdigest()
    if a.compare_tensors:
        inputs[str(a.compare_tensors)] = hashlib.sha256(a.compare_tensors.read_bytes()).hexdigest()
    environment = {k: os.environ[k] for k in ("QDEV", "TT_VISIBLE_DEVICES") if k in os.environ}
    (a.directory / "provenance.json").write_text(
        json.dumps(dict(source=frozen, inputs=inputs, environment=environment), indent=2) + "\n")
    rows = []
    for name, policy in json.loads(a.matrix.read_text()).items():
        cmd = [sys.executable, "-m", "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
               "--decoder", a.decoder, "--mesh", str(a.mesh), "--layer", str(a.layer), "--length", str(a.length),
               "--steps", str(a.steps), "--batch", str(a.batch), "--trace", "--audit", "--warm-prefill",
               "--output", str(a.directory / f"{name}.json")]
        if a.save_tensors:
            cmd.extend(("--save-tensors", str(a.directory / f"{name}.pt")))
        if a.compare_tensors:
            cmd.extend(("--compare-tensors", str(a.compare_tensors)))
        if policy is not None:
            cmd.extend(("--optimization-json", json.dumps(policy)))
        if a.input_activations:
            cmd.extend(("--input-activations", str(a.input_activations)))
        if a.split:
            cmd.extend(("--split", str(a.split)))
        env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
        print("START", name, flush=True)
        start = time.monotonic()
        with (a.directory / f"{name}.log").open("w") as log:
            run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(name=name, policy=policy, command=cmd, exit_code=run.returncode,
                   elapsed_seconds=time.monotonic() - start, source_stable=source_manifest() == frozen)
        out = a.directory / f"{name}.json"
        if out.exists():
            d = json.loads(out.read_text())
            row.update(prefill_pcc=d["prefill_pcc_min"], decode_pcc=d["decode_pcc_min"],
                       prefill_host_us=d["warmed_prefill_host_us"],
                       continuation_host_us=d.get("warmed_continuation_host_us"),
                       prefill_token_pcc_min=(d.get("prefill_token_pcc") or [[None]])[0][0],
                       prefill_tokens_below_0995=d.get("prefill_tokens_below_0995"),
                       prefill_repeat_identical=d.get("prefill_repeat_identical"),
                       prefill_repeat_difference=d.get("prefill_repeat_difference"),
                       decode_host_us=statistics.median(d["traced_replay_host_us"]),
                       mesh_shape=d.get("mesh_shape"), device_ids=d.get("device_ids"),
                       output_replicas_identical=d.get("output_replicas_identical"),
                       control_equivalence=d.get("control_equivalence"),
                       weight_memory=d.get("weight_memory"),
                       construction_memory=d.get("construction_memory"))
        rows.append(row)
        (a.directory / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("END", json.dumps(row), flush=True)
        if not row["source_stable"]:
            raise SystemExit("Source changed during a frozen matrix; row is not sign-off evidence")
        if run.returncode < 0:
            raise SystemExit("Abnormal child termination: inspect hardware before another candidate")


if __name__ == "__main__":
    main()
