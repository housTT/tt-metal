# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve compact logs and hash checkpoint versus workspace-only evidence."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

DOC = Path(__file__).resolve().parent
REPO = DOC.parents[4]


def main():
    candidates = list((DOC / "logs").glob("*.log")) + list((DOC / "triage").glob("*.txt"))
    candidates += list((DOC / "tracy").rglob("*_device*_window.csv"))
    candidates += [path for path in DOC.glob("*.json") if path.stat().st_size > 500_000]
    for path in candidates:
        meta = path.with_suffix(".provenance.json")
        if meta.exists() and "exit_code" not in json.loads(meta.read_text()):
            continue
        path.with_name(path.name + ".gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        if path.suffix == ".json" and path.name.startswith("prefill_integration_full32_"):
            # Per-rank recurrent-state hashes make this exact report large.
            # Keep complete independently readable lanes below the repo file cap.
            payload = json.loads(path.read_text())
            destination = path.with_suffix("")
            destination.mkdir(exist_ok=True)
            payload["lane_artifacts"] = {}
            for name, lane in payload.pop("lanes").items():
                target = destination / f"{name}.json.gz"
                target.write_bytes(gzip.compress((json.dumps(lane, indent=2) + "\n").encode(), mtime=0))
                assert target.stat().st_size < 500_000, target
                payload["lane_artifacts"][name] = target.name
            payload["complete_workspace_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (destination / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    files = sorted(
        p for p in DOC.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "artifact_manifest.json"
    )
    relative = [str(p.relative_to(REPO)) for p in files]
    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin"], input="\n".join(relative), text=True, capture_output=True, cwd=REPO
    )
    assert ignored.returncode in (0, 1), ignored.stderr
    ignored = set(ignored.stdout.splitlines())
    entries = []
    for path, name in zip(files, relative):
        entries.append(
            dict(
                path=name,
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                storage="workspace_only" if name in ignored else "checkpoint",
            )
        )
    model = DOC.parents[1]
    sources = [
        model / "tt/model.py",
        model / "tt/generator.py",
        model / "doc/context_contract.json",
        model / "doc/full_model/test_generator_host_contract.py",
    ]
    result = dict(
        repository=str(REPO),
        hardware="four Blackhole chips on two P300c boards",
        storage_note="Large raw profiler captures, full ops CSV and saved tensors remain at exact paths in this persistent workspace. Compact per-rank signposted CSV.gz, reports, command/source/log provenance and JSON evidence are checkpointed. Manifest excludes itself and Python bytecode.",
        entries=entries,
        implementation_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    (DOC / "artifact_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
