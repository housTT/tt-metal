# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the archived Z diagnostic; device execution belongs to the parent lane."""

import argparse
import ast
import datetime
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

MODULE = "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_z_diagnostic"
ASSERTION = """        assert all(
            row["finite"] and min(row["user_pcc"]) >= 0.995 for row in control_rows
        ), "separate gate control failed for a logical user"
"""

# This block runs only after the original comparison has already failed. It
# creates no additional device tensors and preserves the original assertion.
POSTMORTEM = """
import hashlib

def post_stats(value):
    flat = value.float().flatten()
    finite = bool(flat.isfinite().all())
    result = dict(finite=finite, nonfinite=int((~flat.isfinite()).sum()), nonzero=int(flat.count_nonzero()))
    if finite:
        result.update(minimum=float(flat.min()), maximum=float(flat.max()),
                      std=float(flat.double().std(unbiased=False)), constant=bool((flat == flat[0]).all()))
    return result

def post_comparison(actual, reference):
    stats, ref_stats = post_stats(actual), post_stats(reference)
    finite = stats["finite"] and ref_stats["finite"]
    return dict(actual=stats, reference=ref_stats,
                exact=finite and torch.equal(actual, reference),
                pcc=H.pcc(reference, actual) if finite else None,
                pcc_defined=finite and not (stats.get("constant") or ref_stats.get("constant")),
                maxdiff=float((actual - reference).abs().max()) if finite else None)

def post_read(tensor):
    return [ttnn.to_torch(part).float().clone() for part in ttnn.get_device_tensors(tensor)]

def post_digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

print(json.dumps(dict(name="v2_postmortem_start", original_assertion_failed=True)), flush=True)
first_merged_digests = [post_digest(value) for value in merged_host]
first_z_digests = [post_digest(value) for value in z_host]
ttnn.synchronize_device(mesh)
post_core, post_z, post_merged = post_read(core), post_read(z), post_read(merged)
post_weights = post_read(decoder.w["gdn_norm"])
post_rows = []
for rank, (c, raw, m, weight) in enumerate(zip(post_core, post_z, post_merged, post_weights)):
    dim = decoder.cfg.linear_value_head_dim
    assert weight.numel() == dim, "unexpected norm weight shape"
    norm = c * torch.rsqrt(c.square().mean(dim=-1, keepdim=True) + decoder.cfg.norm_eps)
    norm = norm * weight.reshape(1, 1, 1, dim)
    cpu_merged = norm.reshape(args.batch, decoder.cfg.linear_num_value_heads, 1, dim)
    cpu_merged = cpu_merged.permute(0, 2, 1, 3).reshape(public_shape)
    users = []
    for user in range(args.batch):
        users.append(dict(user=user, core=post_stats(c[user]), raw_z=post_stats(raw[user]),
                          raw_z_reread_vs_first=post_comparison(raw[user], z_host[rank][user]),
                          merged_reread_vs_first=post_comparison(m[user], merged_host[rank][user]),
                          first_merged_vs_cpu_norm=post_comparison(merged_host[rank][user], cpu_merged[user]),
                          reread_merged_vs_cpu_norm=post_comparison(m[user], cpu_merged[user]),
                          first_gate_vs_first_cpu=post_comparison(separated_host[rank][user], oracle[rank][user])))
    post_rows.append(dict(rank=rank, users=users, core_digest=post_digest(c), raw_z_digest=post_digest(raw),
                          merged_digest=post_digest(m), first_merged_digest=first_merged_digests[rank],
                          first_z_digest=first_z_digests[rank]))
print(json.dumps(dict(name="v2_failure_postmortem", ranks=post_rows,
                     first_merged_host_immutable=first_merged_digests == [post_digest(v) for v in merged_host],
                     first_z_host_immutable=first_z_digests == [post_digest(v) for v in z_host],
                     digest_extent="logical_float32_host_bytes", raw_tensor_artifacts=[])), flush=True)
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_overlay(overlay, model_root, archive, provenance, postmortem):
    assert sha256(archive) == provenance["source_archive_sha256"], "source archive hash mismatch"
    with gzip.open(archive, "rt") as source:
        archived = json.load(source)
    package = overlay / "models" / "autoports" / model_root.name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    original_hashes, executed_hashes = {}, {}
    for name, source in archived.items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        if relative.suffix != ".py":
            continue
        original_hash = hashlib.sha256(source.encode()).hexdigest()
        assert original_hash == provenance["source_sha256"][name], name
        original_hashes[name] = original_hash
        if postmortem and name == "tests/multichip_z_diagnostic.py":
            assert source.count(ASSERTION) == 1, "archived failure assertion changed"
            guard = '        if not all(row["finite"] and min(row["user_pcc"]) >= 0.995 for row in control_rows):\n'
            source = source.replace(ASSERTION, guard + textwrap.indent(POSTMORTEM.strip() + "\n", " " * 12) + ASSERTION)
        ast.parse(source, filename=name)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        executed_hashes[name] = sha256(target)
    inputs = {}
    for relative in ("doc/functional_decoder", "doc/optimized_decoder/activations"):
        link = package / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(model_root / relative, target_is_directory=True)
    for relative in ("doc/functional_decoder/hf_config.json", "doc/optimized_decoder/activations/layer0.pt"):
        inputs[relative] = sha256(model_root / relative)
    changes = [name for name in original_hashes if original_hashes[name] != executed_hashes[name]]
    assert changes == (["tests/multichip_z_diagnostic.py"] if postmortem else []), changes
    return dict(original_source_sha256=original_hashes, executed_source_sha256=executed_hashes, input_sha256=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unchanged", "postmortem"), default="unchanged")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--check-only", action="store_true", help="Reconstruct and validate sources without importing TTNN"
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    runs = args.runs if args.runs is not None else (2 if args.mode == "unchanged" else 1)
    if not 1 <= runs <= 3 or args.timeout < 1:
        parser.error("runs must be 1–3 and timeout must be positive")
    report_root = Path(__file__).resolve().parent
    model_root = report_root.parents[1]
    repo_root = model_root.parents[2]
    archive = report_root / "logs/z_fusion_boundary_batch32_v2.sources.json.gz"
    provenance_path = archive.with_name("z_fusion_boundary_batch32_v2.provenance.json")
    provenance = json.loads(provenance_path.read_text())
    bootstrap = (
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "sys.argv=sys.argv[1:];runpy.run_module(sys.argv[0],run_name='__main__',alter_sys=True)"
    )
    manifest_path = (
        args.manifest or report_root / "logs" / f"z_v2_replay_{args.mode}_{uuid.uuid4().hex[:10]}.manifest.json"
    )
    manifest = dict(
        mode=args.mode,
        check_only=args.check_only,
        archive=str(archive),
        archive_sha256=sha256(archive),
        original_provenance_sha256=sha256(provenance_path),
        runner_sha256=sha256(Path(__file__).resolve()),
        original_module_arguments=[MODULE, "--batch", "32"],
        cwd=str(repo_root),
        runs=[],
    )

    def save():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="ornith-z-v2-") as temporary:
        overlay = Path(temporary)
        manifest.update(prepare_overlay(overlay, model_root, archive, provenance, args.mode == "postmortem"))
        save()
        print(
            json.dumps(
                dict(
                    name="archived_v2_overlay",
                    mode=args.mode,
                    manifest=str(manifest_path),
                    archive_sha256=manifest["archive_sha256"],
                    diagnostic_sha256=manifest["executed_source_sha256"]["tests/multichip_z_diagnostic.py"],
                    source_unchanged=args.mode == "unchanged",
                    check_only=args.check_only,
                    cwd=str(repo_root),
                )
            ),
            flush=True,
        )
        if args.check_only:
            return
        environment = os.environ.copy()
        for repeat in range(runs):
            command = [sys.executable, "-c", bootstrap, str(overlay), MODULE, "--batch", "32"]
            run = dict(
                repeat=repeat, command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
            print(json.dumps(dict(name="archived_v2_run_start", **run)), flush=True)
            try:
                result = subprocess.run(command, cwd=repo_root, env=environment, timeout=args.timeout, check=False)
                run["returncode"] = result.returncode
            except subprocess.TimeoutExpired:
                run["returncode"] = 124
            run["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            manifest["runs"].append(run)
            save()
            print(json.dumps(dict(name="archived_v2_run_end", **run)), flush=True)
            if run["returncode"] == 124 or run["returncode"] < 0:
                break
    raise SystemExit(0 if all(run["returncode"] == 0 for run in manifest["runs"]) else 1)


if __name__ == "__main__":
    main()
