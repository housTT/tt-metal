# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Audit preserved run archives and, optionally, the final runtime; no device imports.

Historical failed experiments are valid evidence when their archives verify.
Incomplete runs are errors unless named explicitly or preserved in the interruption
ledger. Neither exception is counted as a completed, verified run.
"""

import argparse
import fnmatch
import gzip
import hashlib
import json
import re
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

COMPLETION_FIELDS = ("returncode", "ended_utc", "log_archive", "log_archive_sha256", "log_sha256")
RUNTIME_BINARIES = ("build/lib/_ttnncpp.so", "build/lib/libtt_metal.so", "ttnn/ttnn/_ttnn.so")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def relative_path(value):
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and bool(path.parts)


class Audit:
    def __init__(self, model_root, repo_root):
        self.model_root = model_root
        self.repo_root = repo_root
        self.errors = []
        self.counts = Counter()
        self.source_cache = {}
        self.current_cache = {}
        self.archive_paths = set()
        self.archive_digests = set()

    def error(self, run, kind, **details):
        self.errors.append({"run": run, "kind": kind, **details})

    def check_hash(self, run, kind, actual, expected, path):
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            self.error(run, "missing_or_invalid_expected_sha256", field=kind, path=str(path))
            return False
        if actual != expected:
            self.error(run, kind, path=str(path), expected=expected, actual=actual)
            return False
        return True

    def archive(self, run, record, kind):
        value = record.get(f"{kind}_archive")
        if not isinstance(value, str) or not value:
            self.error(run, "missing_archive_path", archive_kind=kind)
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.repo_root / path
        try:
            packed = path.read_bytes()
            actual = digest(packed)
            valid = self.check_hash(run, f"{kind}_archive_sha256", actual, record.get(f"{kind}_archive_sha256"), path)
            # Decompress even on hash mismatch, so corruption is distinguished from stale metadata.
            unpacked = gzip.decompress(packed)
        except (OSError, EOFError, ValueError, zlib.error) as error:
            self.error(run, "archive_unreadable", path=str(path), detail=str(error))
            return None
        if valid:
            self.counts[f"{kind}_archives_verified"] += 1
            self.archive_paths.add(str(path.resolve()))
            self.archive_digests.add(actual)
        return unpacked, actual

    def sources(self, run, record):
        result = self.archive(run, record, "source")
        expected = record.get("source_sha256")
        if not isinstance(expected, dict) or not expected or not all(isinstance(key, str) for key in expected):
            self.error(run, "missing_or_invalid_source_sha256")
            return
        if result is None:
            return
        unpacked, archive_hash = result
        if archive_hash not in self.source_cache:
            try:
                contents = json.loads(unpacked)
                if not isinstance(contents, dict) or not all(
                    isinstance(path, str) and relative_path(path) and isinstance(source, str)
                    for path, source in contents.items()
                ):
                    raise ValueError("source archive must map relative paths to source text")
                self.source_cache[archive_hash] = {
                    path: digest(source.encode("utf-8")) for path, source in contents.items()
                }
            except (ValueError, UnicodeError) as error:
                self.error(run, "invalid_source_archive_json", detail=str(error))
                return
        archived = self.source_cache[archive_hash]
        if set(archived) != set(expected):
            self.error(
                run,
                "source_archive_key_set",
                missing=sorted(set(expected) - set(archived)),
                extra=sorted(set(archived) - set(expected)),
            )
        for path in sorted(set(archived) & set(expected)):
            if self.check_hash(run, "archived_source_sha256", archived[path], expected[path], path):
                self.counts["archived_source_entries_verified"] += 1

    def log(self, run, record, provenance):
        result = self.archive(run, record, "log")
        if result is not None and self.check_hash(
            run, "decompressed_log_sha256", digest(result[0]), record.get("log_sha256"), record["log_archive"]
        ):
            self.counts["decompressed_logs_verified"] += 1
        plain = provenance.with_name(f"{run}.log")
        if plain.exists():
            try:
                if self.check_hash(run, "plain_log_sha256", file_digest(plain), record.get("log_sha256"), plain):
                    self.counts["plain_logs_verified"] += 1
            except OSError as error:
                self.error(run, "plain_log_unreadable", path=str(plain), detail=str(error))

    def current_files(self, run, expected, root, kind):
        if not isinstance(expected, dict) or not expected or not all(isinstance(key, str) for key in expected):
            self.error(run, "missing_or_invalid_runtime_manifest", field=kind)
            return
        for relative, expected_hash in sorted(expected.items()):
            if not isinstance(relative, str) or not relative_path(relative):
                self.error(run, "invalid_relative_runtime_path", field=kind, path=relative)
                continue
            path = root / relative
            try:
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns)
                if key not in self.current_cache:
                    self.current_cache[key] = file_digest(path)
                if self.check_hash(run, kind, self.current_cache[key], expected_hash, path):
                    self.counts[f"{kind}_entries_verified"] += 1
            except OSError as error:
                self.error(run, "current_runtime_file_unreadable", path=str(path), detail=str(error))

    def final_runtime(self, run, record):
        if record.get("returncode") != 0:
            self.error(run, "final_run_not_successful", returncode=record.get("returncode"))
        manifest = record.get("source_sha256", {})
        if not isinstance(manifest, dict) or not all(isinstance(key, str) for key in manifest):
            self.error(run, "missing_or_invalid_runtime_manifest", field="source_sha256")
            manifest = {}
        python_sources = {
            path: value for path, value in manifest.items() if path.startswith("tt/") and path.endswith(".py")
        }
        current = {str(p.relative_to(self.model_root)) for p in (self.model_root / "tt").rglob("*.py")}
        if current != set(python_sources):
            self.error(
                run,
                "current_python_runtime_key_set",
                unrecorded_current=sorted(current - set(python_sources)),
                missing_current=sorted(set(python_sources) - current),
            )
        self.current_files(run, python_sources, self.model_root, "current_python_runtime_sha256")
        self.current_files(run, record.get("native_source_sha256"), self.repo_root, "current_native_source_sha256")
        binaries = record.get("runtime_binary_sha256")
        if isinstance(binaries, dict) and not set(RUNTIME_BINARIES) <= set(binaries):
            self.error(run, "missing_required_runtime_binaries", paths=sorted(set(RUNTIME_BINARIES) - set(binaries)))
        self.current_files(run, binaries, self.repo_root, "current_runtime_binary_sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="defaults to DOC/artifact_integrity.json")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--include", action="append", help="run-name glob; repeat for a completed-subset sanity audit")
    parser.add_argument("--ignore-incomplete", action="append", default=[], metavar="EXACT_RUN_NAME")
    parser.add_argument("--final-label", action="append", default=[], help="substring selecting final runs; repeatable")
    args = parser.parse_args()
    doc = args.doc_dir.resolve()
    output = args.output or doc / "artifact_integrity.json"
    patterns = args.include or ["*"]
    if (args.include or args.ignore_incomplete) and output.resolve() == (doc / "artifact_integrity.json").resolve():
        parser.error(
            "subset/active-run audits require an explicit temporary --output; do not overwrite the final audit"
        )
    audit = Audit(doc.parent.parent, args.repo_root.resolve())
    started = utc_now()
    ledger_path = doc / "interrupted_runs.json"
    ledger = {}
    ledger_hash = None
    if ledger_path.exists():
        try:
            raw = ledger_path.read_bytes()
            ledger_hash = digest(raw)
            ledger = json.loads(raw)
            if not isinstance(ledger, dict):
                raise ValueError("interruption ledger must be an object keyed by run name")
        except (OSError, ValueError) as error:
            audit.error(None, "invalid_interruption_ledger", detail=str(error))
            ledger = {}
    paths = sorted(doc.rglob("*.provenance.json"))
    paths = [
        path
        for path in paths
        if any(fnmatch.fnmatchcase(path.name.removesuffix(".provenance.json"), p) for p in patterns)
    ]
    if not paths:
        audit.error(None, "no_provenance_records_selected")
    results, returncodes = [], Counter()
    final_matches = {label: [] for label in args.final_label}
    for path in paths:
        run = path.name.removesuffix(".provenance.json")
        entry = {"run": run, "provenance": str(path.relative_to(doc))}
        results.append(entry)
        prior_errors = len(audit.errors)
        try:
            raw = path.read_bytes()
            entry["provenance_sha256"] = digest(raw)
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("provenance must be an object")
        except (OSError, ValueError) as error:
            audit.error(run, "provenance_unreadable", detail=str(error))
            entry["status"] = "error"
            continue
        selected_final = False
        for label in args.final_label:
            if label in run:
                final_matches[label].append(run)
                selected_final = True
        missing = [field for field in COMPLETION_FIELDS if record.get(field) is None]
        audit.sources(run, record)
        if missing:
            entry["missing_completion_fields"] = missing
            interruption = ledger.get(run)
            if (
                isinstance(interruption, dict)
                and interruption.get("status") == "interrupted"
                and interruption.get("reason")
                and interruption.get("original_provenance_preserved") is True
            ):
                entry["status"] = "ledger_interrupted"
                entry["interruption"] = interruption
                audit.counts["ledger_interrupted_runs"] += 1
            elif run in args.ignore_incomplete:
                entry["status"] = "explicitly_ignored_incomplete"
                audit.counts["explicitly_ignored_incomplete_runs"] += 1
            else:
                entry["status"] = "incomplete"
                audit.error(run, "incomplete_provenance", missing=missing)
            # An interrupted record may already contain an anchored log archive.
            if record.get("log_archive") is not None:
                if record.get("log_sha256") is not None:
                    audit.log(run, record, path)
                else:
                    audit.archive(run, record, "log")
            if selected_final:
                audit.error(run, "final_run_incomplete")
        else:
            audit.counts["completed_runs"] += 1
            if type(record["returncode"]) is not int:
                audit.error(run, "invalid_returncode", value=record["returncode"])
            else:
                returncodes[str(record["returncode"])] += 1
            entry["returncode"] = record["returncode"]
            if run in ledger:
                audit.error(run, "completed_run_listed_as_interrupted")
            audit.log(run, record, path)
            if selected_final:
                audit.final_runtime(run, record)
                audit.counts["final_runtime_runs_checked"] += 1
            entry["status"] = "verified_completed" if len(audit.errors) == prior_errors else "error"
            if entry["status"] == "verified_completed":
                audit.counts["verified_completed_runs"] += 1
        if len(audit.errors) > prior_errors:
            entry["integrity_error_count"] = len(audit.errors) - prior_errors
    for label, matched in final_matches.items():
        if not matched:
            audit.error(None, "final_label_has_no_matching_runs", label=label)
    selected_names = {entry["run"] for entry in results}
    for name in args.ignore_incomplete:
        if name not in selected_names:
            audit.error(name, "explicit_ignore_has_no_matching_record")
    if not args.include:
        for name in ledger:
            if name not in selected_names:
                audit.error(name, "interruption_ledger_provenance_missing")
    audit.counts.update(
        provenance_records=len(results),
        unique_verified_archive_files=len(audit.archive_paths),
        unique_verified_archive_contents=len(audit.archive_digests),
        errors=len(audit.errors),
    )
    partial = bool(args.include or args.ignore_incomplete)
    status = "FAIL" if audit.errors else ("PARTIAL_SANITY" if partial else "PASS_COMPLETED_ARCHIVES")
    document = {
        "status": status,
        "started_utc": started,
        "ended_utc": utc_now(),
        "generator_source_sha256": file_digest(Path(__file__)),
        "selection": {"include": patterns, "ignore_incomplete": args.ignore_incomplete},
        "counts": dict(audit.counts),
        "returncode_counts": dict(sorted(returncodes.items())),
        "interruption_ledger_sha256": ledger_hash,
        "final_runtime_matches": final_matches,
        "current_runtime_scope": "tt/**/*.py file closure, recorded native sources, and recorded runtime binaries; historical reference/tests/docs are checked against their archives only",
        "interpretation": [
            "Integrity verifies recorded content; it does not authenticate provenance or prove correctness of the recorded command.",
            "Historical nonzero return codes are retained and do not alone constitute an integrity error.",
            "Interrupted and explicitly ignored incomplete records are counted separately, never as verified completed runs.",
            "Current source/binary comparisons run only for explicit final labels, whose commands must have completed successfully.",
        ],
        "records": results,
        "errors": audit.errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(output),
                "counts": dict(audit.counts),
                "returncode_counts": document["returncode_counts"],
            }
        )
    )
    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
