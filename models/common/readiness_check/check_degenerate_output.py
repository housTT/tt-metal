# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Machine check for mechanically degenerate generated text.

The checker is deliberately standalone so runner-side gates can invoke it by
file path without importing the (optional) readiness harness. It scans the
standard free-running autoregressive and vLLM qualitative artifacts and uses
exit codes 0/1/2/3 for clean/advisory/critical/checker-error respectively.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

MIN_WORDS_FOR_DUPLICATION = 20
ADJACENT_DUPLICATION_CRITICAL = 0.10
TRIGRAM_LOOP_ADVISORY = 0.50
MIN_WORDS_FOR_LOOP = 50
NEAR_EMPTY_CHARS = 5

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LOAD_FAILED = object()


@dataclass
class Finding:
    severity: str
    artifact: str
    label: str
    metric: str
    value: float
    threshold: float
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    measured: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if any(finding.severity == "critical" for finding in self.findings):
            return 2
        if self.findings:
            return 1
        return 0


def words_of(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def adjacent_duplication(tokens: Sequence[Any]) -> float:
    if len(tokens) < 2:
        return 0.0
    duplicates = sum(1 for left, right in zip(tokens, tokens[1:]) if left == right)
    return duplicates / (len(tokens) - 1)


def trigram_loop_fraction(tokens: Sequence[Any]) -> float:
    if len(tokens) < 3:
        return 0.0
    counts: dict[tuple[Any, ...], int] = {}
    for index in range(len(tokens) - 2):
        trigram = tuple(tokens[index : index + 3])
        counts[trigram] = counts.get(trigram, 0) + 1
    most_common = max(counts, key=counts.get)  # type: ignore[arg-type]
    covered = 0
    index = 0
    while index <= len(tokens) - 3:
        if tuple(tokens[index : index + 3]) == most_common:
            covered += 3
            index += 3
        else:
            index += 1
    return covered / len(tokens)


def check_completion(
    report: Report,
    *,
    artifact: Path,
    label: str,
    text: str | None,
    token_ids: Sequence[int] | None = None,
) -> None:
    sequences: list[tuple[str, Sequence[Any]]] = []
    if text is not None and text.strip():
        sequences.append(("words", words_of(text)))
    if token_ids:
        sequences.append(("token_ids", list(token_ids)))

    if text is not None and len(text.strip()) < NEAR_EMPTY_CHARS:
        report.findings.append(
            Finding(
                severity="advisory",
                artifact=str(artifact),
                label=label,
                metric="near_empty_completion",
                value=float(len(text.strip())),
                threshold=float(NEAR_EMPTY_CHARS),
                detail="Completion is empty or whitespace; verify EOS handling and sampler output.",
            )
        )
        report.measured.append({"artifact": str(artifact), "label": label, "source": "words", "near_empty": True})

    for source, tokens in sequences:
        duplication = adjacent_duplication(tokens)
        loop_fraction = trigram_loop_fraction(tokens)
        report.measured.append(
            {
                "artifact": str(artifact),
                "label": label,
                "source": source,
                "num_tokens": len(tokens),
                "adjacent_duplication": round(duplication, 4),
                "trigram_loop_fraction": round(loop_fraction, 4),
            }
        )

        if len(tokens) >= MIN_WORDS_FOR_DUPLICATION and duplication > ADJACENT_DUPLICATION_CRITICAL:
            report.findings.append(
                Finding(
                    severity="critical",
                    artifact=str(artifact),
                    label=f"{label} ({source})",
                    metric="adjacent_duplication",
                    value=round(duplication, 4),
                    threshold=ADJACENT_DUPLICATION_CRITICAL,
                    detail=(
                        "Adjacent-token duplication this high while text advances is a decode-loop "
                        "input bug signature, not a model-quality property. Compare the same prompt "
                        "against the HF reference."
                    ),
                )
            )
        elif len(tokens) >= MIN_WORDS_FOR_LOOP and loop_fraction > TRIGRAM_LOOP_ADVISORY:
            report.findings.append(
                Finding(
                    severity="advisory",
                    artifact=str(artifact),
                    label=f"{label} ({source})",
                    metric="trigram_loop_fraction",
                    value=round(loop_fraction, 4),
                    threshold=TRIGRAM_LOOP_ADVISORY,
                    detail=(
                        "Completion is dominated by one repeating phrase. This can be normal for "
                        "base checkpoints under greedy decoding; verify against the HF reference."
                    ),
                )
            )


def _load_artifact(report: Report, path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        report.findings.append(
            Finding(
                severity="critical",
                artifact=str(path),
                label="artifact parse",
                metric="unreadable_artifact",
                value=0.0,
                threshold=1.0,
                detail=f"Artifact could not be read or parsed ({error}). Regenerate it.",
            )
        )
        return _LOAD_FAILED


def _record_invalid_artifact(report: Report, path: Path, label: str, detail: str) -> None:
    report.findings.append(
        Finding(
            severity="critical",
            artifact=str(path),
            label=label,
            metric="invalid_artifact_schema",
            value=0.0,
            threshold=1.0,
            detail=detail,
        )
    )


def check_vllm_qualitative(report: Report, path: Path, missing_artifacts: str = "critical") -> None:
    items = _load_artifact(report, path)
    if items is _LOAD_FAILED:
        return
    if not isinstance(items, list):
        _record_invalid_artifact(report, path, "vLLM qualitative outputs", "Expected a JSON list of output objects.")
        return
    findings_before = len(report.findings)
    completions = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _record_invalid_artifact(
                report, path, f"prompt[{index}]", "Expected each vLLM qualitative output to be a JSON object."
            )
            continue
        prompt = str(item.get("prompt", ""))[:60]
        for key in ("greedy_completion", "sampled_completion"):
            if key in item:
                if not isinstance(item[key], str):
                    _record_invalid_artifact(
                        report, path, f"prompt[{index}] {key}", "Expected completion text to be a JSON string."
                    )
                    continue
                completions += 1
                check_completion(
                    report,
                    artifact=path,
                    label=f"prompt[{index}] {key} ({prompt!r})",
                    text=item.get(key) or "",
                )
    if completions == 0 and len(report.findings) == findings_before:
        report.findings.append(
            Finding(
                severity=missing_artifacts,
                artifact=str(path),
                label="vLLM qualitative outputs",
                metric="missing_completion",
                value=0.0,
                threshold=1.0,
                detail="The artifact contains no greedy or sampled completion text.",
            )
        )


def check_autoregressive_meta(report: Report, path: Path, missing_artifacts: str = "critical") -> None:
    metadata = _load_artifact(report, path)
    if metadata is _LOAD_FAILED:
        return
    if not isinstance(metadata, dict):
        _record_invalid_artifact(report, path, "autoregressive metadata", "Expected a JSON object.")
        return
    tt_metadata = metadata.get("tt")
    if not isinstance(tt_metadata, dict):
        _record_invalid_artifact(report, path, "tt metadata", "Expected a JSON object at key 'tt'.")
        return
    text_path = path.parent / "tt_completion.txt"
    try:
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
    except Exception as error:  # noqa: BLE001
        report.findings.append(
            Finding(
                severity="critical",
                artifact=str(text_path),
                label="tt free-running completion",
                metric="unreadable_artifact",
                value=0.0,
                threshold=1.0,
                detail=f"Completion text could not be read ({error}). Regenerate it.",
            )
        )
        return
    token_ids = tt_metadata.get("token_ids")
    if token_ids is not None and (
        not isinstance(token_ids, list) or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        _record_invalid_artifact(
            report,
            path,
            "tt token_ids",
            "Expected 'tt.token_ids' to be a JSON list of non-negative integers.",
        )
        return
    if not (text and text.strip()) and not token_ids:
        report.findings.append(
            Finding(
                severity=missing_artifacts,
                artifact=str(path),
                label="tt free-running completion",
                metric="missing_completion",
                value=0.0,
                threshold=1.0,
                detail="Autoregressive metadata contains neither TT completion text nor TT token IDs.",
            )
        )
        return
    check_completion(
        report,
        artifact=path,
        label="tt free-running completion",
        text=text,
        token_ids=token_ids,
    )
    hf_metadata = metadata.get("hf")
    hf_ids = hf_metadata.get("token_ids") if isinstance(hf_metadata, dict) else None
    tt_ids = tt_metadata.get("token_ids")
    if hf_ids and tt_ids:
        matches = sum(1 for left, right in zip(hf_ids, tt_ids) if left == right)
        report.measured.append(
            {
                "artifact": str(path),
                "label": "hf/tt token agreement (informational)",
                "matching_tokens": matches,
                "compared_tokens": min(len(hf_ids), len(tt_ids)),
            }
        )


def discover(roots: Iterable[Path], scope: str) -> tuple[list[Path], list[Path]]:
    vllm_files: list[Path] = []
    autoregressive_files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.name == "autoregressive_meta.json":
                autoregressive_files.append(root)
            else:
                vllm_files.append(root)
            continue
        if scope in ("all", "vllm"):
            vllm_files.extend(sorted(root.rglob("vllm_qualitative_outputs.json")))
        if scope in ("all", "autoregressive"):
            autoregressive_files.extend(sorted(root.rglob("autoregressive_meta.json")))
    return vllm_files, autoregressive_files


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_model_dirs(root: Path, hf_model: str) -> tuple[list[Path], str]:
    if not root.is_dir():
        return [], f"{root} does not exist"
    target = _squash(hf_model)
    markers = ("tt", "doc", "readiness_vllm")
    candidates = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir()
        and len(path.relative_to(root).parts) <= 3
        and any((path / marker).is_dir() for marker in markers)
    ]
    matches = []
    for path in candidates:
        squashed = _squash(str(path.relative_to(root)))
        if squashed and (squashed in target or target in squashed):
            matches.append(path)
    matches = [
        match for match in matches if not any(other != match and other.is_relative_to(match) for other in matches)
    ]
    if matches:
        return matches, ""
    if candidates:
        listing = ", ".join(str(candidate.relative_to(root)) for candidate in candidates)
        return [], f"no autoport directory under {root} matches {hf_model!r} (found: {listing})"
    return [], f"no autoport directories found under {root}"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path("models/autoports"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--hf-model")
    parser.add_argument("--scope", choices=("all", "vllm", "autoregressive"), default="all")
    parser.add_argument("--missing-artifacts", choices=("advisory", "critical"), default="advisory")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    report = Report()
    if args.paths:
        roots: list[Path] = args.paths
    elif args.model_dir:
        if args.model_dir.is_dir():
            roots = [args.model_dir]
        else:
            roots = []
            report.findings.append(
                Finding(
                    severity=args.missing_artifacts,
                    artifact=str(args.model_dir),
                    label="model directory resolution",
                    metric="missing_model_dir",
                    value=0.0,
                    threshold=1.0,
                    detail=f"--model-dir {args.model_dir} does not exist.",
                )
            )
    elif args.hf_model:
        roots, empty_reason = resolve_model_dirs(args.root, args.hf_model)
        if not roots:
            report.findings.append(
                Finding(
                    severity=args.missing_artifacts,
                    artifact=str(args.root),
                    label="model directory resolution",
                    metric="missing_model_dir",
                    value=0.0,
                    threshold=1.0,
                    detail=empty_reason,
                )
            )
        else:
            print(f"scoped to: {', '.join(str(root) for root in roots)}")
    else:
        roots = [args.root]

    vllm_files, autoregressive_files = discover(roots, args.scope)
    if roots and not vllm_files and not autoregressive_files and not report.findings:
        report.findings.append(
            Finding(
                severity=args.missing_artifacts,
                artifact=", ".join(str(root) for root in roots),
                label="artifact discovery",
                metric="missing_artifacts",
                value=0.0,
                threshold=1.0,
                detail=f"No generation artifacts found (scope={args.scope}).",
            )
        )

    for path in vllm_files:
        check_vllm_qualitative(report, path, args.missing_artifacts)
    for path in autoregressive_files:
        check_autoregressive_meta(report, path, args.missing_artifacts)

    for measurement in report.measured:
        compact = {key: value for key, value in measurement.items() if key not in ("artifact", "label")}
        print(f"measured: {measurement.get('label')} [{measurement.get('artifact')}] {compact}")

    if report.findings:
        print(f"\n{len(report.findings)} finding(s):")
        for finding in report.findings:
            print(f"\n[{finding.severity.upper()}] {finding.metric}={finding.value} (threshold {finding.threshold})")
            print(f"  artifact: {finding.artifact}")
            print(f"  where:    {finding.label}")
            print(f"  {finding.detail}")
    else:
        print("\nNo degenerate output detected.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "findings": [asdict(finding) for finding in report.findings],
                    "measured": report.measured,
                    "exit_code": report.exit_code,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return report.exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise SystemExit(3) from None
