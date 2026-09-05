# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only consistency audit of immutable runs and selected-path evidence."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parent
REPO = DOC.parents[4]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rejected_path = DOC / "geometry_rejections.json"
    rejected = read(rejected_path) if rejected_path.exists() else {}
    records = list((DOC / "logs").glob("*.provenance.json"))
    for path in records:
        record = read(path)
        label = path.name.removesuffix(".provenance.json")
        if record["exit_code"] != 0:
            assert record["exit_code"] == rejected[label]["expected_exit_code"], label
            assert sha(DOC / (label + ".json")) == rejected[label]["artifact_sha256"]
            assert sha(path) == rejected[label]["provenance_sha256"]
        assert record["log_sha256"] == sha(path.with_name(label + ".log")), label
        source = path.with_name(label + ".sources.json.gz")
        assert record["snapshot_sha256"] == sha(source), label
        with gzip.open(source, "rt") as stream:
            payload = json.load(stream)
        assert record["sources_sha256"] == {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in payload.items()
        }, label
    rows = read(DOC / "sweep_results.json")
    with (DOC / "sweep_results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)
    legacy_rows = []
    for row in rows:
        assert row["trace_verified"] and row["token_count"] == 100
        assert row["pass_status"] == "pass"
        assert Path(row["precision_config_path"]).is_file()
        result = read(Path(row["evidence"]))
        assert len(result["runtime"]["layers"]) == 32
        if "propagation_assertions_passed" in result["runtime"]:
            assert result["runtime"]["propagation_assertions_passed"]
        else:
            legacy_rows.append(row["run_id"])
        recorded = dict(result["runtime"]["policy"])
        normalized = dict(row["dtype_policy"])
        for key in ("head_geometry", "config_id"):
            recorded.pop(key, None)
            normalized.pop(key, None)
        assert recorded == normalized
        assert result["teacher_forcing_counters"]["model_replays"] >= 99
    selected = read(DOC / "selected_precision_config.json")
    summary = read(DOC / "selection_summary.json")
    for name in [Path(summary[key]).stem for key in ("teacher_artifact", "token_out_artifact", "context_artifact")]:
        result = read(DOC / (name + ".json"))
        assert result["runtime"]["policy"] == selected
        assert len(result["runtime"]["layers"]) == 32
        record = read(DOC / "logs" / (name + ".provenance.json"))
        assert record["input_artifact_sha256"][str(DOC / "selected_precision_config.json")] == sha(
            DOC / "selected_precision_config.json"
        )
        for path in (DOC.parents[1] / "tt").glob("*.py"):
            assert record["sources_sha256"][str(path.relative_to(REPO))] == sha(path), (name, path)
    perf = read(DOC / summary["token_out_artifact"])
    assert perf["pass"] and perf["cache_context"] == selected["max_context"] == 262144
    assert "--precision-config" not in perf["command"]
    for sample in perf["token_out_no_readback"]:
        counters = sample["loop_counters"]
        assert counters["model_replays"] == counters["sampling_replays"] == 127
        assert all(counters[key] == 0 for key in ("readbacks", "read_waits", "synchronizations", "token_refreshes"))
    capacity = read(DOC / summary["context_artifact"])
    assert capacity["pass"] and [row["logical_prompt_length"] for row in capacity["windows"]] == [262143, 262144]
    assert capacity["windows"][0]["advanced_position"] == 262144
    report = dict(
        pass_status="pass",
        immutable_runs_verified=len(records),
        full_model_rows=len(rows),
        full_model_configs=len({row["config_id"] for row in rows}),
        selected_policy=selected["config_id"],
        production_source_hashes_match_final_runs=True,
        legacy_rows_without_assertion_flag=legacy_rows,
        legacy_policy_evidence="Runtime tensor ledgers plus immutable construction source; earliest baseline/canonicalLoFi compute summaries are pointer reprs, later rows include actual kernel fields",
        context=262144,
        qualitative_review=summary["qualitative_review"],
    )
    (DOC / "evidence_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
