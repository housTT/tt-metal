"""Unit tests for parity_compare.py and eval_metrics.py (host only, no model, no device).

Run:
  cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/hostrun python -m pytest \
      models/autoports/cloudflare_clef/scripts/test_host_metrics.py --timeout=120

The parity tests use /home/hous/dev/clef/reports/reference/ref_text_bf16.jsonl as the
reference. Against itself every distance is zero and both sides score the same. Against a
perturbed copy (top-2 swapped on every question whose reference margin is at least 0.05 and
whose id is in the first half of the file, a 0.02 shift on one question with a wide margin,
and a top-2 swap on a near-tie question when one exists) the flip counts and the maximum
distance are known in advance.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics
import parity_compare
from clef_paths import read_jsonl

REFERENCE = Path("/home/hous/dev/clef/reports/reference/ref_text_bf16.jsonl")


def swap_top2(distribution: dict) -> dict:
    ordered = sorted(distribution, key=distribution.__getitem__, reverse=True)
    swapped = dict(distribution)
    swapped[ordered[0]], swapped[ordered[1]] = distribution[ordered[1]], distribution[ordered[0]]
    return swapped


@pytest.fixture(scope="module")
def reference_rows():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE} missing")
    return read_jsonl(REFERENCE)


def test_identity_is_all_zero(reference_rows):
    summary = parity_compare.summarize(reference_rows, copy.deepcopy(reference_rows), 0.05, 10, "self")
    overall = summary["overall"]
    assert summary["records"]["joined"] == len(reference_rows)
    assert summary["records"]["option_mismatches"] == 0
    assert overall["questions"] == sum(len(row["probs"]) for row in reference_rows)
    assert overall["max_dp"] == 0.0 and overall["mean_dp"] == 0.0
    assert overall["argmax_flips"] == 0 and overall["flips_at_margin"] == 0 and overall["near_tie_flips"] == 0
    assert overall["labelled_questions"] == sum(len(row.get("_label", {})) for row in reference_rows)
    assert overall["ref_accuracy"] == overall["cand_accuracy"]
    assert overall["ref_brier"] == overall["cand_brier"]
    assert overall["ref_ece"] == overall["cand_ece"]
    assert overall["accuracy_delta_pp"] == 0.0 and overall["ece_shift"] == 0.0
    assert summary["flips"] == []
    assert "all" in parity_compare.markdown(summary)


def test_identity_accuracy_matches_cpu_reference_correct_field(reference_rows):
    summary = parity_compare.summarize(reference_rows, reference_rows, 0.05, 10, "self")
    expected = [flag for row in reference_rows for flag in (row.get("correct") or {}).values()]
    assert summary["overall"]["labelled_questions"] == len(expected)
    assert math.isclose(summary["overall"]["ref_accuracy"], sum(expected) / len(expected))


def test_perturbed_copy_counts_flips_and_distance(reference_rows):
    perturbed = copy.deepcopy(reference_rows)
    half = {row["id"] for row in reference_rows[: len(reference_rows) // 2]}
    expected_margin_flips = 0
    expected_near_tie_flips = 0
    shifted = None
    for row in perturbed:
        for question_id, distribution in row["probs"].items():
            margin = parity_compare.top2_margin(distribution)
            if row["id"] in half and margin >= 0.05:
                row["probs"][question_id] = swap_top2(distribution)
                expected_margin_flips += 1
            elif margin < 0.05 and expected_near_tie_flips == 0:
                row["probs"][question_id] = swap_top2(distribution)
                expected_near_tie_flips += 1
            elif shifted is None and margin >= 0.5:
                top = parity_compare.argmax(distribution)
                other = next(option for option in distribution if option != top)
                moved = dict(distribution)
                moved[top] -= 0.02
                moved[other] += 0.02
                row["probs"][question_id] = moved
                shifted = (row["id"], question_id)
    assert expected_margin_flips > 0 and shifted is not None
    summary = parity_compare.summarize(reference_rows, perturbed, 0.05, 10, "perturbed")
    overall = summary["overall"]
    assert overall["flips_at_margin"] == expected_margin_flips
    assert overall["near_tie_flips"] == expected_near_tie_flips
    assert overall["argmax_flips"] == expected_margin_flips + expected_near_tie_flips
    assert len(summary["flips"]) == overall["argmax_flips"]
    largest_swap = max(
        parity_compare.top2_margin(row["probs"][q])
        for row in reference_rows
        if row["id"] in half
        for q in row["probs"]
        if parity_compare.top2_margin(row["probs"][q]) >= 0.05
    )
    assert math.isclose(overall["max_dp"], largest_swap, rel_tol=1e-9)
    moved = next(q for q in summary["per_question"] if (q["id"], q["question"]) == shifted)
    assert math.isclose(moved["dp"], 0.02, abs_tol=1e-9) and not moved["flip"]
    assert overall["cand_accuracy"] != overall["ref_accuracy"]
    assert overall["cand_brier"] > overall["ref_brier"]


def test_mismatch_and_error_rows_are_counted():
    reference = [
        {"id": "a", "answers": {"q": {"type": "choice"}}, "probs": {"q": {"x": 0.7, "y": 0.3}}, "_label": {"q": "x"}},
        {"id": "b", "answers": {"q": {"type": "choice"}}, "probs": {"q": {"x": 0.4, "y": 0.6}}, "_label": {"q": "y"}},
        {"id": "c", "error": "boom"},
    ]
    candidate = [
        {"id": "a", "probs": {"q": {"x": 0.6, "z": 0.4}}},
        {"id": "b", "probs": {"q": {"x": 0.45, "y": 0.55}}},
        {"id": "d", "probs": {"q": {"x": 1.0}}},
    ]
    summary = parity_compare.summarize(reference, candidate, 0.05, 10, "mix")
    assert summary["records"]["reference_errors"] == 1
    assert summary["records"]["option_mismatches"] == 1
    assert summary["records"]["extra_in_candidate"] == ["d"]
    assert summary["records"]["missing_in_candidate"] == []
    assert summary["overall"]["questions"] == 1
    assert math.isclose(summary["overall"]["max_dp"], 0.05)


def test_ece_and_brier_definitions():
    assert parity_compare.ece([0.95, 0.95], [True, True], 10) == pytest.approx(0.05)
    assert parity_compare.ece([1.0], [True], 10) == 0.0
    assert parity_compare.ece([0.6, 0.6, 0.6, 0.6], [True, False, True, False], 10) == pytest.approx(0.1)
    assert parity_compare.brier({"a": 1.0, "b": 0.0}, "a") == 0.0
    assert parity_compare.brier({"a": 0.5, "b": 0.5}, "a") == pytest.approx(0.5)
    assert parity_compare.label_option("noul", True) == "true" and parity_compare.label_option("noul", False) == "false"
    assert parity_compare.label_option("score", 2) == "2"


def test_eval_metrics_accuracy_and_macro_f1():
    eval_rows = [
        {
            "id": f"r{i}",
            "_source": "PolyAI/banking77 test",
            "questions": {"intent": {"type": "choice"}},
            "_label": {"intent": gold},
        }
        for i, gold in enumerate(["a", "a", "b", "c"])
    ]
    results = [
        {"id": "r0", "answers": {"intent": {"type": "choice", "choice": "a"}}},
        {"id": "r1", "answers": {"intent": {"type": "choice", "choice": "b"}}},
        {"id": "r2", "answers": {"intent": {"type": "choice", "choice": "b"}}},
        {"id": "r3", "probs": {"intent": {"a": 0.1, "b": 0.2, "c": 0.7}}},
    ]
    summary = eval_metrics.score(eval_rows, results, "auto", None)
    assert summary["metric"] == "macro_f1"
    assert summary["scored_questions"] == 4 and summary["correct"] == 3
    assert summary["accuracy"] == pytest.approx(0.75)
    f1_a = 2 * (1.0 * 0.5) / (1.0 + 0.5)
    f1_b = 2 * (0.5 * 1.0) / (0.5 + 1.0)
    assert summary["macro_f1"] == pytest.approx((f1_a + f1_b + 1.0) / 3)
    assert summary["macro_f1_labels"] == 3
    accuracy_only = eval_metrics.score(eval_rows, results, "accuracy", None)
    assert "macro_f1" not in accuracy_only
    partial = eval_metrics.score(eval_rows, results[:3], "accuracy", None)
    assert partial["missing_results"] == ["r3"]


def test_eval_metrics_noul_and_score_predictions():
    eval_rows = [
        {"id": "n", "questions": {"q": {"type": "noul"}}, "_label": {"q": True}},
        {"id": "s", "questions": {"q": {"type": "score"}}, "_label": {"q": 2}},
    ]
    results = [
        {"id": "n", "answers": {"q": {"type": "noul", "noul": 0.51}}},
        {"id": "s", "answers": {"q": {"type": "score", "score": 1.6, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}}}},
    ]
    summary = eval_metrics.score(eval_rows, results, "accuracy", None)
    assert summary["correct"] == 2 and summary["accuracy"] == 1.0
    assert json.dumps(summary)
