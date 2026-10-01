import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from models.autoports.jaredpalmer_kev_9b.tt.encode import SPECIAL, rows_for_record

REF = Path("/home/hous/dev/kev/reports/reference")
EXPECTED_SPECIAL = {
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|fim_suffix|>": 248062,
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((REF / "rows.json").read_text())


@pytest.fixture(scope="module")
def tok(ref):
    return AutoTokenizer.from_pretrained(ref["base"], revision=ref["base_revision"])


def test_special_ids(tok):
    assert {t: tok.convert_tokens_to_ids(t) for t in SPECIAL} == EXPECTED_SPECIAL


def test_rows_match_reference(tok, ref):
    with open(REF / "records.jsonl") as f:
        records = [json.loads(line) for line in f]
    expected = {r["row_key"]: r for r in ref["rows"]}
    seen = 0
    for i, record in enumerate(records):
        for row in rows_for_record(tok, record):
            e = expected[f"{i}:{row.qid}"]
            assert row.ids == e["ids"], f"token ids differ for row {i}:{row.qid}"
            assert row.opt_positions == e["opt_positions"]
            assert row.decide_position == e["decide_position"]
            assert row.qtype == e["type"] and row.option_keys == e["keys"] and row.legend == e["legend"]
            assert row.ids[row.decide_position] == EXPECTED_SPECIAL["<|fim_suffix|>"]
            assert all(row.ids[p] == EXPECTED_SPECIAL["<|box_end|>"] for p in row.opt_positions)
            assert len(row.state_ids) == e["state_tokens"] and len(row.question_ids) == e["question_tokens"]
            seen += 1
    assert seen == len(expected) == 29
