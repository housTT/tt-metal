import json
import os
from pathlib import Path

import pytest

from models.autoports.cloudflare_clef.tt import encode as E

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
REFERENCE = Path("/home/hous/dev/clef/reports/reference/records_text.jsonl")
KEV_DEV = Path("/home/hous/dev/kev/kev/evals/documents-v1/development.jsonl")
N_RECORDS = 50
QUESTION_KEYS = ("type", "instructions", "criteria")


def strip_record(record):
    return {
        "state": record["state"],
        "questions": {
            qid: {k: v for k, v in q.items() if k in QUESTION_KEYS} for qid, q in record["questions"].items()
        },
    }


def load_records():
    records = []
    if REFERENCE.exists():
        with REFERENCE.open() as f:
            records += [strip_record(json.loads(line)) for line in f if line.strip()]
    with KEV_DEV.open() as f:
        for line in f:
            if len(records) >= N_RECORDS:
                break
            records.append(strip_record(json.loads(line)))
    return records[:N_RECORDS]


SYNTHETIC = [
    {
        "state": {"invoice": {"total": 120.5, "lines": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": None}]}},
        "questions": {
            "paid": {"type": "noul", "instructions": "Is the invoice fully paid?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is follow-up?",
                "criteria": ["low", "medium", "high"],
            },
            "team": {
                "type": "choice",
                "instructions": None,
                "criteria": {"billing": "Billing team", "sales": "Sales team"},
            },
        },
    },
    {
        "state": "short",
        "questions": {"q": {"type": "noul"}},
    },
]


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
    return E.load_tokenizer()


@pytest.fixture(scope="module")
def release():
    os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
    return E.release_module()


def check_split(encoded, tok, record):
    head, tail, questions = E.split_for_cache(encoded, tok, record)
    assert head + tail == list(encoded.input_ids)
    assert questions == encoded.questions
    split = len(head)
    for q in questions:
        assert q.question_span[0] >= split and q.question_span[1] <= len(encoded.input_ids)
        for start, end in q.option_spans:
            assert split <= start < end <= len(encoded.input_ids)
    assert len(E.cache_key(head)) == 40
    assert E.cache_key(head) == E.cache_key(list(head))
    return split


@pytest.mark.eager_host_side
def test_encode_matches_release(tok, release):
    records = load_records()
    assert len(records) == N_RECORDS
    seen_q = 0
    for i, record in enumerate(records):
        ours = E.encode(tok, record)
        ref = release.encode_record(tok, record)
        assert ours.input_ids == ref.input_ids, i
        assert ours.record_id == ref.record_id
        assert ours.media == ref.media
        assert len(ours.questions) == len(ref.questions)
        for a, b in zip(ours.questions, ref.questions):
            assert a.question_id == b.question_id
            assert a.question_type == b.question_type
            assert a.question_span == b.question_span
            assert a.option_spans == b.option_spans
            assert a.option_ids == b.option_ids
            seen_q += 1
        assert ours == ref
        check_split(ours, tok, record)
    assert seen_q >= N_RECORDS


@pytest.mark.eager_host_side
def test_split_covers_all_question_types_and_truncation(tok, release):
    for record in SYNTHETIC:
        encoded = E.encode(tok, record)
        assert encoded == release.encode_record(tok, record)
        split = check_split(encoded, tok, record)
        state_ids = release._tokens(tok, release.render(record["state"]))
        assert split == len(E.prefix_ids(tok)) + len(state_ids)
    record = SYNTHETIC[0]
    truncated = E.encode(tok, record, max_state_tokens=3)
    assert truncated == release.encode_record(tok, record, max_state_tokens=3)
    assert check_split(truncated, tok, record) == len(E.prefix_ids(tok)) + 3
    capped = E.encode(
        tok, record, max_length=len(E.prefix_ids(tok)) + 5 + len(truncated.input_ids) - len(E.prefix_ids(tok)) - 3
    )
    assert check_split(capped, tok, record) == len(E.prefix_ids(tok)) + 5


@pytest.mark.eager_host_side
def test_split_rejects_foreign_record(tok, expect_error):
    encoded = E.encode(tok, SYNTHETIC[0])
    with expect_error(ValueError, "do not match"):
        E.split_for_cache(encoded, tok, SYNTHETIC[1])
