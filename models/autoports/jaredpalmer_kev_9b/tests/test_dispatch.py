import json
import os
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from models.autoports.jaredpalmer_kev_9b.tt.dispatch import (
    CostModel,
    Policy,
    ShareResult,
    WorkerView,
    collect,
    merge_results,
    plan,
    share_cost_ms,
)

PERF_SUMMARY = Path(__file__).resolve().parent.parent / "doc" / "optimized" / "perf_summary.json"
SHORT_STATE, LONG_STATE, MID_STATE, ONE_BLOCK_STATE = 30, 2392, 370, 140
QUESTION = 40
KEY = "state-key"


class Row:
    def __init__(self, state_tokens, question_tokens=QUESTION):
        self.state_ids = [0] * state_tokens
        self.question_ids = [0] * question_tokens


def rows(n, state_tokens, question_tokens=QUESTION):
    return [Row(state_tokens, question_tokens) for _ in range(n)]


def workers(n=4, backlog=0.0, cached=()):
    return [
        WorkerView(i, backlog if isinstance(backlog, (int, float)) else backlog[i], {KEY} if i in cached else set())
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def model():
    return CostModel.from_perf_summary(PERF_SUMMARY)


def shape(result):
    return [(wid, len(idx)) for wid, idx in result]


def test_cost_model_from_perf_summary(model):
    data = json.loads(PERF_SUMMARY.read_text())["engine_ms"]
    assert dict(model.tail_ms) == {
        int(k.split("_")[-1]): v["traced_policy"] for k, v in data.items() if k.startswith("tail_bucket_")
    }
    assert model.tail_cost_ms(SHORT_STATE, QUESTION) == data["tail_bucket_128"]["traced_policy"]
    assert model.tail_cost_ms(MID_STATE, QUESTION) == data["tail_bucket_256"]["traced_policy"]
    assert model.tail_cost_ms(0, 5000) == data["tail_bucket_2048"]["traced_policy"]
    per_block = (data["state_2048"]["traced_policy"] / 16 + data["state_2392"]["traced_policy"] / 18) / 2
    assert model.state_ms_per_block == round(per_block, 2)
    assert model.state_cost_ms(SHORT_STATE) == 0.0
    assert model.state_cost_ms(LONG_STATE) == pytest.approx(18 * model.state_ms_per_block)
    assert model.state_cost_ms(LONG_STATE) == pytest.approx(data["state_2392"]["traced_policy"], rel=0.03)


def test_cost_model_from_dict_and_fallbacks(expect_error):
    m = CostModel.from_perf_summary(
        {
            "engine_ms": {
                "tail_bucket_128": {"traced": 100.0},
                "tail_bucket_256": 150.0,
                "state_256": {"eager": 120.0},
                "state_65536_plus_question_40": {"traced_policy": 1.0},
            }
        }
    )
    assert m.tail_ms == ((128, 100.0), (256, 150.0)) and m.state_ms_per_block == 60.0
    assert m.bucket(1000) == 256
    with expect_error(ValueError, ""):
        CostModel.from_perf_summary({"engine_ms": {"state_256": 1.0}})
    assert CostModel.load("/nonexistent/perf_summary.json") == CostModel()
    assert CostModel.load(PERF_SUMMARY) == CostModel.from_perf_summary(PERF_SUMMARY)
    assert CostModel.load() == CostModel.from_perf_summary(PERF_SUMMARY)


def test_single_short_request_spreads_over_idle_workers(model):
    result = plan(rows(6, SHORT_STATE), workers(), KEY, model)
    assert result == [(0, [0, 4]), (1, [1, 5]), (2, [2]), (3, [3])]
    views = workers()
    assert [share_cost_ms(rows(6, SHORT_STATE), idx, views[wid], KEY, model) for wid, idx in result] == pytest.approx(
        [209.8, 209.8, 104.9, 104.9]
    )


def test_long_state_cached_on_worker_2(model):
    r = rows(5, LONG_STATE)
    assert plan(r, workers(cached=(2,)), KEY, model) == [(2, [0, 1, 2, 3, 4])]
    replication = model.state_cost_ms(LONG_STATE)
    assert plan(r, workers(backlog=[0, 0, replication, 0], cached=(2,)), KEY, model) == [(2, [0, 1, 2, 3, 4])]
    busy = workers(backlog=[0, 0, 3 * replication, 0], cached=(2,))
    assert plan(r, busy, KEY, model) == [(0, [0, 1, 2, 3, 4])]
    assert share_cost_ms(r, [0, 1, 2, 3, 4], busy[0], KEY, model) == pytest.approx(replication + 5 * 104.9)
    assert share_cost_ms(r, [0, 1, 2, 3, 4], busy[2], KEY, model) == pytest.approx(5 * 104.9)


def test_long_new_state_is_prefilled_once(model):
    assert plan(rows(5, LONG_STATE), workers(), KEY, model) == [(0, [0, 1, 2, 3, 4])]
    assert plan(rows(5, LONG_STATE), workers(backlog=[500, 0, 0, 0]), KEY, model) == [(1, [0, 1, 2, 3, 4])]


def test_mid_state_replicates_only_when_the_share_pays_for_it(model):
    result = plan(rows(5, MID_STATE), workers(), KEY, model)
    assert shape(result) == [(0, 2), (1, 2), (2, 1)]
    state = model.state_cost_ms(MID_STATE)
    tail = model.tail_cost_ms(MID_STATE, QUESTION)
    assert tail > state
    assert plan(rows(5, MID_STATE, 10), workers(), KEY, model) == [(0, [0, 2, 4]), (1, [1, 3])]
    assert model.tail_cost_ms(MID_STATE, 10) < state


def test_one_block_state_drops_a_free_worker_that_saves_nothing(model):
    result = plan(rows(6, ONE_BLOCK_STATE), workers(), KEY, model)
    assert shape(result) == [(0, 2), (1, 2), (2, 2)]


def test_under_load_degrades_to_whole_request_balanced(model):
    views = workers(backlog=9600.0)
    counts = {w.id: 0 for w in views}
    for n in range(64):
        result = plan(rows(6, SHORT_STATE), views, f"key-{n}", model)
        assert len(result) == 1 and result[0][1] == list(range(6))
        wid, idx = result[0]
        counts[wid] += 1
        views[wid].backlog_ms += share_cost_ms(rows(6, SHORT_STATE), idx, views[wid], f"key-{n}", model)
    assert counts == {0: 16, 1: 16, 2: 16, 3: 16}


def test_ramp_from_idle_to_loaded(model):
    views = workers()
    fanned, whole = 0, 0
    for n in range(64):
        r = rows(6, SHORT_STATE)
        result = plan(r, views, f"key-{n}", model)
        fanned += len(result) > 1
        whole += len(result) == 1
        for wid, idx in result:
            views[wid].backlog_ms += share_cost_ms(r, idx, views[wid], f"key-{n}", model)
    assert fanned >= 1 and whole >= 60
    loads = [v.backlog_ms for v in views]
    assert max(loads) - min(loads) <= 6 * 104.9 + 1e-6


def test_cache_affinity_under_load(model):
    views = workers(backlog=[5000, 5000, 5000, 5000], cached=(3,))
    assert plan(rows(5, LONG_STATE), views, KEY, model) == [(3, [0, 1, 2, 3, 4])]
    views = workers(backlog=[4000, 5000, 5000, 5000], cached=(3,))
    assert plan(rows(5, LONG_STATE), views, KEY, model) == [(3, [0, 1, 2, 3, 4])]
    views = workers(backlog=[3000, 5000, 5000, 5000], cached=(3,))
    assert plan(rows(5, LONG_STATE), views, KEY, model) == [(0, [0, 1, 2, 3, 4])]


def test_single_question_and_disabled_fanout_go_whole(model, expect_error):
    assert plan(rows(1, SHORT_STATE), workers(), KEY, model) == [(0, [0])]
    assert plan(rows(6, SHORT_STATE), workers(), KEY, model, Policy(fanout=False)) == [(0, [0, 1, 2, 3, 4, 5])]
    assert plan(rows(6, SHORT_STATE), workers(backlog=[0, 300, 300, 300]), KEY, model) == [(0, [0, 1, 2, 3, 4, 5])]
    assert plan(rows(6, SHORT_STATE), workers(backlog=[0, 0, 300, 300]), KEY, model) == [(0, [0, 2, 4]), (1, [1, 3, 5])]
    assert plan(rows(6, SHORT_STATE), workers(1), KEY, model) == [(0, [0, 1, 2, 3, 4, 5])]
    with expect_error(ValueError, ""):
        plan([], workers(), KEY, model)
    with expect_error(ValueError, ""):
        plan(rows(1, SHORT_STATE), [], KEY, model)


def test_plan_is_deterministic_and_cheap(model):
    views = workers(backlog=[120, 0, 50, 0], cached=(2,))
    first = plan(rows(7, MID_STATE), views, KEY, model)
    t0 = time.perf_counter()
    for _ in range(200):
        assert plan(rows(7, MID_STATE), views, KEY, model) == first
    assert (time.perf_counter() - t0) / 200 < 0.001


def test_merge_order_and_latency_definition(expect_error):
    shares = [
        ShareResult(2, [2], [[0.2]], True, 100.0),
        ShareResult(0, [0, 4], [[0.0], [0.4]], False, 210.0),
        ShareResult(1, [1, 5], [[0.1], [0.5]], True, 190.0),
        ShareResult(3, [3], [[0.3]], True, 90.0),
    ]
    probs, stats = merge_results(6, shares)
    assert probs == [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    assert stats["latency_ms"] == 210.0 and stats["latency_ms_sum"] == 590.0
    assert stats["prefix_cache_hit"] is False and stats["worker"] == 0 and stats["workers"] == [0, 1, 2, 3]
    assert [s["rows"] for s in stats["shares"]] == [[0, 4], [1, 5], [2], [3]]
    single = merge_results(2, [ShareResult(1, [0, 1], [[1.0], [0.5]], True, 12.34)])[1]
    assert (
        single["latency_ms"] == single["latency_ms_sum"] == 12.3
        and single["prefix_cache_hit"] is True
        and single["worker"] == 1
    )
    with expect_error(ValueError, ""):
        merge_results(3, shares[:1])
    with expect_error(ValueError, ""):
        merge_results(2, [ShareResult(0, [0, 1], [[1.0]], True, 1.0)])


def test_collect_merges_when_every_share_completes():
    futures = [Future(), Future(), Future()]
    done = collect([(0, [0, 3], futures[0]), (1, [1], futures[1]), (2, [2], futures[2])], 4)
    futures[1].set_result(([[1.0]], True, 5.0))
    futures[2].set_result(([[2.0]], False, 7.0))
    assert not done.done()
    futures[0].set_result(([[0.0], [3.0]], True, 9.0))
    probs, stats = done.result(timeout=1)
    assert (
        probs == [[0.0], [1.0], [2.0], [3.0]]
        and stats["latency_ms"] == 9.0
        and stats["latency_ms_sum"] == 21.0
        and stats["prefix_cache_hit"] is False
    )


def test_collect_fails_whole_request_on_first_share_error(expect_error):
    futures = [Future(), Future()]
    done = collect([(0, [0], futures[0]), (1, [1], futures[1])], 2)
    futures[1].set_exception(RuntimeError("device hiccup"))
    with expect_error(RuntimeError, "device hiccup"):
        done.result(timeout=1)
    futures[0].set_result(([[0.0]], True, 1.0))
    with expect_error(RuntimeError, "device hiccup"):
        done.result(timeout=1)
    late = Future()
    late.set_exception(ValueError("already failed"))
    done2 = collect([(0, [0], late)], 1)
    with expect_error(ValueError, ""):
        done2.result(timeout=1)


def test_collect_reports_bad_coverage_as_failure(expect_error):
    f = Future()
    done = collect([(0, [0], f)], 2)
    f.set_result(([[0.0]], True, 1.0))
    with expect_error(ValueError, "shares cover rows"):
        done.result(timeout=1)


ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
BASE = (
    "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404"
)
SIX = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems",
        },
    },
    "return_reason": {
        "type": "choice",
        "instructions": "If the customer wants to return something, why?",
        "criteria": {
            "wrong_size": "The item doesn't fit",
            "wrong_item": "A different product was delivered",
            "damaged": "The item arrived broken or faulty",
            "changed_mind": "The item is fine, the customer no longer wants it",
            "other": "A return reason that fits none of the above",
        },
    },
    "requested_resolution": {
        "type": "choice",
        "instructions": "What does the customer want to happen?",
        "criteria": {
            "exchange": "Swap the item for a different one",
            "refund": "Money back",
            "replacement": "The same item sent again",
            "information": "Just an answer, no action needed",
        },
    },
    "tone": {
        "type": "choice",
        "instructions": "What is the customer's tone?",
        "criteria": {"calm": None, "frustrated": None, "angry": None},
    },
    "escalate": {"type": "noul", "instructions": "Does this message require urgent human attention?"},
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}
TICKET = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"


def fake_server(workers=4, **overrides):
    os.environ["KEV_FAKE_ENGINE"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("KEV_RUN", ADAPTER if os.path.isdir(ADAPTER) else "jaredpalmer/kev-9b")
    os.environ.setdefault("HF_MODEL", BASE if os.path.isdir(BASE) else "Qwen/Qwen3.5-9B-Base")
    from models.autoports.jaredpalmer_kev_9b.tt import server as srv

    if not hasattr(srv.Worker, "view"):
        pytest.skip("tt/server.py does not carry doc/multichip/server_patch.diff yet")
    settings = srv.Settings.from_env()
    settings.fake_workers = workers
    for k, v in overrides.items():
        setattr(settings, k, v)
    return srv, srv.Server.start(settings)


def test_server_fans_out_and_merges_in_request_order():
    srv, s = fake_server(4)
    try:
        req = srv.SystemOneRequest.model_validate({"state": TICKET, "model": "kev-latest", "questions": SIX})
        probs, stats, rws = s.submit(req).result(timeout=30)
        assert stats["workers"] == [0, 1, 2, 3] and [sh["rows"] for sh in stats["shares"]] == [[0, 4], [1, 5], [2], [3]]
        assert stats["latency_ms"] == max(sh["latency_ms"] for sh in stats["shares"])
        assert stats["latency_ms_sum"] == pytest.approx(sum(sh["latency_ms"] for sh in stats["shares"]), abs=0.11)
        assert stats["worker"] == 0 and len(probs) == 6 and [r.qid for r in rws] == list(SIX)
        s.policy = Policy(fanout=False)
        probs_whole, stats_whole, _ = s.submit(req).result(timeout=30)
        assert stats_whole["workers"] == [0] and probs_whole == probs
        assert all(w.backlog_ms == 0.0 for w in s.workers)
        card = s.card()
        assert card["dispatch"]["fanout_requests"] == 1 and all("backlog_ms" in w for w in card["workers"])
        body = s.body(req, probs, stats, rws)
        assert set(body) == {"model", "answers", "usage", "latency_ms"} and list(body["answers"]) == list(SIX)
    finally:
        s.close()


def test_server_share_error_fails_request_and_releases_slots(expect_error):
    srv, s = fake_server(4)
    try:
        req = srv.SystemOneRequest.model_validate(
            {"state": f"Brand new ticket. {TICKET}", "model": "kev-latest", "questions": SIX}
        )
        victim = s.workers[3]
        real = victim.engine.question_hidden

        def broken(*a, **k):
            raise RuntimeError("share failed on worker 3")

        victim.engine.question_hidden = broken
        with expect_error(RuntimeError, "share failed on worker 3"):
            s.submit(req).result(timeout=30)
        deadline = time.time() + 10
        while time.time() < deadline and any(w.load() for w in s.workers):
            time.sleep(0.01)
        assert all(w.backlog_ms == 0.0 and w.load() == 0 for w in s.workers)
        assert all(len(w.cache) + len(w.free_slots) == w.cache_size for w in s.workers)
        victim.engine.question_hidden = real
        probs, stats, _ = s.submit(req).result(timeout=30)
        assert len(probs) == 6 and stats["prefix_cache_hit"] is True
    finally:
        s.close()
