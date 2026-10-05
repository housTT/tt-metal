import base64
import json
import math
import os
import re
import threading
from pathlib import Path

import pytest

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ["CLEF_FAKE_ENGINE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CLEF_MODEL", SNAPSHOT if os.path.isdir(SNAPSHOT) else "Cloudflare/clef")
os.environ.setdefault("CLEF_FAKE_WORKERS", "1")
os.environ.pop("CLEF_API_KEY", None)
os.environ.pop("CLEF_TRUNCATE_STATES", None)
os.environ.pop("CLEF_MAX_STATE", None)

from fastapi.testclient import TestClient

from models.autoports.cloudflare_clef.tt import server as srv
from models.autoports.cloudflare_clef.tt.api import SystemOneRequest, check_release_rules

pytestmark = pytest.mark.eager_host_side

REFERENCE = Path("/home/hous/dev/clef/reports/reference")
README_EXAMPLES = REFERENCE / "readme_examples.json"
RECORDS_TEXT = REFERENCE / "records_text.jsonl"
REF_TEXT = REFERENCE / "ref_text_bf16.jsonl"
IMAGE = REFERENCE / "images" / "2a7ddcfe4724ee1403a6291d21347162.png"
IMAGE_B = REFERENCE / "images" / "014c6c7b68db49ae19e30a662d90392e.png"

INVOICE = {
    "model": "clef",
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
}
CHECKOUT = {
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}
BLOG = {
    "model": "clef",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Payments, invoices, and refunds",
                "technical": "Outages, errors, and configuration",
                "sales": "Plans and upgrades",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the customer impact?",
            "criteria": ["No impact", "Minor", "Major", "Critical"],
        },
    },
}


def reference_input_tokens():
    out = {}
    if README_EXAMPLES.is_file():
        data = json.loads(README_EXAMPLES.read_text())
        out["invoice"] = data["usage_example"]["input_tokens"]
        out["checkout"] = data["systemone_example"]["response"]["usage"]["input_tokens"]
    if REF_TEXT.is_file():
        for line in REF_TEXT.read_text().splitlines():
            row = json.loads(line)
            if row.get("id") == "blog_support_triage":
                out["blog"] = row["input_tokens"]
    return out


REF_TOKENS = reference_input_tokens()


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


@pytest.fixture(scope="module")
def client():
    with TestClient(srv.app) as c:
        yield c


def post(client, body, **kw):
    r = client.post("/v1/systemone", json=body, **kw)
    assert r.headers["x-typesafe-request-id"]
    return r.status_code, r.json()


def check_body(r, request):
    assert set(r) == {"model", "answers", "usage", "latency_ms"}
    assert r["model"] == request["model"]
    assert r["usage"] == {"input_tokens": r["usage"]["input_tokens"], "output_tokens": 0}
    assert r["usage"]["input_tokens"] > 0 and r["latency_ms"] >= 0
    assert list(r["answers"]) == list(request["questions"])
    for qid, question in request["questions"].items():
        a = r["answers"][qid]
        if question["type"] == "noul":
            assert set(a) == {"type", "noul"} and 0 <= a["noul"] <= 1
        elif question["type"] == "choice":
            assert set(a) == {"type", "choice", "confidence", "probabilities"}
            assert list(a["probabilities"]) == list(question["criteria"])
            assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
            assert a["confidence"] == a["probabilities"][a["choice"]]
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=0.002)
        else:
            levels = [str(i) for i in range(len(question["criteria"]))]
            assert set(a) == {"type", "score", "confidence", "legend", "probabilities"}
            assert a["legend"] == dict(zip(levels, question["criteria"]))
            assert list(a["probabilities"]) == levels
            assert a["confidence"] == max(a["probabilities"].values())
            assert math.isclose(a["score"], sum(int(k) * v for k, v in a["probabilities"].items()), abs_tol=0.002)


@pytest.mark.parametrize(
    "name,request_body",
    [("invoice", INVOICE), ("checkout", CHECKOUT), ("blog", BLOG)],
)
def test_reference_examples_return_release_shaped_bodies(client, name, request_body):
    code, r = post(client, request_body)
    assert code == 200
    check_body(r, request_body)
    if name in REF_TOKENS:
        assert r["usage"]["input_tokens"] == REF_TOKENS[name]


def test_answers_equal_release_systemone_answer(client):
    s = srv.server()
    probs, stats = s.predict(CHECKOUT)
    body = s.answer(SystemOneRequest.model_validate(CHECKOUT))
    release = s.release
    expected = {qid: release.systemone_answer(CHECKOUT["questions"][qid], probs[qid]) for qid in probs}
    assert body["answers"] == expected
    assert set(probs["department"]) == {"billing", "technical"}
    assert set(probs["urgency"]) == {"0", "1", "2"} and set(probs["outage"]) == {"true", "false"}
    assert all(math.isclose(sum(p.values()), 1.0, abs_tol=1e-5) for p in probs.values())
    assert stats["tokens"] == body["usage"]["input_tokens"] and stats["worker"] == 0


def test_option_order_follows_request_criteria_not_sorted_ids(client):
    request = {
        "model": "clef",
        "state": "zeta first",
        "questions": {"q": {"type": "choice", "criteria": {"zeta": "z", "alpha": "a", "mid": None}}},
    }
    code, r = post(client, request)
    assert code == 200
    assert list(r["answers"]["q"]["probabilities"]) == ["zeta", "alpha", "mid"]
    for v in r["answers"]["q"]["probabilities"].values():
        assert round(v, 4) == v


def test_request_id_is_echoed(client):
    r = client.post("/v1/systemone", json=CHECKOUT, headers={"x-typesafe-request-id": "abc123"})
    assert r.status_code == 200 and r.headers["x-typesafe-request-id"] == "abc123"
    assert r.headers["server-timing"].startswith("app;dur=")


def test_same_request_is_deterministic_and_cached(client):
    first = post(client, BLOG)[1]
    second = post(client, BLOG)[1]
    assert first["answers"] == second["answers"]
    card = client.get("/v1/models").json()["models"][0]
    assert card["prefix_cache"]["hits"] >= 1


@pytest.mark.parametrize(
    "body,message",
    [
        ({"state": "x", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
        ({"model": "clef", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
        ({"model": 1, "state": "x", "questions": {"q": {"type": "noul"}}}, "model and state are required"),
        ({"model": "clef", "state": "x", "questions": {}}, "at least one question is required"),
        ({"model": "clef", "state": "x"}, "at least one question is required"),
        (
            {"model": "clef", "state": "x", "questions": {"q": {"type": "bogus"}}},
            "q: type must be noul, choice, or score",
        ),
        (
            {"model": "clef", "state": "x", "questions": {"q": {"instructions": "i"}}},
            "q: type must be noul, choice, or score",
        ),
        (
            {"model": "clef", "state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},
            "q: criteria must not be empty",
        ),
        (
            {"model": "clef", "state": "x", "questions": {"q": {"type": "score", "criteria": []}}},
            "q: criteria must not be empty",
        ),
        ({"model": "clef", "state": "x", "questions": {"q": {"type": "choice"}}}, "q: criteria must not be empty"),
    ],
)
def test_validation_errors_reproduce_release_wording(client, body, message, expect_error):
    code, r = post(client, body)
    assert code == 422
    assert r["detail"] == message
    with expect_error(ValueError, re.escape(message)):
        check_release_rules(body)


def test_release_validation_agrees_with_check_release_rules():
    release = srv.clef_encode.release_module(srv.Settings.from_env().snapshot)
    source = open(release.__file__).read()
    for message in (
        "model and state are required",
        "at least one question is required",
        "type must be noul, choice, or score",
        "criteria must not be empty",
    ):
        assert message in source


def test_noul_criteria_and_null_state(client):
    code, r = post(
        client,
        {
            "model": "clef",
            "state": None,
            "questions": {
                "billing": {
                    "type": "noul",
                    "instructions": "Is this ticket about billing?",
                    "criteria": {"true": "Explicitly about charges", "false": "Not about charges"},
                },
                "urgency": {"type": "score", "criteria": ["today"]},
            },
        },
    )
    assert code == 200
    assert r["answers"]["urgency"] == {
        "type": "score",
        "score": 0.0,
        "confidence": 1.0,
        "legend": {"0": "today"},
        "probabilities": {"0": 1.0},
    }


def test_oversize_state_422(client):
    code, r = post(client, {**CHECKOUT, "state": "lorem " * 20000})
    assert code == 422
    assert "over the 16,384-token limit" in r["detail"] and "CLEF_TRUNCATE_STATES=1" in r["detail"]


def test_oversize_schema_422(client):
    criteria = {f"option_{i}": f"description of option number {i} with some words" for i in range(600)}
    code, r = post(client, {**CHECKOUT, "questions": {"q": {"type": "choice", "criteria": criteria}}})
    assert code == 422
    assert "over the 4,096-token limit" in r["detail"]


def test_truncate_mode_marks_responses():
    settings = srv.Settings.from_env()
    settings.truncate = True
    settings.warmup = False
    s = srv.Server.start(settings)
    try:
        body = s.answer(SystemOneRequest.model_validate({**CHECKOUT, "state": "lorem " * 20000}))
        assert body["truncated"] is True
        assert body["usage"]["state_tokens_used"] == 16384 and body["usage"]["state_tokens"] > 16384
        assert set(body["usage"]) == {"input_tokens", "output_tokens", "state_tokens", "state_tokens_used"}
        handle, _ = list(s.workers[0].cache.values())[-1]
        assert handle.S == 16384 and body["usage"]["input_tokens"] == 16384 + s.workers[0].engine.received[-1]["T"] - (
            handle.S - handle.S0
        )
        short = s.answer(SystemOneRequest.model_validate(CHECKOUT))
        assert short["truncated"] is False and short["usage"]["state_tokens"] == short["usage"]["state_tokens_used"]
    finally:
        s.close()


def test_prefix_cache_eviction_and_slot_return(expect_error):
    settings = srv.Settings.from_env()
    settings.prefix_cache = 2
    settings.warmup = False
    s = srv.Server.start(settings)
    try:
        w = s.workers[0]
        assert w.cache_size == 2
        reqs = [SystemOneRequest.model_validate({**BLOG, "state": f"Ticket {i}. {BLOG['state']}"}) for i in range(3)]
        fresh = [s.answer(r)["answers"] for r in reqs]
        assert (w.hits, w.misses, len(w.cache)) == (0, 3, 2)
        assert s.answer(reqs[2])["answers"] == fresh[2] and (w.hits, w.misses) == (1, 3)
        assert s.answer(reqs[0])["answers"] == fresh[0] and (w.hits, w.misses) == (1, 4)
        assert sorted(slot for _, slot in w.cache.values()) == [0, 1] and w.free_slots == []
        real = w.engine.prefill_state
        calls = {"n": 0}

        def flaky(ids, slot=0, key=None, media=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("device hiccup")
            return real(ids, slot=slot, key=key, media=media)

        w.engine.prefill_state = flaky
        new = SystemOneRequest.model_validate({**BLOG, "state": "A brand new state that is not cached."})
        with expect_error(RuntimeError, "device hiccup"):
            s.answer(new)
        assert len(w.free_slots) == 1 and len(w.cache) == 1
        assert s.answer(new)["answers"]["team"]["type"] == "choice" and len(w.cache) == 2 and w.free_slots == []
    finally:
        s.close()


def test_auth(client, monkeypatch):
    monkeypatch.setattr(srv, "API_KEY", "secret")
    assert client.post("/v1/systemone", json=CHECKOUT).status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/health").status_code == 200
    assert client.get("/health").status_code == 200
    ok = client.post("/v1/systemone", json=CHECKOUT, headers={"authorization": "Bearer secret"})
    assert ok.status_code == 200
    monkeypatch.setattr(srv, "API_KEY", None)
    assert client.post("/v1/systemone", json=CHECKOUT).status_code == 200


@pytest.mark.skipif(not IMAGE.is_file(), reason="reference image missing")
def test_image_request_reaches_engine_with_pixel_values(client):
    engine = srv.server().workers[0].engine
    before = len(engine.received)
    request = {
        "model": "clef",
        "state": "A New Yorker cartoon. Pick the caption that was written for it.",
        "images": [b64(IMAGE)],
        "questions": {
            "caption": {
                "type": "choice",
                "instructions": "Which caption was written for this cartoon?",
                "criteria": {"A": "Get a hammer.", "B": "Cowabunga, dear.", "C": "I was hoping for eternal rest."},
            }
        },
    }
    code, r = post(client, request)
    assert code == 200
    check_body(r, request)
    calls = engine.received[before:]
    prefill = [c for c in calls if c["method"] == "prefill_state"]
    assert len(prefill) == 1 and prefill[0]["media"] is not None
    media = prefill[0]["media"]
    assert media["pixel_values"] == [320, 1536] and media["image_grid_thw"] == [1, 3]
    assert media["mm_token_type_ids"] == 83 and media["token_offset"] == 36
    assert r["usage"]["input_tokens"] > 36 + 80

    data_url = {**request, "images": [f"data:image/png;base64,{b64(IMAGE)}"]}
    code2, r2 = post(client, data_url)
    assert code2 == 200 and r2["answers"] == r["answers"]
    assert engine.received[-1]["method"] == "schema_hidden"
    hits_before = srv.server().workers[0].hits
    post(client, request)
    assert srv.server().workers[0].hits == hits_before + 1


@pytest.mark.skipif(not (IMAGE.is_file() and IMAGE_B.is_file()), reason="reference images missing")
def test_image_bytes_are_part_of_the_cache_key(client):
    w = srv.server().workers[0]
    base = {
        "model": "clef",
        "state": "Pick the caption.",
        "questions": {"caption": {"type": "choice", "criteria": {"A": "one", "B": "two"}}},
    }
    small = __import__("io").BytesIO()
    from PIL import Image

    Image.open(IMAGE).convert("RGB").resize((224, 224)).save(small, format="PNG")
    other = __import__("io").BytesIO()
    Image.open(IMAGE_B).convert("RGB").resize((224, 224)).save(other, format="PNG")
    a = {**base, "images": [base64.b64encode(small.getvalue()).decode()]}
    b = {**base, "images": [base64.b64encode(other.getvalue()).decode()]}
    misses = w.misses
    ra = post(client, a)[1]
    rb = post(client, b)[1]
    assert w.misses == misses + 2
    assert ra["usage"]["input_tokens"] == rb["usage"]["input_tokens"]
    assert ra["answers"] != rb["answers"]


@pytest.mark.skipif(not IMAGE.is_file(), reason="reference image missing")
def test_media_kwargs_are_part_of_the_cache_key(client):
    w = srv.server().workers[0]
    base = {
        "model": "clef",
        "state": "Pick the caption.",
        "images": [b64(IMAGE)],
        "questions": {"caption": {"type": "choice", "criteria": {"A": "one", "B": "two"}}},
    }
    misses = w.misses
    plain = post(client, base)
    assert plain[0] == 200
    with_kwargs = post(client, {**base, "media_kwargs": {"do_normalize": False}})
    assert with_kwargs[0] == 200
    assert w.misses == misses + 2
    assert plain[1]["usage"]["input_tokens"] == with_kwargs[1]["usage"]["input_tokens"]
    hits = w.hits
    post(client, {**base, "media_kwargs": {"do_normalize": False}})
    assert w.hits == hits + 1


@pytest.mark.skipif(not IMAGE.is_file(), reason="reference image missing")
def test_traced_server_refuses_grids_outside_the_warm_list(client):
    w = srv.server().workers[0]
    request = {
        "model": "clef",
        "state": "Pick the caption.",
        "images": [b64(IMAGE)],
        "questions": {"caption": {"type": "choice", "criteria": {"A": "one", "B": "two"}}},
    }
    assert w.warm_grids is None
    assert client.get("/v1/models").json()["models"][0]["traced_media"] is None
    try:
        w.warm_grids = [(1, 22, 38), (2, 16, 20)]
        code, r = post(client, request)
        assert code == 422
        assert r["detail"].startswith(
            "image grid (1, 16, 20) (t, h, w patches) is not in this traced server's warm list"
        )
        assert "1,22,38; 2,16,20" in r["detail"] and "CLEF_TRACED=0" in r["detail"]
        card = client.get("/v1/models").json()["models"][0]
        assert card["traced_media"]["warm_grids"] == [[1, 22, 38], [2, 16, 20]]
        assert "422" in card["traced_media"]["rule"]
        w.warm_grids = [(1, 16, 20), (1, 22, 38)]
        code, r = post(client, request)
        assert code == 200
        check_body(r, request)
        video = {**request, "images": None, "videos": [[b64(IMAGE), b64(IMAGE)]]}
        code, r = post(client, video)
        assert code == 200
        w.warm_grids = [(1, 22, 38)]
        code, r = post(client, video)
        assert code == 422 and r["detail"].startswith("image grid (1, 16, 20)")
    finally:
        w.warm_grids = None


@pytest.mark.skipif(not IMAGE.is_file(), reason="reference image missing")
def test_traced_server_refuses_more_than_one_grid_per_request(client):
    w = srv.server().workers[0]
    engine = w.engine
    two_images = {
        "model": "clef",
        "state": "Two cartoons. Pick the caption.",
        "images": [b64(IMAGE), b64(IMAGE)],
        "questions": {"caption": {"type": "choice", "criteria": {"A": "one", "B": "two"}}},
    }
    two_videos = {**two_images, "images": None, "videos": [[b64(IMAGE)], [b64(IMAGE)]]}
    assert w.warm_grids is None
    before = len(engine.received)
    code, r = post(client, two_images)
    assert code == 200
    check_body(r, two_images)
    prefill = [c for c in engine.received[before:] if c["method"] == "prefill_state"]
    assert len(prefill) == 1 and prefill[0]["media"]["image_grid_thw"] == [2, 3]
    code, r = post(client, two_videos)
    assert code == 200
    try:
        w.warm_grids = [(1, 16, 20)]
        calls = len(engine.received)
        for body in (two_images, two_videos):
            code, r = post(client, body)
            assert code == 422
            assert r["detail"].startswith("this request carries 2 image or video grids")
            assert "one grid per request" in r["detail"] and "CLEF_TRACED=0" in r["detail"]
        assert len(engine.received) == calls
        rule = client.get("/v1/models").json()["models"][0]["traced_media"]["rule"]
        assert rule.startswith("one image or video grid per request")
        code, r = post(client, {**two_images, "images": [b64(IMAGE)]})
        assert code == 200
    finally:
        w.warm_grids = None


def test_engine_value_error_is_a_422(client):
    w = srv.server().workers[0]
    real = w.engine.prefill_state

    def refuse(ids, slot=0, key=None, media=None):
        raise ValueError("image grid (1, 2, 3) is not in this traced server's warm list []")

    w.engine.prefill_state = refuse
    try:
        code, r = post(client, {**BLOG, "state": "A state the cache has never seen."})
        assert code == 422 and r["detail"].startswith("image grid (1, 2, 3)")
        assert len(w.free_slots) + len(w.cache) == w.cache_size
        assert sorted(w.free_slots + [slot for _, slot in w.cache.values()]) == list(range(w.cache_size))
    finally:
        w.engine.prefill_state = real


@pytest.mark.skipif(not IMAGE.is_file(), reason="reference image missing")
def test_video_request_reaches_engine_with_video_pixel_values(client):
    engine = srv.server().workers[0].engine
    before = len(engine.received)
    frame = b64(IMAGE)
    request = {
        "model": "clef",
        "state": "Two frames of a cartoon.",
        "videos": [[frame, frame]],
        "questions": {"moving": {"type": "noul", "instructions": "Does anything move?"}},
    }
    code, r = post(client, request)
    assert code == 200
    media = [c for c in engine.received[before:] if c["method"] == "prefill_state"][0]["media"]
    assert "pixel_values_videos" in media and media["video_grid_thw"] == [1, 3]


def test_bad_image_payloads_422(client):
    base = {"model": "clef", "state": "x", "questions": {"q": {"type": "noul"}}}
    code, r = post(client, {**base, "images": ["not base64!!"]})
    assert code == 422 and r["detail"].startswith("images[0]")
    code, r = post(client, {**base, "images": [base64.b64encode(b"plain text").decode()]})
    assert code == 422 and "not a decodable image" in r["detail"]
    code, r = post(client, {**base, "images": ["data:image/png,abc"]})
    assert code == 422 and "base64" in r["detail"]
    code, r = post(client, {**base, "videos": [[]]})
    assert code == 422 and r["detail"].startswith("videos[0]")


def test_remote_images_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("CLEF_ALLOW_REMOTE_IMAGES", "0")
    body = {
        "model": "clef",
        "state": "x",
        "images": ["https://example.invalid/a.png"],
        "questions": {"q": {"type": "noul"}},
    }
    code, r = post(client, body)
    assert code == 422 and "CLEF_ALLOW_REMOTE_IMAGES=0" in r["detail"]


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    cards = r.json()["models"]
    assert [c["name"] for c in cards] == ["clef"]
    c = cards[0]
    assert c["backend"] == "fake" and c["max_state_tokens"] == 16384 and c["truncate_states"] is False
    assert c["weights"]["repo"] == "Cloudflare/clef" and c["weights"]["revision"] == srv.DEFAULT_REVISION
    assert set(c["prefix_cache"]) == {"size", "hits", "misses", "cached_states"}
    assert len(c["workers"]) == 1 and c["workers"][0]["media"] is True
    assert c["mesh_shape"] == "1x2" and "precision" in c
    assert c["traced"] is False and c["mode"] == "eager" and c["traced_media"] is None
    assert c["prefix_planner"] is True and c["gdn_conv"] is None


def test_health(client):
    for path in ("/health", "/v1/health"):
        r = client.get(path)
        assert r.status_code == 200 and r.json() == {"status": "ok", "workers": 1, "queued": 0}


def test_concurrent_requests(client):
    codes = []

    def one(i):
        codes.append(post(client, {**BLOG, "state": f"Ticket {i}. {BLOG['state']}"})[0])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 6
    assert all(w["queued"] == 0 for w in client.get("/v1/models").json()["models"][0]["workers"])


def test_two_fake_workers_spread_requests():
    settings = srv.Settings.from_env()
    settings.fake_workers = 2
    settings.warmup = False
    s = srv.Server.start(settings)
    try:
        assert len(s.workers) == 2
        for i in range(4):
            s.answer(SystemOneRequest.model_validate({**BLOG, "state": f"Ticket {i}."}))
        assert sum(w.requests for w in s.workers) == 4
        first = s.workers[0].requests
        s.answer(SystemOneRequest.model_validate({**BLOG, "state": "Ticket 0."}))
        assert s.workers[0].requests + s.workers[1].requests == 5
        assert s.workers[0].hits + s.workers[1].hits == 1
    finally:
        s.close()


@pytest.mark.parametrize(
    "shape,mesh_device,expected",
    [
        ("1x2", "P150x2", (1, 2)),
        ("1x4", "P150x4", (1, 4)),
        ("2x2", None, (2, 2)),
        (None, "P150x2", (1, 2)),
        (None, "P300", (1, 2)),
        (None, "P150x4", (1, 4)),
        (None, "P300x2", (1, 4)),
        (None, "QB2", (2, 2)),
        (None, None, (1, 2)),
        ("", "", (1, 2)),
    ],
)
def test_parse_mesh_shape(shape, mesh_device, expected):
    assert srv.parse_mesh_shape(shape, mesh_device) == expected


def test_mesh_plan_table(expect_error):
    assert srv.mesh_plan((1, 2), 2) == {"fabric": "FABRIC_1D", "open": (1, 2), "submeshes": [], "parent": None}
    assert srv.mesh_plan((1, 2), 4) == {
        "fabric": "FABRIC_1D",
        "open": (1, 4),
        "submeshes": [((1, 2), (0, 0))],
        "parent": "1x4",
    }
    assert srv.mesh_plan((1, 2), 4, offset=(0, 2))["submeshes"] == [((1, 2), (0, 2))]
    assert srv.mesh_plan((1, 2), 2, parent="2x2") == {
        "fabric": "FABRIC_2D",
        "open": (2, 2),
        "submeshes": [((1, 2), (0, 0))],
        "parent": "2x2",
    }
    assert srv.mesh_plan((1, 4), 4)["submeshes"] == [((1, 2), (0, 0)), ((1, 2), (0, 2))]
    assert srv.mesh_plan((2, 2), 4)["fabric"] == "FABRIC_2D"
    with expect_error(ValueError, "CLEF_PARENT_MESH="):
        srv.mesh_plan((1, 2), 4, parent="3x3")
    with expect_error(ValueError, "unsupported mesh shape"):
        srv.mesh_plan((1, 1), 1)
    assert srv.parse_offset("0,2") == (0, 2) and srv.parse_offset(None) == (0, 0)


def test_build_engine_filters_kwargs():
    def init(self, mesh, max_state_len=1, snapshot_slots=1):
        return None

    assert srv.accepted_kwargs(init, dict(max_state_len=5, traced=True, snapshot_slots=2)) == {
        "max_state_len": 5,
        "snapshot_slots": 2,
    }

    def init_kw(self, mesh, **kw):
        return None

    assert srv.accepted_kwargs(init_kw, dict(traced=True)) == {"traced": True}
