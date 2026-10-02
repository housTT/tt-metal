# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from models.autoports.contrastive_lm_clm_v0_1_8b.clm.engine import Engine
from models.autoports.contrastive_lm_clm_v0_1_8b.server import demo_runner
from models.autoports.contrastive_lm_clm_v0_1_8b.server.app import create_app
from models.autoports.contrastive_lm_clm_v0_1_8b.server.demo import validate_start
from models.autoports.contrastive_lm_clm_v0_1_8b.server.trex.engine import Game
from models.autoports.contrastive_lm_clm_v0_1_8b.tests.test_host_engine import CKPT, FakeEmbedderWithIds, needs_ckpt

PLAY_ROW_KEYS = (
    "seed",
    "survived",
    "deaths",
    "best_score",
    "scores",
    "decisions",
    "agreement_with_planner",
    "latency_ms_p50",
    "latency_ms_p95",
    "model_ms_p50",
    "answers_discarded",
    "errors",
    "last_error",
    "best_effort_decisions",
    "shield_interventions",
    "arrival_saves",
    "emergency_saves",
    "input_tokens",
    "game_seconds",
    "wall_seconds",
    "host_stall_seconds_dropped",
    "model",
    "endpoint",
)


class StubPilot:
    active = set()
    held = "run"
    last_action = "run"

    def spectator(self, frame):
        return {"event": "Ready"}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def recv_until(ws, kinds, limit_s=60.0):
    deadline = time.time() + limit_s
    seen = []
    while time.time() < deadline:
        message = json.loads(ws.recv())
        seen.append(message)
        if message["type"] in kinds:
            return message, seen
    raise AssertionError(f"no {kinds} within {limit_s} s; last types {[m['type'] for m in seen[-5:]]}")


@pytest.fixture(scope="module")
def engine():
    if not os.path.isfile(CKPT):
        pytest.skip(f"checkpoint missing: {CKPT}")
    return Engine(FakeEmbedderWithIds(), checkpoint=CKPT, device="cpu", action_cache="64MiB")


@pytest.fixture(scope="module")
def live_server(engine):
    import uvicorn

    port = free_port()
    app = create_app(engine, warmup=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield app, port
    server.should_exit = True
    thread.join(10)


def test_encode_frame_shape():
    game = Game(seed=0)
    game.press_jump()
    for _ in range(600):
        game.step()
    frame = demo_runner.encode_frame(game, StubPilot(), 600, [], [])
    text = json.dumps(frame)
    back = json.loads(text)
    g = back["g"]
    assert back["type"] == "frame" and back["f"] == 600
    assert len(g["t"]) == 5 and 0 <= g["t"][2] <= 4
    assert all(len(o) == 6 and 0 <= o[1] <= 2 for o in g["o"])
    assert len(g["c"]) <= 6
    assert isinstance(g["hi"], int) and isinstance(g["dead"], int) and g["dead"] >= 0
    assert g["rv"] == 600.0 and g["pl"] in (0, 1)


def test_course_row_keys_match_the_upstream_row():
    assert demo_runner.ROW_KEYS[: len(PLAY_ROW_KEYS)] == PLAY_ROW_KEYS
    assert demo_runner.ROW_KEYS[len(PLAY_ROW_KEYS) :] == ("server_ms_p50", "server_ms_p95")


def test_validate_start():
    assert validate_start({})["seeds"] == 5
    assert validate_start({"inflight": 99}) == "inflight must be between 1 and 8"
    assert validate_start({"duration": 1}) == "duration must be between 5 and 180 seconds"
    assert validate_start({"seeds": 2.5}) == "seeds must be an integer"
    assert validate_start({"shield": "yes"}) == "shield must be true or false"
    assert validate_start({"api_key": "k", "duration": 7})["api_key"] == "k"


@needs_ckpt
def test_demo_page_and_static(engine):
    with TestClient(create_app(engine, warmup=False)) as c:
        r = c.get("/demo", follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"] == "/demo/"
        r = c.get("/demo/")
        assert r.status_code == 200 and "demo.js?v=" in r.text and "T-Rex" in r.text
        r = c.get("/demo/demo.js")
        assert (
            r.status_code == 200
            and r.headers["cache-control"] == "no-cache"
            and "/demo/ws" not in r.text
            and "'ws'" in r.text
        )
        r = c.get("/demo/reference_rtx4090.json")
        assert r.status_code == 200 and r.json()["summary"]["survived"] == 5
        assert c.get("/").status_code == 200
    with TestClient(create_app(engine, ui=False, warmup=False)) as c:
        assert c.get("/demo/").status_code == 404
    with TestClient(create_app(engine, demo=False, warmup=False)) as c:
        assert c.get("/demo/").status_code == 404 and c.get("/demo/demo.js").status_code == 404
        assert c.get("/").status_code == 200


@needs_ckpt
def test_demo_ws_validation_and_api_key(engine, monkeypatch):
    monkeypatch.setenv("CLM_DEMO_BASE_URL", f"http://127.0.0.1:{free_port()}")
    app = create_app(engine, api_key="secret", warmup=False)
    with TestClient(app) as c:
        with c.websocket_connect("/demo/ws") as ws:
            hello = ws.receive_json()
            assert hello["running"] is False and hello["api_key_required"] is True
            ws.send_json({"type": "start", "inflight": 99, "api_key": "secret"})
            assert "inflight" in ws.receive_json()["error"]
            ws.send_json({"type": "start", "api_key": "wrong"})
            assert ws.receive_json()["error"] == "invalid API key"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["type"] == "stopped"


@needs_ckpt
def test_demo_short_game_end_to_end(live_server, monkeypatch):
    from websockets.sync.client import connect

    app, port = live_server
    monkeypatch.setenv("CLM_DEMO_BASE_URL", f"http://127.0.0.1:{port}")
    with connect(f"ws://127.0.0.1:{port}/demo/ws", max_size=None) as ws:
        hello = json.loads(ws.recv())
        assert hello["running"] is False
        ws.send(json.dumps({"type": "start", "seeds": 1, "duration": 5, "inflight": 2, "shield": True}))
        course, _ = recv_until(ws, {"course", "error"}, 30)
        assert course["type"] == "course" and course["seed"] == 0 and len(course["warm_ms"]) == 4
        with connect(f"ws://127.0.0.1:{port}/demo/ws", max_size=None) as other:
            assert json.loads(other.recv())["running"] is True
            other.send(json.dumps({"type": "start", "seeds": 1, "duration": 5}))
            busy, _ = recv_until(other, {"busy"}, 10)
            assert busy["config"]["duration"] == 5.0
        summary, seen = recv_until(ws, {"summary", "error", "stopped"}, 60)
    assert summary["type"] == "summary", summary
    frames = [m for m in seen if m["type"] == "frame"]
    decisions = [d for m in frames for d in m["d"] if d["dropped"] is None and d["error"] is None]
    assert len(frames) >= 200
    assert len(decisions) >= 5
    for d in decisions[:20]:
        assert abs(sum(d["p"].values()) - 1.0) < 1e-3
        assert d["latency_ms"] >= d["inference_ms"] >= 0
        assert d["state"].startswith("Dino runner game.") and set(d["criteria"]) == {"jump", "duck", "run"}
    assert any(isinstance(d["server_ms"], (int, float)) for d in decisions)
    assert any(m["type"] == "stats" for m in seen)
    row = next(m["row"] for m in seen if m["type"] == "course_end")
    assert row["decisions"] >= 5 and row["errors"] == 0 and row["game_seconds"] == 5.0
    assert set(row) == set(demo_runner.ROW_KEYS)
    assert summary["summary"]["seeds"] == 1 and summary["summary"]["mean_decisions"] == row["decisions"]
    deadline = time.time() + 5
    while app.state.demo.session is not None and time.time() < deadline:
        time.sleep(0.1)
    assert app.state.demo.session is None


@needs_ckpt
def test_demo_stop_mid_game(live_server, monkeypatch):
    from websockets.sync.client import connect

    app, port = live_server
    monkeypatch.setenv("CLM_DEMO_BASE_URL", f"http://127.0.0.1:{port}")
    with connect(f"ws://127.0.0.1:{port}/demo/ws", max_size=None) as ws:
        json.loads(ws.recv())
        ws.send(json.dumps({"type": "start", "seeds": 2, "duration": 30, "inflight": 2}))
        frames = 0
        while frames < 30:
            frames += json.loads(ws.recv())["type"] == "frame"
        session = app.state.demo.session
        assert session is not None and session.is_alive()
        t0 = time.time()
        ws.send(json.dumps({"type": "stop"}))
        stopped, _ = recv_until(ws, {"stopped", "error"}, 10)
    assert stopped["type"] == "stopped" and time.time() - t0 < 6
    deadline = time.time() + 8
    while (app.state.demo.session is not None or session.is_alive()) and time.time() < deadline:
        time.sleep(0.1)
    assert app.state.demo.session is None
    assert not session.is_alive()
