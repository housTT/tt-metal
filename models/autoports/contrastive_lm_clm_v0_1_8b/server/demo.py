# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .demo_runner import run_session

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PACKAGE_DIR, "clm", "demo")
DEMO_FILES = ("index.html", "demo.css", "demo.js")
GRACE_SECONDS = 10.0
STOP_TIMEOUT_S = 5.0
LIMITS = {"seeds": (1, 10), "seed": (0, 1_000_000), "duration": (5, 180), "inflight": (1, 8)}
DEFAULTS = {"seeds": 5, "seed": 0, "duration": 60.0, "shield": True, "inflight": 6}

log = logging.getLogger("clm.server")


def demo_stamp() -> str:
    return format(int(max(os.path.getmtime(os.path.join(DEMO_DIR, f)) for f in DEMO_FILES)), "x")


def demo_html() -> str:
    with open(os.path.join(DEMO_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    v = demo_stamp()
    return html.replace('href="demo.css"', f'href="demo.css?v={v}"').replace('src="demo.js"', f'src="demo.js?v={v}"')


def validate_start(message: dict) -> dict | str:
    config: dict[str, Any] = {}
    for key in ("seeds", "seed", "inflight"):
        raw = message.get(key, DEFAULTS[key])
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or int(raw) != raw:
            return f"{key} must be an integer"
        low, high = LIMITS[key]
        if not low <= int(raw) <= high:
            return f"{key} must be between {low} and {high}"
        config[key] = int(raw)
    raw = message.get("duration", DEFAULTS["duration"])
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return "duration must be a number of seconds"
    low, high = LIMITS["duration"]
    if not low <= float(raw) <= high:
        return f"duration must be between {low} and {high} seconds"
    config["duration"] = float(raw)
    shield = message.get("shield", DEFAULTS["shield"])
    if not isinstance(shield, bool):
        return "shield must be true or false"
    config["shield"] = shield
    key = message.get("api_key")
    if key is not None and not isinstance(key, str):
        return "api_key must be a string"
    config["api_key"] = key or None
    return config


def public_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "api_key"}


class Session:
    def __init__(self, config: dict, base_url: str, loop: asyncio.AbstractEventLoop, on_finish=None):
        self.on_finish = on_finish
        self.config = dict(config)
        self.config["base_url"] = base_url
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.loop = loop
        self.viewers: set[WebSocket] = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.last_course: str | None = None
        self.last_stats: str | None = None
        self.finished = asyncio.Event()
        context = multiprocessing.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(
            target=run_session, args=(self.config, child), name="trex-demo-runner", daemon=False
        )
        self.process.start()
        child.close()
        self.reader = threading.Thread(target=self._read, name="trex-demo-reader", daemon=True)
        self.reader.start()
        self.broadcaster = loop.create_task(self._broadcast())

    def _read(self) -> None:
        while True:
            try:
                raw = self.conn.recv()
            except (EOFError, OSError):
                break
            self.loop.call_soon_threadsafe(self.queue.put_nowait, raw)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)

    async def _broadcast(self) -> None:
        while True:
            raw = await self.queue.get()
            if raw is None:
                break
            batch = [raw]
            while not self.queue.empty():
                nxt = self.queue.get_nowait()
                if nxt is None:
                    self.queue.put_nowait(None)
                    break
                batch.append(nxt)
            frames = [m for m in batch if m.startswith('{"type":"frame"')]
            others = [m for m in batch if not m.startswith('{"type":"frame"')]
            for m in others:
                if m.startswith('{"type":"course"'):
                    self.last_course = m
                elif m.startswith('{"type":"stats"'):
                    self.last_stats = m
            to_send = others + frames[-1:]
            for ws in list(self.viewers):
                for m in to_send:
                    try:
                        await ws.send_text(m)
                    except Exception:
                        self.viewers.discard(ws)
                        break
        self.finished.set()
        if self.on_finish is not None:
            await self.on_finish()

    def is_alive(self) -> bool:
        return self.process.is_alive()

    async def stop(self) -> None:
        try:
            self.conn.send("stop")
        except (BrokenPipeError, OSError):
            pass
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while self.process.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
        try:
            self.conn.close()
        except OSError:
            pass
        try:
            await asyncio.wait_for(self.finished.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self.broadcaster.cancel()


class DemoManager:
    def __init__(self):
        self.session: Session | None = None
        self.grace: asyncio.TimerHandle | None = None
        self.lock = asyncio.Lock()

    def running(self) -> bool:
        return self.session is not None and self.session.is_alive() and not self.session.finished.is_set()

    async def reap(self) -> None:
        session = self.session
        if session is not None and (not session.is_alive() or session.finished.is_set()):
            await session.stop()
            if self.session is session:
                self.session = None

    async def start(self, config: dict, base_url: str) -> Session | None:
        async with self.lock:
            await self.reap()
            if self.session is not None:
                return None
            self.session = Session(config, base_url, asyncio.get_running_loop(), on_finish=self.reap)
            return self.session

    async def stop(self) -> bool:
        async with self.lock:
            if self.session is None:
                return False
            session, self.session = self.session, None
            await session.stop()
            return True

    def attach(self, ws: WebSocket) -> None:
        if self.grace is not None:
            self.grace.cancel()
            self.grace = None
        if self.session is not None:
            self.session.viewers.add(ws)

    def detach(self, ws: WebSocket) -> None:
        if self.session is not None:
            self.session.viewers.discard(ws)
            if not self.session.viewers and self.grace is None:
                loop = asyncio.get_running_loop()
                self.grace = loop.call_later(GRACE_SECONDS, lambda: loop.create_task(self._grace_stop()))

    async def _grace_stop(self) -> None:
        self.grace = None
        if self.session is not None and not self.session.viewers:
            log.info("clm demo: no viewers for %.0f s, stopping the game", GRACE_SECONDS)
            await self.stop()

    async def shutdown(self) -> None:
        if self.grace is not None:
            self.grace.cancel()
            self.grace = None
        await self.stop()


def register_demo(app: FastAPI) -> None:
    manager = DemoManager()
    app.state.demo = manager

    @app.get("/demo", include_in_schema=False)
    def demo_redirect():
        return RedirectResponse(url="/demo/", status_code=307)

    @app.get("/demo/", include_in_schema=False)
    def demo_page():
        return HTMLResponse(demo_html(), headers={"Cache-Control": "no-cache"})

    class RevalidatingDemoStatic(StaticFiles):
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-cache"
            return response

    @app.websocket("/demo/ws")
    async def demo_ws(websocket: WebSocket):
        await websocket.accept()
        await manager.reap()
        manager.attach(websocket)
        session = manager.session
        hello = {
            "type": "hello",
            "running": manager.running(),
            "config": public_config(session.config) if session else None,
            "started_at": session.started_at if session else None,
            "viewers": len(session.viewers) if session else 1,
            "api_key_required": bool(app.state.api_key),
        }
        await websocket.send_text(json.dumps(hello, separators=(",", ":")))
        if session is not None:
            for cached in (session.last_course, session.last_stats):
                if cached:
                    await websocket.send_text(cached)
        try:
            while True:
                message = await websocket.receive_json()
                kind = message.get("type")
                if kind == "ping":
                    await websocket.send_text('{"type":"pong"}')
                elif kind == "stop":
                    stopped = await manager.stop()
                    if not stopped:
                        await websocket.send_text('{"type":"stopped","rows":[]}')
                elif kind == "start":
                    config = validate_start(message)
                    if isinstance(config, str):
                        await websocket.send_text(json.dumps({"type": "error", "error": config}))
                        continue
                    if app.state.api_key and config.get("api_key") != app.state.api_key:
                        await websocket.send_text(json.dumps({"type": "error", "error": "invalid API key"}))
                        continue
                    if app.state.engine is None:
                        await websocket.send_text(json.dumps({"type": "error", "error": "engine not ready"}))
                        continue
                    base_url = os.environ.get("CLM_DEMO_BASE_URL")
                    if not base_url:
                        server = websocket.scope.get("server") or ("127.0.0.1", 8700)
                        base_url = f"http://127.0.0.1:{server[1]}"
                    started = await manager.start(config, base_url)
                    if started is None:
                        current = manager.session
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "busy",
                                    "config": public_config(current.config) if current else None,
                                    "started_at": current.started_at if current else None,
                                }
                            )
                        )
                        continue
                    manager.attach(websocket)
                    log.info(
                        "clm demo: game started seeds=%s duration=%s inflight=%s base_url=%s",
                        config["seeds"],
                        config["duration"],
                        config["inflight"],
                        base_url,
                    )
                else:
                    await websocket.send_text(json.dumps({"type": "error", "error": f"unknown message type {kind!r}"}))
        except WebSocketDisconnect:
            pass
        finally:
            manager.detach(websocket)

    app.mount("/demo", RevalidatingDemoStatic(directory=DEMO_DIR), name="demo")
