# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(PACKAGE_DIR, "demo")
DEMO_FILES = ("index.html", "demo.css", "demo.js")
PRESETS_FILE = os.path.join(DEMO_DIR, "presets.json")
FEED_FILE = os.path.join(DEMO_DIR, "feed_cases.json")
NO_CACHE = {"Cache-Control": "no-cache"}
QTYPES = ("choice", "score", "noul")


def demo_stamp() -> str:
    return format(int(max(os.path.getmtime(os.path.join(DEMO_DIR, f)) for f in DEMO_FILES)), "x")


def demo_html() -> str:
    with open(os.path.join(DEMO_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    v = demo_stamp()
    return html.replace('href="demo.css"', f'href="demo.css?v={v}"').replace('src="demo.js"', f'src="demo.js?v={v}"')


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_presets() -> List[Dict[str, Any]]:
    return load_json(PRESETS_FILE)


def load_feed() -> Dict[str, Any]:
    return load_json(FEED_FILE)


def validate_preset(preset: Dict[str, Any]) -> None:
    for key in ("id", "title", "state", "questions"):
        if key not in preset:
            raise ValueError("preset lacks %r: %s" % (key, preset.get("id")))
    if not isinstance(preset["questions"], dict) or not preset["questions"]:
        raise ValueError("preset %s: questions must be a non-empty object" % preset["id"])
    for qid, q in preset["questions"].items():
        if q.get("type") not in QTYPES or "instructions" not in q:
            raise ValueError("preset %s: question %s is malformed" % (preset["id"], qid))


def validate_feed(feed: Dict[str, Any]) -> None:
    if not isinstance(feed, dict) or "attribution" not in feed or "cases" not in feed:
        raise ValueError("feed must be an object with 'attribution' and 'cases'")
    att = feed["attribution"]
    for key in ("dataset", "config", "split", "revision"):
        if not att.get(key):
            raise ValueError("feed attribution lacks %r" % key)
    cases = feed["cases"]
    if not isinstance(cases, list) or not 40 <= len(cases) <= 100:
        raise ValueError("feed must carry 40 to 100 cases, got %s" % (len(cases) if isinstance(cases, list) else type(cases)))
    seen = set()
    for case in cases:
        for key in ("id", "workflow", "state", "questions", "gold"):
            if key not in case:
                raise ValueError("feed case lacks %r: %s" % (key, case.get("id")))
        if case["id"] in seen:
            raise ValueError("duplicate feed case id %s" % case["id"])
        seen.add(case["id"])
        if set(case["questions"]) != set(case["gold"]):
            raise ValueError("feed case %s: gold keys differ from question keys" % case["id"])
        for qid, q in case["questions"].items():
            if q.get("type") not in QTYPES:
                raise ValueError("feed case %s: question %s has type %r" % (case["id"], qid, q.get("type")))
            if "label" not in case["gold"][qid]:
                raise ValueError("feed case %s: gold for %s lacks a label" % (case["id"], qid))


class RevalidatingStatic(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def register_demo(app: FastAPI) -> None:
    @app.get("/demo", include_in_schema=False)
    def demo_redirect():
        return RedirectResponse(url="/demo/", status_code=307)

    @app.get("/demo/", include_in_schema=False)
    def demo_page():
        return HTMLResponse(demo_html(), headers=NO_CACHE)

    @app.get("/demo/presets.json", include_in_schema=False)
    def demo_presets():
        return JSONResponse(load_presets(), headers=NO_CACHE)

    @app.get("/demo/feed.json", include_in_schema=False)
    def demo_feed():
        return JSONResponse(load_feed(), headers=NO_CACHE)

    app.mount("/demo", RevalidatingStatic(directory=DEMO_DIR), name="demo")
