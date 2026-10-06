# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .decode import aggregate_usage, check_min_confidence
from .demo import register_demo
from .engine import BadRequest, Engine, LimitError, RequestError, check_batch_limits, check_request_limits, env_flag

LATENCY_HEADER = "X-Inference-Time-Ms"
DEVICE_HEADER = "X-Laya-Device-Ms"
BATCH_HEADER = "X-Laya-Batch"
DEFAULT_PORT = 8710
BODY_REFUSALS = ("hooks", "on_predict_start", "on_predict_end", "hooks_raise", "hooks_timeout")
SANITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_reference.json")
STATE_EN = {
    "ticket": {
        "subject": "Payout failing",
        "messages": [
            {
                "from": "customer",
                "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6,
            }
        ],
    }
}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"},
}
SANITY_QUESTIONS = {"routing": Q_CHOICE}

log = logging.getLogger("laya.server")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def sanity_reference_path(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("LAYA_SANITY_REFERENCE") or SANITY_FILE


def sanity_check(engine: Engine, path: Optional[str] = None) -> Dict[str, Any]:
    path = sanity_reference_path(path)
    results, meta = engine.predict_batch([STATE_EN], SANITY_QUESTIONS)
    answer = results[0]["answers"]["routing"]
    out: Dict[str, Any] = {
        "state": "STATE_EN",
        "question": "Q_CHOICE",
        "choice": answer["choice"],
        "probabilities": answer["probabilities"],
        "device_ms": round(meta["device_ms"], 2),
        "batches": meta["batches"],
        "reference_file": path,
        "reference_choice": None,
        "max_abs_dp": None,
        "ok": None,
    }
    if not os.path.isfile(path):
        log.warning("sanity reference %s missing; startup check recorded without comparison", path)
        return out
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)
    ref_answer = ref["answers"]["routing"]
    out["reference_choice"] = ref_answer["choice"]
    out["reference_backend"] = ref.get("produced_by", {}).get("backend")
    dps = [abs(answer["probabilities"][k] - ref_answer["probabilities"].get(k, 0.0)) for k in answer["probabilities"]]
    out["max_abs_dp"] = round(max(dps), 4) if dps else None
    out["ok"] = answer["choice"] == ref_answer["choice"]
    out["reference_model"] = ref.get("produced_by", {}).get("model") or ref.get("produced_by", {}).get("model_dir")
    if out["ok"]:
        log.info(
            "sanity check ok: STATE_EN routing -> %s (max |dp| %.4f vs stored CPU value in %s)",
            answer["choice"],
            out["max_abs_dp"],
            os.path.basename(path),
        )
    else:
        log.warning(
            "sanity check argmax mismatch: served %s, stored CPU value %s in %s (max |dp| %.4f)",
            answer["choice"],
            ref_answer["choice"],
            os.path.basename(path),
            out["max_abs_dp"],
        )
    return out


def create_app(
    engine: Optional[Engine] = None,
    api_key: Optional[str] = None,
    demo: Optional[bool] = None,
    raw_forward: Optional[bool] = None,
    load: bool = True,
    sanity: Optional[bool] = None,
    sanity_reference: Optional[str] = None,
) -> FastAPI:
    api_key = api_key if api_key is not None else (os.environ.get("LAYA_API_KEY") or None)
    demo = demo if demo is not None else not env_flag("LAYA_NO_DEMO")
    raw_forward = raw_forward if raw_forward is not None else env_flag("LAYA_RAW_FORWARD")
    sanity = sanity if sanity is not None else env_flag("LAYA_SANITY_CHECK", True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-infer")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.lock = asyncio.Lock()
        app.state.started = time.time()
        if app.state.engine is None and load:
            t0 = time.perf_counter()
            app.state.engine = Engine.from_env()
            app.state.owns_engine = True
            log.info("laya model loaded in %.1f s from %s", time.perf_counter() - t0, app.state.engine.model_dir)
        eng = app.state.engine
        if eng is not None:
            info = eng.info()
            if sanity:
                try:
                    app.state.sanity = sanity_check(eng, sanity_reference)
                except Exception as e:
                    log.warning("sanity check failed to run: %s", e)
                    app.state.sanity = {"ok": None, "error": str(e)}
            log.info(
                "laya server ready backend=%s precision=%s mesh=%s seq_buckets=%s row_buckets=%s warm=%d raw_forward=%s demo=%s",
                info["backend"],
                info["precision"],
                info["mesh_shape"],
                info["seq_buckets"],
                info["row_buckets"],
                len(info["warm_shapes"] or []),
                raw_forward,
                demo,
            )
        yield
        if app.state.engine is not None and app.state.owns_engine:
            try:
                app.state.engine.close()
            except Exception as e:
                log.warning("engine close failed: %s", e)
        pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Laya System One API", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.owns_engine = False
    app.state.sanity = None
    app.state.api_key = api_key
    app.state.raw_forward = raw_forward
    app.state.lock = None
    app.state.started = time.time()
    expected = ("Bearer " + api_key).encode("utf-8", "surrogateescape") if api_key else b""

    if env_flag("LAYA_CORS"):
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            allow_credentials=False,
            expose_headers=[LATENCY_HEADER, DEVICE_HEADER, BATCH_HEADER],
        )

    def authorized(authorization: Optional[str]) -> bool:
        if api_key is None:
            return True
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        return hmac.compare_digest(supplied, expected)

    def auth(authorization: Optional[str]) -> None:
        if not authorized(authorization):
            raise HTTPException(401, "invalid or missing bearer token")

    def current_engine() -> Engine:
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "model loading")
        return eng

    async def json_body(request: Request) -> Any:
        try:
            return await request.json()
        except Exception:
            raise HTTPException(400, "request body must be valid JSON")

    def refuse_body_keys(body: Dict[str, Any]) -> None:
        given = sorted(k for k in BODY_REFUSALS if k in body and body[k] is not None)
        if given:
            raise HTTPException(
                422, "%s run inside the server process and cannot be sent to this endpoint" % ", ".join(given)
            )

    def min_confidence_of(body: Dict[str, Any]):
        if body.get("min_confidence") is None:
            return None
        try:
            return check_min_confidence(body["min_confidence"])
        except ValueError as e:
            raise HTTPException(422, str(e))

    def budget_of(eng: Engine, body: Dict[str, Any]):
        try:
            return eng.budget(body)
        except RequestError as e:
            raise HTTPException(422, str(e))

    def check_bool_param(body: Dict[str, Any], key: str) -> None:
        if key in body and body[key] is not None and not isinstance(body[key], bool):
            raise HTTPException(422, "%s must be a boolean" % key)

    def check_int_param(body: Dict[str, Any], key: str) -> None:
        val = body.get(key)
        if val is None:
            return
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise HTTPException(422, "%s must be a positive integer" % key)

    def to_http(e: Exception) -> HTTPException:
        if isinstance(e, HTTPException):
            return e
        if isinstance(e, (RequestError, LimitError, BadRequest)):
            return HTTPException(e.status, str(e))
        return HTTPException(422, str(e)) if isinstance(e, ValueError) else HTTPException(500, "inference failed")

    async def run_locked(fn, *args, **kwargs):
        lock = app.state.lock
        if lock is None:
            app.state.lock = lock = asyncio.Lock()
        async with lock:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            try:
                out = await loop.run_in_executor(pool, partial(fn, *args, **kwargs))
            except Exception as e:
                http = to_http(e)
                if http.status_code == 500:
                    log.exception("inference failed")
                raise http
            return out, (time.perf_counter() - t0) * 1000.0

    def respond(payload: Any, infer_ms: float, meta: Dict[str, Any]) -> JSONResponse:
        headers = {
            LATENCY_HEADER: f"{infer_ms:.2f}",
            "Server-Timing": f"inference;dur={infer_ms:.2f}",
            DEVICE_HEADER: f"{meta.get('device_ms', 0.0):.2f}",
            BATCH_HEADER: ",".join(meta.get("batches") or []) or "none",
        }
        return JSONResponse(payload, headers=headers)

    @app.middleware("http")
    async def latency_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/v1/") and LATENCY_HEADER not in response.headers:
            response.headers[LATENCY_HEADER] = f"{(time.perf_counter() - t0) * 1000:.2f}"
        return response

    @app.get("/health")
    def health():
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"status": "loading", "ready": False}, status_code=503)
        return {
            "status": "ok",
            "ready": True,
            "backend": eng.backend_kind,
            "model": eng.hf_model,
            "revision": eng.revision,
        }

    @app.get("/v1/health")
    def v1_health(authorization: Optional[str] = Header(default=None)):
        eng = app.state.engine
        if eng is None:
            return JSONResponse(
                {"status": "loading", "ready": False, "backend": os.environ.get("LAYA_BACKEND") or "tt"},
                status_code=503,
            )
        out = {"status": "ok", "ready": True, "uptime_s": round(time.time() - app.state.started, 1)}
        out.update(eng.info())
        out["sanity"] = app.state.sanity
        out["raw_forward"] = app.state.raw_forward
        out["api_key_required"] = bool(api_key)
        if not authorized(authorization):
            out = {"status": "ok", "ready": True, "backend": eng.backend_kind, "api_key_required": True}
        return out

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        return {
            "object": "list",
            "data": [
                {
                    "id": eng.hf_model,
                    "object": "model",
                    "owned_by": "convaiinnovations",
                    "revision": eng.revision,
                    "backend": eng.backend_kind,
                    "precision": eng.shapes.get("precision"),
                    "max_len": eng.max_len,
                    "head_max_len": eng.head_max_len,
                    "seq_buckets": eng.buckets.seqs,
                    "row_buckets": eng.buckets.rows,
                }
            ],
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        body = await json_body(request)
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(400, "request body must be an object with a 'questions' field")
        state = body.get("state")
        questions = body["questions"]
        try:
            check_request_limits(state, questions, eng.limits)
        except (BadRequest, LimitError, RequestError) as e:
            raise HTTPException(e.status, str(e))
        refuse_body_keys(body)
        max_len, head_max_len = budget_of(eng, body)
        min_confidence = min_confidence_of(body)
        (results, meta), infer_ms = await run_locked(
            eng.predict_batch,
            [state],
            questions,
            max_len=max_len,
            head_max_len=head_max_len,
            min_confidence=min_confidence,
        )
        return respond(results[0], infer_ms, meta)

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request, authorization: Optional[str] = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        body = await json_body(request)
        if not isinstance(body, dict) or "questions" not in body or "states" not in body:
            raise HTTPException(400, "request body must be an object with 'states' and 'questions' fields")
        states = body["states"]
        questions = body["questions"]
        try:
            check_batch_limits(states, questions, eng.limits)
        except (BadRequest, LimitError, RequestError) as e:
            raise HTTPException(e.status, str(e))
        refuse_body_keys(body)
        max_len, head_max_len = budget_of(eng, body)
        min_confidence = min_confidence_of(body)
        check_int_param(body, "batch_size")
        check_bool_param(body, "sort_by_length")
        (results, meta), infer_ms = await run_locked(
            eng.predict_batch,
            states,
            questions,
            max_len=max_len,
            head_max_len=head_max_len,
            min_confidence=min_confidence,
        )
        payload = {"results": results, "total_usage": aggregate_usage(results)}
        return respond(payload, infer_ms, meta)

    if raw_forward:

        @app.post("/v1/forward")
        async def forward(request: Request, authorization: Optional[str] = Header(default=None)):
            auth(authorization)
            eng = current_engine()
            body = await json_body(request)
            if not isinstance(body, dict):
                raise HTTPException(400, "request body must be an object")
            (out, meta), infer_ms = await run_locked(eng.raw_forward, body)
            return respond(out, infer_ms, meta)

    if demo:
        register_demo(app)

    return app


app = create_app()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m models.autoports.convaiinnovations_laya.server")
    ap.add_argument("--host", default=os.environ.get("LAYA_HOST", "0.0.0.0"))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT") or os.environ.get("LAYA_PORT") or DEFAULT_PORT)
    )
    ap.add_argument("--backend", choices=("tt", "cpu"), default=None)
    ap.add_argument("--log-level", default=os.environ.get("LAYA_LOG_LEVEL", "info"))
    args = ap.parse_args(argv)
    if args.backend:
        os.environ["LAYA_BACKEND"] = args.backend
    import uvicorn

    log.info(
        "laya server starting backend=%s host=%s port=%d", os.environ.get("LAYA_BACKEND", "tt"), args.host, args.port
    )
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level, lifespan="on")
