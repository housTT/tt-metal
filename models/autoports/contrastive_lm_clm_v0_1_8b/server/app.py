# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..clm.embedder import EmbedderError
from ..clm.engine import DEFAULT_MODEL, Engine, ModelNotFound
from ..clm.heads import HIDDEN, default_device

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PACKAGE_DIR, "clm", "static")
UI_FILES = ("index.html", "app.css", "app.js")
DEFAULT_CHECKPOINT = "/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt"
DEFAULT_ACTION_CACHE = "512MiB"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_PORT = 8700
DEFAULT_EMBEDDER = "tt"
DEFAULT_EMBEDDING_MODEL = "qwen3-8b"
EMBEDDER_KINDS = ("tt", "hf", "http")
LATENCY_HEADER = "X-CLM-Latency-Ms"
WARMUP_TEXT = "clm server warm-up"

log = logging.getLogger("clm.server")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def make_embedder(kind: str | None = None, max_tokens: int | None = None) -> tuple[Any, str]:
    kind = (kind or os.environ.get("CLM_EMBEDDER") or DEFAULT_EMBEDDER).lower()
    max_tokens = max_tokens or env_int("CLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    if kind == "tt":
        try:
            from ..tt.encoder import TtQwen3Encoder
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(".tt.encoder"):
                raise ImportError(
                    "CLM_EMBEDDER=tt needs models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder "
                    "(TtQwen3Encoder.from_env); set CLM_EMBEDDER=hf or CLM_EMBEDDER=http instead"
                ) from e
            raise
        return TtQwen3Encoder.from_env(), kind
    if kind == "hf":
        from ..reference.hf_embedder import HfQwen3Embedder

        return HfQwen3Embedder(max_tokens=max_tokens), kind
    if kind == "http":
        from ..clm.embedder import Embedder

        return (
            Embedder(
                url=os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"),
                model=os.environ.get("CLM_EMB_MODEL", DEFAULT_EMBEDDING_MODEL),
                max_tokens=max_tokens,
                api_key=os.environ.get("CLM_EMB_API_KEY"),
            ),
            kind,
        )
    raise ValueError(f"CLM_EMBEDDER must be one of {EMBEDDER_KINDS}, got {kind!r}")


def make_engine(
    embedder: Any, checkpoint: str | None = None, action_cache: Any = None, device: str | None = None
) -> Engine:
    ckpt = checkpoint or os.environ.get("CLM_CKPT") or DEFAULT_CHECKPOINT
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"CLM head checkpoint not found: {ckpt} (set CLM_CKPT)")
    if action_cache is None:
        action_cache = os.environ.get("CLM_ACTION_CACHE", DEFAULT_ACTION_CACHE)
    return Engine(
        embedder,
        checkpoint=ckpt,
        device=device or os.environ.get("CLM_DEVICE") or default_device(),
        action_cache=action_cache,
    )


def warm_up(engine: Engine) -> None:
    try:
        engine.rank(WARMUP_TEXT, [WARMUP_TEXT], model=DEFAULT_MODEL)
    except EmbedderError as e:
        log.warning("clm server warm-up skipped: %s", e)


def embedding_model_name(embedder: Any) -> str:
    return str(getattr(embedder, "name", None) or getattr(embedder, "model", None) or DEFAULT_EMBEDDING_MODEL)


def asset_stamp() -> str:
    return format(int(max(os.path.getmtime(os.path.join(STATIC_DIR, f)) for f in UI_FILES)), "x")


def index_html() -> str:
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    v = asset_stamp()
    return html.replace('href="app.css"', f'href="app.css?v={v}"').replace('src="app.js"', f'src="app.js?v={v}"')


class RevalidatingStatic(StaticFiles):
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def normalize_embedding_input(inp: Any) -> tuple[list, bool]:
    def is_int(x):
        return isinstance(x, int) and not isinstance(x, bool)

    if isinstance(inp, str):
        return [inp], False
    if isinstance(inp, list) and inp:
        if all(isinstance(x, str) for x in inp):
            return list(inp), False
        if all(is_int(x) for x in inp):
            return [list(inp)], True
        if all(isinstance(x, list) and x and all(is_int(t) for t in x) for x in inp):
            return [list(x) for x in inp], True
    raise HTTPException(
        422, "input must be a string, a list of strings, a list of token ids or a list of token-id lists"
    )


def ids_result(result: Any, id_lists: list[list[int]]) -> tuple[np.ndarray, int]:
    if isinstance(result, tuple):
        vectors, tokens = result
        return np.asarray(vectors, dtype=np.float32), int(tokens)
    return np.asarray(result, dtype=np.float32), sum(len(x) for x in id_lists)


def embed_items(
    embedder: Any, items: list, is_ids: bool, truncate: int | None, tokenizer: Any
) -> tuple[np.ndarray, int]:
    max_tokens = int(embedder.max_tokens) if getattr(embedder, "max_tokens", None) else None
    cap = truncate
    if cap is not None and max_tokens:
        cap = min(cap, max_tokens)
    if is_ids:
        limit = cap or max_tokens
        ids = [list(x[-limit:]) if limit else list(x) for x in items]
        if hasattr(embedder, "embed_ids"):
            return ids_result(embedder.embed_ids(ids), ids)
        if tokenizer is None:
            raise HTTPException(501, "token-id input needs an embedder with embed_ids or a tokenizer to decode them")
        return embedder.embed([tokenizer.decode(x) for x in ids])
    if (
        cap is not None
        and (not max_tokens or cap < max_tokens)
        and hasattr(embedder, "tokenize")
        and hasattr(embedder, "embed_ids")
    ):
        ids = [list(embedder.tokenize(t))[-cap:] for t in items]
        return ids_result(embedder.embed_ids(ids), ids)
    return embedder.embed(list(items))


def encode_embedding(vector: np.ndarray, fmt: str):
    if fmt == "base64":
        return base64.b64encode(np.ascontiguousarray(vector, dtype=np.float32).tobytes()).decode("ascii")
    return [float(x) for x in vector]


def create_app(
    engine: Engine | None = None,
    api_key: str | None = None,
    ui: bool = True,
    cors: bool = False,
    embedder_kind: str | None = None,
    tokenizer: Any = None,
    warmup: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            embedder, kind = make_embedder(embedder_kind)
            app.state.embedder_kind = kind
            app.state.engine = make_engine(embedder)
        eng = app.state.engine
        do_warmup = warmup if warmup is not None else env_flag("CLM_WARMUP", True)
        if do_warmup:
            warm_up(eng)
        if eng.arena is not None:
            pools = " + ".join(f"{p['capacity']:,}x{p['dim']}d" for p in eng.arena.stats()["pools"].values())
            log.info("clm vector cache %s MB reserved on %s (%s)", eng.arena.reserved_mb, eng.arena.device, pools)
        log.info("clm models %s heads on %s", [m["name"] for m in eng.models()], eng.device)
        log.info("clm server ready embedder=%s", app.state.embedder_kind)
        yield

    app = FastAPI(title="CLM System One API", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    if embedder_kind:
        app.state.embedder_kind = embedder_kind
    elif engine is None:
        app.state.embedder_kind = (os.environ.get("CLM_EMBEDDER") or DEFAULT_EMBEDDER).lower()
    else:
        app.state.embedder_kind = "custom"
    app.state.tokenizer = tokenizer
    app.state.api_key = api_key
    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            allow_credentials=False,
            expose_headers=[LATENCY_HEADER],
        )

    def current_engine() -> Engine:
        eng = app.state.engine
        if eng is None:
            raise HTTPException(503, "engine not ready")
        return eng

    def auth(authorization: str | None):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")

    def get_tokenizer():
        if app.state.tokenizer is None:
            eng = current_engine()
            tok = getattr(eng.embedder, "tokenizer", None)
            if tok is not None and hasattr(tok, "decode"):
                app.state.tokenizer = tok
            else:
                try:
                    from transformers import AutoTokenizer

                    app.state.tokenizer = AutoTokenizer.from_pretrained(
                        os.environ.get("CLM_TOKENIZER") or os.environ.get("HF_MODEL") or "Qwen/Qwen3-8B"
                    )
                except Exception as e:
                    log.warning("no tokenizer for token-id inputs: %s", e)
                    return None
        return app.state.tokenizer

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(422, f"body is not JSON: {e}") from e
        if not isinstance(body, dict):
            raise HTTPException(422, "body must be a JSON object")
        return body

    def temperature_of(body: dict) -> float:
        try:
            return float(body.get("temperature", 1.0))
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "temperature must be a number") from e

    async def run(fn, *args):
        try:
            return await asyncio.get_running_loop().run_in_executor(None, fn, *args)
        except ModelNotFound as e:
            raise HTTPException(422, str(e.args[0])) from e
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        except EmbedderError as e:
            raise HTTPException(502, str(e)) from e

    @app.middleware("http")
    async def latency_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/v1/") and LATENCY_HEADER not in response.headers:
            response.headers[LATENCY_HEADER] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.get("/health")
    def health():
        eng = app.state.engine
        if eng is None:
            return JSONResponse(
                {
                    "ok": False,
                    "ready": False,
                    "embedder": False,
                    "embedder_kind": app.state.embedder_kind,
                    "models": [],
                    "cache": None,
                },
                status_code=503,
            )
        out = {
            "ok": True,
            "ready": True,
            "embedder": bool(eng.embedder.healthy()),
            "embedder_kind": app.state.embedder_kind,
            "models": [m["name"] for m in eng.models()],
            "cache": eng.arena.stats() if eng.arena else None,
        }
        for attr in ("stats", "info"):
            fn = getattr(eng.embedder, attr, None)
            if callable(fn):
                try:
                    out["embedder_stats"] = fn()
                except Exception as e:
                    out["embedder_stats"] = {"error": str(e)}
                break
        if getattr(eng, "mock", False):
            out["mock"] = True
        return out

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        auth(authorization)
        return {"models": current_engine().models()}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        body = await json_body(request)
        if "state" not in body or not isinstance(body.get("questions"), dict):
            raise HTTPException(422, "body must be {state, model, questions}")
        temperature = temperature_of(body)
        t0 = time.perf_counter()
        out = await run(eng.answer, body["state"], body["questions"], body.get("model") or DEFAULT_MODEL, temperature)
        return JSONResponse(out, headers={LATENCY_HEADER: f"{(time.perf_counter() - t0) * 1000:.1f}"})

    @app.post("/v1/rank")
    async def rank(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        body = await json_body(request)
        if not isinstance(body.get("answers"), list) or not body["answers"]:
            raise HTTPException(422, "body must be {context, question, answers: [..]}")
        if not all(isinstance(a, str) and a for a in body["answers"]):
            raise HTTPException(422, "answers must be non-empty strings")
        temperature = temperature_of(body)
        model = body.get("model") or DEFAULT_MODEL
        t0 = time.perf_counter()
        ranked = await run(
            eng.rank, body.get("context") or "", body["answers"], body.get("question"), model, temperature
        )
        return JSONResponse(
            {"model": model, "ranked": ranked}, headers={LATENCY_HEADER: f"{(time.perf_counter() - t0) * 1000:.1f}"}
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        eng = current_engine()
        body = await json_body(request)
        if "input" not in body:
            raise HTTPException(422, "body must be {model, input, encoding_format}")
        items, is_ids = normalize_embedding_input(body["input"])
        fmt = body.get("encoding_format") or "float"
        if fmt not in ("float", "base64"):
            raise HTTPException(422, "encoding_format must be 'float' or 'base64'")
        truncate = body.get("truncate_prompt_tokens")
        if truncate is not None:
            if not isinstance(truncate, int) or isinstance(truncate, bool) or truncate == 0 or truncate < -1:
                raise HTTPException(422, "truncate_prompt_tokens must be a positive integer or -1")
            if truncate == -1:
                truncate = None
        dims = body.get("dimensions")
        if dims is not None and dims != HIDDEN:
            raise HTTPException(422, f"dimensions must be {HIDDEN} for this encoder")
        model = body.get("model") or embedding_model_name(eng.embedder)
        tokenizer = get_tokenizer() if is_ids and not hasattr(eng.embedder, "embed_ids") else app.state.tokenizer
        t0 = time.perf_counter()
        vectors, tokens = await run(embed_items, eng.embedder, items, is_ids, truncate, tokenizer)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(items):
            raise HTTPException(500, f"embedder returned shape {tuple(vectors.shape)} for {len(items)} inputs")
        data = [
            {"object": "embedding", "index": i, "embedding": encode_embedding(v, fmt)} for i, v in enumerate(vectors)
        ]
        out = {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": int(tokens), "total_tokens": int(tokens)},
        }
        return JSONResponse(out, headers={LATENCY_HEADER: f"{(time.perf_counter() - t0) * 1000:.1f}"})

    if ui and os.path.isdir(STATIC_DIR):

        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def playground():
            return HTMLResponse(index_html(), headers={"Cache-Control": "no-cache"})

        app.mount("/", RevalidatingStatic(directory=STATIC_DIR, html=True), name="playground")

    return app


def app_from_env() -> FastAPI:
    return create_app(
        None, api_key=os.environ.get("CLM_API_KEY") or None, ui=not env_flag("CLM_NO_UI"), cors=env_flag("CLM_CORS")
    )


app = app_from_env()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m models.autoports.contrastive_lm_clm_v0_1_8b.server")
    ap.add_argument("--host", default=os.environ.get("CLM_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--embedder", choices=EMBEDDER_KINDS, default=None)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--cors", action="store_true")
    ap.add_argument("--log-level", default=os.environ.get("CLM_LOG_LEVEL", "info"))
    args = ap.parse_args(argv)
    port = (
        args.port
        if args.port is not None
        else int(os.environ.get("PORT") or os.environ.get("CLM_PORT") or DEFAULT_PORT)
    )
    if args.embedder:
        os.environ["CLM_EMBEDDER"] = args.embedder
    if args.no_ui:
        os.environ["CLM_NO_UI"] = "1"
    if args.cors:
        os.environ["CLM_CORS"] = "1"
    import uvicorn

    log.info(
        "clm server starting embedder=%s host=%s port=%d",
        os.environ.get("CLM_EMBEDDER", DEFAULT_EMBEDDER),
        args.host,
        port,
    )
    uvicorn.run(app_from_env(), host=args.host, port=port, log_level=args.log_level)
