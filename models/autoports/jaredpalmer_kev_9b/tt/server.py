import asyncio
import hashlib
import hmac
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.autoports.jaredpalmer_kev_9b.tt.api import SystemOneRequest, api_request, render, to_answers
from models.autoports.jaredpalmer_kev_9b.tt.dispatch import CostModel, Policy, WorkerView, collect, plan, share_cost_ms
from models.autoports.jaredpalmer_kev_9b.tt.encode import SERVE_MAX_STATE, ContextOverflow, rows_for_record, user_tokens
from models.autoports.jaredpalmer_kev_9b.tt.head import PointerHead

MODEL_NAMES = ("kev-latest", "jev-latest")
DEFAULT_RUN = "jaredpalmer/kev-9b"
DEFAULT_BASE = "Qwen/Qwen3.5-9B-Base"
HIDDEN = 4096
ALIGN = 128
L1_SMALL_SIZE = 24576
COMMAND_QUEUES = 2
API_KEY = os.environ.get("KEV_API_KEY")
OPEN_PATHS = ("/health", "/v1/health")

log = logging.getLogger("kev.tt.server")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

WARMUP = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "returns": "Exchanges, refunds, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def flag(name, default="0"):
    return os.environ.get(name, default) == "1"


@dataclass
class Settings:
    hf_model: str
    run: str
    devices: str
    mesh_shape: tuple
    device_id: int
    fake: bool
    fake_workers: int
    trace_region: int
    prefix_cache: int
    max_state: int
    max_question: int
    truncate: bool
    traced: bool
    fanout: bool = True
    fanout_backlog_ms: float = 200.0
    perf_summary: str = ""

    @classmethod
    def from_env(cls):
        return cls(
            hf_model=os.environ.setdefault("HF_MODEL", DEFAULT_BASE),
            run=os.environ.get("KEV_RUN", DEFAULT_RUN),
            devices=os.environ.get("KEV_DEVICES", "all"),
            mesh_shape=parse_mesh_shape(os.environ.get("KEV_MESH_SHAPE"), os.environ.get("MESH_DEVICE")),
            device_id=int(os.environ.get("KEV_DEVICE_ID", "0")),
            fake=flag("KEV_FAKE_ENGINE"),
            fake_workers=int(os.environ.get("KEV_FAKE_WORKERS", "2")),
            trace_region=int(os.environ.get("KEV_TRACE_REGION", str(1 << 30))),
            prefix_cache=int(os.environ.get("KEV_PREFIX_CACHE", "8")),
            max_state=int(os.environ.get("KEV_MAX_STATE", "65536")),
            max_question=int(os.environ.get("KEV_MAX_QUESTION", "2048")),
            truncate=flag("KEV_TRUNCATE_STATES"),
            traced=flag("KEV_TRACED", "1"),
            fanout=flag("KEV_FANOUT", "1"),
            fanout_backlog_ms=float(os.environ.get("KEV_FANOUT_BACKLOG_MS", "200")),
            perf_summary=os.environ.get("KEV_PERF_SUMMARY", ""),
        )


def parse_mesh_shape(shape, mesh_device):
    if shape:
        r, c = shape.lower().split("x")
        return int(r), int(c)
    name = (mesh_device or "").upper()
    if name in ("P150X4", "P300X2", "QB2"):
        return 2, 2
    return 1, 1


def resolve_run(run):
    if os.path.isdir(run):
        return run
    from huggingface_hub import snapshot_download

    repo, _, revision = run.partition("@")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    return snapshot_download(repo, revision=revision or None, local_files_only=offline)


def release_date(run_dir):
    mtime = (Path(run_dir) / "head.pt").stat().st_mtime
    return datetime.fromtimestamp(mtime, timezone.utc).date().isoformat()


def state_key(state_ids):
    return hashlib.sha1(json.dumps(state_ids).encode()).hexdigest()


def output_tokens(tok, answers):
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)


@dataclass
class FakeHandle:
    key: str
    slot: int


class FakeEngine:
    def __init__(self, snapshot_slots=1 << 30):
        self.snapshot_slots = snapshot_slots
        self.slots = {}

    def prefill_state(self, state_ids, slot=0):
        assert 0 <= slot < self.snapshot_slots, f"slot {slot} out of range"
        key = state_key(state_ids[0].tolist())
        self.slots[slot] = key
        return FakeHandle(key, slot)

    def question_hidden(self, handle, question_ids, positions_in_question):
        if self.slots.get(handle.slot) != handle.key:
            raise RuntimeError(f"slot {handle.slot} no longer holds state {handle.key[:8]}")
        seed_src = json.dumps([handle.key, question_ids.tolist(), list(positions_in_question)]).encode()
        seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:8], "little") % (1 << 62)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(len(positions_in_question), HIDDEN, generator=g)


class Worker:
    def __init__(self, wid, build, head, cache_size):
        self.id = wid
        self.build = build
        self.head = head
        self.cache_size = cache_size
        self.queue = queue.Queue()
        self.cache = OrderedDict()
        self.hits = self.misses = self.requests = 0
        self.inflight = 0
        self.backlog_ms = 0.0
        self.counter = threading.Lock()
        self.ready = threading.Event()
        self.error = None
        self.engine = None
        self.thread = threading.Thread(target=self._run, name=f"kev-worker-{wid}", daemon=True)
        self.thread.start()

    def load(self):
        return self.queue.qsize() + self.inflight

    def submit(self, rows, key, cost_ms=0.0):
        done = Future()
        with self.counter:
            self.inflight += 1
            self.backlog_ms += cost_ms
        self.queue.put((rows, key, done, cost_ms))
        return done

    def view(self):
        return WorkerView(self.id, self.backlog_ms, self.cache)

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    def _run(self):
        try:
            self.engine = self.build()
            slots = getattr(self.engine, "snapshot_slots", 1)
            self.cache_size = min(self.cache_size, slots)
            self.free_slots = list(range(self.cache_size))
        except Exception as e:
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.queue.get()
            if job is None:
                return
            rows, key, done, cost_ms = job
            try:
                done.set_result(self._serve(rows, key))
            except Exception as e:
                done.set_exception(e)
            finally:
                with self.counter:
                    self.inflight -= 1
                    self.backlog_ms -= cost_ms

    def _serve(self, rows, key):
        t0 = time.perf_counter()
        entry = self.cache.get(key)
        hit = entry is not None
        if hit:
            self.cache.move_to_end(key)
            self.hits += 1
            handle, _ = entry
        else:
            self.misses += 1
            slot = 0
            if self.cache_size > 0:
                slot = self.free_slots.pop() if self.free_slots else self.cache.popitem(last=False)[1][1]
            try:
                handle = self.engine.prefill_state(torch.tensor([rows[0].state_ids], dtype=torch.long), slot=slot)
            except Exception:
                if self.cache_size > 0:
                    self.free_slots.append(slot)
                raise
            if self.cache_size > 0:
                self.cache[key] = (handle, slot)
        S = len(rows[0].state_ids)
        probs = []
        for row in rows:
            positions = [p - S for p in row.opt_positions] + [row.decide_position - S]
            h = self.engine.question_hidden(handle, torch.tensor([row.question_ids], dtype=torch.long), positions)
            probs.append(self.head.probs(h[-1], h[:-1]).tolist())
        self.requests += 1
        return probs, hit, round((time.perf_counter() - t0) * 1000, 1)


class Server:
    def __init__(self, settings):
        self.settings = settings
        self.run_dir = resolve_run(settings.run)
        self.release_date = release_date(self.run_dir)
        self.tok = load_tokenizer(settings.hf_model, self.run_dir)
        self.head = PointerHead(self.run_dir)
        self.parent = None
        self.close_devices = lambda: None
        self.lock = threading.Lock()
        self.workers = []
        self.stopping = threading.Event()
        self.cost_model = CostModel.load(settings.perf_summary)
        self.policy = Policy(fanout=settings.fanout, fanout_backlog_ms=settings.fanout_backlog_ms)
        self.fanouts = 0

    @classmethod
    def start(cls, settings):
        server = cls(settings)
        server.open()
        server.warmup()
        return server

    def open(self):
        s = self.settings
        if s.fake:
            builds = [(f"fake{i}", lambda: FakeEngine(max(1, s.prefix_cache))) for i in range(max(1, s.fake_workers))]
            self.device_name = f"fake x{len(builds)}"
        else:
            devices = self.open_devices()
            builds = [
                (str(d.id()) if hasattr(d, "id") else str(i), lambda d=d: build_engine(d, s))
                for i, d in enumerate(devices)
            ]
            self.device_name = f"ttnn {s.mesh_shape[0]}x{s.mesh_shape[1]} x{len(devices)} worker(s)"
        self.workers = [Worker(i, build, self.head, s.prefix_cache) for i, (_, build) in enumerate(builds)]
        for w, (name, _) in zip(self.workers, builds):
            w.ready.wait()
            if w.error is not None:
                self.close()
                raise RuntimeError(f"worker {w.id} ({name}) failed to load") from w.error
            log.info("worker %d ready on %s, prefix cache %d state(s)", w.id, name, w.cache_size)

    def open_devices(self):
        import ttnn

        s = self.settings
        r, c = s.mesh_shape
        kw = dict(l1_small_size=L1_SMALL_SIZE, num_command_queues=COMMAND_QUEUES, trace_region_size=s.trace_region)
        if r * c == 1:
            dev = ttnn.open_device(device_id=s.device_id, **kw)
            dev.enable_program_cache()
            self.parent = dev
            self.close_devices = lambda: ttnn.close_device(dev)
            return [dev]
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(r, c), **kw)
        subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))
        if s.devices != "all":
            subs = [subs[int(i)] for i in s.devices.split(",")]
        for sub in subs:
            sub.enable_program_cache()
        self.parent = mesh
        self.close_devices = lambda: ttnn.close_mesh_device(mesh)
        return subs

    def warmup(self):
        req = SystemOneRequest.model_validate(WARMUP)
        rows, _ = self.encode(req)
        key = state_key(rows[0].state_ids)
        for w in self.workers:
            probs, _, ms = w.submit(rows, key).result()
            log.info("warmup worker=%d questions=%d latency_ms=%.1f", w.id, len(probs), ms)

    def close(self):
        if self.stopping.is_set():
            return
        self.stopping.set()
        for w in self.workers:
            if w.ready.is_set() and w.error is None:
                w.stop()
        self.close_devices()

    def encode(self, req):
        s = self.settings
        try:
            rows = rows_for_record(self.tok, req, truncate=s.truncate)
        except ContextOverflow as e:
            raise HTTPException(422, overflow_message(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        S = len(rows[0].state_ids)
        full = S if S < SERVE_MAX_STATE else len(user_tokens(self.tok, render(req.state))) + 1
        if S > s.max_state:
            if not s.truncate:
                e = ContextOverflow("", state_tokens=full, max_state=s.max_state)
                raise HTTPException(422, overflow_message(e))
            rows = truncate_rows(rows, s.max_state)
            S = s.max_state
        for row in rows:
            Q = len(row.question_ids)
            if Q > s.max_question or (S - (S // ALIGN) * ALIGN) + Q > s.max_question:
                raise HTTPException(
                    422, f"branch too long: {Q} tokens with a {S}-token state (question limit {s.max_question})"
                )
        return rows, full

    def dispatch(self, rows, key):
        by_id = {w.id: w for w in self.workers}
        with self.lock:
            shares = plan(rows, [w.view() for w in self.workers], key, self.cost_model, self.policy)
            parts = []
            for wid, idx in shares:
                w = by_id[wid]
                cost = share_cost_ms(rows, idx, w.view(), key, self.cost_model)
                parts.append((wid, idx, w.submit([rows[i] for i in idx], key, cost)))
            self.fanouts += len(parts) > 1
        return parts

    def submit(self, req):
        if self.stopping.is_set():
            raise HTTPException(503, "the server is stopping")
        rows, full = self.encode(req)
        key = state_key(rows[0].state_ids)
        done = Future()
        inner = collect(self.dispatch(rows, key), len(rows))
        tokens = len(rows[0].state_ids) + sum(len(r.question_ids) for r in rows)

        def finish(f):
            if f.exception() is not None:
                done.set_exception(f.exception())
                return
            probs, merged = f.result()
            stats = {"tokens": tokens, "state_tokens": full, "state_tokens_used": len(rows[0].state_ids), **merged}
            log.info(
                "workers=%s S=%d questions=%d latency_ms=%.1f latency_ms_sum=%.1f cache_hit=%s",
                stats["workers"],
                stats["state_tokens_used"],
                len(rows),
                stats["latency_ms"],
                stats["latency_ms_sum"],
                stats["prefix_cache_hit"],
            )
            done.set_result((probs, stats, rows))

        inner.add_done_callback(finish)
        return done

    def as_request(self, rec):
        return rec if isinstance(rec, SystemOneRequest) else SystemOneRequest.model_validate(api_request(rec))

    def predict(self, rec):
        probs, stats, _ = self.submit(self.as_request(rec)).result()
        return probs, stats

    def answer(self, req):
        return self.body(req, *self.submit(req).result())

    async def answer_async(self, req):
        return self.body(req, *await asyncio.wrap_future(self.submit(req)))

    def body(self, req, probs, stats, rows):
        meta = [{"id": r.qid, "type": r.qtype, "keys": r.option_keys, "legend": r.legend} for r in rows]
        answers = to_answers(probs, meta)
        body = {
            "model": req.model,
            "answers": answers,
            "usage": {"input_tokens": stats["tokens"], "output_tokens": output_tokens(self.tok, answers)},
            "latency_ms": stats["latency_ms"],
        }
        if self.settings.truncate:
            body["usage"].update(state_tokens=stats["state_tokens"], state_tokens_used=stats["state_tokens_used"])
            body["truncated"] = stats["state_tokens"] > stats["state_tokens_used"]
        return body

    def card(self):
        s = self.settings
        cache = {
            "size": s.prefix_cache,
            "hits": sum(w.hits for w in self.workers),
            "misses": sum(w.misses for w in self.workers),
            "cached_states": sum(len(w.cache) for w in self.workers),
        }
        workers = [
            {
                "id": w.id,
                "queued": w.load(),
                "backlog_ms": round(w.backlog_ms, 1),
                "cached_states": len(w.cache),
                "cache_size": w.cache_size,
                "hits": w.hits,
                "misses": w.misses,
                "requests": w.requests,
            }
            for w in self.workers
        ]
        dispatch = {
            "fanout": self.policy.fanout,
            "fanout_backlog_ms": self.policy.fanout_backlog_ms,
            "short_state_tokens": self.policy.short_state_tokens,
            "fanout_requests": self.fanouts,
            "cost_model": {
                "tail_ms": dict(self.cost_model.tail_ms),
                "state_ms_per_block": self.cost_model.state_ms_per_block,
            },
        }
        return {
            "description": f"Kev pointer head on {s.hf_model}, serving {s.run} at temperature {self.head.temperature:.2f} (ttnn)",
            "release_date": self.release_date,
            "run": s.run,
            "base": s.hf_model,
            "lora": self.head.meta.get("lora"),
            "device": self.device_name,
            "backend": "fake" if s.fake else "ttnn",
            "dtype": "bf16",
            "temperature": self.head.temperature,
            "max_state_tokens": min(SERVE_MAX_STATE, s.max_state),
            "max_question_tokens": s.max_question,
            "truncate_states": s.truncate,
            "prefix_cache": cache,
            "workers": workers,
            "dispatch": dispatch,
            "batches": {
                "count": sum(w.requests for w in self.workers),
                "requests": sum(w.requests for w in self.workers),
                "queued": sum(w.load() for w in self.workers),
            },
        }


def load_tokenizer(hf_model, run_dir):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(hf_model)
    except Exception as e:
        log.warning("tokenizer from %s failed (%s); using %s", hf_model, e, run_dir)
        return AutoTokenizer.from_pretrained(run_dir)


def build_engine(device, settings):
    from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine

    return KevEngine(
        device, max_state_len=settings.max_state, snapshot_slots=max(1, settings.prefix_cache), traced=settings.traced
    )


def overflow_message(e):
    if e.max_state is None:
        return str(e)
    return (
        f"state is {e.state_tokens:,} tokens, over the {e.max_state:,}-token limit (the <state> token included): shorten the document or split it across requests"
        f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {e.max_state:,} tokens (responses then say truncated: true)"
    )


def truncate_rows(rows, max_state):
    S = len(rows[0].state_ids)
    shift = max_state - S
    out = []
    for r in rows:
        out.append(
            type(r)(
                r.qid,
                r.qtype,
                r.option_keys,
                r.state_ids[:max_state],
                r.question_ids,
                [p + shift for p in r.opt_positions],
                r.decide_position + shift,
                r.legend,
            )
        )
    return out


@asynccontextmanager
async def lifespan(app):
    settings = Settings.from_env()
    log.info(
        "starting: run=%s base=%s fake=%s mesh=%s devices=%s prefix_cache=%d max_state=%d truncate=%s traced=%s "
        "fanout=%s fanout_backlog_ms=%.0f",
        settings.run,
        settings.hf_model,
        settings.fake,
        settings.mesh_shape,
        settings.devices,
        settings.prefix_cache,
        settings.max_state,
        settings.truncate,
        settings.traced,
        settings.fanout,
        settings.fanout_backlog_ms,
    )
    app.state.server = await asyncio.to_thread(Server.start, settings)
    log.info("ready: %d worker(s) on %s", len(app.state.server.workers), app.state.server.device_name)
    try:
        yield
    finally:
        app.state.server.close()


app = FastAPI(title="kev-tt", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-typesafe-request-id", "server-timing"],
)


@app.middleware("http")
async def typesafe(request, call_next):
    started = time.perf_counter()
    path = request.url.path
    if (
        API_KEY
        and path.startswith("/v1")
        and path not in OPEN_PATHS
        and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}")
    ):
        resp = JSONResponse(
            {"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"},
            401,
            {"www-authenticate": "Bearer"},
        )
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    return resp


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    return await server().answer_async(req)


@app.get("/v1/models")
def models():
    card = server().card()
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


@app.get("/health")
@app.get("/v1/health")
def health():
    s = server()
    return {"status": "ok", "workers": len(s.workers), "queued": sum(w.load() for w in s.workers)}


def main():
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, lifespan="on")


if __name__ == "__main__":
    main()
