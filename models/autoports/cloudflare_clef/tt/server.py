import asyncio
import hashlib
import hmac
import inspect
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.autoports.cloudflare_clef.tt import encode as clef_encode
from models.autoports.cloudflare_clef.tt import head as clef_head
from models.autoports.cloudflare_clef.tt import precision_defaults
from models.autoports.cloudflare_clef.tt.api import SystemOneRequest, api_request, decode_media, to_record

DEFAULT_REPO = "Cloudflare/clef"
DEFAULT_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
DEFAULT_MODEL_NAMES = "clef"
HIDDEN = 5120
ALIGN = 128
TP = 2
L1_SMALL_SIZE = 24576
COMMAND_QUEUES = 2
TRACE_REGION = 1 << 30
ENCODE_MAX_LENGTH = 1 << 31
API_KEY = os.environ.get("CLEF_API_KEY")
OPEN_PATHS = ("/health", "/v1/health")
PARENTS = {"1x4": ("FABRIC_1D", (1, 4)), "2x2": ("FABRIC_2D", (2, 2))}
MESH_DEVICE_SHAPES = {"P150X2": (1, 2), "P300": (1, 2), "P150X4": (1, 4), "P300X2": (1, 4), "QB2": (2, 2)}

log = logging.getLogger("clef.tt.server")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

WARMUP = {
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


def flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def parse_mesh_shape(shape, mesh_device):
    if shape:
        r, c = shape.lower().split("x")
        return int(r), int(c)
    return MESH_DEVICE_SHAPES.get((mesh_device or "").upper(), (1, 2))


def parse_offset(text):
    r, c = (text or "0,0").split(",")
    return int(r), int(c)


def resolve_model(value, revision=None):
    if os.path.isdir(value):
        return str(Path(value).resolve())
    from huggingface_hub import snapshot_download

    repo, _, at_revision = value.partition("@")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    return snapshot_download(repo, revision=at_revision or revision or None, local_files_only=offline)


@dataclass
class Settings:
    model: str
    revision: str
    snapshot: str
    model_names: tuple
    mesh_shape: tuple
    parent_mesh: str
    submesh_offset: tuple
    fake: bool
    fake_workers: int
    trace_region: int
    prefix_cache: int
    max_state: int
    max_tail: int
    truncate: bool
    traced: bool
    planner: bool = True
    allow_remote_images: bool = True
    warmup: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        model = os.environ.get("CLEF_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_REPO
        revision = os.environ.get("CLEF_REVISION", DEFAULT_REVISION)
        snapshot = resolve_model(model, revision)
        os.environ["CLEF_MODEL"] = snapshot
        traced = flag("CLEF_TRACED", "0")
        return cls(
            model=model,
            revision=revision,
            snapshot=snapshot,
            model_names=tuple(
                n.strip() for n in os.environ.get("CLEF_MODEL_NAMES", DEFAULT_MODEL_NAMES).split(",") if n.strip()
            ),
            mesh_shape=parse_mesh_shape(os.environ.get("CLEF_MESH_SHAPE"), os.environ.get("MESH_DEVICE")),
            parent_mesh=os.environ.get("CLEF_PARENT_MESH", ""),
            submesh_offset=parse_offset(os.environ.get("CLEF_SUBMESH_OFFSET")),
            fake=flag("CLEF_FAKE_ENGINE"),
            fake_workers=int(os.environ.get("CLEF_FAKE_WORKERS", "1")),
            trace_region=int(os.environ.get("CLEF_TRACE_REGION", str(TRACE_REGION if traced else 0))),
            prefix_cache=int(os.environ.get("CLEF_PREFIX_CACHE", "4")),
            max_state=int(os.environ.get("CLEF_MAX_STATE", "16384")),
            max_tail=int(os.environ.get("CLEF_MAX_TAIL", "4096")),
            truncate=flag("CLEF_TRUNCATE_STATES"),
            traced=traced,
            planner=flag("CLEF_PLANNER", "1"),
            allow_remote_images=flag("CLEF_ALLOW_REMOTE_IMAGES", "1"),
            warmup=flag("CLEF_WARMUP", "1"),
        )


def mesh_plan(mesh_shape, visible, parent="", offset=(0, 0)):
    r, c = mesh_shape
    if (r, c) == (1, 2):
        if parent or visible > 2:
            name = parent or "1x4"
            if name not in PARENTS:
                raise ValueError(f"CLEF_PARENT_MESH={name!r}; expected one of {sorted(PARENTS)}")
            fabric, shape = PARENTS[name]
            return {"fabric": fabric, "open": shape, "submeshes": [((1, 2), tuple(offset))], "parent": name}
        return {"fabric": "FABRIC_1D", "open": (1, 2), "submeshes": [], "parent": None}
    if (r, c) == (1, 4):
        if parent:
            raise ValueError("CLEF_PARENT_MESH is only supported for CLEF_MESH_SHAPE=1x2")
        return {
            "fabric": "FABRIC_1D",
            "open": (1, 4),
            "submeshes": [((1, 2), (0, 0)), ((1, 2), (0, 2))],
            "parent": None,
        }
    if (r, c) == (2, 2):
        if parent:
            raise ValueError("CLEF_PARENT_MESH is only supported for CLEF_MESH_SHAPE=1x2")
        return {
            "fabric": "FABRIC_2D",
            "open": (2, 2),
            "submeshes": [((1, 2), (0, 0)), ((1, 2), (1, 0))],
            "parent": None,
        }
    raise ValueError(f"unsupported mesh shape {r}x{c}; Clef runs TP={TP} groups on 1x2, 1x4 or 2x2")


def visible_chips():
    import ttnn

    return len(ttnn.get_device_ids())


def open_devices(settings):
    import ttnn

    visible = visible_chips()
    plan = mesh_plan(settings.mesh_shape, visible, settings.parent_mesh, settings.submesh_offset)
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, plan["fabric"]))
    parent = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*plan["open"]),
        l1_small_size=L1_SMALL_SIZE,
        num_command_queues=COMMAND_QUEUES,
        trace_region_size=settings.trace_region,
    )
    groups = []
    for shape, offset in plan["submeshes"]:
        groups.append(parent.create_submesh(ttnn.MeshShape(*shape), offset=ttnn.MeshCoordinate(*offset)))
    if not groups:
        groups = [parent]
    for group in groups:
        group.enable_program_cache()
    log.info(
        "mesh: %d chip(s) visible, %s parent %s chips %s, %d TP group(s) %s",
        visible,
        plan["fabric"],
        list(parent.shape),
        list(parent.get_device_ids()),
        len(groups),
        [list(g.get_device_ids()) for g in groups],
    )

    def close():
        for group in groups:
            ttnn.synchronize_device(group)
        for child in parent.get_submeshes():
            ttnn.close_mesh_device(child)
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    return parent, groups, close, plan


def accepted_kwargs(fn, candidates):
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


def build_engine(mesh, settings):
    from models.autoports.cloudflare_clef.tt.engine import ClefEngine

    candidates = dict(
        max_state_len=settings.max_state,
        max_tail_len=settings.max_tail,
        snapshot_slots=max(1, settings.prefix_cache),
        traced=settings.traced,
        planner=settings.planner,
    )
    return ClefEngine(mesh, **accepted_kwargs(ClefEngine.__init__, candidates))


def build_fake_engine(settings):
    from models.autoports.cloudflare_clef.tt.fake_engine import FakeEngine

    return FakeEngine(
        max_state_len=settings.max_state, max_tail_len=settings.max_tail, snapshot_slots=max(1, settings.prefix_cache)
    )


def media_key(ids, media_digest):
    key = clef_encode.cache_key(ids)
    if media_digest is None:
        return key
    return hashlib.sha1(f"{key}:{media_digest}".encode()).hexdigest()


@dataclass
class Job:
    ids: list
    state_ids: list
    tail_ids: list
    encoded: object
    media: dict
    key: str
    questions: dict


def supports_media(engine):
    try:
        return "media" in inspect.signature(engine.prefill_state).parameters
    except (TypeError, ValueError):
        return bool(getattr(engine, "supports_media", False))


def supports_tail_len(engine):
    try:
        return "tail_len" in inspect.signature(engine.prefill_state).parameters
    except (TypeError, ValueError):
        return False


class Worker:
    def __init__(self, wid, build, head, lm_rows, cache_size):
        self.id = wid
        self.build = build
        self.head = head
        self.lm_rows = lm_rows
        self.cache_size = cache_size
        self.queue = queue.Queue()
        self.cache = OrderedDict()
        self.free_slots = []
        self.hits = self.misses = self.requests = 0
        self.inflight = 0
        self.counter = threading.Lock()
        self.ready = threading.Event()
        self.error = None
        self.engine = None
        self.media_ok = False
        self.tail_len_ok = False
        self.traced = False
        self.warm_grids = None
        self.gdn_conv = None
        self.planner = None
        self.thread = threading.Thread(target=self._run, name=f"clef-worker-{wid}", daemon=True)
        self.thread.start()

    def load(self):
        return self.queue.qsize() + self.inflight

    def submit(self, job):
        done = Future()
        with self.counter:
            self.inflight += 1
        self.queue.put((job, done))
        return done

    def stop(self):
        self.queue.put(None)
        self.thread.join()

    def _run(self):
        try:
            self.engine = self.build()
            slots = getattr(self.engine, "snapshot_slots", 1)
            self.cache_size = min(self.cache_size, slots)
            self.free_slots = list(range(self.cache_size))
            self.media_ok = supports_media(self.engine)
            self.tail_len_ok = supports_tail_len(self.engine)
            self.traced = bool(getattr(self.engine, "traced", False))
            self.gdn_conv = getattr(self.engine, "gdn_conv_impl", None)
            self.planner = getattr(self.engine, "planner", None)
            if self.traced:
                self.warm_grids = sorted(tuple(g) for g, _ in getattr(self.engine, "vision_warmed_grids", []))
        except Exception as error:
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        while True:
            item = self.queue.get()
            if item is None:
                return
            job, done = item
            try:
                done.set_result(self._serve(job))
            except Exception as error:
                done.set_exception(error)
            finally:
                with self.counter:
                    self.inflight -= 1

    def _media_kwargs(self, job):
        if job.media is None:
            return {}
        if not self.media_ok:
            raise HTTPException(422, "this server build does not accept images or videos")
        return {"media": job.media}

    def _serve(self, job):
        t0 = time.perf_counter()
        ids = torch.tensor([job.ids], dtype=torch.long)
        hit = False
        if self.cache_size == 0:
            hidden = self.engine.prefill_hidden(ids, slot=0, **self._media_kwargs(job))
        else:
            entry = self.cache.get(job.key)
            hit = entry is not None
            if hit:
                self.cache.move_to_end(job.key)
                self.hits += 1
                handle, slot = entry
            else:
                self.misses += 1
                slot = self.free_slots.pop() if self.free_slots else self.cache.popitem(last=False)[1][1]
                hint = {"tail_len": len(job.tail_ids)} if self.tail_len_ok else {}
                try:
                    handle = self.engine.prefill_state(
                        torch.tensor([job.state_ids], dtype=torch.long),
                        slot=slot,
                        key=job.key,
                        **self._media_kwargs(job),
                        **hint,
                    )
                except ValueError as error:
                    self.free_slots.append(slot)
                    raise HTTPException(422, str(error))
                except Exception:
                    self.free_slots.append(slot)
                    raise
                self.cache[job.key] = (handle, slot)
            tail = self.engine.schema_hidden(handle, torch.tensor([job.tail_ids], dtype=torch.long))
            hidden = torch.cat([self.engine.prefix_hidden[slot], tail], dim=0)
        probs = clef_head.probs_for_record(self.head, hidden, ids[0], job.encoded, self.lm_rows)
        self.requests += 1
        return probs, hit, round((time.perf_counter() - t0) * 1000, 1)


class Server:
    def __init__(self, settings):
        self.settings = settings
        self.snapshot = settings.snapshot
        self.release = clef_encode.release_module(self.snapshot)
        self.tok = clef_encode.load_tokenizer(self.snapshot)
        self.processor = None
        self.head = clef_head.load_head(self.snapshot)
        self.parent = None
        self.close_devices = lambda: None
        self.plan = None
        self.lock = threading.Lock()
        self.workers = []
        self.stopping = threading.Event()
        self.device_name = "none"
        self.started = time.time()

    @classmethod
    def start(cls, settings):
        server = cls(settings)
        server.open()
        if settings.warmup:
            server.warmup()
        return server

    def processor_for_media(self):
        if self.processor is None:
            with self.lock:
                if self.processor is None:
                    self.processor = clef_encode.load_processor(self.snapshot)
        return self.processor

    def _lm_rows(self):
        from models.autoports.cloudflare_clef.tt.loader import LmHeadRows

        return LmHeadRows(self.snapshot)

    def open(self):
        s = self.settings
        if s.fake:
            builds = [(f"fake{i}", lambda: build_fake_engine(s)) for i in range(max(1, s.fake_workers))]
            self.device_name = f"fake x{len(builds)}"
        else:
            self.parent, groups, self.close_devices, self.plan = open_devices(s)
            builds = [(str(list(g.get_device_ids())), lambda g=g: build_engine(g, s)) for g in groups]
            self.device_name = f"ttnn {s.mesh_shape[0]}x{s.mesh_shape[1]}: {len(groups)} TP={TP} worker(s)"
        self.workers = [
            Worker(i, build, self.head, self._lm_rows(), s.prefix_cache) for i, (_, build) in enumerate(builds)
        ]
        for w, (name, _) in zip(self.workers, builds):
            w.ready.wait()
            if w.error is not None:
                self.close()
                raise RuntimeError(f"worker {w.id} ({name}) failed to load") from w.error
            log.info(
                "worker %d ready on %s, prefix cache %d state(s), media=%s, mode=%s, gdn_conv=%s, planner=%s, "
                "warm_grids=%s",
                w.id,
                name,
                w.cache_size,
                w.media_ok,
                "traced" if w.traced else "eager",
                w.gdn_conv,
                w.planner,
                w.warm_grids,
            )

    def warmup(self):
        req = SystemOneRequest.model_validate(WARMUP)
        job, _ = self.encode(req)
        for w in self.workers:
            probs, _, ms = w.submit(job).result()
            log.info("warmup worker=%d questions=%d latency_ms=%.1f", w.id, len(probs), ms)

    def close(self):
        if self.stopping.is_set():
            return
        self.stopping.set()
        for w in self.workers:
            if w.ready.is_set() and w.error is None:
                w.stop()
        self.close_devices()

    def _encode_record(self, record, processor, max_state_tokens=None):
        try:
            encoded = clef_encode.encode(
                self.tok, record, processor=processor, max_length=ENCODE_MAX_LENGTH, max_state_tokens=max_state_tokens
            )
        except ValueError as error:
            raise HTTPException(422, str(error))
        try:
            state_ids, tail_ids, _ = clef_encode.split_for_cache(encoded, self.tok, record)
        except ValueError as error:
            raise HTTPException(500, f"prefix split failed: {error}")
        return encoded, state_ids, tail_ids

    def encode(self, req):
        s = self.settings
        try:
            images, videos, media_digest = decode_media(req)
        except ValueError as error:
            raise HTTPException(422, str(error))
        record = to_record(req, images, videos)
        processor = self.processor_for_media() if (images or videos) else None
        encoded, state_ids, tail_ids = self._encode_record(record, processor)
        if encoded.media:
            self.check_warm_grids(encoded.media)
        S = len(state_ids)
        stats = {"state_tokens": S, "state_tokens_used": S}
        if S > s.max_state:
            n_state = len(self.release._tokens(self.tok, self.release.render(req.state)))
            fixed = S - n_state
            keep = s.max_state - fixed
            if not s.truncate or keep < 0:
                raise HTTPException(422, overflow_message(S, fixed, s.max_state))
            encoded, state_ids, tail_ids = self._encode_record(record, processor, max_state_tokens=keep)
            stats["state_tokens_used"] = len(state_ids)
        if len(tail_ids) > s.max_tail:
            raise HTTPException(
                422,
                f"schema is {len(tail_ids):,} tokens, over the {s.max_tail:,}-token limit: "
                "use fewer questions or shorter criteria, or split them across requests",
            )
        key = media_key(state_ids, media_digest)
        job = Job(list(encoded.input_ids), state_ids, tail_ids, encoded, encoded.media, key, req.question_dicts())
        stats["tokens"] = len(encoded.input_ids)
        stats["images"] = len(images)
        stats["videos"] = len(videos)
        return job, stats

    def warm_grids(self):
        lists = [w.warm_grids for w in self.workers if w.warm_grids is not None]
        if not lists:
            return None
        return sorted({grid for grids in lists for grid in grids})

    def check_warm_grids(self, media):
        warm = self.warm_grids()
        if warm is None:
            return
        rows = []
        for name in ("image_grid_thw", "video_grid_thw"):
            grids = media.get(name)
            if grids is not None:
                rows.extend(torch.as_tensor(grids).reshape(-1, 3).tolist())
        if len(rows) > 1:
            raise HTTPException(422, single_grid_message(len(rows)))
        for grid in rows:
            if tuple(grid) not in warm:
                raise HTTPException(422, warm_grid_message(tuple(grid), warm))

    def pick_worker(self, key):
        with self.lock:
            for w in self.workers:
                if key in w.cache:
                    return w
            return min(self.workers, key=lambda w: (w.load(), w.id))

    def submit(self, req):
        if self.stopping.is_set():
            raise HTTPException(503, "the server is stopping")
        job, stats = self.encode(req)
        worker = self.pick_worker(job.key)
        done = Future()
        inner = worker.submit(job)

        def finish(f):
            if f.exception() is not None:
                done.set_exception(f.exception())
                return
            probs, hit, ms = f.result()
            stats.update(worker=worker.id, latency_ms=ms, prefix_cache_hit=hit)
            log.info(
                "worker=%d S=%d tail=%d questions=%d images=%d videos=%d latency_ms=%.1f cache_hit=%s",
                worker.id,
                stats["state_tokens_used"],
                len(job.tail_ids),
                len(probs),
                stats["images"],
                stats["videos"],
                ms,
                hit,
            )
            done.set_result((probs, stats, job))

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

    def body(self, req, probs, stats, job):
        answers = {qid: self.release.systemone_answer(job.questions[qid], dist) for qid, dist in probs.items()}
        body = {
            "model": req.model,
            "answers": answers,
            "usage": {"input_tokens": stats["tokens"], "output_tokens": 0},
            "latency_ms": stats["latency_ms"],
        }
        if self.settings.truncate:
            body["usage"].update(state_tokens=stats["state_tokens"], state_tokens_used=stats["state_tokens_used"])
            body["truncated"] = stats["state_tokens"] > stats["state_tokens_used"]
        return body

    def card(self):
        s = self.settings
        workers = [
            {
                "id": w.id,
                "queued": w.load(),
                "cached_states": len(w.cache),
                "cache_size": w.cache_size,
                "hits": w.hits,
                "misses": w.misses,
                "requests": w.requests,
                "media": w.media_ok,
            }
            for w in self.workers
        ]
        backend = "fake engine" if s.fake else "ttnn, TP=2"
        warm = self.warm_grids()
        traced_media = None
        if warm is not None:
            traced_media = {
                "warm_grids": [list(grid) for grid in warm],
                "rule": "one image or video grid per request, and a grid (t, h, w) outside this list is refused "
                "with 422; eager (CLEF_TRACED=0) accepts any grid and several images or videos per request",
            }
        return {
            "description": f"Cloudflare/clef joint schema decisions on Tenstorrent ({backend})",
            "weights": {"repo": DEFAULT_REPO, "revision": s.revision, "snapshot": self.snapshot},
            "device": self.device_name,
            "mesh_shape": f"{s.mesh_shape[0]}x{s.mesh_shape[1]}",
            "mesh_plan": self.plan,
            "backend": "fake" if s.fake else "ttnn",
            "precision": precision_defaults.active(),
            "traced": s.traced,
            "mode": "traced" if s.traced else "eager",
            "gdn_conv": next((w.gdn_conv for w in self.workers if w.gdn_conv), None),
            "prefix_planner": s.planner,
            "traced_media": traced_media,
            "max_state_tokens": s.max_state,
            "max_schema_tokens": s.max_tail,
            "truncate_states": s.truncate,
            "remote_images": s.allow_remote_images,
            "prefix_cache": {
                "size": s.prefix_cache,
                "hits": sum(w.hits for w in self.workers),
                "misses": sum(w.misses for w in self.workers),
                "cached_states": sum(len(w.cache) for w in self.workers),
            },
            "workers": workers,
            "requests": sum(w.requests for w in self.workers),
            "uptime_s": round(time.time() - self.started, 1),
        }


def overflow_message(S, fixed, max_state):
    return (
        f"state is {S:,} tokens ({fixed} of them the system prompt and media placeholders), "
        f"over the {max_state:,}-token limit: shorten the state or split it across requests; "
        f"or start the server with CLEF_TRUNCATE_STATES=1 to read only its first {max_state:,} tokens "
        "(responses then say truncated: true)"
    )


def warm_grid_message(grid, warm):
    grids = "; ".join(",".join(str(x) for x in g) for g in warm)
    return (
        f"image grid {grid} (t, h, w patches) is not in this traced server's warm list [{grids}]: "
        "resize the image to a warmed grid, add the grid to CLEF_VISION_WARM_GRID at startup, "
        "or serve with CLEF_TRACED=0 (the default), which accepts any grid"
    )


def single_grid_message(n_grids):
    return (
        f"this request carries {n_grids} image or video grids; a traced server (CLEF_TRACED=1) takes one grid "
        "per request, because the tower joins several images with a concat program compiled per image count "
        "and no vision program may compile while traces are live: send one image (or one video) per request, "
        "or serve with CLEF_TRACED=0 (the default), which accepts several"
    )


@asynccontextmanager
async def lifespan(app):
    settings = Settings.from_env()
    log.info(
        "starting: model=%s snapshot=%s fake=%s mesh=%s parent=%s offset=%s prefix_cache=%d max_state=%d "
        "max_tail=%d truncate=%s traced=%s planner=%s trace_region=%d",
        settings.model,
        settings.snapshot,
        settings.fake,
        settings.mesh_shape,
        settings.parent_mesh or "auto",
        settings.submesh_offset,
        settings.prefix_cache,
        settings.max_state,
        settings.max_tail,
        settings.truncate,
        settings.traced,
        settings.planner,
        settings.trace_region,
    )
    app.state.server = await asyncio.to_thread(Server.start, settings)
    server_ = app.state.server
    log.info(
        "ready: %d worker(s) on %s, mode=%s, gdn_conv=%s, planner=%s, warm_grids=%s",
        len(server_.workers),
        server_.device_name,
        "traced" if settings.traced else "eager",
        next((w.gdn_conv for w in server_.workers if w.gdn_conv), None),
        settings.planner,
        server_.warm_grids(),
    )
    try:
        yield
    finally:
        app.state.server.close()


app = FastAPI(title="clef-tt", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-typesafe-request-id", "server-timing"],
)


def validation_detail(error):
    errors = error.errors()
    if not errors:
        return "invalid request"
    first = errors[0]
    msg = str(first.get("msg", "invalid request"))
    if msg.startswith("Value error, "):
        return msg[len("Value error, ") :]
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    return f"{loc}: {msg}" if loc else msg


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, error: RequestValidationError):
    return JSONResponse({"detail": validation_detail(error)}, 422)


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
            {"detail": "missing or invalid API key; send Authorization: Bearer <CLEF_API_KEY>"},
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
    return {"models": [{"name": name, **card} for name in server().settings.model_names]}


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
