import json
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

ALIGN = 128
DEFAULT_TAIL_MS = ((128, 104.9), (256, 151.0), (512, 266.4), (1024, 476.4), (2048, 917.3))
DEFAULT_STATE_MS_PER_BLOCK = 57.2
DEFAULT_PERF_SUMMARY = Path(__file__).resolve().parent.parent / "doc" / "optimized" / "perf_summary.json"
PATH_PREFERENCE = ("traced_policy", "traced", "eager")
_STATE_KEY = re.compile(r"^state_(\d+)$")
_TAIL_KEY = re.compile(r"^tail_bucket_(\d+)$")


@dataclass(frozen=True)
class CostModel:
    tail_ms: tuple = DEFAULT_TAIL_MS
    state_ms_per_block: float = DEFAULT_STATE_MS_PER_BLOCK
    align: int = ALIGN

    @classmethod
    def from_perf_summary(cls, source=DEFAULT_PERF_SUMMARY):
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
        engine_ms = data.get("engine_ms", data)
        tails, ratios = {}, []
        for name, entry in engine_ms.items():
            ms = _pick_ms(entry)
            if ms is None:
                continue
            m = _TAIL_KEY.match(name)
            if m:
                tails[int(m.group(1))] = float(ms)
                continue
            m = _STATE_KEY.match(name)
            if m:
                blocks = int(m.group(1)) // ALIGN
                if blocks > 0:
                    ratios.append(float(ms) / blocks)
        if not tails:
            raise ValueError(f"no tail_bucket_<n> entries in {source}")
        per_block = round(sum(ratios) / len(ratios), 2) if ratios else DEFAULT_STATE_MS_PER_BLOCK
        return cls(tail_ms=tuple(sorted(tails.items())), state_ms_per_block=per_block)

    @classmethod
    def load(cls, source=None):
        path = Path(source) if source else DEFAULT_PERF_SUMMARY
        if not path.is_file():
            return cls()
        return cls.from_perf_summary(path)

    def aligned_state(self, state_tokens):
        return (state_tokens // self.align) * self.align

    def state_cost_ms(self, state_tokens):
        return self.state_ms_per_block * (self.aligned_state(state_tokens) // self.align)

    def bucket(self, tail_tokens):
        for b, _ in self.tail_ms:
            if tail_tokens <= b:
                return b
        return self.tail_ms[-1][0]

    def tail_cost_ms(self, state_tokens, question_tokens):
        remainder = state_tokens - self.aligned_state(state_tokens)
        b = self.bucket(remainder + question_tokens)
        return dict(self.tail_ms)[b]

    def row_tail_costs(self, rows):
        return [self.tail_cost_ms(len(r.state_ids), len(r.question_ids)) for r in rows]


def _pick_ms(entry):
    if isinstance(entry, (int, float)):
        return entry
    if isinstance(entry, dict):
        for path in PATH_PREFERENCE:
            if path in entry:
                return entry[path]
    return None


@dataclass(frozen=True)
class Policy:
    fanout: bool = True
    short_state_tokens: int = 256
    fanout_backlog_ms: float = 200.0
    min_idle_workers: int = 2


@dataclass
class WorkerView:
    id: int
    backlog_ms: float = 0.0
    cached_state_keys: object = field(default_factory=frozenset)

    def has(self, key):
        return key is not None and key in self.cached_state_keys


DEFAULT_MODEL = CostModel()
DEFAULT_POLICY = Policy()


def share_cost_ms(rows, indices, view, key, model=DEFAULT_MODEL):
    if not indices:
        return 0.0
    S = len(rows[0].state_ids)
    state = 0.0 if view.has(key) else model.state_cost_ms(S)
    return state + sum(model.tail_cost_ms(S, len(rows[i].question_ids)) for i in indices)


def plan(rows, workers, key=None, model=DEFAULT_MODEL, policy=DEFAULT_POLICY):
    if not rows:
        raise ValueError("plan needs at least one row")
    if not workers:
        raise ValueError("plan needs at least one worker")
    S = len(rows[0].state_ids)
    tails = model.row_tail_costs(rows)
    state_cost = model.state_cost_ms(S)
    offsets = {w.id: w.backlog_ms + (0.0 if w.has(key) else state_cost) for w in workers}
    hits = [w for w in workers if w.has(key)]
    idle = [w for w in workers if w.backlog_ms <= policy.fanout_backlog_ms]
    if not policy.fanout or len(rows) == 1 or len(idle) < policy.min_idle_workers:
        return [_whole(rows, workers, offsets, tails)]
    eligible = sorted({w.id: w for w in hits + idle}.values(), key=lambda w: w.id)
    assignment = _greedy(eligible, offsets, tails)
    if state_cost > 0:
        threshold = 0.0 if model.aligned_state(S) < policy.short_state_tokens else state_cost
        assignment = _prune(eligible, offsets, tails, assignment, hits, threshold)
    return [(wid, sorted(idx)) for wid, idx in sorted(assignment.items()) if idx]


def _whole(rows, workers, offsets, tails):
    total = sum(tails)
    best = min(workers, key=lambda w: (offsets[w.id] + total, w.id))
    return best.id, list(range(len(rows)))


def _greedy(workers, offsets, tails):
    order = sorted(range(len(tails)), key=lambda i: (-tails[i], i))
    load = {w.id: 0.0 for w in workers}
    assigned = {w.id: [] for w in workers}
    for i in order:
        best, best_ms = None, None
        for w in workers:
            candidate = offsets[w.id] + load[w.id] + tails[i]
            if best_ms is None or candidate < best_ms:
                best, best_ms = w.id, candidate
        load[best] += tails[i]
        assigned[best].append(i)
    return assigned


def _makespan(assignment, offsets, tails):
    return max(offsets[wid] + sum(tails[i] for i in idx) for wid, idx in assignment.items() if idx)


def _prune(workers, offsets, tails, assignment, hits, threshold):
    hit_ids = {w.id for w in hits}
    active = list(workers)
    while True:
        current = _makespan(assignment, offsets, tails)
        misses = [w for w in active if w.id not in hit_ids and assignment.get(w.id)]
        if len([w for w in active if assignment.get(w.id)]) < 2 or not misses:
            return assignment
        misses.sort(key=lambda w: (sum(tails[i] for i in assignment[w.id]), -w.id))
        dropped = False
        for w in misses:
            remaining = [v for v in active if v.id != w.id]
            trial = _greedy(remaining, offsets, tails)
            if _makespan(trial, offsets, tails) - current <= threshold:
                active, assignment, dropped = remaining, trial, True
                break
        if not dropped:
            return assignment


@dataclass
class ShareResult:
    worker_id: int
    indices: list
    probs: list
    hit: bool
    ms: float


def merge_results(n_rows, shares):
    covered = sorted(i for s in shares for i in s.indices)
    if covered != list(range(n_rows)):
        raise ValueError(f"shares cover rows {covered}, expected 0..{n_rows - 1}")
    probs = [None] * n_rows
    for s in shares:
        if len(s.probs) != len(s.indices):
            raise ValueError(f"worker {s.worker_id} returned {len(s.probs)} rows for {len(s.indices)} indices")
        for i, p in zip(s.indices, s.probs):
            probs[i] = p
    ordered = sorted(shares, key=lambda s: s.indices[0])
    stats = {
        "latency_ms": round(max(s.ms for s in ordered), 1),
        "latency_ms_sum": round(sum(s.ms for s in ordered), 1),
        "prefix_cache_hit": all(s.hit for s in ordered),
        "worker": ordered[0].worker_id,
        "workers": [s.worker_id for s in ordered],
        "shares": [
            {"worker": s.worker_id, "rows": list(s.indices), "latency_ms": round(s.ms, 1), "prefix_cache_hit": s.hit}
            for s in ordered
        ],
    }
    return probs, stats


def collect(parts, n_rows):
    done = Future()
    lock = threading.Lock()
    pending = {"n": len(parts)}
    results = []

    def finish(worker_id, indices, f):
        with lock:
            if done.done():
                return
            exc = f.exception()
            if exc is not None:
                done.set_exception(exc)
                return
            probs, hit, ms = f.result()
            results.append(ShareResult(worker_id, list(indices), probs, hit, ms))
            pending["n"] -= 1
            if pending["n"]:
                return
        try:
            done.set_result(merge_results(n_rows, results))
        except Exception as e:
            done.set_exception(e)

    for worker_id, indices, f in parts:
        f.add_done_callback(lambda f, w=worker_id, i=indices: finish(w, i, f))
    return done
