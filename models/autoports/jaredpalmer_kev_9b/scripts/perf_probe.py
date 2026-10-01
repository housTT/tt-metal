import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.jaredpalmer_kev_9b.tt.encode import rows_for_record
from models.autoports.jaredpalmer_kev_9b.tt.head import PointerHead

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "doc" / "optimized"
DEVICE_PARAMS = dict(l1_small_size=24576, num_command_queues=2)


def load_bench():
    spec = importlib.util.spec_from_file_location("serving_bench_remote", HERE / "serving_bench_remote.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixed_ids(seed, length):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 100000, (1, length), generator=g, dtype=torch.long)


def sync(dev):
    ttnn.synchronize_device(dev)


class Clock:
    def __init__(self, dev):
        self.dev = dev
        self.marks = []

    def start(self):
        sync(self.dev)
        self.t = time.perf_counter()
        self.marks = []

    def mark(self, name):
        sync(self.dev)
        now = time.perf_counter()
        self.marks.append((name, (now - self.t) * 1000))
        self.t = now

    def split(self):
        return {k: round(v, 2) for k, v in self.marks}


def signpost(name):
    from tracy import signpost as sp

    sp(name)


def summarize(samples):
    keys = samples[0].keys()
    out = {k: round(statistics.median(s[k] for s in samples), 2) for k in keys}
    out["total"] = round(sum(out[k] for k in keys), 2)
    out["n"] = len(samples)
    return out


def eager_question(engine, handle, q, positions, clock):
    clock.start()
    tail = torch.cat([handle.suffix_ids, q], dim=1)
    engine._copy_state(handle.slot, to_live=True)
    clock.mark("restore")
    hidden = engine._run_segment(tail, handle.S0, handle.slot)
    clock.mark("forward")
    offset = handle.S - handle.S0
    out = engine._read_rows(hidden, [offset + p for p in positions])
    ttnn.deallocate(hidden)
    clock.mark("readout")
    return out, clock.split()


def traced_question(engine, handle, q, positions, clock):
    clock.start()
    tail = torch.cat([handle.suffix_ids, q], dim=1)
    engine._replay(("restore", handle.slot))
    clock.mark("restore")
    bucket = engine._segment(tail, handle.S0, handle.slot)
    clock.mark("forward")
    offset = handle.S - handle.S0
    out = engine._gather(bucket, [offset + p for p in positions])
    clock.mark("readout")
    return out, clock.split()


def question_split(engine, handle, q, positions, clock):
    if engine.traced:
        return traced_question(engine, handle, q, positions, clock)
    return eager_question(engine, handle, q, positions, clock)


def profile_hooks(engine, dev, every=8):
    layers = engine.model.layers
    for i, layer in enumerate(layers):
        orig = layer.forward

        def wrapped(*a, _orig=orig, _i=i, **kw):
            out = _orig(*a, **kw)
            if (_i + 1) % every == 0:
                ttnn.ReadDeviceProfiler(dev)
            return out

        layer.forward = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace-region", type=int, default=0)
    ap.add_argument("--traced", action="store_true")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--matmul-policy", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()
    from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine

    bench = load_bench()
    dev = ttnn.open_device(device_id=a.device_id, trace_region_size=a.trace_region, **DEVICE_PARAMS)
    dev.enable_program_cache()
    kw = {"traced": a.traced, "matmul_policy": a.matmul_policy}
    t0 = time.perf_counter()
    engine = KevEngine(dev, n_layers=a.layers, snapshot_slots=a.slots, **kw)
    build_s = time.perf_counter() - t0
    if a.profile:
        profile_hooks(engine, dev, every=min(8, len(engine.model.layers)))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ["HF_MODEL"])
    head = PointerHead(engine.args.adapter_dir if hasattr(engine.args, "adapter_dir") else os.environ["KEV_RUN"])
    clock = Clock(dev)
    reps = 1 if a.profile else a.reps
    result = {
        "mode": "traced" if a.traced else "eager",
        "layers": a.layers or 32,
        "reps": reps,
        "build_s": round(build_s, 1),
        "profile": a.profile,
        "matmul_policy": a.matmul_policy,
    }
    if a.traced:
        tr = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
        result["trace_region_bytes"] = a.trace_region
        result["trace_bytes_allocated_per_bank"] = tr.total_bytes_allocated_per_bank
        result["trace_banks"] = tr.num_banks

    state = fixed_ids(1, 2048)
    handle = engine.prefill_state(state, slot=0)
    tails = {}
    for Q in (50, 200, 450, 1000, 2000):
        q = fixed_ids(Q, Q)
        positions = [Q // 3, (2 * Q) // 3, Q - 1]
        question_split(engine, handle, q, positions, clock)
        if a.profile:
            signpost(f"PERF_TAIL_Q{Q}")
        samples = [question_split(engine, handle, q, positions, clock)[1] for _ in range(reps)]
        if a.profile:
            ttnn.ReadDeviceProfiler(dev)
            signpost(f"PERF_TAIL_Q{Q}_END")
        tails[f"Q{Q}"] = {"tail_tokens": Q, **summarize(samples)}
        logger.info(f"tail Q={Q}: {tails[f'Q{Q}']}")
    result["tail_ms"] = tails

    states = {}
    for S in (2048, 2392):
        ids = fixed_ids(S, S)
        engine.prefill_state(ids, slot=1)
        if a.profile:
            signpost(f"PERF_STATE_S{S}")
        samples = []
        for _ in range(reps):
            sync(dev)
            t = time.perf_counter()
            engine.prefill_state(ids, slot=1)
            sync(dev)
            samples.append({"prefill_state": (time.perf_counter() - t) * 1000})
        if a.profile:
            ttnn.ReadDeviceProfiler(dev)
            signpost(f"PERF_STATE_S{S}_END")
        states[f"S{S}"] = {"state_tokens": S, **summarize(samples)}
        logger.info(f"state S={S}: {states[f'S{S}']}")
    result["state_ms"] = states

    cards = {}
    for case in (bench.SHORT, bench.LONG):
        rows, _ = rows_for_record(tok, bench.request(case, 0)), None
        S = len(rows[0].state_ids)
        ids = torch.tensor([rows[0].state_ids], dtype=torch.long)

        def serve(new):
            sync(dev)
            t = time.perf_counter()
            h = engine.prefill_state(ids, slot=2) if new else handle_c
            t1 = time.perf_counter()
            probs = []
            for row in rows:
                positions = [p - S for p in row.opt_positions] + [row.decide_position - S]
                hid = engine.question_hidden(h, torch.tensor([row.question_ids], dtype=torch.long), positions)
                probs.append(head.probs(hid[-1], hid[:-1]).tolist())
            t2 = time.perf_counter()
            return h, probs, (t1 - t) * 1000, (t2 - t1) * 1000

        handle_c, probs0, _, _ = serve(True)
        if a.profile:
            signpost(f"PERF_CARD_{'SHORT' if case == bench.SHORT else 'LONG'}")
        new = [serve(True)[2:] for _ in range(reps)]
        cached = [serve(False)[2:] for _ in range(reps)]
        if a.profile:
            ttnn.ReadDeviceProfiler(dev)
            signpost(f"PERF_CARD_{'SHORT' if case == bench.SHORT else 'LONG'}_END")
        cards[case] = {
            "state_tokens": S,
            "questions": len(rows),
            "question_tokens": [len(r.question_ids) for r in rows],
            "new_ms": round(statistics.median(s + q for s, q in new), 1),
            "new_state_ms": round(statistics.median(s for s, _ in new), 1),
            "new_questions_ms": round(statistics.median(q for _, q in new), 1),
            "cached_ms": round(statistics.median(s + q for s, q in cached), 1),
            "probs": probs0,
        }
        logger.info(f"card {case}: {cards[case]}")
    result["card_ms"] = cards

    out = Path(a.out) if a.out else DOC / f"perf_probe_{result['mode']}{'_profile' if a.profile else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    logger.info(f"wrote {out}")
    if a.profile:
        ttnn.ReadDeviceProfiler(dev)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
