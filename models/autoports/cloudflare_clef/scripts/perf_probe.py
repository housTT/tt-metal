"""Engine-level latency probe for the Clef TP=2 engine: eager versus traced per bucket and per state length.

Device script (through devrun). With --profile the probe places Tracy signposts around one warmed
1024-token prefill in each mode (PERF_EAGER_1024 / PERF_TRACED_1024) and around one 2,200-token state
prefill per mode, so `python -m tracy -r -p -v scripts/perf_probe.py --profile ...` gives one ops CSV
per window for tt-perf-report. Without --profile it prints medians of --reps runs and writes --out.

Usage:
  python perf_probe.py [--layers 4] [--reps 5] [--profile] [--out /path.json] [--trace-region BYTES]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from loguru import logger

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
os.environ.setdefault("HF_MODEL", SNAPSHOT)

import ttnn
from models.autoports.cloudflare_clef.tt.engine import BUCKETS, REFERENCE_GRIDS, ClefEngine, tp2_mesh

os.environ.setdefault("CLEF_VISION_WARM_GRID", REFERENCE_GRIDS)


def signpost(name):
    from tracy import signpost as sp

    sp(name)


def fixed_ids(seed, length):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (1, length), generator=g, dtype=torch.long)


def timed(fn, mesh, reps):
    fn()
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        samples.append(time.perf_counter() - t)
    return round(statistics.median(samples) * 1000, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace-region", type=int, default=1 << 30)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--parent", default=os.environ.get("CLEF_PARENT", "1x4"))
    a = ap.parse_args()
    out = {"layers": a.layers, "reps": a.reps, "profile": a.profile}
    with tp2_mesh(a.parent, trace_region_size=a.trace_region) as mesh:
        engine = ClefEngine(mesh, n_layers=a.layers, snapshot_slots=a.slots, traced=True)
        out["engine"] = dict(
            timings=engine.timings, trace_mib=round(engine.trace_bytes / 2**20, 1), traces=len(engine.traces)
        )
        schema = fixed_ids(7, 198)
        if a.profile:
            x1024 = fixed_ids(1, 1024)
            x2200 = fixed_ids(2, 2200)
            for traced in (False, True):
                name = "TRACED" if traced else "EAGER"
                engine.traced = traced
                with engine._misses_allowed():
                    engine.prefill_hidden(x1024, slot=0)
                    engine.prefill_hidden(x1024, slot=0)
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_1024")
                    engine.prefill_hidden(x1024, slot=0)
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_1024_END")
                    engine.prefill_state(x2200, slot=1, key=f"p{name}")
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_STATE_2200")
                    handle = engine.prefill_state(x2200, slot=1, key=f"p{name}")
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_STATE_2200_END")
                    engine.schema_hidden(handle, schema)
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_SCHEMA_198")
                    engine.schema_hidden(handle, schema)
                    ttnn.synchronize_device(mesh)
                    signpost(f"PERF_{name}_SCHEMA_198_END")
            engine.traced = True
        else:
            out["buckets"] = {}
            for b in BUCKETS:
                x = fixed_ids(b, b)
                row = {}
                for traced in (True, False):
                    engine.traced = traced
                    with engine._misses_allowed():
                        row["traced_ms" if traced else "eager_ms"] = timed(
                            lambda: engine.prefill_hidden(x, slot=0), mesh, a.reps
                        )
                engine.traced = True
                out["buckets"][b] = row
                logger.info(f"bucket {b}: {row}")
            out["state"] = {}
            for S in (2200, 8192):
                x = fixed_ids(S, S)
                row = {}
                for traced in (True, False):
                    engine.traced = traced
                    name = "traced" if traced else "eager"
                    with engine._misses_allowed():
                        row[f"{name}_prefill_ms"] = timed(
                            lambda: engine.prefill_state(x, slot=0, key=f"{name}{S}"), mesh, 3
                        )
                        handle = engine.prefill_state(x, slot=0, key=f"{name}{S}")
                        row[f"{name}_schema_198_ms"] = timed(lambda: engine.schema_hidden(handle, schema), mesh, a.reps)
                engine.traced = True
                out["state"][S] = row
                logger.info(f"state {S}: {row}")
            out["counters"] = dict(engine.counters)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=1, default=str)
        logger.info(json.dumps(out, default=str))
    print("PERF_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
