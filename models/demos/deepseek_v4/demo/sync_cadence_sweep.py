# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolate the async per-token variance. Warm K-LRU decode, sweep the in-loop sync
cadence (syncs/token). PURE async measurement (no per-section PROF sync). Controlled:
same prefix re-prefilled before each cadence block, same tail measured. Logs per-token
wall-clock + expert-cache hit deltas (mem/disk/cold/lru) so slow tokens can be checked
against upload activity. Reports min/max/mean/median/stdev/CV per cadence."""
import argparse, statistics, time
import torch
import ttnn
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt import fast_decode as FD
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator

T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=20, help="tokens to generate for the corpus")
    ap.add_argument("--meas", type=int, default=16, help="tail tokens to measure each cadence")
    ap.add_argument("--mesh", type=int, default=4)
    ap.add_argument("--cadences", default="0,21,10,5,2,1")
    args = ap.parse_args()
    cadences = [int(x) for x in args.cadences.split(",")]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh))
    try:
        log(f"mesh open n={mesh.get_num_devices()}")
        gen = DeepSeekV4Generator(mesh, num_layers=args.layers)
        tok = gen.tokenizer
        ids = tok(args.prompt, return_tensors="pt").input_ids
        log("cold prefill (disk-load experts)...")
        t = time.perf_counter()
        logits = gen.prefill_fast(ids)
        log(f"cold prefill done {time.perf_counter()-t:.1f}s")
        out_ids, nxt = [], int(logits[0].argmax(-1))
        out_ids.append(nxt)
        for g in range(args.gen - 1):
            t = time.perf_counter()
            logits = gen.decode_fast(nxt)
            nxt = int(logits[0].argmax(-1))
            out_ids.append(nxt)
            log(f"  corpus gen {g+1}/{args.gen-1}: {time.perf_counter()-t:.1f}s -> id={nxt}")
        log(f"corpus COMPLETION={tok.decode(out_ids)!r}")
        fd = gen._fast
        full = ids[0].tolist() + out_ids
        meas = min(args.meas, len(full) - 1)
        prefix = full[: len(full) - meas]
        tail = full[len(full) - meas :]
        cache = fd.fp4_cache

        # extra warm pass so ALL measured experts are resident (isolate dispatch from upload)
        log("warm pass 1 (populate LRU residency)...")
        fd.reset(); fd.prefill(torch.tensor(prefix).reshape(1, -1))
        for wi, t_id in enumerate(tail):
            fd.decode_step(int(t_id))
            if wi % 4 == 0: log(f"  warm1 {wi+1}/{len(tail)}")

        results = {}
        for cad in cadences:
            FD.SYNC_EVERY = cad
            log(f"--- cadence SYNC_EVERY={cad}: reset + prefill prefix ({len(prefix)} tok) ---")
            fd.reset(); fd.prefill(torch.tensor(prefix).reshape(1, -1))
            dts, rows = [], []
            for t_id in tail:
                s0 = dict(cache.stats)
                t = time.perf_counter()
                fd.decode_step(int(t_id))
                ttnn.synchronize_device(mesh)  # token boundary (greedy needs the logits anyway)
                dt = (time.perf_counter() - t) * 1000
                d = {k: cache.stats[k] - s0[k] for k in cache.stats}
                dts.append(dt); rows.append((dt, d))
            mean = statistics.mean(dts); med = statistics.median(dts)
            sd = statistics.pstdev(dts); cv = sd / mean if mean else 0
            results[cad] = (mean, med, sd, cv, min(dts), max(dts))
            log(f"CAD={cad:>3} n={len(dts)} mean={mean:.0f} med={med:.0f} sd={sd:.0f} "
                f"CV={cv:.2f} min={min(dts):.0f} max={max(dts):.0f} tok/s(med)={1000/med:.2f}")
            for j, (dt, d) in enumerate(rows):
                up = d["disk"] + d["cold"]  # uploads this token (disk load or cold build)
                log(f"    cad{cad} tok{j+1:>2} {dt:6.0f}ms | mem={d['mem']} lru={d['lru']} disk={d['disk']} cold={d['cold']} up={up}")
        log("=== SUMMARY (mean / med / sd / CV / min / max ms) ===")
        for cad in cadences:
            mean, med, sd, cv, mn, mx = results[cad]
            log(f"  SYNC_EVERY={cad:>3}: mean={mean:.0f} med={med:.0f} sd={sd:.0f} CV={cv:.2f} "
                f"[{mn:.0f},{mx:.0f}] -> {1000/med:.2f} tok/s")
        print("SWEEP_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception: pass


if __name__ == "__main__":
    main()
