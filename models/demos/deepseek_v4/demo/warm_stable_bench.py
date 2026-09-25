# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prove the fix: warm the FULL host bf4 mem cache (all 11008 experts) so steady-state decode
has NO cold builds / disk loads regardless of the (non-deterministic) routing, then measure
per-token latency. Expect FLAT ~449-600ms (2.0-2.2 tok/s) with disk=0 cold=0 every token —
vs the LRU-off baseline 1.69 tok/s. Also fully populates the disk cache (future cold boots fast)."""
import argparse, os, statistics, time
import torch
import ttnn
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator

T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--meas", type=int, default=16)
    ap.add_argument("--mesh", type=int, default=4)
    ap.add_argument("--skip_warm_all", action="store_true")
    ap.add_argument("--build_only", action="store_true",
                    help="only populate the disk cache (retain=False, low RSS) then exit; no measure")
    ap.add_argument("--pretouch", action="store_true",
                    help="read all cache file payloads into page cache before measuring (tests whether "
                         "the residual per-token spikes are mmap page-cache faults)")
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh))
    try:
        log(f"mesh open n={mesh.get_num_devices()}")
        gen = DeepSeekV4Generator(mesh, num_layers=args.layers)
        tok = gen.tokenizer
        fd = gen._ensure_fast()  # build FastDecoder (+ fp4 cache) now, before warm_all
        cache = fd.fp4_cache

        if args.build_only:
            log(f"build_only: populate disk cache for {args.layers}x{args.experts} experts (retain=False)...")
            cache.warm_all(args.layers, args.experts, log=log, retain=False)
            print("BUILD_ONLY_OK", flush=True)
            return
        if not args.skip_warm_all:
            log(f"warm_all: fill self.c for {args.layers}x{args.experts} experts (retain=True; disk-hits)...")
            cache.warm_all(args.layers, args.experts, log=log, retain=True)

        ids = tok(args.prompt, return_tensors="pt").input_ids
        log("prefill + generate corpus...")
        logits = gen.prefill_fast(ids)
        out, nxt = [int(logits[0].argmax(-1))], None
        nxt = out[0]
        for _ in range(args.gen - 1):
            logits = gen.decode_fast(nxt); nxt = int(logits[0].argmax(-1)); out.append(nxt)
        log(f"corpus COMPLETION={tok.decode(out)!r}")

        if args.pretouch and getattr(cache, "disk_dir", None):
            import glob as _glob
            files = _glob.glob(os.path.join(cache.disk_dir, "*.tensorbin"))
            t = time.perf_counter(); nbytes = 0
            for fp in files:
                with open(fp, "rb") as fh:
                    nbytes += len(fh.read())  # fault the payload into page cache
            log(f"pretouch: read {len(files)} files ({nbytes/1e9:.0f}GB) into page cache in {time.perf_counter()-t:.0f}s")

        full = ids[0].tolist() + out
        meas = min(args.meas, len(full) - 1)
        prefix, tail = full[: len(full) - meas], full[len(full) - meas :]
        log(f"steady-state measure: reset + prefill {len(prefix)} + {len(tail)} tail (pure async)")
        fd.reset(); fd.prefill(torch.tensor(prefix).reshape(1, -1))
        dts = []
        for j, t_id in enumerate(tail):
            s0 = dict(cache.stats)
            t = time.perf_counter()
            fd.decode_step(int(t_id))
            ttnn.synchronize_device(mesh)
            dt = (time.perf_counter() - t) * 1000
            d = {k: cache.stats[k] - s0[k] for k in cache.stats}
            dts.append(dt)
            log(f"  tok{j+1:>2} {dt:6.0f}ms | mem={d['mem']} lru={d['lru']} disk={d['disk']} cold={d['cold']}")
        mean = statistics.mean(dts); med = statistics.median(dts); sd = statistics.pstdev(dts)
        log(f"STEADY mean={mean:.0f}ms med={med:.0f}ms sd={sd:.0f} CV={sd/mean:.2f} "
            f"[{min(dts):.0f},{max(dts):.0f}] -> med {1000/med:.2f} tok/s  mean {1000/mean:.2f} tok/s")
        print("WARM_STABLE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception: pass


if __name__ == "__main__":
    main()
