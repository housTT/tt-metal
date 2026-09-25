# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Root-cause recurring cold expert builds. Decode the SAME token twice from an
identical state in one process. Capture (a) routed expert indices per layer via
ROUTE_TRACE and (b) the expert-cache stats delta. Disambiguates:
  pass2 cold>0 AND routing identical  -> cache-persistence bug (self.c/disk)
  pass2 cold>0 AND routing differs    -> non-deterministic routing (bf16 topk ties)
  pass2 cold==0                       -> cache works; recurring cold was position/warmup artifact
"""
import argparse, time
import torch
import ttnn
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator

T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def decode_once(fd, prefix, tail_id):
    fd.reset()
    fd.prefill(torch.tensor(prefix).reshape(1, -1))
    s0 = dict(fd.fp4_cache.stats)
    D.ROUTE_TRACE = []
    t = time.perf_counter()
    fd.decode_step(int(tail_id))
    ttnn.synchronize_device(fd.device)
    dt = (time.perf_counter() - t) * 1000
    route = {li: tuple(sorted(ids)) for li, ids in D.ROUTE_TRACE}
    D.ROUTE_TRACE = None
    delta = {k: fd.fp4_cache.stats[k] - s0[k] for k in fd.fp4_cache.stats}
    return dt, route, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prefix_extra", type=int, default=4, help="extra generated tokens before the measured one")
    ap.add_argument("--mesh", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh))
    try:
        log(f"mesh open n={mesh.get_num_devices()}")
        gen = DeepSeekV4Generator(mesh, num_layers=args.layers)
        tok = gen.tokenizer
        ids = tok(args.prompt, return_tensors="pt").input_ids
        log("prefill + generate a few tokens to form prefix+target...")
        logits = gen.prefill_fast(ids)
        seq = ids[0].tolist()
        nxt = int(logits[0].argmax(-1)); seq.append(nxt)
        for _ in range(args.prefix_extra):
            logits = gen.decode_fast(nxt); nxt = int(logits[0].argmax(-1)); seq.append(nxt)
        prefix = seq[:-1]
        target = seq[-1]
        fd = gen._fast
        log(f"prefix len={len(prefix)} target_id={target}")

        prev_route = None
        for p in range(args.passes):
            dt, route, delta = decode_once(fd, prefix, target)
            up = delta["disk"] + delta["cold"]
            log(f"PASS {p+1}: {dt:.0f}ms | mem={delta['mem']} lru={delta['lru']} disk={delta['disk']} "
                f"cold={delta['cold']} up={up}")
            if prev_route is not None:
                diffs = [li for li in route if route[li] != prev_route.get(li)]
                log(f"  routing vs prev pass: {len(diffs)}/{len(route)} layers differ"
                    + (f" (layers {diffs[:8]}{'...' if len(diffs)>8 else ''})" if diffs else " -> IDENTICAL"))
            prev_route = route
        print("PROBE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception: pass


if __name__ == "__main__":
    main()
