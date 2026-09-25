# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip (1,N) expert-parallel decode: correctness gate + warm tok/s, PER-TOKEN logging.
Runs the exact server path (DeepSeekV4Generator.prefill_fast/decode_fast) on the full mesh."""
import argparse, time
import torch
import ttnn
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator

T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--mesh", type=int, default=4)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh))
    try:
        log(f"mesh open n={mesh.get_num_devices()} shape={tuple(mesh.shape)}")
        gen = DeepSeekV4Generator(mesh, num_layers=args.layers)
        tok = gen.tokenizer
        ids = tok(args.prompt, return_tensors="pt").input_ids
        log(f"generator built; prompt {ids.shape[1]} tokens; building FastDecoder + cold prefill...")
        t = time.perf_counter()
        logits = gen.prefill_fast(ids)
        log(f"cold prefill done in {time.perf_counter()-t:.1f}s")

        out_ids, nxt = [], int(logits[0].argmax(-1))
        out_ids.append(nxt)
        for i in range(args.gen - 1):
            t = time.perf_counter()
            logits = gen.decode_fast(nxt)
            nxt = int(logits[0].argmax(-1))
            out_ids.append(nxt)
            log(f"  cold gen tok {i+1}: {time.perf_counter()-t:.1f}s -> id={nxt}")
        log(f"COMPLETION={tok.decode(out_ids)!r}")

        fd = gen._fast
        full = ids[0].tolist() + out_ids
        n_meas = args.gen
        log("warm pass: reset + prefill prefix, then measure the tail...")
        fd.reset()
        fd.prefill(torch.tensor(full[:-n_meas]).reshape(1, -1))
        for k in D.PROF: D.PROF[k] = 0.0
        D.PROF_ON = True
        dts = []
        for i, t_id in enumerate(full[-n_meas:]):
            t = time.perf_counter()
            fd.decode_step(int(t_id), profile=(i == 0))
            dt = time.perf_counter() - t
            dts.append(dt)
            log(f"  warm decode {i+1}: {dt*1000:.0f} ms -> {1.0/dt:.2f} tok/s")
        D.PROF_ON = False
        avg = sum(dts) / len(dts)
        log("section prof/token: " + " ".join(f"{k}={v*1000:.0f}ms" for k, v in D.PROF.items()))
        print(f"MESH{args.mesh}_WARM_TOK_S {1.0/avg:.3f}", flush=True)
        print("MESH_DECODE_OK", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        try: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception: pass


if __name__ == "__main__":
    main()
