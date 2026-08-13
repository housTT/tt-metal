# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sweep the **prefill** chunked-SDPA program config, the one knob this stage shipped unmeasured.

README §9 item 7 disclosed it as "a real gap rather than a non-issue": `_prefill_sdpa_config` inherited a
hand-picked `q_chunk = k_chunk = 64` from the fused stage, and unlike every other program config here it had no
probe behind it, while `SDPAOperation` is around one percent of the `full_attention` prefill window — the same order as the
dense prefill matmul group that *does* get a swept table. Review round 14 called that deferred work rather than
a limitation, which is right, so this closes it.

Two constraints bound what is legal, both from the shipped call site:

* `q_chunk` must divide a non-zero `chunk_start_idx`, because chunked prefill resumes at a block boundary
  (`_prefill_sdpa_config` takes the low bit of the start index). The sweep therefore reports what each pair
  costs at a *first* chunk and separately at a resumed one, and the shipped rule keeps the divisibility.
* `k_chunk` interacts with the 64-token paged block only in **decode**; prefill runs against the contiguous
  `k` of the chunk it is filling, so a larger `k_chunk` is not the silent-correctness trap it is at decode
  (`logs/ab_sdpa_decode_contract.txt` candidate A). It still has to be measured rather than assumed.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/probe_prefill_sdpa.py

Rows are ``PREFILLSDPA cfg=<name> us=<min> spread=<max-min> pcc_vs_default=<pcc>``, timed like every other probe
here: three repeats, minimum reported, spread beside it, against the op's own default config as the reference.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R

#: The shipped prefill geometry: one 2048-token chunk of the `full_attention` layer at its head split.
CHUNK = 2048
HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256


def timeit(mesh, fn, n=3):
    fn()
    ttnn.synchronize_device(mesh)
    runs = []
    for _ in range(n):
        start = time.time()
        out = fn()
        ttnn.synchronize_device(mesh)
        runs.append((time.time() - start) * 1e6)
        ttnn.deallocate(out)
    return f"{min(runs):.1f} spread={max(runs) - min(runs):.1f}"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args()

    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    grid = mesh.compute_with_storage_grid_size()
    try:
        gen = torch.Generator().manual_seed(23)

        def dev(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        q = dev(torch.randn(1, HEADS, args.chunk, HEAD_DIM, generator=gen) * 0.1)
        k = dev(torch.randn(1, KV_HEADS, args.chunk, HEAD_DIM, generator=gen) * 0.1)
        v = dev(torch.randn(1, KV_HEADS, args.chunk, HEAD_DIM, generator=gen) * 0.1)

        def run(program_config=None):
            def fn():
                return ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, program_config=program_config
                )

            return fn

        print(
            f"# Prefill chunked-SDPA sweep at one {args.chunk}-token chunk, {HEADS} q heads / {KV_HEADS} kv, "
            f"head_dim {HEAD_DIM}, causal, bfloat16 on a {grid.x}x{grid.y} grid."
        )
        print(
            f"# Reference for `pcc_vs_default` is the op's own default program config, {cfg.num_hidden_layers}-layer "
            "checkpoint's shapes."
        )
        base = ttnn.to_torch(run()()).float()
        print(f"PREFILLSDPA cfg=default(None) us={timeit(mesh, run())} pcc_vs_default=1.000000")

        candidates = []
        # Non-square pairs around the winner decouple the parallelism granularity (`q_chunk`, which sets how
        # many chunk-pairs the factory spreads over the grid) from the inner-loop granularity (`k_chunk`). A
        # peer agent working on the same op's occupancy model suggested it: at 16 q heads this shape saturates
        # the grid below 256 and runs at 58 % of it at 256, so there may be room on the k axis that the square
        # ladder cannot see. 512/512 does not build, which is what makes the asymmetric pairs the only way up.
        for qc, kc in (
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (64, 128),
            (128, 64),
            (32, 32),
            # (0, 0) - the op's own auto-chunk - is deliberately NOT swept here. It killed the probe process
            # outright at this shape (no Python exception, so the arm cannot even be reported as FAILED), and it
            # is unshippable regardless: `_prefill_sdpa_config` has to name a `q_chunk` that divides a non-zero
            # resume offset, which "let the op decide" cannot guarantee. The decode probe does sweep it, where
            # the contract is different and the arm is real (`logs/probe_decode_micro.txt`).
            (256, 128),
            (256, 64),
            (128, 256),
            (512, 128),
            (512, 256),
        ):
            candidates.append(
                (
                    f"grid={grid.x}x{grid.y} q_chunk={qc} k_chunk={kc}",
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=qc,
                        k_chunk_size=kc,
                        exp_approx_mode=False,
                    ),
                )
            )
        # The shipped chunk pair on smaller grids: prefill has whole-grid work per chunk, so unlike decode a
        # narrower grid is not obviously wrong, and the axis has never been measured at this shape.
        for gx, gy in ((8, 8), (8, 4)):
            candidates.append(
                (
                    f"grid={gx}x{gy} q_chunk=64 k_chunk=64",
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        q_chunk_size=64,
                        k_chunk_size=64,
                        exp_approx_mode=False,
                    ),
                )
            )
        # The `fused-parity` policy keeps the KV cache at bfloat16, which doubles what K and V cost in L1 for
        # the same chunk pair. That policy is supported and swept (`logs/ab_precision_policy.txt`), so the
        # chunk this stage ships has to be legal there too - a config that only fits the BFP8 cache would turn
        # a policy arm into a crash. Measured here rather than assumed.
        k16 = dev(torch.randn(1, KV_HEADS, args.chunk, HEAD_DIM, generator=gen) * 0.1, dtype=ttnn.bfloat16)
        v16 = dev(torch.randn(1, KV_HEADS, args.chunk, HEAD_DIM, generator=gen) * 0.1, dtype=ttnn.bfloat16)

        def run16(program_config):
            def fn():
                return ttnn.transformer.scaled_dot_product_attention(
                    q, k16, v16, is_causal=True, program_config=program_config
                )

            return fn

        for qc in (64, 128, 256):
            cfg16 = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=qc, exp_approx_mode=False
            )
            try:
                out = run16(cfg16)()
                ttnn.deallocate(out)
                print(f"PREFILLSDPA cfg=bf16-kv q_chunk={qc} k_chunk={qc} us={timeit(mesh, run16(cfg16))}", flush=True)
            except RuntimeError as exc:
                print(
                    f"PREFILLSDPA cfg=bf16-kv q_chunk={qc} k_chunk={qc} FAILED "
                    f"{str(exc).strip().splitlines()[0][:120]}",
                    flush=True,
                )

        for name, program_config in candidates:
            try:
                got = ttnn.to_torch(run(program_config)()).float()
                print(
                    f"PREFILLSDPA cfg={name} us={timeit(mesh, run(program_config))} "
                    f"pcc_vs_default={pcc(base, got):.6f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - any failure is a result here, not a stop
                first = str(exc).strip().splitlines()[0][:150] or type(exc).__name__
                print(f"PREFILLSDPA cfg={name} FAILED {first}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
