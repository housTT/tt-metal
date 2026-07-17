# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolation micro-benchmark: fastest correct on-device greedy path for the
Kokoro reconstruction readout logits (replicated tiny vocab, sequence-sharded).

The stage-05 token-out path did ttnn.argmax(logits_s, dim=-1) on a TILE tensor,
which the ttnn argmax contract runs SINGLE-CORE (~112 us device, dominant
avoidable terminal cost). ROW_MAJOR last-dim argmax is multi-core. This script
times the candidates on the real per-shard logits shapes and verifies each
returns the greedy token identical to a host argmax.

Run:
  env TT_METAL_HOME=/home/ttuser/dev/tt-metal \
      PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal \
      python models/autoports/hexgrad_kokoro_82m/doc/optimized_full_model/argmax_iso.py
"""
import time

import torch

import ttnn

VOCAB_PAD = 192  # _round_up(178, 32)
REAL_VOCAB = 178


def _timeit(fn, iters=50):
    fn()  # warm
    ttnn.synchronize_device(fn.mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(fn.mesh)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    try:
        for full_seq in (128, 512):
            local = full_seq // 4
            torch.manual_seed(0)
            host = torch.randn(1, 1, local, VOCAB_PAD)
            host[..., REAL_VOCAB:] = -1e9  # pad mask
            # reference in bf16 (matches the on-device logit dtype; fp32 argmax can
            # disagree on near-ties, which is a synthetic-test artifact not a bug)
            ref = host[0, 0].to(torch.bfloat16).float().argmax(-1).tolist()

            log_tile = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

            def _tok(t):
                # replicated -> device 0 copy, flattened token list
                return (
                    ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0]
                    .reshape(-1)
                    .to(torch.long)
                    .tolist()
                )

            # (a) current: TILE dim=-1 (single-core)
            def a():
                t = ttnn.argmax(log_tile, dim=-1, keepdim=False)
                ttnn.deallocate(t)

            a.mesh = mesh

            # (b) untilize -> ROW_MAJOR -> argmax dim=-1 (multi-core)
            def b():
                rm = ttnn.to_layout(log_tile, ttnn.ROW_MAJOR_LAYOUT)
                t = ttnn.argmax(rm, dim=-1, keepdim=False)
                ttnn.deallocate(rm)
                ttnn.deallocate(t)

            b.mesh = mesh

            # correctness for (b)
            rm = ttnn.to_layout(log_tile, ttnn.ROW_MAJOR_LAYOUT)
            tok_b = ttnn.argmax(rm, dim=-1, keepdim=False)
            ok_b = _tok(tok_b) == ref
            ttnn.deallocate(rm)
            ttnn.deallocate(tok_b)

            # correctness for (a)
            tok_a = ttnn.argmax(log_tile, dim=-1, keepdim=False)
            ta_toks = _tok(tok_a)
            ok_a = ta_toks == ref
            ttnn.deallocate(tok_a)
            # device methods must agree with each other (both bf16 on-device)
            rm2 = ttnn.to_layout(log_tile, ttnn.ROW_MAJOR_LAYOUT)
            tokb2 = ttnn.argmax(rm2, dim=-1, keepdim=False)
            ab_agree = _tok(tokb2) == ta_toks
            ttnn.deallocate(rm2)
            ttnn.deallocate(tokb2)
            print(f"    (a)==(b) device agree: {ab_agree}", flush=True)

            ta = _timeit(a)
            tb = _timeit(b)
            print(
                f"full_seq={full_seq} local={local}: "
                f"(a) TILE single-core argmax = {ta*1e3:.1f} us [ok={ok_a}]  |  "
                f"(b) ROW_MAJOR multicore argmax = {tb*1e3:.1f} us [ok={ok_b}]",
                flush=True,
            )
            ttnn.deallocate(log_tile)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
