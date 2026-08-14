# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Expert parallelism vs intermediate-dim tensor parallelism for the routed-expert chain.

This is the one design decision of the multichip stage that is not a program-config knob: with 4
devices and 256 top-8 routed experts, the routed half of the MoE can be split either way.

``ep`` (shipped)
    64 experts per device, each expert **whole** (``moe_intermediate`` 512). Every device runs the
    inherited chain on its own expert block and produces a partial sum over experts; the layer's
    existing MoE collective closes it. Per device the sparse matmul loops once per *local* active
    expert, which is a random variable: 8 distinct experts into 4 blocks of 64, mean 2 per device
    and an expected maximum over the four of 3.51 (the step waits for the slowest device).
``tp``
    all 256 experts on every device with the intermediate sharded 4 ways (gate/up 128 columns each,
    down 128 rows). Every device loops over all 8 active experts, at a quarter of the output width.

The chain measured is exactly ``OptimizedMoE._routed_experts``: packed gate/up ``sparse_matmul``,
the two unpacking slices, the SwiGLU multiply, the score multiply, the down ``sparse_matmul`` and
the expert reduction — i.e. including the ``num_experts``-wide intermediates, which is where the two
schemes differ by more than the matmuls do.

``single`` is the unsharded single-chip chain, for scale.

    python .../doc/multichip_decoder/logs/probe_expert_parallel.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY, TILE, OptimizedMoE


#: Arms: ``(name, experts per device, moe_intermediate per device, active experts per device)``.
#: The EP arm's active count is swept because it is a random variable; the TP arm's is fixed at the
#: model's ``num_experts_per_tok``.
def arms(cfg, tp, phase):
    """Arms at the active-expert counts each phase actually produces.

    The layer always calls ``_routed_experts`` with one 32-token expert group, so the phases differ
    only in how many experts that group activates: 8 at batch-1 decode (one token, top-8) and, for a
    full 32-token prefill group, the expected distinct union of 256 draws --
    ``E*(1-(1-1/E)^(32*top_k))``, i.e. ~162 of 256 globally and ~41 of 64 on one device under EP.
    """
    e, i, k = cfg.num_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
    if phase == "decode":
        return [
            ("single", e, i, [k]),
            ("ep", e // tp, i, list(range(1, k + 1))),
            ("tp", e, i // tp, [k]),
        ]
    union = lambda n: max(1, round(n * (1 - (1 - 1 / n) ** (TILE * k))))
    return [
        ("single", e, i, [union(e)]),
        ("ep", e // tp, i, sorted({union(e // tp), e // tp, max(1, union(e // tp) // 2)})),
        ("tp", e, i // tp, [union(e)]),
    ]


def expected_max_load(draws: int, buckets: int, per_bucket: int) -> float:
    """E[max bucket count] when ``draws`` **distinct** experts land in ``buckets`` equal blocks.

    The gate picks ``draws`` distinct experts out of ``buckets * per_bucket``, so the per-device
    counts are multivariate **hyper**geometric, not multinomial. Modelling it as ``draws``
    independent uniform assignments (i.e. sampling with replacement) is the obvious shortcut and is
    wrong in the safe direction — it overstates clustering, and therefore the slowest device's loop
    count — by about 0.03 experts at the shape this model uses. Small, but the figure is quoted as
    exact in the README, so it is computed exactly here.

    Exact by enumeration over compositions of ``draws`` into ``buckets`` parts, weighted by
    ``prod(C(per_bucket, k_i)) / C(buckets * per_bucket, draws)``. ``draws`` is 8 and ``buckets`` 4,
    so this is cheap and needs no simulation.
    """
    from math import comb

    total = 0.0
    denom = comb(buckets * per_bucket, draws)

    def walk(bucket: int, left: int, counts: list[int]) -> None:
        nonlocal total
        if bucket == buckets - 1:
            full = counts + [left]
            if left > per_bucket:
                return
            weight = 1
            for k in full:
                weight *= comb(per_bucket, k)
            total += weight * max(full) / denom
            return
        for k in range(min(left, per_bucket) + 1):
            walk(bucket + 1, left - k, counts + [k])

    walk(0, draws, [])
    return total


def build_moe(mesh, cfg, experts, inter, policy=DEFAULT_POLICY):
    local = replace(cfg, num_experts=experts, moe_intermediate_size=inter)
    gen = torch.Generator().manual_seed(19)
    gate_up = torch.randn(1, experts, cfg.dim, 2 * inter, generator=gen) * 0.02
    down = torch.randn(1, experts, inter, cfg.dim, generator=gen) * 0.02

    def up(t, dtype):
        return ttnn.as_tensor(
            t.float().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    weights = {
        "expert_gate_up": up(gate_up, policy.expert_gate_up_dtype),
        "expert_down": up(down, policy.expert_down_dtype),
    }
    return OptimizedMoE(mesh, local, weights, policy=policy), local


def routing_for(mesh, local_cfg, tokens, active):
    """Dense routing ``[1, 1, tokens, E]`` whose first ``active`` experts carry weight."""
    host = torch.zeros(1, 1, tokens, local_cfg.num_experts)
    host[..., :active] = 1.0 / active
    return ttnn.from_torch(
        host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def timeit(mesh, fn, iters=10, warmup=3, repeats=3):
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return min(samples), max(samples) - min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--tokens", default="32")
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    cfg = OrnithDecoderConfig.from_hf_config(R.load_text_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    print(f"# routed-expert chain: EP vs intermediate-TP, tp={args.tp}")
    print(
        f"# E[max local active experts] over {args.tp} devices at top-{cfg.num_experts_per_tok}: "
        f"{expected_max_load(cfg.num_experts_per_tok, args.tp, cfg.num_experts // args.tp):.3f}"
    )
    print("# columns: phase arm experts_per_device inter_per_device active tokens us spread")
    try:
        for tokens in [int(v) for v in args.tokens.split(",")]:
            gen = torch.Generator().manual_seed(23)
            x = ttnn.from_torch(
                (torch.randn(1, 1, tokens, cfg.dim, generator=gen) * 0.5).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for name, experts, inter, actives in arms(cfg, args.tp, args.phase):
                moe, local = build_moe(mesh, cfg, experts, inter)
                moe._decode_phase = tokens <= TILE
                moe._call_tokens = tokens
                for active in actives:
                    if active > experts:
                        continue
                    dense = routing_for(mesh, local, tokens, active)

                    def run(dense=dense):
                        return moe._routed_experts(x, dense, tokens, valid_tokens=None)

                    try:
                        us, spread = timeit(mesh, run, iters=args.iters)
                        print(
                            f"MOEPAR {args.phase} {name} {experts} {inter} {active} {tokens} {us:.2f} {spread:.2f}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"MOEPAR {args.phase} {name} {experts} {inter} {active} {tokens} FAIL {type(exc).__name__}"
                        )
                    ttnn.deallocate(dense)
                for tensor in moe.w.values():
                    ttnn.deallocate(tensor)
                del moe
            ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
