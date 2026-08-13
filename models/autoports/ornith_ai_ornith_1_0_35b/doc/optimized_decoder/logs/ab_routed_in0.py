# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer A/B for the two round-14 changes whose evidence has to be a *layer* measurement.

**The routed gate/up `in0` placement.** `tt-perf-report` raises "If possible place input 0 in L1" on the largest
op of the prefill window — but only once the report is given `--active-experts`, because it cannot model a
`sparse_matmul` row whose `nnz` is `std::nullopt` and its advice generator early-returns on such a row. Review
round 14 found every committed report in that state, so the item had never been read. It costs no extra op to
take (the per-group `ttnn.slice` that produces `in0` simply names L1), which is exactly why it needs a layer
measurement rather than an op one: there is no isolated op to compare.

**Every shipped policy's warmed prefill.** `POLICIES` has three entries and until round 14 no test or harness ran
two of them. Running them found that `fused-parity` could not prefill at all — the 2D config's L1 model hardcoded
BFP8's bytes per weight element, so it under-predicted the `in1` circular buffers by ~1.9x under bfloat16 weights
and program construction threw. This is the artifact that keeps all three honest, and the `fused-parity` row is
also the cross-check on the parity claim: it should land on the *fused decoder's* own prefill time, which
README §5.2's generated table carries.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/ab_routed_in0.py

Rows are ``ROUTEDIN0 arm=<name> run=<i> layer=<idx> (<kind>) prefill(warmed) wall/iter=<ms> ms`` and
``POLICYPREFILL policy=<name> run=<i> layer=<idx> (<kind>) prefill(warmed) wall/iter=<ms> ms``. Arms alternate
build-by-build and each arm's first build is discarded, the same protocol as `ab_sdpa_decode_grid.py`.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as impl

CTX = 8192
#: Long enough to run many 32-token expert groups, which is where the `in0` placement is paid.
PREFILL_LEN = 2048
LAYERS = {3: "full_attention", 0: "linear_attention"}


def dev(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def build(mesh, cfg, sd, layer_idx, policy):
    decoder = impl.OptimizedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CTX, policy=policy
    )
    blocks = impl.num_blocks_for_context(CTX)
    decoder.allocate_kv_cache(blocks)
    decoder.allocate_state(1)
    page_table = None
    if decoder.is_full_attention:
        page_table = dev(
            mesh,
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    return decoder, page_table


def prefill_ms(mesh, decoder, page_table, iters=3):
    gen = torch.Generator().manual_seed(41)
    x = (torch.randn(1, PREFILL_LEN, decoder.cfg.dim, generator=gen) * 0.5).to(torch.bfloat16)
    xd = dev(mesh, x)
    ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    start = time.time()
    for _ in range(iters):
        ttnn.deallocate(decoder.prefill_forward(xd, page_table=page_table))
    ttnn.synchronize_device(mesh)
    return (time.time() - start) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="builds per arm; the first is discarded")
    args = ap.parse_args()

    cfg = R.load_text_config()
    print("# Whole-layer A/B: routed gate/up `in0` placement, and every shipped policy's warmed prefill.")
    print(f"# prefill_len={PREFILL_LEN} context={CTX}; arms alternate build-by-build, first build per arm discarded.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    shipped = impl.ROUTED_IN0_MEMORY
    try:
        for layer_idx, kind in LAYERS.items():
            sd = R.load_layer_state_dict(layer_idx)
            arms = {"shipped-in0-L1": ttnn.L1_MEMORY_CONFIG, "in0-DRAM": ttnn.DRAM_MEMORY_CONFIG}
            seen = {arm: 0 for arm in arms}
            for run in range(1, args.runs + 2):
                for arm, memory_config in arms.items():
                    impl.ROUTED_IN0_MEMORY = memory_config
                    decoder, page_table = build(mesh, cfg, sd, layer_idx, impl.POLICIES["optimized"])
                    ms = prefill_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    seen[arm] += 1
                    if seen[arm] == 1:
                        continue  # discard: the first build of an arm pays weight upload and compile
                    print(
                        f"ROUTEDIN0 arm={arm} run={run} layer={layer_idx} ({kind}) prefill(warmed) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            impl.ROUTED_IN0_MEMORY = shipped
            # Every shipped policy at the shipped placement. `fused-parity` is the parity cross-check.
            for name in sorted(impl.POLICIES):
                for run in range(1, 3):
                    decoder, page_table = build(mesh, cfg, sd, layer_idx, impl.POLICIES[name])
                    ms = prefill_ms(mesh, decoder, page_table)
                    del decoder, page_table
                    if run == 1:
                        continue
                    print(
                        f"POLICYPREFILL policy={name} run={run} layer={layer_idx} ({kind}) prefill(warmed) "
                        f"wall/iter={ms:.3f} ms",
                        flush=True,
                    )
            del sd
    finally:
        impl.ROUTED_IN0_MEMORY = shipped
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
