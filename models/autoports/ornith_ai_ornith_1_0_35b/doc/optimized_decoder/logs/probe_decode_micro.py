# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Micro-probes for the non-matmul decode costs the perf report ranks next after the matmuls.

Each section times the shipped spelling against every legal alternative at the layer's own decode
shapes, and checks the alternatives agree numerically where they are supposed to be identical.

* ``norm``   — ``ttnn.rms_norm`` on the ``[1, 1, 32, 2048]`` decode residual: the interleaved form
  the fused stage used (which lands on one core) against width-sharded L1 input/output with an
  explicit ``LayerNormShardedMultiCoreProgramConfig`` over several core counts.
* ``topk``   — ``ttnn.topk(k=8)`` over the 256-wide router logits, against the same call on logits
  padded with ``-inf`` up to the width at which the op's multi-core path becomes legal.
* ``gate``   — the whole router chain: ``topk -> softmax -> scatter`` against the threshold rewrite
  ``topk -> ge(kth) -> where -> softmax(256)``, which produces the same dense vector without the
  scatter's untilize/tilize round trip.
* ``sdpa``   — ``paged_scaled_dot_product_attention_decode`` at the real cache geometry (BFP8 paged
  cache, 2 KV heads, 64-token blocks, 8192-token context, batch 1): the config the fused stage
  shipped, against the op default and against other legal grid / chunk pairs (OPT-002).
* ``state`` — the three float32 recurrent-state matmuls of the ``linear_attention`` decode step
  (``[B, 32, 1, 128] x [B, 32, 128, 128]`` and the ``transpose_a`` outer product), which
  ``tt-perf-report`` flags ``SLOW`` with three open advice items each: core grid, ``in0_block_w``,
  in0 placement, math fidelity and ``fp32_dest_acc_en``.
* ``split``  — packed gate/up (one ``N=1024`` sparse matmul plus two slices) against the separate
  ``N=512`` pair, under the selected BFP4/LoFi policy and the tuned 8-core geometry (OPT-010).

    python .../logs/probe_decode_micro.py --section all
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn

TILE = 32
DIM = 2048
E = 256
TOPK = 8
I = 512


def timeit(mesh, fn, iters=30, warmup=5, repeats=3):
    """Min-of-``repeats`` mean-of-``iters`` microseconds, formatted with the spread across repeats.

    Returns a string, not a number, because every caller prints it: the rows read ``us=<min>
    spread=<max-min>``. Review round 6 pointed out that the sparse, dense and prefill probes had gained a
    measured spread in round 5 while these sections had not, which left every "inside the run-to-run
    spread" argument about a norm, a topk, a gate or an SDPA config unfalsifiable — and one of them
    (``NORM``) turned out to support a documented conclusion the artifact contradicts.
    """
    for _ in range(warmup):
        out = fn()
        for t in out if isinstance(out, (tuple, list)) else (out,):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            out = fn()
            for t in out if isinstance(out, (tuple, list)) else (out,):
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return f"{min(samples):.1f} spread={max(samples) - min(samples):.1f}"


def pcc(a, b):
    a = a.double().flatten() - a.double().flatten().mean()
    b = b.double().flatten() - b.double().flatten().mean()
    d = (a.norm() * b.norm()).item()
    return 1.0 if d == 0 else float((a @ b).item() / d)


def section_norm(mesh, grid, gen):
    x_t = torch.randn(1, 1, TILE, DIM, generator=gen)
    w_t = torch.randn(1, 1, 1, DIM, generator=gen) * 0.1 + 1.0
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    w = ttnn.from_torch(w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    base = ttnn.to_torch(ttnn.rms_norm(x, weight=w, epsilon=1e-6)).float()
    print(f"NORM spelling=interleaved-default us={timeit(mesh, lambda: ttnn.rms_norm(x, weight=w, epsilon=1e-6))}")

    for cores in (4, 8, 16, 32, 64):
        n_tiles = DIM // TILE
        if n_tiles % cores:
            continue
        cols = max((c for c in range(1, grid.x + 1) if cores % c == 0 and cores // c <= grid.y), default=0)
        if not cols:
            continue
        rows = cores // cols
        shard = ttnn.create_sharded_memory_config(
            shape=(TILE, DIM // cores),
            core_grid=ttnn.CoreGrid(x=cols, y=rows),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(cols, rows),
            subblock_w=max(i for i in range(1, 9) if (n_tiles // cores) % i == 0),
            block_h=1,
            block_w=n_tiles // cores,
            inplace=False,
        )
        tag = f"NORM spelling=width-sharded cores={cores}({cols}x{rows}) block_w={n_tiles // cores}"
        try:
            x_sh = ttnn.to_memory_config(x, shard)

            def run(cfg=cfg, x_sh=x_sh, shard=shard):
                return ttnn.rms_norm(x_sh, weight=w, epsilon=1e-6, program_config=cfg, memory_config=shard)

            got = ttnn.to_torch(run()).float()
            print(f"{tag} us={timeit(mesh, run)} pcc={pcc(base, got):.6f}", flush=True)
            ttnn.deallocate(x_sh)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag} FAILED {str(exc).splitlines()[0][:110]}", flush=True)


def section_topk(mesh, gen):
    logits_t = torch.randn(1, 1, TILE, E, generator=gen)
    logits = ttnn.from_torch(logits_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    print(f"TOPK width={E} us={timeit(mesh, lambda: ttnn.topk(logits, k=TOPK, dim=-1, sorted=True))}", flush=True)
    for width in (512, 1024, 2048, 4096, 8192, 16384):
        padded_t = torch.full((1, 1, TILE, width), float("-inf"))
        padded_t[..., :E] = logits_t
        padded = ttnn.from_torch(padded_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        tag = f"TOPK width={width}"
        try:
            values, indices = ttnn.topk(padded, k=TOPK, dim=-1, sorted=True)
            ok = bool((ttnn.to_torch(indices).long() < E).all())
            ttnn.deallocate(values)
            ttnn.deallocate(indices)
            print(
                f"{tag} us={timeit(mesh, lambda p=padded: ttnn.topk(p, k=TOPK, dim=-1, sorted=True))} "
                f"indices_in_range={ok}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{tag} FAILED {str(exc).splitlines()[0][:110]}", flush=True)
        ttnn.deallocate(padded)


def section_gate(mesh, gen):
    logits_t = torch.randn(1, 1, TILE, E, generator=gen)
    logits = ttnn.from_torch(logits_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def scatter_chain():
        values, indices = ttnn.topk(logits, k=TOPK, dim=-1, sorted=True)
        weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
        zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))
        for t in (values, indices, weights):
            ttnn.deallocate(t)
        return dense

    def threshold_chain():
        values, indices = ttnn.topk(logits, k=TOPK, dim=-1, sorted=True)
        kth = ttnn.slice(values, [0, 0, 0, TOPK - 1], [1, 1, TILE, TOPK])
        keep = ttnn.ge(logits, kth)
        masked = ttnn.where(keep, logits, float("-inf"))
        dense = ttnn.typecast(
            ttnn.softmax(masked, dim=-1, numeric_stable=True, compute_kernel_config=ckc), ttnn.bfloat16
        )
        for t in (values, indices, kth, keep, masked):
            ttnn.deallocate(t)
        return dense

    a = ttnn.to_torch(scatter_chain()).float()
    b = ttnn.to_torch(threshold_chain()).float()
    print(f"GATE spelling=topk-softmax-scatter us={timeit(mesh, scatter_chain)}", flush=True)
    print(
        f"GATE spelling=topk-ge-where-softmax us={timeit(mesh, threshold_chain)} "
        f"pcc={pcc(a, b):.6f} max_abs_diff={float((a - b).abs().max()):.3e}",
        flush=True,
    )


def section_sdpa(mesh, grid, gen):
    """Paged flash-decode at the layer's real cache geometry — OPT-002's explicit-config sweep."""
    n_heads, n_kv, head_dim, block, ctx = 16, 2, 256, 64, 8192
    blocks = ctx // block
    cache_t = torch.randn(blocks, n_kv, block, head_dim, generator=gen) * 0.1
    k_cache = ttnn.from_torch(cache_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    v_cache = ttnn.from_torch(cache_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    q = ttnn.from_torch(
        torch.randn(1, 1, n_heads, head_dim, generator=gen) * 0.1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = ttnn.from_torch(
        torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
    )
    cur_pos = ttnn.from_torch(
        torch.tensor([ctx - 8], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
    )
    # The shipped decode call passes **NO** compute_kernel_config, deliberately: the prefill SDPA's
    # HiFi2 + fp32-dest-accumulate config collapses decode PCC on a BFP8 paged cache
    # (`ab_sdpa_decode_contract.txt`, candidate B). Review round 8 found this probe passing exactly that
    # config to every arm INCLUDING the `default(None)` reference, so every conclusion drawn from these rows
    # — "the op default is an order of magnitude slower", the grid ranking, the chunk ranking — was measured
    # under a configuration the layer does not build. The sweep runs at the shipped contract now, and the
    # rejected config is kept as one extra labelled arm so its cost stays visible rather than implicit.
    prefill_ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def make(cfg, ckc=None):
        def call():
            kwargs = {} if ckc is None else {"compute_kernel_config": ckc}
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                cur_pos_tensor=cur_pos,
                page_table_tensor=page_table,
                is_causal=True,
                scale=head_dim**-0.5,
                program_config=cfg,
                **kwargs,
            )

        return call

    base = ttnn.to_torch(make(None)()).float()
    candidates = [("default(None)", None)]
    for gx, gy in ((8, 8), (11, 10), (8, 4), (4, 8)):
        if gx > grid.x or gy > grid.y:
            continue
        # (0, 64) is the separable variant review round 8 asked for: it takes the auto q-chunk while KEEPING
        # k_chunk pinned to the page block, which is the invariant candidate A proves must hold. It is the
        # only arm that separates "the q chunk is what wins" from "the k chunk is what wins".
        for qc, kc in ((32, 64), (32, 128), (32, 32), (0, 0), (0, 64)):
            candidates.append(
                (
                    f"grid={gx}x{gy} q_chunk={qc} k_chunk={kc}",
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        q_chunk_size=qc,
                        k_chunk_size=kc,
                        exp_approx_mode=False,
                    ),
                )
            )
    for name, cfg in candidates:
        try:
            got = ttnn.to_torch(make(cfg)()).float()
            print(f"SDPA cfg={name} us={timeit(mesh, make(cfg))} pcc_vs_default={pcc(base, got):.6f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - illegal configs are data
            print(f"SDPA cfg={name} FAILED {str(exc).splitlines()[0][:110]}", flush=True)
    # One extra arm: the shipped program config under the REJECTED prefill compute-kernel config, so the
    # cost of the thing `ab_sdpa_decode_contract.txt` candidate B rejects on correctness is also on record as
    # a time. Its `pcc_vs_default` is against the no-ckc reference, so a low value here is the isolated
    # op disagreeing with the shipped contract - which is the point.
    shipped = next((cfg for nm, cfg in candidates if nm.startswith("grid=8x8 q_chunk=32 k_chunk=64")), None)
    if shipped is not None:
        got = ttnn.to_torch(make(shipped, prefill_ckc)()).float()
        print(
            f"SDPA cfg=grid=8x8 q_chunk=32 k_chunk=64 ckc=prefill-HiFi2-fp32acc "
            f"us={timeit(mesh, make(shipped, prefill_ckc))} pcc_vs_default={pcc(base, got):.6f}",
            flush=True,
        )


def section_state(mesh, grid, gen):
    """The three float32 recurrent-state matmuls of a `linear_attention` decode step.

    ``q @ h`` and ``k @ h`` are ``[B, HV, 1, DK] x [B, HV, DK, DV]`` batched matmuls (B*HV per-head
    matmuls of 1x128x128); the delta outer product is the same shape with ``transpose_a=True``. The
    state is float32 and is the model's exact carry between steps, so dtype is not a knob — but the
    core grid, math fidelity and ``fp32_dest_acc_en`` are, and the shipped config names only the grid.
    """
    hv, dk, dv = 32, 128, 128
    q = ttnn.from_torch(
        torch.randn(1, hv, 1, dk, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    state = ttnn.from_torch(
        torch.randn(1, hv, dk, dv, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    delta = ttnn.from_torch(
        torch.randn(1, hv, 1, dv, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    base = None
    # Program-config arm: the batched non-mcast family (MatmulMultiCoreReuseProgramConfig) DOES take
    # an in0_block_w, which is what `tt-perf-report`'s "in0_block_w=1 is small" advice on these rows
    # asks for. Kt is 4 tiles, so 2 and 4 are the legal values above 1.
    ckc_ship = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    for in0_block_w in (1, 2, 4):
        for gx, gy in ((8, 4), (grid.x, grid.y)):
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=min(4, dv // TILE),
                per_core_M=1,
                per_core_N=dv // TILE,
            )
            for label, a, b, ta in (("read", q, state, False), ("outer", q, delta, True)):
                tag = f"STATE role={label} progcfg in0_block_w={in0_block_w} grid={gx}x{gy}"
                try:

                    def run(a=a, b=b, ta=ta, cfg=cfg):
                        return ttnn.matmul(
                            a,
                            b,
                            transpose_a=ta,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            compute_kernel_config=ckc_ship,
                            program_config=cfg,
                        )

                    ttnn.deallocate(run())
                    print(f"{tag} us={timeit(mesh, run)}", flush=True)
                except Exception as exc:  # noqa: BLE001 - illegal configs are data
                    print(f"{tag} FAILED {str(exc).splitlines()[0][:100]}", flush=True)

    for fid_name, fid, fp32 in (
        ("HiFi4/fp32acc", ttnn.MathFidelity.HiFi4, True),
        ("HiFi4", ttnn.MathFidelity.HiFi4, False),
        ("HiFi2/fp32acc", ttnn.MathFidelity.HiFi2, True),
        ("HiFi2", ttnn.MathFidelity.HiFi2, False),
        ("LoFi", ttnn.MathFidelity.LoFi, False),
    ):
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        for cores in ((grid.x, grid.y), (8, 4), (4, 8), (8, 8), (1, 1)):
            cg = ttnn.CoreGrid(x=cores[0], y=cores[1])
            for label, a, b, ta in (("read", q, state, False), ("outer", q, delta, True)):
                tag = f"STATE role={label} fidelity={fid_name} grid={cores[0]}x{cores[1]}"
                try:

                    def run(a=a, b=b, ta=ta, ckc=ckc, cg=cg):
                        return ttnn.matmul(
                            a,
                            b,
                            transpose_a=ta,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=ckc,
                            core_grid=cg,
                        )

                    got = ttnn.to_torch(run()).float()
                    if label == "read" and fid_name == "HiFi4/fp32acc" and cores == (grid.x, grid.y):
                        base = got
                    extra = f" pcc_vs_shipped={pcc(base, got):.6f}" if (base is not None and label == "read") else ""
                    print(f"{tag} us={timeit(mesh, run)}{extra}", flush=True)
                except Exception as exc:  # noqa: BLE001 - illegal grids are data
                    print(f"{tag} FAILED {str(exc).splitlines()[0][:100]}", flush=True)


def section_split(mesh, grid, gen):
    """Packed gate/up (N=2I, then two slices) vs the separate N=I pair — OPT-010."""
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import _sparse_matmul_config

    sparse_t = torch.zeros(1, 1, 1, E)
    sparse_t[0, 0, 0, torch.randperm(E, generator=gen)[:TOPK]] = 1.0
    sparsity = ttnn.from_torch(sparse_t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    x = ttnn.from_torch(
        torch.randn(1, 1, TILE, DIM, generator=gen) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    w_t = torch.randn(1, E, DIM, 2 * I, generator=gen) * 0.02
    packed_w = ttnn.from_torch(w_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    gate_w = ttnn.from_torch(w_t[..., :I].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    up_w = ttnn.from_torch(w_t[..., I:].contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False
    )
    tile = ttnn.Tile([TILE, TILE])
    silu = [ttnn.UnaryOpType.SILU]
    packed_cfg = _sparse_matmul_config(TILE, 2 * I, DIM, cores=8, in0_block_w=64)
    split_cfg = _sparse_matmul_config(TILE, I, DIM, cores=8, in0_block_w=64)

    def call(w, cfg, n):
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            program_config=cfg,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat8_b,
        )

    def packed():
        out = call(packed_w, packed_cfg, 2 * I)
        gate = ttnn.slice(out, [0, 0, 0, 0, 0, 0], [1, 1, 1, E, TILE, I])
        up = ttnn.slice(out, [0, 0, 0, 0, 0, I], [1, 1, 1, E, TILE, 2 * I])
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=silu, memory_config=ttnn.L1_MEMORY_CONFIG)
        for t in (out, gate, up):
            ttnn.deallocate(t)
        return hidden

    def split():
        gate = call(gate_w, split_cfg, I)
        up = call(up_w, split_cfg, I)
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=silu, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return hidden

    a = ttnn.to_torch(packed()).float()
    b = ttnn.to_torch(split()).float()
    print(f"SPLIT spelling=packed-gate-up us={timeit(mesh, packed, iters=20)}", flush=True)
    print(
        f"SPLIT spelling=separate-gate-up us={timeit(mesh, split, iters=20)} pcc_vs_packed={pcc(a, b):.6f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["all", "norm", "topk", "gate", "sdpa", "state", "split"])
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    gen = torch.Generator().manual_seed(19)
    print(f"# grid {grid.x}x{grid.y}")
    try:
        if args.section in ("all", "norm"):
            section_norm(mesh, grid, gen)
        if args.section in ("all", "topk"):
            section_topk(mesh, gen)
        if args.section in ("all", "gate"):
            section_gate(mesh, gen)
        if args.section in ("all", "sdpa"):
            section_sdpa(mesh, grid, gen)
        if args.section in ("all", "state"):
            section_state(mesh, grid, gen)
        if args.section in ("all", "split"):
            section_split(mesh, grid, gen)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
