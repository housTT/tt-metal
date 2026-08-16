# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Router-gate spellings at the multichip decode shapes.

The multichip decode profile puts the router chain — ``topk`` (48 us, one core) plus ``softmax``
plus the ``scatter``'s untilize/tilize round trip — at about 17 % of the traced decode window, which
makes it the largest *addressable* block after the two routed sparse matmuls. The multichip stage
inherited the chain unchanged and never priced the fused alternative: the optimized stage rejected
``ttnn.experimental.deepseek.moe.generalized_moe_gate`` as "bfloat16-only" and, in its own words,
**never timed it**. This probe times it.

Every arm produces the same object the layer needs: a dense ``[1, 1, rows, E]`` bfloat16 routing
vector whose non-zero entries are the softmax over the selected top-``k`` logits. Arms are compared
against the shipped chain on that tensor — PCC, max abs diff, and the *selected-set agreement*,
which is the quantity that actually matters for a router (a swapped expert is a different
computation, not a rounded value).

Arms
----
``shipped``            ``topk(fp32) -> softmax -> scatter`` — what the layer runs today.
``topk_bf16``          the same chain with the logits cast to bfloat16 first. Isolates "bfloat16
                       logits" from "the fused kernel", so the fused arm's accuracy delta can be
                       attributed to the dtype rather than to the op.
``topk_unsorted``      ``sorted=False``: the chain never uses the order, only the set and the
                       matched weights.
``fused_gate``         ``generalized_moe_gate`` (one kernel: score + top-k + softmax-over-selected)
                       feeding the same scatter. Preallocated height-sharded bias / input-index /
                       output buffers, built once — the persistent-buffer pattern.
``fused_gate_valid``   the same, but the gate runs on the ``valid`` real rows only rather than on
                       the tile-padded 32. At decode batch 1 that is one core instead of 32.

    python .../logs/probe_gate.py --rows 32 --valid 1
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn

TILE = 32
E = 256
TOPK = 8
FACE = 16
PAD_NEG = -1e9


def timeit(mesh, fn, iters=30, warmup=5, repeats=3, free=True):
    """Min-of-``repeats`` mean-of-``iters`` microseconds as ``us=<min> spread=<max-min>``.

    ``free=False`` for arms whose result *is* a caller-owned persistent buffer — the
    ``generalized_moe_gate`` op writes into and returns the preallocated output tensors, so freeing
    what it returns destroys the buffers the next iteration needs.
    """

    def release(out):
        if not free:
            return
        for t in out if isinstance(out, (tuple, list)) else (out,):
            ttnn.deallocate(t)

    for _ in range(warmup):
        release(fn())
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            release(fn())
        ttnn.synchronize_device(mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return f"{min(samples):.1f} spread={max(samples) - min(samples):.1f}"


def to_host(mesh, tensor) -> torch.Tensor:
    """Device 0's copy of a replicated mesh tensor, as a host float tensor."""
    if mesh.get_num_devices() == 1:
        return ttnn.to_torch(tensor).float()
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh, dim=0))
    return whole[: int(tensor.shape[0])].float()


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.double().flatten() - a.double().mean()
    y = b.double().flatten() - b.double().mean()
    denom = (x.norm() * y.norm()).item()
    return 1.0 if denom == 0 else float((x @ y).item() / denom)


def set_agreement(ref: torch.Tensor, cand: torch.Tensor, rows: int) -> float:
    """Fraction of the first ``rows`` tokens whose non-zero expert set is identical."""
    same = 0
    for r in range(rows):
        a = set((ref[0, 0, r] != 0).nonzero().flatten().tolist())
        b = set((cand[0, 0, r] != 0).nonzero().flatten().tolist())
        same += int(a == b)
    return same / max(1, rows)


def sharded_mem(grid, num_cores, shard=(32, 32)):
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def build_gate_buffers(mesh, grid, rows):
    """The four persistent tensors ``generalized_moe_gate`` needs, built once at setup.

    ``bias`` is all-zero over the 256 real experts (Ornith's router has no score-correction bias; a
    constant shift changes neither the selection nor the normalized output). ``input_indices``
    carries the global expert id per slot. Both are laid out as the op wants them: one 16x16 face per
    token in the top-left of a 32x32 tile, transposed within the face, height-sharded one token per
    core. ``out`` / ``out_idx`` are the preallocated result buffers the op writes into.
    """
    mem = sharded_mem(grid, rows)
    bias_face = torch.nn.functional.pad(
        torch.zeros(1, FACE, FACE, dtype=torch.float32), (0, FACE, 0, FACE, 0, 0), "constant", 0
    ).transpose(1, 2)
    bias = ttnn.from_torch(
        bias_face.repeat(rows, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem
    )
    idx = torch.arange(E, dtype=torch.int32).reshape(1, FACE, FACE).transpose(1, 2)
    in_idx = ttnn.from_torch(
        idx.repeat(rows, 1, 1), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem
    )
    out = ttnn.from_torch(
        torch.zeros(rows, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem
    )
    out_idx = ttnn.from_torch(
        torch.zeros(rows, 32, 32, dtype=torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mem,
    )
    # The op asserts `bias_shape == in_shape`, and the input it is handed is the [rows, 16, 16] face
    # view. The buffers are allocated at the full [rows, 32, 32] tile (that is the shard) and sliced
    # to the face shape once, here, so the forward never pays for it.
    bias = ttnn.slice(bias, [0, 0, 0], [rows, FACE, FACE], memory_config=mem)
    in_idx = ttnn.slice(in_idx, [0, 0, 0], [rows, FACE, FACE], memory_config=mem)
    return bias, in_idx, out, out_idx, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32, help="tile-padded router row count (decode: align_up(batch,32))")
    ap.add_argument("--valid", type=int, default=1, help="real token rows; the rest are exact-zero padding")
    ap.add_argument("--mesh", default="1x4")
    args = ap.parse_args()

    rows, valid = args.rows, args.valid
    shape = (1, 4) if args.mesh == "1x4" else (1, 1)
    if shape != (1, 1):
        from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import (
            DEFAULT_FABRIC_CONFIG,
            fabric_router_config,
        )

        ttnn.set_fabric_config(DEFAULT_FABRIC_CONFIG, router_config=fabric_router_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    gen = torch.Generator().manual_seed(19)
    print(f"# grid {grid.x}x{grid.y} mesh={args.mesh} rows={rows} valid={valid} E={E} k={TOPK}")

    try:
        # Real decode geometry: only `valid` rows carry data; the tile padding is exact zeros, which is
        # what `OptimizedDecoder._block` writes with `_pad_dim`.
        logits_t = torch.randn(1, 1, rows, E, generator=gen)
        logits_t[:, :, valid:, :] = 0.0
        logits = ttnn.from_torch(logits_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        logits_bf16 = ttnn.from_torch(logits_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        # The layer's persistent all-zero scatter base (see `OptimizedMoE._router_zeros_for`).
        zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)

        def scatter_chain(src=logits, sorted_=True):
            values, indices = ttnn.topk(src, k=TOPK, dim=-1, sorted=sorted_)
            weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
            dense = ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))
            for t in (values, indices, weights):
                ttnn.deallocate(t)
            return dense

        def topk_only(src=logits):
            return ttnn.topk(src, k=TOPK, dim=-1, sorted=True)

        ref = to_host(mesh, scatter_chain())
        print(f"GATE arm=shipped us={timeit(mesh, scatter_chain)} pcc=1.000000 setagree=1.000", flush=True)
        print(f"GATEPART part=topk_fp32 us={timeit(mesh, topk_only)}", flush=True)
        print(f"GATEPART part=topk_bf16 us={timeit(mesh, lambda: topk_only(logits_bf16))}", flush=True)

        # The shipped chain, op by op, so the fused arm can be compared against the piece it replaces
        # rather than against the whole block.
        vals, idxs = ttnn.topk(logits, k=TOPK, dim=-1, sorted=True)
        print(
            "GATEPART part=softmax_k us="
            f"{timeit(mesh, lambda: ttnn.softmax(vals, dim=-1, numeric_stable=True, compute_kernel_config=ckc))}",
            flush=True,
        )
        wts = ttnn.typecast(ttnn.softmax(vals, dim=-1, numeric_stable=True, compute_kernel_config=ckc), ttnn.bfloat16)
        print(
            f"GATEPART part=scatter us={timeit(mesh, lambda: ttnn.scatter(zeros, dim=-1, index=idxs, src=wts))}",
            flush=True,
        )
        for t in (vals, idxs, wts):
            ttnn.deallocate(t)

        cand = to_host(mesh, scatter_chain(logits_bf16))
        print(
            f"GATE arm=topk_bf16 us={timeit(mesh, lambda: scatter_chain(logits_bf16))} "
            f"pcc={pcc(ref, cand):.6f} setagree={set_agreement(ref, cand, valid):.3f}",
            flush=True,
        )
        cand = to_host(mesh, scatter_chain(logits, False))
        print(
            f"GATE arm=topk_unsorted us={timeit(mesh, lambda: scatter_chain(logits, False))} "
            f"pcc={pcc(ref, cand):.6f} setagree={set_agreement(ref, cand, valid):.3f}",
            flush=True,
        )

        # ---- how ttnn.topk scales with the searched width ----------------------------------
        # The two-stage decomposition (per-chunk top-k then top-k of the survivors) is only worth a
        # relayout if the op's cost really is linear in the width, so measure the low end of the
        # ladder the optimized stage only measured upward from 256.
        for width in (32, 64, 128, 256):
            probe_t = torch.randn(1, 1, rows, width, generator=gen)
            probe = ttnn.from_torch(probe_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            print(
                f"TOPKW width={width} us={timeit(mesh, lambda p=probe: ttnn.topk(p, k=TOPK, dim=-1, sorted=True))}",
                flush=True,
            )
            ttnn.deallocate(probe)

        # ---- threshold arm: build the DEVICE-LOCAL routing vector without a scatter ---------
        # `MultichipMoE.routing_weights` needs the local 64-wide vector, not the global 256-wide one:
        # the 256-wide `dense` exists only to be narrowed by the `expert_select` one-hot matmul. This
        # arm skips the dense vector entirely. It keeps the global top-8 decision exactly - the
        # threshold `kth` and the softmax denominator are global scalars per token, taken from the
        # same `topk` over all 256 - and applies it to the local logits, which the same one-hot matmul
        # produces exactly (each output sums 255 structural zeros and one logit).
        #
        # The `ramp` is a strictly decreasing perturbation over the expert axis, added before both the
        # topk and the comparison. It makes the >= threshold test select EXACTLY k experts even when
        # logits tie, which matters for two real cases: the tile-padding rows, whose logits are all
        # exactly zero (without it every one of the 256 experts passes `ge`, which is still correct -
        # a padded row's activation is zero - but it would union the whole expert set into the routing
        # sparsity of a prefill group that contains padding); and exact ties between real logits, where
        # `topk` picks one arbitrarily and an unbroken `ge` would select both.
        e_local = E // mesh.get_num_devices()
        eye = torch.eye(E, dtype=torch.float32).reshape(1, 1, E, E)
        mapper = (
            ttnn.shard_tensor_to_mesh_mapper(mesh, dim=-1)
            if mesh.get_num_devices() > 1
            else ttnn.replicate_tensor_to_mesh_mapper(mesh)
        )
        select = ttnn.from_torch(
            eye if mesh.get_num_devices() > 1 else eye[:, :, :, :e_local],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ramp_step = 1e-6
        ramp = ttnn.from_torch(
            (-ramp_step * torch.arange(E, dtype=torch.float32)).reshape(1, 1, 1, E),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )

        def shipped_local():
            """The shipped narrowing, as the baseline this arm must beat and match."""
            dense = scatter_chain()
            local = ttnn.linear(dense, select, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
            ttnn.deallocate(dense)
            return local

        def threshold_local(use_ramp=True):
            src = ttnn.add(logits, ramp) if use_ramp else logits
            values, indices = ttnn.topk(src, k=TOPK, dim=-1, sorted=True)
            top = ttnn.slice(values, [0, 0, 0, 0], [1, 1, rows, 1])  # the max, for the stable exp
            kth = ttnn.slice(values, [0, 0, 0, TOPK - 1], [1, 1, rows, TOPK])  # the selection threshold
            soft = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=ckc)
            inv = ttnn.slice(soft, [0, 0, 0, 0], [1, 1, rows, 1])  # softmax()[0] == 1 / sum(exp(v - max))
            local_logits = ttnn.linear(src, select, dtype=ttnn.float32, compute_kernel_config=ckc)
            keep = ttnn.ge(local_logits, kth)
            expd = ttnn.subtract(local_logits, top, activations=[ttnn.UnaryOpType.EXP])
            scaled = ttnn.multiply(expd, inv)
            local = ttnn.typecast(ttnn.multiply(scaled, keep), ttnn.bfloat16)
            for t in (values, indices, top, kth, soft, inv, local_logits, keep, expd, scaled):
                ttnn.deallocate(t)
            if use_ramp:
                ttnn.deallocate(src)
            return local

        ref_local = to_host(mesh, shipped_local())
        print(f"GATE arm=shipped_local us={timeit(mesh, shipped_local)} pcc=1.000000 setagree=1.000", flush=True)
        for use_ramp in (True, False):
            tag = "threshold_local" + ("" if use_ramp else "_noramp")
            try:
                cand = to_host(mesh, threshold_local(use_ramp))
                sel = int((cand[0, 0, :valid] != 0).sum())
                print(
                    f"GATE arm={tag} us={timeit(mesh, lambda r=use_ramp: threshold_local(r))} "
                    f"pcc={pcc(ref_local[:, :, :valid], cand[:, :, :valid]):.6f} "
                    f"setagree={set_agreement(ref_local, cand, valid):.3f} "
                    f"maxabs_valid={float((ref_local[:, :, :valid] - cand[:, :, :valid]).abs().max()):.3e} "
                    f"selected_per_valid_row={sel / max(1, valid):.2f} "
                    f"padrows_nonzero={int((cand[0, 0, valid:] != 0).sum())}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"GATE arm={tag} FAILED {' | '.join(str(exc).splitlines()[:3])[:400]}", flush=True)

        # ---- fused kernel arms -------------------------------------------------------------
        for tag, gate_rows in (("fused_gate", rows), ("fused_gate_valid", valid)):
            if gate_rows > grid.x * grid.y:
                print(f"GATE arm={tag} SKIPPED rows {gate_rows} exceed {grid.x * grid.y} cores", flush=True)
                continue
            try:
                bias, in_idx, out_buf, out_idx_buf, mem_in = build_gate_buffers(mesh, grid, gate_rows)

                base = zeros if gate_rows == rows else ttnn.slice(zeros, [0, 0, 0, 0], [1, 1, gate_rows, E])
                src0 = logits_bf16 if gate_rows == rows else ttnn.slice(logits_bf16, [0, 0, 0, 0], [1, 1, gate_rows, E])

                def convert_in(gate_rows=gate_rows, src0=src0, mem_in=mem_in):
                    return ttnn.to_memory_config(ttnn.reshape(src0, (gate_rows, FACE, FACE)), memory_config=mem_in)

                faces0 = convert_in()

                def gate_op(faces=faces0, bias=bias, in_idx=in_idx, out_buf=out_buf, out_idx_buf=out_idx_buf):
                    return ttnn.experimental.deepseek.moe.generalized_moe_gate(
                        faces,
                        bias_tensor=bias,
                        input_indices_tensor=in_idx,
                        output_tensor=out_buf,
                        output_indices_tensor=out_idx_buf,
                        eps=1e-20,
                        scaling_factor=1.0,
                        enable_sigmoid=False,
                        topk=TOPK,
                        output_softmax=True,
                    )

                w0, ids0 = gate_op()

                def convert_out(gate_rows=gate_rows, w=w0, ids=ids0, base=base):
                    # [gate_rows, 32, 32] sharded -> the [1, 1, gate_rows, k] pair the scatter wants.
                    # The uint16 index has to become uint32 BEFORE any reshape: reshape_view rejects
                    # uint16 outright, and `ttnn.scatter` wants a uint32 index anyway.
                    wv = ttnn.to_memory_config(w, memory_config=ttnn.L1_MEMORY_CONFIG)
                    iv = ttnn.typecast(ttnn.to_memory_config(ids, memory_config=ttnn.L1_MEMORY_CONFIG), ttnn.uint32)
                    wk = ttnn.reshape(ttnn.slice(wv, [0, 0, 0], [gate_rows, 1, TOPK]), (1, 1, gate_rows, TOPK))
                    ik = ttnn.reshape(ttnn.slice(iv, [0, 0, 0], [gate_rows, 1, TOPK]), (1, 1, gate_rows, TOPK))
                    dense = ttnn.scatter(base, dim=-1, index=ik, src=wk)
                    for t in (wv, iv, wk, ik):
                        ttnn.deallocate(t)
                    return dense

                def fused():
                    faces = convert_in()
                    w, ids = gate_op(faces)
                    dense = convert_out(w=w, ids=ids)
                    ttnn.deallocate(faces)
                    return dense

                print(f"GATEPART part={tag}:convert_in us={timeit(mesh, convert_in)}", flush=True)
                print(f"GATEPART part={tag}:gate_op us={timeit(mesh, gate_op, free=False)}", flush=True)
                print(f"GATEPART part={tag}:convert_out+scatter us={timeit(mesh, convert_out)}", flush=True)
                dense0 = fused()
                cand = to_host(mesh, dense0)
                # Compare on the REAL rows only. The tile padding is a different quantity in the two
                # arms by construction: the shipped chain runs topk on the padding's exact-zero logits
                # and gets an arbitrary 8 experts at 1/8 each, while an arm that gates only the real
                # rows leaves the padding at zero. Neither reaches the layer output - the padded rows'
                # activations are exact zeros - and `_active_expert_mask` already restricts the routing
                # sparsity to the real rows, so a whole-tensor PCC would compare the padding, not the route.
                print(
                    f"GATE arm={tag} us={timeit(mesh, fused)} "
                    f"pcc={pcc(ref[:, :, :valid], cand[:, :, :valid]):.6f} "
                    f"setagree={set_agreement(ref, cand, valid):.3f} "
                    f"maxabs_valid={float((ref[:, :, :valid] - cand[:, :, :valid]).abs().max()):.3e} "
                    f"padrows_nonzero={int((cand[:, :, valid:] != 0).sum())}",
                    flush=True,
                )
                for t in (faces0, dense0):
                    ttnn.deallocate(t)
            except Exception as exc:  # noqa: BLE001
                print(f"GATE arm={tag} FAILED {" | ".join(str(exc).splitlines()[:4])[:600]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
