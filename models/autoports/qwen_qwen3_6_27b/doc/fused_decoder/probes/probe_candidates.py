# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-free probes for the fusing candidates that had to be measured before deciding.

Each probe answers exactly one question that the work log then quotes:

* ``norm``      - how much does width-sharding the decode RMSNorm buy, and at which core grid?
                  (``work_log.md`` §4, ``logs/probe_sharded_norm.log``)
* ``conv``      - depthwise ``ttnn.conv1d`` vs the 4-tap FIR, fp32 vs bf16, multiply+add vs
                  ``ttnn.addcmul``.  (§5 and §7, ``logs/probe_conv.log``)
* ``sharedlhs`` - is merging ``wqkv`` and ``wgate`` into one shared-LHS matmul faster?
                  (§6.2, ``logs/probe_shared_lhs.log``)
* ``ropebatch`` - is the sharded decode ``rotary_embedding_hf`` correct at every batch?
                  (§6.4, ``logs/probe_rope_batch.log``)

Run:  python -m models.autoports.qwen_qwen3_6_27b.doc.fused_decoder.probes.probe_candidates [name ...]
"""

from __future__ import annotations

import sys
import time

import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.doc.fused_decoder.probes.probe_fused_ops import (
    expand_rope_mats,
    pcc,
    rope_permutation,
    torch_partial_rope,
    tt,
)

HIFI4 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


def _cfg():
    return ttnn.WormholeComputeKernelConfig(**HIFI4)


def _bench(device, fn, n=5):
    out = fn()
    if isinstance(out, ttnn.Tensor):
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(n):
        out = fn()
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    return 1e3 * (time.perf_counter() - start) / n


# ------------------------------------------------------------------ decode RMSNorm


def probe_norm(device):
    """Interleaved vs width-sharded RMSNorm on a decode-shaped activation."""
    hidden, rows = 5120, 32
    x = torch.randn(1, 1, rows, hidden)
    weight = torch.randn(1, 1, 1, hidden) * 0.1 + 1
    xt, wt = tt(x, device), tt(weight, device)
    golden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
    base = _bench(device, lambda: ttnn.rms_norm(xt, epsilon=1e-6, weight=wt, compute_kernel_config=_cfg()), n=50)
    print(f"  interleaved (baseline): {1e3 * base:.1f} us/call, cores 1")
    for grid_x, grid_y in ((5, 2), (8, 4), (8, 5), (10, 4), (8, 8)):
        cores = grid_x * grid_y
        if (hidden // 32) % cores:
            print(f"  grid {grid_x}x{grid_y}: skipped, {hidden // 32} tiles not divisible by {cores}")
            continue
        block_w = (hidden // 32) // cores
        for subblock_w in sorted({block_w, min(block_w, 4), 2, 1}, reverse=True):
            if block_w % subblock_w:
                continue
            grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
            )
            mem = ttnn.create_sharded_memory_config(
                shape=(32, block_w * 32), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
            )
            prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_x, grid_y), subblock_w=subblock_w,
                block_h=1, block_w=block_w, inplace=False,
            )
            try:
                xs = ttnn.to_memory_config(xt, mem)
                call = lambda: ttnn.rms_norm(  # noqa: E731
                    xs, epsilon=1e-6, weight=wt, compute_kernel_config=_cfg(),
                    program_config=prog, memory_config=mem,
                )
                got = ttnn.to_torch(ttnn.to_memory_config(call(), ttnn.DRAM_MEMORY_CONFIG)).float()
                elapsed = _bench(device, call, n=50)
                print(f"  sharded {grid_x}x{grid_y} block_w={block_w} subblock_w={subblock_w}: "
                      f"{1e3 * elapsed:.1f} us/call max|err|={float((got - golden).abs().max()):.4g}")
            except Exception as exc:  # noqa: BLE001
                print(f"  sharded {grid_x}x{grid_y} block_w={block_w} subblock_w={subblock_w}: "
                      f"FAILED {str(exc)[:120]}")


# ------------------------------------------------------------------------ causal conv


def probe_conv(device):
    """The 4-tap depthwise FIR at the real width, three ways, plus ttnn.conv1d."""
    length, channels, taps = 2048, 10240, 4
    x = torch.randn(1, 1, length + taps - 1, channels)
    weight = torch.randn(channels, 1, taps) * 0.1
    golden = torch.nn.functional.silu(
        torch.nn.functional.conv1d(x[0, 0].t().unsqueeze(0), weight, groups=channels)[0]
        .t()
        .reshape(1, 1, length, channels)
    )
    for dtype, name in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
        window = tt(x, device, dtype)
        tap_w = [tt(weight[:, 0, j].reshape(1, 1, 1, -1), device, dtype) for j in range(taps)]

        def fir(_w=window, _t=tap_w):
            acc = None
            for j in range(taps):
                piece = ttnn.slice(_w, [0, 0, j, 0], [1, 1, j + length, channels])
                term = ttnn.multiply(piece, _t[j])
                ttnn.deallocate(piece)
                acc = term if acc is None else ttnn.add(acc, term)
            out = ttnn.silu(acc)
            ttnn.deallocate(acc)
            return out

        def fir_addcmul(_w=window, _t=tap_w):
            piece = ttnn.slice(_w, [0, 0, 0, 0], [1, 1, length, channels])
            acc = ttnn.multiply(piece, _t[0])
            ttnn.deallocate(piece)
            for j in range(1, taps):
                piece = ttnn.slice(_w, [0, 0, j, 0], [1, 1, j + length, channels])
                updated = ttnn.addcmul(acc, piece, _t[j], value=1.0)
                ttnn.deallocate(piece)
                ttnn.deallocate(acc)
                acc = updated
            out = ttnn.silu(acc)
            ttnn.deallocate(acc)
            return out

        for label, fn in (("multiply+add", fir), ("addcmul", fir_addcmul)):
            out = fn()
            err = float((ttnn.to_torch(out).float() - golden).abs().max())
            ttnn.deallocate(out)
            print(f"  FIR {name} {label}: {_bench(device, fn):.2f} ms max|err|={err:.4g}")
        ttnn.deallocate(window)
        for t in tap_w:
            ttnn.deallocate(t)

    try:
        x_rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        w_dev = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.conv1d(
            input_tensor=x_rm, weight_tensor=w_dev, device=device, in_channels=channels,
            out_channels=channels, batch_size=1, input_length=length + taps - 1, kernel_size=taps,
            stride=1, padding=0, dilation=1, groups=channels, dtype=ttnn.bfloat16, compute_config=_cfg(),
        )
        print(f"  ttnn.conv1d depthwise: out {out.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ttnn.conv1d depthwise: FAILED {str(exc)[:200]}")


# --------------------------------------------------------------------- shared-LHS


def probe_sharedlhs(device):
    """Two matmuls sharing an LHS vs one matmul over the concatenated weight."""
    for rows in (32, 2048):
        x = tt(torch.randn(1, 1, rows, 5120), device)
        w_qkv = tt(torch.randn(1, 1, 5120, 8192), device)
        w_gate = tt(torch.randn(1, 1, 5120, 6144), device)
        w_fused = tt(torch.randn(1, 1, 5120, 14336), device)

        def split():
            a = ttnn.linear(x, w_qkv, dtype=ttnn.bfloat16, compute_kernel_config=_cfg())
            b = ttnn.linear(x, w_gate, dtype=ttnn.bfloat16, compute_kernel_config=_cfg(), activation="sigmoid")
            ttnn.deallocate(a)
            ttnn.deallocate(b)

        def fused():
            out = ttnn.linear(x, w_fused, dtype=ttnn.bfloat16, compute_kernel_config=_cfg())
            a = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, 8192])
            b = ttnn.slice(out, [0, 0, 0, 8192], [1, 1, rows, 14336])
            gated = ttnn.sigmoid(b)
            for t in (out, a, b, gated):
                ttnn.deallocate(t)

        print(f"  rows={rows}: split={1e3 * _bench(device, split, 20):.1f} us   "
              f"shared-LHS={1e3 * _bench(device, fused, 20):.1f} us")
        for t in (x, w_qkv, w_gate, w_fused):
            ttnn.deallocate(t)


# ------------------------------------------------------------------- decode RoPE


def probe_ropebatch(device):
    """Sharded decode rotary_embedding_hf, per-user PCC, at every batch the layer supports."""
    perm = rope_permutation()
    for batch in (1, 4, 8, 16, 32):
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        head_cfg = ttnn.create_sharded_memory_config(
            shape=(32, 256), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )
        rot_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, 256), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )
        x = torch.randn(1, batch, 32, 256)
        cos = torch.randn(1, batch, 1, 64).clamp(-1, 1)
        sin = torch.randn(1, batch, 1, 64).clamp(-1, 1)
        cos_full, sin_full = expand_rope_mats(cos, sin)
        golden = torch_partial_rope(x, cos, sin)[..., perm]
        out = ttnn.experimental.rotary_embedding_hf(
            ttnn.to_memory_config(tt(x[..., perm].contiguous(), device), head_cfg),
            ttnn.to_memory_config(tt(cos_full, device), rot_cfg),
            ttnn.to_memory_config(tt(sin_full, device), rot_cfg),
            is_decode_mode=True,
        )
        got = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).float()
        per_user = [pcc(golden[:, u], got[:, u]) for u in range(batch)]
        bad = [u for u, p in enumerate(per_user) if p < 0.99]
        print(f"  batch={batch}: overall={pcc(golden, got):.6f} min_user={min(per_user):.6f} bad={bad}")


PROBES = {"norm": probe_norm, "conv": probe_conv, "sharedlhs": probe_sharedlhs, "ropebatch": probe_ropebatch}


def main():
    names = sys.argv[1:] or list(PROBES)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for name in names:
            print(f"== {name}", flush=True)
            PROBES[name](device)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
