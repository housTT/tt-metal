# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Formulations of the GatedDeltaNet depthwise causal conv1d, at the real prefill shape.

After the delta-rule core became one op, this 4-tap FIR over ``conv_dim`` = 10240 channels is
the largest remaining ``linear_attention`` prefill cost.  Its two structural problems are:

* every tap slices the concatenated window at row 1/2/3, i.e. **off** a tile boundary, which
  makes ``ttnn.slice`` an ``untilize_with_unpadding`` + ``tilize_with_val_padding`` sandwich;
* the per-channel tap multiply is a *height-broadcast* binary op, which runs far below the
  bandwidth a same-shape binary op reaches.

Variants compared (all produce the same FIR + SiLU, all checked against torch):

``tile``        what the functional layer does: TILE slices throughout
``rm_shift``    untilize once, shift in ROW_MAJOR, tilize each tap (TILE concat)
``rm_concat``   *also* concatenate in ROW_MAJOR, and fold the SiLU into the last tap's add - shipped
``rm_arith``    untilize once, keep the whole FIR in ROW_MAJOR, tilize once
``aligned_win`` one pre-padded window per tap so every tap slice starts on a tile boundary
``scale_shift`` scale on the TILE tensor first, then untilize once per tap and shift-and-add in
                ROW_MAJOR - the only ordering that moves the per-tap tilize off the critical path

Each runs in float32 and in bfloat16; every result is checked against torch *and* against the
first variant's, so "the formulations agree" is a measurement.  A final microbenchmark isolates
broadcast-vs-same-shape multiply bandwidth.

    python .../probes/probe_causal_conv.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

CONV_DIM = 10240
K = 4
SEQ = 2048


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def timed(fn, device, iters=12):
    """Best/median/stdev over ``iters`` repeats, so a small gap can be told from run-to-run spread."""
    out = fn()
    ttnn.deallocate(out)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)
        got = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
    return (min(samples), statistics.median(samples), statistics.stdev(samples)), got


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        torch.manual_seed(0)
        x = torch.randn(1, 1, SEQ, CONV_DIM) * 0.5
        prefix = torch.randn(1, 1, K - 1, CONV_DIM) * 0.5
        taps = [torch.randn(1, 1, 1, CONV_DIM) * 0.3 for _ in range(K)]
        window_t = torch.cat([prefix, x], dim=2)
        ref = sum(window_t[:, :, j : j + SEQ, :] * taps[j] for j in range(K))
        ref = torch.nn.functional.silu(ref)

        def dev(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        for dtype, tag in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            tx, tp = dev(x, dtype), dev(prefix, dtype)
            tt_taps = [dev(t, dtype) for t in taps]

            def tile_variant():
                window = ttnn.concat([tp, tx], dim=-2)
                acc = None
                for j in range(K):
                    tap = ttnn.slice(window, [0, 0, j, 0], [1, 1, j + SEQ, CONV_DIM])
                    term = ttnn.multiply(tap, tt_taps[j])
                    ttnn.deallocate(tap)
                    if acc is None:
                        acc = term
                    else:
                        acc = ttnn.add(acc, term)
                        ttnn.deallocate(term)
                ttnn.deallocate(window)
                out = ttnn.silu(acc)
                ttnn.deallocate(acc)
                return out

            def rm_shift_variant():
                window = ttnn.concat([tp, tx], dim=-2)
                rm = ttnn.to_layout(window, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(window)
                acc = None
                for j in range(K):
                    piece = ttnn.slice(rm, [0, 0, j, 0], [1, 1, j + SEQ, CONV_DIM])
                    tap = ttnn.to_layout(piece, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(piece)
                    term = ttnn.multiply(tap, tt_taps[j])
                    ttnn.deallocate(tap)
                    if acc is None:
                        acc = term
                    else:
                        acc = ttnn.add(acc, term)
                        ttnn.deallocate(term)
                ttnn.deallocate(rm)
                out = ttnn.silu(acc)
                ttnn.deallocate(acc)
                return out

            def rm_concat_variant():
                """What ships: build the window in ROW_MAJOR too, and fold the SiLU into the last add.

                ``ttnn.concat`` on TILE operands untilizes them, concatenates and re-tilizes, and
                the ROW_MAJOR tap loop throws that tilize away again - so concatenating in
                ROW_MAJOR removes a tilize/untilize round trip over the whole window.
                """
                pieces = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in (tp, tx)]
                rm = ttnn.concat(pieces, dim=-2)
                for piece in pieces:
                    ttnn.deallocate(piece)
                acc = None
                for j in range(K):
                    piece = ttnn.slice(rm, [0, 0, j, 0], [1, 1, j + SEQ, CONV_DIM])
                    tap = ttnn.to_layout(piece, ttnn.TILE_LAYOUT)
                    ttnn.deallocate(piece)
                    term = ttnn.multiply(tap, tt_taps[j])
                    ttnn.deallocate(tap)
                    if acc is None:
                        acc = term
                        continue
                    merged = ttnn.add(acc, term, activations=[ttnn.UnaryOpType.SILU] if j == K - 1 else [])
                    ttnn.deallocate(term)
                    ttnn.deallocate(acc)
                    acc = merged
                ttnn.deallocate(rm)
                return acc

            def rm_arith_variant():
                """Untilize once, keep the whole FIR in ROW_MAJOR, tilize once at the end."""
                window = ttnn.concat([tp, tx], dim=-2)
                rm = ttnn.to_layout(window, ttnn.ROW_MAJOR_LAYOUT)
                ttnn.deallocate(window)
                rm_taps = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in tt_taps]
                acc = None
                for j in range(K):
                    tap = ttnn.slice(rm, [0, 0, j, 0], [1, 1, j + SEQ, CONV_DIM])
                    term = ttnn.multiply(tap, rm_taps[j])
                    ttnn.deallocate(tap)
                    if acc is None:
                        acc = term
                    else:
                        acc = ttnn.add(acc, term)
                        ttnn.deallocate(term)
                ttnn.deallocate(rm)
                for t in rm_taps:
                    ttnn.deallocate(t)
                out = ttnn.to_layout(ttnn.silu(acc), ttnn.TILE_LAYOUT)
                ttnn.deallocate(acc)
                return out

            def aligned_windows_variant():
                """One pre-padded window per tap so every tap slice starts on a tile boundary."""
                acc = None
                for j in range(K):
                    lead = (ttnn.TILE_SIZE - j) % ttnn.TILE_SIZE
                    parts = (
                        [tp, tx]
                        if lead == 0
                        else [
                            ttnn.zeros(
                                (1, 1, lead, CONV_DIM), dtype=tt_taps[j].dtype, layout=ttnn.TILE_LAYOUT, device=device
                            ),
                            tp,
                            tx,
                        ]
                    )
                    window = ttnn.concat(parts, dim=-2)
                    if lead:
                        ttnn.deallocate(parts[0])
                    start = lead + j
                    tap = ttnn.slice(window, [0, 0, start, 0], [1, 1, start + SEQ, CONV_DIM])
                    ttnn.deallocate(window)
                    term = ttnn.multiply(tap, tt_taps[j])
                    ttnn.deallocate(tap)
                    if acc is None:
                        acc = term
                    else:
                        acc = ttnn.add(acc, term)
                        ttnn.deallocate(term)
                out = ttnn.silu(acc)
                ttnn.deallocate(acc)
                return out

            def scale_then_shift_variant():
                """Scale first on the TILE tensor, *then* untilize and shift-and-add in ROW_MAJOR.

                Every other formulation shifts first and multiplies per tap, so each tap pays a
                ``slice`` + ``tilize`` of a full-size window before its multiply - in the committed
                prefill report those pairs are the largest ``layout`` group of the pass.  Scaling
                first replaces them with ``K`` height-broadcast multiplies on the TILE tensor and
                one untilize per tap, at the cost of ``K`` full-size scaled copies existing at once
                and the accumulation happening in ROW_MAJOR.
                """
                windows = []
                for tap in tt_taps:
                    body = ttnn.multiply(tx, tap)
                    prefix = ttnn.multiply(tp, tap)
                    pieces = [ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in (prefix, body)]
                    ttnn.deallocate(body)
                    ttnn.deallocate(prefix)
                    windows.append(ttnn.concat(pieces, dim=-2))
                    for piece in pieces:
                        ttnn.deallocate(piece)
                acc = None
                for j in range(K):
                    term = ttnn.slice(windows[j], [0, 0, j, 0], [1, 1, j + SEQ, CONV_DIM])
                    ttnn.deallocate(windows[j])
                    if acc is None:
                        acc = term
                        continue
                    merged = ttnn.add(acc, term)
                    ttnn.deallocate(term)
                    ttnn.deallocate(acc)
                    acc = merged
                out = ttnn.to_layout(ttnn.silu(acc), ttnn.TILE_LAYOUT)
                ttnn.deallocate(acc)
                return out

            first = None
            for name, fn in (
                ("tile", tile_variant),
                ("rm_shift", rm_shift_variant),
                ("rm_concat", rm_concat_variant),
                ("rm_arith", rm_arith_variant),
                ("aligned_win", aligned_windows_variant),
                ("scale_shift", scale_then_shift_variant),
            ):
                (best, median, stdev), got = timed(fn, device)
                if first is None:
                    first = got
                    agreement = "1.000000 (self)"
                else:
                    agreement = f"{pcc(first, got):.6f}"
                print(
                    f"conv {name:11s} {tag} best_ms={best:8.3f} median_ms={median:8.3f} "
                    f"stdev_ms={stdev:6.3f} pcc_vs_torch={pcc(ref, got):.6f} "
                    f"pcc_vs_first={agreement} max_abs_diff={0.0 if first is got else float((first - got).abs().max()):.3e}",
                    flush=True,
                )
            for t in (tx, tp, *tt_taps):
                ttnn.deallocate(t)

        # --------------------------------------------------- broadcast microbench
        for dtype, tag in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            a = dev(torch.randn(1, 1, SEQ + K - 1, CONV_DIM), dtype)
            small = dev(torch.randn(1, 1, 1, CONV_DIM), dtype)
            full = dev(torch.randn(1, 1, SEQ + K - 1, CONV_DIM), dtype)
            for name, other in (("bcast", small), ("same", full)):
                (best, median, stdev), _ = timed(lambda: ttnn.multiply(a, other), device)
                nbytes = (SEQ + K - 1) * CONV_DIM * (4 if dtype == ttnn.float32 else 2)
                moved = nbytes * (2 if name == "bcast" else 3)
                print(
                    f"multiply {name:5s} {tag} best_ms={best:8.3f} median_ms={median:8.3f} "
                    f"stdev_ms={stdev:6.3f} eff_GBps={moved / (median * 1e-3) / 1e9:7.1f}",
                    flush=True,
                )
            for t in (a, small, full):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
