# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can the recurrent-state update be one ``ttnn.addcmul`` instead of a multiply and an add?

The decode recurrence does ``state_new = state * exp(g) + k^T delta`` as two full passes over the
carried state - at the advertised ``max_batch`` that state is 100 MB of float32, and those two
passes are the largest non-matmul cost of the step.  ``addcmul(a, b, c) = a + b * c`` is exactly
that arithmetic in one op.

§3.14 rejected ``addcmul`` for the conv taps as "a composite in this checkout", and a stage review
showed that reading is wrong: ``ttnn::addcmul`` dispatches a single LLK ternary device op and only
falls back to a decomposition for invalid or subtile-broadcast block-float inputs, neither of which
applies here.  So this probe measures three things at both decode regimes:

``shipped``     ``multiply(state, g, b_activations=[EXP])`` then ``add(..., update)``;
``addcmul``     ``addcmul(update, state, exp_g)`` - one pass, with ``exp(g)`` still its own tiny op;
``in_place``    the same with ``output_tensor=`` aliasing the state buffer, which is what the
                traced decode needs (the state must land at the persistent buffer's address).

The reorder is legal because ``g`` is one scalar per head, so ``k @ (state * g) == (k @ state) * g``
and the state read can consume the *undecayed* state.

It also measures the *conv tap* case §3.14 rejected: the FIR's non-final taps are a plain
``multiply`` + plain ``add`` with no activation riding on them, so the "it would lose the SiLU"
blocker applies only to the last tap.

    python .../probes/probe_addcmul_state.py
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn

NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 48, 128, 128
BATCHES = (1, 32)


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def median_us(fn, device, iters=15):
    out = fn()
    if out is not None:
        ttnn.deallocate(out)
    samples = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e6)
        if out is not None:
            ttnn.deallocate(out)
    return statistics.median(samples), statistics.stdev(samples)


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        torch.manual_seed(0)
        for batch in BATCHES:
            heads = batch * NUM_V_HEADS
            state_host = torch.randn(1, heads, HEAD_K_DIM, HEAD_V_DIM) * 0.1
            g_host = -torch.nn.functional.softplus(torch.randn(1, heads, 1, 1)) * 0.3
            update_host = torch.randn(1, heads, HEAD_K_DIM, HEAD_V_DIM) * 0.05

            def dev(tensor):
                return ttnn.from_torch(tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

            # Everything is uploaded once, outside the timed region: at batch 32 the state is
            # 100 MB and a per-iteration ``from_torch`` of it is an order of magnitude more than
            # the ops under test.  ``shipped`` and ``addcmul`` do not mutate the state, so they can
            # share it; the in-place variant gets its own buffer, refilled between timings.
            g = dev(g_host)
            update = dev(update_host)
            state = dev(state_host)
            scratch = dev(state_host)
            reference = (state_host * g_host.exp() + update_host).float()

            def shipped():
                decayed = ttnn.multiply(state, g, input_tensor_b_activations=[ttnn.UnaryOpType.EXP])
                out = ttnn.add(decayed, update)
                ttnn.deallocate(decayed)
                return out

            def with_addcmul():
                decay = ttnn.exp(g)
                out = ttnn.addcmul(update, state, decay)
                ttnn.deallocate(decay)
                return out

            def in_place():
                """``output_tensor=`` aliasing the state, which the traced decode needs."""
                decay = ttnn.exp(g)
                ttnn.addcmul(update, scratch, decay, output_tensor=scratch)
                ttnn.deallocate(decay)
                return None

            got_shipped = ttnn.to_torch(shipped()).float()
            try:
                got_addcmul = ttnn.to_torch(with_addcmul()).float()
            except Exception as error:  # noqa: BLE001 - the blocker text is the result
                print(
                    f"addcmul batch={batch:2d} rejected: {type(error).__name__}: "
                    f"{str(error).splitlines()[0][:150]}",
                    flush=True,
                )
                for tensor in (g, update, state, scratch):
                    ttnn.deallocate(tensor)
                continue

            shipped_median, shipped_stdev = median_us(shipped, device)
            addcmul_median, addcmul_stdev = median_us(with_addcmul, device)
            try:
                in_place()
                got_place = ttnn.to_torch(scratch).float()
                agreement = pcc(got_place, reference)
                place_median, place_stdev = median_us(in_place, device)
                place = f"in_place_us={place_median:9.1f} ({place_stdev:6.1f}) pcc_in_place={agreement:.6f}"
            except Exception as error:  # noqa: BLE001
                place = f"in_place=REJECTED:{type(error).__name__}:{str(error).splitlines()[0][:70]}"
            print(
                f"addcmul batch={batch:2d} shipped_us={shipped_median:9.1f} ({shipped_stdev:6.1f}) "
                f"addcmul_us={addcmul_median:9.1f} ({addcmul_stdev:6.1f}) {place} "
                f"pcc_shipped_vs_torch={pcc(got_shipped, reference):.6f} "
                f"pcc_addcmul_vs_torch={pcc(got_addcmul, reference):.6f} "
                f"max_abs_diff={float((got_shipped - got_addcmul).abs().max()):.3e}",
                flush=True,
            )
            for tensor in (g, update, state, scratch):
                if tensor.is_allocated():
                    ttnn.deallocate(tensor)
            for tensor in (g, update):
                ttnn.deallocate(tensor)
        # ------------------------------------------------------------------ the conv-tap case
        # One non-final FIR tap at the prefill and decode widths: ``acc + state * w`` as two ops
        # or one.  These carry no activation, so §3.14's blocker does not apply to them.
        conv_dim = 10240
        # The prefill FIR is bfloat16 (§3.7) and the decode FIR is float32 (§3.25), so each row is
        # measured at the dtype that path actually runs - a stage review found this probe measuring
        # bfloat16 for both.
        for rows, label, tap_dtype in ((2048, "prefill", ttnn.bfloat16), (32, "decode", ttnn.float32)):
            acc_host = torch.randn(1, 1, rows, conv_dim) * 0.3
            state_host = torch.randn(1, 1, rows, conv_dim) * 0.3
            tap_host = torch.randn(1, 1, 1, conv_dim) * 0.3

            def dev_tap(tensor):
                return ttnn.from_torch(tensor, dtype=tap_dtype, layout=ttnn.TILE_LAYOUT, device=device)

            acc = dev_tap(acc_host)
            state16 = dev_tap(state_host)
            tap = dev_tap(tap_host)

            def two_ops():
                term = ttnn.multiply(state16, tap)
                out = ttnn.add(acc, term)
                ttnn.deallocate(term)
                return out

            def one_op():
                return ttnn.addcmul(acc, state16, tap)

            try:
                got_two = ttnn.to_torch(two_ops()).float()
                got_one = ttnn.to_torch(one_op()).float()
            except Exception as error:  # noqa: BLE001 - the blocker text is the result
                print(
                    f"conv_tap {label:8s} rows={rows:5d} rejected: {type(error).__name__}: "
                    f"{str(error).splitlines()[0][:140]}",
                    flush=True,
                )
                for tensor in (acc, state16, tap):
                    ttnn.deallocate(tensor)
                continue
            two_median, two_stdev = median_us(two_ops, device)
            one_median, one_stdev = median_us(one_op, device)
            print(
                f"conv_tap {label:8s} rows={rows:5d} dtype={'fp32' if tap_dtype == ttnn.float32 else 'bf16'} "
                f"two_ops_us={two_median:9.1f} ({two_stdev:6.1f}) "
                f"addcmul_us={one_median:9.1f} ({one_stdev:6.1f}) "
                f"pcc_between={pcc(got_two, got_one):.6f} "
                f"max_abs_diff={float((got_two - got_one).abs().max()):.3e}",
                flush=True,
            )
            for tensor in (acc, state16, tap):
                ttnn.deallocate(tensor)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
