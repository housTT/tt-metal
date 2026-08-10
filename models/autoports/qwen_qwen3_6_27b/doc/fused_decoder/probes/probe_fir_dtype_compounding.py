# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why the bfloat16 decode FIR fails even though it is faster: the error compounds.

§3.25 measured the bfloat16 decode causal-conv FIR as clearly faster from batch 4 up, built it,
and reverted it because ``test_traced_decode_batched`` fell to PCC 0.98 against HF.  That failure
is the decisive evidence for a *rejection*, and a stage review pointed out it lived only in prose:
``probe_decode_conv_dtype.py`` shows the FIR itself at PCC 0.99999 per step, which looks harmless.

The gap between "0.99999 per step" and "0.98 after a few steps" is the point.  The decode FIR's
output does not leave the layer - it feeds the recurrent state, and the state carries forward.  So
this probe runs the recurrence the way decode does, N steps deep, with each FIR dtype, against a
float64 torch reference, and reports the PCC of the *state* at every step.  The prefill FIR is
bfloat16 for the same arithmetic and does not have this problem because its output feeds
``chunk_gated_delta_rule``, which casts to bfloat16 anyway and accumulates the state in kernel.

Model-free: synthetic tensors at the real shapes.

    python .../probes/probe_fir_dtype_compounding.py
"""

from __future__ import annotations

import torch

import ttnn

CONV_DIM = 10240
NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM = 48, 128, 128
K_SIZE = 4
BATCH = 4
STEPS = 8


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        heads = BATCH * NUM_V_HEADS
        taps_host = [torch.randn(1, 1, 1, CONV_DIM) * 0.3 for _ in range(K_SIZE)]
        tokens = [torch.randn(1, 1, BATCH, CONV_DIM) for _ in range(STEPS)]
        rows_host = [torch.zeros(1, 1, BATCH, CONV_DIM) for _ in range(K_SIZE)]

        def dev(tensor, dtype):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        for dtype, label in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
            taps = [dev(tap, dtype) for tap in taps_host]
            rows = [row.clone() for row in rows_host]
            # The carried state, float32 in both variants: only the FIR's arithmetic changes.
            state = torch.zeros(1, heads, HEAD_K_DIM, HEAD_V_DIM)
            tt_state = dev(state, ttnn.float32)
            ref_state = state.to(torch.float64)
            values = []
            for step in range(STEPS):
                rows = rows[1:] + [tokens[step]]
                # FIR in the variant's dtype, on device.
                acc = None
                for index, row in enumerate(rows):
                    piece = dev(row, dtype)
                    term = ttnn.multiply(piece, taps[index])
                    ttnn.deallocate(piece)
                    acc = term if acc is None else _add(acc, term)
                conv = ttnn.silu(acc)
                ttnn.deallocate(acc)
                # The recurrence: a rank-1 update of the carried state from the FIR's output, the
                # same shape of arithmetic ``_linear_attention_decode`` does.
                got = ttnn.to_torch(conv).float()
                ttnn.deallocate(conv)
                k = got[..., : HEAD_K_DIM * NUM_V_HEADS].reshape(1, heads, 1, HEAD_K_DIM)
                v = got[..., -HEAD_V_DIM * NUM_V_HEADS :].reshape(1, heads, 1, HEAD_V_DIM)
                state = 0.9 * state + k.transpose(-2, -1) @ v

                reference = torch.zeros(1, 1, BATCH, CONV_DIM, dtype=torch.float64)
                for index, row in enumerate(rows):
                    reference = reference + row.to(torch.float64) * taps_host[index].to(torch.float64)
                reference = torch.nn.functional.silu(reference)
                ref_k = reference[..., : HEAD_K_DIM * NUM_V_HEADS].reshape(1, heads, 1, HEAD_K_DIM)
                ref_v = reference[..., -HEAD_V_DIM * NUM_V_HEADS :].reshape(1, heads, 1, HEAD_V_DIM)
                ref_state = 0.9 * ref_state + ref_k.transpose(-2, -1) @ ref_v
                values.append(pcc(state, ref_state))
            print(
                f"fir_compounding dtype={label} steps={STEPS} "
                + " ".join(f"s{index}={value:.6f}" for index, value in enumerate(values)),
                flush=True,
            )
            for tap in taps:
                ttnn.deallocate(tap)
            ttnn.deallocate(tt_state)
    finally:
        ttnn.close_mesh_device(device)


def _add(acc, term):
    out = ttnn.add(acc, term)
    ttnn.deallocate(acc)
    ttnn.deallocate(term)
    return out


if __name__ == "__main__":
    main()
