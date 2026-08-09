# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only: does the chunked gated delta rule stay exact at chunk_size 128?

``.agents/notes/gdn.md`` gates the whole ``ttnn.transformer.gated_delta_attn_seq`` port on
this: the kernel hardcodes four 32x32 diagonal blocks and therefore *only* supports
chunk_size 128, while HF's ``torch_chunk_gated_delta_rule`` (and this autoport's
``DELTA_CHUNK``) use 64.  The chunked formulation is an exact reformulation of the
recurrence, so the two should agree to float noise; this measures how much noise.

No device is needed.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    fn = m.torch_chunk_gated_delta_rule
    config = ref.load_text_config()
    torch.manual_seed(0)

    nk = config.linear_num_key_heads
    nv = config.linear_num_value_heads
    dk = config.linear_key_head_dim
    dv = config.linear_value_head_dim

    for seq_len in (128, 256, 2048, 2049, 5000):
        q = torch.randn(1, seq_len, nv, dk, dtype=torch.float32) * 0.5
        k = torch.randn(1, seq_len, nv, dk, dtype=torch.float32) * 0.5
        v = torch.randn(1, seq_len, nv, dv, dtype=torch.float32) * 0.5
        beta = torch.sigmoid(torch.randn(1, seq_len, nv, dtype=torch.float32))
        g = -torch.nn.functional.softplus(torch.randn(1, seq_len, nv, dtype=torch.float32)) * 0.1
        out = {}
        for chunk in (64, 128):
            o, state = fn(
                q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), chunk_size=chunk, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
            out[chunk] = (o, state)
        o64, s64 = out[64]
        o128, s128 = out[128]
        print(
            f"seq_len={seq_len:5d}  out_pcc={pcc(o64, o128):.9f}  state_pcc={pcc(s64, s128):.9f} "
            f"  out_max_abs_diff={(o64 - o128).abs().max().item():.3e}"
            f"  out_scale={o64.abs().max().item():.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
