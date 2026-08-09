# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ground truth for the ``ttnn.transformer.gated_delta_attn_seq`` port (``O5``).

``.agents/notes/gdn.md`` names this kernel as the largest available prefill win and points at
``models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py`` in the *built*
tree as the input-preparation reference.  That file cannot be imported by package path from
this checkout (different commit, and ``models`` would collide), so it is loaded here by file
path - for validation only.  The decoder itself ports the recipe explicitly.

This probe answers two questions before any decoder code changes:

1. does the kernel plus that recipe reproduce HF's ``torch_chunk_gated_delta_rule`` at the real
   head shapes, at chunk_size 128?
2. how fast is it against the decoder's current chunked implementation?

Run: ``python probe_gdn_kernel.py``
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402

BUILT = "/home/ttuser/.local/lib/model-bringup/tt-metal"


def load_reference():
    """Import the built tree's seq-kernel adapter by file path, with its package deps."""
    sys.path.insert(0, BUILT)
    spec = importlib.util.spec_from_file_location(
        "ref_delta_rule_seq",
        f"{BUILT}/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_seq.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ref_delta_rule_seq"] = module
    # The module does `from .ttnn_delta_rule_ops import ...`; give it a package to resolve.
    module.__package__ = "models.experimental.gated_attention_gated_deltanet.tt"
    spec.loader.exec_module(module)
    return module


def pcc(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    ref = load_reference()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        masks = ref.create_chunk_masks_seq(128, device)
        for seq_len in (128, 2048):
            torch.manual_seed(0)
            H, K, V = 48, 128, 128
            q = torch.randn(1, seq_len, H, K, dtype=torch.float32) * 0.5
            k = torch.randn(1, seq_len, H, K, dtype=torch.float32) * 0.5
            v = torch.randn(1, seq_len, H, V, dtype=torch.float32) * 0.5
            beta = torch.sigmoid(torch.randn(1, seq_len, H, dtype=torch.float32))
            g = -torch.nn.functional.softplus(torch.randn(1, seq_len, H, dtype=torch.float32)) * 0.1

            golden_out, golden_state = m.torch_chunk_gated_delta_rule(
                q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
                chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )

            def to_dev(t):
                return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            start = time.perf_counter()
            out_tt, state_tt = ref.chunk_gated_delta_rule_seq_adapter(
                to_dev(q), to_dev(k), to_dev(v), to_dev(beta), to_dev(g),
                chunk_size=128, initial_state=None, device=device, cached_masks=masks,
            )
            ttnn.synchronize_device(device)
            elapsed = (time.perf_counter() - start) * 1e3
            got = ttnn.to_torch(out_tt).float().reshape(1, seq_len, H, V)
            got_state = ttnn.to_torch(state_tt).float().reshape(1, H, K, V)

            # The first call includes JIT compilation.  §7 calls this kernel "the largest
            # `linear_attention` prefill opportunity", which is a claim about latency, so time it
            # warmed as well - a rejected candidate's prize should be a number.
            q_d, k_d, v_d, b_d, g_d = (to_dev(t) for t in (q, k, v, beta, g))

            def one():
                o, st = ref.chunk_gated_delta_rule_seq_adapter(
                    q_d, k_d, v_d, b_d, g_d, chunk_size=128, initial_state=None,
                    device=device, cached_masks=masks,
                )
                ttnn.deallocate(o)
                ttnn.deallocate(st)

            one()
            ttnn.synchronize_device(device)
            warm_start = time.perf_counter()
            for _ in range(5):
                one()
            ttnn.synchronize_device(device)
            warmed = (time.perf_counter() - warm_start) * 1e3 / 5
            print(json.dumps({
                "seq_len": seq_len,
                "out_pcc": pcc(golden_out, got),
                "state_pcc": pcc(golden_state, got_state),
                "first_call_ms": elapsed,
                "warmed_ms": warmed,
            }), flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
