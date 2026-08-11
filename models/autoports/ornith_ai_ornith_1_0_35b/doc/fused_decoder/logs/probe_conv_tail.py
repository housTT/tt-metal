# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B for the depthwise-conv history tail: keep it ROW_MAJOR, or materialise it in TILE.

``_causal_conv_prefill`` has to hand ``kernel-1`` rows of conv history to ``_write_conv_state``,
which tilizes one row at a time into the persistent state buffers. Two spellings:

* **ROW_MAJOR tail (shipped)** — return the ROW_MAJOR slice of the padded stream and let
  ``_write_conv_state`` tilize each row as it writes it. ``kernel-1`` layout conversions.
* **TILE tail** — tilize the whole tail once here, then row-slice it in TILE. One extra whole-tail
  conversion, and each row slice of a TILE tensor at a non-tile-aligned offset lowers to
  untilize/slice/tilize anyway.

Both are exercised **on the real decoder** at the 2048-token prefill shape by monkeypatching the
variant in, so what is timed is the layer, not a synthetic tensor. The shipped implementation is the
baseline arm; the other arm is defined here and is genuinely different code — the probe asserts that
before timing, so it can never degenerate into timing one implementation against a copy of itself.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_conv_tail.py
"""

import inspect
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import fused_decoder as FD

SEQ = 2048
CTX = 8192

#: Bound to the shipped implementation in ``main`` so ``tail_tile`` can wrap it.
_SHIPPED_CAUSAL_CONV = None


def tail_tile(self, qkv, logical_len):
    """The TILE-tail variant.

    Deliberately written as a wrapper around the shipped implementation so the two arms can differ
    in **exactly one** thing — where the conv-history tail is tilized — and nothing else. Writing it
    out as a second copy of the whole function is how round 2 of this stage's review found the probe
    had silently become a comparison of one implementation against itself.
    """
    activated, tail_rm = _SHIPPED_CAUSAL_CONV(self, qkv, logical_len)
    tail = ttnn.to_layout(tail_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tail_rm)
    return activated, tail


def _assert_arms_differ(shipped, variant):
    """Guard against a null A/B: the two arms must not compute the tail the same way."""
    a = inspect.getsource(shipped).split("tail =", 1)[1].split("\n")[0].strip()
    b = inspect.getsource(variant).split("tail =", 1)[1].split("\n")[0].strip()
    assert a != b, (
        "the two arms compute `tail` the same way — this A/B would be timing one implementation "
        "against a copy of itself"
    )


def main():
    cfg = R.load_text_config()
    sd = R.load_layer_state_dict(0)
    global _SHIPPED_CAUSAL_CONV
    shipped = FD.FusedDecoder._causal_conv_prefill
    _SHIPPED_CAUSAL_CONV = shipped
    _assert_arms_differ(shipped, tail_tile)
    print("CONVTAIL arms verified distinct: the shipped tail stays ROW_MAJOR, the variant tilizes it")

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        decoder = FD.FusedDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=0, mesh_device=mesh, max_context=CTX)
        decoder.allocate_state(1)
        x = ttnn.from_torch(
            (torch.randn(1, SEQ, cfg.hidden_size, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        reference = None
        for name, impl in (("row-major tail (shipped)", shipped), ("tile tail", tail_tile)):
            FD.FusedDecoder._causal_conv_prefill = impl
            decoder.reset_state()
            out = decoder.prefill_forward(x)
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            if reference is None:
                reference, pcc, exact = got, 1.0, True
            else:
                a = reference.double().flatten() - reference.double().mean()
                b = got.double().flatten() - got.double().mean()
                pcc = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
                exact = torch.equal(reference, got)
            ttnn.synchronize_device(mesh)
            times = []
            for _ in range(5):
                decoder.reset_state()
                start = time.time()
                out = decoder.prefill_forward(x)
                ttnn.synchronize_device(mesh)
                times.append(time.time() - start)
                ttnn.deallocate(out)
            print(
                f"CONVTAIL {name:26s} best={min(times) * 1e3:7.2f} ms "
                f"mean={sum(times) / len(times) * 1e3:7.2f} ms pcc_vs_shipped={pcc:.6f} "
                f"{'bitwise-equal' if exact else 'differs'}",
                flush=True,
            )
        FD.FusedDecoder._causal_conv_prefill = shipped
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
