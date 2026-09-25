# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The two ~2 %-of-step layout ops the ``linear_attention`` decode profile still shows.

The committed optimized ``linear_attention`` decode report has an
``UntilizeWithUnpaddingDeviceOperation`` on **1 core** and a ``TilizeWithValPaddingDeviceOperation``
on **2 cores**, twice per step, together about 2 % of the traced step.  They are not ops this stage
writes: they are how ``ttnn.repeat_interleave`` expands the 16 gated-delta-net key heads to the 48
value heads along ``dim=2``, which is a *tile* axis, so the tensor is untilized, concatenated as 48
row-major pieces and re-tilized.

The norm applied afterwards is per-head over the last dim and the repeated heads are identical, so
``norm(repeat(x)) == repeat(norm(x))``.  :attr:`DecodeGeometry.norm_before_repeat` therefore normalises
the 16 heads first and expands afterwards along ``dim=1`` of ``[1, B*16, 1, dk]`` - a batch axis, which
needs no layout change - and this probe measures both orders at both decode regimes, in-model, with
the PCC alongside so "exactly equivalent" is a measurement and not an argument.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_norm_repeat_order.py
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import time

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import DEFAULT_GEOMETRY, DEFAULT_POLICY, OptimizedDecoder

PREFILL_LEN = 2048
REPLAYS = 8
SAMPLES = 5


def time_traced_decode(lut, mesh, batch):
    """The same capture-once/replay-many timing ``probe_optimized.py`` uses, via the same helper."""
    stats = ref.load_weight_stats()
    runner = H.TracedDecode(lut, batch=batch)
    token = ref.synthetic_hidden_states(lut.config, batch, 1, stats, seed=400)
    positions = torch.full((batch,), PREFILL_LEN)
    runner.warmup(token, positions)
    runner.capture()
    runner.replay(token, positions)
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - start) * 1e3 / REPLAYS)
    runner.release()
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=23887872)
    results: list = []
    try:
        for label, geometry in (
            ("norm before the expand, dim=1 (shipped)", DEFAULT_GEOMETRY),
            (
                "expand before the norm, dim=2 (stage 2)",
                dataclasses.replace(DEFAULT_GEOMETRY, norm_before_repeat=False),
            ),
        ):
            for batch in (1, 32):
                row = {
                    "sweep": "norm_repeat_order",
                    "kind": "linear_attention",
                    "candidate": label,
                    "batch": batch,
                    "norm_before_repeat": geometry.norm_before_repeat,
                }
                try:
                    lut = H.build_layer(
                        mesh,
                        H.LINEAR_LAYER_IDX,
                        max_batch=batch,
                        max_seq_len=4096,
                        decoder_cls=OptimizedDecoder,
                        policy=DEFAULT_POLICY,
                        decode_geometry=geometry,
                    )
                    stats = ref.load_weight_stats()
                    hidden = ref.synthetic_hidden_states(lut.config, 1, PREFILL_LEN, stats)
                    got = H.run_tt_prefill(lut, hidden)
                    cache = DynamicCache(config=lut.config)
                    golden = H.reference_prefill(lut, hidden, cache)
                    row["prefill_pcc"] = H.pcc(golden.reshape(1, PREFILL_LEN, -1), got)
                    H.prepare_decode(lut)
                    # The token is built for the whole batch because ``decode_forward`` asserts its
                    # batch equals ``max_batch``; only user 0 has a reference, since only user 0 was
                    # prefilled.
                    token = ref.synthetic_hidden_states(lut.config, batch, 1, stats, seed=7)
                    golden_decode = H.reference_decode(lut, token[:1], PREFILL_LEN, cache)
                    decoded = H.run_tt_decode(lut, token, torch.full((batch,), PREFILL_LEN))
                    row["decode_pcc"] = H.pcc(golden_decode.reshape(1, 1, -1), decoded[:1])
                    row["decode_ms"], row["decode_std"] = time_traced_decode(lut, mesh, batch)
                except Exception as exc:  # noqa: BLE001 - a blocker is a result
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                finally:
                    H.release_layers()
                results.append(row)
                print(
                    f"  {label:42s} batch={batch:<3d} decode {row.get('decode_ms', float('nan')):7.4f} ms "
                    f"+-{row.get('decode_std', 0):.4f}  prefill_pcc {row.get('prefill_pcc', float('nan')):.6f} "
                    f"decode_pcc {row.get('decode_pcc', float('nan')):.6f}"
                    + (f"  ERROR {row['error']}" if row.get("error") else ""),
                    flush=True,
                )
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
