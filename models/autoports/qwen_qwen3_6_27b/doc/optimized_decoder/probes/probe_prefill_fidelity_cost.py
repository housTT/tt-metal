# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What a prefill-only fidelity raise costs, so the accuracy fix can be priced.

`probe_prefill_fidelity_roles.py` found which roles' fidelity the full-context prefill tail *scale*
depends on: `wqkv` alone fixes the paged V cache's scale (0.988562 -> 0.999276) but barely moves the tail
(0.968952 -> 0.969268), and the MLP is what moves it - to 0.980308 at HiFi2 and 0.985473 at HiFi4.

That is the accuracy side. This is the price side, and it has to be measured before choosing, because the
MLP is most of prefill's FLOPs: `wqkv` is ~8 % of the `full_attention` prefill and the three MLP matmuls
are ~60 %, so "raise the MLP's prefill fidelity" is not a free correctness fix and could plausibly cost
more prefill than this stage's whole layout change won.

Warmed prefill at 2048 tokens, the same length every perf table in this stage uses, through the same
`time_prefill` the candidate sweeps use, for both layer kinds.  Decode is not measured because none of
these arms touches it: `prefill_fidelity_roles` applies at prefill only, which is the entire point.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_prefill_fidelity_cost.py
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import time

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    HIFI2,
    HIFI4,
    OptimizedDecoder,
)

PREFILL_LEN = 2048
SAMPLES = 5
_MLP = ("mlp_gate", "mlp_up", "mlp_down")


def candidates():
    yield "shipped (LoFi at prefill)", DEFAULT_POLICY
    for fidelity, label in ((HIFI2, "HiFi2"), (HIFI4, "HiFi4")):
        for roles, roles_label in (
            (("wqkv",), "wqkv"),
            (_MLP, "MLP"),
            (("wqkv",) + _MLP, "wqkv+MLP"),
            (("wqkv", "o_proj") + _MLP, "wqkv+o_proj+MLP"),
        ):
            yield f"prefill {label} on {roles_label}", dataclasses.replace(
                DEFAULT_POLICY,
                name=f"opt-v1-prefill-{label.lower()}-{roles_label}",
                prefill_fidelity_roles={role: fidelity for role in roles},
            )


def time_prefill(lut, mesh):
    hidden = ref.synthetic_hidden_states(lut.config, 1, PREFILL_LEN, ref.load_weight_stats())
    tt_in = H.tt_hidden_prefill(hidden, mesh)
    rot = H.prefill_rot_mats(lut, PREFILL_LEN, mesh) if lut.is_full_attention else None
    full_pt, per_chunk = H.chunk_page_tables(lut, PREFILL_LEN, 0, mesh)

    def once():
        out = lut.tt_layer.prefill_forward(
            tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
        )
        ttnn.deallocate(out)

    once()
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        once()
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - start) * 1e3)
    ttnn.deallocate(tt_in)
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for kind, layer_idx in (("linear_attention", H.LINEAR_LAYER_IDX), ("full_attention", H.FULL_LAYER_IDX)):
            for label, policy in candidates():
                row = {
                    "sweep": "prefill_fidelity_cost",
                    "kind": kind,
                    "candidate": label,
                    "policy": policy.name,
                }
                try:
                    lut = H.build_layer(
                        mesh,
                        layer_idx,
                        max_batch=1,
                        max_seq_len=8192,
                        decoder_cls=OptimizedDecoder,
                        policy=policy,
                        decode_geometry=DEFAULT_GEOMETRY,
                    )
                    row["prefill_ms"], row["prefill_std"] = time_prefill(lut, mesh)
                except Exception as exc:  # noqa: BLE001 - a blocker is a result
                    row["error"] = f"{type(exc).__name__}: {exc}"[:220]
                finally:
                    H.release_layers()
                results.append(row)
                print(
                    f"  {kind:17s} {label:34s} prefill {row.get('prefill_ms', float('nan')):8.3f} ms "
                    f"+-{row.get('prefill_std', 0):.3f}" + (f"  ERROR {row['error']}" if row.get("error") else ""),
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
