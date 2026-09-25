# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Buy full-context scale margin per millisecond of prefill: mixed per-role prefill fidelities.

The two probes before this one bracket the problem.  `probe_prefill_fidelity_roles.py` found that the
full-context prefill tail *scale* is fixed by raising the **MLP's** prefill fidelity - to 0.980308 at
HiFi2 and 0.985473 at HiFi4, from 0.968952 at LoFi - and that `wqkv` alone fixes the paged V cache's
scale (0.988562 -> 0.999276) without moving the tail.  `probe_prefill_fidelity_cost.py` priced them:
HiFi2 on the MLP costs 16 % of `linear_attention` prefill and 32 % of `full_attention`, while **HiFi4 on
the MLP costs 48 % and lands `linear_attention` prefill at 28.394 ms, slower than the stage-2 baseline's
25.830** - so uniform HiFi4 is not a shippable fix.

That leaves HiFi2-on-everything at 0.980308, which clears the (0.98, 1.02) tolerance by 3e-4.  It is a
deterministic 3e-4 rather than a noisy one, but a gate that passes by 3e-4 is one unrelated change away
from failing, so this probe looks for margin that is cheaper than uniform HiFi4.

The idea it tests: the three MLP matmuls are not equally implicated.  `mlp_down` reduces over 17408
elements - three times the depth of gate/up - and a per-element truncation bias accumulates over the
reduction, so raising `mlp_down` alone may buy most of the accuracy for a third of the width.  Each arm
reports the full-context real-weight tail scale *and* the warmed 2048-token prefill cost, so the choice
is made on accuracy per millisecond rather than on either alone.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_prefill_fidelity_mixed.py
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
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    HIFI2,
    HIFI4,
    OptimizedDecoder,
)

PROMPT = 262143
TAIL = 256
PREFILL_LEN = 2048
SAMPLES = 5


def candidates():
    """``(label, prefill_fidelity_roles)`` - all of them prefill-only by construction."""
    yield "shipped (LoFi at prefill)", None
    yield "HiFi2 on wqkv+MLP", {r: HIFI2 for r in ("wqkv", "mlp_gate", "mlp_up", "mlp_down")}
    yield "HiFi4 on mlp_down only", {"mlp_down": HIFI4}
    yield "HiFi4 on wqkv+mlp_down", {"wqkv": HIFI4, "mlp_down": HIFI4}
    yield "HiFi2 on wqkv+gate/up, HiFi4 on mlp_down", {
        "wqkv": HIFI2,
        "mlp_gate": HIFI2,
        "mlp_up": HIFI2,
        "mlp_down": HIFI4,
    }
    yield "HiFi4 on wqkv+gate/up, HiFi4 on mlp_down", {
        "wqkv": HIFI4,
        "mlp_gate": HIFI4,
        "mlp_up": HIFI4,
        "mlp_down": HIFI4,
    }


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
    return statistics.median(samples)


def accuracy(mesh, policy) -> dict:
    """Full-context real-weight tail scale, the metric the gate is on."""
    context = ref.load_text_config().max_position_embeddings
    lut = H.build_layer(
        mesh,
        H.FULL_LAYER_IDX,
        max_batch=1,
        max_seq_len=context,
        real_weights=True,
        decoder_cls=OptimizedDecoder,
        policy=policy,
        decode_geometry=DEFAULT_GEOMETRY,
    )
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PROMPT, stats)
    got = H.run_tt_prefill(lut, hidden)
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : PROMPT - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, PROMPT - TAIL :, :].contiguous(), cache)
    keys, values = H.read_paged_kv(lut, user_id=0, seq_len=PROMPT)
    ref_keys, ref_values = H.reference_cache_kv(lut, cache, PROMPT)
    out = {
        "prefill_tail_pcc": H.pcc(golden, got[:, -TAIL:, :]),
        "prefill_tail_scale": H.scale_ratio(golden, got[:, -TAIL:, :]),
        "paged_v_cache_scale": H.scale_ratio(ref_values, values),
        "paged_k_cache_scale": H.scale_ratio(ref_keys, keys),
    }
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)
    golden_decode = H.reference_decode(lut, token, PROMPT, cache)
    decoded = H.run_tt_decode(lut, token, torch.tensor([PROMPT]))
    out["decode_scale"] = H.scale_ratio(golden_decode, decoded)
    return out


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for label, roles in candidates():
            policy = (
                DEFAULT_POLICY
                if roles is None
                else dataclasses.replace(
                    DEFAULT_POLICY, name=f"opt-v1-mixed-{abs(hash(label)) % 10000}", prefill_fidelity_roles=roles
                )
            )
            row = {
                "sweep": "prefill_fidelity_mixed",
                "candidate": label,
                "roles": {r: str(f) for r, f in (roles or {}).items()},
                "real_weights": True,
            }
            try:
                row.update(accuracy(mesh, policy))
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                H.release_layers()
            for kind, layer_idx in (("linear_attention", H.LINEAR_LAYER_IDX), ("full_attention", H.FULL_LAYER_IDX)):
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
                    row[f"prefill_ms_{kind}"] = time_prefill(lut, mesh)
                except Exception as exc:  # noqa: BLE001
                    row[f"prefill_ms_{kind}"] = None
                    row.setdefault("cost_error", f"{type(exc).__name__}: {exc}"[:160])
                finally:
                    H.release_layers()
            results.append(row)
            print(
                f"  {label:42s} tail_scale {row.get('prefill_tail_scale', float('nan')):.6f} "
                f"decode_scale {row.get('decode_scale', float('nan')):.6f} "
                f"V_scale {row.get('paged_v_cache_scale', float('nan')):.6f}  "
                f"prefill linear {row.get('prefill_ms_linear_attention') or float('nan'):7.3f} "
                f"full {row.get('prefill_ms_full_attention') or float('nan'):7.3f} ms"
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
