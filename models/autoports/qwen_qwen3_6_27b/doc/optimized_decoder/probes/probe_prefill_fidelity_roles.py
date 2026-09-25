# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The smallest prefill-only fidelity change that fixes the full-context scale, on real weights.

§3.8.2 established the mechanism and ruled out everything else: LoFi feeds the FPU a truncated operand
mantissa, truncation rounds magnitudes toward zero, so it is a systematic **gain loss** rather than
symmetric noise.  The evidence is monotone in operand width - at 262143 tokens on real weights the
`full_attention` prefill tail is scaled by 0.969 with the shipped BFP4/BFP8 weights, 0.966 with bfloat16
attention weights and 0.927 with bfloat16 MLP weights - and `HiFi4 on every projection` fixes it,
taking the tail scale to 0.990969 and the paged V cache's scale from 0.988562 to 1.002101.

But `HiFi4 on every projection` is not shippable: it takes the *decode* scale to 1.021896, just outside
the same (0.98, 1.02) tolerance in the other direction, and it costs about 37 % of prefill.

The failing metric is a **prefill** metric, and `PrecisionPolicy.prefill_fidelity_roles` raises fidelity
at prefill only - the mirror of `prefill_fp32_acc_roles`, for the same reason: prefill fills the KV cache
that the next 262144 reads all depend on, while decode writes one row.  So this sweeps role subsets and
fidelities to find the smallest change that brings the prefill tail inside the gate while leaving the
decode scale at the shipped value.

Roles are added in the order the output path visits them, so the table shows what each one is worth:
`wqkv` (which produces the cached V, the one tensor whose scale error is directly visible), then
`o_proj`, then the MLP.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_prefill_fidelity_roles.py
"""

from __future__ import annotations

import dataclasses
import json
import sys

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

_ATTN = ("wqkv",)
_ATTN_OUT = ("wqkv", "o_proj")
_ALL = ("wqkv", "o_proj", "mlp_gate", "mlp_up", "mlp_down")


def candidates():
    yield "shipped (LoFi everywhere at prefill)", DEFAULT_POLICY
    for fidelity, label in ((HIFI2, "HiFi2"), (HIFI4, "HiFi4")):
        for roles, roles_label in ((_ATTN, "wqkv"), (_ATTN_OUT, "wqkv+o_proj"), (_ALL, "wqkv+o_proj+MLP")):
            yield f"prefill {label} on {roles_label}", dataclasses.replace(
                DEFAULT_POLICY,
                name=f"opt-v1-prefill-{label.lower()}-{len(roles)}",
                prefill_fidelity_roles={role: fidelity for role in roles},
            )


def measure(mesh, policy) -> dict:
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
    assert torch.isfinite(got).all(), "long prefill produced non-finite values"

    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : PROMPT - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, PROMPT - TAIL :, :].contiguous(), cache)
    keys, values = H.read_paged_kv(lut, user_id=0, seq_len=PROMPT)
    ref_keys, ref_values = H.reference_cache_kv(lut, cache, PROMPT)
    out = {
        "prefill_tail_pcc": H.pcc(golden, got[:, -TAIL:, :]),
        "prefill_tail_scale": H.scale_ratio(golden, got[:, -TAIL:, :]),
        "paged_k_cache_scale": H.scale_ratio(ref_keys, keys),
        "paged_v_cache_scale": H.scale_ratio(ref_values, values),
        "paged_v_cache_pcc": H.pcc(ref_values, values),
    }
    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)
    golden_decode = H.reference_decode(lut, token, PROMPT, cache)
    decoded = H.run_tt_decode(lut, token, torch.tensor([PROMPT]))
    out["decode_pcc"] = H.pcc(golden_decode, decoded)
    out["decode_scale"] = H.scale_ratio(golden_decode, decoded)
    return out


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for label, policy in candidates():
            row = {
                "sweep": "prefill_fidelity_roles",
                "kind": "full_attention",
                "candidate": label,
                "policy": policy.name,
                "real_weights": True,
            }
            try:
                row.update(measure(mesh, policy))
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:250]
            finally:
                H.release_layers()
            results.append(row)
            print(
                f"  {label:38s} tail_pcc {row.get('prefill_tail_pcc', float('nan')):.6f} "
                f"tail_scale {row.get('prefill_tail_scale', float('nan')):.6f}  "
                f"decode_scale {row.get('decode_scale', float('nan')):.6f}  "
                f"K_scale {row.get('paged_k_cache_scale', float('nan')):.6f} "
                f"V_scale {row.get('paged_v_cache_scale', float('nan')):.6f}"
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
