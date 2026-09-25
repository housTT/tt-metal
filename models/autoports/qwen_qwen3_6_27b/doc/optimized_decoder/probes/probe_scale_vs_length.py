# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is the output *scale* error a long-context effect, or has it been there at every length?

The full-context tests are the only ones in this stage's suite that assert a **scale** ratio; every
shorter test asserts PCC alone.  So when the 262143-token real-weight run came back with a prefill tail
scaled by 0.969, "long context causes it" was an assumption with no evidence either way - a constant
gain error of the same size would look identical, because nothing shorter looks at scale.

This probe closes that hole the cheap way: the same best-fit scale, on real weights, at lengths from 128
to 8192, for the shipped policy and for the fidelity arms.  It answers a yes/no question that decides how
the finding should be written up:

* if the scale is ~1.0 at short lengths and degrades with length, it is a long-context accumulation
  effect and belongs to the SDPA/state story;
* if it is already off at 128 tokens, it is a **per-matmul systematic gain** - LoFi's mantissa truncation
  rounds magnitudes toward zero - and the full-context test is simply the only place anyone looked.

``k_norm`` and ``q_norm`` normalise any gain error out of K and Q, so the projection that shows it is the
one whose output is not normalised: V.  The paged V cache's scale is therefore reported alongside the
layer output's, and the two together separate "the projection is small" from "attention lost magnitude".

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_scale_vs_length.py
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
    LOFI,
    OptimizedDecoder,
)

LENGTHS = (128, 512, 2048, 8192)


def candidates():
    yield "shipped policy", DEFAULT_POLICY
    yield "attention projections at HiFi2", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-attn-hifi2", attn_fidelity=HIFI2
    )
    yield "attention projections at HiFi4", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-attn-hifi4", attn_fidelity=HIFI4
    )
    yield "every projection at HiFi4", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-hifi4",
        attn_fidelity=HIFI4,
        mlp_fidelity=HIFI4,
        gdn_qkv_fidelity=HIFI4,
        gdn_proj_fidelity=HIFI4,
    )
    yield "attention weights bfloat16, LoFi", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16attn", attn_weight=ttnn.bfloat16, attn_fidelity=LOFI
    )
    yield "attention weights bfloat16, HiFi4", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16attn-hifi4", attn_weight=ttnn.bfloat16, attn_fidelity=HIFI4
    )


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for label, policy in candidates():
            for length in LENGTHS:
                row = {
                    "sweep": "scale_vs_length",
                    "kind": "full_attention",
                    "candidate": label,
                    "policy": policy.name,
                    "seq_len": length,
                    "real_weights": True,
                }
                try:
                    lut = H.build_layer(
                        mesh,
                        H.FULL_LAYER_IDX,
                        max_batch=1,
                        max_seq_len=max(8192, length),
                        real_weights=True,
                        decoder_cls=OptimizedDecoder,
                        policy=policy,
                        decode_geometry=DEFAULT_GEOMETRY,
                    )
                    hidden = ref.synthetic_hidden_states(lut.config, 1, length, ref.load_weight_stats())
                    cache = DynamicCache(config=lut.config)
                    golden = H.reference_prefill(lut, hidden, cache)
                    got = H.run_tt_prefill(lut, hidden)
                    row["prefill_pcc"] = H.pcc(golden, got)
                    row["prefill_scale"] = H.scale_ratio(golden, got)
                    keys, values = H.read_paged_kv(lut, user_id=0, seq_len=length)
                    ref_keys, ref_values = H.reference_cache_kv(lut, cache, length)
                    row["paged_k_cache_scale"] = H.scale_ratio(ref_keys, keys)
                    row["paged_v_cache_scale"] = H.scale_ratio(ref_values, values)
                    row["paged_v_cache_pcc"] = H.pcc(ref_values, values)
                except Exception as exc:  # noqa: BLE001 - a blocker is a result
                    row["error"] = f"{type(exc).__name__}: {exc}"[:220]
                finally:
                    H.release_layers()
                results.append(row)
                print(
                    f"  {label:34s} len={length:<5d} "
                    f"prefill_pcc {row.get('prefill_pcc', float('nan')):.6f} "
                    f"prefill_scale {row.get('prefill_scale', float('nan')):.6f}  "
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
