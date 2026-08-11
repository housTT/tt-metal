# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which precision group costs the **full-context** ``full_attention`` prefill tail its PCC.

`test_full_advertised_context` prefills 262143 tokens and compares the last 256 query rows against
the real HF layer.  At the shipped policy that tail came back at PCC 0.9594 where stage 2 measured
0.998030, and nothing in the 2049-token tests shows it: the same policy is at 0.999103 there.  A 256
000-key attention is a different numerical problem from a 2049-key one - the chunked SDPA merges 512
k chunks, and stage 1 characterised that merge as a one-sided loss in the softmax denominator - so a
precision group that is free at 2049 keys need not be free at 262144.

This probe attributes the loss.  It runs the same reference construction the test uses - a K/V cache
built from ``k_proj``/``v_proj`` + ``k_norm`` + RoPE and the real HF layer over the last 256 queries -
against the device path at four settings, changing one group at a time:

1. the shipped policy;
2. the shipped policy with a **bfloat16 KV cache** (the group the chunked SDPA reads 262144 times);
3. the shipped policy with **bfloat16 attention weights** (the group that produces what is cached);
4. the fused stage's policy, as the control that reproduces stage 2's number.

Each arm reports the tail PCC, the best-fit tail *scale* (the quantity the SDPA merge defect moves,
which PCC cannot see) and the un-paged K/V cache PCC, so a cache-precision cause and an
attention-merge cause are distinguishable rather than conflated.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_long_context_precision.py
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
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    OptimizedDecoder,
)

PROMPT = 262143
TAIL = 256


def candidates():
    yield "shipped policy", DEFAULT_POLICY, DEFAULT_GEOMETRY
    # A bfloat16 cache doubles what the decode SDPA's receive buffers hold, and this op's L1 budget
    # is what the multi-core setting was bought with, so the reduced-cache arms have to move the core
    # count with the dtype or they measure an allocation failure instead of a precision effect.
    yield "shipped + bfloat16 KV cache, SDPA 4 cores", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv", kv_cache=ttnn.bfloat16
    ), dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=4)
    yield "shipped + bfloat16 KV cache, SDPA 1 core", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv", kv_cache=ttnn.bfloat16
    ), dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=1)
    yield "shipped (bfp8 KV), SDPA 1 core", DEFAULT_POLICY, dataclasses.replace(
        DEFAULT_GEOMETRY, sdpa_cores_per_head=1
    )
    yield "shipped + bfloat16 KV cache (SDPA 8 cores)", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv", kv_cache=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + bfloat16 attention weights", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16attn", attn_weight=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + bf16 KV + bf16 attention weights, SDPA 1 core", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv-bf16attn", kv_cache=ttnn.bfloat16, attn_weight=ttnn.bfloat16
    ), dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=1)
    yield "fused-stage policy (control)", FUSED_BASELINE_POLICY, FUSED_BASELINE_GEOMETRY


def measure(mesh, policy, geometry) -> dict:
    context = ref.load_text_config().max_position_embeddings
    lut = H.build_layer(
        mesh,
        H.FULL_LAYER_IDX,
        max_batch=1,
        max_seq_len=context,
        decoder_cls=OptimizedDecoder,
        policy=policy,
        decode_geometry=geometry,
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
        "kv_cache_dtype": str(lut.tt_layer.kv_cache[0].dtype),
        "paged_k_cache_pcc": H.pcc(ref_keys, keys),
        "paged_v_cache_pcc": H.pcc(ref_values, values),
        "prefill_tail_pcc": H.pcc(golden, got[:, -TAIL:, :]),
        "prefill_tail_scale": H.scale_ratio(golden, got[:, -TAIL:, :]),
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
    try:
        for label, policy, geometry in candidates():
            row = {
                "sweep": "long_context_precision",
                "kind": "full_attention",
                "candidate": label,
                "policy": policy.name,
                "sdpa_cores_per_head": geometry.sdpa_cores_per_head,
            }
            try:
                row.update(measure(mesh, policy, geometry))
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                H.release_layers()
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
            print(
                f"  {label:44s} tail_pcc {row.get('prefill_tail_pcc', float('nan')):.6f} "
                f"tail_scale {row.get('prefill_tail_scale', float('nan')):.6f}  "
                f"decode_pcc {row.get('decode_pcc', float('nan')):.6f} "
                f"decode_scale {row.get('decode_scale', float('nan')):.6f}  "
                f"K {row.get('paged_k_cache_pcc', float('nan')):.6f} "
                f"V {row.get('paged_v_cache_pcc', float('nan')):.6f}"
                + (f"  ERROR {row['error']}" if row.get("error") else ""),
                flush=True,
            )
    finally:
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
