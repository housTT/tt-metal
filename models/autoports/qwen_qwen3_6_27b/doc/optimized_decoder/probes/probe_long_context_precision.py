# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What costs the **full-context** ``full_attention`` prefill tail its accuracy, and its *scale*.

`test_full_advertised_context` prefills 262143 tokens and compares the last 256 query rows against the
real HF layer.  A 262144-key attention is a different numerical problem from a 2049-key one - the
chunked SDPA merges 512 k chunks, and stage 1 characterised that merge as a one-sided loss in the
softmax denominator - so a precision group that is free at 2049 keys need not be free here.  This probe
attributes it by changing one group at a time against the same reference construction the test uses: a
K/V cache built from ``k_proj``/``v_proj`` + ``k_norm`` + RoPE and the real HF layer over the last 256
queries.

It grew twice, and both times because a measurement contradicted the reason for the previous arm set:

* first for the **PCC** loss the shipped policy showed on synthetic weights, which §3.8 traced to
  ``wqkv``'s destination-accumulation precision - the arms for the KV cache dtype, the attention weight
  dtype and the SDPA core count are from that round;
* then for the **scale** loss the shipped policy shows on the **real checkpoint**, where none of those
  arms is the cause and the fused control *passes*.  The later arms bisect the remaining difference
  between the control and the shipped configuration - accumulation policy, MLP weight dtype, fidelity,
  and precision-versus-layout - because "the control passes and we changed several things" is not an
  attribution.

Every arm reports the tail PCC, the best-fit tail *scale*, the decode PCC and scale, and the un-paged
K/V cache PCC **and scale**.  The scale columns are the point: a uniform shrink of the V cache gives an
attention output a few percent small while the cache PCC stays at 0.9999, because PCC cannot see a
scale, and that is indistinguishable from an SDPA merge defect if you only look at PCC.

``--real-weights`` runs every arm on the real checkpoint.  Neither stage 1 nor stage 2 ran the
advertised context on real weights - ``real_weights=True`` appears only in their 8192-token tests - so
that mode is where this stage found the failures it is now attributing.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_long_context_precision.py
        [--real-weights]
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
    yield "shipped (bfp8 KV), SDPA 1 core", DEFAULT_POLICY, dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=1)
    yield "shipped + bfloat16 KV cache (SDPA 8 cores)", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv", kv_cache=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + bfloat16 attention weights", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16attn", attn_weight=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + bf16 KV + bf16 attention weights, SDPA 1 core", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16kv-bf16attn", kv_cache=ttnn.bfloat16, attn_weight=ttnn.bfloat16
    ), dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=1)
    # The control passes and the shipped policy does not, on **real weights**, so the gap has to be
    # bisected rather than guessed: these walk the remaining groups one at a time.  The cache dtype, the
    # attention weight dtype and the SDPA core count are all covered above and none of them is it.
    yield "shipped + float32 dest acc on every projection, both phases", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-fp32acc-all", fp32_dest_acc_all=True
    ), DEFAULT_GEOMETRY
    yield "shipped + bfloat16 MLP weights", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16mlp", mlp_weight=ttnn.bfloat16, mlp_down_weight=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + HiFi4 on every projection", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-hifi4",
        attn_fidelity=ttnn.MathFidelity.HiFi4,
        mlp_fidelity=ttnn.MathFidelity.HiFi4,
        gdn_proj_fidelity=ttnn.MathFidelity.HiFi4,
    ), DEFAULT_GEOMETRY
    yield "fused precision on the shipped layout", dataclasses.replace(
        FUSED_BASELINE_POLICY, name="fused-precision-opt-layout"
    ), DEFAULT_GEOMETRY
    yield "shipped precision on the fused layout", DEFAULT_POLICY, FUSED_BASELINE_GEOMETRY
    yield "fused-stage policy (control)", FUSED_BASELINE_POLICY, FUSED_BASELINE_GEOMETRY


#: ``--real-weights`` runs every arm on the real checkpoint instead of the stand-in state dict.  The
#: full-context test is the only place a *state-building* or *cache-building* precision can be checked,
#: and on real weights it finds things the stand-in does not: the shipped policy's real-weight
#: full-context scale is materially worse than its synthetic one, which no 2049-token or synthetic
#: full-context arm shows.
REAL_WEIGHTS = "--real-weights" in sys.argv


def measure(mesh, policy, geometry) -> dict:
    context = ref.load_text_config().max_position_embeddings
    lut = H.build_layer(
        mesh,
        H.FULL_LAYER_IDX,
        max_batch=1,
        max_seq_len=context,
        real_weights=REAL_WEIGHTS,
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
        # The *scale* of the cache, not just its correlation.  A uniform shrink of the V cache produces
        # exactly the signature these runs show - an attention output a few percent small with a cache
        # PCC still at 0.9999 - because PCC is scale-invariant and cannot see it.  On K a scale error
        # changes how peaked the softmax is instead, so the two are reported separately.
        "paged_k_cache_scale": H.scale_ratio(ref_keys, keys),
        "paged_v_cache_scale": H.scale_ratio(ref_values, values),
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
                "real_weights": REAL_WEIGHTS,
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
                f"V {row.get('paged_v_cache_pcc', float('nan')):.6f}  "
                f"K_scale {row.get('paged_k_cache_scale', float('nan')):.6f} "
                f"V_scale {row.get('paged_v_cache_scale', float('nan')):.6f}"
                + (f"  ERROR {row['error']}" if row.get("error") else ""),
                flush=True,
            )
    finally:
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
