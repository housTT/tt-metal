# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint PCC for every precision candidate, at the sequence length the suite uses.

This is the probe that **decides** the precision policy.  ``probe_optimized.py policy`` measures
latency and a synthetic-weight PCC for the same candidates; a synthetic weight tensor is a Gaussian
at the real per-tensor standard deviation, which is close to the worst case for a shared-exponent
block-float format - it has a flat spectrum, no outlier structure and no low-rank concentration for
the shared exponent to exploit.  OPT-012 is explicit that such a case may not veto a lower-precision
policy on its own, and this probe is the real-weight evidence that decides instead.

For each candidate and each layer kind it reports, on the **real** ``Qwen3.6-27B`` checkpoint
weights: prefill PCC at 2049 tokens, decode PCC over four steps from that prefill, and - because
OPT-007 asks for a cache-consuming check when attention-projection precision is what changed - the
minimum PCC over five *traced* decode replays.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_real_weight_policy.py
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
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as OD
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    BFP8_POLICY,
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    OptimizedDecoder,
)

SEQ = 2049
DECODE_STEPS = 4
TRACED_REPLAYS = 5


def candidates():
    bfp8 = BFP8_POLICY
    bfp4_gu = dataclasses.replace(bfp8, name="bfp4-gateup", mlp_weight=ttnn.bfloat4_b)
    bfp4_mlp = dataclasses.replace(bfp4_gu, name="bfp4-mlp", mlp_down_weight=ttnn.bfloat4_b)
    yield "fused-baseline bf16/HiFi4", FUSED_BASELINE_POLICY, FUSED_BASELINE_GEOMETRY
    yield "shipped policy", DEFAULT_POLICY, DEFAULT_GEOMETRY
    yield "bfp8 all + LoFi", bfp8, DEFAULT_GEOMETRY
    yield "bfp4 gate/up only (rest bfp8)", bfp4_gu, DEFAULT_GEOMETRY
    yield "bfp4 MLP incl. down (rest bfp8)", bfp4_mlp, DEFAULT_GEOMETRY
    yield "bfp4 attention only (rest bfp8)", dataclasses.replace(
        bfp8, name="bfp4-attn", attn_weight=ttnn.bfloat4_b, gdn_z_weight=ttnn.bfloat4_b, gdn_out_weight=ttnn.bfloat4_b
    ), DEFAULT_GEOMETRY
    yield "in_proj_qkv at BFP4", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bfp4-gdnqkv", gdn_qkv_weight=ttnn.bfloat4_b
    ), DEFAULT_GEOMETRY
    yield "no fp32 dest acc on the state roles at decode", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-no-state-fp32acc-decode", state_fp32_acc_decode=False
    ), DEFAULT_GEOMETRY
    yield "in_proj_qkv at HiFi2 + no state fp32 dest acc at decode", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-hifi2-gdnqkv-no-state-fp32acc",
        gdn_qkv_fidelity=OD.HIFI2,
        state_fp32_acc_decode=False,
    ), DEFAULT_GEOMETRY
    yield "in_proj_qkv at HiFi2 (stage 2's value)", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-hifi2-gdnqkv", gdn_qkv_fidelity=OD.HIFI2
    ), DEFAULT_GEOMETRY
    yield "bfp4 MLP + bfp4 attention", dataclasses.replace(
        bfp4_mlp,
        name="bfp4-mlp-attn",
        attn_weight=ttnn.bfloat4_b,
        gdn_z_weight=ttnn.bfloat4_b,
        gdn_out_weight=ttnn.bfloat4_b,
    ), DEFAULT_GEOMETRY


def measure(mesh, layer_idx, policy, geometry) -> dict:
    lut = H.build_layer(
        mesh,
        layer_idx,
        max_batch=1,
        max_seq_len=8192,
        real_weights=True,
        decoder_cls=OptimizedDecoder,
        policy=policy,
        decode_geometry=geometry,
    )
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, SEQ, stats)
    cache = DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    got = H.run_tt_prefill(lut, hidden)
    out = {"prefill_pcc": H.pcc(golden, got), "kind": lut.config.layer_types[layer_idx]}
    H.prepare_decode(lut)
    worst = 1.0
    for step in range(DECODE_STEPS):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=100 + step)
        ref_out = H.reference_decode(lut, token, SEQ + step, cache)
        actual = H.run_tt_decode(lut, token, torch.tensor([SEQ + step]))
        worst = min(worst, H.pcc(ref_out, actual))
    out["decode_pcc"] = worst

    # Traced replay from a fresh prefill: the cache-consuming check OPT-007 asks for.
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)
    cache2 = DynamicCache(config=lut.config)
    H.reference_prefill(lut, hidden, cache2)
    runner = H.TracedDecode(lut, batch=1)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=800)
    golden = H.reference_decode(lut, token, SEQ, cache2)
    worst_traced = H.pcc(golden, runner.warmup(token, torch.tensor([SEQ])))
    runner.capture()
    for step in range(1, TRACED_REPLAYS):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=800 + step)
        golden = H.reference_decode(lut, token, SEQ + step, cache2)
        worst_traced = min(worst_traced, H.pcc(golden, runner.replay(token, torch.tensor([SEQ + step]))))
    runner.release()
    out["traced_decode_pcc"] = worst_traced

    # ``in_proj_qkv``'s output *is* the float32 state the causal conv carries, so any candidate that
    # touches its dtype or fidelity has to be checked against HF's own cache object, not only against
    # the layer output.  ``linear_attention`` only.
    if lut.config.layer_types[layer_idx] == "linear_attention":
        H.run_tt_prefill(lut, hidden)
        cache3 = DynamicCache(config=lut.config)
        H.reference_prefill(lut, hidden, cache3)
        conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
        out["conv_state_pcc"] = H.pcc(cache3.layers[layer_idx].conv_states[0].to(torch.float32), conv_state.T)
        out["recurrent_state_pcc"] = H.pcc(
            cache3.layers[layer_idx].recurrent_states[0].to(torch.float32), recurrent_state
        )
    return out


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for kind, layer_idx in (("linear_attention", H.LINEAR_LAYER_IDX), ("full_attention", H.FULL_LAYER_IDX)):
            for label, policy, geometry in candidates():
                row = {"sweep": "real_weight_policy", "kind": kind, "candidate": label, "policy": policy.name}
                try:
                    row.update(measure(mesh, layer_idx, policy, geometry))
                except Exception as exc:  # noqa: BLE001 - a blocker is a result
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                finally:
                    H.release_layers()
                print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
                print(
                    f"  {kind:17s} {label:34s} prefill {row.get('prefill_pcc', float('nan')):.6f}  "
                    f"decode {row.get('decode_pcc', float('nan')):.6f}  "
                    f"traced {row.get('traced_decode_pcc', float('nan')):.6f}  "
                    f"conv {row.get('conv_state_pcc', float('nan')):.6f} "
                    f"rec {row.get('recurrent_state_pcc', float('nan')):.6f}"
                    + (f"  ERROR {row['error']}" if row.get("error") else ""),
                    flush=True,
                )
    finally:
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
