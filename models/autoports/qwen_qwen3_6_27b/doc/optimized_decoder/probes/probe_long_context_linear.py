# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What costs the **full-context** ``linear_attention`` prefill tail its output *scale*.

``probe_long_context_precision.py`` and ``probe_long_context_mlp.py`` attributed the
``full_attention`` full-context tail, and the answer there was ``wqkv``'s destination-accumulation
precision.  The ``linear_attention`` layer has its own, smaller, unattributed move at the same
context: stage 2 measured a 262143-token prefill tail *scale* of 0.996155 and this stage measures
0.982905 against a ``SCALE_TOLERANCE`` floor of 0.98.  It passes, but "passes with 0.003 of margin
and no attribution" is not an answer, and the scale ratio is exactly the quantity PCC cannot see: the
tail PCC is 0.99 either way, so whatever moved is a systematic gain error, not noise.

This probe attributes it the same way, changing one group at a time against a single reference:

1. the shipped policy and geometry;
2. shipped, with the three **gated-delta-net projection weights** back at bfloat16;
3. shipped, with the **MLP weights** back at bfloat16 (the BFP4 gate/up group);
4. shipped, at **HiFi4** on every projection;
5. ``in_proj_qkv`` at LoFi and at HiFi2, and without float32 destination accumulation at decode.
   These three are not attribution arms - they are the *decision* arms.  ``in_proj_qkv`` produces the
   float32 state the recurrence carries for all 262143 tokens, and §3.1 chooses its fidelity and its
   accumulation precision on 2049-token and real-weight evidence; this is where that choice is checked
   at the advertised context, which is the only place a state-building precision can be checked;
6. shipped, plus float32 destination accumulation at prefill on the two *deep* reductions that feed
   the residual stream - ``out_proj`` (192 K tiles) and ``mlp_down`` (544 K tiles).  This is the
   hypothesis the ``full_attention`` result suggests: a gain error that only shows up over a
   262144-token recurrence is what a bfloat16 accumulator does to a deep reduction;
7. the fused stage's **precision** on this stage's **layout**, which separates a precision cause from
   the prefill program configs and the DRAM width-sharded weights this stage introduced;
8. the fused stage's policy *and* geometry, as the control that has to reproduce stage 2's 0.996155.

The reference - a segmented HF prefill over all 262143 tokens plus the decode step after it - costs
more than every device arm combined and does not depend on the arm, so it is built once from the
first arm's weights and reused.  Every arm uses the same synthetic state dict (seed 0), so this is
the same golden for all of them; the probe asserts the weights match before reusing it.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_long_context_linear.py
"""

from __future__ import annotations

import dataclasses
import json
import sys

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    HIFI2,
    HIFI4,
    LOFI,
    OptimizedDecoder,
)

PROMPT = 262143
TAIL = 8192
SEGMENT = 16384

#: ``--real-weights`` runs every arm on the real checkpoint instead of the stand-in state dict.
REAL_WEIGHTS = "--real-weights" in sys.argv

#: The reference, built once on the first arm and reused: ``(golden_tail, ref_conv, ref_recurrent,
#: golden_decode, weight_fingerprint)``.
_REFERENCE = None


def candidates():
    yield "shipped policy + shipped layout", DEFAULT_POLICY, DEFAULT_GEOMETRY
    yield "shipped + bfloat16 GDN projection weights", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-bf16gdn",
        gdn_qkv_weight=ttnn.bfloat16,
        gdn_z_weight=ttnn.bfloat16,
        gdn_out_weight=ttnn.bfloat16,
    ), DEFAULT_GEOMETRY
    yield "shipped + bfloat16 MLP weights", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bf16mlp", mlp_weight=ttnn.bfloat16, mlp_down_weight=ttnn.bfloat16
    ), DEFAULT_GEOMETRY
    yield "shipped + HiFi4 on every projection", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-hifi4",
        attn_fidelity=HIFI4,
        mlp_fidelity=HIFI4,
        gdn_qkv_fidelity=HIFI4,
        gdn_proj_fidelity=HIFI4,
    ), DEFAULT_GEOMETRY
    # ``in_proj_qkv``'s fidelity is the one that builds the state over all 262143 tokens, so both
    # values are measured here whichever one is shipped.  §3.1 decided it on 2049-token and
    # real-weight evidence; this is the check that the decision survives the advertised context.
    yield "in_proj_qkv at LoFi", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-gdnqkv-lofi", gdn_qkv_fidelity=LOFI
    ), DEFAULT_GEOMETRY
    yield "in_proj_qkv at HiFi2", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-gdnqkv-hifi2", gdn_qkv_fidelity=HIFI2
    ), DEFAULT_GEOMETRY
    yield "in_proj_qkv without float32 dest acc at decode", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-no-state-fp32acc-decode", state_fp32_acc_decode=False
    ), DEFAULT_GEOMETRY
    yield "shipped + prefill fp32 acc on out_proj + mlp_down", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-fp32acc-deep",
        prefill_fp32_acc_roles=DEFAULT_POLICY.prefill_fp32_acc_roles + ("out_proj", "mlp_down"),
    ), DEFAULT_GEOMETRY
    yield "fused precision + shipped layout", dataclasses.replace(
        FUSED_BASELINE_POLICY, name="fused-precision-opt-layout"
    ), DEFAULT_GEOMETRY
    yield "fused policy + fused layout (control)", FUSED_BASELINE_POLICY, FUSED_BASELINE_GEOMETRY


def _fingerprint(lut) -> list:
    """A cheap identity for the weights the reference was built from."""
    state = lut.ref_layer.state_dict()
    return [[name, round(float(tensor.to(torch.float64).abs().sum()), 4)] for name, tensor in sorted(state.items())]


def measure(mesh, policy, geometry) -> dict:
    global _REFERENCE
    context = ref.load_text_config().max_position_embeddings
    lut = H.build_layer(
        mesh,
        H.LINEAR_LAYER_IDX,
        max_batch=1,
        max_seq_len=context,
        real_weights=REAL_WEIGHTS,
        decoder_cls=OptimizedDecoder,
        policy=policy,
        decode_geometry=geometry,
    )
    assert not lut.is_full_attention, "this probe is the linear_attention half of the attribution"
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PROMPT, stats)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=99)

    got = H.run_tt_prefill(lut, hidden)
    assert torch.isfinite(got).all(), "long prefill produced non-finite values"

    if _REFERENCE is None:
        golden, cache = H.reference_prefill_segmented(lut, hidden, SEGMENT)
        layer_cache = cache.layers[lut.layer_idx]
        _REFERENCE = (
            golden[:, -TAIL:, :].clone(),
            layer_cache.conv_states[0].to(torch.float32).clone(),
            layer_cache.recurrent_states[0].to(torch.float32).clone(),
            H.reference_decode(lut, token, PROMPT, cache).clone(),
            _fingerprint(lut),
        )
    golden_tail, ref_conv, ref_recurrent, golden_decode, fingerprint = _REFERENCE
    assert fingerprint == _fingerprint(lut), "this arm's weights differ from the reference's"

    conv_state, recurrent_state = H.read_linear_state(lut, user_id=0)
    out = {
        "conv_state_pcc": H.pcc(ref_conv, conv_state.T),
        "recurrent_state_pcc": H.pcc(ref_recurrent, recurrent_state),
        "recurrent_state_scale": H.scale_ratio(ref_recurrent, recurrent_state),
        "prefill_tail_pcc": H.pcc(golden_tail, got[:, -TAIL:, :]),
        "prefill_tail_scale": H.scale_ratio(golden_tail, got[:, -TAIL:, :]),
    }
    H.prepare_decode(lut)
    decoded = H.run_tt_decode(lut, token, torch.tensor([PROMPT]))
    out["decode_pcc"] = H.pcc(golden_decode, decoded)
    out["decode_scale"] = H.scale_ratio(golden_decode, decoded)
    return out


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for label, policy, geometry in candidates():
            row = {
                "sweep": "long_context_linear",
                "kind": "linear_attention",
                "real_weights": REAL_WEIGHTS,
                "candidate": label,
                "policy": policy.name,
                "geometry": "shipped" if geometry is DEFAULT_GEOMETRY else "fused-baseline",
            }
            try:
                row.update(measure(mesh, policy, geometry))
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                H.release_layers()
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
            print(
                f"  {label:46s} tail_pcc {row.get('prefill_tail_pcc', float('nan')):.6f} "
                f"tail_scale {row.get('prefill_tail_scale', float('nan')):.6f}  "
                f"decode_pcc {row.get('decode_pcc', float('nan')):.6f} "
                f"decode_scale {row.get('decode_scale', float('nan')):.6f}  "
                f"conv {row.get('conv_state_pcc', float('nan')):.6f} "
                f"rec {row.get('recurrent_state_pcc', float('nan')):.6f} "
                f"rec_scale {row.get('recurrent_state_scale', float('nan')):.6f}"
                + (f"  ERROR {row['error']}" if row.get("error") else ""),
                flush=True,
            )
    finally:
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
