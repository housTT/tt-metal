# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure the optimized decoder's per-layer device footprint, term by term.

``doc/context_contract.json``'s ``optimized_decoder.footprint_change`` claims a 63 % reduction and
lists a byte count for every term. Those were *modelled* by hand from the shapes and the dtype policy,
which makes them the one group of capability-contract figures no artifact backed — review round 4's
audit port had nothing to check them against. This walks the real device tensors of a built layer
instead, so every term is the allocated padded size at the dtype the shipped policy actually chose.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/optimized_decoder/logs/probe_footprint.py

Both layer kinds, under the selected policy and under the fused-parity policy, each with a paged KV
cache allocated for the full advertised 262144-token context and a batch-1 DeltaNet state. Every row is
``FOOTPRINT <policy> <kind> <term> bytes=<n>``; the ``total`` row is what the contract quotes.
"""

from __future__ import annotations

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    num_blocks_for_context,
)

CONTEXT = 262144
LAYERS = {0: "linear_attention", 3: "full_attention"}

#: Which term of the contract's ``per_term_bytes`` each weight name belongs to. A name that reaches
#: neither this map nor the KV/state terms below lands in ``UNCLASSIFIED``, which the assert at the end
#: refuses — so a weight added later cannot quietly fall out of the total.
TERMS = {
    "router": "moe_shared_and_router_weights",
    "shared_in": "moe_shared_and_router_weights",
    "shared_down": "moe_shared_and_router_weights",
    "expert_gate_up": "moe_routed_expert_weights",
    "expert_down": "moe_routed_expert_weights",
    "attn_in": "projection_weights",
    "o_proj": "projection_weights",
    "gdn_in": "projection_weights",
    "gdn_out": "projection_weights",
    "q_norm": "norms_and_constants",
    "k_norm": "norms_and_constants",
    "attn_norm": "norms_and_constants",
    "ff_norm": "norms_and_constants",
    "gdn_norm": "norms_and_constants",
    "input_norm": "norms_and_constants",
    "post_norm": "norms_and_constants",
    "pre_moe_norm": "norms_and_constants",
    "dt_bias": "norms_and_constants",
    "A_neg": "norms_and_constants",
    "pos_ramp": "norms_and_constants",
    "conv_taps": "conv_weights",
    "conv1d_weights": "conv_weights",
    "conv1d_host": "host_only",
    "gdn_const_tiles": "norms_and_constants",
    "cos": "rope_cos_sin_tables",
    "sin": "rope_cos_sin_tables",
    "trans_mat": "rope_cos_sin_tables",
}


#: Bytes per element. ``Tensor.element_size()`` raises "datum for bfp2, bfp4, bfp8 is invalid" for the
#: block-float dtypes this stage's policy selects, so the shared exponent is counted here: a 16-datum
#: face carries one exponent byte, giving 1 + 1/16 B for bfloat8_b and 0.5 + 1/16 for bfloat4_b.
ELEM_BYTES = {
    ttnn.bfloat16: 2.0,
    ttnn.float32: 4.0,
    ttnn.bfloat8_b: 1.0625,
    ttnn.bfloat4_b: 0.5625,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
    ttnn.uint16: 2.0,
    ttnn.uint8: 1.0,
}


def tensor_bytes(t) -> int:
    """Allocated size of one device tensor: its *padded* volume at its dtype's bytes per element."""
    volume = 1
    for d in t.padded_shape:
        volume *= int(d)
    if t.dtype not in ELEM_BYTES:
        raise AssertionError(f"unmodelled dtype {t.dtype}: add it to ELEM_BYTES")
    return int(round(volume * ELEM_BYTES[t.dtype]))


def walk(value, sink, term):
    """Sum every ttnn device tensor reachable from ``value`` into ``sink[term]``."""
    if isinstance(value, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(value):
            sink[term] = sink.get(term, 0) + tensor_bytes(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            walk(v, sink, term)
    elif isinstance(value, (list, tuple)):
        for v in value:
            walk(v, sink, term)


def measure(mesh, cfg, sd, layer_idx, policy_name):
    decoder = OptimizedDecoder.from_state_dict(
        sd,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh,
        max_context=CONTEXT,
        policy=POLICIES[policy_name],
    )
    blocks = num_blocks_for_context(CONTEXT)
    decoder.allocate_kv_cache(blocks)
    decoder.allocate_state(1)

    sink: dict[str, int] = {}
    holders = [decoder.w]
    moe = getattr(decoder, "moe", None)
    if moe is not None and getattr(moe, "w", None) is not None and moe.w is not decoder.w:
        holders.append(moe.w)
    for holder in holders:
        for name, value in holder.items():
            walk(value, sink, TERMS.get(name, f"UNCLASSIFIED:{name}"))

    # The RoPE cos/sin tables hang off the shared `RotaryEmbedding` helper rather than off `w`, and they
    # are the single largest non-cache term at the full context, so they are walked explicitly.
    walk(getattr(getattr(decoder, "rope", None), "cos_table", None), sink, "rope_cos_sin_tables")
    walk(getattr(getattr(decoder, "rope", None), "sin_table", None), sink, "rope_cos_sin_tables")
    walk(getattr(getattr(decoder, "rope", None), "trans_mat", None), sink, "rope_cos_sin_tables")

    for attr, term in (
        ("k_cache", "paged_kv_cache_k"),
        ("v_cache", "paged_kv_cache_v"),
        ("recurrent_state", "deltanet_recurrent_state_batch1"),
        ("conv_state", "deltanet_conv_state_batch1"),
        ("batch_idxs", "runtime_index_tensors"),
    ):
        walk(getattr(decoder, attr, None), sink, term)

    sink.pop("host_only", None)
    unclassified = sorted(k for k in sink if k.startswith("UNCLASSIFIED"))
    total = sum(sink.values())
    for term in sorted(sink):
        print(f"FOOTPRINT {policy_name} {LAYERS[layer_idx]} {term} bytes={sink[term]}", flush=True)
    print(f"FOOTPRINT {policy_name} {LAYERS[layer_idx]} total bytes={total}", flush=True)
    del decoder
    return total, unclassified


def main():
    cfg = R.load_text_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    print(f"# Measured per-layer device footprint at the full {CONTEXT}-token context, batch 1.")
    print("# Padded volume x element size of every device tensor a built layer holds.")
    totals, problems = {}, []
    try:
        for layer_idx in LAYERS:
            sd = R.load_layer_state_dict(layer_idx)
            for policy_name in ("fused-parity", "optimized"):
                total, unclassified = measure(mesh, cfg, sd, layer_idx, policy_name)
                totals[(policy_name, layer_idx)] = total
                problems += unclassified
            del sd
        for layer_idx, kind in LAYERS.items():
            before, after = totals[("fused-parity", layer_idx)], totals[("optimized", layer_idx)]
            print(
                f"FOOTPRINT delta {kind} before_bytes={before} after_bytes={after} "
                f"reduction_pct={100 * (1 - after / before):.1f}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
    assert not problems, f"unclassified device tensors: {sorted(set(problems))}"


if __name__ == "__main__":
    main()
