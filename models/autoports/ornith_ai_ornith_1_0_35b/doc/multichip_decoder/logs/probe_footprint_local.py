# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure the multichip decoder's **per-device** layer footprint, term by term.

``doc/context_contract.json``'s ``multichip_decoder.footprint_change`` states the per-device weight
and KV-cache bytes that decide the multichip context contract. Tensor and expert parallelism change
both, so the single-chip figures from ``doc/optimized_decoder/logs/probe_footprint.py`` do not
transfer and the multichip numbers must be measured, not divided by four on paper: several terms are
*not* 1/4 of the single-chip term. The kv-cache is halved rather than quartered (2 kv heads over 4
devices, so each device owns one whole head), the k/v projection rows are duplicated across each
sharing pair, the RoPE tables and both RMSNorm gains are replicated, and the DeltaNet ``a``/``b``
gates are padded from 8 to 32 columns to keep them tile-aligned.

    python models/autoports/ornith_ai_ornith_1_0_35b/doc/multichip_decoder/logs/probe_footprint_local.py

Both layer kinds, on the target ``1x4`` mesh under ``FABRIC_1D_RING``, each with a paged KV cache
allocated for the full advertised 262144-token context and a batch-1 DeltaNet state. Every row is
``LOCALFOOTPRINT <kind> <term> bytes=<n>`` where ``<n>`` is what **one** device holds; the ``total``
row is what the contract quotes as ``per_device_worst_case_layer_total_bytes``.

Each row is compared against the committed single-chip measurement,
``doc/optimized_decoder/logs/probe_footprint.txt`` (its ``optimized`` policy rows, which is the
policy this stage inherits unchanged). ``single_chip`` is that artifact's byte count for the same
term and ``vs_ideal`` is ``local / (single_chip / 4)``: 1.000 means the term split perfectly four
ways, above 1 means the term is duplicated or padded, and below 1 cannot happen. The comparison is
read from the artifact rather than re-measured here because a 1x1 and a 1x4 mesh cannot be open in
the same process, and quoting the other stage's number by hand is exactly the transcription error
this probe exists to remove.
"""

from __future__ import annotations

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.reference import hf_reference as R
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

CONTEXT = 262144
LAYERS = {0: "linear_attention", 3: "full_attention"}

#: Same term map as the single-chip probe, plus the two weights only the multichip layer holds.
#: A weight name that reaches neither this map nor the KV/state terms lands in ``UNCLASSIFIED``,
#: which the assert at the end refuses, so a weight added later cannot fall out of the total.
TERMS = {
    "router": "moe_shared_and_router_weights",
    "shared_in": "moe_shared_and_router_weights",
    "shared_down": "moe_shared_and_router_weights",
    "expert_select": "moe_shared_and_router_weights",
    "expert_select_index": "moe_shared_and_router_weights",
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

#: Bytes per element, including the block-float shared exponent (one byte per 16 datums), because
#: ``Tensor.element_size()`` raises for bfp4/bfp8. Identical to the single-chip probe's table.
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


def _bytes_of(padded_shape, dtype) -> int:
    volume = 1
    for d in padded_shape:
        volume *= int(d)
    if dtype not in ELEM_BYTES:
        raise AssertionError(f"unmodelled dtype {dtype}: add it to ELEM_BYTES")
    return int(round(volume * ELEM_BYTES[dtype]))


def tensor_bytes(t) -> int:
    """What **one** device holds of ``t``.

    ``ttnn.get_device_tensors`` returns the per-device shards, so shard 0 is the per-device cost for
    a sharded tensor and the whole tensor for a replicated one — which is the point: replication is
    a real per-device cost and must show up as one. Reading the parent tensor's ``padded_shape``
    instead reports the *global* shape and would claim tensor parallelism saved nothing.

    The shards are asserted equal-sized: every mesh mapper this layer uses splits evenly, so an
    unequal split would be a sharding bug rather than a footprint result.
    """
    per_shard = [_bytes_of(s.padded_shape, s.dtype) for s in ttnn.get_device_tensors(t)]
    if len(set(per_shard)) != 1:
        raise AssertionError(f"uneven shard sizes {per_shard} for a {t.padded_shape} {t.dtype} tensor")
    return per_shard[0]


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


#: The committed single-chip measurement this stage's rows are compared against.
SINGLE_CHIP = (
    __import__("pathlib").Path(__file__).resolve().parents[2] / "optimized_decoder" / "logs" / "probe_footprint.txt"
)


def single_chip_terms():
    """``{(kind, term): bytes}`` from the ``optimized`` rows of the single-chip probe's artifact."""
    out = {}
    for line in SINGLE_CHIP.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "FOOTPRINT" and parts[1] == "optimized":
            out[(parts[2], parts[3])] = int(parts[4].split("=", 1)[1])
    if not out:
        raise AssertionError(f"no 'FOOTPRINT optimized' rows in {SINGLE_CHIP}")
    return out


def measure(mesh, cfg, sd, layer_idx, reference):
    decoder = MC.MultichipDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh, max_context=CONTEXT
    )
    decoder.allocate_kv_cache(num_blocks_for_context(CONTEXT))
    decoder.allocate_state(1)

    sink: dict[str, tuple[int, int]] = {}
    holders = [decoder.w]
    moe = getattr(decoder, "moe", None)
    if moe is not None and getattr(moe, "w", None) is not None and moe.w is not decoder.w:
        holders.append(moe.w)
    for holder in holders:
        for name, value in holder.items():
            walk(value, sink, TERMS.get(name, f"UNCLASSIFIED:{name}"))

    for attr in ("cos_table", "sin_table", "trans_mat"):
        walk(getattr(getattr(decoder, "rope", None), attr, None), sink, "rope_cos_sin_tables")

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
    kind = LAYERS[layer_idx]
    tp = MC.DEFAULT_TP

    def row(term, local, ref):
        ideal = ref / tp
        vs = f"{local / ideal:.3f}" if ideal else "n/a"
        print(f"LOCALFOOTPRINT {kind} {term} bytes={local} single_chip={ref} vs_ideal={vs}", flush=True)

    for term in sorted(sink):
        row(term, sink[term], reference.get((kind, term), 0))
    local_total = sum(sink.values())
    ref_total = reference.get((kind, "total"), 0)
    row("total", local_total, ref_total)
    del decoder
    return local_total, ref_total, unclassified


def main():
    cfg = R.load_text_config()
    ttnn.set_fabric_config(MC.DEFAULT_FABRIC_CONFIG)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MC.DEFAULT_MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    reference = single_chip_terms()
    print(f"# Measured PER-DEVICE layer footprint at the full {CONTEXT}-token context, batch 1, TP=EP=4.")
    print("# bytes=<what one device holds>")
    print(f"# single_chip=<same term in {SINGLE_CHIP.parent.parent.name}/logs/{SINGLE_CHIP.name}, 'optimized' policy>")
    print(f"# vs_ideal=<bytes / (single_chip / {MC.DEFAULT_TP})>: 1.000 is a perfect split, >1 is duplication/padding")
    grid = mesh.compute_with_storage_grid_size()
    print(
        f"DEVICE arch={mesh.arch()} devices={mesh.get_num_devices()} worker_grid={grid.x}x{grid.y} "
        f"worker_l1_unreserved_bytes={ttnn.get_max_worker_l1_unreserved_size()}",
        flush=True,
    )
    problems = []
    try:
        for layer_idx in LAYERS:
            sd = R.load_layer_state_dict(layer_idx)
            _, _, unclassified = measure(mesh, cfg, sd, layer_idx, reference)
            problems += unclassified
            del sd
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    assert not problems, f"unclassified device tensors: {sorted(set(problems))}"


if __name__ == "__main__":
    main()
