# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""In-model candidate sweeps for the optimized Qwen3.6-27B decoder.

Every number this probe prints is measured on the **real layer**, at the real shapes, with the
real weights the caller asks for, through the same warmed-prefill and captured-trace-replay
windows the stage's perf test uses.  The model-free ``probe_matmul_policy.py`` finds the legal
envelope cheaply; this probe is what actually decides, because a matmul that is 20 us faster in
isolation can lose once the reshards, slices and residual layout around it are counted.

Subcommands::

    policy    precision/fidelity candidates at the shipped geometry
    geometry  decode core count and per-role in0_block_w at the shipped policy
    prefill   packed-vs-split MLP and explicit 2D prefill program configs
    final     the shipped configuration and the fused baseline, side by side

Each row prints ``PROBEROW <json>``; ``make_doc_tables.py`` turns those into the work-log tables,
so no table in the documents is transcribed by hand.

    cd /home/ttuser/dev/qwen/tt-metal
    source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh
    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_optimized.py policy
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as OD
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    BFP8_POLICY,
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    FUSED_BASELINE_GEOMETRY,
    FUSED_BASELINE_POLICY,
    DecodeGeometry,
    OptimizedDecoder,
    PrefillGeometry,
)

#: The policy the *geometry* and *prefill* sweeps hold fixed.  Set from the command line so a
#: geometry sweep can be re-run against whichever precision policy the precision sweep selected -
#: OPT-014: a geometry measured under one dtype does not decide the geometry under another.
SWEEP_POLICY = DEFAULT_POLICY

#: ``K_tiles / cores`` per role at the shipped core count, i.e. the set of legal ``in0_block_w``
#: values.  Filled in from the built layer, so it follows the shipped ``DecodeGeometry.cores``.
_K_TILES_PER_CORE: dict = {}

PREFILL_LEN = 2048
DECODE_POS = 2048
#: Trace replays per timing sample.  Large enough that the ~40 us host cost of one
#: ``execute_trace`` + ``synchronize_device`` pair is a small share of the sample, small enough
#: that a sample is a few milliseconds.
REPLAYS = 16
SAMPLES = 9
#: Prefill length the PCC sanity check uses.  Short so an HF forward is cheap; the shipped
#: configuration's full PCC evidence is the test suite's job, not this probe's.
PCC_LEN = 512
PCC_STEPS = 2


def _median_stdev(samples):
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def build(mesh, layer_idx, *, policy, geometry, prefill_geometry=None, batch=1, real_weights=False, cls=None):
    kwargs = {}
    if cls is None or cls is OptimizedDecoder:
        cls = OptimizedDecoder
        kwargs = {"policy": policy, "decode_geometry": geometry, "prefill_geometry": prefill_geometry}
    return H.build_layer(
        mesh,
        layer_idx,
        max_batch=batch,
        max_seq_len=8192,
        real_weights=real_weights,
        decoder_cls=cls,
        **kwargs,
    )


def time_prefill(lut, mesh, length=PREFILL_LEN, samples=3):
    hidden = ref.synthetic_hidden_states(lut.config, 1, length, ref.load_weight_stats())
    tt_in = H.tt_hidden_prefill(hidden, mesh)
    rot = H.prefill_rot_mats(lut, length, mesh) if lut.is_full_attention else None
    full_pt, per_chunk = H.chunk_page_tables(lut, length, 0, mesh)

    def once():
        out = lut.tt_layer.prefill_forward(
            tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
        )
        ttnn.deallocate(out)

    once()
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        once()
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - start) * 1e3)
    ttnn.deallocate(tt_in)
    return _median_stdev(times)


def time_traced_decode(lut, mesh, batch, position=DECODE_POS):
    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, position, stats)
    H.run_tt_prefill(lut, hidden)
    H.prepare_decode(lut)
    ttnn.synchronize_device(mesh)
    runner = H.TracedDecode(lut, batch=batch)
    token = ref.synthetic_hidden_states(lut.config, batch, 1, stats, seed=400)
    positions = torch.full((batch,), position)
    runner.warmup(token, positions)
    runner.capture()
    runner.replay(token, positions)
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, runner.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - start) * 1e3 / REPLAYS)
    runner.release()
    return _median_stdev(times)


def pcc_check(lut, mesh, batch=1):
    """Prefill + decode PCC against the HF layer at a short length, as a candidate sanity check."""
    from transformers.cache_utils import DynamicCache

    stats = ref.load_weight_stats()
    hidden = ref.synthetic_hidden_states(lut.config, 1, PCC_LEN, stats)
    cache = DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    actual = H.run_tt_prefill(lut, hidden)
    prefill_pcc = H.pcc(golden.reshape(1, PCC_LEN, -1), actual)
    H.prepare_decode(lut)
    decode_pcc = 1.0
    for step in range(PCC_STEPS):
        token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=700 + step)
        ref_out = H.reference_decode(lut, token, PCC_LEN + step, cache)
        got = H.run_tt_decode(lut, token, torch.full((batch,), PCC_LEN + step))
        decode_pcc = min(decode_pcc, H.pcc(ref_out.reshape(1, 1, -1), got[:1]))
    return prefill_pcc, decode_pcc


def emit(row: dict) -> None:
    print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
    print(
        f"  {row.get('kind', '?'):17s} {row['candidate']:44s} "
        f"prefill {row.get('prefill_ms', float('nan')):8.3f} +-{row.get('prefill_std', 0):5.3f}  "
        f"decode {row.get('decode_ms', float('nan')):7.4f} +-{row.get('decode_std', 0):6.4f}  "
        f"pcc {row.get('prefill_pcc', float('nan')):.6f}/{row.get('decode_pcc', float('nan')):.6f}"
        + (f"  ERROR {row['error']}" if row.get("error") else ""),
        flush=True,
    )


# ------------------------------------------------------------------ candidate sets


def policy_candidates():
    """Precision candidates, moved one tensor group at a time from the fused baseline.

    The order is the order the work log walks: fidelity alone, then the block-float step, then the
    KV cache, then BFP4 group by group, then the fidelity/BFP4 cross-product that OPT-014 asks for.
    """
    lofi_only = dataclasses.replace(
        FUSED_BASELINE_POLICY,
        name="bf16-lofi",
        attn_fidelity=OD.LOFI,
        mlp_fidelity=OD.LOFI,
        gdn_proj_fidelity=OD.LOFI,
        gdn_qkv_fidelity=OD.HIFI2,
        decode_fidelity=OD.HIFI2,
        fp32_dest_acc_all=False,
    )
    bfp8 = BFP8_POLICY
    bfp8_hifi2_prefill = dataclasses.replace(
        bfp8, name="bfp8-hifi2", attn_fidelity=OD.HIFI2, mlp_fidelity=OD.HIFI2, gdn_proj_fidelity=OD.HIFI2
    )
    bfp4_gu = dataclasses.replace(bfp8, name="bfp4-gateup", mlp_weight=ttnn.bfloat4_b)
    bfp4_mlp = dataclasses.replace(bfp4_gu, name="bfp4-mlp", mlp_down_weight=ttnn.bfloat4_b)
    bfp4_attn = dataclasses.replace(
        bfp8, name="bfp4-attn", attn_weight=ttnn.bfloat4_b, gdn_z_weight=ttnn.bfloat4_b, gdn_out_weight=ttnn.bfloat4_b
    )
    yield "fused-baseline bf16/HiFi4", FUSED_BASELINE_POLICY
    yield "bf16 weights, LoFi prefill / HiFi2 decode", lofi_only
    yield "bfp8 all, LoFi prefill / HiFi2 decode", bfp8
    yield "bfp8 all, LoFi both phases", dataclasses.replace(bfp8, name="bfp8-lofi-both", decode_fidelity=None)
    yield "bfp8 all, HiFi2 both phases", bfp8_hifi2_prefill
    yield "bfp8 all + bf16 KV cache", dataclasses.replace(bfp8, name="bfp8-bf16kv", kv_cache=ttnn.bfloat16)
    yield "bfp4 gate/up only (rest bfp8)", bfp4_gu
    yield "bfp4 gate/up, HiFi2 prefill", dataclasses.replace(
        bfp4_gu, name="bfp4-gateup-hifi2", attn_fidelity=OD.HIFI2, mlp_fidelity=OD.HIFI2, gdn_proj_fidelity=OD.HIFI2
    )
    yield "bfp4 MLP incl. down (rest bfp8)", bfp4_mlp
    yield "bfp4 attention only (rest bfp8)", bfp4_attn
    yield "bfp4 MLP + bfp4 attention", dataclasses.replace(
        bfp4_mlp,
        name="bfp4-mlp-attn",
        attn_weight=ttnn.bfloat4_b,
        gdn_z_weight=ttnn.bfloat4_b,
        gdn_out_weight=ttnn.bfloat4_b,
    )
    yield "in_proj_qkv at HiFi2 (stage 2's value)", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-hifi2-gdnqkv", gdn_qkv_fidelity=OD.HIFI2
    )
    yield "in_proj_qkv at HiFi4", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-hifi4-gdnqkv", gdn_qkv_fidelity=OD.HIFI4
    )
    # The dtype x fidelity cross-product OPT-014 asks for on this role: BFP4 at the shipped LoFi and at
    # stage 2's HiFi2, so "BFP4 is rejected" is a statement about the dtype and not about one pairing.
    yield "in_proj_qkv at BFP4", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bfp4-gdnqkv", gdn_qkv_weight=ttnn.bfloat4_b
    )
    yield "in_proj_qkv at BFP4 + HiFi2", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-bfp4-hifi2-gdnqkv", gdn_qkv_weight=ttnn.bfloat4_b, gdn_qkv_fidelity=OD.HIFI2
    )
    yield "no fp32 dest acc on the state roles at decode", dataclasses.replace(
        DEFAULT_POLICY, name="opt-v1-no-state-fp32acc-decode", state_fp32_acc_decode=False
    )
    yield "in_proj_qkv at HiFi2 + no state fp32 dest acc at decode", dataclasses.replace(
        DEFAULT_POLICY,
        name="opt-v1-hifi2-gdnqkv-no-state-fp32acc",
        gdn_qkv_fidelity=OD.HIFI2,
        state_fp32_acc_decode=False,
    )
    yield "shipped", DEFAULT_POLICY


def geometry_candidates(legal_cores):
    """Decode layout candidates at the shipped precision policy."""
    yield "shipped geometry", DEFAULT_GEOMETRY
    for cores in legal_cores:
        if cores == DEFAULT_GEOMETRY.cores:
            continue
        yield f"cores={cores}", DecodeGeometry(cores=cores, in0_block_w={})
    yield "cores=16, in0_block_w=2 everywhere", DecodeGeometry(
        cores=16, in0_block_w={role: 2 for role in OD.DEFAULT_IN0_BLOCK_W}
    )
    yield "cores=16, in0_block_w=1 everywhere", DecodeGeometry(
        cores=16, in0_block_w={role: 1 for role in OD.DEFAULT_IN0_BLOCK_W}
    )
    yield "packed gate/up at decode", dataclasses.replace(DEFAULT_GEOMETRY, split_gate_up_decode=False)
    yield "cores=16, split gate/up (OPT-010 pair)", DecodeGeometry(cores=16, in0_block_w={})
    yield "cores=16, packed gate/up (OPT-010 pair)", DecodeGeometry(
        cores=16, in0_block_w={}, split_gate_up_decode=False
    )
    yield "fused decode layout (no sharded stream, no DRAM-sharded matmuls)", dataclasses.replace(
        DEFAULT_GEOMETRY, sharded_stream=False, dram_sharded=False
    )
    yield "sharded residual, interleaved matmuls (no DRAM sharding)", dataclasses.replace(
        DEFAULT_GEOMETRY, dram_sharded=False
    )
    yield "SiLU fused into the gate matmul epilogue", dataclasses.replace(DEFAULT_GEOMETRY, fuse_gate_silu=True)
    yield "SDPA 1 core per head (stage 1's pinned value)", dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=1)
    yield "SDPA 8 cores per head", dataclasses.replace(DEFAULT_GEOMETRY, sdpa_cores_per_head=8)
    yield "in_proj_ab DRAM-sharded + separate bias add", dataclasses.replace(DEFAULT_GEOMETRY, dram_sharded_ab=True)
    # Both of these are the *alternative* to the shipped default, which is why they are phrased as the
    # thing being tried: the shipped stream grid is row-wise and the shipped q/k order is expand-then-
    # norm, both because the arm below lost.  ``probe_stream_grid.py`` and ``probe_norm_repeat_order.py``
    # measure each of them on more axes than a single decode time.
    yield "rectangular 8x4 stream core grid", dataclasses.replace(DEFAULT_GEOMETRY, rectangular_stream=True)
    yield "q/k norm before the expand, on a batch axis", dataclasses.replace(DEFAULT_GEOMETRY, norm_before_repeat=True)


def in0_block_w_candidates(layer_kind: str, baseline: dict):
    """Per-role ``in0_block_w`` sweeps, one role at a time from the shipped baseline (OPT-004).

    ``baseline`` is the *actual* shipped map, read out of a built layer's ``config_summary`` rather
    than from the constant, because the constant is clamped at construction to what fits L1
    alongside the resident width-sharded activations - so a sweep that starts from the unclamped
    constant measures a configuration the layer never runs.  Every row therefore differs from the
    shipped configuration in exactly one role.
    """
    for role, shipped in sorted(baseline.items()):
        limit = _K_TILES_PER_CORE[role]
        for value in [d for d in range(1, limit + 1) if limit % d == 0]:
            yield role, value, shipped


# --------------------------------------------------------------------- subcommands


def measure_all(lut, mesh, batch=1, want_prefill=True, want_pcc=True, want_decode=True) -> dict:
    """Prefill latency, PCC and traced-decode latency from **one** built layer.

    One build per candidate rather than three: uploading a layer's weights takes tens of seconds,
    and the three measurements do not interfere as long as they run in this order - the prefill
    timing writes no state the PCC check reads, and the traced-decode timing re-prefills for itself.
    """
    out: dict = {}
    if want_prefill:
        out["prefill_ms"], out["prefill_std"] = time_prefill(lut, mesh)
    if want_pcc:
        out["prefill_pcc"], out["decode_pcc"] = pcc_check(lut, mesh, batch=batch)
    if want_decode:
        out["decode_ms"], out["decode_std"] = time_traced_decode(lut, mesh, batch)
    return out


def run_policy(mesh, kinds, real_weights=False):
    for kind, layer_idx in kinds:
        for label, policy in policy_candidates():
            row = {
                "sweep": "policy",
                "kind": kind,
                "candidate": label + (" [real weights]" if real_weights else ""),
                "policy": policy.name,
                "real_weights": real_weights,
            }
            try:
                geometry = FUSED_BASELINE_GEOMETRY if policy.name == "fused-baseline" else DEFAULT_GEOMETRY
                lut = build(mesh, layer_idx, policy=policy, geometry=geometry, real_weights=real_weights)
                row["config"] = lut.tt_layer.config_summary()
                # A row labelled "shipped" has to *be* the shipped configuration.  An earlier revision
                # of this log had a "shipped" row that was really the bfp4-mlp-attn arm, measured at 16
                # cores with the gate SiLU fused, and every percentage derived from that row was wrong
                # in a way nothing else in the artifacts could catch.  So the label is checked against
                # the built layer rather than trusted.
                if label == "shipped":
                    decode = row["config"]["decode"]
                    assert row["config"]["policy"] == DEFAULT_POLICY.name, (
                        f"the 'shipped' row was built with policy {row['config']['policy']!r}, "
                        f"not {DEFAULT_POLICY.name!r}"
                    )
                    assert decode["cores"] == DEFAULT_GEOMETRY.cores, (
                        f"the 'shipped' row was built at {decode['cores']} cores, " f"not {DEFAULT_GEOMETRY.cores}"
                    )
                    assert (
                        decode["gate_silu_fused"] == DEFAULT_GEOMETRY.fuse_gate_silu
                    ), "the 'shipped' row's fused-SiLU setting does not match the shipped geometry"
                row.update(measure_all(lut, mesh))
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                H.release_layers()
            emit(row)


def run_geometry(mesh, kinds):
    for kind, layer_idx in kinds:
        probe = build(mesh, layer_idx, policy=SWEEP_POLICY, geometry=DEFAULT_GEOMETRY)
        summary = probe.tt_layer.config_summary()
        legal = summary["decode"]["legal_cores"]
        baseline = {
            role: entry["decode_program_config"]["in0_block_w"]
            for role, entry in summary["roles"].items()
            if "decode_program_config" in entry
        }
        _K_TILES_PER_CORE.clear()
        _K_TILES_PER_CORE.update(
            {
                role: entry["decode_program_config"]["input_shard_k_tiles"]
                for role, entry in summary["roles"].items()
                if "decode_program_config" in entry
            }
        )
        H.release_layers()
        for label, geometry in geometry_candidates(legal):
            row = {"sweep": "geometry", "kind": kind, "candidate": label, "geometry": dataclasses.asdict(geometry)}
            try:
                lut = build(mesh, layer_idx, policy=SWEEP_POLICY, geometry=geometry)
                row["config"] = lut.tt_layer.config_summary()
                row.update(measure_all(lut, mesh, want_prefill=False))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                H.release_layers()
            emit(row)
        for role, value, shipped in in0_block_w_candidates(kind, baseline):
            geometry = DecodeGeometry(
                cores=DEFAULT_GEOMETRY.cores,
                in0_block_w={**baseline, role: value},
                split_gate_up_decode=DEFAULT_GEOMETRY.split_gate_up_decode,
                fuse_gate_silu=DEFAULT_GEOMETRY.fuse_gate_silu,
            )
            row = {
                "sweep": "in0_block_w",
                "kind": kind,
                "candidate": f"{role} in0_block_w={value}" + (" (shipped)" if value == shipped else ""),
                "role": role,
                "in0_block_w": value,
                "shipped_in0_block_w": shipped,
                "cores": DEFAULT_GEOMETRY.cores,
            }
            try:
                lut = build(mesh, layer_idx, policy=SWEEP_POLICY, geometry=geometry)
                row["decode_ms"], row["decode_std"] = time_traced_decode(lut, mesh, 1)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            finally:
                H.release_layers()
            emit(row)


def run_prefill(mesh, kinds):
    dram_roles = ("mlp_gate", "mlp_up", "mlp_down", "wqkv", "wgate", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")
    candidates = [
        ("split gate/up both phases (shipped)", DEFAULT_GEOMETRY, PrefillGeometry()),
        (
            "packed gate/up at prefill",
            dataclasses.replace(DEFAULT_GEOMETRY, split_gate_up_prefill=False),
            PrefillGeometry(),
        ),
        (
            "derived grid but 10 rows of cores (8x10)",
            DEFAULT_GEOMETRY,
            PrefillGeometry(grids={role: (8, 10) for role in dram_roles}),
        ),
        (
            "derived grid but 4 rows of cores (8x4)",
            DEFAULT_GEOMETRY,
            PrefillGeometry(grids={role: (8, 4) for role in dram_roles}),
        ),
        (
            "in0_block_w=2 on every prefill projection",
            DEFAULT_GEOMETRY,
            PrefillGeometry(in0_block_w={role: 2 for role in dram_roles}),
        ),
        (
            "in0_block_w=8 on every prefill projection",
            DEFAULT_GEOMETRY,
            PrefillGeometry(in0_block_w={role: 8 for role in dram_roles}),
        ),
        # The *ceiling* on the per-role search, rather than a value forced on every role.  With the L1
        # model exact, a higher ceiling only moves the roles it fits - at 16 that is ``wgate`` and
        # ``in_proj_z`` - so these arms answer "is a deeper reduction block faster where it fits?"
        # without also asking every other role to do something illegal.
        (
            "in0_block_w ceiling 4 (per-role search)",
            DEFAULT_GEOMETRY,
            PrefillGeometry(max_block_w=4),
        ),
        (
            "in0_block_w ceiling 16 (per-role search)",
            DEFAULT_GEOMETRY,
            PrefillGeometry(max_block_w=16),
        ),
        (
            "in0_block_w ceiling 32 (per-role search)",
            DEFAULT_GEOMETRY,
            PrefillGeometry(max_block_w=32),
        ),
        (
            "explicit 2D grid 8x8 on the MLP",
            DEFAULT_GEOMETRY,
            PrefillGeometry(grids={"mlp_gate_up": (8, 8), "mlp_down": (8, 8)}),
        ),
        (
            "explicit 2D grid 11x10 on the MLP",
            DEFAULT_GEOMETRY,
            PrefillGeometry(grids={"mlp_gate_up": (11, 10), "mlp_down": (11, 10)}),
        ),
        (
            "explicit 2D grid 8x8 on every projection",
            DEFAULT_GEOMETRY,
            PrefillGeometry(
                grids={
                    role: (8, 8)
                    for role in (
                        "mlp_gate_up",
                        "mlp_down",
                        "wqkv",
                        "wgate",
                        "o_proj",
                        "in_proj_qkv",
                        "in_proj_z",
                        "out_proj",
                    )
                }
            ),
        ),
    ]
    for kind, layer_idx in kinds:
        for label, geometry, prefill_geometry in candidates:
            row = {"sweep": "prefill", "kind": kind, "candidate": label}
            try:
                lut = build(mesh, layer_idx, policy=SWEEP_POLICY, geometry=geometry, prefill_geometry=prefill_geometry)
                row["config"] = lut.tt_layer.config_summary()
                row.update(measure_all(lut, mesh, want_decode=False))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                H.release_layers()
            emit(row)


def run_isolation(mesh, kinds):
    """The 2x2 that separates the precision change from the layout change, in one harness.

    Neither lever can be credited from the shipped number alone, and neither can be credited from a
    sweep that moved both.  These four arms are the same code with the same weights, measured back to
    back: the diagonal is the stage's speed-up and the off-diagonal is how it splits.
    """
    arms = [
        ("fused precision + fused layout", FUSED_BASELINE_POLICY, FUSED_BASELINE_GEOMETRY),
        ("shipped precision + fused layout", DEFAULT_POLICY, FUSED_BASELINE_GEOMETRY),
        ("fused precision + shipped layout", FUSED_BASELINE_POLICY, DEFAULT_GEOMETRY),
        ("shipped precision + shipped layout", DEFAULT_POLICY, DEFAULT_GEOMETRY),
    ]
    for kind, layer_idx in kinds:
        for label, policy, geometry in arms:
            row = {"sweep": "isolation", "kind": kind, "candidate": label, "policy": policy.name}
            try:
                lut = build(mesh, layer_idx, policy=policy, geometry=geometry)
                row["config"] = lut.tt_layer.config_summary()
                row.update(measure_all(lut, mesh))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                H.release_layers()
            emit(row)


def run_final(mesh, kinds, batches=(1, 32)):
    """The shipped configuration and the fused baseline, in the same harness, at every batch."""
    arms = [
        ("fused decoder (stage 2, as shipped there)", FusedDecoder, None, None),
        ("optimized decoder (shipped)", OptimizedDecoder, DEFAULT_POLICY, DEFAULT_GEOMETRY),
    ]
    for kind, layer_idx in kinds:
        for label, cls, policy, geometry in arms:
            for batch in batches:
                row = {"sweep": "final", "kind": kind, "candidate": f"{label} batch {batch}", "batch": batch}
                try:
                    lut = build(mesh, layer_idx, policy=policy, geometry=geometry, batch=batch, cls=cls)
                    if hasattr(lut.tt_layer, "config_summary"):
                        row["config"] = lut.tt_layer.config_summary()
                    if batch == 1:
                        row["prefill_ms"], row["prefill_std"] = time_prefill(lut, mesh)
                    row["decode_ms"], row["decode_std"] = time_traced_decode(lut, mesh, batch)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                finally:
                    H.release_layers()
                emit(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", choices=("policy", "geometry", "prefill", "isolation", "final"))
    parser.add_argument("--kind", choices=("linear_attention", "full_attention", "both"), default="both")
    parser.add_argument("--real-weights", action="store_true", help="run the PCC check on the real checkpoint")
    parser.add_argument(
        "--sweep-policy",
        default=None,
        help="named policy the geometry/prefill sweeps hold fixed (default: the shipped one)",
    )
    args = parser.parse_args()
    global SWEEP_POLICY
    if args.sweep_policy:
        SWEEP_POLICY = {
            "opt-v1": DEFAULT_POLICY,
            "bfp8-all-lofi": BFP8_POLICY,
            "fused-baseline": FUSED_BASELINE_POLICY,
        }[args.sweep_policy]

    kinds = []
    if args.kind in ("linear_attention", "both"):
        kinds.append(("linear_attention", H.LINEAR_LAYER_IDX))
    if args.kind in ("full_attention", "both"):
        kinds.append(("full_attention", H.FULL_LAYER_IDX))

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        if args.sweep == "policy":
            run_policy(mesh, kinds, real_weights=args.real_weights)
        else:
            {"geometry": run_geometry, "prefill": run_prefill, "isolation": run_isolation, "final": run_final}[
                args.sweep
            ](mesh, kinds)
    finally:
        H.release_layers()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
