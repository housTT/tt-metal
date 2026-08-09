# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Candidate harness for the optimized decoder: one process, many candidates.

Every optimization decision in ``doc/optimized_decoder/work_log.md`` was taken with this
script.  It opens the mesh once, builds the HF reference once per layer kind, then for each
named candidate builds the layer, checks prefill/decode PCC against that reference and times
a **warmed traced** decode and a warmed prefill.  Running the candidates in one process is
what makes a sweep affordable: the reference forward and the real-weight load dominate a
per-candidate pytest invocation.

    python sweep.py --group precision --kinds linear,full [--real] [--seq 2048]

``--real`` uses the real Qwen3.6-27B checkpoint weights for the layer (the evidence that
decides a precision policy - see ``$optimize`` OPT-012); without it the deterministic
synthetic weights of ``weight_stats.json`` are used, which are a stress probe only.

Output is one JSON object per line on stdout, prefixed ``SWEEP ``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")

import torch  # noqa: E402
import ttnn  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tests import harness as H  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as O  # noqa: E402
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder  # noqa: E402

DECODE_REPLAYS = 32
PREFILL_REPEATS = 3


def emit(**payload) -> None:
    print("SWEEP " + json.dumps(payload, sort_keys=True, default=str), flush=True)


# --------------------------------------------------------------------------- candidates


def _p(name, **changes):
    return O.DEFAULT_PRECISION.with_(name=name, **changes)


B16, B8, B4, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32
LOFI, HIFI2, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4

#: Groups of candidates.  Each entry is ``(label, kwargs for build_layer)``.
CANDIDATE_GROUPS: dict = {
    # Baseline vs. the fused stage, then one tensor group at a time.
    "precision": [
        ("fused_stage_baseline", dict(decoder_cls=FusedDecoder)),
        ("opt_bf16_hifi4", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION})),
        ("mlp_bfp8_hifi2", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="mlp_bfp8_hifi2", mlp_gate_up=B8, mlp_down=B8, proj_fidelity=HIFI2)})),
        ("mlp_bfp8_lofi", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="mlp_bfp8_lofi", mlp_gate_up=B8, mlp_down=B8, proj_fidelity=LOFI)})),
        ("mlp_bfp4_lofi", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="mlp_bfp4_lofi", mlp_gate_up=B4, mlp_down=B8, proj_fidelity=LOFI)})),
        ("mlp_bfp4_all_lofi", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="mlp_bfp4_all_lofi", mlp_gate_up=B4, mlp_down=B4, proj_fidelity=LOFI)})),
        ("plus_attn_bfp8", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="plus_attn_bfp8", mlp_gate_up=B4, mlp_down=B8, proj_fidelity=LOFI,
            attn_qkv=B8, attn_gate=B8, attn_out=B8, gdn_qkv=B8, gdn_z=B8, gdn_out=B8)})),
        ("plus_attn_bfp4", dict(decoder_kwargs={"precision": O.FUSED_BASELINE_PRECISION.with_(
            name="plus_attn_bfp4", mlp_gate_up=B4, mlp_down=B8, proj_fidelity=LOFI,
            attn_qkv=B4, attn_gate=B4, attn_out=B4, gdn_qkv=B4, gdn_z=B4, gdn_out=B4)})),
        ("plus_kv_bfp8", dict(decoder_kwargs={"precision": O.DEFAULT_PRECISION.with_(
            name="plus_kv_bfp8")})),
        ("default_sdpa_hifi4", dict(decoder_kwargs={"precision": O.DEFAULT_PRECISION.with_(
            name="default_sdpa_hifi4", sdpa_fidelity=HIFI4)})),
        ("default_sdpa_lofi", dict(decoder_kwargs={"precision": O.DEFAULT_PRECISION.with_(
            name="default_sdpa_lofi", sdpa_fidelity=LOFI)})),
    ],
    # Finer-grained attention/GDN weight precision, re-measured on the *final* topology
    # (DRAM-sharded decode matmuls, split gate/up, sharded residual) per $optimize OPT-007.
    "attn_precision": [
        ("attn_bfp8_gdn_bfp8", dict(decoder_kwargs={"precision": _p("attn_bfp8_gdn_bfp8")})),
        ("attn_bfp4_qkv", dict(decoder_kwargs={"precision": _p("attn_bfp4_qkv", attn_qkv=B4, gdn_qkv=B4)})),
        ("attn_bfp4_out", dict(decoder_kwargs={"precision": _p("attn_bfp4_out", attn_out=B4, gdn_out=B4)})),
        ("attn_bfp4_gate", dict(decoder_kwargs={"precision": _p("attn_bfp4_gate", attn_gate=B4, gdn_z=B4)})),
        ("attn_bfp4_all", dict(decoder_kwargs={"precision": _p(
            "attn_bfp4_all", attn_qkv=B4, attn_gate=B4, attn_out=B4, gdn_qkv=B4, gdn_z=B4, gdn_out=B4)})),
        ("gdn_ba_bf16", dict(decoder_kwargs={"precision": _p("gdn_ba_bf16", gdn_ba=B16)})),
        ("mlp_down_bfp4", dict(decoder_kwargs={"precision": _p("mlp_down_bfp4", mlp_down=B4)})),
        ("proj_hifi2", dict(decoder_kwargs={"precision": _p("proj_hifi2", proj_fidelity=HIFI2)})),
        ("proj_lofi_nofp32acc", dict(decoder_kwargs={"precision": _p("proj_lofi_nofp32acc", proj_fp32_acc=False)})),
        ("gdn_conv_bf16", dict(decoder_kwargs={"precision": _p("gdn_conv_bf16", gdn_conv=B16)})),
        ("gdn_conv_bf16_plus_attn_bfp4", dict(decoder_kwargs={"precision": _p(
            "gdn_conv_bf16_plus_attn_bfp4", gdn_conv=B16, attn_qkv=B4, attn_gate=B4, attn_out=B4)})),
        ("kv_cache_bf16", dict(decoder_kwargs={"precision": _p("kv_cache_bf16", kv_cache=B16)})),
    ],
    # O2/O3/O4/O6 measured as families, each against the same default.
    "topology": [
        ("default", dict()),
        ("no_dram_sharded_decode", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="no_dram_sharded_decode", dram_sharded_decode=False)})),
        ("no_dram_sharded_plus_prefill_pc", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="no_dram_sharded_plus_prefill_pc", dram_sharded_decode=False,
            prefill_program_configs=True)})),
        ("packed_gate_up", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="packed_gate_up", pack_gate_up=True)})),
        ("dram_interleaved_residual", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="dram_interleaved_residual", sharded_decode_residual=False)})),
        ("decode_cores_16", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="decode_cores_16", decode_cores=16)})),
        ("decode_cores_64", dict(decoder_kwargs={"topology": O.TopologyOptions(
            name="decode_cores_64", decode_cores=64)})),
    ],
    # Short-sequence accuracy: the synthetic-weight suite's worst case for the BFP4 MLP.
    "short_seq": [
        ("fused_stage_baseline", dict(decoder_cls=FusedDecoder)),
        ("default_bfp4_gate_up", dict()),
        ("bfp8_gate_up", dict(decoder_kwargs={"precision": _p("bfp8_gate_up", mlp_gate_up=B8)})),
        ("bfp8_gate_up_hifi2", dict(decoder_kwargs={"precision": _p(
            "bfp8_gate_up_hifi2", mlp_gate_up=B8, proj_fidelity=HIFI2)})),
        ("bfp8_everything", dict(decoder_kwargs={"precision": _p(
            "bfp8_everything", mlp_gate_up=B8, mlp_down=B8)})),
    ],
    "default": [("default", dict())],
    "fused": [("fused_stage_baseline", dict(decoder_cls=FusedDecoder))],
}


# --------------------------------------------------------------------------- measurement


class Reference:
    """HF golden prefill / decode for one layer kind, computed once."""

    def __init__(self, mesh_device, layer_idx, seq_len, real, max_seq_len):
        self.layer_idx = layer_idx
        self.seq_len = seq_len
        self.real = real
        self.max_seq_len = max_seq_len
        self.config = ref.load_text_config()
        self.kind = self.config.layer_types[layer_idx]
        stats = ref.load_weight_stats()
        self.hidden = ref.synthetic_hidden_states(self.config, 1, seq_len, stats)
        self.token = ref.synthetic_hidden_states(self.config, 1, 1, stats, seed=777)
        # A throwaway layer only to reach the HF reference module and rotary through the
        # harness's usual construction path.
        lut = H.build_layer(
            mesh_device, layer_idx, max_batch=1, max_seq_len=max_seq_len, real_weights=real,
            decoder_cls=FusedDecoder,
        )
        cache = DynamicCache(config=self.config)
        self.golden_prefill = H.reference_prefill(lut, self.hidden, cache)
        self.golden_decode = H.reference_decode(lut, self.token, seq_len, cache)
        H.release_layers()


def measure(mesh_device, reference, label, build_kwargs, decode_replays=DECODE_REPLAYS):
    kind = reference.kind
    row = {"candidate": label, "kind": kind, "real_weights": reference.real, "seq_len": reference.seq_len}
    build_kwargs = dict(build_kwargs)
    build_kwargs.setdefault("decoder_cls", O.OptimizedDecoder)
    try:
        lut = H.build_layer(
            mesh_device,
            reference.layer_idx,
            max_batch=1,
            max_seq_len=reference.max_seq_len,
            real_weights=reference.real,
            **build_kwargs,
        )
        # ---- prefill: correctness, then warmed wall time
        got = H.run_tt_prefill(lut, reference.hidden)
        row["prefill_pcc"] = H.pcc(reference.golden_prefill, got)

        tt_in = H.tt_hidden_prefill(reference.hidden, mesh_device)
        rot = H.prefill_rot_mats(lut, reference.seq_len, mesh_device) if lut.is_full_attention else None
        full_pt, per_chunk = H.chunk_page_tables(lut, reference.seq_len, 0, mesh_device)

        def one_prefill():
            out = lut.tt_layer.prefill_forward(
                tt_in, user_id=0, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
            )
            ttnn.deallocate(out)

        one_prefill()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(PREFILL_REPEATS):
            one_prefill()
        ttnn.synchronize_device(mesh_device)
        row["prefill_ms"] = (time.perf_counter() - start) * 1e3 / PREFILL_REPEATS

        # ---- decode: correctness of the traced replay, then warmed traced wall time
        H.run_tt_prefill(lut, reference.hidden)
        H.prepare_decode(lut)
        runner = H.TracedDecode(lut, batch=1)
        positions = torch.tensor([reference.seq_len])
        out = runner.warmup(reference.token, positions)
        row["decode_pcc_eager"] = H.pcc(reference.golden_decode, out)
        runner.capture()
        # The captured trace re-runs the same step from the *post-warmup* state, so restore
        # the pre-decode state first and compare the replay against the same golden.
        H.prepare_decode(lut)
        out = runner.replay(reference.token, positions)
        row["decode_pcc_traced"] = H.pcc(reference.golden_decode, out)
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(decode_replays):
            ttnn.execute_trace(mesh_device, runner.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        row["decode_ms"] = (time.perf_counter() - start) * 1e3 / decode_replays
        runner.release()
        for tensor in (tt_in, full_pt):
            if tensor is not None:
                ttnn.deallocate(tensor)
        for group in (rot, per_chunk):
            for tensor in group or ():
                ttnn.deallocate(tensor)
    except Exception as exc:  # a candidate that cannot run is evidence too
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        H.release_layers()
    emit(**row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="precision")
    parser.add_argument("--kinds", default="linear,full")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated candidate labels to keep")
    parser.add_argument("--replays", type=int, default=DECODE_REPLAYS)
    args = parser.parse_args()

    candidates = CANDIDATE_GROUPS[args.group]
    if args.only:
        keep = set(args.only.split(","))
        candidates = [c for c in candidates if c[0] in keep]

    kinds = {"linear": H.LINEAR_LAYER_IDX, "full": H.FULL_LAYER_IDX}
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for key in args.kinds.split(","):
            reference = Reference(mesh_device, kinds[key], args.seq, args.real, args.max_seq)
            for label, build_kwargs in candidates:
                measure(mesh_device, reference, label, build_kwargs, args.replays)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
