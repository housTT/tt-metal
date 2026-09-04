# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paired fusion equivalence and warmed latency, with real checkpoint weights."""

import json
import os
import statistics
import time

import pytest
import torch

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from ..tt.fused_decoder import FusedDecoder
from . import attention_fusion_candidates as A
from . import combined_fusion_candidates as C
from . import decode_cast_candidates as E
from . import final_fusion_candidates as F
from . import gdn_concat_candidates as G
from . import gdn_joint_prefill_candidates as J
from . import linear_fusion_candidates as L
from . import mlp_fusion_candidates as M
from . import mode_fusion_candidates as D
from . import test_functional_decoder as H
from . import transpose_fusion_candidates as T
from .fir_fusion_candidates import SharedFIRRows
from .fusion_candidates import (
    CacheAttention,
    FullWidthAttention,
    MatmulSilu,
    NativeAttention,
    PackedAttention,
    PackedMLP,
    PrefillAttention,
)

pytestmark = H.pytestmark


def selected_candidate():
    return {
        "gdn_concat": G.ConcatGDN,
        "default": FusedDecoder,
        "reuse_outer": T.ReuseOuterGDN,
        "reuse_outer_control": T.ReuseOuterControlGDN,
        "reuse_outer_whole": T.ReuseOuterWholeGDN,
        "reuse_outer_whole_control": T.ReuseOuterWholeControlGDN,
        "key_cast_chain": E.KeyCastChainGDN,
        "all_cast": E.AllCastGDN,
        "post_concat_silu": F.PostConcatSiluConv,
        "query_cast": E.QueryCastGDN,
        "value_cast": E.ValueCastGDN,
        "combined_cast": E.CombinedCastGDN,
        "conv_final_combination": F.ConvFinalCombination,
        "conv1d_1024": L.Conv1d1024GDN,
        "norm_dram_gdn": F.NormDRAMGDN,
        "final_combination": F.FinalCombination,
        "conv1d_narrow": L.Conv1dNarrowGDN,
        "conv1d_narrow256": L.Conv1dNarrow256GDN,
        "slice_l1_attention": A.SliceL1Attention,
        "plain_norm_fir": D.PlainNormFIRGDN,
        "mode_norm_gdn": D.ModeNormGDN,
        "conv1d_gdn": L.Conv1dGDN,
        "joint_gate_gdn": C.JointGateGDN,
        "all_mode_gate_gdn": C.AllModeGateGDN,
        "joint_gate_fir_gdn": C.JointGateFIRGDN,
        "hf_prefill_rope": A.HFPrefillRope,
        "joint_prefill_norm": J.JointPrefillNormGDN,
        "joint_prefill_arithmetic": J.JointPrefillArithmetic,
        "separate_z_silu": J.SeparateZSiluGDN,
        "separate_z_control": J.SeparateZSiluControlGDN,
        "shared_fir_rows": SharedFIRRows,
        "joint_qk_before_repeat": L.JointQKNormBeforeRepeat,
        "bias_projection": L.BiasProjectionGDN,
        "bias_projection_fp32": L.BiasProjectionFP32GDN,
        "hybrid_conv_only": L.HybridConvOnly,
        "hybrid_arithmetic": L.HybridArithmetic,
        "hybrid_kda_norm": L.HybridKdaNorm,
        "joint_qk_norm": L.JointQKNormGDN,
        "direct_k_rope": A.NativeShardedDirectK,
        "mlp_epilogue": M.MatmulSiluEpilogue,
        "mlp_epilogue_control": M.MatmulSiluEpilogueControl,
        "mlp_packed_prefill": M.PackedPrefillSeparateDecode,
        "mlp_packed_epilogue": M.PackedPrefillEpilogueDecode,
        "hybrid_norm": L.HybridNormGDN,
        "hybrid_combined": L.HybridCombinedGDN,
        "kda_conv_decode": L.KDAConvDecodeGDN,
        "gate_add": L.GateAddGDN,
        "beta_chain": L.BetaChainGDN,
        "gate_chain": L.GateChainGDN,
        "combined_gdn": L.CombinedGDN,
        "separate_prefill_gdn": L.SeparatePrefillCombinedGDN,
        "combined_rm_tail": L.CombinedRMTailGDN,
        "separate_rm_tail": L.SeparatePrefillCombinedRMTailGDN,
        "packed_attention": PackedAttention,
        "native_attention": NativeAttention,
        "full_width_attention": FullWidthAttention,
        "prefill_attention": PrefillAttention,
        "cache_attention": CacheAttention,
        "packed_mlp": PackedMLP,
        "matmul_silu": MatmulSilu,
        "flat_gdn": L.FlatGDN,
        "packed_gdn": L.PackedGDN,
        "decode_layout_gdn": L.DecodeLayoutGDN,
        "exp_gdn": L.ExpGDN,
        "transpose_gdn": L.TransposeGDN,
        "query_scale_gdn": L.QueryScaleGDN,
        "arithmetic_gdn": L.ArithmeticGDN,
        "kda_conv_gdn": L.KDAConvGDN,
        "split_packed_gdn": L.SplitPackedGDN,
        "softplus_gdn": L.SoftplusGDN,
        "rank_one_gdn": L.RankOneGDN,
        "kda_norm_gdn": L.KdaNormGDN,
        "mixed_silu_gdn": L.MixedSiluGDN,
        "mixed_silu_a_gdn": L.MixedSiluAGDN,
        "mixed_silu_fp32": L.MixedSiluFP32GDN,
        "mixed_silu_fp32_a": L.MixedSiluFP32AGDN,
        "norm_weight_gdn": L.NormWeightGDN,
        "native_sharded_rope": A.NativeShardedHFRope,
        "llama_qk_rope": A.FusedLlamaQKRope,
        "concat_decode": A.ConcatDecode,
        "concat_decode_sharded": A.ConcatDecodeSharded,
        "joint_rope": A.JointHFRope,
    }[os.environ.get("ORNITH_FUSION_CANDIDATE", "default")]


def test_reuse_outer_matched_timing(mesh_device, monkeypatch):
    """Alternate the prior default-program control and final native-transpose runtime."""
    records, handles, outputs = {}, {}, {}
    prompt = H.make_activations(1, 2048, seed=71)
    token = H.make_activations(1, 1, seed=82)
    for name, cls in (("runtime", T.DefaultOuterGDN), ("native_reuse", FusedDecoder)):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, 0, "real")
        x, d = H.to_device(mesh_device, prompt), H.to_device(mesh_device, token)
        ttnn.deallocate(decoder.prefill_forward(x, page_table=table))
        ttnn.deallocate(x)
        saved = H._snapshot_state(decoder)
        pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
        eager = decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)
        outputs[name] = ttnn.to_torch(eager)
        ttnn.deallocate(eager)
        H._restore_state(decoder, saved)
        handles[name] = decoder, saved, None, None, d, pos, rot, table
        records[name] = []
    # Allocate BOTH models and every persistent input before either capture:
    # an older trace's freed intermediates can overwrite later allocations.
    for name, (decoder, saved, _, _, d, pos, rot, table) in list(handles.items()):
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        handles[name] = decoder, saved, trace, out, d, pos, rot, table
    assert H.pcc(outputs["runtime"], outputs["native_reuse"]) >= H.PCC_BAR
    stress = {}
    try:
        for name, (decoder, saved, trace, out, *_) in handles.items():
            H._restore_state(decoder, saved)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert torch.equal(outputs[name], ttnn.to_torch(out))
        for iteration in range(33):
            order = ("runtime", "native_reuse") if iteration % 2 == 0 else ("native_reuse", "runtime")
            for name in order:
                decoder, saved, trace, out, *_ = handles[name]
                H._restore_state(decoder, saved)
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(256):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                duration = (time.perf_counter() - start) * 1000 / 256
                if iteration >= 2:
                    records[name].append(duration)
                if iteration == 32:
                    stress[name] = ttnn.to_torch(out)
        assert H.pcc(stress["runtime"], stress["native_reuse"]) >= H.PCC_BAR
        states = {}
        for name, (decoder, saved, trace, out, *_) in handles.items():
            H._restore_state(decoder, saved)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert torch.equal(outputs[name], ttnn.to_torch(out))
            states[name] = H._snapshot_state(decoder)
        assert H.pcc(states["runtime"]["rec"], states["native_reuse"]["rec"]) >= H.PCC_BAR
        for a, b in zip(states["runtime"]["conv"], states["native_reuse"]["conv"]):
            assert H.pcc(a, b) >= H.PCC_BAR
    finally:
        for _, _, trace, *_ in handles.values():
            ttnn.release_trace(mesh_device, trace)
    differences = [b - a for a, b in zip(records["runtime"], records["native_reuse"])]
    print(
        "REUSE_OUTER_MATCHED "
        + json.dumps(
            {
                "batch": 1,
                "prefill_len": 2048,
                "windows": 31,
                "replays_per_window": 256,
                "ms": records,
                "median_ms": {name: statistics.median(values) for name, values in records.items()},
                "paired_native_minus_runtime_ms": differences,
                "median_paired_difference_ms": statistics.median(differences),
                "eager_pcc": H.pcc(outputs["runtime"], outputs["native_reuse"]),
                "stress_pcc": H.pcc(stress["runtime"], stress["native_reuse"]),
            }
        )
    )


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_paired_fusion(mesh_device, monkeypatch, layer_idx):
    outputs, timings = {}, {}
    prompt = H.make_activations(1, 2048, seed=71)
    token = H.make_activations(1, 1, seed=82)
    for name, cls in (("functional", FunctionalDecoder), ("fused", selected_candidate())):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real")
        x = H.to_device(mesh_device, prompt)
        d = H.to_device(mesh_device, token)
        pos, rot = H.decode_inputs(mesh_device, torch.tensor([2048]))
        prefill_times = []
        for iteration in range(7):
            decoder.reset_state()
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            out = decoder.prefill_forward(x, page_table=table)
            ttnn.synchronize_device(mesh_device)
            duration = (time.perf_counter() - start) * 1000
            if iteration >= 2:
                prefill_times.append(duration)
            if iteration == 6:
                outputs[name] = {"prefill": ttnn.to_torch(out)}
            ttnn.deallocate(out)
        saved = H._snapshot_state(decoder)

        def forward():
            return decoder.decode_forward(d, current_pos=pos, rot_idxs=rot, page_table=table)

        out = forward()
        outputs[name]["decode"] = ttnn.to_torch(out)
        ttnn.deallocate(out)
        H._restore_state(decoder, saved)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        decode_times = []
        for iteration in range(7):
            H._restore_state(decoder, saved)
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            duration = (time.perf_counter() - start) * 1000 / 32
            if iteration >= 2:
                decode_times.append(duration)
        outputs[name]["stress"] = ttnn.to_torch(out)
        H._restore_state(decoder, saved)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        outputs[name]["trace"] = ttnn.to_torch(out)
        outputs[name]["state"] = H._snapshot_state(decoder)
        if hasattr(decoder, "cache_permutation"):
            inverse = torch.argsort(torch.tensor(decoder.cache_permutation))
            outputs[name]["state"]["k"] = outputs[name]["state"]["k"][..., inverse]
        ttnn.release_trace(mesh_device, trace)
        timings[name] = {
            "prefill_ms": prefill_times,
            "decode_ms": decode_times,
            "prefill_median_ms": statistics.median(prefill_times),
            "decode_median_ms": statistics.median(decode_times),
        }
    correlations = {}
    for mode in ("prefill", "decode", "trace", "stress"):
        correlations[mode] = H.pcc(outputs["functional"][mode], outputs["fused"][mode])
        assert correlations[mode] >= H.PCC_BAR
    for name in outputs:
        assert torch.equal(outputs[name]["decode"], outputs[name]["trace"])
    for key, expected in outputs["functional"]["state"].items():
        actual = outputs["fused"]["state"][key]
        if isinstance(expected, list):
            for a, b in zip(expected, actual):
                assert H.pcc(a, b) >= H.PCC_BAR
        else:
            assert H.pcc(expected, actual) >= H.PCC_BAR
    print("FUSION_PAIR " + json.dumps({"layer": layer_idx, "pcc": correlations, "timings": timings}))


@pytest.mark.parametrize("seq_len", [1, 2, 3, 127, 2047, 2049])
def test_linear_core_equivalence(mesh_device, monkeypatch, seq_len):
    decoders = []
    for cls in (FunctionalDecoder, selected_candidate()):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, _, _ = H.build_decoder(mesh_device, 0, "real")
            decoders.append(decoder)
    results = L.probe_linear_core_equivalence(
        *decoders,
        H.make_activations(1, seq_len, seed=713),
        [H.make_activations(1, 1, seed=1800 + i) for i in range(32)],
    )
    print("LINEAR_CORE_PROBE " + json.dumps({"seq_len": seq_len, **results}))


def test_softplus_localization(mesh_device, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(H, "FunctionalDecoder", L.ArithmeticGDN)
        decoder, _, _ = H.build_decoder(mesh_device, 0, "real")
    results = {}
    for seq in (1, 2048):
        x = H.to_device(mesh_device, H.make_activations(1, seq, seed=713))
        norm = decoder._norm(x, decoder.w["attn_norm"])
        projected = decoder._gdn_project(norm)
        results[str(seq)] = L.probe_softplus_bias_fusion(decoder, projected[2])
        for tensor in (x, norm, *projected):
            ttnn.deallocate(tensor)
    bias = ttnn.to_torch(decoder.w["dt_bias"]).reshape(1, 1, 32).float()
    desired = torch.linspace(-12, 12, 33).reshape(1, 33, 1).expand(1, 33, 32)
    control = H.to_device(mesh_device, (desired - bias).bfloat16())
    results["boundary_control"] = L.probe_softplus_bias_fusion(decoder, control)
    ttnn.deallocate(control)
    print("SOFTPLUS_LOCALIZATION " + json.dumps(results))


def test_convolution_localization(mesh_device, monkeypatch):
    decoders = []
    for cls in (FunctionalDecoder, L.KDAConvGDN):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, _, _ = H.build_decoder(mesh_device, 0, "real", batch=32, max_context=1024)
        decoders.append(decoder)
    prompt = H.make_activations(32, 63, seed=31)
    result = L.probe_linear_conv_equivalence(*decoders, prompt)
    print("CONV_LOCALIZATION " + json.dumps(result))
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    result = L.probe_linear_conv_equivalence(*decoders, prompt, compute_kernel_config=config)
    print("CONV_FP32_DEST_FALSE " + json.dumps(result))


def test_short_prefill_norm_handoff(mesh_device, monkeypatch):
    records, outputs = {}, {}
    prompt = H.make_activations(1, 128, seed=71)
    for name, cls in (("l1_then_dram", D.ModeNormGDN), ("direct_dram", F.NormDRAMGDN), ("l1_input", F.NormL1InputGDN)):
        with monkeypatch.context() as patch:
            patch.setattr(H, "FunctionalDecoder", cls)
            decoder, table, _ = H.build_decoder(mesh_device, 0, "real")
        x = H.to_device(mesh_device, prompt)
        durations = []
        for iteration in range(30):
            decoder.reset_state()
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            out = decoder.prefill_forward(x, page_table=table)
            ttnn.synchronize_device(mesh_device)
            if iteration >= 5:
                durations.append((time.perf_counter() - start) * 1000)
            if iteration == 29:
                outputs[name] = ttnn.to_torch(out)
            ttnn.deallocate(out)
        records[name] = {"ms": durations, "median_ms": statistics.median(durations)}
    assert torch.equal(outputs["l1_then_dram"], outputs["direct_dram"])
    assert torch.equal(outputs["l1_then_dram"], outputs["l1_input"])
    print("SHORT_PREFILL_HANDOFF " + json.dumps(records))
