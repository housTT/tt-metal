# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Optimized-multichip-decoder evidence for Kokoro-82M.

Two device phases in one process:
  Phase A: (1,4) ring mesh -> OptimizedMultichipDecoder (default = sharded norm +
           persistent CCL), the stage-03 multichip BASELINE (same class, opts off),
           and a replicated single-chip OptimizedDecoder for vs-single-chip PCC.
           Emits full PCC sweep + before/after (baseline vs optimized) multichip
           latency in the SAME session, plus the single-chip context number.
  Phase B: single device -> OptimizedDecoder warmed/traced baseline for the
           single-chip-vs-multichip speedup context.

Writes pcc_results.json and perf_summary.json.
"""
import json
import os
import random
import time

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder
from models.autoports.hexgrad_kokoro_82m.tt.optimized_multichip_decoder import OptConfig, OptimizedMultichipDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = os.path.dirname(__file__)
MESH_SHAPE = (1, 4)
_SYM = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
_IPA = ["hɛlˈO wˈɜːld", "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ", "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz"]
ITERS = 50
# stage-03 multichip behaviour: all optimizations off
BASELINE_OPT = OptConfig(norm_sharded=False, fuse_residual=False, persistent_ccl=False)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5)


def _traced_decode_ms(dec, p, mesh):
    dec.decode_forward(
        p["input_ids"],
        p["position_ids"],
        p["token_type_ids"],
        p["attention_mask"],
        batch=p["batch"],
        seq_len=p["padded_seq_len"],
    )
    ttnn.synchronize_device(mesh)
    rec = dec._traces[(p["batch"], p["padded_seq_len"], p["attention_mask"] is not None)]
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, rec["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return round((time.perf_counter() - t0) / ITERS * 1e3, 4)


def _warmed_prefill_ms(dec, p, mesh):
    def run():
        return dec.prefill_forward(
            p["input_ids"],
            p["position_ids"],
            p["token_type_ids"],
            p["attention_mask"],
            batch=p["batch"],
            seq_len=p["padded_seq_len"],
        )

    o = run()
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        o = run()
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    return round((time.perf_counter() - t0) / ITERS * 1e3, 4)


def main():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig, AlbertModel

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(ac).eval()
    hf.load_state_dict(sd, strict=False)
    vocab = cfg["vocab"]
    pool = [vocab[c] for c in _SYM if c in vocab]
    H = ac.hidden_size

    def rep(b, L, seed):
        rng = random.Random(seed)
        rows = []
        for _ in range(b):
            body = [rng.choice(pool) for _ in range(max(L - 2, 0))]
            ids = ([0] + body + [0])[:L]
            while len(ids) < L:
                ids.append(0)
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    def ipa(t):
        return torch.tensor([[0] + [vocab[c] for c in t if c in vocab] + [0]], dtype=torch.long)

    def href(ids, m=None):
        with torch.no_grad():
            return hf(ids, attention_mask=torch.ones_like(ids) if m is None else m).last_hidden_state

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    tp = mesh.get_num_devices()
    mc_perf, before_after = {}, {}
    try:
        opt = OptimizedMultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)  # optimized default
        sc_rep = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)

        def mc_prefill(ids, m=None):
            p = opt.prepare_inputs(ids, attention_mask=m)
            o = opt.prefill_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            return (
                ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
                .reshape(ids.shape[0], p["padded_seq_len"], H)[:, : p["seq_len"], :]
                .float()
            )

        def mc_decode(ids, m=None):
            p = opt.prepare_inputs(ids, attention_mask=m)
            o = opt.decode_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            return (
                ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
                .reshape(ids.shape[0], p["padded_seq_len"], H)[:, : p["seq_len"], :]
                .float()
            )

        def sc_prefill(ids, m=None):
            p = OptimizedDecoder.prepare_inputs(ids, mesh, attention_mask=m)
            o = sc_rep.prefill_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            return ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[
                : ids.shape[0], : p["seq_len"], :
            ].float()

        pref = {
            str(L): {
                "vs_sc": _pcc(mc_prefill(rep(1, L, L)), sc_prefill(rep(1, L, L))),
                "vs_hf": _pcc(mc_prefill(rep(1, L, L)), href(rep(1, L, L))),
            }
            for L in [8, 16, 31, 32, 33, 64, 96, 128, 256, 500, 511, 512]
        }
        deco = {
            str(L): {
                "vs_sc": _pcc(mc_decode(rep(1, L, 5000 + L)), sc_prefill(rep(1, L, 5000 + L))),
                "vs_hf": _pcc(mc_decode(rep(1, L, 5000 + L)), href(rep(1, L, 5000 + L))),
            }
            for L in [16, 32, 64, 128, 500, 511, 512]
        }
        ipa_pcc = {t[:16]: {"vs_hf": _pcc(mc_prefill(ipa(t)), href(ipa(t)))} for t in _IPA}
        batch_pcc = {
            str(b): {
                "vs_sc": _pcc(mc_prefill(rep(b, 128, 1000 + b)), sc_prefill(rep(b, 128, 1000 + b))),
                "vs_hf": _pcc(mc_prefill(rep(b, 128, 1000 + b)), href(rep(b, 128, 1000 + b))),
            }
            for b in [2, 4, 8, 32]
        }
        opt.release_traces()
        sc_rep.release_traces()

        worst_sc = min(
            [
                *[v["vs_sc"] for v in pref.values()],
                *[v["vs_sc"] for v in deco.values()],
                *[v["vs_sc"] for v in batch_pcc.values()],
            ]
        )
        worst_hf = min(
            [
                *[v["vs_hf"] for v in pref.values()],
                *[v["vs_hf"] for v in deco.values()],
                *[v["vs_hf"] for v in ipa_pcc.values()],
                *[v["vs_hf"] for v in batch_pcc.values()],
            ]
        )
        pcc_results = {
            "model": MODEL_ID,
            "mesh_shape": list(MESH_SHAPE),
            "tp": tp,
            "pcc_bar_hf": 0.995,
            "pcc_bar_vs_single_chip": 0.997,
            "prefill_pcc": pref,
            "decode_traced_pcc": deco,
            "ipa_sentence_pcc": ipa_pcc,
            "batch_prefill_pcc": batch_pcc,
            "worst_pcc_vs_single_chip": worst_sc,
            "worst_pcc_vs_hf": worst_hf,
            "note": "Optimized multichip TP=4 (sharded L1 LayerNorm + persistent CCL buffers) vs single-chip TTNN OptimizedDecoder and vs HF.",
        }
        with open(os.path.join(HERE, "pcc_results.json"), "w") as f:
            json.dump(pcc_results, f, indent=2)
        print("WORST PCC vs single-chip:", worst_sc, "vs HF:", worst_hf)
        assert worst_hf >= 0.995, f"worst PCC vs HF {worst_hf} < 0.995"
        assert worst_sc >= 0.997, f"worst PCC vs single-chip {worst_sc} < 0.997"

        # before/after: stage-03 multichip baseline vs optimized (same session)
        base = OptimizedMultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh, opt=BASELINE_OPT)
        opt2 = OptimizedMultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
        for L in [128, 512]:
            ids = rep(1, L, 7)
            pb, po = base.prepare_inputs(ids), opt2.prepare_inputs(ids)
            before_after[str(L)] = {
                "baseline_multichip": {
                    "prefill_warmed_ms": _warmed_prefill_ms(base, pb, mesh),
                    "decode_traced_ms": _traced_decode_ms(base, pb, mesh),
                },
                "optimized_multichip": {
                    "prefill_warmed_ms": _warmed_prefill_ms(opt2, po, mesh),
                    "decode_traced_ms": _traced_decode_ms(opt2, po, mesh),
                },
            }
            mc_perf[str(L)] = before_after[str(L)]["optimized_multichip"]
        base.release_traces()
        opt2.release_traces()
        print("BEFORE/AFTER", json.dumps(before_after, indent=2))
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # Phase B: single-chip context
    dev = ttnn.open_device(device_id=0)
    sc_perf = {}
    try:
        sc = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)
        for L in [128, 512]:
            ids = rep(1, L, 7)
            p = OptimizedDecoder.prepare_inputs(ids, dev)
            sc_perf[str(L)] = {
                "prefill_warmed_ms": _warmed_prefill_ms(sc, p, dev),
                "decode_traced_ms": _traced_decode_ms(sc, p, dev),
            }
        sc.release_traces()
    finally:
        ttnn.close_device(dev)

    results = {}
    for L in ["128", "512"]:
        b = before_after[L]["baseline_multichip"]
        o = before_after[L]["optimized_multichip"]
        results[L] = {
            "single_chip": sc_perf[L],
            "baseline_multichip_tp4": b,
            "optimized_multichip_tp4": o,
            "opt_decode_speedup_vs_baseline_mc": round(b["decode_traced_ms"] / o["decode_traced_ms"], 3),
            "opt_prefill_speedup_vs_baseline_mc": round(b["prefill_warmed_ms"] / o["prefill_warmed_ms"], 3),
            "decode_speedup_vs_single_chip": round(sc_perf[L]["decode_traced_ms"] / o["decode_traced_ms"], 3),
            "decode_efficiency_vs_single_chip": round(sc_perf[L]["decode_traced_ms"] / o["decode_traced_ms"] / 4, 3),
        }
    summary = {
        "model": MODEL_ID,
        "mesh_shape": list(MESH_SHAPE),
        "tp": 4,
        "cluster": "P300_X2 (4x Blackhole p300c, physical 4-ring, FABRIC_1D_RING/Topology.Ring, 2 links)",
        "iters": ITERS,
        "optimizations": ["block-sharded L1 LayerNorm", "persistent CCL output/intermediate buffers"],
        "collectives_per_layer": {"all_gather": 1, "reduce_scatter": 1},
        "weight_dtype": "bfloat8_b (attn fractured, FFN replicated)",
        "activation_dtype": "bfloat16",
        "results": results,
    }
    with open(os.path.join(HERE, "perf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
