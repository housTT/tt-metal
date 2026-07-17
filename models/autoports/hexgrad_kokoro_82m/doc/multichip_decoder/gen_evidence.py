# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multichip-decoder evidence for Kokoro-82M: PCC (vs single-chip TTNN + HF) and
warmed single-chip-baseline vs multi-chip latency / speedup / efficiency.

Runs in one process, two device phases (to keep single-chip and multi-chip
numbers fresh and comparable):
  Phase A: (1,4) mesh + FABRIC_1D  -> MultichipDecoder + replicated single-chip
           OptimizedDecoder (device-0 copy) for PCC isolation.
  Phase B: single device (no fabric) -> OptimizedDecoder warmed/traced baseline.

Writes doc/multichip_decoder/pcc_results.json and perf_summary.json.
"""
import json
import os
import random
import time

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.multichip_decoder import MultichipDecoder
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = os.path.dirname(__file__)
MESH_SHAPE = (1, 4)
_SYM = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
_IPA = ["hɛlˈO wˈɜːld", "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ", "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz"]
ITERS = 50


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5)


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

    def ipa(text):
        return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)

    def href(ids, m=None):
        with torch.no_grad():
            return hf(ids, attention_mask=torch.ones_like(ids) if m is None else m).last_hidden_state

    # ============================ Phase A: mesh ============================
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), trace_region_size=90000000)
    tp = mesh.get_num_devices()
    mc_perf = {}
    try:
        mc = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
        sc_rep = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)

        def mc_prefill(ids, m=None):
            p = mc.prepare_inputs(ids, attention_mask=m)
            o = mc.prefill_forward(
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
            p = mc.prepare_inputs(ids, attention_mask=m)
            o = mc.decode_forward(
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
            for L in [8, 16, 31, 32, 33, 64, 128, 256, 500, 511, 512]
        }
        deco = {
            str(L): {
                "vs_sc": _pcc(mc_decode(rep(1, L, 5000 + L)), sc_prefill(rep(1, L, 5000 + L))),
                "vs_hf": _pcc(mc_decode(rep(1, L, 5000 + L)), href(rep(1, L, 5000 + L))),
            }
            for L in [32, 64, 128, 500, 511, 512]
        }
        ipa_pcc = {t[:16]: {"vs_hf": _pcc(mc_prefill(ipa(t)), href(ipa(t)))} for t in _IPA}
        batch_pcc = {
            str(b): {
                "vs_sc": _pcc(mc_prefill(rep(b, 128, 1000 + b)), sc_prefill(rep(b, 128, 1000 + b))),
                "vs_hf": _pcc(mc_prefill(rep(b, 128, 1000 + b)), href(rep(b, 128, 1000 + b))),
            }
            for b in [2, 4, 8, 32]
        }
        mc.release_traces()
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
            "pcc_bar_vs_single_chip": 0.998,
            "prefill_pcc": pref,
            "decode_traced_pcc": deco,
            "ipa_sentence_pcc": ipa_pcc,
            "batch_prefill_pcc": batch_pcc,
            "worst_pcc_vs_single_chip": worst_sc,
            "worst_pcc_vs_hf": worst_hf,
            "note": "Multichip TP=4 (head-parallel attention + sequence-parallel FFN, sequence-sharded residual) vs single-chip TTNN OptimizedDecoder baseline (isolates sharding/collective) and vs HF (end-to-end).",
        }
        with open(os.path.join(HERE, "pcc_results.json"), "w") as f:
            json.dump(pcc_results, f, indent=2)
        print("WORST PCC vs single-chip:", worst_sc, "vs HF:", worst_hf)
        # Enforce the bars here too (not just record them): the gated test suite
        # covers the same lengths, but this makes the evidence generator itself fail
        # loudly if the tightest length regresses.
        assert worst_hf >= 0.995, f"worst PCC vs HF {worst_hf} < 0.995"
        assert worst_sc >= 0.997, f"worst PCC vs single-chip {worst_sc} < 0.997"

        # multichip latency
        mc2 = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
        for L in [128, 512]:
            ids = rep(1, L, 7)
            p = mc2.prepare_inputs(ids)

            def rpf():
                return mc2.prefill_forward(
                    p["input_ids"],
                    p["position_ids"],
                    p["token_type_ids"],
                    p["attention_mask"],
                    batch=p["batch"],
                    seq_len=p["padded_seq_len"],
                )

            o = rpf()
            ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                o = rpf()
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            pf = (time.perf_counter() - t0) / ITERS * 1e3
            mc2.decode_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            ttnn.synchronize_device(mesh)
            rec = mc2._traces[(1, L, p["attention_mask"] is not None)]
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh, rec["trace_id"], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            dc = (time.perf_counter() - t0) / ITERS * 1e3
            mc_perf[str(L)] = {"prefill_warmed_ms": round(pf, 4), "decode_traced_ms": round(dc, 4)}
        mc2.release_traces()
        print("MC PERF", mc_perf)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # ============================ Phase B: single chip ============================
    dev = ttnn.open_device(device_id=0)
    sc_perf = {}
    try:
        sc = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)
        for L in [128, 512]:
            ids = rep(1, L, 7)
            p = OptimizedDecoder.prepare_inputs(ids, dev)

            def rpf():
                return sc.prefill_forward(
                    p["input_ids"],
                    p["position_ids"],
                    p["token_type_ids"],
                    p["attention_mask"],
                    batch=p["batch"],
                    seq_len=p["padded_seq_len"],
                )

            o = rpf()
            ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                o = rpf()
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            pf = (time.perf_counter() - t0) / ITERS * 1e3
            sc.decode_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            ttnn.synchronize_device(dev)
            rec = sc._traces[(1, L, p["attention_mask"] is not None)]
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(dev, rec["trace_id"], cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dc = (time.perf_counter() - t0) / ITERS * 1e3
            sc_perf[str(L)] = {"prefill_warmed_ms": round(pf, 4), "decode_traced_ms": round(dc, 4)}
        sc.release_traces()
    finally:
        ttnn.close_device(dev)

    # ============================ speedup / efficiency ============================
    perf = {}
    for L in ["128", "512"]:
        sp_dec = round(sc_perf[L]["decode_traced_ms"] / mc_perf[L]["decode_traced_ms"], 3)
        sp_pf = round(sc_perf[L]["prefill_warmed_ms"] / mc_perf[L]["prefill_warmed_ms"], 3)
        perf[L] = {
            "single_chip": sc_perf[L],
            "multichip_tp4": mc_perf[L],
            "decode_speedup": sp_dec,
            "decode_efficiency": round(sp_dec / 4, 3),
            "prefill_speedup": sp_pf,
            "prefill_efficiency": round(sp_pf / 4, 3),
        }
    summary = {
        "model": MODEL_ID,
        "mesh_shape": list(MESH_SHAPE),
        "tp": 4,
        "cluster": "P300_X2 (4x Blackhole p300c, physical 4-ring, FABRIC_1D_RING/Topology.Ring, 2 links)",
        "topology": str(getattr(ttnn.Topology, "Ring", "Ring")),
        "iters": ITERS,
        "collectives_per_layer": {"all_gather": 1, "reduce_scatter": 1},
        "weight_dtype": "bfloat8_b (attn fractured, FFN replicated)",
        "activation_dtype": "bfloat16",
        "results": perf,
    }
    with open(os.path.join(HERE, "perf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
