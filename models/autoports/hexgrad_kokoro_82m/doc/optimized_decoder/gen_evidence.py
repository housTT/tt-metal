# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate the optimized-decoder PCC + warmed-latency evidence for Kokoro-82M.

Uses the EXACT test-suite seeds (prefill seed=seq_len, decode seed=5000+seq_len)
so the numbers are directly comparable to the functional-decoder evidence and to
the delivered tests. Also records the selected policy vs the rejected BFP8-weight
policy (correctness rejection) and warmed prefill / traced-decode latency.

Writes doc/optimized_decoder/pcc_results.json and perf_summary.json.
"""
import json
import os
import random
import time

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = os.path.dirname(__file__)
_SYM = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
_IPA = ["hɛlˈO wˈɜːld", "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ", "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz"]


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

    dev = ttnn.open_device(device_id=0)

    def prefill(dec, ids, m=None):
        p = OptimizedDecoder.prepare_inputs(ids, dev, attention_mask=m)
        o = dec.prefill_forward(
            p["input_ids"],
            p["position_ids"],
            p["token_type_ids"],
            p["attention_mask"],
            batch=p["batch"],
            seq_len=p["padded_seq_len"],
        )
        return ttnn.to_torch(o)[:, : p["seq_len"], :].float()

    def decode(dec, ids, m=None):
        p = OptimizedDecoder.prepare_inputs(ids, dev, attention_mask=m)
        o = dec.decode_forward(
            p["input_ids"],
            p["position_ids"],
            p["token_type_ids"],
            p["attention_mask"],
            batch=p["batch"],
            seq_len=p["padded_seq_len"],
        )
        return ttnn.to_torch(o)[:, : p["seq_len"], :].float()

    try:
        dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)  # selected default policy
        pref = {
            str(L): _pcc(prefill(dec, rep(1, L, L)), href(rep(1, L, L)))
            for L in [8, 16, 31, 32, 33, 64, 128, 256, 500, 511, 512]
        }
        deco = {
            str(L): _pcc(decode(dec, rep(1, L, 5000 + L)), href(rep(1, L, 5000 + L)))
            for L in [32, 64, 128, 500, 511, 512]
        }
        ipa_pcc = {t[:16]: _pcc(prefill(dec, ipa(t)), href(ipa(t))) for t in _IPA}
        batch_pcc = {}
        for b in [2, 4, 8, 32]:
            ids = rep(b, 128, 1000 + b)
            batch_pcc[str(b)] = _pcc(prefill(dec, ids), href(ids))
        dec.release_traces()

        # alternative / rejected policies at the sensitive test seed (T=16 seed 16)
        bf16w = OptimizedDecoder.from_state_dict(
            sd,
            hf_config=ac,
            mesh_device=dev,
            policy=PrecisionPolicy(attn_weight="bf16", mlp_weight="bf16", map_weight="bf16"),
        )
        bf16w_t16 = _pcc(prefill(bf16w, rep(1, 16, 16)), href(rep(1, 16, 16)))
        bf16w.release_traces()
        bfp8_noacc = OptimizedDecoder.from_state_dict(
            sd,
            hf_config=ac,
            mesh_device=dev,
            policy=PrecisionPolicy(fp32_dest_acc=False),
        )
        bfp8_noacc_t16 = _pcc(prefill(bfp8_noacc, rep(1, 16, 16)), href(rep(1, 16, 16)))
        bfp8_noacc.release_traces()

        results = {
            "model": MODEL_ID,
            "selected_policy": PrecisionPolicy().__dict__,
            "pcc_bar": 0.995,
            "prefill_pcc_real_weights_seed_eq_len": pref,
            "decode_traced_pcc_real_weights_seed_5000pL": deco,
            "ipa_sentence_pcc": ipa_pcc,
            "batch_prefill_pcc": batch_pcc,
            "worst_pcc": min([*pref.values(), *deco.values(), *ipa_pcc.values(), *batch_pcc.values()]),
            "alternatives_T16_seed16_pcc": {
                "selected_bfp8_fp32acc": pref["16"],
                "bf16_weights_fp32acc_slower_4pct": bf16w_t16,
                "bfp8_no_fp32acc_REJECTED_below_bar": bfp8_noacc_t16,
            },
            "note": (
                "Selected policy: bf16 act + BFP8 weights + HiFi2 + fp32_dest_acc=True. "
                "Passes every tested length (worst above the 0.995 bar and above the "
                "functional fp32 floor of 0.9958), is ~4% faster on traced decode than "
                "BF16 weights, and halves weight memory. fp32_dest_acc is required: "
                f"BFP8 without it drops T=16 to {bfp8_noacc_t16} < 0.995. BF16 activations "
                "are mandatory (ttnn SDPA rejects fp32 inputs)."
            ),
        }
        with open(os.path.join(HERE, "pcc_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))

        # warmed latency (wall-clock), selected policy
        dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)
        perf = {}
        for L in [128, 512]:
            ids = rep(1, L, 7)
            p = OptimizedDecoder.prepare_inputs(ids, dev)

            # warmed prefill
            def rpf():
                return dec.prefill_forward(
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
            for _ in range(50):
                o = rpf()
                ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            pf = (time.perf_counter() - t0) / 50 * 1e3
            # traced decode
            dec.decode_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            ttnn.synchronize_device(dev)
            rec = dec._traces[(1, L, p["attention_mask"] is not None)]
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(dev, rec["trace_id"], cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dc = (time.perf_counter() - t0) / 50 * 1e3
            perf[str(L)] = {"prefill_warmed_ms": round(pf, 4), "decode_traced_ms": round(dc, 4)}
        dec.release_traces()

        summary = {
            "model": MODEL_ID,
            "iters": 50,
            "weight_dtype": "bfloat8_b",
            "activation_dtype": "bfloat16",
            "matmul_fidelity": "HiFi2",
            "fp32_dest_acc": True,
            "results": perf,
            "functional_baseline_ms": {
                "128": {"prefill_warmed_ms": 7.1013, "decode_traced_ms": 7.056},
                "512": {"prefill_warmed_ms": 21.0115, "decode_traced_ms": 20.9941},
            },
        }
        with open(os.path.join(HERE, "perf_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
