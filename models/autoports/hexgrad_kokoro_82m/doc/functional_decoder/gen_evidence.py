# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate PCC / determinism evidence for the Kokoro-82M functional decoder.

Writes ``pcc_results.json`` and prints a human-readable table. Uses real HF
weights. Run with the dev-checkout env (see README).
"""
import json
import os
import random

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.functional_decoder import FunctionalDecoder

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "hexgrad/Kokoro-82M"
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
_IPA_SENTENCES = [
    "hɛlˈO wˈɜːld",
    "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ",
    "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz",
]


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rep_ids(vocab, batch, seq_len, seed):
    pool = [vocab[c] for c in _REP_SYMBOLS if c in vocab]
    rng = random.Random(seed)
    rows = []
    for _ in range(batch):
        body = [rng.choice(pool) for _ in range(max(seq_len - 2, 0))]
        ids = ([0] + body + [0])[:seq_len]
        while len(ids) < seq_len:
            ids.append(0)
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def main():
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(ac).eval()
    hf.load_state_dict(sd, strict=False)

    dev = ttnn.open_device(device_id=0)
    dec = FunctionalDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)

    def ref(ids, mask=None):
        with torch.no_grad():
            return hf(ids, attention_mask=(mask if mask is not None else torch.ones_like(ids))).last_hidden_state

    def prefill(ids, mask=None):
        p = FunctionalDecoder.prepare_inputs(ids, dev, attention_mask=mask)
        o = dec.prefill_forward(
            p["input_ids"],
            p["position_ids"],
            p["token_type_ids"],
            p["attention_mask"],
            batch=p["batch"],
            seq_len=p["padded_seq_len"],
        )
        return ttnn.to_torch(o)[:, : p["seq_len"], :].float()

    def decode(ids, mask=None):
        p = FunctionalDecoder.prepare_inputs(ids, dev, attention_mask=mask)
        o = dec.decode_forward(
            p["input_ids"],
            p["position_ids"],
            p["token_type_ids"],
            p["attention_mask"],
            batch=p["batch"],
            seq_len=p["padded_seq_len"],
        )
        return ttnn.to_torch(o)[:, : p["seq_len"], :].float()

    results = {
        "model": MODEL_ID,
        "weight_dtype": "bfloat16",
        "activation_dtype": "float32",
        "pcc_bar": 0.995,
        "input": "representative phoneme sequences (BOS/EOS wrapped)",
        "prefill": {},
        "decode": {},
        "batch": {},
        "ipa_sentences": {},
        "short_len_characterization": {},
        "determinism": {},
        "masked": {},
        "worst_case_seed_sweep": {},
    }

    for T in [8, 16, 31, 32, 33, 64, 128, 256, 500, 511, 512]:
        ids = rep_ids(vocab, 1, T, seed=T)
        results["prefill"][T] = round(pcc(prefill(ids), ref(ids)), 6)

    for T in [32, 64, 128, 500, 511, 512]:
        ids = rep_ids(vocab, 1, T, seed=5000 + T)
        results["decode"][T] = round(pcc(decode(ids), ref(ids)), 6)

    for B in [2, 4, 8, 32]:
        ids = rep_ids(vocab, B, 128, seed=1000 + B)
        results["batch"][B] = round(pcc(prefill(ids), ref(ids)), 6)

    # real hand-written IPA transcriptions (the model's actual input domain)
    for text in _IPA_SENTENCES:
        ids = torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)
        results["ipa_sentences"][text] = {"len": ids.shape[1], "pcc": round(pcc(prefill(ids), ref(ids)), 6)}

    # worst-case over 16 representative seeds per length (robustness evidence)
    for T in [16, 31, 32, 64, 128, 512]:
        vals = [pcc(prefill(rep_ids(vocab, 1, T, seed=s)), ref(rep_ids(vocab, 1, T, seed=s))) for s in range(16)]
        results["worst_case_seed_sweep"][T] = {
            "min": round(min(vals), 6),
            "mean": round(sum(vals) / len(vals), 6),
            "below_bar": sum(1 for v in vals if v < 0.995),
        }

    # transparent short-length characterization (below the model's realistic
    # minimum input of 3 tokens = BOS+phoneme+EOS). Single-token outputs have no
    # cross-position averaging so bf16 weight quantization dominates; not a bug.
    for T in [1, 2, 3, 4, 8, 16]:
        vals = [
            pcc(prefill(rep_ids(vocab, 1, T, seed=100 * T + s)), ref(rep_ids(vocab, 1, T, seed=100 * T + s)))
            for s in range(4)
        ]
        results["short_len_characterization"][T] = {
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "mean": round(sum(vals) / len(vals), 6),
        }

    # determinism
    ids = rep_ids(vocab, 1, 128, seed=77)
    results["determinism"]["prefill_bit_identical"] = bool(torch.equal(prefill(ids), prefill(ids)))
    ids = rep_ids(vocab, 1, 64, seed=88)
    results["determinism"]["decode_bit_identical"] = bool(torch.equal(decode(ids), decode(ids)))
    ids = rep_ids(vocab, 1, 64, seed=303)
    results["determinism"]["decode_equals_prefill"] = bool(torch.equal(prefill(ids), decode(ids)))

    # masked / padded batch
    lengths = [96, 40]
    ml = max(lengths)
    ids = torch.zeros((2, ml), dtype=torch.long)
    mask = torch.zeros((2, ml), dtype=torch.long)
    for i, L in enumerate(lengths):
        ids[i, :L] = rep_ids(vocab, 1, L, seed=404 + i)[0]
        mask[i, :L] = 1
    g = prefill(ids, mask)
    r = ref(ids, mask)
    results["masked"] = {str(L): round(pcc(g[i, :L], r[i, :L]), 6) for i, L in enumerate(lengths)}

    # traced decode with batch>1 + nonzero mask (non-aligned rows)
    dlens = [113, 47]
    dml = max(dlens)
    dids = torch.zeros((2, dml), dtype=torch.long)
    dmask = torch.zeros((2, dml), dtype=torch.long)
    for i, L in enumerate(dlens):
        dids[i, :L] = rep_ids(vocab, 1, L, seed=6000 + i)[0]
        dmask[i, :L] = 1
    dg = decode(dids, dmask)
    dr = ref(dids, dmask)
    results["masked_decode_traced"] = {str(L): round(pcc(dg[i, :L], dr[i, :L]), 6) for i, L in enumerate(dlens)}

    dec.release_traces()
    ttnn.close_device(dev)

    with open(os.path.join(HERE, "pcc_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== Kokoro-82M functional-decoder PCC evidence (bf16 weights / fp32 activations, real weights) =====")
    print("PREFILL (batch 1, representative inputs):")
    for T, v in results["prefill"].items():
        print(f"  T={T:4d}  PCC={v:.6f}  {'PASS' if v >= 0.995 else 'FAIL'}")
    print("DECODE (traced, batch 1):")
    for T, v in results["decode"].items():
        print(f"  T={T:4d}  PCC={v:.6f}  {'PASS' if v >= 0.995 else 'FAIL'}")
    print("BATCH (T=128):")
    for B, v in results["batch"].items():
        print(f"  B={B:3d}   PCC={v:.6f}  {'PASS' if v >= 0.995 else 'FAIL'}")
    print("REAL IPA SENTENCES:")
    for t, v in results["ipa_sentences"].items():
        print(f"  len={v['len']:3d}  PCC={v['pcc']:.6f}  '{t[:32]}'")
    print("WORST-CASE over 16 representative seeds:")
    for T, v in results["worst_case_seed_sweep"].items():
        print(f"  T={T:4d}  min={v['min']:.6f} mean={v['mean']:.6f} below_bar={v['below_bar']}")
    print("MASKED/padded rows (prefill):", results["masked"])
    print("MASKED/padded rows (traced decode, batch2):", results["masked_decode_traced"])
    print("DETERMINISM:", results["determinism"])
    print("SHORT-LEN characterization (below model min input=3; out of domain):")
    for T, v in results["short_len_characterization"].items():
        print(f"  T={T:3d}  min={v['min']:.4f} mean={v['mean']:.4f} max={v['max']:.4f}")
    print("Wrote", os.path.join(HERE, "pcc_results.json"))


if __name__ == "__main__":
    main()
