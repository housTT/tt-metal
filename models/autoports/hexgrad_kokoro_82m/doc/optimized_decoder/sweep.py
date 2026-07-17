# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Precision/geometry sweep harness for the Kokoro-82M optimized decoder.

Loads real HF weights once, builds HF references once, then for each requested
PrecisionPolicy builds an OptimizedDecoder and reports:
  * prefill PCC over a set of representative lengths (incl. short + non-aligned),
  * traced-decode latency (warmed, execute_trace-only) at T=512 and T=128.

Usage (dev-checkout ttnn env):
  python models/autoports/hexgrad_kokoro_82m/doc/optimized_decoder/sweep.py <preset>
where <preset> is one of: precision, geometry, final  (see PRESETS below).
Results are written to doc/optimized_decoder/sweeps/<preset>.json
"""
import json
import os
import random
import sys
import time

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import OptimizedDecoder, PrecisionPolicy

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "sweeps")
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_common():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig, AlbertModel

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(ac).eval()
    hf.load_state_dict(sd, strict=False)
    vocab = cfg["vocab"]
    return ac, sd, hf, vocab


def _rep_ids(vocab, seq_len, seed):
    pool = [vocab[c] for c in _REP_SYMBOLS if c in vocab]
    rng = random.Random(seed)
    body = [rng.choice(pool) for _ in range(max(seq_len - 2, 0))]
    ids = ([0] + body + [0])[:seq_len]
    while len(ids) < seq_len:
        ids.append(0)
    return torch.tensor([ids], dtype=torch.long)


def _hf_ref(hf, ids):
    with torch.no_grad():
        return hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state


PCC_LENS = [16, 32, 33, 64, 128, 500, 511, 512]


def _run_prefill(dec, dev, ids):
    prep = OptimizedDecoder.prepare_inputs(ids, dev)
    out = dec.prefill_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    r = ttnn.to_torch(out)[:, : prep["seq_len"], :].float()
    ttnn.deallocate(out)
    return r


def _time_decode(dec, dev, ids, iters=50):
    prep = OptimizedDecoder.prepare_inputs(ids, dev)
    dec.decode_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    ttnn.synchronize_device(dev)
    key = (prep["batch"], prep["padded_seq_len"], prep["attention_mask"] is not None)
    rec = dec._traces[key]
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(dev, rec["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _time_prefill(dec, dev, ids, iters=30):
    prep = OptimizedDecoder.prepare_inputs(ids, dev)

    def run():
        return dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )

    out = run()
    ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = run()
        ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e3


def evaluate(policy, ac, sd, hf, vocab, dev, refs, do_perf=True):
    dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev, policy=policy)
    pccs = {}
    worst = 1.0
    for L in PCC_LENS:
        ids = _rep_ids(vocab, L, seed=5000 + L)
        got = _run_prefill(dec, dev, ids)
        pcc = _pcc(got, refs[L])
        pccs[str(L)] = round(pcc, 5)
        worst = min(worst, pcc)
    res = {
        "policy": policy.label(),
        "policy_fields": policy.__dict__.copy(),
        "prefill_pcc": pccs,
        "worst_pcc": round(worst, 5),
    }
    if do_perf:
        res["decode_ms_512"] = round(_time_decode(dec, dev, _rep_ids(vocab, 512, 7)), 4)
        res["decode_ms_128"] = round(_time_decode(dec, dev, _rep_ids(vocab, 128, 7)), 4)
        res["prefill_ms_512"] = round(_time_prefill(dec, dev, _rep_ids(vocab, 512, 7)), 4)
    dec.release_traces()
    return res


PRESETS = {
    # activation dtype / weight dtype / fidelity exploration
    "precision": [
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bf16",
            mlp_weight="bf16",
            map_weight="bf16",
            matmul_fidelity="HiFi2",
            sdpa_fidelity="HiFi2",
        ),
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bfp8",
            mlp_weight="bfp8",
            map_weight="bfp8",
            matmul_fidelity="HiFi2",
            sdpa_fidelity="HiFi2",
        ),
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bfp8",
            mlp_weight="bfp8",
            map_weight="bfp8",
            matmul_fidelity="LoFi",
            sdpa_fidelity="HiFi2",
        ),
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bfp8",
            mlp_weight="bfp4",
            map_weight="bfp8",
            matmul_fidelity="LoFi",
            sdpa_fidelity="HiFi2",
        ),
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bfp4",
            mlp_weight="bfp4",
            map_weight="bfp8",
            matmul_fidelity="LoFi",
            sdpa_fidelity="HiFi2",
        ),
        PrecisionPolicy(
            activation="bf16",
            attn_weight="bfp8",
            mlp_weight="bfp4",
            map_weight="bfp8",
            matmul_fidelity="LoFi",
            sdpa_fidelity="LoFi",
        ),
    ],
    # after core_grid tuning: re-confirm selected policy vs bf16-weight control
    "final": [
        PrecisionPolicy(),  # selected: bf16 act / bfp8 weights / HiFi2
        PrecisionPolicy(attn_weight="bf16", mlp_weight="bf16", map_weight="bf16"),  # bf16-weight control
    ],
}


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "precision"
    os.makedirs(OUT, exist_ok=True)
    ac, sd, hf, vocab = _build_common()
    refs = {L: _hf_ref(hf, _rep_ids(vocab, L, seed=5000 + L)) for L in PCC_LENS}
    dev = ttnn.open_device(device_id=0)
    results = []
    try:
        for pol in PRESETS[preset]:
            r = evaluate(pol, ac, sd, hf, vocab, dev, refs)
            results.append(r)
            print(json.dumps(r))
    finally:
        ttnn.close_device(dev)
    with open(os.path.join(OUT, f"{preset}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", os.path.join(OUT, f"{preset}.json"))


if __name__ == "__main__":
    main()
