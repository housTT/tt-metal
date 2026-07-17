# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warmed wall-clock latency for prefill (eager) and decode (traced).

Complements the tt-perf-report device-kernel tables with robust end-to-end
host-measured latency (no profiler markers to drop). Writes perf_summary.json.
"""
import json
import os
import random
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.functional_decoder import FunctionalDecoder

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "hexgrad/Kokoro-82M"
_REP = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
ITERS = 50


def build_ids(vocab, T, seed=0):
    pool = [vocab[c] for c in _REP if c in vocab]
    rng = random.Random(seed)
    body = [rng.choice(pool) for _ in range(T - 2)]
    return torch.tensor([[0, *body, 0]], dtype=torch.long)


def main():
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    dev = ttnn.open_device(device_id=0)
    dec = FunctionalDecoder.from_state_dict(sd, hf_config=ac, mesh_device=dev)
    summary = {
        "model": MODEL_ID,
        "iters": ITERS,
        "weight_dtype": "bfloat16",
        "activation_dtype": "float32",
        "results": {},
    }

    for T in [128, 512]:
        prep = FunctionalDecoder.prepare_inputs(build_ids(cfg["vocab"], T), dev)

        # warmed eager prefill
        o = dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            o = dec.prefill_forward(
                prep["input_ids"],
                prep["position_ids"],
                prep["token_type_ids"],
                prep["attention_mask"],
                batch=prep["batch"],
                seq_len=prep["padded_seq_len"],
            )
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        prefill_ms = (time.perf_counter() - t0) / ITERS * 1e3

        # warmed traced decode
        dec.decode_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )
        ttnn.synchronize_device(dev)
        rec = dec._traces[(prep["batch"], prep["padded_seq_len"])]
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(dev, rec["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        decode_ms = (time.perf_counter() - t0) / ITERS * 1e3

        summary["results"][T] = {
            "prefill_warmed_ms": round(prefill_ms, 4),
            "decode_traced_ms": round(decode_ms, 4),
        }
        print(f"T={T:4d}  warmed prefill={prefill_ms:.3f} ms/pass   traced decode={decode_ms:.3f} ms/pass")

    dec.release_traces()
    ttnn.close_device(dev)
    with open(os.path.join(HERE, "perf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", os.path.join(HERE, "perf_summary.json"))


if __name__ == "__main__":
    main()
