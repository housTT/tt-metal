# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Before/after perf harness for the Kokoro full-model token-out path (stage 06).

Measures, in one mesh session on the (1,4) Blackhole p300c ring mesh:
  - decoder-only traced decode ms/token (layer-stack LOWER BOUND) @T=128/512
  - full-model token-out traced decode ms/token (encode+readout+argmax) @T=128/512
  - eager prefill TTFT @128 (host-visible first output)

Writes a JSON tagged with a --tag label so BEFORE (stage-05 code) and AFTER
(stage-06 optimized) runs can be compared apples-to-apples.

  python .../perf_full_model.py --tag before
  python .../perf_full_model.py --tag after
"""
import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.model import KokoroModel

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = Path(__file__).parent
ITERS = 50


def _ids(vocab, seq_len):
    body = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmpsOvɚleɪzidɔɡ" if c in vocab]
    ids = torch.zeros((1, seq_len), dtype=torch.long)
    for i in range(1, seq_len - 1):
        ids[0, i] = body[i % len(body)]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    out = {"tag": args.tag, "mesh": "(1,4) p300c ring", "iters": ITERS, "workloads": {}}
    try:
        model = KokoroModel.from_state_dict(sd, hf_config=config, vocab=vocab, mesh_device=mesh)

        def _min_of(fn, repeats=3):
            best = None
            for _ in range(repeats):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(ITERS):
                    fn()
                dt = (time.perf_counter() - t0) / ITERS * 1e3
                best = dt if best is None else min(best, dt)
            return best

        def time_decoder_only(seq_len):
            ids = _ids(vocab, seq_len)
            prep = model.prepare_inputs(ids)

            def rep():
                model.decoder.decode_forward(
                    prep["input_ids"],
                    prep["position_ids"],
                    prep["token_type_ids"],
                    prep["attention_mask"],
                    batch=prep["batch"],
                    seq_len=prep["padded_seq_len"],
                )

            rep()  # warm + capture
            return _min_of(rep)

        def time_tokenout(seq_len, want_logits=False):
            ids = _ids(vocab, seq_len)
            prep = model.prepare_inputs(ids)

            def rep():
                model.decode_out_traced(prep, want_logits=want_logits)

            rep()  # warm + capture
            return _min_of(rep)

        for T in (128, 512):
            dec = time_decoder_only(T)
            tok = time_tokenout(T)
            out["workloads"][f"T{T}"] = {
                "decoder_only_ms": round(dec, 4),
                "token_out_ms": round(tok, 4),
                "terminal_overhead_ms": round(tok - dec, 4),
                "terminal_overhead_pct": round((tok - dec) / dec * 100, 2),
                "decoder_only_t_s_u": round(1000.0 / dec, 2),
                "token_out_t_s_u": round(1000.0 / tok, 2),
            }
            print(
                f"T={T}: decoder_only={dec:.4f}ms  token_out={tok:.4f}ms  "
                f"overhead={tok-dec:.4f}ms ({(tok-dec)/dec*100:.1f}%)",
                flush=True,
            )

        # eager prefill TTFT @128 (host-visible first output)
        ids128 = _ids(vocab, 128)
        prep = model.prepare_inputs(ids128)
        _ = model._encode_readout_argmax(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            prep["batch"],
            prep["padded_seq_len"],
            want_logits=False,
        )
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        tok_s, _ = model._encode_readout_argmax(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            prep["batch"],
            prep["padded_seq_len"],
            want_logits=False,
        )
        _ = model.gather_tokens(tok_s, prep["batch"], prep["padded_seq_len"], prep["seq_len"])
        ttft = (time.perf_counter() - t0) * 1e3
        out["workloads"]["ttft_eager_ms_T128"] = round(ttft, 4)
        print(f"TTFT eager @128 = {ttft:.4f}ms", flush=True)

        (HERE / f"perf_{args.tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("PERF_OK", args.tag, flush=True)
    finally:
        try:
            model.release()
        except Exception as e:
            print("release err", e)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
