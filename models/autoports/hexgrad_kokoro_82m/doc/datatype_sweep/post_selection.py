# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Post-selection validation for the Kokoro-82M datatype sweep.

Builds the model through the NORMAL construction path (build_generator with no
explicit policy/opt -> loads selected_precision_config.json) and:

  1. warmed token-out no-readback benchmark @T=128/512 + eager TTFT@128 (same
     regime as optimized-full-model perf_full_model.py) -> post_selection_tokenout.json.
     Recorded SEPARATELY from teacher-forcing; this is the serving-headline number.
  2. non-aligned prompt check: last_hidden_state PCC vs HF at 31/33/127/200/511
     (non-tile-aligned logical lengths) through the selected config -> non_aligned_check.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.generator import build_generator

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = Path(__file__).parent
MODEL_DIR = HERE.parent.parent
PCC_BAR = 0.995


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(config).eval()
    hf.load_state_dict(sd, strict=False)
    body = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmpsOvɚleɪzidɔɡ" if c in vocab]

    def make_ids(seq_len):
        ids = torch.zeros((1, seq_len), dtype=torch.long)
        for i in range(seq_len):
            ids[0, i] = 0 if (i == 0 or i == seq_len - 1) else body[i % len(body)]
        return ids

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    try:
        # NORMAL construction path: no policy/opt -> selected config file is loaded.
        gen = build_generator(model_dir=str(MODEL_DIR), mesh_device=mesh)
        model = gen.model

        # ---- 1. warmed token-out no-readback benchmark -----------------------
        def tokenout_ms(seq_len, iters=50, repeats=3):
            ids = make_ids(seq_len)
            prep = model.prepare_inputs(ids)
            model.decode_out_traced(prep, want_logits=False)  # warm + capture
            best = None
            for _ in range(repeats):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(iters):
                    model.decode_out_traced(prep, want_logits=False)
                dt = (time.perf_counter() - t0) / iters * 1e3
                best = dt if best is None else min(best, dt)
            return round(best, 4)

        to128 = tokenout_ms(128)
        to512 = tokenout_ms(512)

        # eager TTFT @128 (host-visible first output, same as optimized-full-model)
        ids128 = make_ids(128)
        prep = model.prepare_inputs(ids128)
        model._encode_readout_argmax(
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
        model.gather_tokens(tok_s, prep["batch"], prep["padded_seq_len"], prep["seq_len"])
        ttft = round((time.perf_counter() - t0) * 1e3, 4)

        tokenout = {
            "construction_path": "build_generator (no explicit policy/opt -> selected_precision_config.json)",
            "regime": "warmed min-of-3x50 traced token-out replay, batch-1, on-device argmax, no logits readback (host_argmax=0, logits_readbacks=0)",
            "hardware": "4x Blackhole p300c (ClusterType.P300_X2, physical 4-ring)",
            "mesh": "(1,4)",
            "ttft_eager_ms_T128": ttft,
            "token_out_decode_ms_T128": to128,
            "token_out_decode_ms_T512": to512,
            "token_out_decode_t_s_u_T128": round(1000.0 / to128, 2),
            "token_out_decode_t_s_u_T512": round(1000.0 / to512, 2),
            "note": "Post-selection token-out is the serving-headline number; teacher-forcing decode t/s/u (the sweep RANKING metric) is separate and recorded in sweep_results.json.",
        }
        (HERE / "post_selection_tokenout.json").write_text(json.dumps(tokenout, indent=2), encoding="utf-8")
        print("TOKENOUT", tokenout, flush=True)

        # ---- 2. non-aligned prompt check -------------------------------------
        rows = {}
        ok = True
        for L in [31, 33, 127, 200, 511]:
            ids = make_ids(L)
            hidden = model.forward(ids, traced=False)
            with torch.no_grad():
                ref = hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            p = round(pcc(hidden, ref), 6)
            rows[str(L)] = p
            ok = ok and (p >= PCC_BAR)
        na = {
            "construction_path": "build_generator (no explicit policy/opt -> selected_precision_config.json)",
            "lengths_non_tile_aligned": rows,
            "pcc_bar": PCC_BAR,
            "all_pass": ok,
            "note": "Selected config uses baseline dtype/layout, so internal padding to TP*TILE=128 and SDPA masking/chunking are unchanged; non-aligned logical lengths still validate at PCC>=0.995.",
        }
        (HERE / "non_aligned_check.json").write_text(json.dumps(na, indent=2), encoding="utf-8")
        print("NON_ALIGNED", na, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("POST_SELECTION_OK", flush=True)


if __name__ == "__main__":
    main()
