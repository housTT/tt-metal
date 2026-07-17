# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model evidence for Kokoro plbert (one mesh session).

Runs on the (1,4) Blackhole p300c ring mesh and writes:
  - results.json (PCC vs HF, prefill/TF top-1/5/100, perf, counters, sampler cmp)
  - readiness_autoregressive/autoregressive_meta.json (+ hf/tt_completion.txt)
  - split_sampling.json (trace feedback / determinism / on-device argmax)
See doc/full_model/README.md for interpretation.
"""
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.generator import build_generator
from models.common.readiness_check.run_prefill_check import run_prefill_check
from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = Path(__file__).parent
MODEL_DIR = HERE.parent.parent  # models/autoports/hexgrad_kokoro_82m


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def load_hf():
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(config).eval()
    hf.load_state_dict(sd, strict=False)
    w_map = sd["encoder.embedding_hidden_mapping_in.weight"]
    w_word = sd["embeddings.word_embeddings.weight"]
    W = w_map.float() @ w_word.float().t()
    return vocab, hf, W


def ids_of(vocab, text):
    return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)


def main():
    vocab, hf, W = load_hf()
    inv = {int(v): k for k, v in vocab.items()}
    results = {"model": MODEL_ID, "mesh": "(1,4) p300c ring", "sections": {}}

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    try:
        gen = build_generator(model_dir=str(MODEL_DIR), mesh_device=mesh)

        # ---- A. PCC full-model last_hidden_state vs HF (incl non-aligned) -----
        pcc_rows = {}
        for seq_len in [8, 31, 33, 64, 127, 128, 200, 256, 511, 512]:
            ids = torch.zeros((1, seq_len), dtype=torch.long)
            body = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmpsOvɚleɪzidɔɡ" if c in vocab]
            for i in range(seq_len):
                ids[0, i] = 0 if (i == 0 or i == seq_len - 1) else body[i % len(body)]
            hidden = gen.model.forward(ids, traced=False)
            with torch.no_grad():
                ref = hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            pcc_rows[seq_len] = round(pcc(hidden, ref), 6)
        # batch > 1
        ids_b = torch.zeros((4, 64), dtype=torch.long)
        body = [vocab[c] for c in "ðəkwɪkbraʊnfɑks" if c in vocab]
        for r in range(4):
            for i in range(1, 63):
                ids_b[r, i] = body[(i + r) % len(body)]
        hidden_b = gen.model.forward(ids_b, traced=False)
        with torch.no_grad():
            ref_b = hf(ids_b, attention_mask=torch.ones_like(ids_b)).last_hidden_state
        pcc_batch = round(pcc(hidden_b, ref_b), 6)
        results["sections"]["pcc_vs_hf"] = {"per_len": pcc_rows, "batch4_T64": pcc_batch, "bar": 0.995}
        print("A pcc_vs_hf", pcc_rows, "batch4", pcc_batch, flush=True)

        # ---- B. prefill readiness check ---------------------------------------
        pre = run_prefill_check(
            model_dir=MODEL_DIR, reference_path=HERE / "readiness_recon_prefill.refpt", mesh_device=mesh
        )
        tot = sum(s["total"] for s in pre)
        results["sections"]["prefill_check"] = {
            "top1": round(sum(s["matches_top1"] for s in pre) / tot, 4),
            "top5": round(sum(s["matches_top5"] for s in pre) / tot, 4),
            "top100": round(sum(s["matches_top100"] for s in pre) / tot, 4),
            "total": tot,
        }
        print("B prefill_check", results["sections"]["prefill_check"], flush=True)

        # ---- C. teacher-forcing readiness check (top-k + perf) ----------------
        tf = run_teacher_forcing(
            model_dir=MODEL_DIR, reference_path=HERE / "readiness_recon_tf.refpt", mesh_device=mesh
        )
        tot = sum(s["total"] for s in tf)
        ttfts = [s["ttft_ms"] for s in tf if s.get("ttft_ms") is not None]
        dtoks = sum(s.get("decode_tokens", 0.0) for s in tf)
        delap = sum(s.get("decode_elapsed_s", 0.0) for s in tf)
        results["sections"]["teacher_forcing"] = {
            "top1": round(sum(s["matches_top1"] for s in tf) / tot, 4),
            "top5": round(sum(s["matches_top5"] for s in tf) / tot, 4),
            "top100": round(sum(s["matches_top100"] for s in tf) / tot, 4),
            "total": tot,
            "ttft_ms": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
            "decode_t_s_u": round(dtoks / delap, 3) if delap > 0 else None,
        }
        print("C teacher_forcing", results["sections"]["teacher_forcing"], flush=True)

        # ---- D. autoregressive_meta.json (HF vs TT full-context recon) --------
        ar_dir = MODEL_DIR / "readiness_autoregressive"
        ar_dir.mkdir(parents=True, exist_ok=True)
        ar_text = "ðə sˈʌn wˈʌz ʃˈaɪnɪŋ ˈOvɚ ðə mˈaʊntənz ænd ðə vˈæli bɪlˈO ænd ðə bˈɜːdz wɜː sˈɪŋɪŋ"
        ar_ids = ids_of(vocab, ar_text)
        prompt_ids = ar_ids[0].tolist()
        with torch.no_grad():
            hf_hidden = hf(ar_ids, attention_mask=torch.ones_like(ar_ids)).last_hidden_state
        hf_tokens = (hf_hidden @ W)[0].argmax(-1).tolist()
        tt_tokens = gen.generate(
            prompt_token_ids=prompt_ids, max_new_tokens=len(prompt_ids), next_input=None, enable_trace=True
        )
        # readable phoneme string for humans; space-joined tokens for the
        # word-level degeneracy check to exercise all reconstructed positions.
        hf_text = " ".join(inv.get(t, f"<{t}>") for t in hf_tokens)
        tt_text = " ".join(inv.get(t, f"<{t}>") for t in tt_tokens)
        (ar_dir / "hf_completion.txt").write_text(hf_text, encoding="utf-8")
        (ar_dir / "tt_completion.txt").write_text(tt_text, encoding="utf-8")
        (ar_dir / "autoregressive_meta.json").write_text(
            json.dumps(
                {
                    "hf_model_id": MODEL_ID,
                    "note": "Non-autoregressive TTS encoder: 'free-running' = full-context phoneme reconstruction (no token-feedback loop; see doc/full_model/README.md).",
                    "prompt_text": ar_text,
                    "prompt_token_ids": prompt_ids,
                    "max_new_tokens": len(prompt_ids),
                    "hf": {"token_ids": hf_tokens, "num_tokens": len(hf_tokens)},
                    "tt": {"token_ids": tt_tokens, "num_tokens": len(tt_tokens)},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        agree = sum(1 for a, b in zip(hf_tokens, tt_tokens) if a == b) / len(hf_tokens)
        adjdup = sum(1 for a, b in zip(tt_tokens, tt_tokens[1:]) if a == b) / max(len(tt_tokens) - 1, 1)
        results["sections"]["autoregressive"] = {
            "hf_tt_token_agreement": round(agree, 4),
            "tt_adjacent_dup": round(adjdup, 4),
            "tt_text": tt_text,
            "hf_text": hf_text,
        }
        print("D autoregressive agree", round(agree, 4), "adjdup", round(adjdup, 4), flush=True)
        print("   TT:", tt_text, flush=True)

        # ---- E. split-sampling trace test -------------------------------------
        # Two decode steps with DIFFERENT context/positions -> outputs must differ;
        # on-device argmax must equal host argmax of the readout logits.
        ctxA = ids_of(vocab, "ðə kwˈɪk brˈaʊn")
        ctxB = ids_of(vocab, "tˈɛnstɔɹɛnt bˈɪldz ˈAI")
        tokA = int(gen.decode_forward(ctxA, sample_on_device=True, enable_trace=True)[0])
        tokB = int(gen.decode_forward(ctxB, sample_on_device=True, enable_trace=True)[0])
        # host cross-check
        _, logA = gen.decode_forward(ctxA, sample_on_device=False, enable_trace=True, want_logits=True)
        hostA = int(torch.argmax(logA[0]))
        # determinism: repeat A twice -> identical
        tokA2 = int(gen.decode_forward(ctxA, sample_on_device=True, enable_trace=True)[0])
        split = {
            "stepA_token": tokA,
            "stepB_token": tokB,
            "outputs_differ": tokA != tokB,
            "on_device_argmax_eq_host": tokA == hostA,
            "deterministic_replay": tokA == tokA2,
            "counters": dict(gen.counters),
        }
        results["sections"]["split_sampling"] = split
        print("E split_sampling", split, flush=True)

        # ---- F. perf: TTFT (prefill 128) + token-out traced decode t/s/u ------
        def warm_and_time(seq_len, want_logits, iters=20):
            ids = torch.zeros((1, seq_len), dtype=torch.long)
            b = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmps" if c in vocab]
            for i in range(1, seq_len - 1):
                ids[0, i] = b[i % len(b)]
            prepared = gen._prepare(ids)
            gen.model.decode_out_traced(prepared, want_logits=want_logits)  # warm/capture
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(iters):
                gen.model.decode_out_traced(prepared, want_logits=want_logits)
            return (time.perf_counter() - t0) / iters * 1000.0

        # TTFT = eager prefill_forward wall time at prompt 128 (host-visible first output)
        ids128 = torch.zeros((1, 128), dtype=torch.long)
        bb = [vocab[c] for c in "ðəkwɪkbraʊnfɑks" if c in vocab]
        for i in range(1, 127):
            ids128[0, i] = bb[i % len(bb)]
        t0 = time.perf_counter()
        _ = gen.prefill_forward(ids128, prompt_lens=[128], return_all_logits=False, enable_trace=False)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        tokout_128 = warm_and_time(128, want_logits=False)
        tokout_512 = warm_and_time(512, want_logits=False)
        tf_decode_128 = warm_and_time(128, want_logits=True)  # token-out incl logits readback-ready
        results["sections"]["perf"] = {
            "workload": "prompt=128 / token-out re-encode (decode==prefill, stateless)",
            "ttft_ms_prefill128_eager": round(ttft_ms, 3),
            "token_out_decode_ms_T128": round(tokout_128, 3),
            "token_out_decode_ms_T512": round(tokout_512, 3),
            "token_out_decode_t_s_u_T128": round(1000.0 / tokout_128, 2),
            "token_out_decode_t_s_u_T512": round(1000.0 / tokout_512, 2),
            "note": "Token-out decode = traced encode+readout+on-device argmax. Teacher-forcing decode t/s/u is in section C.",
        }
        print("F perf", results["sections"]["perf"], flush=True)

        results["counters_final"] = dict(gen.counters)
        (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (HERE / "split_sampling.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        print("EVIDENCE_OK", flush=True)
    finally:
        try:
            gen.teardown()
        except Exception as e:
            print("teardown err", e)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
