# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Evaluate ONE datatype-sweep candidate for Kokoro-82M's plbert full model.

Runs in its own process (one device job at a time; see $tt-device-usage) so each
candidate gets a clean mesh + trace region and device state cannot accumulate
across candidates. Given a precision/opt policy on argv, this:

  1. builds the full model + reconstruction readout through build_generator's
     construction path (build_kwargs={"policy","opt"}), i.e. the SAME path the
     readiness runners and vLLM adapter use, so the measured dtype/fidelity is
     exactly what a served model would use;
  2. does a fast one-encode PCC smoke vs HF (catches broken dtype/layout wiring
     before spending a full accuracy run - the skill's "smoketest first" rule);
  3. runs run_prefill_check + run_teacher_forcing (full-model top-1/5/100 over
     the shared IPA phoneme references, K=100); teacher forcing is trace-verified
     (the runner forces enable_trace=True) and yields the decode t/s/u used to
     rank candidates;
  4. runs the warmed min-of-N traced token-out benchmark @T=128/512;
  5. captures the ttnn dtypes/fidelities actually built on device (propagation
     proof) so a JSON field that the code ignores cannot pass silently.

Writes doc/datatype_sweep/candidates/<id>.json.

Usage:
  python run_candidate.py --id baseline --spec '{"policy": {...}, "opt": {...}}'
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.generator import build_generator
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import PrecisionPolicy
from models.autoports.hexgrad_kokoro_82m.tt.optimized_multichip_decoder import OptConfig
from models.common.readiness_check.run_prefill_check import run_prefill_check
from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

MODEL_ID = "hexgrad/Kokoro-82M"
HERE = Path(__file__).parent
MODEL_DIR = HERE.parent.parent  # models/autoports/hexgrad_kokoro_82m
FULL = MODEL_DIR / "doc" / "full_model"
TOP1_BAR = 0.90
TOP5_BAR = 0.98
PCC_BAR = 0.995
# Non-aligned + aligned lengths; non-aligned lengths guard the chunking contract.
PCC_LENS = [8, 31, 33, 64, 127, 128, 200, 256, 511, 512]


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
    return vocab, config, hf


def dtype_summary(gen):
    """Read back the ttnn dtypes/fidelities actually constructed on device.

    This is the propagation proof: it reads .dtype off the real device tensors
    and the math_fidelity off the real compute-kernel configs the forward path
    uses, so a policy field the code ignores would show up here as a mismatch.
    """
    dec = gen.model.decoder
    w = dec.w

    def dt(t):
        return str(t.dtype).replace("DataType.", "")

    def fid(cfg):
        return str(cfg.math_fidelity).replace("MathFidelity.", "")

    return {
        "attn_qkv_w": dt(w["qkv_w"]),
        "attn_dense_w": dt(w["dense_w"]),
        "mlp_ffn_w": dt(w["ffn_w"]),
        "mlp_ffn_out_w": dt(w["ffn_out_w"]),
        "map_w": dt(w["map_w"]),
        "emb_word": dt(w["word_emb"]),
        "norm_attn_ln_w": dt(w["attn_ln_w"]),
        "readout_w": dt(gen.model.readout_w),
        "readout_b": dt(gen.model.readout_b),
        "matmul_fidelity": fid(dec.matmul_kernel_config),
        "sdpa_fidelity": fid(dec.sdpa_kernel_config),
        "norm_fidelity": fid(dec.norm_kernel_config),
        "matmul_fp32_dest_acc": bool(dec.matmul_kernel_config.fp32_dest_acc_en),
        "ag_ccl_dtype": dt_from_name(dec.ag_ccl_dtype),
        "rs_ccl_dtype": dt_from_name(dec.rs_ccl_dtype),
        "activation_dtype": dt_from_name(dec.activation_dtype),
    }


def dt_from_name(d):
    return str(d).replace("DataType.", "")


def agg_topk(rows):
    tot = sum(s["total"] for s in rows)
    return {
        "top1": round(sum(s["matches_top1"] for s in rows) / tot, 6),
        "top5": round(sum(s["matches_top5"] for s in rows) / tot, 6),
        "top100": round(sum(s["matches_top100"] for s in rows) / tot, 6),
        "total": tot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--spec", required=True, help="JSON: {policy:{...}, opt:{...}, desc:...}")
    args = ap.parse_args()
    spec = json.loads(args.spec)
    policy = PrecisionPolicy(**spec.get("policy", {}))
    opt = OptConfig(**spec.get("opt", {}))

    out = {
        "id": args.id,
        "desc": spec.get("desc", ""),
        "policy": policy.__dict__,
        "opt": opt.__dict__,
        "policy_label": policy.label(),
        "opt_label": opt.label(),
        "hardware": "4x Blackhole p300c (ClusterType.P300_X2, physical 4-ring)",
        "mesh": "(1,4)",
        "reference": "doc/full_model/readiness_recon_{prefill,tf}.refpt (IPA phoneme reconstruction, K=100; AIME24 chat-template N/A for TTS)",
        "bars": {"top1": TOP1_BAR, "top5": TOP5_BAR, "pcc": PCC_BAR},
        "command": f"python doc/datatype_sweep/run_candidate.py --id {args.id} --spec '<spec>'",
        "status": "error",
        "error": None,
    }

    vocab, config, hf = load_hf()

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    body = [vocab[c] for c in "ðəkwɪkbraʊnfɑksʤʌmpsOvɚleɪzidɔɡ" if c in vocab]

    def make_ids(seq_len):
        ids = torch.zeros((1, seq_len), dtype=torch.long)
        for i in range(seq_len):
            ids[0, i] = 0 if (i == 0 or i == seq_len - 1) else body[i % len(body)]
        return ids

    try:
        # --- build through the standard construction path -----------------------
        gen = build_generator(model_dir=str(MODEL_DIR), mesh_device=mesh, policy=policy, opt=opt)
        out["dtype_summary"] = dtype_summary(gen)

        # --- fast PCC smoke vs HF (all lengths incl non-aligned) ---------------
        pcc_rows = {}
        pcc_min = 1.0
        for seq_len in PCC_LENS:
            ids = make_ids(seq_len)
            hidden = gen.model.forward(ids, traced=False)
            with torch.no_grad():
                ref = hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            p = round(pcc(hidden, ref), 6)
            pcc_rows[seq_len] = p
            pcc_min = min(pcc_min, p)
        out["pcc_vs_hf"] = {"per_len": pcc_rows, "min": pcc_min, "bar": PCC_BAR}
        out["pcc_non_aligned"] = {str(L): pcc_rows[L] for L in (31, 33, 127, 200, 511)}
        print(f"[{args.id}] PCC min={pcc_min} rows={pcc_rows}", flush=True)

        # --- prefill accuracy ---------------------------------------------------
        pre = run_prefill_check(
            model_dir=MODEL_DIR,
            reference_path=FULL / "readiness_recon_prefill.refpt",
            mesh_device=mesh,
            build_kwargs={"policy": policy, "opt": opt},
        )
        out["prefill_check"] = agg_topk(pre)
        print(f"[{args.id}] prefill {out['prefill_check']}", flush=True)

        # --- teacher forcing (trace-verified) + decode t/s/u -------------------
        tf = run_teacher_forcing(
            model_dir=MODEL_DIR,
            reference_path=FULL / "readiness_recon_tf.refpt",
            mesh_device=mesh,
            build_kwargs={"policy": policy, "opt": opt},
        )
        tf_agg = agg_topk(tf)
        ttfts = [s["ttft_ms"] for s in tf if s.get("ttft_ms") is not None]
        dtoks = sum(s.get("decode_tokens", 0.0) for s in tf)
        delap = sum(s.get("decode_elapsed_s", 0.0) for s in tf)
        tf_agg["ttft_ms"] = round(sum(ttfts) / len(ttfts), 4) if ttfts else None
        tf_agg["decode_t_s_u"] = round(dtoks / delap, 4) if delap > 0 else None
        tf_agg["traced"] = True  # runner forces enable_trace=True
        out["teacher_forcing"] = tf_agg
        print(f"[{args.id}] TF {tf_agg}", flush=True)

        # --- warmed min-of-N traced token-out benchmark @128/512 ----------------
        def tokenout_ms(seq_len, iters=30, repeats=3):
            ids = make_ids(seq_len)
            prep = gen.model.prepare_inputs(ids)
            gen.model.decode_out_traced(prep, want_logits=False)  # warm + capture
            best = None
            for _ in range(repeats):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(iters):
                    gen.model.decode_out_traced(prep, want_logits=False)
                dt = (time.perf_counter() - t0) / iters * 1e3
                best = dt if best is None else min(best, dt)
            return round(best, 4)

        to128 = tokenout_ms(128)
        to512 = tokenout_ms(512)
        out["token_out"] = {
            "T128_ms": to128,
            "T512_ms": to512,
            "T128_t_s_u": round(1000.0 / to128, 2),
            "T512_t_s_u": round(1000.0 / to512, 2),
            "regime": "warmed min-of-3x30 traced replay, batch-1, on-device argmax, no logits readback",
        }
        print(f"[{args.id}] tokenout {out['token_out']}", flush=True)

        # --- pass/fail on full-model teacher-forcing gate -----------------------
        passed = tf_agg["top1"] >= TOP1_BAR and tf_agg["top5"] >= TOP5_BAR and pcc_min >= PCC_BAR
        out["status"] = "pass" if passed else "fail"
        out["gate"] = {
            "top1_ok": tf_agg["top1"] >= TOP1_BAR,
            "top5_ok": tf_agg["top5"] >= TOP5_BAR,
            "pcc_ok": pcc_min >= PCC_BAR,
        }
        print(f"[{args.id}] STATUS={out['status']} gate={out['gate']}", flush=True)

        gen.teardown()
    except Exception as e:
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(f"[{args.id}] ERROR {out['error']}\n{out['traceback']}", flush=True)
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    (HERE / "candidates" / f"{args.id}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"CANDIDATE_DONE {args.id} {out['status']}", flush=True)


if __name__ == "__main__":
    main()
