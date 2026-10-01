# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os

import numpy as np
import torch

REF_DIR = "/home/hous/dev/clm-v0.1-8B/reference"
CKPT = "/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt"
SCALE = 100.0
MARGIN = 0.10


def head_forward(sd, x):
    h = torch.nn.functional.gelu(x @ sd["inp.weight"].T + sd["inp.bias"])
    h = h @ sd["hidden.0.weight"].T + sd["hidden.0.bias"]
    h = torch.nn.functional.layer_norm(h, (h.shape[-1],), sd["norms.0.weight"], sd["norms.0.bias"])
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.normalize(h @ sd["out.weight"].T + sd["out.bias"], dim=-1)


def decisions(vectors, index, cases, ck):
    out = {}
    for case in cases:
        for qid, q in case["questions"].items():
            si = index.get(("state", q["state_text"]))
            ci = [index.get(("candidate", t)) for t in q["candidate_texts"]]
            if si is None or any(c is None for c in ci):
                continue
            zs = head_forward(ck["state_head"], torch.from_numpy(vectors[si]).float()[None])
            za = head_forward(ck["action_head"], torch.from_numpy(vectors[ci]).float())
            logits = SCALE * (za @ zs[0])
            probs = torch.softmax(logits, dim=0).numpy()
            out[(case["id"], qid)] = dict(zip(q["keys"], probs.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True, help="fidelity *_tt_single.npy or *_tt_batched.npy, corpus order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    corpus = json.load(open(os.path.join(REF_DIR, "fidelity_corpus.json")))
    index = {}
    for i, c in enumerate(corpus):
        index.setdefault((c["role"], c["text"]), i)
    reference = json.load(open(os.path.join(REF_DIR, "typed_decisions_subset_reference.json")))
    subset = {"cases": reference["cases"]}
    ref_q = {(c["id"], qid): q for c in reference["cases"] for qid, q in c["questions"].items()}
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    tt_vectors = np.load(a.vectors).astype(np.float32)
    tt_vectors = tt_vectors / (np.linalg.norm(tt_vectors, axis=-1, keepdims=True) + 1e-12)
    hf = np.load(os.path.join(REF_DIR, "hf_embeddings.npy")).astype(np.float32)
    hf = hf / (np.linalg.norm(hf, axis=-1, keepdims=True) + 1e-12)
    tt_dec = decisions(tt_vectors, index, subset["cases"], ck)
    hf_dec = decisions(hf, index, subset["cases"], ck)
    rows = []
    for key, p_tt in tt_dec.items():
        p_hf = hf_dec[key]
        ref = ref_q.get(key, {})
        ref_label = ref.get("predicted_label") or max(p_hf, key=p_hf.get)
        hf_label = max(p_hf, key=p_hf.get)
        tt_label = max(p_tt, key=p_tt.get)
        srt = sorted(p_hf.values(), reverse=True)
        margin = srt[0] - srt[1] if len(srt) > 1 else 1.0
        tv = 0.5 * sum(abs(p_tt[k] - p_hf[k]) for k in p_hf)
        rows.append(
            {
                "case": key[0],
                "question": key[1],
                "type": ref.get("type"),
                "hf_label": hf_label,
                "ref_label": ref_label,
                "tt_label": tt_label,
                "gold": ref.get("gold_label"),
                "hf_margin": round(margin, 5),
                "tv": round(tv, 5),
                "agree": tt_label == hf_label,
            }
        )
    n = len(rows)
    agree = sum(r["agree"] for r in rows) / n
    conf = [r for r in rows if r["hf_margin"] >= MARGIN]
    agree_conf = sum(r["agree"] for r in conf) / len(conf) if conf else None
    report = {
        "label": a.label,
        "vectors": a.vectors,
        "decisions": n,
        "argmax_agreement": agree,
        "decisions_with_hf_margin_ge_0p10": len(conf),
        "argmax_agreement_margin_ge_0p10": agree_conf,
        "mean_total_variation": sum(r["tv"] for r in rows) / n,
        "accuracy_vs_gold_tt": sum(r["tt_label"] == r["gold"] for r in rows) / n,
        "accuracy_vs_gold_hf": sum(r["hf_label"] == r["gold"] for r in rows) / n,
        "hf_scorer_matches_reference_labels": sum(r["hf_label"] == r["ref_label"] for r in rows) / n,
        "by_type": {
            t: {
                "n": sum(1 for r in rows if r["type"] == t),
                "agreement": sum(r["agree"] for r in rows if r["type"] == t)
                / max(1, sum(1 for r in rows if r["type"] == t)),
            }
            for t in sorted({r["type"] for r in rows})
        },
        "disagreements": [r for r in rows if not r["agree"]],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(
        "DECISION_AGREEMENT",
        json.dumps(
            {
                k: report[k]
                for k in (
                    "label",
                    "decisions",
                    "argmax_agreement",
                    "decisions_with_hf_margin_ge_0p10",
                    "argmax_agreement_margin_ge_0p10",
                    "mean_total_variation",
                    "accuracy_vs_gold_tt",
                    "accuracy_vs_gold_hf",
                    "hf_scorer_matches_reference_labels",
                )
            }
        ),
    )


if __name__ == "__main__":
    main()
