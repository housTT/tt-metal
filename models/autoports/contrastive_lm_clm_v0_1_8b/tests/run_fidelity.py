# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AUTOPORT = os.path.dirname(HERE)
REF_DIR = "/home/hous/dev/clm-v0.1-8B/reference"
CKPT = "/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt"


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def pcc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def head_forward(sd, x):
    h = torch.nn.functional.gelu(x @ sd["inp.weight"].T + sd["inp.bias"])
    h = h @ sd["hidden.0.weight"].T + sd["hidden.0.bias"]
    h = torch.nn.functional.layer_norm(h, (h.shape[-1],), sd["norms.0.weight"], sd["norms.0.bias"])
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.normalize(h @ sd["out.weight"].T + sd["out.bias"], dim=-1).numpy()


def truncate(ids, role, cap):
    if len(ids) <= cap:
        return ids
    return ids[:cap] if role == "candidate" else ids[-cap:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mesh", default="1x1")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    corpus = json.load(open(os.path.join(REF_DIR, "fidelity_corpus.json")))
    ref = np.load(os.path.join(REF_DIR, "hf_embeddings.npy")).astype(np.float32)
    assert len(corpus) == ref.shape[0], (len(corpus), ref.shape)
    if a.limit:
        corpus, ref = corpus[: a.limit], ref[: a.limit]

    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import TtQwen3Encoder, open_mesh, parse_mesh_shape

    mesh = open_mesh(parse_mesh_shape(a.mesh), trace_region_size=200_000_000, l1_small_size=32768)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "precision": a.precision,
        "max_seq_len": a.max_seq_len,
        "max_batch_size": a.max_batch_size,
        "mesh": a.mesh,
        "n_texts": len(corpus),
    }
    try:
        enc = TtQwen3Encoder(
            mesh, max_batch_size=a.max_batch_size, max_seq_len=a.max_seq_len, precision=a.precision, warmup=True
        )
        report["device_name"] = enc.model_args.device_name
        report["load_seconds"] = round(enc.load_seconds, 1)
        cap = a.max_seq_len - 1
        id_lists = [
            truncate(
                enc.tokenizer(c["text"], add_special_tokens=False)["input_ids"] or enc._fallback_ids, c["role"], cap
            )
            for c in corpus
        ]
        t0 = time.perf_counter()
        single = np.stack([enc.embed_ids([ids])[0] for ids in id_lists])
        report["single_seconds"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        batched = enc.embed_ids(id_lists)
        report["batched_seconds"] = round(time.perf_counter() - t0, 2)
        rerun = np.stack([enc.embed_ids([ids])[0] for ids in id_lists[:32]])
        report["encoder_stats"] = enc.stats()
    finally:
        import ttnn

        if "enc" in dir():
            enc.release()
        ttnn.close_mesh_device(mesh)

    refn = l2(ref)
    cos = (single * refn).sum(-1)
    cos_b = (batched * refn).sum(-1)
    iso = (single * batched).sum(-1)
    det = (single[:32] * rerun).sum(-1)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    roles = np.array([c["role"] for c in corpus])
    head_cos = {}
    for role, key in (("state", "state_head"), ("candidate", "action_head")):
        m = roles == role
        if m.any():
            pt = head_forward(ck[key], torch.from_numpy(single[m]).float())
            pr = head_forward(ck[key], torch.from_numpy(refn[m]).float())
            head_cos[role] = {
                "mean": float((pt * pr).sum(-1).mean()),
                "min": float((pt * pr).sum(-1).min()),
                "n": int(m.sum()),
            }
    by_len = {}
    for c, v in zip(corpus, cos):
        b = "<=128" if c["n_tokens"] <= 128 else ("<=1024" if c["n_tokens"] <= 1024 else ">1024")
        by_len.setdefault(b, []).append(float(v))
    report.update(
        {
            "cosine_single_vs_hf": {
                "mean": float(cos.mean()),
                "min": float(cos.min()),
                "p05": float(np.percentile(cos, 5)),
                "median": float(np.median(cos)),
            },
            "cosine_batched_vs_hf": {"mean": float(cos_b.mean()), "min": float(cos_b.min())},
            "cosine_single_vs_batched": {"mean": float(iso.mean()), "min": float(iso.min())},
            "determinism_cosine_first32": {"mean": float(det.mean()), "min": float(det.min())},
            "pcc_single_vs_hf": {
                "mean": float(np.mean([pcc(s, r) for s, r in zip(single, refn)])),
                "min": float(np.min([pcc(s, r) for s, r in zip(single, refn)])),
            },
            "head_projection_cosine": head_cos,
            "cosine_by_length_bucket": {
                k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "n": len(v)} for k, v in by_len.items()
            },
            "nan_count": int(np.isnan(single).sum()),
            "worst": [
                {"id": corpus[i]["id"], "n_tokens": corpus[i]["n_tokens"], "cos": float(cos[i])}
                for i in np.argsort(cos)[:8]
            ],
        }
    )
    np.save(a.out.replace(".json", "_tt_single.npy"), single)
    np.save(a.out.replace(".json", "_tt_batched.npy"), batched)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(
        "FIDELITY_RESULT",
        json.dumps(
            {
                k: report[k]
                for k in (
                    "precision",
                    "cosine_single_vs_hf",
                    "cosine_single_vs_batched",
                    "head_projection_cosine",
                    "nan_count",
                )
            }
        ),
    )


if __name__ == "__main__":
    main()
