# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import bz2
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TALE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tt_transformers", "tests", "tale-of-two-cities.txt.bz2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--lengths", default="32,128,512,1024,2048")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import TtQwen3Encoder, open_mesh, parse_mesh_shape

    with bz2.open(TALE, "rt", encoding="utf-8") as f:
        text = f.read()
    mesh = open_mesh(parse_mesh_shape(a.mesh), trace_region_size=200_000_000, l1_small_size=32768)
    rows = []
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "precision": a.precision,
        "max_seq_len": a.max_seq_len,
        "max_batch_size": a.max_batch_size,
        "mesh": a.mesh,
        "repeats": a.repeats,
    }
    try:
        enc = TtQwen3Encoder(
            mesh, max_batch_size=a.max_batch_size, max_seq_len=a.max_seq_len, precision=a.precision, warmup=True
        )
        report["device_name"] = enc.model_args.device_name
        report["load_seconds"] = round(enc.load_seconds, 1)
        all_ids = enc.tokenizer(text[:200000], add_special_tokens=False)["input_ids"]
        for n in [int(x) for x in a.lengths.split(",")]:
            if n > a.max_seq_len:
                continue
            for b in [int(x) for x in a.batches.split(",")]:
                if b > a.max_batch_size:
                    continue
                ids = [all_ids[i * n : (i + 1) * n] for i in range(b)]
                enc.embed_ids(ids)
                enc.embed_ids(ids)
                times = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    enc.embed_ids(ids)
                    times.append(time.perf_counter() - t0)
                t = np.array(times) * 1000
                row = {
                    "tokens": n,
                    "batch": b,
                    "padded": enc.padded_len(n),
                    "p50_ms": float(np.percentile(t, 50)),
                    "mean_ms": float(t.mean()),
                    "min_ms": float(t.min()),
                    "p95_ms": float(np.percentile(t, 95)),
                    "tokens_per_s": float(n * b / (np.percentile(t, 50) / 1000)),
                    "texts_per_s": float(b / (np.percentile(t, 50) / 1000)),
                }
                rows.append(row)
                print("BENCH_ROW", json.dumps(row), flush=True)
        report["encoder_stats"] = enc.stats()
    finally:
        import ttnn

        if "enc" in dir():
            enc.release()
        ttnn.close_mesh_device(mesh)
    report["rows"] = rows
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("BENCH_DONE", a.out)


if __name__ == "__main__":
    main()
