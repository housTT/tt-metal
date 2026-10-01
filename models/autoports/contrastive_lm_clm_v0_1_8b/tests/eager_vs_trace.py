# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import ttnn
    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import TtQwen3Encoder, open_mesh

    mesh = open_mesh((1, 1), trace_region_size=200_000_000, l1_small_size=32768)
    report = {"precision": a.precision, "seq_len": a.seq_len, "repeats": a.repeats}
    try:
        enc = TtQwen3Encoder(
            mesh, max_batch_size=1, max_seq_len=max(1024, a.seq_len), precision=a.precision, warmup=False
        )
        gen = enc.generator
        ids = torch.zeros(1, a.seq_len, dtype=torch.long)
        ids[0, : a.seq_len] = torch.randint(1000, 30000, (a.seq_len,))
        page_table = enc._page_table_for(1, a.seq_len)
        prepared = gen._prepare_trace_prefill(
            ids,
            page_table=page_table[0:1],
            chunk_page_table=None,
            kv_cache=enc.kv_cache,
            model_id=0,
            batch_size=1,
            user_id=0,
            start_pos=0,
        )
        eager = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            out = gen._prefill_trace_forward(prepared, prepared["device_inputs"])
            host = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
            eager.append(time.perf_counter() - t0)
        eager_vec = enc._pool_and_norm(host, [a.seq_len], a.seq_len)
        del prepared
        enc.warmup()
        traced = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            vec = enc.embed_ids([ids[0].tolist()])
            traced.append(time.perf_counter() - t0)
        cos = float((vec[0] * eager_vec[0] / np.linalg.norm(eager_vec[0])).sum())
        report.update(
            {
                "eager_ms_p50": float(np.percentile(np.array(eager) * 1000, 50)),
                "eager_ms_min": float(np.min(eager) * 1000),
                "traced_ms_p50": float(np.percentile(np.array(traced) * 1000, 50)),
                "traced_ms_min": float(np.min(traced) * 1000),
                "speedup_p50": float(np.percentile(eager, 50) / np.percentile(traced, 50)),
                "cosine_traced_vs_eager": cos,
            }
        )
    finally:
        if "enc" in dir():
            enc.release()
        ttnn.close_mesh_device(mesh)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("EAGER_VS_TRACE", json.dumps(report))


if __name__ == "__main__":
    main()
