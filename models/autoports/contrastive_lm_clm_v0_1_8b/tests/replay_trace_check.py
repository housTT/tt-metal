# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="accuracy")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mesh", default="1x1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tracking = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING")
    skip_pc = os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")

    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder import TtQwen3Encoder, open_mesh, parse_mesh_shape

    mesh = open_mesh(parse_mesh_shape(a.mesh), trace_region_size=200_000_000, l1_small_size=32768)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "precision": a.precision,
        "mesh": a.mesh,
        "TT_METAL_TRACE_ALLOC_TRACKING": tracking,
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": skip_pc,
        "rounds": a.rounds,
        "variants": [],
    }
    try:
        enc = TtQwen3Encoder(
            mesh, max_batch_size=a.max_batch_size, max_seq_len=a.max_seq_len, precision=a.precision, warmup=True
        )
        rng = np.random.default_rng(7)
        vocab = enc.tokenizer.vocab_size
        lens = [n for n in enc.trace_lens if n <= a.max_seq_len]
        variants = [(n, b) for n in lens for b in enc.batch_sizes]
        inputs = {}
        for n, b in variants:
            real = max(4, n - 3)
            inputs[(n, b, "A")] = [rng.integers(1000, vocab - 1000, size=real).tolist() for _ in range(b)]
            inputs[(n, b, "B")] = [rng.integers(1000, vocab - 1000, size=real).tolist() for _ in range(b)]
        first = {}
        results = {}
        for r in range(a.rounds):
            order = variants if r % 2 == 0 else list(reversed(variants))
            for n, b in order:
                for tag in ("A", "B"):
                    t0 = time.perf_counter()
                    out = enc.embed_ids(inputs[(n, b, tag)])
                    dt = time.perf_counter() - t0
                    key = (n, b, tag)
                    if key not in first:
                        first[key] = out
                    rep = float((out * first[key]).sum(-1).min())
                    results.setdefault(key, []).append({"round": r, "seconds": round(dt, 4), "cos_vs_first": rep})
        for n, b in variants:
            a_vs_b = float((first[(n, b, "A")] * first[(n, b, "B")]).sum(-1).max())
            det = min(x["cos_vs_first"] for tag in ("A", "B") for x in results[(n, b, tag)])
            report["variants"].append(
                {
                    "padded_len": n,
                    "batch": b,
                    "replays": len(results[(n, b, "A")]) + len(results[(n, b, "B")]),
                    "min_cos_repeated_replay": det,
                    "max_cos_between_different_inputs": a_vs_b,
                    "updated_input_changes_output": bool(a_vs_b < 0.999),
                    "warm_seconds": [x["seconds"] for x in results[(n, b, "A")][1:]],
                }
            )
        report["encoder_stats"] = enc.stats()
        report["trace_ids"] = {str(k): str(v) for k, v in enc.generator.trace_id_prefill.items()}
        report["unsafe_allocation_error"] = None
    except RuntimeError as exc:
        report["unsafe_allocation_error"] = str(exc)[:2000]
    finally:
        import ttnn

        if "enc" in dir():
            enc.release()
        ttnn.close_mesh_device(mesh)
    report["pass"] = report.get("unsafe_allocation_error") is None and all(
        v["min_cos_repeated_replay"] > 0.9999 and v["updated_input_changes_output"] for v in report["variants"]
    )
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(
        "REPLAY_RESULT",
        json.dumps(
            {"pass": report["pass"], "error": report["unsafe_allocation_error"], "variants": report["variants"]}
        ),
    )


if __name__ == "__main__":
    main()
