import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.jaredpalmer_kev_9b.tt.engine import KevEngine
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter
from models.experimental.gated_attention_gated_deltanet.tt import ttnn_gated_deltanet as gdn_ops
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import chunk_gated_delta_rule_seq_adapter

VARIANTS = ("4d", "4d_l2norm", "flat")


def fused_variant(variant, kw):
    kw = dict(kw)
    q, k, v = kw["q"], kw["k"], kw["v"]
    B, T, H, K = q.shape
    V = v.shape[3]
    if variant == "4d_l2norm":
        kw["q"] = ttnn.to_memory_config(l2_norm_ttnn(q, dim=-1), ttnn.DRAM_MEMORY_CONFIG)
        kw["k"] = ttnn.to_memory_config(l2_norm_ttnn(k, dim=-1), ttnn.DRAM_MEMORY_CONFIG)
    elif variant == "flat":
        kw["q"] = ttnn.reshape(q, [B, T, H * K])
        kw["k"] = ttnn.reshape(k, [B, T, H * K])
        kw["v"] = ttnn.reshape(v, [B, T, H * V])
        kw["qkv_head_dims"] = (H, K, H, V)
    return chunk_gated_delta_rule_fused_adapter(**kw)


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


def position_pccs(o_ref, o, n=5):
    T = o_ref.shape[1]
    idx = sorted(set(torch.linspace(0, T - 1, n).round().long().tolist()))
    return {int(t): round(pcc(o_ref[0, t], o[0, t]), 6) for t in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pieces", default="2048,256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=a.device_id, l1_small_size=24576, num_command_queues=2, trace_region_size=0)
    dev.enable_program_cache()
    engine = KevEngine(dev, n_layers=a.layers, max_state_len=8192, snapshot_slots=1, traced=False, matmul_policy=False)
    n_gdn = len(engine.gdn_layers)
    records = []
    calls = {"i": 0}

    def timed(fn):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        return out, (time.perf_counter() - t0) * 1000

    def both(**kw):
        layer = calls["i"] % n_gdn
        calls["i"] += 1
        (o_seq, s_seq), t_seq = timed(lambda: chunk_gated_delta_rule_seq_adapter(**kw))
        o_seq_t, s_seq_t = ttnn.to_torch(o_seq), ttnn.to_torch(s_seq)
        rec = {
            "call": calls["i"] - 1,
            "gdn_layer": layer,
            "T": int(kw["q"].shape[1]),
            "valid_len": kw.get("valid_len"),
            "q_shape": list(kw["q"].shape),
            "initial_state": None
            if kw["initial_state"] is None
            else [str(kw["initial_state"].dtype), list(kw["initial_state"].shape)],
            "seq_ms": round(t_seq, 2),
            "state_dtype_seq": str(s_seq.dtype),
        }
        for variant in a.variants.split(","):
            try:
                (o_f, s_f), t_f = timed(lambda: fused_variant(variant, kw))
            except Exception as e:
                rec[variant] = {"error": str(e).splitlines()[0][:200]}
                continue
            o_f_t, s_f_t = ttnn.to_torch(o_f), ttnn.to_torch(s_f)
            rec[variant] = {
                "fused_ms": round(t_f, 2),
                "state_dtype": str(s_f.dtype),
                "o_finite": bool(torch.isfinite(o_f_t).all()),
                "o_pcc": round(pcc(o_seq_t, o_f_t), 6),
                "o_pcc_by_position": position_pccs(o_seq_t, o_f_t),
                "state_pcc": round(pcc(s_seq_t, s_f_t.reshape(s_seq_t.shape)), 6),
            }
            ttnn.deallocate(o_f)
            ttnn.deallocate(s_f)
        logger.info(json.dumps(rec))
        records.append(rec)
        Path(a.out).write_text(
            json.dumps({"layers": a.layers, "pieces": a.pieces, "records": records}, indent=1) + "\n"
        )
        return o_seq, s_seq

    gdn_ops.chunk_gated_delta_rule_seq_adapter = both
    g = torch.Generator().manual_seed(7)
    pieces = [int(x) for x in a.pieces.split(",")]
    ids = torch.randint(0, 100000, (1, sum(pieces)), generator=g, dtype=torch.long)
    engine._begin_sequence(ids)
    cs = 0
    for T in pieces:
        ttnn.deallocate(engine._run_segment(ids[:, cs : cs + T], cs, 0))
        cs += T
    logger.info(f"wrote {a.out}")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
