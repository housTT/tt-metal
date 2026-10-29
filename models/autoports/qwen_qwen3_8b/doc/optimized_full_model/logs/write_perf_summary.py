"""perf_summary.json for the optimized full model: roofline, device time per token (reduced profile scaled to 36 layers), end to end."""
import csv
import json
import subprocess
from pathlib import Path

D = Path("models/autoports/qwen_qwen3_8b/doc")
perf = json.loads((D / "full_model" / "perf_batch1_128_128.json").read_text())
contract = json.loads((D / "context_contract.json").read_text())
rows = list(csv.DictReader((D / "optimized_full_model" / "profile" / "decode.csv").open()))
time_col = next(c for c in rows[0] if c.lower().startswith("device time"))


def us(r):
    return float(r[time_col] or 0)


codes = [r["OP Code"] for r in rows]
first_lm = next(i for i, c in enumerate(codes) if c.startswith("Matmul") and "16384" in c)
final_norm_idx = max(i for i in range(first_lm) if codes[i].startswith("LayerNorm"))
layer_start = next(
    i for i, c in enumerate(codes) if not (c.startswith("Embeddings") or c.startswith("InterleavedToSharded"))
)
terminal = {"embedding": 0.0, "final_norm": 0.0, "lm_head": 0.0, "lm_head_layout": 0.0, "sampler": 0.0, "position": 0.0}
layer_ops = []
for i, r in enumerate(rows):
    code, t = codes[i], us(r)
    if i < layer_start:
        terminal["embedding"] += t
    elif i < final_norm_idx:
        layer_ops.append(t)
    elif i == final_norm_idx:
        terminal["final_norm"] += t
    elif code.startswith("Matmul"):
        terminal["lm_head"] += t
    elif code.startswith("ShardedToInterleaved") or (
        code.startswith("Concat")
        and terminal["sampler"] == 0
        and not any(c.startswith("Topk") for c in codes[final_norm_idx:i])
    ):
        terminal["lm_head_layout"] += t
    elif code.startswith("PlusOne"):
        terminal["position"] += t
    else:
        terminal["sampler"] += t
layer_us = sum(layer_ops)
layers = 36
device_ms = (layer_us * layers + sum(terminal.values())) / 1000
weights = contract["full_model"]["weights"]
bytes_per_token = weights["decoder_weight_bytes_per_token"] + weights["lm_head_bytes_bfp8_padded_152576"] + 4096 * 2
kv_bytes = 147456 * 160
dram_gbs = 512.0
roofline_ms = (bytes_per_token + kv_bytes) / (dram_gbs * 1e9) * 1000
e2e = perf["best_token_out"]["decode_ms_per_token"]
summary = {
    "workload": {"profile": "single_user_decode", "prompt_len": 128, "gen_len": 128, "batch": 1},
    "ttft_ms": round(perf["best_token_out"]["ttft_ms"], 2),
    "decode_ms_per_token_e2e": round(e2e, 3),
    "decode_ms_per_token_device": round(device_ms, 3),
    "roofline_ms_per_token_estimate": round(roofline_ms, 3),
    "decode_t_s_u": round(1000 / e2e, 2),
    "device_time_breakdown_us": {
        "decoder_layer_x36": round(layer_us * layers, 1),
        "decoder_layer_each": round(layer_us, 1),
        **{k: round(v, 1) for k, v in terminal.items()},
    },
    "roofline_inputs": {
        "decoder_weight_bytes_per_token": weights["decoder_weight_bytes_per_token"],
        "lm_head_bytes": weights["lm_head_bytes_bfp8_padded_152576"],
        "kv_read_bytes_at_position_160": kv_bytes,
        "dram_bandwidth_gbs_assumed": dram_gbs,
    },
    "host_gap_ms": round(e2e - device_ms, 3),
    "named_limitations": [
        "decoder layers are the stage-3 optimized path (BFP8/BFP4 LoFi DRAM-sharded projections at 79 to 90 percent of DRAM bandwidth, SDPA/norm/head-split/SiLU non-matmul ops about 150 us per layer); the datatype frontier is owned by the datatype-sweep stage",
        "sampler candidate stage is the common sampler's full-row topk_large_indices route (0.73 ms per step); every power-of-two chunking the stock multi-core bitonic accepts measured slower (doc/optimized_full_model/probe_candidates.topk.json)",
        "LM head output is 10 width-sharded pieces converted to DRAM and concatenated (about 0.1 ms per step) because the DRAM-sharded matmul writes its output on the storage cores",
        "decode replay is nonblocking (execute_trace blocking=False) with one 128-byte token readback per step that the caller asks for",
    ],
    "source": {
        "profile": "doc/optimized_full_model/profile/decode.csv (reduced variant: one real layer, real terminal path, one blocking replay of the decode and sampling traces)",
        "e2e": "doc/full_model/perf_batch1_128_128.json best_token_out",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    },
}
(D / "optimized_full_model" / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
