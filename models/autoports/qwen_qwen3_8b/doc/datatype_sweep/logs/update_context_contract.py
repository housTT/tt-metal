"""Add the datatype_sweep section to doc/context_contract.json for the selected precision config (host arithmetic).

Usage: update_context_contract.py <selected config id> [<other kv candidate id> ...]
KV bytes per token follow Qwen3Model.kv_bytes_per_token(dtype); the DRAM budget follows
Qwen3Generator.max_tokens_all_users with the stage-6 measured constants and the per-config resident weights.
"""
import json
import subprocess
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_8b.tt.generator import Qwen3Generator
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import WEIGHT_BYTES_PER_ELEMENT
from models.autoports.qwen_qwen3_8b.tt.precision import decoder_policy, terminal_policy

D = Path("models/autoports/qwen_qwen3_8b/doc")
contract = json.loads((D / "context_contract.json").read_text())
full = contract["full_model"]
H, I, QKV, HEADS, KV_HEADS, HEAD_DIM, LAYERS = 4096, 12288, 6144, 32, 8, 128, 36
CONTEXT = 40960
SIZES = {"qkv": H * QKV, "o_proj": HEADS * HEAD_DIM * H, "gate_proj": H * I, "up_proj": H * I, "down_proj": I * H}
BASE_NON_DECODER = (
    full["weights"]["total_resident_bytes_measured"] - full["weights"]["decoder_stack_resident_bytes"]
) + contract["optimized_full_model"]["weights"]["lm_head_tail_mask_bytes"]


def section(cid: str) -> dict:
    cfg = json.loads((D / "datatype_sweep" / "configs" / f"{cid}.json").read_text())
    per_layer = []
    for idx in range(LAYERS):
        p = decoder_policy(cfg, idx)
        dtypes = {
            "qkv": p.qkv.dtype,
            "o_proj": p.o_proj.dtype,
            "gate_proj": p.gate_up.dtype,
            "up_proj": p.up_role.dtype,
            "down_proj": p.down.dtype,
        }
        per_layer.append(int(sum(SIZES[n] * WEIGHT_BYTES_PER_ELEMENT[d] for n, d in dtypes.items())))
    stack_per_token = sum(per_layer)
    stack_resident = 2 * stack_per_token
    tp = terminal_policy(cfg)
    lm_head = int(H * 152576 * WEIGHT_BYTES_PER_ELEMENT[tp.lm_head_dtype])
    lm_delta = lm_head - full["weights"]["lm_head_bytes_bfp8_padded_152576"]
    resident = BASE_NON_DECODER + stack_resident + lm_delta
    kv_dtype = cfg["kv_cache_dtype"]
    kv_per_token = int(2 * KV_HEADS * HEAD_DIM * WEIGHT_BYTES_PER_ELEMENT[kv_dtype] * LAYERS)
    free = Qwen3Generator.DEVICE_DRAM_BYTES - resident
    tokens = (free - Qwen3Generator.RUNTIME_RESERVE_BYTES) // kv_per_token
    max_tokens = max(32, (tokens // 32) * 32)
    return {
        "config_id": cid,
        "precision_config": f"datatype_sweep/configs/{cid}.json",
        "current_supported_context": CONTEXT,
        "capability_reduction": None,
        "weights": {
            "decoder_weight_bytes_per_token": stack_per_token,
            "decoder_stack_resident_bytes": stack_resident,
            "lm_head_bytes_padded_152576": lm_head,
            "lm_head_dtype": tp.lm_head_dtype,
            "total_resident_bytes_computed": resident,
            "note": "stage-6 measured non-decoder bytes (embedding, RoPE tables, norms, LM head, allocator rounding) plus the per-layer dtype policy twice (DRAM-sharded decode copy and interleaved prefill copy) plus the LM-head tail mask",
        },
        "kv_cache": {
            "dtype": kv_dtype,
            "bytes_per_token_all_layers": kv_per_token,
            "bytes_per_user_at_full_context": kv_per_token * CONTEXT,
            "block_size": 32,
            "page_table_width_blocks": 1280,
        },
        "runtime_reserve_bytes": Qwen3Generator.RUNTIME_RESERVE_BYTES,
        "max_tokens_all_users": int(max_tokens),
        "max_tokens_all_users_formula": "((DEVICE_DRAM_BYTES - total_resident_bytes_computed - runtime_reserve) // kv_bytes_per_token) rounded down to 32 (Qwen3Generator.max_tokens_all_users with kv_cache_dtype)",
        "max_users_at_full_context": int(max_tokens // CONTEXT),
        "batch_32_at_full_context_bytes": kv_per_token * CONTEXT * 32,
        "batch_32_at_full_context_fits": kv_per_token * CONTEXT * 32 <= free - Qwen3Generator.RUNTIME_RESERVE_BYTES,
    }


selected = sys.argv[1]
others = sys.argv[2:]
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
entry = section(selected)
entry[
    "hardware"
] = "one Blackhole P300 chip (TT_VISIBLE_DEVICES=2, 1x1 mesh, p150 mesh graph descriptor); stages 4 and 5 skipped by the operator"
entry["commit"] = sha
entry["kv_candidates_evaluated"] = {cid: section(cid) for cid in others}
entry["notes"] = [
    "The HF-advertised 40960-token context is retained for every request; the number of simultaneous full-context users is bounded by device DRAM (max_users_at_full_context).",
    "Selected KV-cache dtype and weight policy per datatype_sweep/selected_precision_config.json; the capacity numbers above are host arithmetic from the stage-6 measured resident bytes and the per-config dtype sizes, matching Qwen3Generator.max_tokens_all_users without a mesh.",
]
contract["datatype_sweep"] = entry
contract["current_supported_context"] = CONTEXT
(D / "context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
print(json.dumps({k: v for k, v in entry.items() if k != "kv_candidates_evaluated"}, indent=2))
for cid, sec in entry["kv_candidates_evaluated"].items():
    print(
        cid,
        "kv",
        sec["kv_cache"]["dtype"],
        "bytes/token",
        sec["kv_cache"]["bytes_per_token_all_layers"],
        "max_tokens",
        sec["max_tokens_all_users"],
        "users@40960",
        sec["max_users_at_full_context"],
    )
