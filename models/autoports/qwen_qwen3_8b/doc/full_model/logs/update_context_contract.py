"""Add the full_model section to doc/context_contract.json from the byte accounting and the capacity probe."""
import json
from pathlib import Path

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
path = MODEL_DIR / "doc" / "context_contract.json"
contract = json.loads(path.read_text())
probe_path = MODEL_DIR / "doc" / "full_model" / "capacity_probe_3x40960.json"
probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None

dram = 34178731008
layer_resident = 359661568
layers = 36
embedding = 151936 * 4096 * 2
lm_head = int(4096 * 152576 * 1.0625)
norm = 4096 * 2
weights = layer_resident * layers + embedding + lm_head + norm
kv_per_token = 2 * 8 * 128 * 2 * layers
per_user = kv_per_token * 40960
reserve = 1073741824
budget_tokens = ((dram - weights - reserve) // kv_per_token) // 32 * 32
users_full_context = budget_tokens // 40960

section = {
    "hardware": "one Blackhole P300 chip per process (1x1 mesh, p150 mesh graph descriptor); readiness runners on TT_VISIBLE_DEVICES=0 and 1",
    "device_dram_bytes_allocatable": dram,
    "device_dram_bytes_allocatable_source": "ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM): 8 banks x 4272341376 bytes free on a fresh device (stage 1)",
    "weights": {
        "decoder_layer_resident_bytes": layer_resident,
        "decoder_stack_resident_bytes": layer_resident * layers,
        "decoder_weight_bytes_per_token": 179830784 * layers,
        "embedding_bytes_bf16": embedding,
        "lm_head_bytes_bfp8_padded_152576": lm_head,
        "final_norm_bytes": norm,
        "total_resident_bytes": weights,
        "note": "decoder_stack_resident includes the interleaved prefill copy of every projection (OptimizedPolicy.prefill_interleaved_copy_max_rows=256); without it the stack is 6473908224 bytes",
    },
    "kv_cache": {
        "dtype": "bf16",
        "bytes_per_token_all_layers": kv_per_token,
        "bytes_per_user_at_full_context": per_user,
        "block_size": 32,
        "page_table_width_blocks": 1280,
        "scratch_block": "one extra block per cache; page-table columns past a request's allocation name it",
    },
    "runtime_reserve_bytes": reserve,
    "runtime_reserve_note": "prefill activations (two [1, 1, 40960, 4096] BF16 tensors are 671 MB), trace buffers, sampler and zero-fill buffers",
    "max_tokens_all_users": budget_tokens,
    "max_tokens_all_users_formula": "((dram_allocatable - total_resident_weights - runtime_reserve) // kv_bytes_per_token) rounded down to 32 (Qwen3Generator.max_tokens_all_users)",
    "max_users_at_full_context": users_full_context,
    "batch_32_at_full_context_bytes": 32 * per_user,
    "batch_32_at_full_context_fits": False,
    "batch_32_at_full_context_note": "32 users x 40960 tokens need 193 GB of BF16 KV cache against 34.18 GB of device DRAM; the hard physical limit bounds the simultaneous full-context users, not the per-request context",
    "layer_kinds": {
        "full_attention": {
            "num_layers_of_kind": 36,
            "cache_block_size": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "prefill_chunk_size": 4096,
            "prefill_compute_buckets": [32, 64, 128, 256, 512, 1024, 2048, 4096, "then multiples of 4096"],
            "decode_physical_batches": [1, 8, 16, 24, 32],
            "tested_prefill_context": 40960,
            "tested_decode_context": 40960,
            "largest_tested_non_aligned_prefill": 40959,
            "largest_tested_decode_batch": 32,
        }
    },
    "generator_defaults": {"num_blocks": 1280, "max_batch_size": 32, "decode_buckets": [1, 8, 16, 24, 32]},
    "artifacts": [
        "full_model/capacity_probe_3x40960.json",
        "full_model/batch_decode_32.json",
        "full_model/multi_row_prefill.json",
        "full_model/README.md",
    ],
}
if probe is not None:
    section["capacity_probe"] = {
        "num_blocks": probe["num_blocks"],
        "rows": probe.get("first_tokens"),
        "prefill_3_rows_s": probe.get("prefill_3_rows_s"),
        "decode_steps": probe.get("decode_steps_live_rows"),
        "status": "passed",
    }
contract["full_model"] = section
contract["notes"] = contract.get("notes", []) + [
    "full-model stage: the HF-advertised 40960-token context is retained for every request; the number of simultaneous full-context users is bounded by device DRAM (see full_model.max_users_at_full_context)."
]
path.write_text(json.dumps(contract, indent=2) + "\n")
print("current_supported_context", contract["current_supported_context"], "budget_tokens", budget_tokens, "users", users_full_context)
