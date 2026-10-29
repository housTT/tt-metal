"""Add the full_model section to doc/context_contract.json from the byte accounting and the capacity probe."""
import json
from pathlib import Path

MODEL_DIR = Path("models/autoports/qwen_qwen3_8b")
path = MODEL_DIR / "doc" / "context_contract.json"
contract = json.loads(path.read_text())
probe_path = MODEL_DIR / "doc" / "full_model" / "capacity_probe_budget.json"
probe = json.loads(probe_path.read_text()) if probe_path.is_file() else None

dram = 33641860096
layer_resident = 359661568
layers = 36
embedding = 151936 * 4096 * 2
lm_head = int(4096 * 152576 * 1.0625)
norm = 4096 * 2
rope_shared = 4 * 40960 * 128 * 2
norm_weights = layers * 2 * (4096 + 128) * 2 + 4096 * 2
weights_computed = layer_resident * layers + embedding + lm_head + rope_shared + norm_weights
weights = 1866045440 * 8
kv_per_token = 2 * 8 * 128 * 2 * layers
per_user = kv_per_token * 40960
reserve = 2147483648
budget_tokens = ((dram - weights - reserve) // kv_per_token) // 32 * 32
users_full_context = budget_tokens // 40960

section = {
    "hardware": "one Blackhole P300 chip per process (1x1 mesh, p150 mesh graph descriptor); readiness runners on TT_VISIBLE_DEVICES=0 and 1",
    "device_dram_bytes_allocatable": dram,
    "device_dram_bytes_allocatable_source": "ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM) in logs/capacity_probe.log: 8 banks x 4205232512 total_bytes_per_bank with a 512 MB trace region reserved (stage 1 recorded 4272341376 free per bank with trace_region_size=0)",
    "weights": {
        "decoder_layer_resident_bytes": layer_resident,
        "decoder_stack_resident_bytes": layer_resident * layers,
        "decoder_weight_bytes_per_token": 179830784 * layers,
        "embedding_bytes_bf16": embedding,
        "lm_head_bytes_bfp8_padded_152576": lm_head,
        "rope_tables_shared_bytes": rope_shared,
        "norm_weight_bytes": norm_weights,
        "total_resident_bytes_computed": weights_computed,
        "total_resident_bytes_measured": weights,
        "total_resident_bytes_measured_source": "ttnn.get_memory_view after Qwen3Model load: 8 banks x 1866045440 total_bytes_allocated_per_bank (logs/capacity_probe.log)",
        "note": "decoder_stack_resident includes the interleaved prefill copy of every projection (OptimizedPolicy.prefill_interleaved_copy_max_rows=256); without it the stack is 6473908224 bytes. Before RoPE sharing each layer held its own 41.9 MB of tables and a 3 x 1280-block pool did not fit (logs/capacity_probe_3x1280.log)",
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
    "runtime_reserve_note": "prefill activations (two [1, 1, 40960, 4096] BF16 tensors are 671 MB plus chunk intermediates), sampler state, zero-fill rows, dynamic trace storage when the mesh is opened without a trace region",
    "max_tokens_all_users": budget_tokens,
    "max_tokens_all_users_formula": "((free DRAM after weights - runtime_reserve) // kv_bytes_per_token) rounded down to 32; free DRAM measured by ttnn.get_memory_view when a mesh is given, else DEVICE_DRAM_BYTES - WEIGHTS_RESIDENT_BYTES (Qwen3Generator.max_tokens_all_users)",
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
            "warmup_prefill_context": 40960,
            "warmup_prefill_note": "every build prefills a 40960-token synthetic prompt in prepare(); no output check",
            "largest_checked_prefill": 16384,
            "largest_checked_prefill_artifact": "readiness/prefill_resume_check.json (natural text, logits checked at 8 starts)",
            "largest_checked_non_aligned_prefill": 1000,
            "largest_checked_non_aligned_prefill_artifact": "full_model/multi_row_prefill.json (lengths 5, 31, 33, 127, 129, 257, 640, 1000)",
            "largest_checked_decode_position": 314,
            "largest_checked_decode_position_artifact": "readiness_autoregressive/autoregressive_meta.json (59-token prompt plus 256 generated)",
            "largest_tested_decode_batch": 32,
            "full_context_decode_status": "not demonstrated: the capacity probe (decode at 40959 after two full-context rows) hung on the device; see full_model.capacity_probe",
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
section["capacity_probe"] = {
    "status": "not_completed",
    "note": "logs/capacity_probe*.log and triage/: a pool of 3510 blocks allocated and prepared (40960-token warmup prefill included) and the three-row prefill (40959, 40960, 200 tokens) returned, then the first bucket-8 decode step at positions 40959 and 200 (trace capture, replay, sampler, token readback) never completed on chip 1 and on chip 0; max_users_at_full_context is the DRAM arithmetic, not a demonstrated run",
}
if probe is not None:
    section["capacity_probe"].update(
        {
            "num_blocks": probe["num_blocks"],
            "rows": probe.get("first_tokens"),
            "decode_steps": probe.get("decode_steps_live_rows"),
            "status": "passed",
        }
    )
contract["full_model"] = section
note = "full-model stage: the HF-advertised 40960-token context is retained for every request; the number of simultaneous full-context users is bounded by device DRAM (see full_model.max_users_at_full_context)."
contract["notes"] = [n for n in contract.get("notes", []) if n != note] + [note]
path.write_text(json.dumps(contract, indent=2) + "\n")
print(
    "current_supported_context",
    contract["current_supported_context"],
    "budget_tokens",
    budget_tokens,
    "users",
    users_full_context,
)
