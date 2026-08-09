"""Device DRAM headroom + the byte budget of the functional decoder at the full context.

``MeshDevice`` exposes no DRAM-capacity accessor in this build, so the capacity number is
measured the honest way: allocate 1 GiB DRAM tensors until the allocator refuses, and report
the last size that succeeded. Combined with the byte arithmetic below this is what
``../context_contract.json`` cites for "no capability reduction was needed".
"""

import json
import math

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import DEFAULT_BLOCK_SIZE
from models.autoports.qwen_qwen3_6_27b.tt.model_config import decoder_shapes

GIB = 1 << 30

config = ref.load_text_config()
context = config.max_position_embeddings
block = DEFAULT_BLOCK_SIZE

per_kind = {}
for layer_idx in (0, 3):
    s = decoder_shapes(config, layer_idx)
    if s.is_linear:
        weight_elems = (
            s.conv_dim * s.hidden_size  # in_proj_qkv
            + s.value_dim * s.hidden_size  # in_proj_z
            + 2 * s.num_v_heads * s.hidden_size  # in_proj_b / in_proj_a
            + s.value_dim * s.hidden_size  # out_proj
            + 3 * s.intermediate_size * s.hidden_size  # gate / up / down
        )
        entry = {
            "kv_cache_bytes_at_full_context": 0,
            "per_user_state_bytes": (s.conv_kernel_size * s.conv_dim + s.num_v_heads * s.head_k_dim * s.head_v_dim)
            * 4,  # float32 conv + recurrent state; independent of context length
        }
    else:
        weight_elems = (
            2 * s.num_attention_heads * s.head_dim * s.hidden_size  # q_proj emits q and gate
            + 2 * s.num_key_value_heads * s.head_dim * s.hidden_size  # k_proj / v_proj
            + s.num_attention_heads * s.head_dim * s.hidden_size  # o_proj
            + 3 * s.intermediate_size * s.hidden_size
        )
        blocks = math.ceil(context / block)
        entry = {
            "blocks_per_user_at_full_context": blocks,
            "kv_cache_bytes_at_full_context": 2 * blocks * s.num_key_value_heads * block * s.head_dim * 2,
            "per_user_state_bytes": 0,
        }
    entry["layer_type"] = s.layer_type
    entry["weight_bytes_bf16"] = weight_elems * 2
    entry["total_bytes_batch1_full_context"] = (
        entry["weight_bytes_bf16"] + entry["kv_cache_bytes_at_full_context"] + entry["per_user_state_bytes"]
    )
    per_kind[str(layer_idx)] = entry

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
chunk = torch.zeros(1, 1, 1024, GIB // (2 * 1024), dtype=torch.bfloat16)  # 1 GiB of bfloat16
held, failure = [], None
try:
    while len(held) < 64:
        held.append(
            ttnn.from_torch(
                chunk,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
except Exception as exc:  # noqa: BLE001 - the allocator failure *is* the measurement
    failure = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
allocated = len(held)
for tensor in held:
    ttnn.deallocate(tensor)
ttnn.close_mesh_device(mesh)

print(
    "CAPACITY "
    + json.dumps(
        {
            "arch": "blackhole, one chip of a p300c board (TT_VISIBLE_DEVICES selects it)",
            "dram_probe_gib_allocated": allocated,
            "dram_probe_failure": failure,
            "context": context,
            "block_size": block,
            "per_layer_kind": per_kind,
            "worst_case_single_layer_batch1_bytes": max(
                e["total_bytes_batch1_full_context"] for e in per_kind.values()
            ),
        }
    )
)
