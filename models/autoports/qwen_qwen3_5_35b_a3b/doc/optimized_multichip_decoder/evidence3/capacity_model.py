import json
import sys
from pathlib import Path

inventory = Path(sys.argv[1])
headline = Path(sys.argv[2])
out = Path(sys.argv[3])
GIB = 2**30
VOCAB, HIDDEN = 248320, 2048
LINEAR_LAYERS, FULL_LAYERS = 30, 10
CONTEXT = 262144
BFP8 = 1088 / 1024
RESERVE = 2 * GIB
SHAREABLE = ("_rope_cos", "_rope_sin", "rank_buffers", "buffers", "zero_templates", "_zero_recurrent",
             "_zero_conv", "_index_offsets", "region_matrix", "mask", "zero_row", "expert_ids", "ones", "zeros",
             "placeholder", "chunk_constants", "kda_start", "offset", "rotation")
SLOT_STATE = ("_states", "recurrent_state", "fused_conv_state")


def classify(path):
    import re

    names = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)", "." + path)
    if path.startswith("module."):
        return "shareable_constant"
    if "paged_kv_cache_key" in names or "paged_kv_cache_value" in names:
        return "test_kv"
    if any(name in SLOT_STATE for name in names):
        return "slot_state"
    if any(name in SHAREABLE for name in names):
        return "shareable_constant"
    return "layer_weight"


def summarize(path):
    data = json.loads(path.read_text())
    totals = {"layer_weight": 0, "slot_state": 0, "shareable_constant": 0, "test_kv": 0}
    for root in data["roots"]:
        if root["buffer"].endswith("DRAM"):
            totals[classify(root["path"])] += root["bytes"]
    return data, totals


def capacity(mesh):
    commands = json.loads((headline / f"headline_m{mesh}_full" / "commands.json").read_text())
    memory = commands[0]["weight_memory"]
    key = "after_merge" if "after_merge" in memory else "after_cleanup"
    return memory[key]["dram"]["allocated_bytes"] + memory[key]["dram"]["free_bytes"]


result = {"method": ("Per-layer weights from the root-buffer inventory (DRAM payload by unique buffer). Constants "
                     "that are identical in every layer (RoPE tables, dispatch maps, zero templates, index tables) are "
                     "counted once. GDN state counted per configured slot. Embedding BF16 replicated; LM head BFP8 "
                     "sharded by vocabulary. Reserve of 2 GiB for activations, trace and CCL buffers is an estimate "
                     "until the full model measures it."),
          "profiles": {}}
for mesh in (1, 2, 4):
    linear_data, linear = summarize(inventory / f"m{mesh}_layer0.json")
    full_data, full = summarize(inventory / f"m{mesh}_layer3.json")
    slot_bytes = linear["slot_state"] / 32
    kv_heads = max(1, 2 // mesh)
    kv_per_token = FULL_LAYERS * 2 * kv_heads * 256 * BFP8
    weights = LINEAR_LAYERS * linear["layer_weight"] + FULL_LAYERS * full["layer_weight"]
    shared = max(linear["shareable_constant"], full["shareable_constant"]) + min(linear["shareable_constant"],
                                                                                 full["shareable_constant"])
    embedding = VOCAB * HIDDEN * 2
    head = VOCAB * HIDDEN * BFP8 / mesh
    device = capacity(mesh)
    rows = {}
    for slots in (1, 8, 32):
        used = weights + shared + embedding + head + RESERVE + LINEAR_LAYERS * slots * slot_bytes
        tokens = (device - used) / kv_per_token
        rows[str(slots)] = dict(bytes_before_kv=used, kv_tokens=int(tokens), full_context_sequences=tokens / CONTEXT)
    result["profiles"][f"p150x{mesh}" if mesh > 1 else "p150"] = dict(
        tensor_parallel=mesh, device_capacity_bytes=device,
        per_layer_weight_bytes=dict(linear_attention=linear["layer_weight"], full_attention=full["layer_weight"]),
        gdn_state_bytes_per_slot_per_layer=slot_bytes,
        shareable_constant_bytes_counted_once=shared,
        forty_layer_weight_bytes=weights,
        embedding_bytes=embedding, lm_head_bytes=head, reserve_bytes=RESERVE,
        kv_bytes_per_token_all_full_layers=kv_per_token,
        by_gdn_slots=rows,
        fits_one_262144_sequence_with_8_slots=rows["8"]["kv_tokens"] >= CONTEXT,
    )
out.write_text(json.dumps(result, indent=2) + "\n")
for name, p in result["profiles"].items():
    print(name, "weights", round(p["forty_layer_weight_bytes"] / GIB, 2), "GiB;",
          {s: round(r["full_context_sequences"], 2) for s, r in p["by_gdn_slots"].items()}, "x 262144 by slots")
