import json
from pathlib import Path

DOC = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc")
EV = DOC / "full_model" / "evidence2"
PROFILES = {4: "p150x4", 2: "p150x2", 1: "p150"}
KV_BYTES_PER_TOKEN = {4: 5440, 2: 5440, 1: 10880}
SLOT_BYTES = {4: 536576, 2: 1073152, 1: 2146304}

contract = json.loads((DOC / "context_contract.json").read_text())
section = {
    "stage": "full_model",
    "capability_reduction": None,
    "supported_context": 262144,
    "profiles": {},
    "notes": [
        "KV pool and request slots are allocated once at build; the KV pool is allocated before the layers.",
        "Per-layer constants (RoPE tables, dispatch scratch, zero templates, index tables, router templates and "
        "router slice cache) are stored once and shared by all 40 layers.",
        "Unused page-table entries point at the KV scratch block (index num_blocks); entry 0 is a real block.",
        "GDN request slots (32 on every profile) are allocated after the layer weights; allocated before or "
        "between the weights, the p150 load failed on a fragmented 570 MB load-time copy (failed runs in "
        "/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/slots/).",
        "Continuation prefill starts must be multiples of 128 tokens; other starts raise ValueError.",
    ],
    "artifacts": ["full_model/evidence2/", "full_model/memory_handoff_recheck.json"],
}
for mesh, name in PROFILES.items():
    path = EV / f"m{mesh}" / "full_evidence.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    memory = data["memory_after_build"]["dram"]
    pool = data["kv_pool"]
    section["profiles"][name] = dict(
        dram_allocated_bytes_after_build_per_device=memory["allocated_bytes"],
        dram_free_bytes_after_build_per_device=memory["free_bytes"],
        dram_largest_free_bytes_per_bank=memory["largest_free_bytes_per_bank"],
        kv_pool_blocks=pool["num_blocks"], kv_block_tokens=pool["block_size"], kv_pool_tokens=pool["tokens"],
        kv_bytes_per_token_all_full_layers=KV_BYTES_PER_TOKEN[mesh],
        kv_pool_bytes=pool["tokens"] * KV_BYTES_PER_TOKEN[mesh],
        request_slots=pool["request_slots"],
        gdn_state_bytes=30 * pool["request_slots"] * SLOT_BYTES[mesh],
        full_context_sequences_in_pool=pool["tokens"] / 262144,
        long_prompt=dict(length=data["long"]["length"], prefill_and_16_tokens_seconds=data["long"]["seconds"],
                         text=data["long"]["text"]),
        boundary=data["boundary"],
        artifact=f"full_model/evidence2/m{mesh}/full_evidence.json",
        census_after_build=data.get("census_after_build", {}).get("by_category"),
    )
contract["full_model_validation"] = section
(DOC / "context_contract.json").write_text(json.dumps(contract, indent=1) + "\n")
print(json.dumps(section["profiles"], indent=1))
