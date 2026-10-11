import json
import sys
from pathlib import Path

E = sys.argv[1]
doc = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc")
path = doc / "context_contract.json"
contract = json.loads(path.read_text())
section = contract["multichip_decoder_validation"]
tests = {}
for mesh in (4, 2, 1):
    for kind in ("linear", "full"):
        for length in (262143, 262144):
            rel = f"multichip_decoder/{E}/accept_context_m{mesh}/{kind}_{length}.json"
            data = json.loads((doc / rel).read_text())
            tests[f"m{mesh}_{kind}_{length}"] = dict(prefill_pcc=data.get("prefill_pcc"), decode_pcc=data.get("decode_pcc"),
                                                     artifact=rel)
section["native_context_tests"] = tests
model = json.loads((doc / f"multichip_decoder/{E}/capacity_model.json").read_text())
for name, profile in model["profiles"].items():
    entry = section["capacity"][name]
    entry["per_layer_weight_bytes"] = profile["per_layer_weight_bytes"]
    entry["gdn_state_bytes_per_slot_per_layer"] = profile["gdn_state_bytes_per_slot_per_layer"]
    entry["forty_layer_weight_bytes"] = profile["forty_layer_weight_bytes"]
    entry["shareable_constant_bytes_counted_once"] = profile["shareable_constant_bytes_counted_once"]
    entry["kv_tokens_by_gdn_slots"] = {k: v["kv_tokens"] for k, v in profile["by_gdn_slots"].items()}
    entry["full_context_sequences_by_gdn_slots"] = {k: round(v["full_context_sequences"], 2)
                                                    for k, v in profile["by_gdn_slots"].items()}
    entry["fits_one_262144_sequence"] = profile["fits_one_262144_sequence_with_8_slots"]
section["artifacts"] = [f"multichip_decoder/{E}/capacity_model.json", f"multichip_decoder/{E}/inventory/",
                        f"multichip_decoder/{E}/memory_handoff_recheck.json"]
p150 = section["capacity"]["p150"]["full_context_sequences_by_gdn_slots"]
section["capacity_conditions"][-1] = (f"p150 holds one native-context sequence ({p150['8']} x 262144 tokens with 8 slots); "
                                      f"p150 with 32 slots holds {p150['32']}.")
path.write_text(json.dumps(contract, indent=1) + "\n")
print(json.dumps({k: v["full_context_sequences_by_gdn_slots"] for k, v in section["capacity"].items()}))
