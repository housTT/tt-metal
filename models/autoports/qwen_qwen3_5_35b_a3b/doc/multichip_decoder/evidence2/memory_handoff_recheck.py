import hashlib
import json
import sys
from pathlib import Path

E = Path(sys.argv[1])
root = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b")
handoff = json.loads(Path("/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json").read_text())
MIB = 2**20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


profiles = {}
for mesh in (1, 2, 4):
    entry = {}
    for layer, kind in ((0, "linear_attention"), (3, "full_attention")):
        data = json.loads((E / "inventory" / f"m{mesh}_layer{layer}.json").read_text())
        construction = data.get("construction_memory") or []
        entry[kind] = dict(
            allocator_after_construct_mib=round(data["allocator"]["delta_bytes"] / MIB, 1),
            by_consumer_mib={k: round(v["bytes"] / MIB, 1) for k, v in data["by_consumer"].items()},
            construction_steps=[dict(step=s["step"], device_dram_mib=round(s["device_dram_allocated_bytes"] / MIB, 1),
                                     host_rss_gib=round(s["host_rss_bytes"] / 2**30, 2),
                                     host_peak_rss_gib=round(s["host_peak_rss_bytes"] / 2**30, 2)) for s in construction],
            construction_peak_device_mib=round(max([s["device_dram_allocated_bytes"] for s in construction],
                                                   default=data["allocator"]["delta_bytes"]) / MIB, 1),
            released_after_merge=(not construction or construction[-1]["device_dram_allocated_bytes"]
                                  <= construction[1]["device_dram_allocated_bytes"] + 1),
            weight_memory=data.get("weight_memory"),
        )
    profiles[{1: "p150", 2: "p150x2", 4: "p150x4"}[mesh]] = entry
result = dict(
    purpose="Stage 4 recheck of memory-layout-handoff.json against the multichip implementation.",
    handoff_inspected_on=handoff.get("inspected_on"),
    source_sha256={p.name: sha(p) for p in sorted((root / "tt").glob("*.py"))},
    ttnn_patch_sha256=hashlib.sha256((E / "ttnn_patch.diff").read_bytes()).hexdigest(),
    method=("tests/weight_inventory.py --decoder multichip --mesh N: root-buffer census plus allocator samples taken "
            "during construction (start, after each replica, after merge, after release) and host VmRSS/VmHWM."),
    items=[
        dict(item=handoff["required_future_work"][0], status="addressed",
             evidence="construction_steps per profile below; release confirmed by the post-release sample"),
        dict(item=handoff["required_future_work"][1], status="addressed", evidence="ownership table below"),
        dict(item=handoff["required_future_work"][2], status="addressed",
             evidence="no packed monolithic or grouped copies are live; dispatch reads stacked weights"),
        dict(item=handoff["required_future_work"][3], status="addressed",
             evidence="p150 holds one BFP4 gate/up + BFP8 down copy; dispatch BFP8 lists removed (bitwise identical outputs)"),
        dict(item=handoff["required_future_work"][4], status="addressed",
             evidence="moe_fused_swiglu reads expert offsets inside one stacked tensor (stacked_intermediate)"),
        dict(item=handoff["required_future_work"][5], status="open_full_model",
             evidence="RoPE 64 MiB per full layer in the single-layer harness; full model must share one copy per mesh"),
        dict(item=handoff["required_future_work"][6], status="addressed",
             evidence="TP slices and EP slices per rank (tensor_parallel_state, expert_parallel_state); no gather/replicate adapters"),
        dict(item=handoff["required_future_work"][7], status="addressed",
             evidence="p150 dispatch_weights=stacked; p150x2/p150x4 expert_parallel (MESH_POLICIES)"),
        dict(item=handoff["required_future_work"][8], status="open_full_model",
             evidence="capacity_model.json is a model from single-layer measurements; full model must measure"),
    ],
    ownership=[
        dict(representation="decode experts gate/up BFP4 [1,256,2048,2*512/tp] and down BFP8 [1,256,512/tp,2048]",
             consumers=["decode indexed sparse_matmul", "p150 prefill dispatch (stacked mode)"],
             lifetime="layer lifetime"),
        dict(representation="prefill EP experts gate/up BFP4 [1,256/tp,2048,1024] and down BFP8 [1,256/tp,512,2048] (tp>1)",
             consumers=["prefill moe_fused_swiglu (stacked mode)"], lifetime="layer lifetime"),
        dict(representation="replica tensors before merge (tp copies of the local layer)",
             consumers=["merge_replicas host readback"], lifetime="construction only; released after merge"),
        dict(representation="RoPE tables, dispatch maps, zero templates, index tables",
             consumers=["every layer"], lifetime="per layer in the harness; to be shared per mesh in the full model"),
    ],
    profiles=profiles,
)
(E / "memory_handoff_recheck.json").write_text(json.dumps(result, indent=1) + "\n")
for name, entry in profiles.items():
    for kind, v in entry.items():
        print(name, kind, v["allocator_after_construct_mib"], "peak", v["construction_peak_device_mib"],
              "released", v["released_after_merge"],
              "host peak", max([s["host_peak_rss_gib"] for s in v["construction_steps"]], default=None))
