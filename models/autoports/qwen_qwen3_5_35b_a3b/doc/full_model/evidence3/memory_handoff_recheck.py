import hashlib
import json
from pathlib import Path

ROOT = Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b")
EV = ROOT / "doc" / "full_model" / "evidence3"
handoff = json.loads(Path("/home/hous/dev/qwen3.5-35b-a3b/memory-layout-handoff.json").read_text())
work = handoff["required_future_work"]
MIB = 2**20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


profiles = {}
for mesh, name in ((4, "p150x4"), (2, "p150x2"), (1, "p150")):
    data = json.loads((EV / f"m{mesh}" / "full_evidence.json").read_text())
    census = data["census_after_build"]
    profiles[name] = dict(
        allocator_dram_mib=round(census["allocator_dram_bytes"] / MIB, 1),
        named_dram_mib=round(census["named_dram_bytes"] / MIB, 1),
        unnamed_dram_mib=round(census["unnamed_dram_bytes"] / MIB, 1),
        by_category_dram_mib={k: round(v["dram_bytes"] / MIB, 1) for k, v in census["by_category"].items()},
        by_category_buffers={k: v["buffers"] for k, v in census["by_category"].items()},
        largest_buffers=census["largest"],
        dram_free_after_build_mib=round(data["memory_after_build"]["dram"]["free_bytes"] / MIB, 1),
        dram_largest_free_per_bank_mib=round(data["memory_after_build"]["dram"]["largest_free_bytes_per_bank"] / MIB, 1),
        kv_pool_tokens=data["kv_pool"]["tokens"],
        request_slots=data["kv_pool"]["request_slots"],
        long_prompt_tokens=data["long"]["length"],
        boundary=data["boundary"],
    )

result = dict(
    purpose="Stage 6 recheck of memory-layout-handoff.json against the full 40-layer model.",
    handoff_inspected_on=handoff.get("inspected_on"),
    source_sha256={p.name: sha(p) for p in sorted((ROOT / "tt").glob("*.py"))},
    method=("tests/model_census.py inside tests/run_full_evidence.py: after build and prepare(), walk the generator "
            "object graph, group live device buffers by buffer_unique_id, size each as pages x page size per device, "
            "and compare with the allocator total. Tensors reached from several layers are 'shared_across_layers'; "
            "tensors inside a container object shared by layers count under the first layer that holds it."),
    items=[
        dict(item=work[0], status="addressed for device memory",
             evidence="per-category census, 12 largest buffers and allocator totals per profile below; host RSS and "
                      "setup peaks are not re-measured for the full model (loading builds one layer at a time; the "
                      "Stage 5 per-layer setup peaks apply)"),
        dict(item=work[1], status="addressed",
             evidence="Stage 5 ownership table holds; Stage 6 adds KV pool, GDN request-slot pool, embedding, LM "
                       "head, sampler and generator I/O as separate census categories"),
        dict(item=work[2], status="addressed", evidence="unchanged from Stage 5"),
        dict(item=work[3], status="addressed", evidence="unchanged from Stage 5"),
        dict(item=work[4], status="addressed", evidence="unchanged from Stage 5"),
        dict(item=work[5], status="addressed",
             evidence="tt/model.py _share_layer_constants stores RoPE tables (_rope_cos, _rope_sin) and the other "
                      "per-layer constants once after an on-device equality check; census category "
                      "shared_across_layers"),
        dict(item=work[6], status="addressed", evidence="unchanged from Stage 5"),
        dict(item=work[7], status="addressed", evidence="three profiles qualified: full model and sampling in full_model/evidence3/, readiness accuracy in full_model/evidence2/"),
        dict(item=work[8], status="addressed",
             evidence="full-model allocator and census per profile below; serving capacity measured: 262144-token "
                      "context, KV pool tokens and 32 request slots per profile, long prompt 262100 and boundary "
                      "262143 runs"),
    ],
    known_duplicates=[dict(
        what="routed expert weights on p150x2 and p150x4: expert-parallel prefill copy "
             "(experts.dispatch.*) and tensor-parallel decode copy (experts.weights.*)",
        evidence="largest_buffers per profile: both paths at 144 MiB per layer on p150x2; p150 has only "
                 "experts.weights.* (288 MiB per layer)",
        consumers="dispatch copy: prefill dispatch backend; weights copy: traced decode",
        status="both copies have live consumers; removal is Stage 7 memory work "
               "(doc/optimized_multichip_decoder/README.md, Work owned by later stages)")],
    profiles=profiles,
)
out = ROOT / "doc" / "full_model" / "memory_handoff_recheck.json"
out.write_text(json.dumps(result, indent=1) + "\n")
for name, entry in profiles.items():
    print(name, entry["allocator_dram_mib"], entry["by_category_dram_mib"], "unnamed", entry["unnamed_dram_mib"])
