import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from transformers import AutoConfig

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import decoder_class, open_mesh

mesh_size, layer, batch, length, repeats, out = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                                                 int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6]))
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
torch.manual_seed(1234)
x = torch.randn(batch, length + 8, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
mesh = open_mesh(mesh_size)


def host(t):
    return [ttnn.to_torch(s).clone() for s in ttnn.get_device_tensors(t)]


def diff(a, b):
    return [dict(device=d, identical=bool(torch.equal(x_, y_)),
                 elements=int((x_.float() != y_.float()).sum()),
                 max_abs=float((x_.float() - y_.float()).abs().max())) for d, (x_, y_) in enumerate(zip(a, b))]


def band_map(a, b, counts, offsets):
    rows = []
    for d, (x_, y_) in enumerate(zip(a, b)):
        delta = (x_.float() - y_.float()).abs().reshape(-1, x_.shape[-1])
        if not bool((delta > 0).any()):
            continue
        bands = [int((delta[:, i * 128:(i + 1) * 128] > 0).sum()) for i in range(delta.shape[-1] // 128)]
        cnt = counts[d].reshape(-1).long()
        off = offsets[d].reshape(-1).long()
        experts = []
        for e in range(cnt.numel()):
            if cnt[e] == 0:
                continue
            region = delta[off[e]:off[e] + cnt[e]]
            n = int((region > 0).sum())
            if n:
                experts.append(dict(expert=e, rows=int(cnt[e]), differing=n,
                                    differing_rows=int((region > 0).any(-1).sum())))
        outside = int((delta > 0).sum()) - sum(e["differing"] for e in experts)
        rows.append(dict(device=d, bands=bands, experts=experts, differing_outside_regions=outside,
                         differing_rows=int((delta > 0).any(-1).sum()), total_rows=int(delta.shape[0])))
    return rows


try:
    model = decoder_class("multichip").from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh,
                                                       expert_precision="inherited")
    original_empty = ttnn.empty
    ttnn.empty = lambda *a, **k: ttnn.zeros_like(original_empty(*a, **k))
    original_moe = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    neutral = ttnn.from_torch(torch.ones(1, 1, 32, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    calls = []

    def moe(*args, **kwargs):
        result = original_moe(*args, **kwargs)
        ttnn.synchronize_device(mesh)
        a = host(result)
        again = dict(kwargs, output=ttnn.zeros_like(kwargs["output"]))
        b = host(original_moe(*args, **again))
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.add(neutral, neutral).deallocate(True)
        later = dict(kwargs, output=ttnn.zeros_like(kwargs["output"]))
        c = host(original_moe(*args, **later))
        ttnn.synchronize_device(mesh)
        moved = [ttnn.clone(args[0])] + list(args[1:])
        d = host(original_moe(*moved, **dict(kwargs, output=ttnn.zeros_like(kwargs["output"]))))
        ttnn.synchronize_device(mesh)
        spacer = ttnn.zeros_like(args[0])
        moved2 = [ttnn.clone(args[0])] + list(args[1:])
        e = host(original_moe(*moved2, **dict(kwargs, output=ttnn.zeros_like(kwargs["output"]))))
        ttnn.synchronize_device(mesh)
        spacer.deallocate(True)
        calls.append(dict(a=a, counts=host(args[4]), offsets=host(kwargs["expert_region_offsets"]),
                          b_vs_a=diff(a, b), c_vs_a=diff(a, c), d_vs_a=diff(a, d), e_vs_a=diff(a, e),
                          input_addresses=[args[0].buffer_address(), moved[0].buffer_address(), moved2[0].buffer_address()]))
        return result

    ttnn.experimental.deepseek_prefill.moe_fused_swiglu = moe
    original_pcr = ttnn.experimental.deepseek_prefill.post_combine_reduce
    pcr_calls = []

    def pcr(*args, **kwargs):
        result = original_pcr(*args, **kwargs)
        ttnn.synchronize_device(mesh)
        a = host(result)
        b = host(original_pcr(*args, **kwargs))
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.add(neutral, neutral).deallocate(True)
        c = host(original_pcr(*args, **kwargs))
        ttnn.synchronize_device(mesh)
        d = host(original_pcr(ttnn.clone(args[0]), ttnn.clone(args[1]), *args[2:], **kwargs))
        ttnn.synchronize_device(mesh)
        pcr_calls.append(dict(a=a, b_vs_a=diff(a, b), c_vs_a=diff(a, c), d_vs_a=diff(a, d)))
        return result

    ttnn.experimental.deepseek_prefill.post_combine_reduce = pcr
    kw = {}
    if cfg.layer_types[layer] == "full_attention":
        pages = (length + 8 + 31) // 32
        kc = ttnn.zeros((batch * pages + 4, model.config.n_kv_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc), scratch_block=batch * pages + 3)
        table = torch.randperm(batch * pages, dtype=torch.int32).reshape(batch, pages)
        kw["page_table"] = ttnn.from_torch(table, device=mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    px = ttnn.from_torch(x[:, :length].contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    reference, report = None, dict(mesh=mesh_size, layer=layer, batch=batch, length=length, runs=[])
    for r in range(repeats):
        model.reset_request_state(range(batch))
        calls.clear()
        pcr_calls.clear()
        y = model.prefill_forward(px, **kw)
        ttnn.synchronize_device(mesh)
        out_host = host(y)
        y.deallocate(True)
        ttnn.synchronize_device(mesh)
        run = dict(run=r, calls=[])
        for i, call in enumerate(calls):
            entry = dict(call=i, b_vs_a=call["b_vs_a"], c_vs_a=call["c_vs_a"], d_vs_a=call["d_vs_a"],
                         e_vs_a=call["e_vs_a"], input_addresses=call["input_addresses"])
            if reference is not None:
                ref = reference["calls"][i]
                entry["a_vs_run0"] = diff(ref["a"], call["a"])
                entry["inputs_same_counts"] = all(torch.equal(p, q) for p, q in zip(ref["counts"], call["counts"]))
                entry["band_map_vs_run0"] = band_map(ref["a"], call["a"], call["counts"], call["offsets"])
            run["calls"].append(entry)
        run["pcr_calls"] = []
        for i, call in enumerate(pcr_calls):
            entry = dict(call=i, b_vs_a=call["b_vs_a"], c_vs_a=call["c_vs_a"], d_vs_a=call["d_vs_a"])
            if reference is not None and i < len(reference["pcr"]):
                entry["a_vs_run0"] = diff(reference["pcr"][i], call["a"])
            run["pcr_calls"].append(entry)
        if reference is None:
            reference = dict(calls=[dict(a=c["a"], counts=c["counts"]) for c in calls], output=out_host,
                             pcr=[c["a"] for c in pcr_calls])
        else:
            run["output_vs_run0"] = diff(reference["output"], out_host)
        report["runs"].append(run)
        summary = dict(run=r, calls=len(run["calls"]),
                       b_differs=sum(not d["identical"] for c in run["calls"] for d in c["b_vs_a"]),
                       c_differs=sum(not d["identical"] for c in run["calls"] for d in c["c_vs_a"]),
                       d_differs=sum(not d["identical"] for c in run["calls"] for d in c["d_vs_a"]),
                       e_differs=sum(not d["identical"] for c in run["calls"] for d in c["e_vs_a"]),
                       pcr_d_differs=sum(not d["identical"] for c in run["pcr_calls"] for d in c["d_vs_a"]),
                       a_vs_run0_differs=sum(not d["identical"] for c in run["calls"] for d in c.get("a_vs_run0", [])),
                       pcr_calls=len(run["pcr_calls"]),
                       pcr_b_differs=sum(not d["identical"] for c in run["pcr_calls"] for d in c["b_vs_a"]),
                       pcr_c_differs=sum(not d["identical"] for c in run["pcr_calls"] for d in c["c_vs_a"]),
                       pcr_a_vs_run0_differs=sum(not d["identical"] for c in run["pcr_calls"] for d in c.get("a_vs_run0", [])))
        print(json.dumps(summary), flush=True)
        out.write_text(json.dumps(report, indent=1) + "\n")
finally:
    ttnn.close_mesh_device(mesh)
