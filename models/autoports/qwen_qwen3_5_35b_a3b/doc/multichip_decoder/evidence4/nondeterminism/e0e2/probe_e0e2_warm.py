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
name = sys.argv[7] if len(sys.argv) > 7 else "multichip"
replays = int(sys.argv[9]) if len(sys.argv) > 9 else 200
base = int(sys.argv[10]) if len(sys.argv) > 10 else 0
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
torch.manual_seed(1234)
x = torch.randn(batch, length + 8, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
mesh = open_mesh(mesh_size)
records = []


def snap(tag, tensor):
    if isinstance(tensor, (tuple, list)):
        for i, t in enumerate(tensor):
            snap(f"{tag}[{i}]", t)
        return
    if not isinstance(tensor, ttnn.Tensor):
        return
    records.append((tag, [ttnn.to_torch(s).clone() for s in ttnn.get_device_tensors(tensor)]))


def wrap(obj, method, tag):
    original = getattr(obj, method)

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        snap(tag, result)
        return result

    object.__setattr__(obj, method, wrapped) if type(obj).__name__ == "_AllReduceOutput" else setattr(obj, method, wrapped)


try:
    model = decoder_class(name).from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh,
                                                expert_precision="inherited")
    layer_obj = model.layer
    attention = layer_obj.attention
    feed_forward = layer_obj.feed_forward
    inner_attention = object.__getattribute__(attention, "reduced_module") if type(attention).__name__ == "_AllReduceOutput" else attention
    inner_ff = object.__getattribute__(feed_forward, "reduced_module") if type(feed_forward).__name__ == "_AllReduceOutput" else feed_forward
    wrap(inner_attention, "forward", "attention_partial")
    if inner_attention is not attention:
        wrap(attention, "forward", "attention_reduced")
    wrap(inner_ff, "forward", "moe_partial")
    if inner_ff is not feed_forward:
        wrap(feed_forward, "forward", "moe_reduced")
    dispatch = inner_ff.module.experts.dispatch
    if len(sys.argv) > 8 and sys.argv[8] == "lists":
        fused, stacked_down = dispatch.gate[0], dispatch.down[0]
        experts_local, width = fused.shape[1], stacked_down.shape[-2]
        dispatch.up = [ttnn.typecast(ttnn.reshape(fused[:, e:e + 1, :, :width], (2048, width)), ttnn.bfloat8_b)
                       for e in range(experts_local)]
        dispatch.gate = [ttnn.typecast(ttnn.reshape(fused[:, e:e + 1, :, width:], (2048, width)), ttnn.bfloat8_b)
                         for e in range(experts_local)]
        dispatch.down = [ttnn.reshape(stacked_down[:, e:e + 1], (width, 2048)) for e in range(experts_local)]
        dispatch.weight_options = {}
    wrap(dispatch, "route_maps", "dispatch_route_maps")
    original_empty = ttnn.empty

    def zeroed_empty(*args, **kwargs):
        return ttnn.zeros_like(original_empty(*args, **kwargs))

    ttnn.empty = zeroed_empty
    original_moe = ttnn.experimental.deepseek_prefill.moe_fused_swiglu

    moe_calls = []
    keep = {"live": False}

    def moe(*args, **kwargs):
        snap("moe_input", args[0])
        snap("moe_counts", args[4])
        snap("moe_expert_ids", args[5])
        snap("moe_offsets", kwargs.get("expert_region_offsets"))
        result = original_moe(*args, **kwargs)
        snap("moe_output", result)
        if keep["live"]:
            moe_calls.append((args, dict(kwargs), len(records) - 1))
        return result

    ttnn.experimental.deepseek_prefill.moe_fused_swiglu = moe
    original_pcr = ttnn.experimental.deepseek_prefill.post_combine_reduce

    def pcr(*args, **kwargs):
        snap("post_combine_input_rows", args[0])
        snap("post_combine_input_weights", args[1])
        result = original_pcr(*args, **kwargs)
        snap("post_combine_output", result)
        return result

    ttnn.experimental.deepseek_prefill.post_combine_reduce = pcr
    original_fast = ttnn.experimental.fast_reduce_nc

    def fast(*args, **kwargs):
        snap("fast_reduce_input", args[0])
        result = original_fast(*args, **kwargs)
        snap("fast_reduce_output", result)
        return result

    ttnn.experimental.fast_reduce_nc = fast
    shared = inner_ff.module.shared
    wrap(shared, "forward", "shared_partial")
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
    runs = []
    for r in range(repeats):
        keep["live"] = r == repeats - 1
        model.reset_request_state(range(batch))
        records.clear()
        y = model.prefill_forward(px, **kw)
        snap("output", y)
        ttnn.synchronize_device(mesh)
        if r < repeats - 1:
            ttnn.deallocate(y)
            ttnn.synchronize_device(mesh)
        runs.append(list(records))
    report = {"mesh": mesh_size, "layer": layer, "batch": batch, "length": length, "decoder": name,
              "probe": "e0: output deallocated and device synchronized between runs; raw drift stats per capture",
              "captures_per_run": len(runs[0]), "comparisons": []}
    drift_count = {}
    report["reference_run"] = base
    report["reference_note"] = "run 0 compiles programs (cold); runs >= 1 are warm"
    for r in range(base + 1, repeats):
        first = None
        diffs = []
        for index, ((tag, a), (tag_b, b)) in enumerate(zip(runs[base], runs[r])):
            assert tag == tag_b
            per_device = []
            for d, (ta, tb) in enumerate(zip(a, b)):
                if ta.dtype.is_floating_point:
                    delta = (ta.float() - tb.float()).abs()
                    per_device.append(dict(device=d, identical=bool(torch.equal(ta, tb)), max_abs=float(delta.max()),
                                           elements=int((delta > 0).sum())))
                else:
                    per_device.append(dict(device=d, identical=bool(torch.equal(ta, tb)),
                                           elements=int((ta != tb).sum())))
            if not all(p["identical"] for p in per_device):
                diffs.append(dict(index=index, tag=tag, devices=per_device))
                if tag == "moe_output":
                    for p in per_device:
                        if not p["identical"]:
                            drift_count[(index, p["device"])] = drift_count.get((index, p["device"]), 0) + 1
                if first is None:
                    first = dict(index=index, tag=tag, devices=per_device)
        report["comparisons"].append(dict(run=r, first_difference=first, differing_captures=len(diffs),
                                          differing_tags=sorted({d["tag"] for d in diffs}), detail=diffs))
        print(json.dumps({"run": r, "first": first and {"index": first["index"], "tag": first["tag"]},
                          "differing": len(diffs), "tags": sorted({d["tag"] for d in diffs})}), flush=True)
    upstream_tags = ("moe_input", "moe_counts", "moe_expert_ids", "moe_offsets")
    input_drift = [d for c in report["comparisons"] for d in c["detail"] if d["tag"] in upstream_tags]
    report["moe_input_side_drift_captures"] = len(input_drift)
    report["moe_output_drift_by_call_device"] = {f"{i}:{d}": n for (i, d), n in sorted(drift_count.items())}
    target = max(drift_count, key=drift_count.get) if drift_count else None
    if target is None and moe_calls:
        target = (moe_calls[len(moe_calls) // 2][2], 0)
    replay = {}
    if target is not None:
        capture_index, device = target
        call = next(c for c in moe_calls if c[2] == capture_index)
        args, kwargs = call[0], dict(call[1])
        reference = runs[-1][capture_index][1]
        def run_once():
            kwargs["output"] = ttnn.zeros(tuple(kwargs["output"].shape), device=mesh, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            result = original_moe(*args, **kwargs)
            shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(result)]
            ttnn.deallocate(result)
            return shards
        neutral = ttnn.from_torch(torch.randn(1, 1, 4096, 2048, dtype=torch.bfloat16), device=mesh,
                                  layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        for mode in ("back_to_back", "interleaved"):
            outs = []
            for i in range(replays):
                if mode == "interleaved":
                    tmp = ttnn.add(neutral, neutral)
                    ttnn.deallocate(tmp)
                outs.append(run_once())
            per_device = []
            for d in range(len(outs[0])):
                distinct = []
                for o in outs:
                    if not any(torch.equal(o[d], x) for x in distinct):
                        distinct.append(o[d])
                per_device.append(dict(device=d, distinct_outputs=len(distinct),
                                       equal_to_model_call=sum(torch.equal(o[d], reference[d]) for o in outs),
                                       max_abs_vs_first=max(float((o[d].float() - outs[0][d].float()).abs().max()) for o in outs)))
            replay[mode] = per_device
            print(json.dumps({"replay": mode, "per_device": per_device}), flush=True)
        dump = dict(device=device, capture_index=capture_index,
                    args=[[ttnn.to_torch(s) for s in ttnn.get_device_tensors(a)] if isinstance(a, ttnn.Tensor)
                          else [[ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)] for t in a] if isinstance(a, list)
                          else a for a in args],
                    arg_dtypes=[str(a.dtype) if isinstance(a, ttnn.Tensor) else [str(t.dtype) for t in a] if isinstance(a, list) else None for a in args],
                    arg_layouts=[str(a.layout) if isinstance(a, ttnn.Tensor) else [str(t.layout) for t in a] if isinstance(a, list) else None for a in args],
                    kwargs={k: ([ttnn.to_torch(s) for s in ttnn.get_device_tensors(v)], str(v.dtype), str(v.layout)) if isinstance(v, ttnn.Tensor)
                            else (v.x, v.y) if k == "core_grid" else v
                            for k, v in kwargs.items() if k not in ("output", "compute_kernel_config", "dtype")},
                    output_shape=tuple(kwargs["output"].shape), reference=reference)
        torch.save(dump, out.with_suffix(".call.pt"))
        report["replay_target"] = {"capture_index": capture_index, "device": device,
                                   "model_runs_drifting": drift_count.get(target, 0)}
    report["replay"] = replay
    out.write_text(json.dumps(report, indent=1) + "\n")
finally:
    ttnn.close_mesh_device(mesh)
