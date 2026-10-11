import json, os, sys, torch
from pathlib import Path
from transformers import AutoConfig
import ttnn
from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import OptimizedDecoder, DEFAULT_POLICIES
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
layer, length, extra, runs = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
x = torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer{layer}.pt", weights_only=True)["activations"][:, :length].contiguous()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[int(os.environ.get("QDEV", "0"))])
put = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=mesh, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
get = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
rec = {}
late = []
def wrap(obj, name, label, pick=lambda r: r):
    fn = getattr(obj, name)
    def w(*a, **k):
        r = fn(*a, **k)
        p = pick(r)
        if p is not None and getattr(p, "shape", None) is not None and len(p.shape) >= 3 and p.shape[-2] > 1:
            rec.setdefault(label, []).append(get(p))
        return r
    setattr(obj, name, w)
try:
    kind = cfg.layer_types[layer]
    model = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh, optimization={**DEFAULT_POLICIES[kind], **extra})
    L = model.layer
    moe = L.feed_forward.module
    inner = moe.moe if hasattr(moe, "moe") else moe
    wrap(L.attention, "forward", "attention")
    class Call:
        def __init__(self, obj, label, pick=lambda r: r):
            self.obj, self.label, self.pick = obj, label, pick
        def __getattr__(self, n):
            return getattr(self.obj, n)
        def __call__(self, *a, **k):
            r = self.obj(*a, **k)
            p = self.pick(r)
            if p is not None and len(p.shape) >= 3 and p.shape[-2] > 1:
                rec.setdefault(self.label, []).append(get(p))
            return r
    inner.router = Call(inner.router, "router_values", lambda r: r[0] if isinstance(r, tuple) else r)
    inner.router_ids = None
    disp = inner.experts.dispatch
    def dispatch_call(hidden, routing, self=disp):
        values, ids = routing
        length = hidden.shape[2]
        counts, offsets, dispatch_map, inverse, capacity = self.route_maps(routing)
        for lab, t in (("d_counts", counts), ("d_offsets", offsets), ("d_map", dispatch_map), ("d_inverse", inverse)):
            rec.setdefault(lab, []).append(get(ttnn.reshape(t, (1, 1, 1, -1)) if len(t.shape) < 3 else t).reshape(1, 1, -1, 1) if False else ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().reshape(1, -1, 1))
        source = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), (length, 2048))
        dispatched = ttnn.embedding(ttnn.reshape(dispatch_map, (1, capacity)), source, layout=ttnn.ROW_MAJOR_LAYOUT)
        dispatched = ttnn.reshape(dispatched, (1, 1, capacity, 2048))
        output = ttnn.empty((1, 1, capacity, 2048), device=hidden.device(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(dispatched, self.gate, self.up, self.down, counts, self.expert_ids,
            input_m_tiles=self.policy.dispatch_m_tiles or length // 32, core_grid=ttnn.CoreCoord(*self.policy.dispatch_grid), dtype=ttnn.bfloat16,
            output=output, expert_region_offsets=offsets, read_x_at_offset=True, compute_kernel_config=self.compute)
        rec.setdefault("d_result", []).append(get(result))
        restored = ttnn.embedding(ttnn.reshape(inverse, (1, length * 8)), ttnn.reshape(ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT), (capacity, 2048)), layout=ttnn.ROW_MAJOR_LAYOUT)
        if os.environ.get("PSYNC"):
            ttnn.synchronize_device(mesh)
        rec.setdefault("d_restored", []).append(get(ttnn.reshape(restored, (1, length * 8, 2048))))
        if os.environ.get("PSYNC"):
            ttnn.synchronize_device(mesh)
        if self.policy.dispatch_combine == "tiles":
            experts = ttnn.to_layout(ttnn.reshape(restored, (8, 1, length, 2048)), ttnn.TILE_LAYOUT)
            rec.setdefault("t_experts", []).append(get(ttnn.reshape(experts, (1, 8, length, 2048))))
            scores = ttnn.transpose(ttnn.reshape(ttnn.transpose(ttnn.to_layout(values, ttnn.TILE_LAYOUT), -2, -1), (8, 1, 1, length)), -2, -1)
            rec.setdefault("t_scores", []).append(get(ttnn.reshape(scores, (1, 8, length, 1))))
            prod = ttnn.multiply(experts, scores)
            rec.setdefault("t_prod", []).append(get(ttnn.reshape(prod, (1, 8, length, 2048))))
            if os.environ.get("NEUTRAL"):
                junk = ttnn.add(experts, experts)
                junk.deallocate(True)
            out = ttnn.experimental.fast_reduce_nc(prod, dims=[0])
            rec.setdefault("d_out", []).append(get(out))
            return out
        r5 = ttnn.reshape(restored, (1, 1, length, 8, 2048))
        w5 = ttnn.reshape(ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT), (1, 1, length, 8, 1))
        rec.setdefault("w5_mem", []).append(str(w5.memory_config().buffer_type))
        if os.environ.get("WDRAM"):
            w5 = ttnn.to_memory_config(w5, ttnn.DRAM_MEMORY_CONFIG)
        if os.environ.get("PRE") == "clone":
            ttnn.clone(r5, memory_config=ttnn.DRAM_MEMORY_CONFIG).deallocate(True)
        elif os.environ.get("PRE") == "tile_add":
            t = ttnn.to_layout(ttnn.reshape(restored, (1, 1, length * 8, 2048)), ttnn.TILE_LAYOUT)
            ttnn.add(t, t).deallocate(True)
            t.deallocate(True)
        elif os.environ.get("PRE") == "pcr_small":
            ttnn.experimental.deepseek_prefill.post_combine_reduce(r5, w5, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG).deallocate(True)
        out = ttnn.experimental.deepseek_prefill.post_combine_reduce(r5, w5, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rec.setdefault("d_out", []).append(get(out))
        out2 = ttnn.experimental.deepseek_prefill.post_combine_reduce(r5, w5, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rec.setdefault("d_out_e5", []).append(get(out2))
        out2.deallocate(True)
        late.append((ttnn.clone(r5, memory_config=ttnn.DRAM_MEMORY_CONFIG), ttnn.clone(w5)))
        return out
    inner.experts.dispatch = dispatch_call
    inner.experts = Call(inner.experts, "experts")
    wrap(inner.shared, "forward", "shared")
    L.ffn_norm = Call(L.ffn_norm, "ffn_norm")
    px = put(x)
    for i in range(runs):
        model.reset_request_state(range(1))
        rec.setdefault("output", []).append(get(model.prefill_forward(px)))
    for r5, w5 in late:
        o = ttnn.experimental.deepseek_prefill.post_combine_reduce(r5, w5, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rec.setdefault("d_out_late", []).append(get(o))
    print("W5_MEM", rec.pop("w5_mem")[:1], flush=True)
    n_calls = len(rec["d_out"]) // runs
    for k in range(len(rec["d_out"])):
        for other in ("d_out_e5", "d_out_late"):
            a, b = rec["d_out"][k], rec[other][k]
            d = (a - b).abs()
            rows = torch.nonzero(d.reshape(-1, d.shape[-1]).amax(-1) > 0).flatten().tolist()
            print(f"WITHIN call{k} d_out_vs_{other} rows={len(rows)} first={rows[:4]} maxdiff={d.max().item():.3g}", flush=True)
    outs = rec["d_out"]
    for i in range(runs):
        for j in range(i + 1, runs):
            for c in range(n_calls):
                a, b = outs[i * n_calls + c].reshape(-1, 2048), outs[j * n_calls + c].reshape(-1, 2048)
                diff = a != b
                if diff.any():
                    rows = torch.nonzero(diff.any(-1)).flatten()
                    per_row = diff.sum(-1)[rows]
                    ab, bb = a.bfloat16().view(torch.int16).int(), b.bfloat16().view(torch.int16).int()
                    ulp = (ab - bb)[diff]
                    tiles = sorted({(int(r) // 32, int(t) // 32) for r, t in torch.nonzero(diff).tolist()})
                    print(f"E1 run{i}_vs_run{j} call{c} rows={len(rows)} blocks={sorted({int(r)//32 for r in rows})} "
                          f"elements={int(diff.sum())} per_row_min={int(per_row.min())} per_row_max={int(per_row.max())} "
                          f"ulp_min={int(ulp.min())} ulp_max={int(ulp.max())} tiles={len(tiles)}", flush=True)
                else:
                    print(f"E1 run{i}_vs_run{j} call{c} identical", flush=True)
    for label, outs in rec.items():
        n = len(outs) // runs
        for i in range(1, runs):
            for j in range(n):
                a, b = outs[j], outs[i * n + j]
                d = (a - b).abs()
                rows = torch.nonzero(d.reshape(-1, d.shape[-1]).amax(-1) > 0).flatten().tolist()
                if rows:
                    print(f"run{i} {label}[{j}] rows={len(rows)} first={rows[:4]} maxdiff={d.max().item():.3g}", flush=True)
    print("PROBE_DONE", {k: len(v) for k, v in rec.items()}, flush=True)
finally:
    ttnn.close_mesh_device(mesh)
