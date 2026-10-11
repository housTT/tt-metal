import json, os, statistics, sys, time
import torch
from pathlib import Path
from transformers import AutoConfig
import ttnn
from models.autoports.qwen_qwen3_5_35b_a3b.tt import optimized_decoder as od
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict

layer, length = int(sys.argv[1]), int(sys.argv[2])
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
x = torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer{layer}.pt", weights_only=True)["activations"][:, :length].contiguous()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[int(os.environ.get("QDEV", "3"))])
put = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=mesh, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
captured = {}
try:
    model = od.OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
    kw = {}
    if cfg.layer_types[layer] == "full_attention":
        pages = (length + 31) // 32 + 8
        kc = ttnn.zeros((pages + 3, cfg.num_key_value_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc))
        kw = dict(page_table=put(torch.arange(pages, dtype=torch.int32).reshape(1, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
    original = od._LocalDispatch.__call__

    def capture(self, hidden, routing):
        counts, offsets, dispatch_map, inverse, capacity = self.route_maps(routing)
        source = ttnn.reshape(ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT), (hidden.shape[2], 2048))
        dispatched = ttnn.reshape(ttnn.embedding(ttnn.reshape(dispatch_map, (1, capacity)), source, layout=ttnn.ROW_MAJOR_LAYOUT), (1, 1, capacity, 2048))
        captured.update(dispatcher=self, dispatched=ttnn.clone(dispatched), counts=ttnn.clone(counts), offsets=ttnn.clone(offsets), capacity=capacity, length=hidden.shape[2])
        return original(self, hidden, routing)

    od._LocalDispatch.__call__ = capture
    model.prefill_forward(put(x), **kw).deallocate(True)
    ttnn.synchronize_device(mesh)
    od._LocalDispatch.__call__ = original
    router = model.layer.feed_forward.module.moe.router.module.original
    for name in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
        getattr(router, name).deallocate(True)
    l1 = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    d = captured["dispatcher"]
    table = ttnn.reshape(ttnn.typecast(d.expert_ids, ttnn.uint32), (256,))
    L, cap = captured["length"], captured["capacity"]
    grid = ttnn.CoreCoord(*(d.policy.dispatch_grid_large[1:] if d.policy.dispatch_grid_large and L >= d.policy.dispatch_grid_large[0] else d.policy.dispatch_grid))

    def fused():
        out = ttnn.empty((1, 1, cap, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(captured["dispatched"], d.gate, d.up, d.down, captured["counts"], d.expert_ids,
            input_m_tiles=L // 32, core_grid=grid, dtype=ttnn.bfloat16, output=out, expert_region_offsets=captured["offsets"], read_x_at_offset=True, compute_kernel_config=d.compute)

    def hybrid(threshold):
        return lambda: ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(captured["dispatched"], captured["offsets"], captured["counts"], table,
            d.gate, d.up, d.down, L, hybrid_token_threshold=threshold, compute_kernel_config=d.compute)

    def unified():
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(captured["dispatched"], captured["offsets"], captured["counts"], table,
            d.gate, d.up, d.down, L, compute_kernel_config=d.compute)

    reference = ttnn.to_torch(fused(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    counts_host = ttnn.to_torch(captured["counts"], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).flatten()[:256].to(torch.int64)
    offsets_host = ttnn.to_torch(captured["offsets"], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).flatten()[:256].to(torch.int64)
    rows = torch.cat([torch.arange(int(o), int(o) + int(c)) for o, c in zip(offsets_host, counts_host) if c])
    result = dict(layer=layer, length=L, l1_allocated_after_free=l1.total_bytes_allocated_per_bank * l1.num_banks,
                  max_expert_tokens=int(counts_host.max()), median_expert_tokens=float(counts_host.float().median()))
    for name, fn in [("fused", fused), ("unified", unified)] + [(f"hybrid_{t}", hybrid(t)) for t in (32, 64, 128, 256)]:
        try:
            out = fn(); ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(out if out.dtype == ttnn.bfloat16 else ttnn.typecast(out, ttnn.bfloat16), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().reshape(reference.shape)
            a, b = reference.reshape(-1, 2048)[rows], got.reshape(-1, 2048)[rows]
            pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
            times = []
            for _ in range(7):
                ttnn.synchronize_device(mesh); t0 = time.perf_counter_ns(); fn(); ttnn.synchronize_device(mesh); times.append((time.perf_counter_ns() - t0) / 1000)
            result[name] = dict(median_us=statistics.median(times), pcc_vs_fused=pcc)
        except Exception as e:
            result[name] = dict(error=str(e).splitlines()[0][:300])
        print(name, result[name], flush=True)
    print("RESULT", json.dumps(result), flush=True)
    Path(f"hybrid_l{layer}_{length}.json").write_text(json.dumps(result, indent=1))
finally:
    ttnn.close_mesh_device(mesh)
