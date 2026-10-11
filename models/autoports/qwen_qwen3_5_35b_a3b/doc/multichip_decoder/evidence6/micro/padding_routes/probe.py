import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from transformers import AutoConfig

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import decoder_class, open_mesh

mesh_size, layer, length, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
inputs = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer%d.pt" % layer, weights_only=False)
x = inputs["activations"][:, :length].to(torch.bfloat16).contiguous()
mesh = open_mesh(mesh_size)
try:
    model = decoder_class("multichip").from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh,
                                                       expert_precision="inherited")
    dispatch = model.expert_dispatch
    original_route_maps = type(dispatch).route_maps
    counts_log = []

    def route_maps(self, routing):
        result = original_route_maps(self, routing)
        counts_log.append(dict(rows=int(routing[1].shape[2]), valid_rows=self.valid_rows,
                               routes_per_rank=[int(ttnn.to_torch(c).to(torch.int64).sum())
                                                for c in ttnn.get_device_tensors(result[0])]))
        return result

    type(dispatch).route_maps = route_maps
    kw = {}
    if cfg.layer_types[layer] == "full_attention":
        pages = (length + 31) // 32
        kc = ttnn.zeros((pages + 4, model.config.n_kv_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc), scratch_block=pages + 3)
        kw["page_table"] = ttnn.from_torch(torch.arange(pages, dtype=torch.int32).reshape(1, pages), device=mesh,
                                           dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    px = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    report = {"mesh": mesh_size, "layer": layer, "length": length, "modes": {}}
    outputs = {}
    for mode in ("unmasked", "masked", "unmasked", "masked"):
        model.expert_dispatch = dispatch if mode == "masked" else None
        times = []
        for repeat in range(4):
            model.reset_request_state(range(1))
            counts_log.clear()
            ttnn.synchronize_device(mesh)
            started = time.perf_counter_ns()
            y = model.prefill_forward(px, **kw)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter_ns() - started) / 1000)
            if repeat == 3:
                outputs[mode] = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                logged = list(counts_log)
            ttnn.deallocate(y)
        entry = report["modes"].setdefault(mode, {"prefill_us_runs": [], "route_maps_calls": logged})
        entry["prefill_us_runs"].extend(times[1:])
    for mode, entry in report["modes"].items():
        runs = sorted(entry["prefill_us_runs"])
        entry["prefill_us_median"] = runs[len(runs) // 2]
    a, b = outputs["unmasked"].flatten(), outputs["masked"].flatten()
    report["masked_vs_unmasked_output"] = dict(pcc=float(torch.corrcoef(torch.stack((a, b)))[0, 1]),
                                               max_abs=float((a - b).abs().max()))
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({m: (e["prefill_us_median"], e["route_maps_calls"]) for m, e in report["modes"].items()}))
    print(json.dumps(report["masked_vs_unmasked_output"]))
finally:
    ttnn.close_mesh_device(mesh)
