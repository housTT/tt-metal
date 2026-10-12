import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from transformers import AutoConfig

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import decoder_class, open_mesh

mesh_size, layer, lengths, out = int(sys.argv[1]), int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], Path(sys.argv[4])
length = max(lengths)
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
inputs = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer%d.pt" % layer, weights_only=False)
x = inputs["activations"][:, :length].to(torch.bfloat16).contiguous()
mesh = open_mesh(mesh_size)
try:
    decoder = sys.argv[5] if len(sys.argv) > 5 else "multichip"
    model = decoder_class(decoder).from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh,
                                                       expert_precision="inherited")
    dispatch = None
    original_route_maps = type(dispatch).route_maps if dispatch is not None else None
    counts_log = []

    def route_maps(self, routing):
        result = original_route_maps(self, routing)
        counts_log.append(dict(rows=int(routing[1].shape[2]), valid_rows=self.valid_rows,
                               routes_per_rank=[int(ttnn.to_torch(c).to(torch.int64).sum())
                                                for c in ttnn.get_device_tensors(result[0])]))
        return result

    if dispatch is not None:
        type(dispatch).route_maps = route_maps
    kw = {}
    if cfg.layer_types[layer] == "full_attention":
        pages = (length + 31) // 32 + 128
        kc = ttnn.zeros((pages + 4, model.config.n_kv_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc), scratch_block=pages + 3)
        kw["page_table"] = ttnn.from_torch(torch.arange(pages, dtype=torch.int32).reshape(1, pages), device=mesh,
                                           dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    report = {"mesh": mesh_size, "layer": layer, "decoder": decoder, "policy_moe_pieces": model.optimization.prefill_moe_pieces,
              "method": "prefill_forward host time, synchronized; 1 compile run then 6 warm runs; median of warm runs",
              "cases": []}
    for n in lengths:
        px = ttnn.from_torch(x[:, :n].contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        times = []
        for repeat in range(7):
            model.reset_request_state(range(1))
            ttnn.synchronize_device(mesh)
            started = time.perf_counter_ns()
            y = model.prefill_forward(px, **kw)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter_ns() - started) / 1000)
            ttnn.deallocate(y)
        warm = sorted(times[1:])
        case = dict(length=n, compile_run_us=times[0], warm_runs_us=times[1:], warm_median_us=warm[len(warm) // 2],
                    warm_min_us=warm[0], warm_max_us=warm[-1])
        report["cases"].append(case)
        print(json.dumps({k: case[k] for k in ("length", "warm_median_us", "warm_min_us", "warm_max_us")}), flush=True)
        ttnn.deallocate(px)
    out.write_text(json.dumps(report, indent=1) + "\n")
finally:
    ttnn.close_mesh_device(mesh)
