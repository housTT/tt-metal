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
    original_moe = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    calls = []
    def moe(*args, **kwargs):
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        calls.append(dict(rows=int(args[0].shape[-2]), grid=str(kwargs.get("core_grid")),
                          l1_allocated=int(view.total_bytes_allocated_per_bank),
                          l1_largest_free=int(view.largest_contiguous_bytes_free_per_bank)))
        print(json.dumps(calls[-1]), flush=True)
        try:
            bufs = ttnn._ttnn.reports.get_buffers(ttnn.get_device_tensors(args[0])[0].device())
        except Exception as error:
            bufs = []
            print("BUFERR", error, flush=True)
        for b in bufs:
            if "L1" in str(b.buffer_type):
                print("L1BUF", b.address, b.max_size_per_bank, str(b.buffer_layout), flush=True)
        return original_moe(*args, **kwargs)
    ttnn.experimental.deepseek_prefill.moe_fused_swiglu = moe
    report = {"cases": []}
    for n in lengths:
        px = ttnn.from_torch(x[:, :n].contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        print("LENGTH", n, flush=True)
        model.reset_request_state(range(1))
        try:
            y = model.prefill_forward(px, **kw)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(y)
            ok = True
        except Exception as error:
            ok = str(error).splitlines()[-1][:200]
        report["cases"].append(dict(length=n, ok=ok, calls=list(calls)))
        print("RESULT", n, ok, flush=True)
        calls.clear()
        ttnn.deallocate(px)
    out.write_text(json.dumps(report, indent=1) + "\n")
finally:
    ttnn.close_mesh_device(mesh)
