import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import open_mesh
from models.autoports.qwen_qwen3_5_35b_a3b.tt.multichip_decoder import MultichipDecoder

mesh_size, out = int(sys.argv[1]), Path(sys.argv[2])
conv_fill = sys.argv[3] == "fill"
import ttnn
from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import DEFAULT_POLICIES

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
mesh = open_mesh(mesh_size)
H, prompt = cfg.hidden_size, 129
torch.manual_seed(7)
result = dict(mesh=mesh_size, conv_decode_fill=conv_fill, cases=[], states=[])


def put(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous(), device=mesh, dtype=dtype, layout=layout,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def first(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1, H)[0].clone()


def padded_input(row0, pad):
    full = torch.zeros(1, 32, H, dtype=torch.bfloat16)
    full[0, 0] = row0
    if pad is not None:
        full[0, 1:] = pad
    t = put(full)
    return ttnn.reshape(t, (1, 1, H), (1, 32, H))


try:
    layers = {}
    for idx in (0, 3):
        policy = dict(DEFAULT_POLICIES[cfg.layer_types[idx]], conv_decode_fill=conv_fill)
        model = MultichipDecoder.from_state_dict(load_layer_state_dict(ck, idx), hf_config=cfg, layer_idx=idx,
                                                 mesh_device=mesh, optimization=policy)
        kw = {}
        if cfg.layer_types[idx] == "full_attention":
            pages = 16
            kc = ttnn.zeros((pages + 1, model.config.n_kv_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc), scratch_block=pages)
            pt = put(torch.arange(pages, dtype=torch.int32).reshape(1, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            kw = dict(page_table=pt)
        model.prefill_forward(put(torch.randn(1, prompt, H, dtype=torch.bfloat16) * 0.1), **kw).deallocate(True)
        dkw = {}
        if kw:
            dkw = dict(page_table=kw["page_table"],
                       position_tensor=put(torch.tensor([prompt], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
        saved = [(t, ttnn.clone(t)) for row in model._states[:1] for t in row] if cfg.layer_types[idx] != "full_attention" else []
        layers[idx] = (model, dkw, saved)

    def restore(saved):
        for tensor, snapshot in saved:
            ttnn.copy(snapshot, tensor)

    row0 = torch.randn(H, dtype=torch.bfloat16) * 0.5
    for idx in (0, 3):
        model, dkw, saved = layers[idx]
        outputs, states = {}, {}
        for name, pad in (("zero", None), ("large", 1.0e4), ("inf", float("inf")), ("nan", float("nan")),
                          ("random", torch.randn(31, H).to(torch.bfloat16) * 50)):
            restore(saved)
            x_in = padded_input(row0, pad)
            y = model.decode_forward(x_in, **dkw)
            outputs[name] = first(y)
            states[name] = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().clone() for t, _ in saved]
            full = ttnn.to_torch(ttnn.get_device_tensors(ttnn.reshape(ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG), (1, 32, H), (1, 32, H)))[0]).float()
            xin = ttnn.to_torch(ttnn.get_device_tensors(ttnn.reshape(x_in, (1, 32, H), (1, 32, H)))[0]).float()
            result.setdefault("padding_reach", []).append(dict(layer=idx, padding=name,
                input_pad_nonfinite=int((~torch.isfinite(xin[0, 1:])).sum()), input_pad_absmax=float(xin[0, 1:].abs().nan_to_num(1e30).max()),
                output_pad_nonfinite=int((~torch.isfinite(full[0, 1:])).sum()), output_pad_absmax=float(full[0, 1:].abs().nan_to_num(1e30).max())))
            print(json.dumps(result["padding_reach"][-1]), flush=True)
            y.deallocate(True)
        for name, value in states.items():
            result["states"].append(dict(layer=idx, padding=name, tensors=len(value),
                                         bitwise_equal=all(torch.equal(a, b) for a, b in zip(value, states["zero"])),
                                         finite=all(bool(torch.isfinite(a).all()) for a in value)))
            print(json.dumps(result["states"][-1]), flush=True)
        base = outputs["zero"]
        for name, value in outputs.items():
            result["cases"].append(dict(layer=idx, padding=name, row0_bitwise_equal=bool(torch.equal(value, base)),
                                        row0_finite=bool(torch.isfinite(value).all()),
                                        max_abs=float((value - base).abs().nan_to_num(1e30).max())))
            print(json.dumps(result["cases"][-1]), flush=True)
    for a, b in ((0, 3), (3, 0)):
        model_a, dkw_a, saved_a = layers[a]
        model_b, dkw_b, saved_b = layers[b]
        restore(saved_a)
        mid = model_a.decode_forward(padded_input(row0, torch.randn(31, H).to(torch.bfloat16) * 50), **dkw_a)
        restore(saved_b)
        y_raw = first(model_b.decode_forward(mid, **dkw_b))
        restore(saved_b)
        y_zeroed = first(model_b.decode_forward(ttnn.fill_implicit_tile_padding(
            ttnn.to_memory_config(mid, ttnn.L1_MEMORY_CONFIG), 0.0), **dkw_b))
        case = dict(stack=f"{a}->{b}", mid_memory=str(mid.memory_config().memory_layout),
                    row0_bitwise_equal=bool(torch.equal(y_raw, y_zeroed)), row0_finite=bool(torch.isfinite(y_raw).all()),
                    max_abs=float((y_raw - y_zeroed).abs().nan_to_num(1e30).max()))
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
finally:
    out.write_text(json.dumps(result, indent=1))
    ttnn.close_mesh_device(mesh)
