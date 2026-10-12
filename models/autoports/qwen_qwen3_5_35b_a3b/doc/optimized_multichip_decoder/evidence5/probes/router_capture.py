import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from models.autoports.qwen_qwen3_5_35b_a3b.tests.hf_reference import prefill_reference
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import open_mesh
from models.autoports.qwen_qwen3_5_35b_a3b.tests.run_positions import prompt_lengths
from models.autoports.qwen_qwen3_5_35b_a3b.tt.multichip_decoder import MultichipDecoder

mesh_size, out = int(sys.argv[1]), Path(sys.argv[2])
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
cfg._attn_implementation = "eager"
state = load_layer_state_dict(ck, 0)
torch.set_num_threads(16)
recorded = torch.load("/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer0.pt", weights_only=True)["activations"][0]
lengths = prompt_lengths(32, 1234)
cases = []
for row in (24, 17):
    length = lengths[row]
    start = (row * 131) % (recorded.shape[0] - (length + 8) + 1)
    cases.append(dict(name=f"positions_row{row}", x=recorded[None, start:start + length + 8].contiguous(), length=length,
                      steps=8))
cases.append(dict(name="headline_linear", x=recorded[None, :4096 + 40].contiguous(), length=4096, steps=40))

prefix = "model.language_model.layers.0."
with torch.device("meta"):
    ref = Qwen3_5MoeDecoderLayer(cfg, 0)
ref.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()}, assign=True, strict=True)
ref.eval()
rope = Qwen3_5MoeTextRotaryEmbedding(cfg)
cap = {}
gf = ref.mlp.gate.forward
ref.mlp.gate.forward = lambda h: cap.__setitem__("hf", gf(h)) or cap["hf"]
hf = {}
for case in cases:
    _, cache, _ = prefill_reference(ref, cfg, rope, case["x"][:, :case["length"]], 0)
    steps = []
    for i in range(case["steps"]):
        pos = torch.full((1, 1), case["length"] + i, dtype=torch.long)
        token = case["x"][:, case["length"] + i:case["length"] + i + 1]
        ref(token, position_embeddings=rope(token, pos), position_ids=pos, past_key_values=cache)
        logits, _, selected = cap["hf"]
        steps.append(dict(logits=logits[0].float(), selected=sorted(selected[0].tolist())))
    hf[case["name"]] = steps

mesh = open_mesh(mesh_size)
result = dict(mesh=mesh_size, cases={})
try:
    model = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=0, mesh_device=mesh)
    ff = model.layer.feed_forward
    inner = object.__getattribute__(ff, "reduced_module") if type(ff).__name__ == "_AllReduceOutput" else ff
    while not hasattr(inner, "moe"):
        inner = inner.module
    gate = inner.moe.router.module
    original_route = gate.route
    tt = []

    def route(logits):
        values, indices = original_route(logits)
        tt.append(dict(logits=ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(-1, 256)[0],
                       indices=sorted(ttnn.to_torch(ttnn.get_device_tensors(indices)[0]).reshape(-1)[:8].long().tolist())))
        return values, indices

    gate.route = route

    def put(t):
        return ttnn.from_torch(t.contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    for case in cases:
        model.reset_request_state([0])
        model.prefill_forward(put(case["x"][:, :case["length"]])).deallocate(True)
        tt.clear()
        rows = []
        for i in range(case["steps"]):
            y = model.decode_forward(put(case["x"][:, case["length"] + i:case["length"] + i + 1]))
            y.deallocate(True)
            t, h = tt[-1], hf[case["name"]][i]
            order = torch.argsort(h["logits"], descending=True)
            boundary = float(h["logits"][order[7]])
            rows.append(dict(step=i, tt_indices=t["indices"], hf_indices=h["selected"],
                             same_set=t["indices"] == h["selected"],
                             tt_only=sorted(set(t["indices"]) - set(h["selected"])),
                             hf_only=sorted(set(h["selected"]) - set(t["indices"])),
                             hf_boundary_logit=boundary,
                             hf_logits_of_diff={str(e): float(h["logits"][e]) for e in set(t["indices"]) ^ set(h["selected"])},
                             tt_logits_of_diff={str(e): float(t["logits"][e]) for e in set(t["indices"]) ^ set(h["selected"])},
                             max_abs_logit_diff=float((t["logits"] - h["logits"]).abs().max())))
        result["cases"][case["name"]] = rows
        print(case["name"], json.dumps([(r["step"], r["same_set"], r["tt_only"], r["hf_only"], r["hf_logits_of_diff"], r["tt_logits_of_diff"]) for r in rows if not r["same_set"]]), flush=True)
finally:
    out.write_text(json.dumps(result, indent=1))
    ttnn.close_mesh_device(mesh)
