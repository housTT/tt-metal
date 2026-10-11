import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/hous/dev/qwen3.5-35b-a3b/tt-metal")
from transformers import AutoConfig

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import open_mesh
from models.autoports.qwen_qwen3_5_35b_a3b.tt import multichip_decoder as mc

mesh_size, layer, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
import ttnn

ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
mesh = open_mesh(mesh_size)
records = []
original = mc.MultichipDecoder._replica.__func__


def recording(cls, *args, **kwargs):
    replica = original(cls, *args, **kwargs)
    memory = replica.weight_memory
    records.append({key: memory[key]["dram"]["allocated_bytes"] for key in memory})
    return replica


mc.MultichipDecoder._replica = classmethod(recording)
try:
    model = mc.MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh,
                                                expert_precision="inherited")
    result = dict(mesh=mesh_size, layer=layer, replica_weight_memory=records,
                  peak_before_cleanup_bytes=max(r["before_cleanup"] for r in records),
                  construction_memory=model.construction_memory)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"mesh": mesh_size, "layer": layer, "peak_mib": result["peak_before_cleanup_bytes"] / 2**20,
                      "per_replica_mib": [{k: round(v / 2**20) for k, v in r.items()} for r in records]}))
finally:
    ttnn.close_mesh_device(mesh)
