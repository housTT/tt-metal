import sys as _s

_s.meta_path = [m for m in _s.meta_path if "editable" not in getattr(type(m), "__module__", "")]
_s.path = [p for p in _s.path if "model-bringup" not in p]
import json
import random
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.multichip_decoder import MultichipDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
vocab = cfg["vocab"]
_SYM = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
pool = [vocab[c] for c in _SYM if c in vocab]


def rep(b, L, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(b):
        body = [rng.choice(pool) for _ in range(max(L - 2, 0))]
        ids = ([0] + body + [0])[:L]
        while len(ids) < L:
            ids.append(0)
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def run(fabric, topo, label):
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 4)), trace_region_size=90000000)
    try:
        mc = MultichipDecoder.from_state_dict(sd, hf_config=ac, mesh_device=mesh)
        mc.ccl_topology = topo  # force
        for L in (512, 128):
            ids = rep(1, L, 7)
            p = mc.prepare_inputs(ids)
            mc.decode_forward(
                p["input_ids"],
                p["position_ids"],
                p["token_type_ids"],
                p["attention_mask"],
                batch=p["batch"],
                seq_len=p["padded_seq_len"],
            )
            ttnn.synchronize_device(mesh)
            rec = mc._traces[(1, L, p["attention_mask"] is not None)]
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(mesh, rec["trace_id"], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"{label} L={L} decode_traced_ms={(time.perf_counter()-t0)/50*1e3:.4f}")
        mc.release_traces()
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


run(ttnn.FabricConfig.FABRIC_1D, ttnn.Topology.Linear, "LINEAR")
run(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring, "RING")
print("DONE")
