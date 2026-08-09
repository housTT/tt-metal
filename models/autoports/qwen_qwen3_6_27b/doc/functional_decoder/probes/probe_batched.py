import torch, ttnn
from transformers.cache_utils import DynamicCache
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)
stats = ref.load_weight_stats()
for layer_idx, name in ((H.LINEAR_LAYER_IDX,'linear'), (H.FULL_LAYER_IDX,'full')):
    batch = 32
    seq_lens = [64 + 97*u for u in range(batch)]
    lut = H.build_layer(mesh, layer_idx, max_batch=batch, max_seq_len=8192)
    bad = []
    for u, sl in enumerate(seq_lens):
        hidden = ref.synthetic_hidden_states(lut.config, 1, sl, stats, seed=u)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden, user_id=u)
        v = H.pcc(golden, got)
        flag = "" if v >= 0.995 else "  <-- FAIL"
        print(f"{name} user={u:2d} seq={sl:5d} pcc={v:.6f}{flag}", flush=True)
    H.release_layers()
ttnn.close_mesh_device(mesh)
