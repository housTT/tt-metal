import torch, ttnn
from transformers.cache_utils import DynamicCache
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

m = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)
stats = ref.load_weight_stats()
for layer_idx, name in ((H.FULL_LAYER_IDX,'full'), (H.LINEAR_LAYER_IDX,'linear')):
    lut = H.build_layer(m, layer_idx, max_batch=1, max_seq_len=4096)
    for sl in (735, 736, 737, 743, 767, 768):
        hidden = ref.synthetic_hidden_states(lut.config, 1, sl, stats, seed=0)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden)
        v = H.pcc(golden, got)
        err = (golden - got).abs().amax(dim=-1)[0]   # [seq]
        bad = (err > 0.05).nonzero().flatten()
        first_bad = int(bad[0]) if bad.numel() else -1
        pad = -(-sl // (256 if name=='full' else 64)) * (256 if name=='full' else 64) - sl
        print(f"{name} seq={sl} padamt={pad} pcc={v:.6f} first_bad_pos={first_bad} nbad={int(bad.numel())} "
              f"errmax={float(err.max()):.4f} err@0={float(err[0]):.5f} err@last={float(err[-1]):.5f}", flush=True)
    H.release_layers()
ttnn.close_mesh_device(m)
