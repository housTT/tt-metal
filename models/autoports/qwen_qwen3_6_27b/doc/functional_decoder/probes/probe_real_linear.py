"""Localize the real-weight linear_attention PCC failure.

Runs prefill at several lengths with real weights, reports prefill PCC, conv/recurrent state
PCC and decode PCC, plus the same numbers with synthetic weights as the control.
"""
import torch, ttnn
from transformers.cache_utils import DynamicCache
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()

for real in (True, False):
    for sl in (64, 128, 2049):
        lut = H.build_layer(m, H.LINEAR_LAYER_IDX, max_batch=1, max_seq_len=8192, real_weights=real)
        hidden = ref.synthetic_hidden_states(lut.config, 1, sl, stats, seed=0)
        cache = DynamicCache(config=lut.config)
        golden = H.reference_prefill(lut, hidden, cache)
        got = H.run_tt_prefill(lut, hidden)
        p = H.pcc(golden, got)

        conv_state, rec_state = H.read_linear_state(lut, 0)
        ref_conv = cache.layers[lut.layer_idx].conv_states[0].to(torch.float32)
        ref_rec = cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32)
        cp = H.pcc(ref_conv, conv_state.T)
        rp = H.pcc(ref_rec, rec_state)

        H.prepare_decode(lut)
        token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=7)
        gd = H.reference_decode(lut, token, sl, cache)
        gotd = H.run_tt_decode(lut, token, torch.tensor([sl]))
        dp = H.pcc(gd, gotd)

        print(
            f"RESULT real={real} seq={sl} prefill_pcc={p:.6f} conv_pcc={cp:.6f} "
            f"rec_pcc={rp:.6f} decode_pcc={dp:.6f} "
            f"ref_rec_absmax={float(ref_rec.abs().max()):.3e} tt_rec_absmax={float(rec_state.abs().max()):.3e}",
            flush=True,
        )
        H.release_layers()
ttnn.close_mesh_device(m)
