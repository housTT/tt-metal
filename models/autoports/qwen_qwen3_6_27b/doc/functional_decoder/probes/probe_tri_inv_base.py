"""Model-level PCC and wall time of `TRI_INV_BASE` in {8, 16, 32} on real weights.

`probe_blockinv.py` measured only the accuracy of the inverse itself. This measures what
actually matters: the decoder layer's prefill/decode PCC against the HF reference with real
checkpoint weights, and the warmed prefill wall time, since `tt-perf-report` shows the
recursion's single-core 32x32 matmuls are ~43% of `linear_attention` prefill device time.
"""

import time

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd

SEQ = 2048
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()
original = fd.TRI_INV_BASE

for base in (8, 16, 32):
    fd.TRI_INV_BASE = base
    lut = H.build_layer(mesh, H.LINEAR_LAYER_IDX, max_batch=1, max_seq_len=8192, real_weights=True)
    hidden = ref.synthetic_hidden_states(lut.config, 1, SEQ, stats, seed=0)
    cache = DynamicCache(config=lut.config)
    golden = H.reference_prefill(lut, hidden, cache)
    got = H.run_tt_prefill(lut, hidden)
    prefill_pcc = H.pcc(golden, got)

    _, recurrent = H.read_linear_state(lut, 0)
    state_pcc = H.pcc(cache.layers[lut.layer_idx].recurrent_states[0].to(torch.float32), recurrent)

    H.prepare_decode(lut)
    token = ref.synthetic_hidden_states(lut.config, 1, 1, stats, seed=7)
    decode_pcc = H.pcc(H.reference_decode(lut, token, SEQ, cache), H.run_tt_decode(lut, token, torch.tensor([SEQ])))

    tt_in = H.tt_hidden_prefill(hidden, mesh)
    for _ in range(2):  # compile + warm
        ttnn.deallocate(lut.tt_layer.prefill_forward(tt_in, user_id=0))
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(3):
        ttnn.deallocate(lut.tt_layer.prefill_forward(tt_in, user_id=0))
    ttnn.synchronize_device(mesh)
    warmed_ms = (time.perf_counter() - start) * 1e3 / 3

    print(
        f"RESULT TRI_INV_BASE={base:2d} real_weight_prefill_pcc={prefill_pcc:.6f} "
        f"recurrent_state_pcc={state_pcc:.6f} decode_pcc={decode_pcc:.6f} "
        f"warmed_prefill_ms={warmed_ms:.1f}",
        flush=True,
    )
    H.release_layers()

fd.TRI_INV_BASE = original
ttnn.close_mesh_device(mesh)
