"""Last untried SDPA lever: does an fp32 K/V cache and fp32 query fix 262143-token prefill?

Every program-config knob is exhausted - `packer_l1_acc`, the compute grid and the KV block
size all leave the tail PCC at exactly 0.983820, and any k-chunk above 256 fails to fit L1 at
head_dim 256. fp32 destination accumulation was the only thing that moved it (0.970 ->
0.984), which says the loss is in accumulation precision, so the remaining lever is the
precision of the operands themselves.
"""

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

TAIL = 256
LENGTH = 262143
original_sdpa = ttnn.transformer.chunked_scaled_dot_product_attention

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()

for cache_dtype, cast_q in ((ttnn.bfloat16, False), (ttnn.float32, False), (ttnn.float32, True)):
    def sdpa(q, k, v, *args, _cast=cast_q, **kwargs):
        return original_sdpa(ttnn.typecast(q, ttnn.float32) if _cast else q, k, v, *args, **kwargs)

    ttnn.transformer.chunked_scaled_dot_product_attention = sdpa
    label = f"cache_dtype={str(cache_dtype).split('.')[-1]} cast_q_to_fp32={cast_q}"
    lut = H.build_layer(
        mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144, cache_dtype=cache_dtype
    )
    hidden = ref.synthetic_hidden_states(lut.config, 1, LENGTH, stats, seed=0)
    try:
        got = H.run_tt_prefill(lut, hidden)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT {label} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:140]}", flush=True)
        H.release_layers()
        continue
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : LENGTH - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, LENGTH - TAIL :, :].contiguous(), cache)
    print(f"RESULT {label} tail_pcc={H.pcc(golden, got[:, -TAIL:, :]):.6f}", flush=True)
    H.release_layers()
    del hidden, got, cache, golden

ttnn.transformer.chunked_scaled_dot_product_attention = original_sdpa
ttnn.close_mesh_device(mesh)
