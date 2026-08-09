"""Does a larger SDPA k-chunk recover long-context prefill accuracy?

`probe_sdpa_precision.py` showed the loss is inside the attention (the paged K/V cache stays
at PCC 0.99999) and grows sharply past ~65k keys, which points at the online-softmax merge:
the number of merge steps is `context / k_chunk_size`, so a larger chunk means fewer
accumulation steps. This sweeps the chunk size at the full 262143-token context, with fp32
destination accumulation on.
"""

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd

TAIL = 256
LENGTH = 262143

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()
original = fd.SDPA_CHUNK

for chunk in (256, 512, 1024, 2048):
    fd.SDPA_CHUNK = chunk
    lut = H.build_layer(mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144)
    hidden = ref.synthetic_hidden_states(lut.config, 1, LENGTH, stats, seed=0)
    try:
        got = H.run_tt_prefill(lut, hidden)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT sdpa_chunk={chunk} FAILED {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:160]}", flush=True)
        H.release_layers()
        continue
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : LENGTH - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, LENGTH - TAIL :, :].contiguous(), cache)
    print(
        f"RESULT sdpa_chunk={chunk} merge_steps={LENGTH // chunk} "
        f"tail_pcc={H.pcc(golden, got[:, -TAIL:, :]):.6f}",
        flush=True,
    )
    H.release_layers()
    del hidden, got, cache, golden

fd.SDPA_CHUNK = original
ttnn.close_mesh_device(mesh)
