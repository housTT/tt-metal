"""Is the 262143-token accuracy loss driven by the *page table size*, not the merge count?

The error does not grow smoothly with context: with fp32 destination accumulation the tail
PCC is 0.9991 at 65535, 0.9989 at 131071 and 0.9838 at 262143. Doubling the context doubles
the online-softmax merge steps, which cannot explain a 15x jump in error, so the suspect is
the other thing that doubled: the page table handed to
`chunked_scaled_dot_product_attention` (2048 -> 4096 entries at block_size 64).

`block_size` is a constructor parameter, so the same 262143-token context can be expressed
with 4096, 2048, 1024 or 512 blocks. If accuracy tracks the block *count* rather than the
context, this is a page-table-size effect and a larger block size is the fix.
"""

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

TAIL = 256
LENGTH = 262143

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()

for block_size in (64, 128, 256, 512):
    lut = H.build_layer(
        mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144, block_size=block_size
    )
    blocks = 262144 // block_size
    hidden = ref.synthetic_hidden_states(lut.config, 1, LENGTH, stats, seed=0)
    try:
        got = H.run_tt_prefill(lut, hidden)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT block_size={block_size} blocks={blocks} FAILED {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:130]}", flush=True)
        H.release_layers()
        continue
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : LENGTH - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, LENGTH - TAIL :, :].contiguous(), cache)
    keys, values = H.read_paged_kv(lut, 0, LENGTH)
    ref_k, ref_v = H.reference_cache_kv(lut, cache, LENGTH)
    print(
        f"RESULT block_size={block_size} blocks={blocks} "
        f"tail_pcc={H.pcc(golden, got[:, -TAIL:, :]):.6f} "
        f"k_cache_pcc={H.pcc(ref_k, keys):.6f} v_cache_pcc={H.pcc(ref_v, values):.6f}",
        flush=True,
    )
    H.release_layers()
    del hidden, got, cache, golden, keys, values, ref_k, ref_v

ttnn.close_mesh_device(mesh)
