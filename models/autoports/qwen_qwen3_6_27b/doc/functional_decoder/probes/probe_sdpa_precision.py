"""Why does `full_attention` prefill lose PCC at very long context, and does fp32 dest fix it?

The 262143-token test showed the paged K/V cache matching HF at PCC 0.99999 while the prefill
*output* for the last tokens fell to 0.970, so the loss is inside the attention itself, not in
what was written to the cache. This sweeps the context length and A/Bs the SDPA compute
config to find out which.
"""

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd

TAIL = 256
LENGTHS = [8191, 32767, 65535, 131071, 262143]

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()

for fp32_dest in (False, True):
    for length in LENGTHS:
        lut = H.build_layer(mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144)
        lut.tt_layer.sdpa_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest,
            packer_l1_acc=True,
        )
        hidden = ref.synthetic_hidden_states(lut.config, 1, length, stats, seed=0)
        try:
            got = H.run_tt_prefill(lut, hidden)
        except Exception as exc:  # noqa: BLE001 - an op-contract rejection is the answer
            print(f"RESULT fp32_dest={fp32_dest} len={length} FAILED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:140]}", flush=True)
            H.release_layers()
            continue

        cache = DynamicCache(config=lut.config)
        H.fill_reference_kv_cache(lut, hidden[:, : length - TAIL, :].contiguous(), cache)
        golden = H.reference_prefill(lut, hidden[:, length - TAIL :, :].contiguous(), cache)
        tail_pcc = H.pcc(golden, got[:, -TAIL:, :])

        keys, values = H.read_paged_kv(lut, 0, length)
        ref_k, ref_v = H.reference_cache_kv(lut, cache, length)
        print(
            f"RESULT fp32_dest={fp32_dest} len={length} tail_pcc={tail_pcc:.6f} "
            f"k_cache_pcc={H.pcc(ref_k, keys):.6f} v_cache_pcc={H.pcc(ref_v, values):.6f}",
            flush=True,
        )
        H.release_layers()
        del hidden, got, cache, golden, keys, values, ref_k, ref_v

ttnn.close_mesh_device(mesh)
