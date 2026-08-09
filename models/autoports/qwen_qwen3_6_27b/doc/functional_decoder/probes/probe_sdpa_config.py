"""Which SDPA knob recovers 262143-token prefill accuracy?

`probe_sdpa_precision.py` localised the loss to the attention op itself (paged K/V cache PCC
stays at 0.99999) and showed fp32 destination accumulation lifts the tail PCC from 0.970 to
0.984 - better but still under the 0.995 bar. `probe_sdpa_chunk.py` showed a larger *square*
q/k chunk does not fit L1. This sweeps the remaining levers: asymmetric q/k chunks, the
compute grid, and `packer_l1_acc`, which accumulates matmul partials in L1 at the packer's
output precision and is therefore a prime suspect for a long online-softmax merge.
"""

import itertools

import torch
import ttnn
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H
from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd

TAIL = 256
LENGTH = 262143
# (q_chunk, k_chunk, grid, packer_l1_acc)
CANDIDATES = [
    (256, 256, 8, True),   # current default, for reference
    (256, 256, 8, False),
    (128, 512, 8, False),
    (128, 512, 8, True),
    (256, 512, 8, False),
    (64, 1024, 8, False),
    (256, 256, 7, False),
]

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
stats = ref.load_weight_stats()
original = fd._sdpa_program_config

for q_chunk, k_chunk, grid, packer in CANDIDATES:
    def program_config(chunk_start_idx, _q=q_chunk, _k=k_chunk, _g=grid):
        assert chunk_start_idx % _q == 0
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_g, _g),
            q_chunk_size=_q,
            k_chunk_size=_k,
            exp_approx_mode=False,
        )

    fd._sdpa_program_config = program_config
    lut = H.build_layer(mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144)
    lut.tt_layer.sdpa_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer,
    )
    label = f"q={q_chunk} k={k_chunk} grid={grid}x{grid} packer_l1_acc={packer}"
    hidden = ref.synthetic_hidden_states(lut.config, 1, LENGTH, stats, seed=0)
    try:
        got = H.run_tt_prefill(lut, hidden)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT {label} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:130]}", flush=True)
        H.release_layers()
        continue
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : LENGTH - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, LENGTH - TAIL :, :].contiguous(), cache)
    print(f"RESULT {label} tail_pcc={H.pcc(golden, got[:, -TAIL:, :]):.6f}", flush=True)
    H.release_layers()
    del hidden, got, cache, golden

fd._sdpa_program_config = original
ttnn.close_mesh_device(mesh)
