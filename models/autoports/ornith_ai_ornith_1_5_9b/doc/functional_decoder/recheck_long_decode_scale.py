"""Rerun the uninstrumented original test with original or calibrated input scale."""

import os

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import test_functional_decoder as H
from models.autoports.ornith_ai_ornith_1_5_9b.tests.test_contract_extensions import test_native_context_decode_oracle

scale = float(os.environ.get("PROBE_SCALE", ".5"))
order = os.environ.get("PROBE_ORDER", "float")


def make_activations(batch, seq_len, *, seed=0):
    x = torch.randn(batch, seq_len, H.hf_config().hidden_size, generator=torch.Generator().manual_seed(seed))
    return ((x.to(torch.bfloat16) if order == "bf16" else x) * scale).to(torch.bfloat16)


H.make_activations = make_activations
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
mesh.enable_program_cache()
try:
    print(f"RECHECK_SCALE={scale} ORDER={order}", flush=True)
    for _ in range(2):
        test_native_context_decode_oracle(mesh)
finally:
    ttnn.close_mesh_device(mesh)
