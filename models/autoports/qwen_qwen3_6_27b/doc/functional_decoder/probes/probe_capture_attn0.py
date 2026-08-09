"""Capture the real ``attn0`` matrices of the gated delta rule and measure the doubling growth.

Runs the HF reference for layer 0 with the **real** checkpoint weights, intercepts
``torch_chunk_gated_delta_rule`` to recover the strictly-lower-triangular ``attn0`` it builds,
and prints how large the Neumann doubling product's intermediates get before they cancel back
down to the (well-conditioned) inverse.  Writes ``/tmp/attn0_real.pt`` for
``probe_blockinv.py``.  CPU only; no device needed.
"""
import math

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from transformers.cache_utils import DynamicCache

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref

CAPTURED = {}
_original = M.torch_chunk_gated_delta_rule


def capturing(query, key, value, g, beta, chunk_size=64, **kwargs):
    q = M.l2norm(query, dim=-1, eps=1e-6)
    k = M.l2norm(key, dim=-1, eps=1e-6)
    q, k, v, b, gg = [x.transpose(1, 2).contiguous().float() for x in (q, k, value, beta, g)]
    k_beta = k * b.unsqueeze(-1)
    kc = k.reshape(k.shape[0], k.shape[1], -1, chunk_size, k.shape[-1])
    kbc = k_beta.reshape(k_beta.shape[0], k_beta.shape[1], -1, chunk_size, k_beta.shape[-1])
    gcum = gg.reshape(gg.shape[0], gg.shape[1], -1, chunk_size).cumsum(dim=-1)
    decay = ((gcum.unsqueeze(-1) - gcum.unsqueeze(-2)).tril().exp().float()).tril()
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    CAPTURED.setdefault("attn0", -((kbc @ kc.transpose(-1, -2)) * decay).masked_fill(upper, 0))
    return _original(query, key, value, g, beta, chunk_size=chunk_size, **kwargs)


config = ref.load_text_config()
layer = ref.build_reference_layer(0)
layer.linear_attn.chunk_gated_delta_rule = capturing
hidden = ref.synthetic_hidden_states(config, 1, 256, ref.load_weight_stats(), seed=0)
positions = torch.arange(256)
cos, sin = ref.text_position_embeddings(ref.make_rotary(config), positions, batch=1)
with torch.no_grad():
    layer(
        hidden,
        position_embeddings=(cos, sin),
        attention_mask=None,
        position_ids=ref.build_text_position_ids(positions, 1)[0],
        past_key_values=DynamicCache(config=config),
        use_cache=True,
    )

A = CAPTURED["attn0"][0].reshape(-1, 64, 64).double()
identity = torch.eye(64, dtype=torch.float64)
exact = torch.linalg.solve_triangular(identity - A, identity.expand(A.shape[0], 64, 64).contiguous(), upper=False)
print(f"attn0 batch={A.shape[0]} absmax={float(A.abs().max()):.4f} " f"exact |inv|max={float(exact.abs().max()):.4f}")

power, partial = A.clone(), A + identity
print(f"step 0: |A^1|max={float(A.abs().max()):.4e} |partial inv|max={float(partial.abs().max()):.4e}")
for j in range(int(math.log2(64)) - 1):
    power = power @ power
    partial = partial @ (power + identity)
    print(
        f"step {j + 1}: |A^{2 ** (j + 1)}|max={float(power.abs().max()):.4e} "
        f"|partial inv|max={float(partial.abs().max()):.4e}"
    )
print(f"float64 doubling error: {float((partial - exact).abs().max()):.3e}")

torch.save(A.float(), "/tmp/attn0_real.pt")
print("wrote /tmp/attn0_real.pt")
