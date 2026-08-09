"""Localise the long-context ``full_attention`` prefill error to a single stage.

Runs one real prefill, intercepts the last chunk's ``chunked_scaled_dot_product_attention``
call, and then answers, for the last 256 query positions:

* is the device Q (projection + q_norm + partial RoPE) right?     -> ``q_tail_pcc``
* is the device K/V *for the tail positions specifically* right?   -> ``k_tail_pcc``/``v_tail_pcc``
  (the whole-cache PCC that earlier runs reported averages 262143 positions and cannot see a
  tail-only defect)
* is the SDPA kernel right *on its own inputs*?                    -> ``sdpa_selfpcc``
  (torch float32 attention over the device's own Q and the un-paged device K/V cache)
* how much of the final layer error is left after all of that?     -> ``layer_tail_pcc``

Env: LEN (default 262143), TAIL (default 256).
"""

import math
import os

import torch
import ttnn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tests import harness as H

LEN = int(os.environ.get("LEN", "262143"))
TAIL = int(os.environ.get("TAIL", "256"))

captured = {}
_orig_sdpa = ttnn.transformer.chunked_scaled_dot_product_attention


def _spy(q, k, v, page_table, chunk_start, **kw):
    out = _orig_sdpa(q, k, v, page_table, chunk_start, **kw)
    if chunk_start + int(q.shape[2]) >= LEN:  # the chunk that contains the tail
        captured["chunk_start"] = chunk_start
        captured["q"] = ttnn.to_torch(q).to(torch.float32)
        captured["out"] = ttnn.to_torch(out).to(torch.float32)
    return out


def attn_fp32(q_tail, keys, values, positions, n_q_heads, n_kv_heads):
    """float32 attention: q_tail [H, T, D], keys/values [n_kv, n, D] -> [H, T, D]."""
    group = n_q_heads // n_kv_heads
    n = keys.shape[1]
    out = torch.empty(n_q_heads, q_tail.shape[1], q_tail.shape[2], dtype=torch.float32)
    scale = q_tail.shape[2] ** -0.5
    idx = torch.arange(n)
    for h in range(n_q_heads):
        kvh = h // group
        scores = (q_tail[h] @ keys[kvh].t()) * scale
        scores.masked_fill_(idx[None, :] > positions[:, None], float("-inf"))
        out[h] = torch.softmax(scores, dim=-1) @ values[kvh]
        del scores
    return out


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    stats = ref.load_weight_stats()
    lut = H.build_layer(mesh, H.FULL_LAYER_IDX, max_batch=1, max_seq_len=262144)
    s = lut.tt_layer.shapes
    hidden = ref.synthetic_hidden_states(lut.config, 1, LEN, stats, seed=0)

    ttnn.transformer.chunked_scaled_dot_product_attention = _spy
    try:
        got = H.run_tt_prefill(lut, hidden)
    finally:
        ttnn.transformer.chunked_scaled_dot_product_attention = _orig_sdpa

    chunk_start = captured["chunk_start"]
    dev_q = captured["q"][0]  # [H, chunk_len, D]
    dev_attn = captured["out"][0]
    tail_rows = LEN - chunk_start - TAIL  # first tail row inside the chunk
    positions = torch.arange(LEN - TAIL, LEN)

    # ---- reference: HF cache for everything before the tail, then a real prefill of the tail
    cache = DynamicCache(config=lut.config)
    H.fill_reference_kv_cache(lut, hidden[:, : LEN - TAIL, :].contiguous(), cache)
    golden = H.reference_prefill(lut, hidden[:, LEN - TAIL :, :].contiguous(), cache)
    layer_tail_pcc = H.pcc(golden, got[:, -TAIL:, :])
    golden_tail_saved = golden.clone()
    del golden

    ref_k, ref_v = H.reference_cache_kv(lut, cache, LEN)
    del cache

    # ---- reference Q for the tail positions
    attn = lut.ref_layer.self_attn
    with torch.no_grad():
        normed = lut.ref_layer.input_layernorm(hidden[:, LEN - TAIL :, :].contiguous())
        q_full = attn.q_proj(normed).view(1, TAIL, s.num_attention_heads, 2 * s.head_dim)
        q_ref = attn.q_norm(q_full[..., : s.head_dim]).transpose(1, 2)  # [1, H, T, D]
        cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=1)
        q_ref, _ = apply_rotary_pos_emb(q_ref, q_ref, cos, sin)
    q_ref = q_ref[0].to(torch.float32)
    dev_q_tail = dev_q[:, tail_rows : tail_rows + TAIL, :]
    q_tail_pcc = H.pcc(q_ref, dev_q_tail)

    # ---- device K/V, un-paged
    keys, values = H.read_paged_kv(lut, 0, LEN)
    k_full_pcc, v_full_pcc = H.pcc(ref_k, keys), H.pcc(ref_v, values)
    k_tail_pcc = H.pcc(ref_k[:, -TAIL:, :], keys[:, -TAIL:, :])
    v_tail_pcc = H.pcc(ref_v[:, -TAIL:, :], values[:, -TAIL:, :])
    k_chunk_pcc = H.pcc(ref_k[:, chunk_start:LEN, :], keys[:, chunk_start:LEN, :])

    H.release_layers()
    ttnn.close_mesh_device(mesh)

    dev_attn_tail = dev_attn[:, tail_rows : tail_rows + TAIL, :]
    # (a) kernel on its own inputs: torch fp32 over device Q and device K/V
    self_gold = attn_fp32(dev_q_tail, keys, values, positions, s.num_attention_heads, s.num_key_value_heads)
    sdpa_selfpcc = H.pcc(self_gold, dev_attn_tail)
    del self_gold
    # (b) full attention stage vs reference inputs
    ref_gold = attn_fp32(q_ref, ref_k, ref_v, positions, s.num_attention_heads, s.num_key_value_heads)
    sdpa_refpcc = H.pcc(ref_gold, dev_attn_tail)
    # (c) how much of (b) is explained purely by the device's wrong Q?
    qonly_gold = attn_fp32(dev_q_tail, ref_k, ref_v, positions, s.num_attention_heads, s.num_key_value_heads)
    qonly_pcc = H.pcc(ref_gold, qonly_gold)
    # (d) ... and purely by the device's K/V?
    kvonly_gold = attn_fp32(q_ref, keys, values, positions, s.num_attention_heads, s.num_key_value_heads)
    kvonly_pcc = H.pcc(ref_gold, kvonly_gold)

    # ---- token-mean / token-deviation split of the attention output.
    # At very long context a near-uniform softmax makes the attention output almost the same
    # vector for every query token, so a flat PCC is dominated by that shared vector and hides
    # the per-token signal that the rest of the layer actually consumes.
    def split(t):
        m = t.mean(dim=-2, keepdim=True)
        return m, t - m

    ref_m, ref_d = split(ref_gold)
    dev_m, dev_d = split(dev_attn_tail)
    attn_mean_pcc = H.pcc(ref_m, dev_m)
    attn_dev_pcc = H.pcc(ref_d, dev_d)
    mean_over_dev = float(ref_m.norm() / ref_d.norm())

    torch.save(
        {
            "len": LEN,
            "chunk_start": chunk_start,
            "dev_attn_tail": dev_attn_tail,
            "ref_gold_attn": ref_gold,
            "dev_q_tail": dev_q_tail,
            "q_ref_tail": q_ref,
            "layer_got_tail": got[:, -TAIL:, :],
            "layer_golden_tail": golden_tail_saved,
        },
        f"{os.environ.get('DUMP_DIR', '/home/ttuser/dev/qwen/rundir/generated')}/sdpa_localise_{LEN}.pt",
    )

    print(
        f"RESULT len={LEN} chunk_start={chunk_start} layer_tail_pcc={layer_tail_pcc:.6f} "
        f"q_tail_pcc={q_tail_pcc:.6f} k_full_pcc={k_full_pcc:.6f} v_full_pcc={v_full_pcc:.6f} "
        f"k_tail_pcc={k_tail_pcc:.6f} v_tail_pcc={v_tail_pcc:.6f} k_chunk_pcc={k_chunk_pcc:.6f} "
        f"sdpa_selfpcc={sdpa_selfpcc:.6f} sdpa_refpcc={sdpa_refpcc:.6f} "
        f"qonly_pcc={qonly_pcc:.6f} kvonly_pcc={kvonly_pcc:.6f} "
        f"attn_mean_pcc={attn_mean_pcc:.6f} attn_dev_pcc={attn_dev_pcc:.6f} "
        f"mean_over_dev={mean_over_dev:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
