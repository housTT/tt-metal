import sys

import torch

import ttnn

sys.argv = ["x"]
from models.autoports.qwen_qwen3_8b.tests import test_functional_decoder as T
from models.autoports.qwen_qwen3_8b.tt import functional_decoder as FD
from models.autoports.qwen_qwen3_8b.tt.functional_decoder import plan_prefill_chunks

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    fx = T._layer(0)
    dec = T._decoder(mesh, fx)

    def run_len(seq_len, blocks, seed_pt=7, users=4, user_id=2, kv=None, page_table=None, table=None):
        if page_table is None:
            table, page_table = T._page_table(mesh, users, blocks, seed=seed_pt)
            kv = T._kv_cache(mesh, users * blocks)
        hidden = fx.inputs(seq_len + 1, seed=200 + seq_len)
        hf = fx.reference().forward(hidden[:, :seq_len], 0)
        out = T._to_torch(
            dec.prefill_forward(
                T._tt(mesh, hidden[:, :seq_len].unsqueeze(1)), page_table=page_table, kv_cache=kv, user_id=user_id
            )
        )[:, 0]
        tail_ok = T._pcc(hf[:, 8192:], out[:, 8192:]) > 0.99 if seq_len > 8192 else True
        print(f"len {seq_len} blocks {blocks} user {user_id}: pcc {T._pcc(hf, out):.5f} tail_ok {tail_ok}", flush=True)
        return kv, page_table, table, hidden, hf, out, tail_ok

    run_len(8225, 258)
    run_len(8225, 260)
    pt_seq = T._tt(
        mesh, torch.arange(258, dtype=torch.int32).reshape(1, 258), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    run_len(8225, 258, kv=T._kv_cache(mesh, 258), page_table=pt_seq, user_id=0)
    kv, pt, tb, *_ = run_len(8192, 258)
    kv, pt, tb, hidden, hf, out, tail_ok = run_len(8225, 258, kv=kv, page_table=pt, table=tb)
    print(
        "REPRODUCED with the decoder's own chunk selection"
        if not tail_ok
        else "NOT REPRODUCED with the decoder's own chunk selection (workaround active); forcing chunk 256 below",
        flush=True,
    )
    chunks = plan_prefill_chunks(0, 8225, FD.PREFILL_CHUNK_SIZE)
    c3 = chunks[2]
    ref = fx.reference()
    layer = ref.layer
    with torch.no_grad():
        xx = hidden[:, :8225].float()
        normed = layer.input_layernorm(xx)
        pos = torch.arange(8225).unsqueeze(0)
        cos, sin = ref.rotary(xx, pos)
        from transformers.masking_utils import create_causal_mask

        mask = create_causal_mask(
            config=fx.config, inputs_embeds=xx, attention_mask=None, past_key_values=None, position_ids=pos
        )
        attn_ref, _ = layer.self_attn(normed, position_embeddings=(cos, sin), attention_mask=mask)
        attn_ref = attn_ref[:, 8192:]
    x = dec._pad_sequence(T._tt(mesh, hidden[:, :8225].unsqueeze(1)), 0, 31)
    x3 = ttnn.slice(x, [0, 0, 8192, 0], [1, 1, 8256, 4096], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def chunk3(user_row, chunk_fn=None, label=""):
        orig = FD.prefill_sdpa_chunk_size
        if chunk_fn:
            FD.prefill_sdpa_chunk_size = lambda start, length, extent: chunk_fn(start)
        attn_in = dec._rms_norm(x3, dec.weights.input_ln)
        o = dec._attention_prefill(attn_in, c3, user_row, kv)
        FD.prefill_sdpa_chunk_size = orig
        t = T._to_torch(o)[:, 0][:, :33]
        print(f"   chunk3 {label}: pcc {T._pcc(attn_ref, t):.5f} max {float(t.float().abs().max()):.3g}", flush=True)

    row258 = ttnn.slice(pt, [2, 0], [3, 258])
    row264 = T._tt(
        mesh,
        torch.cat([tb[2], torch.zeros(6, dtype=torch.int32)]).reshape(1, 264),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    force256 = lambda s: 256
    chunk3(row258, chunk_fn=force256, label="row 258, forced chunk 256 (unsafe: padded K end 8448 > extent 8256)")
    chunk3(row264, chunk_fn=force256, label="row 264 (slack), forced chunk 256")
    chunk3(row258, chunk_fn=force256, label="row 258, forced chunk 256 (again)")
    chunk3(row258, chunk_fn=lambda s: 64, label="row 258, chunk 64")
    chunk3(row258, chunk_fn=force256, label="row 258, forced chunk 256 (again 2)")
    chunk3(row258, chunk_fn=lambda s: 32, label="row 258, chunk 32")
    chunk3(row258, label="row 258, decoder's own selection (64)")
finally:
    ttnn.close_mesh_device(mesh)
