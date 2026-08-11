# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Feasibility probes for the fused-decoder candidate ops, at Ornith's real shapes.

Each probe answers one question the graph-fusing skill's Step 1/Step 4 needs before a rewrite
is worth attempting: does the dedicated op accept our shapes/dtypes/layouts at all, and does it
compute the same thing as the primitive sequence it would replace?

Run:  python models/autoports/ornith_ai_ornith_1_0_35b/doc/fused_decoder/logs/probe_fused_ops.py
"""

import traceback

import torch

import ttnn

TILE = 32
HIDDEN = 2048
N_HEADS, N_KV, HEAD_DIM, ROPE_DIM = 16, 2, 256, 64
GDN_NK, GDN_NV, GDN_DK, GDN_DV = 16, 32, 128, 128


def pcc(a, b):
    a = a.double().flatten() - a.double().flatten().mean()
    b = b.double().flatten() - b.double().flatten().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def dev(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mc)


RESULTS = []


def probe(name):
    def wrap(fn):
        def run(device):
            try:
                detail = fn(device)
                RESULTS.append((name, "OK", detail))
                print(f"[OK]   {name}: {detail}")
            except Exception as exc:  # noqa: BLE001 - probing for support
                msg = str(exc).strip().splitlines()
                msg = msg[0] if msg else repr(exc)
                RESULTS.append((name, "FAIL", msg[:400]))
                print(f"[FAIL] {name}: {msg[:400]}")
                traceback.print_exc(limit=1)

        return run

    return wrap


def rotate_half_ref(x, cos, sin):
    d = x.shape[-1] // 2
    rot = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return x * cos + rot * sin


@probe("rotary_embedding_hf prefill b=1 head_dim=64")
def p_rope_prefill(device):
    b, h, t = 1, N_HEADS, 128
    x = torch.randn(b, h, t, ROPE_DIM)
    cos = torch.cos(torch.randn(1, 1, t, ROPE_DIM))
    sin = torch.sin(torch.randn(1, 1, t, ROPE_DIM))
    out = ttnn.experimental.rotary_embedding_hf(dev(x, device), dev(cos, device), dev(sin, device))
    got = ttnn.to_torch(out).float()
    return f"pcc={pcc(rotate_half_ref(x, cos, sin), got):.6f} shape={list(out.shape)}"


@probe("rotary_embedding_hf prefill b=4 head_dim=64")
def p_rope_prefill_b4(device):
    b, h, t = 4, N_HEADS, 128
    x = torch.randn(b, h, t, ROPE_DIM)
    cos = torch.cos(torch.randn(1, 1, t, ROPE_DIM))
    sin = torch.sin(torch.randn(1, 1, t, ROPE_DIM))
    out = ttnn.experimental.rotary_embedding_hf(dev(x, device), dev(cos, device), dev(sin, device))
    got = ttnn.to_torch(out).float()
    return f"pcc={pcc(rotate_half_ref(x, cos, sin), got):.6f} shape={list(out.shape)}"


@probe("rotary_embedding_hf decode [1,B,H,64] per-user cos/sin")
def p_rope_decode(device):
    b, h = 4, N_HEADS
    x = torch.randn(1, b, h, ROPE_DIM)
    cos = torch.cos(torch.randn(1, b, 1, ROPE_DIM))
    sin = torch.sin(torch.randn(1, b, 1, ROPE_DIM))
    out = ttnn.experimental.rotary_embedding_hf(dev(x, device), dev(cos, device), dev(sin, device), is_decode_mode=True)
    got = ttnn.to_torch(out).float()
    return f"pcc={pcc(rotate_half_ref(x, cos, sin), got):.6f} shape={list(out.shape)}"


@probe("rotary_embedding_hf decode interleaved (no shard)")
def p_rope_decode_interleaved(device):
    b, h = 4, N_HEADS
    x = torch.randn(1, b, h, ROPE_DIM)
    cos = torch.cos(torch.randn(1, b, 1, ROPE_DIM))
    sin = torch.sin(torch.randn(1, b, 1, ROPE_DIM))
    out = ttnn.experimental.rotary_embedding_hf(
        dev(x, device), dev(cos, device), dev(sin, device), is_decode_mode=False
    )
    got = ttnn.to_torch(out).float()
    return f"pcc(is_decode_mode=False on decode shape)={pcc(rotate_half_ref(x, cos, sin), got):.6f}"


@probe("nlp_create_qkv_heads prefill GQA 16/2 hd=256")
def p_create_heads_prefill(device):
    b, s = 1, 128
    fused = torch.randn(b, 1, s, HEAD_DIM * (N_HEADS + 2 * N_KV))
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        dev(fused, device), num_heads=N_HEADS, num_kv_heads=N_KV, transpose_k_heads=False
    )
    qr = fused[:, 0, :, : N_HEADS * HEAD_DIM].reshape(b, s, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    return f"q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} " f"pcc_q={pcc(qr, ttnn.to_torch(q).float()):.6f}"


@probe("nlp_create_qkv_heads_decode GQA 16/2 hd=256 b=4")
def p_create_heads_decode(device):
    b = 4
    width = HEAD_DIM * (N_HEADS + 2 * N_KV)
    fused = torch.randn(1, 1, b, width)
    tt = dev(fused, device)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(tt, num_heads=N_HEADS, num_kv_heads=N_KV)
    qr = fused[0, 0, :, : N_HEADS * HEAD_DIM].reshape(1, b, N_HEADS, HEAD_DIM)
    got = ttnn.to_torch(q).float()[:, :, :N_HEADS, :]
    return f"q={list(q.shape)} k={list(k.shape)} pcc_q={pcc(qr, got):.6f}"


@probe("nlp_concat_heads_decode [1,B,32,128]")
def p_concat_decode(device):
    b, h, d = 4, GDN_NV, GDN_DV
    x = torch.randn(1, b, h, d)
    out = ttnn.experimental.nlp_concat_heads_decode(dev(x, device), num_heads=h)
    got = ttnn.to_torch(out).float()
    return f"shape={list(out.shape)} pcc={pcc(x.reshape(1, 1, b, h * d), got.reshape(1, 1, b, h * d)):.6f}"


@probe("nlp_concat_heads prefill [B,H,S,D]")
def p_concat_prefill(device):
    b, h, s, d = 1, N_HEADS, 128, HEAD_DIM
    x = torch.randn(b, h, s, d)
    out = ttnn.experimental.nlp_concat_heads(dev(x, device))
    ref = x.permute(0, 2, 1, 3).reshape(b, 1, s, h * d)
    return f"shape={list(out.shape)} pcc={pcc(ref, ttnn.to_torch(out).float()):.6f}"


@probe("ttnn.multiply input_tensor_a_activations=[SILU]")
def p_mul_act(device):
    a = torch.randn(1, 1, 64, 512)
    b = torch.randn(1, 1, 64, 512)
    out = ttnn.multiply(dev(a, device), dev(b, device), input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ref = torch.nn.functional.silu(a) * b
    return f"pcc={pcc(ref, ttnn.to_torch(out).float()):.6f}"


@probe("gdn output gate: SILU folded into the multiply vs applied separately, by |z|")
def p_gdn_gate_fold(device):
    """The one op-merging pattern this stage rejects on the DeltaNet output gate.

    `models/demos/blackhole/qwen36/tt/gdn/tp.py:31-34` says folding the SiLU here
    "overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small
    inputs)". The existing `input_tensor_a_activations=[SILU]` probe above is a hidden case in
    exactly that sense: `torch.randn(1, 1, 64, 512)`, both operands bfloat16.

    The hiding factor turns out to be **dtype, not magnitude**. The real gate multiplies the
    DeltaNet output — which `chunk_gated_delta_rule` returns in FLOAT32 — by a BFLOAT16 `z`, so it
    is a *mixed-dtype* binary op. This probe therefore runs the gate's own shapes (prefill
    ``[1, 2048, linear_v_dim]`` and decode ``[1, 1, linear_v_dim]``) under **both** dtype pairings
    and sweeps |z|, so the log shows the matched-bfloat16 arm agreeing and the real
    float32 x bfloat16 pairing diverging. Both arms are compared against a float32
    ``x * silu(z)`` reference, and non-finite outputs are counted rather than folded into a PCC
    that would read as NaN.
    """
    d = GDN_NV * GDN_DV
    gen = torch.Generator().manual_seed(23)
    out = []
    for label, s, x_dtype in (
        ("prefill f32xbf16(real)", 2048, ttnn.float32),
        ("prefill bf16xbf16", 2048, ttnn.bfloat16),
        ("decode  f32xbf16(real)", 1, ttnn.float32),
    ):
        x = torch.randn(1, s, d, generator=gen)
        xt = dev(x, device, dtype=x_dtype)
        for scale in (1.0, 32.0, 128.0):
            z = torch.randn(1, s, d, generator=gen) * scale
            zt = dev(z, device)
            ref = x.float() * torch.nn.functional.silu(z.float())
            sep = ttnn.to_torch(ttnn.multiply(xt, ttnn.silu(zt))).float()
            fold = ttnn.to_torch(ttnn.multiply(xt, zt, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])).float()
            ttnn.deallocate(zt)
            nf_sep = int((~torch.isfinite(sep)).sum())
            nf_fold = int((~torch.isfinite(fold)).sum())
            p_sep = pcc(ref, sep) if nf_sep == 0 else float("nan")
            p_fold = pcc(ref, fold) if nf_fold == 0 else float("nan")
            out.append(
                f"GATEFOLD {label} max|z|={float(z.abs().max()):7.2f} separate(shipped) pcc={p_sep:.6f} "
                f"nonfinite={nf_sep} | folded pcc={p_fold:.6f} nonfinite={nf_fold}"
            )
            print(out[-1], flush=True)
        ttnn.deallocate(xt)
    b, s = 1, 2048
    x = torch.randn(b, s, d, generator=gen)
    xt = dev(x, device)

    # Latency, once the two arms are known to agree: the fold is worth taking only if it is faster.
    import time

    z = torch.randn(b, s, d, generator=gen)
    zt = dev(z, device)

    def separate():
        act = ttnn.silu(zt)
        res = ttnn.multiply(xt, act)
        ttnn.deallocate(act)
        return res

    def folded():
        return ttnn.multiply(xt, zt, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])

    for name, fn in (("separate(shipped)", separate), ("folded", folded)):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(device)
        start = time.time()
        for _ in range(20):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(device)
        line = f"GATEFOLDTIME {name:18s} {(time.time() - start) / 20 * 1e6:8.1f} us per gate at [{b}, {s}, {d}]"
        out.append(line)
        print(line, flush=True)
    return " ;; ".join(out)


@probe("ttnn.multiply input_tensor_b_activations=[SIGMOID]")
def p_mul_act_b(device):
    a = torch.randn(1, 1, 64, 512)
    b = torch.randn(1, 1, 64, 512)
    out = ttnn.multiply(dev(a, device), dev(b, device), input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ref = a * torch.sigmoid(b)
    return f"pcc={pcc(ref, ttnn.to_torch(out).float()):.6f}"


@probe("ttnn.linear activation='silu'")
def p_linear_act(device):
    a = torch.randn(1, 1, 64, 512)
    w = torch.randn(1, 1, 512, 256)
    out = ttnn.linear(dev(a, device), dev(w, device), activation="silu")
    ref = torch.nn.functional.silu(a @ w)
    return f"pcc={pcc(ref, ttnn.to_torch(out).float()):.6f}"


@probe("chunk_gated_delta_rule use_qk_l2norm=True")
def p_gdn_l2(device):
    from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

    b, t = 1, 128
    eye, tril, ones, masks = build_fused_const_tiles(device, _FUSED_CHUNK_SIZE)
    q = torch.randn(b, t, GDN_NK, GDN_DK)
    k = torch.randn(b, t, GDN_NK, GDN_DK)
    v = torch.randn(b, t, GDN_NV, GDN_DV)
    g = -torch.rand(b, t, GDN_NV).float()
    beta = torch.rand(b, t, GDN_NV).float()
    o, state = ttnn.transformer.chunk_gated_delta_rule(
        dev(q, device),
        dev(k, device),
        dev(v, device),
        dev(g, device, dtype=ttnn.float32),
        dev(beta, device, dtype=ttnn.float32),
        output_final_state=True,
        chunk_size=_FUSED_CHUNK_SIZE,
        use_qk_l2norm=True,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    return f"o={list(o.shape)} layout={o.layout} state={list(state.shape)}"


@probe("chunk_gated_delta_rule output_head_major=True")
def p_gdn_head_major(device):
    from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

    b, t = 1, 128
    eye, tril, ones, masks = build_fused_const_tiles(device, _FUSED_CHUNK_SIZE)
    # The non-flat contract requires q/k to be L2-normalised on the host — that is exactly what the
    # op's own TT_FATAL for `use_qk_l2norm` says. Feeding raw gaussians instead makes the recurrence
    # diverge and the reference come back NaN, which is a harness bug, not an op one.
    q = torch.nn.functional.normalize(torch.randn(b, t, GDN_NK, GDN_DK), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, t, GDN_NK, GDN_DK), dim=-1)
    v = torch.randn(b, t, GDN_NV, GDN_DV)
    g = -torch.rand(b, t, GDN_NV).float() * 0.05
    beta = torch.rand(b, t, GDN_NV).float()
    args = (
        dev(q, device),
        dev(k, device),
        dev(v, device),
        dev(g, device, dtype=ttnn.float32),
        dev(beta, device, dtype=ttnn.float32),
    )
    kw = dict(
        output_final_state=True,
        chunk_size=_FUSED_CHUNK_SIZE,
        use_qk_l2norm=False,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o_ref, _ = ttnn.transformer.chunk_gated_delta_rule(*args, **kw)
    o_hm, _ = ttnn.transformer.chunk_gated_delta_rule(*args, output_head_major=True, **kw)
    # The default output is token-major [B, T, HV, V] ROW_MAJOR; head-major is [B*HV, T, V] TILE.
    # Compare them as the *same* values in the two layouts, and report the shared-value spread as
    # well as PCC, because a degenerate (constant) reference would make PCC nan rather than 1.
    ref = ttnn.to_torch(o_ref).float().reshape(b, t, GDN_NV, GDN_DV).permute(0, 2, 1, 3).reshape(b * GDN_NV, t, GDN_DV)
    got = ttnn.to_torch(o_hm).float().reshape(b * GDN_NV, t, GDN_DV)
    max_abs = float((ref - got).abs().max())
    return (
        f"head_major={list(o_hm.shape)} layout={o_hm.layout} pcc={pcc(ref, got):.6f} "
        f"max_abs_diff_vs_token_major={max_abs:.3e} ref_std={float(ref.std()):.4f}"
    )


@probe("sparse_matmul fused_activation=SILU")
def p_sparse_act(device):
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.moe import _sparse_matmul_config

    E, H, I, groups = 8, 512, 128, 2
    a = torch.randn(1, groups, TILE, H)
    w = torch.randn(1, E, H, I)
    sparsity = torch.ones(1, groups, 1, E)
    cfg = _sparse_matmul_config(TILE, I, 4)
    cfg_act = _sparse_matmul_config(TILE, I, 4)
    cfg_act.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    common = dict(
        nnz=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([TILE, TILE]),
        dtype=ttnn.bfloat16,
    )
    at, wt = dev(a, device), dev(w, device)
    st = dev(sparsity, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    plain = ttnn.sparse_matmul(at, wt, sparsity=st, program_config=cfg, **common)
    fused = ttnn.sparse_matmul(at, wt, sparsity=st, program_config=cfg_act, **common)
    ref = torch.nn.functional.silu(ttnn.to_torch(plain).float())
    return f"pcc(silu(plain) vs fused)={pcc(ref, ttnn.to_torch(fused).float()):.6f}"


@probe("deepseek_moe_fast_reduce_nc dim=1")
def p_ds_reduce(device):
    E, tokens, H = 16, 32, 512
    x = torch.randn(1, E, tokens, H)
    out = ttnn.experimental.deepseek_moe_fast_reduce_nc(dev(x, device), dim=1, split_size=H)
    got = out[0] if isinstance(out, (list, tuple)) else out
    return f"n_out={len(out) if isinstance(out,(list,tuple)) else 1} shape={list(got.shape)} pcc={pcc(x.sum(1, keepdim=True), ttnn.to_torch(got).float()):.6f}"


@probe("ttnn.where dense top-k mask (scatter replacement)")
def p_where_router(device):
    tokens, E, k = 32, 256, 8
    logits = torch.randn(1, 1, tokens, E)
    lt = dev(logits, device, dtype=ttnn.float32)
    vals, _ = ttnn.topk(lt, k=k, dim=-1, sorted=True)
    thr = ttnn.slice(vals, [0, 0, 0, k - 1], [1, 1, tokens, k])
    keep = ttnn.ge(lt, thr)
    masked = ttnn.where(keep, lt, float("-inf"))
    dense = ttnn.softmax(masked, dim=-1, numeric_stable=True)
    got = ttnn.to_torch(dense).float()
    v, i = torch.topk(logits, k, dim=-1)
    ref = torch.zeros_like(logits).scatter_(-1, i, torch.softmax(v, dim=-1))
    return f"pcc={pcc(ref, got):.6f} max_abs_err={float((ref - got).abs().max()):.3e}"


PROBES = [
    p_rope_prefill,
    p_rope_prefill_b4,
    p_rope_decode,
    p_rope_decode_interleaved,
    p_create_heads_prefill,
    p_create_heads_decode,
    p_concat_decode,
    p_concat_prefill,
    p_mul_act,
    p_gdn_gate_fold,
    p_mul_act_b,
    p_linear_act,
    p_gdn_l2,
    p_gdn_head_major,
    p_sparse_act,
    p_ds_reduce,
    p_where_router,
]


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    try:
        for fn in PROBES:
            fn(device)
    finally:
        ttnn.close_mesh_device(device)
    print("\n=== SUMMARY ===")
    for name, status, detail in RESULTS:
        print(f"{status:5s} {name}: {detail}")


if __name__ == "__main__":
    main()
