# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device (traceable) MLA-v4 single-token decode attention — increment 1 of the traced
decode (task #36). Everything stays ON DEVICE between ops (no host round-trip), so a whole
layer can later be captured in a Metal-Trace. Sliding-attention layers first (compressor=None).

Reproduces tt/attention.py::mla_attention / kv_cache_decode.py::mla_attention_decode exactly,
moving the three host fallbacks on-device:
  * interleaved partial RoPE  -> rotate_half(x)=x@R with a fixed [rd,rd] signed-permutation R
    (R[2i,2i+1]=+1, R[2i+1,2i]=-1), so rope = rope*cos + (rope@R)*sin  (all ttnn ops).
  * sink softmax -> ttnn.concat([scores, sink_col], -1); ttnn.softmax(-1); slice[..., :Lkv].
  * -sin un-rotation of the output -> same rope op with sin negated.

The rope slice boundary hd-rd = 512-64 = 448 = 14*32 is tile-aligned, so ttnn.slice is clean.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

import torch

import ttnn
from models.demos.deepseek_v4.reference import real_weights as RW


def _bf4_cache_stamp(mesh):
    """Fingerprint everything the on-disk tilized-bf4 expert bytes depend on, so the cache
    auto-invalidates on a change that matters (and stays warm otherwise). Components:
      model snapshot id + mesh device count C + reorder version + ttnn/tt-metal build fingerprint.
    (ttnn.as_tensor ALSO catches a flatbuffer load failure and rebuilds, so this is belt+suspenders
    against a silent wrong-load after a tt-metal upgrade.)"""
    try:
        snap = os.path.basename(os.path.normpath(RW.find_snapshot()))
    except Exception:
        snap = "unknown"
    C = getattr(mesh, "get_num_devices", lambda: 1)()
    # tt-metal build fingerprint: size+mtime of the compiled ttnn extension (changes on rebuild)
    try:
        so = ttnn._ttnn.__file__
        stt = os.stat(so)
        fmt = hashlib.sha1(f"{stt.st_size}:{int(stt.st_mtime)}".encode()).hexdigest()[:8]
    except Exception:
        fmt = "nofmt"
    REORDER_VERSION = "rv1"  # bump if the gate_up column-reorder / shard layout changes
    return f"{snap}_C{C}_{REORDER_VERSION}_ttnn{fmt}"


def _mesh_n(device):
    """Number of devices in a MeshDevice (1 for a plain single device)."""
    n = getattr(device, "get_num_devices", None)
    try:
        return n() if n is not None else 1
    except Exception:
        return 1


def _dev(t, device, dtype=ttnn.bfloat16):
    """Host tensor -> device. On a >1 MeshDevice the tensor is REPLICATED across all chips
    (every chip runs the attention/mHC/compressor pipeline redundantly with resident replicated
    weights); on a single device it's a plain upload. This one change makes the whole non-MoE
    decode SPMD-correct on the mesh — the MoE is the only part that shards (see moe_device_fp4_mesh)."""
    mapper = ttnn.ReplicateTensorToMesh(device) if _mesh_n(device) > 1 else None
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _host(ty, device):
    """Device -> host. On a >1 MeshDevice every chip holds an identical (replicated) copy, so
    concat on dim 0 and keep chip 0's rows; on a single device it's a plain read."""
    n = _mesh_n(device)
    if n > 1:
        o = ttnn.to_torch(ty, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        return o[: o.shape[0] // n]
    return ttnn.to_torch(ty)


# Tensor-parallel-shard the MLA attention weights across the mesh. This FREES ~4.3GB/chip of DRAM
# for the expert LRU (its purpose); on its own it slightly SLOWS attention (~100->210ms/token from
# the per-layer all-reduce), so it's coupled with the LRU and DEFAULT OFF. Enable the pair with
# DEEPSEEK_V4_ATTN_TP=1 DEEPSEEK_V4_EXPERT_LRU=24 (demonstrated ~2.0 tok/s peak; K needs cooled-
# device tuning for stable steady-state — see REPORT / memory). PCC 0.9999 (demo/attn_tp_pcc.py).
ATTN_TP = os.environ.get("DEEPSEEK_V4_ATTN_TP", "0") == "1"


def build_rope_perm(rope_dim: int, device):
    """Fixed [rd, rd] matrix R with rotate_half(x) = x @ R (interleaved: out[2i]=-x[2i+1], out[2i+1]=x[2i])."""
    R = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for i in range(rope_dim // 2):
        R[2 * i + 1, 2 * i] = -1.0  # contributes to out[2i]
        R[2 * i, 2 * i + 1] = 1.0  # contributes to out[2i+1]
    return _dev(R, device, dtype=ttnn.bfloat16)


class MLAv4DeviceWeights:
    """Resident on-device weights + rope permutation for ONE sliding-attention layer."""

    def __init__(self, attn, cfg, device):
        self.cfg = cfg
        self.device = device
        self.nh = cfg.num_attention_heads
        self.hd = cfg.head_dim
        self.rd = cfg.qk_rope_head_dim
        self.o_groups = cfg.o_groups
        self.eps = cfg.rms_norm_eps
        self.scaling = attn.scaling

        def T(w):  # nn.Linear [out,in] -> device [in,out] for x@W
            return _dev(w.t().contiguous(), device)

        # small always-replicated weights
        self.q_a = T(attn.q_a_proj.weight.data)
        self.q_a_norm = _dev(attn.q_a_norm.weight.data, device, dtype=ttnn.bfloat16)
        self.kv = T(attn.kv_proj.weight.data)  # single shared KV head — replicated (small)
        self.kv_norm = _dev(attn.kv_norm.weight.data, device, dtype=ttnn.bfloat16)
        self.R = build_rope_perm(self.rd, device)
        self.ones_hd = _dev(torch.ones(self.hd, dtype=torch.float32), device, dtype=ttnn.bfloat16)  # unweighted q_b_norm
        oa = attn.o_a_proj.weight.data
        ipg = oa.shape[1]
        rank = oa.shape[0] // self.o_groups
        oa_g = oa.view(self.o_groups, rank, ipg).transpose(1, 2).contiguous()  # [g,ipg,rank]

        # TP: shard the BIG attention linears across the mesh by head/group (frees ~100MB/layer/chip
        # of DRAM for the expert LRU, and parallelizes the q/attention/o compute C-ways). q_b is
        # output-sharded by heads, o_a group-sharded, o_b row-parallel (partial -> all-reduce), sinks
        # sharded by head. q_a/kv/norms/R stay replicated. PCC 0.9999 vs replicated (demo/attn_tp_pcc.py).
        C = _mesh_n(device)
        self.tp = C > 1 and ATTN_TP
        if self.tp:
            self.nhl = self.nh // C  # heads per chip
            self.gl = self.o_groups // C  # o-groups per chip
            shC = ttnn.ShardTensorToMesh(device, dim=1)
            sh0 = ttnn.ShardTensorToMesh(device, dim=0)

            def SH(t, mapper):
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)

            self.q_b = SH(attn.q_b_proj.weight.data.t().contiguous(), shC)  # [q_lora, nh*hd] -> [.,nhl*hd]
            self.o_a = SH(oa_g, sh0)  # [g,ipg,rank] -> [gl,ipg,rank]
            self.o_b = SH(attn.o_b_proj.weight.data.t().contiguous(), sh0)  # [g*rank,H] -> [gl*rank,H]
            self.sinks = SH(attn.sinks.data.reshape(1, self.nh, 1, 1), shC)  # [1,nh,1,1] -> [1,nhl,1,1]
        else:
            self.nhl, self.gl = self.nh, self.o_groups
            self.q_b = T(attn.q_b_proj.weight.data)
            self.o_a = _dev(oa_g, device)
            self.o_b = T(attn.o_b_proj.weight.data)
            self.sinks = _dev(attn.sinks.data.reshape(1, self.nh, 1, 1), device, dtype=ttnn.bfloat16)


class HCACompressorDevice:
    """Resident on-device weights for a Heavily-Compressed-Attention compressor (increment 2a).
    Pools every m=128 tokens into one compressed KV entry via a gated softmax over the window."""

    def __init__(self, hca, cfg, device, R=None, rd=None):
        self.m = hca.compress_rate  # 128
        self.hd = hca.head_dim  # 512
        self.eps = cfg.rms_norm_eps
        self.rd = rd if rd is not None else cfg.qk_rope_head_dim
        self.kv = _dev(hca.kv_proj.weight.data.t().contiguous(), device)  # [H, hd]
        self.gate = _dev(hca.gate_proj.weight.data.t().contiguous(), device)  # [H, hd]
        self.kv_norm = _dev(hca.kv_norm.weight.data, device, dtype=ttnn.bfloat16)
        self.position_bias = _dev(hca.position_bias.data.reshape(1, 1, self.m, self.hd), device)  # [1,1,m,hd]
        self.R = R if R is not None else build_rope_perm(self.rd, device)


def hca_compress_device(hbuf_dev, Wc: HCACompressorDevice, cos_c, sin_c, device):
    """On-device HCA compressor over the full buffer. hbuf_dev [B,S,H]; cos_c/sin_c [1,1,nw,rd]
    (interleaved, at positions arange(nw)*m). Returns ckv [B,1,nw,hd] or None if S<m."""
    m, hd = Wc.m, Wc.hd
    B, S, _ = hbuf_dev.shape
    nw = S // m
    if nw == 0:
        return None
    usable = nw * m
    kv = ttnn.matmul(hbuf_dev, Wc.kv)  # [B,S,hd]
    gate = ttnn.matmul(hbuf_dev, Wc.gate)  # [B,S,hd]
    if usable != S:
        kv = ttnn.slice(kv, [0, 0, 0], [B, usable, hd])
        gate = ttnn.slice(gate, [0, 0, 0], [B, usable, hd])
    ck = ttnn.reshape(kv, [B, nw, m, hd])
    cg = ttnn.reshape(gate, [B, nw, m, hd])
    cg = ttnn.add(cg, Wc.position_bias)  # broadcast [1,1,m,hd]
    # softmax over the window axis (dim=2): move it last, softmax, restore
    cg = ttnn.transpose(cg, 2, 3)  # [B,nw,hd,m]
    probs = ttnn.softmax(cg, dim=-1)
    probs = ttnn.transpose(probs, 2, 3)  # [B,nw,m,hd]
    pooled = ttnn.sum(ttnn.multiply(ck, probs), dim=2)  # [B,nw,hd]
    compressed = ttnn.rms_norm(pooled, epsilon=Wc.eps, weight=Wc.kv_norm)
    compressed = ttnn.reshape(compressed, [B, 1, nw, hd])
    compressed = _rope_dev(compressed, cos_c, sin_c, Wc)  # reuse interleaved partial rope
    for t in (kv, gate, ck, cg, probs, pooled):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return compressed  # [B,1,nw,hd]


class CSACompressorDevice:
    """Resident weights for a Compressed-Sparse-Attention compressor (increment 2b).
    kv/gate project to 2*hd; the pool uses OVERLAPPING 2m windows (second-half channels of the
    current window + first-half channels of the previous window). Within max_context=2048 the
    indexer top-512 selects ALL (<=512) causally-valid entries, so block_bias is a no-op — this
    ports only the pool (the actual math), attended with a zero mask like HCA."""

    def __init__(self, csa, cfg, device, R=None, rd=None):
        self.m = csa.compress_rate  # 4
        self.hd = csa.head_dim  # 512
        self.eps = cfg.rms_norm_eps
        self.rd = rd if rd is not None else cfg.qk_rope_head_dim
        self.kv = _dev(csa.kv_proj.weight.data.t().contiguous(), device)  # [H, 2hd]
        self.gate = _dev(csa.gate_proj.weight.data.t().contiguous(), device)  # [H, 2hd]
        self.kv_norm = _dev(csa.kv_norm.weight.data, device, dtype=ttnn.bfloat16)
        self.position_bias = _dev(csa.position_bias.data.reshape(1, 1, self.m, 2 * self.hd), device)  # [1,1,m,2hd]
        self.R = R if R is not None else build_rope_perm(self.rd, device)


def csa_compress_device(hbuf_dev, Wc: CSACompressorDevice, cos_c, sin_c, device):
    """On-device CSA overlapping-window pool. hbuf_dev [B,S,H]; cos_c/sin_c [1,1,nw,rd].
    Returns ckv [B,1,nw,hd] or None if S<m. Mirrors compressors._ca_cb_pool exactly."""
    m, hd = Wc.m, Wc.hd
    B, S, _ = hbuf_dev.shape
    nw = S // m
    if nw == 0:
        return None
    usable = nw * m
    kv = ttnn.matmul(hbuf_dev, Wc.kv)  # [B,S,2hd]
    gate = ttnn.matmul(hbuf_dev, Wc.gate)  # [B,S,2hd]
    if usable != S:
        kv = ttnn.slice(kv, [0, 0, 0], [B, usable, 2 * hd])
        gate = ttnn.slice(gate, [0, 0, 0], [B, usable, 2 * hd])
    ck = ttnn.reshape(kv, [B, nw, m, 2 * hd])
    cg = ttnn.add(ttnn.reshape(gate, [B, nw, m, 2 * hd]), Wc.position_bias)
    # split channels: [..,:hd] -> "a" (goes to PREVIOUS-window's first half), [..,hd:] -> "b" (this window's second half)
    ck_a = ttnn.slice(ck, [0, 0, 0, 0], [B, nw, m, hd])
    ck_b = ttnn.slice(ck, [0, 0, 0, hd], [B, nw, m, 2 * hd])
    cg_a = ttnn.slice(cg, [0, 0, 0, 0], [B, nw, m, hd])
    cg_b = ttnn.slice(cg, [0, 0, 0, hd], [B, nw, m, 2 * hd])
    # shift "a" forward by one window; window 0 gets zeros (kv) / -inf (gate) so it contributes nothing
    zeros_w = _dev(torch.zeros(B, 1, m, hd), device)
    ninf_w = _dev(torch.full((B, 1, m, hd), -1e9), device)
    if nw > 1:
        ck_a_prev = ttnn.slice(ck_a, [0, 0, 0, 0], [B, nw - 1, m, hd])
        cg_a_prev = ttnn.slice(cg_a, [0, 0, 0, 0], [B, nw - 1, m, hd])
        shifted_kv_a = ttnn.concat([zeros_w, ck_a_prev], dim=1)  # [B,nw,m,hd]
        shifted_gate_a = ttnn.concat([ninf_w, cg_a_prev], dim=1)
    else:
        shifted_kv_a, shifted_gate_a = zeros_w, ninf_w
    new_kv = ttnn.concat([shifted_kv_a, ck_b], dim=2)  # [B,nw,2m,hd]
    new_gate = ttnn.concat([shifted_gate_a, cg_b], dim=2)  # [B,nw,2m,hd]
    # softmax over the 2m window axis (dim=2)
    ng = ttnn.transpose(new_gate, 2, 3)  # [B,nw,hd,2m]
    probs = ttnn.softmax(ng, dim=-1)
    probs = ttnn.transpose(probs, 2, 3)  # [B,nw,2m,hd]
    pooled = ttnn.sum(ttnn.multiply(new_kv, probs), dim=2)  # [B,nw,hd]
    compressed = ttnn.rms_norm(pooled, epsilon=Wc.eps, weight=Wc.kv_norm)
    compressed = ttnn.reshape(compressed, [B, 1, nw, hd])
    compressed = _rope_dev(compressed, cos_c, sin_c, Wc)
    for t in (kv, gate, ck, cg, ck_a, ck_b, cg_a, cg_b, new_kv, new_gate, ng, probs, pooled, zeros_w, ninf_w):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return compressed  # [B,1,nw,hd]


# Cap on mHC Sinkhorn iterations for the throughput path. The 4x4 doubly-stochastic
# normalization converges fast; capping 20->1 is numerically negligible (per prior synthetic
# measurement) but removes ~3300 tiny dispatch-bound ops/token. Set to None to honor the full
# checkpoint value (correctness path).
MHC_SINKHORN_ITERS_CAP = 1


class MHCDevice:
    """Resident weights for one mHC HyperConnection (increment 3). HC=4 streams. base/scale are
    static params → split/precompute on host; only the heavy rms(hc*H)+fn matmul + the tiny
    HC×HC control math run on device (all ttnn, so traceable)."""

    def __init__(self, hc, cfg, device):
        self.HC = cfg.hc_mult
        self.H = cfg.hidden_size
        self.eps = hc.hc_eps
        self.iters = hc.hc_sinkhorn_iters if MHC_SINKHORN_ITERS_CAP is None else min(hc.hc_sinkhorn_iters, MHC_SINKHORN_ITERS_CAP)
        self.norm_eps = getattr(hc.input_norm, "variance_epsilon", 1e-6)
        HC = self.HC
        self.fn = _dev(hc.fn.data.t().contiguous(), device)  # [hc*H, 24]
        base = hc.base.data.float()
        self.pre_b = _dev(base[:HC].reshape(1, 1, HC), device)
        self.post_b = _dev(base[HC : 2 * HC].reshape(1, 1, HC), device)
        self.comb_b = _dev(base[2 * HC :].reshape(1, 1, HC, HC), device)
        s = hc.scale.data.float()
        self.pre_s, self.post_s, self.comb_s = float(s[0]), float(s[1]), float(s[2])
        self.ones = _dev(torch.ones(HC * self.H), device, dtype=ttnn.bfloat16)


def mhc_device(streams, W: MHCDevice):
    """streams [B,1,HC,H] device -> (post [B,1,HC], comb [B,1,HC,HC], collapsed [B,1,H]), all device."""
    B, S, HC, H = streams.shape[0], streams.shape[1], W.HC, W.H
    eps = W.eps
    flat = ttnn.reshape(streams, [B, S, HC * H])
    normed = ttnn.rms_norm(flat, epsilon=W.norm_eps, weight=W.ones)
    proj = ttnn.matmul(normed, W.fn)  # [B,S,24]
    pre_w = ttnn.slice(proj, [0, 0, 0], [B, S, HC])
    post_w = ttnn.slice(proj, [0, 0, HC], [B, S, 2 * HC])
    comb_w = ttnn.slice(proj, [0, 0, 2 * HC], [B, S, 2 * HC + HC * HC])  # [.,HC*HC]
    pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(pre_w, W.pre_s), W.pre_b)), eps)  # [B,S,HC]
    post = ttnn.multiply(ttnn.sigmoid(ttnn.add(ttnn.multiply(post_w, W.post_s), W.post_b)), 2.0)
    comb_logits = ttnn.add(ttnn.multiply(ttnn.reshape(comb_w, [B, S, HC, HC]), W.comb_s), W.comb_b)
    comb = ttnn.add(ttnn.softmax(comb_logits, dim=-1), eps)  # [B,S,HC,HC]
    # normalize dim -2 then sinkhorn (alternate -1,-2), matching host
    comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), eps))
    for _ in range(W.iters - 1):
        comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), eps))
        comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), eps))
    pre4 = ttnn.reshape(pre, [B, S, HC, 1])
    collapsed = ttnn.sum(ttnn.multiply(streams, pre4), dim=2)  # [B,S,H]
    for t in (flat, normed, proj, pre_w, post_w, comb_w, comb_logits, pre, pre4):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return post, comb, collapsed


def _stream_mix(post, comb, x, streams):
    """mHC residual mix: streams = post⊗x + combᵀ@streams. post [B,1,HC], x [B,1,H],
    comb [B,1,HC,HC], streams [B,1,HC,H] -> [B,1,HC,H]. All on device."""
    B, S, HC, H = streams.shape
    post4 = ttnn.reshape(post, [B, S, HC, 1])
    x2 = ttnn.reshape(x, [B, S, 1, H])
    term1 = ttnn.multiply(post4, x2)  # [B,S,HC,H]
    combT = ttnn.transpose(comb, -1, -2)
    term2 = ttnn.matmul(combT, streams)  # [B,S,HC,H]
    out = ttnn.add(term1, term2)
    for t in (post4, x2, term1, combT, term2):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return out


class LayerDeviceWeights:
    """All resident on-device weights for one full V4 decoder layer (attention + MoE)."""

    def __init__(self, layer, cfg, device, layer_idx=None, store=None, fp4_cache=None):
        self.cfg = cfg
        self.eps = cfg.rms_norm_eps
        self.attn_hc = MHCDevice(layer.attn_hc, cfg, device)
        self.ffn_hc = MHCDevice(layer.ffn_hc, cfg, device)
        self.mla = MLAv4DeviceWeights(layer.self_attn, cfg, device)
        comp = getattr(layer.self_attn, "compressor", None)
        self.comp = None
        self.is_hca = False
        if comp is not None:
            self.is_hca = type(comp).__name__.startswith("DeepseekV4HCA")
            self.comp = (HCACompressorDevice if self.is_hca else CSACompressorDevice)(comp, cfg, device, R=self.mla.R)
        self.input_ln = _dev(layer.input_layernorm.weight.data, device, dtype=ttnn.bfloat16)
        self.post_ln = _dev(layer.post_attention_layernorm.weight.data, device, dtype=ttnn.bfloat16)
        self.mlp = layer.mlp  # host module (used by the non-fp4 moe_device path)
        self.comp_rate = comp.compress_rate if comp is not None else None
        # fp4 fast MoE path (streamed experts from `store` via `fp4_cache`) when configured
        self.layer_idx = layer_idx
        self.store = store
        self.fp4_cache = fp4_cache
        # capture MoE host params as CLONES (the scratch layer module is overwritten when the
        # next layer loads, so a reference would be wrong across 43 resident layers)
        g = layer.mlp.gate
        self.gate_bias = g.e_score_correction_bias.data.clone() if hasattr(g, "e_score_correction_bias") else None
        self.is_hash = getattr(layer.mlp, "is_hash", False)
        self.tid2eid = g.tid2eid.clone() if (self.is_hash and hasattr(g, "tid2eid")) else None
        # RESIDENT router gate + shared-expert weights (uploaded once — the shared expert is DENSE,
        # so re-uploading its bf16 weights per token per layer was ~1.2s/token; residency kills it)
        self.gate_w_d = _dev(g.weight.data.t().contiguous(), device)  # [H, n_experts] (replicated)
        se = layer.mlp.shared_experts
        self.is_mesh = _mesh_n(device) > 1
        if self.is_mesh:
            # On the mesh the shared expert is TP-SHARDED (gate/up col-sharded, down row-sharded)
            # so it fuses into the routed experts' single per-layer all_gather+sum reduction.
            self.se_g_d = RW._res_upload(se.gate_proj.weight.data.t().contiguous(), device, ttnn.bfloat16, shard_dim=-1)  # [H, I/C]
            self.se_u_d = RW._res_upload(se.up_proj.weight.data.t().contiguous(), device, ttnn.bfloat16, shard_dim=-1)  # [H, I/C]
            self.se_d_d = RW._res_upload(se.down_proj.weight.data.t().contiguous(), device, ttnn.bfloat16, shard_dim=0)  # [I/C, H]
        else:
            self.se_g_d = _dev(se.gate_proj.weight.data.t().contiguous(), device)  # [H, interm_s]
            self.se_u_d = _dev(se.up_proj.weight.data.t().contiguous(), device)  # [H, interm_s]
            self.se_d_d = _dev(se.down_proj.weight.data.t().contiguous(), device)  # [interm_s, H]


PROF = {"attn": 0.0, "comp": 0.0, "moe": 0.0, "mhc": 0.0}
PROF_ON = False
ROUTE_TRACE = None  # set to a list to record (layer_idx, [expert ids]) per MoE call (locality study)


def _psync(device):
    if PROF_ON:
        ttnn.synchronize_device(device)
        import time as _t

        return _t.perf_counter()
    return 0.0


def decode_layer_device(streams, hbuf, kv_cache, LW: LayerDeviceWeights, cos_m, sin_m, cos_c, sin_c, device, token_id=None):
    """One full V4 decoder-layer decode step, on device (routing reads collapsed_ln to host).
    streams [B,1,HC,H], hbuf [B,S_past,H] or None, kv_cache [B,1,S_past,hd] or None.
    Returns (streams', main_kv, hbuf_full)."""
    import time as _t

    cfg, eps = LW.cfg, LW.eps
    # --- attention half ---
    t = _psync(device)
    post, comb, collapsed = mhc_device(streams, LW.attn_hc)
    collapsed_ln = ttnn.rms_norm(collapsed, epsilon=eps, weight=LW.input_ln)  # [B,1,H]
    if PROF_ON:
        ttnn.synchronize_device(device); PROF["mhc"] += _t.perf_counter() - t; t = _t.perf_counter()
    cln4 = ttnn.reshape(collapsed_ln, [collapsed_ln.shape[0], 1, collapsed_ln.shape[-1]])
    hbuf_full = cln4 if hbuf is None else ttnn.concat([hbuf, cln4], dim=1)
    ckv = None
    if LW.comp is not None:
        fn = hca_compress_device if LW.is_hca else csa_compress_device
        ckv = fn(hbuf_full, LW.comp, cos_c, sin_c, device)
    if PROF_ON:
        ttnn.synchronize_device(device); PROF["comp"] += _t.perf_counter() - t; t = _t.perf_counter()
    attn_out, main_kv = mla_decode_device(collapsed_ln, kv_cache, LW.mla, cos_m, sin_m, device, ckv=ckv)
    streams = _stream_mix(post, comb, attn_out, streams)
    if PROF_ON:
        ttnn.synchronize_device(device); PROF["attn"] += _t.perf_counter() - t; t = _t.perf_counter()
    # --- ffn (MoE) half ---
    post, comb, collapsed = mhc_device(streams, LW.ffn_hc)
    collapsed_ln = ttnn.rms_norm(collapsed, epsilon=eps, weight=LW.post_ln)
    cln_host = _host(collapsed_ln, device)  # one host read/layer for data-dependent expert routing
    if PROF_ON:
        ttnn.synchronize_device(device); PROF["mhc"] += _t.perf_counter() - t; t = _t.perf_counter()
    if getattr(LW, "is_mesh", False) and LW.fp4_cache is not None:
        mlp_out = moe_device_fp4_mesh(cln_host, LW, cfg, device, token_id=token_id)
    elif LW.fp4_cache is not None:
        mlp_out = moe_device_fp4(cln_host, LW, cfg, device, token_id=token_id)
    else:
        mlp_out = moe_device(collapsed_ln, cln_host, LW.mlp, cfg, device)
    streams = _stream_mix(post, comb, mlp_out, streams)
    if PROF_ON:
        ttnn.synchronize_device(device); PROF["moe"] += _t.perf_counter() - t
    return streams, main_kv, hbuf_full


def moe_device(collapsed_ln_dev, collapsed_ln_host, mlp, cfg, device):
    """On-device MoE for one decode token, device fp32 accumulate (one read/layer, not 258).
    Routing (sqrtsoftplus scores + topk + normalize) is host-side from device scores — it only
    picks which experts to stream (data-dependent), matching model.sparse_moe exactly. Expert
    compute + weighted accumulate run ON DEVICE (fused_expert_ondevice + ttnn add). Returns
    device [B,1,H]. mlp = HF DeepseekV4SparseMoeBlock (gate + experts + shared_experts)."""
    from models.demos.deepseek_v4.tt import modules as M

    H = cfg.hidden_size
    interm = cfg.moe_intermediate_size
    limit = cfg.swiglu_limit
    flat = collapsed_ln_host.reshape(-1, H)  # [1,H] host (for routing)
    gate = mlp.gate
    scores = M.sqrtsoftplus_router_scores(flat, gate.weight.data, device)  # [1,256] (host)
    indices = torch.topk(scores + gate.e_score_correction_bias, cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    w = scores.gather(1, indices)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor  # [1,top_k]

    tx = ttnn.from_torch(collapsed_ln_host.reshape(1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gu_all = mlp.experts.gate_up_proj.data  # [E,2I,H]
    dn_all = mlp.experts.down_proj.data  # [E,H,I]
    acc = None
    for k in range(indices.shape[1]):
        e = int(indices[0, k])
        gate_up_T = gu_all[e].t().contiguous()  # [H,2I]
        down_T = dn_all[e].t().contiguous()  # [I,H]
        y = M.fused_expert_ondevice(tx, gate_up_T, down_T, interm, device, limit=limit)  # device [1,H]
        yf = ttnn.multiply(ttnn.typecast(y, ttnn.float32), float(w[0, k]))
        acc = yf if acc is None else ttnn.add(acc, yf)
        ttnn.deallocate(y)
    # shared expert on device
    se = mlp.shared_experts
    shared = M.clamped_swiglu_mlp(
        collapsed_ln_host.reshape(1, H), se.gate_proj.weight.data, se.up_proj.weight.data,
        se.down_proj.weight.data, device, limit=limit,
    )  # host [1,H]
    shared_d = ttnn.from_torch(shared.reshape(1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.add(ttnn.reshape(acc, [1, 1, H]), shared_d)
    ttnn.deallocate(tx)
    ttnn.deallocate(acc)
    ttnn.deallocate(shared_d)
    return out  # [1,1,H]


class Fp4ExpertCache:
    """Lazy host cache of pre-tilized bfloat4_b expert weights keyed (layer_idx, e).
    First touch dequantizes (store) + tilizes to bf4 (one-time ~213ms/expert); reused after."""

    def __init__(self, store):
        self.store = store
        self.c = {}

    def get(self, layer_idx, e):
        k = (layer_idx, e)
        v = self.c.get(k)
        if v is None:
            gu_T, dn_T = RW.expert_fused(self.store, layer_idx, e)  # host bf16 [H,2I],[I,H]
            v = (
                ttnn.from_torch(gu_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT),
                ttnn.from_torch(dn_T, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT),
            )
            self.c[k] = v
        return v


def moe_device_fp4(collapsed_ln_host, LW, cfg, device, token_id=None):
    """MoE for one decode token streaming FP4 experts (the fast path). Routing host-side from
    LW's captured gate params; routed experts pulled from LW.fp4_cache (DMA-only, no host
    tilize/read); on-device fp32 accumulate; shared expert from LW's captured weights.
    Returns device [1,1,H]."""
    H = cfg.hidden_size
    interm = cfg.moe_intermediate_size
    limit = cfg.swiglu_limit
    tx = ttnn.from_torch(collapsed_ln_host.reshape(1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # router scores on device (resident gate weight), read [1,n_experts] to host for topk
    sc = ttnn.matmul(tx, LW.gate_w_d)
    scores = ttnn.to_torch(ttnn.sqrt(ttnn.softplus(sc)))  # [1,n_experts] host
    ttnn.deallocate(sc)
    if LW.is_hash:
        tok = torch.tensor([token_id if token_id is not None else 0])
        indices = LW.tid2eid[tok].long().reshape(1, -1)
    else:
        indices = torch.topk(scores + LW.gate_bias, cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    w = scores.gather(1, indices)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor

    # shared expert (DENSE) — resident weights, on-device matmuls (no re-upload)
    sg = ttnn.clamp(ttnn.matmul(tx, LW.se_g_d), max=limit)
    su = ttnn.clamp(ttnn.matmul(tx, LW.se_u_d), min=-limit, max=limit)
    sact = ttnn.multiply(ttnn.silu(sg), su)
    acc = ttnn.typecast(ttnn.matmul(sact, LW.se_d_d), ttnn.float32)  # [1,H]
    for t in (sg, su, sact):
        ttnn.deallocate(t)
    # routed experts — streamed fp4 (DMA-only), on-device fp32 accumulate
    for k in range(indices.shape[1]):
        e = int(indices[0, k])
        hgu, hdn = LW.fp4_cache.get(LW.layer_idx, e)
        tgu = ttnn.to_device(hgu, device)
        tdn = ttnn.to_device(hdn, device)
        gu = ttnn.matmul(tx, tgu)
        gt = ttnn.clamp(gu[..., :interm], max=limit)
        up = ttnn.clamp(gu[..., interm:], min=-limit, max=limit)
        act = ttnn.multiply(ttnn.silu(gt), up)
        y = ttnn.matmul(act, tdn)
        yf = ttnn.multiply(ttnn.typecast(y, ttnn.float32), float(w[0, k]))
        acc = ttnn.add(acc, yf)
        for t in (tgu, tdn, gu, gt, up, act, y, yf):
            ttnn.deallocate(t)
    out = ttnn.reshape(acc, [1, 1, H])
    ttnn.deallocate(tx)
    return out


class Fp4ExpertCacheMesh:
    """Lazy host cache of pre-tilized, TENSOR-PARALLEL-SHARDED bfloat4_b expert weights for the
    multi-chip mesh, keyed (layer_idx, e). Each expert is split across the C chips: gate/up
    column-sharded on the intermediate dim ([H, I] -> [H, I/C] per chip) and down row-sharded
    ([I, H] -> [I/C, H] per chip). The cached objects are host multi-device tensors (from_torch
    with a ShardTensorToMesh mapper but no device); ttnn.to_device then DMAs each chip's 1/C slice
    over its OWN PCIe in parallel — ~C× the effective host->DRAM bandwidth vs a single chip.
    First touch dequantizes + tilizes + shards (one-time); reused after.

    NOTE: a COALESCED variant (fuse gate_up + device-concat the routed set into one batched matmul)
    measured 1.78x faster in isolation (demo/micro_fp4_coalesce.py) — the host->device DMA
    parallelizes to ~24.6 GB/s aggregate on 4 chips (demo/micro_dma_bw.py) but only with few big
    transfers. That variant HUNG in the integrated pipeline (unresolved) so this per-expert path is
    the working default. See REPORT_11 / memory for the full measurement trail."""

    # suffix ttnn.as_tensor appends to cache_file_name (dtype+layout of the stored tensor)
    _SUF = "_dtype_BFLOAT4_B_layout_TILE.tensorbin"

    def __init__(self, store, mesh):
        self.store = store
        self.mesh = mesh
        self.C = mesh.get_num_devices()
        self.c = {}
        self.shard_gu = ttnn.ShardTensorToMesh(mesh, dim=2)  # split 2I of [1, H, 2I]
        self.shard_dn = ttnn.ShardTensorToMesh(mesh, dim=1)  # split I of  [1, I, H]
        self._perm = None
        # STAMPED disk cache: skips the ~230ms/expert NVFP4-dequant+bf4-tilize+reorder on reboot by
        # persisting the sharded bf4 flatbuffers (load ~0ms, values bit-identical). Set env
        # DEEPSEEK_V4_BF4_CACHE_DIR="" to disable (in-memory only). Path is stamped so it invalidates
        # only when the model / mesh-count / reorder / ttnn-build changes (see _bf4_cache_stamp).
        root = os.environ.get(
            "DEEPSEEK_V4_BF4_CACHE_DIR",
            os.path.join(os.environ.get("TT_METAL_HOME", "/tmp"), "generated", "deepseek_v4_bf4cache"),
        )
        self.disk_dir = os.path.join(root, _bf4_cache_stamp(mesh)) if root else None
        if self.disk_dir:
            os.makedirs(self.disk_dir, exist_ok=True)
        self.stats = {"mem": 0, "disk": 0, "cold": 0, "lru": 0}  # access accounting (isolation)
        # ON-DEVICE expert LRU (env DEEPSEEK_V4_EXPERT_LRU, DEFAULT OFF): keep the K most-recently-
        # used experts PER LAYER resident so repeat routing skips the ~0.6ms/tensor upload. Routing
        # locality is real (~58% hit at K=32, measured demo/locality.py) BUT this is DEVICE-MEMORY
        # BLOCKED here: the 43 layers of REPLICATED attn/mHC weights already fill most of 32GB/chip,
        # so K>=16 (~2GB) thrashes the allocator (warm decode collapses 0.5s->3.5s/token) and K=8
        # (stable) has too low a hit rate to beat the no-residency path (1.45 vs 1.69 tok/s). Becomes
        # viable only after sharding the attention/mHC weights to free DRAM. Default 0 = always stream.
        self.lru_K = int(os.environ.get("DEEPSEEK_V4_EXPERT_LRU", "0"))
        self.dev_lru = {}  # layer_idx -> OrderedDict{e: (gu_dev, dn_dev)}

    def warm_all(self, num_layers, num_experts, log=None, retain=False):
        """Build+dump EVERY (layer, expert) sharded bf4 tensor to the disk cache, so steady-state
        decode never pays a cold build (~300ms) or disk load (~30ms). Resumable: each cold build
        dumps to disk (~9.4MB gu + smaller dn), so a re-run disk-hits and skips it.
        retain=False (default): do NOT keep the built tensors in self.c during the build — the cold
        path creates ~32MB bf16 dequant transients per expert, and retaining thousands of handles +
        dirty-page writeback caused an external OOM SIGKILL mid-build. We gc + fsync periodically to
        bound that. At decode time get() re-loads from disk as a cheap mmap handle (~0.2MB RSS each,
        measured), so retaining here is unnecessary. retain=True keeps them (for a warm in-process run)."""
        import gc, os as _os, time as _t
        t0 = _t.perf_counter()
        total = num_layers * num_experts
        done = 0
        for li in range(num_layers):
            for e in range(num_experts):
                try:
                    self.get(li, e)  # mem/disk/cold(+dump); stores in self.c
                    if not retain:
                        self.c.pop((li, e), None)  # release the handle + any dequant transient
                except Exception as ex:
                    if log:
                        log(f"warm_all skip L{li} e{e}: {ex}")
                done += 1
                if done % 100 == 0:
                    gc.collect()  # release per-expert dequant transients (retain=False bounds RSS)
                if log and done % 200 == 0:
                    r = self.stats
                    log(f"warm_all {done}/{total} ({100*done//total}%) "
                        f"disk={r['disk']} cold={r['cold']} elapsed={_t.perf_counter()-t0:.0f}s")
        if log:
            log(f"warm_all DONE {done}/{total} in {_t.perf_counter()-t0:.0f}s | "
                f"disk={self.stats['disk']} cold={self.stats['cold']}")

    def preload_pagecache(self, log=None):
        """Read every cached expert file's payload into the OS page cache. warm_all/load_tensor only
        mmap the files (lazy) — the 9.4MB bf4 payload is not resident until first to_device faults it
        from disk, which caused per-token spikes (measured: a 4296ms outlier with disk=0/cold=0). A
        one-time sequential read (~29s for 156GB on NVMe) makes every to_device hit warm page cache
        -> flat decode (measured CV 1.15 -> 0.05, 1.98 -> 2.34 tok/s). Page cache is reclaimable, so
        this needs enough free RAM to hold the working set (156GB fit in ~227G here)."""
        import glob as _glob, time as _t
        if not getattr(self, "disk_dir", None):
            return
        files = _glob.glob(os.path.join(self.disk_dir, "*.tensorbin"))
        t0 = _t.perf_counter(); nbytes = 0
        for fp in files:
            try:
                with open(fp, "rb") as fh:
                    nbytes += len(fh.read())
            except OSError:
                pass
        if log:
            log(f"preload_pagecache: {len(files)} files ({nbytes/1e9:.0f}GB) in {_t.perf_counter()-t0:.0f}s")

    def get_device(self, layer_idx, e, device):
        """Return the expert's sharded bf4 weights ON DEVICE, via a per-layer LRU. HIT = no upload;
        MISS = ttnn.to_device (and evict the least-recent if the layer is full). The returned
        tensors are LRU-owned — the caller must NOT deallocate them."""
        if self.lru_K <= 0:
            hgu, hdn = self.get(layer_idx, e)
            return ttnn.to_device(hgu, device), ttnn.to_device(hdn, device), False
        od = self.dev_lru.setdefault(layer_idx, OrderedDict())
        v = od.get(e)
        if v is not None:
            od.move_to_end(e)
            self.stats["lru"] += 1
            return v[0], v[1], True
        hgu, hdn = self.get(layer_idx, e)
        gu = ttnn.to_device(hgu, device)
        dn = ttnn.to_device(hdn, device)
        od[e] = (gu, dn)
        if len(od) > self.lru_K:
            _, (egu, edn) = od.popitem(last=False)  # evict least-recently-used
            for t in (egu, edn):
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return gu, dn, False

    def _gu_perm(self, I):
        if self._perm is None:
            C = self.C
            chunk = I // C
            perm = []
            for cc in range(C):
                perm += list(range(cc * chunk, (cc + 1) * chunk))  # gate chunk cc
                perm += list(range(I + cc * chunk, I + (cc + 1) * chunk))  # up chunk cc
            self._perm = torch.tensor(perm)
        return self._perm

    def get(self, layer_idx, e):
        k = (layer_idx, e)
        v = self.c.get(k)
        if v is not None:
            self.stats["mem"] += 1
            return v
        gu_base = os.path.join(self.disk_dir, f"L{layer_idx}_e{e}_gu") if self.disk_dir else None
        dn_base = os.path.join(self.disk_dir, f"L{layer_idx}_e{e}_dn") if self.disk_dir else None
        # disk HIT: load the sharded bf4 flatbuffers directly (no dequant/tilize/reorder at all)
        if gu_base and os.path.exists(gu_base + self._SUF) and os.path.exists(dn_base + self._SUF):
            try:
                v = (
                    ttnn.load_tensor(gu_base + self._SUF, device=None),
                    ttnn.load_tensor(dn_base + self._SUF, device=None),
                )
                self.c[k] = v
                self.stats["disk"] += 1
                return v
            except RuntimeError:
                pass  # stale/incompatible flatbuffer -> fall through and rebuild
        self.stats["cold"] += 1
        # disk MISS: dequant + reorder + tilize, and (if caching) dump the flatbuffers via as_tensor
        gu_T, dn_T = RW.expert_fused(self.store, layer_idx, e)  # host bf16 [H,2I],[I,H]
        I = dn_T.shape[0]
        gu_re = gu_T[:, self._gu_perm(I)].contiguous().unsqueeze(0)  # [1, H, 2I] reordered
        dn = dn_T.contiguous().unsqueeze(0)  # [1, I, H]
        if gu_base:
            v = (
                ttnn.as_tensor(gu_re, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.shard_gu, cache_file_name=gu_base),
                ttnn.as_tensor(dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.shard_dn, cache_file_name=dn_base),
            )
        else:
            v = (
                ttnn.from_torch(gu_re, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.shard_gu),
                ttnn.from_torch(dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.shard_dn),
            )
        self.c[k] = v
        return v


def moe_device_fp4_mesh(collapsed_ln_host, LW, cfg, device, token_id=None):
    """Multi-chip tensor-parallel MoE for one decode token (the expert-parallel throughput path).

    Every routed expert AND the dense shared expert are TP-sharded across the C chips: gate/up are
    column-sharded so each chip computes its own I/C activation slice, and down is row-sharded so
    each chip yields a [1,H] PARTIAL sum over its I-slice. Because that cross-chip reduction is
    linear, we accumulate ALL experts' (and the shared expert's) partials LOCALLY on each chip and
    do a SINGLE all_gather+sum for the whole layer (not one per expert). Routed-expert weights are
    STREAMED per token (ttnn.to_device of the cached sharded bf4 — 1/C per chip in parallel).
    Returns device [1,1,H] (replicated across chips)."""
    H = cfg.hidden_size
    interm = cfg.moe_intermediate_size
    limit = cfg.swiglu_limit
    C = _mesh_n(device)
    chunk = interm // C  # per-chip I slice
    tx = _dev(collapsed_ln_host.reshape(1, H), device)  # replicated [1,H]

    # router scores on device (resident replicated gate weight), read chip-0 copy for host topk
    sc = ttnn.matmul(tx, LW.gate_w_d)
    scores = _host(ttnn.sqrt(ttnn.softplus(sc)), device)  # [1,n_experts] host
    ttnn.deallocate(sc)
    if LW.is_hash:
        tok = torch.tensor([token_id if token_id is not None else 0])
        indices = LW.tid2eid[tok].long().reshape(1, -1)
    else:
        indices = torch.topk(scores + LW.gate_bias, cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    w = scores.gather(1, indices)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
    E = indices.shape[1]
    if ROUTE_TRACE is not None:  # locality measurement hook
        ROUTE_TRACE.append((LW.layer_idx, [int(x) for x in indices[0].tolist()]))

    # shared expert (DENSE) — resident TP-sharded weights; produces a [1,H] partial per chip
    sg = ttnn.clamp(ttnn.matmul(tx, LW.se_g_d), max=limit)  # [1, I/C]
    su = ttnn.clamp(ttnn.matmul(tx, LW.se_u_d), min=-limit, max=limit)
    sact = ttnn.multiply(ttnn.silu(sg), su)
    local = ttnn.typecast(ttnn.matmul(sact, LW.se_d_d), ttnn.float32)  # [1,H] partial
    for t in (sg, su, sact):
        ttnn.deallocate(t)

    # routed experts — stream the E chosen (2 fused sharded DMAs each), device-concat into a
    # batched [E,...] weight, then ONE batched matmul pair (COALESCED upload = the measured win).
    gu_list, dn_list = [], []
    for k in range(E):
        e = int(indices[0, k])
        gu_d, dn_d, _hit = LW.fp4_cache.get_device(LW.layer_idx, e, device)  # LRU-owned (may skip upload)
        gu_list.append(gu_d)  # [1, H, 2I/C] per chip
        dn_list.append(dn_d)  # [1, I/C, H] per chip
    GU = ttnn.concat(gu_list, dim=0)  # [E, H, 2I/C]
    DN = ttnn.concat(dn_list, dim=0)  # [E, I/C, H]
    txE = _dev(collapsed_ln_host.reshape(1, H).repeat(E, 1).reshape(E, 1, H), device)  # [E,1,H]
    wE = _dev(torch.tensor(w[0].tolist()).reshape(E, 1, 1), device, dtype=ttnn.float32)  # [E,1,1]
    gu = ttnn.matmul(txE, GU)  # [E,1,2I/C] batched
    gt = ttnn.clamp(ttnn.slice(gu, [0, 0, 0], [E, 1, chunk]), max=limit)  # [E,1,I/C]
    up = ttnn.clamp(ttnn.slice(gu, [0, 0, chunk], [E, 1, 2 * chunk]), min=-limit, max=limit)
    act = ttnn.multiply(ttnn.silu(gt), up)  # [E,1,I/C]
    yp = ttnn.matmul(act, DN)  # [E,1,H] partial per chip
    yf = ttnn.multiply(ttnn.typecast(yp, ttnn.float32), wE)
    routed = ttnn.sum(yf, dim=0)  # [1,H] partial
    local = ttnn.add(local, routed)
    # NOTE: gu_list/dn_list are LRU-owned (resident across tokens) — do NOT deallocate them here.
    for t in [GU, DN, txE, wE, gu, gt, up, act, yp, yf, routed]:
        ttnn.deallocate(t)

    # single cross-chip reduction for the whole layer: all_gather the [1,H] partials + sum
    lr = ttnn.reshape(local, [1, 1, 1, H])
    yg = ttnn.all_gather(lr, dim=0)  # [C,1,1,H] on every chip
    reduced = ttnn.sum(yg, dim=0)  # [1,1,H] full (replicated)
    out = ttnn.reshape(reduced, [1, 1, H])
    for t in (tx, local, lr, yg, reduced):
        ttnn.deallocate(t)
    return out


def _rope_dev(x, cos, sin, W):
    """Interleaved partial RoPE on device. x [B,nh,1,hd]; cos/sin [1,1,1,rd] (already interleaved).
    Returns [B,nh,1,hd]. rope = last rd dims; nope = the rest (unchanged)."""
    hd, rd = W.hd, W.rd
    nope = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], hd - rd])
    rope = ttnn.slice(x, [0, 0, 0, hd - rd], [x.shape[0], x.shape[1], x.shape[2], hd])
    rot = ttnn.matmul(rope, W.R)  # rotate_half via fixed matrix
    rotated = ttnn.add(ttnn.multiply(rope, cos), ttnn.multiply(rot, sin))
    out = ttnn.concat([nope, rotated], dim=-1)
    for t in (nope, rope, rot, rotated):
        ttnn.deallocate(t)
    return out


def mla_decode_device(new_ln_dev, kv_cache_dev, W: MLAv4DeviceWeights, cos, sin, device, ckv=None):
    """One-token MLA decode, fully on device.
    new_ln_dev : [B,1,H] device (input_layernorm'd hidden for the new token)
    kv_cache_dev : [B,1,S_past,hd] device post-RoPE main K==V, or None on first token
    cos, sin   : [1,1,1,rd] device, INTERLEAVED (repeat_interleave(2) already applied)
    ckv        : optional compressed KV [B,1,T,hd] (CSA/HCA) concatenated onto main K==V; within
                 max_context its block_bias is a no-op so it's attended with zero extra mask.
    Returns (output_dev [B,1,H], kv_new_dev [B,1,1,hd]).
    """
    cfg, nh, hd, eps = W.cfg, W.nh, W.hd, W.eps
    B = new_ln_dev.shape[0]
    nhl, gl = W.nhl, W.gl  # per-chip heads / o-groups (== nh / o_groups when not TP)

    # --- Q path (q_b output-sharded by heads when TP -> this chip computes its nhl heads) ---
    q_res = ttnn.matmul(new_ln_dev, W.q_a)  # [B,1,q_lora] (replicated input)
    q_res = ttnn.rms_norm(q_res, epsilon=eps, weight=W.q_a_norm)
    q = ttnn.matmul(q_res, W.q_b)  # [B,1,nhl*hd]
    q = ttnn.reshape(q, [B, 1, nhl, hd])
    q = ttnn.transpose(q, 1, 2)  # [B,nhl,1,hd]
    q = ttnn.rms_norm(q, epsilon=eps, weight=W.ones_hd)  # unweighted over hd
    q = _rope_dev(q, cos, sin, W)  # [B,nhl,1,hd]

    # --- KV path (single shared head, REPLICATED — every chip holds the full KV cache) ---
    kv_new = ttnn.matmul(new_ln_dev, W.kv)  # [B,1,hd]
    kv_new = ttnn.rms_norm(kv_new, epsilon=eps, weight=W.kv_norm)
    kv_new = ttnn.reshape(kv_new, [B, 1, 1, hd])
    kv_new = ttnn.transpose(kv_new, 1, 2)  # [B,1,1,hd]
    kv_new = _rope_dev(kv_new, cos, sin, W)  # [B,1,1,hd]
    main_kv = kv_new if kv_cache_dev is None else ttnn.concat([kv_cache_dev, kv_new], dim=2)  # [B,1,S,hd]
    kv_full = main_kv if ckv is None else ttnn.concat([main_kv, ckv], dim=2)  # [B,1,Lkv,hd]

    Lkv = kv_full.shape[2]
    # repeat single KV head -> nhl heads (MQA). ttnn.repeat on dim 1.
    k = ttnn.repeat(kv_full, ttnn.Shape([1, nhl, 1, 1]))  # [B,nhl,Lkv,hd]

    # --- attention core with sinks (sinks sharded by head when TP) ---
    kt = ttnn.transpose(k, 2, 3)  # [B,nhl,hd,Lkv]
    aw = ttnn.matmul(q, kt)  # [B,nhl,1,Lkv]
    aw = ttnn.multiply(aw, W.scaling)
    combined = ttnn.concat([aw, W.sinks], dim=-1)  # [B,nhl,1,Lkv+1]
    probs = ttnn.softmax(combined, dim=-1)
    scores = ttnn.slice(probs, [0, 0, 0, 0], [B, nhl, 1, Lkv])  # drop sink column
    attn_out = ttnn.matmul(scores, k)  # [B,nhl,1,hd]
    attn_out = _rope_dev(attn_out, cos, ttnn.neg(sin), W)  # conjugate -sin un-rotation

    # --- output projection (grouped o_a then o_b) ---
    attn_out = ttnn.transpose(attn_out, 1, 2)  # [B,1,nhl,hd]
    ipg = W.o_a.shape[1]
    xg = ttnn.reshape(attn_out, [B, gl, ipg])  # nhl*hd == gl*ipg
    xg = ttnn.transpose(xg, 0, 1)  # [gl, B, ipg]
    yg = ttnn.matmul(xg, W.o_a)  # [gl, B, rank]
    yg = ttnn.transpose(yg, 0, 1)  # [B, gl, rank]
    grouped = ttnn.reshape(yg, [B, 1, gl * W.o_a.shape[-1]])
    output = ttnn.matmul(grouped, W.o_b)  # [B,1,H] (PARTIAL per chip when TP)
    to_free = [q_res, q, k, kt, aw, combined, probs, scores, attn_out, xg, yg, grouped]
    if W.tp:
        # row-parallel o_b: sum the per-chip partials across the mesh (one all-reduce)
        lr = ttnn.reshape(output, [1, 1, 1, output.shape[-1]])
        ag = ttnn.all_gather(lr, dim=0)  # [C,1,1,H]
        red = ttnn.sum(ag, dim=0)  # [1,1,H]
        to_free += [output, lr, ag]
        output = ttnn.reshape(red, [B, 1, output.shape[-1]])
        to_free.append(red)
    for t in to_free:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return output, main_kv  # main_kv[...,-1:,:] is kv_new for cache append
