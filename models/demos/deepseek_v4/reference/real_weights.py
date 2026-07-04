# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stream the ACTUAL pretrained DeepSeek-V4-Flash weights from a local HF snapshot.

The full model is ~149 GB fp8 (fits host RAM) but ~320 GB bf16 (does NOT). `from_pretrained`
dequantizes to bf16 on a CPU host, so it cannot hold all 43 layers at once. This module reads
the raw fp8 safetensors lazily and dequantizes **one layer at a time** into a reusable scratch
HF module, so the full model runs on Blackhole with per-layer memory.

It maps DeepSeek's native checkpoint names (`layers.N.attn.wq_a.weight`, `ffn.experts.E.w1`, …)
to the HF `transformers` module names, assembling the packed expert tensors and fp8-dequantizing
via the established DeepSeek block-scale routine. Validated against the transformers-loaded model
(allclose) — see tests/test_real_weights.py.
"""
from __future__ import annotations

import glob
import json
import os

import torch
from safetensors import safe_open

from models.demos.deepseek_v3.utils.hf_model_utils import dequantize_weight_tensor

BLOCK = [128, 128]

# MXFP4 e2m1 value table (fp4 experts, expert_dtype=fp4). Same LUT transformers uses.
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant_mxfp4(blk: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize an MXFP4 (fp4 e2m1) packed weight: `blk` int8 [R, Cp] (2 nibbles/byte, low
    nibble = even column, high = odd), `scale` [R, G] `float8_e8m0fnu` per-group multipliers
    (group size = 2*Cp/G = 32). value = FP4_VALUES[nibble] * scale_group. Returns bf16 [R, 2*Cp]."""
    lut = torch.tensor(FP4_VALUES, dtype=dtype)
    b = blk.to(torch.uint8)
    R, Cp = b.shape
    G = scale.shape[1]
    B = Cp // G  # bytes per group (16)
    b = b.reshape(R, G, B)
    out = torch.empty(R, G, 2 * B, dtype=dtype)
    out[..., 0::2] = lut[(b & 0x0F).long()]
    out[..., 1::2] = lut[(b >> 4).long()]
    out = out * scale.float().reshape(R, G, 1).to(dtype)  # e8m0 scale is already the 2^exp multiplier
    return out.reshape(R, G * 2 * B).contiguous()  # [R, 2*Cp] = [interm, hidden] etc.


# HF (per-layer, prefix stripped) -> native (per-layer, prefix stripped). 1:1 weight tensors.
LAYER_MAP = {
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_proj.weight": "attn.wkv.weight",
    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
    "self_attn.o_a_proj.weight": "attn.wo_a.weight",
    "self_attn.o_b_proj.weight": "attn.wo_b.weight",
    "self_attn.sinks": "attn.attn_sink",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn",
    "attn_hc.base": "hc_attn_base",
    "attn_hc.scale": "hc_attn_scale",
    "ffn_hc.fn": "hc_ffn_fn",
    "ffn_hc.base": "hc_ffn_base",
    "ffn_hc.scale": "hc_ffn_scale",
    "mlp.gate.weight": "ffn.gate.weight",
    "mlp.gate.tid2eid": "ffn.gate.tid2eid",
    "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.w1.weight",
    "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.w3.weight",
    "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    # compressor (CSA/HCA layers) + indexer (CSA only) — these ship BF16 (no .scale)
    "self_attn.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
    "self_attn.compressor.gate_proj.weight": "attn.compressor.wgate.weight",
    "self_attn.compressor.position_bias": "attn.compressor.ape",
    "self_attn.compressor.kv_norm.weight": "attn.compressor.norm.weight",
    "self_attn.compressor.indexer.kv_proj.weight": "attn.indexer.compressor.wkv.weight",
    "self_attn.compressor.indexer.gate_proj.weight": "attn.indexer.compressor.wgate.weight",
    "self_attn.compressor.indexer.position_bias": "attn.indexer.compressor.ape",
    "self_attn.compressor.indexer.kv_norm.weight": "attn.indexer.compressor.norm.weight",
    "self_attn.compressor.indexer.q_b_proj.weight": "attn.indexer.wq_b.weight",
    "self_attn.compressor.indexer.scorer.weights_proj.weight": "attn.indexer.weights_proj.weight",
}

# top-level (non-layer) HF -> native
GLOBAL_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "head.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
}


class RealWeightStore:
    def __init__(self, snapshot: str, cache_experts: bool = True):
        self.snapshot = snapshot
        idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]  # native key -> shard filename
        self._handles = {}
        # cache of dequantized routed experts, keyed (layer_idx, expert_id) -> (gate_up, down).
        # Experts are static and (esp. for hash layers) reused every generated token, so this
        # makes token 2+ skip fp4 dequant. Bounded by #unique experts touched (fits host RAM).
        self._expert_cache = {} if cache_experts else None
        # cache of dequantized NON-EXPERT weights, keyed by native key. The streaming forward
        # re-loads every layer's attention/norm/mHC/lm_head weights each token; memoizing the
        # (expensive) fp8 block-dequant makes token 2+ skip it. Non-expert weights total ~16 GB
        # bf16 (fits host RAM); routed experts are excluded (they have their own bounded cache).
        self._deq_cache = {}

    def _f(self, shard):
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.snapshot, shard), framework="pt")
            self._handles[shard] = h
        return h

    def has(self, key):
        return key in self.wm

    def raw(self, key):
        return self._f(self.wm[key]).get_tensor(key)

    def deq(self, key):
        """Fetch a native tensor, dequantizing per its format:
        - fp8 e4m3 + .scale  -> block-128 dequant (attention/shared-expert/lm_head projections)
        - int8 (MXFP4) + .scale -> e2m1 LUT[nibble] × per-group e8m0 float scale (routed experts, expert_dtype=fp4)
        - else (bf16 norms/embed/compressor/sinks/hc/tid2eid) -> as-is."""
        # memoize dequant for non-expert weights (re-loaded every token by the streaming forward)
        cacheable = self._deq_cache is not None and "ffn.experts" not in key
        if cacheable and key in self._deq_cache:
            return self._deq_cache[key]
        w = self.raw(key)
        scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else key + ".scale"
        if w.dtype == torch.float8_e4m3fn and self.has(scale_key):
            out = dequantize_weight_tensor(w, self.raw(scale_key), BLOCK)
        elif w.dtype == torch.int8 and self.has(scale_key):
            out = _dequant_mxfp4(w, self.raw(scale_key))
        elif w.dtype == torch.float8_e4m3fn:
            out = w.to(torch.bfloat16)
        else:
            out = w
        if cacheable:
            self._deq_cache[key] = out
        return out


def load_globals(model, store: RealWeightStore):
    sd = {}
    for hf_k, nat in GLOBAL_MAP.items():
        if hf_k in dict(model.named_parameters()) or hf_k in dict(model.named_buffers()):
            sd[hf_k] = store.deq(nat)
    _assign(model, sd)


def load_layer(layer_module, layer_idx: int, store: RealWeightStore, skip_experts: bool = False):
    """Load real layer-`layer_idx` weights (dequantized) into `layer_module` (HF DecoderLayer).
    `skip_experts=True` leaves the 256 routed-expert tensors untouched (they are streamed on
    demand per selected expert via `expert_gate_up`/`expert_down` — avoids dequantizing all 256)."""
    p = f"layers.{layer_idx}."
    target = dict(layer_module.named_parameters())
    target.update(dict(layer_module.named_buffers()))
    new = {}
    for hf_k in target:
        if hf_k in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj") and skip_experts:
            continue
        if hf_k in LAYER_MAP:
            new[hf_k] = store.deq(p + LAYER_MAP[hf_k])
        elif hf_k == "mlp.experts.gate_up_proj":
            new[hf_k] = _assemble_experts(store, p, ("w1", "w3"), cat=True)
        elif hf_k == "mlp.experts.down_proj":
            new[hf_k] = _assemble_experts(store, p, ("w2",), cat=False)
        elif "inv_freq" in hf_k or "rotary" in hf_k:
            continue  # RoPE inv_freq are computed from config at __init__, not in the checkpoint
        else:
            raise KeyError(f"no native mapping for HF key {hf_k!r} (layer {layer_idx})")
    _assign(layer_module, new)


def _expert(store: RealWeightStore, layer_idx: int, e: int):
    """Cached, PRE-TRANSPOSED expert weights ready for x @ W: gate_up_T [H, 2I], down_T [I, H].
    Transposing once (here, cached) avoids a per-call .t().contiguous() on every token."""
    cache = store._expert_cache
    if cache is not None and (layer_idx, e) in cache:
        return cache[(layer_idx, e)]
    p = f"layers.{layer_idx}.ffn.experts.{e}."
    gate_up = torch.cat([store.deq(p + "w1.weight"), store.deq(p + "w3.weight")], dim=0)  # [2I, H]
    down = store.deq(p + "w2.weight")  # [H, I]
    gate_up_T = gate_up.t().contiguous()  # [H, 2I]
    down_T = down.t().contiguous()  # [I, H]
    if cache is not None:
        cache[(layer_idx, e)] = (gate_up_T, down_T)
    return gate_up_T, down_T


def expert_fused(store: RealWeightStore, layer_idx: int, e: int):
    """(gate_up_T [H,2I], down_T [I,H]) for expert `e`, cached + pre-transposed."""
    return _expert(store, layer_idx, e)


def expert_fused_dev(store: RealWeightStore, device, layer_idx: int, e: int, dtype=None):
    """Device-RESIDENT expert weights (gate_up_T, down_T) as ttnn tensors, cached on `device`.

    First touch: dequantize (host) then upload as `dtype` (default bfloat16 — fast upload with
    no host-side quantization cost, best accuracy, and the working set of ~20-30 routed
    experts/layer still fits in device DRAM ~13 GB/chip) and keep resident. Reused across all
    subsequent tokens → no per-token re-transfer/dequant."""
    import ttnn

    dtype = dtype or ttnn.bfloat16
    key = (layer_idx, e, "dev")
    cache = store._expert_cache
    if cache is not None and key in cache:
        return cache[key]
    gate_up_T, down_T = _expert(store, layer_idx, e)  # host bf16 (cached)
    to = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    dev = (to(gate_up_T), to(down_T))
    if cache is not None:
        cache[key] = dev
    return dev


def _assemble_experts(store, p, parts, cat):
    """Stack per-expert native weights into the HF packed tensor [E, out, in]."""
    e = 0
    tensors = []
    while store.has(f"{p}ffn.experts.{e}.{parts[0]}.weight"):
        pieces = [store.deq(f"{p}ffn.experts.{e}.{part}.weight") for part in parts]
        tensors.append(torch.cat(pieces, dim=0) if cat else pieces[0])
        e += 1
    return torch.stack(tensors, dim=0)


def _assign(module, name_to_tensor):
    """In-place copy tensors into the module's params/buffers, casting to their dtype."""
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    with torch.no_grad():
        for name, t in name_to_tensor.items():
            dst = params.get(name, buffers.get(name))
            if dst is None:
                raise KeyError(f"target {name!r} not found in module")
            dst.copy_(t.to(dst.dtype).reshape(dst.shape))


def build_scratch(scfg):
    """Build the scratch HF model WITHOUT the (expensive) random weight init. from_config's
    `_init_weights` normal_-fills ~34B params (5 full-dim layers × 256 experts) — ~4 min — but
    every weight we use is overwritten by the real-weight loader and the routed-expert tensors
    are never read (streamed from the store). So we no-op `_init_weights`: params are allocated
    uninitialized (fast) while module __init__ still computes the RoPE inv_freq buffers."""
    from transformers import AutoModelForCausalLM
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as md

    orig = md.DeepseekV4PreTrainedModel._init_weights
    md.DeepseekV4PreTrainedModel._init_weights = lambda self, module: None
    try:
        model = AutoModelForCausalLM.from_config(scfg, dtype=torch.bfloat16).eval()
    finally:
        md.DeepseekV4PreTrainedModel._init_weights = orig
    return model


def _mesh_n(device):
    n = getattr(device, "get_num_devices", None)
    try:
        return n() if n is not None else 1
    except Exception:
        return 1


def _res_upload(t, device, dtype, shard_dim=None):
    """Upload a resident weight to `device`. On a MeshDevice: REPLICATE (shard_dim=None) or
    SHARD along `shard_dim` (tensor-parallel). On a single device: plain upload."""
    import ttnn

    mapper = None
    if _mesh_n(device) > 1:
        mapper = (
            ttnn.ShardTensorToMesh(device, dim=shard_dim)
            if shard_dim is not None
            else ttnn.ReplicateTensorToMesh(device)
        )
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def dev_linear(store, device, key, w_out_in, dtype=None):
    """Cache a RESIDENT device weight for a linear, pre-transposed to [in, out]. `w_out_in` is
    the nn.Linear weight [out, in]. Cached in the store keyed by `key` (use the real layer idx).
    Replicated across a mesh (small projections resident on every chip)."""
    import ttnn

    dtype = dtype or ttnn.bfloat16
    cache = store._expert_cache
    if cache is not None and key in cache:
        return cache[key]
    tw = _res_upload(w_out_in.t().contiguous(), device, dtype)
    if cache is not None:
        cache[key] = tw
    return tw


def expert_sharded_dev(store, device, layer_idx: int, e: int, dtype=None):
    """RESIDENT, tensor-parallel-SHARDED expert weights for the mesh MoE (the memory-dominant
    fix, REPORT_7). Returns (tgate [H, I/C], tup [H, I/C], tdown [I/C, H]) as ttnn tensors,
    gate/up column-sharded and down row-sharded across the mesh (TP-exact for SwiGLU), cached
    resident. `C=1` (single device) degenerates to a plain resident upload."""
    import ttnn

    dtype = dtype or ttnn.bfloat16
    key = (layer_idx, e, "shard")
    cache = store._expert_cache
    if cache is not None and key in cache:
        return cache[key]
    p = f"layers.{layer_idx}.ffn.experts.{e}."
    gate_T = store.deq(p + "w1.weight").t().contiguous()  # [H, I]
    up_T = store.deq(p + "w3.weight").t().contiguous()  # [H, I]
    down_T = store.deq(p + "w2.weight").t().contiguous()  # [I, H]
    dev = (
        _res_upload(gate_T, device, dtype, shard_dim=-1),  # [H, I/C]
        _res_upload(up_T, device, dtype, shard_dim=-1),  # [H, I/C]
        _res_upload(down_T, device, dtype, shard_dim=0),  # [I/C, H]
    )
    if cache is not None:
        cache[key] = dev
    return dev


def shared_sharded_dev(store, device, layer_idx: int, se, dtype=None):
    """RESIDENT tensor-parallel-sharded SHARED expert (gate/up col-sharded, down row-sharded).
    `se` is the HF shared_experts module (source of the dequantized weights)."""
    import ttnn

    dtype = dtype or ttnn.bfloat16
    key = (layer_idx, "shared_shard")
    cache = store._expert_cache
    if cache is not None and key in cache:
        return cache[key]
    dev = (
        _res_upload(se.gate_proj.weight.data.t().contiguous(), device, dtype, shard_dim=-1),
        _res_upload(se.up_proj.weight.data.t().contiguous(), device, dtype, shard_dim=-1),
        _res_upload(se.down_proj.weight.data.t().contiguous(), device, dtype, shard_dim=0),
    )
    if cache is not None:
        cache[key] = dev
    return dev


def dev_grouped(store, device, key, w, n_groups, dtype=None):
    """Cache a RESIDENT grouped-linear weight [g, in_per_group, rank] for o_a. `w` is the
    GroupedLinear weight [g*rank, in_per_group]."""
    import ttnn

    dtype = dtype or ttnn.bfloat16
    cache = store._expert_cache
    if cache is not None and key in cache:
        return cache[key]
    ipg = w.shape[1]
    rank = w.shape[0] // n_groups
    wd = w.view(n_groups, rank, ipg).transpose(1, 2).contiguous()  # [g, ipg, rank]
    tw = _res_upload(wd, device, dtype)  # replicated across mesh
    if cache is not None:
        cache[key] = tw
    return tw


def find_snapshot():
    base = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    if not snaps:
        raise FileNotFoundError(f"no DeepSeek-V4-Flash snapshot under {base}")
    return snaps[-1] + "/"
