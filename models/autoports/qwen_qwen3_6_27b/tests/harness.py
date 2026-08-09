# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test-side plumbing: reference runs, paged page tables, device tensor prep, PCC.

Everything here is a *test harness* boundary and is allowed to touch torch; the layer under
test (``tt/functional_decoder.py``) never does outside ``from_state_dict``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import ttnn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from models.autoports.qwen_qwen3_6_27b.reference import hf_reference as ref
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import DEFAULT_BLOCK_SIZE, FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.model_config import FULL_ATTENTION

LINEAR_LAYER_IDX = 0
FULL_LAYER_IDX = 3
PCC_BAR = 0.995

#: Decoder implementation :func:`build_layer` instantiates when no ``decoder_cls`` is given.
#: ``tests/test_fused_decoder.py`` re-points this at ``FusedDecoder`` so the whole functional
#: suite runs unchanged against the fused layer.
DECODER_CLS = FunctionalDecoder


#: Prefix of the machine-readable evidence lines the tests emit (parsed by
#: ``scripts/collect_evidence.py`` into ``doc/functional_decoder/pcc_evidence.json``).
EVIDENCE_PREFIX = "PCCEVIDENCE "


def record(metric: str, value, **fields) -> None:
    """Emit one machine-readable evidence line so a run log carries the numbers, not just PASS."""
    payload = {"metric": metric, "value": value}
    payload.update(fields)
    print(EVIDENCE_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.detach().to(torch.float64).flatten()
    b = actual.detach().to(torch.float64).flatten()
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    assert torch.isfinite(b).all(), "actual tensor contains non-finite values"
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def make_page_table(max_batch: int, max_seq_len: int, block_size: int, seed: int = 1234) -> torch.Tensor:
    """Shuffled virtual→physical block map, ``[max_batch, blocks_per_user]`` int32.

    A permutation (rather than an identity map) is used deliberately so that any latent
    assumption about contiguous or zero-based cache slots shows up as a PCC failure.
    """
    blocks_per_user = math.ceil(max_seq_len / block_size)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(max_batch * blocks_per_user, generator=generator)
    return permutation.reshape(max_batch, blocks_per_user).to(torch.int32)


def to_device_page_table(table: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@dataclass
class LayerUnderTest:
    tt_layer: FunctionalDecoder
    ref_layer: torch.nn.Module
    config: object
    rotary: object
    layer_idx: int
    mesh_device: object
    page_table_torch: torch.Tensor | None
    page_table_tt: ttnn.Tensor | None
    max_batch: int
    max_seq_len: int
    block_size: int

    @property
    def is_full_attention(self) -> bool:
        return self.config.layer_types[self.layer_idx] == FULL_ATTENTION


#: Layers built during the current test, released by the ``release_layers`` autouse fixture.
_CREATED: list = []


def release_layers() -> None:
    """Free every device tensor allocated by ``build_layer`` since the last release."""
    while _CREATED:
        lut = _CREATED.pop()
        layer = lut.tt_layer
        tensors = []
        for value in layer.w.values():
            tensors.extend(value if isinstance(value, list) else [value])
        tensors.extend(layer.const.values())
        tensors.extend(layer.kv_cache or ())
        tensors.extend(t for t in (layer.conv_state, layer.recurrent_state) if t is not None)
        tensors.extend(t for t in layer.user_conv_state if t is not None)
        tensors.extend(t for t in layer.user_recurrent_state if t is not None)
        if lut.page_table_tt is not None:
            tensors.append(lut.page_table_tt)
        for tensor in tensors:
            try:
                ttnn.deallocate(tensor)
            except RuntimeError:
                pass


def build_layer(
    mesh_device,
    layer_idx: int,
    *,
    max_batch: int = 1,
    max_seq_len: int = 4096,
    block_size: int = DEFAULT_BLOCK_SIZE,
    real_weights: bool = False,
    seed: int = 0,
    cache_dtype=ttnn.bfloat16,
    decoder_cls=None,
) -> LayerUnderTest:
    decoder_cls = decoder_cls or DECODER_CLS
    config = ref.load_text_config()
    if real_weights:
        state_dict = ref.load_real_layer_state_dict(layer_idx)
    else:
        state_dict = ref.synthetic_state_dict_from_stats(ref.load_weight_stats(), layer_idx, config, seed=seed)
    ref_layer = ref.build_reference_layer(layer_idx, state_dict={k: v.clone() for k, v in state_dict.items()})

    tt_layer = decoder_cls.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        block_size=block_size,
        cache_dtype=cache_dtype,
    )

    page_table_torch = None
    page_table_tt = None
    if config.layer_types[layer_idx] == FULL_ATTENTION:
        page_table_torch = make_page_table(max_batch, max_seq_len, block_size)
        page_table_tt = to_device_page_table(page_table_torch, mesh_device)

    lut = LayerUnderTest(
        tt_layer=tt_layer,
        ref_layer=ref_layer,
        config=config,
        rotary=ref.make_rotary(config),
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        page_table_torch=page_table_torch,
        page_table_tt=page_table_tt,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        block_size=block_size,
    )
    _CREATED.append(lut)
    return lut


# ------------------------------------------------------------------ reference


def reference_prefill(lut: LayerUnderTest, hidden: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
    """Run the HF layer over ``hidden`` ``[1, seq, hidden]``, populating ``cache``."""
    seq = hidden.shape[1]
    past = cache.get_seq_length(lut.layer_idx) if lut.is_full_attention else 0
    positions = torch.arange(past, past + seq)
    cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=hidden.shape[0])
    text_position_ids = ref.build_text_position_ids(positions, hidden.shape[0])[0]
    mask = None
    if lut.is_full_attention:
        mask = ref.build_causal_mask(lut.config, hidden, cache, text_position_ids)
    with torch.no_grad():
        return lut.ref_layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=text_position_ids,
            past_key_values=cache,
            use_cache=True,
        )


def reference_decode(lut: LayerUnderTest, hidden: torch.Tensor, position: int, cache: DynamicCache):
    """One decode step at absolute ``position`` for ``hidden`` ``[1, 1, hidden]``."""
    positions = torch.tensor([position])
    cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=hidden.shape[0])
    text_position_ids = ref.build_text_position_ids(positions, hidden.shape[0])[0]
    mask = None
    if lut.is_full_attention:
        mask = ref.build_causal_mask(lut.config, hidden, cache, text_position_ids)
    with torch.no_grad():
        return lut.ref_layer(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            position_ids=text_position_ids,
            past_key_values=cache,
            use_cache=True,
        )


def reference_prefill_segmented(lut: LayerUnderTest, hidden: torch.Tensor, segment: int):
    """HF reference over a long prompt, one segment at a time, returning the last segment.

    ``linear_attention`` only.  ``Qwen3_5GatedDeltaNet`` continues exactly from a populated
    cache (it prepends ``conv_state`` and passes ``recurrent_state`` as ``initial_state``), so
    segmenting is mathematically identical to a single call but needs O(segment) memory
    instead of O(seq_len).  The layer has no positional input at all, so segment offsets do
    not matter.
    """
    assert not lut.is_full_attention, "segmented reference is for linear_attention only"
    cache = DynamicCache(config=lut.config)
    last = None
    for start in range(0, hidden.shape[1], segment):
        last = reference_prefill(lut, hidden[:, start : start + segment, :].contiguous(), cache)
    return last, cache


def fill_reference_kv_cache(lut: LayerUnderTest, hidden: torch.Tensor, cache: DynamicCache) -> None:
    """Write into ``cache`` exactly the K/V ``Qwen3_5Attention.forward`` would write.

    Only ``k_proj``/``v_proj``, ``k_norm`` and the partial RoPE are evaluated, so this is O(L)
    in both time and memory - unlike a real forward, whose eager attention would need
    ``num_heads * L**2 * 4`` bytes.  ``test_full_advertised_context`` validates the
    construction against a genuine short ``reference_prefill`` before relying on it.
    """
    assert lut.is_full_attention, "KV cache construction is for full_attention only"
    attn = lut.ref_layer.self_attn
    batch, seq_len, _ = hidden.shape
    past = cache.get_seq_length(lut.layer_idx)
    positions = torch.arange(past, past + seq_len)
    cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=batch)
    with torch.no_grad():
        # Qwen3_5DecoderLayer.forward norms *before* the mixer, so the attention sees
        # input_layernorm(hidden), not hidden.
        normed = lut.ref_layer.input_layernorm(hidden)
        shape = (batch, seq_len, -1, attn.head_dim)
        keys = attn.k_norm(attn.k_proj(normed).view(shape)).transpose(1, 2)
        values = attn.v_proj(normed).view(shape).transpose(1, 2)
        _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
        cache.update(keys, values, lut.layer_idx)


def reference_cache_kv(lut: LayerUnderTest, cache: DynamicCache, upto: int):
    """``(keys, values)`` of ``[n_kv, upto, head_dim]`` out of an HF cache, for comparison."""
    layer = cache.layers[lut.layer_idx]
    return (
        layer.keys[0, :, :upto, :].to(torch.float32),
        layer.values[0, :, :upto, :].to(torch.float32),
    )


# ------------------------------------------------------------------- device


def tt_hidden_prefill(hidden: torch.Tensor, mesh_device) -> ttnn.Tensor:
    assert hidden.shape[0] == 1, "prefill is single-user"
    return ttnn.from_torch(
        hidden.reshape(1, 1, hidden.shape[1], hidden.shape[2]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def tt_hidden_decode(hidden: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """``hidden`` ``[batch, 1, hidden]`` → device ``[1, 1, batch, hidden]``."""
    batch = hidden.shape[0]
    return ttnn.from_torch(
        hidden.reshape(1, 1, batch, hidden.shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def rope_permutation(lut: LayerUnderTest):
    """The layer's head-channel permutation, or ``None`` when it does not use one.

    ``FusedDecoder`` folds a channel permutation into the ``q``/``k`` projection weights so
    Qwen3.5's *partial* rotary embedding becomes a single full-width
    ``rotary_embedding_hf``; ``FunctionalDecoder`` does not.  Tests read this to build the
    matching ``cos``/``sin`` and to un-permute the K cache before comparing it with HF's.
    """
    return getattr(lut.tt_layer, "kv_channel_permutation", None)


def expand_rot_mats(lut: LayerUnderTest, cos: torch.Tensor, sin: torch.Tensor):
    """HF ``[..., rotary_dim]`` cos/sin → whatever the layer under test expects.

    For the functional decoder that is the identity.  For the fused decoder it is the
    ``head_dim``-wide form in permuted channel order: the rotary block's two halves sit either
    side of the rotate-half midpoint and every other channel gets ``cos = 1``, ``sin = 0`` so a
    full-width rotate-half leaves it alone.
    """
    perm = rope_permutation(lut)
    if perm is None:
        return cos, sin
    head_dim = len(perm)
    rot_dim = cos.shape[-1]
    half, mid = rot_dim // 2, head_dim // 2
    lead = cos.shape[:-1]
    cos_full = torch.ones(*lead, head_dim, dtype=cos.dtype)
    sin_full = torch.zeros(*lead, head_dim, dtype=sin.dtype)
    cos_full[..., :half] = cos[..., :half]
    cos_full[..., mid : mid + half] = cos[..., half:rot_dim]
    sin_full[..., :half] = sin[..., :half]
    sin_full[..., mid : mid + half] = sin[..., half:rot_dim]
    return cos_full.contiguous(), sin_full.contiguous()


def prefill_rot_mats(lut: LayerUnderTest, seq_len: int, mesh_device):
    positions = torch.arange(seq_len)
    cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=1)
    cos, sin = expand_rot_mats(lut, cos, sin)
    out = []
    for tensor in (cos, sin):
        out.append(
            ttnn.from_torch(
                tensor[0].reshape(1, 1, seq_len, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        )
    return tuple(out)


def decode_rot_mats_torch(lut: LayerUnderTest, positions: torch.Tensor):
    """``(cos, sin)`` as torch ``[1, batch, 1, rotary_dim]`` for decode."""
    cos, sin = ref.text_position_embeddings(lut.rotary, positions, batch=1)
    cos, sin = expand_rot_mats(lut, cos, sin)
    out = []
    for tensor in (cos, sin):
        per_user = tensor[0]  # [batch, rope_width]
        out.append(per_user.reshape(1, positions.numel(), 1, -1).contiguous())
    return tuple(out)


def to_device(tensor: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def chunk_page_tables(lut: LayerUnderTest, seq_len: int, user_id: int, mesh_device):
    """Per-chunk page-table slices plus the full per-user table, as device tensors."""
    if not lut.is_full_attention:
        return None, None
    user_table = lut.page_table_torch[user_id : user_id + 1, :]
    full = to_device_page_table(user_table, mesh_device)
    per_chunk = []
    for chunk_start, _logical, padded in lut.tt_layer.prefill_chunk_plan(seq_len):
        lo = chunk_start // lut.block_size
        hi = (chunk_start + padded) // lut.block_size
        assert hi <= user_table.shape[1], (
            f"page table covers {user_table.shape[1]} blocks, chunk needs {hi}"
        )
        per_chunk.append(to_device_page_table(user_table[:, lo:hi], mesh_device))
    return full, per_chunk


def run_tt_prefill(lut: LayerUnderTest, hidden: torch.Tensor, user_id: int = 0) -> torch.Tensor:
    seq_len = hidden.shape[1]
    tt_in = tt_hidden_prefill(hidden, lut.mesh_device)
    rot = prefill_rot_mats(lut, seq_len, lut.mesh_device) if lut.is_full_attention else None
    full_pt, per_chunk = chunk_page_tables(lut, seq_len, user_id, lut.mesh_device)
    tt_out = lut.tt_layer.prefill_forward(
        tt_in, user_id=user_id, page_table=full_pt, page_tables_per_chunk=per_chunk, rot_mats=rot
    )
    out = ttnn.to_torch(tt_out).reshape(1, seq_len, -1).to(torch.float32)
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_in)
    return out


def prepare_decode(lut: LayerUnderTest) -> None:
    """Fold per-user prefill state into the batch-wide decode buffers."""
    lut.tt_layer.prepare_decode_state()


def run_tt_decode(lut: LayerUnderTest, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``hidden`` ``[batch, 1, hidden]``, ``positions`` ``[batch]`` → ``[batch, 1, hidden]``."""
    batch = hidden.shape[0]
    tt_in = tt_hidden_decode(hidden, lut.mesh_device)
    rot = None
    pos_tt = None
    if lut.is_full_attention:
        cos, sin = decode_rot_mats_torch(lut, positions)
        rot = (to_device(cos, lut.mesh_device), to_device(sin, lut.mesh_device))
        pos_tt = to_device(positions.to(torch.int32), lut.mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    tt_out = lut.tt_layer.decode_forward(
        tt_in, current_pos=pos_tt, page_table=lut.page_table_tt, rot_mats=rot
    )
    out = ttnn.to_torch(tt_out).reshape(batch, 1, -1).to(torch.float32)
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_in)
    return out


# ------------------------------------------------------------------- state readback


def read_linear_state(lut: LayerUnderTest, user_id: int):
    """``(conv_state [K, conv_dim], recurrent_state [num_v_heads, dk, dv])`` for one user."""
    conv = ttnn.to_torch(lut.tt_layer.user_conv_state[user_id]).to(torch.float32)
    recurrent = ttnn.to_torch(lut.tt_layer.user_recurrent_state[user_id]).to(torch.float32)
    return conv.reshape(conv.shape[-2], conv.shape[-1]), recurrent.reshape(recurrent.shape[-3:])


def read_paged_kv(lut: LayerUnderTest, user_id: int, seq_len: int):
    """Un-page the device KV cache for one user: ``(keys, values)`` of ``[n_kv, seq_len, d]``.

    When the layer stores K in permuted head channels (the fused decoder — see
    :func:`rope_permutation`) the permutation is inverted here, so the caller always gets K in
    HF's channel order.  V is never permuted.
    """
    k_cache, v_cache = lut.tt_layer.kv_cache
    table = lut.page_table_torch[user_id]
    perm = rope_permutation(lut)
    out = []
    for cache in (k_cache, v_cache):
        blocks = ttnn.to_torch(cache).to(torch.float32)  # [num_blocks, n_kv, block, d]
        needed = math.ceil(seq_len / lut.block_size)
        gathered = blocks[table[:needed].to(torch.long)]  # [needed, n_kv, block, d]
        merged = gathered.permute(1, 0, 2, 3).reshape(gathered.shape[1], needed * lut.block_size, -1)
        out.append(merged[:, :seq_len, :])
    if perm is not None:
        inverse = torch.empty(len(perm), dtype=torch.long)
        inverse[torch.tensor(perm, dtype=torch.long)] = torch.arange(len(perm))
        out[0] = out[0][..., inverse]
    return tuple(out)


# ------------------------------------------------------------------ fallback audit


class _ForbidHostFallback:
    """Context manager that makes any host round-trip inside the measured pass an error."""

    _NAMES = ("from_torch", "to_torch", "as_tensor")

    def __enter__(self):
        self._saved = {name: getattr(ttnn, name) for name in self._NAMES}

        def _raise(name):
            def _stub(*args, **kwargs):
                raise AssertionError(f"host fallback: ttnn.{name} called inside a measured pass")

            return _stub

        for name in self._NAMES:
            setattr(ttnn, name, _raise(name))
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(ttnn, name, value)
        return False


def forbid_host_fallback() -> _ForbidHostFallback:
    return _ForbidHostFallback()


# ----------------------------------------------------------------- traced decode


class TracedDecode:
    """Capture and replay ``decode_forward`` with stable input buffers."""

    def __init__(self, lut: LayerUnderTest, batch: int):
        self.lut = lut
        self.batch = batch
        self.mesh_device = lut.mesh_device
        self.trace_id = None
        self.tt_x = None
        self.tt_pos = None
        self.tt_cos = None
        self.tt_sin = None
        self.output = None

    def _host_inputs(self, token: torch.Tensor, positions: torch.Tensor):
        x = token.reshape(1, 1, self.batch, -1)
        if not self.lut.is_full_attention:
            return x, None, None, None
        cos, sin = decode_rot_mats_torch(self.lut, positions)
        return x, positions.to(torch.int32), cos, sin

    def _alloc(self, token: torch.Tensor, positions: torch.Tensor):
        x, pos, cos, sin = self._host_inputs(token, positions)
        self.tt_x = to_device(x, self.mesh_device)
        if self.lut.is_full_attention:
            self.tt_pos = to_device(pos, self.mesh_device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.tt_cos = to_device(cos, self.mesh_device)
            self.tt_sin = to_device(sin, self.mesh_device)

    def _update(self, token: torch.Tensor, positions: torch.Tensor):
        x, pos, cos, sin = self._host_inputs(token, positions)
        pairs = [(x, self.tt_x, ttnn.bfloat16, ttnn.TILE_LAYOUT)]
        if self.lut.is_full_attention:
            pairs += [
                (pos, self.tt_pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                (cos, self.tt_cos, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (sin, self.tt_sin, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ]
        for host, device_tensor, dtype, layout in pairs:
            host_tensor = ttnn.from_torch(host, dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    def _forward(self):
        rot = (self.tt_cos, self.tt_sin) if self.lut.is_full_attention else None
        return self.lut.tt_layer.decode_forward(
            self.tt_x, current_pos=self.tt_pos, page_table=self.lut.page_table_tt, rot_mats=rot
        )

    def warmup(self, token: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self._alloc(token, positions)
        out = self._forward()
        ttnn.synchronize_device(self.mesh_device)
        result = ttnn.to_torch(out).reshape(self.batch, 1, -1).to(torch.float32)
        ttnn.deallocate(out)
        return result

    def capture(self) -> None:
        assert self.tt_x is not None, "call warmup() before capture()"
        ttnn.synchronize_device(self.mesh_device)
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.output = self._forward()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def replay(self, token: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert self.trace_id is not None, "call capture() first"
        self._update(token, positions)
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        return ttnn.to_torch(self.output).reshape(self.batch, 1, -1).to(torch.float32)

    def release(self) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None
