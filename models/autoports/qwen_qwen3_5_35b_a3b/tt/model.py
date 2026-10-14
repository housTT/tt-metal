# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

PREFILL_CHUNK = 8192
PREFILL_MULTIPLE = 128
BLOCK_SIZE = 32
LOGIT_BLOCK = 128


def _load_keys(checkpoint: Path, keys: Sequence[str]):
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = {}
    for key in keys:
        by_file.setdefault(index[key], []).append(key)
    tensors = {}
    for filename, names in by_file.items():
        with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as shard:
            tensors.update({name: shard.get_tensor(name).clone() for name in names})
    return tensors


GATE_TEMPLATES = ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices")


def _find_gates(obj, seen=None, depth=0, match=None):
    from models.common.modules.moe.tt_moe_gate import TTMoEGate

    match = match or (lambda o: isinstance(o, TTMoEGate))
    import types

    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 14:
        return []
    seen.add(id(obj))
    if match(obj):
        return [obj]
    found = []
    if isinstance(obj, types.MethodType):
        children = [obj.__self__, obj.__func__]
    elif isinstance(obj, types.FunctionType):
        children = [cell.cell_contents for cell in (obj.__closure__ or ()) if cell.cell_contents is not None]
    elif isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    else:
        try:
            children = list(object.__getattribute__(obj, "__dict__").values())
        except (AttributeError, TypeError):
            children = []
    for child in children:
        if isinstance(child, (str, bytes, int, float)):
            continue
        found += _find_gates(child, seen, depth + 1, match)
    return found


def _share_gate_templates(layer, shared):
    import torch
    import ttnn

    gates = _find_gates(layer)
    if len(gates) != 1:
        raise RuntimeError(f"expected one TTMoEGate per layer, found {len(gates)}")
    gate = gates[0]
    if not shared:
        shared.update({name: getattr(gate, name) for name in GATE_TEMPLATES})
        return 0
    for name in ("tt_bias", "tt_input_indices"):
        mine = ttnn.to_torch(ttnn.get_device_tensors(getattr(gate, name))[0])
        if not torch.equal(mine, ttnn.to_torch(ttnn.get_device_tensors(shared[name])[0])):
            raise RuntimeError(f"router {name} differs between layers; it cannot be shared")
    for name in GATE_TEMPLATES:
        getattr(gate, name).deallocate(True)
        setattr(gate, name, shared[name])
    return len(GATE_TEMPLATES)


def _share_router_slices(layer, shared):
    routers = _find_gates(layer, match=lambda o: isinstance(getattr(o, "__dict__", None), dict)
                          and isinstance(o.__dict__.get("cached_slices"), dict))
    for router in routers:
        if router.cached_slices:
            raise RuntimeError("router slice cache must be empty when it is shared")
        router.cached_slices = shared
    return len(routers)


SHAREABLE = ("_rope_cos", "_rope_sin", "rank_buffers", "buffers", "zero_templates", "_zero_recurrent", "_zero_conv",
             "_index_offsets", "region_matrix", "mask", "zero_row", "expert_ids", "ones", "zeros", "placeholder",
             "chunk_constants", "kda_start", "offset", "rotation")
SCRATCH = ("rank_buffers", "buffers", "placeholder")


def _collect_tensors(obj, path, out, seen, depth=0):
    import ttnn

    if id(obj) in seen or depth > 16:
        return
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        out.setdefault(path, obj)
        return
    if isinstance(obj, dict):
        items = [(f"{path}[{key!r}]", value) for key, value in obj.items()]
    elif isinstance(obj, (list, tuple)):
        items = [(f"{path}[{i}]", value) for i, value in enumerate(obj)]
    else:
        try:
            items = [(f"{path}.{key}", value) for key, value in object.__getattribute__(obj, "__dict__").items()]
        except (AttributeError, TypeError):
            return
    for child_path, child in items:
        if not isinstance(child, (str, bytes, int, float, bool, type(None))):
            _collect_tensors(child, child_path, out, seen, depth + 1)


def _replace_references(old, new, skip):
    import gc
    import types

    for ref in gc.get_referrers(old):
        if id(ref) in skip or isinstance(ref, types.FrameType):
            continue
        if isinstance(ref, dict):
            for key, value in list(ref.items()):
                if value is old:
                    ref[key] = new
        elif isinstance(ref, list):
            for i, value in enumerate(ref):
                if value is old:
                    ref[i] = new
        elif isinstance(ref, tuple):
            _replace_references(ref, tuple(new if value is old else value for value in ref), skip)
        elif isinstance(ref, types.CellType):
            ref.cell_contents = new


def _device_copies(tensor):
    import ttnn

    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _share_layer_constants(layer, registry):
    import re

    import torch

    found = {}
    _collect_tensors(layer, "", found, set())
    shared = 0
    for path, tensor in found.items():
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path))
        if not names & set(SHAREABLE) or not tensor.is_allocated():
            continue
        key = (path, tuple(tensor.shape), str(tensor.dtype), str(tensor.layout), str(tensor.memory_config()))
        first = registry.setdefault(key, tensor)
        if first is tensor:
            continue
        if not names & set(SCRATCH):
            if any(not torch.equal(a, b) for a, b in zip(_device_copies(tensor), _device_copies(first))):
                raise RuntimeError(f"{path} differs between layers; it cannot be shared")
        shared += 1
        _replace_references(tensor, first, {id(found), id(registry)})
    found.clear()
    return shared


def _kv_shapes(hf_config, tp, num_blocks, block_size):
    from .functional_decoder import FunctionalDecoderConfig
    from .multichip_decoder import DEFAULT_POLICIES, local_hf_config, mesh_policy

    config = FunctionalDecoderConfig.from_hf_config(local_hf_config(hf_config, tp) if tp > 1 else hf_config)
    policy = mesh_policy(DEFAULT_POLICIES["full_attention"], tp, "full_attention")
    return (num_blocks + 1, config.n_kv_heads, block_size, config.head_dim), policy.get("kv_dtype", "bf16")


def _gdn_state_shapes(hf_config, tp):
    from .multichip_decoder import local_hf_config

    text = local_hf_config(hf_config, tp) if tp > 1 else hf_config
    nk, nv = text.linear_num_key_heads, text.linear_num_value_heads
    dk, dv = text.linear_key_head_dim, text.linear_value_head_dim
    return (1, nv, dk, dv), (1, text.linear_conv_kernel_dim - 1, 2 * nk * dk + nv * dv)


def _adopt_request_slots(layer, slots, max_batch_size):
    if layer._states:
        state, conv = layer._states[0]
        for extra_state, extra_conv in slots:
            for a, b in ((state, extra_state), (conv, extra_conv)):
                if (tuple(a.shape), a.dtype, a.layout, a.memory_config()) != (
                        tuple(b.shape), b.dtype, b.layout, b.memory_config()):
                    raise RuntimeError(f"request slot pool does not match layer {layer.layer_idx} state {a.shape}")
        layer._states = layer._states[:1] + list(slots)
    layer.max_batch_size = max_batch_size


@dataclass
class KVPool:
    num_blocks: int
    block_size: int
    caches: list

    @property
    def scratch_block(self) -> int:
        return self.num_blocks


class Qwen35Model:
    def __init__(self, mesh_device, hf_config, layers, embedding, norm, lm_head, layer_ids, load_seconds):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layers = layers
        self.layer_ids = tuple(layer_ids)
        self.embedding = embedding
        self.norm = norm
        self.lm_head = lm_head
        self.load_seconds = load_seconds
        self.tp = mesh_device.get_num_devices()
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.max_seq_len = hf_config.max_position_embeddings
        self.full_layers = [layer for layer in layers if layer.config.is_full_attention_layer(layer.layer_idx)]
        self.linear_layers = [layer for layer in layers if not layer.config.is_full_attention_layer(layer.layer_idx)]
        self.max_batch_size = layers[0].max_batch_size
        self.kv_pool: Optional[KVPool] = None
        self._row_index = None
        self._block_index = None
        self.pad_fallbacks = 0
        self.audit = False

    def _guard(self):
        from contextlib import nullcontext

        if not self.audit:
            return nullcontext()
        from ..tests.runtime_audit import device_only

        return device_only()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        mesh_device,
        *,
        layer_ids: Optional[Sequence[int]] = None,
        max_batch_size: int = 32,
        tensor_cache_path: Optional[Path] = None,
        kv_blocks: Optional[int] = None,
        log=print,
    ) -> "Qwen35Model":
        import gc

        import torch
        import ttnn
        from transformers import AutoConfig

        from ..tests.layer_weights import load_layer_state_dict
        from .fused_decoder import _FusedRMSNorm
        from .multichip_decoder import MultichipDecoder

        checkpoint = Path(checkpoint)
        start = time.perf_counter()
        hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
        if getattr(hf_config, "tie_word_embeddings", False):
            raise ValueError("Qwen3.5-35B-A3B uses an untied LM head")
        ids = tuple(range(hf_config.num_hidden_layers)) if layer_ids is None else tuple(layer_ids)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        cache = None if tensor_cache_path is None else Path(tensor_cache_path)

        def cached(name):
            return None if cache is None else str(cache / f"tp{mesh_device.get_num_devices()}_{name}")

        names = ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight")
        top = _load_keys(checkpoint, names)
        embedding = ttnn.as_tensor(top[names[0]].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh_device, mesh_mapper=replicate, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   cache_file_name=cached("embed_tokens"))
        norm = _FusedRMSNorm(mesh_device, top[names[1]], hf_config.rms_norm_eps)
        lm_head = ttnn.as_tensor(top[names[2]].to(torch.bfloat16).transpose(0, 1).contiguous(), dtype=ttnn.bfloat8_b,
                                 layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                 mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=cached("lm_head_bfp8"))
        del top
        pool = None
        if kv_blocks:
            shape, kv_dtype = _kv_shapes(hf_config, mesh_device.get_num_devices(), kv_blocks, BLOCK_SIZE)
            dtype = ttnn.bfloat8_b if kv_dtype == "bfp8" else ttnn.bfloat16
            full_count = sum(1 for i in ids if hf_config.layer_types[i] == "full_attention")
            pool = KVPool(kv_blocks, BLOCK_SIZE, [
                [ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)] for _ in range(full_count)])
        late_slots = bool(kv_blocks) and max_batch_size > 1
        layers = []
        shared_gate, shared_slices, constants = {}, {}, {}
        for idx in ids:
            t0 = time.perf_counter()
            state = load_layer_state_dict(checkpoint, idx)
            layer = MultichipDecoder.from_state_dict(state, hf_config=hf_config, layer_idx=idx, mesh_device=mesh_device,
                                                     max_batch_size=1 if late_slots else max_batch_size,
                                                     tensor_cache_path=cache)
            layer.optimization = replace(layer.optimization, prefill_bucket_plan=True)
            layers.append(layer)
            del state
            _share_gate_templates(layers[-1], shared_gate)
            if _share_router_slices(layers[-1], shared_slices) != 1:
                raise RuntimeError("expected one router slice cache per layer")
            _share_layer_constants(layers[-1], constants)
            gc.collect()
            log(f"LAYER_LOADED {idx} {time.perf_counter() - t0:.1f}s")
        if late_slots:
            rec_shape, conv_shape = _gdn_state_shapes(hf_config, mesh_device.get_num_devices())
            for layer in layers:
                slots = []
                if layer._states:
                    slots = [(ttnn.zeros(rec_shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG),
                              ttnn.zeros(conv_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                             for _ in range(max_batch_size - 1)]
                _adopt_request_slots(layer, slots, max_batch_size)
        model = cls(mesh_device, hf_config, layers, embedding, norm, lm_head, ids, time.perf_counter() - start)
        if pool is not None:
            model.bind_kv_cache(pool.caches)
        return model

    def allocate_kv_cache(self, num_blocks: int, block_size: int = BLOCK_SIZE) -> KVPool:
        import ttnn

        caches = []
        for layer in self.full_layers:
            dtype = ttnn.bfloat8_b if layer.optimization.kv_dtype == "bfp8" else ttnn.bfloat16
            shape = (num_blocks + 1, layer.config.n_kv_heads, block_size, layer.config.head_dim)
            key = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
            value = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
            layer.bind_paged_kv_cache(key, value, scratch_block=num_blocks)
            caches.append([key, value])
        self.kv_pool = KVPool(num_blocks, block_size, caches)
        return self.kv_pool

    def bind_kv_cache(self, kv_cache, block_size: int = BLOCK_SIZE) -> None:
        if len(kv_cache) != len(self.full_layers):
            raise ValueError(f"kv_cache needs {len(self.full_layers)} [K,V] pairs")
        num_blocks = int(kv_cache[0][0].shape[0]) - 1
        for layer, (key, value) in zip(self.full_layers, kv_cache):
            layer.bind_paged_kv_cache(key, value, scratch_block=num_blocks)
        self.kv_pool = KVPool(num_blocks, block_size, list(kv_cache))

    def zero_kv_cache(self) -> None:
        import ttnn

        for pair in self.kv_pool.caches:
            for tensor in pair:
                ttnn.mul_(tensor, 0.0)

    def reset_request_state(self, request_slots: Optional[Sequence[int]] = None) -> None:
        for layer in self.linear_layers:
            layer.reset_request_state(request_slots)

    def embed(self, tokens):
        import ttnn

        return ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def logits(self, hidden):
        import ttnn

        if hidden.is_sharded():
            hidden = ttnn.sharded_to_interleaved(hidden, ttnn.L1_MEMORY_CONFIG)
        normalized = self.norm(hidden)
        return ttnn.matmul(normalized, self.lm_head, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def physical_length(self, length: int) -> int:
        size = PREFILL_MULTIPLE
        while size < length:
            size *= 2
        return size

    def prefill_chunks(self, start_pos: int, length: int):
        chunks, pos, end = [], start_pos, start_pos + length
        while pos < end:
            size = min(PREFILL_CHUNK - pos % PREFILL_CHUNK, end - pos)
            chunks.append((pos, size))
            pos += size
        return chunks

    def _physical_view(self, out, physical):
        import ttnn

        batch, rows, hidden = out.shape
        padded = tuple(out.padded_shape)
        if rows == physical and padded[1] == physical:
            return out
        if padded[1] >= physical:
            return ttnn.reshape(ttnn.reshape(out, (batch, physical, hidden), padded), (batch, physical, hidden))
        self.pad_fallbacks += 1
        return ttnn.pad(ttnn.reshape(out, (batch, rows, hidden)), ((0, 0), (0, physical - rows), (0, 0)), 0.0)

    def _last_rows(self, hidden, valid_rows):
        import torch
        import ttnn

        batch, physical, _ = hidden.shape
        index = torch.tensor([[valid_rows - 1 + r * physical] for r in range(batch)], dtype=torch.int32)
        if self._row_index is None or tuple(self._row_index.shape) != (batch, 1):
            self._row_index = ttnn.from_torch(index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                              device=self.mesh_device,
                                              mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
        else:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)), self._row_index)
        table = ttnn.to_layout(ttnn.reshape(hidden, (batch * physical, hidden.shape[-1])), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.embedding(self._row_index, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        table.deallocate(True)
        return ttnn.reshape(rows, (batch, 1, hidden.shape[-1]))

    def all_logits(self, hidden, valid_rows, block=LOGIT_BLOCK):
        import torch
        import ttnn

        _, physical, width = hidden.shape
        table = ttnn.to_layout(ttnn.reshape(hidden, (physical, width)), ttnn.ROW_MAJOR_LAYOUT)
        if self._block_index is None:
            self._block_index = ttnn.from_torch(torch.zeros(1, block, dtype=torch.int32), dtype=ttnn.uint32,
                                                layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
                                                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        pieces = []
        for first in range(0, valid_rows, block):
            index = torch.clamp(torch.arange(first, first + block, dtype=torch.int32), max=physical - 1)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(index.reshape(1, block), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)), self._block_index)
            rows = ttnn.embedding(self._block_index, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            logits = self.logits(rows)
            rows.deallocate(True)
            host = ttnn.to_torch(logits, mesh_composer=composer).float().reshape(block, -1)[:, : self.vocab_size]
            logits.deallocate(True)
            pieces.append(host[: min(block, valid_rows - first)])
        table.deallocate(True)
        return torch.cat(pieces)

    def prefill(self, tokens, *, start_pos: int = 0, page_table=None, request_slots=None,
                return_all_logits: bool = False):
        import torch
        import ttnn

        batch, length = tokens.shape
        if batch != 1:
            raise ValueError("model prefill takes one request row; the generator prefills rows separately")
        if length <= 0 or start_pos < 0 or start_pos + length > self.max_seq_len:
            raise ValueError("prompt exceeds the supported context")
        if start_pos % PREFILL_MULTIPLE:
            raise ValueError(f"prefill start {start_pos} is not a multiple of {PREFILL_MULTIPLE}; "
                             "continuation prefill starts on a 128-token boundary")
        if self.full_layers and page_table is None:
            raise ValueError("full attention needs a page table")
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        outputs = []
        hidden = None
        for pos, size in self.prefill_chunks(start_pos, length):
            physical = self.physical_length(size)
            ids = tokens[:, pos - start_pos : pos - start_pos + size].to(torch.int32)
            ids = torch.nn.functional.pad(ids, (0, physical - size))
            tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
                                     mesh_mapper=replicate)
            with self._guard():
                hidden = self.embed(tt_ids)
                tt_ids.deallocate(True)
                for layer in self.layers:
                    kw = dict(valid_len=size, start_pos=pos, request_slots=request_slots)
                    if layer.config.is_full_attention_layer(layer.layer_idx):
                        kw["page_table"] = page_table
                    out = layer.prefill_forward(hidden, **kw)
                    hidden.deallocate(True)
                    hidden = self._physical_view(out, physical)
            if return_all_logits:
                outputs.append(self.all_logits(hidden, size))
                hidden.deallocate(True)
                hidden = None
            elif pos + size < start_pos + length:
                hidden.deallocate(True)
                hidden = None
            else:
                last = self._last_rows(hidden, size)
                hidden.deallocate(True)
                hidden = None
                outputs = self.logits(last)
                last.deallocate(True)
        return outputs

    def decode(self, tokens, position_tensor, page_table, request_slots=None):
        import ttnn

        hidden = self.embed(tokens)
        for layer in self.layers:
            kw = dict(request_slots=request_slots)
            if layer.config.is_full_attention_layer(layer.layer_idx):
                kw.update(position_tensor=position_tensor, page_table=page_table)
            out = layer.decode_forward(hidden, **kw)
            hidden.deallocate(True)
            hidden = out
        logits = self.logits(hidden)
        hidden.deallocate(True)
        return logits
