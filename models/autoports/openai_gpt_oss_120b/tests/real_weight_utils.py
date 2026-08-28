# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layer-local real-weight loading for GPT-OSS 120B functional tests.

The public checkpoint stores expert weights as MXFP4 blocks and scales.  This
module reads only the safetensor shards referenced by one decoder layer and
dequantizes only that layer's two expert tensors.  It deliberately avoids
``from_pretrained`` because constructing the complete 120B model is unnecessary
for a single-layer correctness test.

``load_real_layer_state_dict`` returns local keys (for example,
``self_attn.q_proj.weight``) in exactly the form accepted by
``GptOssDecoderLayer.load_state_dict`` and ``FunctionalDecoder.from_state_dict``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

_INDEX_FILENAME = "model.safetensors.index.json"
_SUPPORTED_LAYERS = (0, 1)

_EXPECTED_LOCAL_SHAPES = {
    "input_layernorm.weight": (2880,),
    "post_attention_layernorm.weight": (2880,),
    "self_attn.sinks": (64,),
    "self_attn.q_proj.weight": (4096, 2880),
    "self_attn.q_proj.bias": (4096,),
    "self_attn.k_proj.weight": (512, 2880),
    "self_attn.k_proj.bias": (512,),
    "self_attn.v_proj.weight": (512, 2880),
    "self_attn.v_proj.bias": (512,),
    "self_attn.o_proj.weight": (2880, 4096),
    "self_attn.o_proj.bias": (2880,),
    "mlp.router.weight": (128, 2880),
    "mlp.router.bias": (128,),
    "mlp.experts.gate_up_proj": (128, 2880, 5760),
    "mlp.experts.gate_up_proj_bias": (128, 5760),
    "mlp.experts.down_proj": (128, 2880, 2880),
    "mlp.experts.down_proj_bias": (128, 2880),
}

_PACKED_EXPERT_SHAPES = {
    "mlp.experts.gate_up_proj_blocks": (128, 5760, 90, 16),
    "mlp.experts.gate_up_proj_scales": (128, 5760, 90),
    "mlp.experts.down_proj_blocks": (128, 2880, 90, 16),
    "mlp.experts.down_proj_scales": (128, 2880, 90),
}


def _weight_map(snapshot_path: Path) -> Mapping[str, str]:
    index_path = snapshot_path / _INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"GPT-OSS safetensor index is missing: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        index = json.load(index_file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a safetensor weight_map")
    return weight_map


def _layer_locations(snapshot_path: Path, layer_idx: int) -> dict[str, str]:
    if layer_idx not in _SUPPORTED_LAYERS:
        raise ValueError(f"real-weight functional tests cover layers {_SUPPORTED_LAYERS}, got {layer_idx}")

    prefix = f"model.layers.{layer_idx}."
    locations = {
        checkpoint_key[len(prefix) :]: shard_name
        for checkpoint_key, shard_name in _weight_map(snapshot_path).items()
        if checkpoint_key.startswith(prefix)
    }
    expected_raw_keys = set(_EXPECTED_LOCAL_SHAPES) - {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} | set(
        _PACKED_EXPERT_SHAPES
    )
    missing = sorted(expected_raw_keys - locations.keys())
    unexpected = sorted(locations.keys() - expected_raw_keys)
    if missing or unexpected:
        raise ValueError(f"Layer {layer_idx} checkpoint-key mismatch: missing={missing}, unexpected={unexpected}")

    missing_shards = sorted({name for name in locations.values() if not (snapshot_path / name).is_file()})
    if missing_shards:
        raise FileNotFoundError(f"Layer {layer_idx} safetensor shards are missing: {missing_shards}")
    return locations


def validate_real_layer_metadata(snapshot_path: str | Path, layer_idx: int) -> dict[str, tuple[int, ...]]:
    """Validate layer keys, shard presence, raw shapes, and checkpoint dtypes without loading weights.

    The returned mapping describes the dense local state dict that
    :func:`load_real_layer_state_dict` will produce.  ``safe_open.get_slice``
    keeps this check metadata-only even though a dense layer is several GiB.
    """

    snapshot_path = Path(snapshot_path)
    locations = _layer_locations(snapshot_path, layer_idx)
    handles = {}
    try:
        for local_key, shard_name in locations.items():
            handle = handles.setdefault(
                shard_name,
                safe_open(snapshot_path / shard_name, framework="pt", device="cpu"),
            )
            tensor_slice = handle.get_slice(f"model.layers.{layer_idx}.{local_key}")
            expected_shape = _PACKED_EXPERT_SHAPES.get(local_key, _EXPECTED_LOCAL_SHAPES.get(local_key))
            actual_shape = tuple(tensor_slice.get_shape())
            if actual_shape != expected_shape:
                raise ValueError(f"Layer {layer_idx} {local_key} shape is {actual_shape}, expected {expected_shape}")
            expected_dtype = "U8" if local_key in _PACKED_EXPERT_SHAPES else "BF16"
            actual_dtype = tensor_slice.get_dtype()
            if actual_dtype != expected_dtype:
                raise TypeError(f"Layer {layer_idx} {local_key} dtype is {actual_dtype}, expected {expected_dtype}")
    finally:
        handles.clear()

    return dict(_EXPECTED_LOCAL_SHAPES)


def load_real_layer_state_dict(
    snapshot_path: str | Path,
    layer_idx: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load and dequantize one real GPT-OSS 120B decoder layer on CPU.

    Only layers 0 (sliding attention) and 1 (full attention) are accepted: they
    cover every meaningful target decoder-layer kind.  Non-expert tensors are
    read verbatim from their indexed shard.  The MXFP4 expert blocks and scales
    are converted to ordinary dense HF tensors in ``dtype``.
    """

    if not dtype.is_floating_point:
        raise TypeError(f"dense layer dtype must be floating point, got {dtype}")

    snapshot_path = Path(snapshot_path)
    locations = _layer_locations(snapshot_path, layer_idx)
    prefix = f"model.layers.{layer_idx}."
    state_dict = {}
    handles = {}
    try:
        for local_key, shard_name in locations.items():
            handle = handles.setdefault(
                shard_name,
                safe_open(snapshot_path / shard_name, framework="pt", device="cpu"),
            )
            state_dict[local_key] = handle.get_tensor(prefix + local_key)
    finally:
        handles.clear()

    for projection in ("gate_up_proj", "down_proj"):
        packed_prefix = f"mlp.experts.{projection}"
        blocks = state_dict.pop(f"{packed_prefix}_blocks")
        scales = state_dict.pop(f"{packed_prefix}_scales")
        state_dict[packed_prefix] = convert_moe_packed_tensors(blocks, scales, dtype=dtype)

    state_dict = {
        key: tensor.to(dtype=dtype) if tensor.is_floating_point() and tensor.dtype != dtype else tensor
        for key, tensor in state_dict.items()
    }
    actual_shapes = {key: tuple(tensor.shape) for key, tensor in state_dict.items()}
    if actual_shapes != _EXPECTED_LOCAL_SHAPES:
        missing = sorted(set(_EXPECTED_LOCAL_SHAPES) - set(actual_shapes))
        unexpected = sorted(set(actual_shapes) - set(_EXPECTED_LOCAL_SHAPES))
        wrong_shapes = {
            key: (actual_shapes[key], _EXPECTED_LOCAL_SHAPES[key])
            for key in actual_shapes.keys() & _EXPECTED_LOCAL_SHAPES.keys()
            if actual_shapes[key] != _EXPECTED_LOCAL_SHAPES[key]
        }
        raise ValueError(
            f"Layer {layer_idx} dense state-dict mismatch: missing={missing}, "
            f"unexpected={unexpected}, wrong_shapes={wrong_shapes}"
        )
    wrong_dtypes = {key: tensor.dtype for key, tensor in state_dict.items() if tensor.dtype != dtype}
    if wrong_dtypes:
        raise TypeError(f"Layer {layer_idx} dense state-dict dtype mismatch: {wrong_dtypes}")
    return state_dict
