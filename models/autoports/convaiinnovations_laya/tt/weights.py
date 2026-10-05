# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Iterable, Optional

import torch

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import (
    DEFAULT_POLICY,
    WEIGHTS_DTYPE,
    expected_tensor_count,
    padded_intermediate,
)

ENCODER_PREFIX = "encoder."
HEAD_LAYER_KEYS = (
    "self_attn.in_proj_weight",
    "self_attn.in_proj_bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "linear1.weight",
    "linear1.bias",
    "linear2.weight",
    "linear2.bias",
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
)
SCORER_KEYS = ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")
ACT_HEAD_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
HEAD_LAYERS = 2


def default_weights_path() -> str:
    env = os.environ.get("LAYA_MODEL_DIR")
    if env:
        return os.path.join(env, "model.safetensors")
    return "/home/hous/dev/laya/state/laya_models/laya/model.safetensors"


def load_state_dict(path: Optional[str] = None, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = path or default_weights_path()
    sd = load_file(path)
    return {k: v.to(dtype) for k, v in sd.items()}


def split_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, object]:
    enc = {k[len(ENCODER_PREFIX) :]: v for k, v in sd.items() if k.startswith(ENCODER_PREFIX)}
    head = {k[len("head.") :]: v for k, v in sd.items() if k.startswith("head.")}
    scorer = {k[len("scorer.") :]: v for k, v in sd.items() if k.startswith("scorer.")}
    act = {k[len("act_head.") :]: v for k, v in sd.items() if k.startswith("act_head.")}
    return {
        "encoder": enc,
        "head": head,
        "scorer": scorer,
        "act_head": act,
        "type_emb": sd["type_emb.weight"],
        "temperature": sd["temperature"],
    }


def encoder_key_map(config) -> Iterable[str]:
    keys = ["embeddings.tok_embeddings.weight", "embeddings.norm.weight", "final_norm.weight"]
    for i in range(config.num_hidden_layers):
        if i > 0:
            keys.append(f"layers.{i}.attn_norm.weight")
        keys += [
            f"layers.{i}.attn.Wqkv.weight",
            f"layers.{i}.attn.Wo.weight",
            f"layers.{i}.mlp_norm.weight",
            f"layers.{i}.mlp.Wi.weight",
            f"layers.{i}.mlp.Wo.weight",
        ]
    return keys


def check_encoder_keys(enc: Dict[str, torch.Tensor], config) -> None:
    want = set(encoder_key_map(config))
    got = set(enc)
    if want != got:
        raise ValueError(f"encoder key map mismatch: missing={sorted(want - got)} extra={sorted(got - want)}")
    if len(got) != expected_tensor_count(config):
        raise ValueError(f"expected {expected_tensor_count(config)} encoder tensors, got {len(got)}")
    h, inter = config.hidden_size, config.intermediate_size
    shapes = {
        "embeddings.tok_embeddings.weight": (config.vocab_size, h),
        "layers.0.attn.Wqkv.weight": (3 * h, h),
        "layers.0.attn.Wo.weight": (h, h),
        "layers.0.mlp.Wi.weight": (2 * inter, h),
        "layers.0.mlp.Wo.weight": (h, inter),
    }
    for k, shape in shapes.items():
        if tuple(enc[k].shape) != shape:
            raise ValueError(f"{k} has shape {tuple(enc[k].shape)}, expected {shape}")


def check_head_keys(parts: Dict[str, object], config) -> None:
    h = config.hidden_size
    for i in range(HEAD_LAYERS):
        for k in HEAD_LAYER_KEYS:
            if f"layers.{i}.{k}" not in parts["head"]:
                raise ValueError(f"missing head key layers.{i}.{k}")
    for k in SCORER_KEYS:
        if k not in parts["scorer"]:
            raise ValueError(f"missing scorer key {k}")
    if tuple(parts["head"]["layers.0.self_attn.in_proj_weight"].shape) != (3 * h, h):
        raise ValueError("head in_proj_weight shape mismatch")
    if tuple(parts["scorer"]["3.weight"].shape) != (1, h):
        raise ValueError("scorer final linear shape mismatch")
    if tuple(parts["type_emb"].shape) != (3, h):
        raise ValueError("type_emb shape mismatch")


def fold_q_scale(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    scaled = weight.clone()
    q_rows = scaled.shape[0] // 3
    scaled[:q_rows] *= head_dim**-0.5
    return scaled


def fold_q_scale_bias(bias: torch.Tensor, head_dim: int) -> torch.Tensor:
    scaled = bias.clone()
    q = scaled.shape[0] // 3
    scaled[:q] *= head_dim**-0.5
    return scaled


def split_wi(wi: torch.Tensor, intermediate_size: int):
    return wi[:intermediate_size, :], wi[intermediate_size:, :]


def pad_up_projection(weight_out_in: torch.Tensor, width: int) -> torch.Tensor:
    out, inp = weight_out_in.shape
    if width == out:
        return weight_out_in
    padded = torch.zeros(width, inp, dtype=weight_out_in.dtype)
    padded[:out] = weight_out_in
    return padded


def pad_down_projection(weight_out_in: torch.Tensor, width: int) -> torch.Tensor:
    out, inp = weight_out_in.shape
    if width == inp:
        return weight_out_in
    padded = torch.zeros(out, width, dtype=weight_out_in.dtype)
    padded[:, :inp] = weight_out_in
    return padded


class _Uploader:
    def __init__(self, device, mesh_mapper=None):
        self.device = device
        self.mesh_mapper = mesh_mapper

    def _kw(self):
        kw = {"device": self.device}
        if self.mesh_mapper is not None:
            kw["mesh_mapper"] = self.mesh_mapper
        return kw

    def linear(self, weight_out_in, dtype):
        return ttnn.from_torch(
            weight_out_in.transpose(-1, -2).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, **self._kw()
        )

    def row(self, vec, dtype):
        return ttnn.from_torch(vec.reshape(1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, **self._kw())

    def norm(self, vec, dtype):
        return ttnn.from_torch(vec.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, **self._kw())

    def embedding(self, table, dtype):
        return ttnn.from_torch(table.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, **self._kw())


def _as_encoder_state_dict(source) -> Dict[str, torch.Tensor]:
    if isinstance(source, dict):
        sd = source
    else:
        sd = source.state_dict()
    if any(k.startswith(ENCODER_PREFIX) for k in sd):
        return split_state_dict(sd)["encoder"]
    return dict(sd)


def prepare_weights(
    source,
    config,
    device,
    policy=DEFAULT_POLICY,
    mesh_mapper=None,
    intermediate_pads=(None, 2816),
    dtype=WEIGHTS_DTYPE,
    layers=None,
):
    """Upload the 170 encoder tensors. Returns {"embeddings", "layers": [{attn_norm, attn, mlp_norm, mlp: {width: {...}}}], "final_norm"}."""
    enc = _as_encoder_state_dict(source)
    check_encoder_keys(enc, config)
    up = _Uploader(device, mesh_mapper)
    linear_dtype = policy.linear_dtype
    head_dim = config.hidden_size // config.num_attention_heads
    inter = config.intermediate_size
    widths = sorted({padded_intermediate(inter, p or 0) for p in intermediate_pads})
    wanted = None if layers is None else set(layers)

    params = {
        "embeddings": {
            "tok_embeddings": up.embedding(enc["embeddings.tok_embeddings.weight"], dtype),
            "norm": up.norm(enc["embeddings.norm.weight"], dtype),
        },
        "layers": [],
        "final_norm": up.norm(enc["final_norm.weight"], dtype),
        "widths": widths,
    }
    for i in range(config.num_hidden_layers):
        if wanted is not None and i not in wanted:
            params["layers"].append(None)
            continue
        p = f"layers.{i}."
        wi_act, wi_gate = split_wi(enc[p + "mlp.Wi.weight"], inter)
        mlp = {}
        for w in widths:
            mlp[w] = {
                "Wi_act": up.linear(pad_up_projection(wi_act, w), linear_dtype),
                "Wi_gate": up.linear(pad_up_projection(wi_gate, w), linear_dtype),
                "Wo": up.linear(pad_down_projection(enc[p + "mlp.Wo.weight"], w), linear_dtype),
            }
        params["layers"].append(
            {
                "attn_norm": None if i == 0 else up.norm(enc[p + "attn_norm.weight"], dtype),
                "attn": {
                    "Wqkv": up.linear(fold_q_scale(enc[p + "attn.Wqkv.weight"], head_dim), linear_dtype),
                    "Wo": up.linear(enc[p + "attn.Wo.weight"], linear_dtype),
                },
                "mlp_norm": up.norm(enc[p + "mlp_norm.weight"], dtype),
                "mlp": mlp,
            }
        )
    return params


def prepare_head_weights(sd, config, device, policy=DEFAULT_POLICY, mesh_mapper=None, dtype=WEIGHTS_DTYPE):
    """Upload type_emb, the two head layers (fused in_proj with the Q-scale fold) and the scorer."""
    parts = split_state_dict(sd) if "type_emb.weight" in sd else sd
    check_head_keys(parts, config)
    up = _Uploader(device, mesh_mapper)
    head_dim = config.hidden_size // config.num_attention_heads
    ldt = policy.head_linear_dtype
    layers = []
    for i in range(HEAD_LAYERS):
        g = lambda k: parts["head"][f"layers.{i}.{k}"]
        layers.append(
            {
                "norm1": {"weight": up.norm(g("norm1.weight"), dtype), "bias": up.norm(g("norm1.bias"), dtype)},
                "in_proj": {
                    "weight": up.linear(fold_q_scale(g("self_attn.in_proj_weight"), head_dim), ldt),
                    "bias": up.row(fold_q_scale_bias(g("self_attn.in_proj_bias"), head_dim), dtype),
                },
                "out_proj": {
                    "weight": up.linear(g("self_attn.out_proj.weight"), ldt),
                    "bias": up.row(g("self_attn.out_proj.bias"), dtype),
                },
                "norm2": {"weight": up.norm(g("norm2.weight"), dtype), "bias": up.norm(g("norm2.bias"), dtype)},
                "linear1": {"weight": up.linear(g("linear1.weight"), ldt), "bias": up.row(g("linear1.bias"), dtype)},
                "linear2": {"weight": up.linear(g("linear2.weight"), ldt), "bias": up.row(g("linear2.bias"), dtype)},
            }
        )
    sc = parts["scorer"]
    out_dtype = ttnn.float32 if policy.scorer_fp32_out else ldt
    scorer = {
        "norm": {"weight": up.norm(sc["0.weight"], dtype), "bias": up.norm(sc["0.bias"], dtype)},
        "dense": {"weight": up.linear(sc["1.weight"], ldt), "bias": up.row(sc["1.bias"], dtype)},
        "out": {
            "weight": up.linear(sc["3.weight"], out_dtype),
            "bias": up.row(sc["3.bias"], out_dtype if policy.scorer_fp32_out else dtype),
        },
    }
    return {
        "type_emb": up.embedding(parts["type_emb"], dtype),
        "layers": layers,
        "scorer": scorer,
        "temperature": parts["temperature"].clone(),
        "act_head": {k: v.clone() for k, v in parts["act_head"].items()},
    }


def deallocate_weights(params) -> None:
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        elif isinstance(node, ttnn.Tensor):
            if node.is_allocated():
                ttnn.deallocate(node)

    walk(params)
