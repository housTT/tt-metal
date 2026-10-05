import contextlib
import math
import os
import time
import warnings
from pathlib import Path

import torch

import ttnn
from models.autoports.cloudflare_clef.tt.loader import (
    VISUAL_PREFIX,
    read_tensors,
    read_weight_map,
    resolve_snapshot,
    revision_sha8,
)
from models.demos.blackhole.qwen36.tt.vision.functional import qwen3_5_vision_transformer_preprocess
from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)

BLOCK_TAPS = (0, 12, 23, 26)
STATE_DICT_PREFIX = "visual"
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
DTYPE_NAMES = {value: key for key, value in DTYPES.items()}
HOST_ONLY_KEYS = {
    "visual.patch_embed.wo.weight",
    "visual.patch_embed.wo.bias",
    "visual.pos_embed.weight",
}


def env_pad_granule():
    return int(os.environ.get("CLEF_VISION_PAD", "2048"))


def env_mask_padding():
    return os.environ.get("CLEF_VISION_PAD_MASK", "1") == "1"


def env_dtype():
    return DTYPES[os.environ.get("CLEF_VISION_DTYPE", "bf16")]


def read_vision_state_dict(snapshot=None):
    snapshot = snapshot or resolve_snapshot()
    weight_map = read_weight_map(snapshot)
    keys = sorted(k for k in weight_map if k.startswith(VISUAL_PREFIX))
    raw = read_tensors(snapshot, weight_map, keys)
    return {k[len(VISUAL_PREFIX) :]: v for k, v in raw.items()}


@contextlib.contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_hf_visual(snapshot=None, state_dict=None, dtype=torch.bfloat16, attn_implementation="sdpa"):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    snapshot = snapshot or resolve_snapshot()
    config = AutoConfig.from_pretrained(snapshot).vision_config
    config._attn_implementation = attn_implementation
    with default_dtype(dtype):
        visual = Qwen3_5VisionModel(config)
    if state_dict is None:
        state_dict = read_vision_state_dict(snapshot)
    visual.load_state_dict(state_dict, strict=True)
    return visual.eval()


def tt_vision_state_dict(hf_state_dict, head_dim):
    state_dict = standardize_hf_keys_multimodal(dict(hf_state_dict))
    state_dict = convert_hf_to_meta(state_dict, head_dim)
    return {f"{STATE_DICT_PREFIX}.{k}": v for k, v in state_dict.items()}


def expected_tt_keys(depth):
    keys = set()
    for index in range(depth):
        block = f"{STATE_DICT_PREFIX}.blocks.{index}"
        for name in ("wq", "wk", "wv", "wo"):
            keys.update({f"{block}.attention.{name}.weight", f"{block}.attention.{name}.bias"})
        for name in ("linear_fc1", "linear_fc2"):
            keys.update({f"{block}.feed_forward.{name}.weight", f"{block}.feed_forward.{name}.bias"})
        for name in ("norm1", "norm2"):
            keys.update({f"{block}.{name}.weight", f"{block}.{name}.bias"})
    for name in ("norm", "linear_fc1", "linear_fc2"):
        keys.update({f"{STATE_DICT_PREFIX}.merger.{name}.weight", f"{STATE_DICT_PREFIX}.merger.{name}.bias"})
    return keys


def padded_rows(n_patches, granule=None):
    granule = env_pad_granule() if granule is None else granule
    rows = max(128, math.ceil(n_patches / granule) * granule)
    if rows > 1024 and rows % 1024:
        rows = math.ceil(rows / 1024) * 1024
    if rows > 2048 and rows % 2048:
        rows = math.ceil(rows / 2048) * 2048
    return rows


def window_bounds(cu_seqlens, rows, mask_padding):
    bounds = [int(v) for v in cu_seqlens]
    n_patches = bounds[-1]
    if rows > n_patches:
        if mask_padding:
            bounds.append(rows)
        else:
            bounds[-1] = rows
    if len(bounds) <= 2 and not (mask_padding and rows > n_patches):
        return None
    return bounds


def vision_shape_report(vision_config, tp, tile=32):
    head_dim = vision_config.hidden_size // vision_config.num_heads
    padded_head_dim = math.ceil(head_dim / tile) * tile
    hidden_dim = math.ceil(vision_config.intermediate_size / (tile * tp)) * tile * tp
    qkv_size = padded_head_dim * 3 * vision_config.num_heads
    mlp_size = vision_config.hidden_size * vision_config.spatial_merge_size**2
    return dict(
        tp=tp,
        depth=vision_config.depth,
        dim=vision_config.hidden_size,
        n_heads=vision_config.num_heads,
        head_dim=head_dim,
        padded_head_dim=padded_head_dim,
        qkv_size=qkv_size,
        local_qkv_size=qkv_size // tp,
        intermediate_size=vision_config.intermediate_size,
        hidden_dim=hidden_dim,
        merger_mlp_size=mlp_size,
        out_hidden_size=vision_config.out_hidden_size,
        divisible=dict(
            n_heads=vision_config.num_heads % tp == 0,
            qkv_size=qkv_size % tp == 0,
            dim=vision_config.hidden_size % tp == 0,
            hidden_dim=hidden_dim % tp == 0,
            merger_mlp_size=mlp_size % tp == 0,
            out_hidden_size=vision_config.out_hidden_size % tp == 0,
        ),
        deepstack_visual_indexes=list(getattr(vision_config, "deepstack_visual_indexes", []) or []),
    )


def env_activation_bf16():
    return os.environ.get("CLEF_VISION_ACT_BF16", "1") == "1"


PRECISIONS = ("accuracy", "upstream")


def env_precision():
    value = os.environ.get("CLEF_VISION_PRECISION", "accuracy")
    if value not in PRECISIONS:
        raise ValueError(f"CLEF_VISION_PRECISION={value!r}; expected one of {PRECISIONS}")
    return value


def tower_precision(model_args):
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.ACTIVATION: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    conf.__name__ = "clef_vision_act_bf16"
    return DecodersPrecision(model_args.n_layers, model_args.model_name, conf)


class ClefVisionArgs(VisionModelArgs):
    def __init__(
        self,
        mesh_device,
        snapshot=None,
        max_batch_size=1,
        max_seq_len=2048,
        weight_cache=None,
        precision=None,
        activation_bf16=None,
        **kwargs,
    ):
        self.snapshot = snapshot or resolve_snapshot()
        os.environ["HF_MODEL"] = self.snapshot
        self.activation_bf16 = env_activation_bf16() if activation_bf16 is None else activation_bf16
        self.precision = precision or env_precision()
        if self.activation_bf16 and "optimizations" not in kwargs:
            kwargs["optimizations"] = tower_precision
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        if self.precision == "accuracy":
            self.vision_weight_dtype = ttnn.bfloat16
            self.vision_sdpa_dtype = ttnn.bfloat16
            self.vision_mlp_compute_kernel_config = self.compute_kernel_config_hifi4
            self.vision_merger_compute_kernel_config = self.compute_kernel_config_hifi4
        self.revision_sha8 = revision_sha8(self.snapshot)
        if weight_cache is None:
            weight_cache = os.environ.get("CLEF_VISION_WEIGHT_CACHE", "0") == "1"
        self.use_weight_cache = weight_cache

    def weight_cache_path(self, dtype=None):
        if not self.use_weight_cache:
            return None
        name = DTYPE_NAMES.get(dtype or ttnn.bfloat16, "bf16")
        mesh = "x".join(str(d) for d in self.cluster_shape)
        return Path(self.model_cache_path) / f"vision_cache_{name}_mesh{mesh}_clef_{self.revision_sha8}"


class ClefVision:
    def __init__(
        self,
        mesh_device,
        args,
        hf_visual_or_state_dict=None,
        dtype=None,
        tt_ccl=None,
        pad_granule=None,
        mask_padding=None,
    ):
        self.mesh_device = mesh_device
        if isinstance(args, VisionModelArgs):
            self.vision_args = args
        else:
            self.vision_args = ClefVisionArgs(
                mesh_device,
                getattr(args, "snapshot", None),
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
            )
        snapshot = getattr(self.vision_args, "snapshot", None) or resolve_snapshot()
        if isinstance(hf_visual_or_state_dict, torch.nn.Module):
            self.hf_visual = hf_visual_or_state_dict
        else:
            self.hf_visual = load_hf_visual(snapshot, state_dict=hf_visual_or_state_dict)
        if isinstance(dtype, str):
            dtype = DTYPES[dtype]
        self.dtype = dtype or env_dtype()
        self.pad_granule = env_pad_granule() if pad_granule is None else pad_granule
        self.mask_padding = env_mask_padding() if mask_padding is None else mask_padding
        started = time.perf_counter()
        self.tower = DropInVisionTransformer(self.hf_visual, self.vision_args, dtype=self.dtype, tt_ccl=tt_ccl)
        self.build_seconds = time.perf_counter() - started
        self.tt_model = self.tower.tt_model
        self.tp = self.vision_args.cluster_shape[1]
        self.dim = self.vision_args.dim
        self.depth = len(self.tt_model.blocks)
        self.out_hidden_size = self.vision_args.hf_config.vision_config.out_hidden_size
        self.spatial_merge_size = self.vision_args.hf_config.vision_config.spatial_merge_size
        self.last_run = {}

    def attach(self, model):
        model.vision_model = self
        model.vision_args = self.vision_args
        return model

    def _replicated(self, tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=layout,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _host_patches(self, pixel_values, grid):
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            patches = self.hf_visual.patch_embed(pixel_values)
            positions = self.hf_visual.fast_pos_embed_interpolate(grid)
        return patches + positions

    def _run_one(self, pixel_values, grid, taps):
        n_patches = int(grid.prod())
        rows = padded_rows(n_patches, self.pad_granule)
        cu_seqlens, (cos_hf, sin_hf) = qwen3_5_vision_transformer_preprocess(
            seq_len=n_patches,
            grid_thw=grid,
            head_dim=self.vision_args.head_dim,
            spatial_merge_size=self.spatial_merge_size,
        )
        patch_input = self._host_patches(pixel_values, grid)
        cos, sin = convert_rope_style_hf_to_meta(cos_hf, sin_hf)
        cos = torch.nn.functional.pad(cos, (0, 0, 0, rows - n_patches), value=1).unsqueeze(0).unsqueeze(0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, rows - n_patches), value=0).unsqueeze(0).unsqueeze(0)
        rot_mats = [self._replicated(cos), self._replicated(sin)]
        bounds = window_bounds(cu_seqlens, rows, self.mask_padding)
        window = None
        if bounds is not None:
            window = self._replicated(
                torch.tensor(bounds, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        x = self.tt_model.prepare_input(patch_input, rows)
        tapped = {}
        for index, block in enumerate(self.tt_model.blocks):
            x = block(x, rot_mats=rot_mats, cu_window_seqlens=window)
            if index in taps:
                tapped[index] = self.hidden_to_torch(x)[:n_patches]
        x = x[:, :, :n_patches, :]
        out = self.tt_model.patch_merger(x)
        for tensor in rot_mats:
            ttnn.deallocate(tensor)
        if window is not None:
            ttnn.deallocate(window)
        self.last_run = dict(n_patches=n_patches, rows=rows, masked=window is not None, window=bounds)
        return out, tapped

    def forward_with_taps(self, pixel_values, grid_thw, taps=()):
        grid_thw = torch.as_tensor(grid_thw).reshape(-1, 3)
        outputs = []
        tapped = []
        offset = 0
        started = time.perf_counter()
        for grid in grid_thw:
            count = int(grid.prod())
            out, taps_i = self._run_one(pixel_values[offset : offset + count], grid.reshape(1, 3), taps)
            offset += count
            outputs.append(out)
            tapped.append(taps_i)
        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = ttnn.concat(outputs, dim=2)
            for tensor in outputs:
                ttnn.deallocate(tensor)
        ttnn.synchronize_device(self.mesh_device)
        self.last_run["seconds"] = time.perf_counter() - started
        self.last_run["images"] = len(outputs)
        return result, tapped

    def forward(self, pixel_values, grid_thw):
        result, _ = self.forward_with_taps(pixel_values, grid_thw)
        return result

    def image_features(self, pixel_values, image_grid_thw):
        out = self.forward(pixel_values, image_grid_thw)
        return ttnn.reshape(out, (-1, out.shape[-1]))

    def video_features(self, pixel_values_videos, video_grid_thw):
        return self.image_features(pixel_values_videos, video_grid_thw)

    def hidden_to_torch(self, tensor):
        full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return full[0, 0, :, : self.dim].float()

    def rows_to_torch(self, tensor):
        last = len(tensor.shape) - 1
        full = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=last))
        return full.reshape(-1, full.shape[-1])[:, : self.out_hidden_size].float()

    def image_features_torch(self, pixel_values, image_grid_thw):
        out = self.forward(pixel_values, image_grid_thw)
        merged = self.rows_to_torch(out)
        ttnn.deallocate(out)
        return merged

    def video_features_torch(self, pixel_values_videos, video_grid_thw):
        return self.image_features_torch(pixel_values_videos, video_grid_thw)

    def block_outputs_torch(self, pixel_values, image_grid_thw, taps=BLOCK_TAPS):
        out, tapped = self.forward_with_taps(pixel_values, image_grid_thw, taps=tuple(taps))
        merged = self.rows_to_torch(out)
        ttnn.deallocate(out)
        return merged, tapped[0] if len(tapped) == 1 else tapped

    def upstream_features_torch(self, pixel_values, image_grid_thw):
        started = time.perf_counter()
        out = self.tower.forward(pixel_values, torch.as_tensor(image_grid_thw).reshape(-1, 3))
        ttnn.synchronize_device(self.mesh_device)
        self.last_run = dict(upstream=True, seconds=time.perf_counter() - started)
        merged = self.rows_to_torch(out)
        ttnn.deallocate(out)
        return merged

    def bfp8_weights(self):
        block = self.tt_model.blocks[0]
        named = dict(
            attention_wqkv=block.attention.wqkv,
            attention_wqkv_bias=block.attention.wqkv_bias_prefill,
            attention_wo=block.attention.wo,
            feed_forward_linear_fc1=block.feed_forward.linear_fc1_weight,
            feed_forward_linear_fc2=block.feed_forward.linear_fc2_weight,
            merger_fc1=self.tt_model.patch_merger.w1,
            merger_fc2=self.tt_model.patch_merger.w2,
        )
        return sorted(name for name, tensor in named.items() if tensor is not None and tensor.dtype == ttnn.bfloat8_b)

    def describe(self):
        return dict(
            dtype=DTYPE_NAMES.get(self.dtype, str(self.dtype)),
            bfp8_weights=self.bfp8_weights(),
            pad_granule=self.pad_granule,
            mask_padding=self.mask_padding,
            activation_bf16=getattr(self.vision_args, "activation_bf16", False),
            precision=getattr(self.vision_args, "precision", "upstream"),
            wqkv_dtype=str(self.tt_model.blocks[0].attention.wqkv.dtype),
            sdpa_dtype=str(self.tt_model.blocks[0].attention.sdpa_dtype),
            mlp_fc1_dtype=str(self.tt_model.blocks[0].feed_forward.linear_fc1_weight.dtype),
            mlp_fidelity=str(self.tt_model.blocks[0].feed_forward._compute_kernel_config().math_fidelity),
            mlp_fp32_acc=bool(self.tt_model.blocks[0].feed_forward._compute_kernel_config().fp32_dest_acc_en),
            merger_fidelity=str(self.tt_model.patch_merger.compute_kernel_config.math_fidelity),
            merger_fp32_acc=bool(self.tt_model.patch_merger.compute_kernel_config.fp32_dest_acc_en),
            attention_activation_dtype=str(self.tt_model.blocks[0].attention.activation_dtype),
            attention_k_dtype=str(self.tt_model.blocks[0].attention.kv_cache_dtype),
            sdpa_fidelity=str(self.tt_model.blocks[0].attention.sdpa_prefill_compute_kernel_cfg.math_fidelity),
            qkv_fidelity=str(self.tt_model.blocks[0].attention.li_qkv_prefill_compute_kernel_cfg.math_fidelity),
            tp=self.tp,
            depth=self.depth,
            weight_cache=str(self.vision_args.weight_cache_path(self.dtype)),
            build_seconds=round(self.build_seconds, 1),
            shapes=vision_shape_report(self.vision_args.hf_config.vision_config, self.tp),
        )
