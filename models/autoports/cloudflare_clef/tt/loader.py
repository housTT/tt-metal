import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

LANGUAGE_PREFIX = "model.language_model."
VISUAL_PREFIX = "model.visual."
LM_HEAD_KEY = "lm_head.weight"
INDEX_FILE = "model.safetensors.index.json"
HEAD_WEIGHTS = "joint_head.safetensors"
HEAD_CONFIG = "joint_head_config.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
PRECISION_TAG_KEYS = (
    ("gu", "QWEN36_MLP_GATE_UP_DTYPE", "bfp4"),
    ("dn", "QWEN36_MLP_DOWN_DTYPE", "bfp8"),
    ("pj", "QWEN36_PROJ_DTYPE", "bfp8"),
)


def resolve_snapshot():
    for env in ("CLEF_MODEL", "HF_MODEL"):
        value = os.environ.get(env)
        if value and os.path.isfile(os.path.join(value, "config.json")):
            return str(Path(value).resolve())
    raise FileNotFoundError("set CLEF_MODEL (or HF_MODEL) to the local Cloudflare/clef snapshot directory")


def revision_sha8(snapshot):
    name = Path(snapshot).name
    if _HEX40.match(name):
        return name[:8]
    import hashlib

    return hashlib.sha1((Path(snapshot) / INDEX_FILE).read_bytes()).hexdigest()[:8]


def precision_cache_tag():
    return "_".join(f"{short}-{os.environ.get(env, default)}" for short, env, default in PRECISION_TAG_KEYS)


def read_weight_map(snapshot):
    return json.loads((Path(snapshot) / INDEX_FILE).read_text())["weight_map"]


def language_keys(weight_map, layer_filter=None):
    keys = []
    for key in weight_map:
        if not key.startswith(LANGUAGE_PREFIX):
            continue
        match = _LAYER_RE.match(key)
        if match is not None and layer_filter is not None and int(match.group(1)) not in layer_filter:
            continue
        keys.append(key)
    return sorted(keys)


def read_tensors(snapshot, weight_map, keys):
    by_file = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    out = {}
    for filename in sorted(by_file):
        with safe_open(str(Path(snapshot) / filename), framework="pt") as shard:
            for key in by_file[filename]:
                out[key] = shard.get_tensor(key)
    return out


class LmHeadRows:
    def __init__(self, snapshot):
        self.snapshot = str(snapshot)
        weight_map = read_weight_map(snapshot)
        self.path = str(Path(snapshot) / weight_map[LM_HEAD_KEY])
        with safe_open(self.path, framework="pt") as shard:
            self.shape = tuple(shard.get_slice(LM_HEAD_KEY).get_shape())
        self._cache = {}

    def __call__(self, token_ids):
        ids = [int(t) for t in (token_ids.tolist() if torch.is_tensor(token_ids) else token_ids)]
        missing = sorted({t for t in ids if t not in self._cache})
        if missing:
            with safe_open(self.path, framework="pt") as shard:
                view = shard.get_slice(LM_HEAD_KEY)
                for t in missing:
                    self._cache[t] = view[t : t + 1][0]
        if not ids:
            return torch.empty((0, self.shape[1]), dtype=torch.bfloat16)
        return torch.stack([self._cache[t] for t in ids])

    def __getitem__(self, token_ids):
        return self(token_ids)


class ClefModelArgs(Qwen36ModelArgs):
    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048, **kwargs):
        self.snapshot = resolve_snapshot()
        os.environ["HF_MODEL"] = self.snapshot
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        self.revision_sha8 = revision_sha8(self.snapshot)
        if mesh_device is not None and self.num_devices > 1:
            self.gdn_gate_fp32 = os.environ.get("QWEN36_GDN_GATE_FP32", "1") == "1"
        self.weight_map = read_weight_map(self.snapshot)
        self.layer_indices = None
        self._lm_head_rows = None

    def weight_cache_path(self, dtype=None):
        base = super().weight_cache_path(dtype)
        return base.with_name(f"{base.name}_clef_{self.revision_sha8}_{precision_cache_tag()}")

    def active_layers(self):
        if self.layer_indices is not None:
            return set(int(i) for i in self.layer_indices)
        return set(range(self.n_layers))

    def raw_language_state_dict(self, layer_filter=None):
        if layer_filter is None:
            layer_filter = self.active_layers()
        keys = language_keys(self.weight_map, layer_filter)
        return read_tensors(self.snapshot, self.weight_map, keys)

    def load_state_dict(self):
        raw = self.raw_language_state_dict()
        state_dict = remap_qwen36_state_dict(raw)
        assert "output.weight" not in state_dict
        assert "tok_embeddings.weight" in state_dict and "norm.weight" in state_dict
        return state_dict

    def mapped_keys(self, layer_filter):
        keys = language_keys(self.weight_map, layer_filter)
        placeholders = {k: torch.empty(0, 0, 0) for k in keys}
        return set(remap_qwen36_state_dict(placeholders))

    @property
    def lm_head_rows(self):
        if self._lm_head_rows is None:
            self._lm_head_rows = LmHeadRows(self.snapshot)
        return self._lm_head_rows

    def load_lm_head_rows(self, token_ids):
        return self.lm_head_rows(token_ids)

    def vision_state_dict(self):
        keys = sorted(k for k in self.weight_map if k.startswith(VISUAL_PREFIX))
        raw = read_tensors(self.snapshot, self.weight_map, keys)
        return {k[len(VISUAL_PREFIX) :]: v for k, v in raw.items()}

    def head_state_dict(self):
        return load_file(str(Path(self.snapshot) / HEAD_WEIGHTS))

    def head_config(self):
        return json.loads((Path(self.snapshot) / HEAD_CONFIG).read_text())
