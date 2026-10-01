import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from models.autoports.jaredpalmer_kev_9b.tt import precision_defaults
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

DEFAULT_RUN = "jaredpalmer/kev-9b"
ADAPTER_PREFIX = "base_model.model."
HF_PREFIX = "model."


def resolve_run(run):
    if os.path.isdir(run):
        return run
    from huggingface_hub import snapshot_download

    repo, _, revision = run.partition("@")
    offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
    return snapshot_download(repo, revision=revision or None, local_files_only=offline)


def adapter_sha256(adapter_dir):
    h = hashlib.sha256()
    with open(Path(adapter_dir) / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_lora_into_state_dict(hf_state_dict, adapter_dir):
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(str(adapter_dir / "adapter_model.safetensors"))
    consumed = 0
    for key in [k for k in lora if k.endswith(".lora_A.weight")]:
        module = key[len(ADAPTER_PREFIX) : -len(".lora_A.weight")]
        target = f"{HF_PREFIX}{module}.weight"
        A, B = lora[key].float(), lora[f"{ADAPTER_PREFIX}{module}.lora_B.weight"].float()
        W = hf_state_dict[target]
        assert W.shape == (B.shape[0], A.shape[1]), (target, W.shape, A.shape, B.shape)
        merged = (W.float() + scale * (B @ A)).to(W.dtype)
        assert not torch.equal(merged, W), f"{target} unchanged by the LoRA merge"
        hf_state_dict[target] = merged
        consumed += 2
    assert consumed == len(lora), f"consumed {consumed} of {len(lora)} adapter tensors"
    return hf_state_dict


class KevModelArgs(Qwen36ModelArgs):
    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048, **kwargs):
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        self.adapter_dir = resolve_run(os.environ.get("KEV_RUN", DEFAULT_RUN))
        self.adapter_sha8 = adapter_sha256(self.adapter_dir)[:8]

    def weight_cache_path(self, dtype=None):
        base = super().weight_cache_path(dtype)
        return base.with_name(f"{base.name}_kev_{self.adapter_sha8}_{precision_defaults.cache_tag()}")

    def load_state_dict(self):
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        text_config = Qwen3_5TextConfig.from_pretrained(self.CKPT_DIR)
        model = Qwen3_5ForCausalLM.from_pretrained(self.CKPT_DIR, config=text_config, dtype="auto")
        hf = merge_lora_into_state_dict(dict(model.state_dict()), self.adapter_dir)
        del model
        state_dict = remap_qwen36_state_dict(hf)
        if "output.weight" not in state_dict and not text_config.tie_word_embeddings:
            raise KeyError("output.weight missing after remap and tie_word_embeddings is false")
        return state_dict
