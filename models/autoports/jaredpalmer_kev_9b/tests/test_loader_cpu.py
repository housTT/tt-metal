import os

import pytest
import torch

from models.autoports.jaredpalmer_kev_9b.tt.loader import KevModelArgs, adapter_sha256

BASE = (
    "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404"
)
ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
CHECKS = {
    "layers.0.linear_attn.qkv_proj.weight": "layers.0.linear_attn.in_proj_qkv",
    "layers.3.self_attn.q_proj.weight": "layers.3.self_attn.q_proj",
    "layers.31.mlp.down_proj.weight": "layers.31.mlp.down_proj",
}


@pytest.mark.eager_host_side
@pytest.mark.timeout(3600)
def test_merged_weights_match_peft():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["HF_MODEL"] = BASE
    os.environ["KEV_RUN"] = ADAPTER
    args = KevModelArgs(mesh_device=None)
    assert args.adapter_sha8 == adapter_sha256(ADAPTER)[:8] == "2b2a70cf"
    sd = args.load_state_dict()
    assert "output.weight" in sd and "tok_embeddings.weight" in sd
    assert sd["layers.0.linear_attn.qkv_proj.weight"].dtype == torch.bfloat16

    from peft import PeftModel
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    text_config = Qwen3_5TextConfig.from_pretrained(BASE)
    base = Qwen3_5ForCausalLM.from_pretrained(BASE, config=text_config, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base.model, ADAPTER).merge_and_unload()
    ref = {k: v for k, v in merged.state_dict().items() if k.split(".weight")[0] in CHECKS.values()}
    for ours_key, ref_module in CHECKS.items():
        ours, theirs = sd[ours_key].float(), ref[f"{ref_module}.weight"]
        diff = (ours - theirs).abs().max().item()
        rel = diff / theirs.abs().max().item()
        mismatched = (sd[ours_key] != theirs.to(torch.bfloat16)).sum().item()
        print(
            f"{ours_key}: max|diff|={diff:.3e} rel={rel:.3e} bf16 elements differing from round(peft fp32)={mismatched}/{theirs.numel()}"
        )
        assert rel < 1e-2, (ours_key, diff, rel)
        assert mismatched < theirs.numel() * 1e-3, (ours_key, mismatched)
