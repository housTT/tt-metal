import os

import pytest
import torch
from safetensors import safe_open

from models.autoports.cloudflare_clef.tt.loader import (
    LM_HEAD_KEY,
    PRECISION_TAG_KEYS,
    ClefModelArgs,
    precision_cache_tag,
    read_weight_map,
)

SNAPSHOT = (
    "/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c"
)
EXPECTED_TEXT = dict(
    dim=5120,
    n_layers=64,
    n_heads=24,
    n_kv_heads=4,
    head_dim=256,
    hidden_dim=17408,
    vocab_size=248320,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    partial_rotary_factor=0.25,
    rope_head_dim=64,
    mrope_section=[11, 11, 10],
    rope_theta=10000000,
)


@pytest.fixture(scope="module")
def args():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CLEF_MODEL", SNAPSHOT)
    os.environ.setdefault("TT_CACHE_PATH", "/home/hous/dev/clef/tt_cache")
    return ClefModelArgs(mesh_device=None)


@pytest.mark.eager_host_side
def test_text_config_resolves(args):
    assert args.snapshot == os.environ["CLEF_MODEL"]
    assert args.revision_sha8 == "2f3de3dd"
    for name, value in EXPECTED_TEXT.items():
        assert getattr(args, name) == value, (name, getattr(args, name), value)
    assert len(args.attention_type_list) == 64
    assert args.attention_type_list.count("linear_attention") == 48
    assert args.attention_type_list.count("full_attention") == 16
    assert all(args.is_full_attention_layer(i) == ((i + 1) % 4 == 0) for i in range(64))
    assert all(args.is_deltanet_layer(i) != args.is_full_attention_layer(i) for i in range(64))
    assert not args.is_moe_layer(0)


@pytest.mark.eager_host_side
def test_cache_name_is_distinct(args):
    import ttnn
    from models.demos.blackhole.qwen36.tt import precision

    for short, env, default in PRECISION_TAG_KEYS:
        parent_value = {
            "QWEN36_MLP_GATE_UP_DTYPE": precision.MLP_GATE_UP_DTYPE,
            "QWEN36_MLP_DOWN_DTYPE": precision.MLP_DOWN_DTYPE,
            "QWEN36_PROJ_DTYPE": precision.PROJ_DTYPE,
        }[env]
        assert precision._DTYPES[os.environ.get(env, default)] == parent_value, env
    path = args.weight_cache_path(ttnn.bfloat16)
    assert path.name == f"tensor_cache_bf16_clef_2f3de3dd_{precision_cache_tag()}"
    assert args.weight_cache_path(ttnn.bfloat8_b).name.startswith("tensor_cache_bfp8_clef_2f3de3dd_")
    assert str(path).startswith(os.environ["TT_CACHE_PATH"])


@pytest.mark.eager_host_side
def test_mapped_keys_match_parent_expectation(args):
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

    config = Qwen3_5TextConfig.from_pretrained(args.snapshot)
    assert config.hidden_size == args.dim and config.vocab_size == args.vocab_size
    config.num_hidden_layers = 4
    config.layer_types = config.layer_types[:4]
    with torch.device("meta"):
        hf_keys = list(Qwen3_5ForCausalLM(config).state_dict().keys())
    expected = set(remap_qwen36_state_dict({k: torch.empty(0, 0, 0) for k in hf_keys}))
    expected.discard("output.weight")
    for layer in (0, 3):
        ours = args.mapped_keys({layer})
        want = {k for k in expected if k.startswith(f"layers.{layer}.") or not k.startswith("layers.")}
        assert ours == want, (layer, ours ^ want)
    assert any(k.startswith("layers.0.linear_attn.") for k in expected)
    assert any(k.startswith("layers.3.self_attn.") for k in expected)


@pytest.mark.eager_host_side
@pytest.mark.timeout(600)
def test_load_state_dict_four_layers(args):
    args.n_layers = 4
    args.attention_type_list = args.attention_type_list[:4]
    sd = args.load_state_dict()
    assert "output.weight" not in sd
    assert set(k.split(".")[1] for k in sd if k.startswith("layers.")) == {"0", "1", "2", "3"}
    assert sd["tok_embeddings.weight"].shape == (248320, 5120) and sd["tok_embeddings.weight"].dtype == torch.bfloat16
    assert sd["layers.0.linear_attn.qkv_proj.weight"].shape == (2 * 2048 + 6144, 5120)
    assert sd["layers.0.linear_attn.q_conv.weight"].shape == (2048, 1, 4)
    assert sd["layers.0.linear_attn.k_conv.weight"].shape == (2048, 1, 4)
    assert sd["layers.0.linear_attn.v_conv.weight"].shape == (6144, 1, 4)
    assert sd["layers.3.self_attn.q_proj.weight"].shape == (24 * 256 * 2, 5120)
    assert sd["layers.3.self_attn.k_proj.weight"].shape == (4 * 256, 5120)
    assert sd["layers.3.self_attn.o_proj.weight"].shape == (5120, 24 * 256)
    assert sd["layers.0.mlp.gate_proj.weight"].shape == (17408, 5120)


@pytest.mark.eager_host_side
def test_lm_head_rows(args):
    ids = [248044, 1, 2]
    rows = args.load_lm_head_rows(ids)
    assert rows.shape == (3, 5120) and rows.dtype == torch.bfloat16
    weight_map = read_weight_map(args.snapshot)
    with safe_open(os.path.join(args.snapshot, weight_map[LM_HEAD_KEY]), framework="pt") as shard:
        full = shard.get_tensor(LM_HEAD_KEY)
    assert full.shape == (248320, 5120)
    assert torch.equal(rows, full[ids])
    assert torch.equal(args.load_lm_head_rows(torch.tensor([2, 2, 248044])), full[[2, 2, 248044]])


@pytest.mark.eager_host_side
def test_vision_and_head_state_dicts(args):
    from models.autoports.cloudflare_clef.tt.head import JointSchemaHead

    weight_map = read_weight_map(args.snapshot)
    visual_keys = [k for k in weight_map if k.startswith("model.visual.")]
    assert len(visual_keys) == 333
    vision = args.vision_state_dict()
    assert len(vision) == 333
    assert set(vision) == {k[len("model.visual.") :] for k in visual_keys}
    assert vision["blocks.0.attn.qkv.weight"].shape == (3 * 1152, 1152)
    assert vision["merger.linear_fc2.weight"].shape[0] == 5120
    head = args.head_state_dict()
    expected = JointSchemaHead(**args.head_config()).state_dict()
    assert set(head) == set(expected)
    assert all(head[k].shape == expected[k].shape for k in expected)
    assert all(v.dtype == torch.bfloat16 for v in head.values())
