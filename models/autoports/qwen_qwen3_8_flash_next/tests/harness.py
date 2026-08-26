# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reference/evidence plumbing for the Qwen3.8-Flash-Next functional layer."""

from __future__ import annotations

import json
import math
import sys
from contextlib import ContextDecorator
from pathlib import Path

import torch

PCC_BAR = 0.995
MODEL_SNAPSHOT = Path(
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/"
    "snapshots/f5d08274bafd880402bd16f5e3e6c514136ec06c"
)
TRANSFORMERS_SITE = Path("/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages")
LAYER_PREFIX = "model.language_model.layers.{layer_idx}."


def import_target_transformers():
    """Load the installed Transformers 5.16 Qwen4Exp implementation.

    The checkout's TTNN environment intentionally owns the interpreter/native
    extension; only pure-Python model/reference packages come from the model
    environment.  Repo-local ``ttnn`` must already be imported by device tests.
    """

    loaded = sys.modules.get("transformers")
    if loaded is not None and getattr(loaded, "__version__", None) != "5.16.0":
        for name in tuple(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
    site = str(TRANSFORMERS_SITE)
    if site not in sys.path:
        sys.path.insert(0, site)
    import transformers

    if transformers.__version__ != "5.16.0":
        raise RuntimeError(f"Qwen4Exp reference requires Transformers 5.16.0, got {transformers.__version__}")
    # The HF layer itself has no accelerate dependency.  Explicitly disable the
    # optional integration because the reference-only site-packages tree owns a
    # different NumPy ABI than the repo's TTNN interpreter.  Importing it would
    # mix the two environments and fail before the layer reference is built.
    import transformers.utils
    import transformers.utils.import_utils

    unavailable = lambda *args, **kwargs: False
    transformers.utils.is_accelerate_available = unavailable
    transformers.utils.import_utils.is_accelerate_available = unavailable
    return transformers


def build_hf_layer(config, layer_idx: int, state: dict[str, torch.Tensor], ple_embeddings=None):
    """Materialize one real HF layer without ever allocating the 360B model."""

    import_target_transformers()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(config.text_config, layer_idx)
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"unexpected HF layer keys: {unexpected}")
    allowed_missing = tuple(name for name in missing if name.startswith("ple.ple_embedding."))
    if tuple(missing) != allowed_missing:
        raise RuntimeError(f"unexpected missing HF layer keys: {missing}")
    if ple_embeddings is not None:

        class PreparedEmbedding(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, input_ids, past_key_values):
                return self.value[:, -input_ids.shape[1] :]

        layer.ple.ple_embedding = PreparedEmbedding(ple_embeddings)
    layer.eval()
    return layer


@torch.no_grad()
def hf_forward(layer, hidden_states, cos, sin, *, ple_embeddings=None):
    """HF full-sequence reference; the last row is the cache-free decode oracle."""

    length = hidden_states.shape[1]
    visible = torch.ones(1, 1, length, length, dtype=torch.bool).tril()
    causal = torch.zeros(1, 1, length, length, dtype=hidden_states.dtype)
    causal.masked_fill_(~visible, torch.finfo(hidden_states.dtype).min)
    kwargs = {}
    if ple_embeddings is not None:
        kwargs["ple_input_ids"] = torch.zeros(1, length, dtype=torch.long)
    return layer(
        hidden_states,
        position_embeddings=(cos[:length].unsqueeze(0), sin[:length].unsqueeze(0)),
        attention_mask=causal,
        conv_mask=None,
        past_key_values=None,
        **kwargs,
    )


def target_config():
    transformers = import_target_transformers()
    return transformers.AutoConfig.from_pretrained(MODEL_SNAPSHOT, local_files_only=True)


def load_real_layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    """Load one decoder layer, excluding the physically impossible full PLE table."""

    from safetensors import safe_open

    index = json.loads((MODEL_SNAPSHOT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = LAYER_PREFIX.format(layer_idx=layer_idx)
    selected = {
        full_name: shard
        for full_name, shard in weight_map.items()
        if full_name.startswith(prefix)
        and ".ple.ple_embedding.ngram_embedding." not in full_name
        and not full_name.endswith("ngram_heads_offsets")
        and not full_name.endswith("ngram_heads_vocab_sizes")
        and not full_name.endswith("layer_multipliers")
    }
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)
    result = {}
    for shard, names in by_shard.items():
        with safe_open(MODEL_SNAPSHOT / shard, framework="pt", device="cpu") as handle:
            for full_name in names:
                result[full_name.removeprefix(prefix)] = handle.get_tensor(full_name)
    return result


def make_partial_state(config, layer_idx: int, seed: int = 17) -> dict[str, torch.Tensor]:
    """Nonzero deterministic weights at every target shape except routed experts.

    Routed expert tensors alone occupy roughly 5 GiB BF16.  Structural tests use
    exact-shaped device zeros for them; the real-checkpoint test covers their
    nonzero sparse path.  All other operations receive nontrivial weights here.
    """

    cfg = config.text_config
    generator = torch.Generator().manual_seed(seed + layer_idx)

    def rand(shape, scale=0.01):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * scale

    h, streams, hc, low = cfg.hidden_size, cfg.hc_count, cfg.hidden_size * cfg.hc_count, cfg.hc_lowrank
    state = {}
    for prefix in ("attn_hyper_connection", "mlp_hyper_connection"):
        state[f"{prefix}.hc_norm.weight"] = rand((hc,), 0.02)
        state[f"{prefix}.input_mix_weight_down.weight"] = rand((low, hc), 0.005)
        state[f"{prefix}.input_mix_weight_up.weight"] = rand((hc, low), 0.005)
        state[f"{prefix}.block_inject_weight.weight"] = rand((streams, hc), 0.005)

    state["mlp.gate.weight"] = rand((cfg.num_experts, h), 0.01)
    for name in ("gate_proj", "up_proj"):
        state[f"mlp.shared_expert.{name}.weight"] = rand((cfg.shared_expert_intermediate_size, h), 0.01)
    state["mlp.shared_expert.down_proj.weight"] = rand((h, cfg.shared_expert_intermediate_size), 0.01)
    state["mlp.shared_expert_gate.weight"] = rand((1, h), 0.01)

    if cfg.layer_types[layer_idx] == "linear_attention":
        qk = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        width = 2 * qk + value
        state["linear_attn.in_proj_qkv.weight"] = rand((width, h), 0.006)
        state["linear_attn.in_proj_z.weight"] = rand((value, h), 0.006)
        state["linear_attn.in_proj_b.weight"] = rand((cfg.linear_num_value_heads, h), 0.006)
        state["linear_attn.in_proj_a.weight"] = rand((cfg.linear_num_value_heads, h), 0.006)
        state["linear_attn.conv1d.weight"] = rand((width, 1, cfg.linear_conv_kernel_dim), 0.02)
        state["linear_attn.dt_bias"] = torch.linspace(-1.0, 1.0, cfg.linear_num_value_heads, dtype=torch.bfloat16)
        state["linear_attn.A_log"] = torch.linspace(-2.0, 1.0, cfg.linear_num_value_heads, dtype=torch.float32)
        state["linear_attn.norm.weight"] = torch.ones(cfg.linear_value_head_dim, dtype=torch.bfloat16)
        state["linear_attn.out_proj.weight"] = rand((h, value), 0.006)
    else:
        q = cfg.num_attention_heads * cfg.head_dim
        kv = cfg.num_key_value_heads * cfg.head_dim
        state["self_attn.q_proj.weight"] = rand((2 * q, h), 0.006)
        state["self_attn.k_proj.weight"] = rand((kv, h), 0.006)
        state["self_attn.v_proj.weight"] = rand((kv, h), 0.006)
        state["self_attn.o_proj.weight"] = rand((h, q), 0.006)
        state["self_attn.q_norm.weight"] = rand((cfg.head_dim,), 0.02)
        state["self_attn.k_norm.weight"] = rand((cfg.head_dim,), 0.02)
        state["self_attn.indexer.index_qk_proj.weight"] = rand(
            ((cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim, h), 0.006
        )
        state["self_attn.indexer.q_layernorm.weight"] = rand((cfg.indexer_head_dim,), 0.02)
        state["self_attn.indexer.k_layernorm.weight"] = rand((cfg.indexer_head_dim,), 0.02)

    if layer_idx + 1 in cfg.ple_layer_ids:
        state["ple.key_proj.weight"] = rand((hc, cfg.ple_embed_dim), 0.005)
        state["ple.value_proj.weight"] = rand((h, cfg.ple_embed_dim), 0.005)
        state["ple.norm_key.weight"] = rand((hc,), 0.02)
        state["ple.norm_query.weight"] = rand((hc,), 0.02)
        state["ple.norm_conv.weight"] = rand((hc,), 0.02)
        state["ple.conv1d.weight"] = rand((hc, 1, cfg.ple_conv_kernel_size), 0.02)
    return state


def pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    x = expected.float().reshape(-1)
    y = actual.float().reshape(-1)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(torch.dot(x, y) / denominator) if denominator else float(torch.equal(expected, actual))


def rope_tables(max_seq_len: int, rotary_dim: int = 64, theta: float = 10_000_000.0):
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    frequencies = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
    embedding = torch.cat([frequencies, frequencies], dim=-1)
    return embedding.cos().bfloat16(), embedding.sin().bfloat16()


def shuffled_page_table(max_seq_len: int, block_size: int = 64, seed: int = 1234):
    blocks = math.ceil(max_seq_len / block_size)
    return torch.randperm(blocks, generator=torch.Generator().manual_seed(seed), dtype=torch.int32).reshape(1, blocks)


class ForbidHostFallback(ContextDecorator):
    """Make a measured pass fail immediately on host conversion."""

    NAMES = ("from_torch", "as_tensor", "to_torch")

    def __enter__(self):
        import ttnn

        self.saved = {name: getattr(ttnn, name) for name in self.NAMES}

        def forbidden(*args, **kwargs):
            raise AssertionError("runtime host fallback attempted")

        for name in self.NAMES:
            setattr(ttnn, name, forbidden)
        return self

    def __exit__(self, *exc):
        import ttnn

        for name, value in self.saved.items():
            setattr(ttnn, name, value)
        return False
