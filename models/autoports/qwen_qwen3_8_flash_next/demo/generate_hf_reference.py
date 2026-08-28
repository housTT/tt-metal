# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the exact 100-token HF AIME24 chat-template readiness reference.

The 336 GiB checkpoint cannot be made transiently resident on this 249 GiB
host.  The reference model therefore retains every non-expert HF tensor and
uses exact mmap-backed expert and PLE modules.  Expert projection remains
ordinary Hugging Face Torch math; this host backing exists only for the HF
oracle and is not part of the delivered TT runtime.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import Qwen38PLEHostStore, SafetensorCheckpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.model import MODEL_ID, MODEL_REVISION


class MmapExperts(torch.nn.Module):
    """HF-equivalent MoE math with bounded exact per-layer expert residency."""

    def __init__(self, checkpoint: SafetensorCheckpoint, layer_idx: int, *, capacity: int = 32):
        super().__init__()
        self.checkpoint = checkpoint
        self.layer_idx = int(layer_idx)
        self.capacity = int(capacity)
        prefix = f"model.language_model.layers.{self.layer_idx}.mlp.experts"
        self.gate_up_key = f"{prefix}.gate_up_proj"
        self.down_key = f"{prefix}.down_proj"
        self.cache: collections.OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = collections.OrderedDict()
        self.reads = 0
        self.hits = 0
        self.read_seconds = 0.0

    def _weights(self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_id = int(expert_id)
        cached = self.cache.get(expert_id)
        if cached is not None:
            self.cache.move_to_end(expert_id)
            self.hits += 1
            return cached
        started = time.perf_counter()
        fused = self.checkpoint.indexed_tensor(self.gate_up_key, expert_id)
        down = self.checkpoint.indexed_tensor(self.down_key, expert_id)
        self.read_seconds += time.perf_counter() - started
        self.reads += 1
        self.cache[expert_id] = (fused, down)
        self.cache.move_to_end(expert_id)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return fused, down

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=512).permute(2, 1, 0)
            active = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert_id in active:
            top_k_pos, token_idx = torch.where(expert_mask[expert_id])
            fused, down = self._weights(expert_id)
            gate, up = torch.nn.functional.linear(hidden_states[token_idx], fused).chunk(2, dim=-1)
            selected = torch.nn.functional.silu(gate) * up
            selected = torch.nn.functional.linear(selected, down)
            selected = selected * top_k_weights[token_idx, top_k_pos, None]
            output.index_add_(0, token_idx, selected.to(output.dtype))
        return output


class MmapPLE(torch.nn.Module):
    def __init__(self, store: Qwen38PLEHostStore):
        super().__init__()
        self.store = store

    def forward(self, input_ids, _past_key_values):
        return self.store.prepare(("hf-aime24",), input_ids)


def _load_oracle(snapshot: Path, *, expert_cache_capacity: int):
    H.import_target_transformers()
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

    config = H.target_config().text_config
    checkpoint = SafetensorCheckpoint(snapshot)
    ple_store = Qwen38PLEHostStore(checkpoint)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    # ``inv_freq`` is a non-persistent derived buffer, so it has no checkpoint
    # entry to assign into the meta skeleton.
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=config)

    model.model.embed_tokens.weight = torch.nn.Parameter(
        checkpoint.tensor("model.language_model.embed_tokens.weight"), requires_grad=False
    )
    experts = []
    for layer_idx, layer in enumerate(model.model.layers):
        state = checkpoint.layer_state(layer_idx, include_experts=False)
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"layer {layer_idx} unexpected keys: {unexpected}")
        allowed = {
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
            "ple.ple_embedding.ngram_embedding.weight",
        }
        if set(missing) - allowed:
            raise RuntimeError(f"layer {layer_idx} unexpected missing keys: {missing}")
        expert = MmapExperts(checkpoint, layer_idx, capacity=expert_cache_capacity)
        layer.mlp.experts = expert
        experts.append(expert)
        if layer.ple is not None:
            layer.ple.ple_embedding = MmapPLE(ple_store)
        del state
        gc.collect()

    final_state = {
        name.removeprefix("model.language_model.hyper_connection_mixer."): checkpoint.tensor(name)
        for name in checkpoint.weight_map
        if name.startswith("model.language_model.hyper_connection_mixer.")
    }
    missing, unexpected = model.model.hyper_connection_mixer.load_state_dict(final_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"final mixer mapping failed: missing={missing}, unexpected={unexpected}")
    model.lm_head.weight = torch.nn.Parameter(checkpoint.tensor("lm_head.weight"), requires_grad=False)
    model.eval()
    meta = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
    if meta:
        raise RuntimeError(f"HF oracle still contains meta tensors: {meta}")
    return model, ple_store, experts


def _aime_prompt(tokenizer, prompt_file: Path):
    item = json.loads(prompt_file.read_text())[0]
    messages = [
        {
            "role": "user",
            "content": item["doc"]["problem"]
            + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    tokens = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    return item, messages, rendered, tokens


@torch.inference_mode()
def generate(output: Path, *, expert_cache_capacity: int = 32, threads: int | None = None) -> dict:
    if threads is not None:
        torch.set_num_threads(threads)
    snapshot = H.MODEL_SNAPSHOT
    prompt_file = Path("models/demos/deepseek_v3/demo/aime_under_8k_prompts.json").resolve()
    H.import_target_transformers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    item, messages, rendered, prompt = _aime_prompt(tokenizer, prompt_file)
    load_started = time.perf_counter()
    model, ple_store, experts = _load_oracle(snapshot, expert_cache_capacity=expert_cache_capacity)
    load_seconds = time.perf_counter() - load_started
    ple_store.reset_request("hf-aime24")

    generated = []
    top100_tokens = []
    top100_values = []
    step_seconds = []
    past = None
    current = prompt
    for step in range(100):
        started = time.perf_counter()
        result = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = result.past_key_values
        logits = result.logits[0, -1].float()
        values, indices = torch.topk(logits, 100)
        token = indices[0].reshape(1, 1)
        generated.append(int(token))
        top100_tokens.append(indices.cpu())
        top100_values.append(values.cpu())
        current = token
        step_seconds.append(time.perf_counter() - started)
        print(f"hf-reference step={step + 1}/100 seconds={step_seconds[-1]:.3f} token={int(token)}", flush=True)

    generated_tensor = torch.tensor(generated, dtype=torch.int64)
    source_sha = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    chat_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    artifact = {
        "metadata": {
            "schema_version": 1,
            "hf_model_id": MODEL_ID,
            "checkpoint_revision": MODEL_REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "prompt_source": "DeepSeek AIME24 first prompt",
            "prompt_source_path": str(prompt_file),
            "prompt_source_sha256": source_sha,
            "prompt_index": int(item["index"]),
            "chat_template": True,
            "chat_template_sha256": chat_sha,
            "generation_length": 100,
            "top_k": 100,
            "sampling": "HF greedy",
            "weight_policy": "persistent non-expert HF BF16 plus exact bounded mmap experts and exact mmap PLE",
            "expert_cache_capacity_per_layer": expert_cache_capacity,
            "generation_command": (
                "python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_hf_reference "
                f"--output {output} --expert-cache-capacity {expert_cache_capacity}"
            ),
        },
        "messages": messages,
        "rendered_prompt": rendered,
        "prompt_tokens": prompt.reshape(-1).cpu(),
        "reference_tokens": generated_tensor,
        "top100_tokens": torch.stack(top100_tokens),
        "top100_values": torch.stack(top100_values),
        "reference_text": tokenizer.decode(generated, skip_special_tokens=True),
        "timing": {
            "model_load_seconds": load_seconds,
            "step_seconds": step_seconds,
            "total_generation_seconds": sum(step_seconds),
        },
        "host_store_metrics": {
            "ple": ple_store.metrics(),
            "expert_reads": sum(expert.reads for expert in experts),
            "expert_hits": sum(expert.hits for expert in experts),
            "expert_read_seconds": sum(expert.read_seconds for expert in experts),
        },
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": artifact["metadata"],
                "prompt_tokens": int(prompt.numel()),
                "reference_text": artifact["reference_text"],
                "timing": artifact["timing"],
                "host_store_metrics": artifact["host_store_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ple_store.close()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-cache-capacity", type=int, default=32)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    generate(args.output, expert_cache_capacity=args.expert_cache_capacity, threads=args.threads)


if __name__ == "__main__":
    main()
