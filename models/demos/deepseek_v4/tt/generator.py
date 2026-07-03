# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash inference Generator (real weights, on Blackhole).

A clean, self-contained generation API over the actual pretrained weights — the model-side
object a serving stack (tt-inference-server / the tt-metal vLLM plugin) wraps, and which an
eval harness (e.g. lm-eval) can drive offline to score accuracy today. It produces CORRECT
tokens (validated end-to-end to PCC ≥ 0.99 vs the HF reference; e.g. "The capital of France
is" -> " Paris."). Weights are streamed per layer from the fp8/fp4 checkpoint and the routed
experts are kept resident on device (see reference/real_weights.py, tt/model.py).

Throughput note: this uses the correctness-first per-token forward. The high-throughput path
(resident+sharded+traced decode, paged attention, continuous batching) is measured in
demo/decode_engine.py + PERF.md and is the remaining productionization — see PRODUCTION_STATUS.md.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoTokenizer

from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import model as TTM

HF_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


class DeepSeekV4Generator:
    """Real-weight DeepSeek-V4 generator on a Tenstorrent device.

    Args:
        device: an open ttnn device / mesh.
        num_layers: model depth (43 = full model).
    """

    def __init__(self, device, num_layers: int = 43):
        self.device = device
        self.num_layers = num_layers
        self.snapshot = RW.find_snapshot()
        self.store = RW.RealWeightStore(self.snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        cfg = AutoConfig.from_pretrained(self.snapshot)
        cfg.num_nextn_predict_layers = 0
        self.cfg = cfg
        self.layer_types = list(cfg.layer_types[:num_layers])
        self.mlp_types = list(cfg.mlp_layer_types[:num_layers])
        # scratch module (5 layers cover all attn/MoE structural combos; no-init fast build)
        scfg = AutoConfig.from_pretrained(self.snapshot)
        scfg.num_hidden_layers = 5
        scfg.num_nextn_predict_layers = 0
        scfg.layer_types = scfg.layer_types[:5]
        scfg.mlp_layer_types = scfg.mlp_layer_types[:5]
        self.scratch = RW.build_scratch(scfg)
        self.eos_id = getattr(cfg, "eos_token_id", None)

    # ---- low-level forward: logits for a full token sequence (prefill / no-KV-cache decode) ----
    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return TTM.tt_forward_streaming(
            self.scratch, self.store, self.layer_types, self.mlp_types, input_ids, self.device, self.num_layers
        )

    @staticmethod
    def _sample(logits_row: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 0.0:
            return int(logits_row.argmax(-1))
        probs = torch.softmax(logits_row.float() / temperature, dim=-1)
        if 0.0 < top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sp, dim=-1)
            keep = cdf <= top_p
            keep[0] = True
            sp = sp * keep
            sp = sp / sp.sum()
            return int(si[torch.multinomial(sp, 1)])
        return int(torch.multinomial(probs, 1))

    def generate_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_at_eos: bool = True,
    ):
        """Autoregressive generation from token ids -> list of generated ids."""
        ids = input_ids.clone()
        out = []
        for _ in range(max_new_tokens):
            logits = self._logits(ids)  # [1, S, vocab]
            nxt = self._sample(logits[0, -1, :], temperature, top_p)
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            if stop_at_eos and self.eos_id is not None and nxt == self.eos_id:
                break
        return out

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_at_eos: bool = True,
    ) -> dict:
        """Text-in -> text-out. Returns {prompt, completion, prompt_tokens, completion_tokens, token_ids}."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        new_ids = self.generate_ids(input_ids, max_new_tokens, temperature, top_p, stop_at_eos)
        return {
            "prompt": prompt,
            "completion": self.tokenizer.decode(new_ids),
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": len(new_ids),
            "token_ids": new_ids,
        }

    # ---- serving hooks (shape a vLLM Generator adapter maps onto) ----
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill a prompt -> next-token logits [1, vocab] (TTFT path)."""
        return self._logits(input_ids)[:, -1, :]

    def decode(self, context_ids: torch.Tensor) -> torch.Tensor:
        """One decode step given the full context -> next-token logits [1, vocab].
        (Correctness path re-runs the sequence; the paged-KV decode is the throughput upgrade.)"""
        return self._logits(context_ids)[:, -1, :]


if __name__ == "__main__":
    import argparse

    import ttnn

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    dev = ttnn.CreateDevice(device_id=0)
    try:
        gen = DeepSeekV4Generator(dev, num_layers=args.layers)
        r = gen.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("PROMPT:", r["prompt"])
        print("COMPLETION:", repr(r["completion"]))
        print("GENERATOR_OK", r["completion_tokens"], "tokens")
    finally:
        ttnn.CloseDevice(dev)
