# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Readiness/serving generator for hexgrad/Kokoro-82M's plbert encoder.

Implements the shared ``models.common.readiness_check.contract.Generator`` ABC
plus the ``build_generator`` factory, wrapping :class:`KokoroModel` (the full
optimized-multichip plbert encoder + tied-embedding reconstruction readout).

Non-autoregressive reality (carried from stages 01-04, four stage-review passes)
-------------------------------------------------------------------------------
Kokoro's plbert is a **bidirectional, stateless, non-autoregressive** encoder:
no KV cache, no paged cache, no current-position advance, no token-by-token
sampling in the real TTS pipeline. The ``Generator`` ABC is written for causal
LMs, so several of its concepts map to *documented N/A* here rather than to
faked machinery:

* ``kv_cache`` / ``page_table`` — **N/A** (stateless). Accepted for API
  compatibility and ignored; ``kv_cache=None`` is the normal call. There is no
  cache to fill, index, or reset.
* ``start_pos`` / current-position advance — **N/A**. Positions are the fixed
  bidirectional ``0..S-1``; there is no per-token position state to advance on
  device.
* on-device *token-feedback loop* (``tt_out_tok`` -> next persistent decode
  token input, device position advance, unchanged-page-table skip) — **N/A**:
  the encoder input **grows** each step (it is not a fixed ``[batch,1]`` decode
  token), so there is no fixed-shape persistent decode-token tensor to feed.

What is genuinely on device and traced (the split-sampling contract we *can*
honour): encode -> reconstruction readout -> argmax is one captured graph; the
greedy token is produced by on-device ``ttnn.argmax`` with **no host argmax and
no full-vocab logits readback** (only the tiny ``[batch]`` token id is read);
teacher-forcing decode runs through the decoder's traced replay. A ``Sampling1D``
top-k/top-p path is provided and compared against the greedy argmax path; the
comparison and the rejected alternative are recorded in ``doc/full_model``.

API levels
----------
Low level (serving): :meth:`prefill_forward` (full encode -> readout logits) and
:meth:`decode_forward` (stateless re-encode of the provided growing context ->
last-position readout logits or on-device sampled token). Both accept explicit
``page_table`` / ``kv_cache`` / ``prompt_lens`` / batch state.

High level: :meth:`generate` — a thin deterministic loop over the low-level
methods that reproduces the growing-prefix reconstruction used by
``run_teacher_forcing`` (teacher forcing) and ``run_autoregressive``
(free-running). ``enable_trace`` is an explicit keyword; teacher forcing always
runs through the traced decode path. ``host_sampling=True`` is the explicit
compatibility mode for tests that need host-side sampling; the measured
token-out path keeps sampling on device.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import torch

from models.autoports.hexgrad_kokoro_82m.tt.model import KokoroModel
from models.autoports.hexgrad_kokoro_82m.tt.precision_config import load_selected
from models.common.readiness_check.contract import Generator

MODEL_ID = "hexgrad/Kokoro-82M"


class _PhonemeTokenizer:
    """Minimal tokenizer exposing ``.decode(list[int]) -> str`` over Kokoro's
    phoneme vocab, for readiness debug output. Kokoro has no HF chat template
    (it is a TTS model, not a causal LM); AIME24 / chat-template references are
    N/A and replaced by real IPA phoneme prompts (see doc/full_model)."""

    def __init__(self, vocab: dict):
        self.vocab = dict(vocab)
        self.inv = {int(v): k for k, v in vocab.items()}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = []
        for i in ids:
            sym = self.inv.get(int(i), "")
            if skip_special_tokens and int(i) == 0:
                continue
            out.append(sym if sym is not None else "")
        return "".join(out)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        body = [self.vocab[c] for c in text if c in self.vocab]
        return ([0] + body + [0]) if add_special_tokens else body


class KokoroGenerator(Generator):
    """Generator over the full optimized-multichip plbert encoder."""

    def __init__(self, *, mesh_device, model: KokoroModel, vocab: dict):
        self.mesh_device = mesh_device
        self.model = model
        self.vocab = vocab
        self.tokenizer = _PhonemeTokenizer(vocab)
        self.vocab_size = model.vocab_size
        self.max_seq_len = model.max_position_embeddings
        # steady-state host-work counters (proof the loop is not doing avoidable
        # per-step host rebuilds beyond the intrinsic growing-input re-encode)
        self.counters = {
            "trace_captures": 0,
            "trace_replays": 0,
            "host_argmax": 0,
            "logits_readbacks": 0,
        }
        self._sampler = None  # lazily built Sampling1D (top-k/top-p path)

    # ---------------------------------------------------------- low level
    def _prepare(self, tokens: torch.Tensor, attention_mask=None):
        return self.model.prepare_inputs(tokens, attention_mask=attention_mask)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens: Optional[List[int]] = None,
        return_all_logits: bool = False,
        enable_trace: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Full bidirectional encode -> reconstruction logits.

        ``page_table`` / ``kv_cache`` are accepted for API compatibility and
        ignored (stateless encoder). Returns ``[batch, seq, vocab]`` when
        ``return_all_logits`` else ``[batch, 1, vocab]`` at each row's final
        (unpadded) prompt position.
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch, seq = tokens.shape
        _, logits = self.model.forward(tokens, traced=enable_trace, return_logits=True)
        # logits: [batch, seq, vocab]
        if return_all_logits:
            return logits
        lens = prompt_lens or [seq] * batch
        last = torch.stack([logits[b, min(lens[b], seq) - 1, :] for b in range(batch)], dim=0)
        return last.unsqueeze(1)  # [batch, 1, vocab]

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos=None,
        *,
        page_table=None,
        kv_cache=None,
        sample_on_device: bool = True,
        enable_trace: bool = True,
        want_logits: bool = False,
        **kwargs: Any,
    ):
        """Stateless re-encode of the provided context; last-position readout.

        For this bidirectional encoder there is no incremental single-token
        decode: ``tokens`` is the current *full* context ``[batch, cur_len]``
        (the growing prefix). Returns on-device-argmax token ids ``[batch]``
        (``sample_on_device``, greedy, no host argmax / no logits readback) or
        readout logits ``[batch, vocab]``. ``start_pos`` / ``page_table`` /
        ``kv_cache`` accepted for API compatibility and ignored (stateless).
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch, cur = tokens.shape
        prepared = self._prepare(tokens)
        padded, seq = prepared["padded_seq_len"], prepared["seq_len"]
        key = (batch, padded, prepared["attention_mask"] is not None, want_logits or not sample_on_device)
        if key not in self.model._out_traces and enable_trace:
            self.counters["trace_captures"] += 1
        if enable_trace:
            tok_s, log_s = self.model.decode_out_traced(prepared, want_logits=want_logits or not sample_on_device)
            self.counters["trace_replays"] += 1
        else:
            tok_s, log_s = self.model._encode_readout_argmax(
                prepared["input_ids"],
                prepared["position_ids"],
                prepared["token_type_ids"],
                prepared["attention_mask"],
                batch,
                padded,
                want_logits=want_logits or not sample_on_device,
            )
        tokens_full = self.model.gather_tokens(tok_s, batch, padded, seq)  # [batch, seq]
        last_tok = tokens_full[:, cur - 1]  # [batch]
        if sample_on_device and not want_logits:
            return last_tok
        logits_full = self.model.gather_logits(log_s, batch, padded, seq)
        self.counters["logits_readbacks"] += 1
        last_logits = logits_full[:, cur - 1, :]  # [batch, vocab]
        return (last_tok, last_logits) if want_logits else last_logits

    # ---------------------------------------------------------- high level
    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input=None,
        enable_trace: bool = True,
        host_sampling: bool = False,
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """Growing-prefix reconstruction loop.

        Step ``i`` re-encodes the current context (prompt + accepted tokens,
        length ``P+i``) and reads the reconstruction argmax at the final
        position. That prediction is recorded; ``next_input(i, pred)`` (teacher
        forcing) or the prediction itself (free running) is appended for the
        next step. This reproduces the growing-prefix reconstruction pinned in
        the teacher-forcing/free-running references.

        Greedy on-device argmax is the default (``host_sampling=False``); the
        explicit host-sampling compatibility mode reads logits and argmaxes on
        host. Teacher forcing always uses ``enable_trace=True``.

        Free-running (``next_input=None``): a bidirectional encoder has **no
        next-token feedback** to iterate (feeding its own prediction back
        collapses, because reconstruction at a sequence boundary is not a
        next-token model). The model-appropriate free-running analog is the
        full-context per-position phoneme reconstruction of the prompt — the
        encoder's actual deterministic "generation". No token-feedback loop
        exists, so decode-loop degeneracy (doubled/collapsed tokens) cannot
        arise by construction; this is recorded honestly in doc/full_model.
        """
        if next_input is None:
            return self._free_running(
                prompt_token_ids, max_new_tokens, enable_trace=enable_trace, host_sampling=host_sampling
            )
        seq = list(int(t) for t in prompt_token_ids)
        preds: List[int] = []
        for i in range(max_new_tokens):
            ctx = torch.tensor([seq], dtype=torch.long)
            if host_sampling:
                logits = self.decode_forward(
                    ctx, sample_on_device=False, enable_trace=enable_trace, want_logits=False
                )  # [1, vocab]
                pred = int(torch.argmax(logits[0]).item())
                self.counters["host_argmax"] += 1
            else:
                tok = self.decode_forward(ctx, sample_on_device=True, enable_trace=enable_trace)  # [1]
                pred = int(tok[0].item())
            preds.append(pred)
            nxt = int(next_input(i, pred)) if next_input is not None else pred
            seq.append(nxt)
            if stop_on_eos and next_input is None and pred == 0 and i > 0:
                break
        return preds

    def _free_running(self, prompt_token_ids, max_new_tokens, *, enable_trace, host_sampling):
        """Full-context per-position reconstruction of the prompt (see generate)."""
        seq = [int(t) for t in prompt_token_ids]
        ctx = torch.tensor([seq], dtype=torch.long)
        prepared = self._prepare(ctx)
        batch, padded, sl = prepared["batch"], prepared["padded_seq_len"], prepared["seq_len"]
        if enable_trace:
            key = (batch, padded, prepared["attention_mask"] is not None, host_sampling)
            if key not in self.model._out_traces:
                self.counters["trace_captures"] += 1
            tok_s, log_s = self.model.decode_out_traced(prepared, want_logits=host_sampling)
            self.counters["trace_replays"] += 1
        else:
            tok_s, log_s = self.model._encode_readout_argmax(
                prepared["input_ids"],
                prepared["position_ids"],
                prepared["token_type_ids"],
                prepared["attention_mask"],
                batch,
                padded,
                want_logits=host_sampling,
            )
        if host_sampling:
            logits = self.model.gather_logits(log_s, batch, padded, sl)  # [1, S, vocab]
            self.counters["logits_readbacks"] += 1
            recon = torch.argmax(logits[0], dim=-1).tolist()
            self.counters["host_argmax"] += len(recon)
        else:
            recon = self.model.gather_tokens(tok_s, batch, padded, sl)[0].tolist()
        # deterministic full-context reconstruction; return up to max_new_tokens
        return recon[:max_new_tokens] if max_new_tokens < len(recon) else recon

    def reset(self) -> None:
        """No per-prompt state to wipe (stateless bidirectional encoder). Traces,
        weights, and compiled programs survive across prompts by design."""
        return None

    def teardown(self) -> None:
        self.model.release()


# --------------------------------------------------------------- factory
def build_generator(model_dir, mesh_device, **kwargs) -> KokoroGenerator:
    """Standard Metal readiness factory.

    Loads the Kokoro ``bert`` (plbert) config/vocab/weights and builds the full
    optimized-multichip model + reconstruction readout on ``mesh_device``.
    ``kwargs``: ``hf_model_id`` (default ``hexgrad/Kokoro-82M``), ``opt``
    (:class:`OptConfig`), ``policy`` (:class:`PrecisionPolicy`).
    """
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig

    hf_model_id = kwargs.get("hf_model_id", MODEL_ID)
    cfg = json.load(open(hf_hub_download(hf_model_id, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(hf_model_id, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}

    # Default precision/opt policy = the datatype-sweep-selected config (stage 07),
    # loaded from doc/datatype_sweep/selected_precision_config.json so the served/
    # measured runtime path uses exactly the swept-and-selected policy. Explicit
    # policy=/opt= kwargs override it (the sweep harness uses this); if the file is
    # absent the loader returns the dataclass defaults (== selected baseline).
    sel_policy, sel_opt, _ = load_selected()
    policy = kwargs.get("policy") or sel_policy
    opt = kwargs.get("opt") or sel_opt
    model = KokoroModel.from_state_dict(
        sd, hf_config=config, vocab=vocab, mesh_device=mesh_device, policy=policy, opt=opt
    )
    return KokoroGenerator(mesh_device=mesh_device, model=model, vocab=vocab)
