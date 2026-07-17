# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Full-model TTNN wrapper for hexgrad/Kokoro-82M's plbert encoder.

Scope and architecture reality
------------------------------
Kokoro-82M is a **non-autoregressive StyleTTS2/ISTFTNet text-to-speech** model.
Its checkpoint (`kokoro-v1_0.pth`) has five components — ``bert`` (plbert),
``bert_encoder``, ``predictor``, ``decoder`` (ISTFTNet vocoder) and
``text_encoder`` — and its only attention-transformer is ``bert`` = plbert, an
HF ``AlbertModel`` (12 weight-tied ``AlbertLayer``s, hidden 768, 12 heads,
head_dim 64, intermediate 2048, vocab 178, max ctx 512). It is **bidirectional,
has no causal decoder, no KV cache, no next-token distribution, and no sampling**
anywhere in the real pipeline. Stages 01-04 brought up exactly this plbert
encoder (functional -> optimized -> multichip -> optimized-multichip) and
established — with four independent stage-review clean-passes — that the
autoregressive contract items (KV cache, paged cache, current-position advance,
MoE, token-by-token sampling) are **N/A**, replaced by a *stateless
decode == prefill* proof.

This module is the **full-model** assembly around the stage-04 optimized
multichip decoder. It preserves that decoder verbatim (import, no re-implement):
TP=4 head-parallel attention + sequence-parallel FFN + sequence-sharded residual
``[b,1,S/TP,H]``, bf16 activations / BFP8 linear weights / HiFi2 / fp32-dest-acc,
block-sharded L1 LayerNorm, persistent CCL buffers, 1 all_gather + 1
reduce_scatter per layer, and the inter-layer residual layout contract. No
fallback to single-chip, replicated, or host-side compute is introduced.

What "full HF forward path" means here
--------------------------------------
For an encoder the full HF forward path is
``AlbertModel(input_ids).last_hidden_state`` — the contextual phoneme embeddings
that feed Kokoro's prosody predictor and vocoder. :meth:`forward` returns exactly
that, gathered from the sequence shards to a full ``[b, S, H]`` tensor at the
*terminal* boundary only (one localized all_gather, measured separately), keeping
the whole decoder stack on the optimized sequence-sharded contract.

Reconstruction readout (LM-head analog for the readiness harness)
-----------------------------------------------------------------
The shared readiness runners (``run_prefill_check`` / ``run_teacher_forcing`` /
``run_autoregressive``) score a per-position vocabulary distribution. plbert
ships **no** LM head. We add a *tied-embedding reconstruction readout* built only
from shipped weights — the ALBERT factorised-embedding tie run in reverse:

    logits = last_hidden_state @ (W_map @ W_word^T)            # [b, S, vocab]

where ``W_map = encoder.embedding_hidden_mapping_in.weight`` ([H,128]) and
``W_word = embeddings.word_embeddings.weight`` ([vocab,128]). This is the model's
MLM-style reconstruction of each phoneme from its *bidirectional* context. It is
a **fidelity/consistency probe**, deliberately not a claim that Kokoro emits
phoneme tokens: the real model output is ``last_hidden_state`` (validated by PCC
>= 0.995 vs HF, carried from stages 01-04). Because HF and TT apply the identical
readout, top-1/5/100 over this readout measures TT-vs-HF *full-model* numerical
fidelity, and the free-running reconstruction gives a genuine, non-degenerate
phoneme completion for the degenerate-output gate.

The readout is a single ``[H, vocab_padded]`` matmul folded on the host, run on
the *sequence shard* (per-token, no cross-token dependency), so only the tiny
logits/token tensors gather at the boundary. On-device greedy argmax stays on the
shard; host-logit readback exists only in the explicit host-sampling
compatibility path.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import TILE, PrecisionPolicy, _round_up
from models.autoports.hexgrad_kokoro_82m.tt.optimized_multichip_decoder import OptConfig, OptimizedMultichipDecoder
from models.common.lightweightmodule import LightweightModule


def _pad_vocab(vocab_size: int) -> int:
    """Round vocab up to a TILE multiple so the readout matmul is tile-aligned."""
    return _round_up(vocab_size, TILE)


class KokoroModel(LightweightModule):
    """Full plbert encoder + reconstruction readout on the TP=4 mesh.

    Wraps :class:`OptimizedMultichipDecoder` (the stage-04 optimized multichip
    encoder) unchanged and adds the terminal gather, the tied-embedding
    reconstruction readout, and on-device greedy argmax. The decode path is the
    decoder's traced full-sequence replay (stateless: ``decode == prefill``).
    """

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        vocab: Optional[dict] = None,
        weights=None,
        readout_weight: Optional[torch.Tensor] = None,
        policy: Optional[PrecisionPolicy] = None,
        opt: Optional[OptConfig] = None,
        decoder: Optional[OptimizedMultichipDecoder] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = hf_config
        self.vocab = vocab or {}
        self.policy = policy or PrecisionPolicy()
        self.opt = opt or OptConfig()
        self.hidden_size = int(hf_config.hidden_size)
        self.vocab_size = int(hf_config.vocab_size)
        self.padded_vocab = _pad_vocab(self.vocab_size)
        self.max_position_embeddings = int(hf_config.max_position_embeddings)

        self.decoder = decoder or OptimizedMultichipDecoder(
            mesh_device=mesh_device, hf_config=hf_config, weights=weights, policy=self.policy, opt=self.opt
        )
        self.tp = self.decoder.tp
        self.mesh_shape = self.decoder.mesh_shape

        # --- reconstruction readout weight ---------------------------------
        # Folded host-side: W_readout[H, vocab] = W_map[H,128] @ W_word[vocab,128]^T.
        # Padded to a tile multiple in the vocab dim; pad columns forced to -inf
        # via an additive bias so on-device argmax never selects a pad column.
        self._build_readout(readout_weight)

        # terminal-gather + readout trace store, keyed by (batch, padded_seq_len, has_mask)
        self._out_traces: dict = {}
        # explicit readout-matmul program configs, cached per M-tile count
        self._readout_pc_cache: dict = {}

    def _build_readout(self, readout_weight: Optional[torch.Tensor]):
        if readout_weight is None:
            raise ValueError(
                "KokoroModel requires the folded readout weight [H, vocab]; use KokoroModel.from_state_dict."
            )
        assert readout_weight.shape == (
            self.hidden_size,
            self.vocab_size,
        ), f"readout_weight must be [H={self.hidden_size}, vocab={self.vocab_size}], got {tuple(readout_weight.shape)}"
        wpad = torch.zeros((self.hidden_size, self.padded_vocab), dtype=torch.float32)
        wpad[:, : self.vocab_size] = readout_weight.float()
        bias = torch.zeros((self.padded_vocab,), dtype=torch.float32)
        bias[self.vocab_size :] = -1.0e9  # mask pad columns out of argmax
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        self.readout_w = ttnn.from_torch(
            wpad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=rep
        )
        self.readout_b = ttnn.from_torch(
            bias.reshape(1, self.padded_vocab),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=rep,
        )

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        vocab: Optional[dict] = None,
        mesh_device,
        policy: Optional[PrecisionPolicy] = None,
        opt: Optional[OptConfig] = None,
    ):
        """Build the full model from a real Kokoro ``bert`` state dict.

        All host->device weight conversion happens here (decoder weights via
        :meth:`OptimizedMultichipDecoder.from_state_dict`, plus the folded
        reconstruction readout weight). Nothing is loaded on the hot path.
        """
        policy = policy or PrecisionPolicy()
        sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        decoder = OptimizedMultichipDecoder.from_state_dict(
            sd, hf_config=hf_config, mesh_device=mesh_device, policy=policy, opt=opt
        )
        w_map = sd["encoder.embedding_hidden_mapping_in.weight"]  # [H, 128]
        w_word = sd["embeddings.word_embeddings.weight"]  # [vocab, 128]
        readout_weight = (w_map.float() @ w_word.float().t()).contiguous()  # [H, vocab]
        return cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            vocab=vocab,
            readout_weight=readout_weight,
            policy=policy,
            opt=opt,
            decoder=decoder,
        )

    # -------------------------------------------------------------- input prep
    def prepare_inputs(self, input_ids: torch.Tensor, *, attention_mask: Optional[torch.Tensor] = None):
        """Delegate to the decoder's sequence-sharded input construction.

        Accepts any logical length 1..max_position_embeddings, including
        non-tile-aligned lengths (padded internally to a TP*TILE multiple).
        """
        return self.decoder.prepare_inputs(input_ids, attention_mask=attention_mask)

    # ---------------------------------------------------------------- readout
    def _readout_program_config(self, m_tiles: int):
        """Explicit 2D program config for the terminal readout (LM-head) matmul.

        Shape (per sequence shard): [b,1,local_seq,H] @ [H, vocab_padded]. The
        stage-05 path used the decoder's auto ``core_grid=CoreGrid(8,10)``, which
        left this matmul SLOW at ``in0_block_w=1`` (~21.6 µs @T=128 eager). Because
        the vocab is small and REPLICATED (no vocab-split / cross-device gather —
        see doc rejection ledger), the readout is a plain sequence-local matmul:
        gx tiles the ``vocab_padded`` N dim, gy divides the M row-tiles, and a
        K-dividing ``in0_block_w`` (gcd(8, H/32)=8) restores utilization
        (~11.8 µs @T=128, ~2x). Cached per M-tile count; valid at every
        tile-aligned local length.
        """
        pc = self._readout_pc_cache.get(m_tiles)
        if pc is not None:
            return pc
        n_tiles = self.padded_vocab // TILE  # 6 for vocab_padded=192
        gx = max(g for g in range(1, 9) if n_tiles % g == 0)
        gy = max(g for g in range(1, 9) if m_tiles % g == 0)
        per_core_n = n_tiles // gx
        in0_block_w = math.gcd(8, self.hidden_size // TILE)  # 8 divides 24 (H=768)
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=per_core_n if per_core_n <= 4 else 1,
            per_core_M=m_tiles // gy,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        self._readout_pc_cache[m_tiles] = pc
        return pc

    def _readout(self, hidden_s):
        """Reconstruction logits on the sequence shard: [b,1,local_seq,H] -> [b,1,local_seq,vocab_padded].

        Per-token matmul (no cross-token dependency), so it runs directly on the
        sequence shard and keeps the terminal gather to the tiny vocab tensor.
        """
        b, _, local_seq, _ = hidden_s.shape
        m_tiles = (b * local_seq) // TILE
        logits = ttnn.linear(
            hidden_s,
            self.readout_w,
            bias=self.readout_b,
            compute_kernel_config=self.decoder.matmul_kernel_config,
            program_config=self._readout_program_config(m_tiles),
            dtype=ttnn.bfloat16,
        )
        return logits

    def _gather_seq(self, x_s):
        """All-gather a sequence-sharded [b,1,S/TP,C] tensor to full [b,1,S,C].

        Terminal-only collective (logits/hidden), separate from the per-layer
        collectives inside the decoder. Uses the decoder's CCL manager/topology.
        """
        return ttnn.experimental.all_gather_async(
            x_s,
            dim=2,
            multi_device_global_semaphore=self.decoder.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.decoder.ccl_num_links,
            topology=self.decoder.ccl_topology,
            barrier_semaphore=self.decoder.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        )

    def _argmax_seq(self, logits_s):
        """On-device greedy argmax over the vocab dim on the sequence shard.

        Returns sequence-sharded token ids [b,1,local_seq] (uint32). Stays on the
        shard; only the tiny token tensor is gathered by the caller.

        Optimization (stage 06): ``ttnn.argmax`` with ``dim=-1`` runs **single
        core** on a TILE input (~112 µs @T=128 / ~454 µs @T=512 device — the
        dominant avoidable terminal cost of the stage-05 path), but runs
        **multi-core** on a ROW_MAJOR input. We untilize the (tiny, replicated
        vocab) logits shard to ROW_MAJOR first, which is 2.7-3x faster end to end
        (43 µs @T=128 / 148 µs @T=512 including the untilize) and returns the
        bit-identical greedy token. ``logits_s`` itself is left in TILE layout so
        the ``want_logits`` host cross-check path is unaffected. The argmax output
        is always UINT32/ROW_MAJOR/INTERLEAVED regardless of input layout, so the
        downstream ``gather_tokens`` contract is unchanged.
        """
        logits_rm = ttnn.to_layout(logits_s, ttnn.ROW_MAJOR_LAYOUT)
        tok_s = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        return tok_s

    # --------------------------------------------------- device-only forwards
    def _encode_shard(self, prepared, *, traced: bool):
        """Run the full 12-layer optimized multichip encoder.

        Returns the sequence-sharded last_hidden_state [b,1,S/TP,H]. ``traced``
        selects the decoder's traced replay (production path) vs eager encode.
        """
        if traced:
            return self.decoder.decode_forward(
                prepared["input_ids"],
                prepared["position_ids"],
                prepared["token_type_ids"],
                prepared["attention_mask"],
                batch=prepared["batch"],
                seq_len=prepared["padded_seq_len"],
            )
        return self.decoder.prefill_forward(
            prepared["input_ids"],
            prepared["position_ids"],
            prepared["token_type_ids"],
            prepared["attention_mask"],
            batch=prepared["batch"],
            seq_len=prepared["padded_seq_len"],
        )

    def forward(
        self, input_ids: torch.Tensor, *, attention_mask=None, traced: bool = False, return_logits: bool = False
    ):
        """Full HF forward path. Returns torch last_hidden_state [b, S, H].

        When ``return_logits`` also returns reconstruction logits [b, S, vocab].
        This is a convenience host wrapper (prepares inputs, runs the encoder,
        gathers, slices to logical length); the generator drives the low-level
        device path directly.
        """
        prepared = self.prepare_inputs(input_ids, attention_mask=attention_mask)
        hidden_s = self._encode_shard(prepared, traced=traced)
        hidden_full = ttnn.to_torch(hidden_s, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))
        b, padded, seq = prepared["batch"], prepared["padded_seq_len"], prepared["seq_len"]
        hidden = hidden_full.reshape(b, padded, self.hidden_size)[:, :seq, :].float()
        if not return_logits:
            return hidden
        logits_s = self._readout(hidden_s)
        logits_full = ttnn.to_torch(logits_s, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))
        logits = logits_full.reshape(b, padded, self.padded_vocab)[:, :seq, : self.vocab_size].float()
        return hidden, logits

    # --------------------------------------------- token-out traced decode
    # One captured graph covering encode -> reconstruction readout -> on-device
    # argmax over the stable sequence-sharded decoder inputs. The greedy token
    # stays on device (only the tiny [b, S] token-id tensor is read back by the
    # caller); no host argmax and no full-vocab logits readback on this path.
    def _encode_readout_argmax(self, in_ids, in_pos, in_tt, in_mask, batch, full_seq, *, want_logits):
        hidden_s = self.decoder._encode(in_ids, in_pos, in_tt, in_mask, batch, full_seq)
        logits_s = self._readout(hidden_s)
        ttnn.deallocate(hidden_s)
        tok_s = self._argmax_seq(logits_s)
        if want_logits:
            return tok_s, logits_s
        ttnn.deallocate(logits_s)
        return tok_s, None

    def capture_out_trace(self, prepared, *, want_logits: bool = False):
        batch = prepared["batch"]
        full_seq = prepared["padded_seq_len"]
        has_mask = prepared["attention_mask"] is not None
        key = (batch, full_seq, has_mask, want_logits)
        rec = self._out_traces.get(key)
        if rec is not None:
            return rec
        dev = self.mesh_device
        in_ids = ttnn.clone(prepared["input_ids"])
        in_pos = ttnn.clone(prepared["position_ids"])
        in_tt = ttnn.clone(prepared["token_type_ids"])
        in_mask = ttnn.clone(prepared["attention_mask"]) if has_mask else None

        # warm every op (readout matmul + argmax) so capture hits no program-cache miss
        w_tok, w_log = self._encode_readout_argmax(
            in_ids, in_pos, in_tt, in_mask, batch, full_seq, want_logits=want_logits
        )
        ttnn.deallocate(w_tok)
        if w_log is not None:
            ttnn.deallocate(w_log)
        ttnn.synchronize_device(dev)

        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        tok_s, log_s = self._encode_readout_argmax(
            in_ids, in_pos, in_tt, in_mask, batch, full_seq, want_logits=want_logits
        )
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)

        rec = {
            "trace_id": trace_id,
            "in_ids": in_ids,
            "in_pos": in_pos,
            "in_tt": in_tt,
            "in_mask": in_mask,
            "tok": tok_s,
            "log": log_s,
        }
        self._out_traces[key] = rec
        return rec

    def decode_out_traced(self, prepared, *, want_logits: bool = False):
        """Traced token-out decode. Returns sequence-sharded (token_ids, logits|None)."""
        rec = self.capture_out_trace(prepared, want_logits=want_logits)
        dev = self.mesh_device
        ttnn.copy(prepared["input_ids"], rec["in_ids"])
        ttnn.copy(prepared["position_ids"], rec["in_pos"])
        ttnn.copy(prepared["token_type_ids"], rec["in_tt"])
        if prepared["attention_mask"] is not None and rec["in_mask"] is not None:
            ttnn.copy(prepared["attention_mask"], rec["in_mask"])
        ttnn.execute_trace(dev, rec["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return rec["tok"], rec["log"]

    def gather_tokens(self, tok_s, batch, padded, seq_len):
        """Gather sequence-sharded token ids [b,1,S/TP] -> torch [b, seq_len]."""
        t = ttnn.to_torch(tok_s, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return t.reshape(batch, padded)[:, :seq_len].to(torch.long)

    def gather_logits(self, log_s, batch, padded, seq_len):
        t = ttnn.to_torch(log_s, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))
        return t.reshape(batch, padded, self.padded_vocab)[:, :seq_len, : self.vocab_size].float()

    def release(self):
        self.decoder.release_traces()
        for rec in self._out_traces.values():
            ttnn.release_trace(self.mesh_device, rec["trace_id"])
        self._out_traces.clear()
