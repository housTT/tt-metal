# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fast all-on-device incremental decode for DeepSeek-V4: 43 resident LayerDeviceWeights +
streamed FP4 experts (tt/mla_v4_device.py). Per token, runs the full 43-layer forward on device
(mHC + MLA attention + compressor + MoE all on-device, validated) with the routed experts
streamed as pre-tilized bfloat4_b (the measured fast path). This is the throughput path the vLLM
generator drives — replaces the ~0.15 tok/s host-recompute KVDecoder.
"""
from __future__ import annotations

import os

import torch

import ttnn

# Insert a device sync every N layers inside the decode loop (0 = off -> pure async,
# one implicit sync/token at the logits read). Drains the async command-queue /
# deferred-dealloc backlog on a fixed cadence to test/kill the per-token spikes.
SYNC_EVERY = int(os.environ.get("DEEPSEEK_V4_SYNC_EVERY", "0"))
from models.demos.deepseek_v4.reference import real_weights as RW
from models.demos.deepseek_v4.tt import mla_v4_device as D
from models.demos.deepseek_v4.tt import modules as M


class FastDecoder:
    def __init__(self, device, store, cfg, scratch, num_layers=43, max_seq=2048):
        self.device = device
        self.store = store
        self.cfg = cfg
        self.scratch = scratch
        self.num_layers = num_layers
        self.max_seq = max_seq
        self.rd = cfg.qk_rope_head_dim
        top = scratch.model
        RW.load_globals(scratch, store)
        self.embed = top.embed_tokens
        self.norm = top.norm
        self.hc_head = top.hc_head
        self.lm_head = scratch.lm_head
        self.layer_types = list(cfg.layer_types[:num_layers])
        self.mlp_types = list(cfg.mlp_layer_types[:num_layers])
        self.type_of_layer = {
            "sliding_attention": "sliding",
            "compressed_sparse_attention": "CSA",
            "heavily_compressed_attention": "HCA",
        }

        def attn_kind(mod):
            c = getattr(mod, "compressor", None)
            return "sliding" if c is None else ("CSA" if "CSA" in type(c).__name__ else "HCA")

        scratch_by_type = {}
        for si, sl in enumerate(top.layers):
            key = (attn_kind(sl.self_attn), "hash" if getattr(sl.mlp, "is_hash", False) else "moe")
            scratch_by_type.setdefault(key, si)

        self.is_mesh = D._mesh_n(device) > 1
        # On a >1 mesh, experts are TP-sharded + streamed across all chips in parallel
        # (moe_device_fp4_mesh); on a single device, whole experts stream over one PCIe.
        self.fp4_cache = D.Fp4ExpertCacheMesh(store, device) if self.is_mesh else D.Fp4ExpertCache(store)
        # build 43 resident layer weight sets
        self.LW = []
        self.comp_rate = []
        for i in range(num_layers):
            key = (self.type_of_layer[self.layer_types[i]], "hash" if self.mlp_types[i] == "hash_moe" else "moe")
            sl = top.layers[scratch_by_type[key]]
            RW.load_layer(sl, i, store, skip_experts=True)
            lw = D.LayerDeviceWeights(sl, cfg, device, layer_idx=i, store=store, fp4_cache=self.fp4_cache)
            self.LW.append(lw)
            self.comp_rate.append(sl.self_attn.compressor.compress_rate if getattr(sl.self_attn, "compressor", None) else None)

        # precompute rope tables (host), interleaved on the fly
        pos = torch.arange(max_seq).unsqueeze(0)
        dummy = torch.zeros(1, max_seq, cfg.hidden_size)
        self.cos_m, self.sin_m = top.rotary_emb(dummy, position_ids=pos, layer_type="main")  # [1,max_seq,rd/2]
        self.cos_c, self.sin_c = top.rotary_emb(dummy, position_ids=pos, layer_type="compress")
        # lm_head resident on device
        self.lm_w = RW.dev_linear(store, device, ("lm_head",), self.lm_head.weight.data)

        # per-layer caches
        self.kv = [None] * num_layers
        self.hbuf = [None] * num_layers
        self.pos = 0

        # Optional: fully warm the host bf4 expert cache at startup (env DEEPSEEK_V4_WARM_ALL=1).
        # MoE routing is non-deterministic (~25/43 layers flip experts on identical input, measured),
        # so a PARTIAL cache keeps cold-building the fluctuating expert tail (~300ms each) -> per-token
        # latency spikes. Warming ALL experts into mem makes every routing choice a cheap mem hit, so
        # steady-state decode is flat regardless of routing (measured warm floor 449ms = 2.2 tok/s vs
        # 1.69 with no expert residency). ~145GB host RAM + one-time build (resumable via disk cache).
        if os.environ.get("DEEPSEEK_V4_WARM_ALL", "0") == "1" and hasattr(self.fp4_cache, "warm_all"):
            ne = getattr(cfg, "n_routed_experts", 256)
            # retain=True keeps the (cheap mmap) handles in self.c so decode gets mem hits. Assumes
            # the disk cache is pre-populated (run warm_stable_bench.py --build_only offline); if not,
            # this cold-builds the missing ones here (slower + higher RSS).
            _wlog = lambda m: print(f"[warm_all] {m}", flush=True)
            self.fp4_cache.warm_all(num_layers, ne, log=_wlog, retain=True)
            # fault the mmap'd payloads into page cache too, else to_device faults from disk mid-decode
            # (measured: without this a token spiked to 4296ms; with it, flat CV=0.05 @ 2.34 tok/s)
            if hasattr(self.fp4_cache, "preload_pagecache"):
                self.fp4_cache.preload_pagecache(log=_wlog)

    def reset(self):
        self.kv = [None] * self.num_layers
        self.hbuf = [None] * self.num_layers
        self.pos = 0

    def _il(self, cos, sin, positions):
        """Gather rope rows at `positions` and interleave -> device [1,1,len,rd]."""
        c = cos[:, positions, :].repeat_interleave(2, dim=-1)  # [1,len,rd]
        s = sin[:, positions, :].repeat_interleave(2, dim=-1)
        L = c.shape[1]
        return D._dev(c.reshape(1, 1, L, self.rd), self.device), D._dev(s.reshape(1, 1, L, self.rd), self.device)

    def decode_step(self, token_id: int, profile: bool = False) -> torch.Tensor:
        import time as _t

        cfg = self.cfg
        p = self.pos
        embed = self.embed(torch.tensor([[token_id]])).to(torch.float32)  # [1,1,H]
        streams = embed.unsqueeze(2).expand(1, 1, cfg.hc_mult, embed.shape[-1]).contiguous()
        streams = D._dev(streams, self.device)
        cos_md, sin_md = self._il(self.cos_m, self.sin_m, [p])  # [1,1,1,rd]
        if profile:
            ttnn.synchronize_device(self.device)
            t0 = _t.perf_counter()
        for i in range(self.num_layers):
            m = self.comp_rate[i]
            if m is not None:
                nw = (p + 1) // m
                cpos = [(j * m) for j in range(nw)] if nw > 0 else [0]
                cos_cd, sin_cd = self._il(self.cos_c, self.sin_c, cpos)
            else:
                cos_cd = sin_cd = None
            streams, main_kv, hbuf_full = D.decode_layer_device(
                streams, self.hbuf[i], self.kv[i], self.LW[i], cos_md, sin_md, cos_cd, sin_cd, self.device, token_id=token_id
            )
            self.kv[i] = main_kv
            self.hbuf[i] = hbuf_full
            if SYNC_EVERY and ((i + 1) % SYNC_EVERY == 0):
                ttnn.synchronize_device(self.device)
        if profile:
            ttnn.synchronize_device(self.device)
            t_layers = _t.perf_counter() - t0
            t0 = _t.perf_counter()
        # head
        hidden = M.hyper_head(D._host(streams, self.device).reshape(1, 1, cfg.hc_mult, cfg.hidden_size), self.hc_head, self.device)
        hidden = M.rms_norm(hidden, self.norm.weight.data, self.device, eps=cfg.rms_norm_eps)
        logits = M.linear_dev(hidden, self.lm_w, self.device)  # [1,1,vocab]
        if profile:
            ttnn.synchronize_device(self.device)
            print(f"    [prof] 43 layers {t_layers*1000:.0f} ms + head {(_t.perf_counter()-t0)*1000:.0f} ms", flush=True)
        self.pos = p + 1
        return logits.reshape(1, -1)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Seed KV/hbuf by running each prompt token incrementally. Returns last-token logits [1,vocab]."""
        self.reset()
        ids = input_ids.reshape(-1).tolist()
        logits = None
        for t in ids:
            logits = self.decode_step(int(t))
        return logits
