# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN sparse MoE feed-forward block for ornith-ai/Ornith-1.0-35B.

Every Ornith decoder layer — both ``linear_attention`` and ``full_attention`` kinds — uses the
same ``Qwen3_5MoeSparseMoeBlock``: a 256-expert top-8 routed SwiGLU MoE plus a sigmoid-gated
shared expert.

HF semantics being reproduced (``Qwen3_5MoeTopKRouter`` / ``Qwen3_5MoeExperts`` /
``Qwen3_5MoeSparseMoeBlock``)::

    router_logits = x @ gate.weight.T                     # [T, 256], no bias
    p             = softmax(router_logits, dtype=float32)
    w, idx        = topk(p, 8)
    w             = w / w.sum(-1, keepdim=True)
    gate_e, up_e  = linear(x, gate_up_proj[e]).chunk(2, -1)
    y            += down_proj[e] @ (silu(gate_e) * up_e) * w_e
    y            += sigmoid(shared_expert_gate(x)) * shared_expert(x)

Two facts make the device implementation tractable:

* ``topk`` then ``softmax`` over the 8 kept logits is algebraically identical to HF's
  ``softmax`` over 256 then ``topk`` then sum-renormalisation (softmax is monotone, so the
  selection is the same, and renormalising the 8 kept probabilities *is* a softmax over the 8
  kept logits). Doing it in that order avoids a 256-wide softmax and is far better conditioned.
* ``ttnn.sparse_matmul`` skips ``(batch-entry, expert)`` pairs whose ``sparsity`` entry is zero,
  so the routed experts run as three sparse matmuls with the router's dense score vector — no
  host gather, no torch, no per-expert Python loop.

Granularity note: one sparsity entry covers one 32-row batch entry, so the mask is the *union* of
the experts selected by the 32 tokens sharing that entry. Decode packs a step's users into one
tile, where the union is small (≤ 8·batch). Prefill packs 32 consecutive tokens per entry, where
the union approaches all 256 experts, so prefill pays roughly ``E/top_k`` redundant expert FLOPs —
the accepted cost of the single-device active-expert pattern. The router weights are applied after
the down projection, which is what makes the redundant experts cancel exactly (the op zero-fills
skipped output blocks, and unselected experts get a zero score).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn

TILE = 32

#: Tokens per ``sparse_matmul`` call in prefill. Each call materialises
#: ``num_experts * tokens * (2 * moe_intermediate + hidden)`` activation elements, so this bounds
#: the transient DRAM footprint (~0.5 GB at 256 tokens for Ornith's shapes) rather than the math.
PREFILL_GROUP_TOKENS = 256


def _sparse_matmul_config(m: int, n: int, in0_block_w: int):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for one sparse matmul shape.

    The sparse factory requires ``mcast_in0``, ``Kt % in0_block_w == 0`` and — critically — that
    the output blocks exactly tile a rectangle of the chosen grid
    (``num_cores_with_work == in0_mcast_receiver_num_cores``). Pick the largest core count that
    divides ``Nt`` and forms a rectangle no wider than 8, and give each core the whole M extent so
    ``num_blocks_y == 1``.
    """
    n_tiles = int(math.ceil(n / TILE))
    best_cores, best_cx, best_cy = 1, 1, 1
    for num_cores in range(1, min(65, n_tiles + 1)):
        if n_tiles % num_cores:
            continue
        for cy in range(1, 9):
            if num_cores % cy == 0:
                cx = num_cores // cy
                if cx <= 8 and num_cores > best_cores:
                    best_cores, best_cx, best_cy = num_cores, cx, cy
                    break
    per_core_n = n_tiles // best_cores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_cx, best_cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(1, int(math.ceil(m / TILE))),
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _largest_divisor_at_most(value: int, cap: int) -> int:
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


@dataclass
class MoEWeights:
    """Device-resident MoE weights, all produced at setup time."""

    router: ttnn.Tensor  # [1, 1, hidden, num_experts]
    expert_gate: ttnn.Tensor  # [1, E, hidden, moe_intermediate]
    expert_up: ttnn.Tensor  # [1, E, hidden, moe_intermediate]
    expert_down: ttnn.Tensor  # [1, E, moe_intermediate, hidden]
    shared_gate: ttnn.Tensor  # [1, 1, hidden, shared_intermediate]
    shared_up: ttnn.Tensor  # [1, 1, hidden, shared_intermediate]
    shared_down: ttnn.Tensor  # [1, 1, shared_intermediate, hidden]
    shared_router: ttnn.Tensor  # [1, 1, hidden, 1]


def load_moe_weights(mesh_device, config, state_dict, *, dtype=ttnn.bfloat16, prefix="mlp.") -> MoEWeights:
    """Build :class:`MoEWeights` from an HF ``mlp`` substate.

    ``state_dict`` keys expected (module-relative, i.e. what ``transformers`` holds after its
    registered expert fusion): ``gate.weight`` ``[E, hidden]``, ``experts.gate_up_proj``
    ``[E, 2*moe_intermediate, hidden]`` with gate first, ``experts.down_proj``
    ``[E, hidden, moe_intermediate]``, ``shared_expert.{gate,up,down}_proj.weight`` and
    ``shared_expert_gate.weight`` ``[1, hidden]``.

    All ``torch`` work happens here, at setup time; nothing in :class:`OrnithMoE`'s forward path
    touches the host.
    """
    import torch

    def get(name):
        return state_dict[f"{prefix}{name}"]

    def upload(t):
        return ttnn.as_tensor(
            t.to(torch.bfloat16).contiguous() if dtype == ttnn.bfloat16 else t.float().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    inter = config.moe_intermediate_size
    fused = get("experts.gate_up_proj")  # [E, 2*inter, hidden], gate rows first
    if fused.shape[1] != 2 * inter:
        raise ValueError(f"experts.gate_up_proj dim1 {fused.shape[1]} != 2*{inter}")
    expert_gate = fused[:, :inter, :].transpose(-2, -1).unsqueeze(0)  # [1, E, hidden, inter]
    expert_up = fused[:, inter:, :].transpose(-2, -1).unsqueeze(0)
    expert_down = get("experts.down_proj").transpose(-2, -1).unsqueeze(0)  # [1, E, inter, hidden]

    return MoEWeights(
        router=upload(get("gate.weight").transpose(0, 1).reshape(1, 1, config.dim, config.num_experts)),
        expert_gate=upload(expert_gate),
        expert_up=upload(expert_up),
        expert_down=upload(expert_down),
        shared_gate=upload(get("shared_expert.gate_proj.weight").transpose(0, 1).unsqueeze(0).unsqueeze(0)),
        shared_up=upload(get("shared_expert.up_proj.weight").transpose(0, 1).unsqueeze(0).unsqueeze(0)),
        shared_down=upload(get("shared_expert.down_proj.weight").transpose(0, 1).unsqueeze(0).unsqueeze(0)),
        shared_router=upload(get("shared_expert_gate.weight").transpose(0, 1).reshape(1, 1, config.dim, 1)),
    )


class OrnithMoE:
    """Router + 256 routed experts + gated shared expert, on device.

    ``forward`` takes and returns ``[1, 1, tokens, hidden]`` with ``tokens % 32 == 0``; the caller
    owns padding/slicing to a logical length.
    """

    def __init__(self, mesh_device, config, weights: MoEWeights):
        self.device = mesh_device
        self.cfg = config
        self.w = weights

        # Expert matmuls: HiFi4 without fp32 dest accumulation. fp32_dest_acc_en halves the matmul
        # dest register budget, which is a known Blackhole corruption source for this op family
        # (see models/demos/gemma4/tt/experts/prefill.py); HiFi4 alone carries the accuracy.
        self.expert_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # Router and shared expert: highest precision available. Expert *selection* in particular
        # is a discrete decision, so logit error near the top-8/top-9 boundary flips an expert
        # rather than perturbing a value.
        self.dense_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.gate_up_in0_block_w = _largest_divisor_at_most(config.dim // TILE, 16)
        self.down_in0_block_w = _largest_divisor_at_most(config.moe_intermediate_size // TILE, 8)
        # gate/up always run one 32-token tile per batch entry; down runs the whole call's tokens.
        self.gate_up_cfg = _sparse_matmul_config(TILE, config.moe_intermediate_size, self.gate_up_in0_block_w)
        self._down_cfg_cache: dict[int, object] = {}
        self.output_tile = ttnn.Tile([TILE, TILE])

    def _down_cfg(self, tokens: int):
        cfg = self._down_cfg_cache.get(tokens)
        if cfg is None:
            cfg = _sparse_matmul_config(tokens, self.cfg.dim, self.down_in0_block_w)
            self._down_cfg_cache[tokens] = cfg
        return cfg

    # ---------------- router ----------------
    def routing_weights(self, x):
        """Dense routing weights ``[1, 1, tokens, num_experts]`` (bfloat16, zeros off-selection).

        ``topk`` runs on the raw logits (monotone-equivalent to HF's softmax-then-topk) and the
        kept 8 logits are softmaxed, which reproduces HF's renormalised top-8 probabilities.

        The logits are produced in **float32**. Selection is a discrete decision, so logit noise
        near the top-8/top-9 boundary swaps an expert rather than perturbing a value. Measured on
        real layer-0 weights at 512 tokens (``logs/router_precision_ab.txt``): bfloat16 logits
        agree with the HF top-8 set on 95.5 % of tokens (score-vector L1 error 1.15 %, MoE-block
        PCC 0.999573); float32 logits agree on 99.8 % (0.19 %, 0.999816).

        ``ttnn.scatter`` rejects a float32 TILE destination, so the kept weights are cast to
        bfloat16 for the scatter — that only affects the weight *values* (relative error ~0.4 %,
        far below the selection effect), not which experts are chosen.

        The scatter destination is built with ``zeros_like`` + ``typecast`` rather than
        ``ttnn.zeros``: the latter uploads from the host, which is illegal inside a trace capture.
        """
        logits = ttnn.linear(x, self.w.router, dtype=ttnn.float32, compute_kernel_config=self.dense_ckc)
        values, indices = ttnn.topk(logits, k=self.cfg.num_experts_per_tok, dim=-1, sorted=True)
        weights = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.dense_ckc)
        zeros = ttnn.typecast(ttnn.zeros_like(logits), ttnn.bfloat16)
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=ttnn.typecast(weights, ttnn.bfloat16))
        ttnn.deallocate(logits)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        ttnn.deallocate(weights)
        return dense

    # ---------------- experts ----------------
    def _routed_experts(self, x, dense_routing, tokens):
        """Routed-expert output ``[1, 1, tokens, hidden]``.

        ``x``/``dense_routing`` are ``[1, 1, tokens, *]`` with ``tokens % 32 == 0``.

        Sparsity granularity: gate/up use a per-32-token-group union mask (one sparsity entry per
        ``(group, expert)``); down runs after the tokens have been folded into the M dimension, so
        its mask is the union over the whole call. ``nnz`` is always inferred at runtime — the
        masks are data-dependent, and a static ``nnz`` that disagrees with
        ``count_nonzero(sparsity)`` deadlocks the in0-mcast receivers.
        """
        E = self.cfg.num_experts
        H = self.cfg.dim
        I = self.cfg.moe_intermediate_size
        groups = tokens // TILE

        grouped_scores = ttnn.reshape(dense_routing, [1, groups, TILE, E])
        group_mask = ttnn.to_layout(
            ttnn.gtz(ttnn.sum(grouped_scores, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT
        )  # [1, groups, 1, E]
        call_mask = ttnn.to_layout(
            ttnn.gtz(ttnn.sum(dense_routing, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT
        )  # [1, 1, 1, E]

        a = ttnn.reshape(x, [1, groups, TILE, H])
        gate = ttnn.sparse_matmul(
            a,
            self.w.expert_gate,
            sparsity=group_mask,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=self.output_tile,
            program_config=self.gate_up_cfg,
            compute_kernel_config=self.expert_ckc,
            dtype=ttnn.bfloat16,
        )
        up = ttnn.sparse_matmul(
            a,
            self.w.expert_up,
            sparsity=group_mask,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=self.output_tile,
            program_config=self.gate_up_cfg,
            compute_kernel_config=self.expert_ckc,
            dtype=ttnn.bfloat16,
        )
        # sparse_matmul returns [1, groups, 1, E, TILE, N]. Swapping the group and expert axes and
        # collapsing gives expert-major [1, E, tokens, N] with token order preserved.
        gate = ttnn.reshape(ttnn.transpose(gate, 1, 3), [1, E, tokens, I])
        up = ttnn.reshape(ttnn.transpose(up, 1, 3), [1, E, tokens, I])
        hidden = ttnn.multiply(ttnn.silu(gate), up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down = ttnn.sparse_matmul(
            hidden,
            self.w.expert_down,
            sparsity=call_mask,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=self.output_tile,
            program_config=self._down_cfg(tokens),
            is_input_a_sparse=True,
            compute_kernel_config=self.expert_ckc,
            dtype=ttnn.bfloat16,
        )  # [1, E, tokens, H]
        ttnn.deallocate(hidden)
        ttnn.deallocate(group_mask)
        ttnn.deallocate(call_mask)

        # Score-weight and reduce over the expert axis. Blocks the sparsity mask skipped come back
        # exactly zero — verified directly by reading a masked output back after deliberately
        # dirtying DRAM with 3e38 values (see doc/functional_decoder/work_log.md, bug 9) — so the
        # multiply-then-reduce selects the active experts and cancels the rest.
        scores = ttnn.permute(dense_routing, (0, 3, 2, 1))  # [1, E, tokens, 1]
        weighted = ttnn.multiply(down, scores, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(down)
        ttnn.deallocate(scores)
        reduced = ttnn.experimental.fast_reduce_nc(weighted, dims=[1])
        ttnn.deallocate(weighted)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(reduced), [1, 1, tokens, H])

    # ---------------- shared expert ----------------
    def _shared_expert(self, x):
        gate = ttnn.linear(x, self.w.shared_gate, compute_kernel_config=self.dense_ckc)
        up = ttnn.linear(x, self.w.shared_up, compute_kernel_config=self.dense_ckc)
        act = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(act, self.w.shared_down, compute_kernel_config=self.dense_ckc)
        ttnn.deallocate(act)
        gate_scalar = ttnn.sigmoid(ttnn.linear(x, self.w.shared_router, compute_kernel_config=self.dense_ckc))
        gated = ttnn.multiply(out, gate_scalar)
        ttnn.deallocate(out)
        ttnn.deallocate(gate_scalar)
        return gated

    # ---------------- public ----------------
    def forward(self, x):
        """``x``: ``[1, 1, tokens, hidden]`` with ``tokens % 32 == 0``. Returns the same shape."""
        tokens = x.shape[2]
        if tokens % TILE:
            raise ValueError(f"MoE token count must be a multiple of {TILE}; got {tokens}")

        shared = self._shared_expert(x)

        if tokens <= PREFILL_GROUP_TOKENS:
            routed = self._routed_experts(x, self.routing_weights(x), tokens)
        else:
            parts = []
            for start in range(0, tokens, PREFILL_GROUP_TOKENS):
                span = min(PREFILL_GROUP_TOKENS, tokens - start)
                # tokens > PREFILL_GROUP_TOKENS here, so every slice is a strict sub-range and
                # never aliases ``x``.
                chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, start + span, self.cfg.dim])
                part = self._routed_experts(chunk, self.routing_weights(chunk), span)
                ttnn.deallocate(chunk)
                parts.append(part)
            routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            if len(parts) > 1:
                for part in parts:
                    ttnn.deallocate(part)

        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out
