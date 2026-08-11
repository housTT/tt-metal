# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for ornith-ai/Ornith-1.0-35B (``Qwen3_5MoeForConditionalGeneration``).

This is the graph-fused successor to
:mod:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder`. The **semantics are
identical** — same prefill/decode contract, same paged KV cache, same DeltaNet state, same
determinism, same non-aligned sequence-length support — but the op graph is rewritten with
dedicated tt-metal ops, structural rewrites and op merges. Every rewrite is PCC-checked against
the same HuggingFace golden as the functional decoder
(``tests/test_fused_decoder.py``) and is recorded in ``doc/fused_decoder/README.md``.

What is fused (see the README for the measured before/after and the rejected candidates)
-----------------------------------------------------------------------------------------
Dedicated fused ops (highest priority)

* ``full_attention`` head split: ``slice ×3 → reshape → permute`` →
  :func:`ttnn.experimental.nlp_create_qkv_heads` (prefill) /
  :func:`ttnn.experimental.nlp_create_qkv_heads_decode` (decode).
* ``full_attention`` head merge: ``transpose → untilize/reshape/tilize`` →
  :func:`ttnn.experimental.nlp_concat_heads`.
* Partial RoPE: ``slice ×3 → neg → concat → mul ×2 → add → slice → concat`` (10 ops) →
  ``slice → rotary_embedding_hf → slice → concat`` (4 ops in prefill; decode adds two transposes to
  reach the op's ``[batch, seq, heads, dim]`` expectation). A one-op variant also exists — permute the
  head-dim order of ``q_proj``/``k_proj``/``q_norm``/``k_norm`` at load time so a full-width
  rotate-half reproduces the 64-of-256 rotation exactly (:func:`_rope_head_permutation`) — and it
  is exact, but measured slower with a 4× larger table, so it is a selectable mode, not the
  default (:data:`DEFAULT_ROPE_MODE`).
* Gated DeltaNet prefill: the *flat* ``chunk_gated_delta_rule`` contract (rank-3 token-major
  q/k/v) removes three head-split relayouts, both explicit L2 norms and the ``scale`` multiply — all
  of which were device ops; none of this was ever a host round trip — the
  op's prep kernel does the L2 norm and folds the scale. ``output_head_major=True`` returns a
  TILE head-major result, removing the ROW_MAJOR→TILE conversion of the whole activation.
* Gated DeltaNet head merge: :func:`ttnn.experimental.nlp_concat_heads` — for prefill only; at
  ``seq_len 1`` it collapses onto one core and a ``permute + reshape`` is several times cheaper
  (``doc/fused_decoder/work_log.md`` §4.12 carries the measurement; the ratio moves between runs).
* Paged cache: one batched :func:`ttnn.experimental.paged_fill_cache` (``batch_idx_tensor``)
  instead of one call per user, and one :func:`ttnn.experimental.paged_fused_update_cache`
  instead of separate K and V updates.
* Expert-axis reduction: :func:`ttnn.experimental.deepseek_moe_fast_reduce_nc`.

Graph rewrites (second priority)

* Shared-LHS matmul packing, ×4: ``q_proj``/``k_proj``/``v_proj`` (+ the output gate) → one
  ``[hidden, 9216]`` matmul; ``in_proj_{qkv,z,a,b}`` → one ``[hidden, 12352]`` matmul; the shared
  expert's ``gate``/``up``/router → one ``[hidden, 1056]`` matmul; the routed experts' ``gate`` and
  ``up`` → one ``2·moe_intermediate``-wide sparse matmul, which doubles its usable core count.
* MoE score placement: the per-(token, expert) router score is applied to the **input** of the down
  projection (``moe_intermediate`` wide) instead of its **output** (``hidden`` wide) — exactly
  equivalent by linearity, and 4× less elementwise work over the 256-expert axis.
* MoE expert-axis relayout: ``silu`` + ``multiply`` happen in the sparse matmul's native 6-D
  layout so only **one** expert/group permute is needed instead of two, and that permute
  degenerates to a reshape whenever there is a single 32-token group — which, at the shipped
  ``moe_group_tokens = 32``, is *every* call, prefill included, so the multi-group branch is not
  exercised by the default configuration.
* MoE expert group = 32 tokens (:data:`DEFAULT_MOE_GROUP_TOKENS`), so the down projection gets the
  same per-group expert skipping the gate/up projections already had.
* One router per MoE call instead of one per expert group.
* One ``repeat_interleave`` over the adjacent Q/K head pair instead of one each.
* ``reshape + permute`` for both decode head relayouts, which needs no layout conversion at all, and
  a conv-history concat done in ROW_MAJOR so that it does not lower to untilize → concat → tilize of
  the whole 8192-wide stream (the conversions that get it there are counted in README §6).
* An explicit ``core_grid`` on the three recurrent-state matmuls, which ttnn's default heuristic
  otherwise places on 4 cores.

Op merging (third priority)

* ``silu(gate) * up`` → ``ttnn.multiply(gate, up, input_tensor_a_activations=[SILU])``.
* ``attn * sigmoid(gate)`` → ``ttnn.multiply(attn, gate, input_tensor_b_activations=[SIGMOID])``.
* ``softplus(a + dt_bias)`` → ``ttnn.add(a, dt_bias, activations=[SOFTPLUS])``.
* ``transpose(k) @ delta`` → ``ttnn.matmul(k, delta, transpose_a=True)``.
* The query's ``head_k_dim ** -0.5`` scale folded into the L2 norm's multiply.

Prefill / decode contract
-------------------------
Unchanged from the functional decoder — see that module's docstring. In particular
``prefill_forward`` accepts **any** logical ``seq_len``; the 128-token physical alignment and the
2048-token internal chunk are internal.

Neither forward path calls ``torch``, ``ttnn.from_torch``, ``ttnn.to_torch`` or any host fallback
**on an allocated layer**: every weight, constant, cache and state buffer is created in
:meth:`FusedDecoder.from_state_dict` / :meth:`FusedDecoder.allocate_state`. The one exception is the
lazy path: a forward call on a layer whose state was never allocated — or a ``full_attention`` layer
handed a larger batch than it was allocated for — calls ``allocate_state`` itself, and that *is* host
work (see its docstring). It happens at most once per layer per batch,
never inside a captured trace, and never inside anything this stage measures, all of which allocate
explicitly first. ``tests/test_fused_decoder.py::test_lazy_allocation_is_the_only_host_call`` pins
exactly that: the guards fire on the first forward of an unallocated layer and stay silent on the
next. The functional decoder's ``allocate_state`` is host-free, so this is a real divergence and is
recorded as one in ``doc/fused_decoder/README.md`` §6.
"""

from __future__ import annotations

from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import OrnithDecoderConfig
from models.common.lightweightmodule import LightweightModule

TILE = 32

#: Physical alignment of a prefill block. 128 keeps every block start legal for the chunked-SDPA
#: 64-token q/k chunks, the paged cache's 64-token blocks, and the flat ``chunk_gated_delta_rule``
#: contract (which requires the block length to be a multiple of its 32-token internal chunk).
PREFILL_ALIGN = 128

#: Default logical tokens per internal prefill block.
DEFAULT_PREFILL_CHUNK = 2048

#: Paged KV cache block size (tokens per block).
DEFAULT_PAGE_BLOCK_SIZE = 64

#: Tokens per ``sparse_matmul`` call in prefill.
#:
#: One sparsity entry covers one 32-row batch entry, so a call's *down*-projection mask is the
#: union of the experts chosen by **all** its tokens. At 256 tokens/call that union is essentially
#: all 256 experts and the down projection does no skipping at all; at 32 tokens/call it is the
#: same partial expert union the gate/up projections already get. 32 is both the finest legal value
#: (one sparsity entry covers exactly one 32-row batch entry) and the fastest measured one — the sweep
#: over 32/64/128/256/512 on a 2048-token prefill is in ``doc/fused_decoder/logs/ab_moe_group_tokens.txt``
#: and tabulated in that stage's README. It also cuts the transient expert-activation footprint 8x.
#: Decode is unaffected: its token count is one 32-row tile either way.
DEFAULT_MOE_GROUP_TOKENS = 32

#: ``chunk_gated_delta_rule`` internal chunk. 32 is the only value that enables the flat
#: (rank-3 token-major) q/k contract, which is what carries the in-kernel L2 norm.
GDN_CHUNK = 32

#: Channels per ``ttnn.conv1d`` call for the depthwise causal conv.
#:
#: The op cannot serve Ornith's full ``conv_dim = 8192``: height-sharded it runs out of L1 at every
#: legal slice count, at 128 slices and above it refuses the slice count itself, and block-sharded it
#: fails at kernel compile — three distinct blockers, all in
#: ``doc/fused_decoder/logs/probe_conv1d_and_norm.txt`` and catalogued in ``work_log.md`` §4.4. But a
#: depthwise conv is separable over
#: channels, and 4096 is the widest split that runs. It also happens to be the natural split point:
#: channels ``[0, 4096)`` are exactly Q and K, and ``[4096, 8192)`` are exactly V, so the two halves
#: feed the delta rule directly with no output concatenation.
CONV1D_CHANNELS = 4096

#: How partial RoPE is lowered. See :class:`OrnithFusedRope`. Both modes were measured
#: (``doc/fused_decoder/logs/ab_rope_mode.txt``): ``"full"`` is one op instead of four in prefill
#: (six in decode, which adds two transposes), but it is
#: slower on traced decode — the wider cos/sin table costs more in the gather's tilize than the
#: slice and concat it removes — and indistinguishable on prefill, while needing a 4x larger table
#: (head_dim-wide instead of rope_dim-wide) at the full context. So the measured default is
#: ``"partial"``; ``"full"`` is kept because it is the fewer-op graph, because the knob is what
#: makes that comparison reproducible, and because ``test_rope_mode_equivalence`` uses it to prove
#: the head-dim permutation is self-consistent across Q, K and the KV cache.
DEFAULT_ROPE_MODE = "partial"

_SILU = [ttnn.UnaryOpType.SILU]
_SIGMOID = [ttnn.UnaryOpType.SIGMOID]
_SOFTPLUS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_dim(tensor, dim: int, amount: int):
    """Zero-pad ``amount`` elements onto the high end of ``dim``.

    Aliasing warning: when the requested logical padding already fits inside the tensor's
    physical tile padding (e.g. 1 → 32 rows), the result can share ``tensor``'s buffer. Callers
    must therefore never free the pre-pad tensor separately.
    """
    padding = [(0, 0)] * len(tensor.shape)
    padding[dim] = (0, amount)
    return ttnn.pad(tensor, padding, 0.0)


def _slice_owned(tensor, begins, ends):
    """``(slice, owned)`` where ``owned`` says whether the result is a fresh buffer."""
    if all(b == 0 for b in begins) and list(ends) == [int(d) for d in tensor.shape]:
        return tensor, False
    return ttnn.slice(tensor, list(begins), list(ends)), True


def _slice_last(tensor, start: int, end: int):
    """Slice ``[start, end)`` of the last dim, keeping every leading dim whole.

    Rejects a whole-dim range rather than returning one: ``ttnn.slice`` short-circuits a full cover
    to the input tensor itself, so a caller that deallocates the result would free its own input.
    Every shipped call site slices a strict sub-range; the ones that can legitimately want the whole
    tensor branch around this (``_rope_prefill``/``_rope_decode`` at ``rope_dim == head_dim``), and
    anything that cannot know statically should use ``_slice_owned``.
    """
    shape = [int(d) for d in tensor.shape]
    if start == 0 and end == shape[-1]:
        raise ValueError(
            f"_slice_last({start}, {end}) covers the whole last dim of {shape}: ttnn.slice would "
            "alias the input and the caller's deallocate would free it — use _slice_owned"
        )
    begins = [0] * len(shape)
    ends = list(shape)
    begins[-1] = start
    ends[-1] = end
    return ttnn.slice(tensor, begins, ends)


def num_blocks_for_context(context: int, block_size: int = DEFAULT_PAGE_BLOCK_SIZE) -> int:
    """Paged blocks needed to hold ``context`` tokens, rounded to the 32-entry page-table stick."""
    blocks = _align_up(context, block_size) // block_size
    return _align_up(blocks, 32)


def _rope_head_permutation(head_dim: int, rope_dim: int) -> list[int]:
    """Head-dim reordering that turns partial RoPE into a full-width rotate-half.

    Ornith rotates only the first ``rope_dim`` (64) of each ``head_dim`` (256) head, with the
    rotate-half pairing ``(i, i + rope_dim // 2)``. ``ttnn.experimental.rotary_embedding_hf``
    only rotates the *whole* width it is given, pairing ``(j, j + head_dim // 2)``.

    Reordering the head dims so that the rotated pair ``(i, i + rope_dim//2)`` lands on
    ``(i, i + head_dim//2)`` makes the full-width op compute exactly the partial rotation, as long
    as the cos/sin table is 1 / 0 on the pass-through positions (:meth:`OrnithFusedRope`). The
    returned list maps **new position → old position**::

        new[0 : 32]        = old[0 : 32]            # rotate-half "first half" of the 64-wide RoPE
        new[32 : 128]      = old[64 : 160]          # pass-through
        new[128 : 160]     = old[32 : 64]           # rotate-half "second half"
        new[160 : 256]     = old[160 : 256]         # pass-through

    Q and K share the permutation, so ``q·k`` is unchanged; V and ``o_proj`` are untouched, so the
    layer output is bit-for-bit the same math. The KV cache stores permuted K, which is an
    internal detail of this layer.
    """
    half_rope = rope_dim // 2
    half_head = head_dim // 2
    if rope_dim == head_dim:
        return list(range(head_dim))
    if half_rope > half_head:
        raise ValueError(f"rope_dim {rope_dim} cannot exceed head_dim {head_dim}")
    lo = list(range(0, half_rope))
    lo_pass = list(range(rope_dim, rope_dim + (half_head - half_rope)))
    hi = list(range(half_rope, rope_dim))
    hi_pass = list(range(rope_dim + (half_head - half_rope), head_dim))
    perm = lo + lo_pass + hi + hi_pass
    assert sorted(perm) == list(range(head_dim)), "head permutation must be a bijection"
    return perm


class OrnithFusedRope(LightweightModule):
    """Device-resident RoPE tables for :func:`ttnn.experimental.rotary_embedding_hf`.

    Two modes, both of which replace the functional decoder's 10-op hand-written partial rotation
    with the dedicated kernel; ``doc/fused_decoder/README.md`` records the measurement that picks
    the default.

    ``"full"``
        The table is widened to ``head_dim`` with ``cos = 1`` / ``sin = 0`` on the pass-through
        dims. Combined with :func:`_rope_head_permutation` on the Q/K projection and norm weights,
        one full-width rotate-half *is* the partial rotation — **one op, no slice, no concat**.
    ``"partial"``
        The ordinary ``rope_dim``-wide table; the layer slices the rotated head prefix, rotates it
        and concatenates the pass-through tail back on (4 ops in prefill; 6 in decode, which adds two
        transposes to reach the op's expected axis order).
    """

    def __init__(
        self,
        mesh_device,
        config,
        *,
        max_context: int,
        table_context: int | None = None,
        # DEFAULT_ROPE_MODE, not "full": constructing this directly used to silently get the
        # measured-slower head_dim-wide table, while FusedDecoder.from_state_dict passed the fast one.
        mode: str = DEFAULT_ROPE_MODE,
        dtype=ttnn.bfloat16,
    ):
        import torch

        if mode not in ("full", "partial"):
            raise ValueError(f"unknown rope mode {mode!r}")
        self.device = mesh_device
        self.mode = mode
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.width = config.head_dim if mode == "full" else config.rope_dim
        # ttnn.experimental.rotary_embedding_hf requires the rotated width to put rotate_half's
        # midpoint on a tile boundary: `width == TILE_WIDTH || width % 64 == 0`
        # (rotary_embedding_hf_device_operation.cpp). The functional decoder's hand-rolled rotation
        # had no such constraint, so this is a narrowing the dedicated op introduces — Ornith's
        # rope_dim of 64 satisfies it, but a config that did not would otherwise reach a raw TT_FATAL.
        if self.width != 32 and self.width % 64:
            raise ValueError(
                f"rotary_embedding_hf cannot rotate a width of {self.width} (rope_dim={self.rope_dim}, "
                f"head_dim={self.head_dim}, mode={mode!r}): it must be 32 or a multiple of 64"
            )
        self.table_rows = int(table_context or max_context)

        half_rope = self.rope_dim // 2
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.int64).float() / self.rope_dim)
        )
        positions = torch.arange(self.table_rows, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [rows, rope_dim // 2]

        if mode == "full":
            cos = torch.ones(self.table_rows, self.width, dtype=torch.float32)
            sin = torch.zeros(self.table_rows, self.width, dtype=torch.float32)
            slots = ((0, half_rope), (self.head_dim // 2, self.head_dim // 2 + half_rope))
        else:
            cos = torch.empty(self.table_rows, self.width, dtype=torch.float32)
            sin = torch.empty(self.table_rows, self.width, dtype=torch.float32)
            slots = ((0, half_rope), (half_rope, self.rope_dim))
        for dst_lo, dst_hi in slots:
            cos[:, dst_lo:dst_hi] = freqs.cos()
            sin[:, dst_lo:dst_hi] = freqs.sin()

        def upload(t):
            return ttnn.from_torch(
                t.reshape(1, 1, self.table_rows, self.width),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.cos_table = upload(cos)
        self.sin_table = upload(sin)

    def prefill_forward(self, start_pos: int, seq_len: int):
        """cos/sin ``[1, 1, seq_len, width]`` for ``[start_pos, start_pos+seq_len)``."""
        end = start_pos + seq_len
        if end > self.table_rows:
            raise ValueError(f"rope window [{start_pos}, {end}) exceeds the {self.table_rows}-row table")
        out = []
        for table in (self.cos_table, self.sin_table):
            t = ttnn.slice(table, [0, 0, start_pos, 0], [1, 1, end, self.width])
            out.append(ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return out[0], out[1]

    def decode_forward(self, rot_idxs):
        """cos/sin ``[1, 1, batch, width]`` gathered at the per-user positions in ``rot_idxs``.

        ``rot_idxs`` is a ``[1, batch]`` uint32 device tensor, so a captured trace only needs its
        contents refreshed. The batch axis takes the place of the sequence axis, which is what lets
        ``rotary_embedding_hf`` run in its (interleaved) prefill mode on a decode step — its native
        decode mode requires height-sharded inputs *and* height-sharded caches.
        """
        cos = ttnn.embedding(rot_idxs, self.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_table, layout=ttnn.TILE_LAYOUT)
        batch = int(rot_idxs.shape[-1])
        cos = ttnn.reshape(cos, [1, 1, batch, self.width])
        sin = ttnn.reshape(sin, [1, 1, batch, self.width])
        return cos, sin


def _sparse_matmul_config(m: int, n: int, in0_block_w: int):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` for one sparse matmul shape.

    The sparse factory requires ``mcast_in0``, ``Kt % in0_block_w == 0`` and that the output blocks
    exactly tile a rectangle of the chosen grid. Pick the largest core count that divides ``Nt`` and
    forms a rectangle no wider than 8, and give each core the whole M extent.
    """
    import math

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


def _conv_compute_config(arch):
    """Compute-kernel config for the depthwise conv, shared by weight preparation and the forward.

    ``prepare_conv_weights`` bakes the compute config into the prepared layout, so the config used at
    setup and the one passed to :func:`ttnn.conv1d` must be the same object shape — hence one
    definition rather than two matching literals.
    """
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _conv1d_host_weights(conv_w, config):
    """Host-side ``ttnn.conv1d`` weight tensors, one per :data:`CONV1D_CHANNELS` block.

    ``ttnn.prepare_conv_weights`` rejects a device tensor, so the host copies are kept and the
    per-(batch, length) preparation happens in :meth:`FusedDecoder.allocate_state`.
    """
    channels = CONV1D_CHANNELS
    kernel = config.linear_conv_kernel_dim
    if config.conv_dim % channels:
        return []
    return [
        ttnn.from_torch(
            conv_w[idx * channels : (idx + 1) * channels].reshape(channels, 1, 1, kernel),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        for idx in range(config.conv_dim // channels)
    ]


def _drop_prepared(prepared, phys):
    """Discard one length's prepared weights, freeing the device buffers rather than dropping the ref.

    ``prepare_conv_weights`` always moves its result to the device, so a dropped entry is device
    memory. At batch 32 every one of the 16 candidate lengths is dropped, and the same argument the
    probe input is freed under applies here.
    """
    for weight in prepared.pop(phys, ()):
        ttnn.deallocate(weight)


def _prepare_conv1d_weights(mesh_device, host_weights, config, prefill_chunk, batch, compute_config):
    """Pre-prepare ``ttnn.conv1d`` weights for every physical block length the layer can produce.

    ``ttnn.prepare_conv_weights`` is a **host** call whose output layout depends on the batch and the
    input length, so it cannot happen inside a forward pass without reintroducing a host round trip.
    Every internal prefill block is a multiple of :data:`PREFILL_ALIGN` and at most ``prefill_chunk``
    tokens, and the conv sees ``kernel-1`` extra history rows, so the whole set of input lengths is
    known at ``allocate_state`` time and is small (16 for the default 2048-token chunk).

    Returns ``{physical_block_length: [prepared weight per CONV1D_CHANNELS block]}``. A length that
    could not be prepared is simply absent and the layer falls back to the FIR form for it, so a
    ttnn change that breaks weight preparation or the validating execution below costs performance
    rather than correctness.

    That guarantee covers *this* function only. ``_conv1d_halves`` calls ``ttnn.conv1d`` unguarded, so
    a length that passes the probe here and then fails in a forward raises rather than falling back.
    It is not a theoretical gap: the dominant refusal class is a pressure-dependent per-bank L1
    allocation failure, and this probe runs at ``allocate_state`` time when the large prefill
    activations are not yet resident. README §8 records it.
    """
    channels = CONV1D_CHANNELS
    kernel = config.linear_conv_kernel_dim
    if not host_weights:
        return {}
    import torch

    conv_cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    prepared: dict[int, list] = {}
    for phys in range(PREFILL_ALIGN, prefill_chunk + 1, PREFILL_ALIGN):
        length = phys + kernel - 1
        try:
            prepared[phys] = [
                ttnn.prepare_conv_weights(
                    weight_tensor=host,
                    weights_format="OIHW",
                    in_channels=channels,
                    out_channels=channels,
                    batch_size=batch,
                    input_height=1,
                    input_width=length,
                    kernel_size=(1, kernel),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    has_bias=False,
                    groups=channels,
                    device=mesh_device,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=compute_config,
                    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                for host in host_weights
            ]
        except Exception:  # noqa: BLE001 - an unpreparable length falls back to the FIR path
            _drop_prepared(prepared, phys)
            continue
        # Preparing the weights is not proof the op will run: the conv's circular buffers are sized
        # at program build and at large batches exceed L1 ("Statically allocated circular buffers ...
        # grow to N B which is beyond max L1 size"), and — more often, above batch 4 — the sharded
        # input's per-bank allocation is refused outright (bank_manager.cpp:462). Neither is
        # predictable from the shape alone, so each (batch, length) is executed
        # once here, on a zero input, and dropped if it throws. That keeps the decision at setup
        # time and out of the forward path, and it is exact rather than a guessed bound.
        probe = None
        try:
            probe = ttnn.from_torch(
                torch.zeros(batch, length, 1, channels, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            out = ttnn.conv1d(
                input_tensor=probe,
                weight_tensor=prepared[phys][0],
                device=mesh_device,
                in_channels=channels,
                out_channels=channels,
                batch_size=batch,
                input_length=length,
                kernel_size=kernel,
                stride=1,
                padding=0,
                dilation=1,
                groups=channels,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=compute_config,
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(out)
        except Exception:  # noqa: BLE001 - this (batch, length) does not fit; use the FIR path
            _drop_prepared(prepared, phys)
        finally:
            # Freed on both paths and explicitly: at batch 32 this input is ~0.5 GB, and the failing
            # (batch, length) pairs are exactly the large ones, so leaving it to refcounting would
            # hold the largest probes longest and across the next iteration's allocation.
            if probe is not None:
                ttnn.deallocate(probe)
    return prepared


class FusedMoE:
    """Router + 256 routed experts + gated shared expert, graph-fused.

    Same math as :class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.moe.OrnithMoE`; five
    rewrites:

    1. **Shared-LHS packing of the shared expert.** Its ``gate``/``up`` projections and its sigmoid
       router share the input, so they run as one matmul over a concatenated weight and are sliced
       apart (the router column is zero-padded to a tile so the slice offsets stay tile-aligned).
    2. **Shared-LHS packing of the routed experts.** ``gate`` and ``up`` become one ``2·I``-wide
       sparse matmul. The win is core count: the sparse factory gives each core one N-tile output
       block and cannot split M (a group is a single 32-row tile), so ``N = I`` is pinned to
       ``I/32`` cores and ``N = 2·I`` doubles that.
    3. **Score placement.** ``down_proj`` is linear, so scaling its *input* by the router score is
       identical to scaling its output — and the input is ``moe_intermediate`` (512) wide rather
       than ``hidden`` (2048) wide, i.e. 4× less elementwise traffic over the 256-expert axis.
    4. **One expert-axis permute.** ``silu(gate) * up`` is evaluated in the sparse matmul's native
       ``[1, groups, 1, E, 32, N]`` layout (with the SiLU folded into the multiply), so only the
       product is relaid out to the expert-major ``[1, E, tokens, N]`` the down projection wants.
       With a single 32-token group — which, at the shipped ``moe_group_tokens = 32``, is every
       call, prefill included — that relayout is a reshape rather than a permute.
    5. **One router per call.** ``topk`` and the scatter chain are single-core on a 256-wide last
       dim and barely scale with the token count, so routing runs once for the whole call and each
       expert group takes a tile-aligned slice of the dense score vector.
    """

    def __init__(self, mesh_device, config, weights, *, group_tokens: int = DEFAULT_MOE_GROUP_TOKENS):
        self.device = mesh_device
        self.cfg = config
        self.w = weights
        self.group_tokens = group_tokens

        # Expert matmuls: HiFi4 without fp32 dest accumulation (fp32_dest_acc_en halves the matmul
        # dest register budget, a known Blackhole corruption source for this op family).
        self.expert_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.dense_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.gate_up_in0_block_w = _largest_divisor_at_most(config.dim // TILE, 16)
        self.down_in0_block_w = _largest_divisor_at_most(config.moe_intermediate_size // TILE, 8)
        self.gate_up_cfg = _sparse_matmul_config(TILE, 2 * config.moe_intermediate_size, self.gate_up_in0_block_w)
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

        Unchanged from the functional decoder: float32 logits (expert *selection* is a discrete
        decision, so logit noise near the top-8/top-9 boundary swaps an expert rather than
        perturbing a value), ``topk`` on the raw logits, ``softmax`` over the kept 8, scatter into a
        dense vector.

        Two replacements were assessed and rejected, both recorded in ``doc/fused_decoder``:
        ``ttnn.experimental.deepseek.moe.generalized_moe_gate`` — one kernel for softmax + top-k +
        normalize, but bfloat16-only, rejected on that ground alone and **never timed**; the
        99.8 %/95.5 % top-8 set agreement it is rejected against is the functional stage's measured
        bfloat16-vs-float32 router A/B, applied to it because it is bfloat16-only — and a
        ``ge(kth) → where → softmax(256)`` threshold rewrite, which *was* timed and is slower at these
        shapes at identical accuracy (work_log.md §4.3, §4.12).
        """
        logits = ttnn.linear(x, self.w["router"], dtype=ttnn.float32, compute_kernel_config=self.dense_ckc)
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
        """Routed-expert output ``[1, 1, tokens, hidden]``."""
        E = self.cfg.num_experts
        H = self.cfg.dim
        I = self.cfg.moe_intermediate_size
        groups = tokens // TILE

        grouped_scores = ttnn.reshape(dense_routing, [1, groups, TILE, E])
        group_mask = ttnn.to_layout(
            ttnn.gtz(ttnn.sum(grouped_scores, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT
        )  # [1, groups, 1, E]
        if groups == 1:
            # The per-group union over the single group *is* the whole-call union; only the shape
            # differs, and [1,1,1,E] is what the group mask already is.
            call_mask = ttnn.reshape(group_mask, [1, 1, 1, E])
            call_owned = False
        else:
            call_mask = ttnn.to_layout(
                ttnn.gtz(ttnn.sum(dense_routing, dim=-2, keepdim=True)), ttnn.ROW_MAJOR_LAYOUT
            )  # [1, 1, 1, E]
            call_owned = True

        a = ttnn.reshape(x, [1, groups, TILE, H])
        # Shared-LHS packing of gate and up. The motivation is core count, not dispatch: the sparse
        # factory hands each core one N-tile block and cannot split M (a group is a single 32-row
        # tile), so an N=512 matmul is pinned to 512/32 = 16 cores while the packed N=1024 one uses
        # 32. Measured at the decode shapes in doc/fused_decoder/logs/probe_gate_up_pack.txt, which
        # also checks the packed form against the two-matmul one with `torch.equal`, not PCC alone.
        packed = ttnn.sparse_matmul(
            a,
            self.w["expert_gate_up"],
            sparsity=group_mask,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=self.output_tile,
            program_config=self.gate_up_cfg,
            compute_kernel_config=self.expert_ckc,
            dtype=ttnn.bfloat16,
        )
        gate = _slice_last(packed, 0, I)
        up = _slice_last(packed, I, 2 * I)
        ttnn.deallocate(packed)
        # SwiGLU folded into one binary op, still in the sparse matmul's native
        # [1, groups, 1, E, TILE, I] layout so only the product needs a relayout.
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=_SILU, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        if groups == 1:
            # Swapping a pair of extent-1 axes is a pure relabel: drop straight to expert-major.
            hidden = ttnn.reshape(hidden, [1, E, tokens, I])
        else:
            swapped = ttnn.transpose(hidden, 1, 3)
            ttnn.deallocate(hidden)
            hidden = ttnn.reshape(swapped, [1, E, tokens, I])
            del swapped  # the reshape is a view; keeping the handle would outlive the buffer's free

        # Score the *input* of the down projection instead of its output: identical by linearity,
        # and moe_intermediate (512) wide instead of hidden (2048) wide.
        scores = ttnn.permute(dense_routing, (0, 3, 2, 1))  # [1, E, tokens, 1]
        scaled = ttnn.multiply(hidden, scores, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        ttnn.deallocate(scores)

        down = ttnn.sparse_matmul(
            scaled,
            self.w["expert_down"],
            sparsity=call_mask,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=self.output_tile,
            program_config=self._down_cfg(tokens),
            is_input_a_sparse=True,
            compute_kernel_config=self.expert_ckc,
            dtype=ttnn.bfloat16,
        )  # [1, E, tokens, H]
        ttnn.deallocate(scaled)
        if call_owned:
            ttnn.deallocate(call_mask)
        ttnn.deallocate(group_mask)

        # deepseek_moe_fast_reduce_nc over ttnn.experimental.fast_reduce_nc: same latency at these
        # shapes but a materially more accurate accumulation (PCC 0.999999 vs 0.999409 against a
        # float32 sum of 256 bfloat16 expert blocks — doc/fused_decoder/logs/probe_router_and_reduce.txt).
        reduced = ttnn.experimental.deepseek_moe_fast_reduce_nc(down, dim=1, split_size=H)[0]
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(reduced), [1, 1, tokens, H])

    # ---------------- shared expert ----------------
    def _shared_expert(self, x):
        """``sigmoid(router(x)) * down(silu(gate(x)) * up(x))`` in one packed matmul + two ops."""
        inter = self.cfg.shared_expert_intermediate_size
        fused = ttnn.linear(x, self.w["shared_in"], compute_kernel_config=self.dense_ckc)
        gate = _slice_last(fused, 0, inter)
        up = _slice_last(fused, inter, 2 * inter)
        router = _slice_last(fused, 2 * inter, 2 * inter + 1)
        ttnn.deallocate(fused)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=_SILU)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(act, self.w["shared_down"], compute_kernel_config=self.dense_ckc)
        ttnn.deallocate(act)
        gated = ttnn.multiply(out, router, input_tensor_b_activations=_SIGMOID)
        ttnn.deallocate(out)
        ttnn.deallocate(router)
        return gated

    # ---------------- public ----------------
    def forward(self, x):
        """``x``: ``[1, 1, tokens, hidden]`` with ``tokens % 32 == 0``. Returns the same shape."""
        tokens = x.shape[2]
        if tokens % TILE:
            raise ValueError(f"MoE token count must be a multiple of {TILE}; got {tokens}")

        shared = self._shared_expert(x)

        # Routing runs **once** for the whole call, not once per expert group: the router's topk and
        # its scatter chain barely scale with the token count (they are single-core on a 256-wide
        # last dim), so 64 group-sized routers cost 64x what one whole-call router does. Slicing the
        # dense score vector per group is tile-aligned and nearly free.
        dense = self.routing_weights(x)
        if tokens <= self.group_tokens:
            routed = self._routed_experts(x, dense, tokens)
            ttnn.deallocate(dense)
        else:
            parts = []
            for start in range(0, tokens, self.group_tokens):
                span = min(self.group_tokens, tokens - start)
                chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, start + span, self.cfg.dim])
                scores = ttnn.slice(dense, [0, 0, start, 0], [1, 1, start + span, self.cfg.num_experts])
                part = self._routed_experts(chunk, scores, span)
                ttnn.deallocate(chunk)
                ttnn.deallocate(scores)
                parts.append(part)
            ttnn.deallocate(dense)
            routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
            if len(parts) > 1:
                for part in parts:
                    ttnn.deallocate(part)

        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out


class FusedDecoder(LightweightModule):
    """One graph-fused Ornith decoder layer on a TTNN mesh device.

    Construct with :meth:`from_state_dict`; call :meth:`allocate_state` (and, for
    ``full_attention`` layers, :meth:`allocate_kv_cache` / :meth:`attach_kv_cache`) before the
    first forward. The public API matches
    :class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.functional_decoder.FunctionalDecoder`.
    """

    def __init__(
        self,
        mesh_device,
        config: OrnithDecoderConfig,
        layer_idx: int,
        *,
        weights: dict,
        moe,
        rope,
        max_context: int,
        page_block_size: int,
        prefill_chunk: int,
    ):
        self.device = mesh_device
        self.cfg = config
        self.layer_idx = layer_idx
        self.kind = config.layer_kind(layer_idx)
        self.is_full_attention = self.kind == "full_attention"
        self.w = weights
        self.moe = moe
        self.rope = rope
        self.max_context = max_context
        self.page_block_size = page_block_size
        self.prefill_chunk = prefill_chunk

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.conv_compute_kernel_config = _conv_compute_config(mesh_device.arch())
        # rotary_embedding_hf defaults to math_approx_mode=True; the layer is accuracy-graded
        # against an HF float32 golden, so the exact SFPU path is used instead.
        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.k_cache = None
        self.v_cache = None
        self.recurrent_state = None
        self.conv_state = None  # list of ``conv_kernel_dim - 1`` buffers, each [B, 1, conv_dim]
        self.conv1d_lengths = []  # prefill block lengths ttnn.conv1d accepted at the allocated batch
        self.batch_size = None
        self.batch_idxs = None  # [batch] int32 device tensor for the batched paged_fill_cache
        # The recurrent read `q @ state` is a rank-4 [B, 32, 1, 128] x [B, 32, 128, 128] batched
        # matmul, i.e. B*32 per-head matmuls.
        # Left to ttnn's default heuristic it lands on 4 cores; naming the whole grid puts the 32
        # per-head matmuls across it (measured in doc/fused_decoder/logs/probe_decode_micro.txt).
        grid = mesh_device.compute_with_storage_grid_size()
        self.full_core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_context: int | None = None,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
        moe_group_tokens: int = DEFAULT_MOE_GROUP_TOKENS,
        rope_mode: str = DEFAULT_ROPE_MODE,
        dtype=ttnn.bfloat16,
        **kwargs,
    ) -> "FusedDecoder":
        """Build a layer from an HF decoder-layer state dict (same keys as the functional decoder)."""
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        import torch

        config = OrnithDecoderConfig.from_hf_config(hf_config)
        max_context = int(max_context or config.max_position_embeddings)
        if prefill_chunk % PREFILL_ALIGN:
            raise ValueError(f"prefill_chunk {prefill_chunk} must be a multiple of {PREFILL_ALIGN}")
        if moe_group_tokens % TILE:
            raise ValueError(f"moe_group_tokens {moe_group_tokens} must be a multiple of {TILE}")
        if prefill_chunk % page_block_size:
            # _attention_prefill computes blk0 = chunk_start_idx // page_block_size and lets
            # paged_fill_cache fill from offset 0 of the sliced page table, so a chunk boundary that
            # is not also a page boundary would write the chunk at the wrong offset within its block.
            # chunk_start_idx is always a multiple of the chunk size, so this is the condition.
            raise ValueError(
                f"prefill_chunk {prefill_chunk} must be a multiple of page_block_size "
                f"{page_block_size}: chunk boundaries have to land on page boundaries"
            )
        kind = config.layer_kind(layer_idx)

        def upload(t, tensor_dtype=dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                t.to(torch.bfloat16).contiguous() if tensor_dtype == ttnn.bfloat16 else t.float().contiguous(),
                dtype=tensor_dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        weights: dict = {}
        for key, dst in (("input_layernorm.weight", "attn_norm"), ("post_attention_layernorm.weight", "ff_norm")):
            weights[dst] = upload((state_dict[key].float() + 1.0).reshape(1, 1, 1, -1))

        rope = None
        if kind == "full_attention":
            n_heads, n_kv, head_dim = config.n_heads, config.n_kv_heads, config.head_dim
            perm = torch.tensor(
                _rope_head_permutation(head_dim, config.rope_dim) if rope_mode == "full" else range(head_dim),
                dtype=torch.long,
            )

            # q_proj is 2x wide: per head, [query(head_dim) | output-gate(head_dim)].
            qg = state_dict["self_attn.q_proj.weight"].float().reshape(n_heads, 2 * head_dim, config.dim)
            q_rows = qg[:, :head_dim, :][:, perm, :].reshape(n_heads * head_dim, config.dim)
            gate_rows = qg[:, head_dim:, :].reshape(n_heads * head_dim, config.dim)
            k_rows = state_dict["self_attn.k_proj.weight"].float().reshape(n_kv, head_dim, config.dim)[:, perm, :]
            k_rows = k_rows.reshape(n_kv * head_dim, config.dim)
            v_rows = state_dict["self_attn.v_proj.weight"].float()
            # Shared-LHS packing, in the order nlp_create_qkv_heads wants: [q | k | v | gate].
            weights["attn_in"] = upload(torch.cat([q_rows, k_rows, v_rows, gate_rows], dim=0).transpose(0, 1))
            weights["o_proj"] = upload(state_dict["self_attn.o_proj.weight"].transpose(0, 1))
            # Q/K norms act over head_dim only and follow the same head-dim permutation. RMSNorm is
            # permutation-equivariant, so permuting the weight reproduces the unpermuted result.
            for src, dst in (("self_attn.q_norm.weight", "q_norm"), ("self_attn.k_norm.weight", "k_norm")):
                weights[dst] = upload(((state_dict[src].float() + 1.0)[perm]).reshape(1, 1, 1, -1))
            rope = OrnithFusedRope(
                mesh_device,
                config,
                max_context=max_context,
                # Rounded up to a whole prefill chunk and then given one more chunk of headroom:
                # the final block's *physical* window can end past max_context, and those extra rows
                # only ever rotate zero-padded activations.
                table_context=_align_up(max_context, prefill_chunk) + prefill_chunk,
                mode=rope_mode,
            )
        elif kind == "linear_attention":
            prefix = "linear_attn."
            # Shared-LHS packing of the four in-projections. `a` and `b` are num_value_heads (32)
            # wide, i.e. exactly one tile, so every slice offset stays tile-aligned.
            weights["gdn_in"] = upload(
                torch.cat(
                    [
                        state_dict[prefix + "in_proj_qkv.weight"].float(),
                        state_dict[prefix + "in_proj_z.weight"].float(),
                        state_dict[prefix + "in_proj_a.weight"].float(),
                        state_dict[prefix + "in_proj_b.weight"].float(),
                    ],
                    dim=0,
                ).transpose(0, 1)
            )
            weights["gdn_out"] = upload(state_dict[prefix + "out_proj.weight"].transpose(0, 1))
            # Gated-DeltaNet output norm is a *standard* RMSNorm (weights ~ 1), not zero-centered.
            weights["gdn_norm"] = upload(state_dict[prefix + "norm.weight"].float().reshape(1, 1, 1, -1))
            conv_w = state_dict[prefix + "conv1d.weight"].float()
            weights["conv_taps"] = [
                upload(conv_w[:, 0, k].reshape(1, 1, -1)) for k in range(config.linear_conv_kernel_dim)
            ]
            # ttnn.conv1d cannot take all conv_dim channels at once, but a depthwise conv is
            # separable over channels and CONV1D_CHANNELS-wide calls do run — several times faster
            # than the FIR form at 2048 tokens; the two `CONV1DTIME` lines of
            # doc/fused_decoder/logs/probe_conv1d_and_norm.txt carry the measurement, and are cited
            # rather than copied here because they move a little on every re-run. The
            # weights depend on the input length, so every length the layer can produce is prepared
            # here, on the host, once.
            weights["conv1d_host"] = _conv1d_host_weights(conv_w, config)
            weights["conv1d_compute_config"] = _conv_compute_config(mesh_device.arch())
            weights["conv1d_weights"] = {}
            weights["A_neg"] = upload(
                (-state_dict[prefix + "A_log"].float().exp()).reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            weights["dt_bias"] = upload(
                state_dict[prefix + "dt_bias"].float().reshape(1, 1, -1), tensor_dtype=ttnn.float32
            )
            from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import build_fused_const_tiles

            weights["gdn_const_tiles"] = build_fused_const_tiles(mesh_device, GDN_CHUNK)
            weights["pos_ramp"] = upload(
                torch.arange(prefill_chunk, dtype=torch.float32).reshape(1, prefill_chunk, 1),
                tensor_dtype=ttnn.float32,
            )
        else:
            raise ValueError(f"unsupported layer kind {kind!r}")

        moe_weights = cls._load_moe_weights(mesh_device, config, state_dict, upload)
        moe = FusedMoE(mesh_device, config, moe_weights, group_tokens=moe_group_tokens)

        return cls(
            mesh_device,
            config,
            layer_idx,
            weights=weights,
            moe=moe,
            rope=rope,
            max_context=max_context,
            page_block_size=page_block_size,
            prefill_chunk=prefill_chunk,
        )

    @staticmethod
    def _load_moe_weights(mesh_device, config, state_dict, upload, prefix="mlp."):
        """Device-resident MoE weights, with the shared expert's three projections packed.

        ``experts.gate_up_proj`` is ``[E, 2*moe_intermediate, hidden]`` with the gate rows first
        (``transformers``' registered fusion for ``qwen3_5_moe_text``).
        """
        import torch

        def get(name):
            return state_dict[f"{prefix}{name}"]

        inter = config.moe_intermediate_size
        fused = get("experts.gate_up_proj")
        if fused.shape[1] != 2 * inter:
            raise ValueError(f"experts.gate_up_proj dim1 {fused.shape[1]} != 2*{inter}")

        shared_inter = config.shared_expert_intermediate_size
        # [gate | up | shared-expert sigmoid router], the router zero-padded out to one tile so the
        # slice offsets are tile-aligned; the padding columns are exact zeros and never read.
        router_col = get("shared_expert_gate.weight").float().transpose(0, 1).reshape(config.dim, 1)
        shared_in = torch.cat(
            [
                get("shared_expert.gate_proj.weight").float().transpose(0, 1),
                get("shared_expert.up_proj.weight").float().transpose(0, 1),
                router_col,
                torch.zeros(config.dim, TILE - 1, dtype=torch.float32),
            ],
            dim=1,
        )

        return {
            "router": upload(get("gate.weight").float().transpose(0, 1).reshape(1, 1, config.dim, config.num_experts)),
            # [1, E, hidden, 2*moe_intermediate] with the gate half first: one shared-LHS sparse
            # matmul per group instead of two half-width ones (see FusedMoE._routed_experts).
            "expert_gate_up": upload(fused.transpose(-2, -1).unsqueeze(0)),
            "expert_down": upload(get("experts.down_proj").transpose(-2, -1).unsqueeze(0)),
            "shared_in": upload(shared_in.reshape(1, 1, config.dim, 2 * shared_inter + TILE)),
            "shared_down": upload(
                get("shared_expert.down_proj.weight").float().transpose(0, 1).reshape(1, 1, shared_inter, config.dim)
            ),
        }

    # ------------------------------------------------------------------ state
    def allocate_kv_cache(self, num_blocks: int, dtype=ttnn.bfloat16):
        """Allocate and attach a paged KV cache ``[num_blocks, n_kv_heads, page_block_size, head_dim]``."""
        if not self.is_full_attention:
            return None
        shape = [num_blocks, self.cfg.n_kv_heads, self.page_block_size, self.cfg.head_dim]
        self.k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return self.k_cache, self.v_cache

    def attach_kv_cache(self, k_cache, v_cache):
        if not self.is_full_attention:
            raise TypeError("linear_attention layers have no KV cache")
        self.k_cache = k_cache
        self.v_cache = v_cache

    def allocate_state(self, batch_size: int):
        """Allocate the per-batch state. Buffers are persistent and only ever written in place, so
        their device addresses stay valid across trace replays.

        **This is setup, and it is not host-free**: it builds the paged-fill row indices and, for
        ``linear_attention``, prepares and probes the ``ttnn.conv1d`` weights. The forward paths call
        it lazily if the caller never did, so a *first* forward on an unallocated layer performs host
        calls and every subsequent one does not. Call it explicitly before capturing a trace or before
        any measurement, as this stage's tests and benchmarks do.
        """
        # Freed before being replaced: prefill_forward/decode_forward re-enter here when a
        # full_attention layer is handed a larger batch than it was allocated for.
        if self.batch_idxs is not None:
            ttnn.deallocate(self.batch_idxs)
            self.batch_idxs = None
        for prepared in self.w.get("conv1d_weights", {}).values():
            for weight in prepared:
                ttnn.deallocate(weight)
        self.w["conv1d_weights"] = {}

        import torch

        self.batch_size = batch_size
        if self.is_full_attention:
            # Row indices for the single batched paged_fill_cache call.
            self.batch_idxs = ttnn.from_torch(
                torch.arange(batch_size, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            return
        # ttnn.conv1d weights depend on the batch as well as the input length, so they are prepared
        # here rather than in from_state_dict: allocate_state is setup, and this keeps every host
        # call out of the forward paths.
        self.w["conv1d_weights"] = _prepare_conv1d_weights(
            self.device,
            self.w.get("conv1d_host", []),
            self.cfg,
            self.prefill_chunk,
            batch_size,
            self.w["conv1d_compute_config"],
        )
        # Which block lengths actually got a working conv1d program at this batch. Coverage shrinks
        # as the batch grows (the conv's circular buffers are sized per batch), so it is logged
        # rather than assumed: at batch 1 every length works, at batch 32 none do and the whole
        # prefill runs on the FIR fallback. doc/fused_decoder/README.md tabulates the measured set and
        # the per-batch split between the two refusal classes.
        self.conv1d_lengths = sorted(self.w["conv1d_weights"])
        total = self.prefill_chunk // PREFILL_ALIGN
        logger.info(
            f"layer {self.layer_idx} batch {batch_size}: ttnn.conv1d accepted "
            f"{len(self.conv1d_lengths)}/{total} prefill block lengths {self.conv1d_lengths}"
        )
        self.recurrent_state = ttnn.zeros(
            [batch_size, self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.conv_state = [
            ttnn.zeros(
                [batch_size, 1, self.cfg.conv_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            for _ in range(self.cfg.linear_conv_kernel_dim - 1)
        ]

    def reset_state(self):
        """Zero the DeltaNet state in place (addresses preserved for trace replay)."""
        if self.is_full_attention or self.recurrent_state is None:
            return
        ttnn.multiply(self.recurrent_state, 0.0, output_tensor=self.recurrent_state)
        for buf in self.conv_state:
            ttnn.multiply(buf, 0.0, output_tensor=buf)

    # ------------------------------------------------------------------ small helpers
    def _norm(self, x, weight):
        """Zero-centered RMSNorm — the ``+1`` is already folded into ``weight``."""
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.norm_eps)

    def max_gdn_prefill_batch(self) -> int:
        """Largest batch one ``chunk_gated_delta_rule`` launch can serve (``B*HV <= cores``)."""
        grid = self.device.compute_with_storage_grid_size()
        return max(1, (grid.x * grid.y) // self.cfg.linear_num_value_heads)

    # ------------------------------------------------------------------ full attention
    def _attention_prefill(self, x, page_table, chunk_start_idx):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before prefill")
        if page_table is None:
            raise ValueError("full_attention prefill requires a page_table")
        cfg = self.cfg
        b, t = x.shape[0], x.shape[1]
        q_width = cfg.n_heads * cfg.head_dim
        kv_width = 2 * cfg.n_kv_heads * cfg.head_dim

        fused = ttnn.linear(x, self.w["attn_in"], compute_kernel_config=self.compute_kernel_config)
        fused = ttnn.reshape(fused, [b, 1, t, int(fused.shape[-1])])
        q_flat = _slice_last(fused, 0, q_width)
        kv_flat = _slice_last(fused, q_width, q_width + kv_width)
        gate = _slice_last(fused, q_width + kv_width, int(fused.shape[-1]))
        ttnn.deallocate(fused)
        gate = ttnn.reshape(gate, [b, t, q_width])

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q_flat, kv_flat, num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads, transpose_k_heads=False
        )
        ttnn.deallocate(q_flat)
        ttnn.deallocate(kv_flat)

        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        cos, sin = self.rope.prefill_forward(chunk_start_idx, t)
        q = self._rope_prefill(q, cos, sin)
        k = self._rope_prefill(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        blk0 = chunk_start_idx // self.page_block_size
        blk_n = _align_up(chunk_start_idx + t, self.page_block_size) // self.page_block_size
        chunk_page_table, pt_owned = _slice_owned(page_table, [0, blk0], [int(page_table.shape[0]), blk_n])
        # One batched fill per cache: `batch_idx_tensor` writes row u of the input into
        # page_table[batch_idxs[u]], so a batch-32 prefill costs 2 launches, not 64.
        batch_idxs, idx_owned = _slice_owned(self.batch_idxs, [0], [b])
        ttnn.experimental.paged_fill_cache(self.k_cache, k, chunk_page_table, batch_idx_tensor=batch_idxs)
        ttnn.experimental.paged_fill_cache(self.v_cache, v, chunk_page_table, batch_idx_tensor=batch_idxs)
        if idx_owned:
            ttnn.deallocate(batch_idxs)
        if pt_owned:
            ttnn.deallocate(chunk_page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            page_table,
            chunk_start_idx,
            scale=cfg.head_dim**-0.5,
            program_config=self._prefill_sdpa_config(chunk_start_idx, t),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q)
        merged = ttnn.experimental.nlp_concat_heads(attn)
        ttnn.deallocate(attn)
        merged = ttnn.reshape(merged, [b, t, q_width])
        return self._attention_output(merged, gate)

    def _prefill_sdpa_config(self, chunk_start_idx: int, phys_len: int):
        """Chunked-SDPA tiling. ``q_chunk`` must divide ``chunk_start_idx`` when it is non-zero."""
        qk = 64
        if chunk_start_idx:
            qk = min(qk, chunk_start_idx & -chunk_start_idx)
        qk = min(qk, phys_len)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=qk,
            k_chunk_size=qk,
            exp_approx_mode=False,
        )

    def _attention_output(self, attn, gate):
        """Sigmoid output gate folded into the multiply, then the output projection."""
        gated = ttnn.multiply(attn, gate, input_tensor_b_activations=_SIGMOID)
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        out = ttnn.linear(gated, self.w["o_proj"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return out

    def _kv_update_memory_configs(self, batch_size):
        """``(k_cfg, v_cfg, fused)`` height-sharded L1 configs for the paged cache update.

        One user per core, as a row-wise core *range set* of exactly ``batch_size`` cores rather
        than a rectangular ``CoreGrid`` — a rectangle would restrict the batch to values with a
        factor pair fitting the grid in both axes (batch 13 has none on an 11×10 grid).

        ``paged_fused_update_cache`` writes both caches in one launch but requires the two inputs
        to live on **disjoint** cores, so K takes the first ``batch_size`` cores and V the next
        ``batch_size``. That needs ``2 * batch_size`` cores; above that the caller falls back to two
        separate ``paged_update_cache`` launches (``fused=False``), which is what the functional
        decoder always did.
        """
        grid = self.device.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        if batch_size > cores:
            raise ValueError(
                f"decode batch {batch_size} exceeds the {cores} cores available for the "
                f"one-user-per-core paged_update_cache shard on a {grid.x}x{grid.y} grid"
            )

        def config(shard_grid):
            spec = ttnn.ShardSpec(shard_grid, [TILE, self.cfg.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
            return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)

        k_grid = ttnn.num_cores_to_corerangeset(batch_size, grid, row_wise=True)
        if 2 * batch_size > cores:
            return config(k_grid), config(k_grid), False
        whole = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        v_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(batch_size % grid.x, batch_size // grid.x), batch_size, whole, True
        )
        return config(k_grid), config(v_grid), True

    def _rope_prefill(self, x, cos, sin):
        """Rotate ``[batch, heads, seq, head_dim]`` with cos/sin ``[1, 1, seq, width]``."""
        if self.rope.mode == "full":
            out = ttnn.experimental.rotary_embedding_hf(
                x, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(x)
            return out
        rope_dim, head_dim = self.cfg.rope_dim, self.cfg.head_dim
        # `rope_dim == head_dim` (partial_rotary_factor 1.0) means there is nothing to pass through,
        # and the slice would alias `x` — the functional decoder guards the same way. Ornith is 0.25,
        # so this is about not narrowing the supported range, not about a shape reached here.
        source = _slice_last(x, 0, rope_dim) if rope_dim < head_dim else x
        rotated = ttnn.experimental.rotary_embedding_hf(
            source, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
        )
        if rope_dim >= head_dim:
            ttnn.deallocate(x)
            return rotated
        ttnn.deallocate(source)
        passthrough = _slice_last(x, rope_dim, head_dim)
        ttnn.deallocate(x)
        out = ttnn.concat([rotated, passthrough], dim=-1)
        ttnn.deallocate(rotated)
        ttnn.deallocate(passthrough)
        return out

    def _rope_decode(self, x, cos, sin):
        """Rotate ``[1, batch, heads, head_dim]`` with per-user cos/sin ``[1, 1, batch, width]``.

        ``rotary_embedding_hf``'s native decode mode requires height-sharded input *and* caches;
        transposing the batch and head axes lets the interleaved prefill kernel do the same work
        with no reshard, which is how ``models/demos/blackhole/qwen36`` drives it too.
        """
        rope_dim, head_dim = self.cfg.rope_dim, self.cfg.head_dim
        # As in _rope_prefill: a full-width rotation has no pass-through half to re-concatenate,
        # whether that is because the mode is "full" or because rope_dim already covers head_dim.
        partial = self.rope.mode == "partial" and rope_dim < head_dim
        source = _slice_last(x, 0, rope_dim) if partial else x
        swapped = ttnn.transpose(source, 1, 2)
        if partial:
            ttnn.deallocate(source)
        roped = ttnn.experimental.rotary_embedding_hf(
            swapped, cos, sin, compute_kernel_config=self.rope_compute_kernel_config
        )
        ttnn.deallocate(swapped)
        out = ttnn.transpose(roped, 1, 2)
        ttnn.deallocate(roped)
        if partial:
            passthrough = _slice_last(x, rope_dim, head_dim)
            ttnn.deallocate(x)
            merged = ttnn.concat([out, passthrough], dim=-1)
            ttnn.deallocate(out)
            ttnn.deallocate(passthrough)
            return merged
        ttnn.deallocate(x)
        return out

    #: Largest decode batch ``ttnn.experimental.nlp_create_qkv_heads_decode`` accepts
    #: (``nlp_create_qkv_heads_decode_device_operation.cpp``: ``num_users <= 32``). Above it the
    #: layer falls back to the functional decoder's generic split, so the supported batch is
    #: unchanged — only the op used to reach it is.
    DECODE_HEAD_SPLIT_MAX_BATCH = 32

    def _decode_qkv_heads(self, qkv, batch):
        """Split the packed ``[1, 1, batch, (n_heads + 2*n_kv) * head_dim]`` into decode-layout heads.

        Returns ``q`` ``[1, batch, n_heads, head_dim]`` and ``k``/``v``
        ``[1, batch, n_kv_heads, head_dim]``, DRAM-interleaved.

        The dedicated op is used whenever it can be — it does the whole split in one launch — and
        the functional decoder's generic slice/reshape/permute is kept as the fallback for batches
        past the op's hard ``num_users <= 32`` limit, so this stage narrows nothing.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        if batch <= self.DECODE_HEAD_SPLIT_MAX_BATCH:
            heads = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=n_heads, num_kv_heads=n_kv)
            return tuple(ttnn.sharded_to_interleaved(t, ttnn.DRAM_MEMORY_CONFIG) for t in heads)

        width = int(qkv.shape[-1])
        flat = ttnn.reshape(qkv, [batch, 1, width])
        out = []
        start = 0
        for heads in (n_heads, n_kv, n_kv):
            part = _slice_last(flat, start, start + heads * head_dim)
            start += heads * head_dim
            part = ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            part = ttnn.reshape(part, [batch, 1, heads, head_dim])
            part = ttnn.to_layout(part, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # [batch, 1, heads, head_dim] -> [1, batch, heads, head_dim]: a relabel, seq extent is 1.
            out.append(ttnn.reshape(part, [1, batch, heads, head_dim]))
        ttnn.deallocate(flat)
        return tuple(out)

    def _attention_decode(self, x, current_pos, rot_idxs, page_table):
        if self.k_cache is None:
            raise RuntimeError("call allocate_kv_cache()/attach_kv_cache() before decode")
        if page_table is None:
            raise ValueError("full_attention decode requires a page_table")
        cfg = self.cfg
        b = x.shape[0]
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q_width = n_heads * head_dim
        kv_width = 2 * n_kv * head_dim

        fused = ttnn.linear(x, self.w["attn_in"], compute_kernel_config=self.compute_kernel_config)
        width = int(fused.shape[-1])
        # [b, 1, width] -> [1, 1, b, width]. Not a metadata view — ttnn.reshape only returns one when
        # the last dim matches and the second-to-last dims are equal or both tile multiples, and 1 is
        # neither — so this dispatches a tiled reshape that moves b from the batch axis into the tile
        # height. It is one op either way, and the alternative (per-row slicing) is more.
        fused = ttnn.reshape(fused, [1, 1, b, width])
        qkv = _slice_last(fused, 0, q_width + kv_width)
        gate = _slice_last(fused, q_width + kv_width, width)
        ttnn.deallocate(fused)
        gate = ttnn.reshape(gate, [b, 1, q_width])

        q, k, v = self._decode_qkv_heads(qkv, b)
        ttnn.deallocate(qkv)

        q = self._norm(q, self.w["q_norm"])
        k = self._norm(k, self.w["k_norm"])
        cos, sin = self.rope.decode_forward(rot_idxs)
        q = self._rope_decode(q, cos, sin)
        k = self._rope_decode(k, cos, sin)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # paged_fused_update_cache writes both caches in one launch. Both inputs must be
        # height-sharded [1, B, kv_heads padded to 32, head_dim].
        k_cfg, v_cfg, fused_update = self._kv_update_memory_configs(b)
        k_upd = ttnn.to_memory_config(_pad_dim(k, 2, TILE - n_kv), k_cfg)
        v_upd = ttnn.to_memory_config(_pad_dim(v, 2, TILE - n_kv), v_cfg)
        if fused_update:
            ttnn.experimental.paged_fused_update_cache(
                self.k_cache, k_upd, self.v_cache, v_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                self.k_cache, k_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                self.v_cache, v_upd, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_upd)
        ttnn.deallocate(v_upd)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            is_causal=True,
            scale=head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                q_chunk_size=32,
                k_chunk_size=64,
                exp_approx_mode=False,
            ),
        )
        ttnn.deallocate(q)
        # [1, B, n_heads, D] -> [B, 1, n_heads*D]: memory order is already (B, head, dim).
        attn = ttnn.reshape(attn, [b, 1, n_heads * head_dim])
        return self._attention_output(attn, gate)

    # ------------------------------------------------------------------ gated deltanet
    def _gdn_project(self, x):
        """One packed in-projection, sliced into ``(qkv, z, a, b)``."""
        cfg = self.cfg
        nv = cfg.linear_num_value_heads
        fused = ttnn.linear(x, self.w["gdn_in"], compute_kernel_config=self.compute_kernel_config)
        qkv_end = cfg.conv_dim
        z_end = qkv_end + cfg.linear_v_dim
        a_end = z_end + nv
        qkv = _slice_last(fused, 0, qkv_end)
        z = _slice_last(fused, qkv_end, z_end)
        a = _slice_last(fused, z_end, a_end)
        b = _slice_last(fused, a_end, a_end + nv)
        ttnn.deallocate(fused)
        return qkv, z, a, b

    def _gdn_gates(self, a, b_raw, logical_len, seq_len):
        """``(beta, g)`` as float32 ``[B, T, num_v_heads]``, with the padded tail neutralised.

        ``beta = sigmoid(in_proj_b(x))`` and ``g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)``
        (float32). ``softplus`` is folded into the bias add. Zeroing both past ``logical_len``
        makes every padded step an exact identity on the recurrent state.
        """
        beta = ttnn.typecast(ttnn.sigmoid(b_raw), ttnn.float32)
        a32 = ttnn.typecast(a, ttnn.float32)
        softplus = ttnn.add(a32, self.w["dt_bias"], activations=_SOFTPLUS)
        ttnn.deallocate(a32)
        g = ttnn.multiply(self.w["A_neg"], softplus)
        ttnn.deallocate(softplus)

        if logical_len < seq_len:
            # _slice_owned, not ttnn.slice: pos_ramp is [1, prefill_chunk, 1], so at
            # seq_len == prefill_chunk this is a whole-tensor slice, which ttnn.slice short-circuits
            # to the input itself (slice.cpp's `no_step && starts_zero && ends_max` no-op check).
            # Deallocating that would free the layer's persistent ramp weight and fatal the *next*
            # masked prefill. Reachable whenever a chunk's logical length lands in
            # (chunk_size - PREFILL_ALIGN, chunk_size), e.g. seq_len=2000 at the shipped 2048.
            ramp, ramp_owned = _slice_owned(self.w["pos_ramp"], [0, 0, 0], [1, seq_len, 1])
            keep = ttnn.typecast(ttnn.lt(ramp, float(logical_len)), ttnn.float32)
            if ramp_owned:
                ttnn.deallocate(ramp)
            beta_masked = ttnn.multiply(beta, keep)
            g_masked = ttnn.multiply(g, keep)
            ttnn.deallocate(keep)
            ttnn.deallocate(beta)
            ttnn.deallocate(g)
            beta, g = beta_masked, g_masked
        return beta, g

    def _conv1d_halves(self, padded_rm, phys_len, prepared):
        """Depthwise conv1d over the padded stream, one call per :data:`CONV1D_CHANNELS` block.

        Returns the SiLU-activated halves as a **list**, one per :data:`CONV1D_CHANNELS` block,
        *not* a concatenated stream: at 4096 channels the split point is exactly the Q/K | V
        boundary, so ``_gdn_prefill`` consumes ``[q|k]`` and ``[v]`` directly and neither a concat
        nor a 4096-wide re-slice is needed. The SiLU is applied separately rather than through
        ``Conv2dConfig(activation=...)``: folding it is faster but not correct on this depthwise
        conv. Measured at Ornith's own shapes in ``doc/fused_decoder/logs/probe_conv1d_and_norm.txt``
        (``CONV1DACT`` rows) and recorded in ``work_log.md`` §4.15;
        ``models/demos/blackhole/qwen36/tt/gdn/tp.py:367`` reports the same for its conv.
        """
        rm = ttnn.DRAM_MEMORY_CONFIG
        channels = CONV1D_CHANNELS
        batch = int(padded_rm.shape[0])
        length = int(padded_rm.shape[1])
        cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        outs = []
        for idx, weight in enumerate(prepared):
            part = _slice_last(padded_rm, idx * channels, (idx + 1) * channels)
            part = ttnn.reshape(part, [batch, length, 1, channels])
            out = ttnn.conv1d(
                input_tensor=part,
                weight_tensor=weight,
                device=self.device,
                in_channels=channels,
                out_channels=channels,
                batch_size=batch,
                input_length=length,
                kernel_size=self.cfg.linear_conv_kernel_dim,
                stride=1,
                padding=0,
                dilation=1,
                groups=channels,
                dtype=ttnn.bfloat16,
                conv_config=cfg,
                compute_config=self.conv_compute_kernel_config,
                # L1-full slicing: the DRAM-slice path does host reads, which a later traced caller
                # could not use. Prefill is not traced today, but the constraint costs nothing here.
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )
            ttnn.deallocate(part)
            out = ttnn.sharded_to_interleaved(out, rm)
            out = ttnn.to_layout(ttnn.reshape(out, [batch, phys_len, channels]), ttnn.TILE_LAYOUT, memory_config=rm)
            activated = ttnn.silu(out, memory_config=rm)
            ttnn.deallocate(out)
            outs.append(activated)

        return outs

    def _conv1d_weights(self, phys_len):
        """Prepared ``ttnn.conv1d`` weights for a block of ``phys_len`` tokens, or ``None``.

        ``ttnn.prepare_conv_weights`` is a host call whose output layout depends on the input
        length, so the weights for every physical block length the layer can produce are prepared at
        load time (``from_state_dict``) and looked up here. A caller that asks for a ``chunk_size``
        the layer was not built for simply gets ``None`` and the FIR path, rather than a host call
        inside a measured forward pass.
        """
        return self.w.get("conv1d_weights", {}).get(int(phys_len))

    def _causal_conv_prefill(self, qkv, logical_len):
        """Depthwise causal conv1d + SiLU over the fused QKV stream.

        Returns ``(activated, tail)``. ``activated`` is the whole ``[batch, phys_len, conv_dim]``
        stream when the FIR path runs, or the list of :data:`CONV1D_CHANNELS`-wide halves when
        ``ttnn.conv1d`` does; :meth:`_gdn_split_conv_output` consumes either. ``tail`` is the
        ``kernel-1`` real inputs ending at ``logical_len`` — the conv history the next block or the
        next decode step starts from.

        The FIR fallback takes its shifted taps from one ROW_MAJOR concatenation rather than
        untilizing the whole stream per tap. It accumulates with :func:`ttnn.addcmul`, as the
        functional decoder did: this stage briefly used :func:`ttnn.mac` instead, on the belief that
        ``addcmul`` was a three-op composite. It is not — ``ternary.cpp`` dispatches a single
        ``prim::ternary(ADDCMUL)`` LLK whenever the broadcast is valid and no input is block-float,
        which holds for both of these call sites, while ``mac`` is unconditionally
        ``add(multiply(...))``. The swap cost one extra device op per tap on the *decode* conv, which
        is the traced path, so it was reverted. ``work_log.md`` §4.14.
        """
        t = int(qkv.shape[1])
        kernel = self.cfg.linear_conv_kernel_dim
        # Concatenate in ROW_MAJOR: `ttnn.concat` on the token axis of TILE tensors lowers to
        # untilize -> concat -> tilize, and that tilize of the whole 8192-wide stream would be thrown
        # away immediately by the shifted-window slicing below.
        rm = ttnn.DRAM_MEMORY_CONFIG
        pieces = [ttnn.to_layout(buf, ttnn.ROW_MAJOR_LAYOUT, memory_config=rm) for buf in self.conv_state]
        qkv_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=rm)
        padded_rm = ttnn.concat(pieces + [qkv_rm], dim=1)
        for piece in pieces:
            ttnn.deallocate(piece)
        ttnn.deallocate(qkv_rm)

        prepared = self._conv1d_weights(t)
        if prepared is not None:
            # A list of CONV1D_CHANNELS-wide halves; _gdn_prefill splits Q/K/V out of them without
            # ever materialising the concatenated stream.
            activated = self._conv1d_halves(padded_rm, t, prepared)
        else:
            acc = None
            for tap in range(kernel):
                if tap == kernel - 1:
                    # The last tap's window is exactly the (already TILE) input: no slice, no tilize.
                    piece, owned = qkv, False
                else:
                    piece, owned = (
                        ttnn.to_layout(padded_rm[:, tap : tap + t, :], ttnn.TILE_LAYOUT, memory_config=rm),
                        True,
                    )
                if acc is None:
                    acc = ttnn.multiply(piece, self.w["conv_taps"][tap], memory_config=rm)
                else:
                    acc = ttnn.addcmul(acc, piece, self.w["conv_taps"][tap], memory_config=rm)
                if owned:
                    ttnn.deallocate(piece)
            activated = ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(acc)

        # The tail stays ROW_MAJOR: _write_conv_state wants one row at a time and rows 0..kernel-2
        # are not tile-aligned, so tilizing the whole tail here would only be undone immediately.
        # The alternative — tilize the whole tail once here and row-slice it in TILE — is timed
        # against this one on the real decoder at the 2048-token prefill shape by
        # doc/fused_decoder/logs/probe_conv_tail.py, which monkeypatches that variant in. The two
        # produce identical output (`torch.equal` in the probe); this one additionally dispatches one
        # layout conversion fewer, which
        # test_no_layout_churn_in_measured_forward pins at 256 and 2048 tokens. The slice is a
        # strict sub-range, so it owns a fresh buffer and `padded_rm` can be freed.
        tail = padded_rm[:, logical_len : logical_len + kernel - 1, :]
        ttnn.deallocate(padded_rm)
        return activated, tail

    def _gdn_split_conv_output(self, activated):
        """Split the causal-conv output into ``(q, k, v)``, consuming ``activated``.

        ``_causal_conv_prefill`` hands back either one ``[B, T, conv_dim]`` stream (the FIR
        fallback) or the list of :data:`CONV1D_CHANNELS`-wide blocks ``ttnn.conv1d`` produced. In the
        list case the blocks are *not* concatenated first: a block that lies entirely inside one of
        q/k/v is handed over as-is, so on Ornith — ``conv_dim`` 8192 split 2048/2048/4096, blocks
        4096 wide — v is exactly block 1 and needs no slice at all. That removes both the
        8192-wide concat of the conv output and the 4096-wide v slice the concatenated spelling
        needed; work_log.md §4.4 has the ConcatDeviceOperation row it cost. The block/field boundaries are only guaranteed to line up that neatly for this
        config, so the general case is still handled: a field spanning two blocks is concatenated
        from its pieces.
        """
        cfg = self.cfg
        bounds = [0, cfg.linear_q_dim, cfg.linear_q_dim + cfg.linear_k_dim, cfg.conv_dim]
        if not isinstance(activated, list):
            fields = [_slice_last(activated, bounds[i], bounds[i + 1]) for i in range(3)]
            ttnn.deallocate(activated)
            return fields

        channels = CONV1D_CHANNELS
        consumed = [False] * len(activated)
        fields = []
        for i in range(3):
            lo, hi = bounds[i], bounds[i + 1]
            pieces = []
            for blk in range(lo // channels, (hi - 1) // channels + 1):
                base = blk * channels
                start, end = max(lo, base) - base, min(hi, base + channels) - base
                if (start, end) == (0, channels):
                    pieces.append(activated[blk])
                    consumed[blk] = True
                else:
                    pieces.append(_slice_last(activated[blk], start, end))
            if len(pieces) == 1:
                fields.append(pieces[0])
            else:
                fields.append(ttnn.concat(pieces, dim=-1))
                # Every piece is now a copy inside the concat output, including any whole block
                # (already marked consumed, so the sweep below will not double-free it).
                for piece in pieces:
                    ttnn.deallocate(piece)
        for blk, block in enumerate(activated):
            if not consumed[blk]:
                ttnn.deallocate(block)
        return fields

    def _write_conv_state(self, tail_rm):
        """Copy the new conv history into the persistent buffers, preserving addresses.

        ``tail_rm`` is ROW_MAJOR ``[batch, kernel-1, conv_dim]``; each row is tilized on its own.
        """
        for idx, buf in enumerate(self.conv_state):
            row = ttnn.to_layout(tail_rm[:, idx : idx + 1, :], ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(row, buf)
            ttnn.deallocate(row)

    def _chunk_delta_rule(self, q, k, v, g, beta):
        """Chunk-parallel gated delta rule over a whole block, sub-batched if needed.

        ``q``/``k``/``v`` are **flat** rank-3 token-major tensors. That contract lets the op's prep
        kernel L2-normalise Q/K over the head dim and fold the ``K**-0.5`` scale into that norm, so
        the three head-split relayouts, the two explicit device-side L2 norms and the scale multiply all
        disappear into the op's prep kernel.
        It requires ``chunk_size == 32`` and ``T % 32 == 0``, both guaranteed by the 128-token
        physical prefill alignment.

        Two couplings the rank-4 spelling did not have, both documented in
        ``doc/fused_decoder/README.md`` §8:

        * the op infers the key head dim **from the value head dim** on this path
          (``chunk_gated_delta_rule.cpp``: ``K = flat_qk ? V : qs[3]``), so an unequal pair is guarded
          below rather than silently miscomputed;
        * the flat path is only legal on the op's phased branch, which that file selects from the
          ``QWEN_GDN_PHASED`` environment variable. With it set to ``0`` this call raises where the
          functional decoder's rank-4 call still runs.
        """
        # See the docstring: guarded rather than assumed. _gdn_decode raises the same way.
        dk, dv = self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        if dk != dv:
            raise ValueError(
                "the flat rank-3 chunk_gated_delta_rule contract infers the key head dim from the "
                f"value head dim, which assumes linear_key_head_dim == linear_value_head_dim; got "
                f"{dk} != {dv}. Use the functional decoder's rank-4 call for such a config."
            )
        cfg = self.cfg
        eye, tril, ones, masks = self.w["gdn_const_tiles"]
        batch, seq = q.shape[0], q.shape[1]
        nv = cfg.linear_num_value_heads

        def launch(q_, k_, v_, g_, beta_, state_):
            return ttnn.transformer.chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                beta_,
                initial_state=state_,
                output_final_state=True,
                chunk_size=GDN_CHUNK,
                use_qk_l2norm=False,
                output_head_major=True,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )

        step = self.max_gdn_prefill_batch()
        if batch <= step:
            return launch(q, k, v, g, beta, self.recurrent_state)

        cores, states = [], []
        for start in range(0, batch, step):
            end = min(start + step, batch)
            q_s = ttnn.slice(q, [start, 0, 0], [end, seq, cfg.linear_q_dim])
            k_s = ttnn.slice(k, [start, 0, 0], [end, seq, cfg.linear_k_dim])
            v_s = ttnn.slice(v, [start, 0, 0], [end, seq, cfg.linear_v_dim])
            g_s = ttnn.slice(g, [start, 0, 0], [end, seq, nv])
            beta_s = ttnn.slice(beta, [start, 0, 0], [end, seq, nv])
            state_s = ttnn.slice(
                self.recurrent_state, [start, 0, 0, 0], [end, nv, cfg.linear_key_head_dim, cfg.linear_value_head_dim]
            )
            core_s, final_s = launch(q_s, k_s, v_s, g_s, beta_s, state_s)
            for tensor in (q_s, k_s, v_s, g_s, beta_s, state_s):
                ttnn.deallocate(tensor)
            cores.append(core_s)
            states.append(final_s)
        # head-major output is [B*HV, T, V]; concatenating on dim 0 restitches the batch.
        core = ttnn.concat(cores, dim=0)
        final_state = ttnn.concat(states, dim=0)
        for tensor in cores + states:
            ttnn.deallocate(tensor)
        return core, final_state

    def _gdn_out(self, core_head_major, z, batch, seq):
        """Gated output norm + out projection.

        ``core_head_major``: ``[B*num_v_heads, T, head_v_dim]`` (prefill) — the op's head-major
        output — or ``[B, num_v_heads, 1, head_v_dim]`` (decode). ``ttnn.rms_norm`` normalises over
        the last dim either way, and ``nlp_concat_heads`` does the head→token relayout in one op.
        """
        cfg = self.cfg
        nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        normed = ttnn.rms_norm(core_head_major, weight=self.w["gdn_norm"], epsilon=cfg.norm_eps)
        ttnn.deallocate(core_head_major)
        normed = ttnn.reshape(normed, [batch, nv, seq, dv])
        if seq > 1:
            merged = ttnn.experimental.nlp_concat_heads(normed)
            ttnn.deallocate(normed)
            merged = ttnn.reshape(merged, [batch, seq, nv * dv])
        else:
            # All three spellings of this head->token relayout were measured from a captured trace
            # (doc/fused_decoder/logs/probe_decode_micro.txt). nlp_concat_heads wins by more than an
            # order of magnitude for a prefill block but collapses onto one core at seq 1, where the
            # plain permute is several times cheaper. All three agree exactly at seq 1 (`torch.equal`); above it
            # only nlp_concat_heads and permute+reshape are equivalent (the flat relayout does not
            # transpose head<->token), which the same probe records.
            swapped = ttnn.permute(normed, (0, 2, 1, 3))
            ttnn.deallocate(normed)
            merged = ttnn.reshape(swapped, [batch, seq, nv * dv])
            ttnn.deallocate(swapped)
        # Deliberately unfused, and this is the one place in the graph where an op-level A/B is not
        # sufficient evidence. `models/demos/blackhole/qwen36/tt/gdn/tp.py:31-34` reports that
        # folding the SiLU here breaks "in the real layer for large-magnitude z (op-level PCC hid
        # it - small inputs)". Measured at this gate's own shape, the fold looks *good*: the two
        # arms agree to PCC 0.999996 with zero non-finite outputs at every |z| up to ~663, and the
        # folded form is measurably faster (GATEFOLD / GATEFOLDTIME rows in
        # doc/fused_decoder/logs/probe_fused_ops.txt). Landing it on that evidence collapses
        # fused-vs-functional agreement to essentially zero on real checkpoint weights - the
        # isolated probe passes and the model is destroyed. So the rejection stands, now on this
        # stage's own controls rather than on the citation, and work_log.md §4.8 records both: the
        # op-level A/B that would have justified the merge, and the real-weight control that
        # refutes it.
        gated = ttnn.multiply(merged, ttnn.silu(z))
        ttnn.deallocate(merged)
        ttnn.deallocate(z)
        out = ttnn.linear(gated, self.w["gdn_out"], compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(gated)
        return out

    def _gdn_prefill(self, x, logical_len):
        b, t = x.shape[0], x.shape[1]
        qkv, z, a, b_raw = self._gdn_project(x)
        activated, tail = self._causal_conv_prefill(qkv, logical_len)
        ttnn.deallocate(qkv)

        q, k, v = self._gdn_split_conv_output(activated)

        beta, g = self._gdn_gates(a, b_raw, logical_len, t)
        ttnn.deallocate(a)
        ttnn.deallocate(b_raw)

        core, final_state = self._chunk_delta_rule(q, k, v, g, beta)
        for tensor in (q, k, v, g, beta):
            ttnn.deallocate(tensor)
        ttnn.copy(final_state, self.recurrent_state)
        ttnn.deallocate(final_state)
        self._write_conv_state(tail)
        ttnn.deallocate(tail)
        return self._gdn_out(core, z, b, t)

    def _gdn_decode(self, x):
        """One recurrent gated-delta-rule step, updating conv + recurrent state in place."""
        cfg = self.cfg
        b = x.shape[0]
        nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        kernel = cfg.linear_conv_kernel_dim

        qkv, z, a, b_raw = self._gdn_project(x)

        acc = ttnn.multiply(qkv, self.w["conv_taps"][kernel - 1])
        for tap in range(kernel - 1):
            acc = ttnn.addcmul(acc, self.conv_state[tap], self.w["conv_taps"][tap])
        activated = ttnn.silu(acc)
        ttnn.deallocate(acc)
        # Shift the history: oldest out, this step's pre-activation input in. Read-before-write
        # order keeps the in-place chain correct and the buffer addresses stable.
        for idx in range(kernel - 2):
            ttnn.copy(self.conv_state[idx + 1], self.conv_state[idx])
        ttnn.copy(qkv, self.conv_state[kernel - 2])
        ttnn.deallocate(qkv)

        # One relayout for the whole conv output into head-major [B, heads, 1, dim] — the shape both
        # the recurrent step and the output norm want. It is a reshape + permute rather than an
        # explicit ROW_MAJOR round trip; the reshape splits the last dim, so ttnn dispatches a tiled
        # reshape kernel rather than returning a view, but that is one device op against the
        # functional decoder's untilize/reshape/tilize. It is not a layout
        # conversion. Slicing heads off dim 1 is then a cheap batch-dim slice, where the functional
        # decoder needed three separate untilize/reshape/tilize round trips (one per Q/K/V).
        if dk != dv:
            raise ValueError(
                f"the decode head split folds Q/K/V into one {2 * nk + nv}-head relayout, which "
                f"assumes linear_key_head_dim == linear_value_head_dim; got {dk} != {dv}"
            )
        rows = ttnn.reshape(activated, [b, 1, 2 * nk + nv, dk])
        ttnn.deallocate(activated)
        heads = ttnn.permute(rows, (0, 2, 1, 3))
        ttnn.deallocate(rows)
        v = ttnn.slice(heads, [0, 2 * nk, 0, 0], [b, 2 * nk + nv, 1, dv])

        # Q and K are adjacent on the head axis, so one repeat_interleave over the pair does both
        # GQA expansions: [q0..q15, k0..k15] -> [q0,q0,...,q15,q15, k0,k0,...,k15,k15]. That halves
        # the untilize/concat/tilize this op lowers to.
        repeats = nv // nk
        qk = ttnn.slice(heads, [0, 0, 0, 0], [b, 2 * nk, 1, dk])
        ttnn.deallocate(heads)
        if repeats > 1:
            expanded = ttnn.repeat_interleave(qk, repeats, dim=1)
            ttnn.deallocate(qk)
            qk = expanded
        q = ttnn.slice(qk, [0, 0, 0, 0], [b, nv, 1, dk])
        k = ttnn.slice(qk, [0, nv, 0, 0], [b, 2 * nv, 1, dk])
        ttnn.deallocate(qk)

        beta, g = self._gdn_gates(a, b_raw, 1, 1)
        ttnn.deallocate(a)
        ttnn.deallocate(b_raw)
        core = self._delta_rule_step(q, k, v, beta, g)
        for tensor in (q, k, v, beta, g):
            ttnn.deallocate(tensor)
        return self._gdn_out(core, z, b, 1)

    def _delta_rule_step(self, q, k, v, beta, g):
        """Single gated-delta-rule step on the persistent recurrent state.

        ``q``/``k``: ``[B, num_v_heads, 1, head_k_dim]`` (already GVA-expanded), ``v``:
        ``[B, num_v_heads, 1, head_v_dim]``, ``beta``/``g``: ``[B, 1, num_v_heads]`` float32.

        Mirrors HF ``torch_recurrent_gated_delta_rule``: L2-normalise Q/K, scale Q by
        ``head_k_dim ** -0.5``, decay the state, read ``k @ h``, write the ``beta``-weighted delta
        outer product, then read ``q @ h``. Everything stays in DRAM float32 so the state is exact.

        The L2 norm is ``rms_norm(x, eps/K) * K**-0.5`` (the idiom from
        ``models/experimental/gated_attention_gated_deltanet``), and Q's ``K**-0.5`` scale is folded
        into that same multiply, so Q costs two ops rather than four.
        """
        cfg = self.cfg
        b = q.shape[0]
        nv, dk = cfg.linear_num_value_heads, cfg.linear_key_head_dim
        dram = ttnn.DRAM_MEMORY_CONFIG
        eps = 1e-6

        # HF's torch_recurrent_gated_delta_rule applies `scale = 1/sqrt(head_k_dim)` to q. The
        # layer PCC cannot discriminate it (Qwen3_5MoeRMSNormGated cancels any uniform per-(token,
        # head) factor), but the chunked prefill op applies the same scale internally, so dropping
        # it would make prefill and decode disagree on the intermediate `o`.
        q_n = ttnn.rms_norm(q, epsilon=eps / dk)
        q_row = ttnn.typecast(ttnn.multiply(q_n, dk**-1.0, memory_config=dram), ttnn.float32)
        ttnn.deallocate(q_n)
        k_n = ttnn.rms_norm(k, epsilon=eps / dk)
        k_row = ttnn.typecast(ttnn.multiply(k_n, dk**-0.5, memory_config=dram), ttnn.float32)
        ttnn.deallocate(k_n)

        v_row = ttnn.typecast(v, ttnn.float32)
        beta_b = ttnn.reshape(beta, [b, nv, 1, 1])
        decay = ttnn.exp(ttnn.reshape(g, [b, nv, 1, 1]), memory_config=dram)

        state = self.recurrent_state
        ttnn.multiply(state, decay, output_tensor=state)
        ttnn.deallocate(decay)

        v_read = ttnn.matmul(
            k_row,
            state,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.full_core_grid,
        )
        delta = ttnn.multiply(ttnn.subtract(v_row, v_read, memory_config=dram), beta_b, memory_config=dram)
        ttnn.deallocate(v_read)
        ttnn.deallocate(v_row)
        # `beta_b` is deliberately not freed here: it is a reshape of the caller's `beta`, and
        # ttnn.reshape may hand back a view, in which case freeing it would free a tensor the caller
        # frees again. Dropping the Python reference is enough.
        # transpose folded into the matmul: k_row^T @ delta is the rank-1 delta outer product.
        outer = ttnn.matmul(
            k_row,
            delta,
            transpose_a=True,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.full_core_grid,
        )
        ttnn.deallocate(k_row)
        ttnn.deallocate(delta)
        ttnn.add(state, outer, output_tensor=state)
        ttnn.deallocate(outer)

        out = ttnn.matmul(
            q_row,
            state,
            memory_config=dram,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.full_core_grid,
        )
        ttnn.deallocate(q_row)
        return out

    # ------------------------------------------------------------------ public forwards
    def _block(self, x, *, mode, logical_len=None, page_table=None, chunk_start_idx=0, current_pos=None, rot_idxs=None):
        """One decoder block: norm → mixer → residual → norm → MoE → residual."""
        b, t = x.shape[0], x.shape[1]
        attn_in = self._norm(x, self.w["attn_norm"])
        if self.is_full_attention:
            if mode == "prefill":
                mixed = self._attention_prefill(attn_in, page_table, chunk_start_idx)
            else:
                mixed = self._attention_decode(attn_in, current_pos, rot_idxs, page_table)
        else:
            if mode == "prefill":
                mixed = self._gdn_prefill(attn_in, logical_len)
            else:
                mixed = self._gdn_decode(attn_in)
        ttnn.deallocate(attn_in)

        h = ttnn.add(x, mixed)
        ttnn.deallocate(mixed)

        tokens = b * t
        padded_tokens = _align_up(tokens, TILE)
        ff_in = ttnn.reshape(self._norm(h, self.w["ff_norm"]), [1, 1, tokens, self.cfg.dim])
        if padded_tokens != tokens:
            # Do not free the pre-pad tensor: ttnn.pad may alias it.
            ff_in = _pad_dim(ff_in, 2, padded_tokens - tokens)
        ff_out = self.moe.forward(ff_in)
        ttnn.deallocate(ff_in)
        if padded_tokens != tokens:
            trimmed = ttnn.slice(ff_out, [0, 0, 0, 0], [1, 1, tokens, self.cfg.dim])
            ttnn.deallocate(ff_out)
            ff_out = trimmed
        ff_out = ttnn.reshape(ff_out, [b, t, self.cfg.dim])
        out = ttnn.add(h, ff_out)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out

    def prefill_forward(self, x, *, start_pos: int = 0, page_table=None, chunk_size: int | None = None):
        """See the module docstring for the full contract."""
        b, seq_len, dim = x.shape[0], x.shape[1], x.shape[2]
        if dim != self.cfg.dim:
            raise ValueError(f"hidden size {dim} != {self.cfg.dim}")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if start_pos + seq_len > self.max_context:
            raise ValueError(
                f"prefill window [{start_pos}, {start_pos + seq_len}) exceeds supported context {self.max_context}"
            )
        chunk_size = chunk_size or self.prefill_chunk
        if chunk_size % PREFILL_ALIGN:
            raise ValueError(f"chunk_size {chunk_size} must be a multiple of {PREFILL_ALIGN}")
        if start_pos % chunk_size:
            raise ValueError(f"start_pos {start_pos} must be a multiple of chunk_size {chunk_size}")
        if not self.is_full_attention and chunk_size > self.w["pos_ramp"].shape[1]:
            raise ValueError(
                f"chunk_size {chunk_size} exceeds the position ramp built for {self.w['pos_ramp'].shape[1]}"
            )
        # Same rule as the functional decoder: the DeltaNet state is per-row and cannot be resized
        # under a live sequence, so linear_attention pins the batch; full_attention has no such
        # state and may run any batch up to what its buffers were allocated for.
        if self.batch_size is None or (self.is_full_attention and b > self.batch_size):
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")

        outputs = []
        for offset in range(0, seq_len, chunk_size):
            logical = min(chunk_size, seq_len - offset)
            phys = min(chunk_size, _align_up(logical, PREFILL_ALIGN))
            block, owned = _slice_owned(x, [0, offset, 0], [b, offset + logical, dim])
            if phys > logical:
                # ttnn.pad may alias `block`, so ownership does not change.
                block = _pad_dim(block, 1, phys - logical)
            out = self._block(
                block,
                mode="prefill",
                logical_len=logical,
                page_table=page_table,
                chunk_start_idx=start_pos + offset,
            )
            if owned:
                ttnn.deallocate(block)
            if phys > logical:
                trimmed = ttnn.slice(out, [0, 0, 0], [b, logical, dim])
                ttnn.deallocate(out)
                out = trimmed
            outputs.append(out)

        if len(outputs) == 1:
            return outputs[0]
        merged = ttnn.concat(outputs, dim=1)
        for out in outputs:
            ttnn.deallocate(out)
        return merged

    def decode_forward(self, x, *, current_pos=None, rot_idxs=None, page_table=None):
        """See the module docstring for the full contract."""
        b, t, dim = x.shape[0], x.shape[1], x.shape[2]
        if t != 1:
            raise ValueError(f"decode expects seq_len 1, got {t}")
        if dim != self.cfg.dim:
            raise ValueError(f"hidden size {dim} != {self.cfg.dim}")
        if self.is_full_attention and (current_pos is None or rot_idxs is None):
            raise ValueError("full_attention decode requires current_pos and rot_idxs device tensors")
        if self.batch_size is None or (self.is_full_attention and b > self.batch_size):
            self.allocate_state(b)
        elif not self.is_full_attention and b != self.batch_size:
            raise ValueError(f"batch {b} != allocated state batch {self.batch_size}")
        return self._block(x, mode="decode", current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
