# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip TTNN decoder for hexgrad/Kokoro-82M (plbert / ALBERT encoder).

Single-chip baseline
--------------------
This module uses ``tt/optimized_decoder.py`` (stage 02) as its numerical and
op-topology baseline: packed-QKV + ``nlp_create_qkv_heads`` + fused
``scaled_dot_product_attention`` + ``nlp_concat_heads``, fused-gelu FF1, an
explicit 2D ``ffn_output`` program config, and the selected precision policy
(bf16 activations, BFP8 linear weights, HiFi2, fp32 dest-acc). The per-op math is
identical; only the *placement* across devices changes. ``PrecisionPolicy`` and
the helper op builders are imported directly from the optimized decoder so the
two stages cannot drift.

Target hardware / mesh
----------------------
4x Blackhole p300c (``ClusterType.P300_X2``), a physical 4-ring exposed as a
``(1, 4)`` mesh. Tensor-parallel factor ``TP = 4`` over the size-4 mesh axis
(FABRIC_1D / ``Topology.Linear``, 2 ethernet links). The design targets
single-user (batch-1) latency, which is Kokoro's real serving regime.

Parallelization scheme (see doc/multichip_decoder/README.md for the full table)
------------------------------------------------------------------------------
Kokoro's plbert is a *bidirectional, non-autoregressive* encoder: 12 weight-tied
``AlbertLayer``s, no KV cache, no causal mask, no MoE. The only op that mixes
tokens is SDPA; embedding, both LayerNorms, the residual adds and the whole FFN
are per-token. That shapes the plan:

* **Residual stream is sequence-sharded** ``[b, 1, S/TP, H]`` (Megatron
  sequence-parallel). Each device owns a contiguous 1/TP slice of the (padded)
  sequence. This is the layer input *and* output contract, so decoders stack
  with no boundary reshard.
* **Attention = head-parallel TP.** The packed QKV and the attention-output
  (``dense``/WO) weights are fractured across the mesh so each device owns
  ``num_heads/TP`` local heads. To attend the full sequence with only local
  heads, the layer ``all_gather``s the residual to full sequence, runs local-head
  QKV/SDPA/WO, then ``reduce_scatter``s the WO partial back to the
  sequence-shard (reduce over head-groups + scatter over sequence in one op).
* **FFN = sequence/data-parallel.** FF1/FF2 are per-token, so each device runs
  the *full* FFN on its own sequence slice with **replicated** FFN weights. Same
  FLOPs/device as intermediate-TP but **zero** extra collective (intermediate-TP
  would add an all-gather+reduce-scatter per layer for no compute win — see
  rejected alternatives in the README).
* **Embedding + norms** are per-token and run locally on the sequence-shard
  (replicated tables/weights, no collective).

Collectives per layer: exactly **1 all_gather + 1 reduce_scatter** (both bf16,
on the sequence dim). Everything else is local. All 4 devices do 1/TP of the
compute.

Row-parallel bias correctness: the WO ``dense`` bias is a row-parallel output, so
it is *excluded* from the (partial) WO matmul and added once to the
already-reduced, sequence-sharded result after ``reduce_scatter`` (adding it
inside the matmul would sum it TP times). The column-parallel QKV and FF1 biases
are naturally sharded and applied in-matmul. FF2 bias is applied locally (FFN is
not reduced).

Sequence padding: the public API accepts any logical length 1..512. Internally
the sequence is padded to a multiple of ``TP * TILE`` (= 128) so each device's
shard is tile-aligned; padded key positions are masked in SDPA and the output is
sliced back to the logical length at the model boundary. There is **no**
aligned-only public contract.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

# Reuse the optimized decoder's precision policy + op-builder helpers verbatim so
# the two stages cannot numerically drift.
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import TILE, PrecisionPolicy, _dtype, _fidelity, _round_up
from models.common.lightweightmodule import LightweightModule
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    default_topology,
    get_num_links,
    get_tt_ccl,
)


class MultichipDecoder(LightweightModule):
    """Tensor-parallel + sequence-parallel TTNN plbert (ALBERT) encoder on a 1xTP mesh.

    Public forward signatures match :class:`OptimizedDecoder`, except that the
    activations are sequence-sharded across the mesh (layer input/output
    contract). ``prepare_inputs`` builds the sharded device tensors; the standalone
    tests gather at the model boundary only.
    """

    def __init__(
        self, *, mesh_device, hf_config, weights, policy: Optional[PrecisionPolicy] = None, layer_idx: int = 0
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = hf_config
        self.layer_idx = layer_idx
        self.num_layers = int(hf_config.num_hidden_layers)
        self.num_heads = int(hf_config.num_attention_heads)
        self.hidden_size = int(hf_config.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads
        self.eps = float(hf_config.layer_norm_eps)
        self.max_position_embeddings = int(hf_config.max_position_embeddings)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.policy = policy or PrecisionPolicy()
        self.w = weights
        self.activation_dtype = _dtype(self.policy.activation)

        # Mesh / TP geometry. The size-4 axis is the last mesh axis.
        self.mesh_shape = tuple(mesh_device.shape)
        self.tp = mesh_device.get_num_devices()
        assert self.num_heads % self.tp == 0, f"heads {self.num_heads} not divisible by TP {self.tp}"
        self.local_heads = self.num_heads // self.tp
        self.local_hidden = self.local_heads * self.head_dim  # per-device attention width
        # sequence must be a multiple of this so each shard is tile-aligned
        self.seq_multiple = self.tp * TILE

        # CCL manager + topology (persistent semaphores -> trace-safe collectives).
        # Match the topology to the fabric the mesh was actually opened with: the
        # 4x p300c is a physical 4-ring, so FABRIC_1D_RING + Topology.Ring is the
        # fastest config at max context (measured 2.56 (Ring) vs 2.71 (Linear) ms
        # @T=512, final config; see doc/multichip_decoder/sweeps/topology.log).
        # Fall back to Linear when the mesh
        # was opened with the non-ring fabric so the two always agree.
        self.tt_ccl = get_tt_ccl(mesh_device)
        try:
            ring_fabric = ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING
        except Exception:
            ring_fabric = False
        self.ccl_topology = ttnn.Topology.Ring if ring_fabric else default_topology(mesh_device)
        self.ccl_num_links = get_num_links(mesh_device)

        self.matmul_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_fidelity(self.policy.matmul_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=self.policy.fp32_dest_acc,
            packer_l1_acc=True,
        )
        self.norm_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_fidelity(self.policy.sdpa_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=128,
            k_chunk_size=128,
        )
        cg = mesh_device.compute_with_storage_grid_size()
        self.mm_core_grid = ttnn.CoreGrid(y=min(8, cg.y), x=min(10, cg.x))
        self._ffn_out_pc_cache: dict = {}
        self._ffn_in_pc_cache: dict = {}
        self._intermediate_ktiles = int(hf_config.intermediate_size) // TILE
        self._intermediate_ntiles = int(hf_config.intermediate_size) // TILE
        self._hidden_ktiles = self.hidden_size // TILE
        self._hidden_ntiles = self.hidden_size // TILE
        self._traces: dict = {}

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx=0, mesh_device, policy: Optional[PrecisionPolicy] = None, **kwargs
    ):
        """Build the mesh-sharded decoder from a real HF ALBERT state dict.

        All host->device weight conversion + mesh sharding happens here. Attention
        (QKV/WO) weights are fractured across the mesh by head; FFN, embedding and
        norm weights are replicated.
        """
        policy = policy or PrecisionPolicy()
        sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}

        num_heads = int(hf_config.num_attention_heads)
        hidden = int(hf_config.hidden_size)
        head_dim = hidden // num_heads
        tp = mesh_device.get_num_devices()
        local_heads = num_heads // tp
        gw = local_heads * head_dim  # 192 for 3 heads
        mesh_shape = tuple(mesh_device.shape)
        shard_axis_dim = len(mesh_shape) - 1  # size-TP axis is the last mesh axis

        emb_dt = _dtype(policy.embedding)
        norm_dt = _dtype(policy.norm_weight)
        attn_dt = _dtype(policy.attn_weight)
        mlp_dt = _dtype(policy.mlp_weight)
        map_dt = _dtype(policy.map_weight)

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def shard_mapper(tensor_dim):
            # shard `tensor_dim` across the size-TP mesh axis (dims=(row_axis, col_axis))
            dims = [None, None]
            dims[shard_axis_dim] = tensor_dim
            return ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=mesh_shape)

        def to_tt(tensor, *, transpose=False, dtype=ttnn.bfloat16, mapper=replicate):
            t = tensor.t().contiguous() if transpose else tensor.contiguous()
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)

        p = "encoder.albert_layer_groups.0.albert_layers.0."

        # --- head-parallel QKV column layout ---------------------------------
        # Reorder the packed [Q|K|V] output columns so a contiguous shard-by-TP
        # gives device c exactly [Q_c | K_c | V_c] (its local `local_heads` of each
        # projection), which nlp_create_qkv_heads(num_heads=local_heads,
        # num_kv_heads=local_heads) slices into local heads.
        qw = torch.cat(
            [sd[p + "attention.query.weight"], sd[p + "attention.key.weight"], sd[p + "attention.value.weight"]], dim=0
        )  # [3H, H] (out,in)
        qb = torch.cat(
            [sd[p + "attention.query.bias"], sd[p + "attention.key.bias"], sd[p + "attention.value.bias"]], dim=0
        )  # [3H]
        col_order = []
        for c in range(tp):
            for base in (0, hidden, 2 * hidden):  # Q, K, V blocks
                col_order.extend(range(base + c * gw, base + c * gw + gw))
        col_order = torch.tensor(col_order, dtype=torch.long)
        qw = qw[col_order, :]  # reorder output rows (== columns after transpose)
        qb = qb[col_order]

        # WO (dense): row-parallel over the concat-heads input dim. Heads are
        # contiguous, so a contiguous shard-by-TP of the input dim matches device
        # c's local heads [c*gw : (c+1)*gw]. Stored transposed as [in=H, out=H];
        # shard the input (row / dim 0).
        weights = {
            # factorized embeddings + embed->hidden map: replicated (per-token, local)
            "word_emb": to_tt(sd["embeddings.word_embeddings.weight"], dtype=emb_dt),
            "pos_emb": to_tt(sd["embeddings.position_embeddings.weight"], dtype=emb_dt),
            "tt_emb": to_tt(sd["embeddings.token_type_embeddings.weight"], dtype=emb_dt),
            "emb_ln_w": to_tt(sd["embeddings.LayerNorm.weight"], dtype=norm_dt),
            "emb_ln_b": to_tt(sd["embeddings.LayerNorm.bias"], dtype=norm_dt),
            "map_w": to_tt(sd["encoder.embedding_hidden_mapping_in.weight"], transpose=True, dtype=map_dt),
            "map_b": to_tt(sd["encoder.embedding_hidden_mapping_in.bias"], dtype=ttnn.bfloat16),
            # attention: FRACTURED across the mesh
            "qkv_w": to_tt(qw, transpose=True, dtype=attn_dt, mapper=shard_mapper(1)),  # [H, 3*gw] per dev
            "qkv_b": to_tt(qb, dtype=ttnn.bfloat16, mapper=shard_mapper(0)),  # [3*gw] per dev
            "dense_w": to_tt(
                sd[p + "attention.dense.weight"], transpose=True, dtype=attn_dt, mapper=shard_mapper(0)
            ),  # [gw, H] per dev
            "dense_b": to_tt(sd[p + "attention.dense.bias"], dtype=ttnn.bfloat16),  # replicated, added post-RS
            "attn_ln_w": to_tt(sd[p + "attention.LayerNorm.weight"], dtype=norm_dt),
            "attn_ln_b": to_tt(sd[p + "attention.LayerNorm.bias"], dtype=norm_dt),
            # FFN: REPLICATED (per-token, sequence-parallel, no collective)
            "ffn_w": to_tt(sd[p + "ffn.weight"], transpose=True, dtype=mlp_dt),
            "ffn_b": to_tt(sd[p + "ffn.bias"], dtype=ttnn.bfloat16),
            "ffn_out_w": to_tt(sd[p + "ffn_output.weight"], transpose=True, dtype=mlp_dt),
            "ffn_out_b": to_tt(sd[p + "ffn_output.bias"], dtype=ttnn.bfloat16),
            "full_ln_w": to_tt(sd[p + "full_layer_layer_norm.weight"], dtype=norm_dt),
            "full_ln_b": to_tt(sd[p + "full_layer_layer_norm.bias"], dtype=norm_dt),
        }
        return cls(mesh_device=mesh_device, hf_config=hf_config, weights=weights, policy=policy, layer_idx=layer_idx)

    # -------------------------------------------------------------- input prep
    def prepare_inputs(
        self, input_ids: torch.Tensor, mesh_device=None, *, attention_mask: Optional[torch.Tensor] = None
    ):
        """Host-side input construction: sequence-shard ids/positions across the
        mesh, build the (replicated) SDPA mask.

        Accepts any logical sequence length. Pads to a multiple of ``TP*TILE`` so
        each device's shard is tile-aligned. Returns ``attention_mask=None`` only
        when the sequence is already ``TP*TILE``-aligned and fully valid (fast
        path), so max-context batch-1 pays no masking cost.
        """
        mesh_device = mesh_device or self.mesh_device
        assert input_ids.dim() == 2, "input_ids must be (batch, seq_len)"
        batch, seq_len = input_ids.shape
        padded = _round_up(seq_len, self.seq_multiple)

        ids = torch.zeros((batch, padded), dtype=torch.int32)
        ids[:, :seq_len] = input_ids.to(torch.int32)
        pos = torch.zeros((batch, padded), dtype=torch.int32)
        pos[:, :seq_len] = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        tok_type = torch.zeros((batch, padded), dtype=torch.int32)

        valid = torch.zeros((batch, padded), dtype=torch.float32)
        valid[:, :seq_len] = 1.0
        if attention_mask is not None:
            valid[:, :seq_len] *= attention_mask.to(torch.float32)
        need_mask = bool((valid == 0).any().item())

        shard_axis_dim = len(self.mesh_shape) - 1
        seq_dims = [None, None]
        seq_dims[shard_axis_dim] = 1  # shard the sequence (dim 1 of [b, seq])
        seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(seq_dims), mesh_shape=self.mesh_shape)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def dev(t, dtype, layout, mapper):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=mapper)

        mask_dev = None
        if need_mask:
            # Full-sequence additive mask (attention runs on gathered full seq),
            # broadcast over heads: (batch, 1, padded, padded), replicated.
            key_bias = (1.0 - valid) * -1.0e9
            additive = key_bias.view(batch, 1, 1, padded).expand(batch, 1, padded, padded).contiguous()
            mask_dev = dev(additive, ttnn.bfloat16, ttnn.TILE_LAYOUT, replicate)

        return {
            "input_ids": dev(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, seq_mapper),
            "position_ids": dev(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, seq_mapper),
            "token_type_ids": dev(tok_type, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, seq_mapper),
            "attention_mask": mask_dev,
            "seq_len": seq_len,
            "padded_seq_len": padded,
            "batch": batch,
        }

    def _ffn_out_program_config(self, m_tiles: int):
        pc = self._ffn_out_pc_cache.get(m_tiles)
        if pc is not None:
            return pc
        gy = max(g for g in range(1, 9) if m_tiles % g == 0)
        per_core_m = m_tiles // gy
        n_tiles = self._hidden_ntiles
        gx = 8
        per_core_n = (n_tiles + gx - 1) // gx
        in0_block_w = math.gcd(8, self._intermediate_ktiles)
        out_subblock_w = per_core_n if per_core_n <= 4 else 1
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        self._ffn_out_pc_cache[m_tiles] = pc
        return pc

    def _ffn_in_program_config(self, m_tiles: int):
        """Explicit 2D program config for the FF1 (up) matmul with fused gelu.

        On the multichip path FF1 runs on the *local* sequence shard, so M is 1/TP
        of the single-chip M (e.g. 4 tiles at T=512). The default core_grid
        heuristic picks ``in0_block_w=1`` and leaves this matmul SLOW at ~10% DRAM
        util (it is the single largest decode matmul, K=hidden N=intermediate). An
        explicit MatmulMultiCoreReuseMultiCast config with a K-dividing
        ``in0_block_w`` and grid rows chosen to divide M-tiles restores utilization
        (43->33 us at M=128) and stays valid at every tile-padded local length.
        See doc/multichip_decoder/README.md (matmul geometry).
        """
        pc = self._ffn_in_pc_cache.get(m_tiles)
        if pc is not None:
            return pc
        gy = max(g for g in range(1, 9) if m_tiles % g == 0)
        per_core_m = m_tiles // gy
        gx = 8
        per_core_n = (self._intermediate_ntiles + gx - 1) // gx  # 8 for N=64
        in0_block_w = math.gcd(8, self._hidden_ktiles)  # 8 divides 24 (hidden=768)
        out_subblock_w = 2 if per_core_n % 2 == 0 else 1
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
        )
        self._ffn_in_pc_cache[m_tiles] = pc
        return pc

    # -------------------------------------------------------------- collectives
    def _all_gather_seq(self, x):
        """Gather the sequence-sharded residual to full sequence (dim 2)."""
        return ttnn.experimental.all_gather_async(
            x,
            dim=2,
            persistent_output_buffer=None,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.ccl_num_links,
            topology=self.ccl_topology,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        )

    def _reduce_scatter_seq(self, x):
        """Reduce the WO partial over head-groups and scatter it back over the
        sequence (dim 2): reduce+scatter in one collective."""
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )

    # ------------------------------------------------------------- computation
    def _embed(self, input_ids, position_ids, token_type_ids, batch, local_seq):
        """Per-token embedding on the sequence-shard. Output [b, 1, local_seq, H]."""
        ck = self.norm_kernel_config
        adt = self.activation_dtype
        we = ttnn.embedding(input_ids, self.w["word_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        pe = ttnn.embedding(position_ids, self.w["pos_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        te = ttnn.embedding(token_type_ids, self.w["tt_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        emb = ttnn.add(ttnn.add(we, te), pe)
        emb = ttnn.layer_norm(
            emb, weight=self.w["emb_ln_w"], bias=self.w["emb_ln_b"], epsilon=self.eps, compute_kernel_config=ck
        )
        hidden = ttnn.linear(
            emb,
            self.w["map_w"],
            bias=self.w["map_b"],
            compute_kernel_config=self.matmul_kernel_config,
            core_grid=self.mm_core_grid,
            dtype=adt,
        )
        ttnn.deallocate(we)
        ttnn.deallocate(pe)
        ttnn.deallocate(te)
        ttnn.deallocate(emb)
        return ttnn.reshape(hidden, (batch, 1, local_seq, self.hidden_size))

    def _albert_layer(self, hidden_s, attention_mask, batch, full_seq, local_seq):
        """One AlbertLayer, sequence-parallel.

        hidden_s: [b, 1, local_seq, H] (sequence-sharded residual).
        Returns the same layout.
        """
        mm = self.matmul_kernel_config
        adt = self.activation_dtype
        nh, hd = self.local_heads, self.head_dim
        cg = self.mm_core_grid

        # --- gather residual to full sequence for attention -------------------
        hidden_full = self._all_gather_seq(hidden_s)  # [b, 1, full_seq, H]

        # packed local-head QKV (column-parallel: bias applied in-matmul)
        qkv = ttnn.linear(
            hidden_full, self.w["qkv_w"], bias=self.w["qkv_b"], compute_kernel_config=mm, core_grid=cg, dtype=adt
        )
        qkv = ttnn.reshape(qkv, (batch, 1, full_seq, 3 * nh * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=nh, num_kv_heads=nh, transpose_k_heads=False)
        ttnn.deallocate(qkv)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn)  # [b, 1, full_seq, nh*hd]
        attn = ttnn.reshape(attn, (batch, full_seq, nh * hd))

        # WO (row-parallel): NO bias here (added once post-reduce). Partial [b, full_seq, H].
        attn = ttnn.linear(attn, self.w["dense_w"], compute_kernel_config=mm, core_grid=cg, dtype=adt)
        attn = ttnn.reshape(attn, (batch, 1, full_seq, self.hidden_size))

        # reduce over head-groups + scatter back to sequence-shard -> [b,1,local_seq,H]
        attn_s = self._reduce_scatter_seq(attn)
        ttnn.deallocate(attn)
        ttnn.deallocate(hidden_full)

        # add WO bias once (sequence-sharded), residual add, post-LN (all local)
        attn_s = ttnn.add(attn_s, self.w["dense_b"])
        hidden_s = ttnn.layer_norm(
            ttnn.add(attn_s, hidden_s),
            weight=self.w["attn_ln_w"],
            bias=self.w["attn_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=self.norm_kernel_config,
        )
        ttnn.deallocate(attn_s)

        # --- FFN: fully local on the sequence-shard (replicated weights) ------
        ff = ttnn.linear(
            hidden_s,
            self.w["ffn_w"],
            bias=self.w["ffn_b"],
            compute_kernel_config=mm,
            program_config=self._ffn_in_program_config(batch * local_seq // TILE),
            dtype=adt,
        )
        ff = ttnn.linear(
            ff,
            self.w["ffn_out_w"],
            bias=self.w["ffn_out_b"],
            compute_kernel_config=mm,
            program_config=self._ffn_out_program_config(batch * local_seq // TILE),
            dtype=adt,
        )
        hidden_s = ttnn.layer_norm(
            ttnn.add(ff, hidden_s),
            weight=self.w["full_ln_w"],
            bias=self.w["full_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=self.norm_kernel_config,
        )
        ttnn.deallocate(ff)
        return hidden_s

    def _encode(self, input_ids, position_ids, token_type_ids, attention_mask, batch, full_seq):
        local_seq = full_seq // self.tp
        hidden = self._embed(input_ids, position_ids, token_type_ids, batch, local_seq)
        for _ in range(self.num_layers):
            hidden = self._albert_layer(hidden, attention_mask, batch, full_seq, local_seq)
        return hidden

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self, input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None
    ):
        """Eager bidirectional encode. Output is sequence-sharded [b,1,seq/TP,H]."""
        b = batch if batch is not None else input_ids.shape[0]
        tp = seq_len if seq_len is not None else input_ids.shape[-1]
        return self._encode(input_ids, position_ids, token_type_ids, attention_mask, b, tp)

    # ------------------------------------------------------------------- decode
    def capture_decode_trace(self, prepared):
        batch = prepared["batch"]
        tp = prepared["padded_seq_len"]
        has_mask = prepared["attention_mask"] is not None
        key = (batch, tp, has_mask)
        if key in self._traces:
            return self._traces[key]

        dev = self.mesh_device
        in_ids = ttnn.clone(prepared["input_ids"])
        in_pos = ttnn.clone(prepared["position_ids"])
        in_tt = ttnn.clone(prepared["token_type_ids"])
        in_mask = ttnn.clone(prepared["attention_mask"]) if has_mask else None

        warm = self._encode(in_ids, in_pos, in_tt, in_mask, batch, tp)
        ttnn.deallocate(warm)
        ttnn.synchronize_device(dev)

        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        out = self._encode(in_ids, in_pos, in_tt, in_mask, batch, tp)
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)

        record = {
            "trace_id": trace_id,
            "in_ids": in_ids,
            "in_pos": in_pos,
            "in_tt": in_tt,
            "in_mask": in_mask,
            "out": out,
        }
        self._traces[key] = record
        return record

    def decode_forward(self, input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None):
        """Traced replay of the encoder. Output sequence-sharded [b,1,seq/TP,H]."""
        b = batch if batch is not None else input_ids.shape[0]
        tp = seq_len if seq_len is not None else input_ids.shape[-1]
        prepared = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "batch": b,
            "padded_seq_len": tp,
        }
        record = self.capture_decode_trace(prepared)
        dev = self.mesh_device
        ttnn.copy(input_ids, record["in_ids"])
        ttnn.copy(position_ids, record["in_pos"])
        ttnn.copy(token_type_ids, record["in_tt"])
        if attention_mask is not None and record["in_mask"] is not None:
            ttnn.copy(attention_mask, record["in_mask"])
        ttnn.execute_trace(dev, record["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return record["out"]

    def release_traces(self):
        for record in self._traces.values():
            ttnn.release_trace(self.mesh_device, record["trace_id"])
        self._traces.clear()
