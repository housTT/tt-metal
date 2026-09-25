# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-token-prediction (MTP) draft head for Qwen3.8-Flash-Next.

The checkpoint ships one MTP block under ``mtp.*``: a QSA decoder layer with
its own 512-expert MoE and hyper-connections, two ``2560x2560`` fusion
projections, their pre-norms, and its own hyper-connection mixer.  It reuses
the target's token embedding and LM head.  Semantics (SGLang
``qwen4_exp_mtp.py``, ``Qwen4ExpForCausalLMMTP``):

* ``e = fc_embedding(GemmaRMSNorm_2560(embed(token_{i+1})))``
* ``h = GemmaRMSNorm_10240(wide_hidden_i)`` viewed as four ``2560`` streams,
  ``fc_hidden`` applied per stream
* layer input (wide, four streams) ``= h_streams + e`` at position ``i``
* one QSA decoder layer -> own mixer collapse -> shared ``lm_head``
* the argmax is the draft for ``token_{i+2}``.

``wide_hidden_i`` is the four-stream residual after the target's last decoder
layer (the tensor the target's final mixer consumes), at the position that
produced ``token_{i+1}``.  GemmaRMSNorm weights are stored as ``1 + w``, the
same convention the port uses for every hyper-connection norm.

This module implements the draft computation (eager, batch one).  Lossless
verification (a two-row target step and state rollback) is tracked in the plan.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.model import BLOCK_SIZE, HC_COUNT, HC_LOWRANK, HC_WIDTH, HIDDEN_SIZE
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import LINEAR_ATTENTION
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import RESIDUAL_SHARD_WIDTH, MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import _hifi2
from models.autoports.qwen_qwen3_8_flash_next.tt.precision_config import layer_policy

MTP_LAYER_PREFIX = "mtp.layers.0."
MTP_CACHE_TAG = "mtp"


def _upload_replicated(value: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class Qwen38MTPDraftHead:
    """Device-resident MTP block bound to a loaded :class:`Qwen38FullModel`."""

    def __init__(self, model, *, fusion_dtype=ttnn.bfloat16):
        self.model = model
        self.mesh_device = model.mesh_device
        text_config = model.text_config
        qsa_indices = [
            index
            for index in range(int(text_config.num_hidden_layers))
            if text_config.layer_types[index] != LINEAR_ATTENTION
        ]
        if not qsa_indices:
            raise ValueError("the MTP layer needs a QSA shape contract")
        self.shape_layer_idx = qsa_indices[0]
        started = time.perf_counter()
        kwargs = layer_policy(model.precision_config, self.shape_layer_idx)
        kwargs["resident_weight_cache_path"] = model.expert_weight_cache
        kwargs["moe_kernel"] = model.moe_kernel
        if model.expert_mode != "resident_ep4":
            raise ValueError("the MTP draft head requires resident EP4 experts")
        self.layer: MultichipDecoder = MultichipDecoder.from_checkpoint_resident(
            model.checkpoint,
            hf_config=model.hf_config,
            layer_idx=self.shape_layer_idx,
            mesh_device=self.mesh_device,
            max_batch=model.max_batch,
            max_seq_len=model.max_seq_len,
            block_size=BLOCK_SIZE,
            ple_store=None,
            key_prefix=MTP_LAYER_PREFIX,
            cache_tag=MTP_CACHE_TAG,
            **kwargs,
        )
        self.layer.qsa_dense_decode = True
        self.layer.prepare_decode_state()
        self.layer_load_seconds = time.perf_counter() - started

        checkpoint = model.checkpoint
        eps = float(text_config.rms_norm_eps)
        self.rms_norm_eps = eps
        mesh = self.mesh_device
        norm_e = checkpoint.tensor("mtp.pre_fc_norm_embedding.weight")
        norm_h = checkpoint.tensor("mtp.pre_fc_norm_hidden.weight")
        if tuple(norm_e.shape) != (HIDDEN_SIZE,) or tuple(norm_h.shape) != (HC_WIDTH,):
            raise ValueError("unexpected MTP pre-fc norm shapes")
        self.norm_embedding_weight = _upload_replicated((norm_e.float() + 1.0).bfloat16().reshape(1, 1, 1, HIDDEN_SIZE), mesh)
        self.norm_hidden_weight = _upload_replicated((norm_h.float() + 1.0).bfloat16().reshape(1, 1, 1, HC_WIDTH), mesh)
        fc_e = checkpoint.tensor("mtp.fc_embedding.weight")
        fc_h = checkpoint.tensor("mtp.fc_hidden.weight")
        if tuple(fc_e.shape) != (HIDDEN_SIZE, HIDDEN_SIZE) or tuple(fc_h.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
            raise ValueError("unexpected MTP fusion projection shapes")
        # nn.Linear weights are [out, in]; ttnn.linear consumes [in, out].
        self.fc_embedding_weight = _upload_replicated(
            fc_e.to(torch.bfloat16).transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, HIDDEN_SIZE), mesh, dtype=fusion_dtype
        )
        self.fc_hidden_weight = _upload_replicated(
            fc_h.to(torch.bfloat16).transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, HIDDEN_SIZE), mesh, dtype=fusion_dtype
        )
        del norm_e, norm_h, fc_e, fc_h

        mixer = "mtp.hyper_connection_mixer"
        norm = checkpoint.tensor(f"{mixer}.hc_norm.weight")
        down = checkpoint.tensor(f"{mixer}.input_mix_weight_down.weight")
        up = checkpoint.tensor(f"{mixer}.input_mix_weight_up.weight")
        if tuple(down.shape) != (HC_LOWRANK, HC_WIDTH) or tuple(up.shape) != (HC_WIDTH, HC_LOWRANK):
            raise ValueError("unexpected MTP mixer shapes")
        self.mixer_norm_weight = _upload_replicated((norm.float() + 1.0).bfloat16().reshape(1, 1, 1, HC_WIDTH), mesh)
        # The mixer divides the low-rank projection by hc_count before SiLU;
        # fold the exact power-of-two scale into the weight (as the target does).
        self.mixer_down_weight = _upload_replicated(
            (down.float() / HC_COUNT).to(torch.bfloat16).transpose(0, 1).reshape(1, 1, HC_WIDTH, HC_LOWRANK), mesh
        )
        self.mixer_up_weight = _upload_replicated(
            up.to(torch.bfloat16).transpose(0, 1).reshape(1, 1, HC_LOWRANK, HC_WIDTH), mesh
        )
        del norm, down, up
        self.mixer_compute = _hifi2()
        self.fusion_compute = _functional_decoder._hifi4(fp32=True)
        # Scratch position / token registers for standalone (non-batch-state) use.
        self.scratch_pos = _upload_replicated(
            torch.zeros(model.max_batch, dtype=torch.int32), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.token_register = _upload_replicated(
            torch.zeros(1, 1, 1, model.max_batch, dtype=torch.int32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.draft_calls = 0
        self.draft_seconds = 0.0

    # ------------------------------------------------------------ components

    def _embed_next_token(self, token_input):
        """Full-width ``[1,1,M,2560]`` embedding of the device token register."""

        model = self.model
        embedded = ttnn.embedding(
            token_input,
            model.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if embedded.dtype != ttnn.bfloat16:
            converted = ttnn.typecast(embedded, ttnn.bfloat16)
            ttnn.deallocate(embedded)
            embedded = converted
        rows = int(embedded.shape[-2])
        local = ttnn.reshape(embedded, (1, 1, rows, RESIDUAL_SHARD_WIDTH))
        layer = self.layer
        full = ttnn.all_gather(
            local,
            dim=3,
            cluster_axis=layer.collective_axis,
            num_links=layer.collective_num_links,
            topology=layer.collective_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _functional_decoder._free(local, embedded, full)
        _functional_decoder._free(embedded, full)
        return full

    def fuse_inputs(self, wide_hidden, token_input):
        """Return the fractured ``[1,1,4M,640]`` MTP layer input.

        ``wide_hidden`` is the target's fractured ``[1,1,4M,640]`` residual
        after its last decoder layer; ``token_input`` the uint32 device
        register holding ``token_{i+1}`` for each row.
        """

        layer = self.layer
        gathered = layer.gather_residual(wide_hidden)  # [1,1,M,10240]
        if gathered.dtype != ttnn.bfloat16:
            converted = ttnn.typecast(gathered, ttnn.bfloat16)
            ttnn.deallocate(gathered)
            gathered = converted
        rows = int(gathered.shape[-2])
        normed = _functional_decoder.FunctionalDecoder._rms_norm(gathered, self.norm_hidden_weight, self.rms_norm_eps)
        _functional_decoder._free(gathered, normed)
        streams = ttnn.reshape(normed, (1, 1, rows * HC_COUNT, HIDDEN_SIZE))
        projected = ttnn.linear(
            streams,
            self.fc_hidden_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.fusion_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _functional_decoder._free(streams, normed, projected)
        ttnn.deallocate(normed)

        embedded = self._embed_next_token(token_input)  # [1,1,M,2560]
        embedded_normed = _functional_decoder.FunctionalDecoder._rms_norm(
            embedded, self.norm_embedding_weight, self.rms_norm_eps
        )
        ttnn.deallocate(embedded)
        embedded_projected = ttnn.linear(
            embedded_normed,
            self.fc_embedding_weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.fusion_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(embedded_normed)
        # Broadcast the token term over the four streams of each row.
        per_row = ttnn.reshape(projected, (rows, HC_COUNT, HIDDEN_SIZE))
        token_rows = ttnn.reshape(embedded_projected, (rows, 1, HIDDEN_SIZE))
        fused = ttnn.add(per_row, token_rows)
        _functional_decoder._free(per_row, projected, fused)
        _functional_decoder._free(token_rows, embedded_projected, fused)
        ttnn.deallocate(projected)
        ttnn.deallocate(embedded_projected)
        wide = ttnn.reshape(fused, (1, 1, rows, HC_WIDTH))
        local = layer.fracture_residual(wide)
        _functional_decoder._free(wide, fused, local)
        return local

    def collapse(self, residual):
        """MTP hyper-connection mixer: fractured ``[1,1,4M,640]`` -> ``[1,1,M,2560]``."""

        compute = residual if residual.dtype == ttnn.bfloat16 else ttnn.typecast(residual, ttnn.bfloat16)
        gathered = self.layer.gather_residual(compute)
        _functional_decoder._free(compute, residual, gathered)
        normed = _functional_decoder.FunctionalDecoder._rms_norm(
            gathered, self.mixer_norm_weight, self.rms_norm_eps, group_count=HC_COUNT
        )
        _functional_decoder._free(gathered, residual, normed)
        scaled = ttnn.linear(
            normed, self.mixer_down_weight, dtype=ttnn.bfloat16, compute_kernel_config=self.mixer_compute
        )
        low = ttnn.silu(scaled)
        ttnn.deallocate(scaled)
        mix = ttnn.linear(low, self.mixer_up_weight, dtype=ttnn.bfloat16, compute_kernel_config=self.mixer_compute)
        ttnn.deallocate(low)
        rows = int(normed.shape[-2])
        norm_groups = ttnn.reshape(normed, (rows, HC_COUNT, HIDDEN_SIZE))
        mix_groups = ttnn.reshape(mix, (rows, HC_COUNT, HIDDEN_SIZE))
        mixed = ttnn.multiply(norm_groups, mix_groups, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        _functional_decoder._free(norm_groups, normed, mixed)
        _functional_decoder._free(mix_groups, mix, mixed)
        ttnn.deallocate(normed)
        ttnn.deallocate(mix)
        hidden = ttnn.mean(mixed, dim=1, keepdim=True)
        ttnn.deallocate(mixed)
        return ttnn.reshape(hidden, (1, 1, rows, HIDDEN_SIZE))

    # ------------------------------------------------------------------ steps

    def draft_logits(self, wide_hidden, token_input, *, current_pos, page_table):
        """Eager batch-one draft: logits of ``token_{i+2}`` given ``hidden_i`` and ``token_{i+1}``.

        ``current_pos`` is the int32 device position register holding ``i``
        (the same value the target used for ``hidden_i``); ``page_table`` the
        request's device page table.  The MTP layer's own KV/indexer caches are
        updated at position ``i``.
        """

        started = time.perf_counter()
        local = self.fuse_inputs(wide_hidden, token_input)
        output = self.layer.decode_forward_fractured(
            local,
            current_pos=current_pos,
            page_table=page_table,
            rot_mats=self.model.rot_mats,
        )
        _functional_decoder._free(local, output)
        hidden = self.collapse(output)
        ttnn.deallocate(output)
        logits = self.model.project_hidden_logits(hidden)
        ttnn.deallocate(hidden)
        self.draft_calls += 1
        self.draft_seconds += time.perf_counter() - started
        return logits

    def set_scratch_position(self, position) -> None:
        values = torch.as_tensor(position, dtype=torch.int32).reshape(-1)
        if values.numel() == 1:
            values = values.repeat(self.model.max_batch)
        host = ttnn.from_torch(values.reshape(self.model.max_batch), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.scratch_pos)

    def set_tokens(self, tokens) -> None:
        values = torch.as_tensor(tokens, dtype=torch.int32).reshape(-1)
        if values.numel() == 1:
            values = values.repeat(self.model.max_batch)
        host = ttnn.from_torch(
            values.reshape(1, 1, 1, self.model.max_batch), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self.token_register)

    def close(self) -> None:
        for name in (
            "norm_embedding_weight",
            "norm_hidden_weight",
            "fc_embedding_weight",
            "fc_hidden_weight",
            "mixer_norm_weight",
            "mixer_down_weight",
            "mixer_up_weight",
            "scratch_pos",
            "token_register",
        ):
            value = getattr(self, name, None)
            if isinstance(value, ttnn.Tensor) and value.is_allocated():
                ttnn.deallocate(value)
        layer = getattr(self, "layer", None)
        if layer is not None:
            layer.close_host_backing()
