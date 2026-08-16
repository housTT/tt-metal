# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full autoregressive TTNN model for ornith-ai/Ornith-1.0-35B on the 1x4 Blackhole ring.

This wraps the optimized multichip decoder layer delivered by
``doc/optimized_multichip_decoder/`` (:class:`...tt.multichip_decoder.MultichipDecoder`) with the
pieces a decoder layer does not own: token embeddings, the 40-layer stack, the final zero-centered
RMSNorm, a column-parallel LM head, and the on-device sampler that turns the LM head's
vocab-sharded logits into a token without a host round trip.

Everything the decoder stage decided is carried through unchanged:

* **mesh** — one ``1x4`` Blackhole ring under ``FabricConfig.FABRIC_1D_RING`` with the stage's
  fabric router packet size; TP=4 for every dense tensor and EP=4 for the 256 routed experts;
* **precision policy** — ``multichip_decoder``'s ``DEFAULT_POLICY``: BFP4/LoFi routed experts,
  BFP8/HiFi2 dense projections and shared expert, bfloat16/HiFi4/fp32-accumulate router, float32
  DeltaNet state, **bfloat8_b paged KV cache** with bfloat16 ``paged_update_cache`` inputs;
* **collectives** — ``CCL_MODE = "all_reduce"``, two per layer, both inside the layer. The
  deprecated ``ttnn.all_gather`` — which stage 5 measured diverging across devices under sustained
  traced replay — appears nowhere in this file, and the sampler's own gather is redirected off it
  by :class:`OrnithSamplingCCL`;
* **inter-layer residual layout** — ``[batch, seq, 2048]`` bfloat16 TILE DRAM-interleaved,
  **replicated** and bitwise identical on all four devices, with *no* collective at the layer
  boundary. The embedding is replicated for the same reason, and the only place the model
  fractures anything is the LM head, whose vocabulary shard is exactly what the sampler wants.

Public contract
---------------
``prefill_forward(tokens, page_table, prompt_lens, ...)``
    Any logical prompt length in ``[1, supported_context]``, including lengths that are not a
    multiple of the tile, the 64-token page, the 128-token physical alignment or the 2048-token
    internal prefill chunk. The model owns the chunking, the padding, the masking, the cache fill,
    the position bookkeeping and the output slicing.

``ttnn_decode_forward(tokens, current_pos, rot_idxs, page_table)``
    Device-only, trace-safe, and returns **sampler-ready** vocab-sharded logits. It advances
    ``current_pos`` and ``rot_idxs`` on device with ``ttnn.plus_one`` so a captured trace needs no
    per-token host position refresh.

The generator in ``tt/generator.py`` owns trace capture/replay, split sampling and the
readiness-check ``Generator`` contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import (
    CHECKPOINT_TEXT_PREFIX,
    HF_MODEL_ID,
    OrnithDecoderConfig,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import (
    DEFAULT_CCL_NUM_LINKS,
    DEFAULT_CCL_TOPOLOGY,
    DEFAULT_FABRIC_CONFIG,
    DEFAULT_MESH_SHAPE,
    MultichipDecoder,
    fabric_router_config,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    DEFAULT_MOE_GROUP_TOKENS,
    DEFAULT_PAGE_BLOCK_SIZE,
    DEFAULT_POLICY,
    DEFAULT_PREFILL_CHUNK,
    POLICIES,
    PrecisionPolicy,
    num_blocks_for_context,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.tt_ccl import TT_CCL

TILE = 32

#: Largest batch ``ttnn.sampling`` can serve — it runs one core per user and asserts
#: ``1 <= num_users <= 32`` (``sampling_device_operation``). It is also the decoder stage's
#: advertised decode batch bound, so the two agree and neither narrows the other.
MAX_SAMPLING_BATCH = 32

#: Groups the sampler's per-device local top-k is split into.
#:
#: ``ttnn.topk``'s runtime is linear in the reduced width and independent of every other dimension
#: (``doc/full_model/logs/probe_topk.txt``), so the 62080-wide vocabulary shard a 248320-token
#: vocabulary leaves on each of four devices costs ~9.9 ms in one reduction - 47 % of a token-out
#: decode step, and by far its largest single op. Splitting it into 20 groups of 3104 turns that
#: into a 3104-wide reduction plus a 640-wide one and costs ~0.9 ms for the same exact result. 20 is
#: the measured optimum among the divisors of 1940 (= 62080 / 32) that keep every group width tile
#: aligned; see ``doc/full_model/README.md`` §Sampling.
DEFAULT_TOPK_GROUPS = 20

#: L1-small the decoder's CCL ops allocate their semaphores from. Same value as
#: ``tests/test_multichip_decoder.py``'s ``DEVICE_PARAMS``.
DEFAULT_L1_SMALL_SIZE = 24576

#: Trace region for the whole-model decode trace plus the sampling trace. A 40-layer decode capture
#: is far larger than the single-layer captures the decoder stage took with ``trace_region_size=0``.
DEFAULT_TRACE_REGION_SIZE = 200_000_000


def _align_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def open_ornith_mesh(
    mesh_shape=DEFAULT_MESH_SHAPE,
    *,
    l1_small_size: int = DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE,
    fabric: bool = True,
):
    """Open the target mesh with the fabric configuration the decoder stage measured.

    The readiness runners' ``--mesh-device`` labels do not cover a ``1x4`` Blackhole ring and their
    opener passes neither ``l1_small_size`` nor ``trace_region_size``, so every driver in this stage
    opens the mesh through here and calls the runners' *programmatic* entry points.
    """
    if fabric and tuple(mesh_shape) != (1, 1):
        ttnn.set_fabric_config(DEFAULT_FABRIC_CONFIG, router_config=fabric_router_config())
    return ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_shape),
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
    )


def close_ornith_mesh(mesh_device, *, fabric: bool = True):
    ttnn.close_mesh_device(mesh_device)
    if fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------- sampler CCL shim
class OrnithSamplingCCL(TT_CCL):
    """The CCL object :class:`~models.common.sampling.tt_sampling.TTSampling` asks for.

    ``TTSampling._perform_all_gather`` prefers ``tt_ccl.line_all_gather`` and otherwise falls back to
    the **deprecated** ``ttnn.all_gather``. That fallback is exactly the op
    ``doc/optimized_multichip_decoder/README.md`` §4.1 measured producing a different result on one
    device from the others in of order 1 % of sustained traced-replay rounds, and the stage removed
    it from the decoder for that reason. Letting the sampler reintroduce it — on the tensor that
    decides the emitted token, no less — would undo that finding, so this shim provides a
    ``line_all_gather`` built on ``ttnn.experimental.all_gather_async`` with the two gather
    semaphores the op asserts (``all_gather_async_device_operation.cpp:57``) plus a barrier
    semaphore. That spelling is the other 0-of-600 arm in the same table.

    A persistent output buffer was tried first — OPT-009 measured persistence as worth ~11 % inside
    the async family — and **rejected on an op-contract conflict, not on latency**: ``TTSampling``
    ends every call with ``ttnn.deallocate(topk_values_gathered_bf16_interleaved)``
    (``tt_sampling.py``), and that tensor *is* the gather's output buffer, so a buffer handed in as
    ``persistent_output_buffer`` is freed by the first sampling call and the second one dies on
    ``input_tensor.is_allocated()``. The hook therefore advertises no ``buffer_key``, which is also
    what stops ``TTSampling`` from passing one.

    Semaphore *cycling* comes from the base :class:`~models.common.modules.tt_ccl.TT_CCL`, which
    double-buffers each handle set, and it is not optional: one sampling call issues two gathers
    back to back (candidate values, then candidate indices), and an earlier version of this shim
    that returned one fixed handle set for both deadlocked the mesh under a 40-layer decode capture.
    """

    def __init__(self, mesh_device, *, num_links: int | None = None, topology=None):
        super().__init__(mesh_device)
        self.num_devices = mesh_device.get_num_devices()
        self.num_links = DEFAULT_CCL_NUM_LINKS if num_links is None else int(num_links)
        self.topology = DEFAULT_CCL_TOPOLOGY if topology is None else topology
        #: Counts every gather this shim performs, so a test can prove the deprecated op is unused.
        self.gathers = 0

    # -- the TTSampling / Sampling1D hook ---------------------------------------------------
    # Deliberately no `buffer_key` / `dtype` parameter: both samplers probe this signature with
    # `inspect.signature` and only pass what it accepts. See the class docstring for why the
    # persistent-buffer variant cannot be used here.
    def line_all_gather(self, tensor, *, dim, cluster_axis=None, memory_config=None, num_links=None):
        self.gathers += 1
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=self.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=self.num_links if num_links is None else num_links,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        )

    def get_num_links(self, cluster_axis=None):
        return self.num_links


class OrnithSamplingArgs:
    """The duck-typed ``args`` object ``models.common.sampling`` reads.

    Only the attributes ``TTSampling``/``TTPenalties`` actually look up are set; everything else
    falls through to their documented defaults. ``model_config`` is deliberately left without a
    ``SAMPLING_AG_CONFIG`` entry, which keeps ``allow_force_argmax`` **off**: the greedy path then
    runs the same semantically greedy split-sampling graph as every other sampling mode (local
    ``top_k=32`` per vocab shard, gather the 4x32 candidates, ``ttnn.sampling`` with ``k=1, p=0,
    temp=1``) instead of an all-gather of the full 248320-wide logits followed by a global argmax.
    See ``doc/full_model/README.md`` for the measurement behind that choice.
    """

    def __init__(
        self,
        *,
        vocab_size,
        padded_vocab_size,
        cluster_shape,
        max_batch_size,
        max_top_k=32,
        topk_num_groups=DEFAULT_TOPK_GROUPS,
    ):
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.cluster_shape = tuple(cluster_shape)
        self.max_batch_size = max_batch_size
        self.max_top_k = max_top_k
        self.sampling_dp = 1
        self.sub_core_grids = None
        self.sub_core_grid_topk = None
        self.start_core = ttnn.CoreCoord(0, 0)
        self.pad_logits_to_power_of_2 = False
        self.topk_num_groups = int(topk_num_groups)
        self.model_config: dict = {}


# ---------------------------------------------------------------------------- checkpoint access
def resolve_model_path(model_path=None) -> Path:
    """Local snapshot directory for the checkpoint.

    ``model_path`` may be the snapshot itself, or an autoport directory that merely *names* the
    model — in which case the HF cache entry is resolved the same way the decoder stage's reference
    does.
    """
    if model_path is not None:
        candidate = Path(model_path)
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL_ID, allow_patterns=None))


def load_text_config(model_path=None):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    with open(resolve_model_path(model_path) / "config.json") as f:
        raw = json.load(f)
    cfg = Qwen3_5MoeTextConfig(**raw["text_config"])
    cfg._attn_implementation = "eager"
    return cfg


class _CheckpointReader:
    """Streams exactly the tensors asked for out of the sharded safetensors snapshot."""

    def __init__(self, model_path: Path):
        self.path = model_path
        with open(model_path / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]

    def get(self, key: str):
        from safetensors import safe_open

        fname = self.weight_map.get(key)
        if fname is None:
            raise KeyError(f"{key!r} is not in the checkpoint index")
        with safe_open(str(self.path / fname), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def layer_state_dict(self, layer_idx: int) -> dict:
        """Module-relative state dict for one text decoder layer, with the experts fused.

        Same conversion ``transformers`` applies on load, and the same one the decoder stage's
        reference implements: ``gate_up_proj[e] = cat([gate_proj[e], up_proj[e]], dim=0)`` and
        ``down_proj`` a plain stack.
        """
        import torch
        from safetensors import safe_open

        prefix = f"{CHECKPOINT_TEXT_PREFIX}layers.{layer_idx}."
        wanted = {k: v for k, v in self.weight_map.items() if k.startswith(prefix)}
        if not wanted:
            raise KeyError(f"no checkpoint entries under {prefix!r}")

        per_file: dict[str, list[str]] = {}
        for key, fname in wanted.items():
            per_file.setdefault(fname, []).append(key)

        gate: dict[int, torch.Tensor] = {}
        up: dict[int, torch.Tensor] = {}
        down: dict[int, torch.Tensor] = {}
        out: dict = {}
        for fname, keys in per_file.items():
            with safe_open(str(self.path / fname), framework="pt", device="cpu") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    sub = key[len(prefix) :]
                    if sub.startswith("mlp.experts."):
                        rest = sub[len("mlp.experts.") :]
                        expert_str, _, tail = rest.partition(".")
                        expert = int(expert_str)
                        if tail == "gate_proj.weight":
                            gate[expert] = tensor
                        elif tail == "up_proj.weight":
                            up[expert] = tensor
                        elif tail == "down_proj.weight":
                            down[expert] = tensor
                        else:
                            raise KeyError(f"unexpected expert entry {key!r}")
                    else:
                        out[sub] = tensor
        if gate:
            experts = sorted(gate)
            out["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([gate[e], up[e]], dim=0) for e in experts], dim=0)
            out["mlp.experts.down_proj"] = torch.stack([down[e] for e in experts], dim=0)
        return out


# ---------------------------------------------------------------------------- the model
class OrnithModel(LightweightModule):
    """The whole Ornith-1.0-35B text model on one TTNN mesh."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        *,
        layers,
        layer_indices,
        embed_weight,
        norm_weight,
        lm_head_weights,
        max_context: int,
        prefill_chunk: int,
        page_block_size: int,
        policy: PrecisionPolicy,
        vocab_size: int,
        padded_vocab_size: int,
        tp: int,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.cfg = OrnithDecoderConfig.from_hf_config(hf_config)
        self.layers = list(layers)
        #: Which HF layer index each entry of :attr:`layers` is. Equal to ``range(num_hidden_layers)``
        #: for the delivered model; a reduced probe passes a shorter list (one layer of each kind).
        self.layer_indices = list(layer_indices)
        self.embed_weight = embed_weight
        self.norm_weight = norm_weight
        self.lm_head_weights = list(lm_head_weights)
        self.max_context = int(max_context)
        self.prefill_chunk = int(prefill_chunk)
        self.page_block_size = int(page_block_size)
        self.policy = policy
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)
        self.tp = int(tp)
        self.dim = self.cfg.dim
        self.is_reduced = len(self.layers) != self.cfg.num_hidden_layers

        self.lm_head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.proj_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=policy.proj_fp32_acc,
            packer_l1_acc=policy.proj_packer_l1_acc,
        )

        self.max_batch_size = None
        self._packs: dict[int, list[dict]] = {}
        self._active_pack: int | None = None
        self.kv_cache = None
        self.num_blocks = None

        self.tt_ccl = None
        self.sampling = None

        #: Steady-state host-work counters. The generator drives them; they exist here so any driver
        #: can assert the decode loop is not host-stepped. See ``doc/full_model/README.md`` §Trace.
        #: True once a prompt has been written into the DeltaNet state or the paged KV cache, and
        #: False again after :meth:`reset_state`. The generator's trace capture wipes both, so this
        #: is what lets it refuse to capture over a live request instead of silently erasing it
        #: (``tt/generator.py::_ensure_decode_trace``, README §5.3). It lives on the model because
        #: every write path is here - the generator's high-level ``generate`` writes through
        #: ``prefill_forward_single``, not through ``prefill_forward``.
        self.state_is_live = False

        self.counters = {
            "decode_calls": 0,
            "embedding_lookups": 0,
            "position_refreshes": 0,
            "rope_refreshes": 0,
            "token_refreshes": 0,
            "page_table_refreshes": 0,
        }

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls,
        model_path=None,
        *,
        mesh_device,
        max_context: int | None = None,
        prefill_chunk: int = DEFAULT_PREFILL_CHUNK,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        moe_group_tokens: int = DEFAULT_MOE_GROUP_TOKENS,
        policy: PrecisionPolicy | str = DEFAULT_POLICY,
        layer_indices=None,
        override_num_layers: int | None = None,
        lm_head_dtype=None,
        lm_head_max_columns: int | None = None,
        tp: int | None = None,
        hf_config=None,
    ) -> "OrnithModel":
        """Build the model from the real checkpoint.

        ``layer_indices`` / ``override_num_layers`` build a *reduced* model for debugging and
        profiling. ``layer_indices=[0, 3]`` is the reduced profiling variant the ``$full-model``
        skill asks for: one real ``linear_attention`` layer and one real ``full_attention`` layer,
        real weights, real cache and page-table shapes, real terminal norm/LM head/sampling.
        """
        import torch

        if isinstance(policy, str):
            if policy not in POLICIES:
                raise ValueError(f"unknown precision policy {policy!r}; known: {sorted(POLICIES)}")
            policy = POLICIES[policy]
        path = resolve_model_path(model_path)
        hf_config = hf_config if hf_config is not None else load_text_config(path)
        gcfg = OrnithDecoderConfig.from_hf_config(hf_config)
        tp = int(tp or mesh_device.get_num_devices())
        max_context = int(max_context or gcfg.max_position_embeddings)

        if layer_indices is None:
            n = gcfg.num_hidden_layers if override_num_layers is None else int(override_num_layers)
            layer_indices = list(range(min(n, gcfg.num_hidden_layers)))
        layer_indices = [int(i) for i in layer_indices]

        reader = _CheckpointReader(path)

        vocab_size = hf_config.vocab_size
        # The sampler shards the vocabulary by device and every device's shard has to be tile
        # aligned, so the LM head's output width is rounded up to a multiple of 32 * tp. Ornith's
        # 248320 is already a multiple of 128, so nothing is padded and no invalid-vocab mask is
        # needed; the arithmetic stays here so a different checkpoint cannot silently misalign.
        padded_vocab_size = _align_up(vocab_size, TILE * tp)

        logger.info(
            f"building OrnithModel: {len(layer_indices)} layer(s) {layer_indices if len(layer_indices) < 8 else '0..'} "
            f"tp={tp} max_context={max_context} policy={policy.name} vocab={vocab_size}->{padded_vocab_size}"
        )

        embed = reader.get(f"{CHECKPOINT_TEXT_PREFIX}embed_tokens.weight")
        if int(embed.shape[0]) != vocab_size:
            raise ValueError(f"embed_tokens rows {int(embed.shape[0])} != vocab_size {vocab_size}")
        embed_weight = ttnn.as_tensor(
            embed.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        del embed

        # Final norm: HF's Qwen3_5MoeRMSNorm is zero-centered (`x_normed * (1 + w)`), the same
        # convention as the two in-layer norms, so the `+1` is folded in at load time exactly as
        # MultichipDecoder.from_state_dict does.
        norm_weight = ttnn.as_tensor(
            (reader.get(f"{CHECKPOINT_TEXT_PREFIX}norm.weight").float() + 1.0).reshape(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # LM head: column-parallel over the vocabulary, which is both the cheapest split (no partial
        # sums, no collective) and precisely the shard layout the split sampler consumes.
        lm_head_dtype = policy.proj_dtype if lm_head_dtype is None else lm_head_dtype
        head = reader.get("lm_head.weight").float().transpose(0, 1).contiguous()  # [dim, vocab]
        if padded_vocab_size > vocab_size:
            head = torch.cat([head, torch.zeros(head.shape[0], padded_vocab_size - vocab_size)], dim=1)
        per_device = padded_vocab_size // tp
        columns = per_device if lm_head_max_columns is None else min(per_device, int(lm_head_max_columns))
        if per_device % columns:
            raise ValueError(f"lm_head_max_columns {columns} must divide the per-device width {per_device}")
        lm_head_weights = []
        for split in range(per_device // columns):
            parts = [
                head[:, d * per_device + split * columns : d * per_device + (split + 1) * columns] for d in range(tp)
            ]
            lm_head_weights.append(
                ttnn.as_tensor(
                    torch.cat(parts, dim=1).contiguous(),
                    dtype=lm_head_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=1),
                )
            )
        del head

        layers = []
        shared_rope = None
        for layer_idx in layer_indices:
            state_dict = reader.layer_state_dict(layer_idx)
            layer = MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_context=max_context,
                page_block_size=page_block_size,
                prefill_chunk=prefill_chunk,
                moe_group_tokens=moe_group_tokens,
                policy=policy,
                tp=tp,
            )
            del state_dict
            if layer.rope is not None:
                # One RoPE table pair for the whole stack. At the advertised context each pair is
                # ~34 MB per device, so the ten full_attention layers would otherwise carry ~608 MB
                # of identical tables. `doc/context_contract.json`'s per-layer projection calls this
                # out as its own conservatism; the full model does not have to inherit it.
                if shared_rope is None:
                    shared_rope = layer.rope
                else:
                    ttnn.deallocate(layer.rope.cos_table)
                    ttnn.deallocate(layer.rope.sin_table)
                    layer.rope = shared_rope
            layers.append(layer)
            logger.info(f"  layer {layer_idx} ({layer.kind}) loaded")

        return cls(
            mesh_device,
            hf_config,
            layers=layers,
            layer_indices=layer_indices,
            embed_weight=embed_weight,
            norm_weight=norm_weight,
            lm_head_weights=lm_head_weights,
            max_context=max_context,
            prefill_chunk=prefill_chunk,
            page_block_size=page_block_size,
            policy=policy,
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
            tp=tp,
        )

    # ------------------------------------------------------------------ cache / state
    def blocks_per_user(self, context: int | None = None) -> int:
        return num_blocks_for_context(int(context or self.max_context), self.page_block_size)

    def allocate_kv_cache(self, num_blocks: int, dtype=None):
        """Allocate the paged KV cache for every ``full_attention`` layer.

        Returns the readiness-contract shape: a list with one entry per decoder layer, ``[k, v]``
        for the layers that have a cache and ``[]`` for the ``linear_attention`` layers, whose state
        is a fixed-size recurrent matrix rather than a growing cache.
        """
        cache = []
        for layer in self.layers:
            if layer.is_full_attention:
                k, v = layer.allocate_kv_cache(num_blocks, dtype=dtype)
                cache.append([k, v])
            else:
                cache.append([])
        self.kv_cache = cache
        self.num_blocks = int(num_blocks)
        return cache

    def attach_kv_cache(self, kv_cache):
        """Point the layers at a caller-owned cache (the serving path)."""
        if kv_cache is None:
            return self.kv_cache
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries for {len(self.layers)} layers")
        for layer, entry in zip(self.layers, kv_cache):
            if layer.is_full_attention:
                if len(entry) != 2:
                    raise ValueError(f"layer {layer.layer_idx} is full_attention and needs a [k, v] pair")
                layer.attach_kv_cache(entry[0], entry[1])
        self.kv_cache = kv_cache
        if kv_cache and any(e for e in kv_cache):
            first = next(e for e in kv_cache if e)
            self.num_blocks = int(first[0].shape[0])
        return kv_cache

    def allocate_state(self, max_batch_size: int):
        """Allocate the per-batch decoder state for both the prefill and the decode batch.

        Prefill runs **one user at a time at batch 1** and decode runs at ``max_batch_size``. Two
        reasons, both structural rather than convenient:

        * a ``linear_attention`` layer's DeltaNet state is per-row and recurrent, so a batch of
          mixed-length prompts cannot be right-padded into one call — the pad tokens would advance
          the short users' state — and cannot be left-padded either, because ``full_attention``
          RoPE positions are absolute. Per-user prefill is what the paged/recurrent split allows;
        * ``ttnn.conv1d``'s prepared weights depend on the batch, and its coverage *shrinks* as the
          batch grows: at batch 1 every prefill block length gets a conv program and at batch 32
          none do. Prefilling at batch 1 is therefore also the faster path.

        Both packs are built here, at setup, so a forward never reallocates and a captured trace's
        buffer addresses stay valid.
        """
        max_batch_size = int(max_batch_size)
        if max_batch_size < 1 or max_batch_size > MAX_SAMPLING_BATCH:
            raise ValueError(f"max_batch_size {max_batch_size} must be in [1, {MAX_SAMPLING_BATCH}]")
        self.max_batch_size = max_batch_size
        self._packs = {}
        for batch in dict.fromkeys((1, max_batch_size)):
            for layer in self.layers:
                layer.allocate_state(batch)
            self._packs[batch] = [self._capture_pack(layer) for layer in self.layers]
            # `MultichipDecoder.allocate_state` frees the conv1d weights and the paged-fill row
            # indices it finds on the layer before building the new ones, so the pack just captured
            # has to be detached from the layer or the *next* batch's allocation would free the
            # buffers this pack owns - and the batch-1 prefill would then hand `ttnn.conv1d` a
            # deallocated weight.
            for layer in self.layers:
                layer.w["conv1d_weights"] = {}
                layer.batch_idxs = None
        self._share_fused_gate_buffers()
        self._prebuild_slot_masks(max_batch_size)
        self._active_pack = max_batch_size
        self._apply_pack(max_batch_size)

    def _prebuild_slot_masks(self, max_batch_size: int):
        """Build every per-slot merge mask here, at setup, rather than lazily at first use.

        The masks depend only on ``(slot, batch)``, so they can all be made now - and they have to
        be. Built lazily, the first batched ``prefill_forward`` would allocate them **after** trace
        capture, and a long-lived buffer allocated after capture is the §5.1 hazard: the trace's
        intermediates are handed back to the allocator at ``end_trace_capture`` while the captured
        commands still write to those addresses. ``_ensure_traces_replay_safe`` would in practice
        catch it (``_merge_rows`` compiles programs too, so the program cache moves), but that is a
        side effect of another op's compilation, not a guarantee about these tensors.
        """
        if max_batch_size <= 1:
            return
        for slot in range(max_batch_size):
            self._slot_mask(slot, max_batch_size)
            self._slot_mask(slot, max_batch_size, invert=True)

    def _share_fused_gate_buffers(self):
        """One set of fused-router-gate buffers for the whole stack instead of one per layer.

        ``MultichipMoE.prepare_decode_gate`` allocates five **persistent L1** tensors per layer: the
        zero bias, the expert-id table, the two preallocated gate outputs (all HEIGHT_SHARDED, one
        32x32 shard per core) and the ROW_MAJOR scatter base. Their contents depend only on the
        expert count, the top-k and the mesh — not on the layer — so forty layers hold forty
        identical copies, which is ~8 KiB of L1 per core per layer, ~320 KiB per core in total.

        That is not a memory nicety. A decoder-layer test never sees it because it builds one layer;
        the 40-layer stack does, and it fails hard rather than slowly: the surviving L1 headroom
        drops below what ``chunked_scaled_dot_product_attention``'s statically allocated circular
        buffers need at the shipped 256-token prefill chunk, and prefill dies with *"Statically
        allocated circular buffers ... clash with L1 buffers ... L1 buffer allocated at 1196032 and
        static circular buffer region ends at 1430912"*. Sharing one set restores the headroom and
        keeps the decoder stage's prefill SDPA chunk, which is the alternative that would otherwise
        have had to be given up.

        Sharing is safe because the layers run **sequentially**, in the traced graph as well as
        eagerly: a layer's gate output is consumed by its own scatter before the next layer's gate
        call writes the buffer again.
        """
        shared = None
        for layer in self.layers:
            moe = layer.moe
            buffers = getattr(moe, "_gate_buffers", None)
            zeros = getattr(moe, "_gate_zeros", None)
            if not buffers:
                continue
            if shared is None:
                shared = (buffers, zeros)
                continue
            if buffers is shared[0]:
                # Already sharing (a second allocate_state on the same model). Freeing here would be
                # a double free of the tensors layer 0 owns.
                continue
            for entry in buffers.values():
                for tensor in entry[1:5]:
                    ttnn.deallocate(tensor)
            for tensor in (zeros or {}).values():
                ttnn.deallocate(tensor)
            moe._gate_buffers = shared[0]
            moe._gate_zeros = shared[1]
        if shared is not None and len(self.layers) > 1:
            logger.info(
                f"fused-router-gate buffers shared across {len(self.layers)} layers "
                f"(row counts {sorted(shared[0])})"
            )

    @staticmethod
    def _capture_pack(layer) -> dict:
        return {
            "batch_size": layer.batch_size,
            "batch_idxs": layer.batch_idxs,
            "recurrent_state": layer.recurrent_state,
            "conv_state": layer.conv_state,
            "conv1d_weights": layer.w.get("conv1d_weights", {}),
            "conv1d_lengths": list(layer.conv1d_lengths),
        }

    def _apply_pack(self, batch: int) -> None:
        for layer, pack in zip(self.layers, self._packs[batch]):
            layer.batch_size = pack["batch_size"]
            layer.batch_idxs = pack["batch_idxs"]
            layer.recurrent_state = pack["recurrent_state"]
            layer.conv_state = pack["conv_state"]
            layer.w["conv1d_weights"] = pack["conv1d_weights"]
            layer.conv1d_lengths = list(pack["conv1d_lengths"])
        self._active_pack = batch

    def _use_pack(self, batch: int) -> None:
        if self._active_pack != batch:
            self._apply_pack(batch)

    def reset_state(self, *, zero_kv_cache: bool = True):
        """Wipe every per-prompt state: DeltaNet recurrent/conv rows in both packs, and the cache."""
        self.state_is_live = False
        for batch in self._packs:
            self._apply_pack(batch)
            for layer in self.layers:
                layer.reset_state()
        self._apply_pack(self.max_batch_size)
        if zero_kv_cache and self.kv_cache is not None:
            for entry in self.kv_cache:
                for tensor in entry:
                    ttnn.multiply(tensor, 0.0, output_tensor=tensor)

    # ------------------------------------------------------------------ sampling
    def build_sampler(
        self,
        *,
        max_batch_size: int | None = None,
        max_top_k: int = 32,
        pad_to_power_of_2: bool = False,
        topk_num_groups: int = DEFAULT_TOPK_GROUPS,
    ):
        """Construct the shared on-device sampler (``models.common.sampling``).

        Kept on the model rather than in the generator because the sampler's persistent buffers,
        semaphores and index tables are model-shaped setup state, and because the generator has to
        be able to hand ``capture_trace`` the exact logits tensor the model trace produced.
        """
        from models.common.sampling import SamplingGenerator

        batch = self.max_batch_size if max_batch_size is None else int(max_batch_size)
        self.tt_ccl = OrnithSamplingCCL(self.mesh_device)
        args = OrnithSamplingArgs(
            vocab_size=self.vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            cluster_shape=tuple(self.mesh_device.shape),
            max_batch_size=max(TILE, _align_up(batch, TILE)),
            max_top_k=max_top_k,
            topk_num_groups=topk_num_groups,
        )
        args.pad_logits_to_power_of_2 = bool(pad_to_power_of_2)
        self.sampling = SamplingGenerator(args=args, mesh_device=self.mesh_device, tt_ccl=self.tt_ccl)
        return self.sampling

    # ------------------------------------------------------------------ terminal path
    def _final_norm(self, x):
        return ttnn.rms_norm(x, weight=self.norm_weight, epsilon=self.cfg.norm_eps)

    def _lm_head(self, rows):
        """``[1, 1, R, dim]`` -> vocab-sharded logits ``[1, 1, R, padded_vocab / tp]`` per device."""
        outs = [
            ttnn.linear(
                rows,
                weight,
                compute_kernel_config=self.lm_head_compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for weight in self.lm_head_weights
        ]
        if len(outs) == 1:
            return outs[0]
        joined = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for out in outs:
            ttnn.deallocate(out)
        return joined

    def _sampler_rows(self, x, rows: int):
        """Reshape a hidden state to the ``[1, 1, 32, dim]`` block the sampler's tables expect."""
        dims = [int(d) for d in x.shape]
        flat = ttnn.reshape(x, [1, 1, rows, self.dim])
        if rows < TILE:
            # ttnn.pad can alias when the logical padding already fits the physical tile, so the
            # source is never freed here; the caller drops its reference instead.
            flat = ttnn.pad(flat, [(0, 0), (0, 0), (0, TILE - rows), (0, 0)], 0.0)
        elif rows % TILE:
            flat = ttnn.pad(flat, [(0, 0), (0, 0), (0, _align_up(rows, TILE) - rows), (0, 0)], 0.0)
        del dims
        return flat

    # ------------------------------------------------------------------ prefill
    def ttnn_prefill_forward(self, tokens_tt, *, start_pos: int, page_table=None):
        """One prefill chunk through embeddings and the layer stack. Device tensors only.

        ``tokens_tt`` is ``[1, logical_len]`` uint32 ROW_MAJOR; the return is the replicated
        residual ``[1, logical_len, dim]``.
        """
        # DRAM interleaved, explicitly, because that is the decoder stage's inter-layer contract and
        # this op would otherwise *inherit* it: `ttnn.embedding` defaults its output memory config to
        # the INDICES tensor's (`output_mem_config.value_or(input_tensor_arg.memory_config())` in
        # embedding_device_operation.cpp), so the residual's placement would silently follow wherever
        # a caller happened to build its token tensor. Naming it here pins the contract instead.
        # (Tried and refuted as the cause of the 40-layer prefill L1 clash - see README section 5.2;
        # that was the per-layer router buffers.)
        x = ttnn.embedding(tokens_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, [1, int(tokens_tt.shape[-1]), self.dim])
        for layer in self.layers:
            nxt = layer.prefill_forward(x, start_pos=start_pos, page_table=page_table, chunk_size=self.prefill_chunk)
            ttnn.deallocate(x)
            x = nxt
        return x

    def prefill_forward_single(
        self,
        tokens,
        *,
        page_table=None,
        start_pos: int = 0,
        return_all_logits: bool = False,
        return_logits=True,
    ):
        """Prefill one user's prompt and return its logits.

        ``tokens`` is a torch ``[1, S]`` (or ``[S]``) tensor of token ids with **any** logical
        length; ``page_table`` is that user's ``[1, blocks]`` int32 ROW_MAJOR device tensor.

        ``return_logits`` selects the output form:

        ``True``       torch ``[1, 1, vocab]`` for the last prompt position, or ``[1, S, vocab]``
                       when ``return_all_logits`` is set;
        ``"device"``   the **device** logits ``[1, 1, 32, padded_vocab/tp]`` for the last prompt
                       position, sampler-ready and never composed on host. This is what the
                       generator's greedy path uses so the first generated token also comes out of
                       the on-device sampler rather than a host argmax;
        ``False``      nothing (cache-fill only).

        Chunking, physical padding, tail masking, cache fill, position bookkeeping and output
        slicing all happen here, which is what lets the public API accept a prompt length that
        divides nothing.
        """
        import torch

        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[0] != 1:
            raise ValueError("prefill_forward_single takes one user; use prefill_forward for a batch")
        seq_len = int(tokens.shape[1])
        if seq_len < 1:
            raise ValueError("prompt must have at least one token")
        if start_pos + seq_len > self.max_context:
            raise ValueError(
                f"prefill window [{start_pos}, {start_pos + seq_len}) exceeds the supported context {self.max_context}"
            )
        if start_pos % self.prefill_chunk:
            raise ValueError(f"start_pos {start_pos} must be a multiple of the prefill chunk {self.prefill_chunk}")

        self._use_pack(1)
        self.state_is_live = True
        chunk = self.prefill_chunk
        all_logits = [] if return_all_logits else None
        last_logits = None

        for offset in range(0, seq_len, chunk):
            logical = min(chunk, seq_len - offset)
            block = tokens[:, offset : offset + logical].to(torch.int32).contiguous()
            tokens_tt = ttnn.from_torch(
                block,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            hidden = self.ttnn_prefill_forward(tokens_tt, start_pos=start_pos + offset, page_table=page_table)
            ttnn.deallocate(tokens_tt)
            final_chunk = offset + logical >= seq_len
            if return_logits is False:
                ttnn.deallocate(hidden)
                continue
            if return_all_logits:
                all_logits.append(self._chunk_logits_to_host(hidden, logical))
                if final_chunk:
                    last_logits = all_logits[-1][:, -1:, :]
                ttnn.deallocate(hidden)
            elif final_chunk:
                # A one-token chunk makes this slice cover the whole tensor, and `ttnn.slice` then
                # returns an alias rather than a copy - so ownership is tracked instead of assumed,
                # exactly as the decoder's own `_slice_owned` does. Freeing `hidden` here on a
                # logical length of 1 (a one-token prompt, or a prompt one token past the internal
                # chunk) left the LM head reading a deallocated tensor.
                owned = logical > 1
                last = ttnn.slice(hidden, [0, logical - 1, 0], [1, logical, self.dim]) if owned else hidden
                rows = self._sampler_rows(last, 1)
                normed = self._final_norm(rows)
                device_logits = self._lm_head(normed)
                ttnn.deallocate(normed)
                if owned:
                    ttnn.deallocate(last)
                ttnn.deallocate(hidden)
                if return_logits == "device":
                    last_logits = device_logits
                else:
                    last_logits = self._logits_to_host(device_logits)[:, :1, :]
                    ttnn.deallocate(device_logits)
            else:
                ttnn.deallocate(hidden)

        if return_logits is False:
            return None
        if return_all_logits:
            return torch.cat(all_logits, dim=1)
        return last_logits

    #: Rows of hidden state pushed through the LM head at once on the all-logits path. 248320
    #: bfloat16 columns is ~0.5 MB of logits per row, so a full 2048-token chunk in one call would
    #: allocate a quarter of a gigabyte of device logits before anything is read back.
    ALL_LOGITS_ROW_CHUNK = 256

    def _chunk_logits_to_host(self, hidden, logical: int):
        import torch

        pieces = []
        for start in range(0, logical, self.ALL_LOGITS_ROW_CHUNK):
            stop = min(start + self.ALL_LOGITS_ROW_CHUNK, logical)
            piece = ttnn.slice(hidden, [0, start, 0], [1, stop, self.dim])
            rows = self._sampler_rows(piece, stop - start)
            normed = self._final_norm(rows)
            logits = self._lm_head(normed)
            ttnn.deallocate(normed)
            pieces.append(self._logits_to_host(logits)[:, : stop - start, :])
            ttnn.deallocate(logits)
        return torch.cat(pieces, dim=1)

    def _logits_to_host(self, logits):
        """Compose the vocab shards of ``[1, 1, R, padded_vocab/tp]`` into torch ``[1, R, vocab]``."""
        whole = ttnn.to_torch(
            logits, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=3)
        ).float()
        return whole[0, :, :, : self.vocab_size]

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        prompt_lens,
        kv_cache=None,
        return_all_logits: bool = False,
        start_pos=0,
        continue_from_state: bool = False,
    ):
        """Low-level batched prefill: one call, many users, **mixed prompt lengths**.

        ``tokens`` is a torch ``[B, P]`` tensor whose row ``u`` carries ``prompt_lens[u]`` real
        tokens followed by anything (the padding is never read). ``page_table`` is a torch or TT
        ``[B, blocks]`` mapping; ``kv_cache`` may be a caller-owned cache to attach. Users are
        prefilled one at a time at batch 1 and each user's DeltaNet state is merged into its decode
        slot, so a row that is not prefilled keeps whatever state it had (a fixed serving slot with
        an inactive row is unaffected by its neighbours).

        ``continue_from_state`` keeps the per-user prefill state instead of zeroing it, which is what
        a serving caller needs to chunk one long prompt across several calls: pass ``start_pos`` at
        the chunk's absolute offset and ``continue_from_state=True`` for every chunk after the first.
        The default (``False``) starts each user from a clean DeltaNet state, which is what a fresh
        request wants.

        Continuation is **batch 1 only**, and it raises rather than silently doing the wrong thing
        above that. Per-user prefill runs on one shared batch-1 state pack and copies the finished
        state *into* the user's decode slot (:meth:`_merge_prefill_state_into_slot`); there is no
        inverse copy, so at batch > 1 the second chunk of user 1 would continue from the state user 0
        left in the shared pack. Restoring slot -> pack before each user is the missing piece and
        belongs with the serving adapter that needs it.

        Returns torch logits ``[B, 1, vocab]``, or ``[B, P, vocab]`` when ``return_all_logits`` is
        set (positions past a user's prompt length are zero-filled).
        """
        import torch

        self.attach_kv_cache(kv_cache)
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch = int(tokens.shape[0])
        if prompt_lens is None:
            prompt_lens = [int(tokens.shape[1])] * batch
        prompt_lens = [int(v) for v in prompt_lens]
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens has {len(prompt_lens)} entries for batch {batch}")
        if batch > (self.max_batch_size or 1):
            raise ValueError(f"batch {batch} exceeds the allocated decode batch {self.max_batch_size}")
        starts = [int(start_pos)] * batch if isinstance(start_pos, int) else [int(v) for v in start_pos]
        if continue_from_state and batch > 1:
            raise ValueError(
                "continue_from_state is batch-1 only: per-user prefill shares one batch-1 state pack and "
                "there is no slot -> pack restore, so continuing user u would resume from user u-1's state. "
                "Chunk one user per call, or add the inverse copy first."
            )

        # Validate against the PHYSICAL extent, not the logical one: each prefill block is padded up
        # to 128 tokens and the cache is written for the padded block, so a logical length of 129
        # touches token 255, not token 128.
        def _physical_end(start: int, length: int) -> int:
            whole, tail = divmod(length, self.prefill_chunk)
            padded_tail = min(self.prefill_chunk, _align_up(tail, MC.PREFILL_ALIGN)) if tail else 0
            return start + whole * self.prefill_chunk + padded_tail

        physical_end = max(_physical_end(s, n) for s, n in zip(starts, prompt_lens))
        page_rows = self._page_table_rows(page_table, batch, max_position=physical_end - 1)
        out = []
        width = int(tokens.shape[1])
        for user in range(batch):
            if continue_from_state:
                self._use_pack(1)
            else:
                self._reset_prefill_pack()
            user_logits = self.prefill_forward_single(
                tokens[user : user + 1, : prompt_lens[user]],
                page_table=page_rows[user],
                start_pos=starts[user],
                return_all_logits=return_all_logits,
            )
            if return_all_logits and user_logits.shape[1] < width:
                pad = torch.zeros(1, width - user_logits.shape[1], user_logits.shape[2])
                user_logits = torch.cat([user_logits, pad], dim=1)
            out.append(user_logits)
            self._merge_prefill_state_into_slot(user)
            if page_rows[user] is not None:
                ttnn.deallocate(page_rows[user])
                page_rows[user] = None
        self._use_pack(self.max_batch_size)
        return torch.cat(out, dim=0)

    def _page_table_rows(self, page_table, batch: int, *, max_position: int | None = None):
        """One ``[1, blocks]`` int32 ROW_MAJOR device tensor per user.

        Accepts a torch table, a TT device table, or ``None`` (a stack with no ``full_attention``
        layer needs none). ``max_position`` is the highest absolute position this call will address;
        a table too narrow to hold it is rejected here rather than read out of bounds by the paged
        SDPA kernel.
        """
        import torch

        if page_table is None:
            if any(layer.is_full_attention for layer in self.layers):
                raise ValueError("this stack has full_attention layers and needs a page table")
            return [None] * batch
        if isinstance(page_table, ttnn.Tensor):
            host = ttnn.to_torch(
                page_table, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=0)
            )[: int(page_table.shape[0])]
        else:
            host = torch.as_tensor(page_table)
        if host.dim() == 1:
            host = host.unsqueeze(0)
        if host.shape[0] < batch:
            raise ValueError(f"page table has {host.shape[0]} rows for batch {batch}")
        if max_position is not None and max_position >= 0:
            # num_blocks_for_context rounds up to a multiple of 32 blocks, which the chunked/paged
            # SDPA kernel requires of the page table's row stick size - so this rejects both a row
            # too short for the position and a row the kernel could not read at all.
            needed = num_blocks_for_context(int(max_position) + 1, self.page_block_size)
            if int(host.shape[1]) < needed:
                raise ValueError(
                    f"page table has {int(host.shape[1])} blocks per user, but position {int(max_position)} "
                    f"needs {needed} blocks at page_block_size {self.page_block_size} (blocks are rounded up "
                    "to a multiple of 32: the paged SDPA kernel requires that row stick size)"
                )
        return [
            ttnn.from_torch(
                host[u : u + 1].to(torch.int32).contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for u in range(batch)
        ]

    def _reset_prefill_pack(self):
        self._use_pack(1)
        for layer in self.layers:
            layer.reset_state()

    def prefill_request_into_slot(
        self,
        tokens,
        *,
        page_table=None,
        slot: int = 0,
        start_pos: int = 0,
        return_logits=True,
        continue_from_state: bool = False,
    ):
        """Prefill **one** request and leave its state where the decode graph will read it.

        :meth:`prefill_forward_single` writes into the batch-1 prefill pack, which at
        ``max_batch_size > 1`` is a *different* set of tensors from the decode pack the captured
        trace is bound to. On its own it therefore leaves the 30 ``linear_attention`` layers'
        recurrent state in the wrong place: the paged KV half of the prompt is fine (the page table
        is per row), but decode would run those layers against whatever the decode pack holds. This
        is the same reset/merge sequence the batched :meth:`prefill_forward` runs per user, exposed
        for the single-request path so the two cannot drift.
        """
        if continue_from_state:
            self._use_pack(1)
        else:
            self._reset_prefill_pack()
        out = self.prefill_forward_single(
            tokens, page_table=page_table, start_pos=start_pos, return_logits=return_logits
        )
        self._merge_prefill_state_into_slot(slot)
        self._use_pack(self.max_batch_size)
        return out

    def _merge_prefill_state_into_slot(self, slot: int):
        """Copy the batch-1 prefill DeltaNet state into row ``slot`` of the decode-batch state.

        A no-op when the two packs are the same object (``max_batch_size == 1``), which is the
        batch-1 latency path this model is primarily tuned for.
        """
        batch = self.max_batch_size
        if batch == 1:
            return
        import torch

        mask = self._slot_mask(slot, batch)
        inverse = self._slot_mask(slot, batch, invert=True)
        src_pack = self._packs[1]
        dst_pack = self._packs[batch]
        for src, dst in zip(src_pack, dst_pack):
            if src["recurrent_state"] is None:
                continue
            for key in ("recurrent_state",):
                self._merge_rows(src[key], dst[key], mask["r"], inverse["r"], batch)
            for src_buf, dst_buf in zip(src["conv_state"], dst["conv_state"]):
                self._merge_rows(src_buf, dst_buf, mask["c"], inverse["c"], batch)
        del torch

    @staticmethod
    def _merge_rows(src, dst, mask, inverse, batch: int):
        wide = ttnn.repeat(src, ttnn.Shape([batch] + [1] * (len(src.shape) - 1)))
        selected = ttnn.multiply(wide, mask)
        ttnn.deallocate(wide)
        ttnn.multiply(dst, inverse, output_tensor=dst)
        ttnn.add(dst, selected, output_tensor=dst)
        ttnn.deallocate(selected)

    def _slot_mask(self, slot: int, batch: int, *, invert: bool = False):
        import torch

        key = ("mask", slot, batch, invert)
        cached = getattr(self, "_mask_cache", None)
        if cached is None:
            cached = self._mask_cache = {}
        if key in cached:
            return cached[key]
        base = torch.zeros(batch)
        base[slot] = 1.0
        if invert:
            base = 1.0 - base

        def upload(shape, dtype):
            return ttnn.from_torch(
                base.reshape(shape),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        cached[key] = {"r": upload((batch, 1, 1, 1), ttnn.float32), "c": upload((batch, 1, 1), ttnn.bfloat16)}
        return cached[key]

    # ------------------------------------------------------------------ decode
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Host TTNN tensors for the decode trace inputs. Never called inside a captured trace."""
        import torch

        batch = self.max_batch_size
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int32)
        if tokens.numel() > batch:
            raise ValueError(f"{tokens.numel()} tokens for a decode batch of {batch}")
        # The token buffer is always 32 wide because it doubles as ``ttnn.sampling``'s preallocated
        # output tensor, and that op emits one token per sampler row. The decode graph slices the
        # first ``max_batch_size`` entries back out before the embedding lookup.
        padded = torch.zeros(MAX_SAMPLING_BATCH, dtype=torch.int32)
        padded[: tokens.numel()] = tokens
        tokens_tt = ttnn.from_torch(
            padded.reshape(1, 1, 1, MAX_SAMPLING_BATCH),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        current_pos = torch.as_tensor(current_pos).reshape(-1).to(torch.int32)
        positions = torch.full((batch,), -1, dtype=torch.int32)
        positions[: current_pos.numel()] = current_pos
        pos_tt = ttnn.from_torch(
            positions,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # RoPE rows must stay non-negative even for inactive rows: the gather reads the table
        # unconditionally and only the KV write is skipped for a negative position.
        rot_tt = ttnn.from_torch(
            positions.clamp_min(0).reshape(1, batch),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        page_tt = None
        if page_table is not None:
            table = page_table if not isinstance(page_table, ttnn.Tensor) else None
            if table is None:
                raise TypeError("prepare_decode_inputs_host wants a torch page table")
            table = torch.as_tensor(table)
            if table.dim() == 1:
                table = table.unsqueeze(0)
            if table.shape[0] < batch:
                raise ValueError(
                    f"page table has {int(table.shape[0])} rows for a decode batch of {batch}; duplicating "
                    "a row would silently give two slots the same physical blocks"
                )
            highest = int(positions.max().item())
            if highest >= 0:
                needed = num_blocks_for_context(highest + 1, self.page_block_size)
                if int(table.shape[1]) < needed:
                    raise ValueError(
                        f"page table has {int(table.shape[1])} blocks per user, but position {highest} needs "
                        f"{needed} blocks at page_block_size {self.page_block_size} (rounded up to a multiple "
                        "of 32, the row stick size the paged SDPA kernel requires); a short row would make the "
                        "kernel read a block id past the end of the row"
                    )
            page_tt = ttnn.from_torch(
                table[:batch].to(torch.int32).contiguous(),
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return tokens_tt, pos_tt, rot_tt, page_tt

    def ttnn_decode_forward(self, tokens_tt, current_pos, rot_idxs, page_table=None, *, advance_positions=True):
        """Device-only decode step: token in, **sampler-ready vocab-sharded logits** out.

        Trace-safe by construction: every input is a persistent device tensor, no host value reaches
        the graph, and the two position tensors are advanced *inside* the graph with
        ``ttnn.plus_one`` so a fixed-step decode loop never refreshes them from host.

        ``tokens_tt`` is the ``[1, 1, 1, 32]`` uint32 ROW_MAJOR buffer that ``ttnn.sampling`` writes
        its result into, so the sampled token of replay N is literally the token input of replay
        N+1 — no readback, no reconstruction.
        """
        batch = self.max_batch_size
        self._use_pack(batch)
        slot = tokens_tt
        if int(tokens_tt.shape[-1]) != batch:
            slot = ttnn.slice(tokens_tt, [0, 0, 0, 0], [1, 1, 1, batch])
        ids = ttnn.reshape(slot, [batch, 1])
        # Same DRAM-interleaved residual contract as prefill, and named for the same reason; see
        # ttnn_prefill_forward.
        x = ttnn.embedding(ids, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.state_is_live = True
        for layer in self.layers:
            nxt = layer.decode_forward(x, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table)
            ttnn.deallocate(x)
            x = nxt

        rows = self._sampler_rows(x, batch)
        normed = self._final_norm(rows)
        logits = self._lm_head(normed)
        ttnn.deallocate(normed)

        if advance_positions:
            # Device-side position advance. `skip_negative_entries` leaves an inactive row's -1
            # sentinel alone, which is what keeps a fixed serving slot inactive across replays.
            #
            # `rot_idxs` has no such sentinel to skip - it is `max(current_pos, 0)` because the RoPE
            # gather reads its table unconditionally - so an inactive row's RoPE index does advance
            # while its position does not. That is harmless and *bounded*: the row's RoPE output is
            # multiplied into a Q/K it never writes to cache, and the index can only reach the number
            # of replays in one request, which the context bound caps below the table's
            # `align_up(max_context, prefill_chunk) + prefill_chunk` rows. Every request boundary
            # rewrites both tensors from host, so it cannot accumulate across requests. Deriving
            # `rot_idxs` from `current_pos` inside the graph would remove the asymmetry at the cost
            # of two ops per step; see doc/full_model/README.md limitation 7.
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rot_idxs)
        return logits

    def decode_logits_to_host(self, logits, batch: int | None = None):
        """Compose the vocab shards of a decode step into torch ``[batch, vocab]``.

        This is the **host-sampling compatibility** boundary. The optimized measured path never
        calls it: the sampler consumes ``logits`` on device and writes the token straight back into
        the decode token buffer.
        """
        batch = self.max_batch_size if batch is None else int(batch)
        whole = ttnn.to_torch(
            logits, mesh_composer=ttnn.concat_mesh_to_tensor_composer(self.mesh_device, dim=3)
        ).float()
        return whole[0, 0, :batch, : self.vocab_size]

    # ------------------------------------------------------------------ introspection
    def capability(self) -> dict:
        """The advertised capability contract, as the model actually built it."""
        return {
            "hf_model_id": HF_MODEL_ID,
            "mesh_shape": list(self.mesh_device.shape),
            "tp": self.tp,
            "num_layers": len(self.layers),
            "layer_indices": list(self.layer_indices),
            "hf_num_hidden_layers": self.cfg.num_hidden_layers,
            "reduced": self.is_reduced,
            "max_context": self.max_context,
            "hf_advertised_context": self.cfg.max_position_embeddings,
            "max_batch_size": self.max_batch_size,
            "prefill_chunk": self.prefill_chunk,
            "page_block_size": self.page_block_size,
            "num_blocks": self.num_blocks,
            "vocab_size": self.vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "policy": self.policy.name,
            "kv_cache_dtype": str(self.policy.kv_cache_dtype),
            "lm_head_dtype": str(self.lm_head_weights[0].dtype),
            "ccl_mode": MC.CCL_MODE,
            "router_mode": MC.ROUTER_MODE,
        }
