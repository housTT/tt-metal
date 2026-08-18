# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving adapter for ornith-ai/Ornith-1.0-35B on the 1x4 Blackhole ring.

Registered with the TT vLLM plugin as ``TTQwen3_5MoeForConditionalGeneration`` (the checkpoint's HF
architecture with the plugin's ``TT`` prefix) in
``vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models``.

What this file is
-----------------

Interface translation, and nothing else. Every device operation belongs to
:class:`~models.autoports.ornith_ai_ornith_1_0_35b.tt.generator.OrnithGenerator`'s serving
primitives, which in turn drive the same model trace, the same
:class:`~models.common.sampling.SamplingGenerator` and the same split-sampling token-out path the
full-model and datatype-sweep stages measured:

* one captured **model** trace, token in -> vocab-sharded sampler-ready logits out, advancing
  ``current_pos`` and the RoPE index on device with ``ttnn.plus_one``;
* one captured **sampling** trace writing ``tt_out_tok`` straight into the persistent decode token
  buffer, so the sampled token of replay *N* **is** the token input of replay *N+1*.

There is therefore no serving-only sampling strategy here, no host argmax, no full-logits readback on
the measured path, no generic top-k fallback for greedy, and no Python readback/writeback token
feedback loop. The precision policy is the selected datatype-sweep config, loaded by
``tt/precision_config.py`` from ``doc/datatype_sweep/selected_precision_config.json`` because
``policy`` is left unset - weight groups, activation and CCL dtypes, KV-cache dtype, compute
fidelities and (empty) layer exceptions all come from that one artifact.

Cache ownership
---------------

vLLM owns the attention KV cache. :meth:`TTQwen3_5MoeForConditionalGeneration.allocate_kv_cache` is
where the paged cache is created, at the block count vLLM sized, and the generator is built *around*
that cache (``kv_cache=`` on its constructor), so nothing allocates a second one and the captured
trace is bound to vLLM's tensors from the start. The recurrent half of this hybrid stack - the
``linear_attention`` layers' DeltaNet matrix and conv window - is fixed-size per slot and stays
model-owned, which is what the hybrid guidance asks for; :meth:`decode_forward` moves it when vLLM
condenses its batch.

Host sampling
-------------

The TT plugin decides per step whether sampling happens on device. It cannot on this 4-device mesh
whenever a request asks for log-probs (``LogProbsCalculator`` needs 8 or 32 devices), and never for
``min_p``, ``bad_words``, ``logit_bias``, ``allowed_token_ids``, ``min_tokens`` or structured output.
On those steps the plugin passes no ``sampling_params`` and expects logits, so this adapter returns
them - explicitly, and only then. That path is a compatibility mode for the shared tests; it is not
the measured path and does not replace it.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import OrnithGenerator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import (
    MAX_SAMPLING_BATCH,
    OrnithModel,
    load_text_config,
    resolve_model_path,
)
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import HF_MODEL_ID
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import num_blocks_for_context

#: Serving token pool, in tokens, when the deployment does not name one.
#:
#: This is the *shared* paged-KV budget vLLM divides into blocks - the analogue of
#: ``--num-gpu-blocks-override`` - and it is deliberately **not** the advertised context, which stays
#: at the HF 262144 (``doc/context_contract.json``) for every request. One full-context request fits,
#: and so does a batch of 32 requests up to 8192 tokens each. The cache costs a measured 5440 B per
#: token per device (``doc/datatype_sweep/capacity/selected.json``), so a deployment that wants more
#: concurrency at long context can raise ``ORNITH_MAX_TOKENS_ALL_USERS`` against the free DRAM the
#: same artifact records.
DEFAULT_MAX_TOKENS_ALL_USERS = 262144

#: Environment overrides. ``ORNITH_VLLM_LAYER_INDICES`` is the reduced serving target the bring-up
#: loop uses ("0,3" is one real layer of each kind); it is a debugging tool and never final evidence.
ENV_MAX_TOKENS = "ORNITH_MAX_TOKENS_ALL_USERS"
ENV_LAYER_INDICES = "ORNITH_VLLM_LAYER_INDICES"
ENV_NUM_LAYERS = "ORNITH_VLLM_NUM_LAYERS"


_NOT_THE_TT_PATH = (
    "this is the vLLM structural interface, not the TT execution path: the TT plugin's worker calls "
    "prefill_forward()/decode_forward() and the loader calls initialize_vllm_model()"
)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


def _resolve_snapshot(hf_config) -> Any:
    """The local checkpoint directory, from the environment the server was launched with."""
    for key in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(key)
        if value and os.path.isdir(os.path.expanduser(value)):
            return os.path.expanduser(value)
    name = getattr(hf_config, "_name_or_path", "") or HF_MODEL_ID
    if os.path.isdir(os.path.expanduser(name)):
        return os.path.expanduser(name)
    return resolve_model_path(None)


class TTQwen3_5MoeForConditionalGeneration:
    """The TT vLLM model class for Ornith-1.0-35B. **Text-only.**

    The plugin registers this class under both ``TTQwen3_5MoeForConditionalGeneration`` and the plain
    HF architecture, replacing upstream's implementation in a TT process. That is what makes vLLM
    describe the model the way the TT port actually is: text-only (so no multimodal processor is
    required of a class that cannot serve images, and image requests are refused by the API rather
    than silently answered from their text) and not ``IsHybrid`` (so vLLM does not raise
    ``cache_config.block_size`` to fit a GDN state into an attention page - this port keeps its
    recurrent state inside the model and its attention blocks are its own 64-token blocks).

    ``vllm_config``, :meth:`embed_input_ids`, :meth:`forward` and :meth:`compute_logits` are the
    structural interface ``vllm.model_executor.models.interfaces_base`` introspects to decide that a
    registered class is a text-generation model. Nothing calls them on TT: the plugin's worker drives
    :meth:`prefill_forward` and :meth:`decode_forward`, and the loader builds the model through
    :meth:`initialize_vllm_model`. They raise rather than return something plausible.
    """

    #: Read by ``vllm_tt_plugin.platform.TTPlatform.check_and_update_config``.
    #:
    #: ``supports_async_decode`` is claimed because the split really is implemented below -
    #: ``decode_forward(..., read_from_device=False)`` returns device tensors,
    #: ``read_decode_output(..., async_read=True)`` enqueues the minimal read behind the replay that
    #: produced it, and ``process_decode_output_host`` only formats - and because the stale-input
    #: contract that overlap depends on is enforced in :meth:`decode_forward`: a continuing row's
    #: token and position come from the device, never from a host view that may lag by one token.
    #: ``supports_prefix_caching`` stays False: it is not implemented and not tested.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    # ------------------------------------------------------------------ construction
    def __init__(
        self,
        model: OrnithModel,
        *,
        mesh_device,
        max_batch_size: int,
        max_model_len: int,
        hf_config,
        vllm_config=None,
    ):
        # `vllm_config` is part of the introspection surface above and is not used: the TT loader
        # builds this class through `initialize_vllm_model`, which gets the pieces it needs directly.
        del vllm_config
        self.model = model
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.max_batch_size = int(max_batch_size)
        self.max_model_len = int(max_model_len)
        #: Page-table width vLLM will hand us: ``ceil(max_model_len / block_size)``, which is also
        #: what the model's own block arithmetic produces for the advertised context.
        self.page_table_blocks = num_blocks_for_context(self.max_model_len, model.page_block_size)
        #: Built in :meth:`allocate_kv_cache`, around vLLM's cache.
        self.generator: OrnithGenerator | None = None

        #: Per-slot serving bookkeeping. ``_device_token_rows`` marks rows whose *device* token buffer
        #: entry is the authoritative last token (the sampling trace wrote it); ``_prefilled_rows``
        #: marks rows whose last token came from a prefill instead, so the device buffer is stale for
        #: them even though their position chain is continuous.
        self._device_token_rows = torch.zeros(self.max_batch_size, dtype=torch.bool)
        self._prefilled_rows = torch.zeros(self.max_batch_size, dtype=torch.bool)
        self._last_submit_was_prefill = True
        self._last_device_sampling: bool | None = None
        #: The plugin unpacks ``(logits, rope_deltas)`` from prefill for any config that declares
        #: ``mrope_section``, which this one does. The delta is 0 for every text request: HF's
        #: interleaved M-RoPE reduces exactly to 1-D RoPE when all three position grids agree, which
        #: is what ``tt/rope.py`` builds and ``tests/test_functional_decoder.py`` asserts.
        self.uses_mrope = "mrope_section" in (getattr(self._text_config(), "rope_parameters", None) or {})
        #: Steady-state serving counters, for the contract checks.
        self.serving_counters = {
            "prefill_calls": 0,
            "decode_calls": 0,
            "device_sampled_decodes": 0,
            "host_sampled_decodes": 0,
            "full_refreshes": 0,
            "page_table_only_refreshes": 0,
            "no_refresh_steps": 0,
            "slot_remaps": 0,
            "async_reads": 0,
        }

    def _text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)

    # --------------------------------------------------------------- vLLM introspection surface
    # `interfaces_base.is_text_generation_model` is a structural check: a `vllm_config` keyword on
    # `__init__`, `embed_input_ids`, `forward(input_ids, positions)` and `compute_logits`. It is what
    # `ModelConfig` validates `--runner generate` against. None of these run on TT.
    def embed_input_ids(self, input_ids):  # pragma: no cover - never called on TT
        raise NotImplementedError(_NOT_THE_TT_PATH)

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - never called on TT
        raise NotImplementedError(_NOT_THE_TT_PATH)

    def compute_logits(self, hidden_states):  # pragma: no cover - never called on TT
        raise NotImplementedError(_NOT_THE_TT_PATH)

    @staticmethod
    def _refuse_device_log_probs(sampling_params, where: str) -> None:
        """On-device log-probs need 8 or 32 devices, so this mesh must never be asked for them.

        The plugin already routes any log-prob request to its host sampler for exactly that reason
        (``check_perform_device_sampling``), so reaching here means a plugin change moved that boundary.
        Refusing beats returning a bare token tensor, which surfaces as an opaque assertion inside
        ``finalize_decode`` about a tuple it did not get.
        """
        enabled = getattr(sampling_params, "enable_log_probs", None)
        if enabled is None:
            return
        if hasattr(enabled, "any"):
            wanted = bool(enabled.any())  # a torch tensor, as the host-sampling path carries it
        elif isinstance(enabled, (list, tuple)):
            # The device-sampling path carries per-row *lists*, and a list of False values is still a
            # truthy object - `bool([False, False])` is True. Asking each row is the whole point.
            wanted = any(bool(value) for value in enabled)
        else:
            wanted = bool(enabled)
        if wanted:
            raise ValueError(
                f"{where}: on-device sampling was asked for log-probs, which need 8 or 32 devices. "
                "The plugin's host sampler owns log-probs on this 4-device mesh, and reaching here "
                "means that boundary moved."
            )

    @staticmethod
    def _carries_visual(kwargs: dict) -> bool:
        """True only when a request carries **real** pixels.

        The API refuses multimodal content for a text-only model before it reaches the worker, so this
        is the second line rather than the first: it turns any payload that does arrive into a clear
        refusal instead of an answer computed from the text alone.
        """
        for key in ("pixel_values", "pixel_values_videos"):
            value = kwargs.get(key)
            if value is not None and len(value) > 0 and value[0] is not None:
                return True
        return False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        n_layers=None,
        layer_indices=None,
        **kwargs: Any,
    ) -> "TTQwen3_5MoeForConditionalGeneration":
        """Build the TTNN model for serving. Called by ``vllm_tt_plugin.loader.TTModelLoader``.

        ``layer_indices`` and ``n_layers`` build the reduced bring-up target. The loader cannot pass
        either - it calls this with a fixed argument list - so the server path uses the environment
        variables below; the keywords exist for the tests and probes that build an adapter directly.
        """
        del kwargs
        if int(tt_data_parallel) != 1:
            raise ValueError(
                f"tt_data_parallel={tt_data_parallel} is not supported: this port is one TP=4/EP=4 "
                "model instance on a 1x4 Blackhole ring, not several submeshes"
            )
        if optimizations is not None:
            # `optimizations` is tt_transformers' performance/accuracy switch. This port's precision
            # is the datatype-sweep selection, one artifact loaded by `tt/precision_config.py`, and
            # `ORNITH_PRECISION_POLICY` is how a deployment selects a different point on the measured
            # frontier. Silently ignoring the flag would report a policy the model is not running.
            raise ValueError(
                f"optimizations={optimizations!r} is not supported; this port's precision comes from "
                "doc/datatype_sweep/selected_precision_config.json. Use ORNITH_PRECISION_POLICY to "
                "select another measured candidate."
            )
        batch = int(max_batch_size)
        if not 1 <= batch <= MAX_SAMPLING_BATCH:
            raise ValueError(f"max_num_seqs={batch} is outside [1, {MAX_SAMPLING_BATCH}] for this model")

        text_config = getattr(hf_config, "text_config", hf_config)
        advertised = int(getattr(text_config, "max_position_embeddings"))
        max_model_len = advertised if max_seq_len is None else int(max_seq_len)
        if max_model_len > advertised:
            raise ValueError(f"max_model_len={max_model_len} exceeds the advertised context {advertised}")

        snapshot = _resolve_snapshot(hf_config)
        # The plugin replaces the plain `Qwen3_5MoeForConditionalGeneration` architecture in a TT
        # process (work log section 3), which is process-global: *any* checkpoint declaring that
        # architecture resolves to this class, not just this one. That is safe but worth saying out
        # loud, because it is a scope this stage did not test. A foreign checkpoint does not silently
        # serve: `OrnithConfig.from_hf` refuses unknown `layer_types`, and the weight loader refuses a
        # checkpoint that does not nest under `model.language_model.`. What it will not do is fall
        # back to upstream's implementation.
        served_name = str(getattr(hf_config, "_name_or_path", "") or "")
        if HF_MODEL_ID.split("/")[-1].lower() not in (served_name + " " + str(snapshot)).lower():
            logger.warning(
                f"serving {served_name or snapshot!r} through the {HF_MODEL_ID} TT port: this process "
                "registers the TT class for the plain Qwen3.5-MoE architecture, so a different "
                "checkpoint of that architecture also resolves here. Only "
                f"{HF_MODEL_ID} was validated by the vLLM-integration stage."
            )
        # The model's own config parse, not vLLM's: the vision half of this checkpoint's config is
        # not this port's, and `load_text_config` is what every other stage builds from.
        model_config = load_text_config(snapshot)
        build: dict[str, Any] = {}
        indices = os.environ.get(ENV_LAYER_INDICES)
        if layer_indices is not None:
            build["layer_indices"] = [int(v) for v in layer_indices]
            logger.warning(
                f"layer_indices={build['layer_indices']}: building a REDUCED serving target. This is a "
                "bring-up tool; it is not the model and its output is not evidence."
            )
        elif indices:
            build["layer_indices"] = [int(v) for v in indices.replace(" ", "").split(",") if v != ""]
            logger.warning(
                f"{ENV_LAYER_INDICES}={indices}: building a REDUCED serving target. This is a bring-up "
                "tool; it is not the model and its output is not evidence."
            )
        elif n_layers is not None or _env_int(ENV_NUM_LAYERS) is not None:
            build["override_num_layers"] = int(n_layers if n_layers is not None else _env_int(ENV_NUM_LAYERS))
            logger.warning(f"building a REDUCED serving target: {build['override_num_layers']} layer(s)")

        logger.info(
            f"loading Ornith-1.0-35B for vLLM: max_num_seqs={batch}, max_model_len={max_model_len}, "
            f"mesh={tuple(mesh_device.shape)}, snapshot={snapshot}"
        )
        model = OrnithModel.from_pretrained(
            snapshot,
            mesh_device=mesh_device,
            hf_config=model_config,
            max_context=advertised,
            **build,
        )
        adapter = cls(
            model,
            mesh_device=mesh_device,
            max_batch_size=batch,
            max_model_len=max_model_len,
            hf_config=hf_config,
        )
        logger.info(f"model precision policy: {model.policy.name}, KV cache dtype {model.policy.kv_cache_dtype}")
        return adapter

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        """The shared paged-KV token budget vLLM turns into blocks.

        Not a context cap: every request may still use the whole advertised ``max_model_len``. The
        default holds one full-context request (which is what
        ``vllm_tt_plugin.worker._validate_tt_kv_cache_capacity`` requires) or a full batch of shorter
        ones, and ``ORNITH_MAX_TOKENS_ALL_USERS`` raises it for a deployment that wants both.
        """
        del model_name, num_devices, tt_data_parallel, kwargs
        override = _env_int(ENV_MAX_TOKENS)
        if override is not None:
            return override
        if max_num_seqs is not None and int(max_num_seqs) > MAX_SAMPLING_BATCH:
            raise ValueError(f"max_num_seqs={max_num_seqs} exceeds this model's sampler bound {MAX_SAMPLING_BATCH}")
        if max_model_len is None:
            return DEFAULT_MAX_TOKENS_ALL_USERS
        # At least one whole max_model_len request has to fit, and the default floor keeps a short-
        # context deployment from shrinking the pool below what a batch of 32 wants.
        return max(int(max_model_len), DEFAULT_MAX_TOKENS_ALL_USERS)

    @property
    def cache_path(self):
        return None

    # ------------------------------------------------------------------ KV cache (vLLM-owned)
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Create the paged KV cache vLLM sized, and build the generator around it.

        ``kv_cache_shape`` is ``(num_blocks, kv_heads_per_device, block_size, head_size)``: the TT
        plugin already divides the head count by the mesh, which lands on exactly the per-device
        shape this port's ``full_attention`` layers allocate (2 KV heads over 4 devices, one each on
        devices 0/1 and 2/3). ``dtype`` is vLLM's torch view of the cache; the cache is built at the
        **selected precision policy's** ``kv_cache_dtype`` instead, so serving allocates the cache the
        sweep measured rather than a wider one vLLM guessed.

        Returned in the readiness-contract shape - one entry per decoder layer, ``[k, v]`` for the
        ``full_attention`` layers and ``[]`` for the recurrent ones - and handed back to us unchanged
        on every prefill and decode call, which is what makes vLLM the cache's owner.
        """
        num_blocks, kv_heads, block_size, head_size = (int(v) for v in kv_cache_shape)
        model = self.model
        expected_heads = max(1, model.cfg.n_kv_heads // model.tp)
        if block_size != model.page_block_size:
            raise ValueError(
                f"vLLM block_size={block_size} but this model's paged cache uses "
                f"{model.page_block_size}-token blocks; launch the server with "
                f"--block_size {model.page_block_size}"
            )
        if head_size != model.cfg.head_dim:
            raise ValueError(f"vLLM head_size={head_size} != model head_dim {model.cfg.head_dim}")
        if kv_heads != expected_heads:
            raise ValueError(
                f"vLLM per-device kv_heads={kv_heads} != this model's {expected_heads} "
                f"(n_kv_heads={model.cfg.n_kv_heads} over tp={model.tp})"
            )
        if num_blocks < self.page_table_blocks:
            raise ValueError(
                f"the serving pool has {num_blocks} block(s) but one max_model_len "
                f"({self.max_model_len}) request needs {self.page_table_blocks}; raise "
                f"{ENV_MAX_TOKENS}"
            )
        logger.info(
            f"allocating the vLLM-owned paged KV cache: {num_blocks} blocks x {block_size} tokens "
            f"({num_blocks * block_size} tokens shared), dtype {model.policy.kv_cache_dtype} "
            f"(vLLM asked for {dtype}), {num_layers} attention layer slot(s)"
        )
        kv_cache = model.allocate_kv_cache(num_blocks)

        # The generator is built *around* vLLM's cache and with the page-table width vLLM will send,
        # so the decode trace is captured against the tensors and shapes serving actually uses. The
        # placeholder table is all zeros - vLLM's reserved null block - because every real table
        # arrives per step from the scheduler.
        self.generator = OrnithGenerator(
            model,
            tokenizer=None,
            max_batch_size=self.max_batch_size,
            cache_context=self.max_model_len,
            sampling_mode="device",
            kv_cache=kv_cache,
            page_table=torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32),
        )
        logger.info(
            f"generator ready: batch {self.max_batch_size}, page table "
            f"{self.max_batch_size}x{self.page_table_blocks}, caller-owned cache "
            f"(owns_cache={self.generator.owns_cache})"
        )
        return kv_cache

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False, **kwargs):
        """Compile the serving prefill path. Phase 1 (``enable_trace=False``) only.

        This port's prefill is eager by design in every stage - it is chunked, its program set is
        keyed by the *logical* prompt length, and no stage traces it - so there is nothing to capture
        in phase 2. What phase 1 buys is that the first real request does not compile the terminal
        norm, the LM head, the prefill sampling graph and the state merge while the decode traces are
        live; ``OrnithGenerator._ensure_traces_replay_safe`` would notice and re-capture, but the
        re-capture would land inside that request's latency.
        """
        del kwargs
        gen = self._require_generator()
        if enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        length = 64
        logger.info(f"warm-up: compiling the serving prefill path at {length} token(s)")
        tokens = torch.ones(1, length, dtype=torch.int32)
        table = self._warmup_page_table()
        gen.prefill_requests_into_slots(
            tokens,
            [length],
            [0],
            page_table=table,
            kv_cache=kv_cache,
            sample_on_device=False,
            ensure_traces=False,
        )
        if can_sample_on_device:
            gen.prefill_requests_into_slots(
                tokens,
                [length],
                [0],
                page_table=table,
                kv_cache=kv_cache,
                sample_on_device=True,
                ensure_traces=False,
                before_sample=lambda user, slot: self._apply_prefill_sampling(None, user, slot, tokens, [length]),
            )
        # The warm-up wrote prompt state into slot 0 and the null block; wipe it so the decode-trace
        # capture that follows is allowed to run (it refuses over a live prompt) and so no request
        # inherits it.
        gen.reset()
        self._reset_serving_state()

    def warmup_model_decode(
        self,
        *,
        kv_cache=None,
        max_batch_size=None,
        num_blocks=None,
        can_sample_on_device=False,
        enable_trace=False,
        **kwargs,
    ):
        """Compile, then capture, the serving decode path.

        Phase 1 (``enable_trace=False``) runs one eager decode step for every sampling shape a
        serving batch can take - greedy and non-greedy, with and without penalties - plus one that
        returns logits for the plugin's host sampler. That is what makes the phase-2 captures, and any
        later capture ``ensure_sampling_trace`` has to do when a batch flips between greedy and
        sampled, record-only: no program compiles inside a trace capture, and none compiles while a
        captured trace is live.

        Phase 2 captures the model decode trace and the greedy sampling trace over the persistent
        inputs, through the same ``_ensure_decode_trace`` the readiness path uses.
        """
        del kwargs
        gen = self._require_generator()
        if max_batch_size is not None and int(max_batch_size) != self.max_batch_size:
            raise ValueError(f"warmup batch {max_batch_size} != model batch {self.max_batch_size}")
        if num_blocks is not None and int(num_blocks) != self.page_table_blocks:
            raise ValueError(
                f"warmup page table width {num_blocks} != {self.page_table_blocks} "
                f"(ceil({self.max_model_len} / {self.model.page_block_size}))"
            )
        table = self._warmup_page_table()
        if not enable_trace:
            tokens = torch.zeros(self.max_batch_size, dtype=torch.int32)
            positions = torch.zeros(self.max_batch_size, dtype=torch.int32)
            logger.info("warm-up: compiling the eager decode path (logits out)")
            gen.decode_forward(tokens, positions, page_table=table, kv_cache=kv_cache, enable_trace=False)
            if can_sample_on_device:
                for label, params in _warmup_sampling_variants():
                    logger.info(f"warm-up: compiling the eager decode + sampling path ({label})")
                    gen.sampling.apply_decode_state(
                        [params],
                        reset_batch=True,
                        prompt_tokens=torch.zeros(self.max_batch_size, 1, dtype=torch.int32),
                        output_tokens=torch.zeros(self.max_batch_size, 1, dtype=torch.int32),
                    )
                    gen.invalidate_sampling_params_cache()
                    gen.sampling.seed_manager.get_new_values(list(range(self.max_batch_size)))
                    gen.decode_forward(
                        tokens,
                        positions,
                        page_table=table,
                        kv_cache=kv_cache,
                        enable_trace=False,
                        sample_on_device=True,
                    )
            if can_sample_on_device:
                # Leave the sampler on greedy: `_ensure_decode_trace` captures the sampling slot for
                # whatever parameters are current, and greedy is the one a benchmark and most requests
                # use. Any other slot is captured on demand, record-only, by `ensure_sampling_trace`.
                _, greedy = _warmup_sampling_variants()[0]
                gen.sampling.apply_decode_state([greedy], reset_batch=True)
                gen.invalidate_sampling_params_cache()
            gen.reset()
            self.model.reset_state()
            self._reset_serving_state()
            return
        logger.info("warm-up: capturing the model decode trace and the sampling trace")
        gen.ensure_serving_traces()
        gen.ensure_sampling_trace()
        self._reset_serving_state()
        logger.info(
            f"warm-up done: program cache {self.mesh_device.num_program_cache_entries()} entries, "
            f"{gen.trace_recaptures} trace re-capture(s), {gen.sampling_trace_captures} sampling capture(s)"
        )
        self._write_serving_capability()
        # ...and again when the engine-core process exits, so the counters describe the traffic it served
        # rather than the warm-up that wrote the first copy. The warm-up copy is kept as
        # `vllm_serving_capability.json`; this one lands beside it with a `_final` suffix, because a
        # crashed process should leave the warm-up evidence rather than nothing.
        atexit.register(self._write_serving_capability, suffix="_final")

    def _write_serving_capability(self, suffix: str = "") -> None:
        """Record what this serving build is, next to the readiness artifacts.

        Written from inside the engine-core process, because that is the only place that can read the
        *built* model rather than a config file: the precision policy every layer actually carries, the
        KV-cache dtype the cache was allocated at, the capability flags the plugin read, and the
        page-table geometry. A stage that reports a policy it did not serve is the failure this file
        exists to make impossible.

        Written twice: once at the end of warm-up, and once at process exit with ``suffix="_final"``, so
        the counters (refreshes, no-refresh steps, slot remaps, async reads, device- against
        host-sampled decodes) describe the traffic the server actually handled. The warm-up copy is
        always complete even if the process dies later.
        """
        import json

        path = Path(__file__).resolve().parents[1] / "readiness_vllm" / f"vllm_serving_capability{suffix}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.serving_capability(), indent=1) + "\n")
            logger.info(f"wrote the serving capability report to {path}")
        except OSError as exc:  # pragma: no cover - never fail a warm-up over an artifact
            logger.warning(f"could not write the serving capability report: {exc}")

    def _warmup_page_table(self) -> torch.Tensor:
        """An all-null-block page table of exactly the serving shape.

        Zeros are vLLM's reserved placeholder block (``BlockPool.null_block``), so a warm-up write
        lands where no request's blocks are and nothing reads it.
        """
        return torch.zeros(self.max_batch_size, self.page_table_blocks, dtype=torch.int32)

    def _reset_serving_state(self) -> None:
        self._device_token_rows[:] = False
        self._prefilled_rows[:] = False
        self._last_submit_was_prefill = True
        self._last_device_sampling = None

    def _require_generator(self) -> OrnithGenerator:
        if self.generator is None:
            raise RuntimeError("allocate_kv_cache() must run before any forward: it builds the generator")
        return self.generator

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        enable_trace: bool = False,
        prompt_lens=None,
        start_pos=None,
        page_tables_per_layer=None,
        sampling_params=None,
        empty_slots=None,
        **kwargs: Any,
    ):
        """One serving prefill step: every scheduled prompt, each into its own decode slot.

        ``enable_trace`` is accepted and ignored: this port's prefill is eager in every stage (see
        :meth:`warmup_model_prefill`). Multimodal payloads in ``kwargs`` are ignored - vLLM attaches
        empty placeholders to text requests for a checkpoint whose config has a vision tower.
        """
        del enable_trace
        if self._carries_visual(kwargs):
            raise ValueError(
                "this is a text-only TT port of Ornith-1.0-35B: the checkpoint's vision tower is not "
                "implemented, so an image or video request cannot be served. Send text only."
            )
        if page_tables_per_layer is not None:
            raise ValueError(
                "per-layer page tables are for vLLM's hybrid KV-cache groups; this adapter declares "
                "one full-attention group and its recurrent layers are model-owned"
            )
        gen = self._require_generator()
        tokens = torch.as_tensor(tokens)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        count = int(tokens.shape[0])
        slots = list(range(count)) if empty_slots is None else [int(s) for s in empty_slots]
        device_sampling = sampling_params is not None
        if device_sampling:
            self._refuse_device_log_probs(sampling_params, "prefill")
        lens = None if prompt_lens is None else [int(v) for v in prompt_lens]
        logger.info(
            f"prefill: {count} request(s), lengths {lens}, slots {slots}, "
            f"{'device' if device_sampling else 'host'} sampling"
        )
        out = gen.prefill_requests_into_slots(
            tokens,
            lens,
            slots,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=start_pos,
            sample_on_device=device_sampling,
            before_sample=(
                (lambda user, slot: self._apply_prefill_sampling(sampling_params, user, slot, tokens, lens))
                if device_sampling
                else None
            ),
        )
        for slot in slots:
            self._prefilled_rows[slot] = True
            self._device_token_rows[slot] = False
        self._last_submit_was_prefill = True
        self.serving_counters["prefill_calls"] += 1
        if self.uses_mrope:
            # Text-only: HF's interleaved M-RoPE reduces to 1-D RoPE, and the model applies positions
            # itself, so there is no per-request delta for vLLM to carry.
            return out, torch.zeros(count, dtype=torch.long)
        return out

    def _apply_prefill_sampling(self, sampling_params, user: int, slot: int, tokens, prompt_lens) -> None:
        """Push one prefilling request's sampling parameters, penalties and seed to the device.

        The prompt's logits occupy sampler **row 0** (the model pads one row of hidden state to the
        sampler's 32), so this request's parameters are broadcast to every row and its seed replicated
        with them: row 0 then samples with exactly what the request asked for, whatever slot it will
        decode in.
        """
        from models.common.sampling import SamplingParams, broadcast_sampling_params, format_sampling_params

        gen = self._require_generator()
        width = gen.sampling.tt_sampling.max_batch_size
        if sampling_params is None:
            params = format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), width)
        else:
            params = format_sampling_params(broadcast_sampling_params(sampling_params, user, slot_len=width), width)
        length = int(tokens.shape[1]) if prompt_lens is None else int(prompt_lens[user])
        prompt = tokens[user : user + 1, :length].to(torch.long).repeat(width, 1)
        gen.sampling.apply_prefill_state(sampling_params=params, prompt_tokens=prompt, empty_slots=[slot])
        gen.invalidate_sampling_params_cache()

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        *,
        tokens,
        page_table,
        kv_cache,
        start_pos,
        enable_trace: bool = True,
        read_from_device: bool = True,
        page_tables_per_layer=None,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch=None,
        slot_remap=None,
        rope_deltas_all_users=None,
        **kwargs: Any,
    ):
        """One serving decode step over the fixed slot batch.

        Rows are slots: row ``i`` is device state slot ``i``, its position is its absolute KV position
        and ``-1`` means the slot is idle (``ttnn.plus_one(..., skip_negative_entries=True)`` keeps it
        idle across replays, and its KV is not written). ``page_table`` is ``[slots, blocks]``.

        The refresh policy is the whole point of the method, and it is the async-decode contract:

        * **steady state** - nothing is copied to the device. The token comes from ``tt_out_tok`` and
          the positions from ``ttnn.plus_one``, both inside the trace, so a host view that lags by one
          token cannot be staged over them;
        * **page table changed** - only the page table is copied. A growing request gets new blocks,
          and that is scheduler state the device cannot derive;
        * **layout changed** (``reset_batch``, a slot remap, a freshly prefilled slot, a switch to or
          from host sampling) - tokens and positions are re-staged, merged per row against what the
          device holds so that continuing rows keep the device's authoritative pair.

        ``read_from_device=False`` returns device tensors for the async split.
        """
        del rope_deltas_all_users, kwargs
        if page_tables_per_layer is not None:
            raise ValueError("per-layer page tables are not used by this adapter (single KV group)")
        gen = self._require_generator()
        device_sampling = sampling_params is not None
        if device_sampling:
            self._refuse_device_log_probs(sampling_params, "decode")
        if not enable_trace:
            # Only the warm-up's compile phase asks for this; serving runs the captured trace.
            out = gen.decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=False,
                sample_on_device=device_sampling,
            )
            return out if device_sampling else out.unsqueeze(1)
        if kv_cache is not None:
            self.model.attach_kv_cache(kv_cache)

        remap = None
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap).reshape(-1).to(torch.int64)
            if remap.numel() < self.max_batch_size:
                # The plugin sends the full slot width; a short one is padded with identity rather than
                # silently shrinking this adapter's per-slot bookkeeping to the remap's length.
                remap = torch.cat([remap, torch.arange(remap.numel(), self.max_batch_size, dtype=torch.int64)])
            remap = remap[: self.max_batch_size]
            if not torch.equal(remap, torch.arange(remap.numel())):
                gen.remap_serving_slots(remap)
                # Unconditionally, not only on a device-sampling step: the per-slot RNG counters are
                # persistent, so a condense that happened while this step sampled on host would leave the
                # *next* device-sampled step reading another request's stream.
                gen.sampling.seed_manager.apply_slot_remap(remap)
                self._device_token_rows = self._device_token_rows[remap]
                self._prefilled_rows = self._prefilled_rows[remap]
                self.serving_counters["slot_remaps"] += 1
            else:
                remap = None

        sampling_mode_changed = self._last_device_sampling is not None and self._last_device_sampling != device_sampling
        full_refresh = bool(
            reset_batch
            or self._last_submit_was_prefill
            or (not device_sampling)
            or sampling_mode_changed
            or (remap is not None)
            or bool(self._prefilled_rows.any())
        )
        if device_sampling:
            self._apply_decode_sampling(
                sampling_params,
                start_pos,
                reset_batch=full_refresh,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        gen.ensure_replay_safe()
        if device_sampling:
            gen.ensure_sampling_trace()
        trust = self._device_token_rows & ~self._prefilled_rows
        info = gen.stage_serving_decode_inputs(
            tokens,
            start_pos,
            page_table,
            full_refresh=full_refresh,
            device_token_rows=trust if device_sampling else None,
        )
        out = gen.submit_serving_decode(sample_on_device=device_sampling)

        self.serving_counters["decode_calls"] += 1
        self.serving_counters["device_sampled_decodes" if device_sampling else "host_sampled_decodes"] += 1
        if info["tokens"]:
            self.serving_counters["full_refreshes"] += 1
        elif info["page_table"]:
            self.serving_counters["page_table_only_refreshes"] += 1
        else:
            self.serving_counters["no_refresh_steps"] += 1
        self._prefilled_rows[:] = False
        self._device_token_rows[:] = bool(device_sampling)
        self._last_submit_was_prefill = False
        self._last_device_sampling = device_sampling

        if not read_from_device:
            return out
        return self.process_decode_output_host(out, is_tokens=device_sampling)

    def _apply_decode_sampling(self, sampling_params, start_pos, *, reset_batch, prompt_tokens, output_tokens) -> None:
        """Per-row parameters, penalty state and seeds for one on-device decode step."""
        from models.common.sampling import format_sampling_params

        gen = self._require_generator()
        sampling = gen.sampling
        sampling.apply_decode_state(
            [sampling_params],
            reset_batch=bool(reset_batch),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        gen.invalidate_sampling_params_cache()
        positions = [int(v) for v in torch.as_tensor(start_pos).reshape(-1)]
        slots = sampling.seed_manager.max_batch_size
        active = [i for i, pos in enumerate(positions[:slots]) if pos >= 0]
        if active:
            seeds = format_sampling_params(sampling_params, sampling.tt_sampling.max_batch_size).seed
            # Register each request's explicit seed and tie its RNG counter to the absolute decode
            # position, so a request that vLLM moves between slots keeps one reproducible stream.
            sampling.seed_manager.reset_seed_from_slots_if_needed(seeds, active)
            sampling.seed_manager.align_seed_counters_to_positions(seeds, active, positions)
        sampling.seed_manager.get_new_values(active)

    # ------------------------------------------------------------------ async split
    def read_decode_output(self, tt_out, async_read: bool = False):
        """Read one decode step's output to host. The deferred half of the async split.

        ``async_read`` enqueues the device->host copy on the same command queue as the replays that
        produced it and records an event behind it, without waiting: the queue is in order and the
        *next* step's replay is enqueued after this copy, so the copy observes exactly this step's
        token while the host's wait overlaps device work that is already running.
        """
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if not async_read:
            return [tt_out]
        host = tt_out.cpu(blocking=False)
        event = ttnn.record_event(self.mesh_device, 0)
        self.serving_counters["async_reads"] += 1
        return [host], [event]

    def process_decode_output_host(self, tt_out, is_tokens: bool = False):
        """Host formatting only - no device work is submitted here.

        Tokens come back as ``[slots]`` ids, logits as ``[slots, 1, vocab]``; the plugin indexes both
        by row and takes the active prefix.
        """
        tensor = tt_out[0] if isinstance(tt_out, (list, tuple)) else tt_out
        if isinstance(tensor, torch.Tensor):
            return tensor
        gen = self._require_generator()
        if is_tokens:
            return gen.tokens_from(tensor).to(torch.int32)
        return gen.logits_from(tensor)

    # ------------------------------------------------------------------ introspection
    def serving_capability(self) -> dict:
        """What this serving build actually is. Written to the stage's evidence."""
        gen = self.generator
        report = {
            "adapter": type(self).__name__,
            "architecture": "TTQwen3_5MoeForConditionalGeneration",
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_batch_size,
            "page_table_blocks": self.page_table_blocks,
            "uses_mrope": self.uses_mrope,
            "model_capabilities": dict(self.model_capabilities),
            "kv_cache_owner": "vllm",
            "recurrent_state_owner": "model",
            "capability": self.model.capability(),
        }
        if gen is not None:
            report["generator"] = {
                "sampling_mode": gen.sampling_mode,
                "owns_cache": gen.owns_cache,
                "trace_recaptures": gen.trace_recaptures,
                "sampling_trace_captures": gen.sampling_trace_captures,
                "counters": dict(gen.counters),
            }
        report["serving_counters"] = dict(self.serving_counters)
        return report


def _warmup_sampling_variants():
    """The four sampling shapes a serving batch can take, as ``(label, SamplingParams)``.

    ``SamplingGenerator`` keys its traces on (penalties, log-probs, force-argmax) and log-probs cannot
    run on device on a 4-device mesh, so these four are the whole space: greedy or not, penalised or
    not. Compiling all of them eagerly at warm-up is what lets every later capture be record-only.
    """
    from models.common.sampling import SamplingParams

    greedy = dict(temperature=0.0, top_k=1, top_p=1.0)
    random = dict(temperature=0.8, top_k=20, top_p=0.9)
    return [
        ("greedy", SamplingParams(**greedy)),
        (
            "greedy + penalties",
            SamplingParams(**greedy, presence_penalty=0.2, frequency_penalty=0.2, repetition_penalty=1.1),
        ),
        ("sampled", SamplingParams(**random)),
        (
            "sampled + penalties",
            SamplingParams(**random, presence_penalty=0.2, frequency_penalty=0.2, repetition_penalty=1.1),
        ),
    ]
