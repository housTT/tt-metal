# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving-adapter suite for ornith-ai/Ornith-1.0-35B on the 1x4 Blackhole ring.

The device cases drive ``tt/generator_vllm.py`` through the **plugin-facing API** - the same calls
``vllm_tt_plugin``'s loader, worker and model runner make - without a server in the way, on the reduced
two-layer target (one real ``linear_attention`` layer, one real ``full_attention`` layer, real weights,
real cache and page-table shapes, real terminal norm / LM head / sampling / trace behaviour). What they
pin is mechanism, not stack depth: cache ownership, per-slot prefill, the decode refresh policy that
makes async scheduling safe, the async read split, and the slot remap. Serving accuracy and performance
evidence lives in ``doc/vllm_integration/``, measured on the full 40-layer model through a real server.

The host-only cases at the top need no device: registration, the capability flags the plugin reads, and
the interface vLLM introspects.

Run:

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py -x -q
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import model as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator_vllm import (
    DEFAULT_MAX_TOKENS_ALL_USERS,
    TTQwen3_5MoeForConditionalGeneration,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
ADAPTER_SOURCE = MODEL_DIR / "tt" / "generator_vllm.py"

#: The reduced serving target: layer 0 is ``linear_attention``, layer 3 is ``full_attention``.
PROBE_LAYERS = [0, 3]
TEST_CONTEXT = 4096
TEST_BATCH = 4

#: The architecture the TT plugin must resolve to this adapter, and the dotted path it registers.
TT_ARCH = "TTQwen3_5MoeForConditionalGeneration"
HF_ARCH = "Qwen3_5MoeForConditionalGeneration"
ADAPTER_TARGET = "models.autoports.ornith_ai_ornith_1_0_35b.tt.generator_vllm:TTQwen3_5MoeForConditionalGeneration"


# --------------------------------------------------------------------------------------
# host-only: registration, flags, interface
# --------------------------------------------------------------------------------------
def test_the_plugin_registers_this_adapter_for_both_architectures():
    """Both registrations matter, and for different reasons.

    ``TT...`` is the plugin's own convention and what ``check_and_update_config`` validates. The plain
    HF architecture is registered too, replacing upstream's class, because everything ``ModelConfig``
    decides before the plugin prefixes the name - ``is_multimodal_model``, ``IsHybrid`` and therefore
    ``cache_config.block_size`` - is decided from the class it resolves then. See the module docstring
    of the adapter and ``doc/vllm_integration/work_log.md`` §3.
    """
    pytest.importorskip("vllm_tt_plugin", reason="the TT vLLM plugin is not installed in this env")
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_tt_plugin.platform import register_tt_models

    register_tt_models()
    archs = ModelRegistry.get_supported_archs()
    assert TT_ARCH in archs
    assert HF_ARCH in archs
    module, _, class_name = ADAPTER_TARGET.partition(":")
    for arch in (TT_ARCH, HF_ARCH):
        registered = ModelRegistry.models[arch]
        assert (registered.module_name, registered.class_name) == (
            module,
            class_name,
        ), f"{arch} is registered as {registered.module_name}:{registered.class_name}"


def test_the_adapter_is_the_text_generation_interface_vllm_introspects():
    """``ModelConfig`` validates ``--runner generate`` structurally, and refuses the model otherwise.

    It also must not look multimodal: this port has no vision tower, and a multimodal answer here is
    what makes vLLM demand a processor of it and hand it image requests it cannot serve.
    """
    pytest.importorskip("vllm", reason="vLLM is not installed in this env")
    from vllm.model_executor.models.interfaces import supports_multimodal
    from vllm.model_executor.models.interfaces_base import is_text_generation_model, is_vllm_model

    assert is_vllm_model(TTQwen3_5MoeForConditionalGeneration)
    assert is_text_generation_model(TTQwen3_5MoeForConditionalGeneration)
    assert not supports_multimodal(TTQwen3_5MoeForConditionalGeneration)


def test_the_capability_flags_are_the_ones_this_stage_proved():
    flags = TTQwen3_5MoeForConditionalGeneration.model_capabilities
    assert flags["supports_sample_on_device"] is True
    assert flags["supports_async_decode"] is True, "the split submit/read/process path is implemented"
    assert flags["supports_prefix_caching"] is False, "not implemented and not tested, so not claimed"


def test_the_adapter_implements_the_shared_vllm_adapter_contract():
    """Every method ``models/common/readiness_check/contract_vllm.py`` names, with its keywords."""
    from models.common.readiness_check import contract_vllm

    protocol = contract_vllm.VllmGeneratorAdapter
    for name in (
        "initialize_vllm_model",
        "get_max_tokens_all_users",
        "allocate_kv_cache",
        "warmup_model_prefill",
        "warmup_model_decode",
        "prefill_forward",
        "decode_forward",
        "read_decode_output",
        "process_decode_output_host",
    ):
        assert hasattr(protocol, name), f"the shared contract no longer names {name}"
        member = getattr(TTQwen3_5MoeForConditionalGeneration, name, None)
        assert callable(member), f"the adapter is missing {name}"
    decode = inspect.signature(TTQwen3_5MoeForConditionalGeneration.decode_forward).parameters
    for keyword in ("tokens", "page_table", "kv_cache", "start_pos", "enable_trace", "read_from_device"):
        assert keyword in decode, f"decode_forward must accept {keyword}"
    assert decode["read_from_device"].kind is inspect.Parameter.KEYWORD_ONLY
    prefill = inspect.signature(TTQwen3_5MoeForConditionalGeneration.prefill_forward).parameters
    for keyword in ("tokens", "page_table", "kv_cache", "prompt_lens", "start_pos", "sampling_params", "empty_slots"):
        assert keyword in prefill, f"prefill_forward must accept {keyword}"


def test_the_adapter_has_no_sampling_path_of_its_own():
    """The measured serving path is the full-model generator's split sampling, nothing else.

    An AST check, deliberately: the ways this regresses are all *additions* - a host ``argmax`` over a
    logits readback, a top-k pick of its own, a trace replay the adapter drives itself - and each is
    visible here before it is visible in a benchmark. Comments and docstrings are invisible to this
    check, so the words may still be discussed in prose above.
    """
    import ast

    tree = ast.parse(ADAPTER_SOURCE.read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            called.add(node.attr)
        elif isinstance(node, ast.Name):
            called.add(node.id)
    for forbidden, why in {
        "argmax": "a host argmax in the adapter would replace the on-device sampler",
        "topk": "picking a top-k in the adapter would be a second sampling strategy",
        "execute_trace": "trace replay belongs to the generator",
        "decode_logits_to_host": "logits composition belongs to the generator's host-sampling mode",
        "sample": "the adapter must not call the sampler directly; the generator owns the split path",
        "capture_trace": "trace capture belongs to the generator",
    }.items():
        assert forbidden not in called, f"{forbidden}: {why}"


def test_the_serving_token_pool_holds_one_full_context_request(expect_error):
    """``_validate_tt_kv_cache_capacity`` refuses a pool that cannot hold one ``max_model_len`` request."""
    advertised = 262144
    pool = TTQwen3_5MoeForConditionalGeneration.get_max_tokens_all_users(max_model_len=advertised, max_num_seqs=32)
    assert pool >= advertised
    assert TTQwen3_5MoeForConditionalGeneration.get_max_tokens_all_users() == DEFAULT_MAX_TOKENS_ALL_USERS
    # A shorter deployment context must not shrink the pool below the default floor.
    assert TTQwen3_5MoeForConditionalGeneration.get_max_tokens_all_users(max_model_len=4096, max_num_seqs=1) >= (
        DEFAULT_MAX_TOKENS_ALL_USERS
    )
    with expect_error(ValueError, "sampler bound"):
        TTQwen3_5MoeForConditionalGeneration.get_max_tokens_all_users(max_model_len=advertised, max_num_seqs=64)


def test_the_log_probs_refusal_reads_the_rows_not_the_container(expect_error):
    """A list of ``False`` is a truthy object, and getting that wrong refuses every device-sampled step.

    The device-sampling path carries per-row *lists* (the plugin ``.tolist()``s its tensors) while the
    host path carries tensors, so both shapes have to be asked row by row. This is a regression test:
    the first version of this check did ``bool(enabled)`` and broke every greedy prefill.
    """
    from models.common.sampling import SamplingParams

    refuse = TTQwen3_5MoeForConditionalGeneration._refuse_device_log_probs

    def params(enable_log_probs):
        return SamplingParams(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, enable_log_probs=enable_log_probs)

    refuse(params([False] * 4), "decode")  # must not raise
    refuse(params(torch.zeros(4, dtype=torch.bool)), "decode")  # must not raise
    refuse(params(None), "decode")  # must not raise
    for wanted in ([False, True, False, False], torch.tensor([False, True, False, False])):
        with expect_error(ValueError, "log-probs"):
            refuse(params(wanted), "decode")


def test_a_visual_payload_is_refused_rather_than_answered_from_the_text():
    adapter = TTQwen3_5MoeForConditionalGeneration.__new__(TTQwen3_5MoeForConditionalGeneration)
    assert not adapter._carries_visual({})
    assert not adapter._carries_visual({"pixel_values": []})
    assert not adapter._carries_visual({"pixel_values": [None]})
    assert adapter._carries_visual({"pixel_values": [torch.zeros(1)]})
    assert adapter._carries_visual({"pixel_values_videos": [torch.zeros(1)]})


def test_a_reduced_target_does_not_overwrite_the_served_capability_report():
    """A two-layer bring-up adapter must not write into `readiness_vllm/`.

    The capability report is written from inside the engine-core process, which is what makes it
    evidence about the *served* model. This suite builds reduced adapters and warms them up, and the
    writer used to fire there too — replacing the served artifacts with a `reduced: true` report of a
    model nothing served, and quietly breaking the committed evidence's attribution.

    The assertion is on the bytes of the real artifacts, so a regression here fails loudly instead of
    corrupting them silently.
    """
    served = MODEL_DIR / "readiness_vllm"
    before = {
        name: (served / name).read_bytes()
        for name in ("vllm_serving_capability.json", "vllm_serving_capability_final.json")
        if (served / name).exists()
    }

    class ReducedStub:
        _write_serving_capability = TTQwen3_5MoeForConditionalGeneration._write_serving_capability

        def serving_capability(self):
            return {"capability": {"reduced": True, "layer_indices": [0, 3], "policy": "stub"}}

    ReducedStub()._write_serving_capability()
    ReducedStub()._write_serving_capability(suffix="_final")

    for name, blob in before.items():
        assert (served / name).read_bytes() == blob, f"a reduced build rewrote readiness_vllm/{name}"


# --------------------------------------------------------------------------------------
# device: the plugin-facing API on the reduced target
# --------------------------------------------------------------------------------------
DEVICE_PARAMS = [
    {
        "l1_small_size": M.DEFAULT_L1_SMALL_SIZE,
        "trace_region_size": M.DEFAULT_TRACE_REGION_SIZE,
        "fabric_config": MC.DEFAULT_FABRIC_CONFIG,
        "fabric_router_config": MC.fabric_router_config(),
    },
]


def _require_weights():
    try:
        available = (M.resolve_model_path() / "model.safetensors.index.json").is_file()
    except Exception:  # noqa: BLE001 - offline / not downloaded
        available = False
    if not available:
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")


_ADAPTER: dict = {}


def serving_adapter(mesh_device, batch=TEST_BATCH):
    """A cached reduced-target adapter, built exactly the way the plugin builds one."""
    _require_weights()
    key = (id(mesh_device), batch)
    if key in _ADAPTER:
        adapter = _ADAPTER[key]
        adapter.generator.reset()
        adapter._reset_serving_state()
        return adapter
    for stale in [k for k in _ADAPTER if k[0] == id(mesh_device)]:
        _ADAPTER.pop(stale).generator.teardown()

    hf_config = M.load_text_config(M.resolve_model_path())
    adapter = TTQwen3_5MoeForConditionalGeneration.initialize_vllm_model(
        hf_config,
        mesh_device,
        batch,
        max_seq_len=TEST_CONTEXT,
        tt_data_parallel=1,
        optimizations=None,
        layer_indices=PROBE_LAYERS,
    )
    # vLLM's shape: (num_blocks, kv_heads_per_device, block_size, head_size). The pool has to hold one
    # max_model_len request plus a spare block per slot, exactly as the worker sizes it.
    model = adapter.model
    blocks = adapter.page_table_blocks + batch
    shape = (blocks, max(1, model.cfg.n_kv_heads // model.tp), model.page_block_size, model.cfg.head_dim)
    kv_cache = adapter.allocate_kv_cache(shape, torch.bfloat16, len(model.layers))
    adapter.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
    adapter.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=batch,
        num_blocks=adapter.page_table_blocks,
        can_sample_on_device=True,
        enable_trace=False,
    )
    adapter.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=batch,
        num_blocks=adapter.page_table_blocks,
        can_sample_on_device=True,
        enable_trace=True,
    )
    adapter._test_kv_cache = kv_cache
    _ADAPTER[key] = adapter
    return adapter


def serving_page_table(adapter):
    """A vLLM-shaped block table: slot ``u`` owns its own run of real blocks, 0 (the null block) elsewhere."""
    batch = adapter.max_batch_size
    width = adapter.page_table_blocks
    table = torch.zeros(batch, width, dtype=torch.int32)
    per_slot = width // batch
    for slot in range(batch):
        base = 1 + slot * per_slot
        table[slot, :per_slot] = torch.arange(base, base + per_slot, dtype=torch.int32)
    return table


def greedy_params(batch):
    from models.common.sampling import SamplingParams

    return SamplingParams(
        temperature=[0.0] * batch,
        top_k=[1] * batch,
        top_p=[1.0] * batch,
        seed=[None] * batch,
        enable_log_probs=[False] * batch,
        num_logprobs=[0] * batch,
    )


def on_mesh(test):
    """The `1x4` ring with this stack's device parameters. Applied per test: the host-only cases above
    must not open a mesh."""
    test = pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)(test)
    return pytest.mark.parametrize("mesh_device", [MC.DEFAULT_MESH_SHAPE], indirect=True)(test)


@on_mesh
def test_vllm_owns_the_kv_cache_and_the_generator_never_allocates_one(mesh_device):
    adapter = serving_adapter(mesh_device)
    generator = adapter.generator
    assert generator.owns_cache is False, "the serving generator must not own the cache"
    assert generator.kv_cache is adapter._test_kv_cache
    assert adapter.model.kv_cache is adapter._test_kv_cache
    # One [k, v] pair per full_attention layer, [] for the recurrent ones: the contract shape.
    kinds = [bool(entry) for entry in adapter._test_kv_cache]
    assert kinds == [layer.is_full_attention for layer in adapter.model.layers]
    # And the cache carries the *policy's* dtype, not vLLM's torch view.
    for entry, layer in zip(adapter._test_kv_cache, adapter.model.layers):
        if layer.is_full_attention:
            assert entry[0].dtype == adapter.model.policy.kv_cache_dtype


@on_mesh
def test_a_rewritten_block_size_is_refused(mesh_device, expect_error):
    """vLLM raises ``block_size`` for hybrid models to fit a recurrent state into an attention page.

    This port keeps its recurrent state in the model, so a rewritten block size does not describe the
    cache it allocates - and allocating anyway would give every request a page table vLLM indexes
    wrongly. The plugin registration prevents it; this pins the refusal that would catch it anyway.
    """
    adapter = serving_adapter(mesh_device)
    model = adapter.model
    heads = max(1, model.cfg.n_kv_heads // model.tp)
    with expect_error(ValueError, "block_size"):
        adapter.allocate_kv_cache((512, heads, 1072, model.cfg.head_dim), torch.bfloat16, len(model.layers))
    with expect_error(ValueError, "kv_heads"):
        adapter.allocate_kv_cache(
            (512, heads + 1, model.page_block_size, model.cfg.head_dim), torch.bfloat16, len(model.layers)
        )


@on_mesh
def test_prefill_writes_the_slot_vllm_assigned_not_the_row_order(mesh_device):
    """``empty_slots`` is the mapping, and it is not the identity whenever an off-batch request holds a row."""
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    prompts = [[6, 66, 666, 6666, 66], [9, 99, 999, 9999, 99, 9]]
    width = max(len(p) for p in prompts)
    tokens = torch.zeros(len(prompts), width, dtype=torch.int64)
    for row, prompt in enumerate(prompts):
        tokens[row, : len(prompt)] = torch.tensor(prompt)
    slots = [1, 3]
    out = adapter.prefill_forward(
        tokens=tokens,
        page_table=table[slots],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(p) for p in prompts],
        start_pos=[0, 0],
        sampling_params=greedy_params(adapter.max_batch_size),
        empty_slots=slots,
    )
    sampled, rope_deltas = out
    assert tuple(sampled.shape) == (len(prompts),), "one sampled token id per prefilled request"
    assert torch.equal(rope_deltas, torch.zeros(len(prompts), dtype=torch.long)), "text mRoPE delta is 0"
    assert bool(adapter._prefilled_rows[1]) and bool(adapter._prefilled_rows[3])
    assert not bool(adapter._prefilled_rows[0]) and not bool(adapter._prefilled_rows[2])
    # The device token buffer is NOT the prefill sampler's output: it carries the other slots' tokens.
    assert not bool(adapter._device_token_rows[1])


@on_mesh
def test_the_steady_state_decode_copies_nothing_to_the_device(mesh_device):
    """The async-decode contract, at the adapter's own API.

    After the reset step that follows a prefill, later steps must copy no token, no position and no page
    table: the token arrives through ``tt_out_tok`` and the positions advance with ``ttnn.plus_one``,
    both inside the trace.
    """
    adapter = serving_adapter(mesh_device)
    generator = adapter.generator
    table = serving_page_table(adapter)
    prompt = [6, 66, 666, 6666, 66]
    slot = 1
    sampled, _ = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=greedy_params(adapter.max_batch_size),
        empty_slots=[slot],
    )
    batch = adapter.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(sampled[0])
    positions[slot] = len(prompt)

    for key in generator.counters:
        generator.counters[key] = 0
    device_out = adapter.decode_forward(
        tokens=tokens,
        page_table=table,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions,
        enable_trace=True,
        read_from_device=False,
        sampling_params=greedy_params(batch),
        reset_batch=True,
    )
    assert isinstance(device_out, ttnn.Tensor), "read_from_device=False must return device handles"
    host, events = adapter.read_decode_output(device_out, async_read=True)
    for event in events:
        ttnn.event_synchronize(event)
    first = adapter.process_decode_output_host(host, is_tokens=True)
    assert tuple(first.shape) == (batch,)
    after_reset = dict(generator.counters)
    assert after_reset["token_refreshes"] == 1 and after_reset["position_refreshes"] == 1

    emitted = [int(first[slot])]
    for _ in range(4):
        tokens[slot] = emitted[-1]
        positions[slot] = int(positions[slot]) + 1
        device_out = adapter.decode_forward(
            tokens=tokens,
            page_table=table,
            kv_cache=adapter._test_kv_cache,
            start_pos=positions,
            enable_trace=True,
            read_from_device=False,
            sampling_params=greedy_params(batch),
            reset_batch=False,
        )
        host, events = adapter.read_decode_output(device_out, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        emitted.append(int(adapter.process_decode_output_host(host, is_tokens=True)[slot]))

    assert generator.counters["token_refreshes"] == after_reset["token_refreshes"], "no token was re-staged"
    assert generator.counters["position_refreshes"] == after_reset["position_refreshes"]
    assert generator.counters["page_table_refreshes"] == after_reset["page_table_refreshes"]
    assert generator.counters["decode_syncs"] == 0, "the serving loop must not synchronize per token"
    assert adapter.serving_counters["no_refresh_steps"] == 4
    assert len(set(emitted)) > 1 or len(emitted) == 1, "a constant stream would suggest a dead token path"


@on_mesh
def test_a_stale_host_pair_does_not_override_the_device(mesh_device):
    """Under async scheduling vLLM re-sends the *previous* token and position for a continuing row.

    The merge must keep the device's pair for such a row - staging the lagging pair re-runs a position
    that already has a token, which reads as a doubled subword rather than as an error.
    """
    adapter = serving_adapter(mesh_device)
    generator = adapter.generator
    table = serving_page_table(adapter)
    prompt = [1, 2, 3, 4, 5, 6, 7]
    slot = 2
    params = greedy_params(adapter.max_batch_size)

    def run(lag: int):
        adapter.generator.reset()
        adapter._reset_serving_state()
        sampled, _ = adapter.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int64),
            page_table=table[slot : slot + 1],
            kv_cache=adapter._test_kv_cache,
            prompt_lens=[len(prompt)],
            start_pos=[0],
            sampling_params=params,
            empty_slots=[slot],
        )
        batch = adapter.max_batch_size
        emitted = [int(sampled[0])]
        position = len(prompt)
        for step in range(5):
            tokens = torch.zeros(batch, dtype=torch.int64)
            positions = torch.full((batch,), -1, dtype=torch.int64)
            # The prefill boundary is never lagged: that row's token came from prefill sampling, which
            # writes a scratch buffer, so the host is the only authority for it.
            behind = 0 if step == 0 else lag
            tokens[slot] = emitted[-1 - behind] if len(emitted) > behind else emitted[0]
            positions[slot] = position - behind
            device_out = adapter.decode_forward(
                tokens=tokens,
                page_table=table,
                kv_cache=adapter._test_kv_cache,
                start_pos=positions,
                enable_trace=True,
                read_from_device=True,
                sampling_params=params,
                reset_batch=True,
            )
            emitted.append(int(device_out[slot]))
            position += 1
        return emitted

    fresh = run(0)
    stale = run(1)
    assert stale == fresh, f"a one-token-stale host view must not change the stream: {fresh} != {stale}"
    assert generator.counters["decode_syncs"] == 0


@on_mesh
def test_only_a_changed_page_table_is_copied(mesh_device):
    adapter = serving_adapter(mesh_device)
    generator = adapter.generator
    table = serving_page_table(adapter)
    prompt = [11, 22, 33, 44, 55]
    slot = 0
    params = greedy_params(adapter.max_batch_size)
    sampled, _ = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=params,
        empty_slots=[slot],
    )
    batch = adapter.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(sampled[0])
    positions[slot] = len(prompt)
    out = adapter.decode_forward(
        tokens=tokens,
        page_table=table,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions,
        enable_trace=True,
        read_from_device=True,
        sampling_params=params,
        reset_batch=True,
    )
    tokens[slot] = int(out[slot])
    positions[slot] = int(positions[slot]) + 1

    grown = table.clone()
    grown[slot, adapter.page_table_blocks // batch] = adapter.page_table_blocks  # a newly allocated block
    before = dict(generator.counters)
    adapter.decode_forward(
        tokens=tokens,
        page_table=grown,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions,
        enable_trace=True,
        read_from_device=True,
        sampling_params=params,
        reset_batch=False,
    )
    assert generator.counters["page_table_refreshes"] == before["page_table_refreshes"] + 1
    assert generator.counters["token_refreshes"] == before["token_refreshes"], "the token must not be re-staged"
    assert generator.counters["position_refreshes"] == before["position_refreshes"]
    assert adapter.serving_counters["page_table_only_refreshes"] >= 1


@on_mesh
def test_a_slot_remap_moves_the_recurrent_state_bit_for_bit(mesh_device, expect_error):
    """A vLLM batch condense moves a request to another row; its DeltaNet state has to follow it.

    Compared as tensors, and with no decode step in between: the GDN recurrence has no position mask,
    so *every* row's state advances on a step and a before/after comparison across one would prove
    nothing. Token-level comparison is equally useless here - at batch > 1 a neighbouring row changes
    the last bits of every row's logits (``doc/datatype_sweep/README.md`` §9.1).
    """
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    prompt = [9, 99, 999, 9999, 99, 9]
    slot = 3
    params = greedy_params(adapter.max_batch_size)
    adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=params,
        empty_slots=[slot],
    )

    def state_rows():
        captured = {}
        for index, layer in enumerate(adapter.model.layers):
            if layer.is_full_attention or layer.recurrent_state is None:
                continue
            buffers = [("recurrent", layer.recurrent_state)] + [
                (f"conv{i}", buf) for i, buf in enumerate(layer.conv_state)
            ]
            for name, buf in buffers:
                for shard, tensor in enumerate(ttnn.get_device_tensors(buf)):
                    host = ttnn.to_torch(tensor)
                    for row in range(adapter.max_batch_size):
                        captured[(index, name, shard, row)] = host[row].clone()
        return captured

    before = state_rows()
    remap = torch.tensor([0, 1, 3, 2], dtype=torch.int32)  # row 2 takes slot 3, row 3 takes slot 2
    moved = adapter.generator.remap_serving_slots(remap)
    after = state_rows()
    assert moved == sum(1 for layer in adapter.model.layers if not layer.is_full_attention)

    def same(row_before, row_after):
        keys = sorted({key[:3] for key in before if key[3] == row_before})
        return all(torch.equal(before[key + (row_before,)], after[key + (row_after,)]) for key in keys)

    assert same(3, 2), "row 2 must have taken slot 3's state, bit for bit on every shard"
    assert same(2, 3), "row 3 must have taken slot 2's state"
    assert same(0, 0) and same(1, 1), "the untouched rows must not move"
    with expect_error(ValueError, "permutation"):
        adapter.model.remap_state_slots(torch.tensor([0, 0, 2, 3], dtype=torch.int32))


@on_mesh
def test_the_adapter_applies_the_slot_remap_before_the_decode_step(mesh_device):
    """The adapter's job is to hand the remap to the model *and* to the seed manager, then reindex its
    own per-slot bookkeeping the same way, before the step reads any of it."""
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    prompt = [7, 77, 777, 7777]
    slot = 3
    params = greedy_params(adapter.max_batch_size)
    sampled, _ = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=params,
        empty_slots=[slot],
    )
    batch = adapter.max_batch_size
    calls = []
    original = adapter.generator.remap_serving_slots
    adapter.generator.remap_serving_slots = lambda remap: (calls.append([int(v) for v in remap]), original(remap))[1]
    try:
        tokens = torch.zeros(batch, dtype=torch.int64)
        positions = torch.full((batch,), -1, dtype=torch.int64)
        tokens[2] = int(sampled[0])
        positions[2] = len(prompt)
        moved_table = table.clone()
        moved_table[2] = table[3]
        out = adapter.decode_forward(
            tokens=tokens,
            page_table=moved_table,
            kv_cache=adapter._test_kv_cache,
            start_pos=positions,
            enable_trace=True,
            read_from_device=True,
            sampling_params=params,
            reset_batch=True,
            slot_remap=torch.tensor([0, 1, 3, 2], dtype=torch.int32),
        )
    finally:
        adapter.generator.remap_serving_slots = original
    assert calls == [[0, 1, 3, 2]], "the remap must reach the model exactly once, unchanged"
    assert adapter.serving_counters["slot_remaps"] == 1
    assert tuple(out.shape) == (batch,)
    # An identity remap must not be forwarded at all: the plugin sends one whenever it condensed
    # nothing, and moving 30 layers of state for it would be pure cost.
    calls.clear()
    adapter.generator.remap_serving_slots = lambda remap: (calls.append([int(v) for v in remap]), original(remap))[1]
    try:
        tokens[2] = int(out[2])
        positions[2] = int(positions[2]) + 1
        adapter.decode_forward(
            tokens=tokens,
            page_table=moved_table,
            kv_cache=adapter._test_kv_cache,
            start_pos=positions,
            enable_trace=True,
            read_from_device=True,
            sampling_params=params,
            reset_batch=False,
            slot_remap=torch.arange(batch, dtype=torch.int32),
        )
    finally:
        adapter.generator.remap_serving_slots = original
    assert calls == [], "an identity remap must be recognised and skipped"
    assert adapter.serving_counters["slot_remaps"] == 1


@on_mesh
def test_host_sampling_returns_logits_and_never_becomes_the_default(mesh_device):
    """The plugin asks for logits whenever a request needs host-only sampling. That path must exist...

    ...and it must stay a request-driven exception: a step with ``sampling_params`` still samples on
    device, and the adapter's own counters say which happened.
    """
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    prompt = [5, 55, 555, 5555]
    slot = 0
    logits = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=None,
        empty_slots=[slot],
    )[0]
    assert tuple(logits.shape) == (1, 1, adapter.model.vocab_size), "host sampling needs [rows, 1, vocab]"

    batch = adapter.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(logits[0, -1]))
    positions[slot] = len(prompt)
    host_out = adapter.decode_forward(
        tokens=tokens,
        page_table=table,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=True,
    )
    assert tuple(host_out.shape) == (batch, 1, adapter.model.vocab_size)
    assert adapter.serving_counters["host_sampled_decodes"] >= 1
    assert adapter.serving_counters["device_sampled_decodes"] >= 0
    # And a device-sampling step still returns token ids, on the same adapter, right after.
    tokens[slot] = int(torch.argmax(host_out[slot, 0]))
    positions[slot] = int(positions[slot]) + 1
    device_out = adapter.decode_forward(
        tokens=tokens,
        page_table=table,
        kv_cache=adapter._test_kv_cache,
        start_pos=positions,
        enable_trace=True,
        read_from_device=True,
        sampling_params=greedy_params(batch),
        reset_batch=True,
    )
    assert tuple(device_out.shape) == (batch,)
    assert adapter.serving_counters["device_sampled_decodes"] >= 1


@on_mesh
def test_the_async_read_returns_the_same_result_for_tokens_and_for_logits(mesh_device):
    """`read_decode_output(async_read=True)` must work on **either** output tensor.

    The plugin defers the read on every step when async scheduling is on, and what the adapter is
    handed is the token buffer on a device-sampled step and the vocab-sharded logits on a host-sampled
    one. Both go through the generator's `read_output_async`, so this pins that the deferred path and
    the blocking path agree bit for bit on both - which is what makes the async split safe to be the
    default (it is, in this vLLM: README section 5).
    """
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    prompt = [7, 77, 777, 7777]
    slot = 0
    prefill_logits = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[slot : slot + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=None,
        empty_slots=[slot],
    )[0]
    batch = adapter.max_batch_size
    tokens = torch.zeros(batch, dtype=torch.int64)
    positions = torch.full((batch,), -1, dtype=torch.int64)
    tokens[slot] = int(torch.argmax(prefill_logits[0, -1]))
    positions[slot] = len(prompt)

    def step(sampling_params, is_tokens):
        """One decode step read both ways: deferred (async) and blocking."""
        deferred = adapter.decode_forward(
            tokens=tokens,
            page_table=table,
            kv_cache=adapter._test_kv_cache,
            start_pos=positions,
            enable_trace=True,
            read_from_device=False,
            sampling_params=sampling_params,
            reset_batch=True,
        )
        pending, events = adapter.read_decode_output(deferred, async_read=True)
        for event in events:
            ttnn.event_synchronize(event)
        from_async = adapter.process_decode_output_host(pending, is_tokens=is_tokens)
        from_blocking = adapter.process_decode_output_host(
            adapter.read_decode_output(deferred, async_read=False), is_tokens=is_tokens
        )
        return from_async, from_blocking

    # host-sampled step: the deferred tensor is the logits row
    async_logits, blocking_logits = step(None, is_tokens=False)
    assert tuple(async_logits.shape) == (batch, 1, adapter.model.vocab_size)
    assert torch.equal(async_logits, blocking_logits), "the async logits read must match the blocking one"

    tokens[slot] = int(torch.argmax(async_logits[slot, 0]))
    positions[slot] = int(positions[slot]) + 1

    # device-sampled step: the deferred tensor is the persistent token buffer
    async_tokens, blocking_tokens = step(greedy_params(batch), is_tokens=True)
    assert tuple(async_tokens.shape) == (batch,)
    assert torch.equal(async_tokens, blocking_tokens), "the async token read must match the blocking one"
    assert adapter.serving_counters["async_reads"] >= 2, "both deferred reads go through the generator"


@on_mesh
def test_a_nonfinite_idle_row_cannot_reach_a_served_request(mesh_device):
    """An idle slot's recurrent state can go non-finite. It must not reach any other slot.

    Measured precondition, not a hypothetical: at `max_num_seqs=32` the rows no prefill ever wrote
    accumulate float32 saturation and a few `inf`/`NaN` entries within one decode step
    (`doc/vllm_integration/decode_nondeterminism.json`, `state_after_first_step`), and nothing wipes them
    between requests - vLLM reuses slots, and `reset_state()` runs only at warm-up.

    Both of this stage's per-slot state paths *read* the rows they overwrite: `_merge_rows` computes
    `dst * inverse + src * mask`, and `inf * 0` is `NaN` under IEEE. So a poisoned idle row could, in
    principle, poison the request prefilled into it (via the prefill merge) or every row at once (via the
    batch-condense remap, which broadcasts the source row). This test writes the poison itself and pins
    that neither happens.
    """
    adapter = serving_adapter(mesh_device)
    table = serving_page_table(adapter)
    batch = adapter.max_batch_size
    model = adapter.model
    idle, target = batch - 2, 0
    recurrent = [layer.recurrent_state for layer in model.layers if not layer.is_full_attention]
    assert recurrent, "the reduced target must carry at least one linear_attention layer"

    def poison_idle_rows():
        """Write ``inf`` into the idle row of every DeltaNet recurrent buffer, in place."""
        for buf in recurrent:
            shape = [int(d) for d in buf.shape]
            host = ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).float()
            host[idle] = float("inf")
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host.reshape(shape), dtype=buf.dtype, layout=ttnn.TILE_LAYOUT), buf
            )

    def rows_are_finite(*, skip):
        """Is every row except ``skip`` finite, in every recurrent buffer?"""
        for buf in recurrent:
            host = ttnn.to_torch(ttnn.get_device_tensors(buf)[0]).float()
            for row in range(batch):
                if row in skip:
                    continue
                if not torch.isfinite(host[row]).all():
                    return False, row
        return True, None

    prompt = [9, 99, 999, 9999]
    poison_idle_rows()
    finite, row = rows_are_finite(skip={idle})
    assert finite, f"the poison itself leaked into row {row} before anything was served"

    # 1. a request prefilled into a *different* slot must not see the poison...
    logits = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[target : target + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=None,
        empty_slots=[target],
    )[0]
    assert torch.isfinite(logits).all(), "a poisoned idle row reached another slot's prefill logits"
    finite, row = rows_are_finite(skip={idle})
    assert finite, f"prefilling slot {target} left row {row} non-finite"

    # 2. ...and neither must a request prefilled *into the poisoned slot itself*.
    poison_idle_rows()
    logits = adapter.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int64),
        page_table=table[idle : idle + 1],
        kv_cache=adapter._test_kv_cache,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        sampling_params=None,
        empty_slots=[idle],
    )[0]
    assert torch.isfinite(logits).all(), "prefilling the poisoned slot produced non-finite logits"
    finite, row = rows_are_finite(skip=set())
    assert finite, f"prefilling the poisoned slot left row {row} non-finite"

    # 3. a batch condense that *moves* a poisoned row must not spread it either.
    poison_idle_rows()
    remap = list(range(batch))
    remap[target], remap[idle] = idle, target
    moved = adapter.generator.remap_serving_slots(torch.tensor(remap, dtype=torch.int32))
    assert moved >= 1, "the remap must actually move state"
    finite, row = rows_are_finite(skip={target})
    assert finite, f"a remap of a poisoned row left row {row} non-finite"


@on_mesh
def test_the_serving_build_carries_the_selected_precision_config(mesh_device):
    """Serving must run the datatype sweep's selection, and prove it from the *built* model.

    The adapter never passes a policy, so this is also the check that the default really is the
    artifact: weight groups, math fidelities, the KV-cache dtype the cache was allocated at, the
    residual/CCL dtypes and the (empty) layer-exception list, read off the live objects by
    ``precision_summary()`` and compared field by field with
    ``doc/datatype_sweep/selected_precision_config.json``.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt import precision_config as PC

    adapter = serving_adapter(mesh_device)
    model = adapter.model
    selected = PC.load_selected_policy()
    summary = model.precision_summary()
    assert model.policy == selected, "the serving build must default to the selected precision config"
    assert summary["selected_config"] == PC.policy_to_dict(selected)
    assert summary["selected_config"]["layer_exceptions"] == [], "the selection has no layer exceptions"
    assert summary["built"]["lm_head_weight_dtype"] == str(selected.resolved_lm_head_dtype)
    assert summary["built"]["logits_dtype"] == str(selected.logits_dtype)
    for row, layer in zip(summary["per_layer"], model.layers):
        assert row["policy_name"] == layer.policy.name
        assert row["residual_dtype"] == str(selected.residual_dtype)
        assert row["ccl_dtype"] == str(selected.ccl_dtype)
        if layer.is_full_attention:
            # The cache vLLM owns was allocated at the policy's dtype, not at vLLM's torch view.
            assert row["k_cache_dtype"] == row["v_cache_dtype"] == str(selected.kv_cache_dtype)


@on_mesh
def test_the_serving_capability_report_names_the_selected_policy(mesh_device):
    adapter = serving_adapter(mesh_device)
    report = adapter.serving_capability()
    assert report["architecture"] == TT_ARCH
    assert report["kv_cache_owner"] == "vllm"
    assert report["recurrent_state_owner"] == "model"
    assert report["max_model_len"] == TEST_CONTEXT
    assert report["capability"]["policy"] == adapter.model.policy.name
    assert report["capability"]["kv_cache_dtype"] == str(adapter.model.policy.kv_cache_dtype)
    assert report["generator"]["sampling_mode"] == "device"
    assert report["generator"]["owns_cache"] is False
