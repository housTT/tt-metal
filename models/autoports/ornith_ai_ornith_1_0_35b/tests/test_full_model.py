# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model and generator suite for ornith-ai/Ornith-1.0-35B on the 1x4 Blackhole ring.

Most cases run against the **reduced** model — one real ``linear_attention`` layer and one real
``full_attention`` layer, real weights, real cache/page-table shapes, real terminal norm/LM
head/sampling — because the properties under test (contract, shapes, trace/token feedback,
determinism, host-fallback freedom, batch/slot handling) are stack-depth independent and a 40-layer
build costs three minutes. The cases that genuinely need every layer are marked ``long``; the
accuracy gates themselves live in the readiness runners, not here.

Run:

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -x -q
    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_full_model.py -m long -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import model as M
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY
from models.common.readiness_check.contract import Generator

MODEL_DIR = Path(__file__).resolve().parents[1]

#: The reduced probe: layer 0 is ``linear_attention``, layer 3 is ``full_attention``.
PROBE_LAYERS = [0, 3]

#: Paged blocks the suite reserves per user. Small: these tests are about contract and mechanism,
#: not about long context, and `test_context_contract_is_the_advertised_one` checks the advertised
#: number separately from what any one construction allocates.
TEST_CACHE_CONTEXT = 4096

DEVICE_PARAMS = [
    {
        "l1_small_size": M.DEFAULT_L1_SMALL_SIZE,
        "trace_region_size": M.DEFAULT_TRACE_REGION_SIZE,
        "fabric_config": MC.DEFAULT_FABRIC_CONFIG,
        "fabric_router_config": MC.fabric_router_config(),
    },
]

pytestmark = [
    pytest.mark.parametrize("mesh_device", [MC.DEFAULT_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
]


def _snapshot_available() -> bool:
    try:
        return (M.resolve_model_path() / "model.safetensors.index.json").is_file()
    except Exception:  # noqa: BLE001 - offline / not downloaded
        return False


def _require_weights():
    if not _snapshot_available():
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")


_CACHE: dict = {}


def probe_generator(mesh_device, *, batch=1, sampling_mode="device", cache_context=TEST_CACHE_CONTEXT):
    """A cached reduced-model generator. Rebuilt only when a test needs a different shape."""
    _require_weights()
    key = (id(mesh_device), batch, sampling_mode, cache_context)
    if key in _CACHE:
        generator = _CACHE[key]
        generator.reset()
        return generator
    for stale_key in [k for k in _CACHE if k[0] == id(mesh_device)]:
        _CACHE.pop(stale_key).teardown()
    generator = build_generator(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        layer_indices=PROBE_LAYERS,
        max_batch_size=batch,
        cache_context=cache_context,
        sampling_mode=sampling_mode,
    )
    _CACHE[key] = generator
    return generator


def shards(mesh_device, tensor):
    lead = int(tensor.shape[0])
    whole = ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
    return [whole[i * lead : (i + 1) * lead] for i in range(mesh_device.get_num_devices())]


# --------------------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------------------
def test_generator_implements_the_readiness_contract(mesh_device):
    generator = probe_generator(mesh_device)
    assert isinstance(generator, Generator)
    import inspect

    signature = inspect.signature(generator.generate)
    parameter = signature.parameters.get("enable_trace")
    assert parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ), "the teacher-forcing runner requires an explicit enable_trace keyword"
    for name in ("prefill_forward", "decode_forward", "generate", "reset"):
        assert callable(getattr(generator, name))
    assert generator.tokenizer is not None
    assert generator.mesh_device is mesh_device


def test_context_contract_is_the_advertised_one(mesh_device):
    generator = probe_generator(mesh_device)
    capability = generator.model.capability()
    assert capability["hf_advertised_context"] == 262144
    assert capability["max_context"] == 262144, "the model must advertise the HF context"
    assert capability["vocab_size"] == 248320
    assert capability["padded_vocab_size"] % (32 * capability["tp"]) == 0
    assert capability["mesh_shape"] == [1, 4]
    assert capability["tp"] == 4


def test_the_decoder_policy_is_carried_through_unchanged(mesh_device):
    """The full model must not quietly relax any decoder-stage decision."""
    generator = probe_generator(mesh_device)
    model = generator.model
    assert MC.CCL_MODE == "all_reduce", "the stage-5 collective correctness fix must still be selected"
    assert MC.ROUTER_MODE == "fused_gate"
    for layer in model.layers:
        assert isinstance(layer, MultichipDecoder)
        assert layer.tp == 4
        assert layer.policy is DEFAULT_POLICY or layer.policy.name == DEFAULT_POLICY.name
        assert layer.policy.kv_cache_dtype == ttnn.bfloat8_b
        assert layer.policy.proj_dtype == ttnn.bfloat8_b
        assert layer.policy.expert_gate_up_dtype == ttnn.bfloat4_b
        assert layer.policy.router_dtype == ttnn.bfloat16
        if layer.is_full_attention:
            assert layer.k_cache is not None and layer.k_cache.dtype == ttnn.bfloat8_b
            assert layer.v_cache.dtype == ttnn.bfloat8_b
        else:
            assert layer.recurrent_state.dtype == ttnn.float32
        assert layer.page_block_size == 64, "the paged cache block size is part of the carried contract"
        assert layer.cfg.num_experts * layer.tp == model.cfg.num_experts == 256, "EP=4 over 256 routed experts"
        if not layer.is_full_attention:
            for buffer in layer.conv_state:
                assert buffer.dtype == ttnn.bfloat16, "the DeltaNet conv history is bfloat16"
    # The LM head is a *new* dense projection, so it takes the dense projection group's dtype
    # rather than inventing one.
    assert model.lm_head_weights[0].dtype == DEFAULT_POLICY.proj_dtype
    # The mesh/fabric half of the contract. The packet size is a fabric setting applied before
    # open_mesh_device, so what is checkable here is that the value the layer ships is the one the
    # suite's device_params passed.
    assert list(mesh_device.shape) == list(MC.DEFAULT_MESH_SHAPE) == [1, 4]
    assert MC.DEFAULT_FABRIC_CONFIG == ttnn.FabricConfig.FABRIC_1D_RING
    assert MC.fabric_router_config().max_packet_payload_size_bytes == MC.DEFAULT_FABRIC_PACKET_BYTES == 8192
    assert MC.DEFAULT_CCL_TOPOLOGY == ttnn.Topology.Ring and MC.DEFAULT_CCL_NUM_LINKS == 2


def test_the_deprecated_all_gather_is_not_reachable_from_the_sampler(mesh_device):
    """Stage 5 removed ``ttnn.all_gather`` from decode; the sampler must not put it back."""
    generator = probe_generator(mesh_device)
    ccl = generator.model.tt_ccl
    assert isinstance(ccl, M.OrnithSamplingCCL)
    assert callable(getattr(ccl, "line_all_gather", None))
    # Bound methods compare unequal by identity, so match on the underlying function and receiver.
    hook = generator.sampling.tt_sampling._line_all_gather
    assert callable(hook), "TTSampling fell back to ttnn.all_gather because it found no line_all_gather"
    assert (
        hook.__self__ is ccl and hook.__func__ is M.OrnithSamplingCCL.line_all_gather
    ), "TTSampling must have picked up the shim; otherwise it silently falls back to ttnn.all_gather"
    before = ccl.gathers
    generator.generate(prompt_token_ids=[1, 2, 3, 4, 5], max_new_tokens=3, enable_trace=True)
    assert ccl.gathers > before, "the sampler's gathers must go through the shim"


def test_the_inter_layer_residual_is_replicated_on_every_device(mesh_device):
    """The decoder stage's boundary contract: no collective and no fracturing between layers."""
    generator = probe_generator(mesh_device)
    model = generator.model
    generator.reset()
    tokens = torch.tensor([[11, 22, 33, 44, 55, 66, 77]], dtype=torch.int32)
    tokens_tt = ttnn.from_torch(
        tokens,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    page_table = generator._prefill_page_row(0)
    model._use_pack(1)
    # Same call the model makes, memory config included: `ttnn.embedding` inherits the *indices*
    # tensor's memory config when none is given, so leaving it off would test a different call.
    x = ttnn.embedding(tokens_tt, model.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, [1, int(tokens.shape[1]), model.dim])
    for layer in model.layers:
        x = layer.prefill_forward(x, start_pos=0, page_table=page_table, chunk_size=model.prefill_chunk)
        parts = shards(mesh_device, x)
        assert list(x.shape) == [1, int(tokens.shape[1]), model.dim]
        assert x.dtype == ttnn.bfloat16
        assert x.layout == ttnn.TILE_LAYOUT
        assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG, (
            "the inter-layer residual must stay DRAM interleaved - that is the decoder stage's "
            "carried contract, and the residual stream keeps whatever the embedding produced"
        )
        for device in range(1, len(parts)):
            assert torch.equal(parts[0], parts[device]), f"layer {layer.layer_idx} output differs on device {device}"
    ttnn.deallocate(x)


# --------------------------------------------------------------------------------------
# prompt shapes
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("prompt_len", [1, 7, 31, 33, 63, 129, 250, 1000, 2049, 3000])
def test_prefill_accepts_any_logical_prompt_length(mesh_device, prompt_len):
    """No divisibility requirement anywhere in the public path.

    2049 and 3000 straddle the 2048-token internal prefill chunk; 33, 63, 129 and 250 straddle the
    tile, the 64-token page and the 128-token physical alignment.
    """
    generator = probe_generator(mesh_device)
    torch.manual_seed(prompt_len)
    prompt = torch.randint(0, generator.model.vocab_size, (1, prompt_len))
    logits = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[prompt_len])
    assert tuple(logits.shape) == (1, 1, generator.model.vocab_size)
    assert torch.isfinite(logits).all()
    generator.reset()


def test_prefill_all_logits_shape_and_last_position_agree(mesh_device):
    generator = probe_generator(mesh_device)
    torch.manual_seed(3)
    prompt_len = 77
    prompt = torch.randint(0, generator.model.vocab_size, (1, prompt_len))
    generator.reset()
    everywhere = generator.prefill_forward(
        prompt, page_table=None, kv_cache=None, prompt_lens=[prompt_len], return_all_logits=True
    )
    assert tuple(everywhere.shape) == (1, prompt_len, generator.model.vocab_size)
    assert torch.isfinite(everywhere).all()
    generator.reset()
    last = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[prompt_len])
    assert torch.equal(
        torch.argmax(everywhere[0, -1]), torch.argmax(last[0, 0])
    ), "the last-position-only path must predict what the all-positions path predicts"


def test_chunked_prefill_continuation_matches_a_single_call(mesh_device, expect_error):
    """A prompt split across two ``prefill_forward`` calls must equal one call over the whole prompt.

    This is the serving shape: a scheduler that chunks a long prompt calls the low-level prefill once
    per chunk with the chunk's absolute ``start_pos``. The DeltaNet state has to carry over, which is
    what ``continue_from_state`` is for; without it the second chunk would start from a zeroed
    recurrent state and the model would silently forget the first 2048 tokens.
    """
    generator = probe_generator(mesh_device)
    chunk = generator.model.prefill_chunk
    total = chunk + 952  # two chunks, the second deliberately not aligned to anything
    torch.manual_seed(77)
    prompt = torch.randint(0, generator.model.vocab_size, (1, total))

    generator.reset()
    single = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[total])

    generator.reset()
    generator.prefill_forward(prompt[:, :chunk], page_table=None, kv_cache=None, prompt_lens=[chunk], start_pos=0)
    chunked = generator.prefill_forward(
        prompt[:, chunk:],
        page_table=None,
        kv_cache=None,
        prompt_lens=[total - chunk],
        start_pos=chunk,
        continue_from_state=True,
    )
    assert int(torch.argmax(chunked[0, 0])) == int(
        torch.argmax(single[0, 0])
    ), "chunked prefill must predict what a single-call prefill predicts"
    assert torch.allclose(chunked, single, atol=1e-2), "chunked prefill logits diverged from the single call"

    # And the boundary is enforced rather than silently wrong.
    batched = probe_generator(mesh_device, batch=4)
    with expect_error(ValueError, "batch-1 only"):
        batched.prefill_forward(
            torch.zeros(4, 8, dtype=torch.int64),
            page_table=None,
            kv_cache=None,
            prompt_lens=[8] * 4,
            continue_from_state=True,
        )


def test_generate_rejects_a_request_longer_than_the_allocated_cache(mesh_device, expect_error):
    generator = probe_generator(mesh_device)
    with expect_error(ValueError, "exceeds the allocated cache context"):
        generator.generate(prompt_token_ids=[1] * 8, max_new_tokens=TEST_CACHE_CONTEXT, enable_trace=True)


def test_a_page_table_too_narrow_for_the_position_is_rejected(mesh_device, expect_error):
    """A short row is worse than a missing one: the paged kernel would read past the row's end.

    Both public low-level entry points validate blocks-per-user against the highest position they
    are about to address, not just the row count.
    """
    generator = probe_generator(mesh_device)
    model = generator.model
    full = generator.page_table
    blocks = int(full.shape[1])
    assert blocks >= 2, "this test needs a cache of at least two blocks"
    narrow = full[:, :1].contiguous()
    beyond = model.page_block_size  # the first position the one-block row cannot hold

    with expect_error(ValueError, "blocks per user"):
        model.prefill_forward(
            torch.ones(1, beyond + 1, dtype=torch.int32),
            page_table=narrow,
            kv_cache=model.kv_cache,
            prompt_lens=[beyond + 1],
        )
    with expect_error(ValueError, "blocks per user"):
        model.prepare_decode_inputs_host([1], [beyond], narrow)

    # ... and the same call with a wide-enough row is accepted, so the check is not just "raise".
    model.prepare_decode_inputs_host([1], [beyond], full)


# --------------------------------------------------------------------------------------
# split sampling / trace contract
# --------------------------------------------------------------------------------------
def test_split_sampling_feeds_the_token_back_on_device(mesh_device):
    """The sampled token of replay N is the token input of replay N+1, with no host reconstruction.

    Also pins the rest of the split-sampling contract: current position and RoPE index advance
    inside the model trace, the page table is not re-copied while it is unchanged, and both traces
    exist as separate captures.
    """
    generator = probe_generator(mesh_device)
    generator.reset()
    generator._ensure_decode_trace()
    assert generator._trace_id is not None, "a model decode trace must be captured"
    assert generator._sampling_trace_ready, "a separate sampling trace must be captured"
    tok_buf, pos_buf, rot_buf, page_buf = generator._trace_inputs
    assert generator.sampling._trace_states, "the sampling trace must be owned by the common sampler"
    slot = next(iter(generator.sampling._trace_states.values()))
    assert slot["output"][0] is tok_buf, "tt_out_tok must be the persistent decode token buffer"
    assert slot["input"] is generator._trace_logits, "the sampler must consume the model trace's own output"

    prompt = [5, 9, 13, 21, 34]
    generator.model.prefill_forward_single(
        prompt, page_table=generator._prefill_page_row(0), start_pos=0, return_logits=False
    )
    generator._write_positions(torch.tensor([len(prompt)], dtype=torch.int32))
    generator._write_tokens(torch.tensor([7], dtype=torch.int32))
    generator._refresh_page_table_only(generator.page_table)

    def read(buffer):
        return ttnn.to_torch(buffer, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)).reshape(-1)

    observed = []
    page_refreshes_before = generator.counters["page_table_refreshes"]
    token_refreshes_before = generator.counters["token_refreshes"]
    position_refreshes_before = generator.counters["position_refreshes"]
    for _ in range(4):
        token_in = int(read(tok_buf)[0])
        position_in = int(read(pos_buf)[0])
        rope_in = int(read(rot_buf)[0])
        assert position_in == rope_in, "current position and the RoPE index must stay coherent"
        generator._decode_step_traced()
        generator._sample_traced()
        ttnn.synchronize_device(mesh_device)
        shards = read(tok_buf).reshape(mesh_device.get_num_devices(), -1)
        # Every device runs the same sampler over its own vocabulary shard and the shim's
        # all-gather is what makes them agree. If a shard disagreed, the generator's readback
        # (device 0) would silently be one device's private answer.
        assert torch.equal(
            shards, shards[0].expand_as(shards)
        ), f"all {mesh_device.get_num_devices()} devices must sample the same token; got {shards[:, 0].tolist()}"
        token_out = int(shards[0, 0])
        position_out = int(read(pos_buf)[0])
        rope_out = int(read(rot_buf)[0])
        assert position_out == position_in + 1, "the model trace must advance current_pos on device"
        assert rope_out == rope_in + 1, "the model trace must advance the RoPE index on device"
        observed.append((token_in, token_out))

    # The chain above would also be satisfied by a sampler that never wrote anything, so cross-check
    # the last replay's token against a host argmax of the logits the sampler actually consumed.
    # This is also the direct evidence that greedy on device means what torch.argmax means.
    host_logits = generator.model.decode_logits_to_host(generator._trace_logits, batch=1)
    assert observed[-1][1] == int(
        torch.argmax(host_logits[0]).item()
    ), "the on-device greedy token must equal the host argmax of the same logits"

    for step in range(1, len(observed)):
        assert observed[step][0] == observed[step - 1][1], (
            "the token consumed by replay N+1 must be the token the sampler wrote in replay N; " f"got {observed}"
        )
    assert (
        generator.counters["page_table_refreshes"] == page_refreshes_before
    ), "an unchanged page table must not be copied per token"
    assert (
        generator.counters["token_refreshes"] == token_refreshes_before
    ), "free-running decode must not refresh the token from host"
    assert (
        generator.counters["position_refreshes"] == position_refreshes_before
    ), "free-running decode must not refresh positions from host"


def test_a_changed_page_table_is_copied_exactly_once(mesh_device):
    generator = probe_generator(mesh_device)
    generator.reset()
    generator._ensure_decode_trace()
    generator._refresh_page_table_only(generator.page_table)
    before = generator.counters["page_table_refreshes"]
    assert generator._refresh_page_table_only(generator.page_table) is False
    assert generator.counters["page_table_refreshes"] == before

    permuted = generator.page_table.clone()
    permuted[0, :2] = permuted[0, [1, 0]]
    assert generator._refresh_page_table_only(permuted) is True
    assert generator.counters["page_table_refreshes"] == before + 1
    assert generator._refresh_page_table_only(permuted) is False
    assert generator.counters["page_table_refreshes"] == before + 1

    device_table = ttnn.to_torch(
        generator._trace_inputs[3], mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)
    )[: generator.max_batch_size]
    assert torch.equal(device_table.to(torch.int32), permuted), "the refresh must reach the trace input tensor"
    generator._refresh_page_table_only(generator.page_table)


def test_greedy_decode_has_no_host_fallback(mesh_device, monkeypatch):
    """No host tensor traffic between trace replays on the measured token-out path."""
    generator = probe_generator(mesh_device)
    generator.reset()
    generator._ensure_decode_trace()
    generator.model.prefill_forward_single(
        [2, 3, 5, 7, 11], page_table=generator._prefill_page_row(0), start_pos=0, return_logits=False
    )
    generator._write_positions(torch.tensor([5], dtype=torch.int32))
    ttnn.synchronize_device(mesh_device)

    def forbid(name):
        def guard(*args, **kwargs):
            raise AssertionError(f"{name} was called inside a traced decode step")

        return guard

    monkeypatch.setattr(ttnn, "from_torch", forbid("ttnn.from_torch"))
    monkeypatch.setattr(ttnn, "to_torch", forbid("ttnn.to_torch"))
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", forbid("ttnn.copy_host_to_device_tensor"))
    monkeypatch.setattr(ttnn, "argmax", forbid("ttnn.argmax"))
    for _ in range(3):
        generator._decode_step_traced()
        generator._sample_traced()
    ttnn.synchronize_device(mesh_device)


def test_greedy_decode_is_deterministic(mesh_device):
    generator = probe_generator(mesh_device)
    prompt = [101, 202, 303, 404, 505, 606, 707]
    first = generator.generate(prompt_token_ids=prompt, max_new_tokens=12, enable_trace=True)
    second = generator.generate(prompt_token_ids=prompt, max_new_tokens=12, enable_trace=True)
    assert first == second, f"greedy decode must be reproducible: {first} != {second}"


def test_requests_of_different_prompt_lengths_do_not_corrupt_each_other(mesh_device):
    """The regression test for the post-capture compilation hazard.

    A prompt length the generator has not seen compiles new programs, and tt-metal allocates their
    kernel binaries from the same pool the live decode trace writes into
    (``allocator.cpp``: *"buffers ... may be corrupted once a trace is executed"*). Before
    ``OrnithGenerator._ensure_traces_replay_safe`` existed, the second request in a process emitted
    token 0 followed by gibberish and **every** later prefill stayed broken — including a repeat of
    the first request, which is what makes this a corruption rather than a bad request.

    ``doc/full_model/logs/probe_bisect.py`` is the minimal repro; this pins the fix.
    """
    generator = probe_generator(mesh_device)
    short = [11, 12, 13, 14, 15]
    long = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    baseline_short = generator.generate(prompt_token_ids=short, max_new_tokens=6, enable_trace=True)
    baseline_long = generator.generate(prompt_token_ids=long, max_new_tokens=6, enable_trace=True)
    again_short = generator.generate(prompt_token_ids=short, max_new_tokens=6, enable_trace=True)
    again_long = generator.generate(prompt_token_ids=long, max_new_tokens=6, enable_trace=True)
    assert again_short == baseline_short, (
        f"the short prompt stopped reproducing after a different-length request: " f"{baseline_short} -> {again_short}"
    )
    assert again_long == baseline_long, f"the long prompt stopped reproducing: {baseline_long} -> {again_long}"
    # A bare prefill must agree too — that is the surface the corruption showed up on first.
    generator.reset()
    logits = generator.prefill_forward(torch.tensor([short]), page_table=None, kv_cache=None, prompt_lens=[len(short)])
    assert int(torch.argmax(logits[0, 0])) == baseline_short[0]


def test_traces_are_recaptured_when_a_new_program_is_compiled(mesh_device):
    """A newly seen prompt length forces exactly one re-capture, and a repeat forces none."""
    generator = probe_generator(mesh_device)
    generator._ensure_decode_trace()
    novel = list(range(500, 500 + 47))
    before = generator.trace_recaptures
    generator.generate(prompt_token_ids=novel, max_new_tokens=4, enable_trace=True)
    after_first = generator.trace_recaptures
    generator.generate(prompt_token_ids=novel, max_new_tokens=4, enable_trace=True)
    after_second = generator.trace_recaptures
    assert after_first >= before, "a new prompt length compiles programs and must re-capture"
    assert after_second == after_first, "a warmed prompt length must not re-capture again"
    assert (
        generator.mesh_device.num_program_cache_entries() == generator._program_cache_at_capture
    ), "after a warmed request the program cache must match what the live traces were captured over"


def test_prefill_logits_are_deterministic_across_runs(mesh_device):
    generator = probe_generator(mesh_device)
    torch.manual_seed(11)
    prompt = torch.randint(0, generator.model.vocab_size, (1, 64))
    generator.reset()
    first = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[64])
    generator.reset()
    second = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[64])
    assert torch.equal(first, second), "logits for the same prompt must be bit-identical across runs"


def test_host_sampling_compatibility_mode_agrees_with_device_sampling(mesh_device):
    """The explicit host-sampling mode exists, and it picks the same greedy tokens."""
    prompt = [12, 34, 56, 78, 90, 123, 456]
    device_generator = probe_generator(mesh_device, sampling_mode="device")
    on_device = device_generator.generate(prompt_token_ids=prompt, max_new_tokens=10, enable_trace=True)
    host_generator = probe_generator(mesh_device, sampling_mode="host")
    on_host = host_generator.generate(prompt_token_ids=prompt, max_new_tokens=10, enable_trace=True)
    assert on_device == on_host, f"device sampling {on_device} != host sampling {on_host}"


def test_alternating_greedy_and_sampled_requests_stay_correct(mesh_device):
    """The same captured graph serves top-k/top-p, and switching modes does not poison greedy.

    ``$tt-enable-tracing``'s symptom table names this exactly: *"greedy output nondeterministic
    across runs, or wrong after a sampled request — trace cache keyed too coarsely"*. The experiment
    it prescribes is to alternate greedy and sampled requests back to back. ``SamplingGenerator``
    keys its trace slots by ``(penalties, log_probs, force_argmax)``; ``force_argmax`` is off here
    for both modes, so greedy and sampled deliberately share one captured graph and differ only in
    the persistent ``k``/``p``/``temp`` tensors — which is the thing worth proving rather than
    assuming.
    """
    from models.common.sampling import SamplingParams

    generator = probe_generator(mesh_device)
    prompt = [3, 14, 15, 92, 65, 35]
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    sampled = SamplingParams(temperature=0.8, top_k=32, top_p=0.95, seed=1234)

    baseline = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    trace_ids_before = {k: v["id"] for k, v in generator.sampling._trace_states.items()}

    hot = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True, sampling_params=sampled)
    assert len(hot) == 8
    assert all(0 <= t < generator.model.vocab_size for t in hot), f"sampled decode produced out-of-range ids: {hot}"

    after = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True, sampling_params=greedy)
    assert after == baseline, (
        f"greedy stopped reproducing after a sampled request: {baseline} -> {after}. "
        "That is the trace-keyed-too-coarsely signature."
    )
    trace_ids_after = {k: v["id"] for k, v in generator.sampling._trace_states.items()}
    assert trace_ids_after == trace_ids_before, (
        "switching sampling params must not re-capture the sampling trace: with force_argmax off, "
        "k/p/temp are persistent tensors, not part of the captured graph"
    )


def test_the_hf_reference_class_resolver_picks_the_checkpoints_own_architecture(mesh_device):
    """The shared readiness helper this stage added, on the checkpoint that motivated it.

    ``AutoModelForCausalLM`` maps ``qwen3_5_moe`` to ``Qwen3_5MoeForCausalLM``, which expects
    ``model.layers.*`` while this checkpoint stores ``model.language_model.layers.*``: every weight
    would be reported missing and the "reference" would be randomly initialised, silently. No device
    work; the resolver only reads the config.
    """
    from transformers import AutoModelForCausalLM

    from models.common.readiness_check.hf_model import resolve_hf_model_class

    _require_weights()
    resolved, reason = resolve_hf_model_class(M.HF_MODEL_ID)
    assert resolved.__name__ == "Qwen3_5MoeForConditionalGeneration", (
        f"resolver picked {resolved.__name__} ({reason}); the checkpoint declares "
        "Qwen3_5MoeForConditionalGeneration and stores model.language_model.layers.*"
    )
    assert resolved is not AutoModelForCausalLM

    class _NoArchitectures:
        architectures = []

    import models.common.readiness_check.hf_model as hf_model

    original = hf_model.AutoConfig.from_pretrained
    try:
        hf_model.AutoConfig.from_pretrained = lambda *a, **k: _NoArchitectures()
        fallback, _ = resolve_hf_model_class("anything")
        assert fallback is AutoModelForCausalLM, "a config with no usable architecture must fall back"
    finally:
        hf_model.AutoConfig.from_pretrained = original


def test_teacher_forcing_calls_back_once_per_requested_token(mesh_device):
    generator = probe_generator(mesh_device)
    seen = []

    def next_input(step, predicted):
        seen.append((step, predicted))
        return (predicted + 1) % generator.model.vocab_size

    predictions = generator.generate(
        prompt_token_ids=[3, 1, 4, 1, 5, 9, 2, 6],
        max_new_tokens=9,
        next_input=next_input,
        enable_trace=True,
    )
    assert len(predictions) == 9
    assert len(seen) == 9
    assert [step for step, _ in seen] == list(range(9))


# --------------------------------------------------------------------------------------
# batch / slots
# --------------------------------------------------------------------------------------
def test_batched_prefill_and_decode_with_mixed_prompt_lengths(mesh_device):
    """Fixed slots, mixed prompt lengths, and an inactive row that must stay untouched."""
    generator = probe_generator(mesh_device, batch=4)
    generator.reset()
    lengths = [37, 5, 129, 64]
    torch.manual_seed(19)
    width = max(lengths)
    tokens = torch.zeros(4, width, dtype=torch.int64)
    for user, length in enumerate(lengths):
        tokens[user, :length] = torch.randint(0, generator.model.vocab_size, (length,))
    logits = generator.prefill_forward(tokens, page_table=None, kv_cache=None, prompt_lens=lengths)
    assert tuple(logits.shape) == (4, 1, generator.model.vocab_size)
    assert torch.isfinite(logits).all()

    # Row 3 is the inactive slot: position -1 keeps it out of the cache write and out of the
    # position advance.
    positions = torch.tensor(lengths, dtype=torch.int32)
    positions[3] = -1
    first = torch.argmax(logits, dim=-1).reshape(-1)
    step = generator.decode_forward(first, positions, page_table=generator.page_table, enable_trace=True)
    assert tuple(step.shape) == (4, generator.model.vocab_size)
    assert torch.isfinite(step[:3]).all()

    device_positions = ttnn.to_torch(
        generator._trace_inputs[1], mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)
    ).reshape(-1)[:4]
    assert int(device_positions[3]) == -1, "an inactive row must not have its position advanced"
    for user in range(3):
        assert int(device_positions[user]) == lengths[user] + 1


def test_low_level_prefill_then_decode_sees_the_prompt(mesh_device, expect_error):
    """The low-level pair on a generator that has never captured must decode *from the prompt*.

    Regression for a silent one: trace capture warm-compiles a real decode step and then wipes the
    DeltaNet state and the whole paged KV cache. When capture was lazy — first triggered inside
    ``decode_forward`` — a caller who prefilled and then decoded got a fluent-looking token computed
    against an *empty* cache, and every test that only checked ``isfinite`` was happy with it. The
    serving adapter drives exactly this pair, so it is checked against the high-level path, which
    captures before prefilling and therefore always saw the prompt.
    """
    prompt = [11, 222, 3333, 44444, 5, 66]
    reference = probe_generator(mesh_device)
    reference.reset()
    # Two tokens: the first comes out of prefill, the second out of a traced decode step that has to
    # read the prompt's cache. It is the second one this test is about.
    expected = reference.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)

    # The same generator with its traces thrown away: a generator that has never captured, driven
    # only through the low-level surface.
    fresh = probe_generator(mesh_device)
    fresh.reset()
    fresh._release_traces()
    assert fresh._trace_id is None, "the point of this test is a generator that has not captured yet"

    logits = fresh.prefill_forward(torch.tensor([prompt]), page_table=None, kv_cache=None, prompt_lens=[len(prompt)])
    first = int(torch.argmax(logits[0, 0]))
    assert first == int(expected[0]), "greedy prefill must agree with the high-level path's first token"
    step = fresh.decode_forward(
        torch.tensor([first]),
        torch.tensor([len(prompt)]),
        enable_trace=True,
        sample_on_device=True,
    )
    assert int(step[0]) == int(expected[1]), (
        "low-level prefill -> traced decode must produce the token the high-level path produces; "
        "a mismatch means the prompt's cache was wiped between the two"
    )

    # And the ordering that cannot work is refused rather than silently wiping the prompt.
    fresh._release_traces()
    with expect_error(RuntimeError, "captured before a prefill"):
        fresh._ensure_decode_trace()


def test_the_eager_debug_path_does_not_poison_the_traced_one(mesh_device, expect_error):
    """`generate(enable_trace=False)` writes state without capturing; the next traced call must work.

    The liveness guard that stops trace capture from wiping a live prompt (§5.3) has to be cleared
    by `reset()`, and `generate(reset=True)` has to reset *before* it captures — otherwise asking
    for a reset would be refused because of the very state the reset removes.
    """
    generator = probe_generator(mesh_device)
    prompt = [4, 44, 444, 4444]
    generator.reset()
    generator._release_traces()

    eager = generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=False)
    assert len(eager) == 2 and all(0 <= int(t) < generator.model.vocab_size for t in eager)
    assert generator.model.state_is_live, "the eager path writes prompt state"
    assert generator._trace_id is None, "the eager path must not capture"

    traced = generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
    assert traced == eager, f"traced and eager greedy decode must agree; {traced} vs {eager}"

    # reset=False with live state and no traces is the one combination that cannot be served, and
    # it says so instead of wiping the caller's prompt.
    generator._release_traces()
    with expect_error(RuntimeError, "captured before a prefill"):
        generator.generate(prompt_token_ids=prompt, max_new_tokens=1, enable_trace=True, reset=False)
    generator.reset()


def test_host_sampling_mode_refuses_device_sampling(mesh_device, expect_error):
    """`sample_on_device=True` needs a sampler graph; the host compatibility mode has none."""
    generator = probe_generator(mesh_device, sampling_mode="host")
    with expect_error(ValueError, "sampling_mode='device'"):
        generator.decode_forward(torch.tensor([5]), torch.tensor([1]), enable_trace=True, sample_on_device=True)


def test_a_caller_owned_cache_attached_late_forces_a_recapture(mesh_device):
    """A cache attached after capture must re-bind the traces, not decode into the old tensors."""
    generator = probe_generator(mesh_device)
    generator.reset()
    generator._ensure_decode_trace()
    before = generator.trace_recaptures
    own = generator.kv_cache

    borrowed = generator.model.allocate_kv_cache(generator.total_blocks)
    try:
        prompt = [3, 33, 333, 3333]
        generator.prefill_forward(
            torch.tensor([prompt]), page_table=generator.page_table, kv_cache=borrowed, prompt_lens=[len(prompt)]
        )
        token = generator.decode_forward(
            torch.tensor([7]), torch.tensor([len(prompt)]), enable_trace=True, sample_on_device=True
        )
        assert generator.trace_recaptures == before + 1, (
            "attaching a caller-owned cache must re-capture the traces: they were bound to the "
            "generator's own cache tensors"
        )
        assert generator._kv_cache_at_capture is borrowed
        assert 0 <= int(token[0]) < generator.model.vocab_size
    finally:
        # Release first: the traces are bound to `borrowed`'s tensors and must not outlive them.
        generator._release_traces()
        generator.model.attach_kv_cache(own)
        for entry in borrowed:
            for tensor in entry:
                ttnn.deallocate(tensor)
        generator.reset()


@pytest.mark.long
@pytest.mark.timeout(3600)
def test_batch_32_prefill_and_decode(mesh_device):
    """The advertised batch bound, end to end on the reduced model.

    32 is what ``ttnn.sampling`` allows (one core per user, ``1 <= num_users <= 32``) and what the
    decoder stage advertised. Marked ``long`` only because it rebuilds the cached probe generator at
    a different batch and prepares a second per-batch state pack; the properties it checks are the
    same fixed-slot ones ``test_batched_prefill_and_decode_with_mixed_prompt_lengths`` checks at 4.
    """
    generator = probe_generator(mesh_device, batch=32, cache_context=2048)
    generator.reset()
    torch.manual_seed(32)
    lengths = [5 + 3 * user for user in range(32)]
    width = max(lengths)
    tokens = torch.zeros(32, width, dtype=torch.int64)
    for user, length in enumerate(lengths):
        tokens[user, :length] = torch.randint(0, generator.model.vocab_size, (length,))
    logits = generator.prefill_forward(tokens, page_table=None, kv_cache=None, prompt_lens=lengths)
    assert tuple(logits.shape) == (32, 1, generator.model.vocab_size)
    assert torch.isfinite(logits).all()

    first = torch.argmax(logits, dim=-1).reshape(-1)
    positions = torch.tensor(lengths, dtype=torch.int32)
    step = generator.decode_forward(first, positions, page_table=generator.page_table, enable_trace=True)
    assert tuple(step.shape) == (32, generator.model.vocab_size)
    assert torch.isfinite(step).all()
    device_positions = ttnn.to_torch(
        generator._trace_inputs[1], mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)
    ).reshape(-1)[:32]
    for user in range(32):
        assert int(device_positions[user]) == lengths[user] + 1
    logger.info(f"batch-32 decode ok; prompt lengths {lengths[0]}..{lengths[-1]}")


def test_batch_four_generate_agrees_with_batch_one(mesh_device):
    """The high-level path at batch > 1 must decode from the prompt, not from an empty state.

    `generate` prefills one request through the **batch-1** pack while the captured decode trace is
    bound to the **batch-B** pack: at batch 1 they are the same tensors, at batch 4 they are not, so
    without `prefill_request_into_slot`'s merge the 30 recurrent layers decode from a zeroed state.
    The prompt's own token would still be right, which is why this compares the **first decoded**
    token: `doc/full_model/batch_slots.json` records that removing the merge changes it from 240560
    to 169222, and that keeping it reproduces batch 1 exactly.

    Two tokens and no more, deliberately. Batch 4 and batch 1 are not bit-identical — the batch
    changes the matmul and MoE-grouping geometry — so greedy picks a different near-tie from the
    third token on. The same probe shows that divergence is *independent of what the other slots
    hold* (identical and different neighbours give the same tokens), i.e. it is batch numerics, not
    cross-request leakage. README §11 records it.
    """
    prompt = [6, 66, 666, 6666, 66]
    expected = probe_generator(mesh_device, batch=1).generate(
        prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True
    )
    got = probe_generator(mesh_device, batch=4).generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
    assert got == expected, (
        f"batch-4 generate must reproduce batch-1 generate's prompt token and first decoded token; "
        f"{got} vs {expected}"
    )


def test_the_batched_prefill_state_reaches_every_decode_slot(mesh_device):
    """The low-level batched pair, checked against a token rather than against `isfinite`.

    Every row gets the same prompt, so every row's decoded token must be the batch-1 answer. This
    is what fails if `_merge_prefill_state_into_slot` stops copying the prefill state into the
    decode pack — `isfinite` and "the position advanced" both survive that.
    """
    prompt = [8, 88, 888, 8888]
    expected = probe_generator(mesh_device, batch=1).generate(
        prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True
    )
    four = probe_generator(mesh_device, batch=4)
    four.reset()
    tokens = torch.tensor([prompt] * 4)
    logits = four.prefill_forward(tokens, page_table=None, kv_cache=None, prompt_lens=[len(prompt)] * 4)
    first = torch.argmax(logits, dim=-1).reshape(-1)
    assert all(int(v) == int(expected[0]) for v in first), f"prefill: {first.tolist()} vs {expected[0]}"
    step = four.decode_forward(
        first,
        torch.tensor([len(prompt)] * 4),
        page_table=four.page_table,
        enable_trace=True,
        sample_on_device=True,
    )
    assert all(
        int(v) == int(expected[1]) for v in step
    ), f"every slot decoded the same prompt, so every slot must produce {expected[1]}; got {step.tolist()}"


def test_batch_one_and_batch_four_agree_on_the_same_prompt(mesh_device):
    """A prompt in slot 0 of a batch-4 model must predict what the batch-1 model predicts."""
    prompt = [7, 77, 777, 7777, 77777]
    single = probe_generator(mesh_device, batch=1)
    single_out = single.generate(prompt_token_ids=prompt, max_new_tokens=6, enable_trace=True)
    batched = probe_generator(mesh_device, batch=4)
    batched.reset()
    tokens = torch.zeros(4, len(prompt), dtype=torch.int64)
    for user in range(4):
        tokens[user, :] = torch.tensor(prompt)
    logits = batched.prefill_forward(tokens, page_table=None, kv_cache=None, prompt_lens=[len(prompt)] * 4)
    predicted = torch.argmax(logits[:, 0], dim=-1)
    assert int(predicted[0]) == single_out[0], "slot 0 must predict what batch 1 predicts"
    for user in range(1, 4):
        assert int(predicted[user]) == int(predicted[0]), "identical prompts in different slots must agree"


# --------------------------------------------------------------------------------------
# reset / cache ownership
# --------------------------------------------------------------------------------------
def test_reset_wipes_state_but_keeps_traces_and_weights(mesh_device):
    generator = probe_generator(mesh_device)
    prompt = [21, 22, 23, 24, 25]
    before = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    trace_id = generator._trace_id
    generator.reset()
    assert generator._trace_id is trace_id, "reset must not release the captured traces"
    for layer in generator.model.layers:
        if not layer.is_full_attention:
            state = ttnn.to_torch(
                layer.recurrent_state, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0)
            )
            assert torch.count_nonzero(state) == 0, "reset must zero the DeltaNet recurrent state"
        else:
            cache = ttnn.to_torch(layer.k_cache, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
            assert torch.count_nonzero(cache) == 0, "reset must zero the paged KV cache"
    after = generator.generate(prompt_token_ids=prompt, max_new_tokens=8, enable_trace=True)
    assert before == after, "a reset generator must reproduce its own output"


def test_a_caller_can_own_the_cache(mesh_device):
    """Cache ownership is explicit: an external cache can be attached and drives the same path."""
    generator = probe_generator(mesh_device)
    model = generator.model
    external = []
    for layer in model.layers:
        if layer.is_full_attention:
            shape = [generator.total_blocks, layer.cfg.n_kv_heads, layer.page_block_size, layer.cfg.head_dim]
            external.append(
                [
                    ttnn.zeros(shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device),
                    ttnn.zeros(shape, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device),
                ]
            )
        else:
            external.append([])
    owned = generator.kv_cache
    try:
        generator.reset()
        logits = generator.prefill_forward(
            torch.tensor([[31, 41, 59, 26, 53]]),
            page_table=generator.page_table,
            kv_cache=external,
            prompt_lens=[5],
        )
        assert torch.isfinite(logits).all()
        for layer, entry in zip(model.layers, external):
            if layer.is_full_attention:
                assert layer.k_cache is entry[0]
                filled = ttnn.to_torch(entry[0], mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))
                assert torch.count_nonzero(filled) > 0, "prefill must fill the caller's cache"
    finally:
        model.attach_kv_cache(owned)
        for entry in external:
            for tensor in entry:
                ttnn.deallocate(tensor)
    generator.reset()


# --------------------------------------------------------------------------------------
# the sampler's grouped local top-k
# --------------------------------------------------------------------------------------
def test_the_two_topk_width_knobs_refuse_to_combine(mesh_device, expect_error):
    """Padding widens the reduction, grouping narrows it; asking for both is a caller error.

    The `mesh_device` fixture is the module-wide parametrisation, not a dependency: this exercises
    the shared sampler's validation on a stub carrying only the attributes the check reads.
    """
    from models.common.sampling.tt_sampling import TTSampling

    class _Args:
        pass

    stub = _Args()
    stub.topk_num_groups = 20
    stub.multi_step_reduction = False
    stub.max_top_k = 32
    stub.pad_to_power_of_2 = True
    with expect_error(ValueError, "cannot be combined"):
        TTSampling._create_grouped_topk_tensors(stub, 62080, ttnn.uint16)

    # ... and without it the same call gets past that check (and then fails on the stub, which has
    # no device - which is the point: the refusal above is a *validation*, not a side effect).
    stub.pad_to_power_of_2 = False
    try:
        TTSampling._create_grouped_topk_tensors(stub, 62080, ttnn.uint16)
    except Exception as exc:  # noqa: BLE001 - anything but the combination error is fine here
        assert "cannot be combined" not in str(exc)


def test_grouped_local_topk_matches_a_single_reduction(mesh_device):
    """The shared sampler's opt-in grouped top-k is exact, not an approximation."""
    generator = probe_generator(mesh_device)
    sampling = generator.sampling.tt_sampling
    assert sampling.topk_num_groups == M.DEFAULT_TOPK_GROUPS
    per_device = generator.model.padded_vocab_size // 4
    torch.manual_seed(23)
    # Distinct, well separated maxima: a tie is broken arbitrarily by *both* spellings, so a random
    # bfloat16 tensor would compare tie-break policy rather than the reduction.
    values = (torch.rand(1, 1, 32, per_device) * 0.01).float()
    for row in range(32):
        positions = torch.randperm(per_device)[:32]
        values[0, 0, row, positions] = 10.0 + torch.arange(32).flip(0).float() * 0.5
    logits = ttnn.from_torch(
        values.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    reference_values, reference_indices = torch.topk(values.bfloat16().float()[0, 0], k=32, dim=-1)
    grouped_values, grouped_indices = sampling._local_topk_grouped(logits)
    single_values, single_indices = ttnn.topk(
        logits, k=32, dim=-1, indices_tensor=sampling.tt_indices_tensor, stable=sampling._topk_stable
    )

    def host(tensor):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=0))[0, 0]

    assert torch.equal(host(grouped_indices).to(torch.int64), reference_indices.to(torch.int64))
    assert torch.equal(host(single_indices).to(torch.int64), reference_indices.to(torch.int64))
    assert torch.allclose(host(grouped_values).float(), reference_values.float(), atol=1e-2)
    for tensor in (logits, grouped_values, grouped_indices, single_values, single_indices):
        ttnn.deallocate(tensor)


# --------------------------------------------------------------------------------------
# reduced-layer profiling variant
# --------------------------------------------------------------------------------------
# `$full-model` asks for the `tt-perf-report` evidence to come from a reduced variant with one real
# layer of each kind and the real terminal path, not from the 40-layer stack: a Tracy capture of
# ~3300 device ops per decode step produces multi-gigabyte dumps and overruns Tracy's buffers, and
# every op in the window appears in the reduced capture anyway. `tracy/run_profiling.sh` captures the
# same two windows through the standalone `logs/profile_reduced.py`, which the profiler survives;
# these pytest nodes are the same measurement without a profiler attached.
def profiling_generator(mesh_device):
    """The reduced model, built as leanly as a Tracy capture can afford.

    ``prefill_chunk=256`` instead of 2048 only changes **setup**: ``allocate_state`` prepares and
    probes one ``ttnn.conv1d`` program per 128-token block length up to the chunk, so 2048 costs
    sixteen probe runs per layer and 256 costs two. Every program in either signposted window is
    identical — a 128-token prefill block is below both chunk sizes, so it resolves the same SDPA
    config, the same MoE group size and the same conv1d length, and decode does not see the chunk at
    all. What it buys is a capture the profiler's post-processing can actually reassemble: at 2048
    the setup probes alone push the op id past 400 000 and ``process_ops_logs`` fails to match a
    device row.
    """
    _require_weights()
    return build_generator(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        layer_indices=PROBE_LAYERS,
        max_batch_size=1,
        cache_context=TEST_CACHE_CONTEXT,
        prefill_chunk=256,
    )


@pytest.mark.long
@pytest.mark.timeout(3600)
def test_perf_full_model_decode_traced(mesh_device):
    """Warmed token-out decode between ``PERF_DECODE`` signposts, on the reduced model."""
    from tracy import signpost

    generator = profiling_generator(mesh_device)
    generator._ensure_decode_trace()
    iters = 32

    def step():
        ttnn.execute_trace(mesh_device, generator._trace_id, cq_id=0, blocking=False)
        generator._sample_traced()

    for _ in range(4):
        step()
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_DECODE")
    import time as _time

    start = _time.perf_counter()
    for _ in range(iters):
        step()
    ttnn.synchronize_device(mesh_device)
    elapsed = _time.perf_counter() - start
    signpost("PERF_DECODE_END")
    logger.info(
        f"FULL-MODEL PERF decode(traced, reduced {PROBE_LAYERS}) iters={iters} "
        f"wall/iter={elapsed / iters * 1e3:.3f} ms  t/s/u={iters / elapsed:.2f}"
    )
    # Release the captured traces before the mesh fixture closes the device: leaving them live
    # through `close_mesh_device` segfaults under the profiler.
    generator.teardown()


@pytest.mark.long
@pytest.mark.timeout(3600)
def test_perf_full_model_prefill(mesh_device):
    """Warmed 128-token prefill between ``PERF_PREFILL`` signposts, on the reduced model."""
    import time as _time

    from tracy import signpost

    generator = profiling_generator(mesh_device)
    torch.manual_seed(0)
    prompt = torch.randint(0, generator.model.vocab_size, (1, 128))
    generator.reset()
    generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[128])
    generator.reset()
    signpost("PERF_PREFILL")
    start = _time.perf_counter()
    generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[128])
    elapsed = _time.perf_counter() - start
    signpost("PERF_PREFILL_END")
    logger.info(f"FULL-MODEL PERF prefill(128, reduced {PROBE_LAYERS}) wall={elapsed * 1e3:.3f} ms")
    generator.teardown()


# --------------------------------------------------------------------------------------
# all-layer cases
# --------------------------------------------------------------------------------------
@pytest.mark.long
@pytest.mark.timeout(3600)
def test_full_stack_generates_coherent_text(mesh_device):
    """The whole 40-layer stack, a real chat prompt, and a human-readable completion."""
    _require_weights()
    generator = build_generator(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        max_batch_size=1,
        cache_context=8192,
    )
    try:
        tokenizer = generator.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        out = generator.generate(prompt_token_ids=prompt, max_new_tokens=64, enable_trace=True)
        text = tokenizer.decode(out, skip_special_tokens=True)
        logger.info(f"full-stack completion: {text!r}")
        assert len(out) > 4
        words = [w.lower() for w in text.split()]
        duplicated = sum(1 for a, b in zip(words, words[1:]) if a == b)
        assert duplicated / max(len(words) - 1, 1) < 0.1, f"mechanically doubled output: {text!r}"
    finally:
        generator.teardown()


@pytest.mark.long
@pytest.mark.timeout(3600)
def test_full_stack_non_aligned_long_prompt(mesh_device):
    """A long non-aligned prompt through the public path on the complete stack."""
    _require_weights()
    generator = build_generator(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        max_batch_size=1,
        cache_context=8192,
    )
    try:
        torch.manual_seed(5)
        prompt_len = 5003  # not a multiple of the tile, page, alignment or prefill chunk
        prompt = torch.randint(0, generator.model.vocab_size, (1, prompt_len))
        logits = generator.prefill_forward(prompt, page_table=None, kv_cache=None, prompt_lens=[prompt_len])
        assert tuple(logits.shape) == (1, 1, generator.model.vocab_size)
        assert torch.isfinite(logits).all()
    finally:
        generator.teardown()
