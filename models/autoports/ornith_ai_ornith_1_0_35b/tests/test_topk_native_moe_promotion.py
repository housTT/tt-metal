# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Promotion gates for the router-index-native Ornith prefill MoE.

The low-level operator suite proves dispatch, combine, cache-patching and workspace semantics.  This
file owns the model boundary that suite cannot see: comparison with the accepted gathered layer,
prefill-only counter accounting, decode isolation, the exact 40-layer B1/B2/B4 evidence workload,
and the real-layer timing bar.

All native tests build with both per-expert layouts available.  An A/B may therefore turn native
execution off and reach the accepted gathered implementation on the *same* decoder, with the same
weights and input.  Native execution always takes precedence when both switches are selected.

Run the focused layer gates on the four-chip Blackhole ring with::

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/\
        test_topk_native_moe_promotion.py \
        -k "matches_gathered_layer or decode_keeps_sparse" -v -p no:randomly

The full-stack counter proof is deliberately separate and long-running.  It loads all 40 layers and
runs only the isolated promotion workload named in its docstring.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tests import test_full_model as FULL
from models.autoports.ornith_ai_ornith_1_0_35b.tests import test_multichip_decoder as MULTI
from models.autoports.ornith_ai_ornith_1_0_35b.tt import optimized_decoder as OD
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_0_35b.tt.generator_vllm import _topk_native_moe_prefill_capability

# The accepted C25 + gathered-MoE result is 25.46 ms for the real 2048-token linear-attention layer.
# Promotion requires at least a 10% same-process win, rounded to the published 22.91 ms ceiling.
GATHERED_LAYER_BASELINE_MS = 25.46
NATIVE_LAYER_RELATIVE_BAR = 0.90
NATIVE_LAYER_MS_BAR = 22.91
NATIVE_GATHERED_PCC_BAR = 0.9999

PROMOTION_BATCHES = (1, 2, 4)
PROMOTION_TOKENS_PER_USER = 2048
REDUCED_STACK_LAYERS = (0, 1, 3)
EXPECTED_LAYERS = 40
SELECTED_SUB_CHUNK = OD._topk_native_sub_chunk()
EXPECTED_CALLS_PER_LAYER = sum(batch * PROMOTION_TOKENS_PER_USER // SELECTED_SUB_CHUNK for batch in PROMOTION_BATCHES)
EXPECTED_LAYER_CALLS_PER_LAYER = len(PROMOTION_BATCHES)
EXPECTED_AGGREGATE_CALLS = EXPECTED_LAYERS * EXPECTED_CALLS_PER_LAYER
EXPECTED_AGGREGATE_LAYER_CALLS = EXPECTED_LAYERS * EXPECTED_LAYER_CALLS_PER_LAYER

assert SELECTED_SUB_CHUNK in OD.MOE_TOPK_NATIVE_SUPPORTED_SUB_CHUNKS
assert EXPECTED_CALLS_PER_LAYER == (14 if SELECTED_SUB_CHUNK == 1024 else 7)
assert EXPECTED_LAYER_CALLS_PER_LAYER == 3
assert EXPECTED_AGGREGATE_CALLS == (560 if SELECTED_SUB_CHUNK == 1024 else 280)
assert EXPECTED_AGGREGATE_LAYER_CALLS == 120
assert NATIVE_LAYER_MS_BAR == round(GATHERED_LAYER_BASELINE_MS * NATIVE_LAYER_RELATIVE_BAR, 2)

pytestmark = [
    pytest.mark.parametrize("mesh_device", [MULTI.DEFAULT_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", MULTI.DEVICE_PARAMS, indirect=True),
]


def _select_native_and_gathered_weights(monkeypatch) -> None:
    """Select both layouts before construction; native wins only when its live flag stays set."""

    monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
    monkeypatch.setattr(OD, "MOE_TOPK_NATIVE", True)


def _counter_fields(moe) -> dict[str, int]:
    status = moe.topk_native_status()
    return {
        "calls": status["calls"],
        "fallbacks": status["fallbacks"],
        "layer_calls": status["layer_calls"],
        "subchunks": status["subchunks"],
    }


def _dram_state(mesh_device) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    allocated = int(view.total_bytes_allocated_per_bank) * banks
    total = int(view.total_bytes_per_bank) * banks
    return {"allocated": allocated, "free": total - allocated, "total": total}


def _instrument_layer_stages(model, mesh_device, monkeypatch, events: list[str]) -> None:
    """Synchronize stage boundaries so the final event identifies a device-side stall."""

    for layer in model.layers:
        real_moe = layer.moe.forward
        real_reduce = layer._all_reduce

        def logged_moe(*args, _real=real_moe, _layer=layer.layer_idx, **kwargs):
            events.append(f"layer={_layer}:moe:start")
            logger.info(events[-1])
            result = _real(*args, **kwargs)
            ttnn.synchronize_device(mesh_device)
            events.append(f"layer={_layer}:moe:finish")
            logger.info(events[-1])
            return result

        reduce_calls = {"count": 0}

        def logged_reduce(*args, _real=real_reduce, _layer=layer.layer_idx, _calls=reduce_calls, **kwargs):
            _calls["count"] += 1
            label = "attention" if _calls["count"] % 2 else "moe"
            events.append(f"layer={_layer}:{label}-all-reduce:start")
            logger.info(events[-1])
            result = _real(*args, **kwargs)
            ttnn.synchronize_device(mesh_device)
            events.append(f"layer={_layer}:{label}-all-reduce:finish")
            logger.info(events[-1])
            return result

        monkeypatch.setattr(layer.moe, "forward", logged_moe)
        monkeypatch.setattr(layer, "_all_reduce", logged_reduce)


def _mesh_tensor_record(mesh_device, tensor) -> dict:
    """Describe one mesh tensor's declared placement and actual per-device values."""

    topology = tensor.tensor_topology()
    topology_placements = tuple(topology.placements())
    placements = tuple(str(placement) for placement in topology_placements)
    distribution_shape = tuple(int(dim) for dim in topology.distribution_shape())
    mesh_coords = tuple(str(coord) for coord in topology.mesh_coords())
    shards = MULTI.shards(mesh_device, tensor)
    contents_equal = tuple(torch.equal(shards[0], shard) for shard in shards[1:])
    max_abs_diffs = tuple(float((shards[0].float() - shard.float()).abs().max()) for shard in shards[1:])
    placement_count_matches_distribution_rank = len(topology_placements) == len(distribution_shape)
    all_placements_are_replicate = all(
        isinstance(placement, ttnn.PlacementReplicate) for placement in topology_placements
    )
    return {
        "placements": placements,
        "distribution_shape": distribution_shape,
        "mesh_coords": mesh_coords,
        "placement_count_matches_distribution_rank": placement_count_matches_distribution_rank,
        "all_placements_are_replicate": all_placements_are_replicate,
        "fully_replicated_topology": placement_count_matches_distribution_rank and all_placements_are_replicate,
        "contents_equal_to_device0": contents_equal,
        "max_abs_diff_to_device0": max_abs_diffs,
    }


def _native_input_mesh_record(mesh_device, tensor) -> dict:
    """Describe the public composite input and its required ROW_MAJOR dispatch conversion.

    ``TensorTopology`` is the contract the device operation is entitled to trust, while the host
    copies answer the separate diagnostic question of whether every device currently happens to
    contain the same values.  The public composite converts ``x`` to ROW_MAJOR immediately before
    local dispatch, so inspect that conversion too: this distinguishes a wrong model boundary from
    topology loss inside the composite without weakening the native operation's fail-closed check.
    """

    public_x = _mesh_tensor_record(mesh_device, tensor)
    dispatch_x_rm = ttnn.to_layout(
        tensor,
        ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        converted = _mesh_tensor_record(mesh_device, dispatch_x_rm)
    finally:
        # Today the model hands the composite TILE input, so untilize owns a new buffer.  Preserve
        # the helper's safety if that API ever admits ROW_MAJOR directly: ``to_layout`` returns its
        # input unchanged when the layout already matches.
        if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            ttnn.deallocate(dispatch_x_rm)
    return {"public_composite_x": public_x, "dispatch_x_rm": converted}


def _assert_native_input_mesh_contract(mesh_device, tensor, *, label: str) -> dict:
    """Fail with topology *and* content evidence before entering the native C++ operation."""

    record = _native_input_mesh_record(mesh_device, tensor)
    valid = all(
        all(stage["contents_equal_to_device0"])
        and stage["placement_count_matches_distribution_rank"]
        and stage["all_placements_are_replicate"]
        for stage in record.values()
    )
    assert valid, (
        f"{label} violates the native expert-parallel input contract: {record}. "
        "Identical contents with a non-replicate or rank-mismatched placement declaration means "
        "the model boundary must normalize the tensor topology; divergent contents mean the "
        "upstream replicated-residual contract broke."
    )
    return record


def _assert_native_ready(moe) -> None:
    status = moe.topk_native_status()
    assert status["selected"] is True
    assert status["weights_loaded"] is True
    assert status["ready"] is True
    assert status["enabled"] is True
    assert status["refusal"] is None


@pytest.mark.parametrize("layer_idx", MULTI.LAYERS, ids=lambda i: MULTI.LAYER_IDS[i])
def test_topk_native_matches_gathered_layer_and_counts_exact_composites(mesh_device, layer_idx, monkeypatch):
    """Native BF8 output matches gathered output and records only real composite invocations.

    A 2048-token layer is exactly ``2048 / selected_sub_chunk`` native invocations. The accepted
    default remains two 1024-token calls; the guarded 2048-token experiment is one. The invocation
    spy and adjacent counters must agree exactly, while the gathered oracle leaves them at zero.
    """

    _select_native_and_gathered_weights(monkeypatch)
    sub_chunk = OD._topk_native_sub_chunk()
    expected_composites = PROMOTION_TOKENS_PER_USER // sub_chunk
    source = MULTI.default_weight_source()
    decoder, page_table, _ = MULTI.build_decoder(mesh_device, layer_idx, source)
    _assert_native_ready(decoder.moe)
    assert _counter_fields(decoder.moe) == {"calls": 0, "fallbacks": 0, "layer_calls": 0, "subchunks": 0}

    x = MULTI.to_device(
        mesh_device,
        MULTI.make_activations(1, PROMOTION_TOKENS_PER_USER, seed=6100 + layer_idx),
    )
    uploaded_input = _mesh_tensor_record(mesh_device, x)
    logger.info(f"layer {layer_idx} uploaded test input={uploaded_input}")
    assert uploaded_input["fully_replicated_topology"] and all(
        uploaded_input["contents_equal_to_device0"]
    ), f"the test's ReplicateTensorToMesh upload was not replicated: {uploaded_input}"

    # Accepted oracle: same object and uploaded weights, with only the native selector disabled.
    monkeypatch.setattr(OD, "MOE_TOPK_NATIVE", False)
    assert decoder.moe._gather_reason(PROMOTION_TOKENS_PER_USER, False, None) is None
    gathered_tt = decoder.prefill_forward(x, page_table=page_table)
    gathered = MULTI.to_host(mesh_device, gathered_tt)
    ttnn.deallocate(gathered_tt)
    decoder.reset_state()
    assert _counter_fields(decoder.moe) == {"calls": 0, "fallbacks": 0, "layer_calls": 0, "subchunks": 0}

    composite_spans: list[int] = []
    native_input_records: list[dict] = []
    real_native = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe

    def record_native(x_chunk, *args, **kwargs):
        composite_spans.append(int(x_chunk.shape[2]))
        native_input_records.append(
            _assert_native_input_mesh_contract(
                mesh_device,
                x_chunk,
                label=f"layer {layer_idx} native subchunk {len(composite_spans) - 1}",
            )
        )
        return real_native(x_chunk, *args, **kwargs)

    def forbid_dense_path(name):
        def forbidden(*args, **kwargs):
            del args, kwargs
            pytest.fail(f"native prefill entered the forbidden {name} path")

        return forbidden

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "topk_routed_expert_moe", record_native)
    monkeypatch.setattr(decoder.moe, "routing_weights", forbid_dense_path("dense-routing materializer"))
    monkeypatch.setattr(decoder.moe, "_gather_routed_experts", forbid_dense_path("gathered routed experts"))
    monkeypatch.setattr(decoder.moe, "_routed_experts", forbid_dense_path("standalone sparse routed experts"))
    monkeypatch.setattr(OD, "MOE_TOPK_NATIVE", True)
    native_tt = decoder.prefill_forward(x, page_table=page_table)
    native = MULTI.to_host(mesh_device, native_tt)
    ttnn.deallocate(native_tt)
    ttnn.deallocate(x)

    value = MULTI.pcc(gathered, native)
    logger.info(
        f"top-k-native vs gathered layer={layer_idx} ({MULTI.LAYER_IDS[layer_idx]}) "
        f"tokens={PROMOTION_TOKENS_PER_USER} PCC={value:.9f} spans={composite_spans} "
        f"counters={_counter_fields(decoder.moe)} native_inputs={native_input_records}"
    )
    assert torch.isfinite(native.float()).all(), "native layer output contains NaN or Inf"
    assert (
        value >= NATIVE_GATHERED_PCC_BAR
    ), f"top-k-native vs gathered PCC {value} is below the {NATIVE_GATHERED_PCC_BAR} promotion bar"
    assert composite_spans == [sub_chunk] * expected_composites
    assert _counter_fields(decoder.moe) == {
        "calls": expected_composites,
        "fallbacks": 0,
        "layer_calls": 1,
        "subchunks": expected_composites,
    }


@pytest.mark.long
@pytest.mark.timeout(600)
def test_topk_native_reduced_stack_2048_completes_collectives(mesh_device, monkeypatch):
    """The native 2K path survives cache reuse and both collectives across real layer kinds.

    The per-layer oracle tests build one decoder at a time. Serving instead drives multiple decoder
    objects through one model, reusing native-op programs while patching every layer's expert weights
    and output addresses before the routed-expert all-reduce. Keep a reduced real-weight stack here
    so a cache-hit or collective deadlock is caught without paying for all 40 layers.
    """

    if not FULL._snapshot_available():
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
    monkeypatch.setenv("ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK", "2048")
    assert OD._topk_native_sub_chunk() == 2048
    _select_native_and_gathered_weights(monkeypatch)
    generator = build_generator(
        model_dir=FULL.MODEL_DIR,
        mesh_device=mesh_device,
        layer_indices=REDUCED_STACK_LAYERS,
        max_batch_size=1,
        cache_context=4096,
        prefill_chunk=PROMOTION_TOKENS_PER_USER,
    )
    events: list[str] = []
    try:
        _instrument_layer_stages(generator.model, mesh_device, monkeypatch, events)

        prompt_gen = torch.Generator().manual_seed(6350)
        prompt = torch.randint(
            0,
            generator.model.vocab_size,
            (1, PROMOTION_TOKENS_PER_USER),
            generator=prompt_gen,
        )
        logits = generator.prefill_requests_into_slots(
            prompt,
            [PROMOTION_TOKENS_PER_USER],
            [0],
            page_table=None,
            kv_cache=None,
            sample_on_device=False,
            ensure_traces=False,
        )
        assert tuple(logits.shape) == (1, 1, generator.model.vocab_size)
        assert torch.isfinite(logits.float()).all()
        expected = PROMOTION_TOKENS_PER_USER // OD._topk_native_sub_chunk()
        for layer in generator.model.layers:
            assert _counter_fields(layer.moe) == {
                "calls": expected,
                "fallbacks": 0,
                "layer_calls": 1,
                "subchunks": expected,
            }
        assert all(f"layer={index}:moe-all-reduce:finish" in events for index in REDUCED_STACK_LAYERS)
    finally:
        generator.teardown()


@pytest.mark.parametrize("layer_idx", MULTI.LAYERS, ids=lambda i: MULTI.LAYER_IDS[i])
def test_topk_native_decode_keeps_sparse_path_and_does_not_move_counters(mesh_device, layer_idx, monkeypatch):
    """Decode dispatches the accepted sparse MoE and cannot masquerade as native evidence."""

    _select_native_and_gathered_weights(monkeypatch)
    source = MULTI.default_weight_source()
    decoder, page_table, _ = MULTI.build_decoder(mesh_device, layer_idx, source)
    _assert_native_ready(decoder.moe)

    native_input_records: list[dict] = []
    real_native = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe

    def record_native_input(x_chunk, *args, **kwargs):
        native_input_records.append(
            _assert_native_input_mesh_contract(
                mesh_device,
                x_chunk,
                label=f"layer {layer_idx} pre-decode native subchunk {len(native_input_records)}",
            )
        )
        return real_native(x_chunk, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "topk_routed_expert_moe", record_native_input)

    prefill = decoder.prefill_forward(
        MULTI.to_device(
            mesh_device,
            MULTI.make_activations(1, PROMOTION_TOKENS_PER_USER, seed=6200 + layer_idx),
        ),
        page_table=page_table,
    )
    ttnn.deallocate(prefill)
    before = _counter_fields(decoder.moe)
    expected_composites = PROMOTION_TOKENS_PER_USER // OD._topk_native_sub_chunk()
    assert before == {
        "calls": expected_composites,
        "fallbacks": 0,
        "layer_calls": 1,
        "subchunks": expected_composites,
    }
    reason = decoder.moe._topk_native_geometry_reason(OD.TILE, True, None)
    assert reason == "decode intentionally keeps sparse MoE"

    recorder = MULTI._OpRecorder(
        monkeypatch,
        ["sparse_matmul", "experimental.deepseek_prefill.topk_routed_expert_moe"],
    )
    current_pos, rot_idxs = MULTI.decode_inputs(mesh_device, torch.tensor([PROMOTION_TOKENS_PER_USER]))
    decoded_tt = decoder.decode_forward(
        MULTI.to_device(mesh_device, MULTI.make_activations(1, 1, seed=6300 + layer_idx)),
        current_pos=current_pos,
        rot_idxs=rot_idxs,
        page_table=page_table,
    )
    decoded = MULTI.to_host(mesh_device, decoded_tt)
    ttnn.deallocate(decoded_tt)

    after = _counter_fields(decoder.moe)
    logger.info(
        f"top-k-native-selected decode layer={layer_idx}: before={before} after={after} "
        f"native={recorder.count('experimental.deepseek_prefill.topk_routed_expert_moe')} "
        f"sparse={recorder.count('sparse_matmul')} native_inputs={native_input_records}"
    )
    assert torch.isfinite(decoded.float()).all()
    assert after == before, "decode changed a prefill-only native evidence counter"
    assert recorder.count("experimental.deepseek_prefill.topk_routed_expert_moe") == 0
    assert recorder.count("sparse_matmul") == 2, "decode no longer ran the accepted two sparse projections"


@pytest.mark.long
@pytest.mark.timeout(7200)
def test_full_stack_b1_b2_b4_prefill_has_exact_native_counters(mesh_device, monkeypatch):
    """All 40 layers record the selected exact calls/subchunks/layer-calls and no fallback.

    This is the isolated promotion evidence workload, and nothing else runs after construction:
    B1, B2 and B4 each receive 2048 synchronized tokens per user.  Correct device batching flattens
    those three model calls to 2048, 4096 and 8192 rows, hence 14 native composites per layer at the
    accepted 1K span or seven at the guarded 2K span. Serializing users would preserve the aggregate
    composite count but inflate layer-calls, so both arrays are exact gates. A scoped op wrapper
    separately proves that the aggregate counter corresponds to public native composite invocations.
    """

    if not FULL._snapshot_available():
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
    _select_native_and_gathered_weights(monkeypatch)

    observed_composites = 0
    real_native = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe

    def record_native(*args, **kwargs):
        nonlocal observed_composites
        observed_composites += 1
        return real_native(*args, **kwargs)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "topk_routed_expert_moe", record_native)
    generator = build_generator(
        model_dir=FULL.MODEL_DIR,
        mesh_device=mesh_device,
        max_batch_size=8,
        cache_context=4096,
        prefill_chunk=PROMOTION_TOKENS_PER_USER,
    )
    events: list[str] = []
    try:
        logger.info(f"full-stack native gate after build DRAM={_dram_state(mesh_device)}")
        _instrument_layer_stages(generator.model, mesh_device, monkeypatch, events)
        initial = _topk_native_moe_prefill_capability(generator.model)
        assert initial["total_layers"] == initial["expected_layers"] == EXPECTED_LAYERS
        assert initial["enabled"] is True and initial["refusal"] is None
        assert initial["calls_per_layer"] == [0] * EXPECTED_LAYERS
        assert initial["fallbacks_per_layer"] == [0] * EXPECTED_LAYERS
        assert initial["per_layer_layer_calls"] == [0] * EXPECTED_LAYERS
        assert initial["per_layer_subchunks"] == [0] * EXPECTED_LAYERS

        for batch in PROMOTION_BATCHES:
            generator.reset()
            logger.info(f"full-stack native gate before B{batch} DRAM={_dram_state(mesh_device)}")
            if batch == 1:
                # Serving warmup deliberately uses one repeated token. Preserve that skew-prone
                # routing shape here: random prompts failed to exercise expert counts above 1024.
                prompts = torch.ones((batch, PROMOTION_TOKENS_PER_USER), dtype=torch.int64)
            else:
                prompt_gen = torch.Generator().manual_seed(6400 + batch)
                prompts = torch.randint(
                    0,
                    generator.model.vocab_size,
                    (batch, PROMOTION_TOKENS_PER_USER),
                    generator=prompt_gen,
                )
            logits = generator.prefill_requests_into_slots(
                prompts,
                [PROMOTION_TOKENS_PER_USER] * batch,
                list(range(batch)),
                page_table=None,
                kv_cache=None,
                sample_on_device=False,
                ensure_traces=False,
            )
            assert tuple(logits.shape) == (batch, 1, generator.model.vocab_size)
            assert torch.isfinite(logits.float()).all()
            logger.info(
                f"full-stack native gate after B{batch} DRAM={_dram_state(mesh_device)} "
                f"last_stage={events[-1] if events else None}"
            )

        final = _topk_native_moe_prefill_capability(generator.model)
        logger.info(
            f"full-stack top-k-native evidence calls={final['calls']} subchunks={final['subchunks']} "
            f"layer_calls={final['layer_calls']} fallbacks={final['fallbacks']} "
            f"observed_composites={observed_composites}"
        )
        assert final["schema"] == "ornith-topk-native-moe-prefill/1"
        assert final["selected"] is final["weights_loaded"] is final["enabled"] is True
        assert final["ready_layers"] == final["enabled_layers"] == EXPECTED_LAYERS
        assert final["refusal"] is None
        assert final["calls_per_layer"] == [EXPECTED_CALLS_PER_LAYER] * EXPECTED_LAYERS
        assert final["per_layer_subchunks"] == [EXPECTED_CALLS_PER_LAYER] * EXPECTED_LAYERS
        assert final["per_layer_layer_calls"] == [EXPECTED_LAYER_CALLS_PER_LAYER] * EXPECTED_LAYERS
        assert final["fallbacks_per_layer"] == [0] * EXPECTED_LAYERS
        assert final["calls"] == final["subchunks"] == EXPECTED_AGGREGATE_CALLS
        assert final["layer_calls"] == EXPECTED_AGGREGATE_LAYER_CALLS
        assert final["fallbacks"] == 0
        assert observed_composites == EXPECTED_AGGREGATE_CALLS
        assert [row["layer_index"] for row in final["layers"]] == list(range(EXPECTED_LAYERS))
    finally:
        generator.teardown()


@pytest.mark.long
@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
def test_topk_native_real_2048_layer_meets_promotion_timing(mesh_device, monkeypatch):
    """The real linear-attention layer is <=22.91 ms and >=10% faster than gathered.

    Both arms use one decoder and one input.  Each is warm-compiled before five interleaved samples;
    reset and its synchronization stay outside the timed window.  The median rejects a one-off host
    scheduling win while retaining the same wall-clock definition as the accepted 25.46 ms gathered
    baseline.
    """

    if not MULTI._snapshot_available():
        pytest.skip("Ornith-1.0-35B checkpoint snapshot not available")
    _select_native_and_gathered_weights(monkeypatch)
    decoder, page_table, _ = MULTI.build_decoder(mesh_device, MULTI.LINEAR_LAYER, "real")
    _assert_native_ready(decoder.moe)
    assert decoder.moe._gather_reason(PROMOTION_TOKENS_PER_USER, False, None) is None
    x = MULTI.to_device(
        mesh_device,
        MULTI.make_activations(1, PROMOTION_TOKENS_PER_USER, seed=6500),
    )

    def run(*, native: bool) -> float:
        monkeypatch.setattr(OD, "MOE_TOPK_NATIVE", native)
        monkeypatch.setattr(OD, "MOE_GATHER_EXPERTS", True)
        decoder.reset_state()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        out = decoder.prefill_forward(x, page_table=page_table)
        ttnn.synchronize_device(mesh_device)
        elapsed = time.perf_counter() - start
        ttnn.deallocate(out)
        return elapsed * 1e3

    # Compile both branches before measurement.  The second pass also catches a cache-hit-only bug.
    for _ in range(2):
        run(native=False)
        run(native=True)

    gathered_samples: list[float] = []
    native_samples: list[float] = []
    for _ in range(5):
        gathered_samples.append(run(native=False))
        native_samples.append(run(native=True))
    gathered_ms = statistics.median(gathered_samples)
    native_ms = statistics.median(native_samples)
    ratio = native_ms / gathered_ms
    logger.info(
        f"top-k-native real layer-0 promotion timing: gathered={gathered_ms:.3f} ms "
        f"native={native_ms:.3f} ms ratio={ratio:.4f}; "
        f"gathered_samples={gathered_samples} native_samples={native_samples}"
    )
    ttnn.deallocate(x)

    assert (
        native_ms <= NATIVE_LAYER_MS_BAR
    ), f"native 2048-token layer median {native_ms:.3f} ms exceeds the {NATIVE_LAYER_MS_BAR:.2f} ms bar"
    assert ratio <= NATIVE_LAYER_RELATIVE_BAR, (
        f"native/gathered timing ratio {ratio:.4f} exceeds {NATIVE_LAYER_RELATIVE_BAR:.2f} "
        f"({native_ms:.3f}/{gathered_ms:.3f} ms)"
    )
