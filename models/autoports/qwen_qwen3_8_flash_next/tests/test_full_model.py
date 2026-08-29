# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract and reduced real-weight gates for the Qwen3.8 full model."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.demo.full_model import (
    run_autoregressive,
    run_prefill_check,
    run_qualitative_suite,
    run_teacher_forcing,
    write_autoregressive_artifacts,
    write_report,
)
from models.autoports.qwen_qwen3_8_flash_next.tests import harness as H
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator, build_generator
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import PLEDeviceStaging
from models.autoports.qwen_qwen3_8_flash_next.tt.model import (
    HIDDEN_SIZE,
    LM_HEAD_POLICIES,
    REQUIRED_L1_SMALL_SIZE,
    VOCAB_SIZE,
    Qwen38FullModel,
    _dtype_name,
    _lm_head_rank_slices,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import FABRIC_PACKET_BYTES, MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision_config import load_precision_config

_SOURCE_DIGEST_PATHS = (
    "tt/functional_decoder.py",
    "tt/generator.py",
    "tt/host_weight_cache.py",
    "tt/model.py",
    "tt/multichip_decoder.py",
    "tt/optimized_decoder.py",
    "tt/precision_config.py",
    "tests/test_full_model.py",
)


def _source_provenance() -> dict[str, object]:
    """Bind retained evidence to the exact dirty source bytes used by pytest."""

    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    files = {}
    for relative in _SOURCE_DIGEST_PATHS:
        payload = (root / relative).read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        files[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {"algorithm": "sha256", "digest": digest.hexdigest(), "files": files}


def _record_source_provenance(record_property) -> dict[str, object]:
    provenance = _source_provenance()
    record_property("source_digest_algorithm", provenance["algorithm"])
    record_property("source_digest", provenance["digest"])
    record_property("source_digest_files", json.dumps(provenance["files"], sort_keys=True))
    return provenance


def _device_params(*, trace_region_size=1_073_741_824, l1_small_size=REQUIRED_L1_SMALL_SIZE):
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PACKET_BYTES
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": router,
        "l1_small_size": l1_small_size,
        "trace_region_size": trace_region_size,
    }


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_PRECISION_SMOKE") != "1",
    reason="explicit datatype-policy construction smoke",
)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_precision_config_reduced_construction_and_summary(bh_1d_mesh_device, device_params):
    """Construct all layer kinds and prove every policy leaf reaches runtime."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
        prepack_host_experts=True,
    )
    try:
        summary = model.precision_propagation_summary()
        assert summary["all_fields_consumed"] is True
        assert all(item["passed"] for item in summary["checks"].values())

        class PrecisionSmokeTokenizer:
            @staticmethod
            def decode(tokens, **_kwargs):
                return " ".join(str(int(token)) for token in tokens)

        generator = Qwen38Generator(model, PrecisionSmokeTokenizer())
        prompt = (torch.arange(129, dtype=torch.int64).reshape(1, -1) + 17) % VOCAB_SIZE
        output = generator.generate_batch(
            prompt,
            max_new_tokens=3,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("precision-smoke-129",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 3)
        assert model.trace_replays == 1
        summary["non_aligned_prompt_tokens"] = 129
        summary["traced_generated_tokens"] = output.tolist()
        summary["runtime_fallback_audit"] = model.runtime_fallback_audit(generator.state)
        write_report(summary, _evidence_dir() / "precision_propagation_smoke.json")
        print({"precision_propagation": summary})
    finally:
        model.close(best_effort=True)


def test_generator_interface_and_policy_are_explicit():
    assert tuple(inspect.signature(build_generator).parameters) == ("model_dir", "mesh_device", "kwargs")
    generate = inspect.signature(Qwen38Generator.generate).parameters
    assert generate["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert generate["next_input"].kind is inspect.Parameter.KEYWORD_ONLY
    assert generate["sampling_mode"].default is None
    selected, _ = load_precision_config()
    assert selected["logits_sampling"]["sampling_mode"] == "device"
    source = inspect.getsource(Qwen38FullModel)
    assert "MultichipDecoder.from_checkpoint_host_backed" in source
    assert "HostBackedSegmentedDecodeTrace" in source
    assert "tt_out_tok=state.token_input" in source
    assert "ttnn.plus_one(state.current_pos" in source
    assert "Sampling1D" in source
    assert "argmax" not in inspect.getsource(Qwen38FullModel.decode_token_out_traced)
    embed_source = inspect.getsource(Qwen38FullModel.embed_tokens)
    assert "ttnn.upsample" in embed_source
    assert "ttnn.repeat_interleave" not in embed_source
    assert "if rows == 1:" in embed_source
    assert embed_source.index("if rows == 1:") < embed_source.index("ttnn.upsample")
    assert "synchronize_device" not in inspect.getsource(PLEDeviceStaging._upload_replicated)
    assert Qwen38Generator.required_device_params["l1_small_size"] == REQUIRED_L1_SMALL_SIZE


def test_lm_head_dram_frontier_has_exact_vocab_and_tile_geometry():
    """Keep every A/B point sampler-safe before touching weights or hardware."""

    frontier = {name: policy for name, policy in LM_HEAD_POLICIES.items() if policy.dram_sharded}
    assert tuple(frontier) == (
        "bfp8_hifi2_dram_s1_c40",
        "bfp8_hifi2_dram_s4_c40",
        "bfp8_hifi2_dram_s5_c40",
        "bfp8_hifi2_dram_s5_c40_b1",
        "bfp8_hifi2_dram_s8_c40",
        "bfp8_hifi2_dram_s10_c20",
    )
    expected_per_core_n = (97, 25, 20, 20, 13, 20)
    local_vocab = VOCAB_SIZE // 2
    for (name, policy), per_core_n in zip(frontier.items(), expected_per_core_n):
        split_sizes = policy.split_sizes(32_768)
        rank_slices = _lm_head_rank_slices(split_sizes)
        assert policy.weight_dtype == "bfp8"
        assert policy.fidelity == "hifi2"
        assert sum(split_sizes) == local_vocab
        assert all(split_size % 32 == 0 for split_size in split_sizes)
        assert policy.worker_cores is not None
        assert (HIDDEN_SIZE // 32) % policy.worker_cores == 0
        assert ((HIDDEN_SIZE // 32) // policy.worker_cores) % policy.in0_block_w == 0
        assert all(
            (split_size // 32 + policy.worker_cores - 1) // policy.worker_cores == per_core_n
            for split_size in split_sizes
        )
        if name != "bfp8_hifi2_dram_s1_c40":
            assert per_core_n <= 25

        offset = 0
        for split_size, slices in zip(split_sizes, rank_slices):
            assert slices == (
                (offset, offset + split_size),
                (local_vocab + offset, local_vocab + offset + split_size),
            )
            offset += split_size
        assert offset == local_vocab
    assert LM_HEAD_POLICIES["bfp4_lofi"].sampler_dtype == "bf16"


def test_autoregressive_writer_emits_runner_contract(tmp_path):
    report = {
        "prompt_tokens": 3,
        "generation_tokens": 2,
        "hf_tokens": [7, 8],
        "tt_tokens": [7, 9],
        "hf_completion": "HF completion",
        "tt_completion": "TT completion",
    }
    write_autoregressive_artifacts(report, tmp_path)
    metadata = json.loads((tmp_path / "autoregressive_meta.json").read_text())
    assert metadata["hf"]["token_ids"] == [7, 8]
    assert metadata["tt"]["token_ids"] == [7, 9]
    assert (tmp_path / "hf_completion.txt").read_text() == "HF completion\n"
    assert (tmp_path / "tt_completion.txt").read_text() == "TT completion\n"


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_embedding_to_terminal_gather_smoke(bh_1d_mesh_device, device_params, record_property):
    """Isolate endpoint collectives and check real checkpoint top-k numerics."""

    _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
        lm_head_policy=os.getenv("QWEN38_LM_HEAD_POLICY", "bfp8_hifi2"),
    )
    try:
        state = model.new_batch_state([1], request_ids=("endpoint",))
        model.copy_tokens(state, torch.tensor([17], dtype=torch.int64))
        residual = model.embed_tokens(model.decode_token_input)
        gathered = model.layers[-1].gather_residual(residual)
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert tuple(gathered.shape) == (1, 1, 1, 10_240)
        ttnn.deallocate(gathered)

        hidden_tt = model.final_hidden(residual)
        row_seconds = []
        logits = None
        for iteration in range(8):
            started = time.perf_counter()
            candidate = model.project_hidden_logits(hidden_tt)
            ttnn.synchronize_device(bh_1d_mesh_device)
            elapsed = time.perf_counter() - started
            if iteration:
                row_seconds.append(elapsed)
            if logits is not None:
                ttnn.deallocate(logits)
            logits = candidate
        tt_logits = model.logits_to_torch(logits).reshape(-1)
        sampled = model.sample_logits(logits, state)
        sampled_token = int(model.sampled_tokens_to_torch(sampled, state)[0])
        model.set_sampling_params(top_k=5, top_p=0.9, temperature=0.8, seeds=[12_345])
        sampled_topk = model.sample_logits(logits, state)
        sampled_topk_token = int(model.sampled_tokens_to_torch(sampled_topk, state)[0])

        configuration = model.lm_head_configuration()
        all_rows_shape = None
        if configuration["dram_sharded"]:
            hidden33 = ttnn.repeat(hidden_tt, (1, 1, 33, 1))
            all_rows = model.project_hidden_logits(hidden33)
            ttnn.synchronize_device(bh_1d_mesh_device)
            all_rows_host = model.logits_to_torch(all_rows)
            all_rows_shape = tuple(all_rows_host.shape)
            assert all_rows_shape == (1, 1, 33, VOCAB_SIZE)
            assert torch.equal(all_rows_host[..., 0, :], all_rows_host[..., -1, :])
            ttnn.deallocate(all_rows)
            ttnn.deallocate(hidden33)
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(residual)
        ttnn.deallocate(logits)

        prefix = "model.language_model"
        hidden = model.checkpoint.tensor(f"{prefix}.embed_tokens.weight")[17].float()
        hyper = hidden.repeat(4)
        groups = hyper.reshape(4, 2_560)
        groups = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + 1e-6)
        norm_weight = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.hc_norm.weight").float()
        normed = groups.flatten() * (1.0 + norm_weight)
        down = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.input_mix_weight_down.weight").float()
        up = model.checkpoint.tensor(f"{prefix}.hyper_connection_mixer.input_mix_weight_up.weight").float()
        mixing = torch.sigmoid(
            torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(normed, down) / 4.0), up)
        ).reshape(4, 2_560)
        final_hidden = (mixing * normed.reshape(4, 2_560)).mean(0)
        lm_head = model.checkpoint.tensor("lm_head.weight").float()
        reference = torch.mv(lm_head, final_hidden)
        centered_tt = tt_logits - tt_logits.mean()
        centered_reference = reference - reference.mean()
        pcc = float(
            torch.dot(centered_tt, centered_reference)
            / torch.sqrt(torch.dot(centered_tt, centered_tt) * torch.dot(centered_reference, centered_reference))
        )
        reference_top5 = set(torch.topk(reference, 5).indices.tolist())
        reference_top100 = set(torch.topk(reference, 100).indices.tolist())
        tt_top5 = torch.topk(tt_logits, 5).indices.tolist()
        tt_top100 = torch.topk(tt_logits, 100).indices.tolist()
        top5_overlap = len(reference_top5.intersection(tt_top5))
        top100_overlap = len(reference_top100.intersection(tt_top100))
        host_argmax = int(torch.argmax(tt_logits))
        tt_top5_set = set(tt_top5)
        row_mean = sum(row_seconds) / len(row_seconds)
        print(
            {
                "lm_head": configuration,
                "row_seconds": row_seconds,
                "row_mean_seconds": row_mean,
                "pcc": pcc,
                "top5_overlap": top5_overlap,
                "top100_overlap": top100_overlap,
                "device_token": sampled_token,
                "host_argmax": host_argmax,
                "device_topk_topp_token": sampled_topk_token,
                "all_rows_shape": all_rows_shape,
            }
        )
        record_property("lm_head_policy", model.lm_head_policy)
        record_property("lm_head_configuration", json.dumps(configuration, sort_keys=True))
        record_property("lm_head_row_mean_seconds", row_mean)
        record_property("lm_head_pcc", pcc)
        record_property("lm_head_top5_overlap", top5_overlap)
        record_property("lm_head_top100_overlap", top100_overlap)
        record_property("lm_head_device_greedy", sampled_token)
        record_property("lm_head_device_topk_topp", sampled_topk_token)
        record_property("lm_head_all_rows_shape", str(all_rows_shape))
        assert pcc >= 0.995
        assert sampled_token == host_argmax
        assert sampled_topk_token in tt_top5_set
        assert top5_overlap >= 4
        assert top100_overlap >= 98
        assert int(torch.argmax(tt_logits)) in reference_top100
    finally:
        model.close(best_effort=True)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_repeated_nonaligned_embedding_expansion_has_no_tiled_reshape_stall(
    bh_1d_mesh_device,
    device_params,
    record_property,
):
    """Alternate aligned-edge shapes through the endpoint expansion cache.

    A serving run previously completed two length-one prefills and the first
    length-63 prefill, then hung on the repeated length-63 embedding expansion.
    Exercise that exact 1/63/1/63 cache pattern twice and verify the four
    token-major hyper streams remain exact copies on both TP ranks.
    """

    del device_params
    _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    lengths = (1, 63, 1, 63, 1, 63)
    try:
        for iteration, length in enumerate(lengths):
            ids = ((torch.arange(length, dtype=torch.int32) * 17 + 29) % VOCAB_SIZE).reshape(1, 1, 1, length)
            token_host = ttnn.from_torch(
                ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(bh_1d_mesh_device),
            )
            token_device = ttnn.to_device(
                token_host,
                bh_1d_mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            residual = model.embed_tokens(token_device)
            ttnn.synchronize_device(bh_1d_mesh_device)
            assert tuple(residual.shape) == (1, 1, length * 4, 1280)
            for shard in ttnn.get_device_tensors(residual):
                host = ttnn.to_torch(shard).reshape(length, 4, 1280)
                for stream in range(1, 4):
                    assert torch.equal(host[:, 0], host[:, stream])
            ttnn.deallocate(residual)
            ttnn.deallocate(token_device)
            record_property(f"iteration_{iteration}_logical_tokens", length)
        record_property("logical_token_sequence", ",".join(str(length) for length in lengths))
        record_property(
            "endpoint_expansion",
            "decode(rows=1):view-repeat-view;prefill(rows>1):row-major upsample(scale=(1,4))",
        )
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_full_model_endpoint_smoke(bh_1d_mesh_device, device_params):
    """One real optimized layer plus the real embedding/final-mix/LM head."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([1], request_ids=("reduced-real",))
        logits = model.prefill_forward(torch.tensor([[17]], dtype=torch.int64), state=state)
        ttnn.synchronize_device(bh_1d_mesh_device)
        host = model.logits_to_torch(logits)
        assert tuple(host.shape) == (1, 1, 1, VOCAB_SIZE)
        assert torch.isfinite(host).all()
        ttnn.deallocate(logits)
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_split_greedy_trace_contract(bh_1d_mesh_device, device_params):
    """Prove model/sampler split traces, direct feedback, positions, and page refresh."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([3], request_ids=("split-trace",))
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        prefill_logits = model.prefill_forward(torch.tensor([[17, 29, 31]], dtype=torch.int64), state=state)
        prefill_argmax = model.logits_to_torch(prefill_logits).reshape(-1).argmax().reshape(1)
        first = model.sample_logits(prefill_logits, state)
        first_host = model.sampled_tokens_to_torch(first, state)
        assert torch.equal(first_host, prefill_argmax)
        ttnn.deallocate(prefill_logits)

        _, trace_token = model.decode_token_out_traced(state, first_host.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        capture_token = model.sampled_tokens_to_torch(trace_token, state)
        capture_position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert capture_position.tolist() == [4]
        copies_after_capture = (state.token_host_copies, state.position_host_copies)

        assert model.update_page_table(state, state.page_table_host.clone()) is False
        changed_pages = state.page_table_host.flip(-1)
        assert model.update_page_table(state, changed_pages) is True
        assert state.page_table_unchanged_skips == 1
        assert model.trace_page_table_changes == 1

        # Inspect the regenerated terminal tensor before its consumer trace is
        # allowed to reuse that corruptible address, then run sampling alone.
        trace_logits = model.replay_model_only_traced(state, capture_token.reshape(1, 1))
        ttnn.synchronize_device(bh_1d_mesh_device)
        replay_argmax = model.logits_to_torch(trace_logits).reshape(-1).argmax().reshape(1)
        ttnn.execute_trace(bh_1d_mesh_device, model.sampling_trace_id, cq_id=0, blocking=False)
        model.advance_positions_traced()
        ttnn.synchronize_device(bh_1d_mesh_device)
        trace_token = state.token_input
        replay_token = model.sampled_tokens_to_torch(trace_token, state)
        assert torch.equal(replay_token, replay_argmax)
        replay_position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert replay_position.tolist() == [5]
        assert (state.token_host_copies, state.position_host_copies) == copies_after_capture
        assert model.trace_replays == 0
    finally:
        model.close(best_effort=True)


@pytest.mark.skipif(os.getenv("RUN_QWEN38_SAMPLER_AB") != "1", reason="explicit sampler strategy benchmark")
@pytest.mark.parametrize("device_params", [_device_params(trace_region_size=268_435_456)], indirect=True)
def test_reduced_real_weight_split_greedy_sampler_strategy_ab(bh_1d_mesh_device, device_params, record_property):
    """Compare semantically greedy Sampling1D argmax and local-topk paths."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    try:
        state = model.new_batch_state([3], request_ids=("sampler-ab",))
        logits = model.prefill_forward(torch.tensor([[17, 29, 31]], dtype=torch.int64), state=state)
        expected = model.logits_to_torch(logits).reshape(-1).argmax().reshape(1)
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        evidence = {}
        for name, force_argmax in (("full_vocab_argmax", True), ("local_top32_k1", False)):
            model._sampling_force_argmax = force_argmax
            model.sample_logits(logits, state)
            ttnn.synchronize_device(bh_1d_mesh_device)
            samples = []
            started = time.perf_counter()
            for _ in range(7):
                sampled = model.sample_logits(logits, state)
                ttnn.synchronize_device(bh_1d_mesh_device)
                samples.append(model.sampled_tokens_to_torch(sampled, state))
            seconds = time.perf_counter() - started
            assert all(torch.equal(sample, expected) for sample in samples)
            evidence[name] = {"iterations": 7, "total_seconds": seconds, "mean_seconds": seconds / 7}
            record_property(f"{name}_mean_seconds", seconds / 7)
        record_property("selected_strategy", "full_vocab_argmax")
        print({"split_greedy_sampler_strategy_ab": evidence})
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_non_greedy_split_trace_contract(bh_1d_mesh_device, device_params):
    """Top-k/top-p trace replay owns seeded sampling, feedback, and mode changes."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    try:
        state = model.new_batch_state([3], request_ids=("non-greedy-trace",))

        def token_value(tokens):
            return int(model.sampled_tokens_to_torch(tokens, state)[0])

        def position_value():
            shard = ttnn.get_device_tensors(state.current_pos)[0]
            return int(ttnn.to_torch(shard).reshape(-1)[0])

        def topk_set(logits, k=4):
            host = model.logits_to_torch(logits).reshape(-1)
            return set(torch.topk(host, k).indices.tolist())

        def seed_snapshot():
            shard = ttnn.get_device_tensors(model.sampling._seeds)[0]
            return tuple(int(value) for value in ttnn.to_torch(shard).reshape(-1)[: model.max_batch])

        model.set_sampling_params(top_k=4, top_p=0.95, temperature=0.8, seeds=[12345])
        prefill_logits = model.prefill_forward(torch.tensor([[17, 18, 19]]), state=state)
        first_top4 = topk_set(prefill_logits)
        first = model.sample_logits(prefill_logits, state)
        first_host = token_value(first)
        first_seed = seed_snapshot()
        assert first_host in first_top4
        ttnn.deallocate(prefill_logits)

        logits0, tok0 = model.decode_token_out_traced(state, torch.tensor([[first_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok0_host = token_value(tok0)
        assert tok0 is state.token_input
        assert model._sampling_force_argmax is False
        assert tok0_host in topk_set(logits0)
        assert position_value() == 4
        capture_seed = seed_snapshot()
        assert capture_seed != first_seed
        copies_after_capture = (state.token_host_copies, state.position_host_copies)

        assert model.update_page_table(state, state.page_table_host.clone()) is False
        changed_pages = state.page_table_host.roll(shifts=1, dims=-1)
        assert model.update_page_table(state, changed_pages) is True
        assert state.page_table_unchanged_skips == 1
        assert model.trace_page_table_changes == 1

        logits1, tok1 = model.decode_token_out_traced(state, torch.tensor([[tok0_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok1_host = token_value(tok1)
        assert tok1 is state.token_input
        assert tok1_host in topk_set(logits1)
        assert position_value() == 5
        assert (state.token_host_copies, state.position_host_copies) == copies_after_capture
        assert model.trace_replays == 1
        assert seed_snapshot() != capture_seed

        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        assert model._trace_ready is False
        assert model.sampling_trace_id is None
        assert model._sampling_force_argmax is True
        logits2, tok2 = model.decode_token_out_traced(state, torch.tensor([[tok1_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        tok2_host = token_value(tok2)
        assert tok2_host == int(model.logits_to_torch(logits2).reshape(-1).argmax())
        assert position_value() == 6

        model.set_sampling_params(top_k=4, top_p=0.95, temperature=0.8, seeds=[12345])
        assert model._trace_ready is False
        assert model.sampling_trace_id is None
        assert model._sampling_force_argmax is False
        logits3, tok3 = model.decode_token_out_traced(state, torch.tensor([[tok2_host]]))
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert token_value(tok3) in topk_set(logits3)
        assert position_value() == 7
        assert model.sampling_seed_host_copies == 4
    finally:
        model.close(best_effort=True)


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_model_only_trace_compatibility(bh_1d_mesh_device, device_params):
    """Host-logit compatibility deploys model and position, never sampling."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=128,
        layer_indices=(0,),
    )
    generator = Qwen38Generator(model, object())
    try:
        state = generator.allocate_batch_state([3], request_ids=("model-only",))
        logits = model.prefill_forward(torch.tensor([[17, 29, 31]]), state=state)
        ttnn.deallocate(logits)
        host_logits, _ = generator.decode_forward(
            torch.tensor([41]),
            state=state,
            enable_trace=True,
            host_sampling_compatibility=True,
            read_from_device=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert torch.isfinite(host_logits).all()
        assert model._trace_execution_mode == "model_only"
        assert model.sampled_tokens_to_torch(state.token_input).tolist() == [41]
        position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert position.tolist() == [4]

        generator.decode_forward(
            torch.tensor([42]),
            state=state,
            enable_trace=True,
            host_sampling_compatibility=True,
            read_from_device=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        assert model.sampled_tokens_to_torch(state.token_input).tolist() == [42]
        position = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert position.tolist() == [5]
    finally:
        generator.close()


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_mixed_prompts_inactive_row_and_eager_decode(bh_1d_mesh_device, device_params):
    """Mixed non-aligned prompts preserve fixed slots and inactive PLE history."""

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=3,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, object())
    try:
        prompts = torch.full((3, 33), model.pad_token_id, dtype=torch.int64)
        prompts[0, 0] = 17
        prompts[1] = torch.arange(33, dtype=torch.int64) + 29
        state = generator.allocate_batch_state(
            [1, 33, 0],
            active_mask=[True, True, False],
            request_ids=("short", "nonaligned", "inactive"),
        )
        inactive_history = model.ple_store._histories["inactive"].clone()
        logits = model.prefill_forward(prompts, state=state)
        host = model.logits_to_torch(logits)
        ttnn.deallocate(logits)
        assert tuple(host.shape) == (1, 1, 3, VOCAB_SIZE)
        assert torch.isfinite(host).all()
        assert torch.count_nonzero(host[..., 2, :]) == 0
        assert torch.equal(model.ple_store._histories["inactive"], inactive_history)

        output = model.decode_forward(
            torch.tensor([101, 102, model.pad_token_id]),
            state=state,
            enable_trace=False,
            on_device_sampling=True,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        tokens = model.sampled_tokens_to_torch(output)
        assert tuple(tokens.shape) == (3,)
        positions = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1)
        assert positions.tolist() == [2, 34, -1]
        assert torch.equal(model.ple_store._histories["inactive"], inactive_history)
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_FULL_MODEL_BATCH32") != "1",
    reason="explicit all-48-layer batch-32 capability gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_48_layer_batch32_eager_fixed_slots(bh_1d_mesh_device, device_params, record_property):
    """Prove the serving-facing eager cohort at the largest supported batch."""

    _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=32,
        max_seq_len=4096,
        # Full-model performance/Watcher gates exercise the selected model-load
        # preload.  This capability test skips the host-only preload because it
        # cannot affect device capacity, fixed-slot state, or eager semantics.
        prepack_host_experts=False,
    )
    generator = Qwen38Generator(model, object())
    prompts = torch.full((32, 33), model.pad_token_id, dtype=torch.int64)
    prompts[0, 0] = 17
    prompts[31] = torch.arange(33, dtype=torch.int64) + 29
    prompt_lens = [1] + [0] * 30 + [33]
    active_mask = [True] + [False] * 30 + [True]
    pages = model._default_page_table_host.clone()
    pages[1::2] = pages[1::2].flip(-1)

    def histories(prefix):
        return {
            request_id: model.ple_store._histories[request_id].clone()
            for request_id in (f"{prefix}-{slot}" for slot in range(32))
        }

    def run(prefix):
        request_ids = tuple(f"{prefix}-{slot}" for slot in range(32))
        output = generator.generate_batch(
            prompts,
            max_new_tokens=2,
            enable_trace=False,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            prompt_lens=prompt_lens,
            page_table=pages,
            request_ids=request_ids,
            active_mask=active_mask,
            stop_on_eos=False,
        )
        ttnn.synchronize_device(bh_1d_mesh_device)
        state = generator.state
        position_shard = ttnn.get_device_tensors(state.current_pos)[0]
        positions = ttnn.to_torch(position_shard).reshape(-1).tolist()
        assert tuple(output.shape) == (32, 2)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        assert state.active_slots == (0, 31)
        assert positions == [2] + [-1] * 30 + [34]
        assert torch.equal(state.page_table_host, pages)
        assert model.update_page_table(state, pages.clone()) is False
        assert state.page_table_unchanged_skips == 1
        assert state.token_host_copies == 1
        assert state.position_host_copies == 1
        assert state.page_table_host_copies == 1
        assert state.compact_token_readbacks == 2
        assert model._trace_ready is False
        assert model.trace_replays == 0
        current_histories = histories(prefix)
        assert current_histories[f"{prefix}-0"].tolist() == [17, int(output[0, 0])]
        assert current_histories[f"{prefix}-31"].tolist() == [61, int(output[31, 0])]
        for slot in range(1, 31):
            assert current_histories[f"{prefix}-{slot}"].tolist() == [model.ple_store.eos_token_id] * 2
        return output.clone(), current_histories

    try:
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.kv_cache) == 12
        first_output, first_histories = run("batch32-first")
        second_output, second_histories = run("batch32-second")
        assert torch.equal(first_output[[0, 31]], second_output[[0, 31]])
        for request_id, expected in first_histories.items():
            assert torch.equal(model.ple_store._histories[request_id], expected)
        for slot in range(1, 31):
            assert second_histories[f"batch32-second-{slot}"].tolist() == [model.ple_store.eos_token_id] * 2
        audit = model.runtime_fallback_audit(generator.state)
        assert not any(audit["prohibited_host_work"].values())
        record_property("full_layers", len(model.layers))
        record_property("batch", model.max_batch)
        record_property("active_slots", "0,31")
        record_property("prompt_lens", "1,33")
        record_property("repeated_active_outputs", first_output[[0, 31]].tolist())
        record_property("planned_bytes_per_device", 23_393_673_304)
        record_property("headroom_bytes_per_device", 10_831_847_336)
    finally:
        generator.close()


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_reduced_real_weight_generator_token_out_trace_smoke(bh_1d_mesh_device, device_params):
    """Real GDN+PLE+QSA stack exercises split model/sampling/position traces."""

    class DummyTokenizer:
        def decode(self, ids, **_kwargs):
            return " ".join(str(int(value)) for value in ids)

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        layer_indices=(0, 1, 3),
    )
    generator = Qwen38Generator(model, DummyTokenizer())
    try:
        prompt = torch.tensor([[17, 18, 19]], dtype=torch.int64)
        output = generator.generate_batch(
            prompt,
            max_new_tokens=3,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("trace-smoke",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 3)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        audit = model.runtime_fallback_audit(generator.state)
        assert audit["prohibited_host_work"]["optimized_sampling_or_argmax"] is False
        assert audit["prohibited_host_work"]["token_feedback_reconstruction"] is False
        assert audit["counters"]["token_host_copies"] == 2
        assert audit["counters"]["position_host_copies"] == 2
        assert audit["counters"]["page_table_host_copies"] == 1
        assert audit["counters"]["compact_token_readbacks"] == 3
        assert model.trace_replays == 1
        assert model.sampling_trace_id is not None
        assert model.position_trace_id is not None
        assert model.last_decode_timing is not None
        assert "expert_service_seconds" in model.last_decode_timing
        assert "ple_service_seconds" in model.last_decode_timing
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_FULL_MODEL") != "1", reason="explicit full-stack hardware gate")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_48_layer_token_out_trace_smoke(bh_1d_mesh_device, device_params):
    """All 48 real layers execute through the canonical token-out split trace."""

    class DummyTokenizer:
        def decode(self, ids, **_kwargs):
            return " ".join(str(int(value)) for value in ids)

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    slot_shard_counts = {
        layer.shapes.layer_idx: tuple(
            len(ttnn.get_device_tensors(slot.gate_up)) for slot in layer.host_expert_cache.slots
        )
        for layer in model.layers
        if layer.host_expert_cache is not None
    }
    assert all(count == 2 for counts in slot_shard_counts.values() for count in counts), slot_shard_counts
    generator = Qwen38Generator(model, DummyTokenizer())
    try:
        output = generator.generate_batch(
            torch.tensor([[17, 18, 19]], dtype=torch.int64),
            max_new_tokens=3,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("full-stack-smoke",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 3)
        assert torch.all((0 <= output) & (output < VOCAB_SIZE))
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.layer_traces) == 48
        assert model.trace_replays == 1
        assert model.last_decode_timing is not None
        print(
            {
                "tokens": output.tolist(),
                "metrics": generator.last_metrics,
                "audit": model.runtime_fallback_audit(generator.state),
            }
        )
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_CONTEXT_CAPACITY") != "1",
    reason="explicit advertised-context full-stack capacity gate",
)
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_advertised_context_construction(bh_1d_mesh_device, device_params, record_property):
    """Construct every weight/cache/endpoint at the HF 262,144-token limit."""

    provenance = _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=262_144,
        # Exact packed-host residency is orthogonal to device context capacity
        # and is proven by the selected full-model performance/Watcher gates.
        prepack_host_experts=False,
    )
    try:
        assert model.is_full_stack
        assert len(model.layers) == 48
        assert len(model.kv_cache) == 12
        assert tuple(model._default_page_table_host.shape) == (1, 4096)
        assert tuple(model.rot_mats[0].shape) == (1, 1, 262_144, 64)
        assert tuple(model.rot_mats[1].shape) == (1, 1, 262_144, 64)
        assert all(layer.host_expert_cache.capacity == 10 for layer in model.layers)
        persistent = {
            "token": model.decode_token_input,
            "position": model.decode_current_pos,
            "page_table": model.decode_page_table,
            "sampling_k": model.sampling_k,
            "sampling_p": model.sampling_p,
            "sampling_temperature": model.sampling_temp,
            "sampling_index_offsets": model.sampling._index_offsets,
            "sampling_seeds": model.sampling._seeds,
            "sampling_user_ids": model.sampling._user_ids,
            "logprob_mask": model.sampling._log_probs_calculator.mask,
            "logprob_output": model.sampling._log_probs_calculator.output_tensor,
        }
        cache_tensors = [
            tensor for _layer, kv_cache, indexer_cache in model.kv_cache for tensor in (*kv_cache, indexer_cache)
        ]
        report = {
            "config_id": model.precision_config["config_id"],
            "precision_config_path": model.precision_config_path,
            "source_provenance": provenance,
            "max_seq_len": model.max_seq_len,
            "kv_cache_policy": model.precision_config["kv_cache"],
            "qsa_layer_count": len(model.kv_cache),
            "cache_tensor_count": len(cache_tensors),
            "cache_tensor_dtypes": sorted({_dtype_name(tensor.dtype) for tensor in cache_tensors}),
            "cache_tensors": [
                {
                    "shape": tuple(tensor.shape),
                    "padded_shape": tuple(tensor.padded_shape),
                    "volume": tensor.volume(),
                    "dtype": _dtype_name(tensor.dtype),
                    "layout": str(tensor.layout),
                }
                for tensor in cache_tensors
            ],
            "persistent_inputs": {
                name: {
                    "shape": tuple(tensor.shape),
                    "padded_shape": tuple(tensor.padded_shape),
                    "volume": tensor.volume(),
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in persistent.items()
            },
        }
        write_report(report, _evidence_dir() / "advertised_context_construction.json")
        print({"advertised_context_construction": report})
        record_property("config_id", report["config_id"])
        record_property("kv_cache_dtype", report["kv_cache_policy"]["dtype"])
        record_property("cache_tensor_count", report["cache_tensor_count"])
        record_property("max_seq_len", report["max_seq_len"])
    finally:
        model.close(best_effort=True)


REFERENCE = Path(__file__).parents[1] / "doc/full_model/readiness_aime24_chat.refpt"
QUALITATIVE_REFERENCE = Path(__file__).parents[1] / "doc/full_model/qualitative_shared_suite.refpt"


def _evidence_dir() -> Path:
    """Keep later-stage refreshes from overwriting completed-stage evidence."""

    output = Path(
        os.getenv(
            "QWEN38_EVIDENCE_DIR",
            str(Path(__file__).parents[1] / "doc/full_model"),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_DATATYPE_SWEEP") != "1",
    reason="explicit full-model datatype candidate gate",
)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_datatype_sweep_candidate(bh_1d_mesh_device, device_params, record_property):
    """Full 48-layer traced accuracy/performance source for one dtype policy."""

    from transformers import AutoTokenizer

    provenance = _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_teacher_forcing(
            generator,
            REFERENCE,
            enable_trace=True,
            decode_rows=99,
        )
        report.update(
            config_id=model.precision_config["config_id"],
            precision_config_path=model.precision_config_path,
            precision_propagation=model.precision_propagation_summary(),
            source_provenance=provenance,
            model_only_trace_replays=model.model_only_trace_replays,
            measurement_regime=(
                "fully-warm 512 packed experts/layer; traced model-only teacher forcing; "
                "full-logits D2H included; sampling/token feedback excluded"
            ),
            host_service_totals=model.host_service_totals(),
            runtime_fallback_audit=model.runtime_fallback_audit(generator.state),
        )
        if os.getenv("QWEN38_SWEEP_RUN_AUTOREGRESSIVE") == "1":
            report["aime24_autoregressive_100"] = run_autoregressive(generator, REFERENCE, enable_trace=True)
        report["passes_gate"] = (
            report["top1_percent"] >= 90.0 and report["top5_percent"] >= 98.0 and report["top100_percent"] == 100.0
        )
        write_report(report, _evidence_dir() / "candidate_result.json")
        print({"datatype_sweep_candidate": report})
        for key in ("top1_percent", "top5_percent", "top100_percent", "ttft_seconds"):
            record_property(key, report[key])
        record_property("config_id", report["config_id"])
        record_property("traced", report["traced"])
        record_property("model_only_trace_replays", report["model_only_trace_replays"])
        record_property("decode_tokens_per_second_per_user", report["decode_tokens_per_second_per_user"])
        assert report["traced"] is True
        assert report["model_only_trace_replays"] == 98
        assert report["passes_gate"], report
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_prefill_accuracy(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_prefill_check(generator, REFERENCE)
        print({"prefill_accuracy": report})
        for key in ("top1_percent", "top5_percent", "top100_percent"):
            record_property(f"prefill_{key}", report[key])
        assert report["top5_percent"] >= 98.0
        assert report["top100_percent"] == 100.0
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_teacher_forcing_accuracy(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_teacher_forcing(
            generator,
            REFERENCE,
            enable_trace=True,
            decode_rows=int(os.getenv("QWEN38_TEACHER_ROWS", "99")),
        )
        report["precision_propagation"] = model.precision_propagation_summary()
        report["host_service_totals"] = model.host_service_totals()
        report["source_provenance"] = _source_provenance()
        write_report(report, _evidence_dir() / "teacher_forcing_accuracy.json")
        print({"teacher_forcing_accuracy": report})
        for key in ("top1_percent", "top5_percent", "top100_percent"):
            record_property(f"decode_{key}", report[key])
            record_property(f"prefill_{key}", report["prefill"][key])
        for key in (
            "trace_capture_seconds",
            "decode_measured_tokens",
            "decode_seconds",
            "decode_seconds_per_token",
            "decode_tokens_per_second_per_user",
        ):
            record_property(key, report[key])
        assert report["top5_percent"] >= 98.0
        assert report["top100_percent"] == 100.0
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_TRACE_DIAGNOSTIC") != "1", reason="explicit trace-state diagnostic")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_eager_and_traced_teacher_prefix_match(bh_1d_mesh_device, device_params):
    """Localize early decode drift to eager model state or trace replay state."""

    from transformers import AutoTokenizer

    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        eager = run_teacher_forcing(generator, REFERENCE, enable_trace=False, decode_rows=4)
        eager_position = ttnn.to_torch(ttnn.get_device_tensors(generator.state.current_pos)[0]).reshape(-1).tolist()
        traced = run_teacher_forcing(generator, REFERENCE, enable_trace=True, decode_rows=4)
        traced_position = ttnn.to_torch(ttnn.get_device_tensors(generator.state.current_pos)[0]).reshape(-1).tolist()
        evidence = {
            "eager": eager,
            "traced": traced,
            "eager_position": eager_position,
            "traced_position": traced_position,
        }
        print({"teacher_prefix_diagnostic": evidence})
        assert eager_position == traced_position == [205], evidence
        assert eager["tt_top1_tokens"] == traced["tt_top1_tokens"], evidence
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_ACCURACY") != "1", reason="explicit full-stack accuracy gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_aime24_autoregressive_quality(bh_1d_mesh_device, device_params, record_property):
    from transformers import AutoTokenizer

    provenance = _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_autoregressive(generator, REFERENCE, enable_trace=True)
        report["source_provenance"] = provenance
        output_dir = _evidence_dir()
        write_report(report, output_dir / "aime24_autoregressive_100_report_final.json")
        write_autoregressive_artifacts(report, output_dir)
        print({"autoregressive": report})
        assert not report["tt_review"]["mechanically_degenerate"]
        assert report["generation_tokens"] == 100
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_QUALITATIVE_SUITE") != "1", reason="explicit shared qualitative gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_shared_qualitative_suite(bh_1d_mesh_device, device_params, record_property):
    """Run three prompt-correct 128-token HF/TT chat comparisons."""

    from transformers import AutoTokenizer

    provenance = _record_source_provenance(record_property)
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        report = run_qualitative_suite(generator, QUALITATIVE_REFERENCE, enable_trace=True)
        report["source_provenance"] = provenance
        report["config_id"] = model.precision_config["config_id"]
        report["precision_config_path"] = model.precision_config_path
        report["precision_propagation"] = model.precision_propagation_summary()
        report["host_service_totals"] = model.host_service_totals()
        write_report(report, _evidence_dir() / "qualitative_shared_suite_final.json")
        print({"qualitative_shared_suite": report})
        record_property("prompt_ids", ",".join(item["id"] for item in report["prompts"]))
        record_property("generation_length", report["metadata"]["generation_length"])
        record_property("config_id", report["config_id"])
        for item in report["prompts"]:
            assert not item["hf_review"]["mechanically_degenerate"], item
            assert not item["tt_review"]["mechanically_degenerate"], item
            assert item["runtime_fallback_audit"]["prohibited_host_work"]["optimized_sampling_or_argmax"] is False
    finally:
        generator.close()


@pytest.mark.skipif(os.getenv("RUN_QWEN38_PERF") != "1", reason="explicit full-stack performance gate")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_batch1_prompt128_generate128_performance(bh_1d_mesh_device, device_params, record_property):
    """Primary readiness workload: full 48-layer prompt128/generate128 token-out."""

    from transformers import AutoTokenizer

    provenance = _record_source_provenance(record_property)
    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    prompt = torch.as_tensor(reference["prompt_tokens"], dtype=torch.int64).reshape(1, -1)[:, :128]
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model_kwargs = {}
    if "QWEN38_LM_HEAD_POLICY" in os.environ:
        model_kwargs["lm_head_policy"] = os.environ["QWEN38_LM_HEAD_POLICY"]
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        prepack_host_experts=os.getenv("QWEN38_PREPACK_ALL_EXPERTS", "1") == "1",
        **model_kwargs,
    )
    generator = Qwen38Generator(model, AutoTokenizer.from_pretrained(H.MODEL_SNAPSHOT, local_files_only=True))
    try:
        output = generator.generate_batch(
            prompt,
            max_new_tokens=128,
            enable_trace=True,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            request_ids=("batch1-perf-128-128",),
            stop_on_eos=False,
        )
        assert tuple(output.shape) == (1, 128)
        assert model.trace_replays == 126
        report = {
            "workload": {"batch": 1, "prompt_tokens": 128, "generated_tokens": 128},
            "source_provenance": provenance,
            "metrics": generator.last_metrics.report(),
            "last_decode_timing": model.last_decode_timing,
            "host_preload": model.host_preload_report,
            "config_id": model.precision_config["config_id"],
            "precision_config_path": model.precision_config_path,
            "precision_propagation": model.precision_propagation_summary(),
            "runtime_fallback_audit": model.runtime_fallback_audit(generator.state),
        }
        expert_metrics = tuple(report["runtime_fallback_audit"]["experts"].values())
        report["host_service_totals"] = {
            name: sum(float(metrics.get(name, 0)) for metrics in expert_metrics)
            for name in (
                "hits",
                "misses",
                "packed_host_hits",
                "packed_host_misses",
                "h2d_bytes",
                "zero_d2d_bytes",
                "source_pack_seconds",
                "h2d_seconds",
                "index_h2d_bytes",
                "index_upload_seconds",
                "deferred_dma_misses",
                "dma_completion_syncs",
            )
        }
        report["host_service_all_totals"] = model.host_service_totals()
        if "QWEN38_EVIDENCE_DIR" in os.environ:
            write_report(report, _evidence_dir() / "full_model_performance.json")
        print({"full_model_performance": report})
        for key, value in report["metrics"].items():
            record_property(key, value)
        for key, value in report["runtime_fallback_audit"]["counters"].items():
            record_property(key, value)
        for key, value in report["runtime_fallback_audit"]["prohibited_host_work"].items():
            record_property(f"prohibited_{key}", value)
        for key, value in report["host_service_totals"].items():
            record_property(f"host_service_{key}", value)
        record_property("config_id", report["config_id"])
        record_property("precision_propagation_all_fields_consumed", True)
        record_property("generator_host_sampling_compatibility_calls", generator.host_sampling_compatibility_calls)
        assert report["metrics"]["traced"] is True
        assert report["metrics"]["sampling_mode"] == "device"
    finally:
        generator.close()


@pytest.mark.skipif(
    os.getenv("RUN_QWEN38_HOST_COLD_WARM") != "1",
    reason="explicit full-stack cold/warm host-store gate",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
def test_full_model_cold_and_warm_chunked_prefill(bh_1d_mesh_device, device_params, record_property):
    """Repeat prompt128 through real expert/PLE stores without full host residency."""

    _record_source_provenance(record_property)
    reference = torch.load(REFERENCE, map_location="cpu", weights_only=False)
    prompt = torch.as_tensor(reference["prompt_tokens"], dtype=torch.int64).reshape(1, -1)[:, :128]
    bh_1d_mesh_device.reshape(ttnn.MeshShape(*MultichipDecoder.TARGET_MESH))
    model = Qwen38FullModel(
        snapshot=H.MODEL_SNAPSHOT,
        hf_config=H.target_config(),
        mesh_device=bh_1d_mesh_device,
        max_batch=1,
        max_seq_len=4096,
        prepack_host_experts=False,
    )
    generator = Qwen38Generator(model, object())

    def totals():
        expert = [layer.host_expert_cache.metrics() for layer in model.layers]
        return {
            "packed_host_hits": sum(value["packed_host_hits"] for value in expert),
            "packed_host_misses": sum(value["packed_host_misses"] for value in expert),
            "expert_h2d_bytes": sum(value["h2d_bytes"] for value in expert),
            "source_pack_seconds": sum(value["source_pack_seconds"] for value in expert),
            "ple_table_rows_read": model.ple_store.metrics()["table_rows_read"],
            "ple_table_bytes_read": model.ple_store.metrics()["table_bytes_read"],
        }

    try:
        windows = []
        previous = totals()
        model.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        for label in ("cold", "warm"):
            started = time.perf_counter()
            logits = generator.prefill_forward(
                prompt,
                prompt_lens=[128],
                request_ids=(f"{label}-prefill",),
                read_from_device=False,
            )
            ttnn.synchronize_device(bh_1d_mesh_device)
            seconds = time.perf_counter() - started
            sampled = model.sample_logits(logits, generator.state)
            top1 = int(model.sampled_tokens_to_torch(sampled, generator.state)[0])
            ttnn.deallocate(logits)
            current = totals()
            windows.append(
                {
                    "label": label,
                    "seconds": seconds,
                    "top1": top1,
                    "delta": {key: current[key] - previous[key] for key in current},
                }
            )
            previous = current
        print({"full_model_cold_warm_prefill": windows})
        for window in windows:
            label = window["label"]
            record_property(f"{label}_seconds", window["seconds"])
            record_property(f"{label}_top1", window["top1"])
            for key, value in window["delta"].items():
                record_property(f"{label}_{key}", value)
        assert windows[0]["delta"]["packed_host_misses"] > 0
        assert windows[1]["delta"]["packed_host_hits"] > 0
        assert windows[1]["delta"]["packed_host_misses"] == 0
        assert windows[1]["delta"]["ple_table_rows_read"] == 0
        assert windows[0]["top1"] == windows[1]["top1"]
    finally:
        generator.close()
