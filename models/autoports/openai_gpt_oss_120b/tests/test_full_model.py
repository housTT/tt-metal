# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.generator import GREEDY, SAMPLER_DECISION, Generator, TraceEvidence
from models.autoports.openai_gpt_oss_120b.tt.model import (
    HF_CONTEXT_LENGTH,
    MODEL_LAYERS,
    FullModelCapacityError,
    StreamingCheckpoint,
    _LayerAdapter,
    capacity_evidence,
    require_resident_capacity,
)
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

SNAPSHOT = Path(
    "/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/"
    "snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def test_full_stack_capacity_selects_only_physical_tp4_target():
    p150 = capacity_evidence(tp=1)
    p150x2 = capacity_evidence(tp=2)
    p150x4 = capacity_evidence(tp=4)

    assert not p150.fits
    assert not p150x2.fits
    assert p150x4.fits
    assert p150.largest_context_for_batch == 0
    assert p150x2.largest_context_for_batch == 0
    assert p150x4.largest_context_for_batch == HF_CONTEXT_LENGTH
    assert p150x4.kv_cache_bytes == 1_283_457_024
    assert capacity_evidence(tp=4, max_batch_size=10).fits
    assert not capacity_evidence(tp=4, max_batch_size=11).fits
    assert capacity_evidence(tp=4, max_batch_size=11).largest_context_for_batch == 130_880


@pytest.mark.parametrize("tp", [1, 2])
def test_capacity_failure_is_explicit_and_has_no_fallback(tp, expect_error):
    with expect_error(FullModelCapacityError, "forbidden fallback"):
        require_resident_capacity(
            tp=tp,
            max_batch_size=1,
            max_context_length=HF_CONTEXT_LENGTH,
            num_layers=MODEL_LAYERS,
        )


def test_reduced_stack_needs_explicit_probe_authorization(expect_error):
    with expect_error(FullModelCapacityError, "explicit hardware probes"):
        require_resident_capacity(
            tp=4,
            max_batch_size=1,
            max_context_length=128,
            num_layers=2,
        )
    assert require_resident_capacity(
        tp=4,
        max_batch_size=1,
        max_context_length=128,
        num_layers=2,
        allow_reduced_model=True,
    ).fits


@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="pinned GPT-OSS checkpoint is not present")
def test_streaming_checkpoint_covers_every_layer_and_terminal_tensor():
    checkpoint = StreamingCheckpoint(SNAPSHOT)
    for layer_idx in range(MODEL_LAYERS):
        prefix = f"model.layers.{layer_idx}."
        layer_keys = {key[len(prefix) :] for key in checkpoint.weight_map if key.startswith(prefix)}
        assert "self_attn.q_proj.weight" in layer_keys
        assert "mlp.experts.gate_up_proj_blocks" in layer_keys
        assert "mlp.experts.down_proj_scales" in layer_keys
    terminal = checkpoint.terminal_state_dict()
    assert terminal["model.embed_tokens.weight"].shape == (201088, 2880)
    assert terminal["model.norm.weight"].shape == (2880,)
    assert terminal["lm_head.weight"].shape == (201088, 2880)


def test_aime_reference_provenance_matches_payload():
    reference_dir = Path("models/autoports/openai_gpt_oss_120b/doc/full_model/references")
    provenance = json.loads((reference_dir / "aime24_chat_100_top100.provenance.json").read_text())
    reference_path = reference_dir / provenance["artifact"]
    command_path = reference_dir / provenance["command_provenance"]["command_file"]
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == provenance["artifact_sha256"]
    assert reference_path.stat().st_size == provenance["artifact_size_bytes"]
    assert (
        hashlib.sha256(command_path.read_bytes()).hexdigest() == provenance["command_provenance"]["command_file_sha256"]
    )
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)["entries"][0]
    assert _tensor_sha256(reference["prompt_tokens"]) == provenance["prompt"]["prompt_tokens_int64_sha256"]
    assert _tensor_sha256(reference["generated_tokens"]) == provenance["generation"]["generated_tokens_int64_sha256"]
    assert _tensor_sha256(reference["topk_tokens"]) == provenance["generation"]["topk_tokens_int32_sha256"]


class _FakeDecoder:
    def __init__(self):
        self.self_attn = object()
        self.kv_cache = ["k", "v"]
        self.calls = []

    def prefill_forward(self, hidden_states, **kwargs):
        self.calls.append(("prefill", hidden_states, kwargs))
        return "prefill-output"

    def decode_forward(self, hidden_states, **kwargs):
        self.calls.append(("decode", hidden_states, kwargs))
        return "decode-output"


@pytest.mark.parametrize("is_decode", [False, True])
def test_layer_adapter_preserves_explicit_cache_page_position_and_batch_state(is_decode):
    decoder = _FakeDecoder()
    adapter = _LayerAdapter(decoder)
    hidden = torch.zeros(1, 1, 2, 2880) if is_decode else "residual"
    position = torch.tensor([7, -1], dtype=torch.int32) if is_decode else "position"
    output = adapter(
        hidden,
        position_embeddings=["cos", "sin"],
        position_idx=position,
        page_table="pages",
        kv_cache="cache",
        is_decode=is_decode,
        user_id=3,
        batch_size=2,
    )
    mode, hidden, kwargs = decoder.calls[-1]
    assert output == f"{mode}-output"
    if is_decode:
        assert hidden.shape == (1, 1, 2, 2880)
    else:
        assert hidden == "residual"
    assert kwargs["page_table"] == "pages"
    assert kwargs["kv_cache"] == "cache"
    assert kwargs["batch_size"] == 2
    if is_decode:
        assert torch.equal(kwargs["current_position"], position)
    else:
        assert kwargs["user_id"] == 3


def test_readiness_generate_signature_explicitly_requires_trace_keyword():
    signature = inspect.signature(Generator.generate)
    assert "enable_trace" in signature.parameters
    assert signature.parameters["enable_trace"].default is True
    for method in ("prefill_forward", "decode_forward", "generate", "reset"):
        assert callable(getattr(Generator, method))


def test_sampling_decision_uses_canonical_shared_trace_owner():
    assert SAMPLER_DECISION["selected"].endswith("SamplingGenerator")
    assert SAMPLER_DECISION["rejected"].endswith("Sampling1D")
    assert SAMPLER_DECISION["force_argmax"] is False
    assert "top_k=1" in SAMPLER_DECISION["greedy_semantics"]


def test_generator_uses_complete_hf_generation_stop_set():
    generator = Generator.__new__(Generator)
    generator.model_args = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[200002, 199999, 200012]))
    generator.model = SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=200002))
    generator.tokenizer = SimpleNamespace(eos_token_id=200002)
    assert generator._eos_token_ids() == {200002, 199999, 200012}


def test_shared_qualitative_hf_control_contract():
    model_dir = Path("models/autoports/openai_gpt_oss_120b")
    hf_artifact = json.loads(
        (model_dir / "doc/full_model/qualitative/qualitative_hf_chat.json").read_text(encoding="utf-8")
    )
    tt_outputs = json.loads(
        (model_dir / "doc/full_model/qualitative/qualitative_tt_chat.json").read_text(encoding="utf-8")
    )
    provenance = hf_artifact["provenance"]
    assert provenance["model_id"] == "openai/gpt-oss-120b"
    assert provenance["model_revision"] == SNAPSHOT.name
    assert provenance["generation"] == {
        "batch_size": 6,
        "greedy": True,
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 128,
        "eos_token_ids": [200002, 199999, 200012],
        "pad_token_id": 199999,
    }
    hf_outputs = hf_artifact["outputs"]
    assert len(hf_outputs) == len(tt_outputs) == 6
    for expected_id, (hf_entry, tt_entry) in enumerate(zip(hf_outputs, tt_outputs, strict=True)):
        assert hf_entry["id"] == tt_entry["id"] == expected_id
        assert hf_entry["prompt"] == tt_entry["prompt"]
        assert hf_entry["prompt_token_ids"] == tt_entry["prompt_token_ids"]
        assert 1 <= len(hf_entry["completion_token_ids"]) <= 128


def test_trace_evidence_distinguishes_reset_page_change_and_steady_replay():
    generator = Generator.__new__(Generator)
    generator._last_page_table = None
    generator._last_sampling_mode = None
    generator._decode_started = False
    generator.trace_evidence = TraceEvidence()
    pages = torch.arange(8, dtype=torch.int32).reshape(1, 8)

    generator._record_decode_staging(
        page_table=pages,
        sampling_mode="device",
        enable_trace=True,
        reset_batch=True,
    )
    generator._record_decode_staging(
        page_table=pages,
        sampling_mode="device",
        enable_trace=True,
        reset_batch=False,
    )
    changed = pages.clone()
    changed[0, 0] = 7
    generator._record_decode_staging(
        page_table=changed,
        sampling_mode="device",
        enable_trace=True,
        reset_batch=False,
    )

    assert generator.trace_evidence.trace_replays == 3
    assert generator.trace_evidence.full_input_refreshes == 1
    assert generator.trace_evidence.page_table_reuses == 1
    assert generator.trace_evidence.page_table_only_refreshes == 1
    assert generator.trace_evidence.token_input_host_refreshes == 1
    assert generator.trace_evidence.position_rope_host_refreshes == 1
    assert generator.trace_evidence.page_table_host_refreshes == 2
    assert generator.trace_evidence.steady_token_input_host_refreshes == 0
    assert generator.trace_evidence.steady_position_rope_host_refreshes == 0
    assert generator.trace_evidence.steady_page_table_host_refreshes == 1


def test_mixed_prefill_keeps_distinct_physical_rows_with_local_page_coordinate():
    generator = Generator.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=2, max_context_len=128)
    generator.model = SimpleNamespace(n_layers=1, vocab_size=8)
    generator._kv_cache = [["k", "v"]]
    generator._inner = SimpleNamespace(mode="decode")
    generator._dirty_cache = False
    captured = []

    def fake_prefill_one(self, token_ids, page_table_row, layer_cache, *, return_all_logits):
        del self, layer_cache, return_all_logits
        captured.append((token_ids.clone(), page_table_row.clone()))
        return torch.zeros(1, 8)

    generator._prefill_one = MethodType(fake_prefill_one, generator)
    tokens = torch.tensor([[11, 12, 13], [21, 22, 0]], dtype=torch.long)
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    output = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=generator._kv_cache,
        prompt_lens=[3, 2],
    )

    assert output.shape == (2, 1, 8)
    assert [row.tolist() for _, row in captured] == [[[0, 1]], [[2, 3]]]
    assert [row.tolist() for row, _ in captured] == [[[11, 12, 13]], [[21, 22]]]


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_PROBE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_PROBE=1 with the pinned checkpoint for the real-weight probe",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_two_layer_full_model_split_sampling_probe(mesh_device, device_params, reset_seeds):
    """Exercise terminal tensors, both layer kinds, and traced split-greedy feedback."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        max_batch_size=1,
        max_context_length=128,
        num_layers=2,
        allow_reduced_model=True,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    prompt = torch.tensor([[200006, 1734, 25, 392, 876, 13, 200007]], dtype=torch.long)
    pages = generator.page_table[:1]

    host_logits = generator.prefill_forward(
        prompt,
        page_table=pages,
        kv_cache=kv_cache,
        prompt_lens=[prompt.shape[1]],
    )
    expected_first = int(torch.argmax(host_logits[0, 0]).item())
    signpost("FULL_MODEL_TOKEN_OUT")
    predictions = generator.generate(
        prompt[0].tolist(),
        4,
        enable_trace=True,
        sampling_mode="device",
    )
    signpost("FULL_MODEL_TOKEN_OUT_END")
    assert predictions[0] == expected_first
    assert all(0 <= token < model.vocab_size for token in predictions)
    evidence = generator.trace_evidence
    assert evidence.trace_replays == 3
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == 2
    assert evidence.page_table_only_refreshes == 0
    assert evidence.host_argmax_calls == 0
    assert evidence.full_logits_readbacks == 0
    assert evidence.forced_token_refreshes == 0

    trace_inputs = generator._inner.trace_inputs_decode[True][0]

    def first_shard_to_torch(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])

    persistent_token = first_shard_to_torch(trace_inputs[0]).reshape(-1)
    persistent_position = first_shard_to_torch(trace_inputs[1]).reshape(-1)
    persistent_pages = first_shard_to_torch(trace_inputs[3]).reshape(pages.shape)
    assert int(persistent_token[0]) == predictions[-1]
    assert int(persistent_position[0]) == prompt.shape[1] + len(predictions) - 1
    assert torch.equal(persistent_pages, pages)

    changed_pages = pages.clone()
    changed_pages[0, 1] = pages[0, 0]
    changed_result = generator.decode_forward(
        torch.tensor([[predictions[0]]], dtype=torch.long),
        torch.tensor([prompt.shape[1] + len(predictions) - 1], dtype=torch.int64),
        page_table=changed_pages,
        kv_cache=kv_cache,
        enable_trace=True,
        sampling_mode="device",
        sampling_params=None,
        reset_batch=False,
    )
    persistent_token = first_shard_to_torch(trace_inputs[0]).reshape(-1)
    persistent_position = first_shard_to_torch(trace_inputs[1]).reshape(-1)
    persistent_pages = first_shard_to_torch(trace_inputs[3]).reshape(changed_pages.shape)
    assert int(persistent_token[0]) == int(changed_result[0])
    assert int(persistent_position[0]) == prompt.shape[1] + len(predictions)
    assert torch.equal(persistent_pages, changed_pages)
    assert evidence.trace_replays == 4
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == 2
    assert evidence.page_table_only_refreshes == 1
    generator.teardown()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_PROBE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_PROBE=1 with the pinned checkpoint for the real-weight probe",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_two_layer_mixed_prompt_fixed_slots(mesh_device, device_params, reset_seeds):
    """Fill separate cache slots for mixed prompts and preserve an inactive decode row."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        max_batch_size=2,
        max_context_length=128,
        num_layers=2,
        allow_reduced_model=True,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    prompts = torch.tensor(
        [
            [200006, 1734, 25, 392, 876, 13, 200007],
            [200006, 1734, 25, 220, 200007, 0, 0],
        ],
        dtype=torch.long,
    )
    pages = generator.page_table[:2]
    logits = generator.prefill_forward(
        prompts,
        page_table=pages,
        kv_cache=kv_cache,
        prompt_lens=[7, 5],
    )
    assert logits.shape == (2, 1, model.vocab_size)
    assert torch.isfinite(logits).all()

    first_tokens = torch.argmax(logits[:, 0], dim=-1).reshape(2, 1)
    first_tokens[1, 0] = 0
    decoded = generator.decode_forward(
        first_tokens,
        torch.tensor([7, -1], dtype=torch.int64),
        page_table=pages,
        kv_cache=kv_cache,
        enable_trace=True,
        sampling_mode="device",
        sampling_params=GREEDY,
        reset_batch=True,
        prompt_tokens=prompts,
    )
    assert decoded.shape == (2,)
    assert 0 <= int(decoded[0]) < model.vocab_size
    trace_inputs = generator._inner.trace_inputs_decode[True][0]
    persistent_positions = ttnn.to_torch(ttnn.get_device_tensors(trace_inputs[1])[0]).reshape(-1)
    assert int(persistent_positions[0]) == 8
    assert int(persistent_positions[1]) == -1
    generator.teardown()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_PROBE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_PROBE=1 with the pinned checkpoint for the real-weight probe",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_two_layer_split_greedy_matches_host_argmax(mesh_device, device_params, reset_seeds):
    """Verify repeated split sampling is semantically identical to host argmax."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    reference = torch.load(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt",
        map_location="cpu",
        weights_only=False,
    )["entries"][0]
    prompt = reference["prompt_tokens"][0].tolist()
    forced = reference["generated_tokens"][0, :16].tolist()

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        max_batch_size=1,
        max_context_length=2048,
        num_layers=2,
        allow_reduced_model=True,
    )
    generator = Generator(model, args, kv_cache=kv_cache)

    def teacher(step, predicted):
        del predicted
        return forced[step]

    device_predictions = generator.generate(
        prompt,
        len(forced),
        next_input=teacher,
        enable_trace=True,
        sampling_mode="device",
    )
    device_metrics = dict(generator.last_generation_metrics)
    device_evidence = generator.trace_evidence.to_dict()
    host_predictions = generator.generate(
        prompt,
        len(forced),
        next_input=teacher,
        enable_trace=True,
        sampling_mode="host",
    )
    host_metrics = dict(generator.last_generation_metrics)
    host_evidence = generator.trace_evidence.to_dict()
    artifact = {
        "steps": len(forced),
        "device_predictions": device_predictions,
        "host_argmax_predictions": host_predictions,
        "exact_match": device_predictions == host_predictions,
        "device_metrics": device_metrics,
        "host_metrics": host_metrics,
        "device_trace_evidence": device_evidence,
        "host_trace_evidence": host_evidence,
    }
    artifact_path = Path(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/artifacts/split_greedy_host_comparison.json"
    )
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    assert device_predictions == host_predictions
    assert device_evidence["host_argmax_calls"] == 0
    assert device_evidence["full_logits_readbacks"] == 0
    assert host_evidence["host_argmax_calls"] == len(forced)
    generator.teardown()


def _run_batch2_logit_reproducibility(
    mesh_device,
    *,
    num_layers: int,
    max_context_length: int,
    tensor_cache_path: str,
    artifact_path: Path | None = None,
):
    """Exercise the explicit validation-only host-logit compatibility boundary."""

    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    reference = torch.load(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt",
        map_location="cpu",
        weights_only=False,
    )["entries"][0]
    prompt = reference["prompt_tokens"][0].to(torch.long)
    assert prompt.numel() % 32 != 0
    prompts = prompt.repeat(2, 1)

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path=tensor_cache_path,
        max_batch_size=2,
        max_context_length=max_context_length,
        num_layers=num_layers,
        allow_reduced_model=num_layers != MODEL_LAYERS,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    pages = generator.page_table[:2].clone()
    assert not torch.equal(pages[0], pages[1])

    runs = []
    for _ in range(2):
        generator.reset()
        prefill = generator.prefill_forward(
            prompts,
            page_table=pages,
            kv_cache=kv_cache,
            prompt_lens=[prompt.numel(), prompt.numel()],
        )[:, 0, :].clone()
        next_tokens = torch.argmax(prefill, dim=-1).reshape(2, 1)
        decode = generator.decode_forward(
            next_tokens,
            torch.tensor([prompt.numel(), prompt.numel()], dtype=torch.int64),
            page_table=pages,
            kv_cache=kv_cache,
            enable_trace=True,
            sampling_mode="host",
            reset_batch=True,
            force_host_tokens=True,
            prompt_tokens=prompts,
        ).clone()
        runs.append({"prefill": prefill, "decode": decode})

    comparisons = {}
    for phase in ("prefill", "decode"):
        tensor_runs = [run[phase] for run in runs]
        comparisons[phase] = {
            "batch_rows_equal_run_0": torch.equal(tensor_runs[0][0], tensor_runs[0][1]),
            "batch_rows_equal_run_1": torch.equal(tensor_runs[1][0], tensor_runs[1][1]),
            "run_0_equals_run_1_row_0": torch.equal(tensor_runs[0][0], tensor_runs[1][0]),
            "run_0_equals_run_1_row_1": torch.equal(tensor_runs[0][1], tensor_runs[1][1]),
            "max_abs_diff_between_rows": float(torch.max(torch.abs(tensor_runs[0][0] - tensor_runs[0][1]))),
            "max_abs_diff_between_runs": float(torch.max(torch.abs(tensor_runs[0] - tensor_runs[1]))),
            "different_values_between_rows": int(torch.count_nonzero(tensor_runs[0][0] != tensor_runs[0][1])),
            "different_values_between_runs": int(torch.count_nonzero(tensor_runs[0] != tensor_runs[1])),
            "finite": [[bool(torch.isfinite(row).all()) for row in tensor] for tensor in tensor_runs],
            "shape": list(tensor_runs[0].shape),
            "dtype": str(tensor_runs[0].dtype),
            "raw_tensor_sha256": [[_tensor_sha256(row) for row in tensor] for tensor in tensor_runs],
            "argmax_tokens": [torch.argmax(tensor, dim=-1).tolist() for tensor in tensor_runs],
            "top100_token_sha256": [
                [_tensor_sha256(torch.topk(row, 100).indices.to(torch.int64)) for row in tensor]
                for tensor in tensor_runs
            ],
        }
    if artifact_path is not None:
        artifact = {
            "checkpoint_revision": SNAPSHOT.name,
            "mesh_shape": list(mesh_device.shape),
            "num_layers": num_layers,
            "max_batch_size": 2,
            "configured_context": max_context_length,
            "prompt_length": prompt.numel(),
            "prompt_sha256": _tensor_sha256(prompt),
            "physical_page_table_rows": [0, 1],
            "page_table_rows_distinct": not torch.equal(pages[0], pages[1]),
            "boundary": "explicit validation-only host full-logit compatibility mode",
            "comparisons": comparisons,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    for phase in ("prefill", "decode"):
        assert comparisons[phase]["batch_rows_equal_run_0"]
        assert comparisons[phase]["batch_rows_equal_run_1"]
        assert comparisons[phase]["run_0_equals_run_1_row_0"]
        assert comparisons[phase]["run_0_equals_run_1_row_1"]
    generator.teardown()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_PROBE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_PROBE=1 with the pinned checkpoint for the real-weight probe",
)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_two_layer_batch2_logit_reproducibility(mesh_device, device_params, reset_seeds):
    del device_params, reset_seeds
    _run_batch2_logit_reproducibility(
        mesh_device,
        num_layers=2,
        max_context_length=512,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        artifact_path=Path(
            "models/autoports/openai_gpt_oss_120b/doc/full_model/artifacts/logit_reproducibility_probe.json"
        ),
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_ACCEPTANCE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 with the pinned checkpoint for the full-stack gate",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_36_layer_full_context_token_out_smoke(mesh_device, device_params, reset_seeds):
    """Allocate the full contract and run trace-fed greedy token-out decode."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_tensor_cache",
        max_batch_size=1,
        max_context_length=HF_CONTEXT_LENGTH,
        num_layers=MODEL_LAYERS,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    prompts_path = Path("models/demos/deepseek_v3/demo/aime_under_8k_prompts.json")
    prompt_text = json.loads(prompts_path.read_text(encoding="utf-8"))[0]["prompt"]
    prompt = args.encode_prompt(prompt_text)
    assert len(prompt) % 32 != 0, "acceptance prompt must exercise public non-aligned prefill"
    predictions = generator.generate(prompt, 4, enable_trace=True, sampling_mode="device")
    decoded = args.tokenizer.decode(predictions, skip_special_tokens=False)
    assert len(predictions) == 4
    assert len(set(predictions)) > 1
    assert all(0 <= token < model.vocab_size for token in predictions)
    evidence = generator.trace_evidence
    assert evidence.trace_replays == 3
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == 2
    assert evidence.token_input_host_refreshes == 1
    assert evidence.position_rope_host_refreshes == 1
    assert evidence.page_table_host_refreshes == 1
    assert evidence.steady_token_input_host_refreshes == 0
    assert evidence.steady_position_rope_host_refreshes == 0
    assert evidence.steady_page_table_host_refreshes == 0
    assert evidence.caller_visible_token_synchronizations == 4
    assert evidence.validation_full_logit_synchronizations == 0
    assert evidence.full_logits_readbacks == 0
    assert evidence.host_argmax_calls == 0
    artifact = {
        "checkpoint_revision": SNAPSHOT.name,
        "mesh_shape": list(mesh_device.shape),
        "num_layers": MODEL_LAYERS,
        "configured_context": HF_CONTEXT_LENGTH,
        "prompt_length": len(prompt),
        "predicted_tokens": predictions,
        "decoded": decoded,
        "capacity": model.capacity.to_dict(),
        "metrics": generator.last_generation_metrics,
        "trace_evidence": evidence.to_dict(),
    }
    artifact_path = Path(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/artifacts/full_model_token_out_smoke.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    generator.teardown()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_ACCEPTANCE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 with the pinned checkpoint for the full-stack gate",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_real_weight_36_layer_batch2_logit_reproducibility(mesh_device, device_params, reset_seeds):
    del device_params, reset_seeds
    _run_batch2_logit_reproducibility(
        mesh_device,
        num_layers=MODEL_LAYERS,
        max_context_length=HF_CONTEXT_LENGTH,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_tensor_cache",
        artifact_path=Path("models/autoports/openai_gpt_oss_120b/doc/full_model/artifacts/logit_reproducibility.json"),
    )


def _readiness_runner(name: str):
    """Load the pipeline readiness runner without shadowing this checkout's models package."""

    import importlib

    import models.common

    sibling_common = "/home/ttuser/dev/scratch/tt-metal/models/common"
    if sibling_common not in models.common.__path__:
        models.common.__path__.append(sibling_common)
    return importlib.import_module(f"models.common.readiness_check.{name}")


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_READINESS") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_READINESS=1 for the all-layer AIME24 readiness gates",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_run_prefill_check_aime24_top100(mesh_device, device_params, reset_seeds):
    """Run the canonical prefill readiness entry point on the fresh HF reference."""

    del device_params, reset_seeds
    runner = _readiness_runner("run_prefill_check")
    model_dir = Path("models/autoports/openai_gpt_oss_120b").resolve()
    reference = model_dir / "doc/full_model/references/aime24_chat_100_top100.refpt"
    stats = runner.run_prefill_check(
        model_dir=model_dir,
        reference_path=reference,
        mesh_device=mesh_device,
        build_kwargs={
            "snapshot_path": SNAPSHOT,
            "tensor_cache_path": "/tmp/gpt_oss_120b_full_model_tensor_cache",
            "max_seq_len": HF_CONTEXT_LENGTH,
            "max_batch_size": 1,
        },
    )
    assert stats[0]["top5"] >= 0.98
    assert stats[0]["top100"] == 1.0
    artifact = model_dir / "doc/full_model/artifacts/prefill_readiness.json"
    artifact.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_READINESS") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_READINESS=1 for the all-layer AIME24 readiness gates",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_run_teacher_forcing_aime24_top100(mesh_device, device_params, reset_seeds):
    """Run canonical traced teacher forcing on the fresh 100-token HF reference."""

    del device_params, reset_seeds
    runner = _readiness_runner("run_teacher_forcing")
    model_dir = Path("models/autoports/openai_gpt_oss_120b").resolve()
    reference = model_dir / "doc/full_model/references/aime24_chat_100_top100.refpt"
    stats = runner.run_teacher_forcing(
        model_dir=model_dir,
        reference_path=reference,
        mesh_device=mesh_device,
        build_kwargs={
            "snapshot_path": SNAPSHOT,
            "tensor_cache_path": "/tmp/gpt_oss_120b_full_model_tensor_cache",
            "max_seq_len": HF_CONTEXT_LENGTH,
            "max_batch_size": 1,
        },
    )
    assert stats[0]["top5"] >= 0.98
    assert stats[0]["top100"] == 1.0
    artifact = model_dir / "doc/full_model/artifacts/teacher_forcing_readiness.json"
    artifact.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_AUTOREGRESSIVE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_AUTOREGRESSIVE=1 for the full AIME24 comparison",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_run_autoregressive_aime24_hf_and_tt(mesh_device, device_params, reset_seeds, monkeypatch):
    """Produce runner-standard HF/TT completions with the fresh exact HF control."""

    del device_params, reset_seeds
    runner = _readiness_runner("run_autoregressive")
    model_dir = Path("models/autoports/openai_gpt_oss_120b").resolve()
    reference = torch.load(
        model_dir / "doc/full_model/references/aime24_chat_100_top100.refpt",
        map_location="cpu",
        weights_only=False,
    )["entries"][0]
    hf_tokens = reference["generated_tokens"][0].tolist()
    monkeypatch.setattr(runner, "_hf_generate_greedy", lambda **kwargs: list(hf_tokens))

    built = []
    original_import = runner._import_build_generator

    def tracked_import(path):
        build = original_import(path)

        def tracked_build(**kwargs):
            generator = build(**kwargs)
            built.append(generator)
            return generator

        return tracked_build

    monkeypatch.setattr(runner, "_import_build_generator", tracked_import)
    output_dir = model_dir / "doc/full_model/artifacts/autoregressive"
    paths = runner.run_autoregressive(
        model_dir=model_dir,
        hf_model_id=str(SNAPSHOT),
        prompt_file=model_dir / "doc/full_model/prompts/autoregressive_chat_prompt.txt",
        mesh_device=mesh_device,
        output_dir=output_dir,
        max_new_tokens=100,
        build_kwargs={
            "snapshot_path": SNAPSHOT,
            "tensor_cache_path": "/tmp/gpt_oss_120b_full_model_tensor_cache",
            "max_seq_len": HF_CONTEXT_LENGTH,
            "max_batch_size": 1,
        },
    )
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["prompt_token_ids"] == reference["prompt_tokens"][0].tolist()
    assert meta["hf"]["token_ids"] == hf_tokens
    assert meta["tt"]["num_tokens"] == 100
    assert len(set(meta["tt"]["token_ids"])) > 16
    meta["tt"]["metrics"] = built[0].last_generation_metrics
    meta["tt"]["trace_evidence"] = built[0].trace_evidence.to_dict()
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_QUALITATIVE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_QUALITATIVE=1 for the shared six-prompt qualitative suite",
)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (1, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "require_exact_physical_num_devices": True,
                TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
            },
            id="p150x4",
        )
    ],
    indirect=True,
)
def test_shared_qualitative_chat_suite(mesh_device, device_params, reset_seeds):
    """Generate the shared prompt suite through traced on-device greedy sampling."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    model_dir = Path("models/autoports/openai_gpt_oss_120b")
    prompts = [
        prompt.strip()
        for prompt in (model_dir / "doc/full_model/prompts/shared_qualitative_prompts.txt")
        .read_text(encoding="utf-8")
        .split("\n\n")
        if prompt.strip()
    ]
    assert len(prompts) == 6
    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_tensor_cache",
        max_batch_size=1,
        max_context_length=HF_CONTEXT_LENGTH,
        num_layers=MODEL_LAYERS,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    outputs = []
    for index, prompt in enumerate(prompts):
        prompt_tokens = args.encode_prompt(prompt)
        completion_tokens = generator.generate(
            prompt_tokens,
            128,
            enable_trace=True,
            sampling_mode="device",
            stop_on_eos=True,
        )
        completion = args.tokenizer.decode(completion_tokens, skip_special_tokens=False)
        assert 1 <= len(completion_tokens) <= 128
        assert len(set(completion_tokens)) > min(8, len(completion_tokens) // 2)
        outputs.append(
            {
                "id": index,
                "prompt": prompt,
                "prompt_token_ids": prompt_tokens,
                "completion_token_ids": completion_tokens,
                "completion": completion,
                "metrics": dict(generator.last_generation_metrics),
                "trace_evidence": generator.trace_evidence.to_dict(),
            }
        )
    artifact = model_dir / "doc/full_model/qualitative/qualitative_tt_chat.json"
    artifact.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    generator.teardown()
