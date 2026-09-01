# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tt import generator as generator_module
from models.autoports.openai_gpt_oss_120b.tt import precision as precision_module
from models.autoports.openai_gpt_oss_120b.tt.generator import GREEDY, SAMPLER_DECISION, Generator, TraceEvidence
from models.autoports.openai_gpt_oss_120b.tt.model import (
    DRAM_SHARDED_LM_HEAD,
    HF_CONTEXT_LENGTH,
    INTERLEAVED_LM_HEAD,
    MODEL_LAYERS,
    FullModelCapacityError,
    Model,
    StreamingCheckpoint,
    _LayerAdapter,
    capacity_evidence,
    require_resident_capacity,
)
from models.autoports.openai_gpt_oss_120b.tt.precision import (
    DEFAULT_PRECISION_CONFIG_PATH,
    PrecisionConfig,
    load_precision_config,
)
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

SNAPSHOT = Path(
    "/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/"
    "snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
)

_RUNTIME_SOURCE_PATHS = (
    "models/autoports/openai_gpt_oss_120b/tests/test_full_model.py",
    "models/autoports/openai_gpt_oss_120b/tt/generator.py",
    "models/autoports/openai_gpt_oss_120b/tt/model.py",
    "models/autoports/openai_gpt_oss_120b/tt/multichip_decoder.py",
    "models/autoports/openai_gpt_oss_120b/tt/precision.py",
    "models/demos/gpt_oss/tt/attention/__init__.py",
    "models/demos/gpt_oss/tt/attention/operations.py",
    "models/demos/gpt_oss/tt/attention/prefill.py",
    "models/demos/gpt_oss/tt/model.py",
    "models/demos/gpt_oss/tt/rms_norm.py",
    "models/common/sampling/generator.py",
    "models/common/sampling/tt_sampling.py",
    "models/tt_transformers/tt/generator.py",
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _runtime_source_provenance() -> dict:
    repo = Path(__file__).resolve().parents[4]
    files = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in _RUNTIME_SOURCE_PATHS}
    state_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "repo": str(repo),
        "source_branch": _git_text(repo, "branch", "--show-current"),
        "source_commit": _git_text(repo, "rev-parse", "HEAD"),
        "source_dirty": bool(_git_text(repo, "status", "--porcelain", "--", *_RUNTIME_SOURCE_PATHS)),
        "runtime_source_state_sha256": hashlib.sha256(state_payload).hexdigest(),
        "runtime_source_files": files,
    }


def _runner_provenance(module) -> dict:
    source_path = Path(inspect.getsourcefile(module)).resolve()
    runner_repo = source_path.parents[3]
    return {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "repo": str(runner_repo),
        "branch": _git_text(runner_repo, "branch", "--show-current"),
        "commit": _git_text(runner_repo, "rev-parse", "HEAD"),
        "last_change_commit": _git_text(runner_repo, "log", "-1", "--format=%H", "--", str(source_path)),
        "git_blob": _git_text(runner_repo, "hash-object", str(source_path)),
        "dirty": bool(_git_text(runner_repo, "status", "--porcelain", "--", str(source_path))),
    }


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
    # KV capacity is reserved in whole 128-token decode K chunks.  The prior
    # page-only estimate advertised 130880 while its padded final SDPA read
    # exceeded DRAM; 130816 is the largest physically safe logical value.
    assert capacity_evidence(tp=4, max_batch_size=11).largest_context_for_batch == 130_816


@pytest.mark.parametrize(
    ("dtype", "tp", "expected_total_bytes", "expected_kv_bytes", "expected_fits", "expected_context"),
    [
        (ttnn.bfloat4_b, 1, 72_479_340_672, 2_717_908_992, False, 0),
        (ttnn.bfloat4_b, 2, 38_064_149_376, 1_358_954_496, False, 0),
        (ttnn.bfloat4_b, 4, 20_939_093_376, 679_477_248, True, HF_CONTEXT_LENGTH),
        (ttnn.bfloat8_b, 1, 74_895_259_776, 5_133_828_096, False, 0),
        (ttnn.bfloat8_b, 2, 39_272_108_928, 2_566_914_048, False, 0),
        (ttnn.bfloat8_b, 4, 21_543_073_152, 1_283_457_024, True, HF_CONTEXT_LENGTH),
        (ttnn.bfloat16, 1, 79_425_108_096, 9_663_676_416, False, 0),
        (ttnn.bfloat16, 2, 41_537_033_088, 4_831_838_208, False, 0),
        (ttnn.bfloat16, 4, 22_675_535_232, 2_415_919_104, True, HF_CONTEXT_LENGTH),
    ],
)
def test_kv_candidate_capacity_uses_physical_dtype_tile_bytes(
    dtype, tp, expected_total_bytes, expected_kv_bytes, expected_fits, expected_context
):
    evidence = capacity_evidence(tp=tp, kv_cache_dtype=dtype)

    assert evidence.total_bytes_per_device == expected_total_bytes
    assert evidence.kv_cache_bytes == expected_kv_bytes
    assert evidence.fits is expected_fits
    assert evidence.largest_context_for_batch == expected_context


def test_selected_precision_is_the_default_and_all_candidates_validate(monkeypatch):
    monkeypatch.delenv("GPT_OSS_120B_PRECISION_CONFIG", raising=False)
    selected = load_precision_config()
    candidate_root = DEFAULT_PRECISION_CONFIG_PATH.parent / "candidates"
    candidates = [load_precision_config(path) for path in sorted(candidate_root.glob("*.json"))]

    assert selected.source_path == DEFAULT_PRECISION_CONFIG_PATH.resolve()
    assert selected.config_id == "ds00_baseline"
    assert selected.to_dict() == load_precision_config(candidate_root / "ds00_baseline.json").to_dict()
    assert [candidate.config_id for candidate in candidates] == [
        "ds00_baseline",
        "ds01_attention_bfp4_lofi",
        "ds02_attention_bfp4_hifi2",
        "ds03_expert_bfp4_hifi2",
        "ds04_canonical_bfp8_hifi2",
        "ds05_kv_bfp4",
        "ds06_kv_bf16",
        "ds07_expert_ccl_bfp8",
        "ds08_all_ccl_bfp4",
        "ds09_expert_intermediate_bfp8",
        "ds10_attention_bfp8_hifi2",
        "ds11_lm_head_lofi",
        "ds12_attention_hifi2_lm_head_lofi",
    ]


def test_precision_mutations_propagate_or_fail_explicitly(monkeypatch, expect_error):
    raw = load_precision_config().to_dict()
    sentinel = object()
    monkeypatch.setitem(precision_module._DTYPES, "bfloat16", sentinel)
    mutated = PrecisionConfig(copy.deepcopy(raw))

    assert mutated.terminal_dtypes()["normalization"] is sentinel
    assert all(
        mutated.decoder_policy_for_layer(layer).normalization_weight_dtype is sentinel for layer in range(MODEL_LAYERS)
    )

    ignored_terminal = copy.deepcopy(raw)
    ignored_terminal["layer_exceptions"] = {"0": {"weight_groups": {"lm_head": "bfloat8_b"}}}
    with expect_error(ValueError, "unsupported layer exception weight_groups"):
        PrecisionConfig(ignored_terminal)

    materialized = copy.deepcopy(raw)
    materialized["logits_sampling_dtype_assumptions"]["full_logits_gather"]["mode"] = "full_vocab"
    with expect_error(ValueError, "requires full_logits_gather.mode=not_materialized"):
        PrecisionConfig(materialized)

    non_bf16_norm = copy.deepcopy(raw)
    non_bf16_norm["weight_groups"]["normalization"] = "bfloat8_b"
    with expect_error(ValueError, "requires bfloat16 normalization weights"):
        PrecisionConfig(non_bf16_norm)


def _fake_precision_runtime_model(config: PrecisionConfig) -> Model:
    model = Model.__new__(Model)
    terminal = config.terminal_dtypes()
    model.precision_config = config
    model.embedding_weight = SimpleNamespace(dtype=terminal["embedding"])
    model.norm = SimpleNamespace(tt_weight=SimpleNamespace(dtype=terminal["normalization"]))
    model.lm_head_weight = SimpleNamespace(dtype=terminal["lm_head_weight"])
    model.lm_head_output_dtype = terminal["lm_head_output"]
    model.lm_head_compute_kernel_config = SimpleNamespace(math_fidelity=terminal["lm_head_math_fidelity"])
    model.sampling_accumulator_dtype = terminal["sampling_accumulator"]
    model.full_logits_gather_policy = terminal["full_logits_gather"]
    model.topk_values_gather_dtype = terminal["topk_values_gather_dtype"]
    sampling_tensors = {
        name: SimpleNamespace(dtype=terminal["sampling_accumulator"])
        for name in ("p_tensor", "temp_tensor", "_greedy_col")
    }
    model.sampling = SimpleNamespace(
        tt_sampling=SimpleNamespace(
            **sampling_tensors,
            _allow_force_argmax_sampling=False,
            _force_argmax_sampling=False,
        )
    )
    model.layers = []
    for layer_idx in range(MODEL_LAYERS):
        policy = config.decoder_policy_for_layer(layer_idx)
        attention = SimpleNamespace(
            weights=SimpleNamespace(
                wqkv=SimpleNamespace(dtype=policy.attention_weight_dtype),
                o_proj=SimpleNamespace(dtype=policy.attention_weight_dtype),
            ),
            activation_ccl_dtype=policy.attention_activation_ccl_dtype,
            residual_dtype=policy.residual_dtype,
            prefill_projection_input_dtype=policy.attention_projection_input_dtype,
            decode_projection_compute_kernel_config=SimpleNamespace(math_fidelity=policy.projection_math_fidelity),
            prefill_projection_compute_kernel_config=SimpleNamespace(
                math_fidelity=policy.prefill_projection_math_fidelity
            ),
            program_config=SimpleNamespace(math_fidelity=policy.attention_sdpa_math_fidelity.name),
        )
        mlp = SimpleNamespace(
            indexed_gate_up=SimpleNamespace(dtype=policy.expert_weight_dtype),
            indexed_down=SimpleNamespace(dtype=policy.expert_weight_dtype),
            router=SimpleNamespace(
                weight=SimpleNamespace(dtype=policy.router_weight_dtype),
                compute_config=SimpleNamespace(math_fidelity=policy.router_math_fidelity),
            ),
            activation_ccl_dtype=policy.expert_activation_ccl_dtype,
            expert_intermediate_dtype=policy.expert_intermediate_dtype,
            expert_compute_kernel_config=SimpleNamespace(math_fidelity=policy.expert_math_fidelity),
        )
        decoder = SimpleNamespace(
            policy=policy,
            self_attn=attention,
            mlp=mlp,
            input_layernorm=SimpleNamespace(tt_weight=SimpleNamespace(dtype=policy.normalization_weight_dtype)),
            post_attention_layernorm=SimpleNamespace(
                tt_weight=SimpleNamespace(dtype=policy.normalization_weight_dtype)
            ),
            kv_cache=[SimpleNamespace(dtype=policy.kv_cache_dtype), SimpleNamespace(dtype=policy.kv_cache_dtype)],
        )
        model.layers.append(SimpleNamespace(decoder=decoder))
    return model


def test_precision_runtime_validation_uses_actual_terminal_and_both_layer_norms(expect_error):
    model = _fake_precision_runtime_model(load_precision_config())
    model._validate_precision_runtime()
    evidence = model.precision_runtime_evidence()

    assert evidence["schema_version"] == 2
    assert sorted(layer for group in evidence["layer_runtime_groups"] for layer in group["layers"]) == list(
        range(MODEL_LAYERS)
    )
    assert all(
        group["input_normalization_weight"] == "bfloat16" and group["post_attention_normalization_weight"] == "bfloat16"
        for group in evidence["layer_runtime_groups"]
    )
    for tensor in (
        model.norm.tt_weight,
        model.layers[0].decoder.input_layernorm.tt_weight,
        model.layers[0].decoder.post_attention_layernorm.tt_weight,
    ):
        original = tensor.dtype
        tensor.dtype = ttnn.bfloat8_b
        with expect_error(RuntimeError, "normalization"):
            model._validate_precision_runtime()
        tensor.dtype = original

    model.sampling.tt_sampling._allow_force_argmax_sampling = True
    with expect_error(RuntimeError, "full-vocabulary argmax"):
        model._validate_precision_runtime()


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


def test_generator_page_table_covers_padded_final_decode_chunk():
    generator = Generator.__new__(Generator)
    generator.model_args = SimpleNamespace(
        max_batch_size=2,
        max_context_len=130,
        physical_kv_context_len=256,
    )

    page_table = generator.allocate_page_table()

    assert page_table.shape == (2, 4)
    assert page_table.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_vllm_cache_owner_has_no_private_page_table(monkeypatch, expect_error):
    monkeypatch.setattr(generator_module, "_TTGenerator", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        Generator,
        "allocate_page_table",
        lambda self: pytest.fail("vLLM ownership must not allocate a private page table"),
    )
    generator = Generator(
        SimpleNamespace(mesh_device=object(), kv_cache=[]),
        SimpleNamespace(tokenizer=object()),
        cache_owner="vllm",
    )

    assert generator.page_table is None
    with expect_error(RuntimeError, "scheduler-owned page table"):
        generator._require_private_page_table()


def test_prepared_decode_trace_uses_no_write_positions_for_live_kv():
    generator = Generator.__new__(Generator)
    generator.model = SimpleNamespace(n_layers=1)
    live_kv = torch.arange(8, dtype=torch.int32)
    original_kv = live_kv.clone()
    captured = {}

    def prepare_decode_trace(token_chunks, position_chunks, *, page_table, **kwargs):
        positions = torch.cat(position_chunks)
        captured.update(tokens=torch.cat(token_chunks), positions=positions, page_table=torch.cat(page_table))
        # Mirror paged_update_cache's public -1 skip contract.  This fake
        # deliberately mutates live KV for every active position so the test
        # catches a regression back to the former synthetic position zero.
        for position in positions.tolist():
            if position >= 0:
                live_kv[position] = -1
        return {"on_device_sampling": kwargs["on_device_sampling"]}

    generator._inner = SimpleNamespace(data_parallel=2, _prepare_decode_trace_text=prepare_decode_trace)

    prepared = generator.prepare_model_decode_trace(
        kv_cache=["layer-cache"],
        max_batch_size=4,
        num_blocks=3,
    )

    assert captured["tokens"].shape == (4, 1)
    assert captured["positions"].tolist() == [-1, -1, -1, -1]
    assert captured["page_table"].shape == (4, 3)
    assert torch.equal(live_kv, original_kv)
    assert prepared["on_device_sampling"] is True


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
    assert "sampling_params" in signature.parameters
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

    # Staging records the requested decode shape only. Actual replay counters
    # advance exclusively at successful ttnn.execute_trace submissions.
    assert generator.trace_evidence.trace_replays == 0
    assert generator.trace_evidence.model_execute_submissions == 0
    assert generator.trace_evidence.sampling_execute_submissions == 0
    assert generator.trace_evidence.full_input_refreshes == 1
    assert generator.trace_evidence.page_table_reuses == 1
    assert generator.trace_evidence.page_table_only_refreshes == 1
    assert generator.trace_evidence.token_input_host_refreshes == 1
    assert generator.trace_evidence.position_rope_host_refreshes == 1
    assert generator.trace_evidence.page_table_host_refreshes == 2
    assert generator.trace_evidence.steady_token_input_host_refreshes == 0
    assert generator.trace_evidence.steady_position_rope_host_refreshes == 0
    assert generator.trace_evidence.steady_page_table_host_refreshes == 1


def test_reset_clears_resident_cache_before_reusing_fixed_slots():
    class CacheOwner:
        def __init__(self):
            self.clear_calls = 0

        def clear_kv_caches(self):
            self.clear_calls += 1

    generator = Generator.__new__(Generator)
    cache_owner = CacheOwner()
    generator.model = cache_owner
    generator.cache_owner = "model"
    generator._inner = SimpleNamespace(
        mode="decode",
        prev_page_table=torch.ones(1, 2),
        _prev_on_device_sampling=True,
        _slots_prefilled_since_decode={0},
    )
    generator._dirty_cache = True
    generator._last_page_table = torch.ones(1, 2, dtype=torch.int32)
    generator._last_sampling_mode = "device"
    generator._decode_started = True
    generator._greedy_sampling_prepared = True
    generator.trace_evidence = TraceEvidence(decode_calls=7)

    generator.reset()

    assert cache_owner.clear_calls == 1
    assert not generator._dirty_cache
    assert generator._inner.mode is None
    assert generator._inner.prev_page_table is None
    assert generator._inner._prev_on_device_sampling is None
    assert generator._inner._slots_prefilled_since_decode == set()
    assert generator._last_page_table is None
    assert generator._last_sampling_mode is None
    assert not generator._decode_started
    assert not generator._greedy_sampling_prepared
    assert generator.trace_evidence == TraceEvidence()

    generator.reset()
    assert cache_owner.clear_calls == 1


def test_unseen_prefill_variant_releases_live_decode_and_sampling_traces(monkeypatch):
    class FakeSampling:
        reset_calls = 0

        def reset_trace(self):
            self.reset_calls += 1

    sampling = FakeSampling()
    inner = SimpleNamespace(
        model=[SimpleNamespace(sampling=sampling)],
        model_args=[SimpleNamespace(mesh_device="mesh")],
        trace_ids_decode={True: {0: 41}},
        trace_inputs_decode={True: {0: ["persistent-input"]}},
        trace_output_decode={True: {0: "persistent-output"}},
        mode="decode",
        prev_page_table=torch.ones(1, 2),
        _prev_on_device_sampling=True,
        _slots_prefilled_since_decode={0},
    )
    generator = Generator.__new__(Generator)
    generator.mesh_device = "mesh"
    generator._inner = inner
    warmed = generator._prefill_program_signature(128, "device_sampling")
    generator._compiled_prefill_variants = {warmed}
    generator._lifetime_prefill_variant_compilations = 1
    generator._lifetime_decode_trace_releases_for_prefill_compile = 0
    generator._greedy_sampling_prepared = True
    generator.trace_evidence = TraceEvidence()
    synchronized = []
    released = []
    monkeypatch.setattr(ttnn, "synchronize_device", synchronized.append)
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: released.append((mesh, trace_id)))

    unseen = generator._prepare_prefill_variants([3, 128], path="device_sampling")

    short_prompt = generator._prefill_program_signature(3, "device_sampling")
    assert unseen == {short_prompt}
    assert synchronized == ["mesh"]
    assert released == [("mesh", 41)]
    assert sampling.reset_calls == 1
    assert inner.trace_ids_decode == {}
    assert inner.trace_inputs_decode == {}
    assert inner.trace_output_decode == {}
    assert inner.mode is None
    assert inner.prev_page_table is None
    assert inner._prev_on_device_sampling is None
    assert inner._slots_prefilled_since_decode == set()
    assert not generator._greedy_sampling_prepared
    assert generator.trace_evidence.decode_trace_releases_for_prefill_compile == 1

    generator._record_compiled_prefill_variants(unseen)
    assert generator._prepare_prefill_variants([3, 128], path="device_sampling") == set()
    assert synchronized == ["mesh"]
    assert released == [("mesh", 41)]
    assert generator._lifetime_prefill_variant_compilations == 2
    assert generator._lifetime_decode_trace_releases_for_prefill_compile == 1


def test_split_greedy_submission_has_one_explicit_collection_boundary():
    class FakeSampling:
        def __init__(self):
            self.calls = []

        def sample(self, **kwargs):
            self.calls.append(kwargs)
            return "sampled-device-output"

    sampling = FakeSampling()
    inner_model = SimpleNamespace(sampling=sampling)

    class FakeInner:
        model = [inner_model]
        mode = "decode"
        trace_inputs_decode = {True: {0: ["persistent-token"]}}
        decode_calls = []

        @staticmethod
        def _decode_token_feedback_buffer(model, trace_inputs):
            del model
            return trace_inputs[0]

        def decode_forward(self, **kwargs):
            self.decode_calls.append(kwargs)
            if kwargs.get("defer_device_sampling"):
                return ["device-logits"]
            return ["initialized-device-output"]

        @staticmethod
        def read_decode_output(output, async_read=False):
            assert output == ["sampled-device-output"]
            assert not async_read
            return ["host-token"]

        @staticmethod
        def process_decode_output_host(output, is_tokens=False):
            assert output == ["host-token"]
            assert is_tokens
            return torch.tensor([17])

    generator = Generator.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=1, max_context_len=128)
    generator._inner = FakeInner()
    generator._kv_cache = ["cache"]
    generator._last_page_table = None
    generator._last_sampling_mode = None
    generator._decode_started = False
    generator._greedy_sampling_prepared = False
    generator._dirty_cache = False
    generator.trace_evidence = TraceEvidence()
    pages = torch.arange(2, dtype=torch.int32).reshape(1, 2)
    tokens = torch.tensor([[11]], dtype=torch.long)

    device_output = generator.decode_forward(
        tokens,
        torch.tensor([7]),
        page_table=pages,
        kv_cache=generator._kv_cache,
        sampling_params=GREEDY,
        reset_batch=True,
        slot_remap=torch.tensor([0], dtype=torch.int32),
        read_from_device=False,
    )
    assert device_output == ["initialized-device-output"]
    for position in (8, 9):
        device_output = generator.decode_forward(
            tokens,
            torch.tensor([position]),
            page_table=pages,
            kv_cache=generator._kv_cache,
            reuse_greedy_sampling_state=True,
            read_from_device=False,
        )
    collected = generator.read_decode_output(device_output)

    assert collected.tolist() == [17]
    assert len(sampling.calls) == 2
    assert all(call["tt_out_tok"] == "persistent-token" for call in sampling.calls)
    evidence = generator.trace_evidence
    assert evidence.device_token_out_submissions == 3
    assert evidence.sampling_state_host_refreshes == 1
    assert evidence.fixed_greedy_sampling_replays == 2
    assert evidence.decode_output_collections == 1
    assert evidence.sampled_token_readbacks == 1
    assert evidence.caller_visible_token_synchronizations == 1
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == 2


def test_minimal_token_read_selects_one_tp_replica_per_distinct_row(expect_error):
    shards = list(range(8))

    assert Generator._select_minimal_token_shards(shards, users_row_sharded=False, mesh_cols=4) == [0]
    assert Generator._select_minimal_token_shards(shards, users_row_sharded=True, mesh_cols=4) == [0, 4]

    with expect_error(RuntimeError, "mesh_cols=3"):
        Generator._select_minimal_token_shards(shards, users_row_sharded=True, mesh_cols=3)


def test_mixed_prefill_keeps_distinct_physical_rows_with_local_page_coordinate():
    generator = Generator.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=2, max_context_len=128)
    generator.model = SimpleNamespace(n_layers=1, vocab_size=8)
    generator._kv_cache = [["k", "v"]]
    generator._inner = SimpleNamespace(mode="decode")
    generator._inner.trace_ids_decode = {}
    generator._compiled_prefill_variants = set()
    generator._lifetime_prefill_variant_compilations = 0
    generator._lifetime_decode_trace_releases_for_prefill_compile = 0
    generator._dirty_cache = False
    generator.trace_evidence = TraceEvidence()
    captured = []

    def fake_prefill_one(
        self,
        token_ids,
        page_table_row,
        layer_cache,
        *,
        return_all_logits,
        page_tables_per_layer=None,
    ):
        del self, layer_cache, return_all_logits, page_tables_per_layer
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
    os.environ.get("GPT_OSS_120B_SAMPLER_TRACE_PROBE") != "1",
    reason="set GPT_OSS_120B_SAMPLER_TRACE_PROBE=1 for the TP4 sampling-trace isolation",
)
@pytest.mark.timeout(600)
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
def test_tp4_sampling_trace_tracks_production_state_matrix(mesh_device, device_params, reset_seeds):
    """Isolate sampler trace replay with production-shaped mutable state.

    Cover the two differences omitted by the original sampler-only probe: the
    decode token-feedback output tensor and the per-step greedy parameter
    refresh performed by the canonical synchronous generator.  Also queue a
    no-readback replay train to exercise the optimized split boundary.
    """

    del device_params, reset_seeds
    vocab_size = 201_088
    padded_vocab_size = 262_144
    batch_size = 32
    args = SimpleNamespace(
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab_size,
        cluster_shape=tuple(mesh_device.shape),
        sampling_all_gather_axis=1,
        sampling_dp=1,
        max_batch_size=1,
        max_top_k=32,
        sub_core_grids=None,
        sub_core_grid_topk=None,
        use_topk_logprobs=True,
        model_config={},
    )
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=mesh_device.shape)

    def host_logits(token, runner_up):
        logits = torch.full((1, 1, batch_size, padded_vocab_size), -10.0, dtype=torch.bfloat16)
        logits[..., token] = 23.5
        logits[..., runner_up] = 23.0
        logits[..., vocab_size:] = -float("inf")
        return ttnn.from_torch(
            logits,
            device=None,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

    host_a = host_logits(5310, 1131)
    host_b = host_logits(1131, 5310)
    tt_logits = ttnn.to_device(host_a, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    host_feedback = ttnn.from_torch(
        torch.zeros(1, 1, 1, batch_size, dtype=torch.int32),
        device=None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_feedback = ttnn.to_device(host_feedback, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def read_first_token(sampled):
        tokens = sampled[0] if isinstance(sampled, tuple) else sampled
        return int(ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).reshape(-1)[0])

    cases = []
    expected = [5310 if step % 2 == 0 else 1131 for step in range(128)]
    for use_feedback in (False, True):
        for refresh_params in (False, True):
            sampling.reset_trace()
            sampling.reset_sampling_params(format_sampling_params(GREEDY, batch_size))
            sampling.seed_manager.reset_seed(None, list(range(batch_size)))
            feedback = tt_feedback if use_feedback else None
            observed = []
            for step, expected_token in enumerate(expected):
                ttnn.copy_host_to_device_tensor(host_a if expected_token == 5310 else host_b, tt_logits)
                if refresh_params:
                    sampling.apply_decode_state([GREEDY], reset_batch=(step == 0))
                sampling.seed_manager.get_new_values()
                sampled = sampling.sample(tt_logits, tt_out_tok=feedback, enable_trace=True)
                ttnn.synchronize_device(mesh_device)
                observed.append(read_first_token(sampled))
            cases.append(
                {
                    "feedback": use_feedback,
                    "refresh_params_each_step": refresh_params,
                    "collection": "per_step",
                    "expected": expected,
                    "observed": observed,
                    "first_divergence": next(
                        (index for index, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]),
                        None,
                    ),
                }
            )

            sampling.reset_trace()
            sampling.reset_sampling_params(format_sampling_params(GREEDY, batch_size))
            sampling.seed_manager.reset_seed(None, list(range(batch_size)))
            sampled = None
            for step, expected_token in enumerate(expected):
                ttnn.copy_host_to_device_tensor(host_a if expected_token == 5310 else host_b, tt_logits)
                if refresh_params:
                    sampling.apply_decode_state([GREEDY], reset_batch=(step == 0))
                sampling.seed_manager.get_new_values()
                sampled = sampling.sample(tt_logits, tt_out_tok=feedback, enable_trace=True)
            ttnn.synchronize_device(mesh_device)
            queued_token = read_first_token(sampled)
            cases.append(
                {
                    "feedback": use_feedback,
                    "refresh_params_each_step": refresh_params,
                    "collection": "final_only",
                    "expected_final": expected[-1],
                    "observed_final": queued_token,
                    "first_divergence": None if queued_token == expected[-1] else len(expected) - 1,
                }
            )

    artifact = {
        "mesh_shape": list(mesh_device.shape),
        "steps_per_case": len(expected),
        "cases": cases,
    }
    artifact_path = Path(
        "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/" "tp4_sampling_trace_isolation.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    assert all(case["first_divergence"] is None for case in cases)
    sampling.reset_trace()


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

    output_tokens = int(os.environ.get("GPT_OSS_120B_PROBE_OUTPUT_TOKENS", "4"))
    isolation_lengths = [
        int(value) for value in os.environ.get("GPT_OSS_120B_ASYNC_ISOLATION_LENGTHS", "").split(",") if value
    ]
    prompt_extra_tokens = int(os.environ.get("GPT_OSS_120B_PROBE_PROMPT_EXTRA_TOKENS", "0"))
    probe_prompt_len = 7 + prompt_extra_tokens
    assert output_tokens >= 2
    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        max_batch_size=1,
        max_context_length=max(
            128,
            probe_prompt_len + output_tokens,
            *(probe_prompt_len + value for value in isolation_lengths),
        ),
        num_layers=2,
        allow_reduced_model=True,
        lm_head_policy=os.environ.get("GPT_OSS_120B_LM_HEAD_POLICY", INTERLEAVED_LM_HEAD),
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    profile_drain = os.environ.get("GPT_OSS_120B_PROFILE_DRAIN") == "1"
    if profile_drain:
        ttnn.ReadDeviceProfiler(mesh_device)
    prompt = torch.tensor(
        [[200006, 1734, 25, 392, 876, 13, *([13] * prompt_extra_tokens), 200007]],
        dtype=torch.long,
    )
    pages = generator.page_table[:1]

    host_logits = generator.prefill_forward(
        prompt,
        page_table=pages,
        kv_cache=kv_cache,
        prompt_lens=[prompt.shape[1]],
    )
    expected_first = int(torch.argmax(host_logits[0, 0]).item())
    if profile_drain:
        ttnn.ReadDeviceProfiler(mesh_device)

    # The selected split sampler must retain the generic traced top-k/top-p
    # topology used by serving requests, not merely its fast greedy settings.
    # Explicit request seeds intentionally bypass sampler trace replay, so this
    # contract probe is unseeded and counts the actual sampling trace executor.
    sampled_params = SamplingParams(temperature=0.8, top_k=20, top_p=0.9)
    sampling = model.sampling
    sampling_trace_replays = 0
    original_execute_trace = sampling._execute_trace

    def counted_execute_trace(key):
        nonlocal sampling_trace_replays
        sampling_trace_replays += 1
        return original_execute_trace(key)

    sampling._execute_trace = counted_execute_trace
    signpost("FULL_MODEL_TOP_K_TOP_P")
    top_k_top_p_metrics = generator.run_device_token_out(
        prompt[0].tolist(),
        5,
        enable_trace=True,
        sampling_params=sampled_params,
    )
    signpost("FULL_MODEL_TOP_K_TOP_P_END")
    sampling._execute_trace = original_execute_trace
    active_sampling_traces = [(key, slot) for key, slot in sampling._trace_states.items() if slot.get("id") is not None]
    assert len(active_sampling_traces) == 1
    sampling_key, sampling_slot = active_sampling_traces[0]
    trace_inputs = generator._inner.trace_inputs_decode[True][0]
    feedback_buffer = generator._inner._decode_token_feedback_buffer(model, trace_inputs)
    tt_out_tok_identity = isinstance(sampling_slot["output"], tuple) and sampling_slot["output"][0] is feedback_buffer
    assert tt_out_tok_identity
    assert not sampling_key.force_argmax
    assert not sampling.seed_manager.has_active_request_seed()
    # Prefill sampling captured the compatible terminal-shape trace, so all
    # four decode submissions replay it (including the first model decode).
    assert sampling_trace_replays == 4
    top_k_top_p_evidence = generator.trace_evidence
    assert top_k_top_p_evidence.trace_replays == 4
    assert top_k_top_p_evidence.device_token_out_submissions == 4
    assert top_k_top_p_evidence.sampling_state_host_refreshes == 1
    assert top_k_top_p_evidence.fixed_sampling_state_replays == 3
    assert top_k_top_p_evidence.fixed_greedy_sampling_replays == 0
    assert top_k_top_p_evidence.decode_output_collections == 1
    assert top_k_top_p_evidence.sampled_token_readbacks == 2
    assert top_k_top_p_evidence.caller_visible_token_synchronizations == 2
    assert top_k_top_p_evidence.full_input_refreshes == 1
    assert top_k_top_p_evidence.page_table_reuses == 3
    assert top_k_top_p_evidence.steady_token_input_host_refreshes == 0
    assert top_k_top_p_evidence.steady_position_rope_host_refreshes == 0
    assert top_k_top_p_evidence.steady_page_table_host_refreshes == 0
    assert top_k_top_p_evidence.full_logits_readbacks == 0
    assert 0 <= top_k_top_p_metrics["first_token"] < model.vocab_size
    assert 0 <= top_k_top_p_metrics["final_token"] < model.vocab_size

    def first_shard_to_torch(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])

    persistent_token = first_shard_to_torch(trace_inputs[0]).reshape(-1)
    persistent_position = first_shard_to_torch(trace_inputs[1]).reshape(-1)
    persistent_rope_position = first_shard_to_torch(trace_inputs[2]).reshape(-1)
    assert int(persistent_token[0]) == top_k_top_p_metrics["final_token"]
    assert int(persistent_position[0]) == prompt.shape[1] + 4
    assert int(persistent_rope_position[0]) == prompt.shape[1] + 4
    top_k_top_p_artifact = Path(
        "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/" "top_k_top_p_trace_contract.json"
    )
    top_k_top_p_artifact.parent.mkdir(parents=True, exist_ok=True)
    top_k_top_p_artifact.write_text(
        json.dumps(
            {
                "mesh_shape": list(mesh_device.shape),
                "layers": 2,
                "sampling_params": {"temperature": 0.8, "top_k": 20, "top_p": 0.9, "seed": None},
                "sampling_trace_id": str(sampling_slot["id"]),
                "sampling_trace_replays": sampling_trace_replays,
                "sampling_force_argmax": sampling_key.force_argmax,
                "tt_out_tok_feedback_identity": tt_out_tok_identity,
                "metrics": top_k_top_p_metrics,
                "persistent_token": int(persistent_token[0]),
                "persistent_position": int(persistent_position[0]),
                "persistent_rope_position": int(persistent_rope_position[0]),
                "trace_evidence": top_k_top_p_evidence.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if profile_drain:
        ttnn.ReadDeviceProfiler(mesh_device)

    signpost("FULL_MODEL_TOKEN_OUT")
    predictions = generator.generate(
        prompt[0].tolist(),
        output_tokens,
        enable_trace=True,
        sampling_mode="device",
    )
    signpost("FULL_MODEL_TOKEN_OUT_END")
    if profile_drain:
        ttnn.ReadDeviceProfiler(mesh_device)
    assert predictions[0] == expected_first
    assert all(0 <= token < model.vocab_size for token in predictions)
    evidence = generator.trace_evidence
    assert evidence.trace_replays == output_tokens - 1
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == output_tokens - 2
    assert evidence.page_table_only_refreshes == 0
    assert evidence.host_argmax_calls == 0
    assert evidence.full_logits_readbacks == 0
    assert evidence.forced_token_refreshes == 0
    trace_inputs = generator._inner.trace_inputs_decode[True][0]

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
    assert evidence.trace_replays == output_tokens
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == output_tokens - 2
    assert evidence.page_table_only_refreshes == 1

    signpost("FULL_MODEL_ASYNC_TOKEN_OUT")
    token_out_metrics = generator.run_greedy_token_out(
        prompt[0].tolist(),
        output_tokens,
        enable_trace=True,
    )
    signpost("FULL_MODEL_ASYNC_TOKEN_OUT_END")
    if profile_drain:
        ttnn.ReadDeviceProfiler(mesh_device)
    evidence = generator.trace_evidence
    assert token_out_metrics["first_token"] == predictions[0]
    assert token_out_metrics["final_token"] == predictions[-1]
    trace_inputs = generator._inner.trace_inputs_decode[True][0]
    persistent_token = first_shard_to_torch(trace_inputs[0]).reshape(-1)
    persistent_position = first_shard_to_torch(trace_inputs[1]).reshape(-1)
    assert int(persistent_token[0]) == token_out_metrics["final_token"]
    assert int(persistent_position[0]) == prompt.shape[1] + output_tokens - 1
    assert evidence.device_token_out_submissions == output_tokens - 1
    assert evidence.sampling_state_host_refreshes == 1
    assert evidence.fixed_sampling_state_replays == output_tokens - 2
    assert evidence.fixed_greedy_sampling_replays == output_tokens - 2
    assert evidence.decode_output_collections == 1
    assert evidence.sampled_token_readbacks == 2  # one TTFT token plus one final decode token
    assert evidence.caller_visible_token_synchronizations == 2
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == output_tokens - 2
    assert evidence.steady_token_input_host_refreshes == 0
    assert evidence.steady_position_rope_host_refreshes == 0
    assert evidence.steady_page_table_host_refreshes == 0

    if isolation_lengths:
        isolation = []
        for length in isolation_lengths:
            sync_predictions = generator.generate(
                prompt[0].tolist(),
                length,
                enable_trace=True,
                sampling_mode="device",
            )
            sync_predictions_repeat = generator.generate(
                prompt[0].tolist(),
                length,
                enable_trace=True,
                sampling_mode="device",
            )
            split_metrics = generator.run_greedy_token_out(
                prompt[0].tolist(),
                length,
                enable_trace=True,
            )
            isolation_trace_inputs = generator._inner.trace_inputs_decode[True][0]
            isolation_token = first_shard_to_torch(isolation_trace_inputs[0]).reshape(-1)
            isolation_position = first_shard_to_torch(isolation_trace_inputs[1]).reshape(-1)
            isolation_rope_position = first_shard_to_torch(isolation_trace_inputs[2]).reshape(-1)
            isolation.append(
                {
                    "output_tokens": length,
                    "logical_context_length": args.max_context_len,
                    "physical_kv_context_length": args.physical_kv_context_len,
                    "decode_k_chunk_size": args.decode_k_chunk_size,
                    "sync_final_token": sync_predictions[-1],
                    "sync_repeat_final_token": sync_predictions_repeat[-1],
                    "sync_runs_exact": sync_predictions == sync_predictions_repeat,
                    "sync_first_divergence": next(
                        (
                            index
                            for index, (left, right) in enumerate(zip(sync_predictions, sync_predictions_repeat))
                            if left != right
                        ),
                        None,
                    ),
                    "sync_predictions": sync_predictions,
                    "sync_repeat_predictions": sync_predictions_repeat,
                    "split_final_token": split_metrics["final_token"],
                    "persistent_token": int(isolation_token[0]),
                    "persistent_position": int(isolation_position[0]),
                    "persistent_rope_position": int(isolation_rope_position[0]),
                    "exact": sync_predictions == sync_predictions_repeat
                    and sync_predictions[-1] == split_metrics["final_token"],
                }
            )
        artifact_path = Path(
            "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
            "split_token_out_length_isolation.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(isolation, indent=2) + "\n", encoding="utf-8")
        assert all(row["exact"] for row in isolation)
    generator.teardown()


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_FULL_MODEL_PROFILE") != "1" or not SNAPSHOT.is_dir(),
    reason="set GPT_OSS_120B_FULL_MODEL_PROFILE=1 for the isolated full-path profiler probe",
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
def test_real_weight_two_layer_optimized_full_model_profile(mesh_device, device_params, reset_seeds):
    """Capture one isolated warmed operation window for each full-path phase."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    phase = os.environ.get("GPT_OSS_120B_FULL_MODEL_PROFILE_PHASE")
    supported_phases = {"prefill", "teacher_forcing_decode", "split_token_out_decode"}
    assert phase in supported_phases, f"select one profile phase from {sorted(supported_phases)}"
    profile_drain = os.environ.get("GPT_OSS_120B_PROFILE_DRAIN") == "1"

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_probe_cache",
        max_batch_size=1,
        max_context_length=256,
        num_layers=2,
        allow_reduced_model=True,
        lm_head_policy=os.environ.get("GPT_OSS_120B_LM_HEAD_POLICY", INTERLEAVED_LM_HEAD),
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    reference = torch.load(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt",
        map_location="cpu",
        weights_only=False,
    )["entries"][0]
    prompt = reference["prompt_tokens"][0, :128].to(torch.long).unsqueeze(0)
    teacher_tokens = reference["generated_tokens"][0, :4].to(torch.long)
    pages = generator.page_table[:1]

    def drain_profiler():
        ttnn.synchronize_device(mesh_device)
        if profile_drain:
            ttnn.ReadDeviceProfiler(mesh_device)

    evidence = {
        "phase": phase,
        "mesh_shape": list(mesh_device.shape),
        "layers": 2,
        "prompt_length": 128,
        "lm_head_policy": model.lm_head_policy,
        "lm_head_candidate_manifest": (
            model.dram_sharded_lm_head.manifest if model.dram_sharded_lm_head is not None else None
        ),
    }
    if phase == "prefill":
        generator._device_prefill_sample(prompt, pages, sampling_params=GREEDY)
        generator.reset()
        drain_profiler()
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_PREFILL")
        predicted = generator._device_prefill_sample(prompt, pages, sampling_params=GREEDY)
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_PREFILL_END")
        drain_profiler()
        assert 0 <= predicted < model.vocab_size
        evidence["predicted_token"] = predicted
    elif phase == "teacher_forcing_decode":
        generator._device_prefill_sample(prompt, pages, sampling_params=GREEDY)
        for step in range(2):
            prediction = generator.decode_forward(
                teacher_tokens[step].reshape(1, 1),
                torch.tensor([prompt.shape[1] + step], dtype=torch.int64),
                page_table=pages,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_mode="device",
                sampling_params=GREEDY,
                reset_batch=True,
                force_host_tokens=True,
                prompt_tokens=prompt if step == 0 else None,
                read_from_device=True,
            )
        drain_profiler()
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_TEACHER_FORCING_DECODE")
        prediction = generator.decode_forward(
            teacher_tokens[2].reshape(1, 1),
            torch.tensor([prompt.shape[1] + 2], dtype=torch.int64),
            page_table=pages,
            kv_cache=kv_cache,
            enable_trace=True,
            sampling_mode="device",
            sampling_params=GREEDY,
            reset_batch=True,
            force_host_tokens=True,
            read_from_device=True,
        )
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_TEACHER_FORCING_DECODE_END")
        drain_profiler()
        evidence["predicted_token"] = int(prediction[0])
    else:
        first_token = generator._device_prefill_sample(prompt, pages, sampling_params=GREEDY)
        device_output = generator.decode_forward(
            torch.tensor([[first_token]], dtype=torch.long),
            torch.tensor([prompt.shape[1]], dtype=torch.int64),
            page_table=pages,
            kv_cache=kv_cache,
            enable_trace=True,
            sampling_mode="device",
            sampling_params=GREEDY,
            reset_batch=True,
            prompt_tokens=prompt,
            read_from_device=False,
        )
        device_output = generator.decode_forward(
            torch.tensor([[first_token]], dtype=torch.long),
            torch.tensor([prompt.shape[1] + 1], dtype=torch.int64),
            page_table=pages,
            kv_cache=kv_cache,
            enable_trace=True,
            sampling_mode="device",
            reuse_greedy_sampling_state=True,
            read_from_device=False,
        )
        generator.read_decode_output(device_output)
        drain_profiler()
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_SPLIT_TOKEN_OUT_DECODE")
        device_output = generator.decode_forward(
            torch.tensor([[first_token]], dtype=torch.long),
            torch.tensor([prompt.shape[1] + 2], dtype=torch.int64),
            page_table=pages,
            kv_cache=kv_cache,
            enable_trace=True,
            sampling_mode="device",
            reuse_greedy_sampling_state=True,
            read_from_device=False,
        )
        signpost("OPTIMIZED_FULL_MODEL_PROFILE_SPLIT_TOKEN_OUT_DECODE_END")
        drain_profiler()
        final_token = int(generator.read_decode_output(device_output)[0])
        evidence.update(
            {
                "first_token": first_token,
                "final_token": final_token,
                "trace_evidence": generator.trace_evidence.to_dict(),
            }
        )
        assert evidence["trace_evidence"]["steady_token_input_host_refreshes"] == 0
        assert evidence["trace_evidence"]["steady_position_rope_host_refreshes"] == 0
        assert evidence["trace_evidence"]["steady_page_table_host_refreshes"] == 0

    artifact = Path(
        "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/profiler/"
        f"final_source/{model.lm_head_policy}/{phase}/phase_evidence.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    if model.lm_head_policy == DRAM_SHARDED_LM_HEAD:
        assert model.dram_sharded_lm_head is not None
        assert model.dram_sharded_lm_head.manifest["num_splits"] == 8
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
    artifact = Path(
        "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
        "mixed_prompt_fixed_slots_inactive_row.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "mesh_shape": list(mesh_device.shape),
                "layers": 2,
                "batch_size": 2,
                "prompt_lengths": [7, 5],
                "prompt_rows_distinct": not torch.equal(prompts[0], prompts[1]),
                "page_table_rows_distinct": not torch.equal(pages[0], pages[1]),
                "decode_start_positions": [7, -1],
                "persistent_positions_after_decode": [
                    int(persistent_positions[0]),
                    int(persistent_positions[1]),
                ],
                "inactive_row_preserved": int(persistent_positions[1]) == -1,
                "active_token": int(decoded[0]),
                "trace_evidence": generator.trace_evidence.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
            "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
            "batch2_logit_reproducibility_probe.json"
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
    """Run non-aligned correctness and warmed prompt-128/token-128 performance."""

    del device_params, reset_seeds
    from models.autoports.openai_gpt_oss_120b.tt.model import build_model

    isolation_lengths = [
        int(value) for value in os.environ.get("GPT_OSS_120B_FULL_ASYNC_ISOLATION_LENGTHS", "").split(",") if value
    ]

    model, args, kv_cache = build_model(
        mesh_device,
        snapshot_path=SNAPSHOT,
        tensor_cache_path="/tmp/gpt_oss_120b_full_model_tensor_cache",
        max_batch_size=1,
        max_context_length=HF_CONTEXT_LENGTH,
        num_layers=MODEL_LAYERS,
    )
    generator = Generator(model, args, kv_cache=kv_cache)
    reference = torch.load(
        "models/autoports/openai_gpt_oss_120b/doc/full_model/references/aime24_chat_100_top100.refpt",
        map_location="cpu",
        weights_only=False,
    )["entries"][0]
    benchmark_prompt = reference["prompt_tokens"][0, :128].to(torch.long).tolist()
    assert len(benchmark_prompt) == 128

    lifecycle_pre_release = None
    if os.environ.get("GPT_OSS_120B_FULL_TRACE_LIFECYCLE_ISOLATION") == "1":
        generator.generate(benchmark_prompt, 4, enable_trace=True, sampling_mode="device")
        before_release_a = generator.generate(benchmark_prompt, 16, enable_trace=True, sampling_mode="device")
        before_release_b = generator.generate(benchmark_prompt, 16, enable_trace=True, sampling_mode="device")
        lifecycle_pre_release = {
            "exact": before_release_a == before_release_b,
            "first_divergence": next(
                (index for index, pair in enumerate(zip(before_release_a, before_release_b)) if pair[0] != pair[1]),
                None,
            ),
            "first": before_release_a,
            "second": before_release_b,
            "decode_trace_release_count": generator._lifetime_decode_trace_releases_for_prefill_compile,
        }

    prompts_path = Path("models/demos/deepseek_v3/demo/aime_under_8k_prompts.json")
    prompt_text = json.loads(prompts_path.read_text(encoding="utf-8"))[0]["prompt"]
    prompt = args.encode_prompt(prompt_text)
    assert len(prompt) % 32 != 0, "acceptance prompt must exercise public non-aligned prefill"
    predictions = generator.generate(prompt, 4, enable_trace=True, sampling_mode="device")
    non_aligned_token_out = generator.run_greedy_token_out(prompt, 4, enable_trace=True)
    decoded = args.tokenizer.decode(predictions, skip_special_tokens=False)
    assert len(predictions) == 4
    assert len(set(predictions)) > 1
    assert all(0 <= token < model.vocab_size for token in predictions)
    assert non_aligned_token_out["first_token"] == predictions[0]
    assert non_aligned_token_out["final_token"] == predictions[-1]

    # Preserve an apples-to-apples optimized measurement for the completed
    # full-model stage's prompt-214/generation-100 baseline.  The four-token
    # request above has already compiled and warmed this non-aligned bucket.
    signpost("OPTIMIZED_FULL_MODEL_ASYNC_PROMPT214_GEN100")
    non_aligned_warmed_token_out = generator.run_greedy_token_out(
        prompt,
        100,
        enable_trace=True,
    )
    signpost("OPTIMIZED_FULL_MODEL_ASYNC_PROMPT214_GEN100_END")
    assert non_aligned_warmed_token_out["prompt_tokens"] == len(prompt)
    assert non_aligned_warmed_token_out["output_tokens"] == 100
    assert non_aligned_warmed_token_out["first_token"] == predictions[0]

    # The first request for a new padded prefill variant exercises the safe
    # request-boundary trace release and compilation path.  Performance starts
    # only after that bucket and the recaptured decode/sampling traces are warm.
    warmup_predictions = generator.generate(
        benchmark_prompt,
        4,
        enable_trace=True,
        sampling_mode="device",
    )
    assert len(warmup_predictions) == 4
    assert generator._lifetime_prefill_variant_compilations == 2
    assert generator._lifetime_decode_trace_releases_for_prefill_compile == 1

    signpost("OPTIMIZED_FULL_MODEL_SYNC_PROMPT128_GEN128")
    baseline_predictions = generator.generate(
        benchmark_prompt,
        128,
        enable_trace=True,
        sampling_mode="device",
    )
    signpost("OPTIMIZED_FULL_MODEL_SYNC_PROMPT128_GEN128_END")
    baseline_metrics = dict(generator.last_generation_metrics)
    baseline_evidence = generator.trace_evidence.to_dict()

    if isolation_lengths:
        assert all(2 <= length <= len(baseline_predictions) for length in isolation_lengths)
        sync_repeat = generator.generate(
            benchmark_prompt,
            len(baseline_predictions),
            enable_trace=True,
            sampling_mode="device",
        )
        sync_first_divergence = next(
            (index for index, (left, right) in enumerate(zip(baseline_predictions, sync_repeat)) if left != right),
            None,
        )
        divergence_logits = None
        if sync_first_divergence is not None:
            pages = generator.page_table[:1]
            padded_per_device = model.sampling.tt_sampling.padded_vocab_size // mesh_device.get_num_devices()

            def capture_true_bucket_at_divergence():
                generator.reset()
                generator._device_prefill_sample(
                    torch.tensor([benchmark_prompt], dtype=torch.long),
                    pages,
                    sampling_params=GREEDY,
                )
                for output_index in range(1, sync_first_divergence):
                    generator.decode_forward(
                        torch.tensor([[baseline_predictions[output_index - 1]]], dtype=torch.long),
                        torch.tensor([len(benchmark_prompt) + output_index - 1], dtype=torch.int64),
                        page_table=pages,
                        kv_cache=generator.kv_cache,
                        enable_trace=True,
                        sampling_mode="device",
                        sampling_params=GREEDY,
                        reset_batch=True,
                        force_host_tokens=True,
                        read_from_device=False,
                    )

                # Replay only the real on-device-sampling model bucket.  The
                # host-sampling bucket has a different output/padding contract
                # and is not an oracle for the tensor bound to the sampler trace.
                generator._inner._slots_prefilled_since_decode.add(0)
                tt_logits = generator._inner.decode_forward(
                    tokens=torch.tensor([[baseline_predictions[sync_first_divergence - 1]]], dtype=torch.long),
                    start_pos=torch.tensor([len(benchmark_prompt) + sync_first_divergence - 1], dtype=torch.int64),
                    page_table=pages,
                    kv_cache=generator._outer_cache(generator.kv_cache),
                    enable_trace=True,
                    read_from_device=False,
                    sampling_params=None,
                    reset_batch=True,
                    defer_device_sampling=True,
                )
                ttnn.synchronize_device(mesh_device)
                logits = tt_logits[0][0] if isinstance(tt_logits[0], tuple) else tt_logits[0]
                device_rows = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(logits)]
                row = torch.cat(device_rows, dim=-1)[0, 0, 0, : model.vocab_size].float()
                global_max = row.max()
                shard_ties = []
                for device_index in range(mesh_device.get_num_devices()):
                    start = device_index * padded_per_device
                    end = min(start + padded_per_device, model.vocab_size)
                    shard = row[start:end]
                    shard_max = shard.max()
                    shard_ties.append(
                        {
                            "device": device_index,
                            "valid_start": start,
                            "valid_end": end,
                            "max": float(shard_max),
                            "exact_local_max_count": int((shard == shard_max).sum()),
                            "exact_global_max_count": int((shard == global_max).sum()),
                            "lowest_local_max_token": start + int(torch.argmax(shard)),
                        }
                    )

                # The synchronization above is an isolation discriminator: if
                # this sample is correct while the natural adjacent traces are
                # not, the bug is a model-trace -> sampler-trace dependency.
                sampled = generator._replay_prepared_greedy_sampling(tt_logits, enable_trace=True)
                sampled_token = int(generator.read_decode_output(sampled)[0])
                return {
                    "host_global_argmax": int(torch.argmax(row)),
                    "global_max": float(global_max),
                    "global_exact_max_count": int((row == global_max).sum()),
                    "sample_after_model_sync": sampled_token,
                    "shards": shard_ties,
                }

            true_bucket_captures = [capture_true_bucket_at_divergence() for _ in range(2)]
            divergence_logits = {
                "oracle_bucket": "trace_output_decode[True][0]",
                "output_index": sync_first_divergence,
                "baseline_sample": baseline_predictions[sync_first_divergence],
                "repeat_sample": sync_repeat[sync_first_divergence],
                "padded_per_device": padded_per_device,
                "captures": true_bucket_captures,
            }
        isolation = []
        for length in isolation_lengths:
            split_metrics = generator.run_greedy_token_out(
                benchmark_prompt,
                length,
                enable_trace=True,
            )
            trace_inputs = generator._inner.trace_inputs_decode[True][0]
            persistent_token = ttnn.to_torch(ttnn.get_device_tensors(trace_inputs[0])[0]).reshape(-1)
            persistent_position = ttnn.to_torch(ttnn.get_device_tensors(trace_inputs[1])[0]).reshape(-1)
            persistent_rope_position = ttnn.to_torch(ttnn.get_device_tensors(trace_inputs[2])[0]).reshape(-1)
            isolation.append(
                {
                    "output_tokens": length,
                    "sync_token": baseline_predictions[length - 1],
                    "sync_repeat_token": sync_repeat[length - 1],
                    "split_token": split_metrics["final_token"],
                    "sync_runs_exact_through_length": (baseline_predictions[:length] == sync_repeat[:length]),
                    "split_endpoint_exact": split_metrics["final_token"] == baseline_predictions[length - 1],
                    "persistent_token": int(persistent_token[0]),
                    "persistent_position": int(persistent_position[0]),
                    "persistent_rope_position": int(persistent_rope_position[0]),
                }
            )
        isolation_artifact = Path(
            "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
            "p150x4_full_stack_split_isolation.json"
        )
        isolation_artifact.parent.mkdir(parents=True, exist_ok=True)
        isolation_artifact.write_text(
            json.dumps(
                {
                    "sync_runs_exact": baseline_predictions == sync_repeat,
                    "sync_first_divergence": sync_first_divergence,
                    "sync_predictions": baseline_predictions,
                    "sync_repeat_predictions": sync_repeat,
                    "before_unseen_prefill_trace_release": lifecycle_pre_release,
                    "decode_trace_release_count": generator._lifetime_decode_trace_releases_for_prefill_compile,
                    "first_divergence_sampler_ready_logits": divergence_logits,
                    "rows": isolation,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        assert sync_first_divergence is None
        assert all(row["split_endpoint_exact"] for row in isolation)

    signpost("OPTIMIZED_FULL_MODEL_ASYNC_PROMPT128_GEN128")
    optimized_metrics = generator.run_greedy_token_out(
        benchmark_prompt,
        128,
        enable_trace=True,
    )
    signpost("OPTIMIZED_FULL_MODEL_ASYNC_PROMPT128_GEN128_END")
    evidence = generator.trace_evidence
    assert optimized_metrics["first_token"] == baseline_predictions[0]
    assert optimized_metrics["final_token"] == baseline_predictions[-1]
    assert evidence.trace_replays == 127
    assert evidence.full_input_refreshes == 1
    assert evidence.page_table_reuses == 126
    assert evidence.token_input_host_refreshes == 1
    assert evidence.position_rope_host_refreshes == 1
    assert evidence.page_table_host_refreshes == 1
    assert evidence.steady_token_input_host_refreshes == 0
    assert evidence.steady_position_rope_host_refreshes == 0
    assert evidence.steady_page_table_host_refreshes == 0
    assert evidence.device_token_out_submissions == 127
    assert evidence.sampling_state_host_refreshes == 1
    assert evidence.fixed_greedy_sampling_replays == 126
    assert evidence.decode_output_collections == 1
    assert evidence.sampled_token_readbacks == 2
    assert evidence.caller_visible_token_synchronizations == 2
    assert evidence.validation_full_logit_synchronizations == 0
    assert evidence.full_logits_readbacks == 0
    assert evidence.host_argmax_calls == 0

    lower_bound = json.loads(
        Path("models/autoports/openai_gpt_oss_120b/doc/full_model/artifacts/profiler/profiler_analysis.json").read_text(
            encoding="utf-8"
        )
    )["layer_stack_lower_bound"]
    baseline_decode_ms = 1000.0 / baseline_metrics["decode_tokens_per_second_per_user"]
    optimized_decode_ms = 1000.0 / optimized_metrics["decode_tokens_per_second_per_user"]
    stack_ms = float(lower_bound["decoder_stack_ms"])
    terminal_ms = float(lower_bound["named_terminal_device_ms"]["sum"])
    stack_plus_terminal_ms = stack_ms + terminal_ms
    avoidable_gap_ms = optimized_decode_ms - stack_plus_terminal_ms
    overhead_over_stack_plus_terminal_percent = 100.0 * avoidable_gap_ms / stack_plus_terminal_ms
    assert overhead_over_stack_plus_terminal_percent <= 15.0
    artifact = {
        "checkpoint_revision": SNAPSHOT.name,
        "mesh_shape": list(mesh_device.shape),
        "num_layers": MODEL_LAYERS,
        "configured_context": HF_CONTEXT_LENGTH,
        "prompt_length": len(prompt),
        "predicted_tokens": predictions,
        "decoded": decoded,
        "non_aligned_token_out": non_aligned_token_out,
        "warmed_non_aligned_prompt214_gen100": non_aligned_warmed_token_out,
        "capacity": model.capacity.to_dict(),
        "precision_runtime_evidence": model.precision_runtime_evidence(),
        "warmed_prompt128_gen128": {
            "prompt_length": len(benchmark_prompt),
            "output_tokens": len(baseline_predictions),
            "sync_token_readback_baseline": baseline_metrics,
            "split_token_out": optimized_metrics,
            "first_token_exact_match": optimized_metrics["first_token"] == baseline_predictions[0],
            "final_token_exact_match": optimized_metrics["final_token"] == baseline_predictions[-1],
            "sync_decode_ms_per_token": baseline_decode_ms,
            "split_token_out_decode_ms_per_token": optimized_decode_ms,
            "speedup": baseline_decode_ms / optimized_decode_ms,
            "sync_trace_evidence": baseline_evidence,
            "split_token_out_trace_evidence": evidence.to_dict(),
            "prefill_variant_lifecycle": {
                "compiled": [
                    {
                        "padded_length": padded_length,
                        "page_rounded_length": page_rounded_length,
                        "last_token_tile_start": last_token_tile_start,
                        "path": path,
                    }
                    for padded_length, page_rounded_length, last_token_tile_start, path in sorted(
                        generator._compiled_prefill_variants
                    )
                ],
                "compile_count": generator._lifetime_prefill_variant_compilations,
                "decode_trace_release_count": generator._lifetime_decode_trace_releases_for_prefill_compile,
            },
            "decoder_layer_stack_lower_bound": lower_bound,
            "optimized_full_path_closure": {
                "decoder_stack_ms": stack_ms,
                "named_terminal_device_ms": terminal_ms,
                "stack_plus_terminal_ms": stack_plus_terminal_ms,
                "measured_split_token_out_ms": optimized_decode_ms,
                "avoidable_gap_ms": avoidable_gap_ms,
                "overhead_over_stack_plus_terminal_percent": overhead_over_stack_plus_terminal_percent,
                "gate_percent": 15.0,
                "gate_pass": overhead_over_stack_plus_terminal_percent <= 15.0,
            },
        },
    }
    artifact_path = (
        Path("models/autoports/openai_gpt_oss_120b/doc/datatype_sweep/artifacts/selected")
        / "post_selection_token_out.json"
        if os.environ.get("GPT_OSS_120B_DATATYPE_SWEEP_SELECTED") == "1"
        else Path(
            "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
            "p150x4_prompt128_gen128_perf.json"
        )
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
        artifact_path=Path(
            "models/autoports/openai_gpt_oss_120b/doc/optimized_full_model/artifacts/"
            "batch2_logit_reproducibility.json"
        ),
    )


def _readiness_runner(name: str):
    """Load the pipeline readiness runner without shadowing this checkout's models package."""

    import importlib

    import models.common
    import models.common.readiness_check

    sibling_common = "/home/ttuser/dev/scratch/tt-metal/models/common"
    if sibling_common not in models.common.__path__:
        models.common.__path__.append(sibling_common)
    sibling_readiness = f"{sibling_common}/readiness_check"
    if sibling_readiness not in models.common.readiness_check.__path__:
        models.common.readiness_check.__path__.append(sibling_readiness)
    return importlib.import_module(f"models.common.readiness_check.{name}")


def _datatype_sweep_readiness_kwargs(model_dir: Path):
    config_path = os.environ.get("GPT_OSS_120B_DATATYPE_SWEEP_CONFIG")
    if not config_path:
        if os.environ.get("GPT_OSS_120B_DATATYPE_SWEEP_DEFAULT_REFRESH") == "1":
            artifact_dir = model_dir / "doc/datatype_sweep/artifacts/default_selected_refresh"
            return {"runtime_evidence_path": artifact_dir / "runtime_precision_evidence.json"}, artifact_dir
        return {
            "runtime_evidence_path": model_dir / "doc/optimized_full_model/artifacts/runtime_precision_evidence.json"
        }, None
    config_path = Path(config_path).expanduser().resolve()
    config_id = json.loads(config_path.read_text(encoding="utf-8"))["config_id"]
    artifact_dir = model_dir / "doc/datatype_sweep/artifacts" / config_id
    return {
        "precision_config": config_path,
        "runtime_evidence_path": artifact_dir / "runtime_precision_evidence.json",
    }, artifact_dir


def _validated_trace_measurement(runtime_evidence: dict) -> dict:
    assert set(runtime_evidence) == {
        "schema_version",
        "config_id",
        "config_path",
        "config_sha256",
        "default_selected_config_path",
        "terminal",
        "layer_runtime_groups",
        "validation",
        "measurement",
    }
    assert runtime_evidence["schema_version"] == 2
    assert set(runtime_evidence["terminal"]) == {
        "embedding_weight",
        "normalization_weight",
        "lm_head_weight",
        "lm_head_output",
        "lm_head_math_fidelity",
        "sampling_accumulator",
        "sampling_device_buffers",
        "full_logits_gather",
        "topk_values_gather_dtype",
    }
    groups = runtime_evidence["layer_runtime_groups"]
    assert sorted(layer for group in groups for layer in group["layers"]) == list(range(MODEL_LAYERS))
    assert all(
        "input_normalization_weight" in group and "post_attention_normalization_weight" in group for group in groups
    )
    measurement = runtime_evidence["measurement"]
    counters = measurement["trace_counters"]
    before = measurement["trace_handles_before_timing"]
    after = measurement["trace_handles"]
    expected = measurement["expected_decode_calls"]
    trace_measurement = {
        "requested": measurement["generation_metrics"]["enable_trace"],
        "warmed_before_timing": measurement["warmed_before_timing"],
        "expected_decode_steps": expected,
        "model_trace_handles_before_timing": before["model_decode_trace_count"],
        "sampling_trace_handles_before_timing": before["sampling_trace_count"],
        "model_trace_handles_after_timing": after["model_decode_trace_count"],
        "sampling_trace_handles_after_timing": after["sampling_trace_count"],
        "model_execute_submissions": counters["model_execute_submissions"],
        "sampling_execute_submissions": counters["sampling_execute_submissions"],
        "unclassified_execute_submissions": counters["unclassified_execute_submissions"],
        "trace_verified": measurement["trace_verified"],
    }
    assert trace_measurement == {
        "requested": True,
        "warmed_before_timing": True,
        "expected_decode_steps": 99,
        "model_trace_handles_before_timing": 1,
        "sampling_trace_handles_before_timing": 1,
        "model_trace_handles_after_timing": 1,
        "sampling_trace_handles_after_timing": 1,
        "model_execute_submissions": 99,
        "sampling_execute_submissions": 99,
        "unclassified_execute_submissions": 0,
        "trace_verified": True,
    }
    return trace_measurement


def _run_warmed_teacher_forcing(
    *, runner, model_dir: Path, reference: Path, mesh_device, build_kwargs: dict, repetitions: int
) -> list[dict]:
    build_generator_fn = runner._import_build_generator(model_dir)
    generator = build_generator_fn(model_dir=model_dir, mesh_device=mesh_device, **build_kwargs)
    runtime_path = Path(build_kwargs["runtime_evidence_path"])
    runner_provenance = _runner_provenance(runner)
    source_provenance = _runtime_source_provenance()
    try:
        warm_acc = runner.TokenAccuracy(reference)
        for entry_idx in range(warm_acc.num_entries):
            if entry_idx > 0:
                generator.reset()
            runner._run_one_entry(generator=generator, acc=warm_acc, entry_idx=entry_idx)

        repeated_rows = []
        for repetition in range(repetitions):
            acc = runner.TokenAccuracy(reference)
            generator.reset()
            generator.mark_warmed_measurement()
            if acc.num_entries != 1:
                raise RuntimeError("datatype-sweep readiness reference must contain exactly one entry")
            stats = runner._run_one_entry(generator=generator, acc=acc, entry_idx=0)
            runtime_evidence = json.loads(runtime_path.read_text(encoding="utf-8"))
            stats["trace_measurement"] = _validated_trace_measurement(runtime_evidence)
            stats["runner_provenance"] = runner_provenance
            stats["runtime_source_provenance"] = source_provenance
            stats["precision_config_id"] = runtime_evidence["config_id"]
            stats["precision_config_path"] = runtime_evidence["config_path"]
            stats["precision_config_sha256"] = runtime_evidence["config_sha256"]
            stats["measurement_repetition"] = repetition + 1
            repeated_rows.append(stats)
            print(runner._format_row(f"warmed[{repetition + 1}]", stats))

        result = copy.deepcopy(repeated_rows[-1])
        for key in ("elapsed_s", "e2e_t/s/u", "ttft_ms", "decode_elapsed_s", "decode_t/s/u"):
            result[key] = statistics.median(row[key] for row in repeated_rows)
        result["warmed_repetitions"] = repeated_rows
        result["measurement_repetition"] = "median"
        return [result]
    finally:
        generator.teardown()


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
    sweep_kwargs, sweep_artifact_dir = _datatype_sweep_readiness_kwargs(model_dir)
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
            **sweep_kwargs,
        },
    )
    assert stats[0]["top5"] >= 0.98
    assert stats[0]["top100"] == 1.0
    artifact = (
        sweep_artifact_dir / "prefill_readiness.json"
        if sweep_artifact_dir is not None
        else model_dir / "doc/optimized_full_model/artifacts/prefill_readiness.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
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
    """Run canonical traced teacher forcing after an unmeasured same-process warmup."""

    del device_params, reset_seeds
    runner = _readiness_runner("run_teacher_forcing")
    model_dir = Path("models/autoports/openai_gpt_oss_120b").resolve()
    sweep_kwargs, sweep_artifact_dir = _datatype_sweep_readiness_kwargs(model_dir)
    reference = model_dir / "doc/full_model/references/aime24_chat_100_top100.refpt"
    build_kwargs = {
        "snapshot_path": SNAPSHOT,
        "tensor_cache_path": "/tmp/gpt_oss_120b_full_model_tensor_cache",
        "max_seq_len": HF_CONTEXT_LENGTH,
        "max_batch_size": 1,
        **sweep_kwargs,
    }
    repetitions = int(os.environ.get("GPT_OSS_120B_DATATYPE_SWEEP_REPETITIONS", "1"))
    if repetitions < 1:
        raise ValueError("GPT_OSS_120B_DATATYPE_SWEEP_REPETITIONS must be at least one")
    stats = _run_warmed_teacher_forcing(
        runner=runner,
        model_dir=model_dir,
        reference=reference,
        mesh_device=mesh_device,
        build_kwargs=build_kwargs,
        repetitions=repetitions,
    )
    artifact = (
        sweep_artifact_dir / "teacher_forcing_readiness.json"
        if sweep_artifact_dir is not None
        else model_dir / "doc/optimized_full_model/artifacts/teacher_forcing_readiness.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    if sweep_artifact_dir is None:
        assert stats[0]["top5"] >= 0.98
        assert stats[0]["top100"] == 1.0


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
    output_dir = model_dir / "doc/optimized_full_model/artifacts/autoregressive"
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
    artifact = (
        model_dir / "doc/datatype_sweep/artifacts/selected/qualitative_tt_chat.json"
        if os.environ.get("GPT_OSS_120B_DATATYPE_SWEEP_SELECTED") == "1"
        else model_dir / "doc/optimized_full_model/qualitative/qualitative_tt_chat.json"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    generator.teardown()
