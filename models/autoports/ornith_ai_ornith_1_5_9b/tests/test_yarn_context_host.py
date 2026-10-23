# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only context wiring tests; no model weights or devices are opened."""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from models.autoports.ornith_ai_ornith_1_5_9b.reference.hf_reference import load_text_config, resolve_model_path
from models.autoports.ornith_ai_ornith_1_5_9b.tt.context import (
    configure_context,
    serving_rope_parameters,
    validated_serving_context,
)
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}


class YarnContextTests(unittest.TestCase):
    def setUp(self):
        self.text = load_text_config(resolve_model_path())
        self.hf = AutoConfig.from_pretrained(resolve_model_path(), local_files_only=True)
        self.contract = {"current_supported_context": 262144}

    def test_native_default_and_explicit_extension(self):
        before = copy.deepcopy(self.text.to_dict())
        native = configure_context(self.text, 262144)
        extended = configure_context(self.text, 1000000, YARN)
        self.assertEqual(native.to_dict(), before)
        self.assertEqual(self.text.to_dict(), before)
        self.assertEqual(extended.max_position_embeddings, 1000000)
        for key, value in before.items():
            if key not in ("max_position_embeddings", "rope_parameters"):
                self.assertEqual(extended.to_dict()[key], value, key)
        self.assertEqual(extended.rope_parameters, {**self.text.rope_parameters, **YARN})

    def test_invalid_extension_rejected(self):
        cases = [
            (1000000, None),
            (1048577, YARN),
            (1000000, {**YARN, "original_max_position_embeddings": 32768}),
            (1000000, {**YARN, "factor": float("inf")}),
            (1000000, {**YARN, "factor": float("nan")}),
            (1000000, {**YARN, "factor": True}),
            (1000000, {**YARN, "partial_rotary_factor": 1.0}),
            (1000000, {**YARN, "rope_theta": 10000}),
            (1000000, {**YARN, "rope_type": "dynamic"}),
            (0, YARN),
            (True, YARN),
        ]
        for context, rope in cases:
            with self.subTest(context=context, rope=rope), self.assertRaises(ValueError):
                configure_context(self.text, context, rope)

    def test_scaled_frequencies_match_hf_at_long_positions(self):
        config = configure_context(self.text, 1000000, YARN)
        inv_freq, scaling = ROPE_INIT_FUNCTIONS["yarn"](config, device="cpu")
        positions = torch.tensor([[0, 262143, 262144, 999999]])
        rotary = Qwen3_5TextRotaryEmbedding(config=config)
        cos, sin = rotary(torch.zeros(1, 4, 1), positions[None].expand(3, 1, 4))
        phases = positions.float()[..., None] * inv_freq
        phases = torch.cat((phases, phases), dim=-1)
        torch.testing.assert_close(cos, phases.cos() * scaling, rtol=0, atol=0)
        torch.testing.assert_close(sin, phases.sin() * scaling, rtol=0, atol=0)
        self.assertEqual(cos.shape[-1], 64)

    def test_legacy_and_text_fields_and_conflicts(self):
        self.assertIsNone(serving_rope_parameters(self.hf))
        self.hf.rope_scaling = dict(YARN)
        self.assertEqual(serving_rope_parameters(self.hf), YARN)
        self.hf.text_config.rope_parameters.update(YARN)
        self.assertEqual(serving_rope_parameters(self.hf), self.hf.text_config.rope_parameters)
        self.hf.rope_scaling["factor"] = 2.0
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            serving_rope_parameters(self.hf)

    def proof(self, profiles=("p150x4",)):
        # Synthetic contract exercises gating only; it is never written as real evidence.
        return {
            **self.contract,
            "optional_yarn": {
                "profiles": {
                    profile: {
                        "status": "pass",
                        "max_context": 1000000,
                        "rope_parameters": {**self.text.rope_parameters, **YARN},
                    }
                    for profile in profiles
                }
            },
        }

    def test_native_limit_retained(self):
        self.assertEqual(validated_serving_context(self.hf, self.contract, 262144, 4), (262144, None))
        with self.assertRaises(ValueError):
            validated_serving_context(self.hf, self.contract, 1000000, 4)

    def test_yarn_requires_matching_profile_evidence(self):
        self.hf.rope_scaling = dict(YARN)
        for count in (1, 2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "lacks validated"):
                validated_serving_context(self.hf, self.contract, 1000000, count)
        self.assertEqual(validated_serving_context(self.hf, self.proof(), 1000000, 4), (1000000, YARN))
        with self.assertRaisesRegex(ValueError, "lacks validated"):
            validated_serving_context(self.hf, self.proof(), 1000000, 1)
        self.hf.rope_scaling["factor"] = 2.0
        with self.assertRaisesRegex(ValueError, "differs"):
            validated_serving_context(self.hf, self.proof(), 524288, 4)

    def test_serving_passes_config_into_actual_constructor(self):
        self.hf.rope_scaling = dict(YARN)
        mesh = SimpleNamespace(get_num_devices=lambda: 4)
        model = SimpleNamespace(mesh_device=mesh, page_block_size=64, precision={})
        with (
            patch.object(Path, "read_text", return_value=json.dumps(self.proof())),
            patch(
                "models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm.OrnithModel.from_pretrained",
                return_value=model,
            ) as load,
            patch("models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm.atexit.register"),
        ):
            adapter = TTOrnithForCausalLM.initialize_vllm_model(self.hf, mesh, 1, max_seq_len=1000000)
        self.assertEqual(load.call_args.kwargs["max_context"], 1000000)
        self.assertEqual(load.call_args.kwargs["rope_parameters"], YARN)
        self.assertEqual(adapter.max_model_len, 1000000)
        self.assertGreaterEqual(adapter.page_table_blocks * 64, 1000000)

    def test_generator_factory_routes_context_and_cache_options_to_model(self):
        model = SimpleNamespace(model_path=resolve_model_path())
        namespace = "models.autoports.ornith_ai_ornith_1_5_9b.tt.generator"
        with (
            patch(namespace + ".OrnithModel.from_pretrained", return_value=model) as load,
            patch(namespace + ".OrnithGenerator") as generator,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
        ):
            build_generator(
                resolve_model_path(),
                object(),
                max_context=1000000,
                rope_parameters=YARN,
                prefix_cache_bytes=1048576,
                prefix_cache_max_tokens=4096,
                max_batch_size=1,
            )
        self.assertEqual(load.call_args.kwargs["rope_parameters"], YARN)
        self.assertEqual(load.call_args.kwargs["prefix_cache_bytes"], 1048576)
        self.assertEqual(load.call_args.kwargs["prefix_cache_max_tokens"], 4096)
        self.assertNotIn("rope_parameters", generator.call_args.kwargs)
        self.assertEqual(generator.call_args.kwargs["max_batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
