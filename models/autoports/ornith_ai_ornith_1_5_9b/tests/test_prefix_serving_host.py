# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-owned serving cache wiring; these tests never open devices."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM

NS = "models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm"


class PrefixServingTests(unittest.TestCase):
    def initialize(self, env):
        model = SimpleNamespace(
            mesh_device=SimpleNamespace(get_num_devices=lambda: 4), page_block_size=64, precision={}
        )
        hf = SimpleNamespace(rope_parameters={})
        with (
            patch.dict(NS + ".os.environ", env, clear=True),
            patch(NS + ".validated_serving_context", return_value=(262144, None)),
            patch(NS + ".OrnithModel.from_pretrained", return_value=model) as load,
            patch(NS + ".atexit.register"),
        ):
            adapter = TTOrnithForCausalLM.initialize_vllm_model(hf, model.mesh_device, 1)
        return adapter, load.call_args.kwargs

    def test_opt_in_reaches_model_without_scheduler_apc(self):
        _, options = self.initialize(
            {"ORNITH_PREFIX_CACHE_BYTES": "268435456", "ORNITH_PREFIX_CACHE_MAX_TOKENS": "2048"}
        )
        self.assertEqual(options["prefix_cache_bytes"], 268435456)
        self.assertEqual(options["prefix_cache_max_tokens"], 2048)
        self.assertFalse(TTOrnithForCausalLM.model_capabilities["supports_prefix_caching"])

    def test_default_is_disabled(self):
        _, options = self.initialize({})
        self.assertEqual(options["prefix_cache_bytes"], 0)

    def test_bad_limits_fail_before_weight_load(self):
        for key, value in (
            ("ORNITH_PREFIX_CACHE_BYTES", "-1"),
            ("ORNITH_PREFIX_CACHE_BYTES", "oops"),
            ("ORNITH_PREFIX_CACHE_MAX_TOKENS", "0"),
            ("ORNITH_PREFIX_CACHE_MAX_TOKENS", "63"),
        ):
            with (
                self.subTest(key=key, value=value),
                patch.dict(NS + ".os.environ", {key: value}, clear=True),
                patch(NS + ".validated_serving_context", return_value=(262144, None)),
                patch(NS + ".OrnithModel.from_pretrained") as load,
            ):
                with self.assertRaisesRegex(ValueError, "ORNITH_PREFIX_CACHE"):
                    TTOrnithForCausalLM.initialize_vllm_model(
                        SimpleNamespace(), SimpleNamespace(get_num_devices=lambda: 4), 1
                    )
                load.assert_not_called()

    def test_teardown_releases_traces_before_prefix_and_is_idempotent(self):
        adapter, _ = self.initialize({})
        events = []
        cache = SimpleNamespace(
            counters={"hits": 2, "reused_tokens": 128}, size_bytes=1024, budget_bytes=4096, max_tokens=64
        )
        adapter.model.prefix_cache = cache
        adapter.model.clear_prefix_cache = Mock(side_effect=lambda: events.append("prefix"))
        adapter.generator = SimpleNamespace(counters={}, teardown=lambda: events.append("traces"))
        self.assertEqual(adapter.prefix_cache_stats()["reused_tokens"], 128)
        adapter.teardown()
        adapter.teardown()
        self.assertEqual(events, ["traces", "prefix"])

    def test_teardown_clears_prefix_if_cache_binding_failed(self):
        adapter, _ = self.initialize({})
        adapter.model.clear_prefix_cache = Mock()
        adapter.teardown()
        adapter.model.clear_prefix_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
