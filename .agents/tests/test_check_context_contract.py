from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_context_contract.py"
SPEC = importlib.util.spec_from_file_location("check_context_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ProfileManifestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.limits = {"P150": 50624, "P150x2": 262144, "P150x4": 262144}

    def scan(self, data, relative_path="readiness_vllm/unified_gate_manifest.json", limits=True):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return CHECK.scan_caps(self.root, 262144, self.limits if limits else None)[0]

    def test_aggregate_manifest_matches_profile_contract(self):
        data = {"profiles": {name: {"max_model_len": limit} for name, limit in self.limits.items()}}
        self.assertEqual(self.scan(data), [])

    def test_reduced_profile_cannot_reduce_its_recorded_limit(self):
        findings = self.scan({"profiles": {"P150": {"max_model_len": 50623}}})
        self.assertEqual(len(findings), 1)
        self.assertIn("below 50624", findings[0])

    def test_full_context_profiles_cannot_borrow_reduced_limit(self):
        findings = self.scan({"profiles": {name: {"max_model_len": 50624} for name in self.limits}})
        self.assertEqual(len(findings), 2)
        self.assertTrue(all("below 262144" in item for item in findings))

    def test_unknown_profile_keeps_global_floor(self):
        findings = self.scan({"profiles": {"unknown": {"max_model_len": 50624}}})
        self.assertEqual(len(findings), 1)
        self.assertIn("below 262144", findings[0])

    def test_global_and_sibling_values_keep_global_floor(self):
        findings = self.scan(
            {
                "max_model_len": 50624,
                "profiles": {"P150": {"max_model_len": 50624}},
                "other": {"max_model_len": 50624},
            }
        )
        self.assertEqual(len(findings), 2)
        self.assertTrue(all("below 262144" in item for item in findings))

    def test_nested_settings_and_arrays_inherit_explicit_profile(self):
        data = {"profiles": {"P150": {"runs": [{"config": {"max_model_len": 50624}}]}}}
        self.assertEqual(self.scan(data), [])

    def test_no_profile_contract_keeps_global_floor(self):
        findings = self.scan({"profiles": {"P150": {"max_model_len": 50624}}}, limits=False)
        self.assertEqual(len(findings), 1)
        self.assertIn("below 262144", findings[0])

    def test_profile_directory_still_uses_its_contract(self):
        self.assertEqual(self.scan({"max_model_len": 50624}, "readiness_vllm/P150/config.json"), [])
        findings = self.scan({"max_model_len": 50623}, "readiness_vllm/P150/config.json")
        self.assertEqual(len(findings), 1)
        self.assertIn("below 50624", findings[0])

    def test_aggregate_override_does_not_escape_into_profile_directory(self):
        findings = self.scan({"profiles": {"P150": {"max_model_len": 50624}}}, "readiness_vllm/P150x2/config.json")
        self.assertEqual(len(findings), 1)
        self.assertIn("below 262144", findings[0])

    def test_stage_report_roots_use_explicit_profiles(self):
        data = {"profiles": {name: {"config": {"max_model_len": limit}} for name, limit in self.limits.items()}}
        for stage in ("vllm_integration", "optimized_vllm", "tti_release"):
            with self.subTest(stage=stage):
                self.assertEqual(self.scan(data, f"doc/{stage}/perf_summary.json"), [])

    def test_stage_report_preserves_other_context_floors(self):
        findings = self.scan(
            {
                "max_model_len": 50624,
                "profiles": {
                    "P150": {"max_model_len": 50624},
                    "P150x2": {"max_model_len": 50624},
                    "unknown": {"max_model_len": 50624},
                },
            },
            "doc/optimized_vllm/perf_summary.json",
        )
        self.assertEqual(len(findings), 3)
        self.assertTrue(all("below 262144" in item for item in findings))

    def test_stage_report_cannot_reduce_recorded_profile_limit(self):
        findings = self.scan({"profiles": {"P150": {"max_model_len": 50623}}}, "doc/optimized_vllm/perf_summary.json")
        self.assertEqual(len(findings), 1)
        self.assertIn("below 50624", findings[0])

    def test_nested_report_does_not_gain_aggregate_override(self):
        findings = self.scan({"profiles": {"P150": {"max_model_len": 50624}}}, "doc/optimized_vllm/P150x2/config.json")
        self.assertEqual(len(findings), 1)
        self.assertIn("below 262144", findings[0])


if __name__ == "__main__":
    unittest.main()
