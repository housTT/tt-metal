import json

import pytest

from models.autoports.cloudflare_clef.tt import precision_defaults as pd

pytestmark = pytest.mark.eager_host_side

KNOB_ENV = tuple(pd.KNOBS) + ("CLEF_PRECISION", "QWEN35_GDN_STATE_BF16", "QWEN36_NOT_A_KNOB")


@pytest.fixture
def clean_env(monkeypatch):
    for key in KNOB_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def selected_file(tmp_path):
    def write(runtime_flags):
        path = tmp_path / "selected_precision_config.json"
        path.write_text(json.dumps({"config_id": "test", "runtime_flags": runtime_flags}))
        return path

    return write


def test_stage1_profile_lists_the_seven_live_knobs():
    assert set(pd.STAGE1_PROFILE) == {
        "QWEN36_MLP_GATE_UP_DTYPE",
        "QWEN36_MLP_DOWN_DTYPE",
        "QWEN36_PROJ_DTYPE",
        "QWEN36_MATMUL_FIDELITY",
        "QWEN36_GDN_GATE_FP32",
        "CLEF_VISION_PRECISION",
        "CLEF_VISION_ACT_BF16",
    }
    assert "QWEN36_GDN_QKV_FP32" not in pd.STAGE1_PROFILE
    assert pd.profile("stage1") == pd.STAGE1_PROFILE
    assert pd.profile("stage1") is not pd.STAGE1_PROFILE


def test_selected_profile_parses_runtime_flags(selected_file):
    path = selected_file({"QWEN36_MLP_DOWN_DTYPE": "bfp8", "QWEN35_GDN_STATE_BF16": 1, "UNRELATED": "x"})
    assert pd.selected_profile(path) == {"QWEN36_MLP_DOWN_DTYPE": "bfp8", "QWEN35_GDN_STATE_BF16": "1"}


def test_selected_profile_is_none_without_file_or_flags(tmp_path, selected_file):
    assert pd.selected_profile(tmp_path / "missing.json") is None
    assert pd.selected_profile(selected_file({})) is None
    assert pd.selected_profile(selected_file({"UNRELATED": "x"})) is None


def test_profile_name_resolution_order(clean_env, selected_file):
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({"QWEN36_MLP_DOWN_DTYPE": "bfp8"}))
    assert pd.profile_name() == "selected"
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({}))
    assert pd.profile_name() == "stage1"
    clean_env.setenv("CLEF_PRECISION", "stage1")
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({"QWEN36_MLP_DOWN_DTYPE": "bfp8"}))
    assert pd.profile_name() == "stage1"
    clean_env.setenv("CLEF_PRECISION", "anything")
    assert pd.profile_name() == "anything"


def test_selected_profile_overlays_stage1(clean_env, selected_file):
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({"QWEN36_MLP_DOWN_DTYPE": "bfp8"}))
    values = pd.profile()
    assert values["QWEN36_MLP_DOWN_DTYPE"] == "bfp8"
    assert {k: v for k, v in values.items() if k != "QWEN36_MLP_DOWN_DTYPE"} == {
        k: v for k, v in pd.STAGE1_PROFILE.items() if k != "QWEN36_MLP_DOWN_DTYPE"
    }


def test_shipped_selected_file_equals_stage1(clean_env):
    assert pd.SELECTED_PATH.exists()
    assert pd.profile_name() == "selected"
    assert pd.selected_profile() == pd.STAGE1_PROFILE
    assert pd.profile() == pd.STAGE1_PROFILE


def test_profile_errors(clean_env, tmp_path, expect_error):
    clean_env.setattr(pd, "SELECTED_PATH", tmp_path / "missing.json")
    with expect_error(FileNotFoundError, "has no runtime_flags"):
        pd.profile("selected")
    with expect_error(ValueError, "expected 'selected' or one of"):
        pd.profile("nonsense")


def test_apply_respects_existing_environment(clean_env, selected_file):
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({"QWEN36_MLP_DOWN_DTYPE": "bfp8"}))
    clean_env.setenv("QWEN36_MATMUL_FIDELITY", "HiFi4")
    active = pd.apply()
    assert active["QWEN36_MATMUL_FIDELITY"] == "HiFi4"
    assert active["QWEN36_MLP_DOWN_DTYPE"] == "bfp8"
    assert active["QWEN36_PROJ_DTYPE"] == "bfp8"
    assert set(active) == set(pd.KNOBS)


def test_active_lists_extra_knobs_of_the_selected_profile(clean_env, selected_file):
    clean_env.setattr(pd, "SELECTED_PATH", selected_file({"QWEN35_GDN_STATE_BF16": "1"}))
    active = pd.apply()
    assert active["QWEN35_GDN_STATE_BF16"] == "1"
    assert set(active) == set(pd.KNOBS) | {"QWEN35_GDN_STATE_BF16"}
