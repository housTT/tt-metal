# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pinned container weight resolution without downloading weights."""

from pathlib import Path
from unittest.mock import patch

from models.autoports.ornith_ai_ornith_1_5_9b.reference.hf_reference import HF_MODEL_ID, HF_REVISION, resolve_model_path


def test_container_resolves_exact_snapshot_even_if_developer_path_exists():
    with (
        patch.dict("os.environ", {"HF_MODEL": HF_MODEL_ID}, clear=True),
        patch("huggingface_hub.snapshot_download", return_value="/hf/hub/pinned") as download,
    ):
        assert resolve_model_path() == Path("/hf/hub/pinned")
    assert download.call_args.kwargs["repo_id"] == HF_MODEL_ID
    assert download.call_args.kwargs["revision"] == HF_REVISION
    assert download.call_args.kwargs["allow_patterns"] == ["*.json", "*.jinja", "*.safetensors"]


def test_explicit_snapshot_override_preserved():
    with (
        patch.dict("os.environ", {"ORNITH_MODEL_PATH": "/model/pinned", "HF_MODEL": HF_MODEL_ID}, clear=True),
        patch("huggingface_hub.snapshot_download") as download,
    ):
        assert resolve_model_path() == Path("/model/pinned")
    download.assert_not_called()


def test_different_container_model_cannot_load_ornith_silently(expect_error):
    with (
        patch.dict("os.environ", {"HF_MODEL": "different/checkpoint"}, clear=True),
        patch("huggingface_hub.snapshot_download") as download,
        expect_error(ValueError, "only supports"),
    ):
        resolve_model_path()
    download.assert_not_called()
