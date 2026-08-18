# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The selected precision config, checked without a device.

``tests/test_full_model.py`` parametrizes ``mesh_device`` at module scope, so every case in it opens
the 1x4 ring. These three cases are pure policy resolution - schema completeness, the round trip, the
override chain and per-layer exception resolution - and they are the ones that must keep working even
where no hardware is attached, so they live here instead. The built-model half of the same contract
(the artifact's fields against the device tensors and the constructed compute-kernel configs) is
``test_full_model.py::test_the_selected_precision_config_is_the_built_policy``.

Run:

    pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_precision_config.py -q
"""

from __future__ import annotations

import json

import ttnn


def test_the_selected_precision_config_artifact_is_complete_and_round_trips(expect_error):
    """Every ``PrecisionPolicy`` field must survive a write/read cycle through the artifact.

    ``tt/precision_config.py`` asserts at import time that its JSON schema covers the dataclass, so
    a policy field added later cannot silently vanish from the selected config. This test closes the
    other half: the file on disk is complete enough to rebuild the policy exactly, so what the sweep
    recorded is what a later build constructs.
    """
    from dataclasses import fields

    from models.autoports.ornith_ai_ornith_1_0_35b.tt import precision_config as PC

    assert (
        PC.SELECTED_PRECISION_CONFIG_PATH.is_file()
    ), f"{PC.SELECTED_PRECISION_CONFIG_PATH} is the model's default precision policy and must exist"
    selected = PC.load_selected_policy()
    payload = PC.policy_to_dict(selected)
    assert PC.policy_from_dict(payload) == selected
    for field in fields(selected):
        assert field.name in PC.SCHEMA, f"{field.name} has no place in the selected-config schema"

    # A *partial* file must raise rather than fall back to a dataclass default no measurement chose.
    # That includes `layer_exceptions`: an omitted key is a partial file, not "no exceptions".
    for section, key, _kind in PC.SCHEMA.values():
        partial = json.loads(json.dumps(payload))
        if section == "layer_exceptions":
            partial.pop("layer_exceptions")
        elif not section:
            partial.pop(key)
        else:
            partial[section].pop(key)
        with expect_error(ValueError, "precision config is missing"):
            PC.policy_from_dict(partial)


def test_the_precision_policy_can_be_overridden_back_to_the_safe_baseline(monkeypatch, expect_error):
    """The escape hatch the skill asks for: one change returns the build to the pre-sweep policy."""
    from models.autoports.ornith_ai_ornith_1_0_35b.tt import precision_config as PC
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY as DECODER_STAGE_POLICY
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import FUSED_PARITY_POLICY

    assert PC.resolve_policy(None) == PC.load_selected_policy()
    monkeypatch.setenv(PC.POLICY_ENV_VAR, "optimized")
    assert PC.resolve_policy(None) is DECODER_STAGE_POLICY
    monkeypatch.setenv(PC.POLICY_ENV_VAR, "fused-parity")
    assert PC.resolve_policy(None) is FUSED_PARITY_POLICY
    monkeypatch.delenv(PC.POLICY_ENV_VAR)
    assert PC.resolve_policy("optimized") is DECODER_STAGE_POLICY
    assert PC.resolve_policy(PC.SELECTED_PRECISION_CONFIG_PATH) == PC.load_selected_policy()
    with expect_error(ValueError, "unknown precision policy"):
        PC.resolve_policy("no-such-policy")


def test_layer_exceptions_resolve_per_layer_and_leave_other_layers_identical(expect_error):
    """``for_layer`` is what makes a layer exception a property of the built layer.

    Checked without a device because it is pure policy resolution; the built-model half is
    ``test_the_selected_precision_config_is_the_built_policy``.
    """
    from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY as P

    plain = P.replace(name="plain")
    assert plain.for_layer(0) is plain, "a policy with no exceptions must return itself"

    excepted = plain.replace(
        name="excepted",
        layer_exceptions=((0, "proj_dtype", ttnn.bfloat8_b), (39, "proj_dtype", ttnn.bfloat8_b)),
    )
    inner = excepted.for_layer(7)
    assert inner is excepted and inner.proj_dtype == excepted.proj_dtype
    for idx in (0, 39):
        edge = excepted.for_layer(idx)
        assert edge is not excepted
        assert edge.proj_dtype == ttnn.bfloat8_b
        assert edge.layer_exceptions == ()
        assert edge.name == f"excepted@L{idx}"
        # A name-keyed table must not lose its measured entry just because a layer is exceptional.
        assert edge.base_name == "excepted"
    assert excepted.exception_layers() == (0, 39)
    with expect_error(ValueError, "unknown policy field"):
        plain.replace(layer_exceptions=((0, "not_a_field", 1),)).for_layer(0)
