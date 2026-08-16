# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loading the HuggingFace reference model for a readiness run.

``AutoModelForCausalLM`` maps a config's ``model_type`` to exactly one class. That is right for a
plain text checkpoint, and wrong for a checkpoint whose text decoder is wrapped in a multimodal
model: the auto class then expects ``model.layers.*`` while the checkpoint stores
``model.language_model.layers.*``, ``from_pretrained`` reports every weight as missing, and the
runner gets a **randomly initialised** model with no error. A reference or an HF control generated
that way is silently meaningless.

``load_hf_reference_model`` prefers the architecture the checkpoint itself declares in
``config.architectures``, which is the class whose parameter names the checkpoint was saved from.
For an ordinary text checkpoint the declared architecture *is* what ``AutoModelForCausalLM``
resolves to, so nothing changes; only the wrapped-multimodal case takes a different path.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM


def resolve_hf_model_class(hf_model_id: str, *, trust_remote_code: bool = True):
    """The class to instantiate for ``hf_model_id``, and the reason it was chosen."""
    import transformers

    try:
        config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
    except Exception:  # noqa: BLE001 - fall back to the auto class and let it produce the error
        return AutoModelForCausalLM, "config unavailable; using AutoModelForCausalLM"

    architectures = list(getattr(config, "architectures", None) or [])
    for name in architectures:
        declared = getattr(transformers, name, None)
        if declared is None or not hasattr(declared, "from_pretrained") or not hasattr(declared, "generate"):
            continue
        return declared, f"checkpoint declares architectures={architectures!r}"
    return AutoModelForCausalLM, f"no usable declared architecture in {architectures!r}"


def load_hf_reference_model(hf_model_id: str, *, trust_remote_code: bool = True, **kwargs: Any):
    """Load the HF reference model, using the checkpoint's declared architecture when there is one."""
    model_class, reason = resolve_hf_model_class(hf_model_id, trust_remote_code=trust_remote_code)
    print(f"Loading {hf_model_id} as {model_class.__name__} ({reason})")
    return model_class.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code, **kwargs)


__all__ = ["load_hf_reference_model", "resolve_hf_model_class"]
