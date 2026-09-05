# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from models.common.readiness_check.generate import (
    _chat_or_plain_prompt_tokens,
    _generation_stop_ids,
    _normal_token_ids,
    _resolve_prompt_text,
    _safe_pad_id,
)


@pytest.mark.parametrize(
    "tokens",
    [
        [3, 5, 8],
        [[3, 5, 8]],
        torch.tensor([3, 5, 8]),
        BatchEncoding({"input_ids": [3, 5, 8], "attention_mask": [1, 1, 1]}),
        BatchEncoding({"input_ids": torch.tensor([[3, 5, 8]])}),
    ],
)
def test_chat_prompt_accepts_tokenizer_return_forms(tokens):
    def apply_chat_template(messages, *, add_generation_prompt, tokenize):
        assert messages == [{"role": "user", "content": "prompt"}]
        assert add_generation_prompt is True
        assert tokenize is True
        return tokens

    tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template)
    assert _chat_or_plain_prompt_tokens(tokenizer, "prompt", chat_template=True) == [3, 5, 8]


@pytest.mark.parametrize("tokens", [[], [[]], BatchEncoding({"input_ids": []})])
def test_chat_prompt_rejects_empty_tokenization(tokens):
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: tokens)
    with pytest.raises(ValueError, match="no tokens"):
        _chat_or_plain_prompt_tokens(tokenizer, "prompt", chat_template=True)


def test_chat_prompt_does_not_silently_drop_batch_rows():
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [[3], [5]])
    with pytest.raises(ValueError, match="one prompt"):
        _chat_or_plain_prompt_tokens(tokenizer, "prompt", chat_template=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, []), (7, [7]), ([7, None, 9], [7, 9])],
)
def test_normal_token_ids(value, expected):
    assert _normal_token_ids(value) == expected


def test_generation_stop_ids_combines_and_deduplicates_tokenizer_and_model_ids():
    tokenizer = SimpleNamespace(
        eos_token_id=[2, 3],
        eot_token_id=None,
        unk_token_id=0,
        get_vocab=lambda: {"<|eot_id|>": 4},
        convert_tokens_to_ids=lambda token: 4 if token == "<|eot_id|>" else 0,
    )
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[3, 5]))

    assert _generation_stop_ids(tokenizer, model) == [2, 3, 5, 4]


def test_generation_stop_ids_requires_eos_or_eot():
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        eot_token_id=None,
        unk_token_id=0,
        get_vocab=lambda: {},
        convert_tokens_to_ids=lambda _token: 0,
    )
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=None))

    with pytest.raises(RuntimeError, match="Could not determine eos/eot"):
        _generation_stop_ids(tokenizer, model)


def test_safe_pad_id_prefers_non_bos_pad_and_falls_back_to_stop_id():
    assert _safe_pad_id(SimpleNamespace(pad_token_id=8, bos_token_id=1), [2]) == 8
    assert _safe_pad_id(SimpleNamespace(pad_token_id=1, bos_token_id=1), [2]) == 2
    assert _safe_pad_id(SimpleNamespace(pad_token_id=None, bos_token_id=1), []) is None


def test_resolve_prompt_text_supports_literal_and_file_sources(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")
    unused_aime_file = tmp_path / "aime.json"

    assert (
        _resolve_prompt_text(
            "text",
            prompt="literal prompt",
            prompt_file=None,
            aime24_prompts_file=unused_aime_file,
            aime24_prompt_index=0,
        )
        == "literal prompt"
    )
    assert (
        _resolve_prompt_text(
            "file",
            prompt=None,
            prompt_file=prompt_file,
            aime24_prompts_file=unused_aime_file,
            aime24_prompt_index=0,
        )
        == "file prompt"
    )


def test_resolve_prompt_text_requires_source_specific_argument(tmp_path):
    with pytest.raises(ValueError, match="--prompt is required"):
        _resolve_prompt_text(
            "text",
            prompt=None,
            prompt_file=None,
            aime24_prompts_file=tmp_path / "aime.json",
            aime24_prompt_index=0,
        )
