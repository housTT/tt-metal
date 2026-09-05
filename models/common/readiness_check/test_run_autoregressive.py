# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import pytest
import torch
from transformers import BatchEncoding

from models.common.readiness_check.generate import _chat_or_plain_prompt_tokens

runner = importlib.import_module("models.common.readiness_check.run_autoregressive")


class PromptTokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        return ([1] if add_special_tokens else []) + [ord(char) for char in text]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt
        rendered = f"<user>{messages[0]['content']}</user>\n<assistant>\n"
        if tokenize:
            return BatchEncoding({"input_ids": self.encode(rendered, add_special_tokens=False)})
        return rendered

    def decode(self, tokens, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(chr(token) for token in tokens)


@pytest.mark.parametrize("chat_template", [False, True])
def test_autoregressive_preserves_exact_hf_and_tt_prompt_ids(monkeypatch, tmp_path, chat_template):
    tokenizer = PromptTokenizer()
    raw_prompt = "  Explain this.\n"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(raw_prompt)
    observed = {}

    class HFModel:
        def eval(self):
            return self

        def to(self, device):
            assert device.type == "cpu"
            return self

        def generate(self, input_ids, **kwargs):
            observed["hf_ids"] = input_ids[0].tolist()
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 2
            return torch.cat((input_ids, torch.tensor([[72, 70]])), dim=1)

    def load_model(hf_model_id, **kwargs):
        assert hf_model_id == "test-local-checkpoint"
        assert kwargs == {"trust_remote_code": True}
        return HFModel()

    def load_tokenizer(hf_model_id, **kwargs):
        assert hf_model_id == "test-local-checkpoint"
        return tokenizer

    class TTGenerator:
        def generate(self, *, prompt_token_ids, max_new_tokens, next_input):
            observed["tt_ids"] = prompt_token_ids
            assert max_new_tokens == 2
            assert next_input is None
            return [84, 84]

        def teardown(self):
            observed["teardown"] = True

    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "_import_build_generator", lambda path: lambda **kwargs: TTGenerator())
    kwargs = {"chat_template": True} if chat_template else {}  # Exercise legacy default.
    paths = runner.run_autoregressive(
        model_dir=tmp_path,
        hf_model_id="test-local-checkpoint",
        prompt_file=prompt_file,
        mesh_device=None,
        output_dir=tmp_path / "output",
        max_new_tokens=2,
        **kwargs,
    )
    prompt = raw_prompt if chat_template else raw_prompt.strip()
    expected = _chat_or_plain_prompt_tokens(tokenizer, prompt, chat_template=chat_template)
    assert observed == {"hf_ids": expected, "tt_ids": expected, "teardown": True}
    meta = json.loads(paths["meta"].read_text())
    assert meta["prompt_token_ids"] == expected
    assert meta["prompt_text"] == prompt
    assert meta["chat_template"] is chat_template
    assert meta["prompt_mode"] == ("chat" if chat_template else "completion")
    if chat_template:
        assert meta["rendered_prompt"].endswith("<assistant>\n")
        assert tokenizer.encode(meta["rendered_prompt"], add_special_tokens=False) == expected
        assert tokenizer.encode(meta["rendered_prompt"].strip(), add_special_tokens=False) != expected
    else:
        assert meta["rendered_prompt"] == raw_prompt.strip()
    assert paths["hf_completion"].read_text() == "HF"
    assert paths["tt_completion"].read_text() == "TT"
