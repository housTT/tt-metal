# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import transformers

from models.common.readiness_check import run_vllm_server


class _NoTemplateTokenizer:
    chat_template = None

    @staticmethod
    def encode(prompt, *, add_special_tokens):
        assert prompt == "A prompt without a template"
        assert add_special_tokens is False
        return [10, 20, 30]


class _CompletionEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(text=f"completion-{len(self.calls)}")])


def test_qualitative_no_chat_template_uses_raw_completion(monkeypatch, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("A prompt without a template")
    completion_endpoint = _CompletionEndpoint()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(AssertionError("chat API used")))
        ),
        completions=completion_endpoint,
    )

    monkeypatch.setattr(run_vllm_server.openai, "OpenAI", lambda **_: fake_client)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: _NoTemplateTokenizer())
    monkeypatch.setattr(run_vllm_server.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))

    run_vllm_server._run_qualitative_prompts(
        server_url="http://localhost:8000",
        hf_model="no-template-model",
        prompts_file=prompts_file,
        output_dir=tmp_path,
    )

    result = run_vllm_server.json.loads((tmp_path / "vllm_qualitative_outputs.json").read_text())[0]
    assert result["prompt_format"] == "raw_completion_no_chat_template"
    assert result["rendered_prompt"] == "A prompt without a template"
    assert result["prompt_token_ids"] == [10, 20, 30]
    assert [call["extra_body"] for call in completion_endpoint.calls] == [{"top_k": 1}, {"top_k": 32}]
