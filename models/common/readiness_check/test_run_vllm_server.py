from types import SimpleNamespace

from models.common.readiness_check.run_vllm_server import _server_command, _stream_choice_is_token_event


def test_stream_choice_is_token_event_counts_empty_text_tokens():
    assert _stream_choice_is_token_event(SimpleNamespace(text="hello", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text=" ", finish_reason=None))
    assert _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason=None))


def test_stream_choice_is_token_event_ignores_terminal_empty_chunks():
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="stop"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text="", finish_reason="length"))
    assert not _stream_choice_is_token_event(SimpleNamespace(text=None, finish_reason=None))


def test_server_command_preserves_the_historical_single_frontend(monkeypatch):
    monkeypatch.setattr(
        "models.common.readiness_check.run_vllm_server._tt_config_flag",
        lambda: "--additional-config",
    )
    command = _server_command(
        hf_model="org/model",
        max_num_seqs=8,
        block_size=64,
        port=8100,
        max_model_len=262144,
        tt_config={"sample_on_device_mode": "all"},
        additional_args=["--disable-log-stats"],
    )

    assert command[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
    assert "--api-server-count" not in command
    assert command[-1] == "--disable-log-stats"


def test_server_command_uses_the_multi_frontend_orchestrator(monkeypatch):
    monkeypatch.setattr(
        "models.common.readiness_check.run_vllm_server._tt_config_flag",
        lambda: "--additional-config",
    )
    command = _server_command(
        hf_model="org/model",
        max_num_seqs=8,
        block_size=64,
        port=8100,
        max_model_len=262144,
        tt_config={"input_queue_batching_delay": 2.0},
        additional_args=[],
        api_server_count=4,
    )

    assert command[1:5] == ["-m", "vllm.entrypoints.cli.main", "serve", "org/model"]
    count = command.index("--api-server-count")
    assert command[count + 1] == "4"
