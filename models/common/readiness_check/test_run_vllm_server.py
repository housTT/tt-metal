# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.common.readiness_check import run_vllm_server
from models.common.readiness_check.run_vllm_server import _qualitative_prompt_mode, _request_qualitative_completion


@pytest.mark.parametrize(
    ("chat_template", "expected"),
    [("{{ messages }}", "chat"), (None, "completion")],
)
def test_qualitative_prompt_mode_follows_checkpoint_chat_template(monkeypatch, chat_template, expected):
    tokenizer_factory = Mock(return_value=SimpleNamespace(chat_template=chat_template))
    monkeypatch.setattr(run_vllm_server.AutoTokenizer, "from_pretrained", tokenizer_factory)

    assert _qualitative_prompt_mode("org/model") == expected
    tokenizer_factory.assert_called_once_with("org/model")


def test_request_qualitative_completion_uses_chat_endpoint_for_chat_models():
    chat_create = Mock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="chat response"))])
    )
    completion_create = Mock()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        completions=SimpleNamespace(create=completion_create),
    )

    result = _request_qualitative_completion(
        client=client,
        hf_model="org/instruct-model",
        prompt="Question",
        prompt_mode="chat",
        temperature=0.0,
    )

    assert result == "chat response"
    chat_create.assert_called_once_with(
        messages=[{"role": "user", "content": "Question"}],
        model="org/instruct-model",
        max_tokens=256,
        temperature=0.0,
    )
    completion_create.assert_not_called()


def test_request_qualitative_completion_uses_raw_endpoint_for_base_models():
    completion_create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(text="base response")]))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())),
        completions=SimpleNamespace(create=completion_create),
    )

    result = _request_qualitative_completion(
        client=client,
        hf_model="org/base-model",
        prompt="Continue",
        prompt_mode="completion",
        temperature=0.7,
        top_p=0.9,
    )

    assert result == "base response"
    completion_create.assert_called_once_with(
        prompt="Continue",
        model="org/base-model",
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )


def test_tt_config_flag_matches_installed_engine_args():
    pytest.importorskip("vllm")
    from vllm.engine.arg_utils import EngineArgs

    fields = {field.name for field in dataclasses.fields(EngineArgs)}
    if "plugin_config" in fields:
        expected = "--plugin-config"
    elif "additional_config" in fields:
        expected = "--additional-config"
    else:
        pytest.fail("Installed vLLM has no supported TT configuration field")
    assert run_vllm_server._tt_config_flag() == expected


def test_launch_passes_namespaced_config_and_creates_private_session(monkeypatch, tmp_path):
    pytest.importorskip("vllm")
    flag = run_vllm_server._tt_config_flag()
    popen = Mock()
    monkeypatch.setattr(run_vllm_server.subprocess, "Popen", popen)
    config = {"sample_on_device_mode": "all", "trace_region_size": 1024}

    proc = run_vllm_server._launch_server(
        hf_model="org/model",
        mesh_device="P150x4",
        max_num_seqs=1,
        block_size=64,
        port=18001,
        log_file=tmp_path / "server.log",
        max_model_len=2048,
        tt_config=config,
        additional_args=[],
    )

    assert proc is popen.return_value
    cmd = popen.call_args.args[0]
    assert json.loads(cmd[cmd.index(flag) + 1]) == {"tt": config}
    assert popen.call_args.kwargs["env"]["MESH_DEVICE"] == "P150x4"
    assert popen.call_args.kwargs["start_new_session"] is True


def test_cli_accepts_p150x4_for_local_serving(monkeypatch, tmp_path):
    launch = Mock()
    monkeypatch.setattr(run_vllm_server, "_launch_server", launch)
    for name in ["_check_port_available", "_wait_for_server", "_hold_until_signal", "_shutdown"]:
        monkeypatch.setattr(run_vllm_server, name, Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_vllm_server",
            "--model-dir",
            str(tmp_path),
            "--hf-model",
            "org/model",
            "--stages",
            "serve",
            "--mesh-device",
            "P150x4",
        ],
    )

    run_vllm_server._main()

    assert launch.call_args.kwargs["mesh_device"] == "P150x4"


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux process groups and subreaper support")
def test_shutdown_kills_orphan_child_after_launcher_exits(tmp_path):
    # Isolate subreaper state in a helper interpreter so the orphan can be reaped
    # without changing pytest's process ownership or leaving a zombie behind.
    child_script = textwrap.dedent(
        """
        import json, os, signal, sys, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        Path(sys.argv[1]).write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0)}))
        time.sleep(45)
        """
    )
    launcher_script = textwrap.dedent(
        """
        import subprocess, sys, time
        from pathlib import Path
        subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1]])
        deadline = time.monotonic() + 5
        while not Path(sys.argv[1]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Dummy child did not become ready")
            time.sleep(0.01)
        """
    )
    script = textwrap.dedent(
        """
        import ctypes
        import json
        import os
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path
        from models.common.readiness_check.run_vllm_server import _shutdown

        assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
        ready = Path(sys.argv[1])
        child_script, launcher_script = sys.argv[2:]
        launcher = subprocess.Popen(
            [sys.executable, "-c", launcher_script, str(ready), child_script],
            start_new_session=True,
        )
        child = None
        reaped = False
        try:
            assert launcher.wait(timeout=7) == 0
            child = json.loads(ready.read_text())
            assert child["pgid"] == child["sid"] == launcher.pid
            assert child["pgid"] != os.getpgrp()
            assert os.getpgid(child["pid"]) == launcher.pid
            _shutdown(launcher, ready.parent / "absent-server.log")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                pid, status = os.waitpid(child["pid"], os.WNOHANG)
                if pid:
                    reaped = True
                    assert os.WIFSIGNALED(status)
                    assert os.WTERMSIG(status) == signal.SIGKILL
                    print("ORPHAN_REAPED_AFTER_SIGKILL")
                    break
                time.sleep(0.01)
            assert reaped, "Orphan child survived runner shutdown"
        finally:
            # Only signal the private group created immediately above.
            try:
                os.killpg(launcher.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            launcher.wait(timeout=5)
            if child is not None and not reaped:
                os.waitpid(child["pid"], 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "child-ready.json"), child_script, launcher_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORPHAN_REAPED_AFTER_SIGKILL" in result.stdout
