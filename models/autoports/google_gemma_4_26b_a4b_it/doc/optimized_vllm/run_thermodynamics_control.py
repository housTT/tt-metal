#!/usr/bin/env python3
"""Compare the retained sampled prefix on original and selected TT adapters."""

import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_profiles import HF_MODEL, MODEL, PROFILES, PYTHON, ROOT

DOC = Path(__file__).resolve().parent
ADAPTER = MODEL / "tt/generator_vllm.py"
BASE_COMMIT = "6eb0427423392d7c6a7f87a511be892b8bf677ae"
SELECTED_SHA256 = "649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325"


def main():
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2).close()
    except urllib.error.URLError:
        pass
    else:
        raise RuntimeError("An existing server must finish before source changes")
    selected = ADAPTER.read_bytes()
    assert hashlib.sha256(selected).hexdigest() == SELECTED_SHA256
    baseline = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{ADAPTER.relative_to(ROOT)}"], cwd=ROOT)
    source_path = MODEL / "readiness_vllm/P150x4/optimized_vllm/after_warmed/vllm_qualitative_outputs.json"
    source = json.loads(source_path.read_text())[3]
    prefix = source["sampled_completion"].split("It states that energy transfers", 1)[0]
    assert prefix.endswith("processes. ")
    prompt = source["rendered_prompt"] + prefix
    out = DOC / "thermodynamics_control"
    out.mkdir(exist_ok=False)
    backup = out / "adapter_backup.py"
    backup.write_bytes(selected)
    ledger = {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "hf_model": HF_MODEL,
        "hf_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        "purpose": "Teacher-forced continuation of actual sampled chat response, not a replacement qualitative suite",
        "prompt": prompt,
        "sampled_prefix": prefix,
        "request_sequence": "Three identical unseeded positive-temperature continuation requests per fresh server, retained without selection",
        "adapter_restored": False,
        "runs": [],
    }
    ledger_path = out / "manifest.json"

    def save():
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")

    env = dict(os.environ)
    assert not [key for key in env if ("PROFILER" in key or "TRACY" in key) and env[key] not in ("", "0")]
    env.update(
        HF_HOME="/home/hous/.cache/huggingface",
        HF_HUB_OFFLINE="1",
        TT_METAL_HOME=str(ROOT),
        TT_GEMMA4_TEXT_VER="google_gemma_4_26b_a4b_it_autoport",
        TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}',
    )
    _, mesh, context, config = PROFILES["P150x4"]
    try:
        for label, code in (("original", baseline), ("optimized", selected)):
            ADAPTER.write_bytes(code)
            command = [
                str(PYTHON),
                "-m",
                "models.common.readiness_check.run_vllm_server",
                "--stages",
                "serve",
                "--model-dir",
                str(MODEL),
                "--hf-model",
                HF_MODEL,
                "--mesh-device",
                mesh,
                "--port",
                "8000",
                "--max-num-seqs",
                "32",
                "--block-size",
                "64",
                "--max-model-len",
                str(context),
                "--tt-config",
                json.dumps(config),
                "--output-subdir",
                f"P150x4/optimized_vllm/thermodynamics_{label}",
                "--additional-server-args",
                "--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4",
            ]
            run = {"label": label, "adapter_sha256": hashlib.sha256(code).hexdigest(), "argv": command, "responses": []}
            ledger["runs"].append(run)
            save()
            with (out / f"{label}_launch.log").open("w") as log:
                server = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    deadline = time.monotonic() + 1200
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"Server exited early: {server.returncode}")
                        try:
                            urllib.request.urlopen("http://localhost:8000/health", timeout=2).close()
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError("Server startup deadline")
                            time.sleep(2)
                    for index in range(3):
                        body = {
                            "model": HF_MODEL,
                            "prompt": prompt,
                            "max_tokens": 256,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 32,
                        }
                        request = urllib.request.Request(
                            "http://localhost:8000/v1/completions",
                            data=json.dumps(body).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=180) as response:
                            raw = response.read()
                        raw_path = out / f"{label}_{index}.json"
                        raw_path.write_bytes(raw)
                        run["responses"].append(
                            {
                                "request": body,
                                "response_path": str(raw_path),
                                "response_sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        )
                        save()
                        print(label, index, json.loads(raw)["choices"][0]["text"], flush=True)
                finally:
                    if server.poll() is None:
                        server.terminate()
                        try:
                            server.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            os.killpg(server.pid, signal.SIGKILL)
                            server.wait()
                    run["server_runner_exit_code"] = server.returncode
                    save()
    finally:
        ADAPTER.write_bytes(backup.read_bytes())
        assert ADAPTER.read_bytes() == selected
        ledger["adapter_restored"] = True
        save()
        backup.unlink()


if __name__ == "__main__":
    main()
