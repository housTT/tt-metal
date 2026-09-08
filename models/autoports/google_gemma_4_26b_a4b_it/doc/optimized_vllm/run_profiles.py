#!/usr/bin/env python3
"""Serialize reproducible serving evidence through the shared readiness runner."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
MODEL = ROOT / "models/autoports/google_gemma_4_26b_a4b_it"
DOC = MODEL / "doc/optimized_vllm"
PYTHON = ROOT / "python_env/bin/python"
HF_MODEL = "google/gemma-4-26B-A4B-it"
PROFILES = {
    "P150": (1, "N150", 50624, {"trace_region_size": 220000000}),
    "P150x2": (
        2,
        "N300",
        262144,
        {
            "trace_region_size": 220000000,
            "fabric_config": "FABRIC_2D",
            "parent_mesh_shape": [2, 2],
            "submesh_offset": [0, 0],
        },
    ),
    "P150x4": (4, "P300x2", 262144, {"trace_region_size": 220000000, "fabric_config": "FABRIC_1D_RING"}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--full-gates", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ)
    forbidden = [key for key in env if ("PROFILER" in key or "TRACY" in key) and env[key] not in ("", "0")]
    if forbidden:
        raise RuntimeError(f"Serving profiler environment is forbidden: {forbidden}")
    env.update(
        HF_HOME="/home/hous/.cache/huggingface",
        HF_HUB_OFFLINE="1",
        TT_METAL_HOME=str(ROOT),
        TT_GEMMA4_TEXT_VER="google_gemma_4_26b_a4b_it_autoport",
        TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}',
    )
    runner = [str(PYTHON), "-m", "models.common.readiness_check.run_vllm_server"]
    for profile in args.profiles:
        tp, mesh, context, config = PROFILES[profile]
        subdir = f"{profile}/optimized_vllm/{args.phase}"
        out = MODEL / "readiness_vllm" / subdir
        out.mkdir(parents=True, exist_ok=True)
        if (out / "run_manifest.json").exists():
            raise FileExistsError(f"Preserve existing evidence: {out}")
        common = [
            "--model-dir",
            str(MODEL),
            "--hf-model",
            HF_MODEL,
            "--max-num-seqs",
            "32",
            "--block-size",
            "64",
            "--max-model-len",
            str(context),
            "--output-subdir",
            subdir,
            "--tt-config",
            json.dumps(config),
        ]
        launch = (
            runner
            + ["--stages", "serve", "--mesh-device", mesh, "--port", "8000"]
            + common
            + [
                "--additional-server-args",
                "--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4",
            ]
        )
        manifest = {
            "profile": profile,
            "phase": args.phase,
            "mesh": mesh,
            "max_model_len": context,
            "max_num_seqs": 32,
            "tt_config": config,
            "sampling_mode": "all",
            "async_scheduling": True,
            "commands": [],
            "started": time.time(),
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "vllm_source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "vllm", text=True
            ).strip(),
            "vllm_worker_sha256": hashlib.sha256(
                (ROOT.parent / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py").read_bytes()
            ).hexdigest(),
            "vllm_benchmark_sha256": hashlib.sha256(
                (ROOT.parent / "vllm/vllm/benchmarks/serve.py").read_bytes()
            ).hexdigest(),
            "fabric_router_sha256": hashlib.sha256(
                (ROOT / "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp").read_bytes()
            ).hexdigest(),
            "source_hashes": {
                name: hashlib.sha256((MODEL / "tt" / name).read_bytes()).hexdigest()
                for name in ("model.py", "generator.py", "generator_vllm.py")
            },
        }
        manifest_path = out / "run_manifest.json"

        def record(command, code=None):
            manifest["commands"].append({"argv": command, "shell": shlex.join(command), "exit_code": code})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        def run(command, log_name):
            print(f"{profile} {args.phase}: {log_name}", flush=True)
            with (out / log_name).open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record(command, result.returncode)
            result.check_returncode()

        record(launch)
        with (out / "launch.log").open("w") as log:
            server = subprocess.Popen(
                launch, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 1200
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server runner exited {server.returncode}: {out / 'launch.log'}")
                    try:
                        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server health timeout")
                    time.sleep(2)
                attach = runner + ["--server-url", "http://localhost:8000"] + common
                # This vLLM version skips its initial endpoint request by
                # default. Explicitly warm both workloads before measurement.
                run(
                    attach + ["--stages", "benchmark", "--additional-benchmark-args", "--num-warmups 1"],
                    "benchmark_runner.log",
                )
                if args.full_gates:
                    tests = MODEL / "tests"
                    endpoint = ["--server-url", "http://localhost:8000", "--model", HF_MODEL]
                    run(
                        [str(PYTHON), str(tests / "run_vllm_feature_checks.py")]
                        + endpoint
                        + [
                            "--expected-max-model-len",
                            str(context),
                            "--output",
                            str(out / "openai_feature_checks.json"),
                        ],
                        "feature_checks.log",
                    )
                    run(
                        [str(PYTHON), str(tests / "run_vllm_async_overlap.py")]
                        + endpoint
                        + ["--output", str(out / "async_overlap_state_test.json")],
                        "async_overlap.log",
                    )
                    baseline = MODEL / "readiness_vllm/standalone_baselines"
                    run(
                        [str(PYTHON), str(tests / "run_vllm_logit_determinism.py")]
                        + endpoint
                        + [
                            "--standalone-baseline",
                            str(baseline / profile / f"logit_oracle_tp{tp}.json"),
                            "--standalone-batch-control",
                            str(baseline / "P150x4/logit_oracle_tp4.json"),
                            "--output",
                            str(out / "logit_determinism.json"),
                        ],
                        "logit_determinism.log",
                    )
                    run(attach + ["--stages", "sampling,qualitative", "--sampling-profile", "full"], "gates_runner.log")
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
                manifest["server_runner_exit_code"] = server.returncode
                manifest["finished"] = time.time()
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{profile} {args.phase}: complete", flush=True)


if __name__ == "__main__":
    main()
