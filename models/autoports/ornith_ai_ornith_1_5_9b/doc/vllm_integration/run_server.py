"""Reproducible stage invocation of the shared serving runner."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--layers")
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--stages", default="serve")
    parser.add_argument("--sampling-profile", default="smoke")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--allow-host-sampling", action="store_true")
    args = parser.parse_args()
    doc = Path(__file__).resolve().parent
    model = doc.parents[1]
    repo = model.parents[2]
    tt_config = {
        "trace_region_size": 100000000,
        "l1_small_size": 32768,
        "fabric_config": "FABRIC_1D_RING",
        "fabric_max_packet_payload_size_bytes": 8192,
    }
    extra = [
        "--hf-overrides",
        json.dumps({"architectures": ["OrnithForCausalLM"]}),
        "--async-scheduling" if args.async_scheduling else "--no-async-scheduling",
    ]
    cmd = [
        sys.executable,
        "-m",
        "models.common.readiness_check.run_vllm_server",
        "--stages",
        args.stages,
        "--model-dir",
        str(model),
        "--hf-model",
        "/home/hous/dev/ornith-1.5-9b/upstream",
        "--mesh-device",
        "P150x4",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        "262144",
        "--sampling-profile",
        args.sampling_profile,
        "--tt-config",
        json.dumps(tt_config),
        "--additional-server-args",
        shlex.join(extra),
    ]
    env = dict(
        os.environ,
        USER="hous",
        PYTHONUNBUFFERED="1",
        ORNITH_VLLM_ALLOW_HOST_SAMPLING="1" if args.allow_host_sampling else "0",
    )
    env.pop("ORNITH_VLLM_LAYER_INDICES", None)
    if args.layers:
        env["ORNITH_VLLM_LAYER_INDICES"] = args.layers
    for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER"):
        if env.get(key):
            raise RuntimeError(f"Serving run must not inherit {key}")
    manifest = {
        "argv": cmd,
        "command": shlex.join(cmd),
        "layers": args.layers,
        "host_sampling_compatibility": args.allow_host_sampling,
        "runtime_environment": {
            key: env.get(key)
            for key in (
                "USER",
                "ORNITH_VLLM_LAYER_INDICES",
                "ORNITH_VLLM_ALLOW_HOST_SAMPLING",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
            )
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                model / "tt/generator_vllm.py",
                model / "tt/generator.py",
                model / "tt/model.py",
                repo / "models/common/readiness_check/run_vllm_server.py",
                repo / "models/common/sampling/tt_penalties.py",
                repo / "models/common/sampling/tt_sampling.py",
                repo.parent / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py",
                repo.parent / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py",
                repo.parent / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py",
            )
        },
    }
    with (doc / f"{args.label}.runner.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest["runner_pid"] = proc.pid
        (doc / f"{args.label}.command.json").write_text(json.dumps(manifest, indent=2) + "\n")

        def stop(signum, frame):
            proc.send_signal(signum)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        result = proc.wait()
    manifest["returncode"] = result
    (doc / f"{args.label}.command.json").write_text(json.dumps(manifest, indent=2) + "\n")
    server_log = model / "readiness_vllm/server.log"
    if server_log.exists():
        shutil.copyfile(server_log, doc / f"{args.label}.server.log")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
