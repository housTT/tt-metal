<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Running the tt-inference-server `benchmarks` workflow against DeepSeek-V4-Flash

This documents the full, wired path to run tt-inference-server's standard `benchmarks`
workflow against the DeepSeek-V4-Flash bring-up on a QuietBox (4× Blackhole = `p300x2`).

## What was wired (so the workflow is runnable)

| Piece | Where | Purpose |
|---|---|---|
| `ModelSpecTemplate` (`DeepSeek-V4-Flash`, `deepseek_v4_impl`, `P300X2`) | `tt-inference-server/workflows/model_spec.py` | passes `run.py` `--model` argparse, resolves the spec for `p300x2`, auto-derives `vllm_args`, and auto-generates `BENCHMARK_CONFIGS` sweep tasks |
| `TTDeepseekV4ForCausalLM` registration | `tt-metal/vllm/plugins/vllm-tt-plugin/.../platform.py` | the TT vLLM plugin prepends `TT` to the HF arch (`DeepseekV4ForCausalLM`) and looks it up in `ModelRegistry` |
| vLLM plugin adapter | `tt-metal/models/demos/deepseek_v4/tt/generator_vllm.py` | `initialize_vllm_model` / `prefill_forward` / `decode_forward` / `allocate_kv_cache` in the plugin's duck-typed contract |
| empty-`override_tt_config` omission | `tt-inference-server/workflows/model_spec.py` | this vLLM build doesn't register `--override-tt-config`; an empty one is now omitted |

## Provisioning (one-time)

The TT vLLM fork is not on PyPI; it is cloned + installed editable. In this environment it is
installed into the tt-metal venv (`/home/ttuser/.tenstorrent-venv`, which already has `ttnn`):

```bash
cd $TT_METAL_HOME
git clone --branch dev https://github.com/tenstorrent/vllm.git vllm     # already present
VENV=/home/ttuser/.tenstorrent-venv
# core vLLM as a no-hardware backend (TT support comes from the plugin); torch pinned to
# keep the ttnn ABI (ttnn needs torch 2.12; vLLM pins 2.10 which would break it).
$VENV/bin/pip install "setuptools>=77"
printf 'torch==2.12.0+cpu\ntorchvision==0.27.0+cpu\ntorchaudio==2.11.0+cpu\n' > /tmp/c.txt
cd vllm && VLLM_TARGET_DEVICE=empty $VENV/bin/pip install -e . --no-build-isolation \
    -c /tmp/c.txt --extra-index-url https://download.pytorch.org/whl/cpu
$VENV/bin/pip install --no-deps -e plugins/vllm-tt-plugin
```

Verify the stack (all should succeed): `import ttnn`, `from vllm import ModelRegistry`,
plugin platform activates (`Platform plugin tt is activated`), and
`TTDeepseekV4ForCausalLM in ModelRegistry.get_supported_archs()`.

## Run the benchmarks workflow

```bash
cd /home/ttuser/code/tt-inference-server
python run.py \
  --workflow benchmarks \
  --model DeepSeek-V4-Flash \
  --tt-device p300x2 \
  --local-server \
  --tt-metal-home /home/ttuser/.local/lib/model-bringup/tt-metal \
  --tt-metal-python-venv-dir /home/ttuser/.tenstorrent-venv \
  --vllm-dir /home/ttuser/.local/lib/model-bringup/tt-metal/vllm \
  --host-hf-cache /home/ttuser/.cache/huggingface \
  --disable-trace-capture \
  --limit-samples-mode smoke-test \
  --service-port 8000
```
(also written to `~/run_deepseek_v4_benchmarks.sh`.)

This starts the local vLLM server, waits for `/health`, then drives `vllm bench serve`
(from the benchmarks venv, upstream `vllm==0.13.0`) and finally the reports workflow.

`--limit-samples-mode smoke-test` runs the bounded `(ISL=16, OSL=4)` workload so the sweep
**completes** at the current served-forward speed (~20 s/token). Drop the flag for the full
ISL/OSL sweep — practical once the on-device served forward lands (see below).

Two fixes were required for the workflow to run here:
- `workflows/model_spec.py`: the `DeepSeek-V4-Flash` `ModelSpecTemplate` `version` must be
  ≥ `0.11.0` (`validate_setup._check_image_version_supported`) — set to `0.14.0`.
- the empty `override_tt_config` is omitted from `vllm_args` (this vLLM build doesn't
  register `--override-tt-config`).

## Performance caveat (important, honest)

The served model currently uses the **correctness-first forward** (per-token weight streaming;
`decode_forward` recomputes over the running context). It produces correct tokens and lets the
server come up and respond, but per-token latency is tens of seconds — so the **full default
sweep will not complete in a practical time**. The perf harness that hits ≥5 tok/s
(`demo/decode_engine.py`, resident + sharded + traced) is measured separately (see `PERF.md`)
and must be folded into `decode_forward` for the server to sustain a full sweep. Until then:

- The workflow **runs**: server bring-up, model registration, request/response, and the report
  pipeline are all exercised.
- For a bounded smoke run that completes, cap the load, e.g. a single short request via the
  server's OpenAI endpoint (`/v1/chat/completions`, `max_tokens` small), or set
  `ONLY_BENCHMARK_TARGETS` / a tiny `perf_reference` target.

See `PRODUCTION_STATUS.md` for the throughput-fold plan (paged KV + continuous batching + the
resident/traced decode path).
