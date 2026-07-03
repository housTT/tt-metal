<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash — Production / tt-inference-server integration status

Honest status of what is ready to wire into tt-inference-server's evals + benchmarks, and what
remains. No overclaiming: "production ready" here means **correct, servable inference + the
standard perf/spec surfaces**, with the throughput-serving path explicitly scoped as remaining.

## Ready now

| Piece | Artifact | State |
|---|---|---|
| Correct inference on real weights | `tt/generator.py::DeepSeekV4Generator` | ✅ generates correct tokens (`"The capital of France is" → " Paris."`); PCC ≥ 0.99 vs HF per-module + e2e |
| Text-in/text-out + sampling + EOS | `DeepSeekV4Generator.generate(...)` | ✅ greedy + temperature/top-p; drivable **offline for accuracy evals** (lm-eval etc.) |
| vLLM adapter surface | `tt/generator_vllm.py::DeepseekV4ForCausalLM` | ✅ `initialize_vllm_model` / `prefill_forward` / `decode_forward` in the production shape (mirrors deepseek_v3) |
| tt-inference-server model spec | `tt_inference_server_spec.json` (device `P300X2`) | ✅ entry to merge into `release_model_spec.json` |
| Standard perf benchmark | `tests/test_perf.py` (`models_performance_bare_metal`) + `PERF.md` | ✅ passes ≥5 t/s/u, <5 s TTFT; canonical `prep_perf_report` CSV + `BenchmarkData` |
| Per-module + e2e correctness | `tests/test_*_pcc.py`, `tests/test_real_e2e.py`, `tests/test_real_weights.py` | ✅ all PCC ≥ 0.99 on Blackhole |

## Remaining for full production serving (throughput path)

The current inference path is **correctness-first**: it re-runs the sequence per token and
transfers MoE weights per token, so single-sequence latency is the slow path (~tens of s/token
on real weights) even though the *optimized* op structure is measured at 5.6–6.2 t/s/u
(`decode_engine.py`, `PERF.md`). To make serving fast enough for large eval sweeps:

1. **Fold the resident+sharded+traced decode engine onto the real weights** — unify
   `demo/decode_engine.py` (fast, mesh-resident, traced) with `tt/generator.py` (correct, real
   weights). This is the single biggest item.
2. **Paged KV-cache attention + continuous batching** — implement `allocate_kv_cache` and the
   paged decode so the vLLM `Generator` base (`models/tt_transformers/tt/generator.py`) can
   batch users. Reuse `models/demos/deepseek_v3_d_p/tt/moe/` EP dispatch/combine for the
   256-expert routing across chips, and paged SDPA for the sliding + compressed KV.
3. **Chunked prefill** — TTFT exceeds 5 s beyond ~2–3K context without it.
4. **vLLM install + registration** — the tt-metal vLLM plugin (`tenstorrent/vllm`) is not
   installed in this env; install it and register `DeepseekV4ForCausalLM` in the plugin's model
   map so `initialize_vllm_model` is invoked by the server.

## How to drive it from tt-inference-server

**1. Register the model spec** (merge `tt_inference_server_spec.json` into the server's registry):
```bash
python - <<'PY'
import json
reg = "/home/ttuser/code/tt-inference-server/release_model_spec.json"
add = json.load(open("models/demos/deepseek_v4/tt_inference_server_spec.json"))["model_specs"]
d = json.load(open(reg)); d["model_specs"].update(add); json.dump(d, open(reg,"w"), indent=2)
print("registered deepseek-ai/DeepSeek-V4-Flash @ P300X2")
PY
```

**2. Accuracy evals now (offline, single-sequence — no vLLM needed):** drive `DeepSeekV4Generator`
directly. Example lm-eval-style loop:
```python
from models.demos.deepseek_v4.tt.generator import DeepSeekV4Generator
import ttnn
dev = ttnn.CreateDevice(device_id=0)
gen = DeepSeekV4Generator(dev, num_layers=43)
for prompt in eval_prompts:
    completion = gen.generate(prompt, max_new_tokens=64, temperature=0.0)["completion"]
    # score completion against the eval's reference
```

**3. Benchmarks now:** the standard tt-metal perf harness passes and emits the canonical CSV:
```bash
pytest models/demos/deepseek_v4/tests/test_perf.py -m models_performance_bare_metal -s
```

**4. Full server (after the throughput items above + vLLM install):** once
`DeepseekV4ForCausalLM` is registered in the tt-metal vLLM plugin, run via the server's `run.py`
with the `P300X2` spec — evals/benchmarks then flow through the standard tt-inference-server
workflow.
