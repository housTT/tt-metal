# vLLM integration work log (resumed 2026 Oct 23)

The 2026 Sep 08 stage log is `work_log_20260908.md`; its README is
`README_20260908.md`. This log covers the resume under the current
`tt-model-bringup` skill version (skills commit `62814d62`), on one Blackhole
P300 chip as the P150 profile.

## Environment and serving pair

- tt-metal worktree `/home/hous/dev/tt-metal-wt-prefill`, branch
  `hous/gemma4-prefill-features`, start commit `4f3c2b13fa`.
- Serving pair: `/home/hous/dev/vllm` at `f96662a7d239c077e1435b658c0e7d316f9c4799`
  (`vllm 0.1.dev14202+gf96662a7d`) with its in-tree plugin
  `plugins/vllm-tt-plugin` installed editable; working tree clean. The pair
  was fixed by the run environment. The plugin checkout has no
  `docs/MODEL_INTERFACE.md`; the adapter contract was derived from
  `platform.py`, `model_runner.py`, `worker.py`, and `tests/tt`. Local plugin
  and vLLM commits relative to upstream are recorded in `plugin_changes.md`.
- Devices: chip 0 serves; chip 1 runs generator-level checks; chips 2 and 3
  unused. Every device run sets `TT_MESH_GRAPH_DESC_PATH` to the p150 mesh
  graph descriptor (without it tt-metal fails with "Custom fabric mesh graph
  descriptor path must be specified").
- The packaged runner's `serve` stage rejects the nested plugin layout, so
  every readiness stage attached to the running server with `--server-url`
  and `--server-log`.

## Server launch (chip 0)

```bash
cd /home/hous/dev/tt-metal-wt-prefill
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TT_METAL_HOME=/home/hous/dev/tt-metal \
    PYTHONPATH=/home/hous/dev/tt-metal-wt-prefill:/home/hous/dev/tt-metal \
    TT_VISIBLE_DEVICES=0 \
    TT_MESH_GRAPH_DESC_PATH=/home/hous/dev/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport \
    GEMMA4_OPT_PREFILL_SDPA_Q_CHUNK=128 GEMMA4_OPT_PREFILL_SDPA_K_CHUNK=128 \
    TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
    VLLM_TT_PLUGIN_ROOT=/home/hous/dev/vllm/plugins/vllm-tt-plugin \
    /home/hous/dev/tt-metal/python_env/bin/python -m models.common.readiness_check.run_vllm_server \
    --stages serve --model-dir models/autoports/google_gemma_4_26b_a4b_it \
    --hf-model google/gemma-4-26B-A4B-it --mesh-device N150 --max-num-seqs 32 \
    --block-size 64 --max-model-len 131072 --port 8000 \
    --output-subdir P150_prefill_features --tt-config '{"trace_region_size":220000000}' \
    --additional-server-args '--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --enable-prefix-caching'
```

Registration: the engine log records
`Getting max_tokens_all_users=94528 ... from generator '<class 'models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm.Gemma4ForCausalLM'>'`
(selected by `TT_GEMMA4_TEXT_VER`), `Chunked prefill is enabled with
max_num_batched_tokens=2048`, `Automatic prefix caching is enabled`, KV pool
1,509 blocks (`Maximum concurrency for 131,072 tokens per request: 1.19x`).

## P150 serving context

`doc/context_contract.json` now records `current_supported_context` 131072 for
the P150 profile with `limiting_reason` `device_dram`. The earlier 50,624 came
from a conservative construction-time projection; the measured post-load DRAM
(5.42 GiB free of 31.63 GiB) and the hybrid KV manager's per-group block
accounting give a 1,509-block pool, enough for one 131,072-token request
(1,109 blocks) but not for a 262,144-token request (2,133 blocks). The
superseded 2026-09-08 P150 artifacts that recorded 50,624 are under
`readiness_vllm/superseded_20260908/` (`INDEX.json` lists originals and
hashes; files carrying the old cap are stored as `.json.xz`). Two stage-10
files in `doc/optimized_vllm/` were compressed the same way and the reason is
noted in that README.

## Gate state at resume

The stage gate exited 2 at resume: `check_context_contract` flagged the
50,624 manifests against the 131,072 contract, `doc/full_model/serving_contract.json`
did not exist, and no `readiness/prefill_resume_check.json`,
`readiness_vllm/vllm_run_summary.json`, `prefix_cache_check.json`, or
`vllm_prefill_burst_benchmark.json` existed.
