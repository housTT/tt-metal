# Qwen3.8-Flash-Next TTI release handoff

## Release status

- Current state: corrected resume-14 `ci-nightly` release is running; this section is
  replaced with the terminal exit, report path, row classifications, and final
  readiness status after aggregation.
- Intended terminal classification: `release-readiness-ci-subset-pass` only if
  every sampled mandatory eval, API/spec test, and benchmark row passes.
- Unrestricted full-set readiness is not claimed by this Stage 11 run.

## Model and implementation

- HF model: `Qwen/Qwen3.8-Flash-Next`, checkpoint revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Autoport implementation check: the source spec, TTI-generated runtime spec,
  and live server all point to
  `models/autoports/qwen_qwen3_8_flash_next`; the server imports
  `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm.Qwen4ExpForConditionalGeneration`.
- No evaluated artifact selects `models/tt_transformers`, `models/demos`, or
  another packaged implementation.
- Supported/served context: 262,144 tokens, matching `doc/context_contract.json`.
  Eval and benchmark requests were not capped or aligned to internal chunk,
  tile, block, page, or trace sizes.
- Prompt mode: structured chat. Meta's model-independent task data contains an
  outer Llama wrapper; the TTI adapter removes only that exact wrapper and
  passes `--apply_chat_template`, so the live server applies Qwen's checkpoint
  chat template. The template hash and shared-suite evidence are recorded in
  `qualitative_release_summary.json`.
- Reasoning parser: `qwen3`. Hidden reasoning is not scored. A TTI-owned,
  version/source-guarded lm-eval 0.4.4 patch maps null final content to an
  unanswered empty string; ordinary final content is unchanged.

## Server topology

- Reservation host: `tt-quietbox`; server mode: existing external autoport
  OpenAI-compatible vLLM server; port 8021.
- Launch control: direct bounded process session; no server tmux session.
- Docker: not used. Docker image/version: N/A.
- Device placement: `TT_VISIBLE_DEVICES=0,1`, P300 1x2 mesh, explicit P300 mesh
  graph descriptor, `FABRIC_1D`, strict fabric initialization, 8192-byte fabric
  packets.
- vLLM settings: `max_num_seqs=2`, block size 64, max model length 262,144,
  async scheduling, on-device sampling, decode-only trace, 1 GiB trace region,
  24,576-byte small L1, served model name `Qwen/Qwen3.8-Flash-Next`, qwen3
  reasoning parser.
- Cold start for the resume-14 server: approximately 6m04s to health.

Server command:

```text
env TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 \
  TT_MESH_GRAPH_DESC_PATH=/home/ttuser/dev/qwen3.8-flash-next/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
  VLLM_PLUGINS=tt,tt_model_registry QWEN38_VLLM_LOG_METRICS=1 \
  QWEN38_HOST_MISS_WAVE_POLICY=serial QWEN38_HOST_STAGING_DEPTH=1 \
  PYTHONPATH=/home/ttuser/dev/vllm-tt-plugin/src:/home/ttuser/dev/qwen3.8-flash-next/tt-metal:/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages \
  python_env/bin/python /home/ttuser/dev/muse-glimmer/tt-metal/models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --output-dir models/autoports/qwen_qwen3_8_flash_next/doc/tti_release/server \
  --stages serve --mesh-device P300 --port 8021 --max-num-seqs 2 \
  --sampling-profile full --block-size 64 --max-model-len 262144 \
  --tt-config '{"trace_region_size":1073741824,"fabric_config":"FABRIC_1D","fabric_packet_payload_bytes":8192,"l1_small_size":24576,"trace_mode":"decode_only"}' \
  '--additional-server-args=--async-scheduling --served-model-name Qwen/Qwen3.8-Flash-Next --reasoning-parser qwen3'
```

## TTI checkout and commands

- Checkout:
  `/home/ttuser/dev/qwen3.8-flash-next/tti-release/qwen3_8_flash_next/tt-inference-server`.
- Repo tag: `v0.20.0`; official tag SHA
  `6ab1de736f303b899f84ae07d184d33f9889946e`.
- TTI Git HEAD at launch: `d7713b7af9b1b0bf4be1f21666add4d14c556f36`
  plus the reviewed null-content, API-timeout, and prompt-mode regression
  diff; the final local checkpoint SHA is recorded after `stage-review`
  clean-pass. The exact three-file launch-time diff SHA256 is
  `457e23c3672b046862555a3dda229eec4af48c8870291508c049a85f97a27e16`.
- tt-metal Git HEAD at launch:
  `60f1562e8ecf709bd778cb86c32dbfa5a06f8cb1`.
- vLLM TT plugin SHA: `a48857ac68b17c31303e4809f348caaebbf10f74`.
- Matching checkout CLI was verified with that checkout's `python3 run.py --help`.

Smoke command (spec supplies the server mode, port, context, and trace flag):

```text
timeout 600 python3 run.py --workflow benchmarks \
  --runtime-model-spec-json /home/ttuser/dev/qwen3.8-flash-next/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/tti_release/specs/smoke_runtime_model_spec.json
```

Release command (spec supplies `workflow=release`, `docker_server=false`,
`local_server=false`, port 8021, and `limit_samples_mode=ci-nightly`):

```text
env ONLY_BENCHMARK_TARGETS=1 timeout 108000 python3 run.py --workflow release \
  --runtime-model-spec-json /home/ttuser/dev/qwen3.8-flash-next/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/tti_release/specs/release_runtime_model_spec.json
```

No token, `.env`, authentication secret, cache, model weight, Docker layer,
persistent TT cache, or large raw eval dump is copied into this handoff.

## Smoke result

- Health: HTTP 200.
- Bounded OpenAI-compatible chat request: HTTP 200, `finish_reason=stop`, two
  completion tokens; only length/hash metadata is retained.
- TTI no-Docker benchmark repeated against the resume-14 server: logical
  8-token input / 8-token output, one request, trace capture disabled,
  `completed=1`, `failed=0`, TTFT 1265.57 ms, mean TPOT 220.61 ms, output
  throughput 2.85 token/s.
- Generated smoke runtime spec proves `docker_server=false`,
  `local_server=false`, port 8021, workflow `benchmarks`, full 262,144 context,
  and implementation path `models/autoports/qwen_qwen3_8_flash_next`.

## CI-subset choice and runtime estimate

- Mode: `ci-nightly`, explicitly 10 `meta_ifeval` samples and 5
  `meta_gpqa_cot` samples. Every selected request keeps the full context and its
  original generation allowance (4096 and 32,768 tokens respectively).
- Measured IFEval cost was about 54 minutes for 10 samples. The first serial
  full-budget GPQA sample took 2h37m. The corrected harness runs two requests
  concurrently, but the host-backed server has one physical decode lane and a
  live two-request aggregate sample measured about 2.6 token/s. A conservative
  all-budget subset projection is therefore roughly 20 hours, not 8h45m.
- Unrestricted projection from those measurements is about 48.7 hours for 541
  IFEval rows plus 586.1 hours for 448 GPQA rows at concurrency two: roughly
  634.8 hours / 26.5 days. This is prohibitive for the reservation window, so
  Stage 11 uses the permitted nightly-equivalent subset.
- Every accuracy number in the terminal report is a CI-subset result, not a
  full-set accuracy claim.

## Host-backed weights

- Contract: `doc/host_weight_contract.json`; release manifest:
  `host_weight_release_manifest.json`.
- Expert store: all 24,576 layer/expert entries are retained in exact packed
  BFP4 tile form; 68,080,435,200 host bytes; 10 fixed slots per layer/rank;
  1,459,814,400 device bytes per rank for the full stack; serial miss waves,
  staging depth 1; exact owner H2D and non-owner zero D2D paths.
- PLE store: 128 checkpoint shards, 2,500,012 rows per shard,
  102,400,491,520 host bytes; n-gram size 3; 8192-row host cache;
  819,200 device staging bytes per rank; BF16 tile staging. Exact EOS/history,
  reset, prefill-to-decode, and request-isolation evidence is linked by the
  manifest.
- Manifest provenance: checkpoint revision above; checkpoint-index SHA256
  `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`;
  all checkpoint/shard and packing/lookup source hashes are enumerated in the
  manifest. No weight is copied into release evidence.
- Host assumptions: 170,480,926,720 required RAM bytes; safetensors shards
  remain mmap-backed; selected PLE rows only; pinned host staging is unavailable
  in the installed CPU Torch and contiguous CPU rows feed TT runtime staging;
  direct owner-rank PCIe H2D on one P300 board. Measured cold preload is
  240.979s.

## Hardware recovery

- The earlier implicit-placement server opened all four local devices and
  reproduced a device-2 remote-Ethernet strict-init timeout after two bounded
  reset sequences and a physical QuietBox reboot. The release contract requires
  dies 0–1; setting its already-declared `TT_VISIBLE_DEVICES=0,1` restricted
  topology discovery to the intended P300 board and cleared the failure.
- Before resume-13 serving, bounded `tt-smi list/reset/list` again showed all
  four devices. The exact 1x2 strict-Fabric configuration opened and closed
  successfully on devices 0–1 before the durable server launch.
- This is classified as recovered infrastructure/placement, not a model result
  or reduced topology.
- An initial resume-13 release client used a 12-hour outer wrapper. After live
  concurrency-two telemetry showed that five full 32K GPQA responses could
  legitimately exceed that wrapper, only the TTI client was stopped; its
  incomplete output directory was archived, the server returned to zero active
  requests, and the identical release was restarted with a bounded 30-hour
  wrapper. No task, sample, context, prompt, generation budget, or score changed.
- That rerun exposed a separate TTI/lm-eval 0.4.4 harness defect: its generic
  API adapter accepted but discarded `timeout=7200`, then aiohttp's 300-second
  cumulative timer also charged requests while they waited for the sole
  connector slot. Three retries caused a deterministic approximately
  15-minute abort while the server remained healthy. AutoFix added a pinned,
  source-guarded compatibility patch that retains the configured timeout and
  gates each attempt before starting its HTTP timer. The real installed venv
  probe passed and the focused TTI suite passes (`144 passed`); context,
  generation budgets, concurrency, and sample counts are unchanged. Source-only
  reports are in `autofix_meta_timeout/`.
- Concurrent work outside Stage 11 changed the autoport to an uncommitted TP4
  resident-expert implementation after the resume-13 release process ended.
  That path contradicts this handoff's immutable TP2/EP2 host-weight contract.
  The exact concurrent edits are preserved recoverably in Git stash object
  `033206aade35273dd5ce3c478b86c85b445c2d28`; this release runs the committed,
  already-optimized autoport at `60f1562e8ecf709bd778cb86c32dbfa5a06f8cb1`.
  The stash is reapplied after release cleanup so no concurrent work is lost or
  mixed into this release evidence.

## Copied report and row classification

- Final report: pending current release aggregation.
- Release workflow exit: pending.
- Failed or missing rows: pending terminal report review.
- Stage review: pending after checks and cleanup.
- Local checkpoint commits: pending clean-pass; never pushed.
