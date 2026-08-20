# TTI release — ornith-ai/Ornith-1.0-35B (generated TTNN autoport)

Stage 11 handoff notes for the `tt-inference-server` release workflow run against the generated
autoport at `models/autoports/ornith_ai_ornith_1_0_35b`.

**Final status: `release-readiness-ci-subset-pass`.**
`run.py --workflow release` EXIT_CODE=0 (the release workflow completed successfully), the release report's Acceptance Criteria is **PASS with 0
blockers**, and every required sampled row passed or is recorded below with its classification.
Because the evals ran under `--limit-samples-mode ci-nightly`, **every accuracy number in this stage is
a CI-subset result, not full-set accuracy**, and is not comparable to a full-set release threshold.

Three qualifiers belong with that headline, not buried below:

1. **Every gradable row in the report is `NA`.** 20/20 benchmark rows are ungraded (no `perf_reference`
   for this model) and 2/2 eval rows have no reference score to compare against, and `EXPERIMENTAL`
   status disables eval and benchmark enforcement anyway. The PASS proves the workflow ran end to end
   and nothing errored; it is not a quality gate (§11).
2. **The shipped configuration has no API-conformance coverage.** `Spec Tests` is `NA`. The suite was
   run separately and 15 of 22 rows failed; 13 are proven harness incompatibility with a
   reasoning-parser server, 2 trace to a real pre-existing model defect. They are **disclosed, not
   waived** (§12).
3. **One carried-forward readiness gap was confirmed here:** repeated identical long generations are not
   byte-reproducible on this server (§12, Class C). Pre-existing, from the vLLM-integration stage.

---

## 1. Topology and server mode

| | |
|---|---|
| server mode | **external / API** (`server_mode: API` in the report). No `--docker-server`, no `--local-server`, no Docker container at any point. |
| host / context | `tt-quietbox-part-2`, the reservation context that owns the tt-metal checkout. There is no separate physical loudbox host in this topology; Docker was never needed, so the Docker fallback was not used. |
| device | 4x Blackhole `p300c` chips, `1x4` mesh under `FABRIC_1D_RING`. TTI device name **`p300x2`** (`DeviceTypes.P300X2` = "BH QuietBox 2", 2x P300 cards = 4 chips, `MESH_DEVICE` `(1, 4)`), which is the TTI label for exactly this hardware. |
| inference server | the autoport's own vLLM server, started from the tt-metal checkout by `models.common.readiness_check.run_vllm_server`, on port **8100**. TTI ran purely as an HTTP client against it. |
| tt-metal commit | `278e0ba` (`278e0ba8a50`) — the checkout HEAD when the TTI spec was written. Note: branch history was later rewritten, so `278e0ba` is no longer an ancestor of this branch and the `code_link` in the report does not resolve from it. The autoport implementation at `278e0ba` is identical to the delivered tree apart from the two regression tests added by §6. |
| vLLM checkout | `/home/ttuser/dev/ornith/vllm` at `5380fd4`, **plus this stage's uncommitted plugin fix** (§6) |

## 2. tt-inference-server version

| | |
|---|---|
| repo | `https://github.com/tenstorrent/tt-inference-server` |
| tag | **`v0.20.0`** |
| SHA | `6ab1de736f303b899f84ae07d184d33f9889946e` |
| work root | `/home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/tt-inference-server` |
| Docker image | **none — Docker was not used** |

`v0.20.0` was selected on evidence, not assumption: it is the newest release tag, and it contains
`6e396b43` ("Support external runtime specs in release workflows", PR #4345) — verified with
`git merge-base --is-ancestor 6e396b4 HEAD` — which is the merge the `$tti-release` skill requires for a
custom `--runtime-model-spec-json` not to be re-validated against the built-in `MODEL_SPECS`. The
checkout's own `python3 run.py --help` was used for every flag spelling: this tag uses `--tt-device`
(not `--device`) and `--runtime-model-spec-json` (not `--model-spec-json`).

## 3. Release integration — how TTI was pointed at the autoport

`v0.20.0` has no built-in entry for this model, so three **local, additive** edits were made to the TTI
checkout (full diff committed as `tti_local_registration.diff`, 201 insertions, 0 deletions):

1. `workflows/model_spec.py` — new `ornith_1_0_35b_autoport` `ImplSpec` with
   `code_path = models/autoports/ornith_ai_ornith_1_0_35b`.
2. `workflows/model_specs/prod/llm.yaml` — a `ProdModelSpecTemplate` for `ornith-ai/Ornith-1.0-35B` on
   `P300X2`: `max_context 262144`, `max_tokens_all_users_override 264192`, `max_concurrency 32`,
   `status EXPERIMENTAL`, `eval_max_retries 1`, and the `vllm_args`/`env_vars`/`override_tt_config` that
   the external server actually runs with. `264192` is not a guess: it is the
   `GPU KV cache size: 264,192 tokens` this run's server logged (block_size 64 x 4128 blocks).
3. `reference_config/evals/eval_config.py` — an `EvalConfig` for `ornith-ai/Ornith-1.0-35B` (§5).

Registering in the catalog is what makes `EVAL_CONFIGS` resolvable, because `EVAL_CONFIGS` is built as
`{model_spec.model_name: ...  for model_spec in MODEL_SPECS.values()}` and `validate_setup` asserts
`model_spec.model_name in EVAL_CONFIGS` for `--workflow release`. Benchmarks need no registry entry —
`build_benchmark_config` derives the sweep from the resolved spec.

A custom runtime spec was **also** exported and passed with `--runtime-model-spec-json`, with its
embedded `cli_args` already correct so the JSON is right on its own and does not depend on CLI flags to
override it (generator: `make_autoport_spec.py`, copied as `report/ornith_autoport_release_spec.json`):

```
cli_args.docker_server = false      cli_args.local_server = false
cli_args.service_port  = 8100       cli_args.workflow     = release
cli_args.runtime_model_spec_json = <this file>
```

### Autoport implementation check — target `models/autoports/ornith_ai_ornith_1_0_35b`

Target path: **`models/autoports/ornith_ai_ornith_1_0_35b`**. The TTI-written run spec
(`report/runtime_model_spec_2026-08-19_20-19-16_*.json`) **matches it**:

```
impl.code_path      models/autoports/ornith_ai_ornith_1_0_35b
impl.impl_name      ornith-1-0-35b-autoport
model_id            id_ornith-1-0-35b-autoport_Ornith-1.0-35B_p300x2
hf_model_repo       ornith-ai/Ornith-1.0-35B
code_link           https://github.com/tenstorrent/tt-metal/tree/278e0ba/models/autoports/ornith_ai_ornith_1_0_35b
runtime_config      docker_server=False local_server=False service_port=8100 workflow=release
```

The release report's metadata carries `"model_impl": "ornith-1-0-35b-autoport"`. No copied report, run
spec, benchmark JSON or eval result mentions `models/tt_transformers`, `models/demos`, or any other
packaged implementation — verified by grep over the whole copy-back.

The server side is the same code: it was launched with `--model-dir models/autoports/ornith_ai_ornith_1_0_35b`
and its log shows `models.autoports.ornith_ai_ornith_1_0_35b.tt.model` /
`...tt.generator_vllm` doing the weight load, prefill and decode.

## 4. Context contract

`doc/context_contract.json` advertises and supports **262144**. Nothing in this stage lowered it:

| where | value |
|---|---|
| `doc/context_contract.json` supported | 262144 |
| server `--max-model-len` | 262144 (`/v1/models` reports `max_model_len: 262144`) |
| TTI spec `device_model_spec.max_context` | 262144 |
| TTI spec `vllm_args.max_model_len` | 262144 |
| eval `model_kwargs.max_length` (both tasks) | 262144 |
| benchmark sweep | full standard sweep, ISL up to **131072**, all 20 points ran |

No request was shortened, aligned or bucketed to make anything pass. The only per-task caps are eval
*generation* budgets (`max_gen_toks` 8192 / 32768), which are generation budgets inside a 262144-token
context, matching what the other Qwen-family reasoning models in TTI use.

One advisory note, so it is not misread: `.agents/scripts/check_context_contract.py` reports an
`ADVISORY CONTEXT CAP` for `doc/tti_release/repro/hang_scheduler_sim.py:133`
(`max_model_len=8192`). That file is a **host-only scheduler simulator** with a fake two-layer config —
it never opens a device and never serves a request. It is not a served context and does not cap
anything; the same advisory class already fires for every earlier stage's reduced-target probe log.

**Non-aligned prompt lengths.** TTI's `vllm bench serve --dataset-name random` emits lengths that are not
multiples of the autoport's internal 128 alignment. Those lengths were served, not waived: the release
sweep and the fix-verification run both show prefills at e.g. **16448, 16450, 16511** completing normally.
The one failure at a non-aligned length was a real autoport bug and was **fixed**, not aligned around — §6.

## 5. Evals — task selection and the meta_ifeval / meta_gpqa_cot decision

Tasks configured: **`ifeval`** and **`r1_gpqa_diamond`**, both via the chat API.

`meta_ifeval` and `meta_gpqa_cot` are **not applicable to this model and were not run**. This is not a
waiver of a failure — those two tasks cannot be constructed for a non-Llama checkpoint:

* `workflows/workflow_venvs.py::setup_evals_meta` derives the dataset as `f"{hf_model_repo}-evals"`, i.e.
  it would look for a nonexistent `ornith-ai/Ornith-1.0-35B-evals` Hugging Face repo (the real ones are
  `meta-llama/Llama-3.1-*-Instruct-evals`), and their prompts are pre-rendered in the Llama-3 chat format
  and consumed with `apply_chat_template=False`.
* In `tt-inference-server` v0.20.0 **every** model configured with `meta_ifeval`/`meta_gpqa_cot`/`meta_gpqa`
  is a `meta-llama/*` checkpoint, and **every** non-Llama LLM uses `ifeval`/`leaderboard_ifeval` plus a
  GPQA variant. `ifeval` and `r1_gpqa_diamond` are this model's equivalents of those two gates.
* `r1_gpqa_diamond` (not `gpqa_diamond_generative_n_shot`) is the right GPQA variant for a thinking model
  — it is what Qwen3-*, Kimi-K2.*, MiniMax-M* and gemma-4-* use in this checkout, and the n-shot variant's
  bare "(C)" exemplars suppress reasoning.

Results (**CI subset**, `--limit-samples-mode ci-nightly` → `--limit 0.05`):

| task | samples | full set | score | reference | accuracy check | classification |
|---|---|---|---|---|---|---|
| `ifeval` | 28 | 541 | **82.08** (mean of prompt/inst x strict/loose) | none | 🟨 N/A | **no-reference — measured only** |
| `r1_gpqa_diamond` | 10 | 198 | **50.00** (`exact_match,none`, stderr 16.7) | none | 🟨 N/A | **no-reference — measured only** |

Both rows render `N/A` rather than PASS/FAIL because **no reference score exists to compare against**:
the Ornith-1.0-35B model card publishes only agentic-coding benchmarks (Terminal-Bench 2.1, SWE-bench
Verified/Pro/Multilingual, NL2Repo, Claw-eval, SWE Atlas) — no IFEval and no GPQA — and no GPU reference
run of these two tasks exists for this checkpoint. `published_score` and `gpu_reference_score` were
deliberately left `None` rather than invented; `gpu_reference_score_ref` records why in the report
itself. These are **not** `readiness-fail`: nothing failed, there is simply nothing to compare to. An
N/A check is not treated as a pass either — the numbers are published as measured values only.
Establishing a GPU reference for these two tasks is the follow-up that would turn them into real gates.

Sanity on the absolute values: 82.08 IFEval and 50.0 GPQA-Diamond are in the expected band for a 35B
Qwen3.5-derived reasoning model, so neither looks like a broken forward pass. The `r1_gpqa_diamond`
subset is 10 samples with a 16.7-point standard error. An earlier release attempt on the same code and
the same 10-sample subset scored **60.0** — committed as `evals/prior_run_same_subset/` so the spread is
checkable, not just asserted. That run was abandoned mid-benchmark by the §6 hang, so it is not the
shipped result; its eval half had already completed and is kept only as this variance evidence (its
`ifeval` on the same 28 samples was 77.97 against the shipped 82.08).
**Do not read a 10-sample CI-subset GPQA number as an accuracy measurement.**

### Why ci-nightly, and the projected unrestricted runtime

Projected from this stage's own measured wall-clock (`ifeval` 28 samples in 19.6 min → 0.70 min/sample;
`r1_gpqa_diamond` 10 samples in 79.1 min → 7.91 min/sample):

| task | full set | per-sample extrapolation | batch-aware extrapolation |
|---|---|---|---|
| `ifeval` | 541 | ~6.3 h | ~6.3 h (28 ran as ~1 batch at `num_concurrent=32`, so ~19 batches x 19.6 min) |
| `r1_gpqa_diamond` | 198 | ~26.1 h | **~9 h** (the 10 ran as one concurrent batch in 79 min; 198 at `num_concurrent=32` is ~7 batches, and a full 32-wide batch is slower than a 10-wide one) |
| **total** | | ~32 h | **~15 h** |

The per-sample column is the naive extrapolation and overstates GPQA, because those 10 samples ran
concurrently rather than serially. The batch-aware column is the honest one: **~15 h**. Both are far
past this stage's window, so the conclusion is unchanged, but the number quoted should be ~15 h.

~15 h of evals alone is prohibitive for this stage's window, so `--limit-samples-mode ci-nightly` was
used. It reduces **eval sampling only**: nothing else was reduced — the context stayed at 262144, the
full 20-point benchmark sweep ran to ISL 131072, and no request was shortened. Effective limits:
`--limit 0.05` for both tasks → 28/541 and 10/198 samples. Verified in the written eval command, not
assumed. Hence `release-readiness-ci-subset-pass`, never `release-readiness-pass`.

### Prompt format / tokenizer decision (`$qualitative-check`)

Ornith-1.0-35B is an **instruct + reasoning** checkpoint: `Qwen2Tokenizer`, `chat_template_present: true`,
and its `chat_template.jinja` pre-fills `<think>\n` into the generation prompt. Both eval tasks therefore
set `use_chat_api=True` so the **server** applies the checkpoint's own chat template
(`eval_class` resolves to `local-chat-completions`, and the written command shows
`base_url=http://127.0.0.1:8100/v1/chat/completions --apply_chat_template`). The server runs
`--reasoning-parser qwen3`, the model card's own recipe, so the `<think>` trace lands in the response's
`reasoning` field and `message.content` holds the final answer that lm-eval scores. No raw-completion
prompting of this instruct checkpoint was used as release evidence, and no chat prompt was invented.

Rendered-prompt evidence, captured live against this server
(`../optimized_vllm` holds the earlier chat-rendered qualitative control):

```
POST /v1/chat/completions  {"messages":[{"role":"user","content":"Reply in one short sentence."}],
                            "max_tokens":2048,"temperature":0}
-> finish_reason "stop", usage 16 prompt / 818 completion
-> message.reasoning = 3019 chars of <think> trace
-> message.content   = "\n\nConsider it done."
```

## 6. The one real model bug this stage found, and its fix

**Symptom.** During the release benchmark sweep point `isl=16384 osl=128 concurrency=16`, the server
deadlocked at request 30/32: `VLLM::EngineCore` spun forever in `_process_engine_step` with
`model_executed=False` while the scheduler still held one RUNNING request, and **all four devices were
idle** (`tools/tt-triage.py`, `triage/tt-triage.txt.gz`: `dump_op_mesh` all idle, `dump_running_operations` pass, `check_eth_status`
pass, `check_arc` pass). Evidence: `hang/`, `triage/`.

**Root cause.** Under async scheduling the plugin's `TTScheduler` discards a preempted request's
in-flight decode tokens, but `TTModelRunner` never reconciled its own
`CachedRequestState.output_token_ids`. On resume the runner's copy of the request was one token longer
than the scheduler's, so `prompt_lens < InputBatch.num_tokens` made a **complete** re-prefill look like
an unfinished chunk, the runner sampled nothing, the async `num_output_placeholders` was never retired,
and vLLM's max-tokens guard skipped that request on every subsequent step. At ISL 16384 with concurrency
16 the KV pool (264192 tokens) is exactly saturated by 16 x 16384 = 262144, so preemption is expected —
and the odd prefill lengths in the hang log, **16511 = 16384 + 127** and **16448 = 16384 + 64**, are
resumed requests carrying their already-accepted output tokens, not oddly-sized prompts.

**Fix** (`$autofix`, report in `AUTOFIX.md`; committed in the vLLM checkout as `52f5bcd` — see §10):
`plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py` (+65) and `model_runner.py` (+25) reconcile
`req_state.output_token_ids` against `CachedRequestData.num_output_tokens`, mirroring upstream
`gpu_model_runner._update_states`. Two regression tests were added to
`models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py`.

Two residual risks that `AUTOFIX.md` records and this fix does **not** close: `num_output_tokens == 0`
at resume is still unreconciled (shared with upstream `gpu_model_runner`; it cannot deadlock unless
`max_tokens <= 1`, but it costs one corrupted output token), and no test asserts output *correctness*
across a preemption — only that the request completes. See §11.

Note what was **not** done: the request length was not aligned or rounded, the benchmark was not
shortened, concurrency was not lowered, and the context was not capped. Per the stage contract, a valid
prompt length that is not a multiple of an internal chunk size is the autoport's problem to solve.

**Verification.**
* Unit: the two new tests pass.
* Deterministic host-level A/B over 36 configurations: **unpatched 22 HANG, patched 0 HANG, 36/36
  completed 32/32** (`repro/ab_sweep.txt`).
* Hardware, the exact command that hung: **`Successful requests: 32`, `Failed requests: 0`, 279.82 s**
  (`fixverify/`). The resumed-preemption path was genuinely exercised — prefills at 16448, 16450 and
  16511 — and the mechanism-level signature flipped from 2 `host sampling` prefills to **0**.
* End-to-end: the same sweep point ran clean inside the full release workflow (`benchmarks/`, 20/20
  points, 907 requests, **0 failed** across the whole sweep).

## 7. Commands

```bash
# --- autoport vLLM server (external; kept running for the whole TTI session) -----------------
cd /home/ttuser/dev/ornith/tt-metal
export TT_METAL_HOME=/home/ttuser/dev/ornith/tt-metal
export PYTHONPATH="$TT_METAL_HOME:${PYTHONPATH:-}"
/home/ttuser/dev/ornith/ornith-pyenv/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/ornith_ai_ornith_1_0_35b \
  --hf-model ornith-ai/Ornith-1.0-35B \
  --mesh-device "(1, 4)" --max-num-seqs 32 --max-model-len 262144 \
  --server-timeout 3600 --port 8100 \
  --tt-config '{"trace_region_size": 200000000, "l1_small_size": 24576,
                "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192}' \
  --additional-server-args "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml"

# --- TTI smoke, no Docker (health + one OpenAI request were run first, see §8) ---------------
cd /home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/tt-inference-server
export CACHE_ROOT=/home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/tti_cache_smoke
python3 run.py --model Ornith-1.0-35B \
  --runtime-model-spec-json ../ornith_autoport_smoke_spec.json \
  --tt-device p300x2 --workflow benchmarks --service-port 8100 \
  --no-auth --skip-system-sw-validation --disable-trace-capture \
  --limit-samples-mode smoke-test

# --- TTI release ------------------------------------------------------------------------------
export CACHE_ROOT=/home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/tti_cache_release
python3 run.py --model Ornith-1.0-35B \
  --runtime-model-spec-json ../ornith_autoport_release_spec.json \
  --tt-device p300x2 --workflow release --service-port 8100 \
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly
```

Environment that mattered: `CACHE_ROOT` (per-run TTI `workflow_logs` root), `HF_HOME`
(`/home/ttuser/.cache/huggingface`, the snapshot the earlier stages already downloaded — nothing was
re-downloaded), `HF_TOKEN` (read from the existing `~/.cache/huggingface/token`; **never printed, never
copied, and confirmed absent from every copied artifact**), `TT_METAL_HOME`, `PYTHONPATH`.
`JWT_SECRET` was not needed (`--no-auth`; the run spec records `jwt_secret: null`).
`--skip-system-sw-validation` is acceptable because `tt-smi` health was validated directly (§9).
No profiler or watcher variable was set at any point.

## 8. Smoke (run before the release workflow, as required)

1. `GET /health` → **200**.
2. One OpenAI-compatible request → `POST /v1/chat/completions`, `finish_reason: stop`, reasoning and
   content both well-formed (quoted in §5).
3. Tiny TTI benchmark, no Docker, `--disable-trace-capture`, `--limit-samples-mode smoke-test` →
   `run.py` exit **0**; one 16-in/4-out sweep point; benchmark JSON `completed=8 failed=0`; TTI-written
   run spec `docker_server=false`, `service_port=8100`,
   `impl.code_path=models/autoports/ornith_ai_ornith_1_0_35b`. Log: `logs/tti_smoke_benchmarks.log.gz`.

## 9. Hardware actions and recovery

All device work was done from the reservation context with `tt-smi` at
`/home/ttuser/.tenstorrent-venv/bin/tt-smi`. Every reset used the bounded `list / reset / list` sequence
and was followed by a `1x4` mesh open/close smoke.

| when | event | action | result |
|---|---|---|---|
| stage start | routine pre-flight | `tt-smi -ls / -r / -ls` + mesh smoke | all exit 0, 4x `p300c`, `MESH_SMOKE_OK` (`logs/device_reset_stage11.txt`, `logs/mesh_smoke_stage11.txt`) |
| after the §6 hang | host-side deadlock, devices idle | tt-triage captured **before** killing; stale `run_vllm_server`/`EngineCore` killed; `tt-smi -ls / -r / -ls` + mesh smoke | all exit 0, `MESH_SMOKE_OK` (`logs/device_reset_after_hang.txt`, `logs/mesh_smoke_after_hang.txt`) |
| 20:05:47Z | server died during device init: `TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to become active again` (`llrt.cpp:594`) — an **ERISC/infrastructure fault**, not a model fault | stale processes killed, `tt-smi -ls / -r / -ls` + mesh smoke | all exit 0, all four boards present, `MESH_SMOKE_OK` (`logs/device_reset_erisc_recovery.txt`, `logs/mesh_smoke_after_erisc.txt`); server relaunched cleanly at 20:13:15Z |
| 00:39:03Z | the **same** ERISC fault on a later server start, while gathering the §12 round-2 controls | same recovery | all exit 0, `MESH_SMOKE_OK` (`logs/device_reset_erisc_recovery2.txt`, `logs/mesh_smoke_after_erisc2.txt`); server up at 00:45:13Z |


The ERISC fault appeared **both** times a server was started without an intervening `tt-smi -r` after a
previous server had exited, and did not appear on any start that followed a reset. Treat a reset as a
required step between server launches on this host rather than as recovery-only.

One reset was enough each time; no second reset, no lock clearing, and no host reboot was required.
The two ERISC faults recorded above are the only ARC/Ethernet faults this stage saw. `$autofix` was used
for the model/serving hang (§6) and **not** for either ERISC fault, which were handled as infrastructure
recovery.

## 10. Report, artifacts and repo state

* **Final release report:
  `models/autoports/ornith_ai_ornith_1_0_35b/doc/tti_release/report/report_ornith-ai__Ornith-1.0-35B_2026-08-19T215826+0000.md`**
* report data JSON, TTI-written run spec, and both custom specs: `report/`
* 20 trimmed benchmark JSONs (per-request arrays dropped): `benchmarks/`
* 2 eval `results_*.json`: `evals/` — the raw `samples_*.jsonl` dumps were deliberately **not** copied
* run logs (gzipped): `logs/`; device reset / mesh-smoke records: `logs/`
* hang evidence: `hang/`, `triage/`; fix report and repro: `AUTOFIX.md`, `repro/`
* fix verification: `fixverify/`; spec-test probe and controls: `spec_tests_probe/`
* TTI checkout diff: `tti_local_registration.diff`
* `report/READ_THIS_FIRST.md` — how to read the copied release report (CI-subset labels, what the
  ungraded PASS means, what `Spec Tests: NA` means)
* `report/serving_capability_stage11_b32.json` and `report/serving_capability_stage11_b32_warmup.json` —
  the serving-capability records this stage's server
  wrote (`max_num_seqs 32`, `num_blocks 4128`, `max_model_len 262144`). The tracked
  `readiness_vllm/vllm_serving_capability.json` and `..._final.json` are the optimized-vLLM stage's own
  final b1 records (`max_num_seqs 1`, `num_blocks 4097`; `doc/vllm_integration/README.md` cites the same
  file for its 4097-block figure) and were **restored** to that stage's contents,
  because the shared runner rewrites both paths on every server launch and they are that stage's
  evidence, not this one's. Commit `0e8198d3513` (an earlier session of this stage) had committed the
  b32 version of `vllm_serving_capability.json` over it, which contradicted
  `doc/vllm_integration/README.md` ("4097 blocks") and `doc/optimized_vllm/README.md` ("the final b1
  server's warm-up-time report"); that is undone here.

Nothing copied contains `.env`, tokens, weights, the HF cache, Docker layers, persistent TT caches,
profiler CSV bulk, or raw eval sample dumps. Total copy-back: **under 600 KiB** (`du -sh` on this directory; it was ~550 KiB when these notes were finalised, and the exact figure moves with edits to this file, so treat the bound as the claim).

`report_data_*.json` and `runtime_model_spec_*.json` each appear twice, once at the root of
`doc/tti_release/` and once under `report/`. That is deliberate: the runner-side gate
(`.agents/prompts/model_bringup_multigoal/11-tti-release.check.sh`) globs
`<release_dir>/*model*spec*.json` and `<release_dir>/*report*data*.json` non-recursively, so the
implementation-path proof has to sit at the top level. The copies are byte-identical.

### Commits (local only — nothing was pushed)

| repo | branch | commit | contents |
|---|---|---|---|
| `tt-metal` (this repo) | `agentic-research/hous/ornith-1.0-35B` | `5803ef097a8d3b24f68c2482eabb54f89c2cb2fc` | `doc/tti_release/` evidence + the two `tests/test_generator_vllm.py` regression tests (this line's own SHA is recorded by the follow-up commit that fixes it) |
| `/home/ttuser/dev/ornith/vllm` | `dev` | `52f5bcd6c9f7c472a55237beff8245af3874518b` | the §6 plugin fix (`vllm_tt_plugin/input_batch.py`, `model_runner.py`) |

The plugin fix lives in a separate checkout outside this repo, so it is committed there and must be
carried forward with the model; `AUTOFIX.md` holds the full analysis. Neither commit was pushed.

## 11. Row-by-row classification

**Read the Acceptance PASS narrowly.** The spec this stage wrote sets `status: EXPERIMENTAL`, which is
the honest status for a first bring-up and matches what `Qwen3.6-27B` carries. But in
`workflows/workflow_types.py`, `ModelStatusTypes.EXPERIMENTAL.required_target_tiers == []` and
`evals_enforced == False`, so at this status an eval FAIL and every benchmark-tier failure would also
have been demoted to non-blocking. Combined with the fact that **every gradable row in the report is
`NA`** — 20/20 benchmark rows (no `perf_reference` for this model, so `llm_module/target_checks.py`
renders them ungraded precisely so "no targets" cannot read as "passed") and 2/2 eval rows (no reference
score) — the `Acceptance Criteria: PASS, 0 blockers` headline proves that **the workflow ran end to end
and nothing errored**. It is not a quality gate. The substantive quality signals in this stage are
IFEval 82.08 on the CI subset, 907/907 benchmark requests completing across the full context sweep, and
the §12 control outputs.

| row | result | classification |
|---|---|---|
| Acceptance Criteria | PASS, 0 blockers | pass — but see the paragraph above: it is ungraded |
| Benchmarks — all 20 sweep points, ISL 128 → 131072 | ran, 907 requests, **0 failed** | pass (ungraded: no perf targets configured, `EXPERIMENTAL`) |
| `ifeval` (CI subset, 28/541) | 82.08 | **no-reference (N/A)** — see §5; not a failure, not a pass |
| `r1_gpqa_diamond` (CI subset, 10/198) | 50.00 | **no-reference (N/A)** — see §5; 10 samples, stderr 16.7 |
| `meta_ifeval`, `meta_gpqa_cot` | not run | **not applicable** — Meta-Llama-only datasets, structurally impossible for this checkpoint (§5). Not a waived failure. |
| Spec Tests | `NA` (no blocks) | **no coverage in the shipped configuration** — the suite was run separately and 15/22 failed; 13 are proven harness incompatibility, 2 trace to the Class C defect. **Disclosed, not waived** (§12) |
| §6 async-resume deadlock | was blocking | **fixed** and re-verified end to end |
| Class C — long generations not byte-reproducible | pre-existing, confirmed here | **readiness gap, carried forward** from the vLLM-integration stage (§12) |

No row in the shipped release report is a `readiness-fail`. Outside it, the stage records one
carried-forward readiness gap (Class C) and one uncovered area (API conformance).

### Residual risks carried into handoff

* **Class C** — repeated identical long generations differ (§12). Pre-existing; not fixed here.
* **`num_output_tokens == 0` at resume is still not reconciled** by the §6 fix (`AUTOFIX.md` residual
  risk 1, shared with upstream `gpu_model_runner`). It cannot deadlock unless `max_tokens <= 1`, but for
  larger `max_tokens` it costs one wasted step and one corrupted output token.
* **No test asserts output *correctness* across a preemption** (`AUTOFIX.md` residual risk 3). The §6
  verification proves the request *completes*; the release sweep now routinely enters that regime at
  `isl=16384 concurrency=16`.
* **No graded quality row exists for this model.** Establishing a GPU reference for `ifeval` and
  `r1_gpqa_diamond`, and a `model_performance_reference.json` entry for the benchmark tiers, is what
  would turn this from a completed workflow into a real gate.

## 12. Spec tests: why the report shows N/A, what was measured anyway, and what it found

The shipped release configuration produced `Spec Tests: NA (no blocks present)` because
`test_module/dispatch.py` found no spec-test suite matching `model='Ornith-1.0-35B' device='p300x2'`.
Rather than accept that silently, the model was temporarily registered into TTI's
`VLLMParamConformanceTest` matrix and the suite was **run in full** (46 min,
`logs/tti_spectests_probe.log.gz`). Result: **15 of 22 parametrisations failed, 7 passed**
(`test_n[2]`, `test_n[3]`, `test_max_tokens[5]`, `test_max_tokens[10]`, `test_logprobs`,
`test_determinism_parameters[top_k-1]`, `test_determinism_parameters[top_p-0.01]`).

Every failure was replayed against the same server with the suite's own payloads plus paired
controls — first round in `spec_tests_probe/controls.json`, then a **second, stronger round** in
`spec_tests_probe/controls_v2.json` that compares the **whole generation** (reasoning trace + content,
SHA-256) across three repeats instead of only the short final answer, and adds the one row the first
round missed. The second round changed two of the first round's conclusions; what follows is the
corrected account.

### Class A — `content is None` (12 rows): the reasoning parser, confirmed

`test_penalties` (9), `test_coherence_verbatim_echo`, `test_stop[stop_seq0]`, `test_non_uniform_seeding`.
The suite asserts on `message["content"]` as a string. For a thinking model served with
`--reasoning-parser qwen3`, a generation that never leaves the `<think>` block within the requested
budget correctly returns `content: null` with the text in `reasoning`. Controls:

| suite payload | as the suite sends it | same payload, `enable_thinking=false` |
|---|---|---|
| verbatim echo, `max_tokens 32` | `finish_reason length`, 32 tokens, `content=None`, 123 reasoning chars | `content = "The quick brown fox jumps over the lazy dog."` — **exact verbatim echo** |
| penalties baseline, `max_tokens 1024` | `finish_reason length`, 1024 tokens, `content=None`, 4163 reasoning chars | proper repetitive story, 225 tokens |
| `stop: ["Stop"]`, `max_tokens 1024` | stop fired inside the think block at 31 tokens, `content=None` | `content = "1, 2, 3, 4, 5\n"`, `"Stop"` correctly absent — **the `stop` parameter works** |
| `test_non_uniform_seeding`, `seed 0`, `max_tokens 64` | 64 tokens, `content=None` on all three repeats | joke returned, 22 tokens, **byte-identical on all three repeats** |

`test_non_uniform_seeding` asserts two things: that `seed 0` repeats agree, and that different seeds
differ. Only the first is replayed above, because that is the assertion the `content=None` crash happens
in; the second is trivially satisfied given Class C.

The coherence guard's actual property — "the model must echo an exact sentence verbatim" — **passes**;
it also passes with thinking left on and a 4096-token budget. No forward-pass or decode-trace corruption
is being masked.

### Class B — `ReadTimeout (read timeout=30)` (3 rows): reachable, and the reason is Class C

`test_determinism_parameters[temperature-0.0]`, `test_seed_reproducibility`, `test_stop[stop_seq1]`.
The suite's default 30 s client read timeout is genuinely reachable on this device: this autoport
decodes at ~7 tok/s/user on a `max_num_seqs=32` server (the padded 32-row decode step, a known
`doc/optimized_vllm` limitation), and a thinking model emits a variable-length reasoning block before a
two-word answer. Measured over three repeats each (`controls_v2.json`):

| control | completion tokens across 3 repeats | max elapsed | exceeds the suite's 30 s? |
|---|---|---|---|
| greedy, `temperature 0.0` | 149 / 155 / 182 | 25.5 s | no |
| `seed 42, temperature 0.5` | 111 / 226 / 257 | **35.7 s** | **yes** |
| `test_stop[stop_seq1]` (stop strings the model never emits, runs to `max_tokens 1024`) | — | ~142 s | yes |

So the timeout is not a fixed property of the payload — it is driven by how long the reasoning block
happens to be on a given run, which is Class C below.

### Class C — long generations are not reproducible: a real, carried-forward model defect

This is the correction to the first round of controls, which compared only `message.content` — for these
prompts a literal `"\n\nParis."` — and on that basis wrongly reported the model as
"greedy- and seed-reproducible". Comparing the **full generation** shows the opposite:

| repeated identical request | full generation identical across 3 runs? | final content identical? |
|---|---|---|
| greedy `temperature 0.0`, thinking on (149–182 tokens) | **NO** | yes |
| `seed 42, temperature 0.5`, thinking on (111–257 tokens) | **NO** | yes |
| `seed 0, temperature 0.7`, thinking on, `max_tokens 64` | **NO** | yes (both `None`) |
| `seed 0, temperature 0.7`, thinking **off** (22 tokens) | **YES** | yes |
| greedy `temperature 0.0`, thinking **off** (2 tokens) | **YES** | yes |

Short generations are byte-reproducible; long ones diverge. This is the vLLM-integration stage's open
defect — *identical greedy requests are not bit-reproducible above padded decode batch 4*
(`doc/vllm_integration/README.md`, carried forward in `doc/optimized_vllm/README.md`) — visible at
request concurrency 1, because the shipped `max_num_seqs=32` server always executes the padded 32-row
decode step regardless of how many requests are in flight. It is **not** introduced by this stage and
**not** caused by the §6 fix; it is the pre-existing defect showing up in a new place.

Anomaly record, per the review standard:

```text
Observed anomaly:      repeated identical requests produce different generations at concurrency 1
Evidence:              spec_tests_probe/controls_v2.json — 149/155/182 and 111/226/257 completion
                       tokens over three repeats of one payload; SHA-256 of the full generation differs
Affected path:         padded 32-row decode step + on-device sampling
Control or comparison: the same payloads with thinking off (2 and 22 tokens) are byte-identical over
                       three repeats, so short generations do reproduce; prior-stage records show
                       bit-reproducibility at padded batch <= 4 and loss at >= 8
Likely subsystem:      decode batch padding / on-device sampling (pre-existing, vllm_integration stage)
Investigation:         replayed three times per payload with full-generation hashing, with and without
                       thinking, at concurrency 1
Resolution:            readiness gap, carried forward — not fixed in this stage, not masked either.
                       It does not change any accuracy number here (both evals ran once), but a
                       reproducibility claim must not be made for this model until it is fixed.
```

### Why the suite is not in the shipped configuration — corrected

An earlier draft of these notes claimed "the only model in the entire v0.20.0 catalog that sets a
reasoning parser is `Qwen/Qwen3.6-27B`". That is wrong as written and is corrected here.
`metadata.reasoning_parser_name` is set for many models (`gpt-oss-20b`/`120b`, `Qwen3-8B`, `Qwen3-32B`,
`QwQ-32B`, all `DeepSeek-R1*`, `gemma-4-31B-it`). The precise, load-bearing statement is about the field
that actually puts `--reasoning-parser` on the server command line:

> In `release_model_spec.json` (v0.20.0), **`device_model_spec.vllm_args.reasoning_parser` is set for
> exactly one model: `Qwen/Qwen3.6-27B`** (P300X2 and P150X8). It is `None` for all four models in the
> `VLLMParamConformanceTest` matrix — `qwen3_32b`, `llama_3_1_8b`, `llama_70b_family`, `gpt_oss_20b` —
> so upstream runs that suite only against servers launched **without** a reasoning parser, which is
> what decides whether `message.content` can be `None`. `Qwen3.6-27B`, the one model actually served
> with the parser, is not in the matrix.

Upstream does have a reasoning-aware conformance suite —
`VLLMQwen3StreamingParamConformanceTest` (`llm_module/test_vllm_qwen3_streaming.py`), which accumulates
`delta["reasoning_content"]` and sends `chat_template_kwargs {"thinking": true, "enable_thinking": true}`
— but it is a Qwen3-32B-specific streaming regression suite, not a general conformance suite, and it is
matrixed only for `qwen3_32b`. There is no general reasoning-aware conformance suite in v0.20.0.

**Status of these rows, stated without softening.** The temporary registration was reverted, so
`tti_local_registration.diff` contains only the three §3 edits and the shipped release report has **no
Spec Tests block at all**. That means the shipped configuration carries **zero API-conformance
coverage**, and the 15 failures above are **disclosed, not waived** — no upstream issue was filed, so
none of them meets this skill's bar for `issue-waived`. Concretely:

* 12 Class A rows and `test_stop[stop_seq1]`: **harness incompatibility**, mechanism proven by control.
  A reasoning-parser server cannot satisfy a suite that requires `content` to be a string.
* `test_determinism_parameters[temperature-0.0]` and `test_seed_reproducibility`: **timeouts caused by
  Class C**, i.e. by a real model defect, not by a harness assumption alone.
* Class C itself: a **carried-forward readiness gap**, recorded above.

Two follow-ups this stage did not do and is not claiming to have done: file the upstream issue for the
reasoning-parser conformance gap (`llm_module/test_vllm_chat_completions.py` assumes
`choices[0].message.content` is a string and defaults to a 30 s read timeout), and fix Class C.

## 13. Cleanup

No autoport vLLM server, no benchmark client, no `tt-inference-server` Docker container and no tmux
session are left running (`tmux` is not installed on this host; long jobs used `setsid nohup`). No `.env`
was created in the TTI checkout. The reservation was not released.
