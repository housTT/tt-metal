# GPT-OSS 120B TTI release handoff

## Outcome

- Readiness: **PASS — `release-readiness-ci-subset-pass`** for the generated
  autoport on
  P150x4. This is not an unrestricted full-set or unrestricted performance
  claim.
- Final repaired release handoff/report merger: `EXIT_CODE=0`, no acceptance
  blockers.
- Accuracy: AIME 2025 13/15 (86.6667%), GPQA Diamond CoT 7/7 (100%),
  MMLU generative 84.6732%, and full canonical IFEval 463/541 (85.5823%);
  all four accuracy gates pass.
- API/spec tests: Logger Fork Safety passes; Vllm Chat Completions passes
  22/22, including both stop-string cases.
- Benchmarks: all 21 requested rows completed with exact input/output lengths,
  counts, metrics, and numeric zero error counts. One B1 row is narrowly
  `issue-waived`: three aggregate-throughput failures come from an internally
  inconsistent batch-scaled target, while two higher-tier TTFT targets are
  genuinely unmet; the other 20 rows are ungraded (`NA`).
- Context: exact 131072-token support is preserved. The boundary request
  `130944 + 128 = 131072` passed twice, and non-aligned logical length 10000
  remained unmodified.
- Stage review: **`clean-pass`**. The independent metric-integrity review first
  returned `more-work-needed`; every listed finding was fixed. The fresh
  post-remediation review found one final docs-only smoke-cap mismatch, which
  was corrected at `076a5e46ff7fd470e2f54f98614edbd563abd069` and rereviewed
  clean. The complete verdict is `stage_review_final.md`.

The original monolithic `run.py --workflow release` attempt exited 1 because
of repairable TTI GPQA, benchmark, spec-timeout, coherence, and stop-semantics
problems. AutoFix isolated those problems and the initially omitted mandatory
IFEval gate. The corrected GPQA, full benchmark, full spec-test, and full
canonical IFEval workflows each exited 0, and the fail-closed
evidence-validating merger then exited 0. The outcome above does not mislabel
the initial monolithic exit or the rejected low-effort IFEval attempt.

## Autoport implementation check: `models/autoports/openai_gpt_oss_120b`

- Model: `openai/gpt-oss-120b`.
- Target code: `models/autoports/openai_gpt_oss_120b`.
- Adapter: `models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py`.
- Both copied source specs and the TTI-written runtime spec record
  `impl.code_path=models/autoports/openai_gpt_oss_120b`.
- The live server log identified
  `models.autoports.openai_gpt_oss_120b.tt.model`.
- Import-origin checks resolved TTNN, official vLLM, the standalone TT plugin,
  and the generated adapter only below `/home/ttuser/dev/gpt-oss-20b`, ending
  in `WORKSPACE_IMPORTS_OK`; see `import_origins.txt`.
- The copied report/spec scan contains no legacy packaged implementation
  identifier.

No stock or demo model, forked vLLM, or alternate checkout was run or
evaluated.

## Server mode, host, and context

- Mode: external OpenAI-compatible autoport vLLM server; TTI was client-only.
- Docker: not used. The inherited image below is provenance only.
- Reservation host: `tt-quietbox-part-2`.
- Hardware: four Blackhole p300c boards, opened as P150x4 mesh `(1, 4)`.
- Endpoint: `http://127.0.0.1:8000`.
- Context: 131072 tokens, exactly matching `doc/context_contract.json` and all
  source/runtime spec fields.
- Server sizing: max 32 sequences, block size 64, hybrid paged BFP8 KV cache.
- Request policy: valid logical lengths were not capped or aligned. Only
  mathematically invalid prompt-plus-completion requests are rejected.

Every serve operation first sourced
`.agents/scripts/gpt_oss_workspace_env.sh` and repeated the import-origin
check. The final server was owned by tmux session
`gpt-oss-120b-tti-server-r4`.

## Provenance

| Component | Version / local commit |
| --- | --- |
| tt-metal autoport base evaluated by the run | `76e51849603f6ff7d05f37e15b4938016d7e946e` |
| completed optimized-vLLM stage | `bcb3f87dd50d4813bb2c917701dedf3f06b13545` |
| official vLLM base / final local repair | `568afb3a13806beb53bb2e6bd518269357b237c0` / `54dea57d98ccfaef072908f085d9296d544ba1fe` |
| standalone TT vLLM plugin | `053c0782aa11028924c21cb061ffa76576705cad` |
| TTI base / intermediate / IFEval / final metric repair | `f07a31d2a2f908aa04098685034e7a5bde7554ea` / `b15d3ae6ac5ae2a00ecffc2e795d37246bb4d5e4` / `ddfba898209f0aaada2294d9230801d053edcf80` / `8459ba8dc6e690e7987235182a0d87c67bacc4a7` |
| inherited release tag | v0.17.0, `48055de3d1b444e0dbce23cc378590eb6abc2ca5` |
| inherited image, not used | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.17.0-8c48a10-f52987a` |
| TTI client checkout | VERSION 0.21.0 |
| HF model revision | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |

All listed repository changes were committed locally and were never pushed.
The tt-metal handoff commit is recorded below after creation.

## Commands and key environment

Server command:

```bash
cd /home/ttuser/dev/gpt-oss-20b/tt-metal
source .agents/scripts/gpt_oss_workspace_env.sh
VLLM_SYSTEM_START_DATE=2026-09-01 \
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b \
  --served-model-name openai/gpt-oss-120b \
  --block-size 64 --max-num-seqs 32 --max-model-len 131072 --port 8000 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":750000000,"fabric_config":"FABRIC_1D_RING"}}' \
  --async-scheduling --disable-log-stats \
  --structured-outputs-config '{"reasoning_parser":"openai_gptoss","enable_in_reasoning":false}'
```

Initial release command (the embedded spec also contains these settings):

```bash
cd /home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tt-inference-server
export CACHE_ROOT=/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache
export PERSISTENT_VOLUME_ROOT=/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/persistent_volume
export SERVICE_PORT=8000
python3 run.py \
  --model gpt-oss-120b \
  --runtime-model-spec-json /home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/specs/autoport_release_spec.json \
  --tt-device p150x4 --engine vllm --workflow release \
  --server-url http://127.0.0.1:8000 --service-port 8000 \
  --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly
```

The corrected constituent reruns used the same command shape and external
endpoint with embedded `workflow=evals`, `benchmarks`, and `spec_tests`
respectively. The GPQA repair spec selected exactly samples 0 through 6. The
final IFEval runtime proof is copied as `runtime_model_spec_validation.json`.
The checkout's own `run.py --help` spelling is summarized in
`run_py_help_summary.txt`.

The final fail-closed report merge used the original release plus the passing
GPQA, benchmark, spec-test, and full IFEval reports, and additionally required
the exact GPQA raw aggregate/runtime spec, IFEval raw aggregate/runtime spec,
all 21 raw benchmark artifacts, and the single-row issue waiver. It passed with
`acceptance=PASS`, `blocker_keys=none`, official vLLM commit
`54dea57d98ccfaef072908f085d9296d544ba1fe`, and TTI commit
`8459ba8dc6e690e7987235182a0d87c67bacc4a7`.

The mandatory IFEval recovery used the copied `run_tti_ifeval_repair.sh` and
`autoport_ifeval_repair_spec.json`. Its logical selector is `meta_ifeval`, its
lm-eval execution task is canonical `ifeval` over `google/IFEval`, and its
embedded spec has `workflow=evals`, `docker_server=false`,
`local_server=false`, `service_port=8000`, `disable_trace_capture=true`, and
no sample limit. The full run used `eval_max_concurrency=1`; an eight-way
throughput probe was stopped when eight requests took 7m24s, substantially
slower than serial execution, without disturbing the server.

Exact full IFEval client command:

```bash
cd /home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b
bash run_tti_ifeval_repair.sh
```

The generated command used canonical `--tasks ifeval`,
`--apply_chat_template`, no `--limit` or `--samples`, concurrency 1,
`reasoning_effort=medium`, `max_gen_toks=4096`, `do_sample=false`,
`temperature=0`, seed 42, and `max_length=131072`. It completed 541/541
requests from 2026-09-03 00:39:36 UTC through 04:17:04 UTC; the workflow
exited 0 after 13046.7 seconds.

The source specs set `docker_server=false`, `local_server=false`,
`service_port=8000`, and the intended workflow before execution; command-line
flags were not used to conceal incorrect embedded values. Key non-secret
settings were `VLLM_TARGET_DEVICE=tt`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
`VLLM_RPC_TIMEOUT=900000`, `TORCHDYNAMO_DISABLE=1`, `MESH_DEVICE=P150x4`, and
`ARCH_NAME=blackhole`. No token or secret value was printed or copied.

## Smoke, release results, and prompt format

Before full release, the no-Docker smoke passed in order:

1. health endpoint HTTP 200;
2. one OpenAI-compatible request HTTP 200 and `finish_reason=length` at its
   deliberately tiny 8-token connectivity cap;
3. one TTI benchmark with `disable_trace_capture=true`, 1/1 completed.

The smoke retained only hashes, counts, timings, and finish metadata. The tiny
benchmark measured mean TTFT 481.0119 ms, mean TPOT 78.2444 ms, and 7.7742
output tok/s. Exact full-context evidence is in
`full_context_boundary_summary.json`.

The model declares an HF chat template. Release calls used
`/v1/chat/completions`; lm-eval used `--apply_chat_template`; the smoke used
`tokenizer.apply_chat_template(add_generation_prompt=True)`. The copied
`prompt_format_summary.json` contains only hashes and token counts. The prior
optimized-vLLM qualitative set had 12 manually inspected responses and a
qualified pass; two thermodynamics answers were incomplete at their requested
cap, so this handoff does not overstate qualitative completeness.

Final compact evidence:

- `accuracy_summary.json`: all four gates pass. `meta_gpqa_cot` is
  satisfied by GPQA Diamond CoT 7/7. Logical `meta_ifeval` is satisfied by
  the full canonical `ifeval` task over `google/IFEval`: 463/541 strict
  prompts, 85.58225508317929%, versus the 74.29% minimum.
- `ifeval_summary.json`, `ifeval_aggregate_results.json`,
  `runtime_model_spec_validation.json`, and the final merged report: full-scope
  count, configuration, corrected timing, score, and provenance evidence
  without sample outputs. The pre-repair TTI source report is retained only in
  the bounded TTI work area because its stale timing denominator would make a
  contradictory customer handoff artifact.
- `gpqa_summary.json`, `gpqa_aggregate_results.json`, and
  `gpqa_runtime_model_spec.json`: the exact seven selected IDs, flexible score,
  external autoport wiring, and corrected `2246.9087665929983 / 7 =
  320.9869666561426` seconds per selected request. The copied aggregate has no
  sample or response field.
- `benchmark_summary.csv`: 21/21 rows complete, zero request failures/errors,
  exact lengths/counts, and no missing metrics. The 10000-token non-aligned
  workload passed without alignment.
- `spec_test_summary.json`: acceptance true, zero blockers, 22/22 Vllm Chat
  Completions plus Logger Fork Safety pass.
- `report_data_release_ci_nightly.json` and
  `report_release_ci_nightly.md`: four-gate merged readiness PASS with
  validated links to each authoritative source report and raw benchmark
  projection. The merger exited 0 with `acceptance=PASS` and
  `blocker_keys=none`.

The one benchmark waiver is limited to `ISL=128, OSL=128, concurrency=1,
requests=8`. It completed 8/8 and reproduced the optimized autoport baseline.
The exact failed set is `functional.tput`, `complete.tput`, `target.tput`,
`complete.ttft`, and `target.ttft`. The three throughput failures derive from a
B32-like aggregate target attached to the B1 row; the two TTFT failures are
genuine unmet higher performance tiers. All five are informational under TTI's
committed `EXPERIMENTAL` policy and remain visibly failed in the report. See
`benchmark_target_ISSUE_WAIVER.md`. The waiver does not cover correctness,
failures, missing metrics, shortened requests, or any other row, and it does
not establish unrestricted performance readiness.

An unrestricted release run was not a practical nightly gate: the configured
full task set represents approximately 1,070,170,112 maximum output tokens;
at the measured 57.674 output tok/s that is about 214.8 device-days before
prefill and harness overhead. The effective nightly-equivalent limits were
AIME 50% (15 samples), GPQA 3.5% (7 samples), and MMLU 15% (approximately
2,127 samples). Canonical IFEval was run at its full 541-sample scope—stronger
than its 5% nightly slice—because the initial 28-sample diagnostic scored
20/28 and did not pass the configured gate. That diagnostic was neither
masked nor waived.

The first full canonical IFEval run used low reasoning and a 1,280-token
generation budget. Although all 541 requests completed, its strict score was
397/541 (73.3826%), below the gate; it was rejected and not waived. A fixed
20-ID A/B isolated the generation budget and reasoning policy without printing
responses. The accepted medium+4096 policy matches the cited public reference
semantics and raised the full strict score to 463/541. Details are in
`ifeval_AUTOFIX.md` and `ifeval_reference_parity_AUTOFIX.md`.

## Recovery and validation

- Corrected custom-spec external mode, service port, workflow, and autoport
  path before execution; Docker was not used as a workaround.
- Retained the exact-context router repair and verified the maximum logical
  request twice without context reduction.
- Repaired GPQA dataset/harness wiring and reran 7/7 successfully.
- Repaired TTI's GPT-OSS IFEval wiring by preserving the logical
  `meta_ifeval` report identity while executing canonical `ifeval`; added a
  task selector orthogonal to sample limiting, strict-metric fail-closed
  behavior, authoritative nonzero subprocess-exit handling, explicit rejection
  of both `NA` and `FAIL` rows, and the required readiness-status spelling.
- Repaired explicit eval-sample accounting: selected logical IDs now override
  lm-eval's full-dataset count for acceptance and timing, canonical aliases are
  preserved, and expanded group results fail closed rather than reusing one
  count for every subtask. GPQA's authoritative raw duration now uses seven—not
  198—as its denominator; IFEval uses the newest 541-sample artifact.
- Repaired the benchmark endpoint, deterministic temperature behavior, and
  streaming raw-evidence merger; the corrected full 21-row workflow exited 0.
- Repaired TTI spec timeouts, reasoning-aware coherence validation, and exact
  completion checks. The TTI host suite passed 308 tests.
- The final touched-surface suite passed 304 tests, the final merger suite
  passed 36 tests, and Python compilation plus diff checks passed. The
  checkout's pre-commit wrapper could not start because its expected
  `.pre-commit/bin/activate` is absent; the available copyright hook passed
  earlier, and no formatter/linter dependency was installed.
- AutoFix found that vLLM's detokenizer removed a matched stop string while
  the Harmony parser rebuilt reasoning from untrimmed token IDs. The official
  vLLM checkout now trims parsed non-stream output consistently with
  `stop_reason`; its focused regression suite passed 5 tests, and the live
  endpoint plus both TTI stop cases passed.
- A broader optional vLLM test module could not collect because the environment
  lacks optional `pytest_asyncio`; focused tests, pycompile, live endpoint, and
  the full TTI suite cover the changed path.
- No ARC, ERISC, remote-Ethernet, or reset failure occurred in the final runs.
  An earlier interrupted AIME attempt was an external process interruption;
  health/import checks passed and no reset was warranted.
- Removed an ignored zero-byte TTI `.env`; it contained no data and no secret
  was read or copied.
- Post-shutdown `tt-smi -ls --local` listed all four boards as visible and
  resettable; a fresh P150x4 `(1, 4)` open/close ended `MESH_SMOKE_OK`.

## Reports, exclusions, and cleanup

Authoritative generated report:

`/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/final_release_report/report_id_openai-gpt-oss-120b-autoport_p150x4_release-repaired_2026-09-03T043907+0000.md`

Its JSON peer is under the adjacent `data/` directory. The corrected spec
report is:

`/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/spec_tests/data/report_data_id_openai-gpt-oss-120b-autoport_p150x4_2026-09-02_20-59-57.json`

Small reports, summaries, specs, repair notes, the benchmark CSV, and the smoke
log were copied to this directory. The copied IFEval and GPQA aggregates contain
only configuration, counts, and aggregate metrics; neither has a sample or
response field. Raw completions/reasoning, per-sample eval JSONL, token IDs,
weights, caches, persistent TT cache, Docker layers, and large server/eval logs
were intentionally excluded.

Cleanup completed: the owned server exited gracefully, its tmux session was
removed, no owned vLLM process remains, no TTI container was created, and only
the pre-existing unrelated tmux session `0` remains. Existing unrelated
Supabase containers were left untouched. Post-shutdown `tt-smi -ls --local`
showed all four boards visible and resettable; a fresh P150x4 `(1, 4)`
open/close ended `MESH_SMOKE_OK`, so no reset was performed. The
runtime-mutated serving capability file was restored to SHA-256
`e63ca5f81e1496be55c709c06c3881637e1efeccf5d16703ff82c998b64f625a`.
The empty harness-created TTI `.env`, incidental 52-byte `uv.lock`, and
temporary GPQA cache symlink were removed; none contained or exposed a secret.

## Local handoff commits and final review

- Official vLLM repair: `54dea57d98ccfaef072908f085d9296d544ba1fe`.
- TTI intermediate repair: `b15d3ae6ac5ae2a00ecffc2e795d37246bb4d5e4`.
- TTI IFEval repair: `ddfba898209f0aaada2294d9230801d053edcf80`.
- TTI final metric-integrity repair:
  `8459ba8dc6e690e7987235182a0d87c67bacc4a7`.
- tt-metal initial report/capability handoff:
  `44c461da3a59e8fa9bcf6d9aedb91cab34208dd0`.
- tt-metal IFEval remediation handoff:
  `9a5d198fee3b4406fe417e2579a79311ad4e2ca5`.
- tt-metal metric-integrity/report handoff:
  `417bb4e7c8e2b3ab111676c6baa79f5007a96bb0`.
- tt-metal contradictory source-report cleanup:
  `d478af00c792e7beadcd7c2841c7cc8d0ed71ff1`.
- Independent stage review: provisional `more-work-needed` findings are in
  `stage_review_metric_findings.md`; final `clean-pass` is in
  `stage_review_final.md`, committed at
  `f5dc6d3847a2e2e6e83008cb90fae3d79d28f042`.

No commit was pushed.
