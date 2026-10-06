# Track S work log (server, demo page, manifests)

All times UTC, 2026 Oct 5. Host `qb2-120-p11t01`, tt-metal venv (Python 3.10.21, torch 2.11.0+cpu, transformers
5.12.1, fastapi 0.136.3, uvicorn 0.53.0, httpx 0.28.1, pytest 9.0.3). No device was opened by this track; no git
commit, checkout, stash or push was run.

## Sources read before writing

- `/home/hous/dev/laya/PLAN.md` in full, including amendments A1 to A4.
- pip `laya` 0.3.27 in `/home/hous/dev/laya/evals/.venv/lib/python3.12/site-packages/laya/`: `serve.py` (limits,
  422 and 413 rules, headers, batch envelope), `agent.py` (`_check_question`, `_to_internal`, `_encode_state`,
  `_decode_answers`, `predict_batch`, usage fields, temperature clamp), `common.py` (`build_sequence` stats,
  `build_head`, `collapsed_options`, `unpermute_probs`, `answer_confidence`, `clamp_temperature`), `confidence.py`
  (`check_min_confidence`, `apply_confidence_gate`). The clone at `/home/hous/dev/laya/evals/vendor/laya` has the same
  `agent.py` and `common.py` (diff -q identical); its `serve.py` differs only in `_resolve_model` (a 422 for unknown
  model paths), which does not change the answer shape.
- Vendored Hub files `vendor/rl_common.py`, `vendor/rl_agent_api.py`, `vendor/rl_agent_config.json`.
- CLM patterns: `server/app.py`, `server/demo.py`, `tests/test_host_engine.py`, `tt-model.yaml`, `requirements.lock`,
  `doc/release/RUN_NOTES.md`; workspace scripts `package-build.sh`, `serve-tt.sh`, `server-smoke.sh`,
  `scratch/wrapper_server.py`, `evals/trex/demo_ws_client.py`, the screenshot step in `final-build12-demo.sh`.
- tt-model 0.1.0 source at `/home/hous/dev/tt-model-manager/src/tt_kernel/`: `container_manifest.py`
  (`load_container_manifest`, `validate_sources_exist(root=...)`), `launchers.py` (`TtDitServerLauncher`: runtime
  keys `app`, `packages`, `lock`, `mesh_shape_env`; `serve_argv`; `serve_env` sets `HF_MODEL` and `MESH_DEVICE`),
  `build.py` (`CODE_IGNORE`).
- The authors' `research/scripts/bench_latency.py` (STATE_EN, Q_CHOICE, Q_NOUL, `qs(n)`, `timed`), `bench_apps.py`
  (question texts for the presets), `bench_local.py` (Part B protocol), the Hub README (issue #185 wording).

## Timeline

- 21:05 to 21:12: read the plan and sources; wrote `server/__init__.py`, `decode.py`, `engine.py`, `demo.py`,
  `app.py`, `__main__.py`.
- 21:12: first CPU smoke (`logs/p2_cpu_smoke_20261005T211221Z.log`): model loaded in 20.6 s; the adapter picked up
  Track R's `reference/laya_reference.py` (`LayaReference`), which already existed; the Hub `collate_items` raised
  `KeyError: 'target'` (the Hub version requires `target`, `label`, `episode`, `ep_step`; pip's does not). Fixed by
  adding those keys to the items as `rl_agent_api.py` does.
- 21:14: second smoke (`logs/p2_cpu_smoke_20261005T211451Z.log`): STATE_EN routing = billing 0.8899 / technical
  0.0417 / sales 0.0685 (batch `1x256`, 15.5 s under load), a 5-question call (`8x256`, 21.6 s), a 4-state x
  3-question batch with `min_confidence=0.5` (`16x256`, 27.1 s), gating fields present. `act_probability` reads 1.0,
  as the authors' issue #185 says.
- 21:13 to 21:16: `shim/laya_tt_backend.py`, `demo/presets.json`, `demo/feed_cases.json` (generated from the parquet
  with the eval venv's pyarrow: the first 15 test cases of each of the 4 workflows, 60 cases, 300 decisions, 193 KB;
  the `agent_trace` preset state, questions and gold were re-read from the dataset row to avoid transcription
  errors), the five workspace scripts.
- 21:16: started a CPU implementation benchmark (eager vs sdpa, 8 vs 16 threads). Load average was 48 on 16 threads
  (three other python processes at 300 to 800 percent CPU), so the numbers (7 to 22 s per 256-token row) measure
  contention, not the model. Stopped it by PID at 21:21 (`logs/p2_cpu_bench_20261005T211623Z.log`, ends with FAIL
  and the reason). Orchestrator confirmed: defer CPU timing until the load average is under 8, use at most 4 threads.
- 21:21 to 21:24: `LAYA_CPU_THREADS` now reaches `LayaReference(threads=...)` and is re-applied after construction;
  `LAYA_CPU_ATTN` and `LAYA_CPU_IMPL` added. Fixture job rerun with 4 threads
  (`logs/p2_make_fixture_20261005T212334Z.log`, DONE): wrote `server/sanity_reference.json`,
  `tests/fixtures/recorded_logits.json` (4 rows padded to 4x512, kmax 12) and `tests/fixtures/recorded_answers.json`.
- 21:17 to 21:27: `demo/index.html`, `demo/demo.css`, `demo/demo.js` (node `--check` passes), the three test files,
  both manifests, `demo-script.md`.
- 21:25: import closures (`package/IMPORT_CLOSURE.md`): the CPU path loads 13 autoport files and no `models/common`
  file, ttnn not loaded; the tt modules that exist load 21 files including `tests/pcc_utils.py` (via
  `tt/runner_infra_upstream.py`); `tt/engine.py` does not exist yet.
- 21:28: `test_host_engine.py`: 25 passed (two expectation errors in my own gate test fixed on the way; the code
  followed pip semantics each time: an answer without any usable confidence is `unevaluated`).
- 21:29 to 21:31: `test_server_cpu.py` + `test_demo_page.py` first run (`logs/p2_tests_cpu_20261005T212928Z.log`):
  26 passed, 1 failed in 104 s; the failure was the dash check reading verbatim dataset text in `feed_cases.json`
  (an em dash inside a state string) and the test file itself holding the literal characters. Fixed: the scan skips
  the dataset file and uses escapes. Second demo run: the "no external assets" check matched the SVG namespace URI of
  the inline favicon; the check now matches only `src=`, `href=`, `url(`, `@import` and `fetch(` with `http`.
- 21:30: manifests validated offline (`package/BUILD_PLAN.md` has the table): semantics ok for both; sources ok under
  the main checkout, missing under the still-empty worktree; `serve_argv` identical to `serve-tt.sh`.
- 21:35: `test_demo_page.py`: 5 passed (`logs/p2_tests_demo_20261005T213504Z.log`, DONE).

## Results

| suite | result | log |
|---|---|---|
| `tests/test_host_engine.py` (no model) | 25 passed in 7.2 s | foreground |
| `tests/test_server_cpu.py` (CPU reference, 4 threads) | 22 passed (in the 27-test combined run: 26 passed, 1 failed, the failure in the demo suite) | `/home/hous/dev/laya/logs/p2_tests_cpu_20261005T212928Z.log` |
| `tests/test_demo_page.py` | 5 passed in 25.2 s after the two test fixes | `/home/hous/dev/laya/logs/p2_tests_demo_20261005T213504Z.log` |

Slowest calls under load (4 threads, load average about 30): every preset decides 28.8 s (5 presets), batch and
chunking 9.3 s, STATE_EN example 7.2 s.

## Decisions recorded

1. Temperatures are clamped to [0.5, 5.0] as pip 0.3.27 does (`LAYA_TEMPERATURE_CLAMP=0` restores the Hub rule).
   Only `choice:11+` (0.1006) is affected. Reason: pip is the wire target and the evaluation baseline (E0 compares
   decoded JSON against pip).
2. `usage.truncated_questions` is pip's field name; the plan wrote `questions_truncated`.
3. `labels` on noul questions is a 422 (the Hub `render_options` cannot render them).
4. `model`, `task`, `lang`, `lang_guess`, `batch_size`, `sort_by_length` are accepted and ignored; no `routing` block.
5. `/health` returns 503 until the lifespan finished, like `/v1/health`, so `curl -f` loops wait for readiness.
6. The CPU backend is Track R's `LayaReference` (eager attention) by default, per the plan; the vendored sdpa build is
   behind `LAYA_CPU_IMPL=vendored`. No speed claim is made for either under today's load.
7. The feed file is an object with `attribution` and `cases`; the Appendix B verify line was rewritten accordingly.
8. Both manifests ship `tests/__init__.py` and `tests/pcc_utils.py` because `tt/runner_infra_upstream.py` imports
   them; drop the two entries when Track T removes the upstream runners.
9. The sibling manifest proposes `LAYA_ROW_BUCKETS_1024` for the 1024-token row set; Track T may rename it.
10. `evals/README.md` did not exist; the stats JSON shape is defined in `doc/serving/README.md` for Track E.

## Ports and jobs

Ports used by this track: none (all server tests ran through `TestClient`; no uvicorn process was started). Scripts
default to 8731 (`serve-cpu.sh`), 8710 (`serve-tt.sh`, the package port) and 8738 (`screenshot-demo.sh` wrapper).
Background jobs and their logs: `p2_cpu_smoke_*` (2), `p2_cpu_bench_*` (stopped), `p2_make_fixture_*` (2, the first
stopped and relaunched with 4 threads), `p2_tests_cpu_*`, `p2_tests_demo_*` (2). All pid files removed.

## Open items for other tracks

- Track T: `tt/engine.py` with `LayaEngine.from_env()`, `forward` and `shapes()` per the contract in `README.md`;
  `last_device_ms` and `close()` are optional but used; read `LAYA_MODEL_DIR`.
- Track E: `evals/demo/feed_client.py` should write the stats JSON shape in `README.md` (`schema`
  `laya-demo-feed-stats/1`, `source` `feed_client`).
- Orchestrator: fill the `PLACEHOLDER` values in both manifests and `demo-script.md`; switch `runtime.packages` to
  `runtime.lock` after the first build; CPU timing only when the load average is under 8.

## 2026 Oct 6, 00:15 to 00:30 UTC: host path profiling (orchestrator follow-up from the device tracks)

Trigger: the stage 7 served E5 table (client 11.0 / 26.5 / 46.0 / 212.9 ms against device 9.2 / 22.7 / 40.2 / 191.7
ms at 1 / 5 / 10 / 50 questions, engine host tail 0.1 to 0.5 ms). From that table the split is: inside the engine
call (server header minus device) 1.0 / 3.0 / 4.8 / 19.4 ms, HTTP layer (client minus server header) 0.8 / 0.8 /
1.0 / 1.8 ms.

- 00:17: stage accumulators added to `predict_batch` and `encode_state` (`meta["stages"]`, `engine.last_stages`;
  nothing on the wire). Load average 0.35.
- 00:17: before wire capture through the real CPU engine (`logs/p2_wire_before_20261006T001700Z.log`, 15 requests,
  `scratch/wire_before/`), then the before benchmark with a zero-forward backend on port 8732
  (`logs/p2_host_bench_before_20261006T001814Z.log`): 50 questions 17.6 ms host path, of which build_rows 9.7 (the
  vendored `build_sequence` re-tokenizing the state per question), build_heads 3.8, head_stats 2.1, collate 1.1,
  decode 0.6; HTTP layer 0.86 to 1.09 ms; 64x5 batch 121 ms. This reproduces the served gap.
- 00:20: `build_heads` (vendored builder on an empty state, once per question per request), rows as head plus state
  slice with the vendored truncation expressions, fallback to the full vendored call when the head alone reaches
  `max_len`; equivalence test against direct vendored calls (84 rows compared) passes.
- 00:22: 512-entry LRU of heads keyed by normalized question, `max_len`, `head_max_len`; one batch tokenizer call
  per request for all states; equivalence tests for both. After capture and benchmark on port 8733
  (`logs/p2_wire_after_20261006T002242Z.log`): all 15 responses byte-identical (`cmp`); 50 questions 2.0 ms host
  path (collate 0.98, decode 0.51, build_heads 0.11, build_rows 0.09); 64x5 batch 12.5 ms; HTTP layer 0.64 to 1.05 ms.
- Not changed: `vendor/collate_items` (largest remaining stage), the numpy decode, the response encoder (orjson is
  installed but float formatting identity is not guaranteed), uvicorn backends (uvloop 0.22.1 and httptools 0.8.0
  are installed and auto-selected).
- Ports used: 8732 and 8733 (in-process uvicorn inside the benchmark, closed at exit). Pid files removed.
- 00:30: full CPU run after the changes (`logs/p2_tests_cpu_20261006T002425Z.log`): 54 passed, 3 warnings in 33.74s.

## 2026 Oct 6, 00:33 to 00:37 UTC: combinable autorun for the review screenshot

`/home/hous/dev/laya/evidence/demo_p150_b1.png` (taken with `?autorun=feed`) showed an empty Answers panel.
`demo.js` now reads `autorun` as a comma list: `decide` selects `preset` (default: the first preset, already applied
at load) and awaits Decide; `feed` then starts the feed, so `?autorun=decide,feed&rate=4&seconds=20` shows answer
cards, the Last call tiles and the live feed in one screenshot. The footer documents the options (`id="page-options"`);
`test_demo_page.py::test_page_documents_autorun_options` checks the page text and the two mode checks in `demo.js`.
`/home/hous/dev/laya/bin/screenshot-demo.sh` defaults to `AUTORUN=decide,feed` and appends `&preset=$PRESET` when
`PRESET` is set. `node --check` and `bash -n` pass; no em or en dash in the edited files. `test_demo_page.py`: 6 passed
in 18.4 s (`/home/hous/dev/laya/logs/p2_tests_demo_20261006T003702Z.log`).
