# kev-9b stage 4 (datatype sweep): work log

Times are Eastern Time, 2026 Oct 01. Agent: stage 4 preparation (harness only; the full sweep runs on request). Constraints in force: device commands only through `/home/hous/dev/kev/bin/devrun` with `timeout`, one device process at a time, at most one short device smoke in this task; `tt/engine.py`, `tt/server.py`, `tests/test_engine.py` not touched (stage 3 owns them).

## 20:38 to 20:44: reading

- `/home/hous/dev/kev/STATUS.md`, the datatype-sweep skill (no references directory; only `agents/openai.yaml`), `/home/hous/dev/kev/reports/review_stage1.md`, `../functional/README.md`, `/home/hous/dev/kev/reports/kev_internals.md` (sections "agreement()" and 8), the plan Stage 4 section.
- Precision sites verified in the worktree: `qwen36/tt/mlp.py` (bfp4 gate/up at `_build_gate_up`, TP dram-sharded and interleaved `shard_w`, single-device `load`; bfp8 down; LoFi configs in `Qwen36MLP.__init__`), `qwen36/tt/attention/weights.py:22` (bfp8), `qwen36/tt/gdn/weights.py:83, 117, 213, 228` (bfp8, the last two are the derived decode-only fused weights), `gdn/gated_deltanet.py:38-47` and `attention/gated_attention.py:26-35` (LoFi, fp32 acc), SDPA HiFi2 in `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:157-159`, `QWEN_GDN_FP32_STATE` at `ttnn_delta_rule_seq.py:201`, `QWEN_SDPA_BF8` in `qwen36/tt/attention/tp.py:138` and `qwen36/tt/model.py:2702`.
- Cache naming: `ttnn/ttnn/operations/core.py:891, 897` (parent dirs created; `_dtype_<D>_layout_<L>.tensorbin` suffix).
- `KevEngine` constructor at the time of writing: `KevEngine(device, args_cls=None, max_state_len=8192, n_layers=None, chunk_size=2048, snapshot_slots=8)`; `prefill_hidden(token_ids [1,T], positions)`. The harness introspects the signature and passes only `args_cls`, `max_state_len`, `n_layers` when present.
- Stage 1 control numbers to reproduce (`/home/hous/dev/kev/reports/stage1_control_merged.json`): 29/29 agree, max_dp 0.10236984, mean_dp 0.03664658, min_pcc 0.97677, median 0.99267, 17.9 s device time for 17,389 tokens, engine load 27.4 s.

## 20:44: qwen36 knob

- Added `models/demos/blackhole/qwen36/tt/precision.py` (env to dtype and fidelity, read at import, invalid value raises). Replaced the hard-coded dtypes and `MathFidelity.LoFi` in `mlp.py`, `attention/weights.py`, `gdn/weights.py`, `gdn/gated_deltanet.py`, `attention/gated_attention.py`. `git diff --stat`: 6 files, 63 insertions, 22 deletions (precision.py tracked with `git add -N`). No behaviour change with the environment unset: every default equals the value it replaced.
- The stage 3 perf probe process that was running had imported the old modules before this edit; any new process imports the new ones and, with the env unset, gets identical constants.

## 20:44 to 20:47: harness and subset

- `scripts/dtype_sweep.py` and `scripts/dtype_sweep_summary.py` written; `py_compile` clean; the summary script exercised on a synthetic CSV in the scratchpad (Pareto marking, 1.0 pp window, 0-flip filter, 2 percent time tie broken by max_dp, incomplete rows listed).
- `ln -sfn /home/hous/dev/kev/tt_cache /home/hous/dev/kev/tt_cache_baseline` so the baseline reuses the stage 1 cache.
- `hostrun python scripts/dtype_sweep.py --materialize-only` (host only, no ttnn import): 200 records, 291 rows, 171,555 tokens, longest row 5024, dropped none, -> `/home/hous/dev/kev/reports/sweep/subset200.jsonl` (1.0 MB). 15 rows are identical to reference rows.
- `df -h /`: 386 GB available. Disk estimate per cache in the README (new total about 53 GB).

## 20:48: smoke queued

- Lock held by the stage 3 probe (`perf_probe.py --profile`, pid 2510418, `timeout 2400`). Smoke queued behind it:
  `devrun timeout 900 python models/autoports/jaredpalmer_kev_9b/scripts/dtype_sweep.py --variant baseline --rows-only --device-id 2 > /home/hous/dev/kev/logs/stage4_smoke_baseline_rows.log`.
- Expected: 29/29, max_dp 0.1024, mean_dp 0.0366 within 1e-3 of the stage 1 control.

## 20:48:18 to 20:48:57: smoke result (failed before any row ran)

- Log: `/home/hous/dev/kev/logs/stage4_smoke_baseline_rows.log`. Exit 1. Outputs set aside as `/home/hous/dev/kev/reports/sweep/baseline_rows_smoke_failed_2048.json` and `sweep_results_smoke_failed_2048.csv`.
- What worked: variant env applied (`QWEN36_*` at their defaults, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache_baseline`); device 2 opened with the stage 1 params; `KevEngine` built from the current signature (`traced=True, read_rows=32, kv_reserve_bytes=3221225472, max_state_len=65536` are new since stage 1; the harness passed `args_cls` and `max_state_len=8192`); all 403 weight files loaded as-is from `tt_cache_baseline/P150/tensor_cache_bfp8_kev_2b2a70cf` with the expected dtypes in the file names (`gate_proj`/`up_proj` `BFLOAT4_B`, `down_proj` and every projection `BFLOAT8_B`); `_fit_slots`: KV per slot 0.31 GiB, DRAM free 22.27 GiB after weights, 8 slots.
- What failed: the new engine default `traced=True` runs `_setup_traces()` in the constructor, which warms every bucket through the trace-path forward `Qwen36Model._forward_prefill_chunk` and then captures traces. The first warm-up forward threw in `ttnn.transformer.gated_delta_attn_seq` (GDN chunk kernel) 6 s after `_fit_slots`: `Statically allocated circular buffers in program 679 clash with L1 buffers on core range [0-0 - 0-0]. L1 buffer allocated at 1193984 and static circular buffer region ends at 1360896` (`tt_metal/impl/program/program.cpp:2525`). Python frames: `engine.py` `__init__` -> `_setup_traces` -> `_forward_body` -> `model.py:988 _forward_prefill_chunk` -> `layer.py:258` -> `gdn/decode.py:42 recurrent_forward` -> `ttnn_gated_deltanet.py:702` -> `ttnn_delta_rule_seq.py:723`. No reference row ran, so no agreement numbers exist from this run.
- Classification: stage 3 traced path, work in progress (the engine revision that ran did not yet log `warming bucket`; the revision read at 20:50 does, and its line numbers differ from the traceback). Not caused by the precision knob: the env values equal the old constants and the loaded files are the stage 1 cache. Not caused by the eager sweep path, which was never reached. The eager path (`traced=False`: `_run_segment` -> `_forward_prefill_chunk_masked`, `_read_rows`) is the stage 1 code the sweep is meant to measure.
- Fix in the harness: `build_engine` now passes `traced=False` whenever the constructor has a `traced` parameter; the CSV `status` keeps only the first line of an exception and the JSON keeps the full text.
- Not re-run: the task allowed one device smoke, and it is spent. A re-run of the rows-only baseline is the first command of the sweep list in the README, and it is the proof that 29/29, 0.1024, 0.0366 reproduce under the new engine's eager path.
- For the stage 3 owner: a `KevEngine(device)` with defaults under `l1_small_size=24576, num_command_queues=2, trace_region_size=0` fails in the trace warm-up with the L1 clash above, so any caller that does not pass `traced=False` (or a trace region and whatever L1 budget the traced path needs) cannot build the engine right now.

## 21:45 to 21:47: sweep agent start, smoke repeated

Agent: stage 4 sweep. Read `STATUS.md`, this directory, the datatype-sweep skill, `../optimized/README.md`. Engine signature now `KevEngine(device, args_cls=KevModelArgs, max_state_len=65536, n_layers=None, chunk_size=2048, snapshot_slots=8, traced=True, read_rows=32, kv_reserve_bytes=3<<30, matmul_policy=None)`; the harness passes `args_cls`, `max_state_len=8192`, `traced=False`. `matmul_policy=None` means the engine reads `KEV_MATMUL_POLICY` (default on), so the first sweep ran with the stage 3 `ttnn.linear` rebind active.

- Smoke, device 2, `devrun timeout 900`, log `/home/hous/dev/kev/logs/stage4_smoke_baseline_rows.log`: engine built in 27.9 s (eager), 29 rows in 12.8 s, 29/29 agree, max |dp| 0.10236984, mean |dp| 0.03664658, min PCC 0.976770, median 0.992665. Identical to the stage 1 control (`/home/hous/dev/kev/reports/stage1_control_merged.json`). Acceptance met; no harness fix was needed.
- Harness: `propagation_check` now also records `traced` and `matmul_policy` read back from the engine.
- `matplotlib` 3.11.2 is in the tt-metal venv; plots are generated by the summary script.
- `tests/test_engine.py` has no `l32` id for the reference test (`test_reference_records[device_params0]`), so the step 6 confirmation uses `-k reference_records`.

## 21:47 to 21:53: first full sweep, matmul policy on, three variants failed

Loop `baseline mlp_bfp8 all_bfp8_hifi2 all_bfp8_gdnfp32`, logs now `/home/hous/dev/kev/logs/stage4_sweep_<variant>_policy_on.log`, outputs `/home/hous/dev/kev/reports/sweep/<variant>_policy_on.json`, CSV rows renamed `<variant>_policy_on`.

- `baseline_policy_on` (21:47 to 21:50, rc 0): reference 29/29, 0.10237 / 0.03665, min PCC 0.97677; subset acc 0.8213, Brier 0.2481, ECE 0.0822, NLL 0.4555; warm row 0.382 s (0.699 ms per token), engine load 27.1 s. By suite: hard-v1 0.8246 (114 rows), devtools-v1 0.7349 (83), documents-v1 0.8936 (94).
- `mlp_bfp8_policy_on`, `all_bfp8_hifi2_policy_on`, `all_bfp8_gdnfp32_policy_on` (rc 1 each, within 1 to 2 s of the first reference row): `TT_THROW: Statically allocated dataflow buffers on core range [0-0 - 10-7] grow to 1891328 B which is beyond max L1 size of 1572864 B` (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2682`), raised from `engine.py:87 policy_linear` -> `_original_linear(..., program_config=cfg)` called by `mlp.py:208` (`w1`, the gate projection, `activation="silu"`), reached through `_run_segment` -> `_forward_prefill_chunk_masked`. The stage 3 `MATMUL_POLICY` 2D program configs were tuned with bfp4 gate/up weights; a bfp8 in1 block is about 1.9x the bytes of a bfp4 block (1088 versus 576 B per tile), so the gate/up entries with `in0_block_w=16` and `per_core_N=35` no longer fit L1 (estimate, not measured per entry). `QWEN9B_MLP_DOWN_AUTO=1` does not apply: the failing matmul is `w1`, and `policy_linear` discards the caller's `program_config` for shapes in the policy. The caches were built and are intact (`tt_cache_mlp_bfp8`, `tt_cache_all_bfp8`: 9.8 GB each, `du -sh`).
- Decision: the policy is a process-level env switch (`KEV_MATMUL_POLICY`), outside the files this agent may edit. The sweep is re-run with the policy off for every variant so time is comparable, the policy-on baseline is kept as evidence of the production path as of stage 3, and the policy incompatibility with bfp8 gate/up is reported to the stage 3 owner.
- Harness: `--matmul-policy {0,1}` (sets `KEV_MATMUL_POLICY`), `--down-auto` (sets `QWEN9B_MLP_DOWN_AUTO=1`), `--tag` (suffix on the variant name and JSON), CSV column `matmul_policy` (read back from the engine). Existing CSV rewritten with the column (`sweep_results_before_policy_off.csv` is the pre-rewrite copy).

## 21:53 to 22:18: full sweep, matmul policy off, six variants complete

Logs `/home/hous/dev/kev/logs/stage4_sweep_<variant>.log`, JSON `/home/hous/dev/kev/reports/sweep/<variant>.json`, each `devrun timeout 3600`, device 2, `--matmul-policy 0`. Engine log line confirms `traced=False matmul_policy=False`. `df -h /` before the bf16 variants (22:09): 358 GB available. Neither bf16 variant needed `QWEN9B_MLP_DOWN_AUTO=1`; both built and ran at the first attempt.

| variant | rc | load s | DRAM free after weights (max_state 8192) | ref flips (margin / near-tie) | max dp | mean dp | min PCC | acc | Brier | ECE | warm row s | cache GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 27.5 | 22.27 GiB | 0 (0 / 0) | 0.1024 | 0.0366 | 0.9768 | 0.8213 | 0.2481 | 0.0856 | 0.577 | 8.86 |
| mlp_bfp8 | 0 | 47.4 (cache built in the policy-on run) | 20.77 GiB | 1 (0 / 1) | 0.0876 | 0.0264 | 0.9873 | 0.8316 | 0.2428 | 0.0778 | 0.579 | 10.47 |
| all_bfp8_hifi2 | 0 | | 20.77 GiB | 0 (0 / 0) | 0.0894 | 0.0271 | 0.9883 | 0.8247 | 0.2435 | 0.0713 | 0.598 | 10.47 (shared) |
| all_bfp8_gdnfp32 | 0 | 28.3 | 20.77 GiB | 1 (0 / 1) | 0.0876 | 0.0266 | 0.9873 | 0.8316 | 0.2429 | 0.0779 | 0.578 | 10.47 (shared) |
| mlp_bf16 | 0 | 43.9 (cache built) | 16.55 GiB | 0 (0 / 0) | 0.0722 | 0.0246 | 0.9888 | 0.8316 | 0.2433 | 0.0725 | 0.604 | 15.0 |
| all_bf16 | 0 | 38.9 (cache built) | 13.67 GiB | 1 (0 / 1) | 0.0661 | 0.0243 | 0.9893 | 0.8247 | 0.2439 | 0.0677 | 0.604 | 16.95 |

- The only flipped reference row in any variant is `0:choice` (337 tokens, 6 options). fp32: 0.2895 / 0.3211 for options 0 / 1, a top-2 margin of 0.0315. Baseline agrees by 0.8 pp (0.2790 / 0.2868); mlp_bfp8 and all_bfp8_gdnfp32 give 0.2779 / 0.2561; all_bf16 gives 0.2877 / 0.2833. The ECE for the policy-off baseline (0.0856) differs from the policy-on baseline (0.0822) because the matmul blocking changes rounding slightly; accuracy and Brier are equal to 4 digits.
- `QWEN_GDN_FP32_STATE` is read at call time (`ttnn_delta_rule_seq.py:201`) and the typecast runs, but the kev engine copies the new state into the persistent bf16 buffer (`gdn/decode.py:100-107` `ttnn.copy(new_state, gdn.recurrent_state)`; buffer dtype bf16 at `qwen36/tt/model.py:2678`), so the fp32 state is re-quantised to bf16 between chunks and `all_bfp8_gdnfp32` tracks `mlp_bfp8` to within 0.0002 mean |dp|. The knob has no useful effect on this engine.
- `mlp_bfp8_nofp32acc` was not run: neither the harness nor the engine exposes an fp32-accumulate knob; every `compute_kernel_config` in `qwen36/tt/mlp.py`, `attention/gated_attention.py`, `gdn/gated_deltanet.py` hard-codes `fp32_dest_acc_en=True`, and the skill forbids writing a config field the model would ignore. Stage 3's `matmul_sweep.json` has the per-matmul cost of fp32 accumulation for the owner of the qwen36 knobs.

## 22:03: selection-rule amendment (recorded verbatim)

> Rule amendment from the orchestrator, record it verbatim in the work log and summary: an argmax flip counts against a variant only when the fp32 reference's top-2 probability margin on that question is at least 0.05. A flip on a near-tie row (margin under 0.05, like the 0.2895 vs 0.3211 case) is reported but not disqualifying, because it measures the tie, not the variant. Keep max |dp| and mean |dp| as tie-breakers, and keep the 1.0 pp subset-accuracy window. Also report, for every variant, the count of "margin >= 0.05" flips and the count of near-tie flips separately. Continue the sweep.

Implemented in `scripts/dtype_sweep_summary.py`: `classify_flips` reads each variant's `reference_rows`, classes a flip by the fp32 top-2 margin (`FLIP_MARGIN = 0.05`), reports `flips_margin` and `flips_neartie` per variant, requires `flips_margin == 0` for eligibility, and breaks time ties by (max |dp|, mean |dp|, time). Rows named `*_policy_on` are shown in the table but excluded from the front and the selection (different measurement regime).

## 22:18: selection

Best accuracy 0.8316 (mlp_bfp8, all_bfp8_gdnfp32, mlp_bf16). Window 1.0 pp: every variant except the baseline (0.8213 is 1.03 pp below). Margin flips: none anywhere. Fastest eligible: mlp_bfp8 0.5786 s and all_bfp8_gdnfp32 0.5785 s (within 2 percent; 0.1 ms apart). Tie-break: max |dp| equal (0.0876, same row), mean |dp| 0.0264 versus 0.0266, so `mlp_bfp8` is selected. It is also the simpler configuration (no extra env knob, and the GDN knob is inert in this engine).

## 22:19 to 22:26: traced production-path probes (step 3)

`scripts/perf_probe.py --traced --trace-region 1073741824 --device-id 2` (`--matmul-policy` only for the policy-on baseline), variant env from `dtype_sweep.py --print-env`, each under `devrun timeout 1800`; JSON `/home/hous/dev/kev/reports/sweep/perf_probe_<name>.json`, logs `/home/hous/dev/kev/logs/stage4_probe_<name>.log`. A first launch of the four probes failed instantly because the loop used `set -- $spec` under zsh (no word splitting); relaunched with explicit arguments.

| name | tail 128 / 256 / 512 / 1024 / 2048 ms | state 2048 / 2392 ms | short card new / cached | long card new / cached | build s | slots | trace MiB |
|---|---|---|---|---|---|---|---|
| baseline_policy_on | 105.0 / 150.7 / 266.4 / 476.3 / 916.7 | 898.7 / 1048.9 | 605.9 / 605.4 | 1528.4 / 524.9 | 37.2 | 8 | 273.8 |
| baseline (policy off) | 107.0 / 167.2 / 468.0 / 645.8 / 1528.0 | 1510.2 / 1675.9 | 619.1 / 618.0 | 2152.9 / 535.7 | 38.3 | 8 | 273.1 |
| mlp_bfp8 (policy off) | 107.0 / 167.3 / 468.3 / 646.9 / 1537.2 | 1519.8 / 1685.9 | 619.6 / 619.9 | 2163.0 / 537.6 | 38.7 | 8 (19.76 GiB free) | 273.1 |
| mlp_bf16 (policy off) | 114.0 / 174.0 / 509.7 / 673.5 / 1581.2 | 1563.7 / 1736.9 | 659.6 / 660.0 | 2247.5 / 570.6 | 37.6 | 6 (15.54 GiB free) | 272.1 |

Chip 2 reproduces the stage 3 chip 0 policy-on numbers within 1 ms. The two 0.8316 variants with distinct weights were probed (mlp_bfp8, mlp_bf16); all_bfp8_gdnfp32 shares mlp_bfp8's weights and engine path.

## 22:26 to 22:28: summary written (step 4)

`hostrun python scripts/dtype_sweep_summary.py --write`: `selected_precision_config.json`, `sweep_results.csv`, `sweep_results.json`, `pareto_table.md`, `top1_perf_pareto.png`, `fp32_agreement_perf_pareto.png` (matplotlib 3.11.2 in the venv). First render had the selected point hidden under the coincident gdnfp32 point and colliding labels; fixed (coincident grouping, collision offsets, selected drawn last, tight bbox, 0.5 percent time tolerance in the dominance test so the two 0.1 ms-apart variants are both on the front) and re-rendered. No `top5` plot: kev has no top-5 token metric; the agreement plot is the second view.

## 22:28 to 22:30: engine default switched to the selected config (step 6)

- New `tt/precision_defaults.py`: profiles `selected` (bfp8 / bfp8 / bfp8, LoFi, GDN state bf16, `QWEN_SDPA_BF8=0`, `KEV_MATMUL_POLICY=0`) and `baseline` (bfp4 gate / up, policy on), chosen by `KEV_PRECISION` (default `selected`), applied with `os.environ.setdefault` at import; `cache_tag()` returns `gu-<x>_dn-<y>_pj-<z>` from the active knobs. `tt/loader.py` imports it before `models.demos.blackhole.qwen36.tt.model_config` and `KevModelArgs.weight_cache_path` appends `_<tag>`. `tt/engine.py`, `tt/server.py`, `tests/test_engine.py` and the qwen36 defaults are untouched. Host check: after `from ...tt.loader import KevModelArgs`, the qwen36 `precision` module reads bfp8 / bfp8 / bfp8 / LoFi, `weight_cache_path` ends in `_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8`, `KEV_MATMUL_POLICY=0`.
- `KEV_MATMUL_POLICY=0` is part of the selected profile because the stage 3 policy overflows L1 with bfp8 gate / up (section 21:47). Production cost until the policy is re-tuned: long card 2163 ms versus 1528 ms, short card 620 versus 606 ms (probe table above).
- Caches: `tt_cache_mlp_bfp8/P150/tensor_cache_bfp8_kev_2b2a70cf` moved to `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8` (the server's default root; `mv` on one filesystem), symlinks left under both names in `tt_cache_mlp_bfp8/P150`; `tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp4_dn-bfp8_pj-bfp8 -> tensor_cache_bfp8_kev_2b2a70cf` for the baseline profile; the all_bfp8, mlp_bf16, all_bf16 directories renamed to their tagged names with an old-name symlink each.
- `doc/context_contract.json`: `functional.weight_precision` updated and a `stage4_precision` block added (KV dtype unchanged, 8 slots at 65536 still fit, policy note).
- Device confirmation, 22:29 to 22:30, chip 2: `devrun timeout 1800 python -m pytest models/autoports/jaredpalmer_kev_9b/tests/test_engine.py -k reference_records --device-id 2 -s` with only `HF_MODEL`, `KEV_RUN`, `MESH_DEVICE=P150`, `HF_HUB_OFFLINE=1`, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache` set (no `QWEN36_*`, no `KEV_*`), log `/home/hous/dev/kev/logs/stage4_default_reference_records.log`. Engine log: `cache=/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8 traced=True matmul_policy=False`; 403 cache files loaded from that directory, none written (no rebuild). Result line: `rows=29 min_pcc=0.987337 argmax_agree=28/29 max_abs_prob_diff=0.087636 mean_abs_prob_diff=0.026433 traced=True`, equal to the eager sweep numbers for mlp_bfp8 (0.987337 / 28 / 0.087636 / 0.026433) and the traced engine therefore matches the eager one on this config. The test then fails its own `assert agree == len(rows)` (`test_engine.py:193`, `28 == 29`) on the near-tie row `0:choice`; under the amended rule that flip is reported, not disqualifying. The test file is owned by stage 3 and was not edited; its owner should apply the margin rule or the assertion stays red under the selected default. `-k "reference_records and l32"` selects nothing (no `l32` id on that test), hence `-k reference_records`.

## Disk

`df -h /`: 378 GB available at 21:45, 358 GB at 22:09 (before the bf16 variants), 328 GB at 22:18. Written by the sweep: 49.6 GB of weight caches (`tt_cache/.../_gu-bfp8_dn-bfp8_pj-bfp8` 10.47 GB, `tt_cache_all_bfp8` 9.8 GB, `tt_cache_mlp_bf16` 14 GB, `tt_cache_all_bf16` 16 GB by `du -sh`) plus logs and reports under 200 MB. Non-selected caches (about 40 GB) left in place for the stage review.
