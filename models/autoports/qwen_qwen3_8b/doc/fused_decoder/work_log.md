# Qwen3-8B fused decoder: work log

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), RoPE (rotary position embedding), SDPA (scaled dot-product
attention), MLP (multi-layer perceptron), DRAM (device dynamic RAM), L1
(per-core local memory), ET (Eastern Time).

Stage 2 (fused decoder) for `Qwen/Qwen3-8B` in
`/home/hous/dev/tt-metal-wt-qwen3` (branch `hous/qwen3-8b-autoport`, started
at HEAD `64f03408f1`). Hardware: Blackhole P300, one chip per process
(`TT_VISIBLE_DEVICES=0` or `1`), 1x1 mesh, grid 11x10. Times are ET.

## 1. Discovery (2026 Oct 28, 00:07 to 00:25 ET)

- Skills read from `/home/hous/dev/skills/plugins/tt-model-bringup` (plugin
  0.1.21): `model-bringup`, `graph-fusing`, `tt-device-usage`,
  `tt-enable-tracing`, `stage-review`; the stage prompt
  `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_02-fused-decoder.txt`.
  The Skill tool resolved `model-bringup` to
  `~/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/`; that
  copy differs from the dev root and was not used.
- `scripts/environment.py` validated `TT_AUTODEBUG_ROOT=/home/hous/dev/skills/plugins/tt-autodebug`
  (tt-autodebug 0.1.8). `timeout 60 tt-smi -ls --local`: four p300c chips.
  Mesh smoke on chip 0: `MESH_SMOKE_OK`.
- Stage-1 artifacts read: `doc/functional_decoder/README.md`, `work_log.md`
  (the padded-K SDPA read and the 8-wide decode Q-placement rule),
  `perf/full_attention/{prefill,decode}.txt`, `tt/functional_decoder.py`,
  `tests/test_functional_decoder.py`.

## 2. Step 1 and 2 of the skill: op inventory and op-sequence table

Op inventory explored in tt-metal: `ttnn/cpp/ttnn/operations/experimental/`
(`minimal_matmul`, `matmul_decode`, `paged_cache/fused_update_cache`,
`transformer/{rotary_embedding_llama_fused_qk, dit_minimal_matmul_addcmul_fused,
nlp_*}`), `normalization/{layernorm, rmsnorm}`, `transformer/sdpa_decode`,
`models/common/modules/{mlp,rmsnorm,rope,attention}`,
`models/tt_transformers/tt/attention.py` (Qwen3 Q/K norm uses the same
reshard hack, "Hack until RMSNorm supports height-sharded output config"),
`models/demos/blackhole/qwen36/tt/` (`minimal_matmul` with `fuse_swiglu`),
`models/demos/deepseek_v3_b1/micro_ops/`, `models/tt_dit/utils/tensor.py`
(`prepare_for_fused_swiglu`).

Decode op table (stage-1 perf, batch 1, 35 ops, 1,266 us): rms_norm (85 us,
1 core), QKV matmul 123, create heads 34, [s2i, rms_norm, i2s] x2 for Q and K,
typecast, [embedding, tilize, transpose, i2s] x2 for cos/sin, rope x2,
paged_update_cache x2 (8 each), SDPA 41, i2s, concat heads, s2i, O matmul 81,
slice, add, rms_norm (84 us, 1 core), gate 253, up 254, mul(silu) 12,
down 237, add. Prefill op table (21 ops, 10,956 us): rms_norm 100, QKV 1,607,
create heads 88, q/k norm 46 + 19, slice x2, rope 111 + 50, fill x2, SDPA 248,
concat 60, O 1,086, add 56, rms_norm 97, gate 1,941, up 1,940, mul 210,
down 3,213, add 57. Every "SLOW" row is a HiFi4 `ttnn.linear`.

## 3. Step 3 and 4: experiments (chips 0 and 1, 00:15 to 00:33 ET)

Scratch scripts under `/tmp/qwen3_fused/` (copied to `experiments/` with
their logs). Random inputs, timing by trace replay of 4 to 20 iterations.

| Experiment | Finding |
| --- | --- |
| `exp1_contracts.py` (batch 1) | E1 sharded rms_norm on height-sharded heads: TT_FATAL layernorm_device_operation.cpp:167. E2 create-heads always height-sharded. E3 rms_norm interleaved in, height-sharded out: PCC 1.0. E4 SDPA sharded output: TT_FATAL sdpa_decode_device_operation.cpp:414 (GQA). E5 linear on width-sharded concat: equal, 85.8 vs 85.0 us. E6 row-major gather + reshape + tilize: bit-equal, 11.2 vs 13.0 us; sharded embedding output: shard alignment TT_FATAL. E7 paged_fused_update_cache: TT_FATAL inputs must not overlap. E8 linear bias as batch-1 residual: PCC 0.9999998, 85.4 vs 88.3 us. E9 MLP variants at 32 and 1024 rows: silu-in-matmul, packed gate/up + slices, packed + swiglu: none faster in decode. |
| `exp2_adapt.py` (batch 1 and 32) | variant (a) K on separate cores still overlapped; variant (b) V reshard + fused update: bit-equal outputs and caches, attention subgraph 321.7 -> 314.3 us (batch 1), 417.6 -> 416.4 us (batch 32). Gate/up packing same weights: bit-equal, decode 760.5 -> 762.8 us, prefill 1024 7,377 -> 7,218 us. |
| `exp3_inf.py` | the batch-1 `inf` seen in exp1/exp2 comes from uninitialized padding rows (users 1..31) of the concat-heads output; the SDPA output is finite and only row 0 is consumed. `overlap_qk_coregrid=False` has no effect with an interleaved QKV input (validation line 83). |
| `exp4_minimal.py` | minimal_matmul and addcmul fusion: prefill 1024 rows QKV 1,619 -> 556 us, O+res 1,154 -> 390, down+res 3,288 -> 1,062, swiglu-fused 4,173 -> 2,478 (HiFi4), PCC vs fp32 unchanged; decode 32 rows: every variant 1.7x to 2.1x slower than ttnn.linear. |
| `exp5_decode_configs.py` | 48 MinimalMatmulConfig settings per decode op: best QKV 213 vs linear 128 us, swiglu 734 vs 522, down 401 vs 241. `matmul_decode` needs L1 width-sharded operands (TT_FATAL line 35). |
| `exp6_view.py` | tiled reshape [1,1,32,24576] -> [1,384,32,64] is not a view (new buffer); interleaved-weight decode MLP 989 vs 757 us. |
| `exp7_prefill_mlp.py` | separate-weight `minimal_matmul` x2 + mul beats fuse_swiglu and linear at 1024 rows (2,317 / 2,654 / 4,185 us) and 4096 rows (8,658 / 9,724 / 29,756 us); QKV, O+res, down+res 3x faster at both sizes. |
| `exp8_crossover.py` | ttnn.linear wins at <= 128 rows for every projection, mixed at 192, minimal/fused wins at >= 256: threshold 256. |

Decisions: separate gate/up weights for both modes (no layout conflict, no
weight duplication); prefill uses `minimal_matmul` and the addcmul-fused
residual projections from 256 rows; decode keeps `ttnn.linear` and applies
the structural rewrites E3, E6, E7(b), E8.

## 4. Implementation (00:33 to 00:36 ET)

- `tt/fused_decoder.py`: `FusedDecoder(FunctionalDecoder)` overriding
  `_prefill_chunk`, `_attention_prefill`, `_mlp_prefill`,
  `_prefill_matmul(_residual)`, `decode_forward`, `_attention_decode`,
  `_decode_head_norm`, `_decode_rope`, `_mlp_decode`,
  `_decode_matmul_residual`; `_decode_v_update_memory_config` places V on the
  rectangle below the Q/K rectangle; `ones_row` is a device constant.
- `tests/test_fused_decoder.py`: binds the stage-1 gates to `FusedDecoder`
  through a module-scoped fixture (decoder factory, JSON writer into
  `doc/fused_decoder`, provenance with both decoder hashes), adds the
  fused-vs-functional on-device equivalence test, the repeated-run stress
  test, the kernel-selection and V-rectangle host tests and the fused hot-path
  audit. First device run failed because `LayerFixture.inputs` reads the real
  token ids from the base artifact directory; fixed by patching only the
  JSON writer.

## 5. Runs and the one reverted rewrite (00:36 to 00:44 ET)

- Fresh "before" profile of the functional decoder on chip 1 (Tracy, same
  harness as stage 1): decode 1,265 us / 35 ops, prefill 10,966 us / 21 ops;
  tables in `perf/before/`.
- Fused default suite on chip 0: 12 passed, 4 skipped (137.9 s). Fused
  profile on chip 1: prefill 4,816 us / 19 ops, decode 1,255 us / 28 ops, but
  the O matmul on the width-sharded concat input ran on 32 cores (85 us)
  versus 64 cores (81 us) before. E5 reverted: `sharded_to_interleaved` kept
  before the O projection.
- Context tests on chip 1 (3 passed, 40 s); tracker run on chip 0 (5 passed,
  86 s); positive control raised as expected.
- Pre-commit hooks run on both new files (black reformatted the decoder);
  final hashes `fused_decoder.py 330c9fe1e680…`, `test_fused_decoder.py
  63292be2c086…`.

## 6. Final battery on the committed-content files (13:24 to 13:35 ET)

A tool-call delay of about 12.7 hours (04:44 to 17:24 UTC) separated the
pre-commit step from the final battery; no device work ran in that gap and
no process was left on a chip (checked with `ps` and `uptime`). The first
attempt at the battery failed before touching a device (zsh passed an
unquoted environment variable as one word, exit 127) and was re-issued with
the environment inline.

| Step (ET) | Command (environment as in README) | Result |
| --- | --- | --- |
| 13:24 default suite, chip 0 | `TT_VISIBLE_DEVICES=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k 'not host and not audit' -p no:cacheprovider` | 12 passed, 4 skipped (128.9 s) |
| 13:26 tracker, chip 0 | `TT_VISIBLE_DEVICES=0 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN3_EVIDENCE_SUFFIX=tracked python -m pytest -q -s ... -k 'test_traced_decode or test_repeated_runs_stress'` | 5 passed (86.5 s); `trace_alloc_tracking/console.log` |
| 00:44 positive control, chip 0 | `TT_METAL_TRACE_ALLOC_TRACKING=1 ... python doc/fused_decoder/trace_alloc_tracking/positive_control.py` | raised "Found 1 device buffer(s) still alive before trace replay" |
| 13:24 Tracy, chip 1 | `TT_VISIBLE_DEVICES=1 QWEN3_PERF=1 QWEN3_PERF_TRACE_WARMUPS=2 QWEN3_PERF_TRACE_ITERATIONS=1 python -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_fused -m pytest -q -s 'models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py::test_fused_decoder_perf_profile[blackhole-batch1-device_params0-mesh_device0]' -p no:cacheprovider` | 1 passed; CSV `generated/profiler/qwen3_fused/reports/2026_10_28_17_33_11/`; tables in `perf/full_attention/` |
| 13:33 context tests, chip 1 | `TT_VISIBLE_DEVICES=1 QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s ... -k advertised_context` | 3 passed (39.4 s) |
| 13:34 watcher, chip 1 | `TT_VISIBLE_DEVICES=1 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8b/doc/fused_decoder/watcher QWEN3_EVIDENCE_SUFFIX=watcher python -m pytest -q -s ... -k 'test_traced_decode or test_repeated_runs_stress'` | 5 passed (81.8 s); `watcher/summary.json` clean, 6 dumps, 0 fatal-pattern matches |
| 13:37 context contract | `python /home/hous/dev/skills/plugins/tt-model-bringup/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B --stage fused-decoder --require-contract --strict-caps` | see section 7 |
| 00:36 host tests | `python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_fused_decoder.py -k 'host or audit' -p no:cacheprovider` | 10 passed |

Every evidence JSON under `doc/fused_decoder` (26 files) records
`fused_decoder_sha256 330c9fe1e680…` and `test_sha256 63292be2c086…`, the
files as committed. Console logs of the runs live in `/tmp/qwen3_fused/logs/`
(scratch, not committed) except the tracker console log, which is committed.

Two chips were used concurrently (one process per chip) as the environment
permits; watcher and profiler never shared a process or a chip.

Host-work counters for the traced decode loop (`test_traced_decode`): one
capture per batch size, three measured replays, token and current_pos
refreshed by `ttnn.copy_host_to_device_tensor` only when the test changes
them, page table never refreshed after setup, no synchronization inside the
replayed region. Positions advance on the host in this layer-level harness;
the generator stage owns on-device advancement.

## 7. Contract check, commits and review

Filled in below as each step completes.

- 13:37 ET context contract: `check_context_contract.py --model-dir
  models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B --stage
  fused-decoder --require-contract --strict-caps` printed `Context contract
  OK for models/autoports/qwen_qwen3_8b: target=40960, supported=40960 (full
  HF context)` (exit 0). `doc/context_contract.json` is unchanged.
- 13:39 ET stage commit `e646fde4cd` on `hous/qwen3-8b-autoport` (never
  pushed): `tt/fused_decoder.py`, `tests/test_fused_decoder.py`,
  `doc/fused_decoder/` (68 files; the two raw 1.4 MB Tracy CSVs stay
  untracked, the per-phase CSVs were force-added past the `*.csv` ignore).
- 13:40 to 13:56 ET stage review 1 (fresh xhigh subagent, read-only, reviewer
  mode): `more-work-needed`. Required work: (P1) the continuation window still
  runs the inherited per-token cache-write loop (27 groups, 351 of 381 ops,
  about 1.2 ms of 2.87 ms) which a batched `paged_update_cache` /
  `paged_fused_update_cache` over n pseudo-users could replace, and the README
  tilize/untilize claim was wrong for that window; (P2) the SDPA-decode kernel
  is 4 to 5 us slower in both fused profiles (44.6 and 46.2 us) than in both
  functional profiles (40.8 and 40.9 us) with the same program config and
  cores, unclassified; (P2) the module docstring still described the reverted
  concat-to-linear rewrite; (P2) work-log section 7 was empty and the watcher
  dump count (6) counted the "completed" lines (3 dumps). Other concerns:
  "same session" wording, continuation host time 5.45 to 5.77 ms (single
  sample, concurrent chip-0 run), concurrent use of chips 0 and 1 against the
  one-at-a-time rule, exp2 variant (b) V placement differs from the shipped
  rectangle at batch 32, `from_state_dict` bypasses the multiple-of-32 check
  on `prefill_minimal_matmul_min_rows`, batch-1 garbage padding rows should be
  stated in the limitations.
- Closed without hardware in the second commit (below): the docstring line,
  the "same session" wording, the dump count in `watcher/summary.json` and
  the README, the README limitations (batch-1 padding rows, 2y core-area
  requirement), this section. The decoder change is docstring-only
  (`git diff e646fde4cd HEAD -- models/autoports/qwen_qwen3_8b/tt/fused_decoder.py`);
  every evidence JSON and `watcher/summary.json` record the pre-change hash
  `330c9fe1e680…`, and no device run was repeated because the time budget was
  exhausted (the coordinator stopped new device work at 13:24 ET).
- Open after this stage (require device runs): the P1 batched continuation
  cache write with re-profiled continuation window and the four continuation
  tests; the P2 SDPA-decode A/B (two-op vs fused cache update, 5 measured
  replays) with an anomaly-ledger entry; the batch-32 decode before/after
  profile and the 224-row crossover point listed under hard-check gaps. The
  final recorded verdict for this stage is `more-work-needed`.

## 8. Budget note

The 6-hour budget started at 00:07 ET. Discovery, experiments, implementation,
the first green suite and the pre-commit step were done by 00:44 ET. A
12.7-hour tool-call delay (00:44 to 13:24 ET) separated the pre-commit step
from the final battery; no device work ran during it. The battery, docs,
commit and review took 13:24 to 13:56 ET.
