# Qwen3-8B datatype sweep: work log

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), LM (language model), SDPA (scaled dot-product attention), DRAM
(device dynamic RAM), BFP8/BFP4 (block floating point, 8- and 4-bit
mantissa), LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), TTFT (time to
first token), t/s/u (tokens per second per user), CCL (collective
communication library), ET (Eastern Time).

Stage 8 (datatype sweep) for `Qwen/Qwen3-8B` in
`/home/hous/dev/tt-metal-wt-qwen3` (branch `hous/qwen3-8b-autoport`, started
at HEAD `2e52cebf61`). Hardware: one Blackhole P300 chip,
`TT_VISIBLE_DEVICES=2` only (chips 0 and 1 wedged, chip 3 stale sysmem
mapping, all pending an operator reset), 1x1 mesh, p150 mesh graph
descriptor. Build and `python_env` shared from `/home/hous/dev/tt-metal`.
Cap: 8 hours from 10:00 ET Oct 30 (ends 18:00 ET). Stages 4 and 5 were
skipped by the operator; one chip is the target topology, so there is no CCL
dtype to sweep (`ccl_dtype: null` in every config).

Every device command runs from the checkout root through
`doc/full_model/logs/stage_env.sh 2 <command>` (the brief's environment)
under `timeout`. `PY` below is `/home/hous/dev/tt-metal/python_env/bin/python`;
`D` is `models/autoports/qwen_qwen3_8b/doc/datatype_sweep`, `L` is `$D/logs`.

## 1. Discovery and startup (09:55 to 10:05 ET)

- Skills read from disk (`/home/hous/dev/skills/plugins/tt-model-bringup`,
  0.1.21): `datatype-sweep`, `tt-device-usage`, `qualitative-check`,
  `stage-review`, `model-bringup` (startup),
  `vllm-integration/references/prefill-features.md`; the stage prompt
  `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_08-datatype-sweep.txt`.
- Startup: `scripts/environment.py` exit 0 (`TT_MODEL_BRINGUP_ROOT`,
  `TT_AUTODEBUG_ROOT=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6`,
  enabled per `installed_plugins.json`); `timeout 60 tt-smi -ls --local`
  lists four p300c chips; `fuser /dev/tenstorrent/2` shows no holder.
- 09:58 ET mesh smoke on chip 2 (`doc/full_model/logs/mesh_smoke.py`, 180 s
  timeout): open, add, 53 MB BFP8 upload, close, exit 0
  (`logs/mesh_smoke_chip2.log`).
- Stage 7 evidence read: `doc/optimized_full_model/{README,work_log}.md`,
  `doc/full_model/README.md`, `doc/context_contract.json`,
  `readiness/readiness_stage7_chip2.json`, the stage-3 dtype ledger
  (`doc/optimized_decoder/README.md` "Rejected candidates", work log 3.8,
  3.9, 3.11, 8.4). Reference `bringup/references/qwen3_8b_aime24_chat.refpt`:
  AIME24 prompt 0, chat template, 158 prompt tokens, 100 generated tokens,
  top-100 captured. It meets the "100 generated tokens" requirement, so it is
  reused, not regenerated.
- Baseline policy (stage 7, `DEFAULT_POLICY` + `DEFAULT_TERMINAL_POLICY`):
  BFP8 LoFi QKV/O/gate/down (1 DRAM reader), BFP4 LoFi up (3 readers), BF16
  activations and residual, BF16 KV cache, HiFi4 fp32-acc norms and SDPA,
  BF16 embedding, BFP8 LM head (LoFi decode, HiFi2 fp32-acc prefill), BF16
  decode logits into the common sampler, fp32 prefill logits (readiness).
  Stage 7 readiness on this policy: prefill check top-1 0.990 / top-5 1.000 /
  top-100 1.000; teacher forcing top-1 0.980 / top-5 1.000 / top-100 1.000,
  TTFT 48.24 ms, 46.76 t/s/u.
- Acceptance bar (no user bar given, skill default): top-1 >= 0.90, top-5 >=
  0.98, top-100 at the readiness expectation (1.000, unchanged).

### 1.1 Gap found before the sweep: no consumed precision artifact

`build_generator` constructed `DEFAULT_POLICY` and `TerminalPolicy` from
code, with environment overrides for the LM head and the prefill kernel
table only; the KV-cache dtype was a constructor kwarg no runner could set,
and the model applied one policy to every layer (no layer exceptions). The
skill requires a config artifact that the construction path consumes by
default, with layer exceptions and fidelities, so the plumbing was added
first (section 2).

## 2. Precision config plumbing (10:05 to 10:35 ET, host only)

- `tt/precision.py` (new): `load_precision_config` reads
  `QWEN3_PRECISION_CONFIG=<path>`, else `doc/datatype_sweep/selected_precision_config.json`
  when it exists, else the built-in stage-7 baseline (`baseline_config()`);
  `decoder_policy(config, layer_idx)` builds the `OptimizedPolicy` for one
  layer (weight groups `qkv`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  with `dtype`, `fidelity`, `fp32_acc`, `readers`, `in0_block_w`; norm and
  SDPA fidelity; prefill fp32 accumulation and BFP8 prefill fidelity) with
  that layer's `layer_exceptions` applied; `layer_policies` collects the
  exceptions; `terminal_policy` builds the `TerminalPolicy` (embedding,
  final norm, LM head dtype and fidelities, prefill logits dtype);
  `kv_cache_dtype`; `describe_runtime` reads every dtype and fidelity back
  from the live weight tensors, compute-kernel configs, program configs
  (readers) and the cache tensors; `check_propagation` compares that with
  the config.
- `tt/model.py`: `Qwen3Model(layer_policies=...)` builds the named layers
  with their own policy (residual core count must match);
  `weight_bytes()` sums per layer; `kv_bytes_per_token(dtype)`.
- `tt/generator.py`: `build_generator` loads the config, logs one line
  (`precision config <id> from <path>: ...`), constructs policy, layer
  policies, terminal policy and `kv_cache_dtype` from it (the existing
  `QWEN3_*` environment overrides still apply on top), and stores
  `generator.precision_config` / `precision_config_path`; `teardown()` logs
  `runtime counters at teardown: {...}` so every readiness log carries the
  trace counters; `max_tokens_all_users(kv_cache_dtype=...)`.
- `tests/test_full_model.py`: provenance records the config id, path and
  layer exceptions.
- Candidate configs written by `logs/make_configs.py` to `configs/<id>.json`
  (table in section 3). Host check: every config loads, resolves per-layer
  policies and the terminal policy (`logs/make_configs.py` output).
- 10:02 ET propagation check on chip 2 (`logs/propagation_check.py`, two
  real layers, `QWEN3_PRECISION_CONFIG` per config, 1200 s timeout, 34 s
  total): `propagation_check_reduced.json`, `logs/propagation_check_reduced.log`.
  For `c5_mlp_bfp4_inner` the live layer 0 reads back BFP8 gate/down (the
  exception) and layer 1 BFP4 gate/up/down with readers 3/3/2; `c3_kv_bfp8`
  reads back `bfloat8_b` K and V caches; `c9_lm_head_bfp4` a BFP4 LM head;
  `c2_bfp8_hifi2` HiFi2 decode compute on QKV/O/gate/down and LoFi on up;
  `problems=[]` for all four, 8 greedy tokens generated through the traced
  decode on each (`PROPAGATION_OK`).

## 3. Candidate matrix

| id | change against the baseline | why |
| --- | --- | --- |
| c0_stage7_baseline | none | refreshed baseline |
| c1_all_bfp8_lofi | up BFP8 LoFi | canonical conservative policy (stage-3 `ALL_BFP8_POLICY`) |
| c2_bfp8_hifi2 | HiFi2 decode on the BFP8 groups (QKV/O/gate/down) | BFP8 LoFi versus HiFi2 on the dominant decode projections (skill requirement) |
| c3_kv_bfp8 | BFP8 KV cache | yes/no switch from the skill; stage 3 rejected it on a short-prefix per-user bar |
| c4_up_bfp4_hifi2 | up BFP4 HiFi2 | BFP4+HiFi2 comparison for the selected BFP4 group |
| c5_mlp_bfp4_inner | gate/up/down BFP4 LoFi on layers 1 to 34 | the skill's coarse step 4 (MLP BFP4, first and last layer excluded) |
| c6_mlp_bfp4_all | gate/up/down BFP4 LoFi on every layer | step 6 (extend to the edge layers) |
| c7_gateup_bfp4_inner | gate/up BFP4 LoFi inner, down BFP8 | restore step 1 (down first) |
| c8_attn_bfp4_inner | QKV/O BFP4 LoFi inner | attention BFP4 (stage 3 layer-17 delta 0.9737) |
| c9_lm_head_bfp4 | LM head BFP4 LoFi | 0.66 GB of the 7.1 GB read per token |
| c10_mlp_bfp4_inner_hifi2 | c5 with HiFi2 on the BFP4 MLP groups | BFP4+HiFi2 comparison for the material BFP4 groups |
| c11_down_bfp4_inner | down BFP4 LoFi inner next to the BFP4 up | restore step 1 variant (gate back to BFP8) |

Readers for BFP4 rows follow stage 3 (`qkv` 3, `o_proj` 2, `gate_proj` 3,
`up_proj` 3, `down_proj` 2); BFP8 rows keep 1 reader (stage-3 3.4). Every
candidate keeps BF16 activations and residual (the sharded decode path), the
HiFi4 fp32-acc norms and SDPA, the LoFi BFP8 prefill fidelity and the split
sampler on BF16 logits.

## 4. Sweep on chip 2 (`logs/sweep_chain_chip2.sh`, 10:04 ET onward)

For each candidate: `run_prefill_check` then `run_teacher_forcing`
(`--mesh-device N150`, the AIME24 reference), each under `timeout 1500`,
`QWEN3_PRECISION_CONFIG=$D/configs/<id>.json`, logs `logs/<id>_{prefill_check,teacher_forcing}.log`,
sidecar `readiness_<id>_chip2.json` (commit, chip, exit codes, headline
numbers, the logged precision-config line and the runtime counters).

### 4.1 Chain 1 results (12 candidates, 10:04 to 10:45 ET, commit `2e52cebf61` plus the uncommitted plumbing of section 2)

Teacher forcing (TF) is the ranking metric: every TF row below replayed the
decode trace 99 times with 0 eager decode steps and 0 seeded eager sampling
steps (`runtime_counters` in the sidecars). Prefill check (PC) and TF share
the AIME24 reference (100 tokens).

| id | PC top-1/5/100 | TF top-1/5/100 | TTFT ms | TF decode t/s/u | bar | note |
| --- | --- | --- | ---: | ---: | --- | --- |
| c0_stage7_baseline | 0.990/1.000/1.000 | 0.980/1.000/1.000 | 48.28 | 46.70 | pass | equals stage 7 (0.980, 48.24 ms, 46.76) |
| c5_mlp_bfp4_inner | 0.940/1.000/1.000 | 0.970/1.000/1.000 | 45.55 | 51.58 | pass | +10.4 percent decode |
| c6_mlp_bfp4_all | 0.940/1.000/1.000 | 0.920/1.000/1.000 | 45.42 | 51.84 | pass | +0.5 percent over c5, top-1 down 5 points |
| c7_gateup_bfp4_inner | 0.970/1.000/1.000 | 0.950/1.000/1.000 | 46.88 | 49.96 | pass | down BFP8 restores nothing measurable, costs 1.6 t/s/u |
| c11_down_bfp4_inner | 0.960/1.000/1.000 | 0.960/1.000/1.000 | 47.00 | 48.09 | pass | |
| c8_attn_bfp4_inner | 0.960/1.000/1.000 | 0.950/1.000/1.000 | 47.77 | 48.82 | pass | stage 3 rejected on a layer-17 delta PCC of 0.9737; the full model passes the bar |
| c9_lm_head_bfp4 | 0.980/1.000/1.000 | 0.970/1.000/1.000 | 48.02 | 47.24 | pass | +1.2 percent |
| c1_all_bfp8_lofi | error | error | | | error | `TT_THROW` L1 overflow in the 512-row up-projection prefill kernel (section 4.2) |
| c2_bfp8_hifi2 | 0.990/1.000/1.000 | 0.970/1.000/1.000 | 48.30 | 35.15 | pass | HiFi2 on the BFP8 decode rows: 25 percent slower, no accuracy gain; LoFi kept |
| c4_up_bfp4_hifi2 | 0.990/1.000/1.000 | 0.980/1.000/1.000 | 48.26 | 46.38 | pass | BFP4 up at HiFi2: 0.7 percent slower, same accuracy; LoFi kept |
| c10_mlp_bfp4_inner_hifi2 | 0.940/1.000/1.000 | 0.970/1.000/1.000 | 45.62 | 50.28 | pass | BFP4 MLP at HiFi2: 2.5 percent slower than c5 at the same top-1; LoFi kept |
| c3_kv_bfp8 | 0.990/1.000/1.000 | 0.980/1.000/1.000 | 48.42 | 46.84 | pass | +0.3 percent at batch 1 (within run-to-run noise); the extra cache cases of section 6 follow |

Reading: the MLP groups carry the win (gate+up+down BFP4 LoFi is 3.0 ms
per token off the 21.4 ms decode step); restoring down (c7) or gate (c11)
to BFP8 gives back 1.6 to 3.5 t/s/u without a measurable accuracy gain on
this reference; attention BFP4 and the BFP4 LM head are independent smaller
wins that also pass; every HiFi2 variant is slower at equal accuracy.
Round 2 (section 4.3) combines the passing groups.

### 4.2 c1 failure and the dtype-aware prefill kernel row (10:35 to 10:46 ET)

`logs/c1_all_bfp8_lofi_prefill_check.log`: `prepare()` warming the 512-row
prefill raised `TT_THROW ... Statically allocated circular buffers on core
range [0-0 - 10-7] grow to 1891328 B which is beyond max L1 size of 1572864 B`
in `ttnn.linear` for the up projection. The stage-7 kernel table has one
row per role; the `up_proj` rows were probed with BFP4 weights (a BFP4 tile
is half a BFP8 tile) and its 512-row entry is the 2D config with
`in0_block_w` 16, which overflows L1 with BFP8 tiles (the BFP8-probed
`gate_proj` row at 512 is `minimal_matmul`). Fix in
`OptimizedPolicy.prefill_kernel(role, rows, dtype)`: for the gate and up
rows (one shape) the table row is chosen by the weight dtype (`up_proj`
for BFP4, `gate_proj` for BFP8); `_linear` passes the role's dtype. Effect
on chain 1: candidates with a BFP4 gate (c5, c6, c7, c10) ran their inner
gate rows on the BFP8-probed table before the fix; their decode numbers are
unaffected (decode uses the DRAM-sharded kernels), the final selected config
is re-measured on the fixed code (section 5), and c1 is rerun in round 2.
Rows with no same-shape BFP4-probed table (down, QKV, O under BFP4) keep
the BFP8-probed kernels, which are legal with the smaller tiles; a BFP4
re-probe of those prefill rows is a TTFT lever left to the optimized-vLLM
stage (section 9).

### 4.3 Chips 0 and 1 released (coordinator message read at 10:53 ET)

The operator reset chips 0 and 1 at 10:47 ET; the coordinator allows one
process per chip on chips 0, 1 and 2 (chip 3 stays off limits). Mesh smoke
on both (`logs/mesh_smoke_chip{0,1}.log`): `MESH_SMOKE_OK`, exit 0. Use
from here on: chip 2 keeps every ranking run (teacher-forcing decode t/s/u
is host wall time, so the numbers that rank candidates come from runs that
were the only device job on the host, or are re-measured that way at the
end; sidecars carry `timing_regime`); chips 0 and 1 take accuracy-only
work whose timing is not used: the short-prefix cases and the batch-32
same-input case (chip 0, `logs/chip0_chain.sh`), and the second AIME24
reference (prompt 1, `bringup/references/qwen3_8b_aime24_chat_p1.refpt`)
on the contenders (chip 1, `logs/sweep_chain.sh 1 ... aime_p1`). Round-2
candidates still running on chip 2 after 10:54 ET (c13 onward) overlap
with those jobs; their decode t/s/u is marked `concurrent` and the final
selected config is re-measured serially (section 5).

### 4.4 Round 2 on chip 2 (combination candidates, 10:46 to 11:14 ET)

| id | PC top-1/5/100 | TF top-1/5/100 | TTFT ms | TF decode t/s/u | regime | bar |
| --- | --- | --- | ---: | ---: | --- | --- |
| c1_all_bfp8_lofi (rerun on the fixed kernel row) | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 50.21 | 43.62 | serialized | pass; the most accurate policy on this reference and the slowest BFP8 one (up BFP8 reads 25 MB more per token than the baseline) |
| c12_all_bfp4_inner | 0.910/1.000/1.000 | 0.890/1.000/1.000 | 44.72 | 54.11 | serialized | fail (top-1 0.89 < 0.90) |
| c13_all_bfp4_inner_lm_bfp4 | 0.900/1.000/1.000 | 0.900/1.000/1.000 | 58.93 | 52.66 | concurrent | pass on this reference at the bar |
| c17_mlp_attn_bfp4_all_layers | 0.920/1.000/1.000 | 0.910/1.000/1.000 | 45.54 | 50.97 | concurrent | pass |
| c16_all_bfp4_all_layers | 0.910/1.000/1.000 | 0.910/1.000/1.000 | 47.24 | 51.95 | concurrent | pass |
| c15_mlp_bfp4_inner_lm_bfp4 | 0.950/1.000/1.000 | 0.940/1.000/1.000 | 47.95 | 49.50 | concurrent | pass |
| c14_mlp_bfp4_all_lm_bfp4 | see section 4.6 | | | | concurrent | |

The concurrent rows ran while chips 0 and 1 executed readiness jobs
(section 4.3); their host-wall numbers are 3 to 6 percent low (c17 carries
two more BFP4 layers than c12 and reads 50.97 against c12's serialized
54.11) and do not rank anything; section 5 re-measures every passing
contender serially. `logs/mark_regime.py` wrote `timing_regime` into the
chip-2 sidecars.

### 4.5 Acceptance rule applied to the contenders

Two 100-token samples at top-1 0.90 differ by noise (binomial standard
deviation about 3 tokens at p = 0.9), and c12 (0.89) against c13 (0.90)
shows the main reference alone cannot separate the aggressive candidates
from the bar. The stage therefore applies the bar to every reference it
evaluated: a contender passes only if top-1 >= 0.90 and top-5 >= 0.98 on
the main AIME24 reference (the gate), on the second AIME24 prompt
(`qwen3_8b_aime24_chat_p1.refpt`, chip 1) and on the 32-token short-prefix
book reference (`qwen3_8b_book32.refpt`, chip 0, also the short-prefix
case the operator requires for the cache candidate). Top-100 stays at
1.000 everywhere. The fastest serialized teacher-forcing decode t/s/u
among the configs that pass all three is selected; the qualitative suite on
the selected config (section 5) can still reject it.

### 4.6 Round 3: robustness references and the cache cases (10:54 to 11:30 ET, chips 0, 1 and 2)

Second AIME24 prompt (`aime_p1`, chip 1; c12 on chip 1 after the chain
driver was stopped, its sidecar written by hand from the finished log) and
the 32-token short-prefix book reference (`book32`, chips 0 and 2); PC /
TF top-1, top-5 in brackets where below 1.000:

| id | main TF top-1 | aime_p1 PC / TF | book32 PC / TF | all references |
| --- | ---: | --- | --- | --- |
| c0_stage7_baseline | 0.980 | 0.990 / 0.980 | 0.970 / 0.970 | pass |
| c1_all_bfp8_lofi | 1.000 | 1.000 / 1.000 | not run | pass |
| c3_kv_bfp8 | 0.980 | not run | 0.970 / 0.960 (top-5 0.990) | pass |
| c5_mlp_bfp4_inner | 0.970 | 0.920 (0.990) / 0.920 (0.990) | 0.900 / 0.910 | pass (book32 PC at the bar) |
| c6_mlp_bfp4_all | 0.920 | 0.920 (0.990) / 0.910 (0.990) | 0.920 / 0.920 | pass |
| c9_lm_head_bfp4 | 0.970 | 0.990 / 0.990 | not run | pass |
| c12_all_bfp4_inner | 0.890 | 0.910 (0.990) / 0.900 (0.990) | not run | fail (main) |
| c13_all_bfp4_inner_lm_bfp4 | 0.900 | not run (queue stopped) | 0.910 (0.990) / 0.870 (0.980) | fail (book32 TF 0.870) |
| c14_mlp_bfp4_all_lm_bfp4 | 0.940 | 0.920 (0.990) / 0.900 (0.990) | 0.910 (0.990) / 0.910 (0.980) | pass (p1 TF and book32 TF top-5 at the bar) |
| c15_mlp_bfp4_inner_lm_bfp4 | 0.940 | 0.910 (0.990) / 0.910 (0.990) | 0.900 (0.990) / 0.900 | pass (book32 at the bar) |
| c16_all_bfp4_all_layers | 0.910 | not run (queue stopped) | 0.890 (0.970) / 0.860 (0.960) | fail |
| c17_mlp_attn_bfp4_all_layers | 0.910 | not run (queue stopped) | 0.930 (0.980) / 0.890 (0.960) | fail |

Every configuration with BFP4 attention projections (c12, c13, c16, c17)
fails at least one reference; the short-prefix book prompt is the
sensitive one, which matches the stage-3 layer-17 finding (attention BFP4
delta PCC 0.9737). The MLP-BFP4 family passes everywhere, with the
edge-layer and LM-head extensions sitting on the bar on the short prefix.
The chip-1 queue for c13/c16/c17 `aime_p1` was stopped (bash drivers
killed between runs; the running c12 job finished on its own and released
the chip) because those three had already failed `book32`.

Cache cases for c3_kv_bfp8 (operator requirement), chip 0,
`batch32_same_input_{c3_kv_bfp8,c0_stage7_baseline}_chip0.json`,
`logs/batch32_same_input_*.log`: 32 users with the same 32-token book
prompt, 100 greedy tokens in the 32-row bucket, against the solo arm and
the HF greedy continuation of the reference. Both configs: every user
bit-equal to its solo arm (32 of 32, one distinct output), 198 decode trace
replays over the two arms with 2 deferred first-use captures (buckets 1
and 32) and no recapture; first divergence from HF greedy at token
27 for both (a near-tie flip that the BF16 baseline shares, after which the
greedy trajectories differ: 28 and 29 matching tokens of 100). Short
prefix (book32 reference): c3 PC 0.970 / TF 0.960 with top-5 0.990 against
the baseline's 0.970 / 0.970 / 1.000. With no batch-1 gain (46.84 against
46.70 t/s/u, 0.3 percent, inside run-to-run noise), a one-token top-5 loss
on the short prefix and the stage-3 per-user rejection, the BFP8 cache is
rejected; its capacity numbers are recorded in `context_contract.json`
(`datatype_sweep.kv_candidates_evaluated`) for reference.

## 5. First selection, superseded by section 7 (11:29 to 11:38 ET)

### 5.1 Serialized re-measurement of the contenders (chip 2, `logs/serial_tf_chip2.sh`, 11:29 to 11:36 ET)

Teacher forcing only, main reference, `fuser` clear on chips 0, 1 and 2
before the chain, nothing else on the host; sidecars
`readiness_<id>_serial_chip2.json`, logs `logs/<id>_serial_teacher_forcing.log`.

| id | TF top-1/5/100 | TTFT ms | decode t/s/u (serialized) | round-2 concurrent row | gain over the baseline (46.70) |
| --- | --- | ---: | ---: | ---: | ---: |
| c14_mlp_bfp4_all_lm_bfp4 | 0.940/1.000/1.000 | 44.91 | 52.44 | 50.01 | +12.3 percent |
| c15_mlp_bfp4_inner_lm_bfp4 | 0.940/1.000/1.000 | 45.02 | 52.13 | 49.50 | +11.6 percent |
| c6_mlp_bfp4_all | 0.920/1.000/1.000 | 45.02 | 51.89 | 51.84 (serialized in chain 1) | +11.1 percent |
| c5_mlp_bfp4_inner | 0.970/1.000/1.000 | 45.20 | 51.47 | 51.58 (serialized in chain 1) | +10.2 percent |

Run-to-run noise on this host is about 0.2 percent (c5 51.58 against
51.47, baseline 46.76 in stage 7 against 46.70 today); the 0.6 percent
step from c15 to c14 and the 0.5 percent step from c6 to c15 are above it
and structural (two more BFP4 MLP layers, 0.66 GB of LM-head bytes halved),
so the "within noise, prefer the simpler" clause does not apply between
them. The attention-BFP4 family (c12, c13, c16, c17, up to 54.11 t/s/u)
fails the short-prefix reference (section 4.6) and is out.

### 5.2 Selected at 11:37 ET: `c14_mlp_bfp4_all_lm_bfp4` (withdrawn in section 7)

The fastest configuration that passes top-1 >= 0.90 and top-5 >= 0.98 on
every reference evaluated (main AIME24: PC 0.960 / TF 0.940, top-5 1.000;
AIME24 prompt 1: PC 0.920 / TF 0.900, top-5 0.990; book32: PC 0.910 / TF
0.910, top-5 0.990 / 0.980; top-100 1.000 everywhere), at 52.44 t/s/u
serialized teacher-forcing decode (baseline 46.70, +12.3 percent) and TTFT
44.91 ms (48.28). Policy: QKV and O BFP8 LoFi (1 reader); gate, up and down
BFP4 LoFi on all 36 layers (3/3/2 readers); LM head BFP4 LoFi (prefill
LoFi, fp32 accumulation); BF16 activations, residual, embedding, norms
(HiFi4 fp32 acc), SDPA (HiFi4 fp32 acc), KV cache; BF16 decode logits, fp32
prefill logits (readiness paths), BF16 sampler input; no CCL (one chip).
Written by `logs/write_selected.py` to `selected_precision_config.json`
(the `selection` block carries the numbers and evidence paths;
`runtime_flags` the override environment). Restore: delete the file or
point `QWEN3_PRECISION_CONFIG` at `configs/c0_stage7_baseline.json`.
Alternatives on the frontier, one file swap away: `c15` (edge layers keep
the stage-7 MLP, 52.13 t/s/u, same main accuracy) and `c5` (stage-7 LM
head and edge layers, 51.47 t/s/u, main top-1 0.970) for a caller that
wants more top-1 margin than the bar.

Margin note: the selected config sits 4 points above the top-1 bar on the
main reference and 0 to 1 point above it on the two robustness references
(100-token samples, binomial standard deviation about 3 tokens at p = 0.9).
The benchmark stage's fixed lm-eval subsets are the next, larger sample;
if they show a loss the bar did not catch, `c5` is the recorded fallback.

### 5.3 Context contract (`logs/update_context_contract.py`, host arithmetic)

`doc/context_contract.json` gained a `datatype_sweep` section:
`current_supported_context` 40960 (unchanged, `capability_reduction`
null); the selected config's resident weights are smaller (BFP4 MLP and
LM head: decoder stack 2 x 4.66 GB instead of 2 x 6.47 GB, LM head 0.35 GB
instead of 0.66 GB), so with the unchanged BF16 cache (147456 bytes per
token) the all-users token budget rises from 112320 to 139008 tokens and
three full-context users fit (two before); batch 32 at full context still
does not (193 GB of cache against 34 GB of DRAM, a hard device limit, as in
stage 6). `kv_candidates_evaluated`
records the BFP8 cache (78336 bytes per token, 5 full-context users) and
the baseline for comparison; the BFP8 cache is rejected (section 4.6), so
no capacity number in the contract depends on it.

## 6. Post-selection evidence on c14 through the default path (chip 2, 11:38 to 11:53 ET, `logs/post_selection_chip2.sh`)

`selected_precision_config.json` held c14 (artifacts of this run were renamed with a `c14` tag before the final selection's run, see the table); no `QWEN3_PRECISION_CONFIG` in
the environment (every log shows `precision config c14_mlp_bfp4_all_lm_bfp4
from .../selected_precision_config.json`). One false start at 11:36 ET
(the selection file was not yet written because a shell variable was not
word-split; the chain driver was stopped between steps, the orphaned
baseline prefill check finished on its own, and the chain was relaunched
at 11:38 ET on a free chip 2).

| Step | Result | Artifact |
| --- | --- | --- |
| `run_prefill_check` / `run_teacher_forcing` | PC 0.960 / 1.000 / 1.000; TF 0.940 / 1.000 / 1.000, TTFT 44.75 ms, 52.53 t/s/u (99 replays, 0 eager) | `readiness_c14_default_chip2.json`, `logs/c14_default_{prefill_check,teacher_forcing}.log` |
| `test_full_model.py` (36 layers, `QWEN3_EVIDENCE_SUFFIX=stage8`) | 10 passed in 187 s; token-out 18.654 ms/token = 53.61 t/s/u, TTFT 38.97 ms, teacher-forced 18.69 ms/token, sampler traced; batch buckets 8/16/24/32 bit-equal to solo; multi-row prefill rows 5 to 1000 top-1 equal, rerun and reordered bit-equal; program cache 655 flat across the post-capture workload; 64-token generate 63 replays / 0 eager | `../full_model/*.stage8_c14.json` (provenance `precision_config_id c14_mlp_bfp4_all_lm_bfp4`; renamed from `*.stage8.json` before the final run) |
| 36-layer propagation check | `problems=[]`, one distinct layer policy, BFP4 LM head, BF16 caches | `propagation_check_c14_36.json` |
| qualitative suite (128 tokens, chat template, HF and stage-7 controls) | six coherent English `<think>` answers, 0.0 adjacent duplication, no control tokens; first divergence from HF at 34, 4, 12, 14, 7, 11 (stage 7: 34, 37, 32, 35, 33, 22) | `qualitative_c14/` |
| resumed prefill, default thresholds | `status=fail`: PCC min 0.9652 at start 32 and 0.8793 at start 68 (every L), top-1 0.981 to 0.984; hit window 2528 top-1 0.938; controls fail everywhere (stage 7 on the baseline: PCC min 0.9970 / 0.9886, top-1 0.996) | `../../readiness/prefill_resume_check_stage8_c14_default_thresholds.json`, `logs/c14_default_prefill_resume_check.log` |
| batch-32 same input (book32 prompt) | 32 of 32 users bit-equal to solo; the first generated token differs from HF greedy (`almost` against `no`, HF top-2 at position 0; c14 continues coherently with a different licence sentence) | `batch32_same_input_c14_default_chip2.json` |
| reduced-variant profile | decode rows `LoFi BF16 x BFP4` for gate (71 us), up (71), down (91) and the LM-head splits (124 each, FLOP-bound at 78 percent), `LoFi BF16 x BFP8` for QKV (58) and O (41) | `profile_c14/decode.txt` |

### 6.1 Resume-check regression attributed (chip 0, 11:52 to 11:56 ET, `logs/resume_margin_probe.py`)

`resume_margin_probe_c14_L1500_s68_fp32.json` and siblings (renamed with the c14 tag after review 1) compare both TT
paths (fresh prefill `ref`, continuation `cand`) with HF fp32 logits over
the window: at the worst position (285) `ref` against `cand` is PCC
0.8793 while `ref` against HF is 0.6327 and `cand` against HF 0.7139 (top-1
agrees three ways, HF margin 6.9); over the 257 positions both paths agree
with HF top-1 at 0.823 / 0.820 and PCC mean 0.9348 / 0.9348. The
continuation path is not worse than the fresh path, so the resume
mechanism is intact; the failure is the precision of the policy itself at
hard positions, where BFP4 MLP rows push both TT paths far from HF and
their mutual PCC follows. The same probe on the stage-7 baseline
(`doc/optimized_full_model/resume_margin_probe_L1500_s68_fp32.json`) had
both paths at 0.918 top-1 against HF and PCC mean 0.9867. `c5` fails the
same gate the same way (PCC min 0.9449 at start 68,
`prefill_resume_check_stage8_c5_mlp_bfp4_inner_default_thresholds.json`);
`c15` too (`..._c15_..._default_thresholds.json`).

### 6.2 Reading

The readiness references (HF-greedy continuations, large margins) rate
c14 at 0.94 top-1; natural book text rates the same policy at 0.82
against the baseline's 0.92. That is a 10-point full-model top-1 loss the
gate references did not see, and it fails the stated bar (top-1 >= 0.90)
on a 257-position sample. Selecting c14 on the easy references while this
evidence sits in the folder would be a weak dismissal, so the stage adds
natural text as a fourth acceptance reference (section 7) and re-selects.

## 7. Natural-text reference and final selection (11:57 to 12:25 ET)

### 7.1 `logs/natural_text_check.py` (chips 0, 1 and 2 in parallel, accuracy only)

TT prefill logits of the first 1500 book tokens (`prefill_logits`, fp32)
against HF fp32 logits over positions 32 to 1499 (1468 real next-token
positions): top-1 and top-5 agreement with HF, PCC. Artifacts
`natural_text_<id>_chip<N>.json`, logs `logs/natural_text_<id>.log`.

| id | top-1 vs HF | top-5 vs HF | PCC mean | PCC min | disagreements with HF margin >= 3 | bar (0.90 / 0.98) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c0_stage7_baseline | 0.9189 | 0.9980 | 0.9866 | 0.8574 | 1 | pass |
| c1_all_bfp8_lofi | 0.9680 | 1.0000 | 0.9973 | 0.5666 | | pass |
| c9_lm_head_bfp4 | 0.9094 | 0.9980 | 0.9793 | 0.8517 | | pass |
| c11_down_bfp4_inner | 0.8910 | 0.9932 | 0.9734 | 0.5655 | | fail |
| c8_attn_bfp4_inner (run at 12:38 ET, after the first review started) | 0.8965 | 0.9898 | 0.9711 | 0.5478 | | fail |
| c10_mlp_bfp4_inner_hifi2 (12:41 ET) | 0.8297 | 0.9789 | 0.9420 | 0.3011 | | fail (equal to c5 at LoFi) |
| c7_gateup_bfp4_inner | 0.8467 | 0.9871 | 0.9553 | 0.5026 | 34 | fail |
| c6_mlp_bfp4_all | 0.8345 | 0.9782 | 0.9418 | 0.5108 | 39 | fail |
| c5_mlp_bfp4_inner | 0.8297 | 0.9789 | 0.9420 | 0.3011 | 41 | fail |
| c14_mlp_bfp4_all_lm_bfp4 | 0.8256 | 0.9755 | 0.9344 | 0.5079 | 36 | fail |
| c15_mlp_bfp4_inner_lm_bfp4 | 0.8195 | 0.9755 | 0.9346 | 0.2994 | 41 | fail |

Every configuration with a BFP4 gate, up+down, or full BFP4 MLP loses 7
to 10 points of top-1 against HF on real text and falls below the bar;
the single-BFP4-projection policies (baseline up only; `c9` adds the
BFP4 LM head) stay above it. The first script revision also wrote two
"book token" fields that compared position i with token i; they were
removed from the JSON files and the script fixed (the HF agreement fields
are unaffected).

### 7.2 Acceptance rule, final form

A candidate passes when top-1 >= 0.90 and top-5 >= 0.98 hold on the main
AIME24 reference (PC and TF), the second AIME24 prompt, the 32-token
short-prefix book reference and the natural-text reference, wherever the
candidate was run; top-100 stays 1.000 on the readiness references. The
fastest serialized traced teacher-forcing decode among the passing
candidates is selected. Passing set: c0, c1, c3 (rejected on the cache
cases, section 4.6), c9. Fastest: `c9_lm_head_bfp4` (47.24 t/s/u in chain
1, re-measured in 7.3), the stage-7 policy with a BFP4 LoFi LM head.

### 7.3 Round 4: c9 robustness and LM-head reader variants (12:08 to 12:20 ET)

- chip 0, book32 pair: PC 0.950 / 0.990 / 1.000, TF 0.970 / 1.000 / 1.000
  (`readiness_c9_lm_head_bfp4_book32_chip0.json`).
- chip 1, resumed prefill at the default thresholds: every PCC gate passes
  (0.9970 / 0.9886 minima, the stage-7 values), top-1 fails only the two hit
  windows by one near-tie flip (1472: 27 of 28, 16352: 31 of 32), so
  `status=fail`; lowered thresholds (0.90 / 0.93): `status=pass`
  (`../../readiness/prefill_resume_check_stage8_c9_lm_head_bfp4_{default_thresholds,lowered}.json`,
  `logs/resume_check_c9_lm_head_bfp4{,_lowered}.log`).
- chip 2, serialized teacher forcing after chips 0 and 1 were free
  (`logs/serial_tf_chip2.sh`): c9 47.24 t/s/u, TTFT 48.05 (chain 1: 47.24,
  48.02); c18 (LM head readers 2) 47.17; c19 (readers 4) `TT_FATAL:
  num_workers_per_dram_bank must be in [1, 3]`
  (`matmul_utilities.cpp:345`), an exact op-contract blocker. Readers 1
  kept. Both variants carry the same dtype and fidelity as c9, so no new
  accuracy run was needed for the rejected geometry.

### 7.4 Final selection: `c9_lm_head_bfp4` (12:17 ET)

`logs/collect_results.py c9_lm_head_bfp4` (18 full candidates plus the two
timing-only reader variants), `logs/write_selected.py c9_lm_head_bfp4`,
`logs/update_context_contract.py c9_lm_head_bfp4 c3_kv_bfp8 c0_stage7_baseline c14_mlp_bfp4_all_lm_bfp4`.
Policy: the stage-7 policy with a BFP4 LoFi LM head (decode bf16 acc,
prefill LoFi fp32 acc, 1 reader). Numbers: section 1 of the README.
Context contract: resident weights 14.62 GB computed (LM head 0.35 GB
instead of 0.66 GB), token budget 114432 (112320), two full-context users
(unchanged), context 40960 retained; `kv_candidates_evaluated` holds c3,
c0 and the withdrawn c14. Restore: delete `selected_precision_config.json`
or set `QWEN3_PRECISION_CONFIG=configs/c0_stage7_baseline.json`.

## 8. Post-selection evidence on c9 through the default path (chip 2, 12:20 to 12:35 ET, `logs/post_selection_chip2.sh`)

`logs/post_selection_chip2.status`: `selected_prefill_check` (PC 0.980 /
1.000 / 1.000), `selected_teacher_forcing` (TF 0.970 / 1.000 / 1.000, TTFT
48.03 ms, 47.30 t/s/u), `selected_pytest_full_model_36` (10 passed in 198
s; token-out 20.776 ms/token = 48.13 t/s/u, TTFT 41.71 ms, teacher-forced
20.80 ms/token, sampler traced, program cache 666 flat, batch buckets and
multi-row prefill bit-equal to solo), `selected_propagation_36`
(`problems=[]`), `selected_qualitative_tt` (six coherent answers, first
divergence from HF equal or later than stage 7 on every prompt),
`selected_prefill_resume_check` (the same rows as the chip-1 run: PCC gates
pass, two hit-window near-tie flips), `selected_batch32_same_input` (32 of
32 bit-equal to solo, HF divergence at token 27 as the baseline),
`selected_profile_reduced` + `perf_report.sh` (`profile/`: LM-head splits
`LoFi BF16 x BFP4` at 124 us, the decoder rows those of stage 7). The
README section 5 table lists every number and artifact.

## 9. Commits and stage review (12:36 ET onward)

Local commits on `hous/qwen3-8b-autoport` (never pushed; the supervisor pushes):

- `cd64cfe5aa` Qwen3-8B datatype sweep: precision config artifact consumed by build_generator (layer exceptions, KV dtype, terminal policy), runtime readback, dtype-aware gate/up prefill kernel row, selected c9 config (`tt/precision.py`, `tt/generator.py`, `tt/model.py`, `tt/optimized_decoder.py`, `tests/test_full_model.py`, `configs/`, `selected_precision_config.json`, `logs/make_configs.py`).
- `2a6cc0d845` Qwen3-8B datatype sweep: 20-candidate sweep on chip 2 with robustness references on chips 0 and 1, natural-text reference, BFP4 LoFi LM head selected (c9), post-selection default-path evidence, context contract, Pareto charts, docs (everything else under `doc/datatype_sweep/`, `doc/context_contract.json`, `doc/full_model/*.stage8*.json`, `readiness/prefill_resume_check_stage8_*.json`, `bringup/references/qwen3_8b_aime24_chat_p1.refpt`; the 1.5 MB `qwen3_8b_book32.refpt` stays uncommitted because of the 500 KB hook, with its regeneration command in the README).

The repository's pre-commit hooks reformatted the stage's helper scripts
(black); no source behaviour changed.

Stage review 1 (12:39 to 12:52 ET, prompt `logs/stage_review_prompt.md`,
fresh read-only subagent): `more-work-needed`
(`logs/review/stage_review_1.md`, full text `stage_review_1_full.md`).
Required work: P1 c8 and c10 recorded as pass without robustness rows
(section 9.1); P2 no margin probe for the c9 hit-window flips (section
9.2); P2 context-contract note against the no-mesh fallback and the
gitignored CSV (section 9.2).

### 9.1 Candidate gap closed during the review (12:38 to 12:43 ET, chips 0, 1 and 2)

`c8_attn_bfp4_inner` (48.82 t/s/u, faster than the selected c9) had no
robustness rows and `c10` no natural-text row. Run while the reviewer
worked (read-only): c8 aime_p1 PC 0.970 / TF 0.970 (chip 1), book32 PC
0.950 / TF 0.950 with top-5 0.990 (chip 0), natural text 0.8965 / 0.9898
(chip 2): fails the natural-text bar. c10 natural text 0.8297 / 0.9789
(chip 2), identical to c5, so HiFi2 does not recover the BFP4 MLP loss.
`sweep_results.{json,csv}` regenerated with these rows; the selection is
unchanged.

### 9.2 Review-1 fixes (12:52 to 13:05 ET)

- P2 resume probe: `logs/resume_margin_probe.py 1500:1472 16384:16352`
  through the default path on chip 0 (`logs/resume_margin_probe_c9_1472.log`,
  `resume_margin_probe_c9_L1500_s1472_fp32.json`,
  `resume_margin_probe_c9_L16384_s16352_fp32.json`): the 1472 window flips
  at position 1487 (fresh-path margin 0.078, continuation margin 0.031, HF
  margin 0.641, HF top-1 equal to the fresh path), the 16352 window at
  16369 (0.125 / 0.016 / 0.393); both TT paths agree with HF equally over
  the windows and their mutual PCC stays at 0.9998 / 0.9995: near ties of
  the BFP4 LM head. The c14 probes were renamed `resume_margin_probe_c14_*`.
  README section 5 now quotes the top-1 rows against stage 7 and keeps
  "unchanged" for the PCC minima.
- P2 context contract: `Qwen3Generator.max_tokens_all_users(model)`
  without a mesh now computes the resident bytes from the loaded model
  (`Qwen3Model.weight_bytes()`, which now includes the LM-head tail mask)
  plus the stage-6 allocator rounding (`ALLOCATOR_ROUNDING_BYTES`), giving
  114432 for the selected build; the stage-6 constants (112320) apply only
  with neither model nor mesh; the mesh path is authoritative. Measured on
  chip 2 (`logs/token_budget_check.py`, `token_budget_check.json`):
  model-only 114432, no-model 112320, with the mesh 72736 (after the
  generator's own KV pool). The contract note and
  `logs/update_context_contract.py` say which path gives which number.
  `sweep_results.csv` and `profile/*.csv` are force-added past the `*.csv`
  ignore; `logs/*.log` stay in the worktree (README section 10).
- `tt/precision.py` `check_propagation` now also compares decode
  `fp32_acc`, `packer_l1_acc`, `in0_block_w`, per-layer prefill fidelity
  and accumulation, norm and SDPA fidelity, and the LM head's prefill
  accumulation, readers and `in0_block_w`; rerun on the 36-layer selected
  build on chip 2 (`logs/selected_propagation_36.log`, 12:54 ET):
  `PROPAGATION_OK`.
- Other concerns: README section 1 states that the natural-text reference
  runs the prefill path and names the decode LM-head coverage; the config
  schema note (README section 1, `tt/precision.py` docstring) says the
  per-group fidelity governs decode only; README section 4 says how the
  propagation script resolves the config; `write_sidecar.py` no longer
  parses top-k fields from resume-check logs and the two sidecar entries
  were corrected; the c8 sentence and the natural-text label were
  rewritten; comment lines were removed from the helper shell scripts; the
  c3 capacity trade is named in README section 7 for the vLLM stage.
- Hard-check gaps: the top-5 chart marks its single frontier point and
  says why; the stale fields were removed from the three remaining
  natural-text JSON files; README section 8 records the verdict.

Commit `26856835e8`: review-1 fixes (sections 9.1 and 9.2; black reformatted
`tt/precision.py` and `logs/token_budget_check.py`, no behaviour change).
Stage review 2 (fresh read-only subagent, same prompt plus the review-1
record): verdict recorded below when it returns.
