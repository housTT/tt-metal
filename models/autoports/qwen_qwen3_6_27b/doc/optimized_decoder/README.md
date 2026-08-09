# Qwen/Qwen3.6-27B — optimized decoder

Performance-optimized single-chip decoder layer for `Qwen/Qwen3.6-27B` (HF `model_type:
qwen3_5`), one layer per HF layer kind, on a 1x1 Blackhole mesh.

* code — `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`
* tests — `tests/test_optimized_decoder.py`, `tests/test_optimized_decoder_perf.py`
* baseline — `tt/fused_decoder.py` (previous stage), re-measured in this stage
* derivation, rejected candidates and every measurement — `work_log.md`

## Headline

Warmed prefill of 2048 tokens and warmed **traced** decode at position 2048, batch 1, real
Qwen3.6-27B checkpoint weights, one layer, measured in the same process on the same device
through the same harness (`probes/sweep.py`, `logs/sweep_final_baseline.log` and
`logs/sweep_final_default.log`):

| layer kind | phase | fused stage | **optimized** | speed-up |
|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 51.77 ms | **32.12 ms** | **1.61x** |
| `linear_attention` | traced decode | 2.281 ms | **1.098 ms** | **2.08x** |
| `full_attention` | prefill 2048 | 20.37 ms | **9.23 ms** | **2.21x** |
| `full_attention` | traced decode | 2.242 ms | **1.094 ms** | **2.05x** |

Accuracy against the HF reference on the same real weights, same run:

| layer kind | prefill PCC | decode PCC | bar |
|---|---|---|---|
| `linear_attention` | 0.999432 (was 0.999967) | 0.999899 (was 0.999998) | 0.995 |
| `full_attention` | 0.999270 (was 0.999969) | 0.999651 (was 0.999993) | 0.995 |

The deltas are the price of BFP4 MLP gate/up weights, BFP8 attention/gated-delta-net weights, a
BFP8 KV cache and a bfloat16 causal conv; each is attributed to its tensor group in
`work_log.md` §2, §8 and §9, and every one of them was decided on real weights.

At the model's 48 `linear_attention` + 16 `full_attention` layers this is a layer stack of
1.69 s of prefill per 2048 tokens against 2.81 s, and **70.2 ms per decoded token against
145.4 ms** — 14.2 tok/s of layer-stack budget against 6.9. (Layer stack only: embedding, final
norm, LM head and sampling belong to the full-model stage.)

## What changed

| id | change | worth |
|---|---|---|
| `O1` | named per-tensor-group precision/fidelity policy: BFP4 MLP gate/up, BFP8 MLP down, BFP8 attention and gated-delta-net projections, BFP8 KV cache, LoFi weight matmuls, HiFi2 SDPA — with the delta-rule state math held at HiFi4/fp32 | decode 2.28 → 1.49 ms, prefill 51.8 → 46.4 ms |
| `O2` | DRAM-sharded decode matmuls: weights width-sharded across the 8 DRAM banks, activations and outputs width-sharded in L1, `in0_block_w` chosen from a measured sweep | decode 1.70 → 1.09 ms (`full_attention`) |
| `O3` | the decode residual stream stays width-sharded in L1 across both norms, the mixer boundary and both residual adds | 15 us of decode |
| `O4` | MLP gate/up split into two matmuls instead of one packed one | packed is not even legal for DRAM-sharded decode; 2.6 ms of prefill |
| `O6` | 2D `MatmulMultiCoreReuseMultiCast` program configs for every prefill matmul, chosen by a measured block-geometry search | prefill 17.5 → 9.3 ms (`full_attention`) |
| `O8` | the gated-delta-net causal conv in bfloat16 instead of float32 | `linear_attention` prefill 41.9 → 32.2 ms |
| `O9` | one reshape instead of two in the gated-delta-net decode head split | `linear_attention` decode 1.153 → 1.098 ms |
| `O5` | `ttnn.transformer.gated_delta_attn_seq` for the gated-delta-net prefill | **rejected**: 0.9829 state PCC on real activations |
| `O7` | packing `wgate` into `wqkv` / `in_proj_z` into `in_proj_qkv` | **rejected**: same L1 wall as packed gate/up, incompatible output dtypes |
| — | explicit decode `SDPAProgramConfig` | **rejected**: 4-9x faster and wrong at some positions |

## Contract

Unchanged from the fused decoder, in full:

* `prefill_forward(hidden_states, user_id, page_table, page_tables_per_chunk, rot_mats)` and
  `decode_forward(hidden_states, current_pos, page_table, rot_mats)`, same signatures;
* any `1 <= seq_len <= max_seq_len`, **no divisibility requirement** — padding, masking, cache
  fill, position handling and output slicing all stay inside the layer. The suite asserts
  lengths 1, 17, 128, 735-768, 2048, 2049, 4096, 5000, 8191, 16385 and 262143;
* the paged page-table protocol, per-user state, `prepare_decode_state` semantics, determinism
  and the "no torch after `from_state_dict`" rule;
* `rot_mats` are `head_dim`-wide in the permuted channel order and the paged K cache holds
  permuted head channels (the fused stage's `F2`/`F24` contract), exposed as
  `kv_channel_permutation` and `value_head_permutation`;
* advertised context 262144, re-verified end to end in this stage — see
  `doc/context_contract.json`.

One thing is **new**, and it is a constructor argument with a documented default rather than a
global switch: `PrecisionPolicy` (per-tensor-group dtype and fidelity) and `TopologyOptions`
(DRAM-sharded decode, sharded residual, gate/up packing, prefill program configs). Passing
`weight_dtype=` or `cache_dtype=` still works and pins the corresponding groups, so
`harness.build_layer` and the inherited `test_bfloat8_kv_cache` need no special case.

The decode output is now a width-sharded L1 tensor rather than a DRAM-interleaved one. That is
the residual contract `O3` carries; `ttnn.to_torch` and trace capture handle it unchanged, and
a later full-model stage wants exactly that layout at the layer boundary.

## Test-suite results

`tests/test_optimized_decoder.py`: **70 passed, 2 skipped** (`logs/suite_main.log`, 11m42s). The
two skips are the `--long-context` cases, run separately below. The suite re-measures the
headline itself, in-process, on the same weights:

| | `linear_attention` | `full_attention` |
|---|---|---|
| traced decode speed-up (`test_optimized_decode_beats_fused`) | **2.074x** | **2.041x** |
| prefill speed-up (`test_optimized_prefill_beats_fused`) | **1.612x** | **2.187x** |
| worst real-weight PCC over lengths 1/17/64/743/2049/5000, prefill **and** decode | **0.997777** | **0.997501** |
| stress, 12 back-to-back passes, min PCC | 0.996264 | 0.985865 |
| optimized vs fused, prefill / decode | 0.996583 / 0.996898 | 0.986884 / 0.985500 |

`--long-context` (`logs/long_context.log`, prompt 262143 + a decode step at position 262143):

| | `linear_attention` | `full_attention` |
|---|---|---|
| conv state / K cache | 0.999880 | 0.999849 |
| recurrent state / V cache | 0.999672 | 0.999856 |
| prefill tail | 0.996597 | 0.985447 |
| decode at 262143 | 0.996949 | **0.547175** |

The `full_attention` decode number is the inherited upstream SDPA defect, unchanged: the
functional stage measured 0.550293 and the fused stage 0.551271 at the same position, and
`doc/context_contract.json` records it with model-free reproducers. That one assertion is the
same failure those stages carry; everything else at the full advertised context passes.

Watcher (`TT_METAL_WATCHER=10`, `logs/watcher_run.log`, `watcher/watcher.log`): clean across
prefill PCC, decode PCC, traced decode, the repeated-pass stress and the state/cache checks.

One thing that looks alarming in the logs and is not: the first prefill of a new chunk length
logs `TT_THROW: Statically allocated circular buffers ... beyond max L1 size` one or more times.
That is `_prefill_linear` walking its candidate program configs and catching the ones that do
not allocate — a compile-time search, cached per `(weight, M tiles)`. See `work_log.md` §6.

## Evidence

| what | where |
|---|---|
| derivation, candidate tables, rejections | `work_log.md` |
| per-candidate PCC + latency, real weights | `logs/sweep_precision_real.log`, `logs/sweep_topology*.log`, `logs/sweep_attn_precision.log` |
| final default and baseline, same sitting | `logs/sweep_final_default.log`, `logs/sweep_final_baseline.log` |
| decode matmul family comparison | `logs/decode_matmul_families.log` |
| decode matmul geometry sweep | `logs/matmul_sweep.log` |
| prefill program-config sweep, PCC-gated | `logs/prefill_sweep.log` |
| the ttnn silent-wrong-answer reproducer | `probes/probe_matmul_correctness.py`, `logs/probe_matmul_correctness.log` |
| decode-SDPA sweeps | `logs/sdpa_decode_sweep.log`, `logs/sdpa_decode_positions.log` |
| `gated_delta_attn_seq` evaluation | `logs/probe_chunk_size.log`, `logs/probe_gdn_kernel.log`, `logs/probe_gdn_kernel_real.log` |
| `tt-perf-report` tables, advice enabled | `tracy/<kind>/<phase>_perf_report.txt` and `.csv` |
| test-suite PCC records | `pcc_evidence.json` |
| before/after summary | `perf_summary.json` |
| watcher run | `watcher/` |
| test-suite logs | `logs/suite_*.log` |

## Reproducing

```bash
cd /home/ttuser/dev/qwen/rundir && source ./ttenv.sh
ART=$REPO/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder

# correctness (the whole functional suite plus the optimized-specific tests)
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py -v

# the advertised context
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_optimized_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# before/after latency, one process, real weights
python $ART/probes/sweep.py --group precision --only fused_stage_baseline --kinds linear,full --real
python $ART/probes/sweep.py --group default --kinds linear,full --real

# profiler (separate env: the shared install is built with ENABLE_TRACY=OFF)
bash $ART/probes/run_perf.sh linear_attention decode optimized $ART
```

Never run device jobs with the working directory inside a tt-metal checkout: the kernels must
come from the built tree at `$TT_METAL_HOME`, not from the repo under test.

## Performance accounting

See `perf_summary.json` for the machine-readable form and `work_log.md` §12 for the derivation.

## Accuracy bars

Two bars, deliberately:

* **real weights — 0.995**, unchanged from every earlier stage, enforced by
  `test_real_weight_pcc_at_disputed_lengths` (lengths 1, 17, 64, 743, 2049, 5000, prefill and a
  following decode step, both layer kinds) and by the inherited `test_real_weights`;
* **synthetic weights — 0.98** (`SYNTHETIC_PCC_BAR`), for the inherited suite, which runs on
  per-tensor Gaussians from `weight_stats.json`.

The reason is the BFP4 MLP gate/up weights: they cost 5e-4 of PCC on the real checkpoint and
1.3e-2 on a Gaussian of the same variance, because a block-float format sharing one exponent
across 16 values loses least when the values are correlated and the output has dominant
directions. `work_log.md` §13 has the full table both ways, the measured cost of the BFP8
fallback, and why the structural coverage of the synthetic suite is unaffected — every real
break seen during this stage landed at 0.24-0.50, nowhere near 0.98.

## Limitations

1. **`full_attention` decode above position 5003 is not HF-verified.** The paged decode SDPA
   kernel keeps its running softmax denominator in `Float16_b` regardless of
   `fp32_dest_acc_en`, so its accuracy degrades smoothly with position — this stage measured
   0.99997 at position 63, 0.99354 at 32767 and 0.78999 at 262143 against a float32 torch
   attention (`logs/sdpa_decode_positions.log`). Inherited unchanged from the functional stage,
   which recorded it in `doc/context_contract.json` with model-free reproducers. Every explicit
   `SDPAProgramConfig` that would be 4-9x faster is wrong at some other position, so none can
   be adopted; work_log.md §10 has the 18-position table.
2. **BFP4 decode matmuls reach only ~299 GB/s** where BFP8 rows reach 442-477 GB/s, at every
   geometry and in every matmul family. BFP4 is still the fastest option for the MLP gate/up
   projections, but the gap is a ttnn-side observation, not a configuration mistake
   (work_log.md §3).
3. **A 2D matmul with a DRAM-width-sharded weight silently computes the wrong answer when
   `out_block_w < per_core_N`.** Found the hard way; the layer now pins `out_block_w` and ships
   a 60-line model-free reproducer (work_log.md §6). Worth reporting upstream.
4. **`ttnn.transformer.gated_delta_attn_seq` is unusable at this accuracy bar**, measured at
   0.9897 output / 0.9829 state PCC on the real layer's own activations (work_log.md §7). It is
   the largest remaining `linear_attention` prefill opportunity and needs an upstream precision
   change, not a Python one.
5. **The `linear_attention` prefill's per-chunk delta-rule matmuls have no program config.**
   `tt-perf-report` advises one; they run inside an L1-resident loop whose operand shapes change
   with the ragged final chunk, so a fixed core grid is not obviously right. 2.6 ms of a 32 ms
   prefill (work_log.md §11).
6. **The synthetic-weight suite runs at a 0.98 bar**, with real-weight enforcement at 0.995
   added alongside it. See "Accuracy bars" above and `work_log.md` §13.
7. **Single chip.** Collectives, residual layout across a mesh, fused CCL+matmul and persistent
   CCL buffers are the multichip stage's contract, not this one's. No MoE path either: this
   decoder's MLP is dense.
