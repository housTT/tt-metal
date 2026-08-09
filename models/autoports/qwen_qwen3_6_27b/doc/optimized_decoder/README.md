# Qwen/Qwen3.6-27B — optimized decoder

Performance-optimized single-chip decoder layer for `Qwen/Qwen3.6-27B` (HF `model_type:
qwen3_5`), one layer per HF layer kind, on a 1x1 Blackhole mesh.

* code — `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`
* tests — `tests/test_optimized_decoder.py`, `tests/test_optimized_decoder_perf.py`
* baseline — `tt/fused_decoder.py` (previous stage), re-measured in this stage
* derivation, rejected candidates and every measurement — `work_log.md`

## Headline

Warmed prefill of 2048 tokens and warmed **traced** decode at position 2048, batch 1, real
Qwen3.6-27B checkpoint weights, one layer, on the same device through the same harness
(`probes/sweep.py`; baseline `logs/sweep_final_baseline.log`, optimized
`logs/sweep_final_default.log` — two runs of one harness back to back on the final code, not one
process). The **in-process** comparison is the suite's: `test_optimized_decode_beats_fused` and
`test_optimized_prefill_beats_fused` build both decoders in one process on the same weights and
compare them, and they agree with the table below to three digits — see "Test-suite results".

| layer kind | phase | fused stage | **optimized** | speed-up |
|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 51.73 ms | **28.899 ms** | **1.79x** |
| `linear_attention` | traced decode | 2.28 ms | **1.098 ms** | **2.07x** |
| `full_attention` | prefill 2048 | 20.43 ms | **9.286 ms** | **2.20x** |
| `full_attention` | traced decode | 2.24 ms | **1.094 ms** | **2.05x** |

Accuracy against the HF reference on the same real weights, same run:

| layer kind | prefill PCC | decode PCC | bar |
|---|---|---|---|
| `linear_attention` | 0.999428 (was 0.999967) | 0.999899 (was 0.999998) | 0.995 |
| `full_attention` | 0.999270 (was 0.999969) | 0.999651 (was 0.999993) | 0.995 |

The deltas are the price of BFP4 MLP gate/up weights, BFP8 attention and gated-delta-net
projections, a BFP8 KV cache and a bfloat16 causal conv; each is attributed to its tensor group
in `work_log.md` §2, §8 and §9, and every one of them was decided on real weights at the
disputed lengths, not at 2048 alone. §23 is the one that got away: a BFP4 output gate that
passed every gate here and was still rejected, because sweeping the decode-token draw put it
below the bar.

At the model's 48 `linear_attention` + 16 `full_attention` layers this is a layer stack of
1.54 s of prefill per 2048 tokens against 2.81 s, and **70.2 ms per decoded token against
145.2 ms** — 14.2 tok/s of layer-stack budget against 6.9. (Layer stack only: embedding, final
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
| `O10` | shape-aware program configs for the batched delta-rule matmuls | `linear_attention` prefill 32.12 → 30.90 ms |
| `O5` | `ttnn.transformer.gated_delta_attn_seq` for the gated-delta-net prefill | **rejected**: 0.9829 state PCC on real activations, unchanged when every Python-side precision knob is raised to HiFi4/fp32 |
| `O7` | packing `wgate` into `wqkv` / `in_proj_z` into `in_proj_qkv` | **rejected on measurement**: attention pair 5.6 us faster in decode and 466 us slower in prefill before split cost; GDN pair blocked by incompatible output dtypes |
| — | explicit decode `SDPAProgramConfig` | **rejected**: 4-9x faster and wrong at some positions |
| — | BFP4 `attn_out` | **rejected on real weights**: 0.993815 prefill / 0.994999 decode at seq 17 on `full_attention`, below the 0.995 bar — caught by `test_real_weight_pcc_at_disputed_lengths` after the 2048-token measurement said it was fine (`work_log.md` §9) |
| — | BFP4 `gdn_out` | **rejected on real weights**, on its own layer kind's numbers: worth 7.3 us of `linear_attention` decode and above the bar on every single-draw measurement, but 0.994194 worst over six decode-token draws at seq 743, below the bar on three of the six (`work_log.md` §9) |
| — | `proj_fp32_acc=False` | **rejected on real weights**: worth 0.5 % of decode, but over six decode-token draws at seq 743 it takes `linear_attention`'s worst real-weight decode PCC from 0.996689 to 0.995077 — onto the 0.995 bar — and puts two synthetic structural cases at 0.9789 against the 0.98 bar (`work_log.md` §9) |
| `O12` | the gated-delta-net chunk loop's recurrent state in L1 | `linear_attention` prefill 30.96 → 29.13 ms |
| `O13` | the per-chunk total decay as a reduction instead of a last-row slice of the cumulative sum | `linear_attention` prefill 29.13 → 28.85 ms; 309 us of untilize/slice/tilize removed |
| `O14` | BFP4 `full_attention` output gate | **adopted, gated, then rejected**: worth 6.7 us of `full_attention` decode and it passes every gate this stage has, including the suite's own real-weight test at 0.996131 — but sweeping only the decode-token draw puts its worst at **0.994316**, below the 0.995 bar, where BFP8 reads 0.996181 on the same six draws (`work_log.md` §23) |
| — | `ttnn.conv1d` for the gated-delta-net causal conv | **rejected**: exact L1 blocker at 10240 depthwise channels, and at every channel split tried |
| — | matching ttnn's own row-wise decode shard grid (`O11`) | **rejected**: the sharded layernorm that carries the residual refuses a non-rectangular grid, and no core count is rectangular both ways — costs one 1.6-1.8 us reshard per decode step (`work_log.md` §19) |

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

`tests/test_optimized_decoder.py`: **72 passed, 2 skipped, 3 warnings in 758.77s (0:12:38)** (`logs/suite_main.log`). The
two skips are the `--long-context` cases, run separately below. The suite re-measures the
headline itself, in-process, on the same weights:

| | `linear_attention` | `full_attention` |
|---|---|---|
| traced decode speed-up (`test_optimized_decode_beats_fused`) | **2.074x** | **2.039x** |
| prefill speed-up (`test_optimized_prefill_beats_fused`) | **1.785x** | **2.179x** |
| worst real-weight PCC over lengths 1/17/64/743/2049/5000, prefill **and** decode | **0.996689** | **0.996181** |
| stress, 12 back-to-back passes, min PCC | 0.996257 | 0.985865 |
| optimized vs fused, prefill / decode | 0.996582 / 0.996900 | 0.986884 / 0.985500 |
| BF16/HiFi4 structural prefill, worst over the same lengths (bar 0.999) | 0.999898 | 0.999434 |

`--long-context` (`logs/long_context.log`, prompt 262143 + a decode step at position 262143):

| | `linear_attention` | `full_attention` |
|---|---|---|
| conv state / K cache | 0.999880 | 0.999849 |
| recurrent state / V cache | 0.999666 | 0.999856 |
| prefill tail | 0.996595 | 0.985447 |
| decode at 262143 | 0.996961 | **0.547175** |

The `full_attention` decode number is the inherited upstream SDPA defect, unchanged: the
functional stage measured 0.550293 and the fused stage 0.551271 at the same position, and
`doc/context_contract.json` records it with model-free reproducers. That one assertion is the
same failure those stages carry; everything else at the full advertised context passes.

Watcher (`TT_METAL_WATCHER=10`, `logs/watcher_run.log`, `watcher/watcher.log`): **30 passed, 44 deselected, 3 warnings in 295.49s (0:04:55)**, clean across
prefill PCC, decode PCC, traced decode, the repeated-pass stress and the state/cache checks.

One thing that looks alarming in the logs and is not: the first prefill of a new chunk length
logs `TT_THROW: Statically allocated circular buffers ... beyond max L1 size` one or more times.
That is `_prefill_linear` walking its candidate program configs and catching the ones that do
not allocate — a compile-time search, cached per `(weight, M tiles)`. See `work_log.md` §6. The
same pattern covers the batched delta-rule matmuls (`O10`, `work_log.md` §17), where a shape the
reuse program refuses is remembered as `None` after one failed dispatch.

A residual risk worth naming: that search keeps the *first* configuration that allocates, and L1
occupancy at the moment of the first call is part of what "allocates" means. Every failure
observed in every shipped run here is the shape-only kind (`grow to 1585920 B which is beyond
max L1 size`), which is occupancy-independent, so the choice was deterministic — the determinism
and 12-repeat stress tests pass bit-identically. A first call under materially different L1
pressure could in principle pin a slower config for the rest of the process.

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
| draw sensitivity of the real-weight decode PCC | `probes/probe_draw_sensitivity.py`, `logs/probe_draws_fp32acc_seq743.log`, `logs/probe_draws_o14_seq17.log` |
| is a BF16 structural gate possible? (§20) | `probes/probe_bf16_structural.py`, `logs/probe_bf16_structural.log` |
| `attn_gate` BFP4 (`O14`) at the disputed lengths, real and synthetic | `logs/sweep_v6_attn_short_{17,743}.log`, `logs/sweep_v8_o14_short_{17,743}.log`, `logs/sweep_v7_gate_synth.log` |
| `O13`, the per-chunk decay reduction | `logs/sweep_o13_glast_check.log` |
| test-suite PCC records, including every per-draw value | `pcc_evidence.json`, regenerated by `probes/collect_pcc_evidence.py --write` |
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

# before/after latency, real weights (two runs of the same harness; the in-process
# comparison is the suite's test_optimized_{decode,prefill}_beats_fused)
python $ART/probes/sweep.py --group precision --only fused_stage_baseline --kinds linear,full --real
python $ART/probes/sweep.py --group default --kinds linear,full --real

# profiler (separate env: the shared install is built with ENABLE_TRACY=OFF)
bash $ART/probes/run_perf.sh linear_attention decode optimized $ART

# regenerate every published number from the artifacts above (never hand-edit them)
python $ART/probes/derive_report_numbers.py --write
python $ART/probes/collect_pcc_evidence.py --write
```

Never run device jobs with the working directory inside a tt-metal checkout: the kernels must
come from the built tree at `$TT_METAL_HOME`, not from the repo under test.

## Performance accounting

See `perf_summary.json` for the machine-readable form and `work_log.md` §12 for the derivation.

## Accuracy bars

Two bars, deliberately:

* **real weights — 0.995**, unchanged from every earlier stage, enforced by
  `test_real_weight_pcc_at_disputed_lengths` (lengths 1, 17, 64, 743, 2049, 5000, prefill and a
  following decode step, both layer kinds). Note the inherited `test_real_weights` asserts
  against `H.PCC_BAR`, which this module relaxes, so it is *not* a second 0.995 gate — the new
  test is the only one.  A precision-independent structural gate (the same lengths at BF16) was
  built and does not work — the prefill program-config search is sized for the shipped weights
  and runs out of L1 at BF16 on short chunks (`work_log.md` §20).  What guards structure instead
  is that a structural break is not a small PCC loss: every one seen during this stage landed at
  0.24-0.50;
* **synthetic weights — 0.98** (`SYNTHETIC_PCC_BAR`), for the inherited suite, which runs on
  per-tensor Gaussians from `weight_stats.json`.

The reason is the BFP4 MLP gate/up weights: they cost 5e-4 of PCC on the real checkpoint and
1.3e-2 on a Gaussian of the same variance, because a block-float format sharing one exponent
across 16 values loses least when the values are correlated and the output has dominant
directions. `work_log.md` §13 has the full table both ways, the measured cost of the BFP8
fallback, and why the structural coverage of the synthetic suite is unaffected — every real
break seen during this stage landed at 0.24-0.50, nowhere near 0.98.

Two things guard against that relaxation being a licence. The first is
`test_real_weight_pcc_at_disputed_lengths` above. The second is new:
**`test_structural_prefill_at_high_precision`** re-runs the same lengths with every weight pinned
to BF16 and HiFi4 — precision removed entirely — at a **0.999** bar against a measured worst of
0.999434. `work_log.md` §20 records why it is prefill-only (the DRAM-sharded decode matmul
cannot be built at BF16) and why an earlier revision wrongly called the whole idea impossible.

The attribution is controlled at the length where it matters most rather than extrapolated:
re-running the full 262143-token case with `OPT_DECODER_PRECISION=bfp8_gate_up` and nothing else
changed moves the `full_attention` prefill tail from 0.985447 to **0.997937** and the
`linear_attention` tail from 0.996595 to **0.999062**, while the conv state, the recurrent state
and both KV caches come back **identical to the last digit**
(`logs/long_context_bfp8_control.log`).

## `$optimize` checklist

Every item, where it was addressed, and the evidence. "N/A" carries a reason, not a shrug.

| item | disposition | evidence |
|---|---|---|
| OPT-001 packed decode QKV | **tried, rejected on measurement** (`O7`). `full_attention` `wqkv`+`wgate` packed is 5.6 us faster in decode but 466 us slower in prefill before the split cost; the gated-delta-net pair is blocked by incompatible output dtypes (`in_proj_ba` must stay float32). | work_log.md §5, `probes/probe_packed_attn.py`, `logs/probe_packed_attn.log` |
| OPT-002 SDPA / KV-cache contract | preserved; an explicit `SDPAProgramConfig` was mandatory and was tried across 18 positions and every legal `q_chunk`/`k_chunk` pair. 4-9x faster and wrong at some position in every case, so none adopted. | work_log.md §10, `logs/sdpa_decode_sweep.log`, `logs/sdpa_decode_positions.log` |
| OPT-003 decode residual layout | `O3`: the residual stays width-sharded in L1 across both norms, the mixer boundary and both residual adds, with `LayerNormShardedMultiCoreProgramConfig` on both norms. Every remaining layout row at a helper boundary is listed with its cost. | work_log.md §4 (boundary table), §19 |
| OPT-004 DRAM-sharded decode geometry | swept per role over core count, `in0_block_w` (to 34) and `per_core_N`, not dtype alone. Shipped `in0_block_w` is 5-17 for every role; **none is ≤ 2**. | work_log.md §3, §15, `probes/decode_matmul_families.py`, `logs/decode_matmul_families.log` |
| OPT-005 logical batch vs tile padding | logical batch is a construction parameter; activations are padded to `round_up(max_batch, 32)` rows and the padding never becomes a user. Page tables, positions and cache slots are per logical user, asserted at batch 4 and 32, eager and traced. | `tests/test_optimized_decoder.py::test_batched_users`, `::test_traced_decode_batched`; `doc/context_contract.json` `batch_contract_unchanged` |
| OPT-006 cumulative sign-off | the shipped configuration is signed off as a whole, not per win: the full suite, the watcher run, the stress pass, the 262143-token case and the headline pair are all re-run on the final code after the last change (`O14`). | work_log.md §16 |
| OPT-007 attention projection precision | its own decode search, field by field, on real weights, re-run on the post-`O2` topology as the item requires. Rejected `attn_qkv`, both output projections, `gdn_qkv`, `gdn_z` and — after adopting and fully gating it — the BFP4 output gate, each on real-weight PCC rather than on margin. | work_log.md §9, §23 |
| OPT-008 row-parallel output projection decompositions | **N/A — single chip.** This stage is a 1x1 mesh with no tensor parallelism, so there is no row-parallel decomposition and no CCL to choose. The multichip stage owns it. | `doc/context_contract.json` `scope`; no CCL op appears in any shipped profile |
| OPT-009 persistent decode CCL buffers | **N/A — single chip**, same reason. | as above |
| OPT-010 packed vs split MLP gate/up | both families measured. Packed is not even legal for the DRAM-sharded decode matmul (one output must be one shard width), and it costs 2.6 ms of prefill; split ships (`O4`). | work_log.md §5 |
| OPT-011 phase-specific activation shards | tried as `O11` — a wider working shard over ttnn's own row-wise core set, to raise the decode matmuls' `in0_block_w`. **Rejected with the exact blocker**: the sharded layernorm that carries the residual refuses a non-rectangular core grid, and no core count is rectangular both ways. Cost of the alternative (one reshard) is in §4's table. | work_log.md §19 |
| OPT-012 synthetic stress must not veto real-weight wins | this is the rule the whole precision policy is built on: BFP4 gate/up costs 1.3e-2 of PCC on a Gaussian and 5e-4 on the real checkpoint, so the synthetic bar moved to 0.98 and a real-weight 0.995 test at the disputed lengths was added. The converse is enforced too — `O14` would have cost no synthetic bar change at all and was rejected anyway, on real weights. | work_log.md §13, §23 |
| OPT-013 prove the policy reached the measured ops | asserted from the device tensors *and* read back out of the shipped profiles: every decode matmul row's input-1 dtype and fidelity is tabulated against the policy. | work_log.md §15, `tests/test_optimized_decoder.py::test_precision_policy_reached_the_weights`, `::test_decode_matmuls_are_dram_sharded` |
| OPT-014 cross precision with geometry | the geometry sweep is precision-locked: each family is measured under the policy's own dtype/fidelity rather than under BF16/HiFi, and the BFP4 rows were re-swept over core/shard geometry separately. | work_log.md §3, §15 |

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
   0.9897 output / 0.9829 state PCC on the real layer's own activations, and **unchanged**
   (0.9898 / 0.9829) when every precision knob Python owns — the preprocessing matmuls and the
   whole `L_inv` solve — is raised from HiFi2 to HiFi4 + fp32 accumulation and the final state
   is read back in float32 (work_log.md §7). The loss is inside the C++ sequential scan, so the
   fix is upstream. Earlier revisions called it the largest `linear_attention` prefill
   opportunity; timing it says otherwise — the adapted path costs **28.89 ms warmed** at 2048
   tokens against 27.11 ms of device time for the *entire* shipped layer, so as it stands it is
   not a latency win either. A fused sequential scan is still the right shape for this problem;
   this one is neither accurate enough nor, as adapted, faster.
5. **The `linear_attention` causal conv carries 1.61 ms of tilize/untilize** (5.6 % of its
   prefill, over 11 ops), because a causal convolution is a sum of row-shifted views and a row
   shift is never tile-aligned. `ttnn.conv1d` — the op that would replace it — refuses the
   contract with an exact L1 blocker at 10240 depthwise channels. This is what is left after
   `O13` removed the one member of that audit that was not the conv at all (§22).
   `full_attention` prefill has no layout op at all. work_log.md §18.
6. **The synthetic-weight suite runs at a 0.98 bar**, with real-weight enforcement at 0.995
   added alongside it. See "Accuracy bars" above and `work_log.md` §13.
7. **Single chip.** Collectives, residual layout across a mesh, fused CCL+matmul and persistent
   CCL buffers are the multichip stage's contract, not this one's. No MoE path either: this
   decoder's MLP is dense.
