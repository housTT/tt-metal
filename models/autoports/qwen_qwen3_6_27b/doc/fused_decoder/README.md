# Qwen/Qwen3.6-27B — fused decoder

Graph-fused TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers.
`FusedDecoder` is a drop-in replacement for stage 1's `FunctionalDecoder`: same constructor,
same `prefill_forward` / `decode_forward` / `prefill_chunk_plan` / `prepare_decode_state`
contract, same paged KV cache, same per-user linear-attention state, same acceptance bar. Only
the op graph changes, with one documented exception — the packed `conv_state` view during
decode, in [Known limitations](#known-limitations).

Hardware and environment are unchanged from stage 1: one Blackhole chip
(`/dev/tenstorrent/2`, `TT_VISIBLE_DEVICES=2`) of the intact p300c board, 1x1 mesh, this
checkout's own `python_env`, sourced through
[`../functional_decoder/ttenv.sh`](../functional_decoder/ttenv.sh).

* Implementation: [`../../tt/fused_decoder.py`](../../tt/fused_decoder.py)
* Stage-1 baseline it is measured against: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* Tests: [`../../tests/test_fused_decoder.py`](../../tests/test_fused_decoder.py),
  [`../../tests/test_fused_decoder_perf.py`](../../tests/test_fused_decoder_perf.py),
  [`../../tests/test_fused_decoder_docs.py`](../../tests/test_fused_decoder_docs.py)
* Every rewrite, with the measurement that kept or rejected it: [`work_log.md`](work_log.md)
* Probes: [`probes/README.md`](probes/README.md)
* Context capability: [`../context_contract.json`](../context_contract.json)
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) — record counts and minima are the generated block in [`work_log.md`](work_log.md) §4
* Perf: [`perf_summary.json`](perf_summary.json) and [`tracy/`](tracy/)
* Watcher: [`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md) (generated from the committed log by [`probes/make_watcher_audit.py`](probes/make_watcher_audit.py))
* Run logs: [`logs/`](logs/) — `suite_main.log`, `long_context.log`, `watcher_run.log`, `doc_gate.log`, and one per probe
* The volatile figures in these documents live between `<!-- GENERATED:... -->` markers and are written by [`probes/make_doc_tables.py`](probes/make_doc_tables.py) out of `perf_summary.json` and the probe logs, so a re-measurement cannot leave them stale

## Performance — before and after

Warmed, one Blackhole chip, from **Tracy device-profiler** runs with the measured window
delimited by signposts. Prefill is one warmed 2048-token pass; decode is **traced** — capture
once, then replay `execute_trace` 8x inside the window, and the number below is the mean replay.
Decode is measured at batch 1 *and* at the advertised `max_batch` of 32, because those are two
different graphs rather than one graph with a wider tensor: at 32 the z-gated norm switches to
the group-reduction form and every recurrence op grows 32-fold. Both implementations were measured by the same script
([`probes/run_perf.sh`](probes/run_perf.sh)) on the same machine, against the same build, with
`<impl>` the only argument that differs; the `.provenance` files next to each CSV carry the
run timestamps. The functional baseline's code is untouched by this stage (`tt/functional_decoder.py` is byte-identical between the two commits and no tt-metal C++
changed), so the pair is like-for-like.

<!-- GENERATED:before_after -->
| layer kind | phase | device time before | device time after | speed-up | ops before | ops after |
|---|---|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 151.210 ms | **26.169 ms** | **5.78x** | 801 | 68 |
| `linear_attention` | traced decode, 1 token, batch 1 | 3.036 ms | **2.355 ms** | **1.29x** | 92 | 66 |
| `linear_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 36.620 ms | **6.170 ms** | **5.93x** | 93 | 69 |
| `full_attention` | prefill, 2048 tokens | 18.633 ms | **17.779 ms** | **1.05x** | 44 | 28 |
| `full_attention` | traced decode, 1 token, batch 1 | 2.276 ms | **2.067 ms** | **1.10x** | 50 | 44 |
| `full_attention` | traced decode, 1 token, batch 32 (advertised `max_batch`) | 3.069 ms | **2.910 ms** | **1.05x** | 49 | 43 |
<!-- END GENERATED:before_after -->

Every row is faster **and** smaller; the stage contract is the first of those, not the second.
The batch-32 `linear_attention` row is the largest single win in the stage after the prefill:
the spelled-out recurrence ran 48 head matmuls per user there.
`tests/test_fused_decoder_docs.py::test_speedup_block_is_consistent` asserts both directions
straight out of `perf_summary.json`, and
`::test_perf_summary_rederives_from_the_report` re-derives every figure in the table by summing
the `Device Time` column of the committed `tt-perf-report` CSVs.

### Where the time goes now

At batch 1 both decode paths are **DRAM-bandwidth bound on bfloat16 weights**, and that is the
ceiling this stage runs into there. The MLP's two matmuls alone are the largest entry of the
`full_attention` decode step: the `32 x 5120 x 34816` gate/up projection moves 356 MB of weights
and the `32 x 17408 x 5120` down projection 178 MB, both at the DRAM roofline the profiler's own
`DRAM` column reports for them. No graph rewrite moves those bytes; a weight-dtype change would,
and that is the datatype-sweep stage's contract, not this one's.

At the advertised `max_batch` of 32 that is only half the story, which is why the table below
carries all six measured passes rather than the four the stage first measured. The weights are
the same 534 MB — one token or thirty-two, a decode step reads every weight once — so the
`matmul` bucket barely moves, and everything that scales *with* the batch becomes visible
instead. In `linear_attention` decode at batch 32 the recurrence's own work is the story: its
`batched_matmul` and `elementwise` buckets grow by more than twenty-fold and nearly ten-fold — the two
`decode b32` columns of the table below against their `decode b1` neighbours — because the
carried state is `[batch * 48, 128, 128]` float32 — 100 MB at batch 32 — and a step decays it,
reads it and writes it back. That is bandwidth against the state, not dispatch overhead, and
`work_log.md` §6.1 records what was measured against it: thirteen core grids at both regimes
(the shipped ones win at both), the `exp`/`sigmoid` folds (taken), and a per-head layout change
that would trade the state's shape for its padding (measured slower, §6). The `full_attention`
decode's batch-32 cost is dominated instead by SDPA — its `sdpa` bucket is eight times the
batch-1 one, the two `sdpa` cells of the table below — which is the one-core-per-head
workaround stage 1 pinned and handed to the optimization stage — 29.7 % of that step at batch 32
against 5.1 % at batch 1.

What is left after fusing, per pass. These are the `breakdown_ms` blocks of
[`perf_summary.json`](perf_summary.json), bucketed from the report's own op codes by
[`probes/make_perf_summary.py`](probes/make_perf_summary.py) — not added up by hand — with an
`other` bucket that stays empty only because every op is classified.

<!-- GENERATED:breakdown -->
| bucket | `linear_attention` prefill | `linear_attention` decode b1 | `linear_attention` decode b32 | `full_attention` prefill | `full_attention` decode b1 | `full_attention` decode b32 |
|---|---|---|---|---|---|---|
| `matmul` (projections, MLP, gated-norm constants) | 14.477 ms | 1.883 ms | 1.907 ms | 13.470 ms | 1.809 ms | 1.808 ms |
| `gated_delta_rule` | 2.848 ms | — | — | — | — | — |
| `sdpa` | — | — | — | 1.280 ms | 0.105 ms | 0.863 ms |
| `batched_matmul` (the decode recurrence) | — | 0.060 ms | 1.345 ms | — | — | — |
| `layout` (tilize/untilize/reshape/permute/concat/slice/shard) | 4.782 ms | 0.179 ms | 0.982 ms | 1.060 ms | 0.037 ms | 0.048 ms |
| `elementwise` | 3.686 ms | 0.205 ms | 1.906 ms | 1.015 ms | 0.042 ms | 0.043 ms |
| `norm` | 0.375 ms | 0.028 ms | 0.031 ms | 0.537 ms | 0.026 ms | 0.029 ms |
| `heads_and_cache` | — | — | — | 0.417 ms | 0.048 ms | 0.119 ms |
| **total** | **26.169 ms** | **2.355 ms** | **6.170 ms** | **17.779 ms** | **2.067 ms** | **2.910 ms** |
<!-- END GENERATED:breakdown -->

Most of the `linear_attention` prefill's `layout` + `elementwise` is the 4-tap causal conv,
which the probe measures in isolation (`logs/probe_causal_conv.log`) and which `work_log.md`
§3.7 records five whole formulations for; the rest is the delta-rule output relayout (§3.13
measures the alternative at 2x the cost), the MLP's two slices (§3.8 measures the alternative
as clearly slower), the three `_split_qkv` slices and the rank-3 conversion of `beta`/`g` (§3.19
measures moving that rank change and finds a tie). The batch-32 decode's `layout` bucket is the
same per-head rank changes as batch 1, 32 times as wide: a `[1, batch * 48, 1, 128]` TILE tensor
carries 32 padded rows for every real one, which is a property of the recurrent state's layout
rather than of the graph over it, and §6 records it as such.
The `full_attention` prefill moves least of the four because it was **already** a fused graph in
stage 1 — `nlp_create_qkv_heads`, `chunked_scaled_dot_product_attention`, `paged_fill_cache` and
`nlp_concat_heads` were all in place — so what remained to fuse there was the partial RoPE, the
MLP's SiLU and the output gate's sigmoid, against the matmul and SDPA time in the table above,
which no graph rewrite touches.

## Correctness

Acceptance bar: **PCC >= 0.995**, the same bar, the same HF reference harness and the same
sequence-length coverage as stage 1. Every row below is the minimum over that measurement's
records in [`pcc_evidence.json`](pcc_evidence.json), read out of it by
[`probes/make_doc_tables.py`](probes/make_doc_tables.py).

<!-- GENERATED:correctness -->
| measurement | `linear_attention` | `full_attention` |
|---|---|---|
| prefill vs HF, seq 1 / 17 / 128 / 2048 / 2049 / 4096 / 5000 | min 0.999828 | min 0.999388 |
| prefill vs HF, longest single-shot reference length | 0.999882 | 0.999415 |
| decode vs HF, 4 steps after prefill 17 / 2048 / 2049 / 5000 | min 0.999869 | min 0.999178 |
| batch 32 and 4, unequal prompts 64..3071, permuted page table - prefill | min 0.999889 | min 0.999383 |
| batch 32 and 4 - decode | min 0.999864 | min 0.999268 |
| **real checkpoint weights** - prefill @ 2049 | 0.999937 | 0.999964 |
| **real checkpoint weights** - decode @ 2049 | 0.999973 | 0.999988 |
| traced decode, replay output vs HF | min 0.999881 | min 0.999436 |
| traced decode at batch 4, per-user positions | min 0.999858 | min 0.999278 |
| paged K cache vs HF after prefill 2049 | — | 0.999989 |
| paged V cache vs HF after prefill 2049 | — | 0.999993 |
| conv state vs HF after prefill 2049 | 0.999995 | — |
| recurrent state vs HF after prefill 2049 | 0.999938 | — |
| page block size 32 and 128 instead of 64 - prefill | — | min 0.999391 |
| page block size 32 and 128 instead of 64 - decode | — | min 0.999479 |
| BFP8 KV cache - prefill @ 2049 | — | 0.999347 |
| BFP8 KV cache - decode @ 2049 | — | 0.999440 |
| pad-below-one-tile lengths 735..768 - prefill | min 0.999900 | min 0.999447 |
| pad-below-one-tile lengths 735..768 - decode | min 0.999882 | min 0.999328 |
| **full context 262143** - prefill tail vs HF | 0.999879 | 0.998030 |
| **full context 262143** - conv state vs HF | 0.999995 | — |
| **full context 262143** - recurrent state vs HF | 0.999934 | — |
| **full context 262143** - paged K cache vs HF | — | 0.999989 |
| **full context 262143** - paged V cache vs HF | — | 0.999993 |
| **full context 262143** - decode at position 262143 | 0.999915 | 0.999266 |
| **full context 262143** - best-fit *scale* vs HF, prefill tail | 0.997682 | 0.997495 |
| **full context 262143** - best-fit *scale* vs HF, decode | 0.996456 | 0.995766 |
| fused vs functional output, prefill and decode @ 2049 | min 0.999921 | min 0.999859 |

Minimum over all 396 PCC records: **0.998030**, against a bar of 0.995. No exception, no waiver, no open gap.
<!-- END GENERATED:correctness -->

### Delta against the functional stage

Every fused figure sits within a few times 1e-4 of the functional one, in both directions:

<!-- GENERATED:delta -->
| | functional | fused | delta |
|---|---|---|---|
| `linear_attention` full-context prefill tail | 0.999947 | 0.999879 | -6.8e-05 |
| `linear_attention` full-context recurrent state | 0.999984 | 0.999934 | -5.0e-05 |
| `linear_attention` full-context decode @ 262143 | 0.999955 | 0.999915 | -4.0e-05 |
| `full_attention` full-context prefill tail | 0.998031 | 0.998030 | -4.4e-07 |
| `full_attention` full-context decode @ 262143 | 0.999201 | 0.999266 | +6.5e-05 |
<!-- END GENERATED:delta -->

The `linear_attention` deltas have one cause, and it is a *deliberate* one: the causal-conv FIR
now runs in bfloat16 (`work_log.md` §3.7). Its own output PCC against torch is 0.999990, and it
feeds `chunk_gated_delta_rule`, whose contract casts q/k/v to bfloat16 anyway, so the extra
float32 precision was being discarded one op later. The conv **state** — the quantity that is
carried across chunks and compared against HF's cache object — stays float32 and is cut from the
float32 inputs, which is why `conv_state_pcc` is unchanged at 0.999995. The `full_attention`
deltas are ordinary bfloat16 reassociation from folding the output gate's sigmoid into its
multiply and rewriting the partial RoPE; the decode figure moved *up*.

### Fusing-specific gates

Correct-but-unfused would pass every PCC test above, so the suite also asserts that the fused
graph is the one running:

| test | what it pins |
|---|---|
| `test_fused_ops_are_dispatched` | `chunk_gated_delta_rule`, `rotary_embedding_hf` and `rotate_half` are really dispatched on a real prefill/decode pass; the call counts are recorded in `pcc_evidence.json` rather than asserted, so a graph change that dispatches one more is not a failure |
| `test_fused_graph_is_smaller` | `ttnn` op count per pass falls, counted at the python boundary (so it differs from the device op counts above) - see below |
| `test_fused_matches_functional` | fused and functional agree with **each other** from identical weights and inputs, not only with HF |
| `test_no_layout_round_trip_in_the_measured_pass` (in `test_fused_decoder_docs.py`) | the committed `tt-perf-report` op sequence contains no `Tilize*` immediately followed by an `Untilize*`. Reading the *device* report is the point: `ttnn.concat` and `ttnn.slice` relayout inside themselves, so a python-level trap cannot see them - which is how a round trip over the whole conv window survived three review rounds |
| `test_no_redundant_relayout_in_measured_prefill` | the same property at the python-call level, on the calls the *layer itself* makes. Weaker than the report-level check above and kept alongside it, not instead of it |
| `test_no_relayout_or_host_ops_in_measured_decode` | the layer asks for no `tilize`/`untilize`/`to_layout` in a measured decode, and its reshard count is **exactly** the set each remaining reshard's op contract justifies — 4 for `linear_attention`, 9 for `full_attention`, enumerated in the test |
| `test_repeated_runs_stable` | six prefill+decode cycles bit-identical, with per-bank DRAM allocation unchanged from cycle 1 — no per-cycle device leak in the `_free` aliasing rules |
| `test_no_runtime_host_fallback` | source scan **and** a live run with `from_torch`/`to_torch`/`as_tensor` stubbed to raise |

The python-boundary op counts `test_fused_graph_is_smaller` recorded, read out of
[`pcc_evidence.json`](pcc_evidence.json):

<!-- GENERATED:python_op_counts -->
`full_attention` 53 -> 37 prefill, 55 -> 49 decode; `linear_attention` 780 -> 69 prefill, 78 -> 64 decode
<!-- END GENERATED:python_op_counts -->

### Capability-contract evidence

| claim | evidence | remaining risk |
|---|---|---|
| Both HF layer kinds implemented and correct through the fused graph | the whole fused suite, both kinds in every parametrised case except the five that are `full_attention`-only by construction; the pass counts are the generated block in [`work_log.md`](work_log.md) §4 | Only layers 0 and 3 are instantiated; `decoder_shapes` rejects a third kind |
| Advertised context 262144 preserved, not reduced | `test_full_advertised_context` prefills 262143 and decodes at 262143 for both kinds against a real HF reference; all four PCCs >= 0.998030, all four scale ratios inside ±2 % | Reference is segmented (`linear_attention`) or projection-built and `torch.equal`-validated (`full_attention`), as in stage 1 |
| Fusing did not change capacity | `test_fused_persistent_state_delta` reads the DRAM allocator around a real `from_state_dict` at batch 32: a `linear_attention` layer grows by 9601024 bytes and a `full_attention` layer by exactly 0, against 31 GiB of measured DRAM and a 1818230784-byte worst-case layer — `../context_contract.json` `fused_decoder` block | measured at one `max_batch` (32) |
| Non-aligned logical lengths still work on the public API | 1, 17, 128, 2049, 5000, 8191, 16385, 262143, the 735..768 pad-below-one-tile range, and 31 of the 32 batch-32 prompts | — |
| Paged KV cache still correct under a non-trivial page table | shuffled-permutation page tables; `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares against HF's cache object (K 0.999989, V 0.999993) | one length (2049) at batch 1, plus full context |
| Deterministic | bit-identical prefill and decode for repeated identical inputs, both kinds, plus six repeated whole cycles | — |
| Watcher clean | `watcher/WATCHER_AUDIT.md`, generated from the committed log by `probes/make_watcher_audit.py`: 11 passed, and the offender grep returns zero over the whole log | Watcher subset is 11 tests, not the whole suite |
| Documents match their artifacts | `tests/test_fused_decoder_docs.py` re-derives every perf figure from the CSVs, resolves every cited path, and re-checks the evidence summary and the watcher grep | Prose that quotes no number is not checked |

## Running

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh

# fused suite (~9 min; the two long-context cases are skipped without --long-context)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s

# full advertised context, 262143-token prompt + decode at 262143 (2 tests, ~6 min)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# document/artifact consistency gate (no device)
python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder_docs.py -v

# before/after profiling, one (kind, phase, impl) triple at a time
doc/fused_decoder/probes/run_perf.sh linear_attention prefill functional
doc/fused_decoder/probes/run_perf.sh linear_attention prefill fused
python models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/make_perf_summary.py

# collect every recorded number
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/logs/{suite_main,long_context,watcher_run}.log \
    --out models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/pcc_evidence.json
```

Tests use the same synthetic weights as stage 1, generated deterministically from
[`../functional_decoder/weight_stats.json`](../functional_decoder/weight_stats.json) at the real
config shapes; `test_real_weights` loads the actual checkpoint.

## What is fused

Full narrative, with the measurement that kept or rejected each one, in
[`work_log.md`](work_log.md). In brief, by the skill's priority order:

**Dedicated fused ops.** `ttnn.transformer.chunk_gated_delta_rule` replaces the whole
`linear_attention` prefill delta-rule core — head split, L2 norms, GQA head expansion, scale,
decay cumsum and mask, the recursive WY inverse and the python loop over sub-chunks, ~700 device
ops — with one op, called on the flat rank-3 path at `chunk_size=32` so the norm and the scale
happen in-kernel too. `ttnn.experimental.rotary_embedding_hf` replaces the prefill partial RoPE,
`ttnn.experimental.rotate_half` the decode one's rotate-half, and `ttnn.rms_norm` the decode
GatedDeltaNet Q/K L2 norm.

**Graph rewrites.** The per-head gated RMS norm becomes a group reduction against two constant
matrices so it runs on the flat token-major layout with no tile relayout; the decode RMS norms
run width-sharded across 20 cores instead of on one; the causal-conv FIR runs in bfloat16 with a
float32 carried state; the decode conv state is kept as per-tap batch-major buffers so the shift
is a copy chain instead of an untilize/tilize sandwich; `in_proj_b` and `in_proj_a` become one
matmul; the recurrence matmuls get an explicit core grid; decode V never leaves the memory
config `nlp_create_qkv_heads_decode` produced.

**Op merging.** The MLP's SiLU, the attention output gate's sigmoid and the delta net's
`silu(z)` all fold into the multiply that consumes them; the causal conv's trailing SiLU folds
into the last tap's add; the recurrent-state write-back folds into the add that produces it; and
the prefill KV-cache typecasts are guarded on dtype so the default bfloat16 cache does not pay
for two `bfloat16 -> bfloat16` no-ops.

**Assessed and rejected**, each with an exact blocker or a measurement:
`gated_delta_attn_seq`, `use_qk_l2norm=True`, chunk size 64, `rotary_embedding_hf` decode mode,
a head-channel permutation that would make RoPE a single op, `paged_fused_update_cache`,
`group_attn_matmul`, `hc_sum_reduce` / `repeat_and_interleave_eltwise_mul`, `ttnn.conv1d`,
`output_head_major` on the delta-rule op, `ttnn.addcmul` for the conv taps,
four alternative causal-conv formulations, split gate/up MLP matmuls, packing
`in_proj_qkv`/`in_proj_z` into the a/b matmul, and removing `repeat_interleave` from the decode
GQA head expansion. See [`work_log.md`](work_log.md) §5.

## Known limitations

* **The packed `conv_state` is not written by a fused decode step.** The functional layer
  rewrites `[1, batch, K, conv_dim]` every step; the fused one keeps the same state as `K`
  batch-major per-row buffers, because a decode step then reads whole buffers instead of slicing
  a tile-height axis (§3.11). The buffers *are* the packed rows, so
  `FusedDecoder.current_conv_state()` folds them back exactly and
  `test_conv_state_after_decode_matches_reference` checks the result against HF's own cache
  object after 1 and 5 decode steps. The limitation is the API asymmetry: a consumer written
  against the stage-1 attribute reads a post-prefill window instead of an error, and the
  accessor exists only on the subclass, because this stage's scope is `tt/fused_decoder.py` and
  adding the method to the base class is a stage-1 edit. A serving stage that inspects conv
  state mid-generation must call the accessor.

* **The decode SDPA still runs on one core per head** (the `sdpa` row of the breakdown table
  above — 5.1 % of the `full_attention` decode step at batch 1 and **29.7 %** at the advertised
  `max_batch` of 32, which is the number the optimization stage should plan against). That is not a graph property: stage 1 pins `max_cores_per_head_batch = 1` to
  work around an upstream cross-core tree-reduction defect in `sdpa_decode`, documented there
  with a model-free reproducer and handed to the optimization stage. This stage does not touch
  it.
* **Both decodes are DRAM-bound on bfloat16 weights.** Four fifths of the `linear_attention`
  decode step and seven eighths of the `full_attention` one — at batch 1 — is the `matmul` bucket of the table
  above, at the DRAM roofline. Fusing cannot move those bytes; a weight-dtype change can, and belongs to
  the datatype-sweep stage.
* **`repeat_interleave` still relayouts** inside the decode GQA head expansion - two
  `untilize_with_unpadding` + two `tilize_with_val_padding` in the committed decode report, about
  1 % of the step. It is a relayout internal to a dedicated op; removing it needs a recurrent-state
  head ordering that would break the direct comparison of the on-device state against HF's cache
  object (`work_log.md` §5).
* Everything stage 1 listed as a limitation still holds: `prepare_decode_state()` rewrites every
  batch slot, prefill is single-user per call, batch 32 is tested at `max_seq_len` 8192, and
  trace capture is exercised at batch 1 and 4 rather than 32.
