# Qwen/Qwen3.6-27B — fused decoder

Graph-fused TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers.
`FusedDecoder` is a drop-in replacement for stage 1's `FunctionalDecoder`: same constructor,
same `prefill_forward` / `decode_forward` / `prefill_chunk_plan` / `prepare_decode_state`
contract, same paged KV cache, same per-user linear-attention state, same acceptance bar. Only
the op graph changes.

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
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) (285 records: 264 PCC, 4 full-context scale ratios, 17 non-numeric)
* Perf: [`perf_summary.json`](perf_summary.json) and [`tracy/`](tracy/)
* Watcher: [`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md) (generated from the committed log by [`probes/make_watcher_audit.py`](probes/make_watcher_audit.py))
* Run logs: [`logs/`](logs/) — `suite_main.log`, `long_context.log`, `watcher_run.log`, `doc_gate.log`, and one per probe
* The volatile figures in these documents live between `<!-- GENERATED:... -->` markers and are written by [`probes/make_doc_tables.py`](probes/make_doc_tables.py) out of `perf_summary.json` and the probe logs, so a re-measurement cannot leave them stale

## Performance — before and after

Warmed, batch 1, one Blackhole chip, from **Tracy device-profiler** runs with the measured
window delimited by signposts. Prefill is one warmed 2048-token pass; decode is **traced** —
capture once, then replay `execute_trace` 8x inside the window, and the number below is the
mean replay. Both implementations were measured by the same script
([`probes/run_perf.sh`](probes/run_perf.sh)) on the same machine, against the same build, with
`<impl>` the only argument that differs; the `.provenance` files next to each CSV carry the
run timestamps. The functional baseline's code is untouched by this stage (`tt/functional_decoder.py` is byte-identical between the two commits and no tt-metal C++
changed), so the pair is like-for-like.

<!-- GENERATED:before_after -->
| layer kind | phase | device time before | device time after | speed-up | ops before | ops after |
|---|---|---|---|---|---|---|
| `linear_attention` | prefill, 2048 tokens | 150.971 ms | **26.275 ms** | **5.75x** | 801 | 74 |
| `linear_attention` | traced decode, 1 token | 3.034 ms | **2.395 ms** | **1.27x** | 92 | 68 |
| `full_attention` | prefill, 2048 tokens | 18.588 ms | **17.776 ms** | **1.05x** | 44 | 28 |
| `full_attention` | traced decode, 1 token | 2.270 ms | **2.066 ms** | **1.10x** | 50 | 44 |
<!-- END GENERATED:before_after -->

Every row is faster **and** smaller; the stage contract is the first of those, not the second.
`tests/test_fused_decoder_docs.py::test_speedup_block_is_consistent` asserts both directions
straight out of `perf_summary.json`, and
`::test_perf_summary_rederives_from_the_report` re-derives every figure in the table by summing
the `Device Time` column of the committed `tt-perf-report` CSVs.

### Where the time goes now

The two decode paths are **DRAM-bandwidth bound on bfloat16 weights**, and that is the ceiling
this stage runs into. The MLP's two matmuls alone are the largest entry of the `full_attention`
decode step: the `32 x 5120 x 34816` gate/up projection moves 356 MB of weights and the
`32 x 17408 x 5120` down projection 178 MB, both at 400-415 GB/s, which is this device's DRAM
roofline. No graph rewrite moves those bytes; a weight-dtype change would, and that is the
datatype-sweep stage's contract, not this one's.

What is left after fusing, per pass. These are the `breakdown_ms` blocks of
[`perf_summary.json`](perf_summary.json), bucketed from the report's own op codes by
[`probes/make_perf_summary.py`](probes/make_perf_summary.py) — not added up by hand — with an
`other` bucket that stays empty only because every op is classified.

<!-- GENERATED:breakdown -->
| bucket | `linear_attention` prefill | `linear_attention` decode | `full_attention` prefill | `full_attention` decode |
|---|---|---|---|---|
| `matmul` (projections, MLP, gated-norm constants) | 14.526 ms | 1.926 ms | 13.472 ms | 1.807 ms |
| `gated_delta_rule` | 2.813 ms | — | — | — |
| `sdpa` | — | — | 1.279 ms | 0.105 ms |
| `batched_matmul` (the decode recurrence) | — | 0.058 ms | — | — |
| `layout` (tilize/untilize/reshape/permute/concat/slice/shard) | 4.835 ms | 0.178 ms | 1.049 ms | 0.037 ms |
| `elementwise` | 3.723 ms | 0.206 ms | 1.016 ms | 0.042 ms |
| `norm` | 0.377 ms | 0.028 ms | 0.537 ms | 0.026 ms |
| `heads_and_cache` | — | — | 0.422 ms | 0.048 ms |
| **total** | **26.275 ms** | **2.395 ms** | **17.776 ms** | **2.066 ms** |
<!-- END GENERATED:breakdown -->

Most of the `linear_attention` prefill's `layout` + `elementwise` is the 4-tap causal conv,
which the probe measures in isolation (`logs/probe_causal_conv.log`) and which `work_log.md`
§3.7 records four whole formulations for; the rest is the delta-rule output relayout (§3.13
measures the alternative at 2x the cost) and the MLP's two slices (§3.8 measures the alternative
as clearly slower).
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
| prefill vs HF, seq 1 / 17 / 128 / 2048 / 2049 / 4096 / 5000 | min 0.999831 | min 0.999388 |
| prefill vs HF, longest single-shot reference length | 0.999881 | 0.999415 |
| decode vs HF, 4 steps after prefill 17 / 2048 / 2049 / 5000 | min 0.999872 | min 0.999178 |
| batch 32 and 4, unequal prompts 64..3071, permuted page table - prefill | min 0.999889 | min 0.999383 |
| batch 32 and 4 - decode | min 0.999878 | min 0.999268 |
| **real checkpoint weights** - prefill @ 2049 | 0.999936 | 0.999964 |
| **real checkpoint weights** - decode @ 2049 | 0.999975 | 0.999988 |
| traced decode, replay output vs HF | min 0.999887 | min 0.999436 |
| traced decode at batch 4, per-user positions | min 0.999916 | min 0.999478 |
| paged K cache vs HF after prefill 2049 | — | 0.999989 |
| paged V cache vs HF after prefill 2049 | — | 0.999993 |
| conv state vs HF after prefill 2049 | 0.999995 | — |
| recurrent state vs HF after prefill 2049 | 0.999938 | — |
| page block size 32 and 128 instead of 64 - prefill | — | min 0.999391 |
| page block size 32 and 128 instead of 64 - decode | — | min 0.999479 |
| BFP8 KV cache - prefill @ 2049 | — | 0.999347 |
| BFP8 KV cache - decode @ 2049 | — | 0.999440 |
| pad-below-one-tile lengths 735..768 - prefill | min 0.999900 | min 0.999447 |
| pad-below-one-tile lengths 735..768 - decode | min 0.999878 | min 0.999328 |
| **full context 262143** - prefill tail vs HF | 0.999879 | 0.998030 |
| **full context 262143** - conv state vs HF | 0.999995 | — |
| **full context 262143** - recurrent state vs HF | 0.999933 | — |
| **full context 262143** - paged K cache vs HF | — | 0.999989 |
| **full context 262143** - paged V cache vs HF | — | 0.999993 |
| **full context 262143** - decode at position 262143 | 0.999918 | 0.999266 |
| **full context 262143** - best-fit *scale* vs HF, prefill tail | 0.997673 | 0.997495 |
| **full context 262143** - best-fit *scale* vs HF, decode | 0.996319 | 0.995766 |
| fused vs functional output, prefill and decode @ 2049 | min 0.999917 | min 0.999859 |

Minimum over all 264 PCC records: **0.998030**, against a bar of 0.995. No exception, no waiver, no open gap.
<!-- END GENERATED:correctness -->

### Delta against the functional stage

Every fused figure sits within a few times 1e-4 of the functional one, in both directions:

<!-- GENERATED:delta -->
| | functional | fused | delta |
|---|---|---|---|
| `linear_attention` full-context prefill tail | 0.999947 | 0.999879 | -6.8e-05 |
| `linear_attention` full-context recurrent state | 0.999984 | 0.999933 | -5.1e-05 |
| `linear_attention` full-context decode @ 262143 | 0.999955 | 0.999918 | -3.7e-05 |
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
| `test_fused_graph_is_smaller` | `ttnn` op count per pass falls - <!-- GENERATED:python_op_counts -->PLACEHOLDER<!-- END GENERATED:python_op_counts --> - counted at the python boundary, so it differs from the device op counts above |
| `test_fused_matches_functional` | fused and functional agree with **each other** from identical weights and inputs, not only with HF |
| `test_no_redundant_relayout_in_measured_prefill` | the measured prefill never converts a tensor's layout and immediately converts it back - matched on the tensor, not on a recycled buffer address. This is what caught a `tilize` -> `untilize` round trip over the whole conv window that three review rounds and the decode-only budget had missed |
| `test_no_relayout_or_host_ops_in_measured_decode` | the layer asks for no `tilize`/`untilize`/`to_layout` in a measured decode, and its reshard count stays inside a budget each remaining reshard's op contract justifies (4 of 6 for `linear_attention`, 9 of 12 for `full_attention`) |
| `test_repeated_runs_stable` | six prefill+decode cycles bit-identical, with per-bank DRAM allocation unchanged from cycle 1 — no per-cycle device leak in the `_free` aliasing rules |
| `test_no_runtime_host_fallback` | source scan **and** a live run with `from_torch`/`to_torch`/`as_tensor` stubbed to raise |

### Capability-contract evidence

| claim | evidence | remaining risk |
|---|---|---|
| Both HF layer kinds implemented and correct through the fused graph | 71 fused tests, both kinds in every parametrised case except the five that are `full_attention`-only by construction; `logs/suite_main.log` = `71 passed, 2 skipped` | Only layers 0 and 3 are instantiated; `decoder_shapes` rejects a third kind |
| Advertised context 262144 preserved, not reduced | `test_full_advertised_context` prefills 262143 and decodes at 262143 for both kinds against a real HF reference; all four PCCs >= 0.998030, all four scale ratios inside ±2 % | Reference is segmented (`linear_attention`) or projection-built and `torch.equal`-validated (`full_attention`), as in stage 1 |
| Fusing did not change capacity | `test_fused_persistent_state_delta` reads the DRAM allocator around a real `from_state_dict` at batch 32: a `linear_attention` layer grows by 8257536 bytes and a `full_attention` layer by exactly 0, against 31 GiB of measured DRAM and a 1818230784-byte worst-case layer — `../context_contract.json` `fused_decoder` block | measured at one `max_batch` (32) |
| Non-aligned logical lengths still work on the public API | 1, 17, 128, 2049, 5000, 8191, 16385, 262143, the 735..768 pad-below-one-tile range, and 31 of the 32 batch-32 prompts | — |
| Paged KV cache still correct under a non-trivial page table | shuffled-permutation page tables; `test_linear_state_and_kv_cache_match_reference` un-pages the device cache and compares against HF's cache object (K 0.999989, V 0.999993) | one length (2049) at batch 1, plus full context |
| Deterministic | bit-identical prefill and decode for repeated identical inputs, both kinds, plus six repeated whole cycles | — |
| Watcher clean | `watcher/WATCHER_AUDIT.md`, generated from the committed log by `probes/make_watcher_audit.py`: 11 passed, and the offender grep returns zero over the whole log | Watcher subset is 11 tests, not the whole suite |
| Documents match their artifacts | `tests/test_fused_decoder_docs.py` re-derives every perf figure from the CSVs, resolves every cited path, and re-checks the evidence summary and the watcher grep | Prose that quotes no number is not checked |

## Running

```bash
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh

# fused suite (71 tests + 2 long-context skips, ~9 min)
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
three alternative causal-conv formulations, split gate/up MLP matmuls, packing
`in_proj_qkv`/`in_proj_z` into the a/b matmul, and removing `repeat_interleave` from the decode
GQA head expansion. See [`work_log.md`](work_log.md) §5.

## Known limitations

* **The decode SDPA still runs on one core per head** (the `sdpa` row of the breakdown table
  above, about 5 % of the `full_attention` decode step). That is not a graph property: stage 1 pins `max_cores_per_head_batch = 1` to
  work around an upstream cross-core tree-reduction defect in `sdpa_decode`, documented there
  with a model-free reproducer and handed to the optimization stage. This stage does not touch
  it.
* **Both decodes are DRAM-bound on bfloat16 weights.** Four fifths of the `linear_attention`
  decode step and seven eighths of the `full_attention` one is the `matmul` bucket of the table
  above, at 400-415 GB/s. Fusing cannot move those bytes; a weight-dtype change can, and belongs to
  the datatype-sweep stage.
* **`repeat_interleave` still relayouts** inside the decode GQA head expansion - two
  `untilize_with_unpadding` + two `tilize_with_val_padding` in the committed decode report, about
  1 % of the step. It is a relayout internal to a dedicated op; removing it needs a recurrent-state
  head ordering that would break the direct comparison of the on-device state against HF's cache
  object (`work_log.md` §5).
* Everything stage 1 listed as a limitation still holds: `prepare_decode_state()` rewrites every
  batch slot, prefill is single-user per call, batch 32 is tested at `max_seq_len` 8192, and
  trace capture is exercised at batch 1 and 4 rather than 32.
