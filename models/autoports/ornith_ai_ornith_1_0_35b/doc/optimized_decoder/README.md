# Ornith-1.0-35B — optimized decoder (TTNN, single Blackhole device)

Stage deliverable: a performance-optimized TTNN decoder layer for `ornith-ai/Ornith-1.0-35B`, both
layer kinds, that computes what the fused decoder computes, supports everything the fused decoder
supported, and is materially faster in both phases.

<!-- generated:headline -->
| Window | before (fused) | after (optimized) | speedup |
| --- | --- | --- | --- |
| `linear_attention` prefill, 2048 tokens | 257.88 ms / 7941.5 tok/s | **102.120 ms / 20055.1 tok/s** | **2.53x** |
| `linear_attention` decode, traced | 2.06 ms / 484.7 steps/s | **1.070 ms / 934.9 steps/s** | **1.93x** |
| `full_attention` prefill, 2048 tokens | 243.56 ms / 8408.7 tok/s | **96.350 ms / 21256.3 tok/s** | **2.53x** |
| `full_attention` decode, traced | 1.83 ms / 546.5 steps/s | **0.857 ms / 1167.2 steps/s** | **2.14x** |
<!-- /generated:headline -->

* Implementation: [`tt/optimized_decoder.py`](../../tt/optimized_decoder.py) — self-contained; it
  reuses `tt/model_config.py` and leaves `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tt/moe.py` and `tt/rope.py` untouched, because the fused decoder is this stage's measured
  baseline and is built **alongside** the optimized one in the same process for every "before"
  number.
* Tests: [`tests/test_optimized_decoder.py`](../../tests/test_optimized_decoder.py)
* Capability contract: [`../context_contract.json`](../context_contract.json)
  (`optimized_decoder` section)
* Candidate-by-candidate narrative, the operation-topology audit, rejected options:
  [`work_log.md`](work_log.md)
* One command to regenerate every artifact below: [`logs/run_evidence.sh`](logs/run_evidence.sh)

Hardware: 1×1 Blackhole mesh (`p300c`, 11×10 compute grid).

**Artifact sizes and the committed set.** Two repo rules shape what is on disk here. The pre-commit
hook rejects files over 500 KB, so the three big text logs (the pytest suite log, the watcher console
log and the 4 MB watcher log) and the two decode perf-report tables are committed **gzipped**. And
`.gitignore` carries a blanket `*.csv`, so **every** report CSV is gzipped unconditionally — not just
the ones over the size limit — because a plain `.csv` here is silently not committed at all. Review
round 3 found exactly that hole: `full_attention/decode_perf_report.csv` was 477 600 B, under the
gzip threshold, and therefore absent from the commit while `make_readme.py --check` still passed
against the working tree. `run_evidence.sh` now ends by re-running all three generators against
`git archive HEAD`, which is the check that catches it.

Every generator (`logs/make_readme.py`, `tracy/perf_accounting.py`, `watcher/census.py`) reads
`foo.ext` or `foo.ext.gz` transparently, and `perf_accounting.py` now *fails* rather than writing a
partial summary when an input is missing. The raw `decode_ops.csv.gz` Tracy dumps are the one thing
deliberately not committed: 620–720 KB each even gzipped, and exactly what `$optimize` lists as not
worth copying back. The per-op `decode_perf_report.csv.gz` that every table and the accounting read
*is* committed, for both layer kinds; `tracy/run_profiling.sh` regenerates the raw dumps.

---

## 1. Contract — unchanged

```python
OptimizedDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device,
    max_context=None,            # defaults to text_config.max_position_embeddings (262144)
    page_block_size=64, prefill_chunk=2048,
    moe_group_tokens=32, rope_mode="partial",
    dtype=ttnn.bfloat16,
    policy=DEFAULT_POLICY,       # NEW: per-tensor-group weight dtype + math fidelity
) -> OptimizedDecoder

decoder.allocate_kv_cache(num_blocks, dtype=None)   # None -> policy.kv_cache_dtype
decoder.attach_kv_cache(k_cache, v_cache)
decoder.allocate_state(batch_size)
decoder.reset_state()

decoder.prefill_forward(x, *, start_pos=0, page_table=None, chunk_size=None)
decoder.decode_forward(x, *, current_pos=None, rot_idxs=None, page_table=None)
```

Argument shapes, dtypes and semantics are exactly the fused decoder's — including that `seq_len`
may be **any** value in `[1, max_context - start_pos]` with no tile, page or chunk divisibility
requirement, and that `current_pos`/`rot_idxs` are device tensors so a captured decode trace only
needs its input buffers refreshed. `tests/test_optimized_decoder.py` re-asserts every capability the
fused suite asserted, under the same test names.

The one addition is `policy`. `PrecisionPolicy` names a weight dtype and a math fidelity **per
tensor group** — routed experts, dense projections, shared expert, router, recurrent state, KV
cache, expert activations — so each group can be swept independently and a regression can be
assigned. `POLICIES["fused-parity"]` reproduces the fused decoder's dtypes exactly and is what the
"before" arm of every dtype comparison uses; `allocate_kv_cache` still takes an explicit dtype.

---

## 2. Correctness

Acceptance bar: **PCC ≥ 0.995**, inherited from the functional-decoder stage and not lowered.
Golden = one `Qwen3_5MoeDecoderLayer` in float32 driven through the real HF prefill/decode cache
paths. Layer 0 is `linear_attention`, layer 3 is `full_attention`. PCC is accumulated in float64.

```bash
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_optimized_decoder.py -v -p no:randomly
```

<!-- generated:suite-result -->
**109 passed** in 569.11 s.
<!-- /generated:suite-result -->

Log: [`logs/pytest_full_suite.txt.gz`](logs/pytest_full_suite.txt.gz). The `long` (262144-token) cases are
**not** opt-in — they run in the same invocation, so the tables below and the advertised-context
evidence come from one log.

### 2.1 Prefill PCC against the float32 HF golden

<!-- generated:prefill-pcc -->
| `seq_len` | `linear_attention` | `full_attention` |
| --- | --- | --- |
| 1 | 0.999994 | 0.999913 |
| 7 | 0.999992 | 0.999899 |
| 32 | 0.999974 | 0.999916 |
| 64 | 0.999974 | 0.999912 |
| 128 | 0.999961 | 0.999923 |
| 129 | 0.999961 | 0.999924 |
| 250 | 0.999938 | 0.999931 |
| 2048 | 0.999923 | 0.999946 |
| 2049 | 0.999924 | 0.999946 |
| 3000 | 0.999919 | 0.999949 |
<!-- /generated:prefill-pcc -->

### 2.2 Decode PCC — four steps after a prefill, each step compared

`prefill_len = 130` makes the decode writes cross a 64-token page boundary; `2048` puts them past
an internal chunk boundary.

<!-- generated:decode-pcc -->
| prefill length | step | `linear_attention` | `full_attention` |
| --- | --- | --- | --- |
| 130 | 0 | 0.999985 | 0.999918 |
| 130 | 1 | 0.999982 | 0.999925 |
| 130 | 2 | 0.999932 | 0.999895 |
| 130 | 3 | 0.999989 | 0.999898 |
| 2048 | 0 | 0.999983 | 0.999960 |
| 2048 | 1 | 0.999973 | 0.999955 |
| 2048 | 2 | 0.999865 | 0.999914 |
| 2048 | 3 | 0.999977 | 0.999914 |
<!-- /generated:decode-pcc -->

### 2.3 Optimized vs fused

The two implementations are not bit-identical and are not meant to be — this stage carries BFP4
expert weights and BFP8 projection weights where the fused decoder carried bfloat16. They agree far
more tightly than either agrees with HF all the same, which is what `test_optimized_matches_fused`
asserts at seq 1 / 130 / 300 for both phases against a 0.998 bar; the per-case values are logged into
[`logs/pytest_full_suite.txt.gz`](logs/pytest_full_suite.txt.gz) (`grep optimized-vs-fused`). The
acceptance gate is the HF bar above; that test exists to catch a *structural* divergence — a wrong
active-expert set, a dropped residual, a mis-sliced projection — that the HF bar might absorb.

### 2.4 Everything else the fused stage asserted

Re-asserted under the same test names, against the optimized decoder: paged-KV correctness and
per-user page tables, ragged decode positions, chunked-prefill continuation, non-aligned
`max_context`, determinism, freed-DRAM independence (`test_forward_with_poisoned_free_pool`),
trace-replay PCC, batched prefill/decode at batch 4 and 32, decode batches past both op fallbacks
(40 and 56), `moe_group_tokens` 64 and 256, both RoPE modes, synthetic as well as real weights, the
full 262144-token context at an aligned and a non-aligned length, and chunk-size invariance at the
full context.

### 2.5 Added by this stage

| Test | What it pins |
| --- | --- |
| `test_optimized_matches_fused` | optimized vs fused PCC at seq 1 / 130 / 300, prefill and decode |
| `test_precision_policy_reaches_the_device_tensors` | every weight tensor and the KV cache hold the dtype the policy names (OPT-013, code half) |
| `test_decode_runs_the_tuned_program_configs` | all five dense decode matmuls carry a tuned 1D program config with `in0_block_w ≥ 2`; both routed sparse matmuls carry a program config and write their `num_experts`-wide output to **L1** |
| `test_padded_rows_do_not_route` | the decode MoE's tile-padding rows add no experts to the sparsity, and removing that masking changes the expert count but not the output — with a liveness control |
| `test_optimized_beats_fused_traced_decode` | the optimized traced decode is faster than the fused one, same process, same device, same weights |
| `test_no_layout_churn_in_measured_forward` | the exact layout-conversion budget, itemised, including the conversions the sharded norms add |

---

## 3. What changed

Nothing about *what* the layer computes. The changes are, in decreasing order of measured effect
(the full ordered table with the measurement after each step is [`work_log.md`](work_log.md) §3):

| Kind | Change |
| --- | --- |
| Memory placement | Every routed-expert intermediate — the packed gate/up output, the two unpacking slices, the SwiGLU product, the scored activation, the down output and the expert reduction — moved from DRAM to **L1**. |
| Program config | Both routed `sparse_matmul` calls take a geometry chosen from the call's **active-expert bound**: 8 cores with a wide output block for a decode step, 32 cores with `per_core_N` 1 for a prefill group. |
| Program config | All five dense decode projections take an explicit 1D `mcast_in0` config tuned per role (core grid, `in0_block_w`, `per_core_N`, output subblock, L1 output). |
| Precision | Routed expert weights bfloat16 → **BFP4** with **LoFi**; dense projection weights → **BFP8** with **HiFi2**; routed expert activations → **BFP8**; KV cache → **BFP8**. Router weight, norms, RoPE tables and the DeltaNet float32 state are unchanged. |
| Topology | The decode MoE routes only its **real** token rows; the tile padding no longer contributes experts to the sparsity (16 → 8 active experts at batch 1). |
| Layout | The decode RMSNorms run **width-sharded in L1** on a `LayerNormShardedMultiCoreProgramConfig` instead of on one core. |

Everything the fused stage built is preserved: the packed projections, the dedicated head-split /
RoPE / paged-cache / expert-reduction ops, the flat `chunk_gated_delta_rule` contract, the
`ttnn.conv1d` depthwise path with its FIR fallback, the router hoist and the 32-token expert group.

---

## 4. Precision and fidelity policy

The fused stage deliberately changed no dtype and no math fidelity, so its whole weight set was
bfloat16 and every dense matmul ran HiFi4. This stage tunes each tensor group separately, one at a
time, and every candidate is measured on **real Ornith-1.0-35B weights** — never on synthetic ones.

### 4.1 Selected policy

| Tensor group | fused | optimized | fidelity | why |
| --- | --- | --- | --- | --- |
| routed expert `gate`/`up` | BF16 | **BFP4** | HiFi4 → **LoFi** | the two largest ops in both windows; the BFP8 arm (0.899 vs 0.858 ms) and the HiFi2 arm (0.895) are both measured in §4.2 and both lose |
| routed expert `down` | BF16 | **BFP4** | **LoFi** | BFP8 measured slower (0.875 vs 0.858 ms) *and* 1.9x the weight bytes |
| dense projections (`attn_in`, `o_proj`, `gdn_in`, `gdn_out`) | BF16 | **BFP8** | HiFi4 → **HiFi2** | BFP4 is faster but costs 30–43× the layer error on the real-weight ladder — §4.3 |
| shared expert (`gate|up|router` pack, `down`) | BF16 | **BFP8** | HiFi4 → **HiFi2** | BFP4+LoFi measured *no* gain, so the higher precision is free |
| router projection | BF16 | **BF16** | **HiFi4**, fp32 acc | expert *selection* is a discrete decision; a rounding change here swaps an expert rather than perturbing a value |
| routed expert activations | BF16 | **BFP8** | — | halves every byte of the `num_experts`-wide intermediate chain; the bfloat16 arm is 4.9 % slower (§4.2) |
| paged KV cache | BF16 | **BFP8** | — | halves the cache; prefill fill tensors are cast to it explicitly, decode `paged_update_cache` inputs stay BF16 |
| norms, RoPE tables, DeltaNet state / gates / router logits | BF16 / FP32 | **unchanged** | — | none is a decode-time DRAM cost, so a shared exponent would be loss for nothing |

`test_precision_policy_reaches_the_device_tensors` asserts the built layer's tensors hold exactly
these dtypes, and §5's `tt-perf-report` rows show the same dtypes at the **measured matmuls** — both
halves of OPT-013.

### 4.2 One-group-at-a-time sweep

<!-- generated:policy-sweep -->
| candidate | `full` decode ms | `linear` decode ms | `full` / `linear` prefill screen PCC |
| --- | --- | --- | --- |
| **(selected policy)** | 0.858 | 1.071 | 0.999946 / 0.994385 |
| `proj_dtype=bfloat4_b` | 0.838 | 1.048 | 0.998860 / 0.992463 |
| `proj_fidelity=LoFi` | 0.857 | 1.070 | 0.999937 / 0.994642 |
| `expert_fidelity=HiFi2` | 0.895 | 1.109 | 0.999946 / 0.994385 |
| `expert_gate_up_dtype=bfloat8_b` | 0.899 | 1.121 | 0.999956 / 0.994386 |
| `shared_dtype=bfloat4_b,shared_fidelity=LoFi` | 0.856 | 1.070 | 0.999902 / 0.994359 |
| `expert_down_dtype=bfloat8_b` | 0.875 | 1.091 | 0.999951 / 0.994386 |
| `kv_cache_dtype=bfloat16` | 0.858 | 1.071 | 0.999950 / 0.994385 |
| `expert_act_dtype=bfloat16` | 0.900 | 1.113 | 0.999949 / 0.994390 |
<!-- /generated:policy-sweep -->

### 4.3 The BFP4 attention/projection candidate (OPT-007)

OPT-007 makes a BFP4 trial on the attention projections mandatory, and it was run on real weights,
not synthetic ones. It is **faster**: 0.858 → 0.838 ms (`full_attention`) and 1.093 → 1.073 ms
(`linear_attention`) traced decode, prefill unchanged. It is also the only candidate in the sweep
that moves the accuracy needle, and the decision rests on the same HF-golden ladder the delivered
suite runs — ten prefill lengths from 1 to 3000 including the non-aligned ones, plus four decode
steps, both arms in one process:

<!-- generated:bfp4-pcc -->
| projection weight dtype | worst `full_attention` | worst `linear_attention` | margin above the 0.995 bar |
| --- | --- | --- | --- |
| **BFP8 (selected)** | 0.999883 (decode step=2) | 0.999920 (prefill seq_len=3000) | 4.9e-03 |
| BFP4 | 0.996507 (prefill seq_len=7) | 0.997422 (decode step=2) | 1.5e-03 |

Every BFP4 row clears the bar, so this is not a pass/fail rejection — it is a 30–43x increase in layer error for 2.3 % of one traced decode step and nothing in prefill, in one layer of a 48-layer stack. The routed-expert BFP4 step this stage *did* take is the opposite trade. Rejected on that comparison, and shipped as `POLICIES["bfp4-projections"]` so `$datatype-sweep` can take it without rediscovering it. Full ladder: [`logs/probe_projection_dtype.txt`](logs/probe_projection_dtype.txt).
<!-- /generated:bfp4-pcc -->



## 5. Performance

### 5.1 Method

Warmed measurements, always from the same harness. Prefill = one warmed 2048-token pass after two
warm-up passes. Decode = 32 warmed `execute_trace` replays, so **every decode number in this stage
is traced**; no eager decode number appears anywhere.

The before/after pair in the headline is
[`logs/bench.py`](logs/bench.py) run twice — once with `--impl fused`, once with
`--impl optimized` — on the same device, with the same real checkpoint weights, the same inputs and
the same warmup/iteration counts, back to back
([`logs/ab_fused_vs_optimized.txt`](logs/ab_fused_vs_optimized.txt)). Two processes, not one: `bench.py`
takes a single `--impl`. The genuinely **same-process** comparison is
`test_optimized_beats_fused_traced_decode`, which builds both decoders in one device session and
gates the claim; its numbers (2.064 → 1.068 and 1.830 → 0.858 ms in the suite log) agree with the A/B
file, which is why both are quoted.

`tt-perf-report` device tables come from four separate Tracy captures — one per (layer kind × phase)
— driven by [`tracy/run_profiling.sh`](tracy/run_profiling.sh), with **advice enabled** in the
committed table.

### 5.2 Result

<!-- generated:perf-result -->
| Layer kind | Phase | before | after | speedup |
| --- | --- | --- | --- | --- |
| `linear_attention` | prefill, 2048 tokens | 257.880 ms | **102.120 ms** | **2.53x** (−60.4 %) |
| `linear_attention` | decode, traced (32 replays) | 2.063 ms | **1.070 ms** | **1.93x** (−48.1 %) |
| `full_attention` | prefill, 2048 tokens | 243.560 ms | **96.350 ms** | **2.53x** (−60.4 %) |
| `full_attention` | decode, traced (32 replays) | 1.830 ms | **0.857 ms** | **2.14x** (−53.2 %) |
<!-- /generated:perf-result -->

### 5.3 Where the decode window goes

<!-- generated:decode-breakdown -->
| Op code | `linear_attention` µs/step | `full_attention` µs/step | launches/step (`full`) |
| --- | --- | --- | --- |
| `SparseMatmulDeviceOperation` | 265.8 | 267.7 | 2 |
| `MatmulDeviceOperation` | 154.7 | 98.8 | 5 |
| `BinaryNgDeviceOperation` | 148.2 | 99.8 | 8 |
| `UnaryDeviceOperation` | 87.3 | 75.4 | 4 |
| `SliceDeviceOperation` | 48.5 | 45.6 | 15 |
| `TopKDeviceOperation` | 48.4 | 48.4 | 1 |
| `UntilizeWithUnpaddingDeviceOperation` | 26.3 | 23.3 | 3 |
| `ReshapeViewDeviceOperation` | 26.2 | 3.7 | 1 |
| `LayerNormDeviceOperation` | 23.3 | 19.5 | 4 |
| `DeepseekMoEFastReduceNCDeviceOperation` | 22.3 | 22.2 | 1 |
| `FillPadDeviceOperation` | 9.8 | 19.6 | 4 |
| `SdpaDecodeDeviceOperation` | — | 17.5 | 1 |
| `TernaryDeviceOperation` | 17.0 | — | 0 |
| `PermuteDeviceOperation` | 11.3 | 11.5 | 1 |
| `CopyDeviceOperation` | 11.3 | — | 0 |
| `TypecastDeviceOperation` | 10.1 | 3.5 | 2 |
| **total device time** | **952.0** | **835.2** | |
<!-- /generated:decode-breakdown -->

### 5.4 Dominant matmul search tables

Every dominant matmul role, the candidates measured for it, and the kept/rejected decision. The
`in0_block_w`, core count, `per_core_N` and output block/subblock of the shipped row are asserted in
`test_decode_runs_the_tuned_program_configs`, which prints them into the suite log.

**Routed-expert `sparse_matmul`** — the largest item in both windows. Swept over core count (via
`per_core_N`), grid shape, the whole `in0_block_w` divisor ladder of `Kt`, output block and subblock
width, and output placement, under the selected BFP4/LoFi policy, at four active-expert counts
([`logs/probe_sparse_matmul.txt`](logs/probe_sparse_matmul.txt)):

| active experts | role | fused geometry | best measured | shipped |
| --- | --- | --- | --- | --- |
| 8 (batch-1 decode) | gate/up | 32 cores, `per_core_N` 1, 1×1 block, `in0_block_w` 16 — **255.3 µs** | **8 cores (1×8), `in0_block_w` 32, `per_core_N` 4, `out_block_w` 4, `sub_w` 4, L1 — 153.4 µs** | 8 cores, `in0_block_w` 64 (2 % of 32, inside the spread) |
| 8 | down | 64 cores, `per_core_N` 1, `in0_block_w` 8 — **333.1 µs** | **8 cores (1×8), `in0_block_w` 16, `per_core_N` 8, `out_block_w` 8, `sub_w` 8, L1 — 152.8 µs** | as measured |
| 32 | gate/up | — | 16 cores (8×2), `in0_block_w` 64, `per_core_N` 2 — 288.7 µs | as measured |
| 32 | down | — | 8 cores, `per_core_N` 8 — 234.0 µs | as measured |
| 64 | gate/up | — | 32 cores (4×8), `per_core_N` 1 — 387.3 µs | as measured |
| 64 | down | — | 16 cores (8×2), `per_core_N` 4 — 277.5 µs | as measured |
| ~162 (a 32-token prefill group) | gate/up | — | 32 cores (4×8), `in0_block_w` 64 — 576.3 µs | as measured |
| ~162 | down | — | 32 cores (8×4), `in0_block_w` 16, `per_core_N` 2 — 345.3 µs | as measured |

The shipped rule reproduces the measured winner at all four points from a bound the layer knows
exactly: `cores = clamp(active_bound / k, 8, 32)`, `k` = 2 for gate/up and 4 for down, with
`active_bound = min(num_experts, real_rows * num_experts_per_tok)`. **No shipped `in0_block_w` is
below 16**: gate/up runs 64 (the whole tiled K) and down runs 16 (likewise).

**Dense decode projections** — three families per role, under each role's own weight dtype and math
fidelity ([`logs/probe_dense_matmul.txt`](logs/probe_dense_matmul.txt)):

| role | shape M×K×N | ttnn heuristic | best DRAM-sharded | **shipped 1D `mcast_in0`** |
| --- | --- | --- | --- | --- |
| `attn_in` | 32×2048×9216 | 58.3 µs | 71.4 | **55.9** — 99 cores (11×9), `in0_block_w` 8, `per_core_N` 3, L1 out |
| `o_proj` | 32×4096×2048 | 75.5 | 34.2 | **26.6** — 22 cores (11×2), `in0_block_w` 16, `per_core_N` 3 |
| `gdn_in` | 32×2048×12352 | 76.1 | 94.5 | **73.5** — 110 cores (11×10), `in0_block_w` 8, `per_core_N` 4 |
| `gdn_out` | 32×4096×2048 | 75.6 | 34.3 | **shipped: 33 cores (11×3), `in0_block_w` 8, `per_core_N` 2** — 26.4 µs in the probe, 25 µs in the layer |
| `shared_in` | 32×2048×1056 | 32.3 | 14.4 | **9.4** — 88 cores (11×8), `in0_block_w` 32, `per_core_N` 1 |
| `shared_down` | 32×512×2048 | 16.2 | 14.2 | **9.3** — 55 cores (11×5), `in0_block_w` 16, `per_core_N` 2 |
| `router` (BF16/HiFi4/fp32 acc) | 32×2048×256 | 27.4 | 13.2 | **9.2** — 33 cores (11×3), `in0_block_w` 32, `per_core_N` 1 |

The core counts above are the **program grid** the config names, not the number of cores that end up
with work: `shared_in` names 88 and 33 of them get an output tile, `router` names 33 and 8 do, and
`shared_down` names 55 and 32 do. That is the shape the sweep chose — every smaller grid was measured
too (`probe_dense_matmul.txt` covers 8 → 110 target cores against the whole `in0_block_w` ladder) and
lost — but it is why those three rows still read `SLOW` at 25–55 % of DRAM bandwidth.

`gdn_out` is the one role whose probe rows were measured with a **bfloat16** in0 while the layer's
own activation was float32 — `chunk_gated_delta_rule` returns float32, so the output gate produced a
float32 activation and the projection ran `HiFi2 FP32 x BFP8 => FP32` at 29 µs decode / 222 µs
prefill, against `o_proj`'s 25 / 169 at the identical shape. The fix is to name `bfloat16` on the
gate's multiply, which costs no extra op and makes the row the same `BF16 x BFP8` one the probe
measured: `linear_attention` decode 1.075 → 1.071 ms, prefill 102.42 → 102.30
([`logs/ab_gdn_out_activation.txt`](logs/ab_gdn_out_activation.txt)). Review round 1 found the
dtype mismatch between the probe and the layer.

Two rows keep `out_subblock 1×1` — `shared_in` and `router`, both at `per_core_N` 1. That is
measured, not overlooked: dropping to a core count that gives `per_core_N` ≥ 2 is **slower**
(`shared_in` 11.5 µs at 16 cores / `per_core_N` 2 against 9.4 at 88 cores / `per_core_N` 1), because
these two are bandwidth-bound at 32 rows of M, not block-bound.

**Dense prefill projections** — ttnn's heuristic already picks the 2D family and fills the grid, but
leaves `in0_block_w` at 1, which `tt-perf-report` flags on every dense prefill row. An explicit
`MatmulMultiCoreReuseMultiCastProgramConfig` with the largest inner block the `in1` buffer holds
wins on all of them ([`logs/probe_prefill_matmul.txt`](logs/probe_prefill_matmul.txt)):

| role | heuristic | **shipped 2D config** |
| --- | --- | --- |
| `attn_in` 2048×2048×9216 | 528.5 µs | **471.9** (11×10, `in0_block_w` 8) |
| `o_proj` 2048×4096×2048 | 260.3 | **178.8** (11×10, `in0_block_w` 16) |
| `gdn_in` 2048×2048×12352 | 695.3 | **607.0** (11×10, `in0_block_w` 8) |
| `shared_in` 2048×2048×1056 | 111.9 | **64.9** (11×10, `in0_block_w` 16) |
| `shared_down` 2048×512×2048 | 73.4 | **52.6** (11×10, `in0_block_w` 16) |

`in0_block_w` 16 fails to build for the two wide-output roles ("statically allocated circular
buffers … clash with L1"), which is what `PREFILL_MATMUL_IN1_TILE_BUDGET` encodes. The whole dense
group is 1.07 % of the prefill window — the window is 81.65 % routed-expert `sparse_matmul` — so
this is worth about 0.4 % of prefill end to end, which is what it measures. At a large prefill batch
the fixed output block exceeds L1 and the layer hands those shapes back to ttnn's heuristic; that is
an explicit size check, not a silent fallback.

### 5.5 `tt-perf-report` advice, item by item

Advice is enabled in the committed tables. Every distinct item in **both** decode reports, with the
per-step occurrence count in each and the op rows it is raised on. The counts and the rows are read
out of `tracy/<kind>/decode_perf_report.csv` by [`logs/make_readme.py`](logs/make_readme.py), so an
item that appears on a new row shows up here without anyone editing the table, an item with no
recorded action renders as **unclassified**, and an item this stage *closed* moves to the second
table automatically. Only the action text is written by hand. (Round 2 of the stage review found the
hand-written version of this table under-counting one row and mis-attributing another, which is why
it is generated now — and the row it under-counted, `place input 0 in L1` on the shared expert's down
projection, is one of the two the stage then closed.)

<!-- generated:advice -->
| Advice | `linear` decode | `full` decode | `linear` prefill | `full` prefill | Rows it is raised on | Action |
| --- | --- | --- | --- | --- | --- | --- |
| *nnz=std::nullopt* | 2.00 | 2.00 | 128 | 128 | `SparseMatmulDeviceOperation active=?/256 x 3 (de)`, `SparseMatmulDeviceOperation active=?/256 x 3 (pr)` | **Reporting limitation, not advice.** `nnz` is inferred at runtime because pinning it wedged the device (§9 item 3); the report cannot model DRAM/FLOP utilisation for those rows, so this stage measures their share of device time instead. |
| *use HiFi4 with BF16 activations* | 4.00 | 4.00 | 3 | 3 | `MatmulDeviceOperation 2048 x 2048 x 1056 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 12352 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 9216 (pr)` … | **Rejected with measurement** — the reverse direction of §4.2's fidelity sweep; HiFi4 is what the fused decoder had and it is slower at equal correctness. |
| *look good* | 3.09 | 1.00 | 3 | 3 | `MatmulDeviceOperation 2048 x 2048 x 1056 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 12352 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 9216 (pr)` … | **Not advice** — `tt-perf-report` printing that a row's `in0_block_w` and output subblock are already what it would have suggested. Kept in the table so the generator cannot silently drop a line it does not recognise. |
| *DRAM-sharded program config* | 1.91 | 2.00 | 0 | 0 | `MatmulDeviceOperation 32 x 2048 x 12352 (de)`, `MatmulDeviceOperation 32 x 2048 x 9216 (de)`, `MatmulDeviceOperation 32 x 4096 x 2048 (de)` | **Tried, rejected with measurement** — loses on all seven dense roles, even without its activation-reshard cost (§5.4). |
| *HiFi2 is sufficient* | 2.00 | 0.00 | 0 | 0 | `MatmulDeviceOperation b={32} x 32 x 128 x 12 (de)` | **Tried, rejected with measurement** on the recurrent-state rows: HiFi2 is 14.6 µs against 14.9 for the shipped HiFi4 + fp32-accumulate and LoFi is 14.7 — ≤0.3 µs per matmul, under 0.1 % of the window, for the float32 state that is the model's exact carry between steps. |
| *Output subblock 1x1 is small* | 2.00 | 2.00 | 1 | 1 | `MatmulDeviceOperation 2048 x 2048 x 256 (pr)`, `MatmulDeviceOperation 32 x 2048 x 1056 (de)`, `MatmulDeviceOperation 32 x 2048 x 256 (de)` | **Tried, rejected with measurement** — the `per_core_N` ≥ 2 alternative is slower for both rows (`shared_in` 11.5 vs 9.4 µs, §5.4). |
| *HiFi2 may also work* | 1.00 | 1.00 | 1 | 1 | `MatmulDeviceOperation 2048 x 2048 x 256 (pr)`, `MatmulDeviceOperation 32 x 2048 x 256 (de)` | **Rejected on purpose** (the router row): the matmul is 9 µs and its output decides *which experts run*. The fused stage measured bfloat16 routing agreeing with float32 on only 99.8 % / 95.5 % of top-8 sets, so this group stays BF16/HiFi4/fp32-accumulate. |
| *place input 0 in L1* | 0.00 | 0.00 | 3 | 3 | `MatmulDeviceOperation 2048 x 2048 x 1056 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 12352 (pr)`, `MatmulDeviceOperation 2048 x 2048 x 256 (pr)` … | **Taken for decode, measured and rejected for prefill.** Decode: the two residual norms, the three float32 recurrent-state matmuls (worth 19 µs/step) and the shared expert's SwiGLU product all hand their result to L1, each size-gated so a large batch still uses DRAM — the item is now raised 0 times in both decode reports. The head-dim norms are the one decode exception and cannot move: `paged_scaled_dot_product_attention_decode` rejects a non-sharded Q outside DRAM. Prefill: still raised on three rows, and measured — an L1 `in0` is *slower* on the two that matter (`attn_in` 471.7 vs 459.5 µs, `gdn_in` 654.7 vs 647.0) and worth ~3 µs on `shared_in`, i.e. 0.006 % of a 96 ms prefill window. `probe_prefill_matmul.txt` `in0=DRAM`/`in0=L1` rows. |

Advice items **no longer raised in any of the four committed reports**:

| Advice (no longer raised) | What closed it |
| --- | --- |
| *in0_block_w=1 is small* | **Taken.** The five dense *prefill* rows got explicit 2D configs with `in0_block_w` 8/16 (§5.4), and the three recurrent-state rows got an explicit `MatmulMultiCoreReuseProgramConfig` — that family does expose `in0_block_w`, unlike the `core_grid` spelling the fused stage used: 2 for the two reads (13.9 vs 14.7 µs) and 1 for the `transpose_a` outer product, where `Kt` is 1 tile so 2 and 4 are rejected by the op (12.6 vs 20.4 µs). `probe_decode_micro.txt` `STATE progcfg` rows. |
<!-- /generated:advice -->

## 6. Runtime audit

`test_no_host_fallback_in_forward` runs one full prefill pass and one full decode pass per layer
kind under two simultaneous guards — raising stubs on
`ttnn.from_torch`/`to_torch`/`as_tensor`/`to_device`/`from_device`, and a
`torch.overrides.TorchFunctionMode` that raises on **any** dispatched `torch` operation — after
first asserting that both guards fire. Both passes complete for both kinds, so there is no host
round trip anywhere inside them. The program configs this stage adds are built from Python integers
and cached, so they cost nothing at replay and nothing on the host after the first call.

`test_no_layout_churn_in_measured_forward` pins how many layout/relayout ops the measured passes may
dispatch, **exactly** rather than as an upper bound, and the test itemises every term:

| Layer kind | prefill @256 | prefill @2048 | decode |
| --- | --- | --- | --- |
| `linear_attention` | 12 | 12 | 5 |
| `full_attention` | 3 | 3 | 14 |

Prefill is unchanged from the fused decoder. Decode rises from 1 and 6 by exactly the sharded-norm
boundary: one `to_memory_config` in and one `sharded_to_interleaved` out per width-sharded norm —
two norms on `linear_attention`, four on `full_attention` (it also norms the Q and K head dims).
That pair costs ~3 µs and saves ~9 per residual norm — 22.4 µs interleaved against 13.7 width-sharded
on 8 cores, from the committed `probe_decode_micro.txt` — and nothing in either budget scales with the
sequence length, which is the property the two prefill columns test.

The model itself dispatches **no** explicit `tilize`, `untilize`, `reshard`, `to_torch` or
`from_torch` in either measured path — the `tilize`/`untilize` entry points are watched by the test
above and are 0 in all four cases. What does appear in the device reports is the lowering of three
*composite* ops that have no tile-native form at these shapes, and they are itemised rather than
denied: `ttnn.scatter` in the router (untilize ×2 → scatter → tilize, ~34 µs/step),
`ttnn.repeat_interleave` for the GQA head expansion (untilize → concat → tilize, ~24 µs/step on
`linear_attention`, plus the 22.7 µs stall §7 attributes), and the `topk` index readback. §5.3 counts
them and §7 names what removing them would take.

`test_optimized_path_is_used` additionally asserts the dedicated fused ops are still dispatched and
that the *functional* decoder dispatches none of them beyond the three the implementations share, so
the assertions cannot pass vacuously.



## 7. Performance accounting

<!-- generated:accounting -->
| | `linear_attention` | `full_attention` |
| --- | --- | --- |
| bytes moved per token (weights at their stored dtypes + KV read) | 64,892,928 B | 47,722,496 B |
| DRAM peak used for the roofline (recovered from the report's own DRAM %) | 512.0 GB/s | 512.0 GB/s |
| 1. theoretical roofline | 0.127 ms | 0.093 ms |
| 2. device-time decode (signposted window / 32 replays) | 0.952 ms | 0.835 ms |
| 3. end-to-end decode, **profiled** run | 1.099 ms | 0.917 ms |
| roofline as a fraction of device time | 13.3% | 11.2% |
| dispatch + host gap (3 − 2) | 0.147 ms | 0.082 ms |

Named limitations, in the order they cost time:

* ttnn.sparse_matmul parallelism is capped by the output tile count (Nt) and it loops once per active expert at a single tile of M, so the two routed projections reach roughly 5 % of the FLOP roofline and ~40 GB/s of weight bandwidth even after the geometry sweep; they are 31 % of the window.
* The routed-expert intermediates are num_experts wide where only num_experts_per_tok slots are non-zero, so the zero-fill, the two unpacking slices, the SwiGLU, the score multiply and the expert reduction each touch 32x the useful width. Moving them to L1 removed the DRAM cost; the remaining ~230 us/step is L1 and launch bound and needs an expert-major gather to remove, which is a routing-algorithm change rather than an op-config one.
* ttnn.topk is single-core on the 256-wide routing dim (48 us/step); its multi-core path needs a power-of-two width >= 8192 and padding to it measured ~3x slower.
* About 90 device ops per decode step at a ~1 us op-to-op gap accounts for most of the difference between device time and end-to-end.
<!-- /generated:accounting -->

Row 3 is the **profiled** run, which is what makes it comparable to row 2 — both come from the same
Tracy capture. Against the un-profiled end-to-end of §5.2 the two kinds differ and are stated
separately rather than summarised together:

* `full_attention`: device time is within ~20 µs of the un-profiled end-to-end of §5.2, i.e.
  device-bound. There is no untraced path, per-step synchronization or host readback left to remove.
* `linear_attention`: the same comparison leaves ~10 % — and that gap is **itemised, not waved
  away**. `22.7 µs of it is a single op-to-op stall`, once per replay, in front of the tilize inside
  `repeat_interleave`'s untilize→concat→tilize lowering for the GQA head expansion (op 22 of 97 in
  `tracy/linear_attention/decode_perf_report.csv`). Placing that op's output in L1 recovered ~3 µs;
  the remainder is dispatch *inside* the captured trace, which the layer cannot reach — the only way
  to remove it is to remove the op, and the alternatives (a rank-5 concat on the head axis, or
  carrying 16 KV heads into a 32-head delta rule) are not expressible at these shapes. Two 6–8 µs
  gaps in front of the float32 gate-promotion typecasts and ~118 gaps at ~0.5 µs each make up the
  rest.

The roofline fraction is ~11 %, and the reason is structural rather than a missed knob: a decode
step reads 8 of 256 experts, so its *useful* traffic is small, while the op contract makes it touch
a `num_experts`-wide intermediate five times and loop the sparse matmul once per active expert at a
single tile of M. The four limitations above are the whole gap, and each is named with what it would
take to remove — none of which is an op-config change this stage could make.



## 8. Watcher

A separate run, never combined with the profiler, over the state-, trace- and optimization-critical
test subset. Exact command, subset rationale and classification:
[`watcher/CLASSIFICATION.md`](watcher/CLASSIFICATION.md).

<!-- generated:watcher-result -->
**59 passed** in 328.13 s. The 65,509 lines of the watcher log are fully accounted for by a disjoint census, and a fatal-class grep (asserts, invalid NOC coordinates or addresses, CB out-of-bounds, L1/stack overflow, sanitizer, corruption, hang/deadlock) returns **0 matches**. Watcher recorded a stack watermark on 5 dump(s); the tightest leaves 1332 bytes free.

Artifacts: [`watcher/watcher_log.txt`](watcher/watcher_log.txt), [`watcher/census_summary.txt`](watcher/census_summary.txt), console log [`logs/watcher_pytest.txt`](logs/watcher_pytest.txt).
<!-- /generated:watcher-result -->



## 9. Known limitations

1. **`ttnn.sparse_matmul` materialises a `num_experts`-wide output.** Eight of 256 expert slots are
   non-zero at batch-1 decode, but the op's zero-fill, the two unpacking slices, the SwiGLU, the
   score multiply and the expert reduction all run at the full 256-expert width. Moving them to L1
   removed their DRAM cost and the geometry sweep and BFP8 activations shrank what is left, but they
   remain the second-largest item in the window. Removing them needs **expert-major token
   gathering**, which is a routing-algorithm change rather than an op-config one.
   `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` is the in-tree op that wants that
   layout and it was assessed: it is a *prefill* op that launches one device program per local
   expert (256 per call here, against this decoder's 2), it needs a dispatched/sorted token buffer
   with per-expert counts and offsets that this decoder does not build, and its own documented
   accuracy target is **PCC ≥ 0.97** against the PyTorch reference — below this stage's inherited
   0.995 layer bar. Rejected on that combination, recorded rather than deferred silently.
2. **`ttnn.topk` is single-core on the 256-wide routing dim**, 48 µs/step. Its multi-core path needs
   a power-of-two reduced width of at least 8192; padding the logits with `-inf` up to 8192 is
   *correct* (the returned indices stay in range) but measured ~3× slower than the single-core
   256-wide call, and every intermediate width is worse still. Measured ladder in
   [`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt).
3. **`nnz` is left inferred at runtime** for both routed sparse matmuls, and the reason is a
   reproduced hang, not an inherited argument. A static `nnz` selects a faster fixed-count path worth
   13–16 % elsewhere, and after the padded-row masking the batch-1 decode mask holds exactly
   `num_experts_per_tok` experts. It was measured: `probe_sparse_matmul.py --nnz --active 8` passes
   `nnz=8` against a sparsity tensor with **exactly 8** non-zeros, and the first candidate **wedged
   the device inside `SparseMatmulDeviceOperation`** — despite the factory's comment that the sender
   validates the count on device and "fail[s] loudly instead of deadlocking"
   ([tt-metal #45943](https://github.com/tenstorrent/tt-metal/issues/45943)). `tt-triage` output is
   committed at [`triage/`](triage/); one `tt-smi -r` recovered the card and the mesh smoke passed.
   `work_log.md` §4.8 has the full incident and recovery table. This is a ttnn limitation worth
   reporting upstream: the documented on-device validation does not fire at these shapes.
4. **DRAM-sharded decode matmuls lose here**, on every dense role, even measured without their
   activation-reshard cost, because the op pins its compute grid to the 8 DRAM banks. Table in §5.
5. **Three batch thresholds, all documented rather than discovered.** The dense decode program
   configs apply up to 8 tile rows of activation (batch ≤ 8 at one tile per batch entry); the sharded
   decode norms up to 4; and the recurrent-state program configs up to **batch 3**, because the
   batched non-mcast op parallelises over `batch * num_value_heads` blocks and 32 of those per batch
   row fill an 11×10 grid at 3. Above each threshold the layer falls back to the shape the fused
   decoder used — correct but untuned. Batch-1 single-user latency is this stage's target; larger
   batches are covered for *correctness* up to 56, not for latency. The MoE's L1 placement has a
   fourth, orthogonal bound (`EXPERT_L1_MAX_CALL_TOKENS`, one prefill chunk of tokens per MoE call),
   which the shared expert now shares.
6. **Prefill is not traced**, as in the fused and functional stages. Only decode is captured and
   replayed.
7. **The `linear_attention` prefill still selects its depthwise-conv path by probing at
   `allocate_state`**, inherited unchanged from the fused decoder together with its coverage table
   and its documented risk that a `(batch, length)` which passes the probe can still fail inside a
   forward. This stage did not touch the conv path.

---

## 10. `$optimize` checklist

Every item of the skill's final checklist, with where its evidence is. Items that cannot apply to a
single-device decoder layer say so rather than being ticked.

| Item | Status | Evidence |
| --- | --- | --- |
| MoE routed active-expert `nnz` handling | ✅ tried, **wedged the device**, rejected | §9 item 3, `work_log.md` §4.8, `triage/` |
| Decoder path fully traced with no host fallbacks | ✅ | `test_no_host_fallback_in_forward`, `test_lazy_allocation_is_the_only_host_call`, §6 |
| Decode activations width-sharded in L1 across norm / attention / residual / MLP / output boundaries | ✅ partially, measured | The **norms** are width-sharded in L1 (§3, `probe_decode_micro.txt` `NORM` rows). Carrying the shard *through* the projections needs the DRAM-sharded matmul family — it is the only one that consumes a width-sharded `in0`; `mcast_in0` requires interleaved — and that family loses on all seven roles even measured without its reshard cost (§5.4). So the shipped contract shards the norms and interleaves at the matmul boundary, which is the measured winner, not the convenient one. |
| Prefill activations DRAM interleaved; 2D program configs for large prefill matmuls | ✅ | §5.4 dense-prefill table; `probe_prefill_matmul.txt` |
| Operation-topology audit recorded | ✅ | `work_log.md` §2 — op sequence, repeated same-input matmuls, reshard/layout conversions, candidate replacements, dtype constraints, action taken |
| Multi-device topology families measured as coherent families | n/a | Single-device 1×1 mesh; no collective in the measured path. Multi-device topology is the next stage's. |
| Lower-movement residual candidates measured without an old-contract restore | n/a | Same reason — there is no collective residual contract here. The single-device equivalent (sharded residual carry) is the row above. |
| Best-candidate comparison completed | ✅ | Every change is measured against the strongest prior candidate in `work_log.md` §3, and the final default is re-measured end to end in `logs/ab_fused_vs_optimized.txt` |
| Final default reproduced the selected best candidate | ✅ | §5.2 is the final default run, not a candidate copied forward |
| Final dtype/fidelity policy verified in the measured runtime rows | ✅ | §5.3 rows show `LoFi BF16 x BFP4 => BFP8` on both routed matmuls and `HiFi2 BF16 x BFP8 => BF16` on the dense ones; `test_precision_policy_reaches_the_device_tensors` checks the tensors |
| SDPA / optimized composite ops used | ✅ | `paged_scaled_dot_product_attention_decode` + `chunked_scaled_dot_product_attention`, with an explicit program config swept against the default and three alternatives (`work_log.md` §3.8) |
| Fused/packed repeated same-input projections | ✅ | All four packings inherited from the fused stage and re-validated under BFP4/LoFi: packed gate/up beats the separate pair 212.1 vs 288.3 µs (OPT-010, `probe_decode_micro.txt` `SPLIT` rows) |
| Explicit `memory_config`, `program_config`, `compute_kernel_config` for important ops | ✅ | §5.4, §4.1; `test_decode_runs_the_tuned_program_configs` asserts the decode ones are live |
| Program-config sweep for every dominant matmul role | ✅ | §5.4 — three families × core counts × the full `in0_block_w` divisor ladder × output block/subblock × output placement, per role, for **both phases**. The float32 recurrent-state matmuls take no `in0_block_w` at the API (rank-4 batched `ttnn.matmul` with an explicit `core_grid`); their grid and fidelity are swept in `probe_decode_micro.txt`. |
| Decode compute fidelity swept as a performance knob | ✅ | §4.2 — LoFi vs HiFi2 vs HiFi4 measured separately for the expert, projection and shared groups, and for the float32 recurrent-state group in `probe_decode_micro.txt` (`STATE` rows) |
| Attention projection dtype/fidelity swept separately from MLP | ✅ | §4.3 — the BFP4 attention/projection candidate on a real-weight HF-golden ladder, with the decision and the rejected policy shipped as `POLICIES["bfp4-projections"]` |
| BFP4/LoFi trials for the dominant MLP/expert matmuls | ✅ | Selected: BFP4 + LoFi for both gate/up and down. FF2/down BFP8 measured and rejected as slower *and* larger (§4.2) |
| Shard specs / core grids divide tensor dims cleanly | ✅ | `_sparse_matmul_config` reduces the target core count to a divisor of `Nt`; `_decode_1d_matmul_config` and `_norm_shard` build exact rectangles |
| DRAM-sharded decode matmuls | ✅ tried, rejected with evidence | §5.4, `probe_dense_matmul.txt` — loses on all seven roles |
| Collective topology minimized | n/a | No collectives on one device |
| Fused matmul-CCL ops | n/a | Same |
| Persistent/preallocated CCL buffers | n/a | Same |
| MoE routed active-expert path with `ttnn.sparse_matmul` | ✅ | Kept and tuned; no dense all-expert path exists. `nnz` deliberately inferred (§9 item 3). Expert intermediates in L1 rather than round-tripping DRAM (§3) |
| LM head / sampling terminal path | n/a | A decoder layer has neither; owned by the full-model stage |
| Reduced precision/fidelity experiments on real weights and activations | ✅ | §4.2, §4.3 — every candidate on the real checkpoint; no synthetic-weight result decides any policy |
| Performance accounting reconciled | ✅ | §7 — roofline, device time and end-to-end from the same run, stated **per layer kind**, with the one material gap attributed to a specific op |
| Batch capability preserved | ✅ | Batch 1 is the tuned target; correctness tested at 4, 32, and past both op fallbacks at 40 and 56 |
