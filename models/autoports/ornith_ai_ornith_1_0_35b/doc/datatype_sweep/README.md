# Ornith-1.0-35B — datatype sweep (TTNN, 4-chip Blackhole ring)

A precision sweep over the [optimized full model](../optimized_full_model/), on the same mesh: four
Blackhole `p300c` chips as a `1x4` ring under `FabricConfig.FABRIC_1D_RING`, TP=4 dense and EP=4 over
the 256 routed experts. **24 full-model configurations were evaluated**, each with the whole 40-layer
stack, real checkpoint weights, the AIME24 chat-template readiness reference and 100 generated tokens.

---

## 1. The selected config

[`selected_precision_config.json`](selected_precision_config.json) — **`C06-proj-bfp4-lofi`**. It is
the model's **default**: `OrnithModel.from_pretrained` resolves `policy=None` through
[`tt/precision_config.py`](../../tt/precision_config.py), which loads that file. There is no second
copy of the selected values in code, and no call site has to ask for them.

| group | weights | math fidelity | changed by this stage |
|---|---|---|---|
| routed expert gate/up | `bfloat4_b` | LoFi | no — the decoder stage's |
| routed expert down | `bfloat4_b` | LoFi | no |
| **dense token-mixer projections** (packed attention in-projection, `o_proj`, packed DeltaNet in-projection, `out_proj`) | **`bfloat8_b` → `bfloat4_b`** | **HiFi2 → LoFi** | **yes** |
| **LM head** | **`bfloat8_b` → `bfloat4_b`** | **HiFi2 → LoFi** | **yes** |
| shared expert (packed gate/up/router + down) | `bfloat8_b` | HiFi2 | no |
| router (256-way) | `bfloat16` | HiFi4, fp32 accumulate | no |
| DeltaNet recurrent state | float32 (not a knob) | HiFi4, fp32 accumulate | no |
| SDPA | — | HiFi2, fp32 accumulate | no |
| routed-expert output activation | `bfloat8_b` | — | no |
| **residual / inter-layer activation stream** | `bfloat16` | — | no (measured, rejected) |
| **CCL payload** | as produced — no cast | — | no (measured, rejected) |
| **paged KV cache** | `bfloat8_b`, 64-token blocks | — | no (measured, rejected both ways) |
| **logits / sampling input** | `bfloat16` | — | no (measured, rejected) |
| layer exceptions | none | | measured as `C17`, rejected: slower at equal accuracy |

The two rows in bold are the whole change: the dense projections and the LM head move from
`bfloat8_b`/HiFi2 to `bfloat4_b`/LoFi. Everything else the sweep touched was measured and kept where
it was, with the numbers in §4.

### Results, against the thresholds

| | selected | baseline (`optimized`) | threshold |
|---|---|---|---|
| `run_prefill_check` top-1 | **0.920** | 0.940 | **≥ 0.90** |
| `run_teacher_forcing` top-1 | **0.920** | 0.970 | **≥ 0.90** |
| top-5, both gates | **1.000** | 1.000 | **≥ 0.98** |
| top-100, both gates | **1.000** | 1.000 | **= 1.00** |
| trace-verified teacher-forcing decode | **42.296 t/s/u** (23.643 ms/token) | 41.962 t/s/u | selection metric |
| teacher-forcing TTFT | 178.9 ms | 181.2 ms | — |
| **post-selection token-out decode** | **43.169 t/s/u** (23.165 ms/token) | 42.916 t/s/u | serving headline |
| **post-selection warmed TTFT** | **139.5 ms median** (133.9 min) | 140.0 median | — |

Two performance regimes, deliberately kept apart and labelled:

* **trace-verified teacher-forcing decode** — the sweep's *selection* metric, and the only one the
  candidate matrix is ranked on. Batch 1, greedy, device sampling, the AIME24 prompt plus 100 forced
  tokens through `run_teacher_forcing`'s own per-entry function. The loop is serial **by
  construction** (the harness decides step *N+1*'s input on the host from token *N*), which is why it
  is slower than token-out. Best of **9 warm repeats** in one build; §3 has the estimator.
* **post-selection token-out decode** — the *serving* headline, and the number later reports and vLLM
  comparisons should use. The optimized full-model stage's own warmed no-readback benchmark
  ([`logs/bench_full_model.py`](../optimized_full_model/logs/bench_full_model.py)), prompt 128 /
  generate 128, batch 1, nine repeats, through the ordinary `build_generator` path with **no policy
  argument** — so it measures what a serving adapter gets by default.
  [`post_selection_token_out.json`](post_selection_token_out.json).

### The Pareto frontier

![top-1 against traced decode throughput](top1_perf_pareto.png)

![top-5 against traced decode throughput](top5_perf_pareto.png)

The x axis is the **binding** accuracy — `min(run_prefill_check, run_teacher_forcing)` — because the
gate is both checks, so a config's distance from the bar is set by its worse one. The frontier has
exactly **three** non-dominated points, and every one of them is a real choice:

| frontier point | top-1 | decode | what it buys |
|---|---|---|---|
| **C06 (selected)** | 0.920 | **42.296 t/s/u** | the fastest passing config |
| C23 `proj-bfp4-lofi-lmhead-bfp8-lofi` | 0.930 | 42.141 | +1 point of top-1 for −0.37 % |
| C04 `proj-bfp8-lofi` | **0.960** | 42.081 | +4 points of top-1 for −0.51 % |

`ORNITH_PRECISION_POLICY=doc/datatype_sweep/candidates/C23-proj-bfp4-lofi-lmhead-bfp8-lofi.json`
selects the middle point and `…/C04-proj-bfp8-lofi.json` the conservative one, with no code change.

The **top-5 chart is a single vertical stack at 1.000**: every one of the 24 configurations, including
the one that fails the top-1 gate, scores exactly 1.000 top-5 and 1.000 top-100 on both readiness
checks. Top-5 constrains nothing in this sweep; top-1 is the only accuracy dimension that separates
these configurations, and it is the one the frontier is drawn against.

---

## 2. The headline finding: this model's decode step is launch-bound, so precision buys ~1 %

The selected config reads **24 % fewer weight bytes per decode step per device** — 569,421,184 B
against 749,907,328 B, both summed from the live device tensors by
`bench_full_model.py::_performance_accounting` — and decode gets **0.59 % faster** (token-out) or
**0.80 %** (teacher forcing).

That is the sweep's real result, and it is consistent across all 24 configurations: the whole matrix
spans **41.037 → 42.296 t/s/u, a 3.1 % range**. Twenty-three of the twenty-four *pass* the accuracy
gate, so the passing set spans essentially the same **3.07 %** — the three regressions below are
accurate, they are just slow. Strip those three and the remaining twenty configurations span
**0.92 %**, which is the honest measure of what precision buys here. The optimized
full-model stage already named the cause and measured it — the step sits at 6.3 % of the DRAM roofline
because it is **~100 device ops per layer at one tile of M**, i.e. bound by op launch rather than by
bandwidth. Narrowing weights moves the denominator of a fraction that is not the constraint: the
achieved roofline fraction *falls* from 6.28 % to 4.80 % while wall-clock barely moves.

The corollary is visible three times in the matrix, and it is the reason three arms are **regressions**:
when a dtype change costs even one extra op per layer, it loses.

| arm | what it adds | decode |
|---|---|---|
| C13 `ccl-bfp8` | one `typecast` per collective, two per layer | **−1.33 %** |
| C14 `residual-bfp8` | one `typecast` per RMSNorm output, two per layer | **−1.87 %** |
| C10 `experts-bfp4-hifi2` | no extra op — just a slower fidelity on the dominant matmul | **−2.20 %** |

---

## 3. How a candidate was measured

One process per candidate, one device job at a time
([`logs/run_sweep.sh`](logs/run_sweep.sh) → [`logs/sweep_one.py`](logs/sweep_one.py)). Each process
opens the ring, builds the whole 40-layer model at that candidate's precision config, and runs:

1. **`run_prefill_check`** against [`readiness_aime24_chat.refpt`](../../readiness_aime24_chat.refpt) —
   the AIME24 chat-template reference the full-model stage generated with `--gen-len 100 --top-k 100`
   ([provenance](../../readiness_aime24_chat.meta.json)); 1 entry, 100 tokens, top-k 100.
2. **`run_teacher_forcing`** ten times, each with a fresh `TokenAccuracy`, so every repeat's accuracy
   is computed independently and they must agree (`teacher_accuracy_identical_across_repeats` is true
   for all 24 configs).

The driver calls the two official runners' **own** per-entry functions
(`_run_one_entry_prefill` / `_run_one_entry`) against one generator instead of letting each runner
build its own — a 40-layer build is ~200 s and the two checks are otherwise identical to running the
two CLIs back to back. That shortcut has a control: the baseline additionally ran the official
programmatic `run_prefill_check` and `run_teacher_forcing` entry points, each building its **own**
generator after the driver's was torn down
(`runs/S00-baseline-optimized.json::official_runner_control`, and the same block in the four-repeat
pass):

| | driver (shared generator) | official runners (own generator) |
|---|---|---|
| prefill top-1 / top-5 / top-100 | 0.940 / 1.000 / 1.000 | **0.940 / 1.000 / 1.000** |
| teacher top-1 / top-5 / top-100 | 0.970 / 1.000 / 1.000 | **0.970 / 1.000 / 1.000** |
| teacher decode | 41.819 – 41.962 (9 warm), best **41.962** | **41.737** — one sample |

**Accuracy is identical, which is what the shortcut could plausibly have broken.** Throughput is
0.535 % below the driver's best-of-nine, which is the order of the selected win, so the control
validates the accuracy shortcut exactly and the *absolute* throughput figure only to ~0.5 %. Two
things account for the offset and neither touches the ranking: a single sample cannot beat a best of
nine by construction (the driver's own nine warm repeats already span 0.34 %), and the official
runner's generator is a fresh build with a different allocator layout. **The ranking is unaffected
because every one of the 24 candidates went through the identical driver**, and it does not depend on
the estimator either — best, median, mean and *minimum* warm repeat all put the same four **passing**
configurations in the same order. (C18 is faster than C17 on three of the four estimators and is
absent here for the same reason it is absent everywhere else: it fails the accuracy gate.)

| estimator | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| best warm (reported) | C06 42.296 | C16 42.286 | C05 42.272 | C17 42.160 |
| median warm | C06 42.265 | C16 42.244 | C05 42.211 | C17 42.137 |
| mean warm | C06 42.252 | C16 42.239 | C05 42.211 | C17 42.127 |
| **worst** warm | C06 42.186 | C16 42.175 | C05 42.163 | C17 42.078 |

**The reported decode figure is the best *warm* repeat.** A repeat is warm only if no trace re-capture
landed inside its own timed window. The first repeat after a build normally is not: prefill compiles
that prompt length's programs while the traces are live, so `_ensure_traces_replay_safe` re-captures
once before the first replay, and that lands inside `run_teacher_forcing`'s decode window. It costs
~9 % — it is the entire difference between the optimized full-model stage's archived 38.18 t/s/u and
this stage's 41.96 for the same policy. Ranking on it would rank trace-capture cost. Every candidate
got 9 warm repeats out of 10. The warm spread is **0.18 – 0.48 %** across the 23 passing configs
(0.70 % for C18, the one that fails the accuracy gate), recorded per config in
[`sweep_results.csv`](sweep_results.csv).

**Trace verification.** `generate(enable_trace=True)` is the only path that reaches
`_decode_step_traced`; the eager path is a different function and never captures. Every repeat records
`trace_id_present`, `trace_recaptures_total`, `recaptures_inside_timed_window`, and the loop's own
counters (`decode_syncs == decode_calls == 99`, `pipelined_readback: false`, which is what teacher
forcing is *supposed* to be). No eager or untraced number appears anywhere in the ranking, the charts
or the selection.

Before any candidate cost a 4-minute full-model run it went through a **reduced two-layer smoke**
([`logs/smoke_policy.py`](logs/smoke_policy.py), results in [`logs/smoke/`](logs/smoke/)): one real
`linear_attention` layer, one real `full_attention` layer, real weights, real cache and page-table
shapes, a non-aligned 87-token prompt and 8 traced decode steps. `$datatype-sweep` asks for exactly
that, and it earned its keep twice — §5's two blocked arms both failed there.

---

## 4. The candidate matrix

All 24, ranked by the selection metric. Full rows — every dtype and fidelity field, both gates, all
warm repeats, command, branch, commit, hardware, mesh — are in [`sweep_results.csv`](sweep_results.csv)
and [`sweep_results.json`](sweep_results.json); the per-candidate configs are in
[`candidates/`](candidates/) and the raw run records in [`runs/`](runs/).

| config | change from the baseline | top-1 | top-5 | decode t/s/u | Δ | verdict |
|---|---|---|---|---|---|---|
| **C06** | **dense projections + LM head → BFP4, LoFi** | **0.920** | 1.000 | **42.296** | **+0.80 %** | **selected** |
| C16 | *identical policy to C06* (see below) | 0.920 | 1.000 | 42.286 | +0.77 % | reproducibility control |
| C05 | dense projections + LM head → BFP4, HiFi2 | 0.920 | 1.000 | 42.272 | +0.74 % | tie with C06; LoFi is canonical |
| C18 | C06 + BFP4 KV cache | 0.860 | 1.000 | 42.269 | +0.73 % | **fails the gate** |
| C17 | C06 with layers 0 and 39 kept at BFP8 | 0.920 | 1.000 | 42.160 | +0.47 % | slower at equal top-1 |
| C23 | dense projections BFP4/LoFi, LM head pinned BFP8/LoFi | 0.930 | 1.000 | 42.141 | +0.43 % | **frontier** |
| C22 | dense projections BFP4/LoFi, LM head pinned BFP8/HiFi2 | 0.930 | 1.000 | 42.136 | +0.42 % | tie with C23 |
| C24 | **the union of every other non-negative arm**: C06 + shared BFP4/LoFi + logits BFP8 + SDPA LoFi | 0.930 | 1.000 | 42.126 | +0.39 % | the skill's step-6 extension — slower than C06 |
| C02 | LM head → BFP4, LoFi | 0.920 | 1.000 | 42.119 | +0.37 % | subsumed by C06 |
| C01 | LM head → BFP4, HiFi2 | 0.920 | 1.000 | 42.116 | +0.37 % | subsumed by C06 |
| C04 | dense projections + LM head → LoFi, dtype unchanged | **0.960** | 1.000 | 42.081 | +0.28 % | **frontier** |
| C12 | KV cache → bfloat16 | 0.930 | 1.000 | 41.997 | +0.08 % | inside the spread |
| C15 | logits tensor → BFP8 | 0.940 | 1.000 | 41.991 | +0.07 % | inside the spread |
| C03 | LM head fidelity → LoFi, dtype unchanged | 0.950 | 1.000 | 41.988 | +0.06 % | inside the spread |
| C20 | SDPA → LoFi without fp32 accumulate | 0.950 | 1.000 | 41.980 | +0.04 % | inside the spread |
| C08 | shared expert → BFP4, LoFi | 0.950 | 1.000 | 41.974 | +0.03 % | inside the spread |
| C11 | KV cache → BFP4 | 0.940 | 1.000 | 41.962 | +0.00 % | inside the spread |
| **S00** | **the baseline: the decoder stage's `optimized` policy** | 0.940 | 1.000 | **41.962** | — | baseline |
| C09 | shared expert → BFP4, HiFi2 | 0.950 | 1.000 | 41.937 | −0.06 % | inside the spread |
| C07 | shared expert fidelity → LoFi, dtype unchanged | 0.920 | 1.000 | 41.926 | −0.08 % | inside the spread |
| C21 | router → BFP8 | 0.940 | 1.000 | 41.912 | −0.12 % | inside the spread |
| C13 | CCL payload → BFP8 | 0.940 | 1.000 | 41.405 | −1.33 % | **regression** |
| C14 | residual stream → BFP8 | 0.940 | 1.000 | 41.179 | −1.87 % | **regression** |
| C10 | routed experts BFP4 → **HiFi2** | 0.940 | 1.000 | 41.037 | −2.20 % | **regression** |

### 4.0 The surviving choices were extended, as the skill's step 6 asks

`$datatype-sweep`'s default search says to try extending the surviving choices once a passing config
is found. Three arms measured non-negative on their own without being selected — C08 (shared expert
BFP4/LoFi, +0.03 %), C15 (logits `bfloat8_b`, +0.07 %) and C20 (SDPA LoFi without fp32 accumulate,
+0.04 %) — so **C24** is C06 unioned with all three, the largest legal lower-precision configuration
this sweep can build short of the KV cache. It **passes** (0.940 / 0.930, top-5 and top-100 1.000)
and it is **slower**: 42.126 t/s/u against C06's 42.296, −0.40 %.

That is **not** §2's extra-dispatch mechanism: C24 adds no op at all — it narrows a weight group, a
logits tensor and a fidelity, all in place, and the `tracy` capture shows no extra typecast. The
honest reading is simpler. Each ingredient measured +0.03 %, +0.07 % and +0.04 %, every one of them
well inside the 0.3 % within-build warm spread, i.e. **unresolved**; three unresolved effects do not
compose into a resolved win, and here they compose into a small loss. The other extension — adding
the BFP4 KV cache — is C18, and it fails the accuracy gate (§4.3). Either way, the "fastest evaluated
config" claim is not an artifact of an unexplored union.

### 4.1 BFP4+LoFi coverage

`$datatype-sweep` requires a **BFP4+LoFi** arm for every material BFP4 matmul group considered or
selected, and a **BFP8+LoFi vs BFP8+HiFi2** comparison for the dominant decode projection groups. All
four matmul groups have both:

| group | BFP4+LoFi | BFP4+HiFi2 | BFP8+LoFi | BFP8+HiFi2 | winner |
|---|---|---|---|---|---|
| routed experts (the largest single item in decode) | **41.962** (S00, shipped) | 41.037 (C10) | — dtype is BFP4 | — | **BFP4+LoFi**, by 2.20 % |
| dense token-mixer projections | **42.296** (C06) | 42.272 (C05) | 42.081 (C04) | 41.962 (S00) | **BFP4+LoFi** |
| LM head | **42.119** (C02) | 42.116 (C01) | 41.988 (C03) | 41.962 (S00) | **BFP4+LoFi** |
| shared expert | 41.974 (C08) | 41.937 (C09) | 41.926 (C07) | 41.962 (S00) | all four inside the spread — kept at BFP8/HiFi2 |

C10 is the arm that earns the routed experts' LoFi: the decoder stage selected BFP4+LoFi there, and
this stage tested the *higher*-fidelity direction rather than inheriting the choice. HiFi2 on that
group is the single largest regression in the matrix.

### 4.2 C06 and C16 are the same policy, and that is the reproducibility control

`lm_head_dtype` and `lm_head_fidelity` default to `null`, which means "follow the dense projection
group" — the LM head is the model's one extra dense projection. So **C05, C06, C16, C17 and C18 all
move the LM head too**, and C16 — written as "C06 plus an explicitly pinned BFP4/LoFi head" — resolves
to a policy byte-identical to C06's. Their two independent 10-repeat measurements landed **0.023 %
apart** (42.296 vs 42.286), which is the sharpest statement this sweep can make about its own
build-to-build reproducibility — an order of magnitude tighter than the 0.26 % spread of the
individual warm repeats *within* either build, which is what a best-of-nine estimator is for.

C22 and C23 were then added to *separate* the two groups, and they are what make the frontier
honest: pinning the head back to BFP8 recovers one point of top-1 for 0.37 % of throughput. The
selected config's `selected_precision_config.json` states `lm_head` explicitly rather than leaving it
to the rule.

### 4.3 What the gate actually caught

**C18 is the only failing configuration**, and it is an interaction rather than a single bad choice.
A BFP4 KV cache **alone** (C11) passes comfortably at 0.940 / 0.950 — indistinguishable from the
baseline. Combined with BFP4 dense projections it collapses to **0.870 / 0.860**, ten points below the
baseline and three below the bar, while buying nothing (42.269, slower than C06). That is the whole
argument for keeping the cache at `bfloat8_b`: not that BFP4 caches are bad, but that this model has
no error budget left for one once the projections are BFP4.

### 4.4 Layer exceptions were measured, not assumed

`$datatype-sweep`'s default search excludes the first and last layer from BFP4. C17 is that config —
`layer_exceptions = ((0, proj_dtype, bfloat8_b), (39, proj_dtype, bfloat8_b))`, resolved per layer by
`PrecisionPolicy.for_layer` — and it is **not** selected: it buys nothing on accuracy (0.920, the same
as C06) and costs 0.32 % of throughput. The exception machinery ships anyway, is exercised by
`tests/test_precision_config.py::test_layer_exceptions_resolve_per_layer_and_leave_other_layers_identical`,
and is the first lever to reach for if a later reference makes the bar tighter.

---

## 5. Rejected and blocked, with the evidence

| candidate | measured / observed | decision |
|---|---|---|
| **BFP4 KV cache** (C11 alone, C18 combined) | alone: 41.962 t/s/u, top-1 0.940/0.950 — a tie with the baseline on both. Combined with BFP4 projections: **top-1 0.870/0.860, below the 0.90 bar** | rejected on the gate (§4.3) |
| **bfloat16 KV cache** (C12) | 41.997 t/s/u (+0.08 %, inside its own 0.48 % spread), top-1 0.930; and it costs **1.17 GiB per device** of paged cache at the advertised context against the `optimized` policy it shares its weights with (24.165 → 22.993 GiB free), or 1.38 GiB against the selected config | rejected — no speed, no accuracy, more memory |
| **BFP8 CCL payload** (C13) | **−1.33 %**. Both per-layer collectives carry half the bytes and the layer is *slower*: the cast is one extra `typecast` dispatch per collective, 80 per decode step | rejected on measurement — and it reproduces the multichip stage's own `CCL_CAST_BLOCKFLOAT` null result from the other direction |
| **BFP8 residual stream** (C14) | **−1.87 %**, same mechanism: `ttnn.rms_norm` takes no `dtype`, so a block-float residual has to be widened after each norm. Two `typecast` dispatches per layer against a residual that is one 2048-wide tile row per token (~4 KB) | rejected on measurement. It needed a code adaptation to run at all — see below |
| **BFP8 logits tensor** (C15) | +0.07 %, inside the spread. The 62,464-wide logits shard is written once and read once per step | rejected — no measurable win, and bfloat16 is what the shared `TTSampling` top-k is built on |
| **BFP4 shared expert** (C08 / C09) | +0.03 % / −0.06 %, both inside the spread, at 0.950 top-1 | rejected — the shared expert is not a material decode cost here |
| **LoFi shared expert** (C07) | −0.08 %, and top-1 drops to 0.920 | rejected on measurement |
| **BFP8 router** (C21) | −0.12 %. Expert *selection* is a discrete decision and the group is 256 columns wide | rejected on measurement; the skill's own low-priority arm |
| **LoFi SDPA without fp32 accumulate** (C20) | +0.04 %, inside the spread | rejected — decode SDPA at this context is a small share |
| **HiFi2 routed experts** (C10) | **−2.20 %**, the largest regression in the matrix | rejected — this is the arm that *earns* the shipped BFP4+LoFi |
| **first/last layer exceptions** (C17) | −0.32 % against C06 at identical top-1 | rejected (§4.4) |
| **BFP4 routed-expert activation** (C19) | **blocked, twice.** See below | not measurable |

### 5.1 The residual arm needed an adaptation before it could be rejected

The first attempt at C14 did not build:

```
TT_FATAL @ nlp_create_qkv_heads_decode_device_operation.cpp:41:
  input_tensor.dtype() == FLOAT32 || input_tensor.dtype() == BFLOAT16
```

`ttnn.rms_norm` has no `dtype` argument, so its output takes the residual's dtype, which then reaches
the token-mixer in-projection and the QKV head split. A first API error is not a rejection, so
`OptimizedDecoder._widen_norm_output` was added: it restores `NORM_OUTPUT_DTYPE` when a block-float
residual made the norm produce one, and **dispatches nothing at all** under any shipped policy.
C14 then ran, and lost on measurement — which is what the −1.87 % row records.

### 5.2 C19 is blocked, with two exact blockers and a triage capture

[`blocked/C19-expert-act-bfp4.json`](blocked/C19-expert-act-bfp4.json) has the full record.

1. `ttnn.experimental.deepseek_moe_fast_reduce_nc` — the routed-expert reduction — rejects the dtype
   outright: `TT_FATAL: DeepseekMoEFastReduceNC input only supports specific data types.
   [BFLOAT16, BFLOAT8_B]` (`moreh_helper_functions.cpp:285`). The op's own validator enumerates the
   two it accepts; `bfloat4_b` is not a legal input at all.
2. The adaptation — widen to `bfloat8_b` before the reduction, the same shape of fix §5.1 used, and
   also left in place and inert — moved the failure to trace capture:
   `TT_FATAL: Writes are not supported during trace capture. trace id: 0`, on all four devices,
   followed by a mesh stall. [`triage/tt-triage-C19.txt.gz`](triage/) captured it before the process was
   killed; `check_binary_integrity` reports kernel `.text` mismatches on `eltwise_binary_no_bcast` and
   the interleaved reader/writer on all four devices — the same signature the optimized full-model
   stage's README §9 records for its traced-first-token-sampling hang. A decode step that cannot be
   *captured* cannot be ranked by this sweep at all, since untraced decode is not admissible evidence.

The arm was not pursued further because it is a *narrowing of an activation*, and the two directly
comparable narrowing arms that did run (C13 at −1.33 %, C14 at −1.87 %) both lost for exactly the
reason this one would. There is no evidence it would be faster, and the BFP4+LoFi coverage the skill
requires is about matmul weight groups — all four of which were measured on both fidelities (§4.1).

Hardware was recovered as `$tt-device-usage` prescribes: the stalled process was killed, then bounded
`tt-smi -ls --local` (8 rows) → `tt-smi -r` (all four PCI devices re-initialised) → `tt-smi -ls --local`
(8 rows) → the mesh smoke (`open_ornith_mesh`/`close_ornith_mesh` → `MESH_SMOKE_OK`). No second reset
was needed and no locks were cleared. Recorded as infrastructure recovery, not a model result.

---

## 6. The selected config is consumed, not merely recorded

`$datatype-sweep` is explicit that a JSON field the code ignores does not satisfy the stage. Four
independent things pin it:

1. **The artifact is the default.** `OrnithModel.from_pretrained(policy=None)` — which is what
   `build_generator`, every readiness runner, `bench_full_model.py` and any vLLM adapter that goes
   through them pass — resolves through `tt/precision_config.py::resolve_policy`, which loads
   `doc/datatype_sweep/selected_precision_config.json`. Delete a field from that file and the build
   raises `precision config is missing …`; delete the file and it raises `FileNotFoundError` naming
   the escape hatch.
2. **The schema is checked against the dataclass at import time.** `precision_config.SCHEMA` must
   cover every `PrecisionPolicy` field exactly; a field added later without a schema entry raises on
   import rather than being silently dropped from the artifact.
   `test_the_selected_precision_config_artifact_is_complete_and_round_trips` closes the other half.
3. **`model.precision_summary()` reads the built objects, not the policy.** Per layer: every weight
   tensor's own `dtype`, the K and V cache dtypes, the recurrent-state dtype, and the resolved
   per-layer policy name. Globally: the LM-head weight dtype, the LM head's constructed
   compute-kernel config, the logits dtype and the resolved prefill SDPA chunk.
   `test_the_selected_precision_config_is_the_built_policy` asserts all of it against the artifact,
   field by field, including `layer.compute_kernel_config.math_fidelity == proj_fidelity`,
   `layer.moe.expert_ckc.math_fidelity == expert_fidelity`,
   `layer.sdpa_compute_kernel_config.math_fidelity == sdpa_fidelity` and
   `layer.state_compute_kernel_config.math_fidelity == state_fidelity`.
4. **The two fields this stage *added* are proved to reach their ops.**
   `test_the_residual_and_logits_dtypes_reach_the_ops` builds an override with
   `residual_dtype=bfloat8_b, logits_dtype=bfloat8_b` and asserts the **traced decode logits buffer**
   comes back `BFLOAT8_B` — that buffer is the terminal matmul's own output, so it is `bfloat8_b` if
   and only if the matmul read `policy.logits_dtype`. Both arms were also measured end to end as C14
   and C15.

### 6.1 And the kernels agree — `tt-perf-report` under the selected policy

Constructor state is not a measured runtime row, so [`tracy/`](tracy/) is a `tt-perf-report` decode
capture on the reduced two-layer variant with **no policy argument**, i.e. the selected config by
default ([`logs/run_profiling.sh`](logs/run_profiling.sh)). Against the previous stage's capture of
the same script under the pre-sweep policy
([`../optimized_full_model/tracy/`](../optimized_full_model/tracy/)):

| dense-matmul shape | what it is | baseline policy | **selected policy** |
|---|---|---|---|
| `32 x 2048 x 2560` ×8 | packed attention in-projection | `HiFi2 BF16 x BFP8 => BF16` | **`LoFi BF16 x BFP4 => BF16`** |
| `32 x 2048 x 3136` ×8 | packed DeltaNet in-projection | `HiFi2 BF16 x BFP8 => BF16` | **`LoFi BF16 x BFP4 => BF16`** |
| `32 x 1024 x 2048` ×16 | `o_proj` / `gdn_out` | `HiFi2 BF16 x BFP8 => BF16` | **`LoFi BF16 x BFP4 => BF16`** |
| `32 x 2048 x 62464` ×8 | **LM head** | `HiFi2 BF16 x BFP8 => BF16`, **374 µs** | **`LoFi BF16 x BFP4 => BF16`, 273 µs** |
| `32 x 2048 x 288` ×16 | shared-expert down | `HiFi2 BF16 x BFP8 => BF16` | unchanged |
| `32 x 128 x 2048` ×16 | shared-expert gate/up/router | `HiFi2 BF16 x BFP8 => BF16` | unchanged |
| `32 x 2048 x 256`, `32 x 256 x 64` ×32 | router | `HiFi4 BF16 x BF16 => BF16` | unchanged |
| routed `SparseMatmul` ×32 | experts | `LoFi … x BFP4 => BFP8` | unchanged |

The arithmetic closes exactly: the baseline capture has **72** `HiFi2 BF16 x BFP8 => BF16` rows and
**zero** `LoFi BF16 x BFP4 => BF16`; the selected capture has **40** of the latter and **32** of the
former, and 40 + 32 = 72. Forty rows moved — the four dense projection roles plus the LM head — and
thirty-two stayed, which is the shared expert. No row shows a dtype or a fidelity the selected config
does not name.

The LM-head row also shows §2's launch-bound story from the other side: the weight halves, but the row
goes 374 → 273 µs (−27 %) rather than −47 %, because its DRAM efficiency falls from 69.0 % to 48.6 %.
(Absolute times in a Tracy window are inflated by the profiler; the ratios are the point, and the
un-profiled wall clock is §1's.)

`precision_summary().built` records the same two facts per candidate in
[`sweep_results.csv`](sweep_results.csv)'s `built_lm_head_weight_dtype` and
`built_lm_head_math_fidelity` columns — which is how §4.2's "C06 also moved the LM head" was found
rather than assumed.

`precision_summary().per_layer[*].math_fidelity` — all six constructed compute-kernel fidelities per
layer — was added *after* the sweep ran, so **the archived records in [`runs/`](runs/) do not carry
it**. For the fidelity-only arms (C07, C09, C10, C20, C21) the per-candidate artifact therefore has
the policy JSON plus the LM-head/dense-group built columns, and the *built* per-layer fidelity comes
from two other places instead: `test_the_selected_precision_config_is_the_built_policy`, which
asserts all six constructed compute-kernel configs against the artifact on every run of the suite,
and C10's −2.20 %, which is a fidelity-only change with no dtype movement at all and could not have
produced that signal unless the fidelity reached the kernel. A run record made from here on carries
the rows directly.

### Getting back to the safe baseline

One environment variable, no code change:

```bash
ORNITH_PRECISION_POLICY=optimized     python ...   # the pre-sweep decoder-stage policy
ORNITH_PRECISION_POLICY=fused-parity  python ...   # the bfloat16 / HiFi4 correctness floor
ORNITH_PRECISION_POLICY=doc/autoports/.../C23-....json   # any frontier point, by path
```

`test_the_precision_policy_can_be_overridden_back_to_the_safe_baseline` pins all four resolution
routes (default, env var, registered name, path) and that an unknown name raises.

---

## 7. Capability is preserved, and the headroom improves

[`../context_contract.json`](../context_contract.json)'s new `datatype_sweep` block is recomputed from
[`capacity/`](capacity/) — one measured allocator view per evaluated KV-cache dtype, each building the
whole 40-layer model and allocating the paged cache at the **full advertised 262144-token context**.

| policy | KV dtype | weights+embed+head | KV + per-batch state | resident | **free** | 262144 fits |
|---|---|---|---|---|---|---|
| **selected** | `bfloat8_b` | **5.581 GiB** | 1.679 | 7.268 | **24.377 GiB** | **yes** |
| pre-sweep `optimized` | `bfloat8_b` | 5.794 | 1.679 | 7.481 | 24.165 | yes |
| C11 | `bfloat4_b` | 5.794 | 1.054 | 6.856 | 24.790 | yes |
| C12 | `bfloat16` | 5.794 | 2.851 | 8.652 | 22.993 | yes |

**No advertised capability is reduced.** The selected config makes the resident weight set *smaller*
(the narrower projections and head), so free DRAM at the advertised context goes **up** by 0.21 GiB.
All three cache dtypes fit with room to spare, so the cache dtype was never a capacity decision for
this model at batch 1 — it was a speed and accuracy decision, and §4.3 records how it was made. The
batch bound stays 32 (`ttnn.sampling` asserts `1 <= num_users <= 32`).

**Non-aligned prompt lengths still work end to end.** The selected config changes no KV-cache dtype,
cache layout, trace buffer or prefill chunking — the resolved chunked-SDPA `q_chunk`/`k_chunk` is 256,
the same value the baseline resolves — but the check was rerun anyway, on the full stack with the full
advertised cache allocated ([`long_prompt.json`](long_prompt.json)):

| prompt | 5003 | 8191 | 16381 | 32749 | 65521 | 131071 | **262143** |
|---|---|---|---|---|---|---|---|
| prefill | 5.74 s | 3.46 | 6.95 | 14.24 | 30.17 | 67.55 | **163.49 s** |
| tokens/s | 872 | 2371 | 2356 | 2300 | 2172 | 1941 | **1603** |
| DRAM free after | 24.37 GiB | 24.37 | 24.37 | 24.37 | 24.37 | 24.36 | **24.36 GiB** |

Every row returns finite logits and a valid in-vocabulary sampled token; none of these lengths is a
multiple of the tile, the 64-token page, the 128-token block alignment or the 2048-token prefill
chunk. `test_prefill_accepts_any_logical_prompt_length` (1, 7, 31, 33, 63, 129, 250, 1000, 2049, 3000)
and `test_full_stack_non_aligned_long_prompt` (5003 through the complete stack) both pass on the
selected config.

---

## 8. Qualitative evidence

The shared `$qualitative-check` suite (`models/common/readiness_check/vllm_prompts.txt`, 6 prompts,
the checkpoint's own chat template, greedy, 128 new tokens) ran **twice on device** — once on the
selected config taken by default, once on the pre-sweep policy through `ORNITH_PRECISION_POLICY` —
and is joined to the previous stage's HF column in
[`qualitative_comparison.json`](qualitative_comparison.json) /
[`qualitative_comparison.md`](qualitative_comparison.md). All three arms are **asserted** to have seen
byte-identical rendered prompts (the comparison raises otherwise).

| prompt | selected vs pre-sweep word similarity | identical leading words | word doubling | repeated trigrams |
|---|---|---|---|---|
| haiku about machine learning | 0.509 | 11 | 0.000 | 0.000 |
| supervised vs unsupervised | 0.686 | 5 | 0.000 | 0.000 |
| story completion | 0.559 | 35 | 0.014 | 0.000 |
| three laws of thermodynamics | 0.750 | 54 | 0.000 | 0.028 |
| translate to French | 0.719 | 29 | 0.000 | 0.016 |
| Fibonacci function | 0.798 | 31 | 0.000 | 0.000 |

Nothing is degenerate: no empty completion, no word-doubling rate above 0.014, no repeated-trigram
rate above 0.028, no non-ASCII drift. Both arms produce the checkpoint's characteristic
`Here's a thinking process:` planning style and stay coherent for the full 128 tokens. The
similarities are 0.51–0.80 rather than 1.000 because greedy decoding diverges permanently once one
token differs — the "identical leading words" column is the honest measure of where each pair split.

**A fresh HF control could not be produced on this host.** The 35B CPU reference needs ~70 GiB and
`MemAvailable` is ~50 GiB, with ~198 GiB held outside anything this run can see; the OOM killer took
the process at 60 s. [`host_memory.md`](host_memory.md) has `dmesg`, `/proc/meminfo` and why the
archived HF column is nevertheless the right control (the HF reference is a torch model with no
dependence on the TTNN precision policy, so it is the *same* control for both TT arms, and the
question this stage has to answer is a TT-against-TT one). This is the same persistent host condition
the previous stage recorded.

---

## 9. Anomaly ledger

### 9.1 One of four identical decode slots emits a different token

```
Observed anomaly:  tests/test_full_model.py::test_the_batched_prefill_state_reaches_every_decode_slot
                   failed on the selected config: four slots prefilled with the same 4-token prompt
                   decoded [45568, 78562, 45568, 45568]. It passes on the pre-sweep policy.
Evidence:          logs/post_status.txt records the failing run (`pytest_short rc=1`, 02:33). Its
                   console log was OVERWRITTEN by the later all-pass rerun of the same path, so the
                   standing evidence is the focused probe rather than that log:
                   logs/probe_batch_slot_tie.py at four precisions/arms, artifacts
                   batch_slot_tie_{selected,baseline,fused_parity,no_merge}.json, which reproduce
                   the exact tokens the failure reported.
Affected path:     the low-level batched decode API a serving scheduler drives.
Control:           the SAME probe on `optimized` (bfloat8_b/HiFi2), on `fused-parity`
                   (bfloat16/HiFi4), and with `_merge_prefill_state_into_slot` DISABLED - the
                   negative control for the failure the test exists to catch.
Likely subsystem:  per-slot floating-point reduction order in the batch-4 decode geometry.
Investigation:     full-logit probe at four arms; then a batch-4 teacher-forcing accuracy run on the
                   whole 40-layer stack under both the selected and the pre-sweep policy.
Resolution:        classified - near-tie, not a state bug, and it costs no accuracy at batch 4 on the
                   full stack. The test now asserts the property it actually means, gated on the
                   contender spread, and the negative control fails it.
```

The probe reads the **full logit vector per slot** instead of only the argmax, three repeats each:

| policy | dense projections | max cross-slot logit difference | logit PCC vs slot 0 | top-1/top-2 margin | slots agree |
|---|---|---|---|---|---|
| `fused-parity` | bfloat16 / HiFi4 | 0.094 – 0.109 | 0.99995 | 0.3125 | yes |
| `optimized` | bfloat8_b / HiFi2 | 0.281 – 0.312 | 0.99966 – 0.99976 | 0.3125 – 0.4375 | yes |
| **selected** | bfloat4_b / LoFi | 0.281 – 0.500 | 0.99929 – 0.99965 | **0.00000 – 0.1875** | **no** |

Three things follow, and together they settle it:

* **Cross-slot decode has never been bit-identical, at any precision.** Even the bfloat16/HiFi4 floor
  shows 0.09–0.11 of logit difference between slots and a different top-5 *ordering* in slot 3. A
  batch-4 decode gives each row a different position in the sharded matmuls and collectives, so the
  reduction order differs per row. The test passed before only because the margin was larger than the
  noise.
* **The prefill state does reach every slot.** Every slot's logit vector tracks slot 0's at
  PCC ≥ 0.9993 and shares its top-5 candidate set. A slot whose prefill state never arrived would not
  correlate at all — which is the failure mode the test's own docstring names.
* **On the selected config this prompt is an exact tie.** In slot 0, tokens 45568 and 78562 both read
  **9.5625** — a top-1/top-2 margin of **0.00000**, i.e. the same bfloat16 value. The per-slot noise
  is 1–3 ULP at that magnitude (bfloat16 spacing near 9.6 is 0.0625), so it decides the ranking. It is
  fully deterministic — the same slot flips on all three repeats — so this is a fixed reduction order,
  not nondeterminism.

**The negative control was then run, and it changed the fix.** The same probe with
`_merge_prefill_state_into_slot` disabled — the failure the test exists to catch —
([`batch_slot_tie_no_merge.json`](batch_slot_tie_no_merge.json)):

| with the merge disabled | value |
|---|---|
| batch-4 **prefill** argmax | 58573 in every slot — still correct |
| batch-4 **decode** argmax | **267 in every slot**, against the batch-1 answer 45568 |
| cross-slot logit PCC | **0.99931 – 0.99954** — indistinguishable from a healthy build |
| cross-slot top-5 overlap | complete; every slot agrees with every other |
| top-1/top-2 margin | 1.69 – 1.88 — not a tie at all |
| batch-1 reference | **unaffected** (`[58573, 45568]`), so it stays a valid oracle |

So a *slot-against-slot* check does **not** detect this failure: without the merge all four slots are
equally wrong and therefore still agree. A PCC threshold would have been the wrong assertion to lean
on, and the first rewrite leaned on it. The assertion that actually carries the load is the
**batch-1 comparison**, and it is now in both branches.

The delivered test asserts: the **prefill** argmax is identical across slots (unchanged, still
passes); the device sampler and the read-back logits agree on the greedy token (so batch-4
device-sampled coverage is not lost); every slot's logits track slot 0's at PCC ≥ 0.999 — sensitive
to *one* slot diverging, explicitly not to all four being wrong together; and the decoded token
equals the **batch-1 answer** whenever the top-1/top-2 margin clears the **contender spread** — the
cross-slot spread of the *candidate tokens' own* logits, not the maximum over all 248,320 vocabulary
entries, which is a column no ranking depends on and which would have made the strict branch
unreachable under both shipped policies. Measured:

| policy | contender spread | min margin | strict branch fires |
|---|---|---|---|
| `fused-parity` | 0.0000 | 0.3125 | **yes** |
| `optimized` (pre-sweep) | 0.1875 | 0.3125 | **yes** |
| **no-merge control** | 0.1250 | 1.6875 | **yes — and it fails, 267 ≠ 45568** |
| selected | 0.1875 | 0.00000 | no — the genuine tie |

When the strict branch yields, the batch-1 answer must still be one of the tokens the slots chose and
no slot may choose anything outside the shared shortlist — which the no-merge control also fails.

### 9.1.1 And it costs nothing on the whole stack at batch 4

The probe above is the reduced two-layer variant. The question the vLLM stage inherits is what forty
layers of accumulated BFP4/LoFi error do to a batched request, so the readiness teacher-forcing check
was run on the **full 40-layer stack at batch 4**, the same AIME24 prompt and the same forced
continuation in every slot, logits read back per slot so each is *scored* rather than merely compared
([`logs/probe_batch4_accuracy.py`](logs/probe_batch4_accuracy.py)):

| | slot 0 | slot 1 | slot 2 | slot 3 | gate |
|---|---|---|---|---|---|
| **selected**, top-1 | 0.930 | 0.920 | 0.920 | **0.940** | ≥ 0.90 — every slot passes |
| **selected**, top-5 / top-100 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | ≥ 0.98 / = 1.00 |
| selected, token agreement with slot 0 | — | 0.99 | 0.99 | 0.99 | |
| pre-sweep `optimized`, top-1 | 0.940 | 0.960 | 0.950 | 0.960 | |
| pre-sweep, token agreement with slot 0 | — | **0.96** | **0.97** | **0.98** | |

Every slot of the selected config clears the gate on the full stack, and per-slot token agreement is
**higher** under the selected config (0.99) than under the pre-sweep policy (0.96–0.98) on this
workload — so batch-4 per-slot variation is a pre-existing property of the batched decode geometry,
not something the precision change introduced. Both artifacts:
[`batch4_accuracy_selected.json`](batch4_accuracy_selected.json),
[`batch4_accuracy_baseline.json`](batch4_accuracy_baseline.json).

### 9.2 `Here's a thinking thinking sequence`

```
Observed anomaly:  the story-completion prompt's completion opens with a doubled word.
Evidence:          qualitative_comparison.json, the "Complete this story" row.
Affected path:     free-running generation.
Control:           the HF reference's own completion for the same prompt.
Likely subsystem:  none - the checkpoint.
Investigation:     the archived HF column opens with the identical phrase, as does the pre-sweep
                   policy's arm. All three word-doubling rates are 0.014-0.015.
Resolution:        controlled - a property of the checkpoint, not of any precision policy.
```

### 9.3 Top-1 moves in both directions across the matrix

`C04` scores **0.970** on `run_prefill_check`, three points *above* the baseline's 0.940, while
scoring 0.960 on teacher forcing. **Eight** configurations beat the baseline on one gate and lose on
the other — C03, C04, C07, C08, C09, C14, C20 and C21 — and none beats it on teacher forcing, which
is the baseline's stronger gate. This is the same near-tie churn the optimized full-model stage documented (§1 there): with
top-5 and top-100 pinned at exactly 1.000 everywhere, single-token top-1 differences are rank flips
in the last bits of the logits, in both directions. It is why the charts plot `min(prefill, teacher)`
— the binding gate — rather than either one alone, and why §4 reads ±1–3 tokens as noise rather than
as signal.

---

## 10. Limitations

1. **The whole sweep spans 3.1 % of decode throughput, and the twenty non-regression configurations
   span 0.92 %.** The *passing* set spans 3.07 %, because all three regressions clear the accuracy
   gate — they are slow, not wrong. The step is
   launch-bound (§2). Precision is not the lever that moves this model; op count is. A future stage
   that fuses or removes decode ops should re-run this sweep afterwards, because the balance between
   "bytes saved" and "one more dispatch" is exactly what decides rows like C13 and C14.
2. **Selection is on one reference entry**: the AIME24 chat-template prompt with 100 generated
   tokens, which is the readiness reference the earlier stages generated and the one `$datatype-sweep`
   names. 100 tokens quantises top-1 to 0.01, so a one-token difference is one point. The gates are
   cleared by 2 points (top-1) and 2 points (top-5), and the matrix's own churn (§9.3) is ±3 points,
   so the *ordering* of configurations within 0.02 of each other is not resolved by this reference.
3. **No fresh HF qualitative control** — host memory, §8 and `host_memory.md`.
4. **`prefill_sdpa_chunk` is carried per config but only two values were measured** (256 for the
   BFP8-cache family, 128 for the bfloat16 cache, both inherited from the decoder stage's own
   legality sweep). The wide-cache clamp was also changed to compare element *widths* rather than to
   test for `bfloat8_b` by identity, so a BFP4 cache now resolves 256 instead of being clamped to 128
   for a reason that applies only to wider caches — and the sweep's 161-token prompt never reaches
   chunked prefill, so 256 with a BFP4 cache is unexercised. It does not affect the selected config,
   whose cache is `bfloat8_b`, and the BFP4 cache is rejected on accuracy anyway.
5. **C19 is blocked, not measured** (§5.2).
6. **Batch > 1 is measured for accuracy but not for throughput or capacity.** §9.1.1 runs the
   readiness teacher-forcing check at batch 4 on the full stack under both policies; every gate,
   benchmark and capacity probe elsewhere in the stage is batch 1, which is the vLLM primary
   single-user profile the previous stage established. Batched *throughput* and batched capacity at
   the advertised bound of 32 are the vLLM stage's to measure.
7. **The failing run's own pytest console log was overwritten** by the later all-pass rerun of the
   same path (§9.1). `logs/post_status.txt` preserves the run's `rc=1` and its timestamp, and the
   focused probe reproduces the exact tokens, but the original console is gone.
8. **TTFT is not a metric this stage claims to move.** The teacher-forcing TTFTs span
   **178.2 ms (C16) – 184.8 ms (C14)** across configurations whose prefill work differs by far less
   than that spread, and the
   optimized full-model stage established that this host's TTFT distribution is wider than the effects
   involved. The post-selection warmed TTFT (139.5 ms median) is reported because the benchmark
   produces it, not as a result.

---

## 11. Artifacts

```
doc/datatype_sweep/
├── README.md                              this file
├── work_log.md                            what was done, in order, with the commands
├── selected_precision_config.json         THE SELECTED CONFIG — the model's default policy
├── sweep_results.json / .csv              all 24 evaluated configs, full rows
├── top1_perf_pareto.png                   §1
├── top5_perf_pareto.png                   §1
├── post_selection_token_out.json          §1, the serving headline, selected config
├── post_selection_token_out_baseline.json §1, the same benchmark on the pre-sweep policy
├── long_prompt.json                       §7, the non-aligned prompt walk
├── readiness_qualitative.json             §8, the selected config
├── readiness_qualitative_baseline.json    §8, the pre-sweep policy
├── qualitative_comparison.{json,md}       §8, the three-way join
├── host_memory.md                         §8, why the HF control is archived
├── batch_slot_tie_{selected,baseline,fused_parity,no_merge}.json   §9.1, incl. the negative control
├── batch4_accuracy_{selected,baseline}.json               §9.1.1, batch 4 on the full stack
├── tracy/                                 §6.1, tt-perf-report decode capture, selected policy
├── candidates/                            one precision config per candidate, + index.json
├── runs/                                  one full record per evaluated config (10 repeats)
├── runs_4repeat_firstpass/                the first pass at 4 repeats, kept as a second
│                                          measurement of the same 21 configs (max |Δ| 0.172 %,
│                                          median 0.048 % against the delivered 10-repeat pass)
├── capacity/                              §7, one allocator view per KV-cache dtype
├── blocked/                               §5.2, C19's blocker record and the config that was tried
├── triage/                                §5.2, the tt-triage capture
└── logs/                                  every driver, and every raw console log
```
