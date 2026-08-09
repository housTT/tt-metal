# Qwen/Qwen3.6-27B — fused decoder

Graph-fused TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers.
Same math and same public contract as the functional decoder, fewer and larger ops.

* Implementation: [`../../tt/fused_decoder.py`](../../tt/fused_decoder.py)
* Unfused reference it is compared against: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* Tests: [`../../tests/test_fused_decoder.py`](../../tests/test_fused_decoder.py),
  [`../../tests/test_fused_decoder_perf.py`](../../tests/test_fused_decoder_perf.py)
* Narrative, every rewrite tried and every one rejected: [`work_log.md`](work_log.md)
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) (282 records, 276 numeric)
* Before/after performance: [`perf_summary.json`](perf_summary.json)
* Context capability: [`../context_contract.json`](../context_contract.json)

Hardware, unchanged from the previous stage: two p300c boards of two Blackhole chips each;
board `000004613192404C` holds `/dev/tenstorrent/{0,1}` with chip 0's ARC wedged, board
`0000046131924022` holds `/dev/tenstorrent/{2,3}`, both healthy. Everything here runs on
**device 2**, 1x1 mesh, `TT_VISIBLE_DEVICES=2`, compute grid 11x10.

## Result

Warmed prefill of 2048 tokens and warmed **traced** decode, batch 1, measured with the Tracy
device profiler inside signposted windows. The baseline was **re-measured in this stage**, back
to back with the fused runs from the same test bodies, so the two columns differ only in the
decoder class.

| layer kind | phase | functional | **fused** | Δ | ops in window |
|---|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 150.23 ms | **67.44 ms** | **−55.1 %** | 805 → 761 |
| `linear_attention` | traced decode | 3032.2 µs/token | **2254.5 µs/token** | **−25.6 %** | 96 → 68 |
| `full_attention` | prefill 2048 | 18.60 ms | **17.70 ms** | **−4.8 %** | 44 → 24 |
| `full_attention` | traced decode | 2425.2 µs/token | **2200.6 µs/token** | **−9.3 %** | 50 → 36 |

Host wall time moves the same way: `linear_attention` prefill 161.1 → 71.1 ms, its traced
decode 3.410 → 2.310 ms/token; `full_attention` prefill 20.16 → 19.01 ms, its traced decode
2.480 → 2.255 ms/token.

Both traced-decode paths beat the functional baseline, which is the gate this stage is measured
on. Both are now dominated by matmuls at the DRAM roofline: **89 %** of a `full_attention`
decode step and **82 %** of a `linear_attention` one is matmul or SDPA running at 80–84 % of
peak DRAM bandwidth, which no graph rewrite can move. `full_attention` prefill is the same
story at 83 %.

Artifacts, per layer kind, under [`tracy/`](tracy/) (fused) and [`baseline/tracy/`](baseline/tracy/)
(functional): `<phase>_ops.csv`, `<phase>_ops.csv.provenance`, `<phase>_perf_report.txt`,
`<phase>_perf_report.csv`, `<phase>_perf_report.console.log`, `<phase>_perf_report_stacked.{csv,png}`.

## What was fused

Full derivations, measurements and the rejected candidates are in [`work_log.md`](work_log.md);
this is the index. Priority order is the graph-fusing skill's: dedicated ops, then graph
rewrites, then op merging. **Every entry below was verified against the profiler**, because
several ttnn helpers that look like fusions are composites that lower back to the sequence they
appear to replace — see the "measured, not assumed" note after the tables.

### Dedicated fused ops

| id | unfused sequence | fused | measured |
|---|---|---|---|
| F2 | partial RoPE: `slice ×3 → neg → concat → mul → mul → add → slice → concat`, twice | one `ttnn.experimental.rotary_embedding_hf` per tensor | `full_attention` 20 device ops → 2; 636 → 367 µs of prefill |
| F17 | head split: `slice → reshape → permute` per tensor | two overlapping `ttnn.experimental.nlp_create_qkv_heads` calls | `linear_attention` prefill 11.25 → 3.14 ms, **bit-identical** |
| F3 | L2 norm: `mul → sum → add → rsqrt → mul`, plus the query scale | one `ttnn.rms_norm` | `linear_attention`; 10 device ops → 2 |
| F16 | `multiply → add` per conv tap | `ttnn.addcmul` | `linear_attention` prefill 15.38 → 13.73 ms, 3 fewer 84 MB temporaries |

F2 needs a host-side trick, because `rotary_embedding_hf` rotates the **whole** head while
Qwen3.5 rotates only the leading 64 of 256 channels. Permuting the head channels of `q`/`k`
(and of `q_norm`/`k_norm`) at load time moves the rotary block's second half to the rotate-half
midpoint, and `cos = 1` / `sin = 0` neutralises everything else. A permutation applied
identically to `q` and `k` leaves `q·k` unchanged, so attention is untouched; `v`, the output
gate and `o_proj` keep HF's order. See `rope_channel_permutation` and [`work_log.md` §2](work_log.md).

F17 needed a trick too: `nlp_create_qkv_heads` requires K and V to have the same head count and
this mixer has 16 key heads and 48 value heads. Two *overlapping* calls express it anyway,
because 48 value heads are three consecutive groups of 16. [`work_log.md` §7](work_log.md).

F3 is an identity, not an approximation: `rms_norm(x, eps') = x·√D / √(Σx² + D·eps')`, so
`eps' = 1e-6/D` reproduces HF's `l2norm(x, eps=1e-6)` up to the constant `√D`, which is folded
into the norm weight together with the `1/√head_k_dim` query scale.

### Graph rewrites

| id | rewrite | measured |
|---|---|---|
| F6 | L1 residency for the triangular inverse and the per-chunk recurrence loop | the largest single win: `linear_attention` prefill −45 ms |
| F15 | width-sharded decode RMSNorm | 103.8 → 24.6 µs per norm, twice per decode step, both kinds |
| F7 | `TRI_INV_BASE` 16 → 32 | one recursion level fewer; a 32×32 block fills a whole tile |
| F19 | concatenate the per-chunk outputs along the **sequence** axis, not the chunk axis | removes a permute and a 2.3 ms reshape |
| F20 | flatten the gated norm's output instead of reshaping `z` into head shape | removes a 3.1 ms reshape |
| F8 | decode conv state as `conv_kernel_size − 1` per-tap row buffers | removes 4 untilize/slice/retilize round trips per step |
| F5 | `in_proj_b` + `in_proj_a` → one shared-LHS matmul with a fused bias | `linear_attention` decode −60 µs |
| F18 | explicit 4×8 core grid for that small `32 × 5120 × 128` projection | 63 → 35 µs, clears its `SLOW` flag |

F6 is worth stating plainly because it is not an op-count change and it is where most of the
`linear_attention` prefill win came from. A batched `32×32×32` matmul costs **1.06 µs per batch
element out of DRAM and 0.043 µs out of L1** — 24× — and the recursive triangular inverse is
thousands of them. The DRAM version was 47.9 ms of the 150 ms baseline; it does not appear in
the fused profile's top ops at all.

F15 matters for the same reason: the interleaved layernorm kernel parallelises over tile *rows*,
and a decode activation has exactly one, so both full-width norms ran on a **single core** at
102 µs each — 8.5 % of a decode step, in both layer kinds.

### Op merging

| id | merge | measured |
|---|---|---|
| F1 | SwiGLU: `slice ×2 → silu → multiply` → `slice ×2 → multiply(SiLU on b)` | −1 device op; −471 µs of `full_attention` prefill |
| F9 | attention output gate: `matmul → sigmoid → multiply` → `matmul → multiply(sigmoid on b)` | −1 device op; −166 µs of `full_attention` prefill |
| F10 | gated-delta-net `z`: `matmul → silu → multiply` → `matmul → multiply(SiLU on b)` | −1 device op |
| F11 | `dt_bias`: `matmul → add` → `ttnn.linear(bias=...)` | −1 device op (verified: no add follows the matmul) |
| F13 | decay mask: `tril → exp → tril` → `add(triu −inf) → exp` | −1 device op on a 25 MB tensor |
| F14 | `attn0`: `multiply → tril → neg` → `multiply → multiply(mask)` | −1 device op |

**Measured, not assumed.** `ttnn.swiglu` and `ttnn.linear(activation=...)` were tried first and
are **not** fusions on this build: `swiglu` is a composite that dispatches
`split → swish → multiply` (`unary_composite_op.cpp:293`), and `matmul` applies a fused
activation as a separate `ttnn::unary_chain` unless a program config or `core_grid` is given
(`matmul.cpp:295`). The profiler showed the identical op sequence in both cases, and `swiglu`
additionally reports the tile-padded height as its logical height, which forced an extra
`slice` on every decode step. F1/F9/F10 are the *working* version of those merges: the
activation becomes an `input_tensor_b_activations` argument of the multiply that was already in
the graph, which really does collapse to one `BinaryNgDeviceOperation`. See
[`work_log.md` §5](work_log.md); `ttnn.matmul(transpose_a/transpose_b=True)` is kept for clarity
but is likewise **not** a fusion here (75 `TransposeDeviceOperation`s before and after).

### Assessed and rejected, with the measurement

| candidate | why not |
|---|---|
| `ttnn.transformer.gated_delta_attn_seq` (dedicated chunked gated-delta-rule kernel) | **slower and less accurate**: 167.9 ms for the delta rule alone versus 150 ms for the whole baseline layer, and PCC 0.9872/0.9878 against HF versus a 0.995 bar |
| shared-LHS `wqkv` + `wgate` | slower: decode 373.3 vs 364.2 µs, prefill 5182 vs 3510 µs — both matmuls already run at 82 % of DRAM roofline |
| widened `in_proj_qkv` (48 key heads) + one `nlp_create_qkv_heads` | works and is 13.68 → 5.15 ms in prefill, but costs +189 MB of weights per layer and **+205 µs/token in decode**, which is the gate. F17 gets 11.25 → 3.14 ms with no extra weights instead |
| `ttnn.experimental.paged_fused_update_cache` | requires its two inputs on **disjoint** core ranges; `nlp_create_qkv_heads_decode` puts K and V on the same batch cores, and the reshard costs the dispatch the fusion saves |
| sharded SDPA-decode output feeding `nlp_concat_heads_decode` | rejected by the op for GQA (`sdpa_decode_device_operation.cpp:405`) |
| sharded `q` through the decode SDPA | **wrong**, not just slower: the kernel derives each user's core from the device grid width, so an 8-wide 32-core shard gives PCC ~0.02 for users 8–31 |
| `ttnn.conv1d` (depthwise) for the causal conv | retried with explicit DRAM width slicing at 2/4/8/16/32 slices after the auto-config failure: the config search then succeeds but every slice count exhausts the allocator (`bank_manager.cpp:462`) |
| `ttnn.swiglu`, `ttnn.linear(activation=…)`, `ttnn.matmul(transpose_b=True)` | not fusions on this build — see "Measured, not assumed" above |
| bfloat16 causal conv | 13.73 → 4.11 ms, a further 14 % of `linear_attention` prefill — but it is a precision trade, not a graph rewrite, and belongs to the optimization stage. Numbers in [`work_log.md` §7](work_log.md) |

## Correctness

Acceptance bar: **PCC ≥ 0.995**, the functional stage's bar, unchanged.

The whole functional suite runs against the fused layer — same tests, same parametrisation,
same unfriendly sequence lengths — by re-pointing `harness.DECODER_CLS`
(`tests/test_fused_decoder.py`). **62 passed, 2 skipped** (the two long-context tests, run
separately below). The functional suite was re-run afterwards to prove the shared harness
changes did not disturb it: **55 passed, 2 skipped**, exactly as before
(`logs/suite_functional_regression.log`).

Both evidence files were compared record by record: of the **258 numeric measurements the two
stages share, none regressed by more than 1e-4**; mean change +7.5e-6, worst −9.2e-5, best
+9.8e-4. Minimum over the 276 numeric fused records, excluding the one inherited gap:
**0.998815**, against the functional stage's 0.998817.

| measurement class | n | functional | **fused** |
|---|---|---|---|
| min over all numeric records except the known gap | 276 | 0.998817 | **0.998815** |
| prefill vs HF, lengths 1/17/128/2048/2049/4096/5000 — `linear_attention` | 7 | 0.999888 | 0.999893 |
| prefill vs HF, same lengths — `full_attention` | 7 | 0.999423 | 0.999433 |
| decode vs HF, 4 steps after prefill 17/2048/2049/5000 — `linear_attention` | 16 | 0.999927 | 0.999925 |
| decode vs HF, same — `full_attention` | 16 | 0.998878 | 0.998941 |
| batch 32, 32 unequal prompts — decode, `full_attention` | 32 | 0.999130 | 0.999106 |
| batch 32, 32 unequal prompts — decode, `linear_attention` | 32 | 0.999931 | 0.999925 |
| traced decode, replay output vs HF | 6 | 0.999201 | 0.999161 |
| **real checkpoint weights** — prefill / decode @ 2049 | 2 / 2 | 0.999968 / 0.999987 | 0.999967 / 0.999987 |
| pad-below-one-tile lengths 735..768 — prefill and decode | 24 | 0.999220 | 0.999307 |
| longest reference-checkable prefill (16385 / 8191) | 2 | 0.999436 | 0.999446 |
| BFP8 KV cache — prefill / decode @ 2049 | 2 | 0.999144 | 0.999263 |
| full context 262143 — `linear_attention` prefill tail / decode | | 0.999942 / 0.999946 | 0.999940 / 0.999942 |
| full context 262143 — `full_attention` prefill tail / K / V cache | | 0.998817 / 0.999989 / 0.999993 | 0.998815 / 0.999988 / 0.999993 |
| full context 262143 — `full_attention` decode | | **0.550293** | **0.551271** |

Fused **against the functional decoder directly**, identical weights and inputs, seq 2049:
prefill 0.999898 / decode 0.999840 / K cache 0.999999 / V cache 1.000000 (`full_attention`),
prefill 0.999984 / decode 0.999986 (`linear_attention`).

**One known gap, inherited unchanged and still blocked on the same tt-metal kernel defect:**
`full_attention` decode at position 262143 reaches 0.5513 (functional: 0.5503). TTNN's
SDPA-decode kernel keeps the flash-attention running max, softmax denominator and output
accumulator in `Float16_b` regardless of `fp32_dest_acc_en`
(`sdpa_decode_program_factory.cpp:437`, with the prefill factory carrying the explicit
"disable fp32 cbs (Issue #13364)" comment), and the fix that works for prefill — capping the
merged k chunks — needs a *runtime* quantity in a *compile-time* program config for decode. The
functional stage's model-free reproducers
(`../functional_decoder/probes/probe_sdpa_synthetic.py`, `probe_sdpa_decode_synthetic.py`) are
unchanged and are what an upstream issue needs. Fusing neither caused nor could fix it; the
0.001 difference is ordinary bf16 reordering.

### Proving the fused path is really fused

PCC alone cannot tell a fused graph from a functional one that happens to pass, and — as F1/F9
showed — a Python-level check cannot either, because ttnn composites lower back to the sequence
they look like they replace. `test_fused_graph_is_smaller` therefore captures the **device** op
stream with `ttnn.graph` for one prefill and one decode of each decoder class and asserts the
fused one is strictly smaller:

| | prefill | decode |
|---|---|---|
| `linear_attention` functional → fused | 996 → **916** | 98 → **70** |
| `full_attention` functional → fused | 111 → **71** | 56 → **42** |

(These counts cover the whole call, including the input upload the signposted perf window
excludes, which is why they differ from the table at the top.) It also asserts that
`RotaryEmbeddingHfDeviceOperation` (F2) and `TernaryDeviceOperation` (F16, `addcmul`) appear in
the fused stream and in neither functional one. `test_matches_functional_decoder` then compares
the two implementations against **each other** on identical weights — prefill output, decode
output and the un-paged K/V cache.

### Stress and repeats

`test_repeated_prefill_decode_stress` runs 12 back-to-back prefill+decode passes per layer kind
and asserts (a) every repeat is **bit-identical** to the first, (b) the PCC never drops below
the bar, and (c) `total_bytes_allocated_per_bank` in DRAM does not move after the first warm
pass — a leak in the fused path, or L1 fragmentation, shows up there. `test_determinism`
additionally asserts bit-identical prefill and decode for both kinds, as before.

## Watcher

`TT_METAL_WATCHER=10` over both layer kinds' paged prefill, paged decode, traced decode, the
BFP8 KV-cache path and the device-op-count comparison: **7 passed**, watcher log clean. Full
audit, including the exact command and the grep that finds no fatal/assert/sanitize/corruption
lines: [`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md). Raw log:
`watcher/generated/watcher/watcher.log`. Watcher and the device profiler were run separately, as
`$tt-device-usage` requires.

## Contract

Identical to the functional decoder — `from_state_dict` signature and key set, arbitrary
`1 <= seq_len <= max_seq_len` with no public divisibility requirement, the paged page-table
protocol, per-user state, `prepare_decode_state` semantics, determinism, and "no `torch` after
`from_state_dict`" (asserted by `test_no_runtime_host_fallback`, which now scans whichever
decoder module is under test) — with **one** difference, forced by F2:

* `rot_mats` are `head_dim`-wide (256), not `rotary_dim`-wide (64), and carry `cos = 1`,
  `sin = 0` in the non-rotary channels, in the permuted channel order.
  `fused_decoder.rope_channel_permutation(head_dim, rotary_dim)` returns the permutation;
  `harness.expand_rot_mats` builds the tensors. The layer asserts the width so a
  functional-shaped `rot_mats` fails loudly instead of silently computing the wrong thing.
* Consequently the **paged K cache holds permuted head channels** (V does not — it never sees
  RoPE). `FusedDecoder.kv_channel_permutation` exposes it; `harness.read_paged_kv` inverts it,
  so every cache-vs-HF comparison in the suite is against HF's channel order. Every later stage
  that reads or transfers this layer's K cache must honour it.

Capacity is unchanged: the advertised 262144-token context is still constructed, prefilled and
decoded at, and re-verified in this stage (`logs/long_context.log`). No memory-layout, sharding
or KV-cache-dtype change here alters capacity — the L1 residency of F6/F15 is bounded by
`L1_BUDGET_BYTES` and falls back to DRAM rather than failing, and the KV cache keeps its
`bfloat16` (or caller-chosen `bfloat8_b`) dtype and its `[num_blocks, n_kv, block, head_dim]`
shape. `../context_contract.json` carries a `fused_decoder` section recording exactly this.

## Running

All device jobs run from `/home/ttuser/dev/qwen/rundir` with [`ttenv.sh`](ttenv.sh) sourced.

```bash
cd /home/ttuser/dev/qwen/rundir && source ./ttenv.sh

# full fused suite (64 tests, ~8 min)
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s

# full advertised context, 262144 tokens (2 tests, ~6 min)
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k test_full_advertised_context --long-context -v -s

# functional regression, proving the shared harness change is inert
python -m pytest $REPO/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py -q

# collect every recorded number
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/logs/*.log \
    --out $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/pcc_evidence.json
```

Profiling (one at a time, `ttenv_profiler.sh`, never alongside watcher):

```bash
cd $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
./probes/run_perf.sh linear_attention decode fused                        # -> tracy/
./probes/run_perf.sh linear_attention decode functional "$PWD/baseline"   # -> baseline/tracy/
```

Model-free op probes, under [`probes/`](probes/):

| probe | question | log |
|---|---|---|
| `probe_fused_ops.py` | swiglu semantics, the permuted-RoPE identity, L1 vs DRAM batched matmul | `logs/probe_fused_ops.log` |
| `probe_fused_ops2.py` | `gated_delta_attn_seq` accuracy and cost at the real head geometry | `logs/probe_fused_ops2.log` |
| `probe_candidates.py norm` | sharded decode RMSNorm, core-grid sweep | `logs/probe_norm.log` |
| `probe_candidates.py conv` | FIR fp32/bf16, `addcmul`, `ttnn.conv1d` | `logs/probe_conv.log` |
| `probe_candidates.py sharedlhs` | shared-LHS `wqkv`+`wgate` | `logs/probe_sharedlhs.log` |
| `probe_candidates.py ropebatch` | sharded decode RoPE, per-user PCC at every batch | `logs/probe_ropebatch.log` |
| `probe_review_followups.py`, `probe_headsplit_narrow.py` | head-split variants, `conv1d` explicit slicing, `b\|a` geometry | `logs/probe_review_followups.log`, `logs/probe_headsplit_narrow.log` |
| `smoke_fused.py` | fast prefill+decode PCC loop used during bring-up | — |

## Known limitations

* **`full_attention` decode at very long positions** — inherited unchanged; see above.
* The bfloat16 causal conv (a further ~14 % of `linear_attention` prefill) is measured but not
  taken: it is a precision trade for the optimization stage, not a graph rewrite.
* `linear_attention` prefill is still 9 % causal conv and 14 % elementwise work in the delta
  rule itself; both are bandwidth on 25–50 MB float32 tensors, and the remaining op-level
  bookkeeping around them has been removed. See [`work_log.md` §8](work_log.md).
* The `_decode_norm_config` grid search falls back to the interleaved (single-core) kernel if no
  candidate core count divides the hidden size in tiles. It does not for this model
  (5120 / 32 = 160 tiles, 10 cores), but a different hidden size could land there and would
  silently lose F15's ~150 µs/token.
* Trace capture is exercised at batch 1 and batch 4, not batch 32 — unchanged from the
  functional stage.
* Prefill is single-user per call by construction; batched prefill is a loop over `user_id`.
