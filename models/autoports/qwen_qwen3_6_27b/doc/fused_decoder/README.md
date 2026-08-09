# Qwen/Qwen3.6-27B — fused decoder

Graph-fused TTNN implementation of the Qwen3.6-27B (HF `model_type: qwen3_5`) decoder layers.
Same math and same public contract as the functional decoder, fewer and larger ops.

* Implementation: [`../../tt/fused_decoder.py`](../../tt/fused_decoder.py)
* Unfused reference it is compared against: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* Tests: [`../../tests/test_fused_decoder.py`](../../tests/test_fused_decoder.py),
  [`../../tests/test_fused_decoder_perf.py`](../../tests/test_fused_decoder_perf.py)
* Narrative, every rewrite tried and every one rejected: [`work_log.md`](work_log.md)
* Every measured number: [`pcc_evidence.json`](pcc_evidence.json) (278 records, 268 numeric)
* Before/after performance: [`perf_summary.json`](perf_summary.json)
* Context capability: [`../context_contract.json`](../context_contract.json)

Hardware, unchanged from the previous stage: two p300c boards of two Blackhole chips each;
board `000004613192404C` holds `/dev/tenstorrent/{0,1}` with chip 0's ARC wedged, board
`0000046131924022` holds `/dev/tenstorrent/{2,3}`, both healthy. Everything here runs on
**device 2**, 1x1 mesh, `TT_VISIBLE_DEVICES=2`.

## Result

Warmed prefill of 2048 tokens and warmed **traced** decode, batch 1, measured with the Tracy
device profiler inside signposted windows. Baseline and fused were re-measured back to back in
one sitting from the same test bodies, so the two columns differ only in the decoder class.

| layer kind | phase | device time, functional | device time, **fused** | Δ | ops/pass |
|---|---|---|---|---|---|
| `linear_attention` | prefill 2048 | 150.23 ms | **78.69 ms** | **−47.6 %** | 805 → 769 |
| `linear_attention` | traced decode | 3032.2 µs | **2395.2 µs** | **−21.0 %** | 96 → 86 |
| `full_attention` | prefill 2048 | 18.60 ms | **18.28 ms** | −1.7 % | 44 → 26 |
| `full_attention` | traced decode | 2425.2 µs | **2213.5 µs** | **−8.7 %** | 50 → 39 |

Host wall time moves the same way: `linear_attention` prefill 161.1 → 82.6 ms, its traced
decode 3.410 → 2.851 ms/token, `full_attention` prefill 20.16 → 19.87 ms and its traced decode
2.480 → 2.273 ms/token.

Both traced-decode paths beat the functional baseline, which is the gate this stage is
measured on. `full_attention` prefill barely moves because it was already 13.5 ms of its
18.6 ms in five matmuls sitting at **80–83 % of DRAM roofline** — there is no graph rewrite
left that touches them; the fusion took the op count from 44 to 26 without touching the floor.

Artifacts, per layer kind, under [`tracy/`](tracy/) (fused) and [`baseline/tracy/`](baseline/tracy/)
(functional): `<phase>_ops.csv`, `<phase>_ops.csv.provenance`, `<phase>_perf_report.txt`,
`<phase>_perf_report.csv`, `<phase>_perf_report.console.log`, `<phase>_perf_report_stacked.{csv,png}`.

## What was fused

Full derivations, measurements and the rejected candidates are in [`work_log.md`](work_log.md);
this is the index. Priority order is the graph-fusing skill's: dedicated ops, then graph
rewrites, then op merging.

### Dedicated fused ops

| id | unfused sequence | fused | where it paid |
|---|---|---|---|
| F1 | `slice → slice → silu → multiply` | `ttnn.swiglu` | both kinds, both phases; −3 ops/pass |
| F2 | partial RoPE: `slice ×3 → neg → concat → mul → mul → add → slice → concat` (×2) | one `ttnn.experimental.rotary_embedding_hf` per tensor | `full_attention`: 20 ops → 2; prefill 2.9 ms |
| F3 | L2 norm: `mul → sum → add → rsqrt → mul` (+ query scale) | one `ttnn.rms_norm` | `linear_attention`; −896 µs of BinaryNg in decode |
| F16 | `multiply → add` per conv tap | `ttnn.addcmul` | `linear_attention` prefill 15.38 → 13.73 ms |

F2 needs a host-side trick, because `rotary_embedding_hf` rotates the **whole** head while
Qwen3.5 rotates only the leading 64 of 256 channels. Permuting the head channels of `q`/`k`
(and of `q_norm`/`k_norm`) at load time moves the rotary block's second half to the
rotate-half midpoint, and `cos = 1` / `sin = 0` neutralises everything else. A permutation
applied identically to `q` and `k` leaves `q·k` unchanged, so attention is untouched; `v`,
the output gate and `o_proj` keep HF's order. See `rope_channel_permutation` and
[`work_log.md` §2](work_log.md).

F3 is an identity, not an approximation: `rms_norm(x, eps') = x·√D / √(Σx² + D·eps')`, so
`eps' = 1e-6/D` reproduces HF's `l2norm(x, eps=1e-6)` up to the constant `√D`, which is folded
into the norm weight together with the `1/√head_k_dim` query scale.

### Graph rewrites

| id | rewrite | effect |
|---|---|---|
| F6 | L1 residency for the triangular inverse and the per-chunk recurrence loop | the biggest single win: `linear_attention` prefill −45 ms |
| F7 | `TRI_INV_BASE` 16 → 32 | one recursion level fewer, and a 32×32 block fills a whole tile |
| F15 | width-sharded decode RMSNorm | 103.8 µs → 24.6 µs per norm, twice per decode step, both kinds |
| F5 | `in_proj_b` + `in_proj_a` → one shared-LHS matmul | `linear_attention` decode −60 µs |
| F8 | decode conv reads the state rows directly instead of rebuilding the window | −2 ops/step |

F6 is worth stating plainly because it is not an op-count change and it is where nearly all of
the `linear_attention` prefill win came from. A batched `32×32×32` matmul costs **1.06 µs per
batch element out of DRAM and 0.043 µs out of L1** — 24× — and the recursive triangular
inverse is thousands of them. The DRAM version was 47.9 ms of the 150 ms baseline; it does not
appear in the fused profile's top ops at all.

F15 matters for the same reason: the interleaved layernorm kernel parallelises over tile
*rows*, and a decode activation has exactly one, so both full-width norms ran on a **single
core** at 102 µs each — 8.5 % of a decode step, in both layer kinds.

### Op merging

| id | merge |
|---|---|
| F9 | attention output gate: `matmul → sigmoid` → `ttnn.linear(activation="sigmoid")` |
| F10 | gated-delta-net `z`: `matmul → silu` → `ttnn.linear(activation="silu")` |
| F11 | `dt_bias`: `matmul → add` → `ttnn.linear(bias=...)` |
| F13 | decay mask: `tril → exp → tril` → `add(triu −inf) → exp` |
| F14 | `attn0`: `multiply → tril → neg` → `multiply → multiply(mask)` |

### Assessed and rejected, with the measurement

| candidate | why not |
|---|---|
| `ttnn.transformer.gated_delta_attn_seq` (dedicated chunked gated-delta-rule kernel) | **slower and less accurate**: 167.9 ms for the delta rule alone versus 150 ms for the whole baseline layer, and PCC 0.9872/0.9878 against HF versus a 0.995 bar |
| shared-LHS `wqkv` + `wgate` | slower: decode 373.3 µs vs 364.2 µs, prefill 5182 µs vs 3510 µs — the two matmuls already run at 82 % of DRAM roofline |
| `ttnn.experimental.paged_fused_update_cache` | requires its two inputs on **disjoint** core ranges; `nlp_create_qkv_heads_decode` puts K and V on the same batch cores, and the reshard costs the dispatch the fusion saves |
| sharded SDPA-decode output feeding `nlp_concat_heads_decode` | rejected by the op for GQA (`sdpa_decode_device_operation.cpp:405`) |
| sharded `q` through the decode SDPA | **wrong**, not just slower: the kernel derives each user's core from the device grid width, so an 8-wide 32-core shard gives PCC ~0.02 for users 8–31 |
| `ttnn.conv1d` (depthwise) for the causal conv | no valid slicing configuration at 10240 channels (`op_slicing.cpp:266`) |
| `ttnn.matmul(transpose_a/transpose_b=True)` | accepted and kept, but it is **not** a fusion here: ttnn still lowers it to a separate `TransposeDeviceOperation` (75 transposes before, 75 after) |
| bfloat16 causal conv | 13.73 → 4.11 ms, a further 12 % of `linear_attention` prefill — but it is a precision trade, not a graph rewrite, and belongs to the optimization stage. Numbers recorded in [`work_log.md` §7](work_log.md). |

## Correctness

Acceptance bar: **PCC ≥ 0.995**, the functional stage's bar, unchanged.

The whole functional suite runs against the fused layer — same tests, same parametrisation,
same unfriendly sequence lengths — by re-pointing `harness.DECODER_CLS`
(`tests/test_fused_decoder.py`). **62 passed, 2 skipped** (the two long-context tests, run
separately below). Minimum over the 268 numeric records, excluding the one inherited gap:
**0.998815**, against the functional stage's 0.998817 — a 2e-6 delta, i.e. no material change.

Both evidence files were compared record by record: of the **262 measurements the two stages
share, none regressed by more than 1e-4**; the mean change is +7.5e-6, the worst is −9.2e-5 and
the best is +9.8e-4. Per class (minimum over each class, functional → fused):

| measurement class | n | functional | **fused** |
|---|---|---|---|
| min over all numeric records except the known gap | 268 | 0.998817 | **0.998815** |
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
`full_attention` decode at position 262143 reaches 0.5512 (functional: 0.5503). TTNN's
SDPA-decode kernel keeps the flash-attention running max, softmax denominator and output
accumulator in `Float16_b` regardless of `fp32_dest_acc_en`
(`sdpa_program_factory.cpp:657`, fp32 variant commented out citing tt-metal issue #13364), and
the fix that works for prefill — capping the merged k chunks — needs a *runtime* quantity in a
*compile-time* program config for decode. The functional stage's model-free reproducers
(`../functional_decoder/probes/probe_sdpa_synthetic.py`,
`probe_sdpa_decode_synthetic.py`) are unchanged and are what an upstream issue needs.
Fusing neither caused nor could fix it; the 0.001 difference is ordinary bf16 reordering.

### Fused-path assertions

PCC alone cannot tell a fused graph from a functional one that happens to pass, so
`test_fused_ops_are_used` spies on the dispatch: a measured prefill **and** a measured decode
must dispatch `ttnn.swiglu` (both kinds) and `ttnn.experimental.rotary_embedding_hf`
(`full_attention`), and must **not** dispatch the ops the rewrite retired (`ttnn.neg`,
`ttnn.sigmoid`, `ttnn.rsqrt`). `test_matches_functional_decoder` then compares the two
implementations against **each other** on identical weights — prefill output, decode output
and the un-paged K/V cache — which is the equivalence the transform actually promises.

### Stress and repeats

`test_repeated_prefill_decode_stress` runs 12 back-to-back prefill+decode passes per layer
kind and asserts (a) every repeat is **bit-identical** to the first, (b) the PCC never drops
below the bar, and (c) `total_bytes_allocated_per_bank` in DRAM does not move after the first
warm pass — a leak in the fused path, or L1 fragmentation, shows up there. `test_determinism`
additionally asserts bit-identical prefill and decode for both kinds, as before.

## Watcher

`TT_METAL_WATCHER=10` over both layer kinds' paged prefill, paged decode, traced decode, the
BFP8 KV-cache path and the fused-op assertions: **7 passed**, watcher log clean. Full audit,
including the exact command and the grep that finds no fatal/assert/sanitize/corruption lines:
[`watcher/WATCHER_AUDIT.md`](watcher/WATCHER_AUDIT.md). Raw log:
`watcher/generated/watcher/watcher.log`. Watcher and the device profiler were run separately,
as the tt-device-usage skill requires.

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
  so every cache-vs-HF comparison in the suite is against HF's channel order.

Capacity is unchanged: the advertised 262144-token context is still constructed, prefilled and
decoded at, and re-verified in this stage (`logs/long_context.log`). No memory-layout,
sharding or KV-cache-dtype change here alters capacity — the L1 residency of F6 is bounded by
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

# collect every recorded number
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
    $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/logs/*.log \
    --out $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/pcc_evidence.json
```

Profiling (one at a time, `ttenv_profiler.sh`, never alongside watcher):

```bash
cd $REPO/models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
./probes/run_perf.sh linear_attention decode fused          # -> tracy/
./probes/run_perf.sh linear_attention decode functional "$PWD/baseline"   # -> baseline/tracy/
```

Model-free op probes (`doc/fused_decoder/probes/`): `probe_fused_ops.py` (swiglu semantics,
the permuted RoPE identity, the L1-vs-DRAM batched matmul), `probe_fused_ops2.py`
(`gated_delta_attn_seq` accuracy and cost at the real head geometry, sharded-op constraints),
`smoke_fused.py` (fast prefill+decode PCC loop used during bring-up).

## Known limitations

* **`full_attention` decode at very long positions** — inherited unchanged; see above.
* The bfloat16 causal conv (a further ~13 % of `linear_attention` prefill) is measured but not
  taken: it is a precision trade for the optimization stage, not a graph rewrite.
* `linear_attention` prefill is still layout-bound: 15.3 % of it is five `ReshapeView` ops that
  split the fused projection output into heads, and TTNN has no dedicated head-split op for
  this geometry (16 k-heads / 48 v-heads, so `nlp_create_qkv_heads` does not apply). See
  [`work_log.md` §8](work_log.md).
* Trace capture is exercised at batch 1 and batch 4, not batch 32 — unchanged from the
  functional stage.
* Prefill is single-user per call by construction; batched prefill is a loop over `user_id`.
