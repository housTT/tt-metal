# Ornith-1.0-35B — fused decoder (TTNN, single Blackhole device)

Stage deliverable: a graph-fused TTNN decoder layer for `ornith-ai/Ornith-1.0-35B`, both layer
kinds, that computes exactly what the functional decoder computes and is materially faster.

<!-- generated:perf-headline -->
| Window | before (functional) | after (fused) | device kernel time |
| --- | --- | --- | --- |
| `linear_attention` prefill, 2048 tokens | 338.36 ms / 6052.7 tok/s | **257.48 ms / 7953.9 tok/s** | 337.759 → **256.504 ms** |
| `linear_attention` decode, traced | 2.613 ms / 382.7 steps/s | **2.063 ms / 484.7 steps/s** | 2.533 → **1.987 ms** |
| `full_attention` prefill, 2048 tokens | 316.71 ms / 6466.5 tok/s | **243.49 ms / 8411.1 tok/s** | 316.373 → **242.441 ms** |
| `full_attention` decode, traced | 2.393 ms / 418.0 steps/s | **1.829 ms / 546.8 steps/s** | 2.340 → **1.815 ms** |
<!-- /generated:perf-headline -->

* Implementation: [`tt/fused_decoder.py`](../../tt/fused_decoder.py) — self-contained; it reuses
  `tt/model_config.py` and leaves `tt/functional_decoder.py`, `tt/moe.py` and `tt/rope.py`
  untouched, because the functional decoder is this stage's measured baseline.
* Tests: [`tests/test_fused_decoder.py`](../../tests/test_fused_decoder.py)
* Capability contract: [`../context_contract.json`](../context_contract.json) (`fused_decoder` section)
* Fusing catalogue, bringup narrative, rejected candidates, hardware incident: [`work_log.md`](work_log.md)
* One-command regeneration of every artifact below: [`logs/run_evidence.sh`](logs/run_evidence.sh)
* Figure audit — asserts every measured number quoted in these documents exists in a committed
  artifact, re-evaluates the derived ones by re-computing them from their declared operands, and
  checks that the source files hash to the manifest the evidence was produced with:
  [`audit_figures.py`](audit_figures.py)

**Every numeric table in this document is generated, not transcribed.** The PCC tables in §2 come
from [`logs/make_readme_tables.py`](logs/make_readme_tables.py) and the performance tables above and
in §5 from [`logs/make_readme_perf.py`](logs/make_readme_perf.py); both splice into the
`<!-- generated:… -->` blocks below, and both are re-run in `--check` mode by `audit_figures.py`, so
a table that disagrees with its artifact fails the audit. This exists because round 3 of this stage's
review found eleven hand-transcribed PCC cells that described an earlier run. Percentages the
generators compute are written next to their arithmetic into
[`tracy/derived_figures.txt`](tracy/derived_figures.txt), which is itself an audited artifact.

Hardware: 1×1 Blackhole mesh (`p300c`, 11×10 compute grid).
Weights: `bfloat16`; DeltaNet recurrent state, gated-delta gates and router logits: `float32` —
identical to the functional decoder. **This stage changes no dtype and no math fidelity**; precision
policy belongs to a later stage, and mixing it in here would make the fusing measurements
uninterpretable.

---

## 1. Contract — unchanged

```python
FusedDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device,
    max_context=None,            # defaults to text_config.max_position_embeddings (262144)
    page_block_size=64, prefill_chunk=2048,
    moe_group_tokens=32,         # fused-stage knob, see §4
    rope_mode="partial",         # fused-stage knob, see §4
    dtype=ttnn.bfloat16,
) -> FusedDecoder

decoder.allocate_kv_cache(num_blocks, dtype=ttnn.bfloat16)   # full_attention only
decoder.attach_kv_cache(k_cache, v_cache)                    # or bring your own
decoder.allocate_state(batch_size)
decoder.reset_state()

decoder.prefill_forward(x, *, start_pos=0, page_table=None, chunk_size=None)
decoder.decode_forward(x, *, current_pos=None, rot_idxs=None, page_table=None)
```

Argument shapes, dtypes and semantics are exactly the functional decoder's — including that
`seq_len` may be **any** value in `[1, max_context - start_pos]` with no tile, page or chunk
divisibility requirement, and that `current_pos`/`rot_idxs` are device tensors so a captured decode
trace only needs its input buffers refreshed. `tests/test_fused_decoder.py` re-asserts every
capability the functional suite asserted, under the same test names.

---

## 2. Correctness

Acceptance bar: **PCC ≥ 0.995**, inherited from the functional-decoder stage and not lowered.
Golden = one `Qwen3_5MoeDecoderLayer` in float32 driven through the real HF prefill/decode cache
paths. Layer 0 is `linear_attention`, layer 3 is `full_attention`. PCC is accumulated in float64.

```bash
pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_fused_decoder.py -v -p no:randomly
```

<!-- generated:suite-result -->**97 passed** in 552.42 s<!-- /generated:suite-result -->. Log: [`logs/pytest_full_suite.txt`](logs/pytest_full_suite.txt); every
logged metric is extracted into [`logs/pcc_summary.txt`](logs/pcc_summary.txt) by
[`logs/summarise_pcc.py`](logs/summarise_pcc.py), which keeps all of them rather than an allow-list,
so a new test's evidence cannot silently miss the summary. The `long` (262144-token) cases are
**not** opt-in here — they run in the same invocation, so the tables below and the
advertised-context evidence come from one log.

**The log is not silent, and the noise is accounted for.** The suite passes, but it prints
`critical`-level `TT_FATAL`/`TT_THROW` lines: `allocate_state` decides whether `ttnn.conv1d` can
serve a given `(batch, block length)` by *executing* that program once at setup and catching the
refusal (§8 item 3), and a caught failure still prints. Left unexplained those look like real faults,
so [`logs/classify_suite_criticals.py`](logs/classify_suite_criticals.py) classifies every one into
[`logs/suite_criticals.txt`](logs/suite_criticals.txt) and **exits non-zero if any line is
unclassified**. All of them fall into the two L1-capacity classes that probe can provoke — an L1
buffer allocation refusal, and a circular-buffer set that exceeds L1 — and the same script reports the
resulting per-batch coverage, which the decoder logs itself:

<!-- generated:conv1d-coverage -->
| allocated batch | prefill block lengths `ttnn.conv1d` accepts | what runs |
| --- | --- | --- |
| 1 | 16 / 16 | every block uses `ttnn.conv1d`; this is the measured configuration in §5 |
| 4 | 6 / 16 | the rest use the FIR fallback |
| 8 | 3 / 16 | the rest use the FIR fallback |
| 32 | 0 / 16 | entirely FIR — correct, just not accelerated |
<!-- /generated:conv1d-coverage -->

<!-- generated:conv1d-coverage-cause -->
Coverage falls as the batch rises, and the refusals split between **two different L1 limits**,
which the classifier counts per batch (bank-allocation / circular-buffer): batch 4: 3 / 7, batch 8: 10 / 3, batch 32: 15 / 1.
Which limit dominates depends on the batch: the program-build circular-buffer set is the
majority at batch 4, but from batch 8 up it is the sharded input's
**per-bank allocation** — 15 of 16 refusals at batch 32 — which grows
with the batch because the conv shards its activation over the same cores. That matters for what
a future stage would change: at the batches where coverage actually collapses, a program-config
or CB-sizing fix addresses only the smaller half.
This is a *performance* fallback — every batch produces a result that clears the PCC bar, and
§2.4's batched rows are measured on whichever path that batch selected — but the two paths are
not bit-identical to each other, which §8 item 7 records. Only the `linear_attention` batches
have a conv path at all; the `full_attention`-only rows (13, 40, 56) are unaffected by this.
<!-- /generated:conv1d-coverage-cause -->

### 2.1 Prefill, before and after

Functional figures from `../functional_decoder/logs/pcc_summary.txt`, fused from
`logs/pcc_summary.txt`; same inputs, same golden, same bar. The tables are generated by
[`logs/make_readme_tables.py`](logs/make_readme_tables.py) rather than transcribed.

<!-- generated:prefill-pcc -->
| `seq_len` | boundary role | `linear_attention` before → after | `full_attention` before → after |
| --- | --- | --- | --- |
| 1 | single token | 0.999996 → **0.999996** | 0.999951 → **0.999951** |
| 7 | sub-tile, sub-page | 0.999994 → **0.999994** | 0.999946 → **0.999946** |
| 32 | exactly one tile | 0.999985 → **0.999985** | 0.999962 → **0.999962** |
| 64 | exactly one page block | 0.999988 → **0.999987** | 0.999960 → **0.999960** |
| 128 | exactly the physical alignment | 0.999982 → **0.999982** | 0.999966 → **0.999967** |
| 129 | one past the alignment | 0.999983 → **0.999984** | 0.999965 → **0.999965** |
| 250 | tile-padded height already equals the padded length | 0.999980 → **0.999980** | 0.999971 → **0.999971** |
| 2048 | exactly one internal chunk | 0.999971 → **0.999975** | 0.999978 → **0.999978** |
| 2049 | one past a chunk boundary | 0.999973 → **0.999976** | 0.999978 → **0.999978** |
| 3000 | multi-chunk, non-divisible | 0.999969 → **0.999973** | 0.999980 → **0.999980** |
| 8000 | long, non-divisible (`test_long_context_pcc`) | 0.999965 → **0.999970** | 0.999982 → **0.999982** |
<!-- /generated:prefill-pcc -->

### 2.2 Decode, before and after

`test_decode_pcc` prefills then runs four decode steps, comparing every step. `prefill_len = 130`
makes the decode writes cross a 64-token page boundary. The `8000` row is `test_long_context_pcc`.

<!-- generated:decode-pcc -->
| prefill_len | step | `linear_attention` before → after | `full_attention` before → after |
| --- | --- | --- | --- |
| 130 | 0 | 0.999995 → **0.999995** | 0.999942 → **0.999942** |
| 130 | 1 | 0.999993 → **0.999993** | 0.999959 → **0.999959** |
| 130 | 2 | 0.999969 → **0.999975** | 0.999913 → **0.999915** |
| 130 | 3 | 0.999994 → **0.999995** | 0.999894 → **0.999893** |
| 2048 | 0 | 0.999995 → **0.999995** | 0.999983 → **0.999984** |
| 2048 | 1 | 0.999982 → **0.999991** | 0.999966 → **0.999961** |
| 2048 | 2 | 0.999893 → **0.999909** | 0.999947 → **0.999949** |
| 2048 | 3 | 0.999990 → **0.999992** | 0.999898 → **0.999898** |
| 8000 | 0 | 0.999992 → **0.999991** | 0.999943 → **0.999944** |
<!-- /generated:decode-pcc -->

<!-- generated:pcc-delta -->
**Material delta: none.** Every prefill row moves by at most `6e-6` and every
decode row by at most `2e-5`, against a distance from the worst row (0.999893) to the
0.995 bar of `5e-3` — the *largest* prefill and decode moves are respectively
3.0 and 2.5 orders of magnitude smaller than that margin, and every other move is
smaller still — and the moves go
both ways. The largest single change is an *improvement* (`linear_attention`, prefill 2048,
decode step 2: 0.999893 → 0.999909). The stage's one accuracy-relevant change is
the expert-axis reduction switching to `deepseek_moe_fast_reduce_nc`, whose accumulation is
measurably more accurate than `fast_reduce_nc`'s (PCC 0.999999 vs 0.999409 against a
float32 sum of 256 bfloat16 expert blocks —
[`logs/probe_router_and_reduce.txt`](logs/probe_router_and_reduce.txt)); that is the plausible
source, but no artifact here isolates this row's move to it.
<!-- /generated:pcc-delta -->

### 2.3 Fused-vs-functional agreement

Clearing the HF bar is necessary but not sufficient — a fusing stage has to reproduce the
*implementation* it replaced. `test_fused_matches_functional` builds both decoders from the same
<!-- generated:equivalence-bar -->
state dict and drives them with the same inputs and page table. The bar is 0.9999 — an
order of magnitude tighter than the 0.995 acceptance bar, though not tighter than every measured
HF-golden agreement: 5 of them sit below it, the lowest at 0.999882. What makes this
the stricter check is not the number but the comparison — both sides share dtypes, device and
page table, so the only difference left is the graph. The measured equivalences are 0.999993
–1.000000, i.e. the worst is 1.2 decades inside the bar.
<!-- /generated:equivalence-bar -->

<!-- generated:equivalence-pcc -->
| `seq_len` | `linear_attention` prefill / decode | `full_attention` prefill / decode |
| --- | --- | --- |
| 1 | 1.000000 / 1.000000 | 0.999999 / 0.999999 |
| 130 | 0.999995 / 0.999993 | 0.999996 / 0.999997 |
| 300 | 0.999995 / 0.999997 | 0.999996 / 0.999998 |
<!-- /generated:equivalence-pcc -->

### 2.4 Everything else the functional stage asserted

<!-- generated:capability-table -->
| Property | Test | Result |
| --- | --- | --- |
| M-RoPE reduces to 1-D partial RoPE, **in both lowerings** | `test_rope_matches_hf` (`seq_len` 1 / 64 at start offset 12345, 4096 at 0; `rope_mode` `partial` and `full`) | every prefill and decode-gather cos/sin PCC 0.999999 |
| the two RoPE lowerings are interchangeable | `test_rope_mode_equivalence` | prefill 0.999995, decode 0.999996 |
| batch 4 prefill + batched decode | `test_batched_prefill_decode_pcc[4]` | linear 0.999985 / 0.999994; full 0.999962 / 0.999916 |
| batch 32 prefill + batched decode | `test_batched_prefill_decode_pcc[32]` | linear 0.999984 / 0.999974; full 0.999964 / 0.999935 |
| batched decode with **distinct per-user positions** over a shuffled disjoint page table | `test_batched_decode_ragged_positions[4]` / `[13]` | batch 4: 0.999917–0.999975; batch 13: 0.999894–0.999973 |
| decode batch past the dedicated ops' limits, so both fallbacks run | `test_decode_batch_above_head_split_limit[40]` / `[56]` | 40 (head-split fallback, fused cache update): 0.999933; 56 (head-split **and** two-launch cache update): 0.999934 |
| a batch **smaller** than the allocated one — the per-user-prefill serving pattern | `test_batch_smaller_than_allocated_state` | `full_attention` batch 1 on a batch-8 allocation: prefill 0.999970, decode 0.999956; `linear_attention` raises, as the functional decoder does, because its DeltaNet state is per-row |
| shuffled, offset page table | `test_permuted_page_table` | prefill 0.999968 (first physical slot 106); decode 0.999899 / 0.999928 |
| chunked-prefill continuation (2 calls, `start_pos > 0`) equals one call | `test_prefill_continuation` | linear 0.999979; full 0.999971 |
| decode under captured/replayed trace, PCC measured **from the replay** | `test_traced_decode_pcc` | linear 0.999979 / 0.999994 / 0.999994; full 0.999923 / 0.999931 / 0.999939 |
| real checkpoint weights | `test_real_weights_pcc` | linear 0.999981 / 0.999938; full 0.999970 / 0.999957 |
| synthetic weights from recorded real statistics (the CI path) | `test_synthetic_weights_pcc` | linear 0.999993 / 0.999995; full 0.999980 / 0.999990 |
| correctness independent of freed-DRAM contents | `test_forward_with_poisoned_free_pool` (`seq_len` 1 / 250 / 300) | linear prefill 0.999979–0.999996, decode 0.999935–0.999997; full prefill 0.999909–0.999971, decode 0.999882–0.999930 |
| determinism on repeated identical inputs | `test_determinism_repeated_inputs` | 3/3 runs **bit-identical**, both kinds |
| repeated-run stress | `test_repeated_run_stress` | 12 prefill + 4-step-decode cycles over prompt lengths 96 / 130 / 257: repeats **bit-identical**, DRAM allocation growth **0** bytes |
| `max_context` not a multiple of the alignment | `test_unaligned_max_context` (5000) | completes, finite, non-degenerate — linear tail std 0.5372; full tail std 0.5010 (no PCC: no tractable golden at this length) |
| advertised context — prefill + decode at the last legal slot (262143), prefill at the full 262144 | `test_full_context_prefill_and_decode[262143]` / `[262144]` | completes, finite, non-degenerate — linear tail std 0.5310/0.5303; full tail std 0.5037/0.5036 (no PCC: no tractable golden at this length) |
| full-context result independent of the internal chunking | `test_full_context_chunk_size_invariance` | 262144-token prefill under chunk 2048 vs 1024: tail PCC **1.000000**, both kinds |
| no host fallback in a measured pass | `test_no_host_fallback_in_forward` | clean for both kinds, with positive controls proving both guards fire — §6 |
<!-- /generated:capability-table -->

Paged-KV behaviour is exercised throughout rather than in one test, and the fused rewrites are
exactly where it could have broken: a single batched `paged_fill_cache` in prefill, one
`paged_fused_update_cache` at a device `current_pos` in decode, `chunked_scaled_dot_product_attention`
with a chunk offset, and `paged_scaled_dot_product_attention_decode`. `test_permuted_page_table` is
the one that catches an address/indexing bug an identity page table hides.

---

## 3. The fused graph

Full catalogue with per-rewrite evidence in [`work_log.md`](work_log.md) §3; rejected candidates
with the measurement that rejected each in §4 there. Summary:

| Kind | Rewrite |
| --- | --- |
| Dedicated op | `nlp_create_qkv_heads` / `nlp_create_qkv_heads_decode` for the QKV head split |
| Dedicated op | `nlp_concat_heads` for the prefill head merge (both attention and DeltaNet) |
| Dedicated op | `rotary_embedding_hf` for partial RoPE (10 primitive ops → 4 in prefill; decode adds two transposes for the op's expected axis order) |
| Dedicated op | `chunk_gated_delta_rule`'s **flat rank-3** contract: the op's prep kernel does the Q/K L2 norm and folds the `K**-0.5` scale, removing three head-split relayouts and two device-side L2 norms (never a host round trip) |
| Dedicated op | `chunk_gated_delta_rule(output_head_major=True)`: TILE head-major output, no ROW_MAJOR→TILE conversion of the whole activation |
| Dedicated op | `paged_fill_cache(batch_idx_tensor=…)`: one launch per cache instead of one per user |
| Dedicated op | `paged_fused_update_cache`: K and V decode updates in one launch |
| Dedicated op | `deepseek_moe_fast_reduce_nc` for the expert-axis reduction |
| Dedicated op | `ttnn.conv1d` for the depthwise causal conv in prefill, split into `CONV1D_CHANNELS`-wide calls — several times faster than the FIR form for a full 8192-channel conv over 2048 tokens (`work_log.md` §4.12 tabulates both times, generated from the probe log) |
| Graph rewrite | Shared-LHS matmul packing, ×4: attention in-projection, DeltaNet in-projection, shared expert, routed-expert gate/up |
| Graph rewrite | Router score applied to the **input** of `down_proj` instead of its output |
| Graph rewrite | SwiGLU evaluated in the sparse matmul's native 6-D layout → one expert-axis permute instead of two, and none at all when there is a single 32-token group |
| Graph rewrite | Expert group = 32 tokens, so the down projection gets the same expert skipping the gate/up projections had |
| Graph rewrite | One router per MoE call instead of one per expert group |
| Graph rewrite | One sparsity mask and one down-projection score operand per MoE call instead of one per expert group — the same hoist as the router, one level down (`work_log.md` §4.16) |
| Graph rewrite | One `repeat_interleave` over the adjacent Q/K head pair instead of two |
| Graph rewrite | `reshape + permute` for both decode head relayouts; ROW_MAJOR conv-history concat |
| Graph rewrite | Explicit `core_grid` on the three recurrent-state matmuls |
| Op merging | SwiGLU and the sigmoid attention gate folded into their binary consumer's input activations |
| Op merging | `softplus(a + dt_bias)` folded into the add |
| Op merging | The delta-rule outer product's transpose folded into `matmul(transpose_a=True)` |
| Op merging | The Q `head_k_dim ** -0.5` scale folded into the L2 norm's multiply |

### 3.1 Weight packing

All four packings are concatenations of the same weights, so the device footprint is **unchanged**.
The shared expert's single-column sigmoid router is the only member that needs padding, and the
functional decoder already stores it tile-padded on its own (`tt/moe.py` uploads it as
`[1, 1, dim, 1]` in `TILE_LAYOUT`), so both spellings hold the same bytes. `context_contract.json`'s
`footprint_change` records this; an earlier revision claimed a one-tile growth, which was an artefact
of the capacity formula omitting that padding rather than a real change.

| Packed weight | Width | Replaces |
| --- | --- | --- |
| `attn_in` | 9216 | `q_proj` (8192, query + output gate interleaved per head), `k_proj` (512), `v_proj` (512) |
| `gdn_in` | 12352 | `in_proj_qkv` (8192), `in_proj_z` (4096), `in_proj_a` (32), `in_proj_b` (32) |
| `shared_in` | 1056 | shared `gate_proj` (512), `up_proj` (512), `shared_expert_gate` (1, zero-padded to one tile) |
| `expert_gate_up` | 1024 per expert | `experts.gate_proj` (512) and `experts.up_proj` (512) |

The attention packing also **de-interleaves** `q_proj`: HF stores it as `n_heads × [query | gate]`,
so the fused loader splits those rows host-side and lays the packed weight out as
`[q | k | v | gate]` — the order `nlp_create_qkv_heads` wants, with the gate coming out as one flat
slice that never needs a head split at all.

---

## 4. Configuration chosen, and why

| Knob | Value | Basis |
| --- | --- | --- |
| `moe_group_tokens` | 32 | [`logs/ab_moe_group_tokens.txt`](logs/ab_moe_group_tokens.txt) — §5.3 |
| `rope_mode` | `"partial"` | [`logs/ab_rope_mode.txt`](logs/ab_rope_mode.txt) — §5.3 |
| head merge | `nlp_concat_heads` above `seq_len` 1, `permute + reshape` at 1 | [`logs/probe_decode_micro.txt`](logs/probe_decode_micro.txt) |
| expert `gate`/`up` | packed into one `N=1024` sparse matmul | [`logs/probe_gate_up_pack.txt`](logs/probe_gate_up_pack.txt) |
| expert reduction | `deepseek_moe_fast_reduce_nc` | [`logs/probe_router_and_reduce.txt`](logs/probe_router_and_reduce.txt) |
| conv-history tail layout | ROW_MAJOR | [`logs/probe_conv_tail.txt`](logs/probe_conv_tail.txt) |
| prefill depthwise conv | `ttnn.conv1d`, 4096 channels per call, with the FIR form as the fallback | [`logs/probe_conv1d_and_norm.txt`](logs/probe_conv1d_and_norm.txt) |
| dtypes / math fidelity | unchanged from the functional stage | out of scope for a correctness-preserving graph transform |

---

## 5. Performance

### 5.1 Method

Warmed measurements. Prefill and decode were captured in **separate** Tracy runs per layer kind
(four captures) by [`tracy/run_profiling.sh`](tracy/run_profiling.sh); the exact commands, the
`--op-support-count` requirement, the latency column and unit, and the expected report warnings are
in [`tracy/PROVENANCE.md`](tracy/PROVENANCE.md). Prefill = one warmed 2048-token pass (two warmup
passes first). Decode = 32 warmed `execute_trace` replays, so decode is measured from traced
execution.

The **before** column is the functional stage's committed capture set,
`../functional_decoder/tracy/`, produced by the same script with the same signposts, the same
iteration counts and the same weights.

The wall-clock before/after in the headline table is a stronger comparison still: both
implementations are built and timed **in one process, on one device, with the same real weights and
the same inputs**, by [`logs/bench_ab.py`](logs/bench_ab.py) →
[`logs/ab_functional_vs_fused.txt`](logs/ab_functional_vs_fused.txt).

### 5.2 Result

<!-- generated:perf-result -->
| Layer kind | Phase | device kernel time before → after | wall clock before → after | throughput before → after |
| --- | --- | --- | --- | --- |
| `linear_attention` | prefill, 2048 tokens | 337.759 → **256.504 ms** (−24.1 %) | 338.36 → **257.48 ms** (−23.9 %) | 6052.7 → **7953.9 tok/s** |
| `linear_attention` | decode, traced | 2.533 → **1.987 ms** (−21.6 %) | 2.613 → **2.063 ms** (−21.0 %) | 382.7 → **484.7 steps/s** |
| `full_attention` | prefill, 2048 tokens | 316.373 → **242.441 ms** (−23.4 %) | 316.71 → **243.49 ms** (−23.1 %) | 6466.5 → **8411.1 tok/s** |
| `full_attention` | decode, traced | 2.340 → **1.815 ms** (−22.4 %) | 2.393 → **1.829 ms** (−23.6 %) | 418.0 → **546.8 steps/s** |
<!-- /generated:perf-result -->

<!-- generated:prefill-win-split -->
**Where the prefill win comes from, honestly split.** At `moe_group_tokens=256` — the value
`tt/moe.py` uses — the fused decoder is 306.27 / 287.28 ms
([`logs/ab_moe_group_tokens.txt`](logs/ab_moe_group_tokens.txt)), so graph fusing alone accounts
for 32 of the 81 ms `linear_attention` saving and 29 of the 73 ms `full_attention` one;
the rest is the expert-group constant, which is a one-line change
the functional MoE could also take. Decode is the reverse: it runs a single 32-token group at every
setting, so the whole of its fall — 21.0–23.6 % across both layer kinds and both the device
and wall-clock measures — is graph fusing.
<!-- /generated:prefill-win-split -->

Device kernel time comes from [`tracy/perf_summary.txt`](tracy/perf_summary.txt), generated by
[`tracy/summarise_perf.py`](tracy/summarise_perf.py) from the committed `*_perf_report.csv`; wall
clock and throughput from `logs/ab_functional_vs_fused.txt`. <!-- generated:op-counts -->
Op counts moved in **opposite directions** by phase, for two different reasons. Prefill rose
because of the finer expert grouping, not the fusing; decode fell *because* of the fusing —
`moe_group_tokens` does not affect it at all (one 32-token group either way). Decode fell
(`linear_attention` 3712 → 2880 rows, `full_attention`
3424 → 2368, over 32 replays) while prefill rose (369 → 853 and 335 → 826), because a
2048-token prefill now issues 64 expert groups instead of 8. Device time fell by at least
21.6 % in every one of the four windows regardless, which is the point: launch count is not
the objective, and this stage traded prefill launches for a much larger reduction in redundant
expert FLOPs.

Where the decode fall actually comes from, per replay, from the per-op-code diff of the two
committed `decode_perf_report.csv` tables (every op code that moved, so the list is complete):

* **`linear_attention`, −26 launches per replay.** Removed: `ReshapeView` −7, `Unary` −6, `UntilizeWithUnpadding` −6, `Matmul` −5, `Tilize` −4, `Permute` −2, `TilizeWithValPadding` −2, `BinaryNg` −1, `Concat` −1, `FastReduceNC` −1, `Reduce` −1, `SparseMatmul` −1, `Untilize` −1. Added: `DeepseekMoEFastReduceNC` +1, `Transpose` +1, `Slice` +10 —
  which closes exactly on the −26. The packed gate/up matmul is the `SparseMatmul`
  term, −1 of it.
* **`full_attention`, −33 launches per replay.** Removed: `Unary` −8, `BinaryNg` −6, `UntilizeWithUnpadding` −5, `Matmul` −4, `ReshapeView` −4, `TilizeWithValPadding` −4, `Concat` −2, `PagedUpdateCache` −2, `Permute` −2, `Transpose` −2, `FastReduceNC` −1, `Reduce` −1, `SparseMatmul` −1. Added: `DeepseekMoEFastReduceNC` +1, `NLPCreateQKVHeadsDecode` +1, `PagedFusedUpdateCache` +1, `Slice` +1, `RotaryEmbeddingHf` +2, `ShardedToInterleaved` +3 —
  which closes exactly on the −33. The packed gate/up matmul is the `SparseMatmul`
  term, −1 of it.

So the fall is **relayout and elementwise elimination**, not the MoE packing: the dedicated
head-split, RoPE and cache-update ops each replace a slice/reshape/permute/transpose sequence,
and the flat rank-3 delta-rule contract removes the per-head tilize/untilize round trips. The
gate/up packing is a real launch saving but a small one by count — its value is in FLOPs and
in one fewer `UnaryOpType::FILL` of the `num_experts`-wide output (§5.4), not in the row total.
<!-- /generated:op-counts -->

Human-readable tables: `tracy/<kind>/{prefill,decode}_perf_report.txt`. Machine-readable rows:
`tracy/<kind>/{prefill,decode}_perf_report.csv` (+ `_stacked.csv/.png`). Raw provenance:
`tracy/<kind>/{prefill,decode}_ops.csv.gz` and `_tracy_run.txt`.

### 5.3 The two configuration sweeps

MoE expert-group granularity, warmed 2048-token prefill
([`logs/ab_moe_group_tokens.txt`](logs/ab_moe_group_tokens.txt)) — one sparsity entry covers one
32-token group, so the group size decides how much of the 256-expert axis the **down** projection
can skip:

<!-- generated:moe-group-sweep -->
| tokens per `sparse_matmul` call | `linear_attention` | `full_attention` |
| --- | --- | --- |
| **32** | **257.73 ms** | **243.59 ms** |
| 64 | 283.30 ms | 264.34 ms |
| 128 | 298.34 ms | 277.58 ms |
| 256 | 306.27 ms | 287.28 ms |
| 512 | 307.51 ms | 292.38 ms |
<!-- /generated:moe-group-sweep -->

Partial-RoPE lowering ([`logs/ab_rope_mode.txt`](logs/ab_rope_mode.txt)) — `"full"` is the
fewer-op graph (one op, no slice, no concat) and is exactly equivalent. One warmed comparison per
mode (which is what `run_sweeps.sh` runs) puts it ~1.5 % slower on traced decode — the window that
matters for a decoder — and separates the two prefill figures by less than the run-to-run spread, so
prefill is a tie this evidence cannot break:

<!-- generated:rope-mode-sweep -->
| mode | prefill 2048 | traced decode |
| --- | --- | --- |
| **`partial`** (4 ops in prefill, `rope_dim`-wide table) | 243.65 ms | **1.832 ms** |
| `full` (1 op, `head_dim`-wide table) | 243.44 ms | 1.856 ms |
<!-- /generated:rope-mode-sweep -->

### 5.4 `tt-perf-report` conclusions

`SLOW`-flagged rows — where neither DRAM bandwidth nor FLOP rate comes close to the roofline —
dropped sharply. Regenerated by [`tracy/summarise_slow_ops.py`](tracy/summarise_slow_ops.py) into
[`tracy/slow_ops_summary.txt`](tracy/slow_ops_summary.txt), which lists **every** group (not a
top-N), grouped by geometry, math fidelity **and** core count:

<!-- generated:slow-table -->
| Window | `SLOW` rows before | after | per traced step before → after | share of window before → after |
| --- | --- | --- | --- | --- |
| `linear_attention` prefill | 17 | **5** | n/a (one pass) | 1.4 % → **1.7 %** |
| `linear_attention` decode | 352 | **224** | 11 → **7** | 12.2 % → **9.7 %** |
| `full_attention` prefill | 16 | **5** | n/a (one pass) | 1.3 % → **1.5 %** |
| `full_attention` decode | 256 | **128** | 8 → **4** | 10.3 % → **7.8 %** |
<!-- /generated:slow-table -->

<!-- generated:slow-attribution -->
Two rewrites moved those rows, and each both removed and added: the **router hoist** turned the
functional decoder's per-256-token-group router into one call per prefill, and the **shared-LHS
packings** merged several narrow projections into one wide one that is itself `SLOW`. Counting
only removals would rank the packings first; counting what they add as well, the router hoist is the larger net win in both windows.
Launches and device time, removed − added:

| Window | router hoist | shared-LHS packings | net `SLOW` rows |
| --- | --- | --- | --- |
| `linear_attention` prefill | 8 − 1, 364 − 119 µs = **−245 µs** | 7 − 2, 3344 − 3178 µs = **−166 µs** | 12 |
| `full_attention` prefill | 8 − 1, 361 − 119 µs = **−242 µs** | 6 − 2, 2660 − 2473 µs = **−187 µs** | 11 |

The two net figures account for the absolute `SLOW`-time fall reported just below to within
0 µs for `linear_attention` — the itemisation is exhaustive, with nothing left over. Where a residual does appear it is
  the geometries that appear in
*both* summaries — the same op, timed twice — which neither rewrite removed and which therefore
belong to neither column. Only the `linear_attention` window has an absolute-fall figure below;
the `full_attention` net figures stand on their own.
<!-- /generated:slow-attribution -->

<!-- generated:slow-absolute -->
The prefill *share* rises slightly while the *count* falls by more than two thirds, because the
window itself got shorter. The absolute `SLOW` time fell as well: summing the per-group rows
`slow_ops_summary.txt` lists for `linear_attention` prefill gives 4877 µs before and 4466 µs
after (each row rounded as the summary prints it), so the absolute `SLOW` time fell 8.4 %
while the window itself fell 24.1 %.
<!-- /generated:slow-absolute -->

What the remaining rows say about where the next stage should look:

<!-- generated:moe-share -->
* **The MoE sparse matmuls are the floor.** They are 81.8 % / 81.7 % of both `linear_attention`
  and `full_attention` prefill device time, and 35.8 % / 39.2 % of the decode windows.
  Cutting further means expert-major token gathering, not another graph rewrite — §8 item 1.
  **They are absent from the `SLOW` table for a structural reason, not a good one**: the rows
  carry no numeric `nnz`, so `tt-perf-report` omits their DRAM and FLOP utilisation entirely
  (`Warning: SparseMatmulDeviceOperation rows without numeric nnz were found. Their DRAM/FLOP utilization is omitted; pass --active-experts K to model K active experts per input batch group.`)
  and the `SLOW` rule — neither metric near the roofline — has nothing to test. There are **0**
  `SLOW`-flagged `SparseMatmul` rows in any capture, fused or functional, and that says nothing
  about their efficiency. `tracy/PROVENANCE.md` records the omission; this stage measures their
  *share of device time*, which is what the claim above rests on, and leaves their roofline
  efficiency unmeasured.
* **The shared-LHS packed projections are themselves `SLOW` rows**, at 22.9–23.5 % of
  peak FLOPs on the full grid: that is roughly what HiFi4 alone predicts (4 passes), so it is a math
  fidelity and program-config question, i.e. exactly the next stage's job.
<!-- /generated:moe-share -->

<!-- generated:slow-bullets -->
* **In decode the largest `SLOW` group is `32 x 4096 x 2048`** (the output projection) at
  42.2-42.6 / 45.2-45.5 % of DRAM bandwidth — a dtype/layout target, not a core-count one.
* **`32 × 2048 × 256`** (the router) still runs on 8 cores at
  9.3-9.4 / 9.2-9.4 % of DRAM bandwidth. The recurrent-state matmuls
  (`b={32} 32 × 128 × 128`) are now one 32-core group costing
  1404 µs, where the functional decoder had 3 groups
  (1920 µs on 4 cores, 385 µs on 110 cores, 441 µs on 110 cores) totalling 2746 µs — this stage's `core_grid`
  fix moved the dominant one off 4 cores.
<!-- /generated:slow-bullets -->

<!-- generated:fill-cost -->
* **The `UnaryDeviceOperation` aggregate is almost entirely one MoE cost**, and it is the
  third-largest item in the traced decode window. Splitting it by the raw capture
  ([`tracy/summarise_fill.py`](tracy/summarise_fill.py) →
  [`tracy/fill_summary.txt`](tracy/fill_summary.txt)) attributes
  94.6 % / 99.2 % of it to
  `UnaryOpType::FILL` — `ttnn.sparse_matmul` zero-initialising its `num_experts`-wide output
  before writing the active experts' blocks. Two of `linear_attention` decode's
  3 `FILL` launches are the ones that matter, one per
  `sparse_matmul` call: 175.261 µs clearing the 2048[2048]-wide output of
  the down projection and 88.341 µs clearing the 1024[1024]-wide output of
  the packed gate/up. (Each fill's width is read from the matmul it precedes, not from its own
  reported shape, which the profiler does not always update — see the script.)
  **It is not reachable by graph fusing**: it is inside the op, its width is the `num_experts`
  output the op's contract requires, and the call count is a settled trade at its measured
  optimum rather than a floor the routing imposes — `logs/ab_moe_group_tokens.txt` runs the
  sweep, and larger groups issue *fewer* calls but were slower end to end. Removing the fill
  means never materialising a 256-wide output —
  expert-major gathering, §8 item 1.
  **Not a regression.** The functional decoder's largest fill, 173.460 µs, clears the same
  2048[2048]-wide down-projection output at essentially the same cost; what this
  stage's gate/up packing changed is the *other* end — the baseline pays two narrower fills for
  its two separate projections where this stage pays one wider. The `FILL` sub-total is
  essentially unchanged — 267.3 µs against 265.9 µs,
  +0.5 % — so the fill cost itself did not fall. What fell is the
  `UnaryDeviceOperation` aggregate it sits inside, 333.1 → 282.4 µs.
  `work_log.md` §4.13.
<!-- /generated:fill-cost -->

<!-- generated:binary-cost -->
* **`BinaryNgDeviceOperation` is the fourth-largest decode item, and it is two distinct MoE
  costs, not one.** `BinaryOpType::MUL` is 83.9 % / 95.0 % of the
  aggregate over 17 launches per replay for `linear_attention`, and the two
  largest differ in kind:
    * 69.012 µs **with SiLU folded into its input activation** — the routed
      experts' SwiGLU. Here the multiply *is* the fused form (§3.3); what is left is the
      expert-activation width, the same lever as the zero-fill above.
    * 49.754 µs with **no** folded activation, immediately before the next
      `sparse_matmul`'s zero-fill — the router-score multiply this stage moved ahead of the down
      projection (§3.2). It is a genuinely separate op, and §4.13 / §4.17 record why: the one
      ttnn op that fuses it into the reduction (`deepseek_moe_fast_reduce_nc_fused`) wants the
      gather-by-expert dispatch layout and an L1-resident activation - the same blocker §4.10
      records - rather than there being no op.
  The aggregate is nonetheless **smaller than the baseline's**: 200.0 µs against
  327.8 µs per replay, because that placement moved the
  multiply from the `hidden_size`-wide residual stream to the `moe_intermediate`-wide expert
  activation.
<!-- /generated:binary-cost -->

<!-- generated:slice-cost -->
* **`SliceDeviceOperation` costs 109–115 µs per
  traced decode step** (5.8–6.0 % of the
  window), and it is the price of the shared-LHS packings: one wide matmul, then slices to recover
  the operands, and it is not a `SLOW` row, so
  a reader looking only at the `SLOW` table would miss it. The table above ranks it sixth in `linear_attention` / fifth in `full_attention`;
  the dense `Matmul*` codes it trades against fall -114.0 to -90.6 µs/step while `Slice` rises
  +92.1 to +105.8 µs/step, so the window data alone does not settle the trade in either
  direction — the packing's own measured effect is the 709.0 → 661.4 µs/call in
  [`logs/probe_gate_up_pack.txt`](logs/probe_gate_up_pack.txt), which covers the MoE pair only.
  Its two dominant calls are the consecutive pair immediately after the
  routed-expert `sparse_matmul`, i.e. they unpack that matmul's `2·I`-wide output and are
  themselves a MoE cost.
<!-- /generated:slice-cost -->

Where the traced decode window actually goes, by op code (`tracy/<kind>/decode_perf_report.csv`,
totalled over the 32 replays and divided by them):

<!-- generated:decode-cost-ranking -->
| Rank | op code | `linear_attention` µs/step | `full_attention` µs/step |
| --- | --- | --- | --- |
| 1 | `SparseMatmulDeviceOperation active=?/256 x 32 x 2048 x 1024` | 367.2 | 368.0 |
| 2 | `SparseMatmulDeviceOperation active=?/256 x 32 x 512 x 2048` | 344.4 | 343.8 |
| 3 | `UnaryDeviceOperation` | 282.4 | 270.1 |
| 4 | `BinaryNgDeviceOperation` | 200.0 | 139.8 |
| 5 | `MatmulDeviceOperation 32 x 2048 x 12352` | 128.5 | 0.0 |
| 6 | `SliceDeviceOperation` | 115.1 | 109.3 |
<!-- /generated:decode-cost-ranking -->

`MatmulDeviceOperation 32 × 2048 × 12352` is the gated-DeltaNet packed in-projection, so it has no
`full_attention` counterpart; that layer's equivalent is `32 × 2048 × 9216`. `UnaryDeviceOperation`
and `BinaryNgDeviceOperation` are op-code aggregates rather than single ops; both have their
dominant members attributed above from the raw capture. The remainder of the `BinaryNg` aggregate is
the long tail of small elementwise launches, which `tracy/fill_summary.txt` counts but does not
itemise.

---

## 6. Runtime audit — no host fallback, no unnecessary relayout

`test_no_host_fallback_in_forward` runs one full prefill pass and one full decode pass per layer
kind under two simultaneous guards — raising stubs on `ttnn.from_torch`/`to_torch`/`as_tensor`/
`to_device`/`from_device`, and a `torch.overrides.TorchFunctionMode` that raises on **any**
dispatched `torch` operation — after first asserting that both guards actually fire. Both passes
complete for both kinds, so there is no host round trip anywhere inside them. (Scope limit, same as
the functional stage: `TorchFunctionMode` intercepts Python-level `torch` dispatch, so a host round
trip buried inside a C++ ttnn op would be caught by the `ttnn` entry-point stubs rather than by it.)

`test_no_layout_churn_in_measured_forward` additionally pins how many layout/relayout ops the
measured passes may dispatch (`to_layout`, `to_memory_config`, `sharded_to_interleaved`,
`interleaved_to_sharded`, `tilize`, `untilize`). The budgets are itemised from the shipped graph
rather than fitted, so a regression that reintroduces a per-head relayout moves the count
immediately:

<!-- generated:layout-budget -->
| Layer kind | prefill @256 | prefill @2048 | decode | what they are |
| --- | --- | --- | --- | --- |
| `linear_attention` | 12 | 12 | 1 | prefill: 4 conv ROW_MAJOR conversions (3 state buffers + the QKV stream) + 2 `sharded_to_interleaved` for the two `ttnn.conv1d` halves + 2 `to_layout` calls on their output that dispatch nothing (conv1d already returns TILE) + 3 conv-history row writebacks + **one MoE group mask per MoE call** (not per expert group — work_log §4.16). decode: the MoE group mask only |
| `full_attention` | 3 | 3 | 6 | prefill: 2 RoPE-table `to_layout` tilizes + one MoE group mask per MoE call. decode: 3 `sharded_to_interleaved` off `nlp_create_qkv_heads_decode` + 2 height-shards for the fused cache update + 1 MoE group mask |
<!-- /generated:layout-budget -->

**That budget is constant per internal prefill chunk**, not per token: `prefill_forward` loops over
`chunk_size`-token blocks and each block dispatches the same fixed set, so a 3000-token prefill pays
it twice. Both budgeted lengths are one chunk, so the two prefill columns being equal pins
*expert-group-count* independence — 8 groups at 256 tokens against 64 at 2048 — which is exactly the
regression round 22 caught. It used to scale — the MoE group mask was
rebuilt once per 32-token expert group, so the count reached 75 at 2048 tokens — until `work_log.md`
§4.16 hoisted it to one per MoE call. Review round 22 caught that the budget had not followed the
code and was tolerating dozens of extra relayouts at exactly the length every §5 figure comes from;
`test_no_layout_churn_in_measured_forward` now asserts these counts **exactly** rather than as an
upper bound, and the generator above fails if the two prefill columns ever differ.

Almost every one of those is required by an op contract: `sparse_matmul` demands a ROW_MAJOR
sparsity tensor, `paged_update_cache` demands a height-sharded input, `rms_norm` rejects
height-sharded input, `ttnn.conv1d` returns a *height-sharded* result that has to come back
interleaved before the SiLU, and the FIR fallback's shifted windows are not tile-aligned by
construction. The exceptions are the two `to_layout(..., TILE_LAYOUT)` calls on the `ttnn.conv1d`
output: `Conv1dConfig` aliases `Conv2dConfig`, whose `output_layout` already defaults to
`Layout::TILE`, and the layer passes no override — so those two calls dispatch nothing. The capture
confirms it: no tilize device op appears between each `sharded_to_interleaved` and its SiLU. They are
kept rather than deleted because the conv's output layout is a config default rather than a
documented contract, and they cost nothing on device; they are counted here because this budget
counts `ttnn` entry-point calls, not device launches.

The conv's four ROW_MAJOR conversions are the price of the causal history concatenation, which has to
happen in ROW_MAJOR: on the token axis `ttnn.concat` of TILE tensors lowers to untilize → concat →
tilize, and that tilize of the whole 8192-wide stream would be discarded immediately. Of its remaining
four ops, the 2 `sharded_to_interleaved` are `ttnn.conv1d`'s output contract and the 2 `to_layout`
calls are the no-ops described above. `ttnn.conv1d` did not take
all `conv_dim = 8192` channels in one call at any DRAM slice count measured — three distinct blockers,
reproduced in `work_log.md` §4.4, one of which is an L1 *free-space* refusal and so depends on what
else is resident — so the layer issues one call per `CONV1D_CHANNELS = 4096` block; that split point
is exactly the Q/K | V boundary, so the halves are consumed directly and no concatenation of the conv
output is needed.

**One host-call divergence from the functional decoder, stated rather than glossed.** Both decoders'
`prefill_forward`/`decode_forward` call `allocate_state` themselves if the caller never did. For the
functional decoder that path is host-free (`ttnn.zeros` only); for this one it is not — it builds the
paged-fill row indices with `ttnn.from_torch`, and for `linear_attention` prepares and probes the
`ttnn.conv1d` weights. So a *first* forward on an unallocated layer touches the host, at most once per
layer per batch. It never happens inside a captured trace or inside anything measured here, because
every measurement and every trace capture allocates explicitly first — and
`test_lazy_allocation_is_the_only_host_call` pins **both** halves: the first forward on an unallocated
layer does make host calls, the second makes none. `test_no_host_fallback_in_forward` covers the
allocated path, which is the one §5 measures.

One rewrite went the other way and is recorded as such: this stage carried `ttnn.mac` as the conv-tap
accumulator for thirteen review rounds on the belief that it was cheaper than the functional decoder's
`ttnn.addcmul`. It is not — `addcmul` dispatches a single LLK op for these shapes and dtypes, `mac` is
always two — so it was a pessimisation on the traced decode path and is reverted. `work_log.md` §4.14.

`test_fused_path_is_used` asserts that the dedicated ops the fused graph is defined by are actually
dispatched by a plain prefill and a plain decode, and — as a live negative control — that the
functional decoder dispatches none of them beyond the three the two implementations genuinely share
(`sparse_matmul`, `chunk_gated_delta_rule`, `paged_fill_cache`). The recorder is **cleared after
setup**, so its `conv1d` count is what the forward pass itself dispatched, not the probe calls
`allocate_state` makes; for `linear_attention` the count is asserted exactly (`conv_dim /
CONV1D_CHANNELS` = 2 per prefill block) rather than merely nonzero, which is what distinguishes the
`ttnn.conv1d` path from the FIR fallback.

---

## 7. Watcher

Separate run, never combined with the profiler, with its own log path (exact command in
[`watcher/CLASSIFICATION.md`](watcher/CLASSIFICATION.md)):

<!-- generated:watcher-result -->
**47 passed** in 169.94 s. The 51 324 lines of the watcher log are fully accounted for by a disjoint census summing exactly to 51 324, and a fatal-class grep (asserts, invalid NOC coordinates or addresses, CB out-of-bounds, L1/stack overflow, sanitizer, corruption, hang/deadlock) returns **0 fatal-class matches**. The log carries no stack-headroom evidence.
<!-- /generated:watcher-result -->

The run had `disabled features: None` (Ethernet checks left on).

<!-- generated:watcher-subset -->
The subset is a chosen one, not an exhaustive sweep: it covers every in-place cache/state
writer, trace replay and fused-only rewrite that this stage introduced or changed, and it is
listed from the run's own console log rather than described. Tests that drive prefill or decode
purely to check numerics (the PCC ladders, the weight-source cases, the full-context cases) are
deliberately out — they exercise no memory pattern the tests below do not. The generator
asserts by name that the load-bearing ones are present, so widening the filter cannot silently
drop them. 19 test functions, 47 cases, with 50 cases deselected:

* `test_batch_smaller_than_allocated_state`
* `test_batched_decode_ragged_positions`
* `test_batched_paged_fill_is_one_launch_per_cache`
* `test_batched_prefill_decode_pcc`
* `test_decode_batch_above_head_split_limit`
* `test_decode_pcc`
* `test_determinism_repeated_inputs`
* `test_forward_with_poisoned_free_pool`
* `test_fused_matches_functional`
* `test_fused_path_is_used`
* `test_lazy_allocation_is_the_only_host_call`
* `test_no_layout_churn_in_measured_forward`
* `test_permuted_page_table`
* `test_prefill_continuation`
* `test_repeated_prefill_at_a_masked_chunk_length`
* `test_repeated_run_stress`
* `test_rope_mode_equivalence`
* `test_traced_decode_pcc`
* `test_unaligned_max_context`
<!-- /generated:watcher-subset -->

[`watcher/census.py`](watcher/census.py) reproduces both the census and the fatal-class
grep into [`watcher/census_summary.txt`](watcher/census_summary.txt). Artifacts:
[`watcher/watcher_log.txt`](watcher/watcher_log.txt),
[`watcher/kernel_names.txt`](watcher/kernel_names.txt), console log
[`logs/watcher_pytest.txt`](logs/watcher_pytest.txt). There are no suspected false positives.

On stack usage, whichever way the run lands: watcher prints a watermark only for dumps where the
firmware happened to record one, so whether this log carries stack-headroom evidence varies between
runs of the same subset. The generated sentence above states what **this** log has, read from
`watcher/census_summary.txt`, and `census.py` reports an absence explicitly rather than letting it
read as "no overflow". Either way the conclusion is the same: stack overflow would also surface
through the fatal-class grep, which is clean.

---

## 8. Known limitations

<!-- generated:moe-floor-limitation -->
1. **The MoE remains the floor.** After fusing, the two sparse expert matmuls are
   81.8 % / 81.7 % of the two prefill windows and 35.8 % / 39.2 % of the traced decode
   windows — a clear majority in prefill, the largest single item but not a majority in decode.
   Cutting further needs expert-major token gathering — `unified_routed_expert_ffn`, `moe_compute`
   and `moe_gpt` all want that layout — which is a change to the routing algorithm, not to the op
   graph, and multi-device in every in-tree instance. `work_log.md` §4.10. §5.4 also records why
   their absence from the `SLOW` table is not evidence that they are efficient.
<!-- /generated:moe-floor-limitation -->
2. **`ttnn.topk` is single-core here.** Its multi-core path needs a power-of-two reduced dim of at
   least 8192; Ornith's is 256. Mitigated by hoisting the router out of the expert-group loop
   (64 calls → 1 per 2048-token prefill), not removed.
3. **`ttnn.conv1d` serves the depthwise causal conv only in `CONV1D_CHANNELS`-wide slices, and only
   for `(batch, block length)` pairs that fit L1 — mostly a per-bank allocation limit rather than a
   circular-buffer one; §2 has the measured split.** It cannot take all 8192
   channels at once at any DRAM slice count measured — one of the three blockers is an L1 free-space
   refusal, so it depends on what else is resident — and the layer therefore keeps the FIR form as a
   fallback that `allocate_state` selects by *executing* each candidate shape once at setup. Coverage
   falls as the batch rises — §2's generated table has the per-batch figures — so large-batch prefill
   is correct but not accelerated. `work_log.md` §3.1 has the
   measurement and §4.4 the blockers.

   The fallback is chosen at setup, and **only** at setup: `_conv1d_halves` calls `ttnn.conv1d`
   unguarded, so a `(batch, length)` that passes the setup probe and then fails inside a forward
   raises rather than falling back. That is not purely theoretical, but it is narrower than the whole
   refusal class, and the bound is batch-dependent — the probe runs at `allocate_state` time, when the
   large prefill activations are not yet resident, so only refusals that *could* have gone either way
   can differ between the probe and a forward pass:

<!-- generated:conv1d-forward-risk -->
Of the 28 bank-allocation refusals, 15 are pressure-dependent —
the per-bank share would fit an empty bank, so whether they refuse depends on what else is
resident — and 13 are hard overflows no amount of free space would satisfy. Only the
first kind can behave differently between the setup probe and a forward pass, and the split is
batch-dependent (pressure-dependent / hard): batch 4: 3 / 0, batch 8: 9 / 1, batch 32: 3 / 12. The risk is therefore **largest at batch
8**, where 9 of 10 are the kind that can differ, not at the batch where
coverage is worst.
<!-- /generated:conv1d-forward-risk --> No delivered test has provoked it (the
   probe executes each candidate shape once, which is the same op on the same shapes), and an
   in-forward `try`/`except` around a device launch is not obviously safe for buffer ownership, so this
   is recorded rather than papered over.
4. **`paged_fused_update_cache` needs `2 × batch` cores.** Above that the layer falls back to two
   separate `paged_update_cache` launches, which is what the functional decoder always did — so the
   launch count changes, not the supported batch.
5. **The dedicated decode head split stops at batch 32**, because
   `nlp_create_qkv_heads_decode` asserts `batch <= 32`. Above it the layer falls back to the
   functional decoder's generic slice/reshape/permute split, so the *supported* batch is unchanged —
   only the op used to reach it is. `test_decode_batch_above_head_split_limit` runs it against an HF
   golden at **two** batches, which is what separates the two fallbacks: batch 40 is past the
   head-split limit but still inside `2 × batch ≤ 110`, so the cache update is still the fused one;
   batch 56 is past both (`2 × 56 = 112 > 110`), so it also takes the two-launch cache update. The
   test asserts which of the two is active at each batch, so a change that silently stopped fusing
   the cache update at batch 40 would fail rather than pass quietly.
6. **The flat `chunk_gated_delta_rule` contract couples `linear_attention` prefill to two things
   the functional decoder's rank-4 call did not.** First, the op infers the key head dim *from the
   value head dim* on this path, so a config with `linear_key_head_dim != linear_value_head_dim`
   would be computed with the wrong head count silently; `_chunk_delta_rule` and `_gdn_decode` both
   raise instead. Second, the flat path is legal only on the op's phased branch, which
   `chunk_gated_delta_rule.cpp` selects from the **`QWEN_GDN_PHASED` environment variable**, read per
   call: with `QWEN_GDN_PHASED=0` every fused `linear_attention` prefill raises, where the functional
   decoder still runs. Neither is reachable for Ornith at its shipped config and default environment,
   and both are guarded or documented rather than silent — but both are narrower than the baseline.

   Two constructor validations are narrower for the same reason, and are recorded in
   `context_contract.json` under `fused_decoder.construction_validation_delta`:
   `prefill_chunk % page_block_size == 0` (the functional decoder validates only
   `prefill_chunk % PREFILL_ALIGN`, and would silently write a chunk at the wrong offset inside its
   page), and the RoPE rotated width having to be 32 or a multiple of 64, which is
   `rotary_embedding_hf`'s own requirement. The shipped defaults satisfy both.
7. **The conv path is selected by probing, so the *bits* can depend on setup order.**
   `allocate_state` decides per `(batch, block length)` whether `ttnn.conv1d` fits L1 by executing it
   once and catching the refusal (§2). L1 free space at that moment depends on what else is resident,
   so the same layer with the same weights and the same input can take the `ttnn.conv1d` path in one
   process and the FIR path in another. Both clear the PCC bar and §2's coverage table records the
   split, but they are not bit-identical to each other, and `test_determinism_repeated_inputs`
   repeats within one process after one `allocate_state`, so it does not cover this.
8. **A first `decode_forward` on an unallocated `linear_attention` layer pays the whole prefill
   conv-weight probe sweep** — 16 host `prepare_conv_weights` calls and a probe tensor that reaches
   ~0.5 GB at batch 32 — even though decode never uses `ttnn.conv1d`. The probes are freed in a
   `finally`, so it is a transient spike rather than a leak, and allocating explicitly (as every
   measurement and trace capture here does) avoids it entirely.
9. **Precision and matmul-geometry tuning are deliberately untouched.** The dense matmuls still run
   HiFi4 with `fp32_dest_acc`; §5.4 hands the current `SLOW` list to the next stage.
10. **Prefill is not traced.** Only decode is captured and replayed, as in the functional stage.
11. **`rope_mode="full"` stores K in the cache under a head-dim permutation**, so a cache written by a
    `"full"`-mode layer is not interchangeable with one written by a `"partial"`-mode layer or by the
    functional decoder. `attach_kv_cache` does not and cannot check this. It does not affect the
    shipped configuration — the default is `"partial"` and `test_rope_mode_equivalence` builds each
    mode its own cache — but a future stage sharing caches across layers should treat `rope_mode` as
    part of the cache format. Relatedly, `from_state_dict(dtype=...)` does not reach the conv weights:
    `_conv1d_host_weights` and `Conv1dConfig` are `bfloat16` regardless, so at any other dtype the
    conv1d and FIR paths would carry different weight precision.
