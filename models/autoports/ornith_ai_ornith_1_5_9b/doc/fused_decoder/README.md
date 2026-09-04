# Ornith-1.5-9B fused decoder

**Fused-decoder validation is complete: 85 short-suite tests, nine native/long
checks, 23 Watcher tests, and four profiler captures pass. Independent
[stage review](STAGE_REVIEW.md) returns `clean-pass`. Local checkpoint details
are recorded in [work_log.md](work_log.md).**

This stage fuses the existing decoder graph for `ornith-ai/Ornith-1.5-9B`, pinned
HF revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. It preserves the
[functional decoder contract](../functional_decoder/README.md) and
[model instructions](../../AGENTS.md): both layer kinds, batch 1–32, arbitrary
valid logical prompt lengths, native context 262144, paged cache continuity,
request-lane isolation, and device-only forward execution. Acceptance requires
on-device equivalence at PCC >= 0.995, no material accuracy regression, and a
faster measured graph. Applicable dedicated operators, structural rewrites, and
adjacent-operation folds must be investigated, including supported adaptations
of initially failing operators. Final performance must also be compared with
the best coherent candidates, rather than only the original functional graph.

All timings here use **one Blackhole chip on physical P300c boards**, a 1×1 mesh.
`p150` denotes that single-chip topology; these are not P150-board measurements.
Generic per-device tuning, datatype sweeps, multichip, full-model, and serving
work belong to later stages. The pinned Ornith-1.0-35B implementation supplied
source examples only; none of its accuracy or timing claims is 9B evidence.

## Runtime and persistent-state contract

[FusedDecoder](../../tt/fused_decoder.py) inherits setup/orchestration helpers
from [FunctionalDecoder](../../tt/functional_decoder.py), and implements its own
block and both mixers' prefill/decode computation. The public API is unchanged:

- `from_state_dict(..., hf_config, layer_idx, mesh_device, max_context=None,
  page_block_size=64, prefill_chunk=2048, dtype=ttnn.bfloat16)` prepares weights,
  packed projections, rotary tables, constants, and convolution metadata.
- `allocate_kv_cache(num_blocks)` / `attach_kv_cache(k_cache, v_cache)` bind full
  attention's paged cache; `allocate_state(batch_size)` prepares stable request
  state and all physical-chunk convolution weights before forward or capture.
- `prefill_forward(x, start_pos=0, page_table=None, chunk_size=None)` accepts
  BF16/TILE/DRAM `[B,T,4096]`. Internal physical chunks are multiples of 128;
  the returned tensor retains the logical T. Batched prefill shares T/start.
- `decode_forward(x, current_pos=None, rot_idxs=None, page_table=None)` accepts
  `[B,1,4096]`. Full attention consumes device INT32 ROW_MAJOR `[B]` positions,
  UINT32 ROW_MAJOR `[1,B]` rotary indices, and INT32 ROW_MAJOR
  `[B,pages_per_user]` page tables. Callers provide valid positions and disjoint
  physical page ownership. Decode performs no position readback.

Shapes come from this checkpoint: hidden 4096, dense MLP 12288, 24 linear and
8 full-attention layers. Representative real-weight tests use layers 0 and 3.
Full attention has 16 Q heads, 4 KV heads, head width 256, and partial rotary
width 64. Linear attention has 16 key heads and 32 value heads, both width 128,
and a four-tap causal depthwise convolution.

| Tensor / computation | Retained policy |
| --- | --- |
| Projection weights and activations | BF16; fused setup rejects another weight/activation dtype |
| Full-attention KV cache | BF16/TILE, `[physical_pages,4,64,256]`; no cache-basis permutation selected |
| Page allocation | Page size 64; `num_blocks_for_context` rounds page-table width to 32 entries to cover the SDPA rounded read window |
| Linear recurrent state | FP32/TILE `[B,32,128,128]`, updated in place |
| Convolution history | Three persistent BF16/TILE buffers `[B,1,8192]`; only real logical tokens enter history |
| Linear decay/beta and recurrence | FP32 with explicit BF16 projection/sigmoid/scaling rounding boundaries retained |
| Generic compute config | HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=False` |
| Prefill SDPA | Existing separate HiFi2 config, exact math, FP32 destination; full device grid, Q/K chunks up to 64, reduced to satisfy continuation alignment |
| Decode SDPA | Existing operator-default compute policy; explicit 8×8 program grid, Q chunk 32, K chunk 64, `exp_approx_mode=False`; this call does not pass the generic HiFi4 config |
| Dedicated ops without a compute-config argument | Retain their operator implementation/configuration; the generic HiFi4 setting is not a claim that every dedicated kernel uses HiFi4 |

Logical padding keeps beta and decay exponent zero, so it cannot advance linear
state. Real convolution-history rows exclude padding. State reset/copy preserves
device addresses for trace replay. Unaligned full-attention continuation uses
the inherited device-decode prefix adapter before aligned chunked prefill.
Weights and convolution preparation are outside measured forward execution.

## Current selected graph

The table describes the runtime validated by `final_short_v2` and `final_long_v2`,
including Q/V cast folds, the rounded K unary chain, small-prefill L1 norm input,
and the selected whole-head native outer-product transpose. Both runs record
the current runtime SHA256:
`18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d`.

| Boundary | Selected operations and necessary movement |
| --- | --- |
| Shared block | Entry RMSNorm; mixer; residual add; RMSNorm; dense SwiGLU; down projection; final residual add |
| Full projections | Setup packs Q/K/V/gate into one width-10240 projection; prefill uses dedicated QKV/head split; decode slices QKV directly into L1 and uses `nlp_create_qkv_heads_decode` |
| Full prefill | Q/K RMSNorm; dedicated rotate-half RoPE on 64 dimensions plus unchanged 192-dimension tail; setup-tiled trig tables; batched K/V `paged_fill_cache`; chunked paged SDPA; `nlp_concat_heads`; sigmoid folded into output-gating multiply |
| Full decode | Q/K move to interleaved DRAM for RMSNorm; partial64 slices feed native sharded `rotary_embedding_hf`; shared trig tensors and unchanged tails use the same filled rectangular grid; K moves to cores disjoint from V for `paged_fused_update_cache`; paged SDPA output reshapes directly to the gated projection |
| Arbitrary decode batch | Native HF RoPE executes a filled rectangle. Batches without a suitable one-user-per-core rectangle group users per core; Q then returns to interleaved storage for SDPA. V retains its original head-split shards |
| Linear projections | Setup packs QKV/A/B into width 8256; Z has a separate width-4096 projection on the full device grid. Decode Z has a genuine matmul SiLU epilogue; prefill produces raw BF16 Z on the same grid |
| Linear prefill convolution | One row-major history/input join; eight 1024-channel depthwise Conv1d calls with prepared weights, kernel 4, stride 1, padding 0, height sharding and `act_block_h_override=32`; BF16 convolution output precedes standalone SiLU; concatenate directly into Q/K/V fields; write the three real tail rows back |
| Linear prefill normalization/core | Preserve separate functional rank-4 Q/K RMSNorm and BF16 scale; pass flat V and pre-normalized Q/K into `chunk_gated_delta_rule(use_qk_l2norm=False, output_head_major=True)`. Q/K norm results stay in L1 when `B * physical_T <= 512`, otherwise DRAM. Existing subbatch limits preserve state-row ownership |
| Linear prefill output | FP32 head-major core plus raw Z feed `kda.sigmoid_gated_rms_norm`; multiply its sigmoid-gated output by Z to obtain SiLU gating; output projection |
| Linear decode convolution/norm | Retain functional BF16 multiply/three-addcmul FIR and SiLU; one joint Q/K RMSNorm before repeated value-head rows; split the normalized Q/K fields |
| Linear decode gates | Beta unary chain retains sigmoid → BF16 rounding → FP32 output; A enters FP32 bias-add directly; standalone FP32 softplus and decay scale remain |
| Linear decode recurrence | Q multiply by exact 1/128 emits FP32 directly; K unary chain uses coefficient 181/2048 (the original BF16-rounded 1/sqrt(128)) and an explicit BF16-product round-trip; V enters FP32 subtract directly. Fold EXP into in-place state decay; outer-product `transpose_a=True` uses `MatmulMultiCoreReuseProgramConfig` with per-core M/N=4/4 tiles, K block=1 tile and output subblock=1×4, so each work block contains a complete head. Preserve in-place FP32 state add/read |
| Linear decode output | Ordinary output RMSNorm plus head merge; multiply by already-activated Z; output projection |
| Dense MLP | Prefill packs gate/up into width 24576, then splits; decode retains separate gate/up linears. Both fold SiLU into BF16 multiply |

The remaining Q/K DRAM norm handoff, K disjoint-grid move, convolution BF16 pack
before SiLU, and residual add have concrete operator or numerical contracts.
They are not dismissed solely because a first attempt failed. Their alternatives
and measurements are in the ledger below. The first profiler audit found that
the default matmul program materialized the transpose despite `transpose_a=True`.
The supported whole-head reuse adaptation is now selected after corrected matched
trace measurements and complete short/long correctness reruns. The current
profiler confirms 49 linear decode operations per replay, down from 50 before
the actual transpose fold. Ordinary GDN decode head concatenation
also passed its adaptation but was slower at 1.509661 ms and was rejected.

## Evidence status and measured comparisons

The original functional implementation is the paired comparator. Most isolated
experiments inherit [fusion_baseline.py](../../tests/fusion_baseline.py), a frozen
experimental parent containing accepted early attention changes. It is distinct
from the unfused comparator. Source snapshots identify each run even when later
formatting or runtime integration changes current file hashes.

| Evidence | Recorded result | Scope |
| --- | --- | --- |
| [selected_matrix](logs/selected_matrix.log.gz), [provenance](logs/selected_matrix.provenance.json) | 69 passed, 13 deselected | `conv_final_combination` candidate; nonlong contract matrix, excluding perf |
| [integrated_pair_v1](logs/integrated_pair_v1.log.gz), [provenance](logs/integrated_pair_v1.provenance.json) | 8 passed, 2 deselected | Integrated runtime before the latest cast/L1 changes; two layer pairs plus six raw linear-core probes |
| [combined_cast_v1](logs/combined_cast_v1.log.gz), [all_cast_v1](logs/all_cast_v1.log.gz) | One paired linear test passes in each | Isolated Q/V folds and Q/V plus rounded K chain; subsequently covered by the final integrated suites |
| [short_norm_handoff_v2](logs/short_norm_handoff_v2.log.gz) | One matched test passes with bit-identical outputs | B1, physical T128; L1 consumer versus two DRAM-handoff controls |
| [final_short](logs/final_short.log.gz), [provenance](logs/final_short.provenance.json) | 84 passed, 9 deselected in 123.67 s | Historical runtime before the explicit native transpose |
| [final_short_v2](logs/final_short_v2.log.gz), [provenance](logs/final_short_v2.provenance.json) | 85 passed, 9 deselected in 150.23 s | Current runtime contract matrix, paired timings, six raw-core probes, and corrected 31×256 matched outer-program comparison |
| [final_long](logs/final_long.log.gz), [provenance](logs/final_long.provenance.json) | 9 passed, 73 deselected in 172.40 s | Historical pre-transpose runtime |
| [final_long_v2](logs/final_long_v2.log.gz), [provenance](logs/final_long_v2.provenance.json) | 9 passed, 73 deselected in 169.40 s | Current runtime: both kinds at 262143/262144, chunk invariance, 8001-token HF comparison, and exact-cache final-position oracle |
| [Historical Watcher](logs/watcher_final.log.gz), [provenance](logs/watcher_final.provenance.json) | 23 passed, 59 deselected, 86.03 s | Pre-transpose runtime; separate process without profiler; no Watcher corruption or invalid NoC diagnostics |
| [Current Watcher](logs/watcher_final_v2.log.gz), [audit](watcher_audit.json), [provenance](logs/watcher_final_v2.provenance.json) | 23 passed, 59 deselected in 44.57 s | Current runtime; profiler unset, 92 device-check messages, no matching Watcher corruption or invalid NoC diagnostics |
| [Historical profiler](performance_v1.json) | Four captures pass HF PCC and coverage checks | Pre-transpose runtime; preserved `v1_` CSV/text/ops artifacts |
| [Current profiler](performance.json), `profile_final_*_v2` | All four captures pass HF PCC and complete-coverage checks | Current runtime; linear decode 49 operations per replay, full decode 36; exact reports below |
| [Independent stage review](STAGE_REVIEW.md) | **clean-pass** | No required work; final runtime, artifacts, and all material fusion decisions reviewed |

Paired timings below are synchronized wall-clock medians at **B1, prefill 2048,
decode position 2048**. Seven windows are run; the first two are discarded.
Each decode window contains 32 trace replays. State restoration and transfers
are outside the timed windows. They are layer timings, not generation throughput.

| Recorded graph / layer | Functional → tested prefill ms | Functional → tested trace ms | Minimum paired output PCC |
| --- | ---: | ---: | ---: |
| Current runtime, linear 0 | 40.662267 → 26.180386 | 1.623890 → 1.462064 | 0.99989795 |
| Current runtime, full 3 | 35.449361 → 23.498724 | 1.512215 → 1.263523 | 0.99996715 |
| Q/V cast candidate, linear 0 | 40.637475 → 26.167327 | 1.624063 → 1.472689 | 0.99989795 |
| Q/V/K cast candidate, linear 0 | 40.643788 → 26.169420 | 1.623641 → 1.461613 | 0.99989795 |

These rows come from `final_short_v2`, `combined_cast_v1`, and `all_cast_v1`.
The current runtime reduces linear/full prefill by 35.62%/33.71% and traced
decode by 9.97%/16.45% against the paired functional comparator. Submicrosecond
differences between separate candidate and final runs do not establish which
outer-product program is faster.

The controlled outer-product comparison alternates the preserved default-program
control (`DefaultOuterGDN`, logged as `runtime`) and the selected native-reuse
runtime over 31 measured windows × 256 replays, after two warm windows. In
`final_short_v2`, the selected runtime wins 29/31 windows; the median paired
difference is **−0.252512 µs** per replay, with separate medians
**1.461468797 → 1.461143297 ms**. The earlier
[corrected matched v3](logs/reuse_outer_matched_v3.log.gz) independently wins
25/31 windows: mean −0.248252 µs, sample standard error 0.040963 µs, and paired
median −0.250328 µs. Eager/stress PCC is effectively 1; exact eager/replay checks
before and after timing and recurrent/conv-state comparisons pass. Selection
uses these repeated within-run controls, rather than a roughly 0.5 µs difference
between separate candidate runs. The gain is small and specific to this measured
layer workload.

[candidate_measurements.md](candidate_measurements.md) and
[candidate_measurements.json](candidate_measurements.json) retain paired
baseline/candidate values and samples. If their generated inventory predates a
new log, that log and its provenance are authoritative until regeneration.

The short-norm experiment uses 30 runs with five discarded, B1/T128:
L1 → DRAM 2.493768 ms, direct DRAM 2.499128 ms, retained L1 input 2.465545 ms.
Its bit-identical outputs support the selected small-prefill handoff; these
numbers are not directly comparable with the T2048 table. Moving Conv1d SiLU
after field concatenation passes the pair but measures 26.254829 ms against
the earlier coherent convolution result 26.194263 ms; it is not selected.
See [post-concat probe](logs/post_concat_silu_v1.log.gz) and
[coherent convolution pair](logs/conv_final_combination_v1.log.gz).

The inherited standalone **prefill perf test uses 2048 tokens**. Only its
**decode perf test uses a 128-token prompt and decode position 128**, unlike
the paired position-2048 regime. Prefill has two warm calls; traced decode has
compile/capture plus four warm replays. The default measured decode count is 32,
overridable by `ORNITH_PERF_DECODE_ITERS`; profiler captures must record their
actual count. Full attention rewrites one fixed position while linear recurrence
advances every replay. The measured outputs are checked against the corresponding
HF states. Profiler `Device Time` sums exclude dispatch gaps and must be reported
separately from wall timings. The functional-stage profiler baselines remain
[in their own report](../functional_decoder/README.md#warmed-performance).

The real-weight HF checks below compare [baseline.log](logs/baseline.log.gz) with
[final_short_v2.log](logs/final_short_v2.log.gz), using the same test inputs and real
checkpoint weights. They are HF-versus-TT PCC, distinct from the higher
functional-versus-fused PCC in the paired table.

| HF comparison | Functional baseline PCC | Final fused PCC |
| --- | ---: | ---: |
| Linear 0, real-weight prefill 300 | 0.999366 | 0.999387 |
| Linear 0, decode after 300 | 0.999433 | 0.999427 |
| Full 3, real-weight prefill 300 | 0.999459 | 0.999463 |
| Full 3, decode after 300 | 0.998906 | 0.998911 |
| Linear 0, measured prefill 2048 | 0.99948959 | 0.99951420 |
| Linear 0, measured 32-step recurrent trace after 128 | 0.99969425 | 0.99969729 |
| Full 3, measured prefill 2048 | 0.99949539 | 0.99949588 |
| Full 3, measured fixed-position trace at 128 | 0.99902570 | 0.99889471 |

All clear the retained 0.995 bar. Small changes occur in both directions;
the full-attention measured decode PCC decreases by 0.00013099 and remains
0.99889471. The final per-user, state, continuity, long-context, and paired
checks provide additional coverage beyond these aggregate sanity comparisons.

## Signposted profiler evidence

All four captures use the current runtime SHA256 recorded above, B1, prefill
T2048 or a 128-token prompt with decode position128. The machine-readable summary
records the resulting attention-context length 129. The original captures remain
historical evidence in [performance_v1.json](performance_v1.json) and `v1_` report
artifacts; current results are in [performance.json](performance.json).
Each decode capture measures four replays after four warm replays; all operation
IDs match the warm template with no dropped markers. Device Time is the sum of
kernel durations, excluding dispatch gaps. Compare these rows only with the
matching functional-stage profiler regime, not the paired wall-clock table.

| Layer / mode | Functional → fused kernel ms | Ops per forward | Measured-output HF PCC before → after | CSV / text / run provenance |
| --- | ---: | ---: | ---: | --- |
| linear_attention / prefill | 40.075804 → 25.647288 | 96 → 120 | 0.99948959 → 0.99951420 | [CSV](tracy/linear_attention/prefill_perf_report.csv), [text](tracy/linear_attention/prefill_perf_report.txt), [run](logs/profile_final_linear_attention_prefill_v2.provenance.json) |
| linear_attention / decode | 1.609667 → 1.441265 | 78 → 49 | 0.99980134 → 0.99976799 | [CSV](tracy/linear_attention/decode_perf_report.csv), [text](tracy/linear_attention/decode_perf_report.txt), [run](logs/profile_final_linear_attention_decode_v2.provenance.json) |
| full_attention / prefill | 35.104062 → 23.002021 | 66 → 33 | 0.99949539 → 0.99949588 | [CSV](tracy/full_attention/prefill_perf_report.csv), [text](tracy/full_attention/prefill_perf_report.txt), [run](logs/profile_final_full_attention_prefill_v2.provenance.json) |
| full_attention / decode | 1.422320 → 1.209381 | 69 → 36 | 0.99902570 → 0.99889471 | [CSV](tracy/full_attention/decode_perf_report.csv), [text](tracy/full_attention/decode_perf_report.txt), [run](logs/profile_final_full_attention_decode_v2.provenance.json) |

Linear prefill has more kernels because ordinary depthwise convolution groups
replace the expensive BF16 FIR sequence while retaining its rounding boundary.
It is faster despite the higher count. Native outer-product transpose removes
one linear-decode operation relative to the preserved 50-operation v1 graph.
The small difference between independent v1/v2 kernel-duration sums is not the
selection test; the repeated alternating wall-time control above establishes
the transpose gain. The exact remaining movements and operator contracts are
recorded in the pattern ledger. These are decoder-layer results and do not
claim full-model throughput. Historical CSV/text/ops evidence remains under
the `v1_` prefix, including [linear decode CSV](tracy/linear_attention/v1_decode_perf_report.csv),
[text](tracy/linear_attention/v1_decode_perf_report.txt), and
[complete ops](tracy/linear_attention/v1_decode_ops.csv.gz).

Profiler reproduction (run serially, with Watcher unset):

```bash
python -m tracy -r -p -v --check-exit-code --web-app-port 18940 \
  -o models/autoports/ornith_ai_ornith_1_5_9b/doc/fused_decoder/tracy/linear_attention/raw/prefill \
  -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py::test_perf_prefill \
  -k linear_attention -v -s
```

The four current `_v2` run provenance links give exact output directories and
selectors. Each records the same selected runtime hash and Watcher unset.
Decode selects `test_perf_decode_traced` with `ORNITH_PERF_DECODE_ITERS=4`.
Use fresh output directories for reproduction; preserved raw captures remain
local, while complete ops CSVs are stored as `*_ops.csv.gz`.

## Correctness coverage and test-path guard

[test_fused_decoder.py](../../tests/test_fused_decoder.py) reuses the prior
contract tests and replaces the shared harness factory with the selected fused
class. Its autouse fixture records construction and checks the exact class;
for the runtime it poisons the functional `_block`, `_attention_prefill`,
`_attention_decode`, `_gdn_prefill`, and `_gdn_decode` methods. Any such fallback
fails. Candidate inheritance has a narrower block guard; a candidate result
therefore must not be presented as the final runtime's stronger guard result.

The contract covers real and synthetic weights, finite outputs, per-user HF PCC,
B1/4/32 prefill/decode and trace, B4/13 ragged full-attention positions, shuffled
pages, continuation, repeated identical state, changed page-table contents in
one trace, poisoned allocator contents, and host-fallback guards. Torch and
host/device transfer guards are positively tested before guarding the forward.
Not every integer batch is a separate device test; batch 1–32 remains the API
contract, with the irregular B13 case exercising grouped rotary geometry.

[test_fusion_equivalence.py](../../tests/test_fusion_equivalence.py) checks
functional-versus-fused prefill, eager decode, one restored-state replay, and
the 32-replay stress output at PCC >= 0.995. Eager/replay outputs must be
bit-identical from identical state. Recurrent/convolution or K/V cache state is
also compared. Permuting experimental RoPE candidates expose the inverse cache
permutation for the K comparison; production cache ordering stays unchanged.
Six raw linear-core probes use logical prompt lengths 1/2/3/127/2047/2049 and
32 distinct decode inputs each, checking PCC and core amplitude before output
normalization can hide an error. All pass on the final source: minimum core
PCC 0.99994460, core norm-ratio range 0.99783322–1.00369780, and minimum
recurrent-state PCC 0.99998375 in `final_short_v2`.

Native full-prefill checks establish completed, finite, nonconstant output and
usable decode state at the advertised limit, plus 2048-versus-1024 chunk-tail
agreement. They are not full-length HF-prefill oracles. The separate final-position
oracle uses real layer-3 weights and exact-shape permuted BF16 historical-cache
fixtures at position 262143. Full HF prefill/decode parity is tested through
8001 tokens. These checks pass on the current fused source in `final_long_v2`:

| Final native/long check | Linear 0 | Full 3 |
| --- | --- | --- |
| Native prefill at 262143 and 262144 | Both complete; finite/nonconstant tails | Both complete; finite/nonconstant tails |
| Traced decode after actual prefill, position 262143 | Eager/replay PCC 1.00000000 | Eager/replay PCC 1.00000000 |
| 262144-token prefill, chunk 2048 versus 1024, final 128 rows | PCC 1.000000 | PCC 1.000000 |
| 8001-token HF prefill | PCC 0.999426 | PCC 0.999528 |
| HF decode after 8001 | PCC 0.999315 | PCC 0.999745 |
| Exact-cache final-position HF/traced decode | Not applicable | PCC 0.99965014 |

The final short suite additionally checks unaligned continuation through a
4096-token cache capacity with split 63, HF PCC 0.99951119. Native prefill at
262144 is a full-capacity prefill check; decode occurs after the 262143-token
case at the last legal position. No million-token or full-length HF-prefill
claim follows from these results. The `long` marker is registered without
automatic deselection; a plain fused suite invocation includes its long cases.

## Candidate and rejection ledger

[patterns.md](patterns.md) is the comprehensive pattern-by-pattern inventory,
including applicable rewrites, supported retries, measured rejection reasons,
source-only exclusions, and explicitly unmeasured optional variants.
[work_log.md](work_log.md) records investigation history.
[AUTODEBUG_attention.md](AUTODEBUG_attention.md),
[AUTODEBUG_linear.md](AUTODEBUG_linear.md), and
[AUTOFIX_linear.md](AUTOFIX_linear.md) give exact operator contracts and
localization evidence. [The trace-allocation diagnosis](AUTODEBUG_transpose_trace.md)
and [verified repair](AUTOFIX_transpose_trace.md) preserve the failing matched
trace, positive corruption control, setup-order repair, and passing reruns. Early "pending" sections in chronological notes are not
newer results; consult their later entries and the associated completed logs.

| Family / candidate source | Selected result and material rejected alternatives |
| --- | --- |
| [Initial projection/MLP candidates](../../tests/fusion_candidates.py) | Full Q/K/V/gate packing and fused binary activation selected. MLP packing selected only for prefill; packed decode loses to separate projections |
| [Attention candidates](../../tests/attention_fusion_candidates.py) | Native partial64 sharded HF decode plus direct QKV slice-to-L1 selected. Full-width rotary, joint QK, fused Llama QK with setup basis permutation, direct destination-K grid, dedicated decode concat with both memory handoffs, and HF-prefill rotary were implemented and measured; none replaced the selected graph |
| [Linear candidates](../../tests/linear_fusion_candidates.py) | Flat V/head-major core with functional rank-4 Q/K normalization selected. Raw flat Q/K fails B32 HF PCC 0.99420974. KDA fused causal conv fails B32 HF PCC 0.99315313; isolated hybrid convolution still fails 0.99257131. Ordinary Conv1d was retried at legal channel groups and selected at 1024 |
| [FIR movement candidates](../../tests/fir_fusion_candidates.py) | Shared row-major input/history and real-tail writes evaluated. Ordinary Conv1d wins prefill; decode retains the functional BF16 FIR ordering |
| [MLP epilogue candidates](../../tests/mlp_fusion_candidates.py) | True explicit SiLU epilogues use matched profiler-derived programs and pass, but are slower than the identical-program unfused controls. Default `activation='silu'` without a core grid/program had merely dispatched a separate unary |
| [Joint prefill / Z candidates](../../tests/gdn_joint_prefill_candidates.py) | Joint prefill QK normalization stays correct but adds costly structural work. Separate Z enables a genuine decode SiLU epilogue; its matched raw-Z control uses the same full grid |
| [Gate combinations](../../tests/combined_fusion_candidates.py) and [mode combinations](../../tests/mode_fusion_candidates.py) | Decode-only gate folds, joint decode QK norm before repetition, separate activated decode Z, and raw prefill Z with dedicated gated RMSNorm selected. All-mode gate folding adds no prefill win. Mixed-input SiLU binary folds fail; FP32 adaptations pass but lose speed |
| [Coherent final candidates](../../tests/final_fusion_candidates.py) | Combine ordinary prefill Conv1d, mode-specific norm/Z, attention, and MLP choices. Short-prefill L1 input wins its matched handoff probe; post-concat SiLU does not improve the measured coherent prefill |
| [Cast candidates](../../tests/decode_cast_candidates.py) | Q FP32 output, mixed V subtract, and rounded K unary chain are isolated before combination; candidate trace is 1.461613 ms. Current short/long suites pass with the later transpose adaptation |
| [Outer-product transpose candidates](../../tests/transpose_fusion_candidates.py) | Default `transpose_a=True` still inserted a transpose. Whole-head reuse (M/N=4/4, K block=1, subblock=1×4) supports the actual fold. The preserved default-program control loses in 25/31 then 29/31 matched windows; selected after unchanged PCC/state checks |
| Exact source exclusions | Height-sharded input RMSNorm is unsupported by its validator; full GQA decode SDPA cannot directly output the required concat shards; residual-input RMSNorm exposes no reusable residual-sum output; exposed SwiGLU is a composite; KDA chunk scans do not express this scalar-gate FP32 recurrent contract unchanged |
| Numerical exclusions | FP32 softplus folded into binary bias-add zeros measured negative-input tails that standalone FP32 softplus preserves. Matmul bias and norm-weight folds change rounding and lose to the selected coherent graph. A passing aggregate layer PCC alone did not justify retaining these changes |

Raw logs and same-stem provenance retain failed first attempts as well as repaired
and passing adaptations. Candidate classes that remain available but unmeasured
are not claimed as fixes or measured rejections. The completed graph/profile
audit is recorded in the ledger; independent review confirms the applicable-pattern
gate is closed with no required work.

## Runtime warning audit

| Observed diagnostic | Investigation and resolution |
| --- | --- |
| Device allocations after an active trace | The initial two-trace matched harness allocated decoder B after capturing A; [the diagnostic reproduction](logs/trace_allocation_repro_v1.log.gz) proves A's single replay corrupts 12 persistent B buffers even though B never replays. The harness now allocates both models and all persistent consumer inputs before either capture. `final_short_v2` still emits the conservative allocator warning at 23:28:12.334 during second-capture temporary/output allocation. That output is consumed only after its own replay rewrites it; eager equality before/after timing, 256-replay stress and state checks pass. This warning is recorded, not treated as absent. The one-trace validation path has no corruption-related diagnostics. See [verified AutoFix](AUTOFIX_transpose_trace.md). |
| Conv2D DRAM ignores explicit memory config | `conv2d.cpp:805` fixes the auto-DRAM-sliced output to DRAM interleaved, exactly the requested memory config. Shorter ordinary-convolution paths still need the explicit DRAM output argument. Final native/B32 tests pass; profiler shows device convolution and its required sharding. This is an ignored matching argument, with no host compute fallback. |
| Unknown B850M-C motherboard / tray-id fallback | Physical discovery uses bus ID for tray labeling; it does not change chip architecture or mesh. Four Blackhole chips were listed; both functional and fused runs use one chip on P300c boards. |
| Opening subset of MMIO devices | Expected for the selected 1×1 topology on this four-chip host. Both before/after paths use the same topology; no multichip performance is claimed. |
| `[EXPECTED_ERROR] host fallback` assertions | Deliberate positive controls in the inherited guard test prove the guard catches host transfers. The subsequent guarded fused prefill/decode passes. |
| Deprecated kernel copy-tile overload / Python SWIG types | Toolchain deprecation diagnostics; Watcher-instrumented kernels compile and run. No C++ or toolchain files are changed. |
| Tracy optional web trace copy fails | The extra browser-view copy refers to a missing default host-log path. Each requested output directory contains the actual capture and ops CSV; all four complete replay-ID and raw-duration checks pass. Compact validated CSV/text reports are the performance evidence. |
| Pandas mixed-type CSV warning | Signpost/trace metadata columns mix text and numeric rows. The summary parses explicit signposts and checks every measured row has a kernel duration; it also compares raw nanoseconds with report microseconds. |

No device hang, reset, recovery or unexpected host fallback occurred in the
completed current-runtime validation. The matched two-trace allocation warning
and the separately reproduced earlier corruption are explicitly classified
above; this is not a warning-free run claim. Watcher and profiler use separate
processes. The complete failure/adaptation history remains in the pattern ledger.

## Reproduction and provenance

Run from the repository root in the existing environment. The model reference
loads only the pinned local snapshot `/home/hous/dev/ornith-1.5-9b/upstream`;
`ORNITH_MODEL_PATH` may name another verified copy of that same revision.
The starting functional-stage commit recorded in these experiments is
`ae18ba18bdde6ae20dd628d3f2fd8f59c42ecaf3`; tt-metal base is
`e7638d2859b6a1ef30eb984781cbddf9872a8d62`. A Git HEAD alone is insufficient
to reproduce an uncommitted experiment: use its source hashes/archive.

```bash
source python_env/bin/activate
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export TT_METAL_HOME=/home/hous/dev/ornith-1.5-9b/tt-metal
export TT_METAL_CACHE=/home/hous/dev/ornith-1.5-9b/state/tt-cache
export OMP_NUM_THREADS=8
export ORNITH_WEIGHTS=real
unset ORNITH_FUSION_CANDIDATE TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
```

Only the supervisor's serialized hardware lane runs device workloads. Watcher
and profiler use separate processes. The following are exact pytest argv from
completed provenance records; reproducing an old result requires its archived
source, candidate environment and numerical policy as well as the command.

```bash
# baseline.provenance.json: original functional baseline
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py \
  -k 'perf or real_weights_pcc' -v -s

# selected_matrix.provenance.json: coherent candidate before final cast changes
ORNITH_FUSION_CANDIDATE=conv_final_combination \
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py \
  -k 'not full_context and not long_context and not native_context and not perf' -x -v -s

# integrated_pair_v1.provenance.json: runtime at that recorded source revision
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k 'paired or linear_core' -x -v -s

# all_cast_v1.provenance.json: latest isolated cast combination
ORNITH_FUSION_CANDIDATE=all_cast \
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k 'paired and linear_attention' -x -v -s

# short_norm_handoff_v2.provenance.json: matched B1/T128 norm movement probe
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k short_prefill_norm_handoff -x -v -s
```

For new evidence, [record_run.py](record_run.py) wraps the command without changing
its arguments; choose a fresh run name because existing evidence is not overwritten:

```bash
python models/autoports/ornith_ai_ornith_1_5_9b/doc/fused_decoder/record_run.py \
  reproduction_pair_v1 \
  pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k 'paired or linear_core' -x -v -s
```

Each run stores stdout/stderr, exact argv/cwd, selected environment, start/end
timestamps, return code, Git HEAD, per-file SHA256 for model `tt/`, `reference/`
and `tests/`, and the log hash. Later runs also have a same-stem
`.sources.json.gz` containing those model Python files plus its archive hash.
For example, [integrated source snapshot](logs/integrated_pair_v1.sources.json.gz)
reproduces the model source for that pair; it does not contain the entire
tt-metal checkout, compiled toolchain, Python environment, or weights.
Use the pinned repository/environment and original checkpoint provenance too.
Early runs without source archives have weaker reproduction coverage; their
hashes identify source but cannot recreate missing contents by themselves.

To inspect an archive safely, decompress the JSON and extract files into a
separate scratch checkout, verifying every `source_sha256` entry against the
corresponding UTF-8 text. Do not overwrite the active runner's source to replay
an old candidate. If plain logs are later archived as `.log.gz`, restore their
bytes with gzip and verify the uncompressed log against `log_sha256`.

The completed current-runtime commands below use
`ORNITH_FUSION_CANDIDATE` unset. Their provenance records and source archives
are [final_short_v2](logs/final_short_v2.provenance.json) /
[source snapshot](logs/final_short_v2.sources.json.gz) and
[final_long_v2](logs/final_long_v2.provenance.json) /
[source snapshot](logs/final_long_v2.sources.json.gz):

```bash
# final_short_v2: 85 passed, 9 deselected
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k 'not full_context and not long_context and not native_context' -x -v -s

# final_long_v2: 9 passed, 73 deselected
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py \
  -k 'full_context or long_context or native_context' -x -v -s
```

To repeat the controlled outer-program comparison, use:

```bash
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py \
  -k reuse_outer_matched_timing -x -v -s
```

The original failed setup-order control is opt-in and diagnostic only; its pass
means corruption was reproduced, not model correctness:

```bash
ORNITH_TRACE_ALLOCATION_REPRO=1 \
pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_trace_allocations.py -x -v -s
```

The [current Watcher provenance](logs/watcher_final_v2.provenance.json) records
`TT_METAL_WATCHER=10` (10-second polling), with profiler unset: 23 passed in 44.57 s.
[watcher_audit.json](watcher_audit.json) records 92 check messages, no matching
diagnostics, and compressed final-session dumps with hashes; full stdout covers
all 23 tests. For profiler artifacts,
[render_perf.py](render_perf.py) renders each layer kind's prefill/decode report
between `PERF_PREFILL`/`PERF_PREFILL_END` or `PERF_DECODE`/`PERF_DECODE_END`, adding
`--tracing-mode` for decode. [summarize_perf.py](summarize_perf.py) requires no
dropped markers, one passing measured-output test per capture, complete replay
operation IDs matching the warm template, and report microseconds equal to raw
kernel nanoseconds divided by 1000. All four current captures pass these checks.
[performance.json](performance.json) records durations, counts, HF PCC and report
hashes; [validation.json](validation.json) records the 85/9/23 passing current
runs, paired measurements and raw-core metrics. Historical summaries remain in
[performance_v1.json](performance_v1.json) and [validation_v1.json](validation_v1.json).
The independent [stage review](STAGE_REVIEW.md) is `clean-pass`; local checkpoint
SHAs are recorded in [work_log.md](work_log.md).
