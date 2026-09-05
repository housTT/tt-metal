# Optimized TP4 decoder: evidence synthesis

Source-only review of completed records on 2026-09-05 for Ornith-1.5-9B,
revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, on four Blackhole chips
on two P300c boards, logical 1x4 ring. The initially promoted QKV4 decode
policy failed final validation. Production now uses packed QKV8 for every batch
and packed decode MLP gate/up with three readers after controlled comparisons.
The stable defaults pass all 101 final watcher tests, both async checks and
the three-layer traced stack. Current family/precision comparisons and paired
before/after performance accounting are complete. The independent
[stage review](STAGE_REVIEW.md) returns `clean-pass` with no required work.
Local checkpoint SHAs are recorded in the [work log](work_log.md#local-checkpoints).

Whole-layer numbers are the actual TP4 path at batch 1/context 2048 unless
stated otherwise. Decode is warmed trace replay; prefill is the median of
synchronized warmed forwards. Sample counts and exact configurations belong
to each linked provenance. State restoration and host comparison are outside
timing. The optimized single-chip path is a numerical control, not a timing
baseline for this stage. Probe PCCs compare against that control; direct HF
checks are identified explicitly. Higher PCC against a lower-precision control
does not establish better HF accuracy.

Each run link below names a `.provenance.json` with command, environment,
source/archive hashes and loaded-library hashes; the corresponding `.log.gz`
and `.sources.json.gz` preserve the execution. [Candidate measurements](candidate_measurements.md)
provide the refreshed 379-record whole-layer index at this revision. The
[summarizer](summarize_candidates.py) now names its flag `probe_pass` (formerly
`eligible`): it records that run's probe gates, not stage eligibility. This
synthesis checks the linked completed archives directly. A passing historical
QKV4 probe remains evidence of that run, not acceptance of the failed policy.

## Current policy and completed final validation

The current [MeshConfig source](../../tt/multichip_decoder.py) uses replicated
residuals, native collectives, both-phase packed GDN with the compact Z gate,
**packed decode MLP gate/up in DRAM at 32 input cores/K-block4/R3**, and
**BFP8 decode QKVG at 32 input cores/K-block4/R1 for all batches**. MLP down
uses eight cores/block6/R2; `pack_mlp_decode=True` selects the packed decode path.
Residual cores remain 32, SDPA remains 8x8/chunk256, KV remains BFP8, other
projection groups remain BFP4/LoFi, and GDN recurrence is FP32.

Setup packs raw local gate/up weights into one DRAM allocation shared by
`w["gate_up"]` and `decode_weights["gate_up"]`; it removes the separate decode
copies. Prefill keeps its separate gate/up weights and operations. For B>1,
the decoder compacts user rows internally before sharded slices and restores
their logical batch shape after gating. Callers retain the same interface.

The [promoted-default batch smoke](logs/packed_promoted_batch_smoke.provenance.json)
passes all ten tests with no candidate override: B4/B32 prefill/decode HF
checks and B1/B4/B32 changed-input decode traces, both layer kinds. This includes
the original failing B32 full-attention selector; its three aggregate PCCs are
0.999039/0.999185/0.999153 and all per-user gates pass. The two promoted-default
whole-layer probes also pass numerical/state gates and exact restored trace
after eager execution: [linear decode 0.355676 ms](logs/packed_promoted_probe_layer0.provenance.json)
and [full-attention decode 0.268672 ms](logs/packed_promoted_probe_layer3.provenance.json),
with prefill medians 3.417262/2.740990 ms over 15 warmed samples each. These
promotion probes are superseded for final timing by the measurements below.

The [final_default_v2 watcher suite](logs/final_default_v2_watcher_contracts.provenance.json)
passes all **101 tests**, including the formerly failing B32 full-attention
trace selector, long-prefill/B32 stress, continuation, native capacity and
refreshed prefill traces. Worker watcher uses `TT_METAL_WATCHER=10`; Ethernet
watcher instrumentation remains disabled under the recorded prior kernel-buffer
limitation. Separate two-link async carried-shard checks pass for
[linear](logs/final_default_v2_watcher_async_layer0.provenance.json) and
[full attention](logs/final_default_v2_watcher_async_layer3.provenance.json).

The native262144 test reserves **7,153,385,472 DRAM bytes per device** alongside
24 recurrent layers' L1 state while executing the tested full-attention layer.
This is a capacity reservation, not execution of a complete model. The native
cache oracle passes HF PCC **0.9983894214** at position262143 with permuted
pages and exact eager/trace output; its historical cache fixture is not an HF
full-context prefill. [The warning/anomaly ledger](warning_ledger.md) records
the trace-allocation controls, watcher scope and preserved historical anomalies.

## Historical QKV4 failure and controlled repair

The earlier [QKV4 default smoke](logs/selected_defaults_smoke.provenance.json)
passed six tests: B4/B32 × both layer kinds against HF, plus both B1/T2048
prefill traces. Those traces had exact output/state and refreshed-input replay.
The subsequent [final_default_v1 watcher suite](logs/final_default_v1_watcher_contracts.provenance.json)
**failed with 65 passed and one failed test**, stopping before all 101 planned
cases. Full-attention B32 user 31 at position63 had PCC **0.9942707597 < 0.995**
on the real changed-input trace selector. The earlier smoke does not override
that failure.

Native context262144 capacity reservation, the native cache oracle
(PCC 0.9981916521), full-attention chunk invariance (0.999954), and traced
position262143 had passed before the suite stopped. These are QKV4-era
artifacts; the completed QKV8 final-default rerun above supersedes their scope.

The [fresh diagnosis](AUTODEBUG_qkv_trace.md) and
[completed QKV AutoFix](AUTOFIX_qkv_trace.md) isolate the failed input:

| Exact input/control | First-step minimum HF PCC | Finding |
| --- | ---: | --- |
| [QKV4 B32, 8/block16/R2](logs/qkv_trace_b4_b32.provenance.json) | 0.9942707597 | Reproduces original user31 failure in eager, trace and restored replay |
| [Same user31 extracted into B1](logs/qkv_trace_b4_user31.provenance.json) | 0.9943646417 | A batch-dependent precision fallback would not fix it |
| [Matched QKV8 B32, 8/block16/R2](logs/qkv_trace_b8_b32.provenance.json) | 0.9964473790 | All 96 user/step checks pass with identical inputs and prefill cache |
| [QKV4 HiFi2](logs/qkv_trace_b4_hifi2_b32.provenance.json) | 0.9942707597 | Unchanged failure |
| [QKV4 HiFi4 + FP32 destination](logs/qkv_trace_b4_hifi4_fp32_b32.provenance.json) | 0.9948409006 | Still below the unchanged gate |
| [QKV4 32/block4/R1](logs/qkv_trace_b4_c32_r1_b32.provenance.json) | 0.9945455234 | Alternate geometry still fails |
| [BFP8 gate-only substitution](logs/qkv_trace_gate8_b32.provenance.json) | 0.9966528050 | All 96 pass; initial/final K/V remain bit-identical to QKV4 |

Exact restored outputs, final cache hashes and unchanged snapshots refute
trace/cache-state corruption for these inputs. Gate precision is a sufficient
rescue, not proof that no other component correction could rescue the output.
The substitution uses two full packed projections and supplies no timing claim.

The actual smaller QKV4/gate8 split passes the
[original 96-output trace selector](logs/qkv4_gate8_split_trace32.provenance.json).
Its initial configuration plus all 13
[geometry/reader adaptations](mixed_qkv_geometry_matrix.json) pass whole-layer
probe gates. Best split decode is [0.275440](logs/qkv4_gate8_split_layer3.provenance.json)
ms, versus packed QKV8 [0.271286](logs/qkv8_correct_family_c32_b4_r1_layer3.provenance.json)
at 32/block4/R1 and [0.271270](logs/qkv8_correct_family_c8_b16_r1_layer3.provenance.json)
at 8/block16/R1. The packed geometries differ by only 0.016 us, treated as a tie;
retain the established 32-core geometry. The best actual split is 4.154 us
slower. Original trace selectors pass QKV8 at both tested R2 geometries;
the retained R1 family passes its whole-layer probe and the completed final-default
suite above.

## Interlayer residual contract

TP shards weights and local heads. The replicated family carries the complete
4096-wide BF16 hidden state on every rank. The hidden-sharded family carries
each rank's contiguous 1024-wide slice through both residual additions; its
distributed norms gather statistics and normalized projection inputs inside
the layer. A following layer of the same family consumes the output directly.
No wrapper gather, all-reduce, or reshard is part of either interface.

| Boundary | Replicated family | Hidden-sharded family |
| --- | --- | --- |
| Logical prefill input/output | `[B,T,4096]` per rank | `[B,T,1024]` per rank |
| Prefill output memory | Tiled BF16, DRAM interleaved | Tiled BF16, DRAM interleaved |
| Logical decode input/output | `[B,1,4096]` per rank | `[B,1,1024]` per rank |
| Decode output, B=1 with 32 residual cores | L1 width-sharded, row-major 8x4 grid, shard `[32,128]` | L1 width-sharded, row-major 8x4 grid, shard `[32,32]` |
| Decode output, B=2–32 | **DRAM interleaved**, restored to public shape | **DRAM interleaved**, restored to public shape |

The batch distinction is explicit in `OptimizedDecoder._residual_add`: it
folds users to `[1,B,H]` for the sharded residual addition, then converts to
DRAM and restores `[B,1,H]`. Thus “decode always returns L1 shards” is not
the current contract. Residual-core experiments change internal geometry;
they are not an external conversion requirement. Source: [residual implementation](../../tt/optimized_decoder.py),
[TP block and distributed norms](../../tt/multichip_decoder.py). The B32 HF
diagnostic also records the returned DRAM output in [its archived log](logs/z_user31_hf_current_v2.log.gz).

State remains private to each layer/rank: eight GDN value heads with FP32
recurrent state, four attention query heads and one KV head per rank. The KV
page block is 64 and head dimension 256. The native context contract remains
262144 and supported logical batch range 1–32; candidate context-2048 timing
does not validate that full capacity. Padding, slicing and unaligned continuation
belong inside the decoder. The [final stack run](logs/final_default_v2_stack.provenance.json)
feeds layers **0/3/0** directly at B1, with a 131-token prefill and eight traced
decode steps. It records **zero boundary conversions**, exact eager/trace
output, and minimum decode PCC **0.99992039855** against the composed numerical
control. This validates the tested replicated B1 composition; the public
batched and carried-shard contracts retain the distinct layouts stated above.

## Final measurements and device attribution

[measurements.json](measurements.json) records final default source/runtime,
unchanged numerical gates and the before/after runs. Both kinds use B1/T2048;
prefill has 15 synchronized warmed samples, and decode is the median of five
warmed nonblocking windows of 32 trace replays. State restoration and host
comparison remain outside timing. All final output/state gates and exact
trace/replay-after-eager checks pass.

| Layer kind | Before prefill ms | Final prefill ms | Before decode ms | Final decode ms | Decode latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear attention | [3.463762](logs/before_layer0.provenance.json) | [3.443891](logs/final_default_v2_timing_layer0.provenance.json) | [0.371151](logs/before_layer0.provenance.json) | [0.355672](logs/final_default_v2_timing_layer0.provenance.json) | 4.17% |
| Full attention | [2.858084](logs/before_layer3.provenance.json) | [2.690952](logs/final_default_v2_timing_layer3.provenance.json) | [0.283669](logs/before_layer3.provenance.json) | [0.268965](logs/final_default_v2_timing_layer3.provenance.json) | 5.18% |

| Layer kind | Before prefill / decode PCC | Final prefill / decode PCC |
| --- | --- | --- |
| Linear attention | 0.999963081 / 0.999987989 | 0.999963081 / 0.999990048 |
| Full attention | 0.999977247 / 0.999695988 | 0.999977247 / 0.999697387 |

These PCCs use the optimized single-chip numerical control; direct HF gates
are recorded separately above. Linear prefill's 0.57% lower median is within
the observed sample variation. Full attention's measured prefill median is
5.85% lower, but its prefill math, topology and weight precision are unchanged;
do not attribute that difference to an algorithmic improvement. Both runs'
sample arrays remain available, and their timing ranges overlap.

The paired [performance accounting report](performance_accounting.md) and
[JSON](performance_accounting.json) cover all **eight profiles / 32 rank records**:
both kinds, prefill/decode and before/after, with advice reports, raw operations,
actual dtype/program metadata and per-rank attribution. Final decode matmul
counts drop from 9 to 7 for linear attention and 5 to 4 for full attention;
total operation counts drop 62→59 and 47→46. Packed GDN combines two input
projections, while packed gate/up R3 and down R2 reduce decode projection cost.
Movement and added packed-output slices remain part of the complete totals.

| Decode profile | Same-run host us, before / after | Per-rank kernel range us, before / after | Per-rank interior-gap range us, before / after |
| --- | --- | --- | --- |
| Linear attention | 401.492 / 385.308 | 300.397–302.153 / 286.832–288.344 | 80.641–82.890 / 79.100–81.021 |
| Full attention | 337.783 / 314.762 | 266.755–268.655 / 254.142–254.852 | 49.888–50.183 / 48.478–48.871 |

Profiled host times retain profiler/launch/synchronization overhead and differ
from the unprofiled final timing table. Ranks are never summed, and independent
rank-clock spans are not treated as a synchronized mesh span. Only the initial
pre-window gap is excluded; all interior gaps remain. Runtime metadata confirms
BFP4 packed gate/up with R3, BFP4 down with R2, BFP8 decode QKVG with R1, and
the actual BF16/FP32 activation boundaries.

The report's separate optimistic decode storage floor is 60.192 us for linear
attention and 70.050 us for full attention, using actual consumed weight tile
storage plus minimum KV reads at an assumed 512 GB/s per chip. It excludes
state traffic, cache writes, movement, collectives, repeated reads and compute;
it is not a measured P300c bandwidth limit or an attainable whole-layer target.
The partial operation performance model also excludes its ≤1 ns placeholders;
its coverage is explicit rather than presented as a complete layer roofline.

[Artifact integrity](artifact_integrity.json) reports **466 completed runs**,
one separately recorded interruption, **933 verified archive files** and
**36,591 archived source entries**, with zero integrity errors. Completed runs
include preserved nonzero experimental outcomes, not 466 correctness passes.
All ten final-validation/profile runs match the current recorded Python/native
source closure and runtime binaries. Integrity verifies recorded content;
the numerical and performance conclusions come from the gates and records above.

## Operation-topology audit and measured response

The initial audit used prior-stage advice-enabled reports: input GDN projections
were approximately 19.4+12.5 us, two row-output collectives together 39–50 us,
and movement 35–43 us in decode. These are attribution clues, not this stage's
before/after totals. See [prior attribution](../multichip_decoder/PERF_ANALYSIS.md)
and the fresh [linear](tracy/linear_attention/before/decode_device0_report.txt)/
[full-attention](tracy/full_attention/before/decode_device0_report.txt) baseline reports.

| Material boundary | Compatible experiment | Evidence-led disposition |
| --- | --- | --- |
| QKV/A/B and Z share normalized GDN input | Pack raw per-rank HF weights, retain aligned A/B fields, own slices and Z activation | Packed both-phase GDN and compact Z gate pass; 8x4/block8 decode geometry measured against 8x2, 8x8 and 11x10 |
| Gate/up share MLP input; down consumes gated local channels | Shared input, packed gate/up, interleaved or DRAM projections, R1/R2/R3 | Packed32/block4/R2 ties or loses matched pairs; packed32/block4/R3 wins both repeated pairs for both layer kinds and is promoted, with down kept at R2 |
| Mixer/down row projections followed by collectives | Native all-reduce; RS with carried hidden shards; gathered-input column projections; fused AG-MM/MM-RS | Full families pass after semaphore repair; native replicated decode is faster in these records |
| Residual normalization and layouts | Distributed norm/statistics gather, fused norm/AG-MM, persistent CCL buffers, 16/32/64 residual cores | Hidden-sharded boundaries remain intact; no timed restore to a different family. Residual-core selection remains cumulative |
| Projection movement/readers | Mesh-coordinate R2/R3 support, direct working shards, fewer/wider input cores, padded output capacity | Native fix is built and exercised; local timing does not substitute for whole-layer integration |
| Attention/cache and precision | QKVG BFP4/BFP8, SDPA chunk/grid, activation/CCL casts, grouped weight/fidelity policies, BFP4 KV | QKV4 probe sweeps complete but final HF gate fails; retain QKV8 after adapted controls. Adapted chunk1024 is slower in its measured family; BFP4 KV fails controlled state/HF gates |

Fresh TP4 controls are [linear](logs/before_layer0.provenance.json)
**3.463762 ms prefill / 0.371151 ms decode** and
[full attention](logs/before_layer3.provenance.json)
**2.858084 / 0.283669 ms**. Their prefill/decode PCCs are
0.999963081/0.999987989 and 0.999977247/0.999695988, with exact restored trace.

## Completed current-policy family comparisons

All 32 cases in [the resumed packed-default matrix](packed_default_families_resumed.json)
pass on packed GDN, packed MLP32/block4/R3, down8/block6/R2 and QKV8 at
32/block4/R1. Both additional packed fused-norm/sharded cases in
[the final precision matrix](packed_final_precision.json) also pass. Every row
has exact restored trace after intervening eager work. Linked pairs are
layer0 / layer3 decode milliseconds; replicated outputs are 4096 wide and
carried hidden shards are 1024 wide.

| Packed-MLP family | Replicated hidden | Carried hidden shards |
| --- | --- | --- |
| Native row projection / ordinary norm | [0.355832](logs/packedfinal_default_replicated_layer0.provenance.json) / [0.268733](logs/packedfinal_default_replicated_layer3.provenance.json) | [0.460788](logs/packedfinal_default_sharded_layer0.provenance.json) / [0.381158](logs/packedfinal_default_sharded_layer3.provenance.json) |
| Gather + output-column projection | [0.479143](logs/packedfinal_ag_mm_replicated_layer0.provenance.json) / [0.391346](logs/packedfinal_ag_mm_replicated_layer3.provenance.json) | [0.562181](logs/packedfinal_ag_mm_sharded_layer0.provenance.json) / [0.480527](logs/packedfinal_ag_mm_sharded_layer3.provenance.json) |
| Fused gather + matmul | [0.483263](logs/packedfinal_fused_ag_mm_replicated_layer0.provenance.json) / [0.391383](logs/packedfinal_fused_ag_mm_replicated_layer3.provenance.json) | [0.564682](logs/packedfinal_fused_ag_mm_sharded_layer0.provenance.json) / [0.482656](logs/packedfinal_fused_ag_mm_sharded_layer3.provenance.json) |
| Fused matmul + reduce-scatter | [0.428674](logs/packedfinal_fused_mm_rs_replicated_layer0.provenance.json) / [0.341892](logs/packedfinal_fused_mm_rs_replicated_layer3.provenance.json) | [0.511479](logs/packedfinal_fused_mm_rs_sharded_layer0.provenance.json) / [0.433576](logs/packedfinal_fused_mm_rs_sharded_layer3.provenance.json) |
| Persistent collective buffers | [0.372546](logs/packedfinal_persistent_replicated_layer0_resumed.provenance.json) / [0.285556](logs/packedfinal_persistent_replicated_layer3.provenance.json) | [0.460473](logs/packedfinal_persistent_sharded_layer0.provenance.json) / [0.382324](logs/packedfinal_persistent_sharded_layer3.provenance.json) |
| CCL payload BFP8, packet8192 | [0.364504](logs/packedfinal_ccl_bfp8_replicated_layer0.provenance.json) / [0.279763](logs/packedfinal_ccl_bfp8_replicated_layer3.provenance.json) | [0.466178](logs/packedfinal_ccl_bfp8_sharded_layer0.provenance.json) / [0.390849](logs/packedfinal_ccl_bfp8_sharded_layer3.provenance.json) |
| Attention activations BFP8 | [0.375306](logs/packedfinal_acts8_attention_replicated_layer0.provenance.json) / [0.283775](logs/packedfinal_acts8_attention_replicated_layer3.provenance.json) | [0.474079](logs/packedfinal_acts8_attention_sharded_layer0.provenance.json) / [0.385945](logs/packedfinal_acts8_attention_sharded_layer3.provenance.json) |
| MLP activations BFP8 | [0.363326](logs/packedfinal_acts8_mlp_replicated_layer0.provenance.json) / [0.275223](logs/packedfinal_acts8_mlp_replicated_layer3.provenance.json) | [0.465032](logs/packedfinal_acts8_mlp_sharded_layer0.provenance.json) / [0.386953](logs/packedfinal_acts8_mlp_sharded_layer3.provenance.json) |
| Fused distributed norm + gather/matmul | Not this input contract | [0.467420](logs/packedfinal_fused_norm_sharded_layer0.provenance.json) / [0.395854](logs/packedfinal_fused_norm_sharded_layer3.provenance.json) |

The original `packedfinal_persistent_replicated_layer0` record was interrupted
by runner-container recreation with an empty log; it is not a numerical
failure or accepted measurement. [The interruption manifest](interrupted_runs.json)
preserves that status and names the completed replacement
`packedfinal_persistent_replicated_layer0_resumed`, linked in the table.

Native replicated remains the fastest decode family in these runs. Hidden
shards are carried through the layer, with host reconstruction only after
measurement. Replicated alternatives include their required output gathers
inside timing. Prefill can rank differently, so these are decode decisions,
not claims that every operation or phase becomes faster.

The additional [19 full-attention controls](validated_full_attention_families.json)
hold QKV8 fixed and start from separate DRAM MLP R2, with an explicit packed-MLP
alternative. All 19 pass output/state and both restored-trace gates. Their
full-attention decode milliseconds independently confirm the coherent choice:

| QKV8 control family | Replicated hidden | Carried hidden shards |
| --- | --- | --- |
| Native row projection / ordinary norm | [0.271275](logs/validatedfamily_default_replicated_layer3.provenance.json) | [0.382325](logs/validatedfamily_default_sharded_layer3.provenance.json) |
| Gather + output-column projection | [0.391021](logs/validatedfamily_ag_mm_replicated_layer3.provenance.json) | [0.481687](logs/validatedfamily_ag_mm_sharded_layer3.provenance.json) |
| Fused gather + matmul | [0.395536](logs/validatedfamily_fused_ag_mm_replicated_layer3.provenance.json) | [0.483000](logs/validatedfamily_fused_ag_mm_sharded_layer3.provenance.json) |
| Fused matmul + reduce-scatter | [0.344689](logs/validatedfamily_fused_mm_rs_replicated_layer3.provenance.json) | [0.435389](logs/validatedfamily_fused_mm_rs_sharded_layer3.provenance.json) |
| Packed MLP32/block4/R3 | [0.268686](logs/validatedfamily_packed_mlp_replicated_layer3.provenance.json) | [0.381275](logs/validatedfamily_packed_mlp_sharded_layer3.provenance.json) |
| Persistent collective buffers | [0.289268](logs/validatedfamily_persistent_replicated_layer3.provenance.json) | [0.383171](logs/validatedfamily_persistent_sharded_layer3.provenance.json) |
| CCL payload BFP8, packet8192 | [0.283539](logs/validatedfamily_ccl_bfp8_replicated_layer3.provenance.json) | [0.393053](logs/validatedfamily_ccl_bfp8_sharded_layer3.provenance.json) |
| Attention activations BFP8 | [0.287154](logs/validatedfamily_acts8_attention_replicated_layer3.provenance.json) | [0.386711](logs/validatedfamily_acts8_attention_sharded_layer3.provenance.json) |
| MLP activations BFP8 | [0.281551](logs/validatedfamily_acts8_mlp_replicated_layer3.provenance.json) | [0.391886](logs/validatedfamily_acts8_mlp_sharded_layer3.provenance.json) |
| Fused distributed norm + gather/matmul | Not this input contract | [0.380597](logs/validatedfamily_fused_norm_ag_mm_sharded_layer3.provenance.json) |

Historical controls remain in [the QKV4 topology matrix](final_topology_matrix.json)
(38 passing local probes) and [the QKV4 fidelity/async/SDPA matrix](final_remaining_families.json)
(19 passing local probes). Their native replicated controls were
[0.358910](logs/finalfamily_default_replicated_layer0.provenance.json) /
[0.266112](logs/finalfamily_default_replicated_layer3.provenance.json) ms.
Those layer3 numbers use the rejected QKV4 policy; they do not override the
later HF failure or substitute for current-policy comparisons. The earlier
six async controls passed after semaphore repair, so the initial two-link
first-error rejection is obsolete. Current async comparisons follow below.

## Reader geometry and cumulative integration

R1/R2/R3 denote workers per active DRAM bank, not total input-storage cores.
The probes retain raw HF weights, logical local K/N, dtype and fidelity; log
actual padded weight width, active readers, output storage, fresh-buffer
rebinding, changed-input replay, and alternating timing windows.

The completed packed gate/up confirmations use actual decode activations at
position2048, raw per-rank HF weights, local K4096/N6144, BFP4/LoFi, 32 input
cores and K-block4. [Layer0](logs/reader_packed_alternating_layer0.provenance.json)
and [layer3](logs/reader_packed_alternating_layer3.provenance.json) pass all
three reader counts, fresh input/output rebinding and exact changed-input
trace replay. Timings alternate R1/R2/R3 and R3/R2/R1, with five retained
measurement windows after warmup. Input sharding precedes this local projection
timing; it excludes output gating, down projection and the row collective.

Separate [layer0](logs/reader_packed_profile_layer0.provenance.json) and
[layer3](logs/reader_packed_profile_layer3.provenance.json) Tracy runs pass and
provide device attribution. Ranges below cover all four ranks; their host
times come from those same profiled runs, not the alternating runs.

| Layer / readers | Alternating trace us | Profiled kernel us, rank range | Same profiled-run host us | Weight storage GB/s, rank range |
| --- | ---: | --- | ---: | --- |
| 0 / R1 | 59.09978 | 52.84025–53.02050 | 75.69199 | 266.99–267.90 |
| 0 / R2 | 44.66892 | 37.33275–38.10000 | 54.69448 | 371.54–379.18 |
| 0 / R3 | 41.54530 | 34.03800–34.52500 | 50.94477 | 410.02–415.88 |
| 3 / R1 | 59.09855 | 52.77725–53.01000 | 79.40401 | 267.04–268.22 |
| 3 / R2 | 44.54339 | 37.45225–37.84300 | 53.95550 | 374.07–377.97 |
| 3 / R3 | 41.58538 | 33.77875–34.26650 | 50.70926 | 413.11–419.07 |

The [24-row accounting JSON](reader_packed_device_accounting.json) and
[CSV](reader_packed_device_accounting.csv) retain each rank, source hash and
report command. All readers store the same 14,155,776 weight bytes per chip
without extra padded columns. Per-reader tile-row payload drops from 13,824
to 6,912 to 4,608 bytes. Weight-storage GB/s is stored weight bytes divided by
kernel duration; R3 reaches 80.08–81.85% of the stated 512 GB/s reference model.
That fraction is an accounting estimate, not measured DRAM-bus utilization.
Advice-enabled [layer0](tracy/readers/layer0/r3_device0_report.txt) /
[layer3](tracy/readers/layer3/r3_device0_report.txt) reports and their
[layer0](tracy/readers/layer0/reader_ops.csv.gz) /
[layer3](tracy/readers/layer3/reader_ops.csv.gz) raw operation archives preserve
the attribution. These local results corroborate the R3 whole-layer pairs;
the separate completed whole-layer accounting above supplies final attribution.

| Experiment | Measured result | Implication |
| --- | --- | --- |
| GDN packed N=2112, four input cores, K-block32, output `per_core_N=18` | [R1/R2/R3: 26.270/26.327/27.189 us](logs/reader_gdn_n18.provenance.json), all correctness/replay controls pass | R3 required 72 output tiles; the old four×17 storage provided only 68. Padding to four×18 fixes capacity without changing logical work |
| MLP, eight input cores, gate/up K-block8 and down K-block6 | [R1/R2/R3 gate: 32.941/27.516/32.026 us; up: 32.879/27.487/32.029; down: 33.665/27.708/35.368](logs/reader_sweep_adapted_layer0.provenance.json) | R2 merits whole-layer integration; R3 is not assumed faster |
| Smaller and wider working shards | [Four-core linear](logs/reader_fewer_linear_fewer4_largest.provenance.json), [QKVG eight-core](logs/reader_fewer_full_qkvg8_largest.provenance.json), [32-core](logs/reader_geometry_linear_residual32_largest.provenance.json), [64-core](logs/reader_geometry_linear_wide64_largest.provenance.json), [packed GDN 64-core](logs/reader_geometry_packed_wide64_largest.provenance.json) probes pass | Geometry is tried at fixed logical work; each role still requires its cumulative timing decision |
| Whole-layer DRAM MLP, R1 versus R2 | Linear [0.380645](logs/dram_mlp_r1_layer0_v2.provenance.json) / [0.360492](logs/dram_mlp_r2_layer0_v2.provenance.json) ms; full [0.293548](logs/dram_mlp_r1_layer3_v2.provenance.json) / [0.272283](logs/dram_mlp_r2_layer3_v2.provenance.json) ms | End-to-end support for R2 with this working-shard policy |
| Production packed GDN + DRAM MLP R2 | [Linear 0.358922 ms](logs/production_packed_dram2_qr1_layer0.provenance.json); full QKVG R1/R2/R3 [0.271353](logs/production_packed_dram2_qr1_layer3.provenance.json)/[0.272172](logs/production_packed_dram2_qr2_layer3.provenance.json)/[0.274966](logs/production_packed_dram2_qr3_layer3.provenance.json) | Reader preference is cumulative; earlier local R2 results do not force QKVG R2 |
| Equal-geometry QKVG precision/readers | Eight cores, common output N12, BFP4/BFP8 × R1/R2/R3 × block8/16 all pass the probe. Best recorded BFP4: [R2/block16 0.266045 ms](logs/production_qkv_bfloat4_b_r2_b16_n12_layer3.provenance.json); best BFP8: [R1/block16 0.271374 ms](logs/production_qkv_bfloat8_b_r1_b16_n12_layer3.provenance.json) | Separates precision from geometry and padding; later real-input HF failure rejects the QKV4 policy despite this probe speed |

Moving GDN to DRAM also earned a measured comparison: packed-input,
output-only, whole-mixer and separate-input variants pass at
[0.361181](logs/production_gdn_packed_dram_layer0.provenance.json),
[0.363800](logs/production_gdn_output_dram_layer0.provenance.json),
[0.365592](logs/production_gdn_mixer_dram_layer0.provenance.json), and
[0.367297](logs/production_gdn_separate_dram_layer0.provenance.json) ms, versus
the 0.358922 ms packed-interleaved production control.

The [15-case packed-MLP geometry matrix](packed_mlp_geometry_matrix.json)
now honors an explicit `gate_up` config instead of inheriting `gate_proj`.
All cases pass whole-layer output/state gates and both restored-trace checks.
These layer0 decode values include the actual packed projection, movement,
gating, down projection and collectives.

| Packed gate/up input cores / K-block | R1 ms | R2 ms | R3 ms |
| --- | --- | --- | --- |
| 8 / 4 | [0.376498](logs/packed_mlp_c8_b4_r1_layer0.provenance.json) | [0.363983](logs/packed_mlp_c8_b4_r2_layer0.provenance.json) | [0.372642](logs/packed_mlp_c8_b4_r3_layer0.provenance.json) |
| 8 / 8 | [0.371984](logs/packed_mlp_c8_b8_r1_layer0.provenance.json) | [0.363489](logs/packed_mlp_c8_b8_r2_layer0.provenance.json) | [0.371170](logs/packed_mlp_c8_b8_r3_layer0.provenance.json) |
| 8 / 16 | [0.371469](logs/packed_mlp_c8_b16_r1_layer0.provenance.json) | [0.364070](logs/packed_mlp_c8_b16_r2_layer0.provenance.json) | [0.371382](logs/packed_mlp_c8_b16_r3_layer0.provenance.json) |
| 4 / 16, 32 | — | [0.376368](logs/packed_mlp_c4_b16_r2_layer0.provenance.json), [0.378938](logs/packed_mlp_c4_b32_r2_layer0.provenance.json) | — |
| 16 / 4, 8 | — | [0.370901](logs/packed_mlp_c16_b4_r2_layer0.provenance.json), [0.370815](logs/packed_mlp_c16_b8_r2_layer0.provenance.json) | — |
| 32 / 2, 4 | — | [0.373231](logs/packed_mlp_c32_b2_r2_layer0.provenance.json), [0.358480](logs/packed_mlp_c32_b4_r2_layer0.provenance.json) | — |

The initial best packed32/block4/R2 value was only 0.430 us below the earlier
separate control, 0.358910 ms. All 19 cases in the subsequent
[packed-MLP final matrix](packed_mlp_final_matrix.json) and all ten in the
[R3 confirmation matrix](packed_mlp_r3_confirm_matrix.json) now pass whole-layer
and both restored-trace gates. The 32-core reader/block controls are:

| Packed gate/up K-block, 32 input cores | R1 ms | R2 ms | R3 ms |
| --- | --- | --- | --- |
| 1 | [0.424195](logs/packed_mlp_c32_b1_r1_layer0.provenance.json) | [0.418943](logs/packed_mlp_c32_b1_r2_layer0.provenance.json) | [0.426685](logs/packed_mlp_c32_b1_r3_layer0.provenance.json) |
| 2 | [0.384447](logs/packed_mlp_c32_b2_r1_layer0.provenance.json) | [0.373231](logs/packed_mlp_c32_b2_r2_layer0.provenance.json) | [0.377136](logs/packed_mlp_c32_b2_r3_layer0.provenance.json) |
| 4 | [0.373508](logs/packed_mlp_c32_b4_r1_layer0.provenance.json) | [0.358480](logs/packed_mlp_c32_b4_r2_layer0.provenance.json) | [0.355563](logs/packed_mlp_c32_b4_r3_layer0.provenance.json) |

Paired repeats compare each actual packed32/block4 path with separate DRAM
gate/up R2; down remains eight-core/block6/R2, and full attention uses QKV8.
Each cell is separate / packed decode milliseconds.

| Packed readers / repeat | Linear attention | Full attention |
| --- | --- | --- |
| R2 / 0 | [0.358912](logs/packed32_pair0_separate_layer0.provenance.json) / [0.358883](logs/packed32_pair0_packed_layer0.provenance.json) | [0.271230](logs/packed32_pair0_separate_layer3.provenance.json) / [0.271496](logs/packed32_pair0_packed_layer3.provenance.json) |
| R2 / 1 | [0.358903](logs/packed32_pair1_separate_layer0.provenance.json) / [0.358895](logs/packed32_pair1_packed_layer0.provenance.json) | [0.271537](logs/packed32_pair1_separate_layer3.provenance.json) / [0.271619](logs/packed32_pair1_packed_layer3.provenance.json) |
| R2 / 2 | [0.358922](logs/packed32_pair2_separate_layer0.provenance.json) / [0.358493](logs/packed32_pair2_packed_layer0.provenance.json) | [0.271360](logs/packed32_pair2_separate_layer3.provenance.json) / [0.271683](logs/packed32_pair2_packed_layer3.provenance.json) |
| R3 / 0 | [0.358930](logs/packed32_r3_pair0_separate_layer0.provenance.json) / [0.355915](logs/packed32_r3_pair0_packed_layer0.provenance.json) | [0.271202](logs/packed32_r3_pair0_separate_layer3.provenance.json) / [0.268754](logs/packed32_r3_pair0_packed_layer3.provenance.json) |
| R3 / 1 | [0.358749](logs/packed32_r3_pair1_separate_layer0.provenance.json) / [0.355688](logs/packed32_r3_pair1_packed_layer0.provenance.json) | [0.271532](logs/packed32_r3_pair1_separate_layer3.provenance.json) / [0.268737](logs/packed32_r3_pair1_packed_layer3.provenance.json) |

R2 supplies no consistent material gain: it is approximately tied for linear
attention and slower in all three full-attention pairs. R3 improves linear
decode by 3.015–3.061 us and full attention by 2.448–2.795 us across both pairs.
That repeated whole-layer result supports promotion of packed R3. The same
packed R3 with carried hidden shards passes at
[0.460481](logs/packed32_r3_sharded_layer0.provenance.json) /
[0.381590](logs/packed32_r3_sharded_layer3.provenance.json) ms; it remains slower
than the replicated family while preserving the sharded boundary contract.

## Current-policy precision, communication and prefill placement

All 34 cases in [packed_final_precision.json](packed_final_precision.json)
pass output/state and both restored-trace checks: two fused-norm cases above,
20 weight/fidelity cases, four async cases, four BFP8 packet-size controls and
four paired prefill-placement runs. BF16 remains the public residual type,
GDN recurrence is FP32, and decode QKVG remains BFP8. The following five
interventions otherwise retain packed MLP32/block4/R3 and the selected geometry.
Each cell is layer0 / layer3 decode milliseconds.

| Weight/fidelity intervention | Replicated hidden | Carried hidden shards |
| --- | --- | --- |
| Attention HiFi2, weight policy unchanged | [0.355825](logs/packedprecision_attention_hifi2_replicated_layer0.provenance.json) / [0.285119](logs/packedprecision_attention_hifi2_replicated_layer3.provenance.json) | [0.461438](logs/packedprecision_attention_hifi2_sharded_layer0.provenance.json) / [0.396997](logs/packedprecision_attention_hifi2_sharded_layer3.provenance.json) |
| MLP HiFi2, BFP4 weights | [0.367159](logs/packedprecision_mlp_hifi2_replicated_layer0.provenance.json) / [0.280543](logs/packedprecision_mlp_hifi2_replicated_layer3.provenance.json) | [0.472620](logs/packedprecision_mlp_hifi2_sharded_layer0.provenance.json) / [0.393369](logs/packedprecision_mlp_hifi2_sharded_layer3.provenance.json) |
| Gate/up BFP8, down BFP4, LoFi | [0.384245](logs/packedprecision_gateup8_replicated_layer0.provenance.json) / [0.297586](logs/packedprecision_gateup8_replicated_layer3.provenance.json) | [0.489417](logs/packedprecision_gateup8_sharded_layer0.provenance.json) / [0.410318](logs/packedprecision_gateup8_sharded_layer3.provenance.json) |
| All MLP BFP8, LoFi | [0.400378](logs/packedprecision_mlp8_replicated_layer0.provenance.json) / [0.313507](logs/packedprecision_mlp8_replicated_layer3.provenance.json) | [0.505619](logs/packedprecision_mlp8_sharded_layer0.provenance.json) / [0.426041](logs/packedprecision_mlp8_sharded_layer3.provenance.json) |
| All MLP BFP8, HiFi2 | [0.400907](logs/packedprecision_mlp8_hifi2_replicated_layer0.provenance.json) / [0.314649](logs/packedprecision_mlp8_hifi2_replicated_layer3.provenance.json) | [0.506030](logs/packedprecision_mlp8_hifi2_sharded_layer0.provenance.json) / [0.426505](logs/packedprecision_mlp8_hifi2_sharded_layer3.provenance.json) |

Attention HiFi2 ties the linear control to within 0.007 us but slows full
attention; the other tested interventions are slower for decode. These results
support the retained policy, not a claim that closer agreement with the
single-chip numerical control means better HF accuracy. Earlier grouped
BFP8 weight/fidelity results remain in [the separate-MLP matrix](precision_dram_matrix_v2.json).
Its initial attention-prefill resource failure was adapted by halving only
`gdn_out` K-block to eight before that matrix completed. The exact-input QKV
AutoFix above independently refutes higher fidelity as a QKV4 accuracy repair.

| Communication intervention | Replicated decode L0 / L3 ms | Carried-shard decode L0 / L3 ms |
| --- | --- | --- |
| Async 1 link | [0.376355](logs/packedfinal_async1_replicated_layer0.provenance.json) / [0.289455](logs/packedfinal_async1_replicated_layer3.provenance.json) | Ordinary carried-shard control above uses one-link async reduce-scatter |
| Async 2 links | [0.377077](logs/packedfinal_async2_replicated_layer0.provenance.json) / [0.292201](logs/packedfinal_async2_replicated_layer3.provenance.json) | No new two-link carried-shard row in this matrix |
| CCL payload BFP8, packet4352 | [0.364267](logs/packedfinal_ccl8_packet4352_replicated_layer0.provenance.json) / [0.279579](logs/packedfinal_ccl8_packet4352_replicated_layer3.provenance.json) | [0.472058](logs/packedfinal_ccl8_packet4352_sharded_layer0.provenance.json) / [0.395405](logs/packedfinal_ccl8_packet4352_sharded_layer3.provenance.json) |

The BFP8 payload controls with 4352-byte packets execute correctly but remain
slower than selected BF16 communication. Their replicated decode differs only
slightly from the BFP8/8192 controls above, and their carried-shard results are
slower. Native replicated remains selected; no first-error exclusion is used.

The final prefill placement pairs each use 15 synchronized warmed samples.
They change projection-input placement to L1 within the current packed-default
family; their exact logical inputs, weights, gates and trace checks remain fixed.

| Prefill placement | Linear prefill ms | Full-attention prefill ms |
| --- | --- | --- |
| Default placement | [3.333419](logs/packed_prefill_l1_pair_default_layer0.provenance.json) | [2.667687](logs/packed_prefill_l1_pair_default_layer3.provenance.json) |
| Projection inputs in L1 | [3.452045](logs/packed_prefill_l1_pair_prefill_l1_inputs_layer0.provenance.json) | [2.790130](logs/packed_prefill_l1_pair_prefill_l1_inputs_layer3.provenance.json) |

L1 input placement is slower by 0.118626 ms for linear attention and
0.122443 ms for full attention; retain default placement. These paired
candidate measurements do not replace final before/after release profiling.

Direct HF evidence is separate: [production batch contracts](logs/production_candidate_batch_contracts.provenance.json)
and [BFP4-QKVG eight-core contracts](logs/production_qkv4_c8_batch_contracts.provenance.json)
each pass batches 4/32 × layers 0/3 with 96-token prefills. The affected real
user 31 also passes linear layer0 at B32/T2048 in [the targeted production check](logs/z_user31_hf_current_v2.provenance.json),
whole-layer PCC 0.9989239221 on every rank, with finite/nonconstant core/Z/merged.
That check concerns the GDN investigation; it does not cover the later
full-attention user31 failure at position63. Earlier QKV4 batch successes
likewise do not override the unchanged final trace gate.

## Rejections, adaptations and open attribution

| Finding | Verified repair or controlled retry | Remaining interpretation |
| --- | --- | --- |
| R2/R3 called a unit-mesh-only hop query | Coordinate-aware factory/reader assignment built and installed; [build](logs/reader_native_build.provenance.json), [real TP4 QKVG](logs/reader2_qkvg_layer3.provenance.json), [AutoFix](AUTOFIX_reader_mesh.md) | Required Docker wrapper could not run; installed-toolchain build passed. Model reader geometry alone did not repair the native call |
| Async gather corruption before trace, workers outside inherited 8x8 semaphore grid | Independent [caller-grid](logs/ag_boundary_core_grid_layer0_v2.provenance.json) and [full-grid](logs/ag_boundary_full_grid_layer0.provenance.json) controls pass; failed coherent families retested | Full 11x10 coverage is a verified CCL repair; see [CCL AutoFix](AUTOFIX_ag_trace.md) |
| FP32 × BF16 RHS SiLU fusion corrupt/nonfinite; swapped LHS fails with multiple tiles/worker | Compact `[1,B,1024]` BF16 LHS, explicit FP32 result, restore logical users; [AutoFix and B1/B4/B32 evidence](AUTOFIX_z_fusion.md) | Model-local adaptation, not a general BinaryNG repair; numerical equivalence is not bitwise |
| Historical user-31 constant observation | [Two unchanged archived-v2 replays](logs/z_v2_archived_replay.provenance.json), legacy-grid/restored controls and targeted HF check are healthy | Unknown cause, non-reproducing; do not credit the semaphore fix or call it harmless |
| BFP4 KV fails unchanged state and some direct HF gates | [Initial whole-layer gate](logs/production_cache4_layer3.provenance.json), matched [QKVG8 B1](logs/cache_precision_qkv8_b1.provenance.json), [QKVG4 B1](logs/cache_precision_qkv4_b1.provenance.json)/[permuted](logs/cache_precision_qkv4_b1_permuted.provenance.json), and [QKVG4 B32](logs/cache_precision_qkv4_b32.provenance.json)/[permuted](logs/cache_precision_qkv4_b32_permuted.provenance.json); [completed AutoFix](AUTOFIX_cache_precision.md) | All five diagnostics complete with intentional gate-failure exits. Identical producers give K4/K8 and V4/V8 state PCC about 0.982/0.983, below 0.99. B32 KV4 decode users 8/26 fail HF at 0.993658/0.994209; matched KV8 users pass at 0.996814/0.997055. Page mapping/fill, fused update versus clone, untouched rows and CPU attention controls pass; identity/permuted outputs agree. Retain K8/V8; earlier short HF passes do not override this rejection |
| Packed-prefill K64 exceeds L1 | Smaller output-block adaptation makes [K64](logs/packed_prefill_block64_adapted_layer0.provenance.json) / [K128](logs/packed_prefill_block128_adapted_layer0.provenance.json) pass at 3.712728/3.750639 ms prefill | Slower than [K8](logs/packed_prefill_block8_layer0.provenance.json) / [K32](logs/packed_prefill_block32_layer0.provenance.json), 3.351477/3.335827 ms; a first resource exception was not the rejection |

Five historical QKV4-policy SDPA controls complete the 19-case
[remaining-family matrix](final_remaining_families.json), alongside eight
fidelity and six async cases in that same archive. All 19 exit 0 and pass both restored-trace
checks. Sub-microsecond differences around the 0.266112 ms SDPA control do not
establish a stable improvement; chunk256/grid8x8 remains selected.

| SDPA change, same BFP4-QKVG family | Full-attention decode ms |
| --- | --- |
| Automatic chunk | [0.265995](logs/final_sdpa_chunk0_layer3.provenance.json) |
| Chunk128 | [0.270918](logs/final_sdpa_chunk128_layer3.provenance.json) |
| Chunk512 | [0.267998](logs/final_sdpa_chunk512_layer3.provenance.json) |
| Grid8x4 | [0.266026](logs/final_sdpa_grid8x4_layer3.provenance.json) |
| Grid11x10 | [0.266271](logs/final_sdpa_grid11x10_layer3.provenance.json) |

The [completed chunk1024 AutoFix](AUTOFIX_sdpa_chunk1024.md) identifies static-CB
overlap with live L1 allocations. Releasing projection intermediates and moving
Q/gate to DRAM was insufficient at cap16 during restored eager execution with
the trace still allocated. Reducing cap to eight cores also reduces SDPA CB19
scratch by 81920 bytes; all three adapted v2 cases then pass the unchanged
whole-layer and both restored-trace gates. At matched cap8, chunk1024 is
5.298 us slower than chunk256. The accepted adaptation supplies a measured
rejection; it does not require a general kernel fix or a default lifetime change.

| Lower-live-L1 SDPA adaptation | Full-attention decode ms |
| --- | --- |
| Cap8, chunk1024 | [0.276570](logs/sdpa_low_live_cap8_chunk1024_layer3.provenance.json) |
| Cap8, chunk256 | [0.271272](logs/sdpa_low_live_cap8_chunk256_layer3.provenance.json) |
| Cap16, chunk256 | [0.267339](logs/sdpa_low_live_cap16_chunk256_layer3.provenance.json) |

Residual cores [16 L0](logs/production_residual16_layer0.provenance.json)/
[16 L3](logs/production_residual16_layer3.provenance.json) pass at
0.357983/0.266831 ms; [64 L0](logs/production_residual64_layer0.provenance.json)/
[64 L3](logs/production_residual64_layer3.provenance.json) at
0.362908/0.268284 ms. These are cumulative-selection inputs, not promoted defaults.

## Closure

Hardware validation, paired final timing/profile accounting and artifact
integrity are complete. The independent [stage review](STAGE_REVIEW.md) returns
`clean-pass`; local checkpoint SHAs are in the [work log](work_log.md#local-checkpoints).
The [stage contract](stage_contract.md) remains the acceptance authority.

Only this evidence document was authored for this synthesis. No TTNN import,
device command, code/default change, or stage acceptance was performed here.
