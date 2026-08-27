# Optimized-decoder work log

## Scope, environment, and contract

- Model: `Qwen/Qwen3.8-Flash-Next`, checkpoint revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Stage-owned paths: `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`, `tests/test_optimized_decoder_perf.py`,
  and `doc/optimized_decoder/`.
- Target: one P300c Blackhole chip 0, worker grid `11x10`, TTNN
  `0.75.0rc10.dev880`. Chip 1 and all multichip/full-model/vLLM paths stayed
  outside the goal.
- Hardware commands sourced `../functional_decoder/ttenv.sh`, which selects
  this checkout, its Python environment, `TT_VISIBLE_DEVICES=0`, and the
  single-chip mesh descriptor. Bounded `tt-smi -ls --local` checks passed at
  start and after the final watcher run. One bounded `tt-smi -r` recovery was
  needed after auto-discovery produced an unusable physical-chip mapping;
  subsequent runs pinned the P150 single-chip mesh descriptor and remained
  healthy. A later global-environment launch failed during kernel compilation
  before model execution because it mixed an installed TTNN library with this
  checkout; all signoff evidence uses `python_env/bin/python` and repo-local
  TTNN.

`doc/context_contract.json` is unchanged. The selected dtype, layout, and cache
policy do not reduce capacity: maximum context 262144, page size 64, internal
chunk 128, and tested decode batch 32 remain valid. Logical prefill length is
never required to satisfy `seq_len % chunk_size == 0`.

## Measured workload

The signoff harness does not use zero weights or zero activations. It loads the
same real layer state through `load_real_layer_state` for fused and optimized
implementations. Input is deterministically built from checkpoint
`embed_tokens` rows 2026 through 2283; PLE uses independent checkpoint rows.
The common layer-0/layer-3 activation SHA-256 is
`635cca15a36d85c5053c7627812f17d47137691b1d7fdc08ddaf5417387417dc`;
layer 1 is
`9fa1eef6a6d2680fadb72066643dbca5bc280987cddbf61bc8e7a31cc0062530`.

The observer runs before timing/signposts and then restores the exact runtime
method. Its only host conversion is therefore unmeasured. `ForbidHostFallback`
wraps the measured prefill, trace capture, and trace replays.

Final optimized prefill routing is deterministic across seven samples:

| Layer | Global union | Per-32-row group unions |
| ---: | ---: | --- |
| 0 | 262 | 155, 162, 158, 145 |
| 1 | 325 | 165, 168, 166, 154 |
| 3 | 268 | 148, 143, 140, 140 |

The padded decode tile observer sees unions 19/20/20 for layers 0/1/3. That is
not the executed expert count: batch-one optimized decode takes row 0's ten
top-k IDs and weights on device and passes exactly those IDs to indexed sparse
gate/up and down matmuls. Prefill and `max_batch > 1` retain the exact dynamic
union scan with `nnz=None`; no unsafe report-side `nnz=10` assumption enters
runtime.

## Operation-topology audit before and during tuning

| Region | Measured topology/opportunity | Candidate or replacement | Action and evidence |
| --- | --- | --- | --- |
| Routed gate/up | Same-input packed sparse projection dominates scan decode | BFP8/HiFi2, BFP8/LoFi, BFP4/LoFi; 20/40 cores and K blocks; legal separate gate/up | Selected BFP4/LoFi packed 40 cores, block 16. Exact indexed packed median 1.090623 ms beats well-tuned correct separate 1.160033 ms. |
| Routed down | Sparse expert-major down matmul is the second material expert row | BFP4/LoFi 40/80 cores, blocks 5/10; indexed A+B sparse form | Selected 40 cores/block 5. A profiler-advised 20-core gate candidate passes PCC but is slower at 1.098534 ms. |
| Active experts | Fused scan computes a tile-wide union even for one logical decode row | On-device row-0 UINT16 top-k IDs; indexed B-sparse gate/up and A+B-sparse down; compact L1 | Selected for batch-one decode: 3.406965 to 1.090553 ms controlled median. Dynamic scan remains for prefill/batched semantics. |
| Hyperconnections | Packed down/inject and same-input up projections; residual/norm movement | Tune legal 1D/2D programs and L1 outputs; width-sharded residual/norm chain | Programs/L1 selected. Width-sharded chain was repaired and retried after its first grid error; the later TTNN tensor-allocation/lifetime failure was source-triaged. |
| GDN | Packed qkv/beta/decay, separate z/out, FP32 recurrence/composite | Per-role precision; 1D/2D programs; 55-core padded qkv; six-role DRAM-sharded matrix | BFP8/HiFi2 projections and layer-0-only 55-core packed qkv selected. Layer 1 at 55 cores is invalid numerically (decode PCC 0.98584318), so its automatic legal geometry remains. |
| PLE | Same-input packed key/value projection plus recurrent state | 100-core decode, `10x4x4` prefill, L1 output, exact DRAM-sharded decode | Interleaved/L1 program selected; DRAM-sharded candidate is slower. State semantics/dtype unchanged. |
| QSA projections | Packed q/k/v/gate/index input and attention output | BF16/BFP8/BFP4 fidelity sweep, 110/20-core programs, exact-final DRAM-sharded roles and their composition, index-query L1 | BF16/HiFi2 selected after isolated real PCC. DRAM-sharded QSA input and attention output both win and their composition is the fastest correct path; index-query L1 is slower. |
| Attention/cache | Native paged GQA SDPA with BF16 K/V/index cache | SDPA grid/q/k chunk sweep; exact-final BFP8 cache PCC, trace, and seven-run perf | Native composite retained. Prefill `11x10,q32,k64` and decode `11x10,q0,k32` win/legal. BFP8 cache is correct and faster in prefill, but BF16 has the fastest traced decode. |
| Prefill matmuls | Auto programs and DRAM intermediates | Large per-role 2D programs, block/divisibility retries, L1 output | Legal winners selected. Failed CB/divisibility candidates were adapted rather than dismissed at first error. |
| Decode matmuls | Auto/interleaved programs | Role-scoped 1D grids, L1 output, all six material DRAM-sharded roles plus the two winning attention roles together | Selected per-role 1D grids/L1 and both layer-3 DRAM-sharded attention projections. The other four roles are slower under the exact final fidelity policy. |
| Runtime movement | Tiled/RM recurrence, routing, cache and QSA head-grid boundaries | L1 chaining, width sharding, index-query L1, inspect all format/reshard rows | No host op in final windows. Required format/reshard rows and nine small interleaved/sharded bridges are quantified below; the four bridges added by the two selected DRAM matmuls are retained only because end-to-end traced decode wins. |

Packed expert projection is retained only after the legal, tuned separate
control lost. The other packed projection weights and composite splits are the
completed fused decoder's checkpoint topology, not newly introduced fusion;
this pass tunes their legal packed programs and avoids adding separate dispatch,
repeat input reads, and concatenation movement.

## Selected runtime policy

```text
expert policy: expert_bfp4_lofi_g40b16_d40b5
expert topology: packed
batch-one decode expert mode: indexed
prefill/batched expert mode: exact dynamic union scan
shared projections: bfp8_lofi
GDN projections: bfp8_hifi2
QSA input and attention output: bf16_hifi2
KV cache: bf16

decode 1D output: l1
layer-3 DRAM-sharded decode roles: qsa_input, attn_out
decode 1D cores:
  gdn_qkv_b_a@layer0=55, in_proj_z=48, gdn_out=20,
  attn_hc_down_inject=10, mlp_hc_down_inject=10,
  attn_hc_up=80, mlp_hc_up=80, moe_input=40,
  shared_down_proj=40, qsa_input=110, attn_out=20,
  ple_key_value=100

prefill output: l1
prefill 2D programs (grid x per-core-N-block x K-block):
  attn_hc_down_inject=11x4x8, mlp_hc_down_inject=11x4x8,
  gdn_out=10x4x8, qsa_input=11x4x2,
  attn_out=10x4x8, ple_key_value=10x4x4

prefill SDPA: grid 11x10, q_chunk=32, k_chunk=64
decode SDPA: grid 11x10, q_chunk=0, k_chunk=32
```

The batch-one indexed intermediates are compact and L1-resident: packed
gate/up is `10*32*1280*2 = 819200` bytes and down input is
`10*32*2560*2 = 1638400` bytes. The prefill scan's physical four-group packed
gate/up output would be `4*512*32*1280*2 = 167772160` bytes before other CBs
and buffers, so L1 placement is physically unavailable there; the scan path
uses DRAM while preserving exact dynamic routing.

## Candidate evidence and decisions

### Routed experts and same-input topology

All material precision acceptance used real checkpoint weights; random PCC did
not veto a real-weight winner.

| Candidate | Correctness | Traced decode median | Decision |
| --- | --- | ---: | --- |
| Dynamic scan, final precision/geometry | Same semantics | 3.406965 ms | Replaced for batch-one decode only. |
| Indexed packed 40-core/block-16 gate/up, 40-core/block-5 down | Final L0 PCC 0.99824739/0.99601841 | 1.090553 ms in scan control; 1.090623 ms in topology control | Selected. |
| Indexed separate gate/up, 20 cores/block 16 | All L0/L1/L3 real PCC pass | 1.160033 ms | Rejected; packed is 5.98% faster. |
| Indexed packed gate/up 20 cores/block 16 | L0 0.99620670/0.99588609 | 1.098534 ms | Rejected; correct but 0.72% slower. |

Earlier BFP8/HiFi2, BFP8/LoFi, BFP4/LoFi and 20/40/80-core sweeps are in
`candidate_expert_*`. The final controlled comparisons are
`autofix_expert_{indexed,scan}_real_decode_l0_count3.xml`,
`autofix_{packed,separate}_indexed_real_decode_l0_count3.xml`, and
`autofix_packed_indexed_g20b16_d40b5_real_decode_l0_count3.xml`.

### Precision and fidelity

- Shared BFP8/LoFi and GDN BFP8/HiFi2 passed cumulative real PCC and beat the
  BF16/BFP8 alternatives. GDN recurrence and state remain FP32/BF16.
- QSA BFP8/HiFi2, BFP8/LoFi, and BFP4/LoFi input/output trials fall below the
  representative acceptance bar, so the datatype remains BF16.
- With BF16 fixed, HiFi2 was isolated for QSA input and attention output. The
  combined real decode samples are 2.898105/2.898714/2.898494 ms versus HiFi4
  2.899751/2.900130/2.899714 ms; the sample ranges do not overlap. HiFi2 is
  selected. Its final layer-3 PCC after both sharded roles is
  0.99611741/0.99915755. The seven-run
  prefill median is 18 us slower than the prior HiFi4 signoff sample, a 0.037%
  trade for the best correct traced-decode candidate.
- The final-path BFP8 cache candidate used the selected BF16/HiFi2 QSA
  projections, both DRAM-sharded attention roles, identical real inputs, and
  100 trace replays. It passed layer-3 PCC at 0.99606544/0.99814188 and
  deterministic trace replay. Seven-run medians are 48.451365 ms prefill and
  2.867980 ms decode, versus BF16's 48.573433 and 2.866913 ms. Thus BFP8 wins
  prefill by 0.122068 ms but loses decode by 0.001067 ms (0.0372%); the BF16
  decode range 2.866728--2.867291 ms and BFP8 range
  2.867770--2.868404 ms do not overlap. BF16 is retained as the fastest
  correct traced-decode policy. Evidence:
  `autofix_cache_bfp8_exact_final_real_pcc_trace_l3.xml` and
  `autofix_cache_bfp8_exact_final_perf_l3_count7.xml`.

### Sharding and movement candidates

The independent rereview found that the original DRAM-sharded evidence used an
obsolete QSA HiFi4 policy. `$autofix` reran every material role with the exact
final BF16/HiFi2 attention policy, BFP8/HiFi2 GDN policy, selected expert
geometry, identical real weights/activations, and 100 traced replays. The two
attention roles win individually, so their composition was implemented and
measured rather than deferred.

| DRAM-sharded role | Candidate decode ms | Interleaved final/control ms | Decision |
| --- | ---: | ---: | --- |
| `ple_key_value` | 1.481124 | 1.455374 | Rejected; slower. |
| `in_proj_z` | 1.111638 | 1.090902 | Rejected; slower. |
| `gdn_out` | 1.114356 | 1.090902 | Rejected; slower. |
| `gdn_qkv_b_a` | 1.122247 | 1.090902 | Rejected; slower. |
| `attn_out` | 2.889402 | 2.898317 contemporaneous L3 control | Individual winner. |
| `qsa_input` | 2.876988 | 2.898317 contemporaneous L3 control | Individual winner. |
| `qsa_input,attn_out` | 2.866888 | 2.876958 seven-run QSA-only default | Selected; fastest correct composition. |

The first QSA sharding attempt exposed prefill use of a mutated sharded weight.
It was fixed with distinct prefill/interleaved and decode/sharded copies and
rerun under the exact final policy. Each selected role keeps its own activation
shard config/core count, and comma-separated roles compose without sharing an
invalid singular config. Final combined real PCC is
0.99611741/0.99915755. `autofix_dram_sharded_exact_final_hifi2_*` contains all
six isolated roles, the same-run control, the composition, and real-PCC
evidence. The failed mixed-installed/repo environment launch is retained as
infrastructure evidence and is not a candidate rejection.

For width-sharded residual/norm chaining, attempt 1 found an inconsistent shard
grid. After fixing the bounding box and output grid, the repaired attempt
reached `Tensor is not allocated` at the chained allocation/lifetime boundary.
That run was captured with `tt-triage`; device cores were idle while dispatch
waited downstream, and `AUTOTRIAGE.md` identifies the host tensor lifetime
failure. The candidate code was removed. Evidence:
`autofix_width_sharded_residual_indexed_l0_attempt1.xml` and
`triage/width_sharded_attempt2/{tt-triage.txt,AUTOTRIAGE.md}`.

For QSA, forcing post-RoPE index query to L1 is correct
(0.99612647/0.99912781) but is slower: 2.903632 versus 2.899714 ms controlled
median. It was removed.

### Program and SDPA sweeps

- Decode core winners are the values in the selected policy. The layer-0-only
  55-core qkv candidate pads N to a legal 10560, slices the logical 10336
  output, and passes at 0.99824739/0.99601948. Applying it to layer 1 gives
  decode PCC 0.98584318, so scope is explicit.
- Prefill block/core candidates that hit K divisibility or CB capacity errors
  were retried with legal blocks. Selected QSA uses K block 2, PLE block 4,
  and the other material roles block 8. L1 output advice won and was promoted.
- Controlled real QSA prefill SDPA medians: q32/k32 48.605036 ms, q32/k64
  48.502865 ms, q32/k128 48.533627 ms. q32/k64 is selected.
- Controlled decode q0/k32 is 2.899783 ms. k64 and k128 both violate the
  composite's exact gathered-mask width 2080 divisibility constraint; q0/k32
  is the legal winner and public non-aligned sequence support remains intact.

## Final correctness and semantics

All rows instantiate the exact optimized class/default. Results:

- Normal gate with `TT_METAL_TRACE_ALLOC_TRACKING=1`: 35 passed, seven long
  tests deselected, 125.95 s.
- Exact promoted layer-3 capacity gate: full 262144, non-aligned 262143, and
  traced decode at the advertised maximum all pass, 256.31 s. The earlier
  all-layer long-context suite also passed all seven marked cases; only layer 3
  needed repetition after adding the second sharded setup weight.
- Trace stress: nine passed (three repeats for each representative kind),
  56.92 s.
- Watcher gate: 35 passed, seven long tests deselected, 171.07 s. A broad log
  search found no error/assert/exception/hang/deadlock/timeout/stuck or
  NoC/semaphore/CB failure signature; dump 2 completed at 171.810 s and device
  0 detached at 173.916 s. Post-run `tt-smi -ls --local` showed both P300c
  boards visible and resettable.

| Layer | Kind | fused prefill | final prefill | delta | fused decode | final decode | delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 0.99842572 | 0.99824739 | -0.00017833 | 0.99702853 | 0.99601841 | -0.00101012 |
| 1 | PLE + GDN | 0.99911219 | 0.99837846 | -0.00073373 | 0.99988902 | 0.99922472 | -0.00066430 |
| 3 | QSA | 0.99668270 | 0.99611741 | -0.00056529 | 0.99977344 | 0.99915755 | -0.00061589 |

Paging coverage includes shuffled page-table invariance, independent per-user
tables/positions, underfilled selected-token multisets, two-user prefill then
batched decode, batch-32 decode, and QSA trace replay at current position
262143. Determinism checks compare repeated trace outputs and cache state.

## Final like-for-like performance

Both implementations use the real workload described above. Each median is
seven independent warmed samples; each decode sample averages 100 trace
replays.

| Layer | fused prefill ms | final prefill ms | prefill win | fused decode ms | final decode ms | decode win |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 31.864180 | 22.504434 | 29.374% | 4.090832 | 1.090618 | 73.340% |
| 1 | 36.066132 | 25.594588 | 29.034% | 4.554379 | 1.455579 | 68.040% |
| 3 | 56.996406 | 48.573433 | 14.778% | 5.777473 | 2.866913 | 50.378% |

The result beats the correct fused baseline and every retained correct
traced-decode candidate; no op-count or topology-only proxy is used.

## Final `tt-perf-report`, accounting, and roofline

All six captures use real weights/inputs, signposts, and no host ops. The
unchanged layer-0/layer-1 captures use the same shipped policy; layer 3 was
recaptured after promoting both DRAM-sharded roles. Decode reports use the ten
actually indexed experts. Prefill group unions/totals are layer 0
155/162/158/145 = 620, layer 1 165/168/166/154 = 653, and layer 3
148/143/140/140 = 571. Because `tt-perf-report` accepts only one integer per
group, the scalar averages 155/163/143 model totals 620/652/572: exact for
layer 0 and one-row rounding for layers 1/3. They do not claim top-k=10 or the
maximum group union for every prefill group. A single traced decode replay
keeps attribution complete.
Layer-3 decode emitted a profiler-buffer-full warning after the delimited
signpost, but every one of its 234 report rows has device time and window
accounting reconciles.

| Layer | Mode | Profiled host ms | Device sum ms | Gaps ms | residual ms | report rows | modeled DRAM |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | prefill | 22.807508 | 22.084258 | 0.546694 | 0.176556 | 118 | 15.9%, 82 GB/s |
| 0 | decode | 1.178049 | 1.060899 | 0.087901 | 0.029249 | 132 | 19.0%, 97 GB/s |
| 1 | prefill | 25.918687 | 25.187850 | 0.515870 | 0.214967 | 181 | 14.9%, 76 GB/s |
| 1 | decode | 1.552323 | 1.402573 | 0.120868 | 0.028882 | 176 | 19.1%, 98 GB/s |
| 3 | prefill | 48.956668 | 48.269088 | 0.517290 | 0.170290 | 220 | 6.9%, 35 GB/s |
| 3 | decode | 3.004818 | 2.552567 | 0.418620 | 0.033631 | 234 | 11.3%, 58 GB/s |

Residual is profiled host minus device sum and op gaps; it covers trace replay
synchronization and profiler bookkeeping, not host fallback. Clean-run medians
are lower because profiling is disabled.

For a conservative decode lower bound, packed tile storage at ten experts is:

```text
common fixed + experts = 22,635,520 + 2,764,800*U bytes
L0 fixed (including padded 10560 qkv) = 91,397,120 bytes
L1 fixed = 130,191,360 bytes
L3 fixed = 130,098,176 bytes
```

At `U=10` and 512 GB/s this gives 119,045,120/157,839,360/157,746,176
bytes and 0.232510/0.308280/0.308098 ms for layers 0/1/3. Final profiler
device sums are 4.563x/4.550x/8.285x these weight-only limits; profiled host
is 5.067x/5.035x/9.753x. The remaining distance is explained by many small
dispatches, routing/gather/reduction, recurrence/cache work, and QSA head/index
movement rather than an untried single dominant matmul setting.

Actionable report advice was closed as follows:

- L1 intermediate outputs: tested, won, selected.
- DRAM-sharded decode: all six material roles were rerun under the exact final
  policy; QSA input and attention output win separately and together and are
  selected, while the other four are slower.
- Sparse subblock/core advice: exact 20-core/block-16 gate candidate is correct
  but slower; 40-core/block-16 remains.
- Lower attention fidelity: BF16/HiFi2 is correct, faster, and selected.
- Large prefill programs and SDPA chunks: legal controlled winners selected;
  invalid choices retried/adapted.
- Width-sharded residual/norm: the grid failure was repaired and the subsequent
  implementation was source-triaged, establishing the current TTNN
  allocation/lifetime blocker.

### Final movement audit

| Layer | Mode | format/reshard rows | device time us | Contents |
| ---: | --- | ---: | ---: | --- |
| 0 | prefill | 20 | 442.191 | Required GDN RM/tiled recurrence and public-boundary tilize/untilize/typecast. |
| 0 | decode | 12 | 66.983 | Required recurrence/public-boundary tilize/untilize. |
| 1 | prefill | 35 | 1007.322 | PLE+GDN recurrence and packed-state format boundaries. |
| 1 | decode | 12 | 68.812 | Required recurrence/public-boundary tilize/untilize. |
| 3 | prefill | 44 | 792.698 | QSA rotary/cache/index and public-boundary formats. |
| 3 | decode | 49 | 188.279 | QSA formats plus one required head/cache-grid reshard. |

There are zero torch, `from_torch`, `to_torch`, or host-fallback rows in final
signpost windows. Layer-3 decode additionally has five
`InterleavedToSharded` and four `ShardedToInterleaved` bridge rows totaling
9.092 us device time. Four of those bridge rows enter/leave the two selected
DRAM-sharded matmuls; they are necessary under the TTNN DRAM-sharded API and
are retained only because the composition wins end-to-end. The remaining
bridges are QSA/SDPA layout boundaries. No avoidable conversion remains on the
measured path.

## Commands

All signoff commands used the repo-local Python/TTNN environment and explicit
single-chip topology:

```bash
export PYTHONPATH=/home/ttuser/dev/qwen3.8-flash-next/tt-metal
export TT_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/dev/qwen3.8-flash-next/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

# Comparable fused and optimized performance.
QWEN38_OPT_PERF_IMPLEMENTATION=fused QWEN38_OPT_PERF_DECODE_REPLAYS=100 \
python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder_perf.py \
  --junitxml=.../final_real_fused_perf_count7.xml

QWEN38_OPT_PERF_DECODE_REPLAYS=100 \
python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder_perf.py \
  --junitxml=.../final_real_optimized_dram_qsa_attn_perf_count7.xml

# Exact-final reduced-cache candidate: repeat the optimized command for layer
# 3 with QWEN38_OPT_CACHE_POLICY=bfp8 and retain seven 100-replay samples.
QWEN38_OPT_CACHE_POLICY=bfp8 QWEN38_OPT_PERF_LAYERS=3 \
QWEN38_OPT_PERF_DECODE_REPLAYS=100 \
python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder_perf.py \
  --junitxml=.../autofix_cache_bfp8_exact_final_perf_l3_count7.xml

QWEN38_OPT_CACHE_POLICY=bfp8 TT_METAL_TRACE_ALLOC_TRACKING=1 \
python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all \
  '.../test_optimized_decoder.py::test_real_weights_hf_prefill_decode_pcc[3]' \
  '.../test_optimized_decoder.py::test_decode_trace_replay_and_determinism[3]' \
  --junitxml=.../autofix_cache_bfp8_exact_final_real_pcc_trace_l3.xml

# Correctness, capacity, stress, watcher.
TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m pytest -q \
  --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder.py \
  -m 'not long_context' \
  --junitxml=.../final_dram_qsa_attn_correctness_trace_alloc.xml

python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all \
  --long-context \
  '.../test_optimized_decoder.py::test_full_advertised_context[3]' \
  '.../test_optimized_decoder.py::test_near_max_non_aligned_context[3]' \
  .../test_optimized_decoder.py::test_qsa_traced_decode_at_advertised_context \
  --junitxml=.../final_dram_qsa_attn_long_context.xml

python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all --count=3 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder.py \
  -k decode_trace_replay_and_determinism \
  --junitxml=.../final_dram_qsa_attn_stress.xml

TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=.../watcher_dram_qsa_attn_final \
python_env/bin/python -m pytest -q --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder.py \
  -m 'not long_context' --junitxml=.../final_dram_qsa_attn_watcher.xml

# Capture exact-final layer-3 windows; repeat with mode=prefill and its temp dir.
QWEN38_OPT_PERF_MODE=decode QWEN38_OPT_PERF_LAYERS=3 \
QWEN38_OPT_PERF_DECODE_REPLAYS=1 \
python_env/bin/python -m tracy -r --check-exit-code \
  -o /tmp/qwen38_profile_dram_qsa_attn_ohbbJ3 -m pytest -q \
  --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_optimized_decoder_perf.py

tt-perf-report .../tracy_dram_qsa_attn_final/layer3_qsa/decode_ops.csv \
  --start-signpost OPT_PERF_DECODE_L3 \
  --end-signpost OPT_PERF_DECODE_L3_END \
  --no-color --no-host-ops --active-experts 10 \
  --csv .../layer3_qsa/decode_perf_report.csv \
  --summary-file .../layer3_qsa/decode_perf_report_stacked

tt-perf-report .../tracy_dram_qsa_attn_final/layer3_qsa/prefill_ops.csv \
  --start-signpost OPT_PERF_PREFILL_L3 \
  --end-signpost OPT_PERF_PREFILL_L3_END \
  --no-color --no-host-ops --active-experts 143 \
  --csv .../layer3_qsa/prefill_perf_report.csv \
  --summary-file .../layer3_qsa/prefill_perf_report_stacked
```

Exact-final layer-3 raw profiler databases remain at
`/tmp/qwen38_profile_dram_qsa_attn_ohbbJ3` (decode) and
`/tmp/qwen38_profile_dram_qsa_attn_prefill_ZO15nS` (prefill). Earlier layer
0/1 raw paths are recorded with their capture artifacts. Compact ops/report
CSVs have no missing device-time rows and are checked in; raw data is
reproducible with the commands above.

## Exact final artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `final_dram_qsa_attn_correctness_trace_alloc.xml` | `1d62f65fe02c5517c3085c21734bb705fde4be5e2bf090052f830500958e7f23` |
| `final_dram_qsa_attn_long_context.xml` | `5d0dc0cda5d6b5df232bfa037cc0cf1678f5ec276a84bfc970bedfbe8c403d64` |
| `final_dram_qsa_attn_stress.xml` | `cafae0bfd2c3b857bcc9c869cf0a23c4cd6d5a2c2443042c2e1c7c661cf7dfdd` |
| `final_dram_qsa_attn_watcher.xml` | `df83c7b1b7c310a39e8ed52b2ed77b9016bab03ab2006802e1545d2ea70764d9` |
| `final_real_fused_perf_count7.xml` | `b1997a59a1a69128e71206d70c7c402deda6e4788145ac4534d064236bc2a62f` |
| `final_real_optimized_dram_qsa_attn_perf_count7.xml` | `8b542b82d7083945b5b8c824be575ff3a63d7f7b7ca32d4f55f6d85b5e076ddd` |
| Exact-final BFP8-cache PCC/trace | `933a14ac8b343b6044bcf20db64e493632a5b357b20214497ee42f12a05dc8e4` |
| Exact-final BFP8-cache seven-run perf | `6f505aa711c42a49fa11ab706fd0b7a666dd20eb7c12fa080cbf468f737b0113` |
| `watcher_dram_qsa_attn_final/generated/watcher/watcher.log` | `c3ebb919dedf579fa689b615ed6965f6848eeebbdc20fc98bf94625eb0405c3e` |
| Combined DRAM-sharded candidate count3 / real PCC | `011a4f4e1f0ca791dd118e455f97d00a2ca2ff0acbfb3f1cbca03dce0eeb9662` / `f5ee537cbc2845365143a1a13c15cc3fcb3859998fb2ce6be631e8f0b2f0ac6b` |
| L0 prefill/decode ops CSV | `1cafeb50c2a6123ca27e51e832732cb26ecd97589a234c9773c7911848aa6fc3` / `af5d12764a1312ca06b37444bcd717cff138bd1ba2fa861707842d1f1328688c` |
| L1 prefill/decode ops CSV | `f7b1909522ffa5d68f9f24e67354f2c1d66dd87a5cccad96bbc986fe21066949` / `bdc89b4a11110c4531b0bdecdea0aa45d6200cd8d7e3fd8531cfcad0006b3eee` |
| Exact-final L3 prefill/decode ops CSV | `50c234fff8d40d851e721a62327b5b64f2a17cce84ed8d2dbb83dc71728e5cc6` / `86b8d63ad59565482d05ea58a3915e5aad62fee7cecfb8d75e9264940d5bd1a9` |
| Exact-final L3 prefill/decode report CSV | `a5b1b6c405afa94869584f95100b427c40a13f132676c9ad095f40609c43071b` / `39ef8a152e0398c318bb6e5f332210d8dddb2e211c5678e1862515bd7fc149d6` |

## `$optimize` checklist

- [x] Exact target shapes and all meaningful layer kinds use real checkpoint
  weights for final PCC and material candidate acceptance.
- [x] The pass began with an operation-topology audit covering repeated
  same-input projections, packed/separate candidates, composite ops, movement,
  sharding, and sparse expert execution.
- [x] Canonical per-subgraph datatype/fidelity policy is evidence-backed;
  material BFP4/LoFi expert geometries were tested with real weights.
- [x] Sharded layouts, all material DRAM-sharded decode roles, L1/DRAM output,
  large prefill programs, 1D decode programs, SDPA, cache dtype, memory config,
  compute kernels, and active-expert execution were addressed now.
- [x] First API/runtime errors were repaired and retried; width sharding has a
  repaired attempt plus `tt-triage`, and DRAM sharding was rerun after the
  dual-copy repair.
- [x] Packed expert projection beats a legal tuned separate candidate; no
  newly introduced packed projection is kept on topology alone.
- [x] Native paged SDPA and optimized sparse composite ops remain on device;
  there is no material host replacement.
- [x] Public non-aligned lengths, paged cache semantics, determinism, batch 32,
  layer-kind coverage, and the full context contract are exercised.
- [x] Final prefill/decode PCC is at least 0.995 for each layer kind and all
  deltas are quantified.
- [x] Seven-run warmed prefill and 100-replay traced decode beat the comparable
  real-weight fused baseline for every layer kind and the best correct
  candidate.
- [x] Final prefill/decode `tt-perf-report` CSV/tables and advice conclusions
  exist for all three kinds; roofline and device/gap/host accounting reconcile.
- [x] Final measured windows have no host fallback or torch conversion;
  required format/reshard/DRAM-shard bridge rows are quantified and the added
  bridges have net-win evidence.
- [x] Repeated stress, trace allocation tracking, full/near-maximum context,
  watcher-clean correctness, and post-run health checks pass.
- [x] Capacity is unchanged, so no context-contract edit or capability
  reduction is required.
- [x] No applicable single-device decoder optimization is deferred to a later
  stage. Multichip, full-model, and serving work remain out of scope.
- [x] Independent `$stage-review` clean-pass recorded below.
- [ ] Stage-owned local commit SHA recorded below; nothing pushed.

## Review and commits

The first independent review returned `more-work-needed` for uncontrolled
packed/separate comparison, incomplete sharding/movement/fidelity/SDPA evidence,
non-comparable replay counts, zero-input performance, and missing accounting.
`$autofix` addressed those findings. The next fresh review found one remaining
issue: the six retained DRAM-sharded rows were not measured under the exact
final HiFi2 attention policy. `$autofix` reran the full matrix, discovered the
two attention winners, tested their composition, promoted the combined winner,
and repeated PCC, seven-run performance, trace-allocation, maximum-context,
stress, watcher, and profiler gates. A subsequent fresh review found that the
BFP8-cache dismissal was still based on the older path. `$autofix` reran the
exact-final candidate with real PCC, deterministic trace, and seven 100-replay
samples; it is correct but loses traced decode to BF16 with non-overlapping
sample ranges. The final fresh `$stage-review` independently recomputed the
cache and baseline medians, checked the exact optimized path, gates, hashes,
profiler/advice evidence, context contract, and candidate closures, and
returned `clean-pass` with no required work. The stage-owned commit is pending.
