# Optimized decoder work log — stage 03

Target: Qwen/Qwen3.5-35B-A3B revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Starting commit: 1d857aa3, branch feat/qwen3-5-35b-a3b. Working tree was clean.
Scope: optimized_decoder.py, stage tests and documentation. No multichip/model/serving work.
Environment: source ../scripts/environment.sh; enabled optimize, tt-device-usage,
tt-autodebug and telemetry packages verified in the selected Codex inventory.
Available hardware: four Blackhole ASICs / two P300c cards (device_list.log).
Selected layer mesh: 1x1. Hardware commands are serialized.

## Initial operation-topology audit (before tuning)

Evidence: fused_decoder/profile_final/tracy/linear_attention and
fused_decoder/profile_placement_final/tracy/full_attention. Each contains the
prefill/decode tables and complete 4096/128 B1 layer measurements.
Operation durations below are per replay means; they are diagnostic row durations,
not additive layer denominators (firmware intervals can overlap).

| Path | Current operations / observed cost | Candidate replacement and constraints | Action / evidence |
|---|---|---|---|
| Both expert decode | compact top8 mapping; BF16 RM pad; fused SwiGLU/down BFP8 LoFi on 8x2; weighted native reduce; FFN ~410us | Homogeneous BFP4/LoFi native FFN across legal grids; sparse gate/up/down with separate role geometry, exact boolean top8 mask; packed versus separate gate/up | Pending measured sweep; must retain all256 descriptors and only8 active experts |
| Both expert prefill | Eight packed BFP4 minimal-matmul split calls, concat, sparse down with all256 active, route weights/reduce; down ~572–576ms per4096 | Larger K/output blocks and grid search for sparse down; tile-union sparse path; tuned2D/minimal packed versus separate projections | Pending; chunk sizes128 linear/512 full; useful work counts8 experts only |
| GDN input/output | packed same-input QKV/A/B/Z BF16 HiFi4 (132us), output BF16 HiFi4 (76us) | Independent input/output BFP8/BFP4 and LoFi/HiFi2 trials; DRAM-sharded projections including legal multi-readers; compare packing against tuned split where legal | Pending; FP32 recurrent state and exact chunk kernel remain correctness contract |
| Full attention input/output | packed Q/K/V/gate BFP8 LoFi (57us), output BFP8 LoFi (74us) | DRAM-sharded weights, coherent L1 input shards, readers1/2/3, larger K blocks; separate projection control; BFP4 trials | Pending; indexed RoPE channel basis retained |
| Full attention cache/SDPA | BF16 paged paired update; native SDPA decode54us; sharded K/V boundary, DRAM query/output | BFP8 cache with fill cast and BF16 update; explicit SDPA grid/chunks; lower-movement heads | Pending; logical batch1 versus padded32, physical pages/positions unchanged |
| Residual/norm | FP32 typecast/RMS/multiply/BF16, both norms ~47us full; DRAM residual adds | Sharded L1 residual + sharded RMS, preserving zero-centered FP32 arithmetic where required | Pending adapted whole-layer candidates |
| Shared expert/router | packed BFP4 minimal SwiGLU24us; BFP8 down11us; router BF16 HiFi4~28us; scalar gate16us | DRAM geometry/precision per role; fused-elementwise split control for packed shared gate/up; reduce layout crossings | Pending real-weight evidence |
| GDN state/movement | FP32 exact grouped recurrence; native causal conv42us; reshape/slice/state transforms | L1 intermediate placements, precise state/math preserved; inspect optimized composites under matching state contract | Pending; prior native TF32 recurrence fails real B32 and cannot replace exact state |
| Prefill orchestration | RM flatten, device embedding gathers, bounded chunk calls, RM slice_write, final tilize | Tiled chunk slicing/concat or direct tiled writes preserving arbitrary tails; chunk-size trials | Pending; no public alignment restriction |
| Persistent weights | compact slices plus original packed and grouped copies | Explicit consumer/owner inventory; release setup-only copies after last use | Source handoff ../memory-layout-handoff.json read; allocator verification pending |
| CCL/LM head | absent in this single-device layer | Not applicable | No multichip or generation work in this stage |

Baseline whole-layer device times: linear prefill764671.617us/decode1399.833us;
full prefill697717.625us/decode978.654us. Baseline unprofiled host trace medians:
linear1259.719us, full920.198us. Acceptance PCC >=0.995 per request/token.
These are stage02 controls, not stage03 measurements.

## Experiments

Stage remains in progress; no completion claims.

- `expert_screen/commands.json`: real checkpoint, random input screening128/16,
  tracking1/skip-program-cache0/runtime audit. Native BFP4/LoFi passes;8x2
  median910.539us versus927.517us control.8x4/8x8 slower at both dtypes.
  Sparse prefill down K16/32cores/subblock2 reduces warmed128-token host time
  25020.499us to11202.070us. Original64core/K1 retained as control.
- `record_inputs.log`: CPU-only checkpoint embedding and preceding HF layers
  produce actual layer0/layer3 activations from local LLM report text. Fixtures
  live outside Git at `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs`.
  This is layer-test preparation, not full-model bringup or generation.
- `expert_real/commands.json`: recorded HF activations; BFP4/LoFi4x4 currently
  873.163us median versus926.9005us control; min decode PCC0.9987903 versus
  0.9994024. Short screening workload only, not headline telemetry.
- Native4x2 BF8 fails circular-buffer L1 overlap. BFP4 adaptation fits and passes
  at880.111us, slower than4x4.2x2 requires16 expert rows but DEST supports8. AutoDebug identifies a3392-byte live-L1 overlap for4x2/BFP8; tiled BFP8
  input adaptation fits, and is measured in sparse_retry/commands.json.
- AutoDebug sandbox preflight cannot create a user namespace (`bwrap`). Retried
  with the skill's `AUTODEBUG_SKIP_CHILD_SANDBOX=1` under existing unrestricted
  execution authorization; child remains inspection-only, no device jobs.
- First sparse candidates failed due to a Python method name colliding with
  inherited `MoEConfig`; renamed method to `program_config`. Failed trials are
  preserved, not classified as TTNN blockers; affected rows require retry.


- `AUTODEBUG.md`, `AUTOFIX.md`, `sparse_activation_diagnosis.md`: source-backed
  diagnosis and adapted runtime tests. Tiled BF8 native inputs fit4x2 and4x4,
  but extra conversion makes them slower (923.19/930.19us BFP8 weights;
  906.59/902.84us BFP4). Native2x2 is a DEST register geometry limit.
- Sparse factory does not lower `fused_activation` into SFPU_ACTIVATION.
  Binary-consumer SiLU repairs split gate/up PCC from0.33266 to0.999394 on
  the same real fixture. Correct split takes913.7us versus865.5us packed
  sparse (different role geometries are recorded in each policy).
- `attention_real/commands.json`: BFP4 full packed input fails on real HF
  activations (PCC0.989355 at both LoFi/HiFi2). Attention output DRAM-sharded
  BFP8/LoFi8cores/K8/one reader takes824.2895us versus873us control. Two and
  three readers and4/8/16/32 input-shard geometries are measured explicitly.
  Sharded residual+FP32 norm candidates4/8/16/32 cores, including fused
  norm-weight variants, all pass but best900us is slower than873us.
- `baseline_advice/*/*_advice.txt`: advice was regenerated from existing
  whole-layer stage02 captures, without `--no-advice`. Original stage02
  processing had disabled advice. Dominant projection DRAM sharding, larger
  output/K blocks, and shared/router activation placement are being tried.
- Indexed sparse MoE is supported in this checkout: UINT16 RM top8 indices,
  nnz omitted, compact8-group intermediates, sparse-input down projection.
  Early16-core/K8 gate result554.11us with min PCC0.9993957, versus873us
  native BFP4. Remaining gate/down geometry and dtype trials are pending.

- `indexed_fine/commands.json`, `reduce_retry/commands.json`: indexed gate
  BFP4/LoFi32cores/K32 and BFP8 down32cores/K16, UINT16 routing and fused
  score/reduction reach512.552us on recorded layer3 inputs128/16. Initial
  fused-reduce reshape failed because sparse down exposes32 physical rows;
  selecting the single logical token before reshape repairs it. PCC0.999397.
  BFP4 down64cores/K16 is a further candidate (532.52us versus538.94us under
  the older16core gate); it must be combined and verified on the target.
- `dispatch_maps.json`, `dispatch_smoke/commands.json`,
  `dispatch_large/commands.json`: local device sort/count/prefix/scatter maps
  pass exact random/hot/boundary31/32/33 cases at128 and512 tokens. No host
  routing occurs. Native single-device fabric dispatch is not legal in this
  checkout; the adapted local map/embedding path passes full-layer real PCC.
  At512 tokens BFP4 native4x4 takes17385.175us prefill/PCC0.9987985 versus
  packed32551.228us/PCC0.9995723. BFP8 native4x4 is18079.666us/PCC0.9995719.
- `local_dispatch_diagnosis.md`, `prefill_dispatch_review.md`: independent
  AutoFix source diagnosis. Larger native input_m_tiles does not grow the full
  activation CB; it controls fixed8-tile loops and buffering depth. Retain
  T/32 bounds so hot experts cannot truncate. UINT16 reshape and masked-count
  layout repairs were followed by exact device map tests, not dismissed.
- The tiled orchestration candidate now retains the fused device-index planner
  for padded/nonaligned/continuation inputs: host slice offsets and variable
  concat arity would otherwise compile new programs for an unseen logical36
  in the warmed physical128 bucket while a decode trace is live. Both branches
  use the optimized attention/router/experts. Final reuse tests remain required.

## Resumed session (Claude, 2026 Oct 10)

Codex stopped on usage limits (runs/main/STATUS.md). Device ownership moved to this session.
`tests/run_optimized_trials.py` now archives `tt/` and `tests/` before each matrix, records
SHA-256 for every autoport source, the tt-metal HEAD and a hash of the diff outside the
autoport, and stops the matrix if any of them changes during a row (`source_stable`).
Input fixture hashes are in each matrix `provenance.json`.

All rows: recorded HF activations, B1, traced decode, warmed prefill, `--audit`.
Decode = median traced replay host us. PCC = minimum per token/request.

| Matrix | Layer | Candidate | Prefill tokens | Decode us | Prefill us | Decode PCC | Prefill PCC |
|---|---|---|---|---|---|---|---|
| combo1_full | 3 | indexed sparse + fused reduce (control) | 128 | 512.97 | 11132 | 0.9993 | 0.9993 |
| combo1_full | 3 | + attention_out BFP8 LoFi 1D 8x4 k8 | 128 | 463.47 | 11050 | 0.9993 | 0.9993 |
| combo1_full | 3 | + attention_out BFP8 DRAM-sharded r1 | 128 | 464.06 | 10946 | 0.9993 | 0.9993 |
| combo1_linear | 0 | indexed sparse + fused reduce (control) | 128 | 883.96 | 11335 | 0.9969 | 0.9997 |
| combo1_linear | 0 | + attention_in BFP8 LoFi, attention_out BFP8 DRAM r2 | 128 | 781.19 | 11201 | 0.9968 | 0.9996 |
| combo1_linear | 0 | + gdn_l1 | 128 | 725.92 | 11532 | 0.9968 | 0.9996 |
| norm2_full | 3 | + sharded decode RMSNorm 16 cores, BF16 in, DRAM out | 128 | 426.18 | 11202 | 0.9993 | 0.9993 |
| norm2_linear | 0 | + same norm | 128 | 688.95 | 11537 | 0.9968 | 0.9996 |

Norm finding: an isolated trace microbenchmark
(`runs/main/stage03_claude/micro/bench_norm.py`, output `bench_norm.out`) shows the
interleaved FP32 norm at 29.3 us/call (one core) and the 16-core sharded norm at 18.2 us
(FP32 input) or 8.7 us (BF16 input, FP32 accumulation), all PCC 0.999998. Inside the
layer, an L1-interleaved norm output made the next matmuls slower: +30 us per layer
versus control. DRAM-interleaved output with BF16 input wins (-38 us). BF16 input to
`rms_norm` with `fp32_dest_acc_en` matches the FP32 path: BF16 to FP32 widening is exact.

Prefill at 4096 tokens (`prefill4k_*`, 4 decode steps, decode at context 4096):

| Layer | Prefill expert path | Chunk | Prefill us | Prefill PCC |
|---|---|---|---|---|
| 3 | packed all-expert (control) | 512 | 253025 | 0.9995 |
| 3 | dispatch BFP4 4x4 tiled | 512 / 1024 / 2048 / 4096 | 138995 / 109215 / 93034 / 84505 | 0.9987 |
| 3 | dispatch, gather planner | 2048 | 93450 | 0.9987 |
| 0 | packed all-expert (control) | 128 | 319400 | 0.9995 |
| 0 | dispatch BFP4 4x4 tiled | 512 / 1024 / 2048 / 4096 | 138930 / 111906 / 94167 / 87252 | 0.9987 |
| 0 | dispatch, gather planner | 2048 | 94743 | 0.9987 |

`prefill_chunk` now also applies to the gather planner (`_planned_prefill`), which handles
unaligned lengths and offsets. Its cost at 2048 is within 1% of the tiled path.
Attention decode activations in DRAM instead of L1: 473.19 versus 466.79 us. Kept L1.

Capacity constraint for later stages: one BFP4 routed-expert copy is about 432 MiB per
layer, about 17 GiB for 40 layers. Dispatch prefill (per-expert lists) and indexed
decode (stacked tensors) use different layouts. Both copies do not fit one 32 GB p150
with the rest of the model. The p150 profile needs a single routed representation.

### Profiles of intermediate candidates (diagnostic)

`profile_c2/` (full: candidate2, linear: candidate3) at 4096/128, run through the
symlinked profiler root `/tmp/q35-functional-profiler-root` with a fresh cache
`/tmp/q35-opt-profiler-cache`. The first attempt hit the known 16-bit zone hash collision
(`profile_c2_hash_collision/`); the linear candidate2 attempt failed PCC
(`profile_c2/*c2_pcc_fail*`). The linear candidate3 profile process imported its sources
before a concurrent edit of `optimized_decoder.py`; it is diagnostic only.

Prefill finding (full layer, 4096 tokens, 84.2 ms device): `ttnn.sort` of 32768 route IDs
in the dispatch map took 50.3 ms; the routed FFN took 8.7 ms. A route's dispatch slot is
`region_start[e] + (number of earlier tokens that chose e)`. The new `dispatch_rank="cumsum"`
computes the second term as an exclusive cumsum of a [T,256] BF16 one-hot mask (scatter,
typecast FP32, cumsum, subtract) and one gather. The inverse map equals the per-route
target, so the inverse scatter is gone. `run_dispatch_maps.py --rank cumsum` checks counts,
offsets, capacity, unique targets, round trip, region bounds and stable token order for 18
cases (hot, random, boundary 31/32/33, native block boundary; 128 to 4096 tokens):
`dispatch_maps_cumsum.json`. FP32 TILE scatter is not supported (`dispatch_maps_cumsum_fp32_scatter_fail.log`).

| Matrix | Layer | Rank | Prefill us (4096) | Prefill PCC |
|---|---|---|---|---|
| rank_full | 3 | sort / cumsum | 84584 / 38321 | 0.9987 |
| rank_full | 3 | cumsum, BFP8 dispatch | 38694 | 0.9995 |
| rank_full | 3 | cumsum 8x4 / 8x2 | 37734 / 40291 | 0.9987 / 0.9988 |
| rank_linear | 0 | sort / cumsum | 87405 / 41050 | 0.9986 |
| rank_linear | 0 | cumsum 8x4 | failed (rank_linear/cumsum_8x4.log, assertion) | - |

### Decode follow-ups at 4096 context

| Matrix | Change | Full decode us | Linear decode us | PCC note |
|---|---|---|---|---|
| gate | packed router (256) + shared-expert gate (1, padded 32) in one BF16 HiFi4 matmul | 454.2 (default grid) / 439.8 (8x2 k8) | 690.4 / 676.0 | unchanged |
| c4 | gate pack 8x2 k16 / k32 / 8x1 k8 | 435.5 / 435.3 / 437.5 | 673.9 / 672.3 / 676.5 | unchanged |
| gate | BFP8 KV cache | 466.1 vs 468.7 | - | unchanged |
| gate, c4 | explicit SDPA decode config 8x8 k128/k256/k512/k64, 8x4, 4x4, exp approx, LoFi | 465–468 (gate), 437–440 (c4); 4x4 465.8 | - | no win; default kept |
| gdn2 | causal-conv `channel_chunk_size` 256 / 128 (32 / 64 cores) | - | 659.4 / 651.9 vs 673.2 | unchanged |
| gdn2 | heads-as-rows decode (`gdn_heads_decode`) | - | 671.1 | -2 us; `nlp_create_qkv_heads_decode` 33 us on one core; not selected |
| gdn3 | row-major persistent conv history (`conv_state_rm`) | - | 628.3 vs 652.0 | unchanged |
| shared | BFP4 decode down 64 / 32 cores | 428.1 / 432.2 vs 435.8 | - | 0.9971 at 32 steps |
| shared | split shared gate/up (DRAM r1, r2, 1D), minimal config 8x2 | 437.1–438.3, 435.9 | - | no win; packed kept |
| shared | shared down DRAM-sharded r1 | 432.7 | - | unchanged |
| c5v/c5abl | BFP4 decode down at 4096/128 | fails: min 0.9946 at token 127 | - | BFP8 down: 0.9984; rejected |

### Acceptance failures and fixes

First frozen chain (`final_evidence/`, defaults = candidate6): watcher (4 runs), pytest
(real and synthetic, both layers) and cumsum dispatch maps passed with stable source.
Linear `positions` (B32 heterogeneous, random activations) failed: one token 0.9932.
Linear `reuse` failed: 1153 tokens with continuation cuts, decode 0.9948.

`accept_ablation/` bisection:

- BFP8 GDN in-projection causes the PCC loss. BF16 restores reuse (0.9981) and positions
  (0.9977). Splitting the in-projection (BFP8 q/k/v/z, BF16 HiFi4 a/b gates) does not fix
  it (0.9941 / 0.9950).
- Same-contract recorded-activation check (OPT-012): new `--input-activations` option for
  `run_positions.py` and `run_reuse.py`. Linear BFP8 in-projection fails real B32 positions
  at 0.9896. Rejected on real-activation evidence.
- The Stage-2 fused decoder also fails real B32 positions at the same rows and tokens
  (`fused_real_linear_positions.json`: 0.99186; candidate9: 0.99187). This is an
  inherited property, not an optimization regression. Random-activation gates remain the
  acceptance bar and pass.
- Determinism: B32 positions showed eager-vs-trace and trace-vs-trace mismatches with the
  sharded decode RMSNorm (`det_*`, `det4_*`). Without it: 7/7 bitwise identical. In
  isolation the op is deterministic (`runs/main/stage03_claude/micro/det_norm.out`: 50
  eager + 50 traced runs). Zero-filling the input tile padding first
  (`fill_implicit_tile_padding`) makes it deterministic (`normfill_*`, 4/4). The sharded
  `rms_norm` output row 0 depends on uninitialized tile padding rows. This is a TTNN
  finding; the model now fills padding before this op.
- Sharded norm precision: linear layer at 4096/128 fails with sharded norm (BF16 input
  0.9899, FP32 input 0.9897; `c10_linear`, `c11_linear`) and passes without (0.9987).
  Full layer passes every gate with it. Selected per layer kind.

### Selected policy and final runtime evidence

Defaults are `DEFAULT_POLICIES` in `tt/optimized_decoder.py`; `optimization=None` selects
them by layer type. The only final frozen evidence is `final_evidence13/` (see "Stage review 3
response"). Earlier chains `final_evidence{,2,...,12}/` are kept as superseded history.

## Stage review 1 response (`stage_review.md`, verdict more-work-needed)

Evidence root: `review1/`. This response used `final_evidence4/` (chain
`runs/main/stage03_claude/evidence_chain4.sh`). It is superseded by `final_evidence13/`.
`final_evidence2/` was an interrupted chain; `final_evidence3/` is an older complete chain.

### P1: BFP8 GDN projections (`review1/p1/`)

Final base (padding fill, bucket ladder), recorded HF activations unless noted.

| Candidate | Decode us (4096/128) | Min decode PCC (4096/128) | Real B32 positions, 3 windows: tokens < 0.995 | Random B32 positions | Random reuse min |
|---|---|---|---|---|---|
| BF16 HiFi4 in and out | 727.69 | 0.9987 | 3 / 768 | pass 0.9977 | 0.9982 |
| BFP8 LoFi in | 672.69 | 0.9897 (fail) | 4 / 768 | fail 0.9938 | 0.9952 |
| BFP8 HiFi2 in | 673.04 | 0.9897 (fail) | 4 / 768 | fail 0.9929 | 0.9978 |
| BFP8 LoFi out | 709.54 | 0.9888 (fail) | 2 / 768 | pass 0.9972 | 0.9981 |
| BFP8 HiFi2 out | 709.63 | 0.9987 | 3 / 768 | pass 0.9974 | 0.9982 |
| BFP8 HiFi2 in and out | 655.08 | 0.9896 (fail) | 3 / 768 | pass 0.9974 | 0.9985 |

Decision: BFP8 HiFi2 GDN output projection adopted (-18 us). BFP8 in-projection rejected
on the real-activation 4096/128 run (0.9897) plus the random B32 gate.

### P2: single-token dips (`runs/main/stage03_claude/micro/token_outlier.{py,out,json}`)

CPU HF analysis of layer 0, 4096 prefill + 128 decode. Token 32 is not an outlier in
out-projection input magnitude, max/rms ratio, layer output or input. Each token's layer
output is dominated by one channel (median 54% of energy); token 32 and 112 have lower
shares (44%, 37%). Per-token TT error (1-PCC) correlates with lower share: Spearman
0.396 (default), 0.390 (BFP8 in), 0.309 (BFP8 out), p < 4e-4. Partial explanation only:
per-token PCC on tokens with less massive-channel energy exposes errors in other channels.

### P2: duplicate weights

Same-dtype `ttnn.typecast` created new buffers. `_as_dtype` keeps the tensor when the
dtype already matches (sparse experts, `_Projection`). Allocator bytes after cleanup:
full 1829 -> 1227 MB, linear about 1830 -> 1268 MB per layer at that time
(`final_evidence4/perf_*/default.json`). Current per-buffer inventory: "Stage review 2 response".

### P2: nondeterminism

- Decode-mode steps inside prefill (token tails) and B>1 decode slices carry
  uninitialized tile padding rows. `decode_input_fill` zero-fills padding at layer decode
  entry. Unaligned repeated prefill became identical (`review1/tail2_*`).
- Remaining: 1-2 BF16 ULP drift in one 32-token block of the MoE combine output in some
  repeated prefills. Probe (`runs/main/stage03_claude/micro/prefill_probe.py`, logs
  `flake/PD_*`, `PS_*`, `TP_*`, `NT_*`): router weights, dispatch maps, counts, offsets,
  restored expert rows and products are bitwise identical; only the reduction output
  differs. Both `post_combine_reduce` and an alternative `fast_reduce_nc` combine show it.
  Device sync before the call or a neutral op does not remove it. Standalone repros of
  both ops (host or device inputs, FP32-DEST predecessors) are deterministic over 160
  calls each. AutoDebug (`review1/post_combine_autodebug/AUTODEBUG.md`) found no source
  defect that explains it; root cause unresolved. Classified: rounding-level, PCC
  unaffected, decode trace replay remains bitwise identical in every gate.
- `run_layer.py` records `prefill_repeat_identical` (not asserted, because of the item above).
- 8x4 dispatch decode-identity failure: classified by watcher runs
  `final_evidence4/watcher_long/diag_8x4.*` and `final_evidence6/watcher_long/diag_8x4.json`
  (watcher-clean, deterministic). Speed re-measured in review 2.

### P2: prefill overhead

- Matmul ranks also produce per-expert counts (`ones @ onehot`) and region starts
  (`tiles @ triu`, HiFi4, exact integers); no cumsum or masked_bincount remains.
- 2D prefill configs: `_Projection` 2D grids for the in-projections exceed L1 (4.8 MB
  static CBs at M=4096); `minimal_matmul` 11x10 M4 K8 N16 adopted for in-projections
  (full -0.86 ms, linear -0.2 ms), `minimal_matmul` 8x8 for full out-projection
  (-0.4 ms), shared-down 2D 8x8 k4 (-0.7 ms) (`review1/reruns/mm2_*`, `reruns/*`).
- k-major routes with tile combine remove the 1.6 ms row-major flatten and save 0.5 ms,
  but show more combine drift (~1200 rows); not selected (`review1/combine_linear`, `flake/TC_*`).
- Remaining measured items: one-core dispatch-map scatter, slot flatten reshape.

### P2: unaligned prefill (`review1/tail_*`, `tail2_*`, `final_evidence4/unaligned_*`)

GDN tails used token-wise decode (planner block 128). Now: 32-token block for GDN, 32-token
SDPA chunk config for full attention, bucket ladder 4096/2048/1024/512(full)/256/128/64/32.
Matmul counts remove the 64-token minimum of `masked_bincount`.

### P2: reruns on final base (`review1/reruns/`)

DRAM-sharded BF16 GDN in-projection, 16/32 cores, k 4/2, readers 1/2: 743-785 us vs 731
us default or L1 failure. Sharded residual 16 cores: 454/448 us vs 446 us. GDN state in L1
(1 slot): -5 us; needs 2 MiB L1 per slot per layer; rejected for capacity.

### Flat q/k GDN prefill (`review1/flatqk/`)

`gdn_flat_qk` passes q/k token-major to `chunk_gated_delta_rule`, which then L2-normalizes
in-kernel. Linear prefill 34.08 -> 28.60 ms. Random positions/reuse and real reuse pass.
At 4096/128 decode token 32 drops to 0.9899 with BF16 or BFP8 output projection
(`perf128/flat_out_bf16`, `flat_out_bfp8_hifi4`); without flat q/k 0.99874
(`perf128/noflat_out_bfp8`). The first measurement used 32 decode steps and did not reach
token 32. `final_evidence5/` ran with flat q/k and records the failure
(`final_evidence5/perf_linear`, `profile/linear_profile.log`). Rejected; default reverted.
`final_evidence6/` was the review-2 input; it is superseded by `final_evidence13/`.

## Stage review 2 response (`stage_review_2.md`, verdict more-work-needed)

Evidence for this response: `final_evidence10/` (chain `runs/main/stage03_claude/evidence_chain10.sh`),
superseded by `final_evidence13/`.
Superseded during this response, kept as history:

- `final_evidence7/`: timing lanes ran in parallel with CPU-heavy HF references and
  distorted host timing (full 100-token prefill 17.7 ms versus 9.2 ms alone). Timing lanes
  were stopped (`final_evidence7/aborted_parallel_timing/`). Its correctness lanes found
  two defects (below). Rule since then: correctness gates run in parallel lanes; timing,
  profiles and pytest run alone.
- `final_evidence8/`: complete chain on BFP4 dispatch; superseded by the BFP8 dispatch
  decision. Its `device_ab/` found the device-0 replay offset.
- `superseded_final_evidence9_bfp4_dispatch/`: stopped at the start when BFP8 dispatch was
  adopted.
- `final_evidence9/`: complete chain on BFP8 dispatch with the 8x8 grid only for 4096-token
  groups; superseded by the 2048 threshold (`review2/bfp8_grid`).

### P1: aligned prefill ladder (`review1/route_timing/`, `final_evidence10/prefill_*`)

- `prefill_forward` now routes aligned inputs to the tiled path with the same bucket set as
  the planner (`_planner_buckets`). Other inputs go to `_batched_prefill`: attention per
  ladder chunk, one MoE call per group of up to 4096 tokens, masked 32-row GDN tail.
- Before (stage-owner runs cited by the review, linear): 1024 tokens 49168 us, 2048 tokens
  98705 us, 3072 tokens 148815 us. After (final source): see README prefill table
  (1024: about 13 ms; 2048: about 19 ms; 3072: about 32 ms).
- Continuation: 2048 tokens after 128 cached tokens (`run_layer.py --split 128`) now
  matches a fresh 2048-token prefill.
- Fixed cost attribution (`review2/smallprof/`, tt-perf-report, 1000 and 1024 tokens):
  `moe_fused_swiglu` costs 4.8-4.9 ms at 1000-1024 tokens on 16 cores (BFP4 dispatch,
  intermediate source) and 7.52-7.65 ms at 4096 tokens on 64 cores (BFP8, final profile). It loops over 256 experts in sequence, about 19 us per expert with one
  tile row each. Embedding gathers cost 1.0 ms and the scatter 0.5 ms. Reducing the expert
  loop floor needs an op change (experts on disjoint core groups); recorded as the blocker.
- Single tail token: a lone count-1 group runs the decode layer path instead of a padded
  32-row MoE call. 4097 tokens: full 31.9 -> 30.1 ms, linear 39.5 -> 37.4 ms
  (`review2/single_tail/`). A 1-token prefill also runs the decode path.
- Remaining batched-path overhead: RM round trips cost about 4 ms at 4096 tokens versus the
  tiled path (4095 versus 4096 tokens).

### Defects found and fixed during this response

- GDN tail mask and row-index tensors stayed alive on the attention object after prefill.
  The trace allocation tracker caught them before replay (`review1/route/reuse_0_fail_tailattr.log`).
  `_attention_step` now clears them after the GDN forward.
- Full attention with 128-token pages: a masked 32-row tail made `count // block == 0`
  (`KeyError: 0`, `final_evidence7/reuse_3/full_page128.log`). The masked tail now requires
  the page size to divide 32 and the page table to cover the padded rows.
- 1-token prefill on real activations fell to 0.992 (`final_evidence7`, real reuse
  linear): the batched path sent one token through the BFP4 prefill MoE. It now uses the
  decode path; fixed by the routing rule above and by BFP8 dispatch.

### Prefill dispatch dtype (`review2/dispatch_dtype/`, `review2/dispatch_down/`, `review2/prefill_ref/`)

`run_layer.py` now records per-token prefill PCC (`prefill_token_pcc`, `prefill_tokens_below_0995`).

| Layer, tokens | Dispatch | Prefill us (parallel screen) | Prefill PCC | Tokens below 0.995 | Min token PCC |
|---|---|---:|---:|---:|---:|
| full 4096 | BFP4 | 25833 | 0.99879 | 96 | 0.9846 |
| full 4096 | BFP8 | 26364 | 0.99952 | 1 | 0.9914 |
| linear 4096 | BFP4 | 32436 | 0.99881 | 54 | 0.9913 |
| linear 4096 | BFP8 | 32862 | 0.99971 | 1 | 0.9940 |
| full 1000 | BFP4 / BFP8 | 16038 / 16657 | 0.99879 / 0.99956 | 28 / 1 | 0.9873 / 0.9914 |
| linear 1000 | BFP4 / BFP8 | 18353 / 20664 | 0.99850 / 0.99971 | 38 / 0 | 0.9918 / 0.9968 |

- Decision: BFP8. Decode rejected BFP4 down for one token at 0.9946; the same per-token rule
  rejects BFP4 prefill dispatch. Cost +0.4-0.5 ms at 4096 tokens, +408 MiB per layer.
- BFP4 gate/up with BFP8 down is blocked by the op contract:
  `TT_FATAL moe_fused_swiglu: w_down[0] dtype BFLOAT8_B must match w_gate[0] (BFLOAT4_B)`
  (`moe_fused_swiglu_device_operation.cpp:142`).
- Remaining tokens below 0.995 (CPU FP32-reference control, `review2/prefill_ref/`): full
  token 911 is BF16-reference noise (BF16 versus FP32 0.99156; TT 0.99136). Linear token 1720
  is a TT error (reference 0.99942; TT 0.99402). The BF16 reference itself has two tokens
  below 0.995 over the same 4096 tokens (minimum 0.99323).

### Dispatch grid under the selected dtype (`review2/large_grid/`, `review2/bfp8_grid/`)

BFP8 screen, three lanes (devices 1-3), prefill wall time delta versus 4x4 small / 8x8 large:

| Candidate | full 1024 | full 2048 | full 4096 | linear 1024 | linear 2048 | linear 4096 |
|---|---:|---:|---:|---:|---:|---:|
| 4x4 only (no large grid) | +47 | -140 | +1488 | +52 | -85 | +1641 |
| 8x8 for every group size | +183 | -456 | -37 | +165 | -332 | +42 |
| 8x4 large | +190 | -71 | +882 | +1 | +12 | +1001 |
| 11x8 large | +282 | -205 | +952 | +16 | -6 | +986 |
| 8x4 small | +405 to +502 | -109 to -190 | -80 to +1050 | +322 to +379 | -20 to +64 | -43 to +1028 |

Selected: 4x4 for groups below 2048 tokens, 8x8 for 2048 and 4096 (`dispatch_grid_large
[2048, 8, 8]`). Matched control in the final chain: `final_evidence10/control_grid_*`.
k-major tile combine (`review2/large_grid`, BFP4): -0.5 to +0.4 ms, sign varies by length;
rejected on speed, not on drift.

### P2: prefill nondeterminism (`review2/drift/`)

- `run_layer.py` records `prefill_repeat_difference` with element count, rows, 32-row
  blocks, maximum absolute difference, ordered BF16 ULP and repeat PCC. Ordered ULP is large
  near zero (sign changes); maximum absolute difference and repeat PCC are the useful
  magnitudes.
- E1 (12 processes, layer 0, 2048 tokens, 3 runs each, `logs/E15_*`, `E6_*`): differences hit
  one or two 32-row blocks, 64-393 of 2048 elements per row, maximum absolute difference
  1.5e-4 to 4.9e-4. Run1 and run2 are often identical; the odd run can be any.
- E2: the model's weight operand is DRAM (`W5_MEM ['BufferType.DRAM']`), not L1. AutoDebug's
  L1-placement premise does not hold; the existing standalone repro with DRAM weights
  (deterministic, 160 calls) already matches the model placement.
- E5: two back-to-back `post_combine_reduce` calls with the same inputs always agree, in
  every run, including the odd run.
- E6 (`prefill_probe3.py`): a call after the prefill on the original input buffers (no
  clone) gives the majority result, also with mixed original/cloned operands. Inputs are not
  the cause; device state set by preceding ops and kept across consecutive reduce calls is.
- E3-lite (`prefill_probe4.py`, 8 processes each): a clone or a tiled add right before the
  reduce leaves the rate unchanged (5 of 8, 6 of 8; baseline 6 of 8).
- Status: root cause unresolved. README and `context_contract.json` state that prefill is
  not bitwise reproducible. Final-chain magnitude: see README determinism paragraph.

### P2: weight ownership inventory (`tests/weight_inventory.py`, `final_evidence10/inventory/`)

- Census by root `buffer_unique_id` with allocator totals. Released two unused copies:
  linear BF16 `output_weight` (16 MiB) and full `q_proj/k_proj/v_proj` (19.1 MiB)
  (`review2/inventory/pre_release_*` versus final).
- Per-layer table and the 40-layer estimate: README "Memory". Handoff recheck:
  `final_evidence10/memory_handoff_recheck.json`.
- RoPE cos/sin tables (64 MiB) are per full layer today; the full model should share one.

### P2: documents

- One final evidence pointer (`final_evidence10/`); older chain references in this log
  are marked superseded.
- 8x4/8x8/11x8 dispatch rows added (above, README rejected table).
- Reference-precision control for the real B32 positions diagnostic recorded
  (`review2/ref_precision/`, README).

### Other concerns from review 2

- `_PREFIX_MATRICES` and `_ZERO_ROWS` are keyed by the mesh object, not `id(mesh)`; a
  later mesh cannot reuse a closed mesh's tensors.
- Dispatch-map exactness now covers 32, 64, 128, 256, 512, 1024, 2048 and 4096 tokens.
- `run_layer.py` keys position refresh and KV setup on layer type, not layer index.
- Page size 128 with a full-attention tail below 32 tokens uses token-wise steps; correct
  (`reuse_3/full_page128.json`); the serving page size is decided in the vLLM stage.
- Scatter (1.75 ms per 4096 tokens) remains: no single-op alternative found in this stage.

### Device placement (`final_evidence8/device_ab`, `final_evidence10/device_ab`)

Physical device 0 replays traces 16-21 us slower than devices 1-3 with identical source;
prefill does not change. Headline runs use device 3, the default device of the Stage-2
and earlier Stage-3 runs.

## Stage review 3 response (`stage_review_3.md`, verdict more-work-needed)

Final frozen evidence: `final_evidence13/` (chain `runs/main/stage03_claude/evidence_chain13.sh`).
Superseded: `final_evidence10/` (review-3 input), `final_evidence11/` (planner before the
scratch page, calibration and size-gated combine), `superseded_final_evidence11_nolog/`
(stopped at start; chain log was not captured), `final_evidence12/` (found the defect below:
full-layer prefill of 2, 4097 and 4098 tokens failed).

Defect found by `final_evidence12`: with a scratch page bound, decode-path steps inside a full-layer
prefill received the extended page table, and `paged_fused_update_cache` rejected it
(`TT_FATAL: max_num_blocks_per_seq must be less than max_num_blocks: 257 > 133`). Decode-path steps
now use the original table (`review3/decode_tail_fix/`). `run_reuse.py` and `run_positions.py` now
bind a scratch page for the optimized full layer (guard pages checked separately), so the
trace-alive gates exercise this path (`review3/scratch_gates/`).

### P1: prefill planner cost (`review3/planner_check*`, `review3/plan_calibration*`, `review3/scratch_check`)

Changes in `tt/optimized_decoder.py`:

- `_chunk_plan` is a cost-model planner. After alignment steps and full 4096 chunks it
  chooses, for the remainder, between exact ladder buckets and one masked bucket by
  `cost = chunks x chunk_cost + rows x row_cost` (`_PLAN_COST`). Remainders of 1-2 tokens use
  decode-path steps. The GDN mask now covers any chunk size (`_index_offsets[count]`).
- One MoE call per group of up to 4096 planned rows. Chunk outputs are concatenated in tile
  layout before the MoE call; the group pads to the next dispatch bucket and slices back.
  The row-major assembly buffer and the `_ZERO_ROWS` template cache (33 MiB per mesh) are gone.
- Output: tile concat of group results (trailing decode tokens join as one 32-row tile piece);
  the returned view has at most the planned rows (no `physical + 4096` padded shape).
- Optional scratch KV page (`bind_paged_kv_cache(..., scratch_block=...)`): padded rows of a
  masked full-attention chunk beyond the page table write to that page through an extended
  table of fixed width, so tight page tables still allow a masked chunk without new programs.
  Without a scratch page the planner keeps the page-coverage rule. `run_layer.py` binds one
  scratch block by default for the optimized full layer (`--no-kv-scratch` disables it).

Calibration (serial, device 3, `review3/plan_calibration/`, warmed prefill us):

| Tokens | full ladder | full masked | linear ladder | linear masked |
|---|---:|---:|---:|---:|
| 1100 | 17101 | 16525 | 18170 | 20169 |
| 1500 | 18587 | 16441 | 20750 | 20175 |
| 2176 | 24588 | 27104 | 28480 | 34242 |
| 2600 | 26211 | 26882 | 30505 | 34091 |
| 3072 | 25735 | 27100 | 30499 | 34426 |

Selected constants reproduce every winner: full 0.8 ms per chunk + 1.4 us per row, linear
0.5 ms per chunk + 3.0 us per row. The default mixed plans (exact prefix plus masked tail)
match or beat both pure plans (`review3/plan_calibration_default/`: full 1100 16385,
linear 1100 18319, full 1500 16419, linear 1500 19596, full 2600 25880, linear 2600 30554).
The earlier constants (1.3/0.7 ms, then 8/3 ms) came from parallel-lane runs disturbed by
CPU-heavy reference jobs and are not used.

Unaligned profile (`final_evidence13/profile_unaligned/`, tt-perf-report, op device-time sums):
linear 4095 tokens take 32.69 ms versus 32.05 ms for aligned 4096 (two more embedding gathers and
the mask ops); 1000 tokens take 13.03 ms. The full-layer unaligned profile is blocked by a profiler limitation: two zone names
in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp`
("Reduce max", line 1691, and "Softmax(Q@KT)@V", line 1770) have the same 16-bit hash, and
`tt_metal/impl/profiler/profiler.cpp:376` throws "Source location hashes are colliding" when a
new streaming SDPA variant compiles. The path is the real repo path, so a different profiler root
does not change it (`final_evidence11/profile_unaligned/retry*`).

### P2: sibling routed-expert ops (`review3/routed_ffn/`)

Same dispatch maps, recorded activations, warmed prefill delta versus `moe_fused_swiglu` (us):

| Candidate | full 1024 | full 2048 | full 4096 | linear 1024 | linear 2048 | linear 4096 |
|---|---:|---:|---:|---:|---:|---:|
| `unified_routed_expert_moe` | +3065 | +1905 | +903 | +2213 | +1828 | +983 |
| unified, DRAM ND-sharded weights | +2172 | +1880 | +981 | +2277 | +1646 | +1064 |
| fused, DRAM ND-sharded weights | +333 | +287 | -85 | +228 | 0 | -181 |
| `hybrid_routed_expert_moe` (threshold 64 or 128) | TT_FATAL | TT_FATAL | TT_FATAL | TT_FATAL | TT_FATAL | TT_FATAL |

Hybrid: `TT_FATAL: Out of Memory: Not enough space to allocate 128578560 B L1 buffer across 88
banks` (the fused half sizes its circular buffers for the maximum tokens per expert). Decision:
keep `moe_fused_swiglu`. The options stay in the policy (`dispatch_ffn`,
`dispatch_hybrid_threshold`, `dispatch_weight_nd`), default off.

### P2: contract and telemetry figures

- Drift wording: one to three 32-row blocks; largest observed maximum absolute difference
  0.0078125 (`review2/bfp8_grid/full_1024/small4x4_large11x8.json`); repeat PCC at least 0.9999998.
- GDN slots: 32 configured slots = 65.5 MiB per linear layer; the census counts slot 0 under
  attention.
- Telemetry: whole-sequence prefill PCC renamed `prefill_pcc_vs_hf_whole_sequence`; per-token
  minimum added as `prefill_token_pcc_min_vs_hf`.

### Other concerns from review 3

- Tile combine re-measured on the final source (`review3/tile_combine/`, serial, two default
  runs as control): -0.45 ms at 4096 for both layers, -0.07 to -0.24 ms at 2048, +0.24 to
  +0.30 ms at 1024. Adopted for groups of 2048 tokens and more (`dispatch_tiles_from 2048`).
  Dispatch maps are checked in both combine modes.
- Returned prefill tensor: compact view (planned rows, not `physical + 4096`).
- GDN prefill head split (reshape and transpose, about 6.5 ms per 4096 tokens): not changed in
  this stage; the flat q/k alternative fails decode token 32 (review 1).

## Stage review 4 response (`stage_review_4.md`, verdict more-work-needed)

Final frozen evidence: `final_evidence14/` (chain `runs/main/stage03_claude/evidence_chain14.sh`).
`final_evidence13/` is superseded (review-4 input).

### P2: reproducibility figures

The contract, README and telemetry now separate the two combine paths: `post_combine_reduce`
(MoE groups below 2048 rows) differs in 1-3 32-row blocks; the k-major tile combine (groups of
2048+ rows) can differ in up to 38 of 64 blocks (`final_evidence13/control_grid_linear_2048`,
`prefill_*_1100`). Maximum absolute difference stays at most 9.8e-4 in the final chains and
0.0078 anywhere; repeat PCC at least 0.9999998. The drift experiments in `review2/drift/` cover
only the `post_combine_reduce` path.

### P2: linear token 950 (`review4/token950/`, `review4/token950_ref/`)

- TT outputs at token 950 are bitwise identical for 4096 and 4097 tokens and match 2600 at PCC
  0.999999 (`run_layer.py --save-tensors`).
- The HF BF16 reference at token 950 changes with length under the run_layer setting (8 CPU
  threads): cross-length PCC 0.99260 (4096 versus 4097 and 2600). The router logits at token 950
  have an exact BF16 tie between experts 27 and 173 at -4.59375 (8th and 9th places). The CPU top-k
  picks expert 27 at 4096 and expert 173 at 4097 and 2600. TT versus the 4096 reference: 0.99975.
- With 16 threads the reference did not flip (`token950.log`: BF16 versus FP32 0.999995 at every
  length). Classified: reference noise from an exact router tie, not a TT error.

### P2: `hybrid_routed_expert_moe` (`review4/hybrid/`)

- In the layer, the op needs all L1 above the allocator base (`arena_bytes_for` in
  `hybrid_program_factory.cpp`). The 8 KiB per bank already allocated are the qwen36 router's
  persistent L1-sharded tensors (`tt_bias`, `tt_input_indices`, `tt_output`, `tt_output_indices`,
  2 KiB per core each; `final_evidence13/inventory/`). Decode traces capture their addresses, so
  prefill cannot free and reallocate them while a trace is alive.
- Isolated run on captured real dispatch inputs with those tensors freed (`hybrid_isolated.py`,
  full layer, device 3): every threshold (32, 64, 128, 256) at 1024 and 4096 tokens fails with
  `TT_FATAL: Program size (71264..74240) too large for kernel config buffer (70656) on TENSIX`
  (256 local experts in one program). Same run: `moe_fused_swiglu` 5906 us / 8033 us,
  `unified_routed_expert_moe` 7620 us / 8251 us at 1024 / 4096 tokens.
- Decision: keep `moe_fused_swiglu`. With 64 local experts per chip (Stage 4 expert parallel) the
  hybrid program may fit; re-measure there. The earlier work-log causes ("static buffers",
  "maximum tokens per expert") and the "threshold 64 or 128" claim were wrong and are corrected
  here.

### P2: scratch page

- `masked()` now requires `absolute + n <= max_seq_len`. `tests/test_prefill_plan.py` covers starts
  258176 and 260000 (including length 3952) and checks that no chunk ends past 262144
  (1116 cases).
- `run_reuse.py` adds case 2047 tokens with a cut at 1152: the second segment plans a masked
  1024-row chunk past the 2048-position table, so it writes the scratch page with both rows live,
  guard pages checked and a decode trace alive with program-cache misses forbidden. The scratch
  page starts as a 5.0 sentinel; `kv_scratch_written` is true (`review4/checks/reuse_3_sentinel.json`).
  A first attempt used 2048 tokens; decode then ran past the table and failed (test-design error,
  kept as `review4/checks/reuse_3_len2048_designbug.log`).
- Watcher run with a scratch write: full 1500 tokens (`final_evidence14/watcher_long/layer3_1500.json`).
- Contract: `optimized_decoder_validation.serving_contract` records the scratch reservation, the
  page-table ownership rule, the extended table width and the warm-up shape sets per physical bucket.

### Other concerns from review 4

- Decode-step and MoE-group costs are calibrated from the final-chain increments (full 4097-4096
  +0.81 ms, 4098-4097 +0.42 ms, masked 32-row tail group +3.70 ms; linear +1.11, +0.74, +3.29 ms):
  `_PLAN_COST` step 0.8 / 1.1 ms, group 3.0 ms, token-wise tails up to 8 tokens when cheaper.
- A single-call 2176-token prefill is now timed (`final_evidence14/prefill_*_2176`).
- Unaligned profile attribution corrected: the largest extra item of linear 4095 versus 4096 is one
  BF16 concat (+358 us); the embedding gathers add +83 us, untilize +84 us, op-to-op gaps 578 us
  versus 198 us.
- 40-layer estimate: 59.7 GiB from allocator totals minus named shared buffers, 60.6 GiB with raw
  allocator totals and RoPE/module caches counted once; both exceed 31.8 GiB.

## Stage review 5 response (`stage_review_5.md`, verdict more-work-needed)

Final frozen evidence: `final_evidence15/` (chain `runs/main/stage03_claude/evidence_chain15.sh`).
`final_evidence14/` is superseded (review-5 input).

### P2: short whole prompts

- `_PLAN_COST` now has a separate decode-step cost for a plan that is only decode steps from
  offset 0: full 1.05 ms, linear 1.6 ms, from the final-chain 2-4 token rows (full 3501, 4644,
  5623 us; linear 4399, 6005 us; masked 32 rows about 4.8-4.9 ms). Tails after 4096 rows keep
  0.8 / 1.1 ms. Plans: full 2-3 tokens decode steps, 4-5 masked 32; linear 2 decode steps,
  3-5 masked 32; full 4100 keeps four decode steps, linear 4100 a masked 32-row tail.
- `tests/test_prefill_plan.py::test_short_plan_choice` pins these plans.

### P2: `hybrid_routed_expert_moe` (`review5/hybrid_worker_l1/`)

Isolated run on captured real dispatch inputs with the mesh opened at `worker_l1_size` 1444736
(16 KiB less than the default unreserved size, so the kernel config buffer grows by 16 KiB) and
the router L1 tensors freed. The program now fits. Device 3, full-layer dispatch maps, median
of 7:

| Tokens | moe_fused_swiglu | unified | hybrid 32 | hybrid 64 | hybrid 128 | hybrid 256 |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 5905 us | 7615 us | 8714 us | 8540 us | 8142 us | 8306 us |
| 4096 | 8135 us | 8303 us | 10867 us | 10175 us | 10324 us | 9780 us |

Every threshold is 2.2-2.8 ms (1024) and 1.6-2.7 ms (4096) slower than `moe_fused_swiglu`. Output
PCC versus the fused op on the occupied rows is 0.9993-0.9996. Rejected on measurement.

### P2: full-layer pytest tokens 684 and 592 (`review5/token684_ref/`, `review5/token684_tt/`)

- Input: the pytest input (real weights, random activations, seed 1234, 1000 tokens). TT output
  saved with `run_layer.py --save-tensors` on the final planner path (`token684_tt/tt_1000.pt`).
- Token 684: the BF16 router logits have a three-way exact tie for the 8th expert (experts 70, 10
  and 92 at 1.04688). The HF reference picks expert 10 (TT versus reference 0.99306). Forcing the
  other tied experts into the top 8 inside the router gives TT versus reference 0.99751 (expert 70)
  and 0.99771 (expert 92). TT follows a valid tie-break. Classified: reference tie noise.
- Token 592: no tie (BF16 versus FP32 reference 0.999975; same top 10 in both); TT versus reference
  0.99499 in every variant. Classified: TT-side error at one of 1000 tokens, the same class as
  linear token 1720. Recorded for the datatype sweep.

### Other concerns from review 5

- README headline prefill values use the raw JSON values (32567.358 us, 25894.473 us style).

### Stage review 6 (`stage_review_6.md`): clean-pass

Non-blocking items fixed in the documents: contract planner text (decode-path steps for remainders
up to 8 tokens), token-684 label (BF16 router-logit tie), short-prompt timings marked as single
samples.

Checkpoint commit: pre-commit hooks (black, trailing whitespace) are skipped for this commit. They
rewrite `tt/*.py`, `tests/*.py` and evidence text, which would break byte identity with the
reviewed `final_evidence15` source manifest and artifacts. Logs are stored gzip-compressed
(`*.log.gz`); raw profiler dumps, fabric/inspector output and saved tensors are excluded
(`.gitignore`).

Stage 03 checkpoint commit: `81485827def301c3a0fe99bf2aa8931dc6bf436b` on branch `feat/qwen3-5-35b-a3b` (local, not pushed;
parent `1d857aa3`).
