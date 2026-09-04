# AutoDebug: remaining linear-attention graph fusions

Inspection-only report, 2026-09-04. Target: `ornith-ai/Ornith-1.5-9B`, fused-decoder stage, branch `hous/ornith-1.5-9b`, starting HEAD `ae18ba18bdde6ae20dd628d3f2fd8f59c42ecaf3`. No devices opened, implementation changed, or measurements performed by this investigator. Read root/model AGENTS, autofix/autodebug/graph-fusing skills, both decoder implementations, paired tests, existing logs, profiler tables, operator implementations/tests, and the pinned 35B reference. Full attention is outside this report.

## Finding

The small improvement is explained by the current graph: the linear mixer still inherits all functional GDN methods. Its only implemented fusion is the shared MLP's SiLU folded into multiply. Several substantial applicable fusions remain, including an existing dedicated flat DeltaNet interface. This is incomplete graph-fusion coverage, not evidence that linear attention cannot be accelerated.

`logs/dedicated_pair.log:35` supplies the strongest current comparison: real-weight functional/fused PCC is effectively 1.0 for prefill, decode and replay; five-sample medians are 40.635466/40.245919 ms prefill and 1.623659/1.619570 ms traced decode. The latter saves 4.09 microseconds (0.25%). `baseline.log` and `dedicated_v2.log` independently report 1.624/1.619 ms and identical measured-output HF decode PCC 0.99969425. These are observed results, not projected gains from the hypotheses below.

The pinned reference is particularly useful here:

`/home/hous/dev/ornith-1.5-9b/references/ornith-1.0-35b/models/autoports/ornith_ai_ornith_1_0_35b/tt/fused_decoder.py`

Its GDN dimensions are the same 16 key heads, 32 value heads and 128-wide heads. Its hidden width and MoE differ; adapt the GDN mechanisms and derive dimensions from the 9B config. Do not transplant its measurements or MoE implementation. Below, `35B:` means this file.

## Actual graph and cost evidence

Source: `tt/functional_decoder.py:587–859`, shared block at 861; committed `../functional_decoder/tracy/linear_attention/{decode,prefill}_perf_report.txt`. Decode table has four replays, 312 operations = 78 operations per step. Values below are representative **single-replay kernel durations**, not end-to-end timing predictions.

| Segment | Inputs and sequence | Movement / measured cost |
|---|---|---|
| Input norm + QKV | BF16 `[B,T,4096]` → RMSNorm → linear width 8192 | B=1 decode 37 + 165 us |
| Decode conv | QKV times last tap; three `addcmul`; SiLU; three in-place history copies | BF16 history `[B,1,8192]` ×3; about 44 us including copies |
| QKV split | Three slices, each ROW_MAJOR → reshape → TILE | Q/K `[B,T,16,128]`, V `[B,T,32,128]`; about 42 us |
| A/B gates | Two independent width-32 matmuls; sigmoid(B); FP32 cast(A), bias, softplus, A_neg multiply | Two nearly single-core decode matmuls cost 30–32 us each |
| Head expansion and L2 | Q and K each repeat_interleave, RMSNorm, inverse-sqrt scale, FP32 cast; another Q scale | Repeat each lowers to untilize/concat/tilize |
| Decode recurrence | reshape Q/K/V and gates; exp(g); state multiply; K@state; subtract; beta multiply; transpose; outer matmul; state add; Q@state | FP32 state `[B,32,128,128]`; state decay ~15 us, read ~12 us, outer ~14 us, update ~17 us, final read ~60 us; final read uses only 4 cores |
| Z/output | Width-4096 Z projection; ROW_MAJOR head split; per-head RMSNorm; SiLU(Z) times core; ROW_MAJOR head merge; output linear | Z ~80 us, output linear ~84 us, plus relayouts |
| Shared dense MLP | residual; RMSNorm; gate/up linears; SiLU×up; down linear; residual | gate/up ~253/255 us, down ~235 us; gate/up/down prefill 7.751/7.756/5.585 ms |

Prefill has 96 operations and ~40.076 ms summed kernels. It spells out history concatenation, four FIR slices/MACs and activation; QKV head splits cost ~0.83 ms; explicit L2 ~0.38 ms. The chunk operator then performs additional Q/K/V transposes (~1.03 ms), Q/K repeats (~0.15 ms), and scale (~0.095 ms). Its return conversion plus caller tilize costs ~0.61 ms. Z split plus output merge cost another ~0.88 ms. This identifies material fusion targets independently of matmul tuning.

## Ranked verify/refute hypotheses

Each experiment should be an isolated candidate class first. Change one tightly related graph segment, compare to the latest accepted fused decoder, then retain only with real-weight equivalence and faster measured path. Preserve BF16 projection/conv weights and activations, FP32 gates/state/core, existing HiFi4 compute, state addresses, arbitrary logical length, and batch 1–32. No precision sweep or general program tuning is proposed.

### H1 — use the existing flat DeltaNet prefill contract (highest priority)

**Evidence/contract:** Current `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:152–209,277–309,349–363` already accepts flat rank-3 raw Q/K/V. The reader maps value heads to key heads; the prep kernel normalizes Q/K and folds Q scale. `output_head_major=True` returns FP32 TILE `[B*HV,T,V]`. The public docstring does not fully advertise flat input, so inspect the implementation rather than dismissing this based on its documented rank-4 signature.

Minimal adaptation: `35B:1670–1803,1852–1878`. Slice conv output to raw `[B,T,2048]`, `[B,T,2048]`, `[B,T,4096]`; do **not** call `_split_heads` or `l2_norm_ttnn`; call chunk op with `chunk_size=32`, `use_qk_l2norm=False`, `output_head_major=True`, existing device constants and FP32 initial state/gates. Despite that flag being false, flat input selects in-kernel normalization. Retain existing batch subdivision and make its slices rank-3. Output normalization consumes the head-major output; H2 describes the associated merge.

Constraints: `K==V==128`; physical T divisible by 32; `QWEN_GDN_PHASED` cannot be 0; batch×32 limited by core count per launch. Validate these deliberately. The op explicitly rejects `use_qk_l2norm=True`; that flag is not the way to enable this path. Raw flat Q/K must never be normalized twice.

**Prediction:** remove the external head splits/L2 and internal transposes/repeats/scale, and return conversion. Prove with profiler operation removal, not merely a lower Python call count. Focused numerical check compares prefill core **before gated RMSNorm**, final recurrent state and continuation decode to functional using real projected/conv inputs. Then paired prefill timings and whole-layer HF PCC at logical lengths 1,127,128,129,2047,2048,2049 and batches 1,4,32; existing test cases may cover most. Query-scale errors can be hidden by output RMSNorm, so output-only PCC is insufficient.

### H2 — maintain head-major GDN core; perform gating in flat layout

**Evidence:** Current `_gdn_out` splits Z into heads only to merge the result after elementwise gating. RMSNorm can operate on `[B*32,T,128]` / `[B,32,1,128]`; Z does not need head splitting at all. `35B:1805–1850` is an exact structural adaptation.

Prefill: normalize head-major FP32 core, reshape `[B,32,T,128]`, use `ttnn.experimental.nlp_concat_heads`, reshape `[B,T,4096]`, multiply by `silu(z)` in its native flat BF16 layout. Decode: normalize `[B,32,1,128]`, permute `(0,2,1,3)`, reshape `[B,1,4096]`; then flat gating. This removes Z's split and the functional output round-trip. Compare concat-heads versus permute+reshape at decode rather than assuming the dedicated operator wins: the reference documents its single-core decode limitation.

**Experiment:** real core/Z tuples before and after H1, compare gated output and layer output/PCC, then prefill/decode paired timing. Test B=1,4,32. Keep separate SiLU initially: the reference documents non-finite mixed-FP32/BF16 input-activation folding on this exact boundary (`35B:1830–1844`). If rechecking this on the new build, use actual FP32 core × BF16 Z and assert finite outputs; a matched-BF16 microtest does not refute the failure.

### H3 — dedicate prefill causal conv + activation + split

`ttnn.experimental.kda.qkv_causal_conv1d_silu` is the most relevant newly available KDA op. Its nanobind/validation under `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/` specify exactly four taps and return independent TILE BF16 Q/K/V. Ornith's widths 2048/2048/4096 match. Inputs must be interleaved ROW_MAJOR BF16 `[1,T,8192]`, history `[1,3,8192]`, tap tensors TILE BF16 with volume 8192; T must be positive and divisible by 32. `QkvCausalConv1dSiluProgramConfig(channel_chunk_size=...)` is required and its size must divide 8192. Compute `math_approx_mode=False` is required. History is **not updated** by the op.

Minimal experiment: replace only `_causal_conv` + field slices for B=1 prefill. Convert each history buffer and QKV to ROW_MAJOR, concatenate history only, invoke the op, separately select history tail using **logical_len**, not physical padded T. Feed outputs into H1 flat interface when accepted. Use a modest legal channel chunk (e.g. 256/512) as an implementation choice; do not copy tests' 1536 for an 8192 width. Initially compare conv outputs and tail against functional with real QKV/taps, including nonzero initial history and logical_len=1,127,129,2047. Then compare recurrent state and prefill→decode continuity.

For B>1 the op is B=1-only: serialize per-user calls only if measured beneficial, preserving request isolation; never concatenate users along T. For decode T=1 it is not a direct replacement. A padded T=32 experiment could extract token0 and manually update actual history, but this computes 31 unnecessary tokens and needs latency evidence; static legality alone does not prove a win.

Fallback dedicated candidate: ordinary `ttnn.conv1d(groups=8192)` or two 4096-channel depthwise calls (`35B:1540–1669`). Supply input history, pre-prepare weights at setup and keep state updates trace-safe. Prefer KDA first because it fuses split/activation and avoids the ordinary conv's sharded-output handling. The 35B and `models/demos/blackhole/qwen36/tt/gdn/tp.py` report incorrect conv activation folding; evaluate separate SiLU and fused activation on real shapes before retaining. This path's L1/workspace constraints must be checked, not hidden behind a broad exception. Rejection of ordinary conv does not reject the KDA conv.

### H4 — merge shared-LHS GDN projections

Current QKV, Z, A, B share exactly the same normalized x and are independent. Pack host weights in order `[QKV|Z|A|B]`, total width `8192+4096+32+32=12352`, then one BF16 linear and four tile-aligned slices. `35B:990–1001,1489–1502` supplies the pattern. Thread projected Z/A/B through `_gdn_prefill`, `_gdn_decode`, gates and output to prevent accidentally recomputing them. Free old unused device projections once construction is correct; no host packing in forward.

**Experiment:** first compare all four projected fields individually with real weights and x, then whole-layer prefill/decode/state PCC and paired timings. Two narrow gate projections alone cost ~62 us decode, but packing may change matmul geometry; no speedup is guaranteed. If full packing regresses one mode, isolate packing QKV+Z and packing A+B as two smaller hypotheses rather than claiming shared-LHS packing is exhausted after one configuration. Keep dtype/fidelity fixed.

### H5 — one decode QKV relayout and one adjacent Q/K repeat

`35B:1880–1934` reshapes activated `[B,1,8192]` once to `[B,1,64,128]`, permutes to `[B,64,1,128]`, slices V and adjacent QK, repeats QK once along head axis, then slices Q/K. Return all three head-major to recurrence. This removes three ROW_MAJOR round-trips, one Q/K expansion sequence and recurrent Q/K/V reshapes. Current functional `_split_heads` warning is reason to test this actual tiled reshape+permute, not to assume all tiled reshapes are free views or incorrect.

**Experiment:** compare Q/K/V elementwise before normalization with recognizable per-head/channel ramps and real conv output; verify `[q0,q0,...,q15,q15]` and matching K order. Follow with real recurrent state/core and B1/B4/B32 trace PCC and state-address checks. Test poisoned free pool to expose padding-dependent relayout mistakes. Benchmark the combined representation change because its consumers must accept the new shapes.

### H6 — fuse recurrent arithmetic without changing state precision

The search found no drop-in dedicated single-token GDN kernel with the current FP32 state/core contract. `gated_delta_attn_seq` is a prepared **128-token chunk scan**, not such an op; KDA contracts below also differ. Three small op merges remain directly testable:

1. `exp(g)` then in-place state multiply → `multiply(state, reshaped_g, input_tensor_b_activations=[EXP], output_tensor=state)`. Both inputs are FP32. Existing idiom: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py:448–453`. Validate recurrent state before K read and across 32 distinct trace replays.
2. `transpose(k_row) @ delta` → `matmul(k_row, delta, transpose_a=True)` (`35B:1986–1995`). Retain HiFi4 and FP32. Compare outer and state before output norm. Rank/broadcast support and padded K dimension matter.
3. Q normalization's `K**-0.5` then post-cast query scale `K**-0.5` → one `K**-1` multiply, preserving explicit query scale mathematically (`35B:1953–1963`). This moves a rounding point; check core and state against current graph, rather than claiming bit identity. A weight folded into `rms_norm` could remove another multiply, but BF16 norm-weight representability/rounding require a separate probe; K scaling is an exact power of two for query here, while key `1/sqrt(128)` is not.

Also inspect rank-1 update identity `state + k_col * delta_row` using broadcast `addcmul` instead of padded matmul+add. This is a meaningful graph rewrite (the contraction dimension is logically 1); exact output/state and trace PCC must establish padded lanes are not contributing. If ternary broadcasting is unsupported, that is a concrete rejection. A generic `addmm` name is not evidence of a fused rank-4 in-place update; inspect lowering before replacing.

The final recurrent read is 60 us on four cores while the other read is 12 us. Explicit matmul core-grid tuning in 35B could address this, but it is performance tuning rather than graph fusion and is **deferred** under this investigation's scope. Do not combine it with a fusion experiment and attribute the total gain to fusion.

### H7 — dense MLP gate/up packing and anchor activation

Gate and up share normalized h and width 12288. Pack into one `[4096,24576]` weight and slice two halves, retaining folded binary SiLU. This is applicable even though the skill gives ≥3 peers as its common example. Compare the two projected halves and real whole-layer results; time prefill and decode independently because these matmuls account for >50% of prefill and ~0.5 ms decode but packing may not improve their bandwidth geometry.

Alternative isolated merge: keep separate projections but `linear(..., activation="silu")` for gate, then multiply activated gate with up. Check precise support and numerical behavior on actual matmul configuration; this competes with the already fused binary activation, not an extra cumulative gain. `ttnn.swiglu` is a **composite**: `eltwise/unary/device/unary_composite_op.cpp:294–303` splits, calls swish, then multiply. Do not assume changing to that name reduces device operations. DeepSeek routed-expert fused kernels have MoE dispatch/weight contracts and are not directly suitable for this dense MLP.

### H8 — fold gate bias/softplus and audit residual norm

`g = A_neg * softplus(a32 + dt_bias)` → `soft = ttnn.add(a32, dt_bias, activations=[SOFTPLUS]); g = multiply(A_neg, soft)` as `35B:1504–1534`. Use the same softplus parameters as `ttnn.softplus`; inspect UnaryWithParam signature if required. Both inputs are FP32. Compare gate output over real a values and padded-token masks; then state/core and trace equivalence. A linear-bias fold would move FP32 bias into the BF16 projection boundary and is not automatically equivalent under fixed precision.

`rms_norm(..., residual_input_tensor=mixed)` can internally add x+mixed, but this block still needs the unnormalized h for its final residual. Standard `rms_norm` returns one normalized tensor, so replacing `h=add(x,mixed); norm(h)` with residual-input norm while retaining the final h would duplicate the addition. Reject that naive substitution structurally. A specialized operation returning both h and norm(h) would be applicable only if found with a matching dtype/layout contract. Likewise, folding gdn_norm weights into the output projection changes a rounding boundary and does not remove a standalone op: weight is already inside RMSNorm.

## KDA candidates: precise compatibility ledger

All source paths are under `ttnn/cpp/ttnn/operations/experimental/kda/`, with contract tests in `tests/ttnn/nightly/unit_tests/operations/experimental/kda/`. These operators must be considered, but shared names are not shared contracts.

| Operator | Compatibility verdict and reason |
|---|---|
| `qkv_causal_conv1d_silu` | Applicable prefill hypothesis H3; exact four-tap formula and asymmetric Q/K/V widths supported. B=1 and T%32 limit require deliberate handling. |
| `sigmoid_gated_rms_norm` | Direct replacement **invalid**: computes norm(core)×weight×sigmoid(z); Ornith requires SiLU(z)=z×sigmoid(z). Possible prefill adaptation is op followed by flat multiply z. It also requires `[B*H,T,V]`, BF16 flat gate `[B,T,H*V]`, weight rank1 `[V]`, T%32=0, and `packer_l1_acc=False`; request FP32 output. Compare this two-op adaptation with H2 on real core/Z. At T=1, validation excludes direct decode. |
| `prepare_chunk_recurrence` | Direct fixed-precision replacement **invalid**: requires BF16 per-key g `[1,T,H*K]`, versus Ornith's FP32 per-head g. Would need head repetition and broadcast gate, then a prohibited downcast. It is also B=1 and implicitly normalizes raw Q/K. Existing flat GDN H1 keeps FP32 gates and does GVA in its reader. |
| `recurrent_chunk_scan` | Consumes seven prepared 32-token terms; state FP32 but token output is forced BF16, versus current FP32 core. No immediate precision-preserving drop-in for decode. Preparing a padded single-token chunk incurs multiple operations and altered output precision; exclude under fixed-precision scope. |
| `summarize_chunk_recurrence`, `affine_exclusive_scan`, `reduce_affine_transforms` | Grouped prefill state propagation components, not direct single-step fusion. Would require explicit per-chunk affine forms, grouping and output reconstruction that the existing GDN chunk op already implements. These do not replace the current primitive decode recurrence without a separate algorithm design. Defer under this stage's fusion scope; do not claim their presence proves a decode kernel exists. |
| `transformer.gated_delta_attn_seq` | Requires eight prepared FP32 tensors, C=K=V=128, optional state `[BH,K,V]`; not raw Q/K/V decode. Existing dedicated chunk GDN is the relevant prefill op. |

## Commands and acceptance procedure for the hardware owner

Run only through the main agent's existing serialized hardware wrapper/environment. These are **suggested inner commands, not commands executed by this investigation**:

```bash
ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fusion_equivalence.py -k linear_attention -x -v -s
ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py -k 'linear_attention and (real_weights_pcc or traced_decode_pcc or perf)' -x -v -s
ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_fused_decoder.py -k 'linear_attention and (prefill_continuation or batched or determinism or poisoned or repeated_identical)' -x -v -s
```

Add each isolated candidate to `tests/fusion_candidates.py` and select it explicitly in the paired harness; current harness recognizes `default` and `packed_attention` only. The first command tests the accepted class unless that map is extended. Compare **each candidate against the latest accepted fused graph**, retaining an independent original-functional control before final promotion. Record class/source hash, real checkpoint revision, sample list/median, measured output PCC, and full state comparison for every attempt. Use exact test node IDs for contract extensions whose parametrization lacks the `linear_attention` id; a zero-selected run is not evidence.

Prior bar is PCC≥0.995. Also require finite values, no material HF accuracy regression, real state continuity across nonaligned prefill and decode, unchanged trace buffer identities and equal restored-state eager/replay outputs. The current paired harness compares recurrent/conv state after one replay and times 32 advances but discards that timed final state: recurrence-changing hypotheses need comparison after the 32 advances too, with changing inputs in an untimed correctness arm. A pair of wrong intermediate scales can be invisible after RMSNorm, so H1/H6 must check pre-norm core amplitude as well as PCC (relative error or allclose).

After accepted changes, regenerate operation tables, repeat the candidate scan, and run the complete prior fused stage contract, including native context gates and watcher evidence under the main hardware owner. Exhaustion means every materially applicable entry above is measured or ruled out by a precise source contract; the present 0.25% decode gain does not meet that standard.

## Investigation status

H1–H8 are source-supported hypotheses awaiting isolated device verification. H1/H2/H4/H5 have concrete corresponding code in the pinned 35B reference; H3 adds a newly available operator that reference did not use. No implementation fixes are claimed by this report. No build required: the only artifact authored here is this Markdown report.
