# Optimized multichip decoder work log

## 2026-09-05: starting state and operation-topology audit

Started from clean branch `hous/gemma-4-26b-a4b-it` at
`537f844d043` (`Document Gemma 4 multichip commit`).  This pass optimizes
`tt/multichip_decoder.py` in place.  It does not start full-model, generator,
LM-head, sampling, or vLLM work.

The accepted baseline is the completed `doc/multichip_decoder` stage.  The
public inter-layer residual is replicated `[1,1,logical_M,2816]`; decode uses
an internal 22-core height-sharded L1 residual for every profile/layer except
TP4 full attention, which uses the correctness-gated interleaved R0 family.
TP2/TP4 row-parallel O, dense-down, and active-expert-down each finish with a
persistent asynchronous BF16 all-reduce.  There is no collective between
decoder layers.

| Boundary / sequence | Repeated same-input matmuls or material movement | Candidate family and constraints | Starting action |
| --- | --- | --- | --- |
| input norm -> attention | Q/K/V are already packed into one rank-local QKV matmul; R22 inputs are converted to DRAM before the multichip QKV override | Preserve packed QKV; compare DRAM-sharded QKV plus a phase-specific input shard for TP1/TP2/TP4, with BF16 sliding and BFP8 full policies held fixed | measure by profile; do not split QKV |
| QKV -> heads -> RoPE -> paged cache -> SDPA | head creation, L1/interleaved conversions, BF16 cache updates, explicit paged SDPA; full attention has extra transpose/rotary movement | Keep real logical B1 separate from tile padding; compare explicit attention configs and BFP8 cache under the same paged/trace contract | measure and retain cache-consuming PCC |
| local attention O -> residual | row-parallel local O, then BF16 persistent async all-reduce to replicated hidden; TP4 O is DRAM-sharded | compare local-matmul+AR, fused AG+local-output matmul, fused matmul+RS, and delayed-gather fractured residual; account through next norm/consumer | prior fractured complete layer was 29.47% slower; recheck only against new compatible family/config evidence |
| attention residual -> dense MLP | gate/up share one input. R22 baseline uses separate gate/up; TP4 R0 decode uses packed gate/up DRAM-sharded, then two slices. TP1/TP2 have no default DRAM roles | compare packed and tuned separate families under identical dtype/fidelity and residual layout; include split/GELU/mul/down/CCL | profile-specific A/B |
| dense down -> residual | row-parallel down plus BF16 persistent async all-reduce; selected path restores inherited residual layout after the collective | try lower-movement family only through following norm/router/expert consumer; immediate restore alone is not a rejection | use full compatible layer measurement |
| router | replicated FP32 matmul, TopK/scatter; TP4/full R0 router and norm rows are material | compare router input L1, legal program/subblock/fidelity variants, and R22 TP4/full with localized accuracy repair | audit advice and A/B |
| MoE gate/up | gate-selected top-8 sparse execution. TP4 decode packs gate/up; TP1/TP2 decode is separate. Prefill currently issues separate sparse gate/up per chunk | retain active-expert execution; compare packed/separate, validate whether fixed `nnz=8` metadata survives score conversion, and try indexed top-k plus role-specific geometry | profile-specific A/B; dense all-expert is forbidden |
| MoE down -> residual | sparse down, score weighting/reduction, then BF16 persistent async all-reduce | compare L1 versus DRAM intermediates and compatible fractured-residual carry-forward | measure whole routed chain |
| final norms/add | R22 stays sharded locally; TP4/full R0 has several interleaved norm rows | make TP4/full R22 correct with a minimal precision/layout exception if possible; otherwise preserve R0 only with current PCC/perf evidence | first material correctness target |
| repeated decode CCL | three hidden all-reduces per TP2/TP4 layer, three rotating persistent buffers/semaphores, Linear/1-link TP2 and Ring/2-link TP4 | compare persistence, BF16/BFP8 payload, link/placement policy, and fused CCL+matmul families | persistence already selected; rerun final default and dtype/topology matrix |
| inter-layer contract | public output is replicated DRAM; no gather/reshard/all-reduce is inserted between layers | only replace with fractured carry-forward if a complete next-layer-compatible family wins | preserve unless measured winner changes it |

Starting profiler findings from the selected TP4 capture:

- Sliding decode: sparse matmuls 19.57%, DRAM-interleaved matmuls 13.90%,
  persistent all-reduce 8.46%, width-sharded matmuls 7.39%, and layout movement
  (`ShardedToInterleaved` + `InterleavedToSharded`) 5.68% of classified device
  time.  Material advice targets QKV/router input placement, router subblocks,
  and attention fidelity.
- Full decode: interleaved norms 27.71%, sparse matmuls 13.17%,
  DRAM-interleaved matmuls 10.80%, all-reduce 5.68%, and width-sharded matmuls
  4.47%.  The R0 residual exception is therefore the first whole-layer target.
- Prefill: the selected S=1024 report is dominated by active-expert sparse
  gate/up work, with many low-utilization 32-row chunks.  Packing and chunk
  geometry must be compared without changing gate-selected execution.

No runtime experiment had been started when this audit was written.

## Baseline freeze

The starting implementation at `537f844d043a202f921053d465a20f52f3021431`
was measured with real checkpoint weights at S=1024/B1. Each profile ran one
untimed prefill warmup and five requested decode warmups plus trace validation
before 30 trace replays. Frozen JSON and JUnit files are under
`baseline/<profile>/`.

| Profile | Sliding prefill/decode ms | Full prefill/decode ms |
| --- | ---: | ---: |
| P150 | 96.431136 / 0.964689 | 107.013618 / 1.080224 |
| P150x2 | 70.733865 / 0.845147 | 77.815577 / 0.943894 |
| P150x4 | 78.695409 / 0.677389 | 87.299281 / 0.984365 |

P150 and P150x2 create 1x1 and 1x2 submeshes from the QB2 2x2 control plane.
P150x4 uses the real 1x4 target mesh. Artifact device IDs are `[1]`, `[1,0]`,
and `[1,0,3,2]`; no timing is from replicated fallback.

## Candidate sequence and decisions

1. Collective output was converted directly from the CCL's width-sharded L1
   output to the following R22 residual instead of restoring through DRAM.
   The isolated delta is small/noisy but removes redundant DRAM movement and
   passes TP4 B32 PCC.
2. Dense execution stayed inside the coherent R22 family through gate/up,
   activation, local down, all-reduce, and the following residual consumer.
   P150 decode improved 4.3-4.7%; P150x2 improved 5.7-7.0%. Selected on TP1/2.
3. TP4 sliding's first R22 dense attempt failed because local Kt=17 was not
   divisible by block 3. The legal block-1 retry completed but regressed decode
   from 0.677389 to 0.690094 ms. TP4 full R22 retries scored 0.994723 default,
   0.994419 BF16 attention, 0.994771 HiFi2 MLP/experts, and 0.994771 BF16
   experts. TP4 keeps sliding R22 without coherent dense and full R0.
4. Active-expert gate/up was packed for both phases and every profile. One
   rank-local `[up,gate]` weight feeds one sparse matmul and is reused by
   prefill/decode. Releasing separate gate/up tensors makes it capacity-neutral
   and removes the prior TP4 extra copy. This supplies the dominant prefill win.
5. One/two/three-reader TP1 O measurements completed. Sliding decode is
   0.864045/0.872459/0.924485 ms; full is
   0.890058/0.885859/0.906084 ms. Every option is slower than the final
   0.759652/0.885008 ms final default and is rejected on whole-layer latency. The
   corrected five-full-layer capacity ledger leaves 268,670,464 bytes, so
   capacity is not used as the rejection rationale.
6. TP2 O is neutral on sliding (0.740116 versus 0.739822 ms) and improves full
   from 0.837662 to 0.801357 ms. Full-only O is selected. Its five-layer
   physical BFP8 copy is 61,276,160 bytes and leaves 861,289,472 bytes of
   conservative headroom.
7. BFP8 CCL produces 0.991261/0.994844 B32 PCC and regresses decode to
   0.685657/0.995042 ms. BFP8 activation completes after layout diagnostics
   but produces 0.994058/0.994514 PCC. Both families are rejected.
8. TP4 Ring with one link measures 0.720692/1.030042 ms, 6.0%/4.4% slower
   than two links. Nonpersistent CCL is only 0.2-0.3% nominally faster and
   removes trace-safe explicit ownership. BF16/two-link/rotating persistent
   resources remain selected on TP4; TP2 retains Linear/one-link.
9. The profiler's 88-core advice was exercised by wiring the inherited legal
   110-core 2D QKV/O prefill family into the multichip path. All six completed:
   P150 79.963791/78.735242 ms, P150x2 53.087146/57.384931 ms, and P150x4
   63.860760/70.320307 ms. It ranges from 0.24% faster to 0.71% slower and
   loses both TP4 cases; it was removed from defaults as noise/regression.
10. Sparse-prefill input was moved from DRAM to L1 on all six release cases.
    P150 measures 80.190674/78.727656 ms, P150x2
    52.808566/57.433407 ms, and P150x4 63.492598/70.236147 ms. Mixed deltas
    span a 0.18% gain to a 0.40% regression, so the extra conversion is rejected.
11. The 110-core attention family was combined with explicit L1 QKV/O inputs
    for a second six-case advice trial. P150 measures
    80.173652/78.745712 ms, P150x2 52.775410/57.423895 ms, and P150x4
    63.488310/70.795482 ms. The mixed deltas include a 1.20% TP4 full
    regression, so neither the grid nor L1-input combination is selected.
12. Final report advice was applied to TP4 sliding QKV with the other selected
    DRAM roles unchanged. It passed PCC and two alternating candidate/default
    controls measured 0.584680/0.585390 ms versus 0.594262/0.594501 ms decode,
    a stable 1.50-1.65% win. QKV is therefore selected only for TP4 sliding;
    the full-attention path cannot consume this configuration outside R22.
    The retained 25-layer BF16 copy is 288,358,400 bytes/device and remains
    within the context capacity contract.

### Coherent-family matrix

| Family | Evidence and result | Decision |
| --- | --- | --- |
| residual layout | complete R22 dense through following residual; TP1/2 win, adapted TP4 loses/fails PCC | select TP1/2; profile-specific TP4 layouts |
| collective placement | CCL output goes directly to residual conversion; no inter-layer CCL | select |
| fused CCL+matmul | exact-shape inherited matmul-RS/AG-matmul plus complete fractured layer; 29.474% slower end to end | reject; exact op shapes unchanged |
| packed vs separate projections | QKV/dense already packed; expert gate/up packing saves one material sparse op | select packed experts |
| activation/CCL dtype | completed BFP8 activation and CCL PCC failures | retain BF16 activation/CCL/cache |
| persistent buffers | nonpersistent timing within noise; rotating resources survive stacked replay | retain persistent |
| DRAM-sharded decode | TP1 1/2/3-reader rejected by measured whole-layer latency; TP2 full O and TP4 sliding QKV selected; inherited TP4 O/packed-gate-up/down retained | profile-specific selection |

The lower-movement residual family was never ranked by immediately restoring
the old DRAM contract. It was measured through dense/all-reduce/residual and
stacked next-layer consumers. TP4 full includes four completed repairs after
the first layout/program issue.

## Final performance and profiler evidence

Final default warmed timing is:

| Profile | Sliding prefill/decode ms | Full prefill/decode ms |
| --- | ---: | ---: |
| P150 | 80.051155 / 0.759652 | 78.798871 / 0.885008 |
| P150x2 | 52.653659 / 0.646949 | 57.376768 / 0.722745 |
| P150x4 | 63.521466 / 0.585234 | 70.092498 / 0.914335 |

These measurements use one prefill warmup and five requested decode warmups
plus trace validation before 30 traced replays. They are the final optimized
numbers retained under `final/{p150,p150x2,p150x4}/`. All use decoder SHA-256
`9c6735c56ff48309215845e3f73aa640c54fdb7e5ca0a5ff3b36155b409473ba`
with test SHA-256
`54a87ce28b179efc740bd5ea1078340348a2fd7dee92f01379e3c814916f8071`
and fallback throwing. The six representative real-weight PCC cases were also
repeated on this path; all pass 0.995.

Tracy ran independently from watcher with two warmups and three trace
iterations for all six cases. Raw CSVs are retained under
`profiler/<profile>_<kind>/raw_ops.csv.xz` using lossless xz compression so
each file satisfies the repository's 500 KB limit. Each was processed twice
with exact prefill/decode signposts and once with `--no-merge-devices`.
Reproduce a report by first decompressing the raw provenance to a temporary
CSV. Example:

```bash
env -u TT_METAL_WATCHER GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
  GEMMA4_RANGE_DOWNLOAD=1 \
  GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=2 \
  GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=3 \
  python_env/bin/python -m tracy -r -p \
  -o gemma4_optimized_multichip_p150x4_sliding \
  -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_perf_profile[blackhole-sliding_attention-batch1-mesh_device0-device_params0]'

xz -dc profiler/p150x4_sliding/raw_ops.csv.xz > /tmp/gemma4_raw_ops.csv
python_env/bin/tt-perf-report /tmp/gemma4_raw_ops.csv --arch blackhole \
  --active-experts 8 \
  --start-signpost PERF_DECODE_layer0_sliding_attention_seq1024_batch1 \
  --end-signpost PERF_DECODE_layer0_sliding_attention_seq1024_batch1_END \
  --csv decode_ops.csv --summary-file decode_summary
```

Same-run device-operation time per replay is 0.717020/0.845685 ms on P150,
0.608808/0.687962 ms on P150x2, and 0.549745/0.883933 ms on P150x4. Host minus
device-operation time is dispatch/synchronization overhead. Per-device max/min
time is at most 1.0062x. Modeled decode DRAM roofline ranges from 7.9% to 37.5%;
summary totals are in `perf_summary.json`; dominant-op tables and advice are in
the human-readable `*_report.txt` files and CSVs. `profiler/provenance.json`
supplies same-run hash and operation-accounting links for all six captures.

## Independent-review remediation and final candidate closure

The initial independent review (`STAGE_REVIEW_INITIAL.md`) found that the
fixed sparse `nnz=8` hint was unsafe after BF16 score conversion, TP2 geometry
and QKV advice were incomplete, precision evidence was too coarse, profiler
provenance was ambiguous, and the watcher limitation was underqualified.
`AUTODEBUG.md` reproduced those findings. The `$autofix` investigation then
split sparse routing, geometry, and evidence/provenance into independent
hypotheses; `AUTOFIX.md` records the resulting actions.

Sparse remediation was measured in two steps:

1. Removing the fixed hint and asking TTNN to scan all 128 runtime scores was
   safe but regressed TP4 decode to 0.790689/1.099992 ms.
2. The selected path carries exact TopK indices plus eight compact scores into
   indexed sparse gate/up and indexed sparse down matmuls. It keeps dynamic
   gate-selected top-8 execution and improves final decode on every profile.
   The non-indexed fallback still omits `nnz`; no path asserts a false count.

TP2 packed-expert `per_core_N=2` with a legal 1x2 subblock measured
0.666375/0.730061 ms decode versus 0.646949/0.722745 final. Applying the
same geometry to prefill was only 0.2-0.3% different and mixed by layer kind.
Two adapted TP2 sliding-attention DRAM-sharded QKV trials completed: block width 11 measured
0.669843 ms and block width 1 measured 0.684791 ms on sliding decode, each
retaining another 550 MiB/device. The final profiler advice was also ported
into the coherent R22 path and rerun on the frozen source. TP1 full QKV
regressed to 0.889607 ms and router-L1 regressed to 0.915695 ms. Gate/up were
noise-level at 0.882128/0.882643 ms. TP2 full down regressed to 0.763932 ms.
TP1 full down was the only nominal winner: three candidate runs measured
0.881037/0.882284/0.881167 ms versus three interleaved-default controls at
0.885303/0.884308/0.884412 ms. Its 0.23-0.48% delta is sub-noise and would
retain another 59.5 MB/device across the five full layers, so it is rejected.
All candidate PCCs pass the recorded-activation 0.995 gate. The analogous TP4
sliding candidate was a stable material winner in alternating controls and is
selected in the final default, with full-attention allocation explicitly
excluded because that path cannot consume QKV DRAM sharding outside R22.

Role-isolated BFP8 attention, dense MLP, MoE, and residual/norm runs completed
on TP2 and TP4. Individual timings were noise-level or slower. A combined
all-role run improved TP2 prefill to 45.391/49.099 ms but slowed decode to
0.664679/0.726787 ms and failed TP4 B1 PCC (0.974848/0.976697), TP2 full decode
PCC (0.991173), and TP4 B32. A matching-dtype TP2 BFP8 CCL retry failed sliding
decode PCC at 0.994785 and was slower. BF16 therefore remains default for all
activation roles, residual/norm, persistent buffers, and CCL payloads.

The final precision gate uses prompt-derived activations, not seeded random
hidden states. `precision/recorded_layer_inputs.{pt,json}` records a 32-token
prompt's inputs to layer 0 and layer 5 after executing the preceding HF layers
with checkpoint revision `4d7ae4984b7d`; its content hash is
`c89172f7d09d9da3c96aca73362c213265519f6dda365a8cb84402f71dcdc44a`.
The initial default passed layer 0 but exposed full-attention prefill PCC
0.992689. Isolated attention-BF16, expert-BF16, dense-down-BF16, and all-HiFi4
probes stayed near 0.9927. The shared BF16 dense-weight override covers gate,
up, and down; it reached 0.999831 prefill and 0.999859 decode and is selected
for full-attention layers. Final recorded-input
PCC across TP1/TP2/TP4 is at least 0.999510. A crossed TP2 BF16/BFP8 CCL B32
run passed numerically but BFP8 was slower: 6.5485/6.6342 ms versus
6.5356/6.6256 ms for sliding/full. The final coherent R22 TP1/TP2 constructors
also omit the packed dense tensor they never consume; this more than offsets
most of the five full layers' BF16 storage delta. Accounting for all three
dense matrices (and TP4's retained packed dense gate/up copy) leaves projected
P150/P150x2 headroom of 268,670,464/861,289,472 bytes.

The default packed-both constructor originally uploaded separate device gate
and up tensors before allocating the shared packed replacement. That transient
peak exceeded the final P150 margin. AutoFix changed setup to upload the one
rank-local packed tensor directly from the host and never allocate separate
device gate/up tensors. This removes 285,474,816 bytes from the TP1 full-layer
constructor peak. Both meaningful layer kinds then passed a fresh final-source
50,624-token physical boundary run with fallback throwing.

TP2 B32 uses its inherited stress threshold 0.99 and passes at
0.994905/0.994361. The indexed-off control produces the same PCC within 1e-7
while taking about 14.7 ms versus 6.6 ms, so compaction is not responsible for
the small TP2-versus-single-chip numerical delta.

A final watcher retry exposed a deterministic TP4 full B32 score of 0.994989
against the completed stage's older BFP8-dense synthetic oracle. AutoFix
repeated the miss three times, refuted watcher drift through eager/trace/replay
bit equality, and rejected dense-HiFi2 and temporary expert-HiFi2 responses.
Fresh prompt-derived single-chip references use the same selected precision
policy as each layer kind: BF16 attention for sliding and BF16 dense
gate/up/down for full. The unchanged LoFi TP4 path then scores
0.999764/0.999833 sliding/full; TP2 scores 0.999854/0.999895. The final
12-case watcher suite uses those matched references and passes in full.

Tracy captured all six final profile/layer cases separately from watcher. The
frozen `9c6735c5`/`54a87ce2` source/test pair was captured in six fresh
isolated runs and generated the retained raw CSVs. The human reports were
regenerated without `--csv`; CSV reports, summaries, per-device data,
compressed raw inputs, and same-run timing/provenance are all retained.

Active-Ethernet watcher instrumentation was attempted and fails before model
execution because the 30,064-byte fabric program exceeds the 26,624-byte
ACTIVE_ETH watcher config buffer. That failure and a post-failure healthy
`tt-smi` snapshot are retained. With only ETH watcher instrumentation disabled,
worker and idle-Ethernet monitoring passed 12 risk-matched TP2/TP4 real-weight,
B32, nonaligned S=33, and stacked-trace cases with no watcher errors.

## Correctness, fallback, stress, and health gates

Final direct gate:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_RECORDED_ACTIVATIONS=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/precision/recorded_layer_inputs.pt \
  GEMMA4_RANGE_DOWNLOAD=1 \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'p150_proxy_real_weights_prefill_decode or p150x2_proxy_real_weights_prefill_decode or multichip_real_weights_prefill_decode'
```

Six passed. Final HF PCC artifacts are separated by profile under
`final/{current_pcc_p150,current_pcc_p150x2,current_correctness}/`. Static
source tests prove the runtime inherits the TTNN
optimized decoder, contains no Torch conversion in hot methods, preserves
top-8 sparse execution, validates environment overrides, and owns padding.

Final risk-matched stress used fallback throwing and TP2/TP4 B32, both S=33
nonaligned traces, and two-layer TP2/TP4 mixed-attention stacks.

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_RECORDED_ACTIVATIONS=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/precision/recorded_layer_inputs.pt \
  GEMMA4_MULTICHIP_REFERENCE_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/final/tp4_full_b32_matched_reference \
  GEMMA4_MULTICHIP_ARTIFACT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/final/watcher_current \
  GEMMA4_RANGE_DOWNLOAD=1 \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'multichip_batch32_trace_and_optimized_pcc or multichip_non_aligned_prefill_and_decode_trace or tp2_stacked_mixed_attention_shared_persistent_ccl_trace or tp4_stacked_mixed_attention_shared_persistent_ccl_trace'
```

Six passed. Both stacks use 20 bit-exact replays. Path counters are unchanged
during replay and record only in-layer attention/dense/expert reductions.

Watcher was kept separate from profiling. Active ETH was attempted first and
its size failure is retained under `final/watcher_eth/`. The complete clean
run disabled only ETH instrumentation:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_RECORDED_ACTIVATIONS=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/precision/recorded_layer_inputs.pt \
  GEMMA4_MULTICHIP_REFERENCE_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/final/tp4_full_b32_matched_reference \
  GEMMA4_MULTICHIP_ARTIFACT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/final/watcher_current \
  GEMMA4_RANGE_DOWNLOAD=1 \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  -k 'p150x2_proxy_real_weights_prefill_decode or multichip_real_weights_prefill_decode or p150x2_batch32_trace_and_optimized_pcc or multichip_batch32_trace_and_optimized_pcc or multichip_non_aligned_prefill_and_decode_trace or tp2_stacked_mixed_attention_shared_persistent_ccl_trace or tp4_stacked_mixed_attention_shared_persistent_ccl_trace'
```

Twelve passed with no watcher error under
`final/watcher_current/`. Its provenance manifest hashes the copied nonaligned
and stacked evidence as well as the B32 outputs and watcher log. The final
health snapshot in `final/tt_smi_post_final.json` reports four healthy P300Cs,
healthy DRAM, and zero corrected/uncorrected GDDR errors after both watcher
attempts.

Host/static verification enumerated every `test_` function before the first
hardware case (`test_tp4_ring_all_reduce_smoke`) as an explicit pytest node ID,
avoiding fixture-name heuristics.

```bash
HOST_STATIC_NODES=("${(@f)$(python_env/bin/pytest --collect-only -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
  2>/dev/null | sed -n \
  '/<Function test_tp4_ring_all_reduce_smoke/,$d; s/.*<Function \(test[^>]*\)>.*/models\/autoports\/google_gemma_4_26b_a4b_it\/tests\/test_multichip_decoder.py::\1/p')}")
python_env/bin/pytest -q "${HOST_STATIC_NODES[@]}" \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder/final/host_static_current.xml
```

All 45 final host/static cases pass after the sparse-routing remediation.
Touched implementation/test files compile with `python -m py_compile`; JSON passes
`python -m json.tool`. This is a
Python/JSON/Markdown-only change, so `AGENTS.md` does not require a C++ build.

## Optimize checklist

- [x] Baseline frozen before edits for P150/P150x2/P150x4, sliding/full,
  warmed prefill and traced warmed decode.
- [x] Operation-topology audit completed before candidates, including repeated
  same-input matmuls, collectives, conversions, fused paths, and packing.
- [x] Real tensor-parallel target mesh used; no replicated or single-chip
  fallback accepted for TP2/TP4 measurements.
- [x] Layout, collective placement, fused CCL+matmul, packed/separate,
  activation/CCL dtype, persistent buffer, DRAM sharding/readers, and
  precision/fidelity families have evidence.
- [x] Report advice tried or closed by exact physical/shape/capacity evidence;
  material failures were adapted and retried.
- [x] Real-weight representative PCC passes for both layer kinds and all
  profiles; TP2 and TP4 B32 also pass.
- [x] Nonaligned lengths, paged-cache consumption, trace replay, stacked
  resources, fallback throwing, and watcher are covered.
- [x] MoE remains gate-selected top-8 sparse execution.
- [x] Context contract and exact persistent-byte projection updated; no max
  context reduction.
- [x] Final performance/profiler artifacts reproduce the final default path.
- [x] No full-model or vLLM work started; no optimization is deferred.

## Independent gate and local commits

The final independent review is recorded in `STAGE_REVIEW.md` with verdict
`clean-pass` and no required work.

- Stage starting commit: `537f844d043a202f921053d465a20f52f3021431`.
- Optimized implementation and evidence commit:
  `d03bc791f7c2dbf9fc955332b53544e120fe1b16`.
- This commit-record-only work-log update is the final local follow-up commit;
  its SHA is reported in the handoff because a commit cannot contain its own
  identifier.
