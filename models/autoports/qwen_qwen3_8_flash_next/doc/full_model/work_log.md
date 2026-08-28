# Qwen3.8-Flash-Next full-model work log

Date: 2026-08-27 through 2026-08-28

## Scope and inherited baseline

Active skills: `$full-model`, `$host-weight-cache`, `$tt-device-usage`, with
`$tt-enable-tracing`, `$qualitative-check`, `$autofix`, and `$stage-review`
used for their applicable gates. Work started from the completed optimized
host-backed multichip decoder. No vLLM file or registration was created.

Target configuration for every accepted hardware result:

```text
TT_VISIBLE_DEVICES=0,1
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
mesh=1x2 P300 Blackhole, FABRIC_1D, links=2, packet=8192 bytes
l1_small_size=24576, trace_region_size=1073741824
checkpoint=f5d08274bafd880402bd16f5e3e6c514136ec06c
```

Implemented files:

- `tt/model.py`: embedding, 48-layer fractured stack, final hyperconnection
  mixer/norm, LMHead1D, Sampling1D, KV/page/position/fixed-slot state, eager
  and split traced decode, runtime fallback audit, bounded close/reset.
- `tt/generator.py`: standard `build_generator`, low-level
  compile/prefill/decode/token-out APIs, ragged prompt padding and slicing,
  fixed-slot generation/chat, device sampling, and explicit host-sampling
  compatibility.
- `demo/generate_hf_reference.py` and `demo/full_model.py`: bounded exact HF
  reference generation and prefill/teacher/autoregressive readiness runners.
- `tests/test_full_model.py` and `tests/test_full_model_perf.py`: static,
  real-weight, trace, mixed-prompt, accuracy, quality, performance, context,
  cold/warm host-store, sampler A/B, watcher, and profiler gates.

Endpoint checkpoint tensors are mmap-read one at a time and released after TT
materialization. Decoder layers use `from_checkpoint_host_backed`; at no point
is the complete expert checkpoint or PLE table transiently resident in RAM or
TT DRAM.

## Fresh HF readiness reference

Command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
python models/autoports/qwen_qwen3_8_flash_next/demo/generate_hf_reference.py \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/readiness_aime24_chat.refpt
```

Result: exact checkpoint/tokenizer revision, first DeepSeek AIME24 prompt,
checkpoint chat template, 201 prompt tokens, 100 HF greedy tokens, and
`[100,100]` top-token IDs. HF model load was 9.696 s, first step 39.193 s,
and total generation 97.309 s. The bounded oracle made 35,121 expert reads
with 25,143 host-cache hits and 5.272 s expert read time. PLE made 100 lookup
calls, selected 4,800 rows, and read 3,480 table rows. Metadata records
tokenizer/template/source hashes so no stale exact-match assertion is used.

Artifacts: `readiness_aime24_chat.refpt` and
`readiness_aime24_chat.json`.

## Bring-up and trace correctness

Reduced real-weight gates covered embedding-to-terminal top-100, a complete
layer-0 text endpoint, model-only compatibility trace, token-out split trace,
generator token-out, mixed prompts/inactive row, and sampling strategy.

Trace evidence:

- sampled tokens equal full-logits host argmax for capture and replay;
- positions advance 3 -> 4 -> 5 exactly once per transition;
- token/position host-copy counters do not change after capture;
- unchanged page tables skip the copy; a changed page table updates the
  stable buffer without trace rebuild;
- mixed lengths `[1,33,0]` produce positions `[2,34,-1]`; inactive PLE
  history remains absent;
- `full48_after_singleton_fix.xml` passes a three-token, all-48-layer
  token-out trace.

The initial mixed-prompt test wrongly expected an inactive history not to be
initialized at reset; `mixed_prompt_inactive.xml` retains that test failure.
The implementation already initialized every request with the EOS history,
which is the PLE contract. The assertion was corrected to require the
inactive history remain unchanged, and `mixed_prompt_inactive_after_fix.xml`
passes.

## AutoFix history

The exact source investigations and proof chain are in `AUTOFIX.md`.
Material fixes:

1. Remove an endpoint residual double-deallocation (`AUTODEBUG.md`).
2. Make singleton routed-expert banks alias-aware so explicit force
   deallocation cannot destroy persistent slots.
3. Keep canonical GDN/PLE state in DRAM and use the existing shared L1
   multichip trace workspace for active batch-1 compute.
4. Remove only invalid decode 1D role `gdn_qkv_b_a@0:55`; all other selected
   dtype/fidelity/sharding/fabric/host policies remain.

The missing progressing-HF test was made an asserted reusable gate and wrapped
for fused/optimized implementations. The host-backed multichip layer-0
trajectory improves from about 0.96066 -> 0.91035 to
0.999825 -> 0.998188 over 12 transitions after the single-role fix.

## Accuracy

Command form:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
RUN_QWEN38_ACCURACY=1 pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_aime24_teacher_forcing_accuracy \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/aime24_teacher_99_l1_workspace_final.xml
```

Final current-source result, 228.12 s test body:

| Phase | Top-1 | Top-5 | Top-100 |
| --- | ---: | ---: | ---: |
| Prefill | 100% | 100% | 100% |
| 99 traced teacher-forced decode rows | 91.9192% | 100% | 100% |

Trace capture was 3.664468 s. The explicit full-logits compatibility path
measured 98 steady rows in 50.173196 s, 0.511971 s/token, or 1.953234 t/s/u.
This is accuracy/teacher-forcing throughput, not the optimized token-out
number.

Rejected controls are retained: `aime24_teacher_forcing_accuracy.xml`
(47.47/73.74/90.91 decode), `aime24_teacher_12_after_state_dram.xml`
(33.33/83.33/91.67), and `aime24_teacher_12_no_gdn_qkv_1d.xml`
(83.33/100/100). `aime24_teacher_99_final.xml` is the earlier passing
99-row run before the final dynamic workspace-memory cleanup; the final
artifact above reruns the complete gate afterward.

## Autoregressive qualitative check

Command used the same accuracy environment and
`test_full_model_aime24_autoregressive_quality`. Result:

- 100 traced greedy tokens, first divergence at token 5;
- TT is fluent English and remains on the Aya walking-speed problem;
- dominant-token fraction 0.06, zero adjacent repeats, 12 repeated
  four-grams, Latin-letter fraction 1.0, no mechanical degeneration;
- trace audit: 98 steady replays, token copies 2, position copies 2,
  page-table copies 1, compact readbacks 100, no host sampling/argmax and no
  host feedback reconstruction.

Artifacts: `aime24_autoregressive_100_final.xml` and the exact completions,
token review, human verdict, and audit in
`aime24_autoregressive_100_final.json`. A current-source rerun also retains
the exact 100 HF/TT token IDs, full generation metrics, degeneracy counters,
and complete fallback audit in `aime24_autoregressive_100_report_final.json`
and `aime24_autoregressive_100_report_final.xml`; it reproduces divergence at
token 5 and passes.

The independent first stage review correctly found that one math prompt did
not satisfy the shared qualitative-suite gate. A fresh three-prompt HF oracle
was generated with the exact checkpoint tokenizer/chat template, first at the
allowed 64-token minimum. Direct human review found the run non-degenerate but
mostly truncated inside visible reasoning, so it was rejected as weak
evidence. The oracle and TT suite were regenerated at the allowed 128-token
maximum:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.demo.generate_qualitative_reference \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_shared_suite.refpt \
  --expert-cache-capacity 32 --threads 16
RUN_QWEN38_QUALITATIVE_SUITE=1 pytest -q --timeout=900 --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_shared_qualitative_suite \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative_shared_suite_final.xml
```

The 128-token TT run passes all three prompts. Explanation matches HF for 40
tokens and coding for 42; both remain coherent, English, on-topic, and
non-degenerate but hit the 128-token ceiling before a complete answer. The TT
summary diverges immediately yet independently produces a complete correct
one-sentence result. Exact raw prompts/tokens/completions and fallback audits
are in `qualitative_shared_suite_final.json`; exact template hashes are in
`qualitative_prompt_format.json`; the bounded human verdict is in
`QUALITATIVE_REVIEW.md`. The common
`models/common/readiness_check/check_degenerate_output.py` script does not
exist in this checkout, so `_degeneracy` plus direct review is the declared
substitute.

## Common sampler comparison

```bash
RUN_QWEN38_SAMPLER_AB=1 pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_reduced_real_weight_split_greedy_sampler_strategy_ab \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/split_greedy_sampler_strategy_ab_final.xml
```

Both `Sampling1D` strategies equal host argmax on the real-logit A/B.
Full-vocabulary force-argmax measured 0.665837 ms versus 0.906958 ms for
local-top32 k=1 and is selected. Local-top32 was also rejected because it
lacks the generic sampler's lowest-global-index tie adjustment. `TTSampling`
was also reviewed: it is the
generic 2D, max-batch-padded, constructor-parameter surface and is a poorer
fit than `Sampling1D` for the LMHead1D sharded output and per-call state.
No custom sampler was needed.

The review also found that non-greedy device seeds were accepted by the public
generator but ignored by the model. AutoFix added a request-local RNG and a
declared compact per-token H2D update into the persistent `Sampling1D` seed
buffer only for explicit non-greedy mode. Greedy measurements are unchanged.
`non_greedy_split_trace_final.xml` passes `top_k=4`, `top_p=0.95`, temperature
0.8, seed 12345 across eager sample, trace capture/replay, changed and
unchanged page tables, positions 4 through 7, and sampled -> greedy -> sampled
trace invalidation. It records four seed copies and no token/position copies
after capture; every sampled token belongs to the current top-4 set. Greedy
still exactly matches full-vocabulary host argmax.

## Full token-out performance

```bash
RUN_QWEN38_PERF=1 pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_batch1_prompt128_generate128_performance \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/batch1_prompt128_generate128_performance_final.xml
```

Full 48 layers, prompt 128, generate 128, device greedy sampling:

- TTFT 104.640022 s; prefill 104.635425 s;
- trace capture 4.226754 s;
- 126 steady measured tokens in 76.677881 s;
- 0.608555 s/token, **1.643238 t/s/u**;
- 126 steady trace replays, two token initialization/capture copies, two
  position initialization/capture copies, one page-table copy, 128 compact
  readbacks, zero host sampling calls;
- representative final submit: 0.487758 s, exact expert service 0.397370 s,
  PLE 0.000887 s.

The inherited exact-host decoder medians provide an optimistic 111.817691 ms
48-layer decoder sum (8.943 t/s), before endpoints and host service. The
496.737 ms gap to measured token-out is dominated by exact expert service,
source packing/misses, compact control/DMA, endpoints, and orchestration—not
the canonical sampler.

## Host-store cold/warm and reset

```bash
RUN_QWEN38_HOST_COLD_WARM=1 pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_cold_and_warm_chunked_prefill \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/cold_warm_chunked_prefill_final.xml
```

The final prompt-128 cold pass was 104.769271 s with 10,994 packed misses,
60,792,422,400 expert H2D bytes, 92.197182 s source packing, and 1,816 PLE
table rows / 581,120 bytes. The warm pass was 11.315565 s with 10,994 packed
hits, zero packed misses, zero source packing, the same exact expert H2D
volume, and zero PLE table reads. Both passes produced on-device greedy token
248046; the test asserts their equality and records every counter as a JUnit
property.

`tests/test_host_weight_cache.py` covers cache ownership, exact misses/hits,
partial and capacity-one eviction, reload, stale generation, upload-failure
invalidation, ordered/indexed publication, real PLE hashing/rows, EOS and
two-token carry, chunk carry, reset/cancel, repeated lookup, and mixed-request
isolation.

## Context capacity

The first advertised-context construction succeeded but its test assumed
128-wide RoPE while the checkpoint uses 64 rotary columns; the assertion-only
failure is retained in `advertised_context_construction.xml`. The corrected
run passes in `advertised_context_construction_final.xml`:

- all 48 layers, 12 QSA cache sets, page table `[1,4096]`;
- two RoPE tensors `[1,1,262144,64]`;
- persistent page/token/position/sampler payload 26,712 bytes/device;
- RoPE 67,108,864 bytes/device;
- full endpoint runtime charge 67,135,576 bytes/device.

Recomputed total is 10,170,438,744 bytes/device, leaving
24,055,081,896 bytes/device. Advertised context remains 262,144 with no
reduction. Exact values and the largest-feasible evidence are in
`../context_contract.json` and `../host_weight_contract.json`.

The first stage review also correctly distinguished inherited single-layer
batch-32 evidence from the delivered full model. The new all-48 gate builds
`max_batch=32,max_seq_len=4096`, uses active slots 0 and 31 with prompt lengths
1 and 33, leaves 30 fixed slots inactive, supplies distinct/flipped page-table
rows, produces two eager on-device-sampled tokens, and repeats on the same
model with new request IDs. `full48_batch32_eager_fixed_slots.xml` passes in
119.46 s with outputs `[[15,16],[62,63]]` on both epochs. Positions are
`[2,-1,...,-1,34]`; page ownership is exact; inactive PLE histories stay EOS;
old histories are preserved; active histories carry the prompt tail and first
generated token; and device-feedback counters hold. A conservative plan
replaces the batch-1 max-context cache/runtime charges with batch-32 context
4096 charges and totals 23,558,946,904 bytes/device, leaving
10,666,573,736 bytes/device. Segmented tracing remains batch 1; all-48 eager
fixed-slot capability is now proven at batch 32.

## Tracy and tt-perf-report

`profiler_provenance.txt` records the exact commands and hashes. One combined
three-layer profiler attempt overflowed device buffers and was rejected.
Capacity-safe one-layer captures for representatives 0, 1, and 3 all passed
and postprocessed without drops/missing rows. Their signpost-filtered merged
device totals are:

| Representative | Token-out ops/time | Sampling ops/time |
| --- | ---: | ---: |
| GDN layer 0 | 195 / 3995.579 us | 3 / 534.749 us |
| PLE+GDN layer 1 | 265 / 4411.602 us | 3 / 535.116 us |
| QSA layer 3 | 285 / 5350.275 us | 3 / 537.020 us |

The sampler's untilize/all-gather/argmax window is roughly 0.535 ms and does
not dominate full token-out. Raw `.csv.xz`, detailed reports, summaries, and
PNGs are under `tracy_reduced_decode/`.

## Watcher, static gates, review, and commits

The final watcher command runs separately from Tracy:

```bash
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DISABLE_ETH=1 \
RUN_QWEN38_FULL_MODEL=1 pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_48_layer_token_out_trace_smoke \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/full_model/full48_tokenout_watcher_fixed.xml
```

The first full-stack watcher run exposed a BRISC assertion in the non-mux
Linear async all-gather writer. A one-layer split-greedy watcher run reproduced
it before trace capture; AutoTriage mapped the reported line to an
unconditional `get_forward_connection()` on a zero-target line endpoint. The
generic writer now retrieves only existing directions and asserts that a null
direction has zero targets. `reduced_split_trace_watcher_fixed.xml` passes the
focused eager/capture/replay contract, and the original 48-layer command then
passes in 87.56 s with tokens `[16,9,24]`, one steady trace replay, direct
device feedback, and clean watcher teardown. Post-run `tt-smi -s` reports all
four boards, DRAM healthy, and zero uncorrectable GDDR errors. Full evidence is
in `AUTOTRIAGE.md`, `AUTOFIX.md`, and the pre/post watcher logs.

The accepted untracked trace runs also emitted TTNN's generic warning that
allocations younger than an active trace can be unsafe. AutoDebug localized
the likely source to state snapshot clones created after ingress capture, then
required the runtime tracker rather than accepting source reasoning alone.
Both the reduced split trace and the original full-48 token-out trace pass
under `TT_METAL_TRACE_ALLOC_TRACKING=1`, traceback depth 12, and watcher, with
no unsafe-allocation `RuntimeError` and clean replay/teardown. Evidence:
`reduced_split_trace_alloc_tracker.xml`,
`full48_tokenout_trace_alloc_tracker.xml`, and
`AUTODEBUG_TRACE_ALLOC.md`. The tracker proves younger allocations are freed
or marked safe before replay and classifies the generic warning as controlled.

The final pre-rereview CPU/static contract run is
`static_host_contracts_rereview.xml`: 13/13 tests pass. It covers the host-weight
cache and PLE ownership/reset/isolation contracts, recomputed full-stack
capacity including the 67,135,576-byte endpoint runtime, and the explicit
generator/sampling policy. `py_compile`, both JSON parses, and
`git diff --check` also pass. Final post-performance `tt-smi -s` reports all
four P300 boards present, healthy DRAM, and zero uncorrectable GDDR errors.

The first independent xhigh stage review returned `more-work-needed` for the
unclassified trace-allocation warning, missing shared qualitative suite,
unexercised non-greedy trace/seed path, and lack of all-48 batch>1 evidence.
Every finding above was fixed with current-source artifacts before the fresh
rereview. Final verdict and local commit SHAs are appended after review and
commit. Nothing is pushed.

## Declared limitations

- exact host-backed token-out tracing is batch 1; the all-48 host-backed eager
  fixed-slot generator is validated at batch 32 and context 4096;
- cold expert union packing makes prefill slow; even warm prefill must DMA
  exact routed weights into bounded wave slots;
- exact expert/PLE service is serialized and no overlap is claimed;
- Torch pinned allocation is unavailable in the installed CPU-only build;
- the checkpoint exposes xhigh reasoning and can exhaust the qualitative
  suite's maximum 128-token budget before completing a short answer;
- deprecated CCL-argument and nanobind shutdown warnings remain runtime/tool
  issues; accepted gates have no watcher, NoC, assertion, panic, or hang.
