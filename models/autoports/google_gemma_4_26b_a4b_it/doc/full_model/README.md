# Gemma 4 26B A4B full model

This stage completes the repo-local TTNN autoregressive path in `tt/model.py`
and `tt/generator.py`. It stacks all 30 optimized multichip decoder layers,
adds sharded tied embeddings, final norm, vocabulary-sharded LM head, paged KV
state, and split traced sampling. It deliberately does not start vLLM
integration.

## Headline full-model performance

On the P150x4 proxy at batch 1, prompt 128 / generate 128, the public traced
token-out path measures **139.029 ms TTFT**, **46.322 request decode t/s/u**,
and **46.327 warmed decode t/s/u**. The lower-level trace-only split
model-plus-device-sampling boundary measures **46.788 t/s/u** with no loop
readback or synchronization. Allocation-tracked teacher forcing measures
**2.825 t/s/u** on P150x4 and is correctness evidence, not token-out
performance; it intentionally performs host token feedback and tracking. The
boundaries and corresponding artifacts are detailed below.

## Delivered contract

`Gemma4FullModel` accepts distinct 1x1, 1x2, and 1x4 meshes. The matching P150,
P150x2, and P150x4 proxy profiles retain real tensor parallelism throughout:
TP1 uses identity reduction, TP2 uses persistent BF16 Linear all-reduce with
one link, and TP4 uses persistent BF16 Ring all-reduce with two links. TP2 and
TP4 allocate exactly three shared persistent all-reduce buffer/semaphore sets
after all layer weights are placed, then every layer reuses those resources.

The optimized decoder policy is unchanged: exact dynamic top-8 indexed sparse
experts, packed expert gate/up, profile-specific attention/dense weight
exceptions, BF16 activations and CCL payloads, and the replicated BF16 TILE
DRAM inter-layer residual `[1,1,logical_M,2816]`. The 25 sliding layers use
caller-visible BF16 paged caches with 64-token blocks and a 1,024-token window;
the five full-attention layers use BF16 128-token blocks through the supported
profile context. The earlier BFP8 cache rejection remains binding. Tied
embedding/LM-head storage is BFP8_B and sharded across hidden/vocabulary axes,
which is the capacity-required terminal policy recorded by the multichip
stage.

No external datatype-sweep configuration exists in this checkout. The model
therefore intentionally resolves `tt/precision_policy.py` defaults and the
optimized decoder's inherited per-layer table; provenance records the loader
source and hash rather than claiming a missing configuration artifact.

The low-level generator API exposes explicit cache, page tables, positions,
prompt lengths, active rows, and sampling parameters. Logical prompts are
padded and masked internally, so non-tile/page-aligned lengths remain public
inputs. A separate host compatibility mode exists for readiness consumers;
the optimized path never uses it.

## Split traced token-out path

Decode is captured as a model trace ending in sampler-ready sharded logits and
a second sampling trace using `models.common.modules.sampling.Sampling1D`.
`tt_out_tok` aliases the persistent model token input, while current-position
and RoPE tensors advance with `plus_one` inside the model trace. Unchanged page
tables are not copied. A scheduler mapping change copies once into the same
stable device addresses and reuses the existing trace.

The selected greedy path keeps local top-32 candidates and invokes the common
top-k/top-p sampler with `k=1`, `p=0`, and temperature 1. The same API supports
non-greedy top-k/top-p parameters with distinct trace keys. The other common
implementation, `models/common/sampling/generator.py` (TTTv1), was rejected
because its generator-owned penalty/seed state and high-level trace lifecycle
duplicate this model's explicit serving state. `Sampling1D` directly supports
the required 1D TP topology, padding, and `tt_out_tok` contract. Within
`Sampling1D`, force-argmax full-vocabulary gather was also rejected. The
terminal soft-cap makes several vocabulary entries tie at logit 30 for the
probe, so split top-k's token 495 and force-argmax's token 1 are both valid
global maxima. The semantically equivalent split path is much faster on the
same logits: 0.446 ms versus 2.296 ms per sample.

Sampling-parameter changes preserve the prior trace's device token input while
the old trace IDs are released and the new model/sampling pair is captured.
Because sampling capture executes and aliases `tt_out_tok` to that input, the
generator makes a temporary device-side backup and restores it device-to-device
before first replay.
The allocation-tracked TP4 transition regression deliberately supplies stale
host token 0 through greedy -> sampled -> greedy recaptures and compares it
with a correct-token control. Both directions retain the same sampled token,
top-1, and top-100 set, with logits cosine 0.99996 or better. This closes the
recapture boundary without adding a host token-feedback step.

Identity-distinct scheduler page tables are copied once into the existing
generator-owned stable tensors, so their addresses and both trace IDs remain
unchanged. Reusing the same source performs no additional copy; mutating that
source requires an explicit `refresh_page_tables` call. The sampled control in
`sampling_rng_tp4.json` proves this adoption leaves the random stream unchanged.

`mixed_state_tp4.json` records seven replays covering stale host inputs,
active-row switching, an all-inactive step, a page-table update, and retained
teacher-trace reuse after reset. Steady replays perform no host token,
position, RoPE, page-table, synchronization, or readback work. One intentional
page-table change increments only its refresh counter. `token_out_trace_tp4`
records 134 replays; after the request-boundary copies, steady token-out has no
refresh, readback, or synchronization.

## Capacity and capability

All three mandatory profiles construct the complete 30-layer real-weight
model plus terminal tensors, full paged-cache state, page tables, sampler
buffers, trace reserve, and persistent CCL state where applicable.

| Profile | Mesh | Supported context | Exact BF16 KV/device | Allocated/device | Free/device | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P150 | 1x1 | 50,624 | 1,247,805,440 B | 28,699,084,288 B | 5,412,537,856 B | load/capacity/prefill/decode pass |
| P150x2 | 1x2 | 262,144 | 2,789,212,160 B | 17,975,282,688 B | 16,136,339,456 B | load/capacity/prefill/decode pass |
| P150x4 | 1x4 | 262,144 | 2,736,783,360 B | 12,052,469,248 B | 22,059,152,896 B | load/capacity/prefill/decode pass |

The allocator snapshot is post-construction. The retained P150 reduction is
based on the stricter earlier construction/prefill projection, including
source-live peaks and operational reserve: 34,091,067,904 bytes/device with
268,670,464 bytes headroom at 50,624 tokens. It cannot safely claim the HF
262,144-token context. P150x2 and P150x4 retain 861,289,472 and 5,026,182,144
bytes of conservative projected headroom at the advertised context. Exact
formulas and artifacts are in `../context_contract.json`.

The public generator passed nonaligned prefill at 50,623 tokens on P150 and
262,143 tokens on P150x2/P150x4, followed by the one remaining legal traced
decode replay. The next decode is rejected by the host guard at capacity. The
same full 30-layer TP4 path passed fixed-slot batch 32 with prompt length 32,
two decode replays, all 32 sampled rows checked against the global-max logit,
and a 128 MiB trace region. B1-versus-B32 logits have minimum cosine 0.9929.
The allocation-tracked batch-32 run completed without an unsafe-allocation
report.

## Correctness and qualitative evidence

The fresh main reference is the exact pinned checkpoint, AIME24 prompt 0,
Gemma chat template, 161 prompt tokens, 100 generated tokens, and top-100
teacher data. Full-stack prefill scores 95/100, 96/100, and 95/100 top-1 on
P150/P150x2/P150x4; traced teacher forcing scores the same. Every profile is
100/100 top-5 and top-100. The teacher runs use strict fallback and trace
allocation tracking; their 2.815, 2.795, and 2.825 decode t/s/u are integrity
evidence, not primary performance numbers because tracking and host teacher
feedback perturb latency.

Free-running traced generation was refreshed after the final trace-state fix
and reviewed against same-revision HF controls. The shared six-prompt readiness
suite covers haiku, supervised/unsupervised learning, an inventor story,
thermodynamics, French translation, and Fibonacci. Every prompt uses the
pinned tokenizer's chat template and a 64-token greedy HF control; allocation
tracking was enabled and the same TT model was reset between requests. All six
TT outputs are coherent, task-aligned, and free from mechanical repetition or
wrong-language drift. A separate three-prompt run, also 64 tokens per prompt,
retained one trace across `reset()` and passed allocation tracking. All TT outputs
are coherent English, remain on task, and show no prompt echo, control-token
leakage, doubled tokens, mechanical repetition, or cross-request leakage. The
explanation prompt matches HF for the first 44 tokens; the bicycle and rainbow
prompts diverge earlier but remain coherent and structurally comparable to
their controls. A raw story continuation is retained only as labeled stress
coverage: both HF and TT repeat phrases under greedy completion, and it is not
the instruct-model quality verdict. The shared degeneracy checker reports no
critical or advisory finding for the main chat run or raw control.

## Performance and reduced profiler

The primary full-stack P150x4 workload is the public host-visible
`generate()` contract at B1, prompt 128, generate 128 after a separate warmup
request. It measures 139.029 ms TTFT, 44.433 end-to-end t/s/u, 46.322 cold
request decode t/s/u, and 46.327 warmed decode t/s/u. This boundary includes
first-token sampling, caller-visible Python tokens, and 128 minimal token
readbacks. A separate prompt-128 prefill component measures 106.118 ms warmed
through last-token sampler-ready logits (205.931 ms initial).

Two lower-level position-32 probes characterize the optimized device boundary.
The model-only trace through sampler-ready logits is 20.9238 ms/token or
47.7925 decode/s. The split model-plus-sampling trace, including final norm,
vocabulary-sharded LM head, on-device sampling, device token feedback, and
both replays, is 21.3730 ms/token or 46.7880 t/s/u, with zero measured-loop
host readbacks or synchronizations. These component probes use related but
separately captured traces, so their difference is not claimed as a matched
sampling-overhead measurement.

The best decoder-stack lower bound from the preserved per-layer measurements
is 23.4163 ms on P150, 19.7875 ms on P150x2, and 19.2025 ms on P150x4. The
position-32 TP4 logits boundary is 1.7213 ms above its decoder-stack lower
bound; the complete device token-out boundary is 2.1705 ms above it. This is
consistent with terminal norm, LM-head, padding/layout, and sampling work
rather than an inter-layer gather or host-stepped decode loop, but is not a
same-prompt end-to-end attribution.

Tracy was run only on the reduced real-shape TP4 model with layers 0 and 5,
real terminal path, paged state, split traces, and sampling. `tt-perf-report
--arch blackhole --active-experts 8` reports 3.967 ms summed device operation
time and 12.9% modeled DRAM roofline. The largest family is a terminal BF8 to
BF16 untilize at 1.375 ms. This is the BFP8 tiled embedding table being
converted to the row-major BF16 format required by the current embedding op;
it is not LM-head conversion. The four matmuls total 0.697 ms, sampler top-k plus
sampling families are not the dominant cost, and two-layer all-reduce totals
0.108 ms. The compressed raw capture and processed report/CSV/PNG are under
`artifacts/profiler/`.

Focused real-shape terminal trials are retained in
`artifacts/terminal_trials.json`. A persistent row-major BF16 embedding reduces
the measured TP4 embedding path from 1.430 ms to 0.136 ms but would require
692,060,160 additional bytes/device on TP1, exceeding its 268,670,464-byte
conservative headroom and violating the capacity-required BFP8 storage policy.
The generic BFP8 `gather` alternative produces non-finite data and takes 76.050
ms. For the LM head, a monolithic DRAM-sharded TP4 candidate fails L1 circular-
buffer allocation; the valid 8,192-column split measures 0.1231 ms per linear,
so its eight-linear 0.9850 ms lower bound already exceeds the 0.5904 ms
incumbent before concat. The mature TP4 interleaved config improves only 0.43%,
and both pre-scaled softcap alternatives change the top-100 set. All were
rejected; no less-correct or profile-specific terminal path was retained.

## Evidence index

- `artifacts/full_stack_probe_tp{1,2,4}.json` and `final_profiles.xml`:
  allocation-tracked complete 30-layer load, prefill, and traced
  token-feedback decode on each profile.
- `artifacts/capacity_tp{1,2,4}.json`: complete stack at each maximum profile
  context with allocator, cache-shape, terminal-dtype, and persistent-CCL data.
- `artifacts/long_context_probe_tp{1,2,4}.json`: public nonaligned long prefill
  and traced decode using one real layer of each attention kind.
- `artifacts/batch32_probe_tp4.json`: complete 30-layer B32 capability gate.
- `artifacts/readiness_prefill_tp{1,2,4}.json`,
  `readiness_teacher_forcing_tp{1,2,4}.json`, and
  `gemma4_aime24_chat.refpt`: all-profile accuracy gates and exact-revision
  reference.
- `artifacts/generator_generate_tp4.json`: primary public B1 prompt-128,
  generate-128 performance boundary.
- `artifacts/prefill_tp4.json`, `logits_only_trace_tp4.json`, and
  `token_out_trace_tp4.json`: prefill and position-32 component boundaries.
- `artifacts/mixed_state_tp4.json`, `logical_tail_tp4.json`,
  `sampling_rng_tp4.json`, `public_sampling_rng_tp4.json`,
  `generator_generate_tp4.json`, and `sampler_ab_tp4.json`: split-trace state,
  logical-tail, transition/RNG/reset lifecycle, public-generator, and sampler
  decision evidence.
- `artifacts/qualitative/`: rendered prompts, token ids, HF controls, TT
  completions, shared-suite source/assessment, degeneracy reports, and verdict.
- `artifacts/profiler/`: compressed raw Tracy data and `tt-perf-report`
  outputs for the reduced two-layer variant.
- `perf_summary.json` and `artifacts/terminal_trials.json`: machine-readable
  performance boundaries and focused terminal candidate/rejection evidence.

All implementation changes are Python-only, so `AGENTS.md` does not require a
C++ build. Final commands and commit records are retained in `work_log.md`.
