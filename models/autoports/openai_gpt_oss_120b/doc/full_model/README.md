# GPT-OSS 120B full model

## Performance and status

The resident 36-layer P150x4 full model passes the full-model correctness,
readiness, trace, qualitative, profiler, and repeated-logit gates.  Independent
stage review returned `clean-pass`.

| Batch-1 AIME24 path | Prompt | TTFT | Decode | Sampling boundary |
| --- | ---: | ---: | ---: | --- |
| Free-running token-out | 214 tokens | 3.7757 s | **51.6229 t/s/u** over 99 decode tokens | traced on-device top-k=1, device token feedback |
| Teacher forcing | 214 tokens | 3.7515 s | **52.1145 t/s/u** over 99 decode tokens | traced on-device prediction with an explicit host-token compatibility refresh |

These are warmed-program measurements on a `(1,4)` Blackhole mesh.  The
available boards report `p300c`; the model uses the requested P150x4 semantic
topology and exact four-device tensor-parallel policy.  Teacher-forcing and
free-running numbers are kept separate because only the latter is the measured
serving token-out path.  The shared six-prompt suite measured 54.54 t/s/u on
the first capture-bearing prompt and 62.43--62.57 t/s/u on the five subsequent
short-context prompts.

## Correctness

The fresh AIME24 Harmony/chat reference contains 100 HF tokens and top-100
sets from `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
Its SHA-256 is
`7e722ad241eee84148ed62b5accee20bc642a4a1de4cab98ae146a166ee9d2bc`.

| Gate | Top-1 | Top-5 | Top-100 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `run_prefill_check` | 94% | 100% | 100% | pass |
| `run_teacher_forcing` | 95% | 100% | 100% | pass |

The 16-step reduced real-weight split-greedy comparison is exact between the
canonical device sampler and host argmax.  It is semantically greedy:
temperature zero is normalized by the shared sampler to top-k=1/top-p=0.  The
device arm has zero host argmax and zero full-logit readbacks.

The full 36-layer, full-context batch-2 reproducibility gate is bitwise exact
for both prefill and traced decode: identical prompts in distinct physical page
rows match each other, and both rows match across two reset/reuse runs.  Raw
logit and top-100 hashes are identical, all maximum differences are zero, and
the greedy tokens are the first two pinned HF reference tokens (`200005`,
`35644`).  A prior invocation-dependent failure was reproduced with finite,
HF-matching greedy tokens, diagnosed by fresh-context AutoFix, and cleared by a
bounded physical device reset; the unchanged gate then passed twice in separate
processes.  This preserves the exact gate and records stale external
fabric/collective state rather than accepting nondeterministic logits.

## Full-model architecture

`tt/model.py` streams every checkpoint layer once, constructs 36 optimized
`MultichipDecoder` blocks, then attaches the maintained GPT-OSS embedding,
RoPE, final RMSNorm, and TP-sharded LM head.  It does not select a single-chip,
replicated-weight, host-executed, or reduced production path.

The inherited optimized policy is unchanged:

- 1D tensor parallelism on mesh axis 1;
- logical BF16 replicated residuals, L1-interleaved for decode and
  DRAM-interleaved for prefill;
- BFP8 attention weights and paged KV cache, with local KV heads per TP rank;
- LoFi decode projections, HiFi2 packed-QKV prefill, and LoFi output projection;
- BFP8 attention CCL for decode, BF16 prefill attention CCL, and BF16 expert CCL;
- BFP4 expert weights, BF16 replicated router, top-4 routed indexed sparse decode;
- selected 45-core gate/up, 15-core decode-down, and 45-core prefill-down geometry;
- TP4 physical 2944-column attention reduction followed by the logical 2880 slice.

The optimized multichip stage's candidate/rejection ledger remains authoritative;
the full model instantiates `DEFAULT_MULTICHIP_POLICY` for every layer and adds no
fallback candidate.

## Context and resident capacity

The public contract remains 131072 tokens, including non-aligned prompt lengths.
The generator owns prompt slicing/padding, masking, cache fill, positions, and
logical output slicing.  Full resident accounting includes measured decoder
tensors, embedding, final norm, sharded LM head, RoPE, paged BFP8 KV cache/page
state, and a 2 GiB trace/activation reserve.

| Target | Batch-1 full-context bytes/device | GiB/device | Result |
| --- | ---: | ---: | --- |
| P150 | 74,895,259,776 | 69.751646 | rejected; fixed state already exceeds 32 GiB, feasible context 0 |
| P150x2 | 39,272,108,928 | 36.575002 | rejected; fixed state already exceeds 32 GiB, feasible context 0 |
| P150x4 | 21,543,073,152 | 20.063550 | accepted at 131072 tokens |

P150x4 supports batch 10 at the full context under this accounting.  Batch 11's
largest capacity context is 130880 and batch 32's is 44992.  These are physical
capacity results, not an advertised reduction of the batch-1 model context.

## Serving and trace contract

`tt/generator.py` exposes low-level prefill/decode with explicit KV cache,
page table, prompt lengths, positions, batch state, fixed slots, and inactive
rows (`-1`).  Hardware tests cover mixed non-aligned prompts in two physical
cache rows and a decode batch containing one inactive row.

The free-running AIME path recorded 99 decode calls/replays, one initial full
input refresh, 98 unchanged page-table reuses, zero per-token forced refreshes,
zero host argmax calls, and zero full-logit readbacks.  The refreshed resident
36-layer smoke makes the host boundary explicit: one token-input, one
position/RoPE, and one page-table upload at trace setup; zero steady-state
uploads for all three; four caller-visible scalar-token synchronizations; and
zero validation full-logit synchronizations.  The reduced trace probe also
verifies the sampled token, incremented position, unchanged page table, and a
later page-table-only refresh after an explicit page change.  Python reads the
sampled token for the caller but deliberately does not feed it back; the
sampling trace writes `tt_out_tok` into the persistent decode input.

Host sampling remains available only through explicit
`sampling_mode="host"`.  Teacher forcing similarly marks caller-provided tokens
as authoritative.  Normal token-out uses neither boundary.

## Sampler and profiler verdict

The selected common implementation is
`models.common.sampling.generator.SamplingGenerator`.  The stateless
`models.common.modules.sampling.sampling_1d.Sampling1D` alternative is rejected
because it has no matching trace/token-feedback, seed, penalty, log-prob, or
per-request state owner.  No custom sampler was written.

A reduced profiler run retained one sliding and one full-attention layer plus
the real terminal stack.  In a steady capture the sharded LM-head matmul takes
708.553 us and `SamplingDeviceOperation` takes 27.500 us: sampling is 3.88% of
LM-head device time and about 0.14% of the measured 19.371 ms/token full-model
wall time.  Sampling does not dominate, so the canonical LM-head/sampling
contract is accepted.

The best optimized P150x4 layer measurements are 0.396414 ms for each of 18
sliding layers and 0.396354 ms for each of 18 full-attention layers.  Their
14.2698 ms/token summed decoder-stack lower bound accounts for 73.67% of the
19.3712 ms/token full-model token-out wall time.  The 5.1014 ms/token (26.33%)
full-model-only residual includes 0.7086 ms LM head and 0.0275 ms sampling;
the remaining 4.3653 ms is an upper bound for final norm, trace
replay/orchestration, synchronization, caller-visible scalar token readback,
and wall/device timer gaps.  No per-token token, position/RoPE, or page-table
host refresh occurs on the free-running path, so none of that residual is an
accepted steady-state input-rebuild fallback.

## Qualitative verdict and limitations

The HF and TT AIME completions are coherent English analyses of the same
equations.  TT first diverges on the 19th generated token but remains on-topic;
there is no repetition, wrong-language drift, or malformed text.  Both are cut
at the requested 100-token evidence boundary.

All six shared chat prompts are coherent and on-task.  Exact-checkpoint HF
controls use the same six prompt token sequences, the same chat template, and
greedy 128-token settings.  Every HF/TT pair is coherent and semantically
aligned: common prefixes are 38/3/3/17/3/26 tokens, with later differences in
wording or answer structure rather than language drift or collapse.  The TT
translation terminates correctly at the checkpoint's `<|return|>` EOS after 81
tokens; its HF control is merely cut at the 128-token evidence limit.  The
machine checker reports no degeneration across the shared suite and AIME
artifact.  Several long-form answers are intentionally truncated at 128 tokens,
so the evidence demonstrates generation health rather than answer completeness.

P150 and P150x2 cannot host the resident full stack.  P150x4 is therefore the
only production target in this stage.  vLLM work is intentionally absent.

## Artifacts

- `artifacts/prefill_readiness.json`
- `artifacts/teacher_forcing_readiness.json`
- `artifacts/split_greedy_host_comparison.json`
- `artifacts/full_model_token_out_smoke.json`
- `artifacts/logit_reproducibility.json`
- `artifacts/logit_reproducibility_probe.json`
- `artifacts/autoregressive/`
- `artifacts/profiler/`
- `references/aime24_chat_100_top100.refpt`
- `qualitative/qualitative_hf_chat.json`
- `qualitative/qualitative_hf_tt_comparison.json`
- `qualitative/qualitative_tt_chat.json`
- `qualitative/degenerate_check.json`
- `runtime_fallback_audit.md`
- `sampler_decision.md`
- `work_log.md`
