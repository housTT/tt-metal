# Qwen3.6-27B full model

## Result

This stage builds the complete Hugging Face autoregressive text path for
`Qwen/Qwen3.6-27B`: replicated BF16 embedding, the selected 64-layer TP4
optimized multichip decoder, replicated final RMSNorm, a TP4 vocabulary-column
BFP8 LM head, paged cache/state ownership, and the standard Metal readiness
generator. It targets exactly four Blackhole P300c devices in a `1x4` ring; no
single-chip, host-executed decoder, replicated-model, or vLLM path exists.

The optimized decoder policy is preserved unchanged: replicated BF16
TILE/DRAM inter-layer residuals; BFP8 attention projections; BFP4 linear MLP
weights; BFP4 full-attention gate/up and BFP8 down; BF16 attention/full-MLP CCL
payloads; BFP8 linear-MLP CCL payloads; persistent async ring collectives; BFP8
paged KV cache; and the inherited compute-fidelity choices. The only decoder
repair is a numerically stable, exact 64-token GDN block inverse. It keeps the
64-token outer chunk and ten TT matmuls rather than falling back to host math or
16-token execution.

## Correctness and performance

The reference is a fresh 100-token greedy HF continuation for AIME24 prompt 0,
rendered with the checkpoint tokenizer's chat template. Its file timestamp is
2026-08-20 07:55 EDT; it contains one 161-token prompt, 100 generated tokens,
and a `[100,100]` top-k table.

| Gate | Top-1 | Top-5 | Top-100 | Result |
|---|---:|---:|---:|---|
| Full 64-layer prefill | 97/100 | 100/100 | 100/100 | pass |
| Full 64-layer traced teacher forcing | 97/100 | 100/100 | 100/100 | pass |

| Batch-1 performance | Result |
|---|---:|
| TTFT from readiness teacher forcing | 14,620.75 ms |
| Warmed TTFT, prompt 128 in representative 128/128 run | 742.651 ms |
| Teacher-forcing traced decode | 19.22 t/s/u |
| Caller-visible autonomous token-out, prompt 128 / generate 128 | 20.323 t/s/u, 49.205 ms/token |
| Device-only model + sampler trace pair | 22.445 t/s/u, 44.553 ms/token |
| 64-layer model trace | 42.140 ms |
| Canonical Ring force-argmax sampling trace | 2.416 ms |

The teacher-forcing number includes a per-token host write of the ground-truth
token, required only by that accuracy harness. The caller-visible number
measures 127 autonomous decode intervals after the prefill-selected token and
includes the public generator's sampled-ID readback. The device-only
attribution number times the model trace followed by the sampling trace with
direct device token feedback. Sampling is 5.4% of that combined latency and
therefore does not dominate the production stack.

The final Tracy/`tt-perf-report` capture profiles real checkpoint layers 0 and
3, covering both unique layer kinds: linear attention and full attention. It
also includes final norm, the TP LM head, and selected Ring force argmax. Its
4.361 ms summed device time across 195 merged rows contains no top-k, includes
one `SdpaDecodeDeviceOperation`, two paged-cache updates, and four AllReduce
rows. Argmax is 1.418 ms (32.51%), Ring all-gather is 0.884 ms (20.26%), and 29
matmuls total 1.221 ms (28.00%). The older one-linear-layer selected capture
and rejected Linear capture remain as comparison artifacts. The uninstrumented
64-layer split measures sampling at 2.416 ms, 17.44x shorter than its 42.140 ms
model trace. Final compact artifacts use the `two_kind_ring_` prefix.

The optimized-decoder medians provide a layer-stack lower bound: 48 linear
layers x 0.718857 ms plus 16 full-attention layers x 0.476422 ms = 42.127888
ms. The final 64-layer model trace is 42.140081 ms, only 0.012193 ms (0.029%)
above that bound. Combined token-out is 44.552574 ms, leaving 2.424686 ms over
the decoder bound—effectively the measured 2.416293 ms sampler trace. This
cross-check supports preservation of the optimized multichip stack.

## Generator and state contract

`FullModelState` exposes page-table, paged KV-cache, recurrent linear state,
current positions, rotary positions, persistent input token storage, prompt
lengths, active slots, block count, and page-table ownership. Public prefill
accepts arbitrary valid logical lengths, including non-aligned and mixed
lengths. It owns 32-row internal padding, causal masking, cache fill, positions,
and output slicing. Mixed `[3,5]` prompts and inactive rows passed the reduced
hardware gate. Decode uses stable fixed prefix slots `[0, active_batch)`; rows
outside that range keep position `-1` and are skipped by the linear-attention
state path. Batch 32 remains supported.

An internal page table is packed by `ceil(prompt_len/64)` blocks per user from
one physical pool. An external table remains caller-owned, may have a narrower
active width, and is copied only when content changes. Unchanged tables incur
no rebuild or transfer. Reset clears KV and linear state in place and retains
the allocations.

The low-level `prefill_forward` and `decode_forward` APIs accept explicit
cache, page table, position, prompt-length, batch, and state inputs. The default
decode mode returns sampled token IDs. `sampling_mode="host"` and
`host_sampling_compatibility=True` are explicit test-only compatibility
boundaries that read logits; neither is used in optimized measurements.
Low-level device sampling starts with `sampling_request_start=True`, explicit
sampling parameters, and per-row prompt/output history. It owns active prefix
slots, initializes their seeds and penalty state, advances each seed exactly
once per decoded token, preserves inactive rows, and rejects mid-request
parameter changes. Reset clears this request state.

## Split trace and sampling selection

Decode uses two independent TTNN traces. The model trace consumes the
persistent token buffer, advances cache plus both position tensors, and emits
TP vocabulary logits. The common `SamplingGenerator` force-argmax trace finds
each TP shard's local maximum, gathers those candidates over the physical 1x4
Ring, chooses the global greedy token, and writes directly back to the same
token buffer. Python observes output tokens for the API response but does not
feed them back to the model.

The reduced trace test proves:

- selected tokens appear in the persistent input buffer;
- active positions increment while inactive rows remain `-1`;
- changed page tables update the persistent device table;
- identical page tables reuse the prior host snapshot and device allocation;
- mixed non-aligned prompts retain separate positions and compact linear state;
- reset is in-place and trace teardown releases both trace objects;
- final-position capture observes position 262143 at both trace boundaries,
  rather than advancing warmup to an invalid 262144;
- the selected small-Ring gather passes safe Watcher without a Linear barrier.

Both common samplers were tested with semantically greedy parameters. The
selected common `SamplingGenerator` force-argmax trace measured 2.416 ms.
Common `Sampling1D` selected the same token 225721 but measured 10.798 ms; it
was rejected because it is 4.47x slower and lacks the selected wrapper's
sampling-parameter, seed, penalty, logprob, and trace-state management. No
custom device sampler was written. Linear common-sampler routing was also rejected:
safe Watcher reproduced asynchronous all-gather writer stalls in both force
argmax and standard top-k. The retained model-scoped small-Ring opt-in matches
the physical P300c topology while preserving generic Linear behavior for small
logical submeshes.

The standard stochastic path uses the same proven physical-Ring protocol to
gather the TP vocabulary logits once. It then runs four 65,536-wide local
top-k operations and concatenates their values plus persistent global indices
into the common sampler's 128-candidate input. This avoids both the unsafe
Linear candidate collective and a non-terminating 262,144-wide top-k. The
final warning-free seeded-and-penalized safe-Watcher gate passes in 178.05 seconds.

Public `sampling_params` apply from the prefill token onward. The unavoidable
prefill host-logit boundary implements deterministic seeded temperature,
top-k, top-p, presence/frequency, and repetition handling; later optimized
device decode uses the common sampler trace. Explicit host compatibility mode
uses the same parameter-aware policy at every token and copies only the chosen
token into the persistent device input. Focused controls prove non-argmax
token-zero selection, seeded repeatability, top-p filtering, and host feedback.
For device stochastic decode, request setup installs prompt/output history,
resets the selected slot's persistent seed, and advances that seed once per
decoded token; trace replay copies the updated persistent seed without a
per-token host rebuild. Reset clears all request sampling state.
This CPU reference policy is not a token-out sampler candidate: both common
device paths were compared first, and the selected production path remains the
common `SamplingGenerator` trace.

The steady-state greedy host-work ledger is explicit. Each autonomous step
replays one model trace and one sampler trace. Token refreshes, current-position
refreshes, RoPE refreshes, unchanged-page-table refreshes, mask rebuilds, and
explicit synchronizations are all zero. A changed page table is copied once at
the request boundary. Caller-visible generation reads one sampled ID per step;
the device-only attribution loop reads none. Unseeded stochastic mode performs
two seed copies at request setup and zero per steady-state token. Explicitly
seeded stochastic mode performs one seed-tensor copy per token as required by
the common sampler contract; token feedback, positions, and page tables remain
device-resident.

## Context and capacity

The public context remains the HF-advertised 262,144 tokens, interpreted as a
total physical KV-token budget across active prompts. Per-device capacity is:

| Item | Bytes/device |
|---|---:|
| Selected decoder weights | 10,035,920,896 |
| Replicated BF16 embedding | 2,542,796,800 |
| Replicated BF16 final norm | 327,680 |
| TP4 BFP8 LM head | 356,515,840 |
| Full-model weights | 12,935,561,216 |
| BFP8 paged KV cache at max context | 2,281,701,376 |
| Batch-32 linear recurrent state | 1,459,617,792 |
| Persistent CCL pool | 3,317,760 |
| Trace/activation/fragmentation reserve | 4,294,967,296 |
| Planned total | 20,975,165,440 (19.535 GiB) |

That leaves 13,384,572,928 bytes on each 32 GiB device. There is no physical
reason to reduce context, so `doc/context_contract.json` records no capability
reduction and preserves the decoder's maximum-context evidence.
The stochastic common-sampler contract reserves 33,554,432 bytes/device for
four persistent global-index chunks plus 16,384 bytes for offsets, replacing
the prior 8,388,608-byte local index tensor. The 25,165,824-byte delta and the
transient gathered logits fit inside the existing 4 GiB trace/activation/
fragmentation reserve; the planned total therefore does not change.

## Runtime fallback audit

| Boundary | Audit result |
|---|---|
| Model stack | Always constructs `MultichipDecoder` on a 1x4 TP mesh; other mesh shapes raise instead of falling back |
| Residual/layout | Replicated BF16 TILE/DRAM remains the only inter-layer contract; active batch compaction happens with TTNN slice/pad ops inside the trace |
| Cache ownership | KV cache and recurrent state stay device-resident; page-table host copies occur only at request/state boundaries |
| Decode logits | No `to_torch`, argmax, or full-logit readback in `decode_device`; full logits cross to host only in explicit host compatibility mode |
| Sampling/feedback | Canonical greedy and stochastic common-sampler paths are traced and write the next token on device; the host loop only observes returned IDs |
| Reset | Cache/state tensors are zero-filled in place; no host cache reconstruction or silent state replacement |

Prefill necessarily returns host logits for readiness accuracy checks, and the
initial post-prefill token is selected at that public boundary. Every subsequent
optimized decode token uses split device sampling. Teacher forcing explicitly
overwrites the device token with its ground truth after recording each TT
prediction; this is not the autonomous token-out path.

## Qualitative evaluation

The AIME24 HF reference begins with a coherent English plan that identifies the
distance, both timing scenarios, and the requested `s+1/2` case. The independent
100-token HF completion exactly reproduces the fresh reference. TT matches its
first 40 tokens, then changes “coffee break” to the semantically equivalent
“coffee shop visit” and continues a relevant English solution outline. Neither
completion repeats, changes language, or loses the problem; both are truncated
at the requested 100-token inspection boundary. The outputs are stored under
`evidence/autoregressive/`.

A separate shared six-prompt, 64-token chat-template suite covers factual,
reasoning, summarization, coding, multilingual, and repetition controls. All
six TT outputs are coherent and relevant with no wrong-language drift,
prompt-format leakage, or pathological repetition. One phrase repetition in
the reasoning control is present in both HF and TT and is therefore not a TT
regression. Prompt metadata, HF/TT outputs, the automatic degeneracy report,
and the manual per-prompt verdict are under `evidence/qualitative/`.

## Evidence and limitations

Primary final results are summarized in `evidence/final_validation.md` and
backed by machine-readable `evidence/token_out_ring_metrics.json`,
`evidence/teacher_forcing_ring_metrics.json`, `evidence/prefill_ring_metrics.json`,
`evidence/full64_ring_argmax_metrics.junit.xml`,
`evidence/reduced_trace_ring_autofix_final.junit.xml`, and
`evidence/run_autoregressive.log`. The safe-Watcher context proof is
`evidence/max_context_watcher_ring_sampler.junit.xml`. The selected terminal
profile is `evidence/profiler/two_kind_ring_capture.md`,
`two_kind_ring_token_out_report.csv`, `two_kind_ring_token_out_summary.csv`,
and `two_kind_ring_token_out_summary.png`. The older `ring_` files retain the
one-linear-layer selected comparison; unprefixed profiler files retain the
rejected Linear diagnostic.
Reference data is `../../readiness_aime24_chat.refpt`. Exact commands and the
candidate/rejection ledger are in `work_log.md`; `artifact_manifest.sha256`
locks the primary reference, correctness, qualitative, and profiler artifacts.
The older `run_*` and `token_out_64_layer.log` files retain the chronological
pre-AutoFix evidence and are not the source of final Ring performance numbers.

The final AutoFix-4 stochastic gate preserves its complete console in
`evidence/reduced_trace_ring_stochastic_autofix4.log`. Moving penalty-history
maintenance into the captured sampler program removed the prior trace-resident
allocation. The log contains no unsafe-allocation warning, Watcher error/fatal,
or test failure; its paired JUnit records one pass in 178.05 seconds.

The supported hardware contract is TP4 on one `1x4` Blackhole P300c ring.
Active decode slots are stable and contiguous; arbitrary sparse slot IDs are
not advertised. The 262,144-token number is one total cache pool, not that many
tokens independently for all 32 rows. Full-model prefill is dynamic rather
than traced because it owns arbitrary logical lengths and cache fill. This
stage does not start vLLM integration.
