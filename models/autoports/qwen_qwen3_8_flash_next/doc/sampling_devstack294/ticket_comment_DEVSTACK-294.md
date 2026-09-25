# Draft comment for DEVSTACK-294 (review, then post)

Evidence for every number: `models/autoports/qwen_qwen3_8_flash_next/doc/sampling_devstack294/README.md`; latency rows from the community sweep harness run on 2026-09-17 against the new package.

---

Thanks all — new package revision is published (`tt-model serve tt-hous/qwen3.8-flash-next-p300x2 --refresh`).

**1. Repetition collapse.** Fixed as requested: `temperature > 0` requests now sample on the host; greedy stays on device. Cost is ~18 ms/token for stochastic requests (41 → 59 ms below 2k context, 72 → 91 ms above). `QWEN38_DEVICE_SAMPLING_MAX_TOP_K=32` in the serve env restores device sampling.

We could not reproduce the collapse: 24 device-sampled runs (new build, and the 09-11 image as installed) and 8 host-sampled runs of the AIME prompt at the default preset all finish cleanly with the right answer. A new distribution gate shows the device sampler within noise of the vLLM reference (0 draws outside the nucleus in 49k). Sam, can you attach the request bodies of one collapsed run and one clean host run? If the host runs carried a `presence_penalty`, that alone explains the difference (vLLM sends penalised requests to the host in every mode).

**2. Long context.** Cost rule: `E2EL = TTFT + max_tokens × TPOT`. TTFT is linear in prompt length; TPOT is flat above 2k tokens and does not depend on context length. That is why 256 → 64 output tokens saved only ~15 s. `trace_mode: all` and `--max-num-batched-tokens 8192` are no-ops on this port.

The latest revision also fixes the long-context accuracy caveat and speeds up prefill: the sparse-attention block-key cache was written with a different page geometry than it was read (half the block keys invisible to the selector), so decode agreement after a 3k/6k/12k-token document was 73–76 % top-1 against the reference; it is now 90–98 %. The same bug had blocked a larger prefill microchunk, which now ships. Measured on the published package (greedy, one user):

| ISL | TTFT before → now | E2EL now (OSL 256) | TPOT above 2k |
| ---: | ---: | ---: | ---: |
| 16k | 52 s → 38 s | 57 s | 72 ms |
| 64k | 222 s → 163 s | 182 s | 72 ms |
| 131k | 458 s → 338 s | 357 s | 72 ms |

TTFT is now ~2.6 ms per prompt token (~400 tok/s). For RULER at 131k set the client timeout ≥ 10 min; 32k/64k is the practical iteration loop. Was the 300 s a harness timeout? The card states all of this. Next on the prefill roadmap: a capacity-bounded selector view (another −70 % of the selector term above 2k), then traced prefill.

**3. AIME24.** Documented as a limitation (drift beyond ~25–30k reasoning tokens; cap `max_tokens` near 32k). Please re-run the failed items on the new default and note the reasoning-token count of each.

**4. 335 GB.** It is exactly 180.0 B BF16 parameters, nothing duplicated: 123.6 B routed experts, 51.2 B n-gram table (host-mapped, never on device), 3.5 B dense, 1.3 B embeddings/LM head, 0.45 B unused vision tower. ~6.7 B active per token. A pre-converted BFP snapshot would halve it; separate item.
