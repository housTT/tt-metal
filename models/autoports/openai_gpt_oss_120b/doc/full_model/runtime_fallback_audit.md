# Runtime fallback audit

Scope: the resident GPT-OSS 120B full-model construction, generator prefill and
decode paths, cache ownership, sampling boundaries, reset, and teardown.

| Boundary | Production behavior | Audit result |
| --- | --- | --- |
| Decoder stack | All 36 blocks instantiate `MultichipDecoder` with `DEFAULT_MULTICHIP_POLICY` | clean; no mature single-chip or reduced decoder substitute |
| Tensor parallelism | Exact `(1,4)` mesh-axis-1 sharding | clean; no host, single-device, or replicated-weight fallback |
| Inter-layer residual | Logical 2880-wide BF16 replicated residual, L1 decode / DRAM prefill | clean; no host round trip or alternate layout |
| KV cache | Generator-owned paged BFP8 cache unless an explicit low-level cache is supplied | clean; layer-local KV heads and page rows are preserved |
| Non-aligned prefill | Generator passes logical lengths and owns padding, cache fill, positions, and output slicing | clean; 214-token AIME and mixed 7/5-token prompts passed |
| Low-level serving state | Explicit cache, page table, position, prompt length, batch, fixed slot, and inactive-row state | clean; batch-2 inactive `-1` row passed |
| Optimized sampling | Traced `SamplingGenerator` top-k=1 with `tt_out_tok` device feedback | clean; no Python token feedback, host argmax, full-logit readback, or untraced sampling |
| Host compatibility | `sampling_mode="host"` explicitly requests logits/host argmax | isolated compatibility boundary; never selected implicitly |
| Teacher forcing | `next_input` explicitly requests caller-authoritative tokens and full input refreshes | isolated test boundary; not used for measured token-out |
| EOS | Checkpoint `generation_config.json` set `{200002, 199999, 200012}` | clean; no hard-coded `<|end|>` truncation and no post-EOS qualitative tail |
| Page-table update | Initial full refresh, unchanged-table reuse, changed-table-only refresh | clean; persistent page-table contents asserted on hardware |
| Reset | Clears KV cache only after use, clears request/page/mode state, retains warmed traces | clean; full-36 batch-2 logits are bitwise exact across two reset/reuse runs after external device recovery, with no alternate cache owner or silent state carryover |
| Teardown | Idempotently releases and clears trace stores | clean; repeated cleanup cannot double-release traces |
| Capacity failure | P150/TP1 and P150x2/TP2 raise `FullModelCapacityError` | clean; message names forbidden fallback and largest feasible context |

Host boundaries used by validation are deliberate: prefill/top-k accuracy reads
full logits, teacher forcing receives HF tokens, and the explicit host sampler
reads logits.  The free-running performance and qualitative paths use none of
those boundaries.

The reproducibility investigation also audited failure recovery.  A healthy
`tt-smi` snapshot can miss stale external fabric/collective state; a bounded
board reset cleared one such source-unchanged failure, after which the exact
36-layer gate passed twice in fresh processes.  This is device-environment
recovery, not a generator reset substitute or a runtime fallback.  The model
keeps exact logit equality as its acceptance gate.
